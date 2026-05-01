import logging
import time

from extract import statsbomb_local as sb
from transform.features import agg_player_events
from transform.schema import player_id, team_id, event_type
from load.postgres import insert_stats, upsert_match
from config.settings import COMPETITIONS

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def _match_result(match_row, team_cache, home_team_id, away_team_id):
    """
    Return a dict mapping statsbomb team_id -> 'win'/'draw'/'loss'
    for both sides of a match.
    """
    hs = match_row.get("home_score", 0) or 0
    aw = match_row.get("away_score", 0) or 0
    if hs > aw:
        return {home_team_id: "win",  away_team_id: "loss"}
    elif hs < aw:
        return {home_team_id: "loss", away_team_id: "win"}
    return {home_team_id: "draw", away_team_id: "draw"}


def run(
    conn,
    team_cache,
    player_cache,
    weather_cache=None,
    batch_size: int = 1000,
    track_metrics: bool = True,
):
    """
    Process StatsBomb open data into the PostgreSQL database.

    Steps per match
    ---------------
    1. Upsert the match row (fixes issue #2).
    2. For each player, aggregate all event types (fixes issue #5).
    3. Bulk-insert player_match_stats with all columns (fixes issue #12).

    Competition filter (fixes issue #11): only processes competitions listed
    in config.settings.COMPETITIONS.
    """
    if track_metrics:
        from metrics import MetricsTracker
        tracker = MetricsTracker()
        tracker.start()

    comps = sb.competitions()

    # Issue #11 fix: filter to in-scope competitions only
    if COMPETITIONS:
        comps = comps[comps.apply(
            lambda r: (int(r["competition_id"]), int(r["season_id"])) in COMPETITIONS,
            axis=1,
        )]

    logger.info("Found %d in-scope competition-seasons to process", len(comps))

    for _, comp in comps.iterrows():
        comp_start = time.time()
        comp_name  = comp.get("competition_name", f"Comp {comp['competition_id']}")
        season     = comp.get("season_name", str(comp["season_id"]))

        logger.info("=" * 60)
        logger.info("Processing: %s  Season: %s", comp_name, season)
        logger.info("=" * 60)

        try:
            matches = sb.matches(comp["competition_id"], comp["season_id"])

            if len(matches) == 0:
                logger.warning("No matches found for %s", comp_name)
                continue

            logger.info("Found %d matches", len(matches))

            processed_matches = 0
            total_events      = 0
            failed_events     = 0

            for _, match in matches.iterrows():
                sb_match_id = int(match["match_id"])
                logger.info("  Match %d", sb_match_id)

                # ----------------------------------------------------------
                # Issue #2 fix: upsert the match into the matches table
                # ----------------------------------------------------------
                home_sb_id = match["home_team"]["home_team_id"]
                away_sb_id = match["away_team"]["away_team_id"]
                home_pg_id = team_cache.get_or_create(
                    home_sb_id, match["home_team"]["home_team_name"]
                )
                away_pg_id = team_cache.get_or_create(
                    away_sb_id, match["away_team"]["away_team_name"]
                )

                stadium     = match.get("stadium") or {}
                stadium_lat = None
                stadium_lng = None
                if isinstance(stadium, dict):
                    # StatsBomb open data does not include coordinates;
                    # the weather pipeline will fill these from the match row.
                    stadium_name = stadium.get("name")
                else:
                    stadium_name = None

                match_row_db = {
                    "statsbomb_match_id": sb_match_id,
                    "match_date":         match.get("match_date"),
                    "home_team_id":       home_pg_id,
                    "away_team_id":       away_pg_id,
                    "home_score":         match.get("home_score"),
                    "away_score":         match.get("away_score"),
                    "competition":        comp_name,
                    "season":             season,
                    "stadium_name":       stadium_name,
                    "stadium_lat":        stadium_lat,
                    "stadium_lng":        stadium_lng,
                }
                pg_match_id = upsert_match(conn, match_row_db)

                # Fetch weather_id if available from the weather cache
                weather_id = None
                if weather_cache and pg_match_id in weather_cache:
                    weather_id = weather_cache[pg_match_id]

                # Determine result for each team
                result_map = _match_result(
                    match, team_cache, home_sb_id, away_sb_id
                )

                # ----------------------------------------------------------
                # Load events
                # ----------------------------------------------------------
                try:
                    events = sb.events(sb_match_id)
                except Exception as exc:
                    logger.error("Failed to load events for match %d: %s", sb_match_id, exc)
                    failed_events += 1
                    continue

                total_events += len(events)

                if len(events) == 0:
                    logger.warning("  No events for match %d", sb_match_id)
                    continue

                # ----------------------------------------------------------
                # Build stat rows per player
                # ----------------------------------------------------------
                # Collect unique (player_id, team_id) pairs from the events
                player_team_map = {}
                for _, ev in events.iterrows():
                    p = ev.get("player")
                    t = ev.get("team")
                    if isinstance(p, dict) and isinstance(t, dict):
                        player_team_map[p["id"]] = t["id"]

                stat_rows = []
                for sb_pid, sb_tid in player_team_map.items():
                    try:
                        pg_player_id = player_cache.get_or_create(
                            sb_pid,
                            events[events["player"].apply(
                                lambda x, pid=sb_pid: isinstance(x, dict)
                                and x.get("id") == pid
                            )].iloc[0]["player"]["name"],
                        )
                        pg_team_id   = team_cache.get_or_create(
                            sb_tid,
                            events[events["team"].apply(
                                lambda x, tid=sb_tid: isinstance(x, dict)
                                and x.get("id") == tid
                            )].iloc[0]["team"]["name"],
                        )
                    except Exception as exc:
                        logger.debug("Cache miss for player %d: %s", sb_pid, exc)
                        failed_events += 1
                        continue

                    try:
                        # Issue #5 fix: full event aggregation
                        stats = agg_player_events(events, sb_pid, event_type)
                    except Exception as exc:
                        logger.error(
                            "  Error aggregating player %d in match %d: %s",
                            sb_pid, sb_match_id, exc,
                        )
                        failed_events += 1
                        continue

                    result = result_map.get(sb_tid)

                    # Tuple order must match insert_stats() column list exactly
                    stat_rows.append((
                        pg_player_id,
                        pg_match_id,
                        pg_team_id,
                        weather_id,
                        result,
                        stats["goals"],
                        stats["assists"],
                        stats["shots"],
                        stats["xg"],
                        stats["xa"],
                        stats["key_passes"],
                        stats["passes_attempted"],
                        stats["passes_completed"],
                        stats["pass_accuracy"],
                        stats["progressive_passes"],
                        stats["carry_distance"],
                        stats["progressive_carries"],
                        stats["dribbles_completed"],
                        stats["tackles"],
                        stats["interceptions"],
                        stats["clearances"],
                        stats["pressures"],
                        stats["yellow_cards"],
                        stats["red_cards"],
                        stats["minutes_played"],
                        stats["sub_minute"],
                    ))

                # Issue #12 fix: insert_stats now covers all columns
                if stat_rows:
                    try:
                        insert_stats(conn, stat_rows)
                    except Exception as exc:
                        logger.error(
                            "  Failed to insert stats for match %d: %s",
                            sb_match_id, exc,
                        )
                        failed_events += len(stat_rows)

                logger.info(
                    "  Match %d: %d player rows, %d events",
                    sb_match_id, len(stat_rows), len(events),
                )
                processed_matches += 1
                if track_metrics:
                    tracker.match_processed(sb_match_id)

            logger.info(
                "Completed %s: %d/%d matches, %d events, %d failed",
                comp_name, processed_matches, len(matches),
                total_events, failed_events,
            )

            if track_metrics:
                tracker.competition_completed(
                    comp_name, len(matches), total_events,
                    failed_events, time.time() - comp_start,
                )

        except Exception as exc:
            logger.error("Failed to process %s: %s", comp_name, exc)
            if track_metrics:
                tracker.competition_failed(comp_name, str(exc))
            raise

    if track_metrics:
        tracker.finish()
        logger.info("=" * 60)
        logger.info("Pipeline Summary:")
        # Issue #13 fix: use .items() not tuple-unpacking of dict
        for metric, value in tracker.get_metrics().items():
            logger.info("  %s: %s", metric, value)
        logger.info("=" * 60)