"""
pipelines/ingest_statsbomb.py

High-performance StatsBomb ingestion pipeline.

Architecture
------------
                         main process
                        /            \\
              DB writes              ProcessPoolExecutor
              (serial)               (N worker processes)
                  ^                        |
                  |                  _process_match()
            result queue            - pd.read_json (JSON load)
                  |                 - agg_match_by_player (vectorised)
                  |                 - extract_pass_edges
                  \\                        |
                   \\_______ returns MatchResult namedtuple

Performance wins
----------------
1. Vectorised feature computation   -- no iterrows(), mask+sum only
2. Single groupby per match         -- eliminates N re-scans of events DF
3. Single name-lookup pass          -- one dict build per match, not per player
4. Batched DB commits               -- one commit per competition, not per row
5. ProcessPoolExecutor              -- parallel JSON load + feature compute
7. Pass network extracted here      -- eliminates second full read of all JSON files
"""

import logging
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import pandas as pd
from psycopg2.extras import execute_values

from config.settings import COMPETITIONS, DATA_ROOT, DB_DSN
from extract import statsbomb_local as sb
from load.postgres import connect, insert_stats, upsert_match, upsert_pass_edges
from transform.features import (
    agg_match_by_player,
    extract_player_id_col,
    extract_team_id_col,
    extract_type_col,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
)
logger = logging.getLogger(__name__)

# Number of worker processes.  Default = CPU count - 1 (leave one for DB writes).
_WORKERS = max(1, (os.cpu_count() or 2) - 1)

# How many matches to accumulate before flushing to DB (within one competition).
_COMMIT_EVERY = 50


# ---------------------------------------------------------------------------
# Data structures passed between worker and main process
# ---------------------------------------------------------------------------

@dataclass
class MatchResult:
    """Everything a worker produces for one match."""
    sb_match_id:  int
    match_date:   object          # str or date
    home_sb_id:   int
    away_sb_id:   int
    home_name:    str
    away_name:    str
    home_score:   int
    away_score:   int
    comp_name:    str
    season:       str
    stadium_name: Optional[str]

    # {sb_player_id -> (name, sb_team_id)}
    player_team:  dict = field(default_factory=dict)

    # {sb_player_id -> stats_dict}
    player_stats: dict = field(default_factory=dict)

    # pass network edge rows (no pg IDs yet -- still StatsBomb IDs)
    # list of (sb_match_id, sb_team_id, sb_passer_id, sb_receiver_id,
    #          pass_count, avg_xs, avg_ys, avg_xe, avg_ye)
    pass_edges_sb: list = field(default_factory=list)

    error: Optional[str] = None


# ---------------------------------------------------------------------------
# Worker function  (runs in a child process -- no DB connection, no cache)
# ---------------------------------------------------------------------------

def _process_match(
    sb_match_id: int,
    comp_name: str,
    season: str,
    match_date: object,
    home_team: dict,
    away_team: dict,
    home_score: int,
    away_score: int,
    stadium_name: Optional[str],
    data_root: str,
) -> MatchResult:
    """
    Load events for one match, compute all player stats and pass edges.
    Pure CPU / IO -- no DB access.
    Runs in a worker process.
    """
    result = MatchResult(
        sb_match_id  = sb_match_id,
        match_date   = match_date,
        home_sb_id   = home_team["home_team_id"],
        away_sb_id   = away_team["away_team_id"],
        home_name    = home_team["home_team_name"],
        away_name    = away_team["away_team_name"],
        home_score   = home_score,
        away_score   = away_score,
        comp_name    = comp_name,
        season       = season,
        stadium_name = stadium_name,
    )

    try:
        # ---- Fix #6: JSON loading happens in the worker process ----------
        sb.set_root(data_root)
        events = sb.events(sb_match_id)

        if events.empty:
            return result

        # ---- Pre-extract flat columns once per match ---------------------
        type_col      = extract_type_col(events)
        player_id_col = extract_player_id_col(events)
        team_id_col   = extract_team_id_col(events)

        # ---- Fix #3: single pass to build player->name and player->team --
        for _, row in events.iterrows():
            p = row["player"]
            t = row["team"]

            if isinstance(p, dict) and isinstance(t, dict):
                pid = p.get("id")
                tid = t.get("id")

                if isinstance(pid, int) and isinstance(tid, int):
                    result.player_team[pid] = (p.get("name", ""), tid)

        # ---- Fix #1 + #2: vectorised aggregation, single groupby ---------
        result.player_stats = agg_match_by_player(events, type_col, player_id_col)

        # ---- Fix #7: extract pass edges in the same event read -----------
        result.pass_edges_sb = _extract_pass_edges_sb(
            events, type_col, player_id_col, team_id_col, sb_match_id
        )

    except Exception as exc:
        result.error = str(exc)

    return result


def _extract_pass_edges_sb(
    events: pd.DataFrame,
    type_col: pd.Series,
    player_id_col: pd.Series,
    team_id_col: pd.Series,
    sb_match_id: int,
) -> list:
    """
    Extract pass network edges using StatsBomb IDs (no DB lookup needed).
    Returns list of (sb_match_id, sb_team_id, sb_passer_id, sb_receiver_id,
                     pass_count, avg_xs, avg_ys, avg_xe, avg_ye).
    """
    is_pass = type_col == "Pass"
    if not is_pass.any():
        return []

    pass_df = events.loc[is_pass].copy(deep=False)
    pass_df["_pid"] = player_id_col[is_pass].values
    pass_df["_tid"] = team_id_col[is_pass].values

    from collections import defaultdict
    acc = defaultdict(lambda: {"n": 0, "xs": 0.0, "ys": 0.0, "xe": 0.0, "ye": 0.0})

    for _, row in pass_df.iterrows():
        pass_data = row.get("pass") or {}
        # Completed passes only
        if pass_data.get("outcome") is not None:
            continue
        recip = pass_data.get("recipient")
        if not isinstance(recip, dict):
            continue
        recip_id = recip.get("id")
        passer_id = row["_pid"]
        team_id   = row["_tid"]
        if not (passer_id and recip_id and team_id):
            continue

        key = (int(team_id), int(passer_id), int(recip_id))
        loc_s = row.get("location") or []
        loc_e = pass_data.get("end_location") or []
        a = acc[key]
        a["n"] += 1
        if len(loc_s) >= 2:
            a["xs"] += loc_s[0]; a["ys"] += loc_s[1]
        if len(loc_e) >= 2:
            a["xe"] += loc_e[0]; a["ye"] += loc_e[1]

    rows = []
    for (tid, pid, rid), a in acc.items():
        n = a["n"]
        rows.append((
            sb_match_id, tid, pid, rid, n,
            a["xs"] / n, a["ys"] / n,
            a["xe"] / n, a["ye"] / n,
        ))
    return rows


# ---------------------------------------------------------------------------
# Main-process DB writer
# ---------------------------------------------------------------------------

def _write_results(
    conn,
    team_cache,
    player_cache,
    results: list[MatchResult],
    weather_cache: dict,
) -> tuple[int, int]:
    """
    Translate a batch of MatchResult objects into DB rows and flush.
    Returns (stat_rows_inserted, edge_rows_inserted).
    """
    all_stat_rows = []
    all_edge_rows = []

    for res in results:
        if res.error:
            logger.error("Match %d failed in worker: %s", res.sb_match_id, res.error)
            continue

        # ---- Register teams / players in caches (buffered) ---------------
        home_pg = team_cache.get_or_create(res.home_sb_id, res.home_name)
        away_pg = team_cache.get_or_create(res.away_sb_id, res.away_name)
        for sb_pid, (p_name, sb_tid) in res.player_team.items():
            player_cache.get_or_create(sb_pid, p_name)
            team_cache.get_or_create(sb_tid, "")   # name may be blank, already in cache

        # Flush caches -- one round-trip each
        team_cache.flush()
        player_cache.flush()

        # ---- Upsert match row (no commit yet) ----------------------------
        hs, aw = res.home_score, res.away_score
        if hs > aw:
            result_map = {res.home_sb_id: "win",  res.away_sb_id: "loss"}
        elif hs < aw:
            result_map = {res.home_sb_id: "loss", res.away_sb_id: "win"}
        else:
            result_map = {res.home_sb_id: "draw", res.away_sb_id: "draw"}

        stadium = res.stadium_name
        pg_match_id = upsert_match(conn, {
            "statsbomb_match_id": res.sb_match_id,
            "match_date":         res.match_date,
            "home_team_id":       team_cache.resolve(res.home_sb_id),
            "away_team_id":       team_cache.resolve(res.away_sb_id),
            "home_score":         res.home_score,
            "away_score":         res.away_score,
            "competition":        res.comp_name,
            "season":             res.season,
            "stadium_name":       stadium,
            "stadium_lat":        None,
            "stadium_lng":        None,
        })

        weather_id = weather_cache.get(pg_match_id)

        # ---- Build stat rows ---------------------------------------------
        for sb_pid, stats in res.player_stats.items():
            info = res.player_team.get(sb_pid)
            if info is None:
                continue
            _, sb_tid = info
            try:
                pg_pid = player_cache.resolve(sb_pid)
                pg_tid = team_cache.resolve(sb_tid)
            except KeyError:
                logger.debug("Unresolved ID: player=%d team=%d", sb_pid, sb_tid)
                continue

            result = result_map.get(sb_tid)
            all_stat_rows.append((
                pg_pid, pg_match_id, pg_tid, weather_id, result,
                stats["goals"], stats["assists"], stats["shots"],
                stats["xg"], stats["xa"], stats["key_passes"],
                stats["passes_attempted"], stats["passes_completed"],
                stats["pass_accuracy"], stats["progressive_passes"],
                stats["carry_distance"], stats["progressive_carries"],
                stats["dribbles_completed"],
                stats["tackles"], stats["interceptions"],
                stats["clearances"], stats["pressures"],
                stats["yellow_cards"], stats["red_cards"],
                stats["minutes_played"], stats["sub_minute"],
            ))

        # ---- Build pass edge rows ----------------------------------------
        for (sb_match_id_e, sb_tid, sb_pid, sb_rid,
             n, axs, ays, axe, aye) in res.pass_edges_sb:
            try:
                pg_passer   = player_cache.resolve(sb_pid)
                pg_receiver = player_cache.resolve(sb_rid)
                pg_team     = team_cache.resolve(sb_tid)
            except KeyError:
                continue
            all_edge_rows.append((
                pg_match_id, pg_team, pg_passer, pg_receiver,
                n, axs, ays, axe, aye,
            ))

    # ---- Single bulk write + single commit per batch --------------------
    insert_stats(conn, all_stat_rows)
    upsert_pass_edges(conn, all_edge_rows)
    conn.commit()

    return len(all_stat_rows), len(all_edge_rows)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def run(
    conn,
    team_cache,
    player_cache,
    weather_cache: dict = None,
    workers: int = _WORKERS,
    commit_every: int = _COMMIT_EVERY,
    track_metrics: bool = True,
):
    """
    Process all in-scope StatsBomb competitions into PostgreSQL.

    Parameters
    ----------
    conn          : main-process DB connection (used for all writes)
    team_cache    : TeamCache instance
    player_cache  : PlayerCache instance
    weather_cache : {pg_match_id -> weather_id} (optional)
    workers       : number of worker processes for parallel JSON loading
    commit_every  : flush accumulated rows to DB every N matches
    """
    if weather_cache is None:
        weather_cache = {}

    if track_metrics:
        from metrics import MetricsTracker
        tracker = MetricsTracker()
        tracker.start()

    comps = sb.competitions()
    if COMPETITIONS:
        comps = comps[comps.apply(
            lambda r: (int(r["competition_id"]), int(r["season_id"])) in COMPETITIONS,
            axis=1,
        )]

    logger.info("Found %d in-scope competition-seasons  |  workers=%d", len(comps), workers)

    total_stat_rows = 0
    total_edge_rows = 0

    for _, comp in comps.iterrows():
        comp_start = time.time()
        comp_name  = comp.get("competition_name", f"Comp {comp['competition_id']}")
        season     = comp.get("season_name", str(comp["season_id"]))

        logger.info("=" * 60)
        logger.info("Processing: %s  Season: %s", comp_name, season)
        logger.info("=" * 60)

        try:
            matches = sb.matches(comp["competition_id"], comp["season_id"])
            if matches.empty:
                logger.warning("No matches for %s", comp_name)
                continue

            logger.info("  %d matches  (submitting to %d workers)", len(matches), workers)

            # Build the list of futures
            futures = {}
            with ProcessPoolExecutor(max_workers=workers) as pool:
                for _, match in matches.iterrows():
                    sb_match_id  = int(match["match_id"])
                    home_team    = match["home_team"]
                    away_team    = match["away_team"]
                    stadium      = match.get("stadium")
                    stadium_name = stadium.get("name") if isinstance(stadium, dict) else None

                    fut = pool.submit(
                        _process_match,
                        sb_match_id,
                        comp_name,
                        season,
                        match.get("match_date"),
                        home_team if isinstance(home_team, dict) else {"id": 0, "name": ""},
                        away_team if isinstance(away_team, dict) else {"id": 0, "name": ""},
                        int(match.get("home_score") or 0),
                        int(match.get("away_score") or 0),
                        stadium_name,
                        str(DATA_ROOT),
                    )
                    futures[fut] = sb_match_id

                # Collect results and write in batches
                pending: list[MatchResult] = []
                done = 0

                for fut in as_completed(futures):
                    sb_mid = futures[fut]
                    try:
                        res = fut.result()
                    except Exception as exc:
                        logger.error("Worker exception for match %d: %s", sb_mid, exc)
                        continue

                    pending.append(res)
                    done += 1

                    if len(pending) >= commit_every:
                        s, e = _write_results(conn, team_cache, player_cache,
                                              pending, weather_cache)
                        total_stat_rows += s
                        total_edge_rows += e
                        logger.info(
                            "  Flushed %d/%d matches  (+%d stat rows, +%d edges)",
                            done, len(matches), s, e,
                        )
                        pending.clear()

                # Flush remainder
                if pending:
                    s, e = _write_results(conn, team_cache, player_cache,
                                          pending, weather_cache)
                    total_stat_rows += s
                    total_edge_rows += e

            elapsed = time.time() - comp_start
            logger.info(
                "Completed %s in %.1fs  |  stat rows: %d  |  edge rows: %d",
                comp_name, elapsed, total_stat_rows, total_edge_rows,
            )

            if track_metrics:
                tracker.competition_completed(
                    comp_name, len(matches), total_stat_rows, 0, elapsed
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
        for metric, value in tracker.get_metrics().items():
            logger.info("  %s: %s", metric, value)
        logger.info("=" * 60)