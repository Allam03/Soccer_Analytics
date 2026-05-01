"""
pipelines/ingest_pass_network.py

Extract passer -> receiver edges from StatsBomb Pass events and populate
the pass_network_edges table.  This feeds Model 2 (Team Cohesion Analysis).

Edge aggregation: one row per (match_id, team_id, passer_id, receiver_id)
with:
  - pass_count    : number of completed passes on this edge
  - avg_x_start   : mean passer x position
  - avg_y_start   : mean passer y position
  - avg_x_end     : mean receiver x position (pass end_location)
  - avg_y_end     : mean receiver y position
"""

import logging
from collections import defaultdict

from psycopg2.extras import execute_values

from extract import statsbomb_local as sb
from core.caches import TeamCache, PlayerCache
from load.postgres import connect
from config.settings import DB_DSN, COMPETITIONS

logger = logging.getLogger(__name__)


def _extract_edges(events, pg_match_id, team_cache, player_cache):
    """
    Parse all Pass events in `events` and return a list of edge tuples
    ready for bulk insertion.

    Returns
    -------
    list of tuples:
        (match_id, team_id, passer_id, receiver_id,
         pass_count, avg_x_start, avg_y_start, avg_x_end, avg_y_end)
    """
    # Accumulator: (pg_match_id, pg_team_id, passer_pid, receiver_pid)
    #   -> {"count": int, "x_start": [], "y_start": [], "x_end": [], "y_end": []}
    edge_data = defaultdict(lambda: {"count": 0, "xs": [], "ys": [], "xe": [], "ye": []})

    pass_events = events[events["type"].apply(
        lambda t: isinstance(t, dict) and t.get("name") == "Pass"
    )]

    for _, ev in pass_events.iterrows():
        pass_data = ev.get("pass") or {}

        # Only count completed passes (outcome is None in StatsBomb)
        if pass_data.get("outcome") is not None:
            continue

        passer = ev.get("player")
        recip  = pass_data.get("recipient")
        team   = ev.get("team")

        if not (isinstance(passer, dict) and isinstance(recip, dict)
                and isinstance(team, dict)):
            continue

        passer_sb = passer.get("id")
        recip_sb  = recip.get("id")
        team_sb   = team.get("id")

        if not (passer_sb and recip_sb and team_sb):
            continue

        try:
            pg_passer   = player_cache.get_or_create(passer_sb, passer.get("name", ""))
            pg_receiver = player_cache.get_or_create(recip_sb,  recip.get("name", ""))
            pg_team     = team_cache.get_or_create(team_sb,     team.get("name", ""))
        except Exception as exc:
            logger.debug("Cache error building edge: %s", exc)
            continue

        key = (pg_match_id, pg_team, pg_passer, pg_receiver)
        loc_start = ev.get("location") or []
        loc_end   = pass_data.get("end_location") or []

        acc = edge_data[key]
        acc["count"] += 1
        if len(loc_start) >= 2:
            acc["xs"].append(loc_start[0])
            acc["ys"].append(loc_start[1])
        if len(loc_end) >= 2:
            acc["xe"].append(loc_end[0])
            acc["ye"].append(loc_end[1])

    rows = []
    for (mid, tid, pid, rid), acc in edge_data.items():
        avg = lambda lst: sum(lst) / len(lst) if lst else None
        rows.append((
            mid, tid, pid, rid,
            acc["count"],
            avg(acc["xs"]),
            avg(acc["ys"]),
            avg(acc["xe"]),
            avg(acc["ye"]),
        ))
    return rows


def run(conn=None):
    """
    Iterate over all in-scope competition-seasons and populate
    pass_network_edges for every match.  Skips matches that already
    have edges (idempotent).
    """
    if conn is None:
        conn = connect(DB_DSN)

    team_cache   = TeamCache(conn)
    player_cache = PlayerCache(conn)

    # Fetch all matches already in the DB
    with conn.cursor() as cur:
        cur.execute("""
            SELECT m.match_id, m.statsbomb_match_id
            FROM   matches m
            WHERE  NOT EXISTS (
                SELECT 1 FROM pass_network_edges e
                WHERE  e.match_id = m.match_id
            )
        """)
        pending = cur.fetchall()

    logger.info("Pass network: %d matches to process", len(pending))
    inserted_total = 0

    for pg_match_id, sb_match_id in pending:
        try:
            events = sb.events(sb_match_id)
        except Exception as exc:
            logger.error("Could not load events for SB match %d: %s", sb_match_id, exc)
            continue

        edge_rows = _extract_edges(events, pg_match_id, team_cache, player_cache)
        if not edge_rows:
            logger.debug("No pass edges for match %d", sb_match_id)
            continue

        with conn.cursor() as cur:
            execute_values(cur, """
                INSERT INTO pass_network_edges (
                    match_id, team_id, passer_id, receiver_id,
                    pass_count,
                    avg_x_start, avg_y_start, avg_x_end, avg_y_end
                ) VALUES %s
                ON CONFLICT DO NOTHING
            """, edge_rows)
        conn.commit()
        inserted_total += len(edge_rows)
        logger.info("  Match %d: %d edges inserted", sb_match_id, len(edge_rows))

    logger.info("Pass network ingestion complete: %d total edges", inserted_total)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    run()