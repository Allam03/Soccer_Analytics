"""
pipelines/ingest_injuries.py

Load Transfermarkt injury data into the injuries table and backfill
players.date_of_birth where available.

Dataset: https://www.kaggle.com/datasets/irrazional/transfermarkt-injuries

Expected CSV columns (Transfermarkt schema):
    player_id, player_name, season, injury, date_from, date_until,
    duration_days, games_missed

The player_id in this CSV is a Transfermarkt ID, NOT a StatsBomb ID.
Matching strategy:
  1. Exact match on players.transfermarkt_player_id (fastest).
  2. Normalised name fuzzy match on players.norm_name (fallback).
"""

import logging

import pandas as pd
from psycopg2.extras import execute_values

from core.utils import norm_name
from load.postgres import connect
from config.settings import DB_DSN, TRANSFERMARKT_CSV

logger = logging.getLogger(__name__)


def _load_player_map(conn) -> tuple[dict, dict]:
    """
    Return two lookup dicts:
      tm_id_map  : transfermarkt_player_id (str) -> player_id
      norm_map   : norm_name               (str) -> player_id
    """
    tm_id_map = {}
    norm_map  = {}
    with conn.cursor() as cur:
        cur.execute("""
            SELECT player_id, transfermarkt_player_id, norm_name
            FROM players
        """)
        for pid, tmid, nn in cur.fetchall():
            if tmid:
                tm_id_map[str(tmid)] = pid
            if nn:
                norm_map[nn] = pid
    return tm_id_map, norm_map


def _resolve_player(row, tm_id_map, norm_map) -> int | None:
    """Return internal player_id or None if no match found."""
    # Try Transfermarkt ID first
    tm_id = str(row.get("player_id", "") or "")
    if tm_id and tm_id in tm_id_map:
        return tm_id_map[tm_id]

    # Fall back to normalised name
    nn = norm_name(str(row.get("player_name", "") or ""))
    return norm_map.get(nn)


def _update_transfermarkt_ids(conn, df: pd.DataFrame, tm_id_map: dict, norm_map: dict):
    """
    Back-fill players.transfermarkt_player_id where we matched by name.
    This makes future runs faster (ID match path).
    """
    updates = []
    for _, row in df.iterrows():
        tm_id = str(row.get("player_id", "") or "")
        if not tm_id:
            continue
        if tm_id in tm_id_map:
            continue  # already linked
        nn  = norm_name(str(row.get("player_name", "") or ""))
        pid = norm_map.get(nn)
        if pid:
            updates.append((tm_id, pid))

    if updates:
        with conn.cursor() as cur:
            for tm_id, pid in updates:
                cur.execute("""
                    UPDATE players
                    SET transfermarkt_player_id = %s
                    WHERE player_id = %s
                      AND transfermarkt_player_id IS NULL
                """, (tm_id, pid))
        conn.commit()
        logger.info("Back-filled transfermarkt_player_id for %d players", len(updates))


def run(conn=None, csv_path: str = None):
    """
    Parse the Transfermarkt injuries CSV and populate:
      - injuries table
      - players.transfermarkt_player_id (backfill)

    Parameters
    ----------
    conn     : psycopg2 connection (created from DB_DSN if None)
    csv_path : path to the Transfermarkt CSV (defaults to TRANSFERMARKT_CSV)
    """
    if conn is None:
        conn = connect(DB_DSN)
    if csv_path is None:
        csv_path = TRANSFERMARKT_CSV

    logger.info("Loading Transfermarkt injuries from: %s", csv_path)
    df = pd.read_csv(csv_path)
    logger.info("Loaded %d rows", len(df))

    # Normalise column names to lowercase-underscore
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]

    # Parse dates; invalid values become NaT
    for col in ("date_from", "date_until"):
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce").dt.date

    tm_id_map, norm_map = _load_player_map(conn)

    injury_rows = []
    matched     = 0
    unmatched   = 0

    for _, row in df.iterrows():
        pid = _resolve_player(row, tm_id_map, norm_map)
        if pid is None:
            unmatched += 1
            continue

        matched += 1
        injury_rows.append((
            pid,
            str(row.get("injury") or ""),
            row.get("date_from"),
            row.get("date_until"),
            int(row["games_missed"]) if pd.notna(row.get("games_missed")) else None,
            str(row.get("season") or ""),
        ))

    logger.info(
        "Player matching: %d matched, %d unmatched (not in StatsBomb data)",
        matched, unmatched,
    )

    if injury_rows:
        with conn.cursor() as cur:
            execute_values(cur, """
                INSERT INTO injuries (
                    player_id, injury_type, injury_date,
                    return_date, matches_missed, season
                ) VALUES %s
                ON CONFLICT DO NOTHING
            """, injury_rows)
        conn.commit()
        logger.info("Inserted %d injury rows", len(injury_rows))

    _update_transfermarkt_ids(conn, df, tm_id_map, norm_map)

    logger.info("Injuries ingestion complete")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    run()