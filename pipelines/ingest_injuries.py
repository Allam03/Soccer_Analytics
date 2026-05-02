"""
pipelines/ingest_injuries.py

Load Transfermarkt data from two CSVs into the database.

Pass 1 -- transfermarkt_players.csv
    For every Transfermarkt player, try to match them to a row in the
    players table (via norm_name fuzzy match).  When a match is found:
      - Write players.transfermarkt_player_id
      - Write players.date_of_birth
      - Write players.nationality  (country_of_citizenship) if blank
      - Write players.position     if blank

Pass 2 -- transfermarkt_injuries.csv
    For every injury row, look up the internal player_id using
    transfermarkt_player_id (now populated by Pass 1 -- no name matching
    needed at this stage).  Insert into the injuries table.

Matching strategy for Pass 1
-----------------------------
Priority 1: players.transfermarkt_player_id already set and matches.
Priority 2: norm_name(full_name) matches players.norm_name.
Priority 3: norm_name(first_name + ' ' + last_name) matches players.norm_name.

The players CSV uses `transfermarkt_player_id` as its primary key, which is
the same ID that appears as `player_id` in the injuries CSV -- so after
Pass 1 the injuries join is a simple integer lookup with no name matching.
"""

import logging

import pandas as pd
from psycopg2.extras import execute_values

from core.utils import norm_name
from load.postgres import connect
from config.settings import DB_DSN, TRANSFERMARKT_CSV, TRANSFERMARKT_PLAYERS_CSV

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_sb_player_map(conn) -> tuple[dict, dict]:
    """
    Load the current state of the players table.

    Returns
    -------
    tm_id_map : {transfermarkt_player_id (str) -> internal player_id}
    norm_map  : {norm_name (str)               -> internal player_id}
    """
    tm_id_map: dict[str, int] = {}
    norm_map:  dict[str, int] = {}
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


def _resolve_tm_player(row: pd.Series, tm_id_map: dict, norm_map: dict) -> int | None:
    """
    Try to find the internal player_id for a transfermarkt_players row.

    Tries (in order):
      1. transfermarkt_player_id already linked in DB
      2. norm_name(full_name)
      3. norm_name(first_name + last_name)
    """
    raw = row.get("transfermarkt_player_id")
    tm_id = str(int(raw)) if pd.notna(raw) else ""
    if tm_id and tm_id in tm_id_map:
        return tm_id_map[tm_id]

    # Try full_name
    full = str(row.get("full_name") or "").strip()
    if full:
        nn = norm_name(full)
        if nn in norm_map:
            return norm_map[nn]

    # Try first + last
    first = str(row.get("first_name") or "").strip()
    last  = str(row.get("last_name")  or "").strip()
    if first or last:
        nn = norm_name(f"{first} {last}")
        if nn in norm_map:
            return norm_map[nn]

    return None


# ---------------------------------------------------------------------------
# Pass 1: transfermarkt_players.csv -> players table
# ---------------------------------------------------------------------------

def ingest_players(conn, players_csv: str) -> dict[str, int]:
    """
    Match Transfermarkt players to the StatsBomb players table and back-fill
    transfermarkt_player_id, date_of_birth, nationality, and position.

    Returns
    -------
    tm_id_map : {transfermarkt_player_id (str) -> internal player_id}
                The complete map after updates, used by Pass 2.
    """
    logger.info("Pass 1: loading Transfermarkt players from: %s", players_csv)
    df = pd.read_csv(players_csv, low_memory=False)
    logger.info("  %d rows in players CSV", len(df))

    # Normalise column names to lowercase-underscore
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]

    # Parse date_of_birth; keep as Python date
    if "date_of_birth" in df.columns:
        df["date_of_birth"] = pd.to_datetime(df["date_of_birth"], errors="coerce").dt.date

    tm_id_map, norm_map = _load_sb_player_map(conn)

    linked   = 0
    no_match = 0
    updates  = []   # (tm_id, dob, nationality, position, internal_player_id)

    for _, row in df.iterrows():
        internal_pid = _resolve_tm_player(row, tm_id_map, norm_map)
        if internal_pid is None:
            no_match += 1
            continue

        raw = row.get("transfermarkt_player_id")
        tm_id = str(int(raw)) if pd.notna(raw) else None
        dob   = row.get("date_of_birth") if pd.notna(row.get("date_of_birth")) else None

        # Use country_of_citizenship as nationality; fall back to country_of_birth
        nat = (
            str(row.get("country_of_citizenship") or "").strip()
            or str(row.get("country_of_birth") or "").strip()
            or None
        )
        if not nat:
            nat = None

        # Prefer sub_position (e.g. "Centre-Forward") over position ("Attack")
        pos = (
            str(row.get("sub_position") or "").strip()
            or str(row.get("position")  or "").strip()
            or None
        )
        if not pos:
            pos = None

        updates.append((tm_id, dob, nat, pos, internal_pid))

        # Keep tm_id_map current so Pass 2 can use it without a second DB read
        if tm_id:
            tm_id_map[tm_id] = internal_pid

        linked += 1

    logger.info("  Player matching: %d linked, %d not in StatsBomb data", linked, no_match)

    if updates:
        with conn.cursor() as cur:
            for tm_id, dob, nat, pos, pid in updates:
                cur.execute("""
                    UPDATE players
                    SET
                        transfermarkt_player_id = COALESCE(transfermarkt_player_id, %s),
                        date_of_birth           = COALESCE(date_of_birth,           %s),
                        nationality             = COALESCE(nationality,             %s),
                        position                = COALESCE(position,                %s)
                    WHERE player_id = %s
                """, (tm_id, dob, nat, pos, pid))
        conn.commit()
        logger.info("  Updated %d player rows (DOB, TM ID, nationality, position)", len(updates))
    else:
        logger.warning(
            "  No players matched -- check that ingest_statsbomb.py has run first "
            "and that the players CSV path is correct"
        )

    return tm_id_map


# ---------------------------------------------------------------------------
# Pass 2: transfermarkt_injuries.csv -> injuries table
# ---------------------------------------------------------------------------

def ingest_injuries(conn, injuries_csv: str, tm_id_map: dict[str, int]):
    """
    Insert injury records using the tm_id_map built by Pass 1.

    Every lookup is a direct integer match on transfermarkt_player_id --
    no name fuzzy matching needed here.

    injuries CSV columns (Transfermarkt schema)
    -------------------------------------------
    player_id     : Transfermarkt player ID  (= transfermarkt_player_id in players CSV)
    player_name   : string  (logged only for unmatched rows)
    season        : e.g. "2019"
    injury        : injury description string
    date_from     : injury start date
    date_until    : return date
    duration_days : int
    games_missed  : int
    """
    logger.info("Pass 2: loading Transfermarkt injuries from: %s", injuries_csv)
    df = pd.read_csv(injuries_csv, low_memory=False)
    logger.info("  %d rows in injuries CSV", len(df))

    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]

    for col in ("date_from", "date_until"):
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce").dt.date

    injury_rows = []
    matched     = 0
    unmatched   = 0

    for _, row in df.iterrows():
        raw_tm_id = row.get("player_id")
        tm_id     = str(int(raw_tm_id)) if pd.notna(raw_tm_id) else ""

        internal_pid = tm_id_map.get(tm_id)
        if internal_pid is None:
            # Player is in Transfermarkt but not in StatsBomb -- expected for
            # players who never appeared in our in-scope competitions.
            unmatched += 1
            continue

        matched += 1
        injury_rows.append((
            internal_pid,
            str(row.get("injury") or "").strip() or None,
            row.get("date_from")  if pd.notna(row.get("date_from"))  else None,
            row.get("date_until") if pd.notna(row.get("date_until")) else None,
            int(row["games_missed"]) if pd.notna(row.get("games_missed")) else None,
            str(row.get("season") or "").strip() or None,
        ))

    logger.info(
        "  Injury matching: %d matched, %d unmatched "
        "(player not in StatsBomb in-scope competitions)",
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
        logger.info("  Inserted %d injury rows", len(injury_rows))
    else:
        logger.warning(
            "  No injury rows inserted -- verify that Pass 1 linked players correctly"
        )


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def run(
    conn=None,
    players_csv: str = None,
    injuries_csv: str = None,
):
    """
    Run both passes in order.

    Parameters
    ----------
    conn          : psycopg2 connection  (created from DB_DSN if None)
    players_csv   : path to transfermarkt_players.csv
    injuries_csv  : path to transfermarkt_injuries.csv
    """
    if conn is None:
        conn = connect(DB_DSN)
    if players_csv is None:
        players_csv = TRANSFERMARKT_PLAYERS_CSV
    if injuries_csv is None:
        injuries_csv = TRANSFERMARKT_CSV

    # Pass 1: match TM players to StatsBomb players, backfill DOB + metadata
    tm_id_map = ingest_players(conn, players_csv)

    # Pass 2: insert injury records -- direct ID lookup, no name matching
    ingest_injuries(conn, injuries_csv, tm_id_map)

    logger.info("Injuries ingestion complete")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    run()
