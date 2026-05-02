"""
pipelines/ingest_injuries.py

Load Transfermarkt data from two CSVs into the database.

Column name fix
---------------
The Kaggle transfermarkt_players.csv uses 'player_id' as its id column,
NOT 'transfermarkt_player_id'.  _detect_tm_id_col() detects the actual
column name at load time and renames it to 'transfermarkt_player_id' so
all downstream code uses one consistent name.

Without this fix:
  row.get("transfermarkt_player_id") returns None for every row
  -> tm_id is always None
  -> nothing added to tm_id_map
  -> Pass 2 sees an empty map and inserts 0 injuries

Matching strategy (priority order)
------------------------------------
1. transfermarkt_player_id already set in DB (reruns)
2. norm_name(first + last)          e.g. "lionel messi"
3. norm_name(full_name)
4. norm_name(first_initial + last)  e.g. "l messi"
5. last name only, unambiguous      (exactly 1 SB player with that surname)
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

def _to_tm_id(raw) -> str | None:
    """Safely convert a raw CSV player id value to a clean string integer."""
    if not pd.notna(raw):
        return None
    try:
        return str(int(float(raw)))
    except (ValueError, TypeError):
        return None


def _detect_tm_id_col(df: pd.DataFrame) -> str | None:
    """
    Return the column that holds the Transfermarkt integer player id.

    The Kaggle players CSV calls this column 'player_id', not
    'transfermarkt_player_id'.  We try several candidate names in order.
    """
    for candidate in ("transfermarkt_player_id", "player_id", "tm_player_id", "id"):
        if candidate in df.columns:
            logger.info("  TM id column detected as '%s'", candidate)
            return candidate
    logger.error(
        "  Cannot find TM player id column. Columns present: %s", list(df.columns)
    )
    return None


def _load_sb_player_map(conn) -> tuple[dict, dict, dict]:
    """
    Returns
    -------
    tm_id_map   : {str(transfermarkt_player_id) -> internal player_id}
    norm_map    : {norm_name                    -> internal player_id}
    last_nm_map : {last token of norm_name      -> [internal player_id]}
    """
    tm_id_map:   dict[str, int]       = {}
    norm_map:    dict[str, int]       = {}
    last_nm_map: dict[str, list[int]] = {}

    with conn.cursor() as cur:
        cur.execute(
            "SELECT player_id, transfermarkt_player_id, norm_name FROM players"
        )
        for pid, tmid, nn in cur.fetchall():
            if tmid:
                tm_id_map[str(tmid)] = pid
            if nn:
                norm_map[nn] = pid
                last = nn.split()[-1] if nn.split() else ""
                if last:
                    last_nm_map.setdefault(last, []).append(pid)

    logger.info(
        "  DB map: %d with TM id, %d norm entries, %d surname entries",
        len(tm_id_map), len(norm_map), len(last_nm_map),
    )
    return tm_id_map, norm_map, last_nm_map


def _candidate_norms(row: pd.Series) -> list[str]:
    """
    Return normalised name strings to try, most-specific first,
    deduplicated while preserving order.
    """
    first = str(row.get("first_name") or "").strip()
    last  = str(row.get("last_name")  or "").strip()
    full  = str(row.get("full_name")  or "").strip()

    raw = [
        f"{first} {last}" if (first and last) else "",
        full,
        f"{first[0]} {last}" if (first and last) else "",
        last,
        full.strip().split()[-1] if full else "",
    ]

    seen: list[str] = []
    seen_set: set[str] = set()
    for s in raw:
        nn = norm_name(s)
        if nn and nn not in seen_set:
            seen.append(nn)
            seen_set.add(nn)
    return seen


def _resolve_tm_player(
    row: pd.Series,
    tm_id_map: dict,
    norm_map: dict,
    last_nm_map: dict,
) -> tuple[int | None, str]:
    """Return (internal_player_id, match_method) or (None, 'no_match')."""
    # Priority 1: TM ID already in DB
    tm_id = _to_tm_id(row.get("transfermarkt_player_id"))
    if tm_id and tm_id in tm_id_map:
        return tm_id_map[tm_id], "tm_id"

    # Priority 2-4: normalised name candidates
    for nn in _candidate_norms(row):
        if nn in norm_map:
            return norm_map[nn], f"norm:{nn}"

    # Priority 5: unambiguous surname match
    last = str(row.get("last_name") or "").strip()
    if last:
        matches = last_nm_map.get(norm_name(last), [])
        if len(matches) == 1:
            return matches[0], f"surname_only:{norm_name(last)}"

    return None, "no_match"


# ---------------------------------------------------------------------------
# Pass 1: transfermarkt_players.csv -> players table
# ---------------------------------------------------------------------------

def ingest_players(conn, players_csv: str) -> dict[str, int]:
    """
    Match TM players to the StatsBomb players table and backfill
    transfermarkt_player_id, date_of_birth, nationality, position.

    Returns tm_id_map {str(tm_id) -> internal player_id} for Pass 2.
    """
    logger.info("Pass 1: loading %s", players_csv)
    df = pd.read_csv(players_csv, low_memory=False)
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
    logger.info("  %d rows | columns: %s", len(df), list(df.columns))

    # Detect and standardise the TM id column name
    tm_id_col = _detect_tm_id_col(df)
    if tm_id_col is None:
        return {}
    if tm_id_col != "transfermarkt_player_id":
        df = df.rename(columns={tm_id_col: "transfermarkt_player_id"})

    if "date_of_birth" in df.columns:
        df["date_of_birth"] = pd.to_datetime(
            df["date_of_birth"], errors="coerce"
        ).dt.date

    tm_id_map, norm_map, last_nm_map = _load_sb_player_map(conn)

    # Diagnostics
    logger.info("  Sample DB norm_names: %s", list(norm_map.keys())[:5])
    name_cols = [c for c in ("full_name", "first_name", "last_name") if c in df.columns]
    logger.info(
        "  Sample TM names: %s", df[name_cols].head(3).to_dict("records")
    )
    logger.info(
        "  Sample TM ids (raw): %s",
        df["transfermarkt_player_id"].dropna().head(5).tolist(),
    )

    linked   = 0
    no_match = 0
    by_method: dict[str, int] = {}
    updates  = []

    for _, row in df.iterrows():
        internal_pid, method = _resolve_tm_player(
            row, tm_id_map, norm_map, last_nm_map
        )
        if internal_pid is None:
            no_match += 1
            continue

        bucket = method.split(":")[0]
        by_method[bucket] = by_method.get(bucket, 0) + 1

        tm_id = _to_tm_id(row.get("transfermarkt_player_id"))
        dob   = row.get("date_of_birth") if pd.notna(row.get("date_of_birth")) else None

        nat = (
            str(row.get("country_of_citizenship") or "").strip()
            or str(row.get("country_of_birth") or "").strip()
            or None
        ) or None

        pos = (
            str(row.get("sub_position") or "").strip()
            or str(row.get("position")  or "").strip()
            or None
        ) or None

        updates.append((tm_id, dob, nat, pos, internal_pid))

        # Always add to in-memory map so Pass 2 can use it
        if tm_id:
            tm_id_map[tm_id] = internal_pid

        linked += 1

    logger.info(
        "  Pass 1 result: %d linked, %d unmatched | by method: %s",
        linked, no_match, by_method,
    )

    if not updates:
        logger.warning("  0 players matched -- check CSV paths and column names above")
        return tm_id_map

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
    logger.info("  Updated %d player rows", len(updates))

    with conn.cursor() as cur:
        cur.execute(
            "SELECT COUNT(*) FROM players WHERE transfermarkt_player_id IS NOT NULL"
        )
        logger.info(
            "  DB now has %d players with transfermarkt_player_id set",
            cur.fetchone()[0],
        )

    logger.info("  tm_id_map size after Pass 1: %d", len(tm_id_map))
    return tm_id_map


# ---------------------------------------------------------------------------
# Pass 2: transfermarkt_injuries.csv -> injuries table
# ---------------------------------------------------------------------------

def ingest_injuries(conn, injuries_csv: str, tm_id_map: dict[str, int]):
    """
    Insert injury records using tm_id_map (direct integer lookup).

    injuries CSV expected columns (after lowercasing):
      player_id, player_name, season, injury,
      date_from, date_until, duration_days, games_missed
    """
    logger.info("Pass 2: loading %s", injuries_csv)
    df = pd.read_csv(injuries_csv, low_memory=False)
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
    df["games_missed"] = pd.to_numeric(df["games_missed"], errors="coerce")
    logger.info("  %d rows | columns: %s", len(df), list(df.columns))

    # The injuries CSV player id column may also need detection
    inj_id_col = _detect_tm_id_col(df)
    if inj_id_col is None:
        logger.error("  Cannot find player id column in injuries CSV")
        return
    if inj_id_col != "player_id":
        df = df.rename(columns={inj_id_col: "player_id"})

    sample_ids = df["player_id"].dropna().head(5).tolist()
    logger.info(
        "  Sample injury player_id values: %s (dtype=%s)",
        sample_ids, df["player_id"].dtype,
    )
    logger.info(
        "  Sample tm_id_map keys: %s (size=%d)",
        list(tm_id_map.keys())[:5], len(tm_id_map),
    )

    if not tm_id_map:
        logger.error(
            "  tm_id_map is empty -- Pass 1 found 0 TM ids. "
            "Check that the players CSV has a numeric player id column."
        )
        return

    for col in ("date_from", "date_until"):
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce").dt.date

    injury_rows = []
    matched   = 0
    unmatched = 0

    for _, row in df.iterrows():
        tm_id = _to_tm_id(row.get("player_id"))
        if tm_id is None:
            unmatched += 1
            continue

        internal_pid = tm_id_map.get(tm_id)
        if internal_pid is None:
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

    logger.info("  Injury matching: %d matched, %d unmatched", matched, unmatched)

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
        logger.warning("  No injury rows inserted")


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def run(conn=None, players_csv: str = None, injuries_csv: str = None):
    if conn is None:
        conn = connect(DB_DSN)
    if players_csv is None:
        players_csv = TRANSFERMARKT_PLAYERS_CSV
    if injuries_csv is None:
        injuries_csv = TRANSFERMARKT_CSV

    tm_id_map = ingest_players(conn, players_csv)
    ingest_injuries(conn, injuries_csv, tm_id_map)
    logger.info("Injuries ingestion complete")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    run()