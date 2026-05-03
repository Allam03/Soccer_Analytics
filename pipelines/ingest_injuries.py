"""
pipelines/ingest_injuries.py

Load Transfermarkt data from two CSVs into the database.

Matching strategy
-----------------
Priority 1 -- TM ID already in DB (instant, used on reruns)
Priority 2 -- Exact norm_name match (zero fuzzy cost, handles clean names)
Priority 3 -- Token-subset match (handles middle-name mismatches)
Priority 4 -- Blocking + RapidFuzz (handles transliteration / umlaut variants)

    Blocking: candidates are restricted to DB players whose norm_name starts
    with the same first character as the TM last name.  This reduces the
    comparison set from ~2084 to ~50-150 players per lookup.

    Scoring (both conditions must pass):
      last_score  = fuzz.ratio(tm_last_nn, db_last_token)          >= LAST_THRESHOLD
      full_score  = fuzz.token_set_ratio(tm_full_nn, db_full_nn)   >= FULL_THRESHOLD

    This catches "Mæhle" (DB: "mhle") vs "Maehle" (TM: "maehle") because
    fuzz.ratio("mhle", "maehle") ≈ 80 and token_set_ratio passes too.

Thresholds (tunable in settings.py or overridden at call time):
    LAST_THRESHOLD = 80
    FULL_THRESHOLD = 85
"""

import logging
from pathlib import Path

import pandas as pd
from psycopg2.extras import execute_values

try:
    from rapidfuzz import fuzz as _fuzz
    _RAPIDFUZZ_AVAILABLE = True
except ImportError:
    _RAPIDFUZZ_AVAILABLE = False

from core.utils import norm_name
from load.postgres import connect
from config.settings import (
    DB_DSN, TRANSFERMARKT_CSV, TRANSFERMARKT_PLAYERS_CSV, OUTPUT_DIR
)

logger = logging.getLogger(__name__)

# Fuzzy match thresholds (0-100)
LAST_THRESHOLD = 80   # fuzz.ratio on last name token
FULL_THRESHOLD = 85   # fuzz.token_set_ratio on full normalised name


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
    The Kaggle players CSV uses 'player_id', not 'transfermarkt_player_id'.
    """
    for candidate in ("transfermarkt_player_id", "player_id", "tm_player_id", "id"):
        if candidate in df.columns:
            logger.info("  TM id column detected as '%s'", candidate)
            return candidate
    logger.error(
        "  Cannot find TM player id column. Columns present: %s", list(df.columns)
    )
    return None


# ---------------------------------------------------------------------------
# DB player map + fuzzy index
# ---------------------------------------------------------------------------

class PlayerIndex:
    """
    Pre-built lookup structures over the players table.

    Attributes
    ----------
    tm_id_map   : {str(transfermarkt_player_id) -> pg player_id}
    norm_map    : {norm_name                    -> pg player_id}
    token_map   : {frozenset(tokens)            -> pg player_id}
    blocks      : {first_char_of_last_token     -> [(norm_name, last_token, pg player_id)]}
                  Used by the fuzzy pass to avoid O(N) comparisons.
    """

    def __init__(self, conn):
        self.tm_id_map: dict[str, int]              = {}
        self.norm_map:  dict[str, int]              = {}
        self.token_map: dict[frozenset, int]        = {}
        self.blocks:    dict[str, list[tuple]]      = {}

        with conn.cursor() as cur:
            cur.execute(
                "SELECT player_id, transfermarkt_player_id, norm_name FROM players"
            )
            for pid, tmid, nn in cur.fetchall():
                if tmid:
                    self.tm_id_map[str(tmid)] = pid
                if not nn:
                    continue
                self.norm_map[nn] = pid
                tokens = nn.split()
                if not tokens:
                    continue
                self.token_map[frozenset(tokens)] = pid
                last = tokens[-1]
                block_key = last[0] if last else ""
                if block_key:
                    self.blocks.setdefault(block_key, []).append((nn, last, pid))

        logger.info(
            "  PlayerIndex: %d TM-linked, %d norm, %d token-sets, %d blocks",
            len(self.tm_id_map), len(self.norm_map),
            len(self.token_map), len(self.blocks),
        )

    def add(self, tm_id: str | None, pg_pid: int, nn: str):
        """Register a newly linked player so subsequent lookups in the same
        run can use Priority 1 (TM ID) immediately."""
        if tm_id:
            self.tm_id_map[tm_id] = pg_pid
        if nn:
            self.norm_map[nn] = pg_pid


# ---------------------------------------------------------------------------
# Candidate name generator
# ---------------------------------------------------------------------------

def _candidate_norms(row: pd.Series) -> list[str]:
    """
    Normalised name strings to try, most-specific first, deduplicated.
    """
    first = str(row.get("first_name") or "").strip()
    last  = str(row.get("last_name")  or "").strip()
    full  = str(row.get("full_name")  or "").strip()

    raw = [
        f"{first} {last}" if (first and last) else "",
        full,
        f"{first[0]} {last}" if (first and last) else "",
        last,
        full.split()[-1] if full else "",
    ]

    seen: list[str] = []
    seen_set: set[str] = set()
    for s in raw:
        nn = norm_name(s)
        if nn and nn not in seen_set:
            seen.append(nn)
            seen_set.add(nn)
    return seen


# ---------------------------------------------------------------------------
# Individual match strategies
# ---------------------------------------------------------------------------

def _exact_match(candidates: list[str], index: PlayerIndex) -> int | None:
    for nn in candidates:
        pid = index.norm_map.get(nn)
        if pid is not None:
            return pid
    return None


def _token_subset_match(candidates: list[str], index: PlayerIndex) -> int | None:
    for nn in candidates:
        cand_tokens = frozenset(nn.split())
        if not cand_tokens:
            continue
        matches = [
            pid for db_tokens, pid in index.token_map.items()
            if cand_tokens.issubset(db_tokens)
        ]
        if len(matches) == 1:
            return matches[0]
    return None


def _fuzzy_match(
    row: pd.Series,
    candidates: list[str],
    index: PlayerIndex,
    last_threshold: int,
    full_threshold: int,
) -> int | None:
    """
    Block on first character of TM last name, then score with RapidFuzz.

    Scoring
    -------
    Gate 1 -- last_score: max(ratio(tm_last, token) for token in db_name.split())
              Scores TM last name against every token in the DB name and
              takes the best.  This correctly finds "messi" inside
              "lionel andres messi cuccittini" rather than comparing only
              against the final token "cuccittini".

    Gate 2 -- full_score: token_set_ratio(tm_first_last, db_full_name)
              Handles middle-name mismatches and word-order differences.

    Both gates must pass.  Best combined score wins.
    """
    if not _RAPIDFUZZ_AVAILABLE:
        return None

    last_raw = str(row.get("last_name") or "").strip()
    if not last_raw:
        return None

    last_nn   = norm_name(last_raw)
    block_key = last_nn[0] if last_nn else ""
    block     = index.blocks.get(block_key, [])
    if not block:
        return None

    # Use first+last as the full candidate for token_set_ratio
    full_nn = candidates[0] if candidates else ""

    best_score = -1
    best_pid   = None

    for db_nn, _db_last_unused, pid in block:
        # Gate 1: best match of TM last name against ANY token in DB name
        last_score = max(_fuzz.ratio(last_nn, tok) for tok in db_nn.split())
        if last_score < last_threshold:
            continue

        # Gate 2: full-name token set similarity
        full_score = _fuzz.token_set_ratio(full_nn, db_nn)
        if full_score < full_threshold:
            continue

        combined = (last_score + full_score) / 2
        if combined > best_score:
            best_score = combined
            best_pid   = pid

    return best_pid


# ---------------------------------------------------------------------------
# Master resolver
# ---------------------------------------------------------------------------

def _resolve(
    row: pd.Series,
    index: PlayerIndex,
    last_threshold: int,
    full_threshold: int,
) -> tuple[int | None, str]:
    """
    Return (pg_player_id, method_label) or (None, 'no_match').
    """
    # Priority 1: TM ID already in DB
    tm_id = _to_tm_id(row.get("transfermarkt_player_id"))
    if tm_id and tm_id in index.tm_id_map:
        return index.tm_id_map[tm_id], "tm_id"

    candidates = _candidate_norms(row)

    # Priority 2: exact norm match
    pid = _exact_match(candidates, index)
    if pid is not None:
        return pid, "exact"

    # Priority 3: token-subset match
    pid = _token_subset_match(candidates, index)
    if pid is not None:
        return pid, "token_subset"

    # Priority 4: blocking + fuzzy
    pid = _fuzzy_match(row, candidates, index, last_threshold, full_threshold)
    if pid is not None:
        return pid, "fuzzy"

    return None, "no_match"


# ---------------------------------------------------------------------------
# Pass 1: transfermarkt_players.csv -> players table
# ---------------------------------------------------------------------------

def ingest_players(
    conn,
    players_csv: str,
    last_threshold: int = LAST_THRESHOLD,
    full_threshold: int = FULL_THRESHOLD,
) -> dict[str, int]:
    """
    Match TM players to the StatsBomb players table and backfill
    transfermarkt_player_id, date_of_birth, nationality, position.

    Returns tm_id_map {str(tm_id) -> pg player_id} for Pass 2.
    """
    if not _RAPIDFUZZ_AVAILABLE:
        logger.warning(
            "rapidfuzz not installed -- fuzzy matching disabled. "
            "Run: pip install rapidfuzz"
        )

    logger.info("Pass 1: loading %s", players_csv)
    df = pd.read_csv(players_csv, low_memory=False)
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
    df["games_missed"] = pd.to_numeric(df.get("games_missed", pd.Series(dtype=float)), errors="coerce")
    logger.info("  %d rows | columns: %s", len(df), list(df.columns))

    tm_id_col = _detect_tm_id_col(df)
    if tm_id_col is None:
        return {}
    if tm_id_col != "transfermarkt_player_id":
        df = df.rename(columns={tm_id_col: "transfermarkt_player_id"})

    if "date_of_birth" in df.columns:
        df["date_of_birth"] = pd.to_datetime(
            df["date_of_birth"], errors="coerce"
        ).dt.date

    index = PlayerIndex(conn)

    # Diagnostics
    logger.info("  Sample DB norm_names: %s", list(index.norm_map.keys())[:5])
    name_cols = [c for c in ("full_name", "first_name", "last_name") if c in df.columns]
    logger.info("  Sample TM names: %s", df[name_cols].head(3).to_dict("records"))
    logger.info(
        "  Fuzzy thresholds: last>=%d  full_token_set>=%d",
        last_threshold, full_threshold,
    )

    linked   = 0
    no_match = 0
    by_method: dict[str, int] = {}
    updates  = []

    for _, row in df.iterrows():
        pid, method = _resolve(row, index, last_threshold, full_threshold)
        if pid is None:
            no_match += 1
            continue

        by_method[method] = by_method.get(method, 0) + 1

        tm_id = _to_tm_id(row.get("transfermarkt_player_id"))
        dob   = row.get("date_of_birth") if pd.notna(row.get("date_of_birth")) else None
        nat   = (
            str(row.get("country_of_citizenship") or "").strip()
            or str(row.get("country_of_birth") or "").strip()
            or None
        ) or None
        pos   = (
            str(row.get("sub_position") or "").strip()
            or str(row.get("position")  or "").strip()
            or None
        ) or None

        updates.append((tm_id, dob, nat, pos, pid))
        index.add(tm_id, pid, norm_name(
            f"{row.get('first_name') or ''} {row.get('last_name') or ''}".strip()
        ))
        linked += 1

    logger.info(
        "  Pass 1 result: %d linked, %d unmatched | by method: %s",
        linked, no_match, by_method,
    )

    if not updates:
        logger.warning("  0 players matched -- check CSV paths and column names")
        return index.tm_id_map

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

    logger.info("  tm_id_map size after Pass 1: %d", len(index.tm_id_map))
    return index.tm_id_map


# ---------------------------------------------------------------------------
# Pass 2: transfermarkt_injuries.csv -> injuries table
# ---------------------------------------------------------------------------

def ingest_injuries(conn, injuries_csv: str, tm_id_map: dict[str, int]):
    """
    Insert injury records using tm_id_map (direct integer lookup).
    """
    logger.info("Pass 2: loading %s", injuries_csv)
    df = pd.read_csv(injuries_csv, low_memory=False)
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
    logger.info("  %d rows | columns: %s", len(df), list(df.columns))

    inj_id_col = _detect_tm_id_col(df)
    if inj_id_col is None:
        logger.error("  Cannot find player id column in injuries CSV")
        return
    if inj_id_col != "player_id":
        df = df.rename(columns={inj_id_col: "player_id"})

    logger.info(
        "  Sample injury player_id values: %s (dtype=%s)",
        df["player_id"].dropna().head(5).tolist(), df["player_id"].dtype,
    )
    logger.info(
        "  Sample tm_id_map keys: %s (size=%d)",
        list(tm_id_map.keys())[:5], len(tm_id_map),
    )

    if not tm_id_map:
        logger.error(
            "  tm_id_map is empty -- Pass 1 matched 0 players. "
            "Check that the players CSV has a numeric player id column."
        )
        return

    for col in ("date_from", "date_until"):
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce").dt.date

    df["games_missed"] = pd.to_numeric(
        df.get("games_missed", pd.Series(dtype=float)), errors="coerce"
    )

    injury_rows: list[tuple]  = []
    unmatched_rows: list[dict] = []
    matched   = 0
    unmatched = 0

    for _, row in df.iterrows():
        tm_id = _to_tm_id(row.get("player_id"))
        if tm_id is None:
            unmatched += 1
            unmatched_rows.append({**row.to_dict(), "_reason": "no_tm_id"})
            continue

        internal_pid = tm_id_map.get(tm_id)
        if internal_pid is None:
            unmatched += 1
            unmatched_rows.append({**row.to_dict(), "_reason": "not_in_statsbomb"})
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

    # Export unmatched rows split by reason
    if unmatched_rows:
        out_dir = Path(OUTPUT_DIR)
        out_dir.mkdir(parents=True, exist_ok=True)
        unmatched_df = pd.DataFrame(unmatched_rows)

        for reason, label in [
            ("not_in_statsbomb", "injuries_unmatched_not_in_statsbomb.csv"),
            ("no_tm_id",         "injuries_unmatched_no_tm_id.csv"),
        ]:
            subset = unmatched_df[unmatched_df["_reason"] == reason].drop(
                columns=["_reason"]
            )
            if subset.empty:
                continue
            out_path = out_dir / label
            subset.to_csv(out_path, index=False)
            logger.info(
                "  Unmatched export (%s): %d rows -> %s",
                reason, len(subset), out_path,
            )


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def run(
    conn=None,
    players_csv: str = None,
    injuries_csv: str = None,
    last_threshold: int = LAST_THRESHOLD,
    full_threshold: int = FULL_THRESHOLD,
):
    if conn is None:
        conn = connect(DB_DSN)
    if players_csv is None:
        players_csv = TRANSFERMARKT_PLAYERS_CSV
    if injuries_csv is None:
        injuries_csv = TRANSFERMARKT_CSV

    tm_id_map = ingest_players(
        conn, players_csv,
        last_threshold=last_threshold,
        full_threshold=full_threshold,
    )
    ingest_injuries(conn, injuries_csv, tm_id_map)
    logger.info("Injuries ingestion complete")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    run()