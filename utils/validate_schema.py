"""
utils/validate_schema.py

Validate that the live PostgreSQL schema matches the expected DDL.
"""

import logging
from typing import List, Tuple

logger = logging.getLogger("schema")
logger.setLevel(logging.WARNING)

# PostgreSQL reports TEXT columns as 'text', not 'varchar'.
# FLOAT columns are reported as 'double precision'.
# INT columns are 'integer'.
_TYPE_ALIASES = {
    "varchar":          "text",
    "character varying":"text",
    "int4":             "integer",
    "int8":             "bigint",
    "float4":           "real",
    "float8":           "double precision",
    "numeric":          "double precision",  # treat as compatible
    "bool":             "boolean",
}


def _normalise_type(t: str) -> str:
    return _TYPE_ALIASES.get(t.lower(), t.lower())


def validate_schema(conn) -> Tuple[bool, List[str], List[str]]:
    """
    Validate that all expected tables and columns exist with compatible types.

    Returns
    -------
    (success, warnings, errors)
    """
    warnings: List[str] = []
    errors:   List[str] = []

    expected = {
        "teams": {
            "team_id":           "integer",
            "team_name":         "text",
            "statsbomb_team_id": "integer",
        },
        "players": {
            "player_id":               "integer",
            "player_name":             "text",
            "norm_name":               "text",
            "statsbomb_player_id":     "integer",
            "transfermarkt_player_id": "text",
            "date_of_birth":           "date",
        },
        "matches": {
            "match_id":           "integer",
            "statsbomb_match_id": "integer",
            "match_date":         "date",
            "home_team_id":       "integer",
            "away_team_id":       "integer",
            "home_score":         "integer",
            "away_score":         "integer",
            "competition":        "text",
            "season":             "text",
            "stadium_name":       "text",
            "stadium_lat":        "double precision",
            "stadium_lng":        "double precision",
        },
        "weather": {
            "weather_id":      "integer",
            "match_id":        "integer",
            "temperature_c":   "double precision",
            "humidity_pct":    "double precision",
            "wind_speed_kmh":  "double precision",
            "precipitation_mm":"double precision",
        },
        "injuries": {
            "injury_id":    "integer",
            "player_id":    "integer",
            "injury_date":  "date",
            "return_date":  "date",
        },
        "player_match_stats": {
            "stat_id":              "integer",
            "player_id":            "integer",
            "match_id":             "integer",
            "team_id":              "integer",
            "weather_id":           "integer",
            "goals":                "integer",
            "assists":              "integer",
            "shots":                "integer",
            "xg":                   "double precision",
            "xa":                   "double precision",
            "key_passes":           "integer",
            "passes_attempted":     "integer",
            "passes_completed":     "integer",
            "pass_accuracy":        "double precision",
            "progressive_passes":   "integer",
            "carry_distance":       "double precision",
            "progressive_carries":  "integer",
            "dribbles_completed":   "integer",
            "tackles":              "integer",
            "interceptions":        "integer",
            "clearances":           "integer",
            "pressures":            "integer",
            "yellow_cards":         "integer",
            "red_cards":            "integer",
            "minutes_played":       "integer",
            "sub_minute":           "integer",
            "days_since_last_injury":"integer",
            "matches_last_30_days": "integer",
            "minutes_last_30_days": "integer",
            "is_injured_next_30d":  "boolean",
        },
        "pass_network_edges": {
            "edge_id":    "integer",
            "match_id":   "integer",
            "team_id":    "integer",
            "passer_id":  "integer",
            "receiver_id":"integer",
            "pass_count": "integer",
            "avg_x_start":"double precision",
            "avg_y_start":"double precision",
            "avg_x_end":  "double precision",
            "avg_y_end":  "double precision",
        },
    }

    try:
        with conn.cursor() as cur:
            for table, cols in expected.items():
                cur.execute("""
                    SELECT column_name, data_type
                    FROM information_schema.columns
                    WHERE table_schema = 'public'
                      AND table_name   = %s
                """, (table,))
                existing = {row[0]: _normalise_type(row[1]) for row in cur.fetchall()}

                if not existing:
                    errors.append(f"Table '{table}' does not exist")
                    continue

                for col, exp_type in cols.items():
                    if col not in existing:
                        errors.append(f"{table}.{col} is missing")
                    elif existing[col] != _normalise_type(exp_type):
                        warnings.append(
                            f"{table}.{col}: expected {exp_type}, "
                            f"got {existing[col]}"
                        )

            # Check for required unique constraint on player_match_stats
            cur.execute("""
                SELECT COUNT(*) FROM information_schema.table_constraints
                WHERE table_name = 'player_match_stats'
                  AND constraint_type = 'UNIQUE'
            """)
            if cur.fetchone()[0] == 0:
                errors.append("player_match_stats missing UNIQUE constraint")

            # Orphan checks
            for fk_table, fk_col, ref_table, ref_col in [
                ("player_match_stats", "player_id", "players",  "player_id"),
                ("player_match_stats", "match_id",  "matches",  "match_id"),
                ("player_match_stats", "team_id",   "teams",    "team_id"),
                ("injuries",           "player_id", "players",  "player_id"),
            ]:
                cur.execute(f"""
                    SELECT COUNT(*) FROM {fk_table} f
                    LEFT JOIN {ref_table} r ON r.{ref_col} = f.{fk_col}
                    WHERE f.{fk_col} IS NOT NULL AND r.{ref_col} IS NULL
                """)
                orphans = cur.fetchone()[0]
                if orphans:
                    warnings.append(
                        f"{fk_table}.{fk_col}: {orphans} orphaned rows "
                        f"(no matching {ref_table}.{ref_col})"
                    )

        success = len(errors) == 0
        level   = logging.WARNING if not success else logging.INFO
        logger.log(level, "Schema validation: %s", "PASSED" if success else "FAILED")
        for w in warnings:
            logger.warning("  Warning: %s", w)
        for e in errors:
            logger.error("  Error: %s", e)

        return success, warnings, errors

    except Exception as exc:
        logger.error("Schema validation raised an exception: %s", exc)
        return False, [], [str(exc)]