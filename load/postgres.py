"""
load/postgres.py

Database write helpers.  All column references use the renamed schema:
  sb_match_id, sb_team_id, sb_player_id  (source identifiers)
  match_id, team_id, player_id           (internal surrogate PKs)

Schema changes reflected here:
  - matches no longer has stadium_name / stadium_lat / stadium_lng;
    these live in the stadiums table, referenced via stadium_id.
  - upsert_stadium() inserts/returns a stadium_id.
  - upsert_match() accepts stadium_id instead of stadium_name/lat/lng.
  - insert_features() bulk-upserts player_match_features rows (computed
    ML columns that were previously part of player_match_stats).
  - pass_network_edges now has a true UNIQUE constraint, so
    ON CONFLICT DO NOTHING is sufficient and correct.
"""

import psycopg2
from psycopg2.extras import execute_values


def connect(dsn: str):
    return psycopg2.connect(dsn)


# ---------------------------------------------------------------------------
# Stadiums
# ---------------------------------------------------------------------------

def upsert_stadium(conn, stadium_name: str,
                   lat: float | None = None,
                   lng: float | None = None) -> int:
    """
    Insert or update a stadium row.  Returns the internal stadium_id.
    Does NOT commit.

    If lat/lng are supplied they fill NULL columns but never overwrite
    existing coordinate data (manual data takes precedence).
    """
    with conn.cursor() as cur:
        cur.execute("""
            INSERT INTO stadiums (stadium_name, stadium_lat, stadium_lng)
            VALUES (%s, %s, %s)
            ON CONFLICT (stadium_name) DO UPDATE
                SET stadium_lat = CASE
                        WHEN stadiums.stadium_lat IS NULL THEN EXCLUDED.stadium_lat
                        ELSE stadiums.stadium_lat
                    END,
                    stadium_lng = CASE
                        WHEN stadiums.stadium_lng IS NULL THEN EXCLUDED.stadium_lng
                        ELSE stadiums.stadium_lng
                    END
            RETURNING stadium_id
        """, (stadium_name, lat, lng))
        return cur.fetchone()[0]


# ---------------------------------------------------------------------------
# Matches
# ---------------------------------------------------------------------------

def upsert_match(conn, row: dict) -> int:
    """
    Insert or update one match row.  Returns the internal match_id.
    Does NOT commit.

    Expected keys
    -------------
    sb_match_id, match_date, home_team_id, away_team_id,
    home_score, away_score, competition, season, stadium_id
    """
    with conn.cursor() as cur:
        cur.execute("""
            INSERT INTO matches (
                sb_match_id, match_date,
                home_team_id, away_team_id,
                home_score, away_score,
                competition, season,
                stadium_id
            ) VALUES (
                %(sb_match_id)s, %(match_date)s,
                %(home_team_id)s, %(away_team_id)s,
                %(home_score)s, %(away_score)s,
                %(competition)s, %(season)s,
                %(stadium_id)s
            )
            ON CONFLICT (sb_match_id) DO UPDATE
                SET home_score  = EXCLUDED.home_score,
                    away_score  = EXCLUDED.away_score,
                    stadium_id  = COALESCE(matches.stadium_id, EXCLUDED.stadium_id)
            RETURNING match_id
        """, row)
        return cur.fetchone()[0]


# ---------------------------------------------------------------------------
# Weather
# ---------------------------------------------------------------------------

def upsert_weather(conn, match_id: int, weather: dict) -> int:
    """
    Insert or update a weather row.  Commits immediately.  Returns weather_id.
    """
    with conn.cursor() as cur:
        cur.execute("""
            INSERT INTO weather (
                match_id, temperature_c, humidity_pct,
                wind_speed_kmh, precipitation_mm, weather_condition
            ) VALUES (%s, %s, %s, %s, %s, %s)
            ON CONFLICT (match_id) DO UPDATE
                SET temperature_c     = EXCLUDED.temperature_c,
                    humidity_pct      = EXCLUDED.humidity_pct,
                    wind_speed_kmh    = EXCLUDED.wind_speed_kmh,
                    precipitation_mm  = EXCLUDED.precipitation_mm,
                    weather_condition = EXCLUDED.weather_condition
            RETURNING weather_id
        """, (
            match_id,
            weather.get("temperature_c"),
            weather.get("humidity_pct"),
            weather.get("wind_speed_kmh"),
            weather.get("precipitation_mm"),
            weather.get("weather_condition"),
        ))
        weather_id = cur.fetchone()[0]
    conn.commit()
    return weather_id


# ---------------------------------------------------------------------------
# Player match stats  (raw event aggregates)
# ---------------------------------------------------------------------------

def insert_stats(conn, rows: list, page_size: int = 500):
    """
    Bulk-upsert player match stat rows.  Does NOT commit.

    Tuple column order:
        player_id, match_id, team_id, weather_id, result,
        goals, assists, shots, xg, xa, key_passes,
        passes_attempted, passes_completed, pass_accuracy,
        progressive_passes,
        carry_distance, progressive_carries,
        dribbles_completed,
        tackles, interceptions, clearances, pressures,
        yellow_cards, red_cards,
        minutes_played, sub_minute
    """
    if not rows:
        return

    with conn.cursor() as cur:
        execute_values(cur, """
            INSERT INTO player_match_stats (
                player_id, match_id, team_id, weather_id, result,
                goals, assists, shots, xg, xa, key_passes,
                passes_attempted, passes_completed, pass_accuracy,
                progressive_passes,
                carry_distance, progressive_carries,
                dribbles_completed,
                tackles, interceptions, clearances, pressures,
                yellow_cards, red_cards,
                minutes_played, sub_minute
            ) VALUES %s
            ON CONFLICT (player_id, match_id) DO UPDATE SET
                goals               = EXCLUDED.goals,
                assists             = EXCLUDED.assists,
                shots               = EXCLUDED.shots,
                xg                  = EXCLUDED.xg,
                xa                  = EXCLUDED.xa,
                key_passes          = EXCLUDED.key_passes,
                passes_attempted    = EXCLUDED.passes_attempted,
                passes_completed    = EXCLUDED.passes_completed,
                pass_accuracy       = EXCLUDED.pass_accuracy,
                progressive_passes  = EXCLUDED.progressive_passes,
                carry_distance      = EXCLUDED.carry_distance,
                progressive_carries = EXCLUDED.progressive_carries,
                dribbles_completed  = EXCLUDED.dribbles_completed,
                tackles             = EXCLUDED.tackles,
                interceptions       = EXCLUDED.interceptions,
                clearances          = EXCLUDED.clearances,
                pressures           = EXCLUDED.pressures,
                yellow_cards        = EXCLUDED.yellow_cards,
                red_cards           = EXCLUDED.red_cards,
                minutes_played      = EXCLUDED.minutes_played,
                sub_minute          = EXCLUDED.sub_minute,
                weather_id          = EXCLUDED.weather_id,
                result              = EXCLUDED.result
        """, rows, page_size=page_size)


# ---------------------------------------------------------------------------
# Player match features  (computed ML columns — separate table)
# ---------------------------------------------------------------------------

def insert_features(conn, rows: list, page_size: int = 500):
    """
    Bulk-upsert player_match_features rows.  Does NOT commit.

    These rows are written by compute_labels.py after ingestion is complete.
    They reference player_match_stats.stat_id via a FK, so stats must be
    committed before calling this function.

    Tuple column order:
        stat_id, player_id, match_id,
        matches_last_30_days, minutes_last_30_days,
        days_since_last_injury, is_injured_next_30d
    """
    if not rows:
        return

    with conn.cursor() as cur:
        execute_values(cur, """
            INSERT INTO player_match_features (
                stat_id, player_id, match_id,
                matches_last_30_days, minutes_last_30_days,
                days_since_last_injury, is_injured_next_30d
            ) VALUES %s
            ON CONFLICT (player_id, match_id) DO UPDATE SET
                matches_last_30_days   = EXCLUDED.matches_last_30_days,
                minutes_last_30_days   = EXCLUDED.minutes_last_30_days,
                days_since_last_injury = EXCLUDED.days_since_last_injury,
                is_injured_next_30d    = EXCLUDED.is_injured_next_30d
        """, rows, page_size=page_size)


# ---------------------------------------------------------------------------
# Pass network edges
# ---------------------------------------------------------------------------

def upsert_pass_edges(conn, rows: list, page_size: int = 500):
    """
    Bulk-insert pass network edge rows.  Does NOT commit.

    The table now has a true UNIQUE(match_id, team_id, passer_id, receiver_id)
    constraint, so ON CONFLICT DO NOTHING correctly deduplicates re-runs.

    Each row: (match_id, team_id, passer_id, receiver_id,
               pass_count, avg_x_start, avg_y_start, avg_x_end, avg_y_end)
    """
    if not rows:
        return
    with conn.cursor() as cur:
        execute_values(cur, """
            INSERT INTO pass_network_edges (
                match_id, team_id, passer_id, receiver_id,
                pass_count,
                avg_x_start, avg_y_start, avg_x_end, avg_y_end
            ) VALUES %s
            ON CONFLICT (match_id, team_id, passer_id, receiver_id) DO NOTHING
        """, rows, page_size=page_size)