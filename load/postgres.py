"""
load/postgres.py

Database write helpers.  All column references use the renamed schema:
  sb_match_id, sb_team_id, sb_player_id  (source identifiers)
  match_id, team_id, player_id           (internal surrogate PKs)
"""

import psycopg2
from psycopg2.extras import execute_values


def connect(dsn: str):
    return psycopg2.connect(dsn)


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
    home_score, away_score, competition, season,
    stadium_name, stadium_lat, stadium_lng
    """
    with conn.cursor() as cur:
        cur.execute("""
            INSERT INTO matches (
                sb_match_id, match_date,
                home_team_id, away_team_id,
                home_score, away_score,
                competition, season,
                stadium_name, stadium_lat, stadium_lng
            ) VALUES (
                %(sb_match_id)s, %(match_date)s,
                %(home_team_id)s, %(away_team_id)s,
                %(home_score)s, %(away_score)s,
                %(competition)s, %(season)s,
                %(stadium_name)s, %(stadium_lat)s, %(stadium_lng)s
            )
            ON CONFLICT (sb_match_id) DO UPDATE
                SET home_score   = EXCLUDED.home_score,
                    away_score   = EXCLUDED.away_score,
                    stadium_name = EXCLUDED.stadium_name,
                    stadium_lat  = EXCLUDED.stadium_lat,
                    stadium_lng  = EXCLUDED.stadium_lng
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
# Player match stats
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
# Pass network edges
# ---------------------------------------------------------------------------

def upsert_pass_edges(conn, rows: list, page_size: int = 500):
    """
    Bulk-insert pass network edge rows.  Does NOT commit.

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
            ON CONFLICT DO NOTHING
        """, rows, page_size=page_size)