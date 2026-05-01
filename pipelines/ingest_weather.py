"""
pipelines/ingest_weather.py

Fetch historical weather data from Open-Meteo for every match that has
stadium coordinates, and upsert into the weather table.

Open-Meteo free archive API (no key required):
https://archive-api.open-meteo.com/v1/archive

Field mapping (from project spec):
  temperature_c    = avg(temperature_2m_max, temperature_2m_min)
  precipitation_mm = precipitation_sum
  wind_speed_kmh   = windspeed_10m_max
  humidity_pct     = relative_humidity_2m_max
"""

import logging
import time

import requests

from load.postgres import connect, upsert_weather
from config.settings import DB_DSN, OPEN_METEO_URL

logger = logging.getLogger(__name__)

# Stadium coordinates for StatsBomb open-data stadiums that lack lat/lng
# in the raw JSON.  Add more as needed.
STADIUM_COORDS = {
    # La Liga
    "Camp Nou":                     (41.3809,  2.1228),
    "Santiago Bernabeu":            (40.4531, -3.6883),
    "Estadio Wanda Metropolitano":  (40.4361, -3.5995),
    "Estadio Ramon Sanchez-Pizjuan":(37.3841, -5.9706),
    "Estadio de Mestalla":          (39.4750, -0.3583),
    "Estadio San Mames":            (43.2642, -2.9494),
    # World Cup 2018 -- Russia
    "Luzhniki Stadium":             (55.7317,  37.5600),
    "Saint Petersburg Stadium":     (59.9724,  30.2219),
    "Fisht Stadium":                (43.4010,  39.9514),
    "Ekaterinburg Arena":           (56.8429,  60.5935),
    "Kazan Arena":                  (55.8483,  49.0675),
    "Nizhny Novgorod Stadium":      (56.3379,  43.9633),
    "Mordovia Arena":               (54.1831,  45.1747),
    "Rostov Arena":                 (47.2289,  39.7158),
    "Volgograd Arena":              (48.7074,  44.5534),
    "Cosmos Arena":                 (53.4133,  50.1725),
    "Kaliningrad Stadium":          (54.7138,  20.5167),
    # World Cup 2022 -- Qatar
    "Lusail Iconic Stadium":        (25.4333,  51.5000),
    "Al Bayt Stadium":              (25.6572,  51.5150),
    "Ahmad Bin Ali Stadium":        (25.2477,  51.4041),
    "Education City Stadium":       (25.3117,  51.4230),
    "Al Thumama Stadium":           (25.2364,  51.5361),
    "Khalifa International Stadium":(25.2632,  51.4500),
    "Stadium 974":                  (25.2735,  51.5497),
    "Al Janoub Stadium":            (25.1270,  51.5000),
}

_DAILY_VARS = (
    "temperature_2m_max,temperature_2m_min,"
    "precipitation_sum,windspeed_10m_max,"
    "relative_humidity_2m_max"
)


def _fetch_weather(lat: float, lng: float, date_str: str) -> dict | None:
    """Call Open-Meteo archive API and return parsed weather dict."""
    params = {
        "latitude":  lat,
        "longitude": lng,
        "start_date": date_str,
        "end_date":   date_str,
        "daily":      _DAILY_VARS,
        "timezone":   "UTC",
    }
    try:
        resp = requests.get(OPEN_METEO_URL, params=params, timeout=15)
        resp.raise_for_status()
        data = resp.json()
        daily = data.get("daily", {})

        t_max = (daily.get("temperature_2m_max") or [None])[0]
        t_min = (daily.get("temperature_2m_min") or [None])[0]
        temp  = (t_max + t_min) / 2 if t_max is not None and t_min is not None else None

        return {
            "temperature_c":    temp,
            "precipitation_mm": (daily.get("precipitation_sum") or [None])[0],
            "wind_speed_kmh":   (daily.get("windspeed_10m_max") or [None])[0],
            "humidity_pct":     (daily.get("relative_humidity_2m_max") or [None])[0],
            "weather_condition": None,
        }
    except Exception as exc:
        logger.warning("Open-Meteo request failed (%s, %s, %s): %s", lat, lng, date_str, exc)
        return None


def run(conn=None):
    """
    For every match that has stadium coordinates (either from the DB row or
    from STADIUM_COORDS above), fetch weather and upsert into the weather
    table.  Skips matches that already have a weather row.
    """
    if conn is None:
        conn = connect(DB_DSN)

    with conn.cursor() as cur:
        cur.execute("""
            SELECT m.match_id, m.match_date, m.stadium_name,
                   m.stadium_lat, m.stadium_lng
            FROM   matches m
            LEFT JOIN weather w ON w.match_id = m.match_id
            WHERE  w.weather_id IS NULL
            ORDER BY m.match_date
        """)
        rows = cur.fetchall()

    logger.info("Weather ingestion: %d matches need weather data", len(rows))
    inserted = 0
    skipped  = 0

    for match_id, match_date, stadium_name, lat, lng in rows:
        # Resolve coordinates: prefer DB value, fall back to lookup table
        if lat is None or lng is None:
            coords = STADIUM_COORDS.get(stadium_name or "")
            if coords is None:
                logger.debug("No coordinates for stadium '%s', skipping", stadium_name)
                skipped += 1
                continue
            lat, lng = coords

        date_str = str(match_date)
        weather  = _fetch_weather(lat, lng, date_str)
        if weather is None:
            skipped += 1
            continue

        upsert_weather(conn, match_id, weather)
        inserted += 1

        # Be polite to the free API
        time.sleep(0.25)

    logger.info(
        "Weather ingestion complete: %d inserted, %d skipped (no coords or API error)",
        inserted, skipped,
    )

    # Return a mapping of match_id -> weather_id so the stats pipeline can
    # attach weather_id foreign keys without a second DB round-trip.
    with conn.cursor() as cur:
        cur.execute("SELECT match_id, weather_id FROM weather")
        return {mid: wid for mid, wid in cur.fetchall()}


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    wc = run()
    print(f"Weather cache built: {len(wc)} entries")