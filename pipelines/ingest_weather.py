"""
pipelines/ingest_weather.py

Fetch historical weather data from Open-Meteo for every match that has
stadium coordinates and upsert into the weather table.

Performance fix #8
------------------
Replaced the sequential requests + time.sleep(0.25) loop with a
ThreadPoolExecutor.  Threads are ideal here (pure I/O -- network wait).
A semaphore caps concurrent in-flight requests to MAX_CONCURRENT (default 4)
so we stay polite to the free API without sleeping between every request.

At 4 concurrent requests with ~300ms average latency each, throughput goes
from 4 req/s (sequential + 0.25s sleep) to ~13 req/s -- roughly 3-4x faster.

Open-Meteo free archive API (no key required):
https://archive-api.open-meteo.com/v1/archive

Field mapping (from project spec):
  temperature_c    = avg(temperature_2m_max, temperature_2m_min)
  precipitation_mm = precipitation_sum
  wind_speed_kmh   = windspeed_10m_max
  humidity_pct     = relative_humidity_2m_max
"""

import logging
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests

from load.postgres import connect, upsert_weather
from config.settings import DB_DSN, OPEN_METEO_URL

logger = logging.getLogger(__name__)

MAX_CONCURRENT = 4          # simultaneous in-flight HTTP requests
REQUEST_TIMEOUT = 15        # seconds per request

# Stadium coordinates for StatsBomb open-data stadiums that lack lat/lng.
STADIUM_COORDS = {
    # La Liga
    "Camp Nou":                      (41.3809,   2.1228),
    "Santiago Bernabeu":             (40.4531,  -3.6883),
    "Estadio Wanda Metropolitano":   (40.4361,  -3.5995),
    "Estadio Ramon Sanchez-Pizjuan": (37.3841,  -5.9706),
    "Estadio de Mestalla":           (39.4750,  -0.3583),
    "Estadio San Mames":             (43.2642,  -2.9494),
    "Estadio de la Ceramica":        (39.9444,  -0.1028),
    "Estadio de Vallecas":           (40.3920,  -3.6600),
    "Estadio de Balaidos":           (42.2117,  -8.7397),
    "Estadio El Madrigal":           (39.9444,  -0.1028),
    "Estadio Municipal de Ipurua":   (43.1864,  -2.4714),
    "Estadio de la Rosaleda":        (36.7167,  -4.4500),
    "Estadio Nuevo Los Carmenes":    (37.1506,  -3.5986),
    "Estadio de Gran Canaria":       (28.1000, -15.4361),
    "Estadio El Molinon":            (43.5314,  -5.6361),
    "Estadio Municipal de Mendizorroza": (42.8494, -2.6819),
    "Power Horse Stadium":           (36.8417,  -2.4556),
    "Estadio de la Liga":            (40.4531,  -3.6883),
    # World Cup 2018 -- Russia
    "Luzhniki Stadium":              (55.7317,  37.5600),
    "Saint Petersburg Stadium":      (59.9724,  30.2219),
    "Fisht Stadium":                 (43.4010,  39.9514),
    "Ekaterinburg Arena":            (56.8429,  60.5935),
    "Kazan Arena":                   (55.8483,  49.0675),
    "Nizhny Novgorod Stadium":       (56.3379,  43.9633),
    "Mordovia Arena":                (54.1831,  45.1747),
    "Rostov Arena":                  (47.2289,  39.7158),
    "Volgograd Arena":               (48.7074,  44.5534),
    "Cosmos Arena":                  (53.4133,  50.1725),
    "Kaliningrad Stadium":           (54.7138,  20.5167),
    # World Cup 2022 -- Qatar
    "Lusail Iconic Stadium":         (25.4333,  51.5000),
    "Al Bayt Stadium":               (25.6572,  51.5150),
    "Ahmad Bin Ali Stadium":         (25.2477,  51.4041),
    "Education City Stadium":        (25.3117,  51.4230),
    "Al Thumama Stadium":            (25.2364,  51.5361),
    "Khalifa International Stadium": (25.2632,  51.4500),
    "Stadium 974":                   (25.2735,  51.5497),
    "Al Janoub Stadium":             (25.1270,  51.5000),
    # UCL 2018/19 -- common venues
    "Wanda Metropolitano":           (40.4361,  -3.5995),
    "Anfield":                       (53.4308,  -2.9608),
    "Johan Cruijff Arena":           (52.3143,   4.9418),
    "Camp Nou":                      (41.3809,   2.1228),
    "Tottenham Hotspur Stadium":     (51.6042,  -0.0664),
    "Estadio da Luz":                (38.7525,  -9.1842),
    "Signal Iduna Park":             (51.4926,   7.4519),
    "Parc des Princes":              (48.8414,   2.2530),
    "Allianz Arena":                 (48.2188,  11.6247),
    "Stamford Bridge":               (51.4816,  -0.1910),
    "Old Trafford":                  (53.4631,  -2.2913),
    "Santiago Bernabeu":             (40.4531,  -3.6883),
    "Juventus Stadium":              (45.1096,   7.6413),
    "San Siro":                      (45.4781,   9.1240),
}

_DAILY_VARS = (
    "temperature_2m_max,temperature_2m_min,"
    "precipitation_sum,windspeed_10m_max,"
    "relative_humidity_2m_max"
)


def _fetch_one(match_id: int, lat: float, lng: float,
               date_str: str, sem: threading.Semaphore) -> tuple:
    """
    Fetch weather for a single match.  Called in a thread.

    Returns (match_id, weather_dict | None).
    The semaphore limits concurrent in-flight requests.
    """
    with sem:
        params = {
            "latitude":   lat,
            "longitude":  lng,
            "start_date": date_str,
            "end_date":   date_str,
            "daily":      _DAILY_VARS,
            "timezone":   "UTC",
        }
        try:
            resp = requests.get(OPEN_METEO_URL, params=params,
                                timeout=REQUEST_TIMEOUT)
            resp.raise_for_status()
            data  = resp.json()
            daily = data.get("daily", {})

            t_max = (daily.get("temperature_2m_max") or [None])[0]
            t_min = (daily.get("temperature_2m_min") or [None])[0]
            temp  = (t_max + t_min) / 2 if (t_max is not None and t_min is not None) else None

            weather = {
                "temperature_c":    temp,
                "precipitation_mm": (daily.get("precipitation_sum")            or [None])[0],
                "wind_speed_kmh":   (daily.get("windspeed_10m_max")            or [None])[0],
                "humidity_pct":     (daily.get("relative_humidity_2m_max")     or [None])[0],
                "weather_condition": None,
            }
            return (match_id, weather)

        except Exception as exc:
            logger.warning(
                "Open-Meteo failed (match=%d, %s, %s, %s): %s",
                match_id, lat, lng, date_str, exc,
            )
            return (match_id, None)


def run(conn=None, max_concurrent: int = MAX_CONCURRENT) -> dict:
    """
    Fetch weather for all matches that don't yet have a weather row.

    Parameters
    ----------
    conn           : psycopg2 connection (created from DB_DSN if None)
    max_concurrent : max simultaneous HTTP requests (default 4)

    Returns
    -------
    {pg_match_id -> weather_id} -- full map after completion,
    for attaching weather_id FKs in the stats pipeline.
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
        pending_rows = cur.fetchall()

    logger.info("Weather ingestion: %d matches need weather data", len(pending_rows))

    # Resolve coordinates up front; discard rows with no coords
    work_items = []   # list of (match_id, lat, lng, date_str)
    skipped = 0
    for match_id, match_date, stadium_name, lat, lng in pending_rows:
        if lat is None or lng is None:
            coords = STADIUM_COORDS.get(stadium_name or "")
            if coords is None:
                logger.debug("No coords for '%s' -- skipping", stadium_name)
                skipped += 1
                continue
            lat, lng = coords
        work_items.append((match_id, lat, lng, str(match_date)))

    logger.info(
        "  %d fetchable, %d skipped (no coordinates)",
        len(work_items), skipped,
    )

    if not work_items:
        with conn.cursor() as cur:
            cur.execute("SELECT match_id, weather_id FROM weather")
            return {mid: wid for mid, wid in cur.fetchall()}

    # Semaphore shared across all threads to cap concurrency
    sem = threading.Semaphore(max_concurrent)

    inserted = 0
    failed   = 0

    # Fix #8: ThreadPoolExecutor -- all requests in-flight concurrently
    # (up to max_concurrent at a time via the semaphore)
    with ThreadPoolExecutor(max_workers=max_concurrent * 2) as pool:
        futures = {
            pool.submit(_fetch_one, mid, lat, lng, date_str, sem): mid
            for mid, lat, lng, date_str in work_items
        }

        for fut in as_completed(futures):
            match_id, weather = fut.result()
            if weather is None:
                failed += 1
                continue
            upsert_weather(conn, match_id, weather)   # commits per row
            inserted += 1

            if inserted % 50 == 0:
                logger.info("  Weather progress: %d / %d", inserted, len(work_items))

    logger.info(
        "Weather ingestion complete: %d inserted, %d failed, %d skipped",
        inserted, failed, skipped,
    )

    with conn.cursor() as cur:
        cur.execute("SELECT match_id, weather_id FROM weather")
        return {mid: wid for mid, wid in cur.fetchall()}


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    wc = run()
    print(f"Weather cache: {len(wc)} entries")