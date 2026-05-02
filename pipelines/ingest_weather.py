"""
pipelines/ingest_weather.py

Fetch historical weather data from Open-Meteo for every match and upsert
into the weather table.

Stadium name normalisation fix
-------------------------------
StatsBomb stadium names often contain accents, diacritics, and spacing
variations (e.g. "Estadio Ramón Sánchez-Pizjuán") that don't match the
plain-ASCII dict keys.  At startup, STADIUM_COORDS is re-indexed under
normalised keys using the same norm_name() logic applied to player names
(strip accents, lowercase, collapse spaces).  Incoming stadium names from
the DB are normalised the same way before lookup.

Concurrent HTTP fix (fix #8)
-----------------------------
ThreadPoolExecutor + Semaphore replaces sequential sleep loop.
"""

import logging
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests

from core.utils import norm_name
from load.postgres import connect, upsert_weather
from config.settings import DB_DSN, OPEN_METEO_URL

logger = logging.getLogger(__name__)

MAX_CONCURRENT  = 4
REQUEST_TIMEOUT = 15

# ---------------------------------------------------------------------------
# Stadium coordinate table  (plain ASCII keys -- normalised at import time)
# ---------------------------------------------------------------------------
_STADIUM_COORDS_RAW = {
    # La Liga
    "Camp Nou":                          (41.3809,   2.1228),
    "Santiago Bernabeu":                 (40.4531,  -3.6883),
    "Estadio Wanda Metropolitano":       (40.4361,  -3.5995),
    "Wanda Metropolitano":               (40.4361,  -3.5995),
    "Estádio Cívitas Metropolitano":     (40.4361,  -3.5995),
    "Estadio Ramon Sanchez-Pizjuan":     (37.3841,  -5.9706),
    "Estadio Ramón Sánchez Pizjuán":     (37.3841,  -5.9706),
    "Estadio de Mestalla":               (39.4750,  -0.3583),
    "Mestalla":                          (39.4750,  -0.3583),
    "Estadio San Mames":                 (43.2642,  -2.9494),
    "San Mames":                         (43.2642,  -2.9494),
    "Estadio de la Ceramica":            (39.9444,  -0.1028),
    "Estadio El Madrigal":               (39.9444,  -0.1028),
    "Estadio de Vallecas":               (40.3920,  -3.6600),
    "Estadio de Balaidos":               (42.2117,  -8.7397),
    "Abanca-Balaídos":                   (42.2117,  -8.7397),
    "Estadio Municipal de Ipurua":       (43.1864,  -2.4714),
    "Estadio de la Rosaleda":            (36.7167,  -4.4500),
    "Estadio La Rosaleda":               (36.7167,  -4.4500),
    "Estadio Nuevo Los Carmenes":        (37.1506,  -3.5986),
    "Estadio de Gran Canaria":           (28.1000, -15.4361),
    "Estadio El Molinon":                (43.5314,  -5.6361),
    "Estadio Municipal El Molinón":      (43.5314,  -5.6361),
    "Estadio Municipal de Mendizorroza": (42.8494,  -2.6819),
    "Estadio de Mendizorroza":           (42.8494,  -2.6819),
    "Power Horse Stadium":               (36.8417,  -2.4556),
    "Estadio Municipal de El Alcoraz":   (42.1361,  -0.4111),
    "Estadio El Alcoraz":                (42.1361,  -0.4111),
    "Estadio de Anoeta":                 (43.3014,  -1.9736),
    "Reale Arena":                       (43.3014,  -1.9736),
    "Estadio Municipal de Butarque":     (40.3517,  -3.7914),
    "Estadio Nuevo Mirandilla":          (36.5064,  -6.2722),
    "Estadio de los Juegos Mediterraneos": (36.8417, -2.4556),
    "Estadio Municipal de Montilivi":    (41.9833,   2.8167),
    "Estadi Municipal de Montilivi":     (41.9833,   2.8167),
    "Estadio RCDE":                      (41.3473,   2.0758),
    "Estadio Benito Villamarin":         (37.3567,  -5.9814),

    # Additional Spain stadiums
    "Coliseum Alfonso Pérez":            (40.3256,  -3.7143),
    "Estadi Mallorca Son Moix":          (39.5899,   2.6303),
    "Estadio Abanca-Riazor":             (43.3687,  -8.4173),
    "Estadio Alfredo Di Stéfano":        (40.4762,  -3.6163),
    "Estadio Ciudad de Valencia":        (39.4933,  -0.3642),
    "Estadio El Sadar":                  (42.7963,  -1.6373),
    "Estadio Manuel Martínez Valero":    (38.2669,  -0.6635),
    "Estadio Municipal José Zorrilla":   (41.6443,  -4.7612),
    "Estadio Nuevo Arcángel":            (37.8886,  -4.7896),
    "Estadio Vicente Calderón": (40.4017, -3.7206),

    # World Cup 2018 -- Russia
    "Luzhniki Stadium":                  (55.7317,  37.5600),
    "Stadion Luzhniki":                  (55.7317,  37.5600),
    "Saint Petersburg Stadium":          (59.9724,  30.2219),
    "Fisht Stadium":                     (43.4010,  39.9514),
    "Ekaterinburg Arena":                (56.8429,  60.5935),
    "Kazan Arena":                       (55.8483,  49.0675),
    "Ak Bars Arena":                     (55.8483,  49.0675),
    "Nizhny Novgorod Stadium":           (56.3379,  43.9633),
    "Stadion Nizhny Novgorod (Nizhniy Novgorod)": (56.3379, 43.9633),
    "Mordovia Arena":                    (54.1831,  45.1747),
    "Rostov Arena":                      (47.2289,  39.7158),
    "Volgograd Arena":                   (48.7074,  44.5534),
    "Cosmos Arena":                      (53.4133,  50.1725),
    "Solidarnost Arena":                 (53.4133,  50.1725),
    "Kaliningrad Stadium":               (54.7138,  20.5167),
    "Stadion Kaliningrad":               (54.7138,  20.5167),
    "Otkritie Bank Arena":      (55.8178, 37.4403),

    # World Cup 2022 -- Qatar
    "Lusail Iconic Stadium":             (25.4333,  51.5000),
    "Al Bayt Stadium":                   (25.6572,  51.5150),
    "Ahmad Bin Ali Stadium":             (25.2477,  51.4041),
    "Education City Stadium":            (25.3117,  51.4230),
    "Al Thumama Stadium":                (25.2364,  51.5361),
    "Khalifa International Stadium":     (25.2632,  51.4500),
    "Stadium 974":                       (25.2735,  51.5497),
    "Al Janoub Stadium":                 (25.1270,  51.5000),

    # UCL 2018/19
    "Anfield":                           (53.4308,  -2.9608),
    "Johan Cruijff Arena":               (52.3143,   4.9418),
    "Tottenham Hotspur Stadium":         (51.6042,  -0.0664),
    "Estadio da Luz":                    (38.7525,  -9.1842),
    "Signal Iduna Park":                 (51.4926,   7.4519),
    "Parc des Princes":                  (48.8414,   2.2530),
    "Allianz Arena":                     (48.2188,  11.6247),
    "Stamford Bridge":                   (51.4816,  -0.1910),
    "Old Trafford":                      (53.4631,  -2.2913),
    "Juventus Stadium":                  (45.1096,   7.6413),
    "San Siro":                          (45.4781,   9.1240),
    "Olimpiyskiy":                       (50.4339,  30.5214),
    "Estadio Jose Alvalade":             (38.7613,  -9.1603),
    "Jan Breydel Stadion":               (51.1944,   3.1600),
    "Estadio do Dragao":                 (41.1614,  -8.5839),
}

# Build a normalised lookup dict once at import time.
# Both keys AND incoming stadium names from the DB are passed through
# norm_name() before comparison, so accents / casing mismatches are
# eliminated automatically.
STADIUM_COORDS: dict[str, tuple] = {
    norm_name(k): v for k, v in _STADIUM_COORDS_RAW.items()
}

_DAILY_VARS = (
    "temperature_2m_max,temperature_2m_min,"
    "precipitation_sum,windspeed_10m_max,"
    "relative_humidity_2m_max"
)


def _resolve_coords(
    stadium_name: str | None,
    lat: float | None,
    lng: float | None,
) -> tuple[float, float] | None:
    """
    Return (lat, lng) for a match, or None if coordinates cannot be found.

    Prefers explicit DB coords, falls back to the normalised lookup table.
    """
    if lat is not None and lng is not None:
        return lat, lng

    if not stadium_name:
        return None

    nn = norm_name(stadium_name)
    coords = STADIUM_COORDS.get(nn)
    if coords:
        return coords

    # Partial match fallback: try if any key starts with the incoming name
    # (handles "Camp Nou" matching "camp nou barcelona" variants)
    for key, val in STADIUM_COORDS.items():
        if nn in key or key in nn:
            return val

    return None


def _fetch_one(
    match_id: int,
    lat: float,
    lng: float,
    date_str: str,
    sem: threading.Semaphore,
) -> tuple[int, dict | None]:
    """Fetch weather for one match in a worker thread."""
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
            resp  = requests.get(OPEN_METEO_URL, params=params, timeout=REQUEST_TIMEOUT)
            resp.raise_for_status()
            daily = resp.json().get("daily", {})

            t_max = (daily.get("temperature_2m_max") or [None])[0]
            t_min = (daily.get("temperature_2m_min") or [None])[0]
            temp  = (t_max + t_min) / 2 if (t_max is not None and t_min is not None) else None

            return match_id, {
                "temperature_c":    temp,
                "precipitation_mm": (daily.get("precipitation_sum")        or [None])[0],
                "wind_speed_kmh":   (daily.get("windspeed_10m_max")        or [None])[0],
                "humidity_pct":     (daily.get("relative_humidity_2m_max") or [None])[0],
                "weather_condition": None,
            }
        except Exception as exc:
            logger.warning(
                "Open-Meteo failed (match=%d %s): %s", match_id, date_str, exc
            )
            return match_id, None


def run(conn=None, max_concurrent: int = MAX_CONCURRENT) -> dict:
    """
    Fetch and store weather for all matches that lack a weather row.

    Returns {pg_match_id -> weather_id}.
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

    work_items = []
    skipped    = 0
    unresolved_stadiums: set[str] = set()

    for match_id, match_date, stadium_name, lat, lng in pending_rows:
        coords = _resolve_coords(stadium_name, lat, lng)
        if coords is None:
            unresolved_stadiums.add(stadium_name or "<null>")
            skipped += 1
            continue
        work_items.append((match_id, coords[0], coords[1], str(match_date)))

    if unresolved_stadiums:
        logger.warning(
            "  %d matches skipped -- no coords for stadiums: %s",
            skipped,
            sorted(unresolved_stadiums),
        )

    logger.info("  %d fetchable, %d skipped (no coordinates)", len(work_items), skipped)

    if not work_items:
        with conn.cursor() as cur:
            cur.execute("SELECT match_id, weather_id FROM weather")
            return {mid: wid for mid, wid in cur.fetchall()}

    sem      = threading.Semaphore(max_concurrent)
    inserted = 0
    failed   = 0

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
            upsert_weather(conn, match_id, weather)
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