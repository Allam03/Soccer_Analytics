"""
config/settings.py

All configuration is read from environment variables.
Copy .env.example to .env and fill in your values before running anything.
"""

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()


def _require(key: str) -> str:
    val = os.getenv(key)
    if not val:
        raise RuntimeError(
            f"Required environment variable '{key}' is not set. "
            "Copy .env.example to .env and fill in your values."
        )
    return val


# ---------------------------------------------------------------------------
# Database
# ---------------------------------------------------------------------------
DB_DSN = _require("DB_DSN")

# ---------------------------------------------------------------------------
# Data paths
# ---------------------------------------------------------------------------
DATA_ROOT                 = Path(_require("DATA_ROOT"))
TRANSFERMARKT_CSV         = Path(_require("TRANSFERMARKT_CSV"))
TRANSFERMARKT_PLAYERS_CSV = Path(_require("TRANSFERMARKT_PLAYERS_CSV"))

# ---------------------------------------------------------------------------
# External APIs
# ---------------------------------------------------------------------------
OPEN_METEO_URL = os.getenv(
    "OPEN_METEO_URL",
    "https://archive-api.open-meteo.com/v1/archive",
)

# ---------------------------------------------------------------------------
# (competition_id, season_id) pairs that are in scope.
#
# La Liga: competition_id=11, seasons 2015/16-2020/21
# UCL 2018/19: competition_id=16, season_id=4
# World Cup 2018: competition_id=43, season_id=3
# World Cup 2022: competition_id=43, season_id=106
# ---------------------------------------------------------------------------
COMPETITIONS = {
    (11, 26),   # La Liga 2015/16
    (11, 27),   # La Liga 2016/17
    (11,  4),   # La Liga 2017/18
    (11,  1),   # La Liga 2018/19
    (11, 42),   # La Liga 2019/20
    (11, 90),   # La Liga 2020/21
    (16,  4),   # UEFA Champions League 2018/19
    (43,  3),   # FIFA World Cup 2018
    (43, 106),  # FIFA World Cup 2022
}