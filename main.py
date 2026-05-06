"""
Pipeline orchestrator. Run init_db.py once before first use.

Flags:
  --train         Train ML models after ingestion
  --skip-ingest   Skip ingestion (steps 1-3), run only labels and optional training
  --workers N     Worker processes for StatsBomb ingestion (default: CPU count - 1)
"""

import argparse
import logging
import psycopg2

from config.settings import DB_DSN, DATA_ROOT
from core.caches import TeamCache, PlayerCache
from extract import statsbomb_local as sb
from pipelines.ingest_statsbomb import run as run_statsbomb
from pipelines.ingest_weather   import run as run_weather
from pipelines.ingest_injuries  import run as run_injuries
from pipelines.compute_labels   import run as run_labels

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
)
logger = logging.getLogger(__name__)

_SEP = "=" * 60


def parse_args():
    p = argparse.ArgumentParser(description="Soccer Analytics ML pipeline")
    p.add_argument("--train",       action="store_true")
    p.add_argument("--skip-ingest", action="store_true")
    p.add_argument("--workers",     type=int, default=None)
    return p.parse_args()


def main():
    args = parse_args()
    sb.set_root(DATA_ROOT)
    conn = psycopg2.connect(DB_DSN)

    if not args.skip_ingest:
        logger.info("%s\nStep 1: StatsBomb ingestion", _SEP)
        team_cache   = TeamCache(conn)
        player_cache = PlayerCache(conn)
        kwargs = {"workers": args.workers} if args.workers is not None else {}
        run_statsbomb(conn, team_cache, player_cache, **kwargs)

        logger.info("%s\nStep 2: Weather ingestion", _SEP)
        weather_cache = run_weather(conn)
        logger.info("Weather cache: %d entries", len(weather_cache))

        logger.info("%s\nStep 3: Injuries ingestion", _SEP)
        run_injuries(conn)

    logger.info("%s\nStep 4: Computing labels", _SEP)
    run_labels(conn)

    if args.train:
        logger.info("%s\nStep 5: Training ML models", _SEP)

        from models.model1_player_clustering import run as train1
        from models.model2_team_cohesion     import run as train2
        from models.model3_injury_risk       import run as train3
        from models.model4_environment       import run as train4
        from models.model5_win_probability   import run as train5

        for label, fn in [
            ("Model 1: Player Clustering",        train1),
            ("Model 2: Team Cohesion",            train2),
            ("Model 3: Injury Risk",              train3),
            ("Model 4: Environmental Impact",     train4),
            ("Model 5: Win Probability",          train5),
        ]:
            logger.info(label)
            fn(conn)

    conn.close()
    logger.info("Pipeline complete.")


if __name__ == "__main__":
    main()