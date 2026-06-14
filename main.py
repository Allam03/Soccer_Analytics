"""
Pipeline orchestrator. Run init_db.py once before first use.

Data sources: StatsBomb event data (primary) and Transfermarkt CSVs (injuries).
The weather (Open-Meteo) pipeline and model were removed; they did not
generalise on this data.

Flags:
  --train         Train ML models after ingestion
  --skip-ingest   Skip ingestion, run only training
  --workers N     Worker processes for StatsBomb ingestion (default: CPU count - 1)
"""

import argparse
import logging
import psycopg2

from config.settings import DB_DSN, DATA_ROOT
from core.caches import TeamCache, PlayerCache
from extract import statsbomb_local as sb
from pipelines.ingest_statsbomb import run as run_statsbomb
from pipelines.extract_shots   import run as run_shots
from pipelines.ingest_injuries import run as run_injuries
from pipelines.compute_labels  import run as run_labels

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
        kwargs = {"workers": args.workers} if args.workers is not None else {}
        run_statsbomb(conn, TeamCache(conn), PlayerCache(conn), **kwargs)

        logger.info("%s\nStep 2: Shot extraction (xG inputs)", _SEP)
        run_shots(conn)

        logger.info("%s\nStep 3: Injuries ingestion (Transfermarkt)", _SEP)
        run_injuries(conn)

        logger.info("%s\nStep 4: Computing labels (workload + injury)", _SEP)
        run_labels(conn)

    if args.train:
        logger.info("%s\nStep 5: Training ML models", _SEP)

        from models.model1_player_clustering import run as train1
        from models.model2_team_cohesion     import run as train2
        from models.model3_injury_risk       import run as train3
        from models.model5_win_probability   import run as train5
        from models.model_xg                 import run as train_xg

        for label, fn in [
            ("Model 1: Player Efficiency & Style Profiling", train1),
            ("Model 2: Team Cohesion / Pass Networks",        train2),
            ("Model 3: Injury Risk",                          train3),
            ("Model 5: Win Probability",                      train5),
            ("Model xG: Expected Goals",                      train_xg),
        ]:
            logger.info(label)
            fn(conn)

    conn.close()
    logger.info("Pipeline complete.")


if __name__ == "__main__":
    main()
