"""
main.py

Full pipeline orchestrator.  Run this after init_db.py.

Execution order
---------------
1. ingest_statsbomb   -- matches, players, teams, player_match_stats
2. ingest_weather     -- weather table (Open-Meteo API)
3. ingest_injuries    -- injuries table + players.date_of_birth (Transfermarkt)
4. ingest_pass_network -- pass_network_edges (Model 2 input)
5. compute_labels     -- is_injured_next_30d, workload features (Model 3)

Then train models (optional, pass --train to enable):
  Model 1  Player clustering
  Model 2  Team cohesion regression
  Model 3  Injury risk classification
  Model 4  Environmental impact regression
  Model 5  Win probability classification
"""

import argparse
import logging
import psycopg2

from config.settings import DB_DSN, DATA_ROOT
from core.caches import TeamCache, PlayerCache
from extract import statsbomb_local as sb
from pipelines.ingest_statsbomb   import run as run_statsbomb
from pipelines.ingest_weather     import run as run_weather
from pipelines.ingest_injuries    import run as run_injuries
from pipelines.ingest_pass_network import run as run_pass_network
from pipelines.compute_labels     import run as run_labels

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
)
logger = logging.getLogger(__name__)


def parse_args():
    p = argparse.ArgumentParser(description="Soccer Analytics ML pipeline")
    p.add_argument("--train",    action="store_true", help="Train ML models after ingestion")
    p.add_argument("--skip-ingest", action="store_true", help="Skip ingestion, go straight to labels / training")
    return p.parse_args()


def main():
    args = parse_args()

    sb.set_root(DATA_ROOT)
    conn = psycopg2.connect(DB_DSN)

    if not args.skip_ingest:
        # ---- 1. StatsBomb events ------------------------------------------
        logger.info("=" * 60)
        logger.info("Step 1: StatsBomb ingestion")
        logger.info("=" * 60)
        team_cache   = TeamCache(conn)
        player_cache = PlayerCache(conn)
        run_statsbomb(conn, team_cache, player_cache)

        # ---- 2. Weather ---------------------------------------------------
        logger.info("=" * 60)
        logger.info("Step 2: Weather ingestion (Open-Meteo)")
        logger.info("=" * 60)
        weather_cache = run_weather(conn)
        logger.info("Weather cache: %d entries", len(weather_cache))

        # ---- 3. Injuries / Transfermarkt ----------------------------------
        logger.info("=" * 60)
        logger.info("Step 3: Injuries ingestion (Transfermarkt)")
        logger.info("=" * 60)
        run_injuries(conn)

        # ---- 4. Pass network edges ----------------------------------------
        logger.info("=" * 60)
        logger.info("Step 4: Pass network extraction")
        logger.info("=" * 60)
        run_pass_network(conn)

    # ---- 5. Compute derived labels / features ----------------------------
    logger.info("=" * 60)
    logger.info("Step 5: Computing labels and workload features")
    logger.info("=" * 60)
    run_labels(conn)

    # ---- 6. Train ML models (optional) -----------------------------------
    if args.train:
        logger.info("=" * 60)
        logger.info("Step 6: Training ML models")
        logger.info("=" * 60)

        from models.model1_player_clustering import run as train1
        from models.model2_team_cohesion     import run as train2
        from models.model3_injury_risk       import run as train3
        from models.model4_environment       import run as train4
        from models.model5_win_probability   import run as train5

        logger.info("Model 1: Player Efficiency and Style Profiling")
        train1(conn)

        logger.info("Model 2: Team Cohesion Analysis")
        train2(conn)

        logger.info("Model 3: Injury Risk Prediction")
        train3(conn)

        logger.info("Model 4: Environmental Impact Analysis")
        train4(conn)

        logger.info("Model 5: Win Probability Modeling")
        train5(conn)

    conn.close()
    logger.info("Pipeline complete.")


if __name__ == "__main__":
    main()