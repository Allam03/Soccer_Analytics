"""
init_db.py

One-time database initialisation script.
Run this BEFORE any pipeline to create all tables, indexes, and views.

Usage:
    python init_db.py
"""

import logging
import psycopg2
from config.settings import DB_DSN

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
logger = logging.getLogger(__name__)


def init_db(conn):
    """Execute the full DDL from schema.sql."""
    schema_path = "schema.sql"
    logger.info("Reading schema from %s ...", schema_path)
    with open(schema_path) as f:
        sql = f.read()

    with conn.cursor() as cur:
        cur.execute(sql)
    conn.commit()
    logger.info("Schema created / verified successfully.")


def verify(conn):
    """Quick sanity check -- list all tables in public schema."""
    with conn.cursor() as cur:
        cur.execute("""
            SELECT table_name
            FROM information_schema.tables
            WHERE table_schema = 'public'
            ORDER BY table_name
        """)
        tables = [r[0] for r in cur.fetchall()]
    logger.info("Tables in database: %s", tables)
    expected = {
        "injuries", "matches", "pass_network_edges",
        "player_match_stats", "players", "teams", "weather",
    }
    missing = expected - set(tables)
    if missing:
        logger.error("Missing tables: %s", missing)
    else:
        logger.info("All expected tables present.")


if __name__ == "__main__":
    logger.info("Connecting to %s ...", DB_DSN.split("@")[-1])
    conn = psycopg2.connect(DB_DSN)
    init_db(conn)
    verify(conn)
    conn.close()
    logger.info("Done.")