"""
pipelines/compute_labels.py

Post-ingestion SQL passes that compute derived columns in player_match_stats:

  1. is_injured_next_30d      -- Model 3 target label
  2. days_since_last_injury   -- Model 3 feature
  3. matches_last_30_days     -- Model 3 feature (workload)
  4. minutes_last_30_days     -- Model 3 feature (workload)

Run this AFTER:
  - ingest_statsbomb.py  (player_match_stats rows exist)
  - ingest_injuries.py   (injuries table populated)
"""

import logging
from load.postgres import connect
from config.settings import DB_DSN

logger = logging.getLogger(__name__)


def compute_injury_label(conn):
    """
    Set is_injured_next_30d = TRUE for any player-match row where the player
    suffered an injury within 30 days after match_date.
    """
    logger.info("Computing is_injured_next_30d label ...")
    with conn.cursor() as cur:
        # Reset first so re-runs are idempotent
        cur.execute("UPDATE player_match_stats SET is_injured_next_30d = FALSE")

        cur.execute("""
            UPDATE player_match_stats pms
            SET    is_injured_next_30d = TRUE
            FROM   matches m
            JOIN   injuries i ON i.player_id = pms.player_id
            WHERE  pms.match_id = m.match_id
              AND  i.injury_date >= m.match_date
              AND  i.injury_date <= m.match_date + INTERVAL '30 days'
        """)
        updated = cur.rowcount
    conn.commit()
    logger.info("  is_injured_next_30d set to TRUE for %d rows", updated)


def compute_days_since_last_injury(conn):
    """
    For each player-match row, set days_since_last_injury to the number of
    days between the match and the player's most recent injury return_date
    before that match.  NULL when no prior injury exists on record.
    """
    logger.info("Computing days_since_last_injury ...")
    with conn.cursor() as cur:
        cur.execute("""
            UPDATE player_match_stats pms
            SET    days_since_last_injury = sub.days_since
            FROM (
                SELECT
                    pms2.stat_id,
                    m2.match_date - MAX(i.return_date) AS days_since
                FROM player_match_stats pms2
                JOIN matches m2  ON m2.match_id  = pms2.match_id
                JOIN injuries i  ON i.player_id  = pms2.player_id
                WHERE i.return_date < m2.match_date
                GROUP BY pms2.stat_id, m2.match_date
            ) sub
            WHERE pms.stat_id = sub.stat_id
        """)
        updated = cur.rowcount
    conn.commit()
    logger.info("  days_since_last_injury updated for %d rows", updated)


def compute_workload(conn):
    """
    Set matches_last_30_days and minutes_last_30_days for every row by
    looking back 30 days from each match_date.

    This is done in a single UPDATE ... FROM subquery to avoid row-by-row
    Python iteration over what can be 50k+ rows.
    """
    logger.info("Computing workload features (matches/minutes last 30 days) ...")

    with conn.cursor() as cur:
        cur.execute("""
            UPDATE player_match_stats pms
            SET
                matches_last_30_days = sub.match_count,
                minutes_last_30_days = sub.minute_sum
            FROM (
                SELECT
                    pms_outer.stat_id,
                    COUNT(pms_inner.stat_id)            AS match_count,
                    COALESCE(SUM(pms_inner.minutes_played), 0) AS minute_sum
                FROM player_match_stats pms_outer
                JOIN matches m_outer ON m_outer.match_id = pms_outer.match_id
                LEFT JOIN player_match_stats pms_inner
                    ON  pms_inner.player_id = pms_outer.player_id
                    AND pms_inner.stat_id  != pms_outer.stat_id
                LEFT JOIN matches m_inner
                    ON  m_inner.match_id   = pms_inner.match_id
                    AND m_inner.match_date >= m_outer.match_date - INTERVAL '30 days'
                    AND m_inner.match_date <  m_outer.match_date
                GROUP BY pms_outer.stat_id
            ) sub
            WHERE pms.stat_id = sub.stat_id
        """)
        updated = cur.rowcount
    conn.commit()
    logger.info("  Workload features updated for %d rows", updated)


def run(conn=None):
    if conn is None:
        conn = connect(DB_DSN)

    compute_workload(conn)
    compute_days_since_last_injury(conn)
    compute_injury_label(conn)

    logger.info("compute_labels complete")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    run()