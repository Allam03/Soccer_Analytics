"""
pipelines/compute_labels.py

Post-ingestion SQL passes that compute derived columns in player_match_stats.

Workload fix
------------
The original query used:
    LEFT JOIN player_match_stats pms_inner ON pms_inner.player_id = ...
    LEFT JOIN matches m_inner ON m_inner.match_id = pms_inner.match_id
        AND m_inner.match_date >= ...

The date filter on the JOIN condition filters which m_inner rows ATTACH to
each pms_inner row, but pms_inner itself is still included in the result set
— COUNT(pms_inner.stat_id) counts every pms_inner row regardless of whether
its m_inner date fell in the window.  This produced values like 219.

Fix: move the date range filter into a WHERE condition, or rewrite as a
correlated subquery so only in-window matches are counted.  The correlated
subquery approach is cleaner and avoids the LEFT JOIN ambiguity.
"""

import logging
from load.postgres import connect
from config.settings import DB_DSN

logger = logging.getLogger(__name__)


def compute_injury_label(conn):
    """Set is_injured_next_30d = TRUE where an injury follows within 30 days."""
    with conn.cursor() as cur:
        cur.execute("""
            UPDATE player_match_stats pms
            SET    is_injured_next_30d = TRUE
            WHERE EXISTS (
                SELECT 1
                FROM matches m
                JOIN injuries i
                ON i.player_id = pms.player_id
                WHERE m.match_id = pms.match_id
                AND i.injury_date >= m.match_date
                AND i.injury_date <= m.match_date + INTERVAL '30 days'
            );
        """)
        updated = cur.rowcount
    conn.commit()
    logger.info("is_injured_next_30d: %d rows set to TRUE", updated)


def compute_days_since_last_injury(conn):
    """Days between each match and the player's most recent prior return_date."""
    with conn.cursor() as cur:
        cur.execute("""
            UPDATE player_match_stats pms
            SET    days_since_last_injury = sub.days_since
            FROM (
                SELECT
                    pms2.stat_id,
                    (m2.match_date - MAX(i.return_date))::INT AS days_since
                FROM player_match_stats pms2
                JOIN matches  m2 ON m2.match_id  = pms2.match_id
                JOIN injuries i  ON i.player_id  = pms2.player_id
                WHERE i.return_date < m2.match_date
                GROUP BY pms2.stat_id, m2.match_date
            ) sub
            WHERE pms.stat_id = sub.stat_id
        """)
        updated = cur.rowcount
    conn.commit()
    logger.info("days_since_last_injury: %d rows updated", updated)


def compute_workload(conn):
    """
    matches_last_30_days and minutes_last_30_days for each player-match row.

    Uses a correlated subquery so the 30-day window filter applies correctly.
    The original LEFT JOIN approach counted rows outside the window.
    """
    with conn.cursor() as cur:
        cur.execute("""
            UPDATE player_match_stats pms
            SET
                matches_last_30_days = sub.match_count,
                minutes_last_30_days = sub.minute_sum
            FROM (
                SELECT
                    pms_outer.stat_id,
                    COUNT(pms_inner.stat_id)                        AS match_count,
                    COALESCE(SUM(pms_inner.minutes_played), 0)::INT AS minute_sum
                FROM player_match_stats pms_outer
                JOIN matches m_outer ON m_outer.match_id = pms_outer.match_id
                -- Inner self-join: only rows strictly before this match date
                -- and within the 30-day window.  Use INNER JOIN + WHERE so
                -- the date predicate filters rows, not just join attachment.
                LEFT JOIN (
                    player_match_stats pms_inner
                    JOIN matches m_inner ON m_inner.match_id = pms_inner.match_id
                ) ON  pms_inner.player_id  = pms_outer.player_id
                  AND pms_inner.stat_id   != pms_outer.stat_id
                  AND m_inner.match_date   >= m_outer.match_date - INTERVAL '30 days'
                  AND m_inner.match_date    < m_outer.match_date
                GROUP BY pms_outer.stat_id
            ) sub
            WHERE pms.stat_id = sub.stat_id
        """)
        updated = cur.rowcount
    conn.commit()
    logger.info("Workload features: %d rows updated", updated)


def run(conn=None):
    if conn is None:
        conn = connect(DB_DSN)

    compute_workload(conn)
    compute_days_since_last_injury(conn)
    compute_injury_label(conn)

    logger.info("compute_labels complete")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    run()