import psycopg2
from psycopg2.extras import execute_values


def connect(dsn):
    return psycopg2.connect(dsn)


def insert_stats(conn, rows):
    with conn.cursor() as cur:
        execute_values(cur, """
            INSERT INTO player_match_stats (
                player_id, match_id, team_id,
                goals, assists, shots, xg, xa,
                passes_attempted, passes_completed, pass_accuracy
            )
            VALUES %s
            ON CONFLICT DO NOTHING
        """, rows)

    conn.commit()
    