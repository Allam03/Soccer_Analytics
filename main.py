import psycopg2

from config.settings import DB_DSN, DATA_ROOT
from core.caches import TeamCache, PlayerCache
from extract import statsbomb_local as sb
from pipelines.ingest_statsbomb import run


def main():

    sb.set_root(DATA_ROOT)

    conn = psycopg2.connect(DB_DSN)

    team_cache = TeamCache(conn)
    player_cache = PlayerCache(conn)

    run(conn, team_cache, player_cache)

    conn.close()


if __name__ == "__main__":
    main()