class TeamCache:
    def __init__(self, conn):
        self.conn = conn
        self.cache = {}
        self._load()

    def _load(self):
        with self.conn.cursor() as cur:
            cur.execute("SELECT team_id, statsbomb_team_id FROM teams")
            for tid, sid in cur.fetchall():
                self.cache[sid] = tid

    def get_or_create(self, sb_id, name):
        if sb_id in self.cache:
            return self.cache[sb_id]

        with self.conn.cursor() as cur:
            cur.execute("""
                INSERT INTO teams (team_name, statsbomb_team_id)
                VALUES (%s,%s)
                ON CONFLICT (statsbomb_team_id)
                DO UPDATE SET team_name=EXCLUDED.team_name
                RETURNING team_id
            """, (name, sb_id))
            tid = cur.fetchone()[0]

        self.conn.commit()
        self.cache[sb_id] = tid
        return tid


class PlayerCache:
    def __init__(self, conn):
        self.conn = conn
        self.sb = {}
        self.norm = {}
        self._load()

    def _load(self):
        with self.conn.cursor() as cur:
            cur.execute("SELECT player_id, statsbomb_player_id, norm_name FROM players")
            for pid, sid, nn in cur.fetchall():
                if sid:
                    self.sb[sid] = pid
                if nn:
                    self.norm[nn] = pid

    def get_or_create(self, sb_id, name):
        if sb_id in self.sb:
            return self.sb[sb_id]

        from core.utils import norm_name
        nn = norm_name(name)

        if nn in self.norm:
            return self.norm[nn]

        with self.conn.cursor() as cur:
            cur.execute("""
                INSERT INTO players (statsbomb_player_id, player_name, norm_name)
                VALUES (%s,%s,%s)
                RETURNING player_id
            """, (sb_id, name, nn))
            pid = cur.fetchone()[0]

        self.conn.commit()
        self.sb[sb_id] = pid
        self.norm[nn] = pid
        return pid