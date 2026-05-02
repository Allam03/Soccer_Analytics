"""
core/caches.py

In-memory look-up caches for teams and players.

Performance changes vs original
--------------------------------
- get_or_create() no longer commits after every new row.
  New rows are accumulated in a pending dict and flushed to the DB in a
  single execute_values() + one commit when flush() is called.
- The pipeline calls flush() once per match (teams) or once per
  competition (players) -- caller controls granularity.
- All reads are pure dict lookups (O(1)), zero DB round-trips for already-
  seen entities.
"""

from psycopg2.extras import execute_values
from core.utils import norm_name


class TeamCache:
    def __init__(self, conn):
        self.conn    = conn
        self.cache   = {}        # statsbomb_team_id -> pg team_id
        self._pending: dict[int, str] = {}   # sb_id -> name (not yet in DB)
        self._load()

    def _load(self):
        with self.conn.cursor() as cur:
            cur.execute("SELECT team_id, statsbomb_team_id FROM teams")
            for tid, sid in cur.fetchall():
                self.cache[sid] = tid

    def get_or_create(self, sb_id: int, name: str) -> int | None:
        """Return pg team_id.  New teams are buffered until flush()."""
        if sb_id in self.cache:
            return self.cache[sb_id]
        # Stage for batch insert; return sentinel None until flushed
        self._pending[sb_id] = name
        return None   # caller must call flush() before using the ID

    def flush(self):
        """Write all pending teams to DB in one round-trip."""
        if not self._pending:
            return
        rows = [(name, sb_id) for sb_id, name in self._pending.items()]
        with self.conn.cursor() as cur:
            execute_values(cur, """
                INSERT INTO teams (team_name, statsbomb_team_id)
                VALUES %s
                ON CONFLICT (statsbomb_team_id)
                DO UPDATE SET team_name = EXCLUDED.team_name
                RETURNING team_id, statsbomb_team_id
            """, rows)
            for tid, sid in cur.fetchall():
                self.cache[sid] = tid
        self.conn.commit()
        self._pending.clear()

    def resolve(self, sb_id: int) -> int:
        """Return pg team_id -- call only after flush()."""
        return self.cache[sb_id]


class PlayerCache:
    def __init__(self, conn):
        self.conn  = conn
        self.sb    = {}      # statsbomb_player_id -> pg player_id
        self.norm  = {}      # norm_name           -> pg player_id
        self._pending: dict[int, tuple[str, str]] = {}  # sb_id -> (name, nn)
        self._load()

    def _load(self):
        with self.conn.cursor() as cur:
            cur.execute("SELECT player_id, statsbomb_player_id, norm_name FROM players")
            for pid, sid, nn in cur.fetchall():
                if sid:
                    self.sb[sid] = pid
                if nn:
                    self.norm[nn] = pid

    def get_or_create(self, sb_id: int, name: str) -> int | None:
        """Return pg player_id.  New players are buffered until flush()."""
        if sb_id in self.sb:
            return self.sb[sb_id]

        nn = norm_name(name)
        if nn in self.norm:
            pid = self.norm[nn]
            # Back-fill statsbomb_player_id -- stage it, will be done in flush()
            self.sb[sb_id] = pid
            self._pending_backfill = getattr(self, "_pending_backfill", {})
            self._pending_backfill[sb_id] = pid
            return pid

        if sb_id not in self._pending:
            self._pending[sb_id] = (name, nn)
        return None   # caller must call flush() before using the ID

    def flush(self):
        """Write all pending players to DB in one round-trip."""
        # Back-fill statsbomb IDs for name-matched players
        backfill = getattr(self, "_pending_backfill", {})
        if backfill:
            with self.conn.cursor() as cur:
                execute_values(cur, """
                    UPDATE players AS p
                    SET statsbomb_player_id = v.sb_id
                    FROM (VALUES %s) AS v(sb_id, player_id)
                    WHERE p.player_id = v.player_id
                      AND p.statsbomb_player_id IS NULL
                """, [(sb_id, pid) for sb_id, pid in backfill.items()])
            self._pending_backfill = {}

        if not self._pending:
            if backfill:
                self.conn.commit()
            return

        rows = [(sb_id, name, nn) for sb_id, (name, nn) in self._pending.items()]
        with self.conn.cursor() as cur:
            execute_values(cur, """
                INSERT INTO players (statsbomb_player_id, player_name, norm_name)
                VALUES %s
                ON CONFLICT (statsbomb_player_id) DO NOTHING
                RETURNING player_id, statsbomb_player_id, norm_name
            """, rows)
            for pid, sid, nn in cur.fetchall():
                self.sb[sid]   = pid
                self.norm[nn]  = pid

        self.conn.commit()
        self._pending.clear()

    def resolve(self, sb_id: int) -> int:
        """Return pg player_id -- call only after flush()."""
        return self.sb[sb_id]