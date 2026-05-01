# Soccer_Analytics/utils/validate_schema.py
import logging
from typing import List, Tuple, Optional

from load.postgres import connect

logging.getLogger('schema').setLevel(logging.WARNING)


def validate_schema(conn) -> Tuple[bool, List[str], List[str]]:
    """
    Validate that the database schema matches expectations.
    
    Returns:
        (success: bool, warnings: List[str], errors: List[str])
    """
    warnings = []
    errors = []
    
    try:
        with conn.cursor() as cur:
            # Check teams table
            cur.execute("""
                SELECT column_name, data_type, is_nullable
                FROM information_schema.columns
                WHERE table_schema = 'public'
                AND table_name = 'teams'
                ORDER BY ordinal_position
            """)
            
            expected_teams_cols = {
                'team_id': ('int4', False),
                'team_name': ('varchar', True),
                'statsbomb_team_id': ('int4', False),
            }
            
            existing_teams_cols = {row[0]: (row[1], row[2]) for row in cur.fetchall()}
            
            for col_name, (expected_type, is_nullable) in expected_teams_cols.items():
                if col_name not in existing_teams_cols:
                    errors.append(f"teams table missing required column: {col_name}")
                else:
                    actual_type, is_nullable_actual = existing_teams_cols[col_name]
                    if actual_type.upper() != expected_type.upper():
                        warnings.append(f"teams.{col_name} has type {actual_type}, expected {expected_type}")
            
            # Check players table
            cur.execute("""
                SELECT column_name, data_type, is_nullable
                FROM information_schema.columns
                WHERE table_schema = 'public'
                AND table_name = 'players'
                ORDER BY ordinal_position
            """)
            
            expected_players_cols = {
                'player_id': ('int4', False),
                'player_name': ('varchar', True),
                'norm_name': ('varchar', True),
                'statsbomb_player_id': ('int4', True),
            }
            
            existing_players_cols = {row[0]: (row[1], row[2]) for row in cur.fetchall()}
            
            for col_name, (expected_type, is_nullable) in expected_players_cols.items():
                if col_name not in existing_players_cols:
                    errors.append(f"players table missing required column: {col_name}")
                else:
                    actual_type, is_nullable_actual = existing_players_cols[col_name]
                    if actual_type.upper() != expected_type.upper():
                        warnings.append(f"players.{col_name} has type {actual_type}, expected {expected_type}")
            
            # Check player_match_stats table
            cur.execute("""
                SELECT column_name, data_type, is_nullable
                FROM information_schema.columns
                WHERE table_schema = 'public'
                AND table_name = 'player_match_stats'
                ORDER BY ordinal_position
            """)
            
            expected_stats_cols = {
                'player_id': ('int4', False),
                'match_id': ('int4', False),
                'team_id': ('int4', True),
                'goals': ('int4', True),
                'assists': ('int4', True),
                'shots': ('int4', True),
                'xg': ('numeric', True),
                'xa': ('numeric', True),
                'passes_attempted': ('int4', True),
                'passes_completed': ('int4', True),
                'pass_accuracy': ('numeric', True),
            }
            
            existing_stats_cols = {row[0]: (row[1], row[2]) for row in cur.fetchall()}
            
            for col_name, (expected_type, is_nullable) in expected_stats_cols.items():
                if col_name not in existing_stats_cols:
                    errors.append(f"player_match_stats table missing required column: {col_name}")
                else:
                    actual_type, is_nullable_actual = existing_stats_cols[col_name]
                    if actual_type.upper() != expected_type.upper():
                        warnings.append(f"player_match_stats.{col_name} has type {actual_type}, expected {expected_type}")
            
            # Check indexes
            cur.execute("""
                SELECT indexname, indexdef
                FROM pg_indexes
                WHERE schemaname = 'public'
                AND (indexdef LIKE '%player_match_stats%' OR indexdef LIKE '%teams%' OR indexdef LIKE '%players%')
            """)
            
            existing_indexes = {row[0]: row[1] for row in cur.fetchall()}
            
            required_indexes = [
                ('player_match_stats_player_id_match_id_team_id_idx', 
                 'CREATE INDEX player_match_stats_player_id_match_id_team_id_idx ON player_match_stats (player_id, match_id, team_id)'),
                ('teams_statsbomb_team_id_idx', 'CREATE INDEX teams_statsbomb_team_id_idx ON teams (statsbomb_team_id)'),
                ('players_statsbomb_player_id_idx', 'CREATE INDEX players_statsbomb_player_id_idx ON players (statsbomb_player_id)'),
                ('players_norm_name_idx', 'CREATE INDEX players_norm_name_idx ON players (norm_name)'),
            ]
            
            for idx_name, idx_def in required_indexes:
                if idx_name not in existing_indexes:
                    errors.append(f"Missing required index: {idx_name}")
                    warnings.append(f"  {idx_def}")
            
            # Check for orphaned records
            cur.execute("""
                SELECT COUNT(*) as count
                FROM player_match_stats pms
                LEFT JOIN teams t ON pms.team_id = t.team_id
                WHERE t.team_id IS NULL
            """)
            
            orphaned_teams = cur.fetchone()[0]
            if orphaned_teams > 0:
                warnings.append(f"Found {orphaned_teams} player_match_stats records with non-existent teams")
            
            cur.execute("""
                SELECT COUNT(*) as count
                FROM player_match_stats pms
                LEFT JOIN players p ON pms.player_id = p.player_id
                WHERE p.player_id IS NULL
            """)
            
            orphaned_players = cur.fetchone()[0]
            if orphaned_players > 0:
                warnings.append(f"Found {orphaned_players} player_match_stats records with non-existent players")
            
            conn.commit()
            
            success = len(errors) == 0
            logging.warning(f"Schema validation: {'PASSED' if success else 'FAILED'}")
            for warning in warnings:
                logging.warning(f"  Warning: {warning}")
            for error in errors:
                logging.error(f"  Error: {error}")
            
            return success, warnings, errors
            
    except Exception as e:
        logging.error(f"Schema validation failed with exception: {e}")
        return False, [], [str(e)]