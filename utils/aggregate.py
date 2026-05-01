# Soccer_Analytics/utils/aggregate.py
import logging
from typing import Dict, Any, List, Optional
import pandas as pd
import numpy as np

logging.getLogger('aggregate').setLevel(logging.DEBUG)


def aggregate_player_events(events: pd.DataFrame, 
                           player_id: int,
                           event_type: str = 'all') -> Dict[str, Any]:
    """
    Aggregate all events for a single player across a match.
    
    Args:
        events: DataFrame containing all events for the match
        player_id: The player ID to aggregate
        event_type: 'all', 'attack', 'defence', 'setpiece', etc.
    
    Returns:
        Dictionary of aggregated stats for the player
    """
    # Filter events for this player
    player_events = events[events['player_id'] == player_id]
    
    if player_events.empty:
        # Return empty stats with defaults
        return {
            'goals': 0,
            'assists': 0,
            'shots': 0,
            'xg': 0.0,
            'xa': 0.0,
            'passes_attempted': 0,
            'passes_completed': 0,
            'pass_accuracy': 0.0,
            'key_passes': 0,
            'shots_on_target': 0,
            'tackles': 0,
            'interceptions': 0,
            'clearances': 0,
            'yellow_cards': 0,
            'red_cards': 0,
            'minutes_played': 0,
            'positions': []
        }
    
    # Aggregate basic stats
    stats = {
        'goals': player_events['player_id'].count() if 'goals' in player_events.columns else 0,
        'assists': 0,
        'shots': 0,
        'xg': 0.0,
        'xa': 0.0,
        'passes_attempted': 0,
        'passes_completed': 0,
        'pass_accuracy': 0.0,
        'key_passes': 0,
        'shots_on_target': 0,
        'tackles': 0,
        'interceptions': 0,
        'clearances': 0,
        'yellow_cards': 0,
        'red_cards': 0,
        'minutes_played': 0,
        'positions': []
    }
    
    # Event type based aggregation
    if event_type == 'attack':
        attack_types = ['shot', 'goal', 'penalty_miss', 'penalty_saved', 'assisted_goal',
                       'key_pass', 'shot_on_target']
        player_events = player_events[
            player_events['event_type_name'].apply(
                lambda x: x in attack_types if isinstance(x, str) else False
            )
        ]
        
    elif event_type == 'defence':
        defence_types = ['tackle', 'interception', 'clearance', 'header_clearance',
                        'block', 'fouls', 'offsides']
        player_events = player_events[
            player_events['event_type_name'].apply(
                lambda x: x in defence_types if isinstance(x, str) else False
            )
        ]
    
    elif event_type == 'setpiece':
        setpiece_types = ['freekick', 'corner', 'throw_in', 'goal_kick', 'penalty']
        player_events = player_events[
            player_events['event_type_name'].apply(
                lambda x: x in setpiece_types if isinstance(x, str) else False
            )
        ]
    
    elif event_type == 'all':
        pass # Use all events
    
    # Count events for this player
    stats['events_count'] = len(player_events)
    
    # Aggregate specific metrics
    if not player_events.empty:
        stats['events_count'] = len(player_events)
        
        # Goals (event_type_name == 'goal' with outcome == 'on')
        goals = player_events[
            (player_events['event_type_name'] == 'goal') & 
            (player_events.get('outcome', '').astype(str).str.contains('on', na=False))
        ]
        stats['goals'] = len(goals)
        
        # Assists
        assists = player_events[
            (player_events['event_type_name'] == 'assisted_goal')
        ]
        stats['assists'] = len(assists)
        
        # Shots
        shots = player_events[
            (player_events['event_type_name'] == 'shot')
        ]
        stats['shots'] = len(shots)
        
        # Expected Goals
        if 'expected_goal_value' in player_events.columns:
            stats['xg'] = player_events['expected_goal_value'].sum()
        
        # Expected Assists
        if 'expected_assist_value' in player_events.columns:
            stats['xa'] = player_events['expected_assist_value'].sum()
        
        # Passes
        pass_events = player_events[
            player_events['event_type_name'].isin(['pass', 'accurate_pass', 'inaccurate_pass'])
        ]
        if not pass_events.empty:
            stats['passes_attempted'] = pass_events['player_id'].nunique() # Approximation for now
            stats['passes_completed'] = len(pass_events[pass_events.get('accurate', False)])
            if stats['passes_attempted'] > 0:
                stats['pass_accuracy'] = stats['passes_completed'] / stats['passes_attempted']
        
        # Defensive stats
        tackle_events = player_events[
            player_events['event_type_name'] == 'tackle'
        ]
        stats['tackles'] = len(tackle_events)
        
        intercept_events = player_events[
            player_events['event_type_name'] == 'interception'
        ]
        stats['interceptions'] = len(intercept_events)
        
        clearance_events = player_events[
            player_events['event_type_name'] == 'clearance'
        ]
        stats['clearances'] = len(clearance_events)
        
        # Cards
        yellow_events = player_events[
            (player_events['event_type_name'] == 'foul') & 
            (player_events.get('outcome', '').astype(str).str.contains('yellow', na=False))
        ]
        stats['yellow_cards'] = len(yellow_events)
        
        red_events = player_events[
            (player_events['event_type_name'] == 'foul') & 
            (player_events.get('outcome', '').astype(str).str.contains('red', na=False))
        ]
        stats['red_cards'] = len(red_events)
        
        # Minutes played (approximate based on last event time)
        if not player_events.empty:
            first_time = min(player_events['timestamp'].min(), default=0)
            last_time = max(player_events['timestamp'].max(), default=0)
            # Note: This is a rough approximation. Ideally, use player minute data.
            stats['minutes_played'] = max(0, int((last_time - first_time).total_seconds() / 60))
        
        return stats
    return {}


def aggregate_match_stats(events: pd.DataFrame, match_id: int) -> Dict[str, Any]:
    """
    Aggregate all stats for a single match.
    
    Args:
        events: DataFrame of all events for a match
        match_id: The match ID
    
    Returns:
        Dictionary containing match-level aggregated stats
    """
    match_events = events[events['match_id'] == match_id]
    
    if match_events.empty:
        return {
            'match_id': match_id,
            'home_team': None,
            'away_team': None,
            'home_score': None,
            'away_score': None,
            'total_events': 0,
            'home_goals': 0,
            'away_goals': 0,
            'possession_home': 0,
            'possession_away': 0
        }
    
    # Basic counts
    total_events = len(match_events)
    home_goals = len(match_events[
        (match_events['event_type_name'] == 'goal') & 
        (match_events.get('outcome', '').astype(str).str.contains('home', na=False))
    ])
    away_goals = len(match_events[
        (match_events['event_type_name'] == 'goal') & 
        (match_events.get('outcome', '').astype(str).str.contains('away', na=False))
    ])
    
    return {
        'match_id': match_id,
        'home_team': None, # Would need team data join
        'away_team': None,
        'home_score': home_goals,
        'away_score': away_goals,
        'total_events': total_events,
        'home_goals': home_goals,
        'away_goals': away_goals,
        'possession_home': 0, # Would need matchbox data join
        'possession_away': 0
    }


def process_batch(events_df: pd.DataFrame) -> List[Dict[str, Any]]:
    """
    Process a large batch of events.
    
    Args:
        events_df: Pandas DataFrame of Statsbomb events
    
    Returns:
        List of aggregated player stats
    """
    player_stats = []
    match_stats = {}
    
    # Group by match for efficiency
    matches = events_df.groupby('match_id')
    
    for match_id, match_events in matches:
        # Validate batch first (optional, expensive for large batches)
        # validate_batch(match_events)
        
        match_agg = aggregate_match_stats(match_events, match_id)
        match_stats[match_id] = match_agg
        
        # Process each player in the match
        unique_players = match_events['player_id'].unique()
        
        for player_id in unique_players:
            player_events = match_events[match_events['player_id'] == player_id]
            try:
                player_stat = aggregate_player_events(player_events, player_id)
                player_stat['match_id'] = match_id
                player_stat['player_id'] = player_id
                player_stats.append(player_stat)
            except Exception as e:
                logging.error(f"Error processing player {player_id} in match {match_id}: {e}")
                # Create empty stat for this player to ensure they have a record
                empty_stat = {
                    'match_id': match_id,
                    'player_id': player_id,
                    'goals': 0,
                    'assists': 0,
                    'shots': 0,
                    'xg': 0.0,
                    'xa': 0.0,
                    'passes_attempted': 0,
                    'passes_completed': 0,
                    'pass_accuracy': 0.0,
                    'tackles': 0,
                    'interceptions': 0,
                    'clearances': 0,
                    'yellow_cards': 0,
                    'red_cards': 0,
                    'minutes_played': 0,
                    'positions': []
                }
                player_stats.append(empty_stat)
    
    return player_stats, match_stats


__all__ = ['aggregate_player_events', 'aggregate_match_stats', 'process_batch']