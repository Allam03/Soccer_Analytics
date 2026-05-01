# Soccer_Analytics/utils/validate_data.py
import logging
import statistics
from typing import Dict, Any, Optional, List
import pandas as pd

logging.getLogger('data').setLevel(logging.WARNING)


def validate_event(ev: Dict[str, Any]) -> bool:
    """
    Validate a single event.
    
    Args:
        ev: Event dictionary from Statsbomb
    
    Returns:
        True if event is valid, False otherwise
    """
    # Check required fields
    if not ev.get('player'):
        return False
    if not ev.get('type'):
        return False
    if not ev.get('match'):
        return False
    
    # Validate player has ID
    player_data = ev.get('player', {})
    if not isinstance(player_data, dict):
        return False
    if 'id' not in player_data:
        return False
    
    # Validate event type
    event_type_data = ev.get('type', {})
    if isinstance(event_type_data, dict):
        if 'name' not in event_type_data:
            return False
    else:
        # It's already a string
        pass
    
    # Validate match has ID
    match_data = ev.get('match', {})
    if not isinstance(match_data, dict):
        return False
    if 'id' not in match_data:
        return False
    
    return True


def validate_batch(events: pd.DataFrame) -> Dict[str, Any]:
    """
    Validate a batch of events and return statistics
    
    Args:
        events: DataFrame of events from Statsbomb
    
    Returns:
        dict with validation stats
    """
    valid_count = 0
    invalid_count = 0
    invalid_reasons = []
    
    for idx, ev in events.iterrows():
        if not validate_event(ev):
            invalid_count += 1
            reason = "Missing required fields"
            if not ev.get('player'):
                reason = "Missing player data"
            elif not ev.get('type'):
                reason = "Missing event type"
            elif not ev.get('match'):
                reason = "Missing match data"
            invalid_reasons.append((idx, reason))
        else:
            valid_count += 1
            
    stats = {
        'valid_count': valid_count,
        'invalid_count': invalid_count,
        'valid_rate': valid_count / max(1, valid_count + invalid_count),
        'invalid_reasons': invalid_reasons[:10], # Log top 10 invalid reasons
    }
    
    if invalid_count > 0:
        logging.warning(f"Batch validation: {invalid_count} invalid events found")
        for idx, reason in invalid_reasons[:5]:
            logging.debug(f"  Invalid event {idx}: {reason}")
            
    return stats


def validate_aggregated_stats(stats: Dict[str, Any]) -> bool:
    """
    Validate aggregated stats before insertion.
    
    Args:
        stats: Dictionary of aggregated stats
    
    Returns:
        True if stats are valid, False otherwise
    """
    required_stats = ['goals', 'assists', 'shots', 'xg', 'xa', 
                      'passes_attempted', 'passes_completed', 'pass_accuracy']
    
    for key in required_stats:
        if key not in stats:
            logging.warning(f"Missing stat key: {key}")
            return False
    
    # Validate numeric fields
    numeric_fields = ['goals', 'assists', 'shots', 'xg', 'xa', 
                      'passes_attempted', 'passes_completed']
    for field in numeric_fields:
        value = stats.get(field, 0)
        if not isinstance(value, (int, float)):
            logging.warning(f"Stat {field} is not numeric: {type(value)}")
            return False
        if value < 0:
            logging.warning(f"Stat {field} is negative: {value}")
            return False
    
    # Validate pass accuracy is between 0 and 1
    accuracy = stats.get('pass_accuracy', 0)
    if not (0 <= accuracy <= 1):
        logging.warning(f"Pass accuracy out of range: {accuracy}")
        return False
    
    return True


# Export functions
__all__ = ['validate_event', 'validate_batch', 'validate_aggregated_stats']