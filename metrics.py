# Soccer_Analytics/metrics.py
import time
import logging
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, Any, Optional

logging.getLogger('metrics').setLevel(logging.INFO)


@dataclass
class MetricsTracker:
    """Track pipeline performance metrics."""
    
    metrics: Dict[str, Any] = field(default_factory=lambda: {
        'matches_processed': 0,
        'total_events': 0,
        'failed_events': 0,
        'total_time': 0,
        'matches_by_competition': defaultdict(int),
        'events_by_match': defaultdict(int),
        'failed_by_match': defaultdict(int),
        'completions': 0,
        'failures': 0,
    })
    
    match_start_times: Dict[str, float] = field(default_factory=dict)
    
    def start(self):
        """Start tracking."""
        self.metrics['start_time'] = time.time()
    
    def finish(self):
        """Finish tracking and compute final metrics."""
        end_time = time.time()
        self.metrics['total_time'] = end_time - self.metrics.get('start_time', end_time)
        self.metrics['end_time'] = end_time
    
    def match_processed(self, match_id: str):
        """Record a successfully processed match."""
        self.metrics['matches_processed'] += 1
        self.metrics['matches_by_competition'][self._current_competition] += 1
        self.metrics['events_by_match'][match_id] = self.metrics['total_events']
    
    def record_event(self):
        """Record an event (increment counters)."""
        self.metrics['total_events'] += 1
    
    def record_failure(self):
        """Record a failure (increment counters)."""
        self.metrics['failed_events'] += 1
    
    def competition_completed(self, comp_name: str, matches_count: int, events: int, failed: int, duration: float):
        """Record a completed competition."""
        self.metrics['completions'] += 1
        self._current_competition = comp_name
        self.metrics['matches_by_competition'][comp_name] = matches_count
        self.metrics['total_events'] += events
        self.metrics['failed_events'] += failed
        self.metrics['competitions_duration'] = duration
    
    def competition_failed(self, comp_name: str, error: str):
        """Record a failed competition."""
        self.metrics['failures'] += 1
        self.metrics['failures_reason'] = f"{comp_name}: {error}"
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get formatted metrics for logging."""
        metrics = dict(self.metrics)
        # Convert defaultdicts to regular dicts
        for key in ['matches_by_competition', 'events_by_match', 'failed_by_match']:
            metrics[key] = dict(metrics[key])
        
        # Add derived metrics
        if metrics['total_events'] > 0:
            metrics['events_per_match'] = metrics['total_events'] / max(1, metrics['matches_processed'])
        else:
            metrics['events_per_match'] = 0
        
        if metrics['matches_processed'] > 0:
            metrics['avg_time_per_match'] = metrics['total_time'] / metrics['matches_processed']
        else:
            metrics['avg_time_per_match'] = 0
        
        metrics['success_rate'] = (1 - metrics['failed_events'] / max(1, metrics['total_events'])) * 100 if metrics['total_events'] > 0 else 100
        
        return metrics