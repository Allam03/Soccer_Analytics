# Soccer_Analytics/pipelines/ingest_statsbomb.py
import logging
import time
from typing import Optional
from pathlib import Path

from extract import statsbomb_local as sb
from transform.features import agg_player_events
from transform.schema import player_id, team_id, event_type
from load.postgres import insert_stats
from config.settings import DATA_ROOT

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def run(
    conn, 
    team_cache, 
    player_cache, 
    batch_size: int = 1000,
    validate_events: bool = True,
    track_metrics: bool = True
):
    """
    Process Statsbomb data with performance optimizations and validation.
    
    Args:
        conn: Database connection
        team_cache: Team cache instance
        player_cache: Player cache instance
        batch_size: Number of rows to process at once
        validate_events: Enable event validation
        track_metrics: Enable metrics tracking
    """
    
    if track_metrics:
        from metrics import MetricsTracker
        tracker = MetricsTracker()
        tracker.start()
    
    comps = sb.competitions()
    
    logger.info(f"Found {len(comps)} competitions to process")
    
    for _, comp in comps.iterrows():
        comp_start = time.time()
        comp_name = comp.get('name', f'Comp {comp["competition_id"]}')
        
        logger.info(f"\n{'='*60}")
        logger.info(f"Processing: {comp_name} (Season: {comp.get('season_id', 'N/A')})")
        logger.info(f"{'='*60}")
        
        try:
            matches = sb.matches(comp["competition_id"], comp["season_id"])
            
            if len(matches) == 0:
                logger.warning(f"No matches found for {comp_name}")
                continue
            
            logger.info(f"Found {len(matches)} matches")
            
            processed_matches = 0
            total_events = 0
            failed_events = 0
            
            for match_idx, match in matches.iterrows():
                match_id = match["match_id"]
                logger.info(f"  Processing match {match_idx + 1}/{len(matches)}: {match_id}")
                
                start_time = time.time()
                
                # Fetch events once per match
                try:
                    events = sb.events(match_id)
                except Exception as e:
                    logger.error(f"Failed to load events for match {match_id}: {e}")
                    failed_events += len(events) if 'events' in dir() else 0
                    continue
                
                total_events += len(events)
                
                if len(events) == 0:
                    logger.warning(f"  No events for match {match_id}")
                    continue
                
                # Batch process events
                stat_rows = []
                batch = []
                batch_count = 0
                
                for ev_idx, ev in events.iterrows():
                    try:
                        # Validate event data
                        if validate_events:
                            if not ev.get('player'):
                                logger.debug(f"  Match {match_id} Event {ev_idx}: No player data, skipping")
                                continue
                            if not ev.get('type'):
                                logger.debug(f"  Match {match_id} Event {ev_idx}: No event type, skipping")
                                continue
                        
                        pid = player_id(ev)
                        if not pid:
                            continue
                        
                        # Get or create player
                        pg_player = player_cache.get_or_create(pid, ev.get("player", {}).get("name"))
                        team = team_id(ev)
                        
                        # Get all events once (already loaded)
                        # Use the same events for all players in this match
                        # Filter events for this specific player
                        pe = events[events["player"].apply(
                            lambda x: isinstance(x, dict) and x.get("id") == pid
                        )]
                        
                        stats = agg_player_events(pe, pid, event_type)
                        
                        stat_rows.append((
                            pg_player,
                            match_id,
                            team,
                            stats["goals"],
                            stats["assists"],
                            stats["shots"],
                            stats["xg"],
                            stats["xa"],
                            stats["passes_attempted"],
                            stats["passes_completed"],
                            stats["pass_accuracy"],
                        ))
                        
                        batch.append(stat_rows[-1])
                        batch_count += 1
                        
                    except Exception as e:
                        logger.error(f"  Match {match_id} Event {ev_idx}: Error processing event - {e}")
                        failed_events += 1
                        continue
                
                # Insert stats in batches if needed
                if stat_rows:
                    try:
                        insert_stats(conn, stat_rows)
                    except Exception as e:
                        logger.error(f"  Failed to insert stats for match {match_id}: {e}")
                        failed_events += len(stat_rows)
                
                elapsed = time.time() - start_time
                logger.info(f"  Match {match_id}: {len(events)} events processed in {elapsed:.2f}s")
                logger.info(f"  Stats inserted: {len(stat_rows)}, Failed: {failed_events}")
                
                processed_matches += 1
                if track_metrics:
                    tracker.match_processed(match_id)
            
            logger.info(f"Completed {comp_name}: {processed_matches}/{len(matches)} matches")
            logger.info(f"Total events processed: {total_events}")
            logger.info(f"Total failed events: {failed_events}")
            logger.info(f"Time elapsed: {time.time() - comp_start:.2f}s")
            
            if track_metrics:
                tracker.competition_completed(comp_name, len(matches), total_events, failed_events, time.time() - comp_start)
                
        except Exception as e:
            logger.error(f"Failed to process {comp_name}: {e}")
            if track_metrics:
                tracker.competition_failed(comp_name, str(e))
            raise
    
    if track_metrics:
        tracker.finish()
        logger.info(f"\n{'='*60}")
        logger.info("Pipeline Summary:")
        for metric, value in tracker.get_metrics():
            logger.info(f"  {metric}: {value}")
        logger.info(f"{'='*60}")
    
    conn.close()