"""
transform/features.py

Vectorised per-player feature aggregation for one match.

Key design
----------
- Caller does ONE groupby over the match events DataFrame and passes each
  player's pre-filtered slice here, eliminating repeated full-DataFrame
  scans.
- No iterrows() anywhere -- all counting and arithmetic uses pandas
  boolean masks and .sum() / .max().
- _dist_to_goal and progressive thresholds are applied with numpy
  vectorised ops on the location columns.
"""

import math
import numpy as np
import pandas as pd

_GOAL_X   = 120.0
_GOAL_Y   = 40.0
_PROG_PASS_THRESHOLD   = 25.0   # yards closer to goal
_PROG_CARRY_THRESHOLD  = 10.0   # yards closer to goal
_PROG_CARRY_MIN_X      = 48.0   # must end in attacking half


# ---------------------------------------------------------------------------
# Vectorised location helpers
# ---------------------------------------------------------------------------

def _vec_dist_to_goal(locs: pd.Series) -> pd.Series:
    """
    Given a Series of [x, y] lists (StatsBomb location format), return a
    float Series of distances to the goal centre (120, 40).
    Returns NaN where the location is missing or malformed.
    """
    def _single(loc):
        if isinstance(loc, (list, tuple)) and len(loc) >= 2:
            try:
                return math.sqrt((loc[0] - _GOAL_X) ** 2 + (loc[1] - _GOAL_Y) ** 2)
            except (TypeError, ValueError):
                pass
        return float("nan")
    return locs.map(_single)


def _vec_dist(starts: pd.Series, ends: pd.Series) -> pd.Series:
    """Euclidean distance between two Series of [x, y] locations."""
    def _single(pair):
        a, b = pair
        if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
            try:
                return math.sqrt((b[0] - a[0]) ** 2 + (b[1] - a[1]) ** 2)
            except (TypeError, ValueError, IndexError):
                pass
        return 0.0
    return pd.Series(zip(starts, ends)).map(_single)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def extract_type_col(events: pd.DataFrame) -> pd.Series:
    """
    Return a plain string Series of event-type names from the nested
    type dict column.  Call this ONCE per match and pass the result into
    agg_match_by_player().
    """
    return events["type"].map(
        lambda t: t.get("name") if isinstance(t, dict) else (t or "")
    )


def extract_player_id_col(events: pd.DataFrame) -> pd.Series:
    """Return integer player-id Series (NaN for events with no player)."""
    return events["player"].map(
        lambda p: p.get("id") if isinstance(p, dict) and isinstance(p.get("id"), int)
        else None
    )


def extract_team_id_col(events: pd.DataFrame) -> pd.Series:
    """Return integer team-id Series."""
    return events["team"].map(
        lambda t: t.get("id") if isinstance(t, dict) and isinstance(t.get("id"), int)
        else None
    )


def agg_match_by_player(
    events: pd.DataFrame,
    type_col: pd.Series,
    player_id_col: pd.Series,
) -> dict[int, dict]:
    """
    Aggregate all events in ONE match into a per-player stats dict.

    Parameters
    ----------
    events        : Full match events DataFrame.
    type_col      : Pre-extracted string Series of event type names
                    (from extract_type_col).
    player_id_col : Pre-extracted player-id Series
                    (from extract_player_id_col).

    Returns
    -------
    {sb_player_id -> stats_dict}
    """
    # Attach the flat columns so we can groupby without re-extracting
    ev = events.copy(deep=False)
    ev["_type"]      = type_col
    ev["_player_id"] = player_id_col

    # Drop events with no player (ball receipts, half-starts, etc.)
    ev = ev.dropna(subset=["_player_id"])
    ev["_player_id"] = ev["_player_id"].astype(int)

    result: dict[int, dict] = {}

    for pid, pe in ev.groupby("_player_id", sort=False):
        result[pid] = _agg_player_slice(pe, pid)

    return result


def _agg_player_slice(pe: pd.DataFrame, pid: int) -> dict:
    """
    Aggregate one player's events (already filtered to that player).
    All operations are mask + .sum() -- no Python-level loops over rows.
    """
    types = pe["_type"]

    # ------------------------------------------------------------------
    # Pre-build type masks (each used multiple times)
    # ------------------------------------------------------------------
    is_shot         = types == "Shot"
    is_pass         = types == "Pass"
    is_carry        = types == "Carry"
    is_dribble      = types == "Dribble"
    is_duel         = types == "Duel"
    is_interception = types == "Interception"
    is_clearance    = types == "Clearance"
    is_pressure     = types == "Pressure"
    is_bad_beh      = types == "Bad Behaviour"
    is_sub          = types == "Substitution"

    # ------------------------------------------------------------------
    # Shots
    # ------------------------------------------------------------------
    shots = int(is_shot.sum())
    xg    = 0.0
    goals = 0
    if shots:
        shot_col = pe.loc[is_shot, "shot"].map(
            lambda s: s if isinstance(s, dict) else {}
        )
        xg    = float(shot_col.map(lambda s: s.get("statsbomb_xg") or 0.0).sum())
        goals = int(shot_col.map(
            lambda s: 1 if _resolve_name(s.get("outcome")) == "Goal" else 0
        ).sum())

    # ------------------------------------------------------------------
    # Passes
    # ------------------------------------------------------------------
    passes_attempted = int(is_pass.sum())
    passes_completed = assists = key_passes = progressive_passes = 0
    xa = 0.0
    if passes_attempted:
        pass_col = pe.loc[is_pass, "pass"].map(
            lambda p: p if isinstance(p, dict) else {}
        )
        # Completed = outcome is None
        passes_completed  = int(pass_col.map(lambda p: p.get("outcome") is None).sum())
        assists           = int(pass_col.map(lambda p: bool(p.get("goal_assist"))).sum())
        key_passes        = int(pass_col.map(
            lambda p: bool(p.get("shot_assist") or p.get("goal_assist"))
        ).sum())
        xa                = float(pass_col.map(lambda p: p.get("xA") or 0.0).sum())

        # Progressive passes -- vectorised distance calculation
        start_locs = pe.loc[is_pass, "location"]
        end_locs   = pass_col.map(lambda p: p.get("end_location"))
        d_start    = _vec_dist_to_goal(start_locs)
        d_end      = _vec_dist_to_goal(end_locs)
        progressive_passes = int(((d_start - d_end) >= _PROG_PASS_THRESHOLD).sum())

    pass_accuracy = (passes_completed / passes_attempted * 100) if passes_attempted else 0.0

    # ------------------------------------------------------------------
    # Carries
    # ------------------------------------------------------------------
    carry_distance      = 0.0
    progressive_carries = 0
    if is_carry.sum():
        carry_col  = pe.loc[is_carry, "carry"].map(
            lambda c: c if isinstance(c, dict) else {}
        )
        start_locs = pe.loc[is_carry, "location"]
        end_locs   = carry_col.map(lambda c: c.get("end_location"))
        dists      = _vec_dist(start_locs, end_locs)
        carry_distance = float(dists.sum())

        d_start = _vec_dist_to_goal(start_locs)
        d_end   = _vec_dist_to_goal(end_locs)
        end_x   = end_locs.map(
            lambda e: e[0] if isinstance(e, (list, tuple)) and len(e) >= 1 else 0.0
        )
        mask = ((d_start - d_end) >= _PROG_CARRY_THRESHOLD) & (end_x >= _PROG_CARRY_MIN_X)
        progressive_carries = int(mask.sum().item())

    # ------------------------------------------------------------------
    # Dribbles
    # ------------------------------------------------------------------
    dribbles_completed = 0
    if is_dribble.sum():
        drib_col = pe.loc[is_dribble, "dribble"].map(
            lambda d: d if isinstance(d, dict) else {}
        )
        dribbles_completed = int(
            drib_col.map(lambda d: _resolve_name(d.get("outcome")) == "Complete").sum()
        )

    # ------------------------------------------------------------------
    # Duels / Tackles
    # ------------------------------------------------------------------
    tackles = 0
    if is_duel.sum():
        duel_col = pe.loc[is_duel, "duel"].map(
            lambda d: d if isinstance(d, dict) else {}
        )
        tackles = int(
            duel_col.map(lambda d: _resolve_name(d.get("type")) == "Tackle").sum()
        )

    # ------------------------------------------------------------------
    # Simple counts
    # ------------------------------------------------------------------
    interceptions = int(is_interception.sum())
    clearances    = int(is_clearance.sum())
    pressures     = int(is_pressure.sum())

    # ------------------------------------------------------------------
    # Discipline
    # ------------------------------------------------------------------
    yellow_cards = red_cards = 0
    if is_bad_beh.sum():
        bb_col = pe.loc[is_bad_beh, "bad_behaviour"].map(
            lambda b: b if isinstance(b, dict) else {}
        )
        card_names    = bb_col.map(lambda b: _resolve_name(b.get("card")) or "")
        yellow_cards  = int(card_names.isin({"Yellow Card", "Second Yellow"}).sum())
        red_cards     = int((card_names == "Red Card").sum())

    # ------------------------------------------------------------------
    # Substitution + minutes played
    # ------------------------------------------------------------------
    sub_minute = None
    if is_sub.sum():
        sub_minutes = pe.loc[is_sub, "minute"]
        if len(sub_minutes):
            sub_minute = int(sub_minutes.iloc[0])

    last_minute = int(pe["minute"].max()) if len(pe) and "minute" in pe.columns else 0
    minutes_played = sub_minute if sub_minute is not None else last_minute

    return {
        "goals":               goals,
        "assists":             assists,
        "shots":               shots,
        "xg":                  xg,
        "xa":                  xa,
        "key_passes":          key_passes,
        "passes_attempted":    passes_attempted,
        "passes_completed":    passes_completed,
        "pass_accuracy":       pass_accuracy,
        "progressive_passes":  progressive_passes,
        "carry_distance":      carry_distance,
        "progressive_carries": progressive_carries,
        "dribbles_completed":  dribbles_completed,
        "tackles":             tackles,
        "interceptions":       interceptions,
        "clearances":          clearances,
        "pressures":           pressures,
        "yellow_cards":        yellow_cards,
        "red_cards":           red_cards,
        "minutes_played":      minutes_played,
        "sub_minute":          sub_minute,
    }


# ---------------------------------------------------------------------------
# Backwards-compatible single-player entry point
# (kept so any code that still calls agg_player_events() keeps working)
# ---------------------------------------------------------------------------

def agg_player_events(events: pd.DataFrame, pid: int, event_type_fn) -> dict:
    """
    Single-player aggregation.  Delegates to the vectorised path.
    Kept for backwards compatibility -- prefer agg_match_by_player()
    when processing a full match.
    """
    type_col      = extract_type_col(events)
    player_id_col = extract_player_id_col(events)
    all_stats     = agg_match_by_player(events, type_col, player_id_col)
    return all_stats.get(pid, _empty_stats())


def _empty_stats() -> dict:
    return {
        "goals": 0, "assists": 0, "shots": 0,
        "xg": 0.0, "xa": 0.0, "key_passes": 0,
        "passes_attempted": 0, "passes_completed": 0,
        "pass_accuracy": 0.0, "progressive_passes": 0,
        "carry_distance": 0.0, "progressive_carries": 0,
        "dribbles_completed": 0,
        "tackles": 0, "interceptions": 0, "clearances": 0, "pressures": 0,
        "yellow_cards": 0, "red_cards": 0,
        "minutes_played": 0, "sub_minute": None,
    }


def _resolve_name(val) -> str:
    """Extract .name from a StatsBomb ref-dict, or return the value as-is."""
    if isinstance(val, dict):
        return val.get("name", "")
    return val or ""