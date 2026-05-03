"""
transform/features.py

Vectorised per-player feature aggregation for one match.
"""

import math
import numpy as np
import pandas as pd

_GOAL_X  = 120.0
_GOAL_Y  = 40.0
_PROG_PASS_THRESHOLD  = 25.0
_PROG_CARRY_THRESHOLD = 10.0
_PROG_CARRY_MIN_X     = 48.0


def _vec_dist_to_goal(locs: pd.Series) -> pd.Series:
    def _single(loc):
        if isinstance(loc, (list, tuple)) and len(loc) >= 2:
            try:
                return math.sqrt((loc[0] - _GOAL_X)**2 + (loc[1] - _GOAL_Y)**2)
            except (TypeError, ValueError):
                pass
        return float("nan")
    return locs.map(_single)


def _vec_dist(starts: pd.Series, ends: pd.Series) -> pd.Series:
    def _single(pair):
        a, b = pair
        if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
            try:
                return math.sqrt((b[0]-a[0])**2 + (b[1]-a[1])**2)
            except (TypeError, ValueError, IndexError):
                pass
        return 0.0
    return pd.Series(zip(starts, ends)).map(_single)


def extract_type_col(events: pd.DataFrame) -> pd.Series:
    return events["type"].map(
        lambda t: t.get("name") if isinstance(t, dict) else (t or "")
    )


def extract_player_id_col(events: pd.DataFrame) -> pd.Series:
    return events["player"].map(
        lambda p: p.get("id") if isinstance(p, dict) and isinstance(p.get("id"), int)
        else None
    )


def extract_team_id_col(events: pd.DataFrame) -> pd.Series:
    return events["team"].map(
        lambda t: t.get("id") if isinstance(t, dict) and isinstance(t.get("id"), int)
        else None
    )


def _build_xa_map(events: pd.DataFrame) -> dict[int, float]:
    """
    Build {pass_event_index -> xa} using the direct StatsBomb link:
        pass.assisted_shot_id  ->  shot event uuid  ->  shot.statsbomb_xg

    This is a two-step vectorised lookup with no related_events traversal
    and no heuristic fallbacks.

    Fallback for older spec rows where goal_assist=True but assisted_shot_id
    is absent: use the mean xG of all shots in the match (not 1.0), since we
    know a goal occurred but cannot identify the exact shot.
    """
    type_col = events["type"].map(
        lambda t: t.get("name") if isinstance(t, dict) else (t or "")
    )

    # ------------------------------------------------------------------
    # Step 1: {shot_uuid -> xg} — vectorised, no iterrows
    # ------------------------------------------------------------------
    is_shot = type_col == "Shot"
    shot_xg_dict: dict[str, float] = {}

    if is_shot.any():
        shot_rows = events.loc[is_shot]
        uids = shot_rows["id"]
        xgs  = shot_rows["shot"].map(
            lambda s: float(s.get("statsbomb_xg") or 0.0) if isinstance(s, dict) else 0.0
        )
        shot_xg_dict = dict(zip(uids, xgs))

    mean_shot_xg: float = float(np.mean(list(shot_xg_dict.values()))) if shot_xg_dict else 0.0

    # ------------------------------------------------------------------
    # Step 2: {pass_index -> xa} via pass.assisted_shot_id
    # ------------------------------------------------------------------
    is_pass = type_col == "Pass"
    xa_map: dict[int, float] = {}

    if not is_pass.any():
        return xa_map

    pass_rows = events.loc[is_pass]

    # Extract assisted_shot_id from each pass dict (None when absent)
    assisted_shot_ids = pass_rows["pass"].map(
        lambda p: p.get("assisted_shot_id") if isinstance(p, dict) else None
    )

    # Direct lookup: pass index -> xg of the linked shot
    for idx, shot_uuid in assisted_shot_ids.dropna().items():
        xg = shot_xg_dict.get(shot_uuid)
        if xg is not None:
            xa_map[idx] = xg

    # ------------------------------------------------------------------
    # Fallback: goal_assist=True with no assisted_shot_id (older spec)
    # Use mean match xG — not 1.0 — to avoid systematic overestimation.
    # ------------------------------------------------------------------
    goal_assist_flags = pass_rows["pass"].map(
        lambda p: bool(p.get("goal_assist")) if isinstance(p, dict) else False
    )
    for idx in goal_assist_flags[goal_assist_flags].index:
        if idx not in xa_map:
            xa_map[idx] = mean_shot_xg

    return xa_map


def agg_match_by_player(
    events: pd.DataFrame,
    type_col: pd.Series,
    player_id_col: pd.Series,
) -> dict[int, dict]:
    ev = events.copy(deep=False)
    ev["_type"]      = type_col
    ev["_player_id"] = player_id_col
    ev = ev.dropna(subset=["_player_id"])
    ev["_player_id"] = ev["_player_id"].astype(int)

    # Build xa map once per match across all events (shots and passes both needed)
    xa_map = _build_xa_map(events)

    result: dict[int, dict] = {}
    for pid, pe in ev.groupby("_player_id", sort=False):
        result[pid] = _agg_player_slice(pe, pid, xa_map)
    return result


def _agg_player_slice(pe: pd.DataFrame, pid: int, xa_map: dict) -> dict:
    types = pe["_type"]

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

    # Shots
    shots = int(is_shot.sum())
    xg = goals = 0.0
    if shots:
        shot_col = pe.loc[is_shot, "shot"].map(
            lambda s: s if isinstance(s, dict) else {}
        )
        xg    = float(shot_col.map(lambda s: s.get("statsbomb_xg") or 0.0).sum())
        goals = int(shot_col.map(
            lambda s: 1 if _resolve_name(s.get("outcome")) == "Goal" else 0
        ).sum())

    # Passes
    passes_attempted = int(is_pass.sum())
    passes_completed = assists = key_passes = progressive_passes = 0
    xa = 0.0
    if passes_attempted:
        pass_col = pe.loc[is_pass, "pass"].map(
            lambda p: p if isinstance(p, dict) else {}
        )
        passes_completed = int(pass_col.map(lambda p: p.get("outcome") is None).sum())
        assists          = int(pass_col.map(lambda p: bool(p.get("goal_assist"))).sum())
        key_passes       = int(pass_col.map(
            lambda p: bool(p.get("shot_assist") or p.get("goal_assist"))
        ).sum())

        # xa: sum xa_map values for this player's pass event indices only
        xa = float(sum(xa_map.get(idx, 0.0) for idx in pe.loc[is_pass].index))

        start_locs = pe.loc[is_pass, "location"]
        end_locs   = pass_col.map(lambda p: p.get("end_location"))
        d_start    = _vec_dist_to_goal(start_locs)
        d_end      = _vec_dist_to_goal(end_locs)
        progressive_passes = int(((d_start - d_end) >= _PROG_PASS_THRESHOLD).sum())

    pass_accuracy = (passes_completed / passes_attempted * 100) if passes_attempted else 0.0

    # Carries
    carry_distance = progressive_carries = 0.0
    if is_carry.sum():
        carry_col  = pe.loc[is_carry, "carry"].map(
            lambda c: c if isinstance(c, dict) else {}
        )
        start_locs = pe.loc[is_carry, "location"]
        end_locs   = carry_col.map(lambda c: c.get("end_location"))
        carry_distance = float(_vec_dist(start_locs, end_locs).sum())

        d_start = _vec_dist_to_goal(start_locs)
        d_end   = _vec_dist_to_goal(end_locs)
        end_x   = end_locs.map(
            lambda e: e[0] if isinstance(e, (list, tuple)) and len(e) >= 1 else 0.0
        )
        mask = ((d_start - d_end) >= _PROG_CARRY_THRESHOLD) & (end_x >= _PROG_CARRY_MIN_X)
        progressive_carries = int(mask.sum().item())

    # Dribbles
    dribbles_completed = 0
    if is_dribble.sum():
        drib_col = pe.loc[is_dribble, "dribble"].map(
            lambda d: d if isinstance(d, dict) else {}
        )
        dribbles_completed = int(
            drib_col.map(lambda d: _resolve_name(d.get("outcome")) == "Complete").sum()
        )

    # Tackles
    tackles = 0
    if is_duel.sum():
        duel_col = pe.loc[is_duel, "duel"].map(
            lambda d: d if isinstance(d, dict) else {}
        )
        tackles = int(
            duel_col.map(lambda d: _resolve_name(d.get("type")) == "Tackle").sum()
        )

    interceptions = int(is_interception.sum())
    clearances    = int(is_clearance.sum())
    pressures     = int(is_pressure.sum())

    # Discipline
    yellow_cards = red_cards = 0
    if is_bad_beh.sum():
        bb_col = pe.loc[is_bad_beh, "bad_behaviour"].map(
            lambda b: b if isinstance(b, dict) else {}
        )
        card_names   = bb_col.map(lambda b: _resolve_name(b.get("card")) or "")
        yellow_cards = int(card_names.isin({"Yellow Card", "Second Yellow"}).sum())
        red_cards    = int((card_names == "Red Card").sum())

    # Sub + minutes
    sub_minute = None
    if is_sub.sum():
        sub_mins = pe.loc[is_sub, "minute"]
        if len(sub_mins):
            sub_minute = int(sub_mins.iloc[0])

    last_minute    = int(pe["minute"].max()) if len(pe) and "minute" in pe.columns else 0
    minutes_played = sub_minute if sub_minute is not None else last_minute

    return {
        "goals":               int(goals),
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
        "progressive_carries": int(progressive_carries),
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


def agg_player_events(events: pd.DataFrame, pid: int, event_type_fn) -> dict:
    """Backwards-compatible single-player entry point."""
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
    if isinstance(val, dict):
        return val.get("name", "")
    return val or ""