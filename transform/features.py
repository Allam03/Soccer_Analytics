"""
transform/features.py

Vectorised per-player feature aggregation for one match.

xa fix
------
StatsBomb open data does not include an 'xA' field directly on the pass dict
in the way the commercial feed does.  In the open data, expected assists are
derived by looking at passes that preceded a shot and using the shot's
statsbomb_xg as the xA value for the passer.

Approach:
1. Build a mapping from (shot event index) -> xg for all shot events.
2. For each completed pass, look up the next shot event in the same
   possession sequence using the 'shot_assist' flag or by checking whether
   the very next shot event belongs to the same possession.  If found,
   assign that shot's xg as the pass's xA contribution.

Simpler approximation used here (avoids tracking possession chains):
- A pass with shot_assist=True is a key pass.
- Its xA is set to the average xg of all shots in the match weighted by
  the pass location distance to goal -- this is an approximation.
- A pass with goal_assist=True gets xA = 1.0 (the goal happened).

Actually the cleanest and most accurate approach for open data:
StatsBomb stores 'shot_assist': True on the pass, and separately stores
the shot's statsbomb_xg.  We can match them via the 'related_events' list
on each event.  However, related_events requires iterating pairs.

Practical fix: use the 'through_ball', 'switch', 'cross', and technique
flags to detect key passes, and for xA use the shot xg of shots that
immediately follow (within the same team's possession, next 2 events).
This is what most open-data pipelines do.

For correctness we use: if pass has shot_assist flag -> find the linked
shot via related_events and assign its xg as xa.  Falls back to 0 if
related_events is missing (older spec).
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


def _build_xa_map(events: pd.DataFrame) -> dict:
    """
    Build {event_uuid -> xa_value} for all passes that assisted a shot.

    Strategy: for each shot event, check its 'related_events' list.
    Any UUID in that list that belongs to a Pass event is the key pass;
    assign the shot's statsbomb_xg as xa for that pass.

    Falls back to the 'xA' field on the pass dict if present (commercial feed).
    Returns a dict from event index (integer) to xa float.
    """
    # Build uuid -> (index, type_name, xg) for shots
    shot_by_uuid: dict[str, float] = {}
    pass_idx_by_uuid: dict[str, int] = {}

    for idx, row in events.iterrows():
        t = row.get("type")
        t_name = t.get("name") if isinstance(t, dict) else t

        uid = row.get("id")  # StatsBomb event UUID

        if t_name == "Shot":
            shot_data = row.get("shot") or {}
            xg = shot_data.get("statsbomb_xg") or 0.0
            if uid:
                shot_by_uuid[uid] = xg
            # Also register related events pointing back to the assist pass
            related = row.get("related_events") or []
            for rel_uid in related:
                # We'll resolve these below
                shot_by_uuid.setdefault(f"__shot_for_{rel_uid}", xg)

        if t_name == "Pass":
            if uid:
                pass_idx_by_uuid[uid] = idx

    # Build the xa_map: pass_index -> xa
    xa_map: dict[int, float] = {}

    for idx, row in events.iterrows():
        t = row.get("type")
        t_name = t.get("name") if isinstance(t, dict) else t
        if t_name != "Pass":
            continue

        pass_dict = row.get("pass") or {}

        # Method 1: explicit xA field (commercial feed)
        explicit_xa = pass_dict.get("xA")
        if explicit_xa is not None:
            xa_map[idx] = float(explicit_xa)
            continue

        # Method 2: related_events on this pass -> find linked shot xg
        uid = row.get("id")
        if uid:
            shot_xg = shot_by_uuid.get(f"__shot_for_{uid}")
            if shot_xg is not None:
                xa_map[idx] = shot_xg
                continue

        # Method 3: shot_assist flag -> use related_events on the shot side
        if pass_dict.get("shot_assist") or pass_dict.get("goal_assist"):
            related = row.get("related_events") or []
            for rel_uid in related:
                xg = shot_by_uuid.get(rel_uid)
                if xg is not None:
                    xa_map[idx] = xg
                    break
            else:
                # goal_assist but no linked xg found: use 1.0 as proxy
                if pass_dict.get("goal_assist"):
                    xa_map[idx] = 1.0

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

    # Build xa map once per match (uses all events including shots)
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

        # xa: sum from pre-built xa_map (indexed by event dataframe index)
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