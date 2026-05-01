import math


# Goal mouth x-coordinate in StatsBomb's 120x80 pitch coordinate system.
# Used to compute "progressive" carries and passes (moving toward goal).
_GOAL_X = 120.0


def _dist(loc_a, loc_b):
    """Euclidean distance between two [x, y] locations."""
    if not (isinstance(loc_a, (list, tuple)) and isinstance(loc_b, (list, tuple))):
        return 0.0
    try:
        return math.sqrt((loc_b[0] - loc_a[0]) ** 2 + (loc_b[1] - loc_a[1]) ** 2)
    except (IndexError, TypeError):
        return 0.0


def _dist_to_goal(loc):
    """Distance from a location to the centre of the goal (120, 40)."""
    if not isinstance(loc, (list, tuple)) or len(loc) < 2:
        return None
    try:
        return math.sqrt((loc[0] - _GOAL_X) ** 2 + (loc[1] - 40.0) ** 2)
    except (IndexError, TypeError):
        return None


def agg_player_events(events, pid, event_type_fn):
    """
    Aggregate all StatsBomb events for one player in one match into a flat
    stats dictionary that maps directly onto player_match_stats columns.

    Parameters
    ----------
    events : pd.DataFrame
        All events for the match (not pre-filtered -- this function filters
        by player internally so it can also inspect substitution events that
        reference the player as the player being replaced).
    pid : int
        StatsBomb player id.
    event_type_fn : callable
        Function that returns the event-type name string from a row.
    """
    # Filter to events belonging to this player
    pe = events[events["player"].apply(
        lambda x: isinstance(x, dict) and x.get("id") == pid
    )]

    stats = {
        # attacking
        "goals":               0,
        "assists":             0,
        "shots":               0,
        "xg":                  0.0,
        "xa":                  0.0,
        "key_passes":          0,
        # passing
        "passes_attempted":    0,
        "passes_completed":    0,
        "pass_accuracy":       0.0,
        "progressive_passes":  0,
        # carrying
        "carry_distance":      0.0,
        "progressive_carries": 0,
        # dribbling
        "dribbles_completed":  0,
        # defending
        "tackles":             0,
        "interceptions":       0,
        "clearances":          0,
        "pressures":           0,
        # discipline
        "yellow_cards":        0,
        "red_cards":           0,
        # time
        "minutes_played":      0,
        "sub_minute":          None,
    }

    last_minute = 0

    for _, ev in pe.iterrows():
        t = event_type_fn(ev)
        minute = ev.get("minute") or 0
        if minute > last_minute:
            last_minute = minute

        # ---- Shots --------------------------------------------------------
        if t == "Shot":
            stats["shots"] += 1
            shot = ev.get("shot") or {}
            stats["xg"] += shot.get("statsbomb_xg") or 0.0
            outcome = shot.get("outcome")
            if isinstance(outcome, dict):
                outcome = outcome.get("name", "")
            if outcome == "Goal":
                stats["goals"] += 1

        # ---- Passes -------------------------------------------------------
        elif t == "Pass":
            pass_data = ev.get("pass") or {}
            stats["passes_attempted"] += 1

            # A completed pass has no outcome (field absent / None)
            outcome = pass_data.get("outcome")
            if outcome is None:
                stats["passes_completed"] += 1

            # Key pass: pass that directly precedes a shot
            if pass_data.get("shot_assist") or pass_data.get("goal_assist"):
                stats["key_passes"] += 1

            # Assist: pass flagged as goal_assist
            if pass_data.get("goal_assist"):
                stats["assists"] += 1

            # xA
            stats["xa"] += pass_data.get("xA") or 0.0

            # Progressive pass: end location moves ball >=25 yards closer
            # to the opponent goal than the start location.
            start_loc = ev.get("location")
            end_loc   = pass_data.get("end_location")
            d_start   = _dist_to_goal(start_loc)
            d_end     = _dist_to_goal(end_loc)
            if d_start is not None and d_end is not None:
                if (d_start - d_end) >= 25:
                    stats["progressive_passes"] += 1

        # ---- Carries ------------------------------------------------------
        elif t == "Carry":
            carry_data = ev.get("carry") or {}
            start_loc  = ev.get("location")
            end_loc    = carry_data.get("end_location")
            dist       = _dist(start_loc, end_loc)
            stats["carry_distance"] += dist

            # Progressive carry: end location moves ball >=10 yards closer
            # to opponent goal AND ends in the final 60% of the pitch (x>=48)
            d_start = _dist_to_goal(start_loc)
            d_end   = _dist_to_goal(end_loc)
            if d_start is not None and d_end is not None:
                end_x = end_loc[0] if isinstance(end_loc, (list, tuple)) else 0
                if (d_start - d_end) >= 10 and end_x >= 48:
                    stats["progressive_carries"] += 1

        # ---- Dribbles -----------------------------------------------------
        elif t == "Dribble":
            dribble_data = ev.get("dribble") or {}
            outcome = dribble_data.get("outcome")
            if isinstance(outcome, dict):
                outcome = outcome.get("name", "")
            if outcome == "Complete":
                stats["dribbles_completed"] += 1

        # ---- Duels / Tackles ----------------------------------------------
        elif t == "Duel":
            duel_data = ev.get("duel") or {}
            dtype = duel_data.get("type")
            if isinstance(dtype, dict):
                dtype = dtype.get("name", "")
            if dtype == "Tackle":
                stats["tackles"] += 1

        # ---- Interceptions ------------------------------------------------
        elif t == "Interception":
            stats["interceptions"] += 1

        # ---- Clearances ---------------------------------------------------
        elif t == "Clearance":
            stats["clearances"] += 1

        # ---- Pressure -----------------------------------------------------
        elif t == "Pressure":
            stats["pressures"] += 1

        # ---- Discipline (Bad Behaviour) -----------------------------------
        elif t == "Bad Behaviour":
            bb = ev.get("bad_behaviour") or {}
            card = bb.get("card")
            if isinstance(card, dict):
                card = card.get("name", "")
            if card in ("Yellow Card", "Second Yellow"):
                stats["yellow_cards"] += 1
            elif card == "Red Card":
                stats["red_cards"] += 1

        # ---- Substitution -------------------------------------------------
        elif t == "Substitution":
            stats["sub_minute"] = minute

    # ---- Minutes played ---------------------------------------------------
    # If the player was substituted off, minutes played = sub_minute.
    # Otherwise use the last minute they had an event recorded.
    if stats["sub_minute"] is not None:
        stats["minutes_played"] = stats["sub_minute"]
    else:
        stats["minutes_played"] = last_minute

    # ---- Pass accuracy ----------------------------------------------------
    if stats["passes_attempted"] > 0:
        stats["pass_accuracy"] = (
            stats["passes_completed"] / stats["passes_attempted"] * 100
        )

    return stats