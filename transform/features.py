def agg_player_events(events, pid, event_type_fn):

    pe = events[events["player"].apply(
        lambda x: isinstance(x, dict) and x.get("id") == pid
    )]

    stats = {
        "goals": 0,
        "assists": 0,
        "shots": 0,
        "xg": 0,
        "xa": 0,
        "passes_attempted": 0,
        "passes_completed": 0,
    }

    for _, ev in pe.iterrows():
        t = event_type_fn(ev)

        if t == "shot":
            stats["shots"] += 1
            stats["xg"] += ev.get("shot", {}).get("statsbomb_xg") or 0
            if ev.get("shot", {}).get("outcome") == "goal":
                stats["goals"] += 1

        elif t == "pass":
            stats["passes_attempted"] += 1
            if ev.get("pass", {}).get("outcome") in [None, "", "incomplete"]:
                stats["passes_completed"] += 1

            if "goal_assist" in (ev.get("pass", {}).get("flags") or []):
                stats["assists"] += 1

            stats["xa"] += ev.get("pass", {}).get("xA", 0) or 0

    if stats["passes_attempted"]:
        stats["pass_accuracy"] = stats["passes_completed"] / stats["passes_attempted"] * 100
    else:
        stats["pass_accuracy"] = 0

    return stats