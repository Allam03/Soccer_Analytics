"""
models/eval_utils.py

Shared honest-evaluation helpers for the supervised models.

Why this exists
---------------
The original models used plain shuffled k-fold cross-validation. Because the
rows are not independent -- the two team-rows of a match, every minute-snapshot
of a match, and every player-season of the same player share information -- a
shuffled split puts near-duplicates of a row in both the train and test folds.
That leaks and inflates the reported metrics (most starkly Model 5B in-game:
0.89 shuffled vs 0.64 grouped-by-match).

These helpers provide:
  * grouped_cv()             -- (Stratified)GroupKFold cross-validation grouped
                                by match_id, so no match straddles the
                                train/test boundary.
  * grouped_cv_multi()       -- same grouping, but returns R2/MAE/RMSE in one
                                pass (cross_validate instead of cross_val_score)
                                for model-comparison tables.
  * holdout_season()         -- a single out-of-time test on a held-out season
                                (FIFA World Cup 2022 by default), the
                                strictest generalisation check available here.
  * leave_one_season_out()   -- repeats holdout_season() for every season in
                                turn, so the 2022 result isn't read as the only
                                out-of-time evidence when one season (2015/16)
                                dominates the row count.
  * attach_season()          -- map match_id -> season onto a feature frame.

The StatsBomb free data is one full La Liga season (2015/16) + five
Barcelona-only La Liga seasons + two World Cups, so BALANCED_SEASONS marks the
slice with genuine team diversity (used to de-bias Model 5).
"""

import numpy as np
from sklearn.base import clone
from sklearn.metrics import get_scorer, mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import (
    cross_val_score, cross_validate, GroupKFold, StratifiedGroupKFold,
)

# Held-out season for the out-of-time generalisation test.
TEST_SEASON = "2022"  # FIFA World Cup 2022 -- recent, balanced, out-of-distribution

# Seasons with league-wide team diversity (no Barcelona over-representation).
BALANCED_SEASONS = {"2015/2016", "2018", "2022"}


def season_of_matches(conn) -> dict:
    with conn.cursor() as cur:
        cur.execute("SELECT match_id, season FROM matches")
        return {int(m): s for m, s in cur.fetchall()}


def attach_season(df, conn, match_col: str = "match_id"):
    """Return a copy of df with a 'season' column mapped from match_id."""
    df = df.copy()
    df["season"] = df[match_col].map(season_of_matches(conn))
    return df


def grouped_cv(estimator, X, y, groups, scoring, n_splits: int = 5,
               stratified: bool = False):
    """
    Cross-validate with GroupKFold (or StratifiedGroupKFold for classifiers)
    grouped by `groups` (match_id). Returns (mean, std).
    """
    cv = (
        StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=42)
        if stratified else GroupKFold(n_splits=n_splits)
    )
    scores = cross_val_score(estimator, X, y, cv=cv, groups=groups, scoring=scoring)
    return scores.mean(), scores.std()


def holdout_season(estimator, X, y, seasons, scoring, test_season: str = TEST_SEASON):
    """
    Train on every season except `test_season`, evaluate once on it.
    Returns (score, n_test). Returns (None, n_test) if the split is degenerate.
    X may be a DataFrame (boolean row-masking works for both df and ndarray).
    """
    seasons = np.asarray(seasons)
    y = np.asarray(y)
    test = seasons == test_season
    if test.sum() == 0 or (~test).sum() == 0:
        return None, int(test.sum())
    est = clone(estimator).fit(X[~test], y[~test])
    score = get_scorer(scoring)(est, X[test], y[test])
    return float(score), int(test.sum())


def grouped_cv_multi(estimator, X, y, groups, n_splits: int = 5):
    """
    GroupKFold cross-validation returning R2, MAE and RMSE in a single pass
    (one fit per fold instead of one fit per metric). Returns a dict:
        {"r2_mean", "r2_std", "mae_mean", "mae_std", "rmse_mean", "rmse_std"}
    """
    cv = GroupKFold(n_splits=n_splits)
    scoring = {
        "r2":   "r2",
        "mae":  "neg_mean_absolute_error",
        "rmse": "neg_root_mean_squared_error",
    }
    res = cross_validate(estimator, X, y, cv=cv, groups=groups, scoring=scoring)
    return {
        "r2_mean":   float(res["test_r2"].mean()),
        "r2_std":    float(res["test_r2"].std()),
        "mae_mean":  float(-res["test_mae"].mean()),
        "mae_std":   float(res["test_mae"].std()),
        "rmse_mean": float(-res["test_rmse"].mean()),
        "rmse_std":  float(res["test_rmse"].std()),
    }


def leave_one_season_out(estimator, X, y, seasons, min_test_rows: int = 30):
    """
    Repeats holdout_season() for every distinct season present in `seasons`.

    Train on all-other-seasons, test once on the held-out season -- run in
    turn for each season. Seasons with fewer than `min_test_rows` rows are
    skipped (too few points for a stable R2 estimate) and reported as
    skipped rather than silently dropped.

    Returns a list of dicts: {"season", "n_test", "r2", "mae", "rmse"} or
    {"season", "n_test", "skipped": True} for thin seasons.
    """
    seasons_arr = np.asarray(seasons)
    y = np.asarray(y)
    results = []
    for season in sorted(set(seasons_arr.tolist())):
        test = seasons_arr == season
        n_test = int(test.sum())
        if n_test < min_test_rows or (~test).sum() == 0:
            results.append({"season": season, "n_test": n_test, "skipped": True})
            continue
        est = clone(estimator).fit(X[~test], y[~test])
        pred = est.predict(X[test])
        results.append({
            "season": season,
            "n_test": n_test,
            "r2":   float(r2_score(y[test], pred)),
            "mae":  float(mean_absolute_error(y[test], pred)),
            "rmse": float(np.sqrt(mean_squared_error(y[test], pred))),
        })
    return results
