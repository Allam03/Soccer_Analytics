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
  * grouped_cv()     -- (Stratified)GroupKFold cross-validation grouped by
                        match_id, so no match straddles the train/test boundary.
  * holdout_season() -- a single out-of-time test on a held-out season
                        (FIFA World Cup 2022 by default), the strictest
                        generalisation check available in this data.
  * attach_season()  -- map match_id -> season onto a feature frame.

The StatsBomb free data is one full La Liga season (2015/16) + five
Barcelona-only La Liga seasons + two World Cups, so BALANCED_SEASONS marks the
slice with genuine team diversity (used to de-bias Model 5).
"""

import numpy as np
from sklearn.base import clone
from sklearn.metrics import get_scorer
from sklearn.model_selection import (
    cross_val_score, GroupKFold, StratifiedGroupKFold,
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
