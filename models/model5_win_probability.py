"""
models/model5_win_probability.py

Model 5: Win Probability Modeling
Type: Multiclass classification  (win / draw / loss)
Objective: Predict match outcome using pre-match team stats and
           in-game cumulative features.

Two sub-models
--------------
A. Pre-match model  -- uses season-to-date averages as of the match date
B. In-game model    -- uses minute-by-minute cumulative stats
                       (requires minute_by_minute_cumulative_stats to be
                        computed; see note below)

Note on minute-by-minute data
------------------------------
True minute-by-minute snapshots require replaying each event in order and
saving running totals -- a separate compute step not yet built.  This model
scaffold trains on full-match aggregate features, which is a valid starting
point.  The in-game sub-model slot is included as a stub to extend later.

Features (pre-match / full-match)
  - rolling avg xg, shots, passes_attempted, pass_accuracy,
    tackles, pressures (team level, last 5 matches)
  - red_cards in this match
  - substitutions made in this match
  - is_home flag

Target: result (win=2, draw=1, loss=0)
"""

import logging
from typing import Dict, Any

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import classification_report
import joblib

logger = logging.getLogger(__name__)

RESULT_MAP = {"win": 2, "draw": 1, "loss": 0}

FEATURES = [
    "avg_xg_last5",
    "avg_shots_last5",
    "avg_passes_last5",
    "avg_pass_acc_last5",
    "avg_tackles_last5",
    "avg_pressures_last5",
    "red_cards_match",
    "subs_made",
    "is_home",
]


def load_features(conn) -> pd.DataFrame:
    """
    Build team-match rows with rolling 5-match averages and in-match
    disciplinary/substitution events.
    """
    # Step 1: aggregate team-level stats per match
    query = """
        SELECT
            pms.match_id,
            pms.team_id,
            m.match_date,
            m.home_team_id,
            MAX(pms.result)                              AS result,
            SUM(pms.xg)                                  AS team_xg,
            SUM(pms.shots)                               AS team_shots,
            AVG(pms.passes_attempted)                    AS team_passes,
            AVG(pms.pass_accuracy)                       AS team_pass_acc,
            SUM(pms.tackles)                             AS team_tackles,
            SUM(pms.pressures)                           AS team_pressures,
            SUM(pms.red_cards)                           AS red_cards,
            COUNT(pms.sub_minute) FILTER
                (WHERE pms.sub_minute IS NOT NULL)        AS subs_made
        FROM player_match_stats pms
        JOIN matches m ON m.match_id = pms.match_id
        WHERE pms.result IS NOT NULL
        GROUP BY pms.match_id, pms.team_id, m.match_date, m.home_team_id
        ORDER BY m.match_date
    """
    with conn.cursor() as cur:
        cur.execute(query)
        cols = [d[0] for d in cur.description]
        rows = cur.fetchall()
    df = pd.DataFrame(rows, columns=cols)

    if df.empty:
        logger.warning("No rows returned -- check that result column is populated")
        return df

    # Step 2: compute rolling 5-match averages per team (excluding current match)
    df = df.sort_values(["team_id", "match_date"])

    for col, alias in [
        ("team_xg",       "avg_xg_last5"),
        ("team_shots",    "avg_shots_last5"),
        ("team_passes",   "avg_passes_last5"),
        ("team_pass_acc", "avg_pass_acc_last5"),
        ("team_tackles",  "avg_tackles_last5"),
        ("team_pressures","avg_pressures_last5"),
    ]:
        df[alias] = (
            df.groupby("team_id")[col]
            .transform(lambda s: s.shift(1).rolling(5, min_periods=1).mean())
        )

    df["is_home"]         = (df["team_id"] == df["home_team_id"]).astype(int)
    df["red_cards_match"] = df["red_cards"].fillna(0)
    df["subs_made"]       = df["subs_made"].fillna(0)

    # Drop rows where rolling features are not yet available (first match per team)
    df = df.dropna(subset=["avg_xg_last5"])

    return df


def encode_labels(df: pd.DataFrame) -> np.ndarray:
    return df["result"].map(RESULT_MAP).values


def run(conn, output_dir: str = "artifacts/model5") -> Dict[str, Any]:
    import os
    os.makedirs(output_dir, exist_ok=True)

    logger.info("Model 5: loading features ...")
    df = load_features(conn)

    if df.empty:
        logger.error("Empty feature set -- cannot train Model 5")
        return {}

    logger.info("  %d team-match rows", len(df))
    logger.info("  Class distribution:\n%s", df["result"].value_counts())

    X = df[FEATURES].fillna(0).values
    y = encode_labels(df)

    scaler = StandardScaler()
    X_sc   = scaler.fit_transform(X)

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # Logistic Regression baseline
    lr = LogisticRegression(
        class_weight="balanced", max_iter=1000, random_state=42, multi_class="multinomial"
    )
    lr_acc = cross_val_score(lr, X_sc, y, cv=cv, scoring="accuracy")
    logger.info("LR  accuracy (5-fold): %.3f +/- %.3f", lr_acc.mean(), lr_acc.std())

    # Random Forest
    rf = RandomForestClassifier(
        n_estimators=300, max_depth=6, class_weight="balanced",
        random_state=42, n_jobs=-1
    )
    rf_acc = cross_val_score(rf, X_sc, y, cv=cv, scoring="accuracy")
    logger.info("RF  accuracy (5-fold): %.3f +/- %.3f", rf_acc.mean(), rf_acc.std())

    # Gradient Boosting (primary)
    gbc = GradientBoostingClassifier(
        n_estimators=300, max_depth=4, learning_rate=0.05,
        random_state=42
    )
    gbc_acc = cross_val_score(gbc, X_sc, y, cv=cv, scoring="accuracy")
    logger.info("GBC accuracy (5-fold): %.3f +/- %.3f", gbc_acc.mean(), gbc_acc.std())

    # Fit on full data
    lr.fit(X_sc, y)
    rf.fit(X_sc, y)
    gbc.fit(X_sc, y)

    logger.info("\nClassification report (train set -- for diagnostics only):\n%s",
                classification_report(y, gbc.predict(X_sc),
                                      target_names=["loss", "draw", "win"]))

    importances = pd.Series(gbc.feature_importances_, index=FEATURES)
    logger.info("GBC feature importances:\n%s",
                importances.sort_values(ascending=False))

    joblib.dump(scaler, f"{output_dir}/scaler.pkl")
    joblib.dump(lr,     f"{output_dir}/lr.pkl")
    joblib.dump(rf,     f"{output_dir}/rf.pkl")
    joblib.dump(gbc,    f"{output_dir}/gbc.pkl")
    df.to_parquet(f"{output_dir}/features.parquet", index=False)

    logger.info("Model 5 artefacts saved to %s", output_dir)
    return {"lr": lr, "rf": rf, "gbc": gbc, "scaler": scaler}


if __name__ == "__main__":
    import psycopg2
    from config.settings import DB_DSN
    logging.basicConfig(level=logging.INFO)
    conn = psycopg2.connect(DB_DSN)
    run(conn)
    conn.close()