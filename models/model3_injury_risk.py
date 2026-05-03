"""
models/model3_injury_risk.py

Model 3: Injury Risk Prediction
Type: Binary classification
Target: is_injured_next_30d
Algorithms: XGBoost (primary), Random Forest, Logistic Regression (baseline)

Features
--------
minutes_played, matches_last_30_days, minutes_last_30_days,
days_since_last_injury, age_at_match, sub_minute,
xg, xa, pressures, tackles, carry_distance,
interceptions, clearances

Schema change
-------------
Workload and injury label columns have moved from player_match_stats to
player_match_features.  The load_features query now JOINs both tables.

Requires
--------
- ingest_injuries.py has run (injuries table populated)
- compute_labels.py has run (player_match_features rows exist with
  is_injured_next_30d set, workload features filled)
- players.date_of_birth populated from Transfermarkt
"""

import logging
from typing import Dict, Any

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import (
    classification_report, roc_auc_score, average_precision_score
)
from sklearn.utils.class_weight import compute_class_weight
import joblib

logger = logging.getLogger(__name__)

FEATURES = [
    "minutes_played",
    "matches_last_30_days",
    "minutes_last_30_days",
    "days_since_last_injury",
    "age_at_match",
    "sub_minute_flag",        # 1 if sub_minute IS NOT NULL
    "xg",
    "xa",
    "pressures",
    "tackles",
    "carry_distance",
    "interceptions",
    "clearances",
]


def load_features(conn) -> pd.DataFrame:
    query = """
        SELECT
            pms.stat_id,
            pms.player_id,
            pms.match_id,
            pms.minutes_played,
            pmf.matches_last_30_days,
            pmf.minutes_last_30_days,
            pmf.days_since_last_injury,
            pms.sub_minute,
            pms.xg,
            pms.xa,
            pms.pressures,
            pms.tackles,
            pms.carry_distance,
            pms.interceptions,
            pms.clearances,
            pmf.is_injured_next_30d        AS label,
            COALESCE(
                EXTRACT(YEAR FROM AGE(m.match_date, p.date_of_birth))::INT,
                25                          -- median fallback when DOB unknown
            ) AS age_at_match
        FROM player_match_stats pms
        JOIN player_match_features pmf ON pmf.stat_id  = pms.stat_id
        JOIN matches m                 ON m.match_id   = pms.match_id
        JOIN players p                 ON p.player_id  = pms.player_id
        WHERE pms.minutes_played >= 1
    """
    with conn.cursor() as cur:
        cur.execute(query)
        cols = [d[0] for d in cur.description]
        rows = cur.fetchall()
    df = pd.DataFrame(rows, columns=cols)

    # Derived feature
    df["sub_minute_flag"] = df["sub_minute"].notna().astype(int)

    # Fill remaining nulls (days_since_last_injury = -1 means no history)
    df["days_since_last_injury"] = df["days_since_last_injury"].fillna(-1)

    return df


def preprocess(df: pd.DataFrame):
    X = df[FEATURES].fillna(0).values
    y = df["label"].astype(int).values
    return X, y


def run(conn, output_dir: str = "artifacts/model3") -> Dict[str, Any]:
    import os
    os.makedirs(output_dir, exist_ok=True)

    logger.info("Model 3: loading features ...")
    df = load_features(conn)
    logger.info("  %d rows  |  positive rate: %.2f%%",
                len(df), df["label"].mean() * 100)

    X, y = preprocess(df)

    scaler = StandardScaler()
    X_sc   = scaler.fit_transform(X)

    # Class weights to handle imbalance (typical injury rate ~5-15%)
    classes = np.unique(y)
    weights = compute_class_weight("balanced", classes=classes, y=y)
    class_weight = dict(zip(classes, weights))
    logger.info("Class weights: %s", class_weight)

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # Baseline: Logistic Regression
    lr = LogisticRegression(
        class_weight=class_weight, max_iter=1000, random_state=42
    )
    lr_auc = cross_val_score(lr, X_sc, y, cv=cv, scoring="roc_auc")
    logger.info("LR AUC-ROC (5-fold): %.3f +/- %.3f", lr_auc.mean(), lr_auc.std())

    # Random Forest
    rf = RandomForestClassifier(
        n_estimators=300, max_depth=8, class_weight=class_weight,
        random_state=42, n_jobs=-1
    )
    rf_auc = cross_val_score(rf, X_sc, y, cv=cv, scoring="roc_auc")
    logger.info("RF  AUC-ROC (5-fold): %.3f +/- %.3f", rf_auc.mean(), rf_auc.std())

    # XGBoost (primary)
    try:
        from xgboost import XGBClassifier
        scale_pos = (y == 0).sum() / max(1, (y == 1).sum())
        xgb = XGBClassifier(
            n_estimators=400, max_depth=4, learning_rate=0.05,
            scale_pos_weight=scale_pos, eval_metric="aucpr",
            random_state=42, n_jobs=-1, verbosity=0,
        )
        xgb_auc = cross_val_score(xgb, X_sc, y, cv=cv, scoring="roc_auc")
        logger.info("XGB AUC-ROC (5-fold): %.3f +/- %.3f",
                    xgb_auc.mean(), xgb_auc.std())
        xgb.fit(X_sc, y)
        joblib.dump(xgb, f"{output_dir}/xgb.pkl")
    except ImportError:
        logger.warning("xgboost not installed -- skipping XGBClassifier")
        xgb = None

    # Fit final models on full data
    lr.fit(X_sc, y)
    rf.fit(X_sc, y)

    # Feature importances from RF
    importances = pd.Series(rf.feature_importances_, index=FEATURES)
    logger.info("RF feature importances:\n%s",
                importances.sort_values(ascending=False))

    joblib.dump(scaler, f"{output_dir}/scaler.pkl")
    joblib.dump(lr,     f"{output_dir}/lr.pkl")
    joblib.dump(rf,     f"{output_dir}/rf.pkl")
    df.to_parquet(f"{output_dir}/features.parquet", index=False)

    logger.info("Model 3 artefacts saved to %s", output_dir)
    return {"lr": lr, "rf": rf, "xgb": xgb, "scaler": scaler}


if __name__ == "__main__":
    import psycopg2
    from config.settings import DB_DSN
    logging.basicConfig(level=logging.INFO)
    conn = psycopg2.connect(DB_DSN)
    run(conn)
    conn.close()