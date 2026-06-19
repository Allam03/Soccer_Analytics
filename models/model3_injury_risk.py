"""
models/model3_injury_risk.py

Model 3: Injury Risk Prediction
Type: Binary classification
Target: is_injured_next_30d
Algorithms: XGBoost (primary), Random Forest, Logistic Regression (baseline)

Fix: data leakage in cross-validation.
Previously StandardScaler was fit on the full dataset before cross_val_score,
meaning the scaler had seen test-fold values during fitting. This inflated
reported CV AUC scores. The scaler is now wrapped inside a Pipeline so it is
fit only on the training fold during each CV split. The final scaler fit on
the full dataset for inference is unchanged.
"""

import logging
from typing import Dict, Any

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.utils.class_weight import compute_class_weight
import joblib

logger = logging.getLogger(__name__)

FEATURES = [
    "minutes_played",
    "matches_last_30_days",
    "minutes_last_30_days",
    "days_since_last_injury",
    "age_at_match",
    "sub_minute_flag",
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
                25
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

    df["sub_minute_flag"] = df["sub_minute"].notna().astype(int)
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

    classes = np.unique(y)
    weights = compute_class_weight("balanced", classes=classes, y=y)
    class_weight = dict(zip(classes, weights))
    logger.info("Class weights: %s", class_weight)

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # Wrap each estimator in a Pipeline with its own scaler so that
    # cross_val_score fits the scaler only on training folds. This prevents
    # test-fold data from leaking into the scaler's mean/variance, which
    # previously inflated reported CV AUC scores.
    lr_pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(
            class_weight=class_weight, max_iter=1000, random_state=42
        )),
    ])
    rf_pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", RandomForestClassifier(
            n_estimators=300, max_depth=8, class_weight=class_weight,
            random_state=42, n_jobs=-1
        )),
    ])

    lr_auc = cross_val_score(lr_pipe, X, y, cv=cv, scoring="roc_auc")
    logger.info("LR AUC-ROC (5-fold): %.3f +/- %.3f", lr_auc.mean(), lr_auc.std())

    rf_auc = cross_val_score(rf_pipe, X, y, cv=cv, scoring="roc_auc")
    logger.info("RF  AUC-ROC (5-fold): %.3f +/- %.3f", rf_auc.mean(), rf_auc.std())

    xgb_auc = None
    try:
        from xgboost import XGBClassifier
        scale_pos = (y == 0).sum() / max(1, (y == 1).sum())
        xgb_pipe = Pipeline([
            ("scaler", StandardScaler()),
            ("clf", XGBClassifier(
                n_estimators=400, max_depth=4, learning_rate=0.05,
                scale_pos_weight=scale_pos, eval_metric="aucpr",
                random_state=42, n_jobs=-1, verbosity=0,
            )),
        ])
        xgb_auc = cross_val_score(xgb_pipe, X, y, cv=cv, scoring="roc_auc")
        logger.info("XGB AUC-ROC (5-fold): %.3f +/- %.3f",
                    xgb_auc.mean(), xgb_auc.std())
        xgb_pipe.fit(X, y)
        # Save only the fitted estimator and a separate scaler for inference
        # so api_server.py can call scaler.transform() + model.predict_proba()
        # without depending on sklearn Pipeline internals.
        joblib.dump(xgb_pipe.named_steps["clf"],    f"{output_dir}/xgb.pkl")
        joblib.dump(xgb_pipe.named_steps["scaler"], f"{output_dir}/scaler.pkl")
        xgb = xgb_pipe.named_steps["clf"]
    except ImportError:
        logger.warning("xgboost not installed -- skipping XGBClassifier")
        xgb = None

    # Fit final LR and RF on full data for completeness; primary artifact
    # is XGB when available, RF otherwise.
    scaler_final = StandardScaler()
    X_sc = scaler_final.fit_transform(X)

    lr = LogisticRegression(
        class_weight=class_weight, max_iter=1000, random_state=42
    )
    lr.fit(X_sc, y)

    rf = RandomForestClassifier(
        n_estimators=300, max_depth=8, class_weight=class_weight,
        random_state=42, n_jobs=-1
    )
    rf.fit(X_sc, y)

    importances = pd.Series(rf.feature_importances_, index=FEATURES)
    logger.info("RF feature importances:\n%s",
                importances.sort_values(ascending=False))

    # Only write scaler.pkl from the full-data fit if XGB was not available
    # (XGB pipeline already wrote it above with a properly CV-isolated scaler).
    if xgb is None:
        joblib.dump(scaler_final, f"{output_dir}/scaler.pkl")

    joblib.dump(lr, f"{output_dir}/lr.pkl")
    joblib.dump(rf, f"{output_dir}/rf.pkl")
    df.to_parquet(f"{output_dir}/features.parquet", index=False)

    logger.info("Model 3 artifacts saved to %s", output_dir)

    metrics = {
        "lr_roc_auc": float(lr_auc.mean()),
        "lr_roc_auc_std": float(lr_auc.std()),
        "rf_roc_auc": float(rf_auc.mean()),
        "rf_roc_auc_std": float(rf_auc.std()),
        "positive_rate": float(df["label"].mean()),
        "primary": "xgb" if xgb is not None else "rf",
    }
    if xgb_auc is not None:
        metrics["xgb_roc_auc"] = float(xgb_auc.mean())
        metrics["xgb_roc_auc_std"] = float(xgb_auc.std())
    metrics["feature_importances"] = {
        f: float(v) for f, v in zip(FEATURES, rf.feature_importances_)
    }

    return {
        "lr": lr, "rf": rf, "xgb": xgb, "scaler": scaler_final,
        "_registry": {
            "model_key": "model3_injury_risk",
            "version": "1.0",
            "display_name": "Injury Risk",
            "task": "classification",
            "algorithm": "XGBoost / RandomForest / LogisticRegression (balanced)",
            "target": "is_injured_next_30d",
            "features": list(FEATURES),
            "metrics": metrics,
            "n_train_rows": int(len(df)),
            "artifact_path": output_dir,
            "prediction_table": "model3_features",
        },
        "_predictions": {"model3_features": df},
    }


if __name__ == "__main__":
    import psycopg2
    from config.settings import DB_DSN
    logging.basicConfig(level=logging.INFO)
    conn = psycopg2.connect(DB_DSN)
    run(conn)
    conn.close()
