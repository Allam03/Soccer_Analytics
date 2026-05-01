"""
models/model4_environment.py

Model 4: Environmental Impact Analysis
Type: Regression
Objective: Quantify the effect of weather conditions and venue type
           on player and team performance metrics.

Target variables (one model per target)
  - xg            (expected goals -- proxy for offensive performance)
  - pass_accuracy (passing quality)
  - pressures     (defensive intensity)

Features
  - temperature_c, precipitation_mm, wind_speed_kmh, humidity_pct
  - venue_type (home / away -- derived from team side)
  - stadium_name (one-hot encoded, top venues only)

Requires ingest_weather.py to have run.
"""

import logging
from typing import Dict, Any

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge, Lasso
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score
import joblib

logger = logging.getLogger(__name__)

WEATHER_FEATURES  = ["temperature_c", "precipitation_mm", "wind_speed_kmh", "humidity_pct"]
CATEGORICAL_FEATURES = ["venue_type", "stadium_name"]
TARGETS = ["xg", "pass_accuracy", "pressures"]


def load_features(conn) -> pd.DataFrame:
    query = """
        SELECT
            pms.stat_id,
            pms.player_id,
            pms.match_id,
            pms.xg,
            pms.pass_accuracy,
            pms.pressures,
            pms.minutes_played,
            w.temperature_c,
            w.precipitation_mm,
            w.wind_speed_kmh,
            w.humidity_pct,
            m.stadium_name,
            CASE
                WHEN pms.team_id = m.home_team_id THEN 'home'
                ELSE 'away'
            END AS venue_type
        FROM player_match_stats pms
        JOIN matches m  ON m.match_id  = pms.match_id
        JOIN weather w  ON w.match_id  = pms.match_id
        WHERE pms.minutes_played >= 45
          AND w.temperature_c IS NOT NULL
    """
    with conn.cursor() as cur:
        cur.execute(query)
        cols = [d[0] for d in cur.description]
        rows = cur.fetchall()
    df = pd.DataFrame(rows, columns=cols)

    # Keep only stadiums with enough data for meaningful encoding
    top_stadiums = df["stadium_name"].value_counts().head(20).index
    df["stadium_name"] = df["stadium_name"].where(
        df["stadium_name"].isin(top_stadiums), other="Other"
    )

    return df


def build_pipeline(model) -> Pipeline:
    """Wrap a model in a preprocessing + estimator pipeline."""
    preprocessor = ColumnTransformer(transformers=[
        ("num", StandardScaler(), WEATHER_FEATURES),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False),
         CATEGORICAL_FEATURES),
    ])
    return Pipeline([("prep", preprocessor), ("model", model)])


def run(conn, output_dir: str = "artifacts/model4") -> Dict[str, Any]:
    import os
    os.makedirs(output_dir, exist_ok=True)

    logger.info("Model 4: loading features ...")
    df = load_features(conn)
    logger.info("  %d player-match rows with weather data", len(df))

    if len(df) == 0:
        logger.error("No rows with weather data -- run ingest_weather.py first")
        return {}

    X = df[WEATHER_FEATURES + CATEGORICAL_FEATURES]
    results = {}

    for target in TARGETS:
        y = df[target].fillna(0).values
        logger.info("--- Target: %s ---", target)

        ridge_pipe = build_pipeline(Ridge(alpha=1.0))
        lasso_pipe = build_pipeline(Lasso(alpha=0.01, max_iter=2000))
        gbr_pipe   = build_pipeline(
            GradientBoostingRegressor(
                n_estimators=200, max_depth=3, learning_rate=0.05, random_state=42
            )
        )

        for name, pipe in [("Ridge", ridge_pipe), ("Lasso", lasso_pipe), ("GBR", gbr_pipe)]:
            cv_r2 = cross_val_score(pipe, X, y, cv=5, scoring="r2")
            logger.info("  %s  R2=%.3f +/- %.3f", name, cv_r2.mean(), cv_r2.std())

        # Fit the best model (GBR) on full data
        gbr_pipe.fit(X, y)
        joblib.dump(gbr_pipe, f"{output_dir}/gbr_{target}.pkl")

        # Extract feature importances from the GBR step
        enc_cats = (
            gbr_pipe.named_steps["prep"]
            .named_transformers_["cat"]
            .get_feature_names_out(CATEGORICAL_FEATURES)
        )
        feature_names = WEATHER_FEATURES + list(enc_cats)
        imps = pd.Series(
            gbr_pipe.named_steps["model"].feature_importances_,
            index=feature_names,
        )
        logger.info("  Top features for %s:\n%s", target,
                    imps.sort_values(ascending=False).head(6))
        results[target] = gbr_pipe

    df.to_parquet(f"{output_dir}/features.parquet", index=False)
    logger.info("Model 4 artefacts saved to %s", output_dir)
    return results


if __name__ == "__main__":
    import psycopg2
    from config.settings import DB_DSN
    logging.basicConfig(level=logging.INFO)
    conn = psycopg2.connect(DB_DSN)
    run(conn)
    conn.close()