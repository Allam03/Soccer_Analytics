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

FEATURES_PRE_MATCH = [
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

FEATURES_IN_GAME = [
    "minute",
    "goals_so_far",
    "xg_so_far",
    "shots_so_far",
    "pass_acc_so_far",
    "pressures_so_far",
    "red_cards_so_far",
    "goal_diff_so_far",   # derived: goals_so_far - opponent_goals_so_far
    "xg_diff_so_far",     # derived
    "is_home",
    "avg_xg_last5",       # pre-match context carried into in-game features
    "avg_pass_acc_last5",
]

# Keep old name as alias for the pre-match sub-model
FEATURES = FEATURES_PRE_MATCH


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

def load_in_game_features(conn) -> pd.DataFrame:
    """
    Build in-game training rows from match_minute_snapshots.

    Each row = one team, one minute, with result label (win/draw/loss at FT).
    Only includes minutes where snapshots exist (i.e. events occurred).
    """
    query = """
        SELECT
            mms.match_id,
            mms.team_id,
            mms.minute,
            mms.goals_so_far,
            mms.xg_so_far,
            mms.shots_so_far,
            mms.passes_so_far,
            mms.pass_acc_so_far,
            mms.pressures_so_far,
            mms.red_cards_so_far,
            -- opponent goals at same minute (latest snapshot <= this minute)
            COALESCE((
                SELECT opp.goals_so_far
                FROM   match_minute_snapshots opp
                WHERE  opp.match_id  = mms.match_id
                  AND  opp.team_id  != mms.team_id
                  AND  opp.minute   <= mms.minute
                ORDER  BY opp.minute DESC
                LIMIT  1
            ), 0) AS opp_goals_so_far,
            COALESCE((
                SELECT opp.xg_so_far
                FROM   match_minute_snapshots opp
                WHERE  opp.match_id  = mms.match_id
                  AND  opp.team_id  != mms.team_id
                  AND  opp.minute   <= mms.minute
                ORDER  BY opp.minute DESC
                LIMIT  1
            ), 0.0) AS opp_xg_so_far,
            -- full-time result for this team (from player_match_stats)
            (SELECT DISTINCT pms.result
             FROM   player_match_stats pms
             WHERE  pms.match_id = mms.match_id
               AND  pms.team_id  = mms.team_id
             LIMIT  1) AS result,
            -- is_home flag
            CASE WHEN m.home_team_id = mms.team_id THEN 1 ELSE 0 END AS is_home,
            -- last-5 rolling averages (reuse pre-match query logic via subquery)
            COALESCE((
                SELECT AVG(pms2.xg)
                FROM   player_match_stats pms2
                JOIN   matches m2 ON m2.match_id = pms2.match_id
                WHERE  pms2.team_id  = mms.team_id
                  AND  m2.match_date < m.match_date
                ORDER  BY m2.match_date DESC
                LIMIT  5
            ), 0) AS avg_xg_last5,
            COALESCE((
                SELECT AVG(pms2.pass_accuracy)
                FROM   player_match_stats pms2
                JOIN   matches m2 ON m2.match_id = pms2.match_id
                WHERE  pms2.team_id  = mms.team_id
                  AND  m2.match_date < m.match_date
                ORDER  BY m2.match_date DESC
                LIMIT  5
            ), 0) AS avg_pass_acc_last5
        FROM match_minute_snapshots mms
        JOIN matches m ON m.match_id = mms.match_id
        -- Sample every 5 minutes to keep training set manageable
        WHERE mms.minute % 5 = 0
          AND mms.minute > 0
        ORDER BY mms.match_id, mms.team_id, mms.minute
    """
    with conn.cursor() as cur:
        cur.execute(query)
        cols = [d[0] for d in cur.description]
        rows = cur.fetchall()
    df = pd.DataFrame(rows, columns=cols)
    if df.empty:
        return df

    df["goal_diff_so_far"] = df["goals_so_far"] - df["opp_goals_so_far"]
    df["xg_diff_so_far"]   = df["xg_so_far"]   - df["opp_xg_so_far"]
    return df

def encode_labels(df: pd.DataFrame) -> np.ndarray:
    return df["result"].map(RESULT_MAP).values


def run(conn, output_dir: str = "artifacts/model5") -> Dict[str, Any]:
    import os
    os.makedirs(output_dir, exist_ok=True)

    # ── Sub-model A: pre-match ──────────────────────────────────────────────
    logger.info("Model 5A: loading pre-match features ...")
    df_pre = load_features(conn)

    artifacts = {}

    if not df_pre.empty:
        logger.info("  %d team-match rows", len(df_pre))
        X_pre = df_pre[FEATURES_PRE_MATCH].fillna(0).values
        y_pre = encode_labels(df_pre)

        scaler_pre = StandardScaler()
        X_sc_pre   = scaler_pre.fit_transform(X_pre)

        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

        gbc_pre = GradientBoostingClassifier(
            n_estimators=300, max_depth=4, learning_rate=0.05, random_state=42
        )
        acc = cross_val_score(gbc_pre, X_sc_pre, y_pre, cv=cv, scoring="accuracy")
        logger.info("Pre-match GBC accuracy (5-fold): %.3f +/- %.3f", acc.mean(), acc.std())
        gbc_pre.fit(X_sc_pre, y_pre)

        joblib.dump(scaler_pre, f"{output_dir}/scaler_pre.pkl")
        joblib.dump(gbc_pre,    f"{output_dir}/gbc_pre.pkl")
        df_pre.to_parquet(f"{output_dir}/features_pre.parquet", index=False)
        artifacts["gbc_pre"] = gbc_pre
        artifacts["scaler_pre"] = scaler_pre
    else:
        logger.error("Empty pre-match feature set")

    # ── Sub-model B: in-game ────────────────────────────────────────────────
    logger.info("Model 5B: loading in-game features ...")
    df_ig = load_in_game_features(conn)

    if not df_ig.empty and df_ig["result"].notna().any():
        logger.info("  %d team-minute rows", len(df_ig))
        df_ig = df_ig.dropna(subset=["result"])
        X_ig = df_ig[FEATURES_IN_GAME].fillna(0).values
        y_ig = encode_labels(df_ig)

        scaler_ig = StandardScaler()
        X_sc_ig   = scaler_ig.fit_transform(X_ig)

        gbc_ig = GradientBoostingClassifier(
            n_estimators=300, max_depth=5, learning_rate=0.05, random_state=42
        )
        cv_ig = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        acc_ig = cross_val_score(gbc_ig, X_sc_ig, y_ig, cv=cv_ig, scoring="accuracy")
        logger.info("In-game GBC accuracy (5-fold): %.3f +/- %.3f", acc_ig.mean(), acc_ig.std())
        gbc_ig.fit(X_sc_ig, y_ig)

        joblib.dump(scaler_ig, f"{output_dir}/scaler_ingame.pkl")
        joblib.dump(gbc_ig,    f"{output_dir}/gbc_ingame.pkl")
        df_ig.to_parquet(f"{output_dir}/features_ingame.parquet", index=False)
        artifacts["gbc_ig"] = gbc_ig
        artifacts["scaler_ig"] = scaler_ig
    else:
        logger.warning(
            "No in-game snapshot data — run ingestion first, then retrain Model 5"
        )

    logger.info("Model 5 artefacts saved to %s", output_dir)
    return artifacts


if __name__ == "__main__":
    import psycopg2
    from config.settings import DB_DSN
    logging.basicConfig(level=logging.INFO)
    conn = psycopg2.connect(DB_DSN)
    run(conn)
    conn.close()