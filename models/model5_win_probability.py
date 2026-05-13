"""
models/model5_win_probability.py

Model 5: Win Probability Modeling
Type: Multiclass classification  (win / draw / loss)

Two sub-models
--------------
A. Pre-match model  -- uses season-to-date rolling averages as of match date
B. In-game model    -- uses minute-by-minute cumulative stats
                       (requires match_minute_snapshots table)

Key fixes vs original
---------------------
load_features():
  The rolling-5 subqueries used
      ORDER BY m2.match_date DESC  LIMIT 5
  inside a scalar aggregate subquery.  PostgreSQL does not allow ORDER BY /
  LIMIT inside a subquery that is used as a scalar expression in a SELECT
  list together with GROUP BY.  Fixed by replacing each correlated subquery
  with a lateral join that first limits the rows and then aggregates, which
  is valid SQL and executes efficiently with the existing indexes.

load_in_game_features():
  Same pattern: the avg_xg_last5 / avg_pass_acc_last5 correlated subqueries
  used ORDER BY … LIMIT 5 illegally.  Fixed identically with lateral joins.
"""

import logging
from typing import Dict, Any

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
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
    "goal_diff_so_far",
    "xg_diff_so_far",
    "is_home",
    "avg_xg_last5",
    "avg_pass_acc_last5",
]

FEATURES = FEATURES_PRE_MATCH


def load_features(conn) -> pd.DataFrame:
    """
    Build team-match rows with rolling 5-match averages.

    Fix: rolling averages are computed with a LATERAL subquery that applies
    ORDER BY + LIMIT before the AVG(), which is valid in PostgreSQL.  The
    original correlated subquery used ORDER BY / LIMIT inside a scalar
    aggregate context, which PostgreSQL rejects.
    """
    query = """
        WITH team_match_agg AS (
            -- Step 1: aggregate player rows up to team-match level
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
                COUNT(pms.sub_minute)
                    FILTER (WHERE pms.sub_minute IS NOT NULL) AS subs_made
            FROM player_match_stats pms
            JOIN matches m ON m.match_id = pms.match_id
            WHERE pms.result IS NOT NULL
            GROUP BY pms.match_id, pms.team_id, m.match_date, m.home_team_id
        )
        SELECT
            tma.match_id,
            tma.team_id,
            tma.match_date,
            tma.home_team_id,
            tma.result,
            tma.red_cards                         AS red_cards_match,
            tma.subs_made,
            (tma.team_id = tma.home_team_id)::INT AS is_home,
            -- Rolling 5-match averages via LATERAL (ORDER BY + LIMIT inside
            -- a subquery that feeds into AVG is only valid this way)
            COALESCE(r5.avg_xg,       0) AS avg_xg_last5,
            COALESCE(r5.avg_shots,    0) AS avg_shots_last5,
            COALESCE(r5.avg_passes,   0) AS avg_passes_last5,
            COALESCE(r5.avg_pass_acc, 0) AS avg_pass_acc_last5,
            COALESCE(r5.avg_tackles,  0) AS avg_tackles_last5,
            COALESCE(r5.avg_pressures,0) AS avg_pressures_last5
        FROM team_match_agg tma
        LEFT JOIN LATERAL (
            SELECT
                AVG(prev.team_xg)       AS avg_xg,
                AVG(prev.team_shots)    AS avg_shots,
                AVG(prev.team_passes)   AS avg_passes,
                AVG(prev.team_pass_acc) AS avg_pass_acc,
                AVG(prev.team_tackles)  AS avg_tackles,
                AVG(prev.team_pressures)AS avg_pressures
            FROM (
                SELECT team_xg, team_shots, team_passes,
                       team_pass_acc, team_tackles, team_pressures
                FROM team_match_agg prev_inner
                WHERE prev_inner.team_id   = tma.team_id
                  AND prev_inner.match_date < tma.match_date
                ORDER BY prev_inner.match_date DESC
                LIMIT 5
            ) prev
        ) r5 ON TRUE
        ORDER BY tma.team_id, tma.match_date
    """
    with conn.cursor() as cur:
        cur.execute(query)
        cols = [d[0] for d in cur.description]
        rows = cur.fetchall()
    df = pd.DataFrame(rows, columns=cols)

    # Drop rows where no prior matches existed (cold start)
    df = df.dropna(subset=["avg_xg_last5"])
    return df


def load_in_game_features(conn) -> pd.DataFrame:
    """
    Build in-game training rows from match_minute_snapshots.

    Fix: avg_xg_last5 / avg_pass_acc_last5 used correlated subqueries with
    ORDER BY … LIMIT 5 in a scalar position, which PostgreSQL rejects.
    Replaced with LATERAL joins, identical fix to load_features().
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
            (SELECT DISTINCT pms.result
             FROM   player_match_stats pms
             WHERE  pms.match_id = mms.match_id
               AND  pms.team_id  = mms.team_id
             LIMIT  1) AS result,
            CASE WHEN m.home_team_id = mms.team_id THEN 1 ELSE 0 END AS is_home,
            -- Rolling 5-match avg xg via LATERAL
            COALESCE(r5xg.avg_xg,       0) AS avg_xg_last5,
            COALESCE(r5pa.avg_pass_acc,  0) AS avg_pass_acc_last5
        FROM match_minute_snapshots mms
        JOIN matches m ON m.match_id = mms.match_id
        -- Rolling avg xg: last 5 matches before this match date for this team
        LEFT JOIN LATERAL (
            SELECT AVG(sub.xg) AS avg_xg
            FROM (
                SELECT SUM(pms2.xg) AS xg
                FROM   player_match_stats pms2
                JOIN   matches m2 ON m2.match_id = pms2.match_id
                WHERE  pms2.team_id   = mms.team_id
                  AND  m2.match_date  < m.match_date
                GROUP  BY pms2.match_id
                ORDER  BY MAX(m2.match_date) DESC
                LIMIT  5
            ) sub
        ) r5xg ON TRUE
        -- Rolling avg pass accuracy: last 5 matches
        LEFT JOIN LATERAL (
            SELECT AVG(sub.pass_acc) AS avg_pass_acc
            FROM (
                SELECT AVG(pms2.pass_accuracy) AS pass_acc
                FROM   player_match_stats pms2
                JOIN   matches m2 ON m2.match_id = pms2.match_id
                WHERE  pms2.team_id   = mms.team_id
                  AND  m2.match_date  < m.match_date
                GROUP  BY pms2.match_id
                ORDER  BY MAX(m2.match_date) DESC
                LIMIT  5
            ) sub
        ) r5pa ON TRUE
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
        logger.info("Pre-match GBC accuracy (5-fold): %.3f +/- %.3f",
                    acc.mean(), acc.std())
        gbc_pre.fit(X_sc_pre, y_pre)

        joblib.dump(scaler_pre, f"{output_dir}/scaler_pre.pkl")
        joblib.dump(gbc_pre,    f"{output_dir}/gbc_pre.pkl")
        df_pre.to_parquet(f"{output_dir}/features_pre.parquet", index=False)
        artifacts["gbc_pre"]    = gbc_pre
        artifacts["scaler_pre"] = scaler_pre
        logger.info("Model 5A artifacts saved.")
    else:
        logger.error("Empty pre-match feature set — check that result column "
                     "is populated in player_match_stats")

    # ── Sub-model B: in-game ────────────────────────────────────────────────
    logger.info("Model 5B: loading in-game features ...")
    df_ig = load_in_game_features(conn)

    if not df_ig.empty and df_ig["result"].notna().any():
        logger.info("  %d team-minute rows", len(df_ig))
        df_ig = df_ig.dropna(subset=["result"])
        X_ig  = df_ig[FEATURES_IN_GAME].fillna(0).values
        y_ig  = encode_labels(df_ig)

        scaler_ig = StandardScaler()
        X_sc_ig   = scaler_ig.fit_transform(X_ig)

        gbc_ig = GradientBoostingClassifier(
            n_estimators=300, max_depth=5, learning_rate=0.05, random_state=42
        )
        cv_ig  = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        acc_ig = cross_val_score(gbc_ig, X_sc_ig, y_ig,
                                 cv=cv_ig, scoring="accuracy")
        logger.info("In-game GBC accuracy (5-fold): %.3f +/- %.3f",
                    acc_ig.mean(), acc_ig.std())
        gbc_ig.fit(X_sc_ig, y_ig)

        joblib.dump(scaler_ig, f"{output_dir}/scaler_ingame.pkl")
        joblib.dump(gbc_ig,    f"{output_dir}/gbc_ingame.pkl")
        df_ig.to_parquet(f"{output_dir}/features_ingame.parquet", index=False)
        artifacts["gbc_ig"]    = gbc_ig
        artifacts["scaler_ig"] = scaler_ig
        logger.info("Model 5B artifacts saved.")
    else:
        logger.warning(
            "No in-game snapshot data — run the full ingestion pipeline "
            "first, then retrain Model 5."
        )

    logger.info("Model 5 complete.")
    return artifacts


if __name__ == "__main__":
    import psycopg2
    from config.settings import DB_DSN
    logging.basicConfig(level=logging.INFO)
    conn = psycopg2.connect(DB_DSN)
    run(conn)
    conn.close()