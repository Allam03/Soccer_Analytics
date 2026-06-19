"""
models/model5_win_probability.py

Model 5: Win Probability Modeling
Type: Multiclass classification  (win / draw / loss)

Honest evaluation:
  * The dataset is now the complete 2015/16 season of all five major European
    leagues plus the men's international tournaments, so it is balanced across
    ~100 clubs and 50+ nations -- no single-team prior to lean on.
  * Cross-validation uses StratifiedGroupKFold grouped by match_id. A shuffled
    split leaks badly here because the two team-rows of a match (5A) and all
    ~90 minute-rows of a match (5B) share the same label and form features.
  * A held-out season (FIFA World Cup 2022) gives an out-of-time test.
  * Scalers stay inside Pipelines during CV; artifacts are still saved as
    separate scaler/estimator .pkl files in the format api_server.py expects.
"""

import logging
from typing import Dict, Any

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import joblib

from models.eval_utils import attach_season, grouped_cv, holdout_season, TEST_SEASON

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
    Build team-match rows with rolling 5-match averages via LATERAL subquery.
    """
    query = """
        WITH team_match_agg AS (
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
    df = df.dropna(subset=["avg_xg_last5"])
    return df


def load_in_game_features(conn) -> pd.DataFrame:
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
            COALESCE(r5xg.avg_xg,       0) AS avg_xg_last5,
            COALESCE(r5pa.avg_pass_acc,  0) AS avg_pass_acc_last5
        FROM match_minute_snapshots mms
        JOIN matches m ON m.match_id = mms.match_id
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

    artifacts = {}
    pre_m = pre_s = pre_ho = pre_naive = None
    ig_m = ig_s = ig_ho = ig_naive = None

    # ── Sub-model A: pre-match ──────────────────────────────────────────────
    logger.info("Model 5A: loading pre-match features ...")
    df_pre = load_features(conn)
    df_pre = attach_season(df_pre, conn)

    if not df_pre.empty:
        logger.info("  %d team-match rows", len(df_pre))
        X_pre = df_pre[FEATURES_PRE_MATCH].fillna(0).values
        y_pre = encode_labels(df_pre)
        groups  = df_pre["match_id"].values
        seasons = df_pre["season"].values

        # Pipeline keeps the scaler isolated to training folds; GroupKFold by
        # match prevents the two team-rows of a match leaking across folds.
        gbc_pipe_pre = Pipeline([
            ("scaler", StandardScaler()),
            ("clf", GradientBoostingClassifier(
                n_estimators=300, max_depth=4, learning_rate=0.05, random_state=42
            )),
        ])
        pre_m, pre_s = grouped_cv(gbc_pipe_pre, X_pre, y_pre, groups, "accuracy", stratified=True)
        pre_naive = np.bincount(y_pre).max() / len(y_pre)
        logger.info("Pre-match accuracy (StratifiedGroupKFold): %.3f +/- %.3f "
                    "(naive majority=%.3f, lift=%+.3f)", pre_m, pre_s, pre_naive, pre_m - pre_naive)
        pre_ho, n = holdout_season(gbc_pipe_pre, X_pre, y_pre, seasons, "accuracy")
        if pre_ho is not None:
            logger.info("Pre-match accuracy held-out %s (n=%d): %.3f", TEST_SEASON, n, pre_ho)
        gbc_pipe_pre.fit(X_pre, y_pre)

        # Save scaler and estimator separately so api_server.py can call
        # scaler.transform() and gbc.predict_proba() independently.
        gbc_pre    = gbc_pipe_pre.named_steps["clf"]
        scaler_pre = gbc_pipe_pre.named_steps["scaler"]

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
    df_ig = attach_season(df_ig, conn)

    if not df_ig.empty and df_ig["result"].notna().any():
        logger.info("  %d team-minute rows", len(df_ig))
        df_ig = df_ig.dropna(subset=["result"])
        X_ig  = df_ig[FEATURES_IN_GAME].fillna(0).values
        y_ig  = encode_labels(df_ig)
        groups_ig  = df_ig["match_id"].values
        seasons_ig = df_ig["season"].values

        gbc_pipe_ig = Pipeline([
            ("scaler", StandardScaler()),
            ("clf", GradientBoostingClassifier(
                n_estimators=300, max_depth=5, learning_rate=0.05, random_state=42
            )),
        ])
        # GroupKFold by match is essential here: all ~90 minute-rows of a match
        # share the same final-result label and identical rolling-form features,
        # so a shuffled split lets the model memorise matches (0.89 shuffled vs
        # ~0.64 grouped).
        ig_m, ig_s = grouped_cv(gbc_pipe_ig, X_ig, y_ig, groups_ig, "accuracy", stratified=True)
        ig_naive = np.bincount(y_ig).max() / len(y_ig)
        logger.info("In-game accuracy (StratifiedGroupKFold by match): %.3f +/- %.3f "
                    "(naive majority=%.3f)", ig_m, ig_s, ig_naive)
        ig_ho, n = holdout_season(gbc_pipe_ig, X_ig, y_ig, seasons_ig, "accuracy")
        if ig_ho is not None:
            logger.info("In-game accuracy held-out %s (n=%d): %.3f", TEST_SEASON, n, ig_ho)
        gbc_pipe_ig.fit(X_ig, y_ig)

        gbc_ig    = gbc_pipe_ig.named_steps["clf"]
        scaler_ig = gbc_pipe_ig.named_steps["scaler"]

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

    metrics: dict[str, Any] = {}
    if pre_m is not None:
        metrics["prematch_accuracy"] = float(pre_m)
        metrics["prematch_accuracy_std"] = float(pre_s)
        metrics["prematch_naive"] = float(pre_naive)
    if pre_ho is not None:
        metrics["prematch_accuracy_heldout"] = float(pre_ho)
    if ig_m is not None:
        metrics["ingame_accuracy"] = float(ig_m)
        metrics["ingame_accuracy_std"] = float(ig_s)
        metrics["ingame_naive"] = float(ig_naive)
    if ig_ho is not None:
        metrics["ingame_accuracy_heldout"] = float(ig_ho)

    result: dict[str, Any] = dict(artifacts)
    result["_registry"] = {
        "model_key": "model5_win_probability",
        "version": "1.0",
        "display_name": "Win Probability",
        "task": "classification",
        "algorithm": "GradientBoostingClassifier (pre-match + in-game)",
        "target": "match result (win / draw / loss)",
        "features": list(FEATURES_PRE_MATCH),
        "metrics": metrics,
        "n_train_rows": int(len(df_pre)) if not df_pre.empty else 0,
        "artifact_path": output_dir,
        "prediction_table": "model5_features_pre",
    }
    if not df_pre.empty:
        result["_predictions"] = {"model5_features_pre": df_pre}
    return result


if __name__ == "__main__":
    import psycopg2
    from config.settings import DB_DSN
    logging.basicConfig(level=logging.INFO)
    conn = psycopg2.connect(DB_DSN)
    run(conn)
    conn.close()
