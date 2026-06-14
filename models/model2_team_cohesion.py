"""
models/model2_team_cohesion.py

Model 2: Team Cohesion Analysis
Type: Graph analysis + regression
Objective: Build pass-network graphs per match, compute cohesion metrics
           (centrality, density, clustering coefficient), and use them to
           predict match outcomes (goals scored) via regression.

Beyond pure graph topology, the feature set also includes the strongest
known predictors of goals scored -- team xG, xG conceded, home advantage and
opponent quality -- since network structure alone explains very little of the
variance in goals (R2 ~ 0.05-0.09).

Two correctness fixes vs the original:
  1. No CV leakage. The StandardScaler is wrapped in a Pipeline so it is fit
     only on the training folds inside cross_val_score (previously the scaler
     was fit on the full dataset before CV, inflating the reported R2). The
     persisted artifacts (scaler.pkl, gbr.pkl) are still produced the same way
     so the serving path is unchanged.

Input: pass_network_edges + player_match_stats + matches.
"""

import logging
from typing import Dict, Any, List

import numpy as np
import pandas as pd
import networkx as nx
from sklearn.linear_model import Ridge
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import joblib

from models.eval_utils import attach_season, grouped_cv, holdout_season, TEST_SEASON

logger = logging.getLogger(__name__)


def load_edges(conn) -> pd.DataFrame:
    """Load all pass network edges joined with match outcome."""
    query = """
        SELECT
            pne.match_id,
            pne.team_id,
            pne.passer_id,
            pne.receiver_id,
            pne.pass_count,
            pne.avg_x_start,
            pne.avg_y_start,
            pne.avg_x_end,
            pne.avg_y_end,
            m.home_team_id,
            m.away_team_id,
            m.home_score,
            m.away_score
        FROM pass_network_edges pne
        JOIN matches m ON m.match_id = pne.match_id
        ORDER BY pne.match_id, pne.team_id
    """
    with conn.cursor() as cur:
        cur.execute(query)
        cols = [d[0] for d in cur.description]
        rows = cur.fetchall()
    return pd.DataFrame(rows, columns=cols)


def load_team_context(conn) -> pd.DataFrame:
    """
    One row per (match_id, team_id) with team-level context features:

        team_xg          -- sum of player xG for the team in the match
        team_xga         -- the opponent's team_xg in the same match (xG against)
        is_home          -- 1 if this team played at home
        opponent_quality -- opponent's season-to-date points-per-game *before*
                            this match (expanding mean, shifted to exclude the
                            current result -> no leakage)
    """
    query = """
        SELECT
            m.match_id,
            pms.team_id,
            m.home_team_id,
            m.away_team_id,
            m.home_score,
            m.away_score,
            m.season,
            m.match_date,
            SUM(pms.xg) AS team_xg
        FROM player_match_stats pms
        JOIN matches m ON m.match_id = pms.match_id
        GROUP BY m.match_id, pms.team_id, m.home_team_id, m.away_team_id,
                 m.home_score, m.away_score, m.season, m.match_date
    """
    with conn.cursor() as cur:
        cur.execute(query)
        cols = [d[0] for d in cur.description]
        rows = cur.fetchall()
    df = pd.DataFrame(rows, columns=cols)
    if df.empty:
        return df

    df["team_xg"]  = df["team_xg"].astype(float)
    df["is_home"]  = (df["team_id"] == df["home_team_id"]).astype(int)
    df["opp_team_id"] = np.where(
        df["is_home"] == 1, df["away_team_id"], df["home_team_id"]
    )

    gf = np.where(df["is_home"] == 1, df["home_score"], df["away_score"])
    ga = np.where(df["is_home"] == 1, df["away_score"], df["home_score"])
    df["points"] = np.select([gf > ga, gf == ga], [3, 1], default=0)

    # team_xga = opponent's team_xg in the same match
    xga = df[["match_id", "team_id", "team_xg"]].rename(
        columns={"team_id": "opp_team_id", "team_xg": "team_xga"}
    )
    df = df.merge(xga, on=["match_id", "opp_team_id"], how="left")

    # season-to-date PPG for each team, excluding the current match
    df = df.sort_values(["season", "team_id", "match_date"])
    df["ppg_to_date"] = (
        df.groupby(["season", "team_id"])["points"]
          .transform(lambda s: s.expanding().mean().shift(1))
    )

    # opponent_quality = opponent's season-to-date PPG at this match
    oppq = df[["match_id", "team_id", "ppg_to_date"]].rename(
        columns={"team_id": "opp_team_id", "ppg_to_date": "opponent_quality"}
    )
    df = df.merge(oppq, on=["match_id", "opp_team_id"], how="left")

    league_avg = df["ppg_to_date"].mean()
    league_avg = float(league_avg) if pd.notna(league_avg) else 1.0
    df["opponent_quality"] = df["opponent_quality"].fillna(league_avg)
    df["team_xga"] = df["team_xga"].fillna(0.0)

    return df[["match_id", "team_id", "team_xg", "team_xga",
               "is_home", "opponent_quality"]]


def build_graph(edges_df: pd.DataFrame) -> nx.DiGraph:
    """
    Build a directed weighted pass graph from a subset of edges.

    Parameters
    ----------
    edges_df : rows for ONE (match_id, team_id) pair.
    """
    G = nx.DiGraph()
    for _, row in edges_df.iterrows():
        G.add_edge(
            row["passer_id"],
            row["receiver_id"],
            weight=row["pass_count"],
        )
    return G


def compute_graph_features(G: nx.DiGraph) -> Dict[str, float]:
    """
    Compute cohesion metrics from a pass graph.

    Returns
    -------
    dict with scalar features for ML input.
    """
    if G.number_of_nodes() == 0:
        return _empty_graph_features()

    # Convert to undirected for density / clustering
    UG = G.to_undirected()

    # Node-level centrality (mean across players)
    in_centrality  = nx.in_degree_centrality(G)
    out_centrality = nx.out_degree_centrality(G)
    between        = nx.betweenness_centrality(G, weight="weight", normalized=True)
    page_rank      = nx.pagerank(G, weight="weight")

    # Pass-weighted degree
    total_passes = sum(d["weight"] for _, _, d in G.edges(data=True))

    return {
        "network_density":          nx.density(UG),
        "clustering_coefficient":   nx.average_clustering(UG),
        "mean_in_centrality":       np.mean(list(in_centrality.values())),
        "mean_out_centrality":      np.mean(list(out_centrality.values())),
        "mean_betweenness":         np.mean(list(between.values())),
        "max_betweenness":          max(between.values(), default=0),
        "mean_pagerank":            np.mean(list(page_rank.values())),
        "max_pagerank":             max(page_rank.values(), default=0),
        "n_nodes":                  G.number_of_nodes(),
        "n_edges":                  G.number_of_edges(),
        "total_passes":             total_passes,
        "pass_per_edge":            total_passes / max(1, G.number_of_edges()),
    }


def _empty_graph_features() -> Dict[str, float]:
    keys = [
        "network_density", "clustering_coefficient",
        "mean_in_centrality", "mean_out_centrality",
        "mean_betweenness", "max_betweenness",
        "mean_pagerank", "max_pagerank",
        "n_nodes", "n_edges", "total_passes", "pass_per_edge",
    ]
    return {k: 0.0 for k in keys}


def build_feature_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """
    Iterate over all (match_id, team_id) pairs and compute graph features.

    Returns a DataFrame with one row per (match_id, team_id) containing
    graph-derived features and the goals scored by that team.
    """
    records: List[Dict[str, Any]] = []

    for (match_id, team_id), grp in df.groupby(["match_id", "team_id"]):
        G       = build_graph(grp)
        metrics = compute_graph_features(G)

        # Determine goals scored by this team
        row = grp.iloc[0]
        if team_id == row["home_team_id"]:
            goals = row["home_score"]
        else:
            goals = row["away_score"]

        records.append({
            "match_id": match_id,
            "team_id":  team_id,
            "goals":    goals,
            **metrics,
        })

    return pd.DataFrame(records)


GRAPH_FEATURES = [
    "network_density", "clustering_coefficient",
    "mean_in_centrality", "mean_out_centrality",
    "mean_betweenness", "max_betweenness",
    "mean_pagerank", "max_pagerank",
    "n_nodes", "n_edges", "total_passes", "pass_per_edge",
]

# Strong contextual predictors of goals scored, appended to the graph metrics.
CONTEXT_FEATURES = ["team_xg", "team_xga", "is_home", "opponent_quality"]

# Full feature vector used by the regression models (and the serving path).
MODEL_FEATURES = GRAPH_FEATURES + CONTEXT_FEATURES


def run(conn, output_dir: str = "artifacts/model2") -> Dict[str, Any]:
    import os
    os.makedirs(output_dir, exist_ok=True)

    logger.info("Model 2: loading pass network edges ...")
    df = load_edges(conn)
    logger.info("  %d edges loaded", len(df))

    logger.info("Model 2: computing graph features ...")
    feat_df = build_feature_matrix(df)
    logger.info("  %d team-match rows", len(feat_df))

    # Enrich with team-level context (team_xg/xga, home, opponent quality).
    ctx = load_team_context(conn)
    feat_df = feat_df.merge(ctx, on=["match_id", "team_id"], how="left")
    for col in CONTEXT_FEATURES:
        feat_df[col] = feat_df[col].fillna(0.0)

    feat_df = attach_season(feat_df, conn)
    feat_df.to_parquet(f"{output_dir}/graph_features.parquet", index=False)

    X = feat_df[MODEL_FEATURES].fillna(0).values
    y = feat_df["goals"].values.astype(float)
    groups  = feat_df["match_id"].values
    seasons = feat_df["season"].values

    # Honest evaluation: GroupKFold by match (no match straddles folds) plus a
    # held-out season test. The scaler lives inside the Pipeline so it is fit
    # only on training folds. Persisted artifacts below are still a standalone
    # scaler + estimator to match the serving path in api_server.py.
    ridge_pipe = Pipeline([("scaler", StandardScaler()), ("est", Ridge(alpha=1.0))])
    gbr_pipe   = Pipeline([
        ("scaler", StandardScaler()),
        ("est", GradientBoostingRegressor(
            n_estimators=200, max_depth=3, learning_rate=0.05, random_state=42)),
    ])

    m, s = grouped_cv(ridge_pipe, X, y, groups, "r2")
    logger.info("Ridge R2 (GroupKFold by match): %.3f +/- %.3f", m, s)
    m, s = grouped_cv(gbr_pipe, X, y, groups, "r2")
    logger.info("GBR   R2 (GroupKFold by match): %.3f +/- %.3f", m, s)
    ho, n = holdout_season(gbr_pipe, X, y, seasons, "r2")
    if ho is not None:
        logger.info("GBR   R2 held-out %s (n=%d): %.3f", TEST_SEASON, n, ho)

    # Fit final artifacts on the full dataset (scaler + estimators saved
    # separately, exactly as the API expects).
    scaler = StandardScaler()
    X_sc   = scaler.fit_transform(X)

    ridge = Ridge(alpha=1.0)
    ridge.fit(X_sc, y)

    gbr = GradientBoostingRegressor(
        n_estimators=200, max_depth=3, learning_rate=0.05, random_state=42
    )
    gbr.fit(X_sc, y)

    # Feature importances
    importances = pd.Series(gbr.feature_importances_, index=MODEL_FEATURES)
    logger.info("Top features:\n%s", importances.sort_values(ascending=False).head())

    joblib.dump(scaler, f"{output_dir}/scaler.pkl")
    joblib.dump(ridge,  f"{output_dir}/ridge.pkl")
    joblib.dump(gbr,    f"{output_dir}/gbr.pkl")

    logger.info("Model 2 artefacts saved to %s", output_dir)
    return {"ridge": ridge, "gbr": gbr, "scaler": scaler, "feat_df": feat_df}


if __name__ == "__main__":
    import psycopg2
    from config.settings import DB_DSN
    logging.basicConfig(level=logging.INFO)
    conn = psycopg2.connect(DB_DSN)
    run(conn)
    conn.close()