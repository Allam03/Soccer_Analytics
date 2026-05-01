"""
models/model2_team_cohesion.py

Model 2: Team Cohesion Analysis
Type: Graph analysis + regression
Objective: Build pass-network graphs per match, compute cohesion metrics
           (centrality, density, clustering coefficient), and use them to
           predict match outcomes (goals scored) via regression.

Input: pass_network_edges table (populated by ingest_pass_network.py)
"""

import logging
from typing import Dict, Any, List

import numpy as np
import pandas as pd
import networkx as nx
from sklearn.linear_model import Ridge
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
import joblib

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


def run(conn, output_dir: str = "artifacts/model2") -> Dict[str, Any]:
    import os
    os.makedirs(output_dir, exist_ok=True)

    logger.info("Model 2: loading pass network edges ...")
    df = load_edges(conn)
    logger.info("  %d edges loaded", len(df))

    logger.info("Model 2: computing graph features ...")
    feat_df = build_feature_matrix(df)
    logger.info("  %d team-match rows", len(feat_df))

    feat_df.to_parquet(f"{output_dir}/graph_features.parquet", index=False)

    X = feat_df[GRAPH_FEATURES].fillna(0).values
    y = feat_df["goals"].values.astype(float)

    scaler = StandardScaler()
    X_sc   = scaler.fit_transform(X)

    # Ridge regression (baseline)
    ridge  = Ridge(alpha=1.0)
    cv_r2  = cross_val_score(ridge, X_sc, y, cv=5, scoring="r2")
    logger.info("Ridge R2 (5-fold CV): %.3f +/- %.3f", cv_r2.mean(), cv_r2.std())
    ridge.fit(X_sc, y)

    # Gradient boosting (main model)
    gbr    = GradientBoostingRegressor(
        n_estimators=200, max_depth=3, learning_rate=0.05, random_state=42
    )
    cv_gbr = cross_val_score(gbr, X_sc, y, cv=5, scoring="r2")
    logger.info("GBR    R2 (5-fold CV): %.3f +/- %.3f", cv_gbr.mean(), cv_gbr.std())
    gbr.fit(X_sc, y)

    # Feature importances
    importances = pd.Series(gbr.feature_importances_, index=GRAPH_FEATURES)
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