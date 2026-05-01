"""
models/model1_player_clustering.py

Model 1: Player Efficiency and Style Profiling
Type: Unsupervised clustering
Algorithms: KMeans, DBSCAN, GaussianMixture

Objective: Classify players into tactical archetypes using per-match
statistical fingerprints aggregated to player-season level.

Features used
-------------
passes_attempted, shots, tackles, interceptions, clearances,
carry_distance, progressive_carries, key_passes, progressive_passes,
pressures, dribbles_completed
"""

import logging
from typing import Dict, Any

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans, DBSCAN
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
import joblib

logger = logging.getLogger(__name__)

FEATURES = [
    "passes_attempted",
    "shots",
    "tackles",
    "interceptions",
    "clearances",
    "carry_distance",
    "progressive_carries",
    "key_passes",
    "progressive_passes",
    "pressures",
    "dribbles_completed",
]

ARCHETYPE_LABELS = {
    # Populated after fitting -- maps cluster id -> human-readable label.
    # Example: {0: "Defensive Midfielder", 1: "Wide Attacker", ...}
}


def load_features(conn) -> pd.DataFrame:
    """
    Pull per-player season averages from the DB.

    Returns one row per (player_id, season) with mean values for all
    FEATURES, plus player_name and position for interpretability.
    """
    query = """
        SELECT
            p.player_id,
            p.player_name,
            p.position,
            m.season,
            COUNT(pms.stat_id)                       AS matches_played,
            AVG(pms.passes_attempted)                AS passes_attempted,
            AVG(pms.shots)                           AS shots,
            AVG(pms.tackles)                         AS tackles,
            AVG(pms.interceptions)                   AS interceptions,
            AVG(pms.clearances)                      AS clearances,
            AVG(pms.carry_distance)                  AS carry_distance,
            AVG(pms.progressive_carries)             AS progressive_carries,
            AVG(pms.key_passes)                      AS key_passes,
            AVG(pms.progressive_passes)              AS progressive_passes,
            AVG(pms.pressures)                       AS pressures,
            AVG(pms.dribbles_completed)              AS dribbles_completed
        FROM player_match_stats pms
        JOIN players p ON p.player_id = pms.player_id
        JOIN matches  m ON m.match_id  = pms.match_id
        WHERE pms.minutes_played >= 45
        GROUP BY p.player_id, p.player_name, p.position, m.season
        HAVING COUNT(pms.stat_id) >= 5
    """
    with conn.cursor() as cur:
        cur.execute(query)
        cols = [d[0] for d in cur.description]
        rows = cur.fetchall()
    return pd.DataFrame(rows, columns=cols)


def preprocess(df: pd.DataFrame) -> tuple[np.ndarray, StandardScaler]:
    """Scale features to zero mean and unit variance."""
    X      = df[FEATURES].fillna(0).values
    scaler = StandardScaler()
    X_sc   = scaler.fit_transform(X)
    return X_sc, scaler


def train_kmeans(X: np.ndarray, n_clusters: int = 7) -> KMeans:
    model = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    model.fit(X)
    score = silhouette_score(X, model.labels_)
    logger.info("KMeans  k=%d  silhouette=%.4f", n_clusters, score)
    return model


def train_dbscan(X: np.ndarray, eps: float = 0.8, min_samples: int = 5) -> DBSCAN:
    model = DBSCAN(eps=eps, min_samples=min_samples)
    model.fit(X)
    n_clusters = len(set(model.labels_)) - (1 if -1 in model.labels_ else 0)
    noise      = (model.labels_ == -1).sum()
    logger.info("DBSCAN  n_clusters=%d  noise_points=%d", n_clusters, noise)
    return model


def train_gmm(X: np.ndarray, n_components: int = 7) -> GaussianMixture:
    model = GaussianMixture(
        n_components=n_components, covariance_type="full", random_state=42
    )
    model.fit(X)
    logger.info("GMM  n_components=%d  BIC=%.1f", n_components, model.bic(X))
    return model


def select_k(X: np.ndarray, k_range=range(4, 12)) -> int:
    """Use silhouette score to pick the best K for KMeans."""
    best_k, best_score = 5, -1.0
    for k in k_range:
        labels = KMeans(n_clusters=k, random_state=42, n_init=10).fit_predict(X)
        score  = silhouette_score(X, labels)
        logger.info("  k=%d  silhouette=%.4f", k, score)
        if score > best_score:
            best_k, best_score = k, score
    logger.info("Best k: %d (score=%.4f)", best_k, best_score)
    return best_k


def run(conn, output_dir: str = "artifacts/model1") -> Dict[str, Any]:
    """
    Full training run.

    Returns a dict with fitted models, scaler, and the labelled DataFrame.
    """
    import os
    os.makedirs(output_dir, exist_ok=True)

    logger.info("Model 1: loading features ...")
    df = load_features(conn)
    logger.info("  %d player-season rows loaded", len(df))

    X, scaler = preprocess(df)

    # Dimensionality reduction for DBSCAN (helps with curse of dimensionality)
    pca  = PCA(n_components=6, random_state=42)
    X_pca = pca.fit_transform(X)
    logger.info("PCA explained variance: %.1f%%", pca.explained_variance_ratio_.sum() * 100)

    # Select best K then train all three algorithms
    best_k = select_k(X)

    kmeans = train_kmeans(X, n_clusters=best_k)
    dbscan = train_dbscan(X_pca)
    gmm    = train_gmm(X, n_components=best_k)

    df["cluster_kmeans"] = kmeans.labels_
    df["cluster_dbscan"] = dbscan.labels_
    df["cluster_gmm"]    = gmm.predict(X)

    # Persist artefacts
    joblib.dump(scaler, f"{output_dir}/scaler.pkl")
    joblib.dump(kmeans, f"{output_dir}/kmeans.pkl")
    joblib.dump(dbscan, f"{output_dir}/dbscan.pkl")
    joblib.dump(gmm,    f"{output_dir}/gmm.pkl")
    joblib.dump(pca,    f"{output_dir}/pca.pkl")
    df.to_parquet(f"{output_dir}/player_clusters.parquet", index=False)

    logger.info("Model 1 artefacts saved to %s", output_dir)

    return {
        "kmeans": kmeans,
        "dbscan": dbscan,
        "gmm":    gmm,
        "scaler": scaler,
        "pca":    pca,
        "df":     df,
    }


if __name__ == "__main__":
    import psycopg2
    from config.settings import DB_DSN
    logging.basicConfig(level=logging.INFO)
    conn = psycopg2.connect(DB_DSN)
    run(conn)
    conn.close()