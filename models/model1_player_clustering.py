"""
models/model1_player_clustering.py

Model 1: Player Efficiency and Style Profiling
Type: Unsupervised clustering
Algorithms: KMeans, DBSCAN, GaussianMixture

Objective: Classify players into tactical archetypes using statistical
fingerprints aggregated to the player-team-season level.

Accuracy fixes vs the original
------------------------------
1. Per-90 normalisation. Raw per-match averages of counting stats are
   dominated by minutes played and by zero-inflation (54-62% zeros in
   shots/dribbles), which collapses the clusters (silhouette ~ 0.22-0.28).
   Every counting feature is now expressed per 90 minutes
   (SUM(stat) / SUM(minutes) * 90), removing the minutes confound.
2. Goalkeepers are excluded from the outfield clustering -- their outfield
   stat profile is degenerate (near-zero everywhere) and forms a trivial
   cluster that inflates within-cluster homogeneity without meaning.

Features used (all per-90)
--------------------------
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

def position_group(pos) -> str:
    """
    Map a StatsBomb position name (player_match_stats.starting_position, e.g.
    'Left Center Back', 'Right Wing Back', 'Center Attacking Midfield',
    'Center Forward') to a coarse role group. Keyword order matters: 'back'
    is checked before 'wing' so wing-backs are defenders.
    """
    if not pos:
        return "UNK"
    p = pos.lower()
    if "goalkeeper" in p:
        return "GK"
    if "back" in p:                       # full/centre/wing backs
        return "DEF"
    if "midfield" in p:                   # def / central / attacking midfield
        return "MID"
    if "wing" in p or "forward" in p or "striker" in p:
        return "FWD"
    return "UNK"

# Data-driven archetype labelling.
#
# KMeans cluster ids are arbitrary, so a fixed id->name map is meaningless: the
# cluster that happens to be "0" on one training run is a different group on the
# next. Instead we describe each cluster from its own centroid. Each signature
# below is a set of +/- feature weights; we score every cluster's z-scored
# feature profile against every signature and assign the best-matching label.
# This makes the labels reflect what the cluster actually is, and it survives
# re-training and changes in k.
SIGNATURES: dict[str, dict[str, int]] = {
    "Ball-Winning Midfielder": {"tackles": 1, "pressures": 1, "interceptions": 1, "shots": -1},
    "Deep-Lying Playmaker":    {"passes_attempted": 1, "progressive_passes": 1, "shots": -1, "tackles": -1},
    "Advanced Playmaker":      {"key_passes": 1, "progressive_passes": 1, "dribbles_completed": 1},
    "Dribbling Winger":        {"dribbles_completed": 1, "progressive_carries": 1, "carry_distance": 1},
    "Goalscorer":              {"shots": 1, "key_passes": 1, "tackles": -1, "clearances": -1},
    "Ball-Playing Defender":   {"progressive_passes": 1, "clearances": 1, "interceptions": 1, "shots": -1},
    "No-Nonsense Defender":    {"clearances": 1, "interceptions": 1, "progressive_passes": -1},
    "Pressing Forward":        {"pressures": 1, "shots": 1, "clearances": -1},
}


def label_clusters(profile_z: pd.DataFrame) -> dict[int, str]:
    """
    Assign a human-readable archetype to each cluster from its z-scored centroid.

    profile_z: index = cluster id, columns = FEATURES, values = z-scores of the
    cluster mean across clusters (how much this cluster over/under-indexes on
    each feature). Returns {cluster_id: label}. Collisions are resolved by giving
    the stronger-scoring cluster the label and the other its next-best signature.
    """
    # score[cid][label] = dot(centroid_z, signature weights)
    scored: list[tuple[float, int, str]] = []
    for cid in profile_z.index:
        z = profile_z.loc[cid]
        for name, weights in SIGNATURES.items():
            s = sum(w * float(z.get(f, 0.0)) for f, w in weights.items())
            scored.append((s, int(cid), name))
    scored.sort(reverse=True)  # strongest matches first

    labels: dict[int, str] = {}
    used: set[str] = set()
    for _, cid, name in scored:
        if cid in labels or name in used:
            continue
        labels[cid] = name
        used.add(name)
    # Any cluster still unlabelled (more clusters than signatures) gets a fallback.
    for cid in profile_z.index:
        labels.setdefault(int(cid), "Hybrid Role")
    return labels


def load_features(conn) -> pd.DataFrame:
    """
    Pull per-90 statistical fingerprints from the DB.

    Returns one row per (player_id, team_id, season). Every counting feature
    is normalised to per-90 minutes -- SUM(stat) / SUM(minutes) * 90 -- so the
    fingerprint reflects playing style rather than how many minutes a player
    accumulated. team_id is included so the serving layer can map a player to
    their archetype directly. Position is the player's dominant
    starting_position (StatsBomb), used only to drop goalkeepers.
    """
    query = """
        SELECT
            p.player_id,
            p.player_name,
            pms.team_id,
            mode() WITHIN GROUP (ORDER BY pms.starting_position) AS position,
            m.season,
            COUNT(pms.stat_id)              AS matches_played,
            SUM(pms.minutes_played)         AS minutes_total,
            SUM(pms.passes_attempted)::float    / NULLIF(SUM(pms.minutes_played),0) * 90 AS passes_attempted,
            SUM(pms.shots)::float               / NULLIF(SUM(pms.minutes_played),0) * 90 AS shots,
            SUM(pms.tackles)::float             / NULLIF(SUM(pms.minutes_played),0) * 90 AS tackles,
            SUM(pms.interceptions)::float       / NULLIF(SUM(pms.minutes_played),0) * 90 AS interceptions,
            SUM(pms.clearances)::float          / NULLIF(SUM(pms.minutes_played),0) * 90 AS clearances,
            SUM(pms.carry_distance)::float      / NULLIF(SUM(pms.minutes_played),0) * 90 AS carry_distance,
            SUM(pms.progressive_carries)::float / NULLIF(SUM(pms.minutes_played),0) * 90 AS progressive_carries,
            SUM(pms.key_passes)::float          / NULLIF(SUM(pms.minutes_played),0) * 90 AS key_passes,
            SUM(pms.progressive_passes)::float  / NULLIF(SUM(pms.minutes_played),0) * 90 AS progressive_passes,
            SUM(pms.pressures)::float           / NULLIF(SUM(pms.minutes_played),0) * 90 AS pressures,
            SUM(pms.dribbles_completed)::float  / NULLIF(SUM(pms.minutes_played),0) * 90 AS dribbles_completed
        FROM player_match_stats pms
        JOIN players p ON p.player_id = pms.player_id
        JOIN matches  m ON m.match_id  = pms.match_id
        WHERE pms.minutes_played >= 45
        GROUP BY p.player_id, p.player_name, pms.team_id, m.season
        HAVING COUNT(pms.stat_id) >= 5
    """
    with conn.cursor() as cur:
        cur.execute(query)
        cols = [d[0] for d in cur.description]
        rows = cur.fetchall()
    df = pd.DataFrame(rows, columns=cols)
    if not df.empty:
        df["position_group"] = df["position"].map(position_group)
        # Goalkeepers have a degenerate outfield-stat profile -- exclude them
        # from the archetype clustering.
        df = df[df["position_group"] != "GK"].reset_index(drop=True)
    return df


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

    # Derive archetype labels from the KMeans centroids (data-driven, not a
    # hard-coded id->name guess). Profile each cluster by its mean per-90 feature
    # vector, z-score across clusters, then match to the closest signature.
    profile   = df.groupby("cluster_kmeans")[FEATURES].mean()
    profile_z = (profile - profile.mean()) / profile.std(ddof=0).replace(0, 1)
    labels    = label_clusters(profile_z)
    df["archetype"] = df["cluster_kmeans"].map(labels)
    logger.info("Cluster archetypes: %s", labels)

    import json
    with open(f"{output_dir}/cluster_labels.json", "w", encoding="utf-8") as f:
        json.dump({str(k): v for k, v in labels.items()}, f, indent=2)

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
        "labels": labels,
    }


if __name__ == "__main__":
    import psycopg2
    from config.settings import DB_DSN
    logging.basicConfig(level=logging.INFO)
    conn = psycopg2.connect(DB_DSN)
    run(conn)
    conn.close()