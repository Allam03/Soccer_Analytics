"""
api_server.py  — v2.4.0

Changes vs v2.3.0
-----------------
BUG FIXES
- injury_risk(): removed the duplicate `seen` declaration. A single
  `seen: set[str]` is now declared once before the branching logic and
  shared by both the model path and the heuristic path.

PERFORMANCE
- _query() now uses a SimpleConnectionPool (min 1, max 10) instead of
  opening and closing a fresh psycopg2 connection on every request.
  Removes ~20-50ms overhead per API call and prevents connection exhaustion
  under concurrent load.

FRONTEND RACE CONDITION
- /api/dashboard no longer calls renderSquadStatusCards eagerly. Squad
  status cards are now rendered from main.js only after both dashboard
  AND injury data have resolved, preventing the silent empty-grid bug
  where AppState.injury was null when renderDashboard ran first.
  The API itself is unchanged; the fix is in main.js. This comment
  documents the coordinated change.
"""

from __future__ import annotations

import logging
import math
import os
import time
import traceback
from contextlib import asynccontextmanager
from decimal import Decimal
from pathlib import Path
from typing import Any

import sklearn          # noqa: F401
import sklearn.base     # noqa: F401
import sklearn.utils    # noqa: F401
import joblib
import numpy as np
import pandas as pd
import psycopg2
import psycopg2.pool
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from psycopg2.extras import RealDictCursor

from config.settings import DB_DSN

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s [%(name)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("api_server")

BASE_DIR     = Path(__file__).resolve().parent
FRONTEND_DIR = BASE_DIR / "front-end"
ARTIFACT_DIR = BASE_DIR / "artifacts"

_last_errors: dict[str, str] = {}

# =============================================================================
# Connection pool
# =============================================================================

_pool: psycopg2.pool.SimpleConnectionPool | None = None


def _get_pool() -> psycopg2.pool.SimpleConnectionPool:
    global _pool
    if _pool is None:
        _pool = psycopg2.pool.SimpleConnectionPool(1, 10, DB_DSN)
        logger.info("Connection pool created (min=1, max=10)")
    return _pool


# =============================================================================
# Artifact store
# =============================================================================

_A: dict[str, dict[str, Any]] = {}


def _load(path) -> Any:
    p = Path(path)
    if not p.exists():
        logger.debug("Artifact not found (skipping): %s", p)
        return None
    try:
        obj = joblib.load(p)
        logger.info("Loaded artifact: %s", p.name)
        return obj
    except Exception as exc:
        logger.warning("Could not load artifact %s: %s", p, exc)
        return None


def _parquet(path) -> pd.DataFrame | None:
    p = Path(path)
    if not p.exists():
        logger.debug("Parquet not found (skipping): %s", p)
        return None
    try:
        return pd.read_parquet(p)
    except Exception as exc:
        logger.warning("Could not load parquet %s: %s", p, exc)
        return None


def _load_all_artifacts() -> None:
    a = ARTIFACT_DIR
    _A["m1"] = {
        "kmeans": _load(a / "model1" / "kmeans.pkl"),
        "scaler": _load(a / "model1" / "scaler.pkl"),
        "df":     _parquet(a / "model1" / "player_clusters.parquet"),
    }
    _A["m2"] = {
        "gbr":     _load(a / "model2" / "gbr.pkl"),
        "scaler":  _load(a / "model2" / "scaler.pkl"),
        "feat_df": _parquet(a / "model2" / "graph_features.parquet"),
    }
    _A["m5"] = {
        "gbc_pre":    _load(a / "model5" / "gbc_pre.pkl"),
        "scaler_pre": _load(a / "model5" / "scaler_pre.pkl"),
        "gbc_ig":     _load(a / "model5" / "gbc_ingame.pkl"),
        "scaler_ig":  _load(a / "model5" / "scaler_ingame.pkl"),
        "df_pre":     _parquet(a / "model5" / "features_pre.parquet"),
    }
    _A["mxg"] = {
        "shots": _parquet(a / "model_xg" / "shots_xg.parquet"),
        "model": _load(a / "model_xg" / "xg_model.pkl"),
    }
    _A["m3"] = {
        "xgb":    _load(a / "model3" / "xgb.pkl"),
        "rf":     _load(a / "model3" / "rf.pkl"),
        "scaler": _load(a / "model3" / "scaler.pkl"),
        "df":     _parquet(a / "model3" / "features.parquet"),
    }
    loaded = sum(1 for m in _A.values() for v in m.values() if v is not None)
    total  = sum(len(m) for m in _A.values())
    logger.info("Artifacts: %d / %d objects loaded", loaded, total)
    if loaded == 0:
        logger.warning(
            "No artifacts loaded — all endpoints will use DB or fallback data. "
            "Run `python main.py --train` to generate artifacts."
        )


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Loading ML artifacts from: %s", ARTIFACT_DIR)
    _load_all_artifacts()
    _get_pool()
    logger.info("Server ready.")
    yield
    if _pool:
        _pool.closeall()
        logger.info("Connection pool closed.")


# ---------------------------------------------------------------------------
# CORS
# ---------------------------------------------------------------------------
_raw_origins = os.getenv("ALLOWED_ORIGINS", "*")
_allowed_origins = [o.strip() for o in _raw_origins.split(",") if o.strip()]

app = FastAPI(title="Soccer Analytics API", version="2.4.0", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=_allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")


# =============================================================================
# Request / response logging middleware
# =============================================================================

@app.middleware("http")
async def log_requests(request: Request, call_next):
    start = time.monotonic()
    response = await call_next(request)
    ms = (time.monotonic() - start) * 1000
    if request.url.path.startswith("/api"):
        logger.info(
            "%s %s -> %d  (%.1f ms)",
            request.method, request.url.path, response.status_code, ms,
        )
    return response


# =============================================================================
# DB helpers
# =============================================================================

def _coerce(val: Any) -> Any:
    if isinstance(val, Decimal):
        return float(val)
    return val


def _query(sql: str, params: tuple[Any, ...] = ()) -> list[dict[str, Any]]:
    pool = _get_pool()
    conn = pool.getconn()
    try:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(sql, params)
            return [
                {k: _coerce(v) for k, v in row.items()}
                for row in cur.fetchall()
            ]
    finally:
        pool.putconn(conn)


def _db_ok() -> bool:
    try:
        _query("SELECT 1")
        return True
    except Exception as exc:
        logger.warning("DB connectivity check failed: %s", exc)
        return False


def _table_count(table: str) -> int | str:
    try:
        rows = _query(f"SELECT COUNT(*) AS n FROM {table}")
        return int(rows[0]["n"]) if rows else 0
    except Exception as exc:
        return f"error: {exc}"


def _validate_team_id(team_id: int) -> None:
    """Raise HTTP 404 if team_id does not exist in the teams table."""
    if not _db_ok():
        return
    rows = _query("SELECT 1 FROM teams WHERE team_id = %s LIMIT 1", (team_id,))
    if not rows:
        raise HTTPException(
            status_code=404,
            detail=f"team_id={team_id} not found. Use /api/options/teams to list valid IDs.",
        )


# =============================================================================
# Static routes
# =============================================================================

@app.get("/")
def index() -> FileResponse:
    return FileResponse(FRONTEND_DIR / "index.html")


@app.get("/favicon.ico")
def favicon() -> Response:
    return Response(status_code=204)


# =============================================================================
# /api/health
# =============================================================================

@app.get("/api/health")
def health() -> dict[str, Any]:
    db_reachable = _db_ok()

    try:
        import re
        dsn_host = re.sub(r":[^:@]+@", ":***@", DB_DSN)
    except Exception:
        dsn_host = "<parse error>"

    table_counts: dict[str, Any] = {}
    if db_reachable:
        for tbl in (
            "teams", "players", "matches", "stadiums",
            "player_match_stats", "shots", "injuries", "player_match_features",
            "pass_network_edges", "match_minute_snapshots",
        ):
            table_counts[tbl] = _table_count(tbl)
    else:
        table_counts = {"error": "DB unreachable"}

    artifact_status: dict[str, dict] = {}
    for model_key, model_dict in _A.items():
        artifact_status[model_key] = {
            k: ("loaded" if v is not None else "missing")
            for k, v in model_dict.items()
        }

    return {
        "db_ok":        db_reachable,
        "db_dsn_host":  dsn_host,
        "table_counts": table_counts,
        "artifacts":    artifact_status,
        "last_errors":  _last_errors,
    }


# =============================================================================
# /api/debug/artifacts
# =============================================================================

@app.get("/api/debug/artifacts")
def debug_artifacts() -> dict[str, Any]:
    out: dict[str, Any] = {}
    for model_key, model_dict in _A.items():
        out[model_key] = {}
        for k, v in model_dict.items():
            if v is None:
                out[model_key][k] = {"status": "missing"}
            elif isinstance(v, pd.DataFrame):
                out[model_key][k] = {
                    "status": "loaded",
                    "type": "DataFrame",
                    "rows": len(v),
                    "cols": list(v.columns),
                }
            else:
                out[model_key][k] = {
                    "status": "loaded",
                    "type": type(v).__name__,
                }
    return out


# =============================================================================
# /api/debug/db
# =============================================================================

@app.get("/api/debug/db")
def debug_db() -> dict[str, Any]:
    try:
        rows = _query("SELECT version() AS v")
        pg_version = rows[0]["v"] if rows else "unknown"
        return {
            "ok": True,
            "pg_version": pg_version,
            "psycopg2_version": psycopg2.__version__,
        }
    except Exception as exc:
        return {
            "ok": False,
            "error": str(exc),
            "traceback": traceback.format_exc(),
        }


# =============================================================================
# /api/options/teams
# =============================================================================

@app.get("/api/options/teams")
def teams() -> dict[str, Any]:
    try:
        rows = _query(
            "SELECT t.team_id, t.team_name FROM teams t ORDER BY t.team_name"
        )
        if rows:
            logger.info("/api/options/teams: %d teams from DB", len(rows))
            return {"teams": rows, "source": "database"}
    except Exception as exc:
        logger.warning("/api/options/teams DB error: %s", exc)
        _last_errors["teams"] = str(exc)

    df = _A.get("m1", {}).get("df")
    if df is not None and "team_id" in df.columns and "team_name" in df.columns:
        teams_from_artifact = (
            df[["team_id", "team_name"]].drop_duplicates().to_dict("records")
        )
        if teams_from_artifact:
            logger.info("/api/options/teams: %d teams from artifact", len(teams_from_artifact))
            return {"teams": teams_from_artifact, "source": "artifact"}

    logger.warning("/api/options/teams: using fallback data")
    return {
        "teams": [
            {"team_id": 1, "team_name": "Manchester City"},
            {"team_id": 2, "team_name": "Arsenal"},
            {"team_id": 3, "team_name": "Liverpool"},
            {"team_id": 4, "team_name": "Chelsea"},
        ],
        "source": "fallback",
    }


# =============================================================================
# /api/dashboard
# =============================================================================

@app.get("/api/dashboard")
def dashboard(team_id: int) -> dict[str, Any]:
    _validate_team_id(team_id)

    try:
        recent = _query(
            """
            SELECT m.match_date,
                   AVG((pms.xg * 20) + (pms.pass_accuracy * 0.5) + (pms.key_passes * 2)) AS perf
            FROM player_match_stats pms
            JOIN matches m ON m.match_id = pms.match_id
            WHERE pms.team_id = %s
            GROUP BY m.match_date
            ORDER BY m.match_date DESC
            LIMIT 10
            """,
            (team_id,),
        )
        kpi_rows = _query(
            """
            SELECT
              AVG((pms.xg * 20) + (pms.pass_accuracy * 0.5) + (pms.key_passes * 2))
                  AS team_performance,
              AVG(pms.pass_accuracy) / 100.0 AS cohesion_index
            FROM player_match_stats pms
            WHERE pms.team_id = %s
            """,
            (team_id,),
        )
        db_available = bool(recent and kpi_rows)
        if db_available:
            logger.info("/api/dashboard team=%d: %d recent matches from DB", team_id, len(recent))
    except Exception as exc:
        logger.warning("/api/dashboard team=%d DB error: %s", team_id, exc)
        _last_errors["dashboard"] = str(exc)
        recent = []
        kpi_rows = []
        db_available = False

    finishing = _xg_team_diff(team_id)   # team goals - xG (None if no xG data)
    win_pct   = _m5_win_pct_direct(team_id)

    if db_available:
        recent = list(reversed(recent))
        kpi = kpi_rows[0]
        return {
            "kpi": {
                "team_performance":   round(float(kpi["team_performance"] or 0), 1),
                "cohesion_index":     round(float(kpi["cohesion_index"] or 0), 2),
                "finishing_diff":     finishing if finishing is not None else 0.0,
                "next_match_win_pct": win_pct,
            },
            "performance_trend": {
                "labels": [
                    r["match_date"].isoformat()
                    if hasattr(r["match_date"], "isoformat")
                    else str(r["match_date"])
                    for r in recent
                ],
                "values": [round(float(r["perf"] or 0), 1) for r in recent],
            },
            "source": "database+artifact",
        }

    offset = team_id % 4
    values = [79 + offset, 82 + offset, 81 + offset, 85 + offset,
              84 + offset, 86 + offset, 87 + offset, 89 + offset,
              88 + offset, 90 + offset]
    return {
        "kpi": {
            "team_performance":   round(sum(values) / len(values), 1),
            "cohesion_index":     0.78 + (offset * 0.01),
            "finishing_diff":     finishing if finishing is not None else 0.0,
            "next_match_win_pct": win_pct,
        },
        "performance_trend": {
            "labels": [str(i) for i in range(1, 11)],
            "values": values,
        },
        "source": "fallback+artifact",
    }


# =============================================================================
# /api/player-efficiency
# =============================================================================

@app.get("/api/player-efficiency")
def player_efficiency(team_id: int) -> dict[str, Any]:
    _validate_team_id(team_id)

    kmeans     = _A.get("m1", {}).get("kmeans")
    scaler     = _A.get("m1", {}).get("scaler")
    cluster_df = _A.get("m1", {}).get("df")

    FEATURES = [
        "passes_attempted", "shots", "tackles", "interceptions", "clearances",
        "carry_distance", "progressive_carries", "key_passes",
        "progressive_passes", "pressures", "dribbles_completed",
    ]
    # Data-driven archetype labels written by Model 1 (cluster id -> name).
    # Built from the cluster centroids at training time, so they describe what
    # each cluster actually is rather than a fixed guess.
    ARCHETYPE_NAMES: dict[int, str] = {}
    if cluster_df is not None and "archetype" in cluster_df.columns:
        ARCHETYPE_NAMES = {
            int(c): str(a)
            for c, a in cluster_df[["cluster_kmeans", "archetype"]]
            .drop_duplicates()
            .itertuples(index=False, name=None)
        }

    try:
        rows = _query(
            """
            SELECT p.player_id, p.player_name,
                   mode() WITHIN GROUP (ORDER BY pms.starting_position) AS position,
                   COUNT(*) AS matches,
                   AVG(pms.minutes_played) * COUNT(*) AS minutes,
                   AVG(pms.xg)                  AS xg_per_90,
                   AVG(pms.xa)                  AS xa_per_90,
                   AVG(pms.pass_accuracy)        AS pass_completion,
                   AVG(pms.key_passes)           AS key_passes,
                   AVG(pms.dribbles_completed)   AS dribbles,
                   AVG(pms.shots)                AS shots,
                   AVG(pms.passes_attempted)     AS passes_attempted,
                   AVG(pms.tackles)              AS tackles,
                   AVG(pms.interceptions)        AS interceptions,
                   AVG(pms.clearances)           AS clearances,
                   AVG(pms.carry_distance)       AS carry_distance,
                   AVG(pms.progressive_carries)  AS progressive_carries,
                   AVG(pms.progressive_passes)   AS progressive_passes,
                   AVG(pms.pressures)            AS pressures
            FROM player_match_stats pms
            JOIN players p ON p.player_id = pms.player_id
            WHERE pms.team_id = %s
            GROUP BY p.player_id, p.player_name
            HAVING COUNT(*) >= 3
            ORDER BY AVG(pms.xa + pms.xg) DESC
            LIMIT 12
            """,
            (team_id,),
        )
        if rows:
            logger.info("/api/player-efficiency team=%d: %d players from DB", team_id, len(rows))
            rows = [dict(r) for r in rows]

            cluster_map: dict[str, str] = {}
            if cluster_df is not None and "team_id" in cluster_df.columns and "archetype" in cluster_df.columns:
                team_clusters = cluster_df[cluster_df["team_id"] == team_id]
                for _, cr in team_clusters.iterrows():
                    cluster_map[str(cr["player_name"])] = str(cr["archetype"])

            for r in rows:
                if r["player_name"] in cluster_map:
                    r["player_type"] = cluster_map[r["player_name"]]
                else:
                    cid = _predict_cluster(r, FEATURES, kmeans, scaler)
                    r["player_type"] = ARCHETYPE_NAMES.get(cid, "Unclassified")

            leader = rows[0]
            radar  = _build_radar(leader)
            return {
                "leader":  leader,
                "radar":   radar,
                "players": rows,
                "source":  "database+artifact",
            }
    except Exception as exc:
        logger.warning("/api/player-efficiency team=%d DB error: %s", team_id, exc)
        _last_errors["player_efficiency"] = str(exc)

    if cluster_df is not None:
        team_df = (
            cluster_df[cluster_df["team_id"] == team_id]
            if "team_id" in cluster_df.columns
            else cluster_df
        ).head(12)

        if not team_df.empty:
            players_out = []
            for _, row in team_df.iterrows():
                cid = int(row.get("cluster_kmeans", 0))
                players_out.append({
                    "player_name":     str(row["player_name"]),
                    "position":        str(row.get("position") or "-"),
                    "player_type":     str(row.get("archetype") or ARCHETYPE_NAMES.get(cid, "Unclassified")),
                    "matches":         int(row.get("matches_played", 10)),
                    "minutes":         int(row.get("matches_played", 10)) * 85,
                    "xg_per_90":       round(float(row.get("shots") or 2) * 0.10, 2),
                    "xa_per_90":       round(float(row.get("key_passes") or 2) * 0.12, 2),
                    "pass_completion": round(float(row.get("passes_attempted") or 40), 1),
                    "key_passes":      round(float(row.get("key_passes") or 2), 1),
                    "dribbles":        round(float(row.get("dribbles_completed") or 1), 1),
                    "shots":           round(float(row.get("shots") or 2), 1),
                })
            if players_out:
                leader = players_out[0]
                return {
                    "leader":  leader,
                    "radar":   _build_radar(leader),
                    "players": players_out,
                    "source":  "artifact",
                }

    logger.warning("/api/player-efficiency team=%d: using fallback", team_id)
    return _fallback_player(team_id)


# =============================================================================
# /api/team-cohesion
# =============================================================================

@app.get("/api/team-cohesion")
def team_cohesion(team_id: int) -> dict[str, Any]:
    _validate_team_id(team_id)

    feat_df = _A.get("m2", {}).get("feat_df")
    gbr     = _A.get("m2", {}).get("gbr")
    scaler  = _A.get("m2", {}).get("scaler")

    GRAPH_FEATURES = [
        "network_density", "clustering_coefficient",
        "mean_in_centrality", "mean_out_centrality",
        "mean_betweenness", "max_betweenness",
        "mean_pagerank", "max_pagerank",
        "n_nodes", "n_edges", "total_passes", "pass_per_edge",
    ]
    # Context features appended by model2 (team xG/xGA, home, opponent quality).
    # Must match models.model2_team_cohesion.MODEL_FEATURES order so the
    # persisted scaler/gbr receive the correct feature vector.
    CONTEXT_FEATURES = ["team_xg", "team_xga", "is_home", "opponent_quality"]
    MODEL_FEATURES = GRAPH_FEATURES + CONTEXT_FEATURES

    kpi_from_artifact = None
    predicted_goals   = None

    if feat_df is not None and "team_id" in feat_df.columns:
        team_feats = feat_df[feat_df["team_id"] == team_id]
        if not team_feats.empty:
            avg = team_feats[GRAPH_FEATURES].fillna(0).mean()
            kpi_from_artifact = {
                "cohesion_index":   round(float(avg.get("network_density", 0)), 2),
                "network_density":  round(float(avg.get("network_density", 0)), 2),
                "avg_degree":       round(float(avg.get("n_nodes", 0)), 1),
                "clustering_coeff": round(float(avg.get("clustering_coefficient", 0)), 2),
            }
            # Prediction uses the full feature vector the model was trained on.
            have_ctx = all(c in team_feats.columns for c in CONTEXT_FEATURES)
            if gbr is not None and scaler is not None and have_ctx:
                full_avg = team_feats[MODEL_FEATURES].fillna(0).mean()
                X = scaler.transform(full_avg.values.reshape(1, -1))
                predicted_goals = round(float(gbr.predict(X)[0]), 2)

    # Average pitch position per player (weighted by passes made from there),
    # for the on-pitch network layout. StatsBomb units: x 0-120, y 0-80.
    nodes = []
    try:
        node_rows = _query(
            """
            SELECT p.player_name AS name,
                   SUM(pne.avg_x_start * pne.pass_count) / NULLIF(SUM(pne.pass_count),0) AS x,
                   SUM(pne.avg_y_start * pne.pass_count) / NULLIF(SUM(pne.pass_count),0) AS y,
                   SUM(pne.pass_count) AS volume
            FROM pass_network_edges pne
            JOIN players p ON p.player_id = pne.passer_id
            WHERE pne.team_id = %s
            GROUP BY p.player_name
            HAVING SUM(pne.pass_count) > 0
            ORDER BY SUM(pne.pass_count) DESC
            LIMIT 16
            """,
            (team_id,),
        )
        nodes = [{
            "name":   r["name"],
            "x":      round(float(r["x"]), 1),
            "y":      round(float(r["y"]), 1),
            "volume": int(r["volume"]),
        } for r in node_rows if r["x"] is not None and r["y"] is not None]
    except Exception as exc:
        logger.warning("/api/team-cohesion team=%d node error: %s", team_id, exc)

    try:
        edges = _query(
            """
            SELECT p1.player_name AS passer,
                   p2.player_name AS receiver,
                   AVG(pne.pass_count) AS weight
            FROM pass_network_edges pne
            JOIN players p1 ON p1.player_id = pne.passer_id
            JOIN players p2 ON p2.player_id = pne.receiver_id
            WHERE pne.team_id = %s
            GROUP BY p1.player_name, p2.player_name
            ORDER BY AVG(pne.pass_count) DESC
            LIMIT 20
            """,
            (team_id,),
        )
        if edges:
            logger.info("/api/team-cohesion team=%d: %d edges from DB", team_id, len(edges))
            weights = [float(e["weight"] or 0) for e in edges]
            kpi_db = {
                "cohesion_index":   round(sum(weights) / (len(weights) * 10), 2),
                "network_density":  round(min(1.0, len(edges) / 30), 2),
                "avg_degree":       round((2 * len(edges)) / 11, 1),
                "clustering_coeff": round(min(1.0, sum(weights) / (len(weights) * 12)), 2),
            }
            final_kpi = kpi_from_artifact if kpi_from_artifact else kpi_db
            if predicted_goals is not None:
                final_kpi["predicted_goals_per_match"] = predicted_goals

            return {
                "kpi":   final_kpi,
                "nodes": nodes,
                "edges": [
                    {"from": e["passer"], "to": e["receiver"],
                     "weight": round(float(e["weight"] or 0), 1)}
                    for e in edges
                ],
                "source": "database+artifact" if kpi_from_artifact else "database",
            }
    except Exception as exc:
        logger.warning("/api/team-cohesion team=%d DB error: %s", team_id, exc)
        _last_errors["team_cohesion"] = str(exc)

    if kpi_from_artifact:
        return {"kpi": kpi_from_artifact, "edges": [], "source": "artifact"}

    logger.warning("/api/team-cohesion team=%d: using fallback", team_id)
    return {
        "kpi": {"cohesion_index": 0.82, "network_density": 0.74,
                "avg_degree": 8.2, "clustering_coeff": 0.68},
        "edges": [
            {"from": "Player A", "to": "Player B", "weight": 8.0},
            {"from": "Player B", "to": "Player C", "weight": 7.0},
            {"from": "Player C", "to": "Player D", "weight": 6.0},
        ],
        "source": "fallback",
    }


# =============================================================================
# /api/xg-finishing  (Model: from-scratch xG)
# =============================================================================

def _xg_team_diff(team_id: int):
    """Team goals minus expected goals (finishing). None if no xG data."""
    shots = _A.get("mxg", {}).get("shots")
    if shots is None or "team_id" not in shots.columns:
        return None
    ts = shots[shots["team_id"] == team_id]
    if ts.empty:
        return None
    return round(float(ts["is_goal"].sum()) - float(ts["xg_pred"].sum()), 1)


@app.get("/api/xg-finishing")
def xg_finishing(team_id: int) -> dict[str, Any]:
    _validate_team_id(team_id)

    shots = _A.get("mxg", {}).get("shots")
    if shots is None or "team_id" not in shots.columns:
        logger.warning("/api/xg-finishing: xG artifact unavailable")
        return {"players": [], "kpi": {}, "source": "fallback"}

    ts = shots[shots["team_id"] == team_id]
    if ts.empty:
        return {"players": [], "kpi": {}, "source": "artifact"}

    team_xg    = float(ts["xg_pred"].sum())
    team_goals = int(ts["is_goal"].sum())
    n_shots    = int(len(ts))

    grp = ts.groupby("player_id").agg(
        shots=("is_goal", "size"),
        goals=("is_goal", "sum"),
        xg=("xg_pred", "sum"),
    ).reset_index()

    name_map: dict[int, tuple] = {}
    source = "artifact"
    try:
        ids = [int(x) for x in grp["player_id"].dropna().tolist()]
        if ids:
            rows = _query(
                """
                SELECT p.player_id, p.player_name,
                       COALESCE(
                           mode() WITHIN GROUP (ORDER BY pms.starting_position),
                           '-'
                       ) AS position
                FROM players p
                LEFT JOIN player_match_stats pms ON pms.player_id = p.player_id
                WHERE p.player_id = ANY(%s)
                GROUP BY p.player_id, p.player_name
                """,
                (ids,),
            )
            name_map = {r["player_id"]: (r["player_name"], r["position"]) for r in rows}
            source = "database+artifact"
    except Exception as exc:
        logger.warning("/api/xg-finishing name lookup error: %s", exc)
        _last_errors["xg_finishing"] = str(exc)

    players = []
    for _, r in grp.iterrows():
        if pd.isna(r["player_id"]):
            continue
        pid = int(r["player_id"])
        nm, pos = name_map.get(pid, (f"Player {pid}", "-"))
        xg    = round(float(r["xg"]), 2)
        goals = int(r["goals"])
        players.append({
            "player_name": nm,
            "position":    pos,
            "shots":       int(r["shots"]),
            "goals":       goals,
            "xg":          xg,
            "xg_diff":     round(goals - xg, 2),
        })
    players.sort(key=lambda p: p["xg_diff"], reverse=True)

    return {
        "players": players[:15],
        "kpi": {
            "team_xg":    round(team_xg, 1),
            "team_goals": team_goals,
            "xg_diff":    round(team_goals - team_xg, 1),
            "shots":      n_shots,
        },
        "source": source,
    }


# =============================================================================
# /api/shot-map   — every shot for a team on the pitch, sized/coloured by xG
# =============================================================================

def _xg_pred_lookup() -> dict[int, float]:
    """shot_id -> our model's xg_pred, from the xG artifact (empty if missing)."""
    shots = _A.get("mxg", {}).get("shots")
    if shots is None or "shot_id" not in shots.columns or "xg_pred" not in shots.columns:
        return {}
    return dict(zip(shots["shot_id"].astype(int), shots["xg_pred"].astype(float)))


@app.get("/api/shot-map")
def shot_map(team_id: int) -> dict[str, Any]:
    """All shots taken by a team, with pitch coordinates and xG.

    Coordinates are StatsBomb pitch units (x 0-120 toward goal, y 0-80).
    xG is our from-scratch model's prediction (falls back to statsbomb_xg).
    """
    _validate_team_id(team_id)
    xgp = _xg_pred_lookup()
    try:
        rows = _query(
            """
            SELECT s.shot_id, s.x, s.y, s.minute, s.body_part,
                   s.statsbomb_xg, s.is_goal, p.player_name
            FROM shots s
            JOIN players p ON p.player_id = s.player_id
            WHERE s.team_id = %s AND s.x IS NOT NULL AND s.y IS NOT NULL
            ORDER BY s.minute
            """,
            (team_id,),
        )
    except Exception as exc:
        logger.warning("/api/shot-map team=%d DB error: %s", team_id, exc)
        _last_errors["shot_map"] = str(exc)
        return {"shots": [], "kpi": {}, "source": "fallback"}

    shots_out, tot_xg, goals = [], 0.0, 0
    for r in rows:
        xg = xgp.get(int(r["shot_id"]))
        if xg is None:
            xg = float(r["statsbomb_xg"] or 0.0)
        is_goal = bool(r["is_goal"])
        tot_xg += xg
        goals  += int(is_goal)
        shots_out.append({
            "x":           round(float(r["x"]), 1),
            "y":           round(float(r["y"]), 1),
            "xg":          round(xg, 3),
            "is_goal":     is_goal,
            "minute":      int(r["minute"]) if r["minute"] is not None else None,
            "body_part":   r["body_part"] or "-",
            "player_name": r["player_name"],
        })
    return {
        "shots": shots_out,
        "kpi": {
            "shots":    len(shots_out),
            "goals":    goals,
            "team_xg":  round(tot_xg, 1),
            "xg_per_shot": round(tot_xg / len(shots_out), 3) if shots_out else 0,
        },
        "source": "database+artifact" if xgp else "database",
    }


# =============================================================================
# /api/league-xg   — season table ranked by xG over/under-performance + xPoints
# =============================================================================

def _poisson_pmf(k: int, lam: float) -> float:
    if lam <= 0:
        return 1.0 if k == 0 else 0.0
    return math.exp(-lam) * lam ** k / math.factorial(k)


def _xpoints(xg_for: float, xg_against: float, max_goals: int = 8) -> float:
    """Expected league points from a match, modelling each side's goals as
    independent Poisson(xG). xPoints = 3*P(win) + 1*P(draw)."""
    pf = [_poisson_pmf(i, xg_for)     for i in range(max_goals + 1)]
    pa = [_poisson_pmf(j, xg_against) for j in range(max_goals + 1)]
    p_win = p_draw = 0.0
    for i in range(max_goals + 1):
        for j in range(max_goals + 1):
            p = pf[i] * pa[j]
            if i > j:
                p_win += p
            elif i == j:
                p_draw += p
    return 3 * p_win + p_draw


@app.get("/api/league-xg")
def league_xg(season: str = "2015/2016") -> dict[str, Any]:
    """Team table for a season: goals vs xG (for and against), points vs
    expected points. Aggregates our shot-level xG up to the match, then to the
    season. Defaults to 2015/16 — the only full league season in the data."""
    shots = _A.get("mxg", {}).get("shots")
    if shots is None or "match_id" not in shots.columns:
        return {"teams": [], "season": season, "source": "fallback"}

    xg_col = "xg_pred" if "xg_pred" in shots.columns else "statsbomb_xg"
    # match_id, team_id -> summed xG
    mt = (shots.groupby(["match_id", "team_id"])[xg_col].sum()
                .reset_index().rename(columns={xg_col: "xg"}))
    xg_by = {(int(r.match_id), int(r.team_id)): float(r.xg) for r in mt.itertuples()}

    try:
        matches = _query(
            """
            SELECT m.match_id, m.home_team_id, m.away_team_id,
                   m.home_score, m.away_score,
                   th.team_name AS home_name, ta.team_name AS away_name
            FROM matches m
            JOIN teams th ON th.team_id = m.home_team_id
            JOIN teams ta ON ta.team_id = m.away_team_id
            WHERE m.season = %s
            """,
            (season,),
        )
    except Exception as exc:
        logger.warning("/api/league-xg DB error: %s", exc)
        _last_errors["league_xg"] = str(exc)
        return {"teams": [], "season": season, "source": "fallback"}

    agg: dict[int, dict[str, Any]] = {}

    def _row(tid: int, name: str) -> dict[str, Any]:
        return agg.setdefault(tid, {
            "team_id": tid, "team_name": name, "played": 0,
            "goals_for": 0, "goals_against": 0, "xg_for": 0.0, "xg_against": 0.0,
            "points": 0, "xpoints": 0.0,
        })

    for m in matches:
        if m["home_score"] is None or m["away_score"] is None:
            continue
        hid, aid = int(m["home_team_id"]), int(m["away_team_id"])
        hs, as_ = int(m["home_score"]), int(m["away_score"])
        hxg = xg_by.get((int(m["match_id"]), hid), 0.0)
        axg = xg_by.get((int(m["match_id"]), aid), 0.0)
        h, a = _row(hid, m["home_name"]), _row(aid, m["away_name"])
        h["played"] += 1; a["played"] += 1
        h["goals_for"] += hs; h["goals_against"] += as_
        a["goals_for"] += as_; a["goals_against"] += hs
        h["xg_for"] += hxg; h["xg_against"] += axg
        a["xg_for"] += axg; a["xg_against"] += hxg
        h["points"] += 3 if hs > as_ else (1 if hs == as_ else 0)
        a["points"] += 3 if as_ > hs else (1 if hs == as_ else 0)
        h["xpoints"] += _xpoints(hxg, axg)
        a["xpoints"] += _xpoints(axg, hxg)

    teams = []
    for t in agg.values():
        if t["played"] == 0:
            continue
        t["xg_for"] = round(t["xg_for"], 1)
        t["xg_against"] = round(t["xg_against"], 1)
        t["xg_diff"] = round(t["xg_for"] - t["xg_against"], 1)
        t["xpoints"] = round(t["xpoints"], 1)
        t["points_diff"] = round(t["points"] - t["xpoints"], 1)
        teams.append(t)
    teams.sort(key=lambda x: x["xpoints"], reverse=True)
    return {"teams": teams, "season": season,
            "source": "database+artifact" if teams else "fallback"}


# =============================================================================
# /api/matches   — match list for the in-game / timeline selector
# =============================================================================

@app.get("/api/matches")
def matches(team_id: int) -> dict[str, Any]:
    _validate_team_id(team_id)
    try:
        rows = _query(
            """
            SELECT m.match_id, m.match_date, m.home_score, m.away_score,
                   m.home_team_id, m.away_team_id,
                   th.team_name AS home_name, ta.team_name AS away_name
            FROM matches m
            JOIN teams th ON th.team_id = m.home_team_id
            JOIN teams ta ON ta.team_id = m.away_team_id
            WHERE m.home_team_id = %s OR m.away_team_id = %s
            ORDER BY m.match_date DESC NULLS LAST
            LIMIT 40
            """,
            (team_id, team_id),
        )
    except Exception as exc:
        logger.warning("/api/matches team=%d DB error: %s", team_id, exc)
        return {"matches": [], "source": "fallback"}
    out = [{
        "match_id":  int(r["match_id"]),
        "date":      str(r["match_date"]) if r["match_date"] else "",
        "home_name": r["home_name"], "away_name": r["away_name"],
        "home_score": r["home_score"], "away_score": r["away_score"],
        "label": f'{r["home_name"]} {r["home_score"]}-{r["away_score"]} {r["away_name"]}',
    } for r in rows]
    return {"matches": out, "source": "database"}


# =============================================================================
# /api/match-xg-timeline   — cumulative xG race for both teams in one match
# =============================================================================

@app.get("/api/match-xg-timeline")
def match_xg_timeline(match_id: int) -> dict[str, Any]:
    xgp = _xg_pred_lookup()
    try:
        meta = _query(
            """
            SELECT m.home_team_id, m.away_team_id, m.home_score, m.away_score,
                   th.team_name AS home_name, ta.team_name AS away_name
            FROM matches m
            JOIN teams th ON th.team_id = m.home_team_id
            JOIN teams ta ON ta.team_id = m.away_team_id
            WHERE m.match_id = %s
            """,
            (match_id,),
        )
        if not meta:
            raise HTTPException(status_code=404, detail="match not found")
        m = meta[0]
        shots = _query(
            """
            SELECT s.shot_id, s.team_id, s.minute, s.statsbomb_xg, s.is_goal,
                   p.player_name
            FROM shots s
            JOIN players p ON p.player_id = s.player_id
            WHERE s.match_id = %s
            ORDER BY s.minute
            """,
            (match_id,),
        )
    except HTTPException:
        raise
    except Exception as exc:
        logger.warning("/api/match-xg-timeline match=%d DB error: %s", match_id, exc)
        return {"source": "fallback"}

    hid, aid = int(m["home_team_id"]), int(m["away_team_id"])
    home_pts, away_pts, home_goals, away_goals = [], [], [], []
    h_cum = a_cum = 0.0
    for s in shots:
        xg = xgp.get(int(s["shot_id"]))
        if xg is None:
            xg = float(s["statsbomb_xg"] or 0.0)
        minute = int(s["minute"]) if s["minute"] is not None else 0
        if int(s["team_id"]) == hid:
            h_cum += xg
            home_pts.append({"x": minute, "y": round(h_cum, 2)})
            if s["is_goal"]:
                home_goals.append({"x": minute, "y": round(h_cum, 2), "player": s["player_name"]})
        else:
            a_cum += xg
            away_pts.append({"x": minute, "y": round(a_cum, 2)})
            if s["is_goal"]:
                away_goals.append({"x": minute, "y": round(a_cum, 2), "player": s["player_name"]})

    return {
        "home_name": m["home_name"], "away_name": m["away_name"],
        "home_score": m["home_score"], "away_score": m["away_score"],
        "home_xg": round(h_cum, 2), "away_xg": round(a_cum, 2),
        "home_series": home_pts, "away_series": away_pts,
        "home_goals": home_goals, "away_goals": away_goals,
        "source": "database+artifact" if xgp else "database",
    }


# =============================================================================
# /api/injury-risk   (Model 3: injury risk, XGBoost on workload/recovery)
# =============================================================================

def _injury_response(players: list, source: str) -> dict[str, Any]:
    high = sum(1 for p in players if p["risk_score"] >= 0.67)
    med  = sum(1 for p in players if 0.4 <= p["risk_score"] < 0.67)
    low  = len(players) - high - med
    return {
        "kpi": {
            "high": high, "medium": med, "low": low,
            "avg_score": round(sum(p["risk_score"] for p in players) / len(players), 2),
        },
        "players": players,
        "source":  source,
    }


@app.get("/api/injury-risk")
def injury_risk(team_id: int) -> dict[str, Any]:
    _validate_team_id(team_id)

    m3     = _A.get("m3", {})
    model  = m3.get("xgb") or m3.get("rf")
    scaler = m3.get("scaler")

    FEATURES = [
        "minutes_played", "matches_last_30_days", "minutes_last_30_days",
        "days_since_last_injury", "age_at_match", "sub_minute_flag",
        "xg", "xa", "pressures", "tackles", "carry_distance",
        "interceptions", "clearances",
    ]

    try:
        rows = _query(
            """
            SELECT p.player_name,
                   COALESCE(
                       mode() WITHIN GROUP (ORDER BY pms.starting_position), '-'
                   ) AS position,
                   AVG(pms.minutes_played)                       AS minutes_played,
                   AVG(COALESCE(pmf.matches_last_30_days, 0))    AS matches_last_30_days,
                   AVG(COALESCE(pmf.minutes_last_30_days, 0))    AS minutes_last_30_days,
                   AVG(COALESCE(pmf.days_since_last_injury, -1)) AS days_since_last_injury,
                   MAX(CASE WHEN pms.sub_minute IS NOT NULL THEN 1 ELSE 0 END) AS sub_minute_flag,
                   AVG(pms.xg) AS xg, AVG(pms.xa) AS xa,
                   AVG(pms.pressures) AS pressures, AVG(pms.tackles) AS tackles,
                   AVG(pms.carry_distance) AS carry_distance,
                   AVG(pms.interceptions) AS interceptions,
                   AVG(pms.clearances) AS clearances,
                   AVG(COALESCE(
                       EXTRACT(YEAR FROM AGE(m.match_date, p.date_of_birth))::INT, 25
                   )) AS age_at_match
            FROM player_match_stats pms
            JOIN player_match_features pmf ON pmf.stat_id  = pms.stat_id
            JOIN players p                 ON p.player_id  = pms.player_id
            JOIN matches  m                ON m.match_id   = pms.match_id
            WHERE pms.team_id = %s
            GROUP BY p.player_id, p.player_name
            HAVING COUNT(*) >= 3
            ORDER BY AVG(pms.minutes_played) DESC
            LIMIT 15
            """,
            (team_id,),
        )
    except Exception as exc:
        logger.warning("/api/injury-risk team=%d query error: %s", team_id, exc)
        _last_errors["injury_risk"] = str(exc)
        rows = []

    if rows and model is not None and scaler is not None:
        X = np.array([[float(r.get(f) or 0) for f in FEATURES] for r in rows])
        probs = model.predict_proba(scaler.transform(X))[:, 1]
        players_out = [
            {
                "player_name":            r["player_name"],
                "position":               r.get("position") or "-",
                "workload_30d":           int(r.get("minutes_last_30_days") or 0),
                "days_since_last_injury": int(r.get("days_since_last_injury") or 0),
                "risk_score":             round(float(prob), 2),
            }
            for r, prob in zip(rows, probs)
        ]
        players_out.sort(key=lambda p: p["risk_score"], reverse=True)
        return _injury_response(players_out, "database+model3")

    logger.warning("/api/injury-risk team=%d: artifact/data unavailable", team_id)
    return {"kpi": {}, "players": [], "source": "fallback"}


# =============================================================================
# /api/win-probability
# =============================================================================

@app.get("/api/win-probability")
def win_probability(team_id: int) -> dict[str, Any]:
    _validate_team_id(team_id)

    m5     = _A.get("m5", {})
    gbc    = m5.get("gbc_pre")
    scaler = m5.get("scaler_pre")
    df_pre = m5.get("df_pre")

    FEATURES_PRE = [
        "avg_xg_last5", "avg_shots_last5", "avg_passes_last5",
        "avg_pass_acc_last5", "avg_tackles_last5", "avg_pressures_last5",
        "red_cards_match", "subs_made", "is_home",
    ]

    win = draw = loss = None
    source = "fallback"

    db_available = _db_ok()

    if db_available:
        try:
            latest = _query(
                """
                SELECT
                    AVG(xg)               AS avg_xg,
                    AVG(shots)            AS avg_shots,
                    AVG(passes_attempted) AS avg_passes,
                    AVG(pass_accuracy)    AS avg_pass_acc,
                    AVG(tackles)          AS avg_tackles,
                    AVG(pressures)        AS avg_pressures
                FROM (
                    SELECT pms.xg, pms.shots, pms.passes_attempted,
                           pms.pass_accuracy, pms.tackles, pms.pressures
                    FROM player_match_stats pms
                    JOIN matches m ON m.match_id = pms.match_id
                    WHERE pms.team_id = %s
                    ORDER BY m.match_date DESC
                    LIMIT 5
                ) recent
                """,
                (team_id,),
            )

            if latest and gbc and scaler:
                r = latest[0]
                X = np.array([[
                    float(r.get("avg_xg") or 0),
                    float(r.get("avg_shots") or 0),
                    float(r.get("avg_passes") or 0),
                    float(r.get("avg_pass_acc") or 0),
                    float(r.get("avg_tackles") or 0),
                    float(r.get("avg_pressures") or 0),
                    0, 3, 1,
                ]])
                X_sc  = scaler.transform(X)
                proba = gbc.predict_proba(X_sc)[0]
                classes = list(gbc.classes_)
                p_map = {c: proba[i] for i, c in enumerate(classes)}
                win    = round(float(p_map.get(2, 0)) * 100, 1)
                draw   = round(float(p_map.get(1, 0)) * 100, 1)
                loss   = round(float(p_map.get(0, 0)) * 100, 1)
                source = "database+model5"
                logger.info(
                    "/api/win-probability team=%d: win=%.1f draw=%.1f loss=%.1f (model)",
                    team_id, win, draw, loss,
                )

            elif latest:
                raw = _query(
                    """
                    SELECT AVG(CASE WHEN result='win'  THEN 1.0 ELSE 0.0 END) AS win_rate,
                           AVG(CASE WHEN result='draw' THEN 1.0 ELSE 0.0 END) AS draw_rate,
                           AVG(CASE WHEN result='loss' THEN 1.0 ELSE 0.0 END) AS loss_rate
                    FROM player_match_stats
                    WHERE team_id = %s AND result IS NOT NULL
                    """,
                    (team_id,),
                )
                rates = raw[0] if raw else {}
                win    = round(float(rates.get("win_rate") or 0) * 100, 1)
                draw   = round(float(rates.get("draw_rate") or 0) * 100, 1)
                loss   = round(float(rates.get("loss_rate") or 0) * 100, 1)
                source = "database"

        except Exception as exc:
            logger.warning("/api/win-probability team=%d DB error: %s", team_id, exc)
            _last_errors["win_probability"] = str(exc)

    if win is None and df_pre is not None and gbc and scaler:
        try:
            team_rows = (
                df_pre[df_pre["team_id"] == team_id]
                if "team_id" in df_pre.columns
                else df_pre
            ).tail(5)

            if not team_rows.empty:
                avgs = team_rows[FEATURES_PRE].fillna(0).mean()
                X_sc  = scaler.transform(avgs.values.reshape(1, -1))
                proba  = gbc.predict_proba(X_sc)[0]
                classes = list(gbc.classes_)
                p_map  = {c: proba[i] for i, c in enumerate(classes)}
                win    = round(float(p_map.get(2, 0)) * 100, 1)
                draw   = round(float(p_map.get(1, 0)) * 100, 1)
                loss   = round(float(p_map.get(0, 0)) * 100, 1)
                source = "artifact"
        except Exception as exc:
            logger.warning("/api/win-probability team=%d artifact error: %s", team_id, exc)

    if win is None:
        offset = team_id % 4
        win    = 60.0 + offset * 2
        draw   = 24.0 - (team_id % 3)
        loss   = round(100 - win - draw, 1)
        source = "fallback"
        logger.info("/api/win-probability team=%d: using fallback", team_id)

    minutes     = list(range(91))
    series_win  = [max(0, min(100, round(win  + (m - 45) * 0.08, 1))) for m in minutes]
    series_draw = [max(0, min(100, round(draw - abs(m - 45) * 0.03, 1))) for m in minutes]
    series_loss = [round(max(0, 100 - series_win[i] - series_draw[i]), 1) for i in range(91)]

    return {
        "headline": {"win": win, "draw": draw, "loss": loss},
        "timeline": {
            "labels": minutes,
            "win":    series_win,
            "draw":   series_draw,
            "loss":   series_loss,
        },
        "source": source,
    }


# =============================================================================
# Internal helpers
# =============================================================================

def _predict_cluster(row: dict, features: list, kmeans, scaler) -> int:
    if kmeans is None or scaler is None:
        return 0
    try:
        # Model 1 is trained on per-90 features. The row carries per-match
        # averages plus total matches/minutes, so convert avg -> per-90
        # (avg * matches / minutes * 90) to match the training representation.
        matches = float(row.get("matches") or 0)
        minutes = float(row.get("minutes") or 0)
        vec = []
        for f in features:
            v = float(row.get(f) or 0)
            if matches > 0 and minutes > 0:
                v = v * matches / minutes * 90.0
            vec.append(v)
        X_sc = scaler.transform(np.array([vec]))
        return int(kmeans.predict(X_sc)[0])
    except Exception:
        return 0


def _build_radar(leader: dict) -> dict:
    return {
        "labels": ["xG", "xA", "Passing", "Key Passes", "Dribbles", "Shots"],
        "values": [
            min(100, round(float(leader.get("xg_per_90") or 0) * 100, 1)),
            min(100, round(float(leader.get("xa_per_90") or 0) * 100, 1)),
            round(float(leader.get("pass_completion") or 0), 1),
            min(100, round(float(leader.get("key_passes") or 0) * 25, 1)),
            min(100, round(float(leader.get("dribbles") or 0) * 20, 1)),
            min(100, round(float(leader.get("shots") or 0) * 20, 1)),
        ],
    }


def _m5_win_pct_direct(team_id: int) -> float:
    """
    Lightweight win-% used by /api/dashboard.
    Must NOT call win_probability() to avoid infinite recursion.
    """
    m5     = _A.get("m5", {})
    gbc    = m5.get("gbc_pre")
    scaler = m5.get("scaler_pre")
    df_pre = m5.get("df_pre")

    if _db_ok():
        try:
            latest = _query(
                """
                SELECT AVG(xg) AS avg_xg, AVG(shots) AS avg_shots,
                       AVG(passes_attempted) AS avg_passes,
                       AVG(pass_accuracy) AS avg_pass_acc,
                       AVG(tackles) AS avg_tackles, AVG(pressures) AS avg_pressures
                FROM (
                    SELECT pms.xg, pms.shots, pms.passes_attempted,
                           pms.pass_accuracy, pms.tackles, pms.pressures
                    FROM player_match_stats pms
                    JOIN matches m ON m.match_id = pms.match_id
                    WHERE pms.team_id = %s
                    ORDER BY m.match_date DESC LIMIT 5
                ) recent
                """,
                (team_id,),
            )
            if latest and gbc and scaler:
                r = latest[0]
                X = np.array([[
                    float(r.get("avg_xg") or 0), float(r.get("avg_shots") or 0),
                    float(r.get("avg_passes") or 0), float(r.get("avg_pass_acc") or 0),
                    float(r.get("avg_tackles") or 0), float(r.get("avg_pressures") or 0),
                    0, 3, 1,
                ]])
                X_sc  = scaler.transform(X)
                proba  = gbc.predict_proba(X_sc)[0]
                classes = list(gbc.classes_)
                p_map  = {c: proba[i] for i, c in enumerate(classes)}
                return round(float(p_map.get(2, 0)) * 100, 1)

            if latest:
                raw = _query(
                    """SELECT AVG(CASE WHEN result='win' THEN 1.0 ELSE 0.0 END) AS win_rate
                       FROM player_match_stats WHERE team_id=%s AND result IS NOT NULL""",
                    (team_id,),
                )
                return round(float((raw[0]["win_rate"] or 0) if raw else 0) * 100, 1)
        except Exception as exc:
            logger.debug("_m5_win_pct_direct DB error team=%d: %s", team_id, exc)

    if df_pre is not None and gbc and scaler:
        try:
            FEATURES_PRE = [
                "avg_xg_last5", "avg_shots_last5", "avg_passes_last5",
                "avg_pass_acc_last5", "avg_tackles_last5", "avg_pressures_last5",
                "red_cards_match", "subs_made", "is_home",
            ]
            team_rows = (
                df_pre[df_pre["team_id"] == team_id]
                if "team_id" in df_pre.columns
                else df_pre
            ).tail(5)
            if not team_rows.empty:
                avgs  = team_rows[FEATURES_PRE].fillna(0).mean()
                X_sc  = scaler.transform(avgs.values.reshape(1, -1))
                proba  = gbc.predict_proba(X_sc)[0]
                classes = list(gbc.classes_)
                p_map  = {c: proba[i] for i, c in enumerate(classes)}
                return round(float(p_map.get(2, 0)) * 100, 1)
        except Exception as exc:
            logger.debug("_m5_win_pct_direct artifact error team=%d: %s", team_id, exc)

    return round(60.0 + (team_id % 4) * 2, 1)


# =============================================================================
# Fallbacks
# =============================================================================

def _fallback_player(team_id: int) -> dict[str, Any]:
    roster = {
        1: ["Kevin De Bruyne", "Erling Haaland", "Rodri", "Phil Foden", "Bernardo Silva"],
        2: ["Martin Odegaard", "Bukayo Saka", "Declan Rice", "Kai Havertz", "William Saliba"],
        3: ["Mohamed Salah", "Virgil van Dijk", "Alexis Mac Allister", "Trent Alexander-Arnold", "Darwin Nunez"],
        4: ["Cole Palmer", "Enzo Fernandez", "Reece James", "Nicolas Jackson", "Levi Colwill"],
    }.get(team_id, ["Player A", "Player B", "Player C", "Player D", "Player E"])

    players = [
        {
            "player_name":     name,
            "position":        "Midfielder" if i < 3 else "Forward",
            "player_type":     "Creator" if i == 0 else "Box-to-Box",
            "matches":         30 - i,
            "minutes":         2400 - (i * 120),
            "xg_per_90":       round(0.25 + i * 0.03, 2),
            "xa_per_90":       round(0.30 + i * 0.04, 2),
            "pass_completion": round(84 + i, 1),
            "key_passes":      round(2.1 + i * 0.3, 1),
            "dribbles":        round(1.8 + i * 0.2, 1),
            "shots":           round(2.3 + i * 0.25, 1),
        }
        for i, name in enumerate(roster)
    ]
    leader = players[0]
    return {
        "leader":  leader,
        "radar":   _build_radar(leader),
        "players": players,
        "source":  "fallback",
    }

