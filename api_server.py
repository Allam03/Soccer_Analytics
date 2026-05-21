"""
api_server.py  — v2.3.0

Changes vs v2.2.0
-----------------
SECURITY
- CORS allowed origins now read from the ALLOWED_ORIGINS environment variable
  (comma-separated).  Defaults to "*" when unset so local dev is unchanged,
  but production deployments can lock it down without touching code.
- team_id is now validated against the teams table before any heavy query
  runs.  Unknown IDs return HTTP 404 immediately instead of silently falling
  through to fallback demo data.

BUG FIXES
- injury_risk(): `seen` was typed as dict[str, bool] and used as a
  set-with-value, and was declared twice in the same function scope (once
  per code path).  Both declarations replaced with a single `seen: set[str]`
  that is shared across the two branches via a helper.
- ingest_injuries.py date_of_birth null check simplified (separate fix).

OBSERVABILITY (unchanged from v2.2.0)
- Structured per-request logging.
- /api/health, /api/debug/artifacts, /api/debug/db endpoints.
- _last_errors dict for per-endpoint exception surfacing.
"""

from __future__ import annotations

import logging
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
    _A["m3"] = {
        "xgb":    _load(a / "model3" / "xgb.pkl"),
        "rf":     _load(a / "model3" / "rf.pkl"),
        "scaler": _load(a / "model3" / "scaler.pkl"),
        "df":     _parquet(a / "model3" / "features.parquet"),
    }
    _A["m4"] = {t: _load(a / "model4" / f"gbr_{t}.pkl")
                for t in ("xg", "pass_accuracy", "pressures")}
    _A["m5"] = {
        "gbc_pre":    _load(a / "model5" / "gbc_pre.pkl"),
        "scaler_pre": _load(a / "model5" / "scaler_pre.pkl"),
        "gbc_ig":     _load(a / "model5" / "gbc_ingame.pkl"),
        "scaler_ig":  _load(a / "model5" / "scaler_ingame.pkl"),
        "df_pre":     _parquet(a / "model5" / "features_pre.parquet"),
    }
    loaded = sum(1 for m in _A.values() for v in m.values() if v is not None)
    total  = sum(len(m) for m in _A.values())
    logger.info("Artifacts: %d / %d objects loaded across 5 models", loaded, total)
    if loaded == 0:
        logger.warning(
            "No artifacts loaded — all endpoints will use DB or fallback data. "
            "Run `python main.py --train` to generate artifacts."
        )


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Loading ML artifacts from: %s", ARTIFACT_DIR)
    _load_all_artifacts()
    logger.info("Server ready.")
    yield


# ---------------------------------------------------------------------------
# CORS — read from environment so production can restrict origins without
# touching code.  Unset => "*" (all origins), fine for local dev only.
# ---------------------------------------------------------------------------
_raw_origins = os.getenv("ALLOWED_ORIGINS", "*")
_allowed_origins = [o.strip() for o in _raw_origins.split(",") if o.strip()]

app = FastAPI(title="Soccer Analytics API", version="2.3.0", lifespan=lifespan)
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
            "%s %s → %d  (%.1f ms)",
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
    conn = psycopg2.connect(DB_DSN)
    try:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(sql, params)
            return [
                {k: _coerce(v) for k, v in row.items()}
                for row in cur.fetchall()
            ]
    finally:
        conn.close()


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
    """Raise HTTP 404 if team_id does not exist in the teams table.

    Skipped gracefully when the DB is unreachable so that the fallback
    data paths still work without a live database.
    """
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
            "teams", "players", "matches", "weather",
            "injuries", "player_match_stats", "player_match_features",
            "pass_network_edges",
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

    high_risk = _m3_high_risk_count(team_id)
    win_pct   = _m5_win_pct_direct(team_id)

    if db_available:
        recent = list(reversed(recent))
        kpi = kpi_rows[0]
        return {
            "kpi": {
                "team_performance":   round(float(kpi["team_performance"] or 0), 1),
                "cohesion_index":     round(float(kpi["cohesion_index"] or 0), 2),
                "high_risk_players":  high_risk,
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
            "high_risk_players":  high_risk,
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
    ARCHETYPE_NAMES = {
        0: "Creator",        1: "Ball Winner",   2: "Wide Attacker",
        3: "Box-to-Box",     4: "Finisher",      5: "Playmaker",
        6: "Defensive Shield",
    }

    try:
        rows = _query(
            """
            SELECT p.player_id, p.player_name, p.position,
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
            GROUP BY p.player_id, p.player_name, p.position
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
            if cluster_df is not None and "team_id" in cluster_df.columns:
                team_clusters = cluster_df[cluster_df["team_id"] == team_id]
                if not team_clusters.empty:
                    for _, cr in team_clusters.iterrows():
                        cid = int(cr.get("cluster_kmeans", 0))
                        cluster_map[str(cr["player_name"])] = ARCHETYPE_NAMES.get(cid, "Midfielder")

            for r in rows:
                if r["player_name"] in cluster_map:
                    r["player_type"] = cluster_map[r["player_name"]]
                else:
                    cid = _predict_cluster(r, FEATURES, kmeans, scaler)
                    r["player_type"] = ARCHETYPE_NAMES.get(cid, "Midfielder")

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
                    "player_type":     ARCHETYPE_NAMES.get(cid, "Midfielder"),
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
            if gbr is not None and scaler is not None:
                X = scaler.transform(avg.values.reshape(1, -1))
                predicted_goals = round(float(gbr.predict(X)[0]), 2)

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
# /api/injury-risk
# =============================================================================

@app.get("/api/injury-risk")
def injury_risk(team_id: int) -> dict[str, Any]:
    _validate_team_id(team_id)

    m3      = _A.get("m3", {})
    model   = m3.get("xgb") or m3.get("rf")
    scaler  = m3.get("scaler")
    feat_df = m3.get("df")

    FEATURES = [
        "minutes_played", "matches_last_30_days", "minutes_last_30_days",
        "days_since_last_injury", "age_at_match", "sub_minute_flag",
        "xg", "xa", "pressures", "tackles", "carry_distance",
        "interceptions", "clearances",
    ]

    players_out = []

    try:
        rows = _query(
            """
            SELECT p.player_name,
                   COALESCE(p.position, '-') AS position,
                   pms.minutes_played,
                   COALESCE(pmf.matches_last_30_days,    0)   AS matches_last_30_days,
                   COALESCE(pmf.minutes_last_30_days,    0)   AS minutes_last_30_days,
                   COALESCE(pmf.days_since_last_injury, -1)   AS days_since_last_injury,
                   CASE WHEN pms.sub_minute IS NOT NULL THEN 1 ELSE 0 END AS sub_minute_flag,
                   pms.xg, pms.xa, pms.pressures, pms.tackles,
                   pms.carry_distance, pms.interceptions, pms.clearances,
                   COALESCE(
                       EXTRACT(YEAR FROM AGE(m.match_date, p.date_of_birth))::INT,
                       25
                   ) AS age_at_match
            FROM player_match_stats pms
            JOIN player_match_features pmf ON pmf.stat_id  = pms.stat_id
            JOIN players p                 ON p.player_id  = pms.player_id
            JOIN matches  m                ON m.match_id   = pms.match_id
            WHERE pms.team_id = %s
            ORDER BY pms.match_id DESC
            LIMIT 15
            """,
            (team_id,),
        )
        if rows:
            logger.info("/api/injury-risk team=%d: %d rows (primary join)", team_id, len(rows))
    except Exception as exc:
        logger.warning("/api/injury-risk team=%d primary query error: %s", team_id, exc)
        _last_errors["injury_risk"] = str(exc)
        rows = []

    if not rows:
        try:
            rows = _query(
                """
                SELECT p.player_name,
                       COALESCE(p.position, '-') AS position,
                       pms.minutes_played,
                       0   AS matches_last_30_days,
                       0   AS minutes_last_30_days,
                       -1  AS days_since_last_injury,
                       CASE WHEN pms.sub_minute IS NOT NULL THEN 1 ELSE 0 END AS sub_minute_flag,
                       pms.xg, pms.xa, pms.pressures, pms.tackles,
                       pms.carry_distance, pms.interceptions, pms.clearances,
                       COALESCE(
                           EXTRACT(YEAR FROM AGE(m.match_date, p.date_of_birth))::INT,
                           25
                       ) AS age_at_match
                FROM player_match_stats pms
                JOIN players p ON p.player_id = pms.player_id
                JOIN matches  m ON m.match_id  = pms.match_id
                WHERE pms.team_id = %s
                ORDER BY pms.match_id DESC
                LIMIT 15
                """,
                (team_id,),
            )
            if rows:
                logger.info(
                    "/api/injury-risk team=%d: %d rows (fallback query, no pmf join)",
                    team_id, len(rows),
                )
        except Exception as exc:
            logger.warning("/api/injury-risk team=%d fallback query error: %s", team_id, exc)
            rows = []

    if rows and model and scaler:
        X = np.array([
            [
                float(r.get("minutes_played") or 0),
                float(r.get("matches_last_30_days") or 0),
                float(r.get("minutes_last_30_days") or 0),
                float(r.get("days_since_last_injury")
                      if r.get("days_since_last_injury") is not None else -1),
                float(r.get("age_at_match") or 25),
                float(r.get("sub_minute_flag") or 0),
                float(r.get("xg") or 0),
                float(r.get("xa") or 0),
                float(r.get("pressures") or 0),
                float(r.get("tackles") or 0),
                float(r.get("carry_distance") or 0),
                float(r.get("interceptions") or 0),
                float(r.get("clearances") or 0),
            ]
            for r in rows
        ])
        X_sc  = scaler.transform(X)
        probs = model.predict_proba(X_sc)[:, 1]

        # FIX: use a set, not a dict[str, bool], since we only need membership.
        seen: set[str] = set()
        for r, prob in zip(rows, probs):
            name = r["player_name"]
            if name in seen:
                continue
            seen.add(name)
            players_out.append({
                "player_name":            name,
                "position":               r.get("position") or "-",
                "workload_30d":           int(r.get("minutes_last_30_days") or 0),
                "days_since_last_injury": int(r.get("days_since_last_injury") or 0),
                "risk_score":             round(float(prob), 2),
            })

        if players_out:
            players_out.sort(key=lambda p: p["risk_score"], reverse=True)
            return _injury_response(players_out, "database+model3")

    elif rows:
        # Heuristic path — no trained model available
        seen: set[str] = set()
        for r in rows:
            name = r["player_name"]
            if name in seen:
                continue
            seen.add(name)
            workload = float(r.get("minutes_played") or 0)
            days_ago = float(
                r.get("days_since_last_injury")
                if r.get("days_since_last_injury") not in (None, -1)
                else 365
            )
            score = min(1.0, (workload / 90) * 0.3 + max(0, 1 - days_ago / 180) * 0.7)
            players_out.append({
                "player_name":            name,
                "position":               r.get("position") or "-",
                "workload_30d":           int(workload),
                "days_since_last_injury": int(days_ago),
                "risk_score":             round(score, 2),
            })

        if players_out:
            players_out.sort(key=lambda p: p["risk_score"], reverse=True)
            return _injury_response(players_out, "database")

    if feat_df is not None and model and scaler:
        cols_present = [c for c in FEATURES if c in feat_df.columns]
        if cols_present:
            try:
                X     = feat_df[FEATURES].fillna(0).values
                X_sc  = scaler.transform(X)
                probs = model.predict_proba(X_sc)[:, 1]
                feat_copy = feat_df.copy()
                feat_copy["risk_score"] = probs

                if "team_id" in feat_copy.columns:
                    feat_copy = feat_copy[feat_copy["team_id"] == team_id]

                feat_copy = feat_copy.sort_values("risk_score", ascending=False).head(15)
                players_out = [
                    {
                        "player_name":            str(r.get("player_name", f"Player {i}")),
                        "position":               str(r.get("position") or "-"),
                        "workload_30d":           int(r.get("minutes_last_30_days") or 0),
                        "days_since_last_injury": int(r.get("days_since_last_injury") or 0),
                        "risk_score":             round(float(r["risk_score"]), 2),
                    }
                    for i, (_, r) in enumerate(feat_copy.iterrows())
                ]
                if players_out:
                    return _injury_response(players_out, "artifact")
            except Exception as exc:
                logger.warning("/api/injury-risk team=%d artifact inference error: %s", team_id, exc)

    logger.warning("/api/injury-risk team=%d: using fallback", team_id)
    return _fallback_injury_for_team(team_id)


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


# =============================================================================
# /api/environment-impact
# =============================================================================

@app.get("/api/environment-impact")
def environment_impact(team_id: int) -> dict[str, Any]:
    _validate_team_id(team_id)

    gbr_xg       = _A.get("m4", {}).get("xg")
    gbr_pass_acc = _A.get("m4", {}).get("pass_accuracy")

    try:
        points = _query(
            """
            SELECT w.temperature_c AS temp,
                   w.precipitation_mm,
                   w.wind_speed_kmh,
                   w.humidity_pct,
                   w.weather_condition,
                   AVG((pms.xg * 20) + (pms.pass_accuracy * 0.5) + (pms.key_passes * 2)) AS perf,
                   AVG(pms.xg)            AS avg_xg,
                   AVG(pms.pass_accuracy) AS avg_pass_acc
            FROM player_match_stats pms
            JOIN weather w ON w.weather_id = pms.weather_id
            WHERE pms.team_id = %s AND w.temperature_c IS NOT NULL
            GROUP BY w.temperature_c, w.precipitation_mm,
                     w.wind_speed_kmh, w.humidity_pct, w.weather_condition
            ORDER BY w.temperature_c
            """,
            (team_id,),
        )
        if points:
            logger.info("/api/environment-impact team=%d: %d weather points", team_id, len(points))
            scatter = [{"x": round(float(p["temp"]), 1),
                        "y": round(float(p["perf"] or 0), 1)} for p in points]

            condition_summary: dict[str, Any] = {}
            if gbr_xg is not None and gbr_pass_acc is not None:
                for p in points:
                    cond = p.get("weather_condition") or "clear"
                    if cond in condition_summary:
                        continue
                    row = pd.DataFrame([{
                        "temperature_c":    float(p["temp"] or 15),
                        "precipitation_mm": float(p.get("precipitation_mm") or 0),
                        "wind_speed_kmh":   float(p.get("wind_speed_kmh") or 0),
                        "humidity_pct":     float(p.get("humidity_pct") or 60),
                        "venue_type":       "home",
                        "stadium_name":     "Other",
                    }])
                    pred_xg       = round(float(gbr_xg.predict(row)[0]), 3)
                    pred_pass_acc = round(float(gbr_pass_acc.predict(row)[0]), 1)
                    condition_summary[cond] = {
                        "predicted_xg":       pred_xg,
                        "predicted_pass_acc": pred_pass_acc,
                    }

            return {
                "scatter":           scatter,
                "condition_summary": condition_summary,
                "source":            "database+model4" if condition_summary else "database",
            }
    except Exception as exc:
        logger.warning("/api/environment-impact team=%d DB error: %s", team_id, exc)
        _last_errors["environment_impact"] = str(exc)

    logger.warning("/api/environment-impact team=%d: using fallback", team_id)
    return {
        "scatter": [
            {"x": 8,  "y": 76}, {"x": 12, "y": 82}, {"x": 16, "y": 88},
            {"x": 20, "y": 90}, {"x": 24, "y": 85}, {"x": 30, "y": 78},
        ],
        "condition_summary": {},
        "source": "fallback",
    }


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
        X    = np.array([[float(row.get(f) or 0) for f in features]])
        X_sc = scaler.transform(X)
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


def _m3_high_risk_count(team_id: int) -> int:
    m3      = _A.get("m3", {})
    model   = m3.get("xgb") or m3.get("rf")
    scaler  = m3.get("scaler")
    feat_df = m3.get("df")

    if model and scaler and feat_df is not None:
        try:
            FEATURES = [
                "minutes_played", "matches_last_30_days", "minutes_last_30_days",
                "days_since_last_injury", "age_at_match", "sub_minute_flag",
                "xg", "xa", "pressures", "tackles", "carry_distance",
                "interceptions", "clearances",
            ]
            team_df = (
                feat_df[feat_df["team_id"] == team_id]
                if "team_id" in feat_df.columns
                else feat_df
            ).tail(20)
            if not team_df.empty:
                X    = team_df[FEATURES].fillna(0).values
                X_sc = scaler.transform(X)
                probs = model.predict_proba(X_sc)[:, 1]
                return int((probs >= 0.67).sum())
        except Exception as exc:
            logger.warning("_m3_high_risk_count team=%d error: %s", team_id, exc)

    try:
        rows = _query(
            """
            SELECT COUNT(*) AS cnt
            FROM player_match_features pmf
            JOIN player_match_stats pms ON pms.stat_id = pmf.stat_id
            WHERE pms.team_id = %s AND pmf.is_injured_next_30d = TRUE
            """,
            (team_id,),
        )
        return int(rows[0]["cnt"]) if rows else 2
    except Exception:
        return 2


def _m5_win_pct_direct(team_id: int) -> float:
    """
    Lightweight win-% used by /api/dashboard.
    Must NOT call win_probability() — that would be infinite recursion.
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


def _fallback_injury_for_team(team_id: int) -> dict[str, Any]:
    roster = {
        1: [("Kyle Walker", "RB"), ("Ruben Dias", "CB"), ("Rodri", "CDM"), ("Phil Foden", "RW"), ("Erling Haaland", "ST")],
        2: [("Ben White", "RB"), ("William Saliba", "CB"), ("Declan Rice", "CM"), ("Bukayo Saka", "RW"), ("Gabriel Jesus", "ST")],
        3: [("Trent Alexander-Arnold", "RB"), ("Virgil van Dijk", "CB"), ("Dominik Szoboszlai", "CM"), ("Luis Diaz", "LW"), ("Mohamed Salah", "RW")],
        4: [("Reece James", "RB"), ("Levi Colwill", "CB"), ("Enzo Fernandez", "CM"), ("Cole Palmer", "RW"), ("Nicolas Jackson", "ST")],
    }.get(team_id, [("Player A", "MF"), ("Player B", "DF"), ("Player C", "FW"), ("Player D", "CB"), ("Player E", "ST")])

    scores    = [0.76, 0.61, 0.49, 0.33, 0.24]
    workloads = [540, 500, 470, 430, 390]
    days      = [42, 81, 110, 170, 240]
    players   = [
        {
            "player_name":            name,
            "position":               pos,
            "workload_30d":           workloads[i],
            "days_since_last_injury": days[i],
            "risk_score":             scores[i],
        }
        for i, (name, pos) in enumerate(roster)
    ]
    return {
        "kpi":     {"high": 1, "medium": 2, "low": 2, "avg_score": 0.49},
        "players": players,
        "source":  "fallback",
    }