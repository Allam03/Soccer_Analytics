from __future__ import annotations

from pathlib import Path
from typing import Any

import psycopg2
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles
from psycopg2.extras import RealDictCursor

from config.settings import DB_DSN

BASE_DIR = Path(__file__).resolve().parent
FRONTEND_DIR = BASE_DIR / "front-end"

app = FastAPI(title="Soccer Analytics API", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")


def _query(sql: str, params: tuple[Any, ...] = ()) -> list[dict[str, Any]]:
    conn = psycopg2.connect(DB_DSN)
    try:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(sql, params)
            return list(cur.fetchall())
    finally:
        conn.close()


def _fallback_dashboard(team_id: int) -> dict[str, Any]:
    offset = team_id % 4
    values = [79 + offset, 82 + offset, 81 + offset, 85 + offset, 84 + offset, 86 + offset, 87 + offset, 89 + offset, 88 + offset, 90 + offset]
    return {
        "kpi": {
            "team_performance": round(sum(values) / len(values), 1),
            "cohesion_index": 0.78 + (offset * 0.01),
            "high_risk_players": 2 + offset,
            "next_match_win_pct": 62 + offset * 2,
        },
        "performance_trend": {"labels": [str(i) for i in range(1, 11)], "values": values},
        "source": "fallback",
    }


def _fallback_player(team_id: int) -> dict[str, Any]:
    team_players = {
        1: ["Kevin De Bruyne", "Erling Haaland", "Rodri", "Phil Foden", "Bernardo Silva"],
        2: ["Martin Odegaard", "Bukayo Saka", "Declan Rice", "Kai Havertz", "William Saliba"],
        3: ["Mohamed Salah", "Virgil van Dijk", "Alexis Mac Allister", "Trent Alexander-Arnold", "Darwin Nunez"],
        4: ["Cole Palmer", "Enzo Fernandez", "Reece James", "Nicolas Jackson", "Levi Colwill"],
    }
    roster = team_players.get(team_id, team_players[1])
    leader_name = roster[0]
    leader = {
        "player_name": leader_name,
        "position": "Midfielder",
        "matches": 34,
        "minutes": 2650,
        "xg_per_90": 0.41,
        "xa_per_90": 0.57,
        "pass_completion": 88.4,
        "key_passes": 3.6,
        "dribbles": 2.1,
        "shots": 2.9,
    }
    players = []
    for idx, name in enumerate(roster):
        players.append(
            {
                "player_name": name,
                "position": "Midfielder" if idx < 3 else "Forward",
                "matches": 30 - idx,
                "minutes": 2400 - (idx * 120),
                "xg_per_90": round(0.25 + idx * 0.03, 2),
                "xa_per_90": round(0.30 + idx * 0.04, 2),
                "pass_completion": round(84 + idx, 1),
                "key_passes": round(2.1 + idx * 0.3, 1),
                "dribbles": round(1.8 + idx * 0.2, 1),
                "shots": round(2.3 + idx * 0.25, 1),
            }
        )

    return {
        "leader": leader,
        "radar": {"labels": ["xG", "xA", "Passing", "Key Passes", "Dribbles", "Shots"], "values": [41, 57, 88, 90, 65, 58]},
        "players": players,
        "source": "fallback",
    }


def _fallback_cohesion() -> dict[str, Any]:
    return {
        "kpi": {"cohesion_index": 0.82, "network_density": 0.74, "avg_degree": 8.2, "clustering_coeff": 0.68},
        "edges": [
            {"from": "Player A", "to": "Player B", "weight": 8.0},
            {"from": "Player B", "to": "Player C", "weight": 7.0},
            {"from": "Player C", "to": "Player D", "weight": 6.0},
        ],
        "source": "fallback",
    }


def _fallback_injury() -> dict[str, Any]:
    return {"kpi": {"high": 1, "medium": 1, "low": 1, "avg_score": 0.54}, "players": [], "source": "fallback"}


def _fallback_injury_for_team(team_id: int) -> dict[str, Any]:
    team_rosters = {
        1: [("Kyle Walker", "RB"), ("Ruben Dias", "CB"), ("Rodri", "CDM"), ("Phil Foden", "RW"), ("Erling Haaland", "ST")],
        2: [("Ben White", "RB"), ("William Saliba", "CB"), ("Declan Rice", "CM"), ("Bukayo Saka", "RW"), ("Gabriel Jesus", "ST")],
        3: [("Trent Alexander-Arnold", "RB"), ("Virgil van Dijk", "CB"), ("Dominik Szoboszlai", "CM"), ("Luis Diaz", "LW"), ("Mohamed Salah", "RW")],
        4: [("Reece James", "RB"), ("Levi Colwill", "CB"), ("Enzo Fernandez", "CM"), ("Cole Palmer", "RW"), ("Nicolas Jackson", "ST")],
    }
    roster = team_rosters.get(team_id, team_rosters[1])
    scores = [0.76, 0.61, 0.49, 0.33, 0.24]
    workloads = [540, 500, 470, 430, 390]
    days = [42, 81, 110, 170, 240]
    players = []
    for i, (name, pos) in enumerate(roster):
        players.append(
            {
                "player_name": name,
                "position": pos,
                "workload_30d": workloads[i],
                "days_since_last_injury": days[i],
                "risk_score": scores[i],
            }
        )
    return {"kpi": {"high": 1, "medium": 2, "low": 2, "avg_score": 0.49}, "players": players, "source": "fallback"}


def _fallback_environment() -> dict[str, Any]:
    return {
        "scatter": [{"x": 8, "y": 76}, {"x": 12, "y": 82}, {"x": 16, "y": 88}, {"x": 20, "y": 90}, {"x": 24, "y": 85}, {"x": 30, "y": 78}],
        "source": "fallback",
    }


def _fallback_winprob(team_id: int) -> dict[str, Any]:
    base = 60 + (team_id % 4) * 2
    draw = 24 - (team_id % 3)
    loss = 100 - base - draw
    minutes = list(range(91))
    win = [max(0, min(100, round(base + (m - 45) * 0.07, 1))) for m in minutes]
    draw_series = [max(0, min(100, round(draw - abs(m - 45) * 0.02, 1))) for m in minutes]
    loss_series = [round(max(0, 100 - win[i] - draw_series[i]), 1) for i in range(len(minutes))]
    return {
        "headline": {"win": base, "draw": draw, "loss": loss},
        "timeline": {"labels": minutes, "win": win, "draw": draw_series, "loss": loss_series},
        "source": "fallback",
    }


@app.get("/")
def index() -> FileResponse:
    return FileResponse(FRONTEND_DIR / "index.html")


@app.get("/favicon.ico")
def favicon() -> Response:
    return Response(status_code=204)


@app.get("/api/options/teams")
def teams() -> dict[str, Any]:
    try:
        rows = _query("SELECT t.team_id, t.team_name FROM teams t ORDER BY t.team_name")
        if rows:
            return {"teams": rows, "source": "database"}
    except Exception:
        pass
    return {
        "teams": [
            {"team_id": 1, "team_name": "Manchester City"},
            {"team_id": 2, "team_name": "Arsenal"},
            {"team_id": 3, "team_name": "Liverpool"},
            {"team_id": 4, "team_name": "Chelsea"},
        ],
        "source": "fallback",
    }


@app.get("/api/dashboard")
def dashboard(team_id: int) -> dict[str, Any]:
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
              AVG((pms.xg * 20) + (pms.pass_accuracy * 0.5) + (pms.key_passes * 2)) AS team_performance,
              AVG(pms.pass_accuracy) / 100.0 AS cohesion_index,
              COUNT(*) FILTER (WHERE pmf.is_injured_next_30d) AS high_risk_players
            FROM player_match_stats pms
            LEFT JOIN player_match_features pmf ON pmf.stat_id = pms.stat_id
            WHERE pms.team_id = %s
            """,
            (team_id,),
        )
        win_rows = _query(
            "SELECT AVG(CASE WHEN pms.result = 'win' THEN 1 ELSE 0 END) * 100 AS win_pct FROM player_match_stats pms WHERE pms.team_id = %s",
            (team_id,),
        )
        if not recent or not kpi_rows:
            return _fallback_dashboard(team_id)
        recent = list(reversed(recent))
        kpi = kpi_rows[0]
        win = win_rows[0] if win_rows else {"win_pct": 0}
        return {
            "kpi": {
                "team_performance": round(float(kpi["team_performance"] or 0), 1),
                "cohesion_index": round(float(kpi["cohesion_index"] or 0), 2),
                "high_risk_players": int(kpi["high_risk_players"] or 0),
                "next_match_win_pct": round(float(win["win_pct"] or 0), 1),
            },
            "performance_trend": {"labels": [r["match_date"].isoformat() for r in recent], "values": [round(float(r["perf"] or 0), 1) for r in recent]},
            "source": "database",
        }
    except Exception:
        return _fallback_dashboard(team_id)


@app.get("/api/player-efficiency")
def player_efficiency(team_id: int) -> dict[str, Any]:
    try:
        rows = _query(
            """
            SELECT p.player_name, p.position, COUNT(*) AS matches, AVG(pms.minutes_played) * COUNT(*) AS minutes,
                   AVG(pms.xg) AS xg_per_90, AVG(pms.xa) AS xa_per_90, AVG(pms.pass_accuracy) AS pass_completion,
                   AVG(pms.key_passes) AS key_passes, AVG(pms.dribbles_completed) AS dribbles, AVG(pms.shots) AS shots
            FROM player_match_stats pms
            JOIN players p ON p.player_id = pms.player_id
            WHERE pms.team_id = %s
            GROUP BY p.player_name, p.position
            HAVING COUNT(*) >= 3
            ORDER BY AVG(pms.xa + pms.xg) DESC
            LIMIT 12
            """,
            (team_id,),
        )
        if not rows:
            return _fallback_player(team_id)
        leader = rows[0]
        radar = {
            "labels": ["xG", "xA", "Passing", "Key Passes", "Dribbles", "Shots"],
            "values": [
                min(100, round(float(leader["xg_per_90"] or 0) * 100, 1)),
                min(100, round(float(leader["xa_per_90"] or 0) * 100, 1)),
                round(float(leader["pass_completion"] or 0), 1),
                min(100, round(float(leader["key_passes"] or 0) * 25, 1)),
                min(100, round(float(leader["dribbles"] or 0) * 20, 1)),
                min(100, round(float(leader["shots"] or 0) * 20, 1)),
            ],
        }
        return {"leader": leader, "radar": radar, "players": rows, "source": "database"}
    except Exception:
        return _fallback_player(team_id)


@app.get("/api/team-cohesion")
def team_cohesion(team_id: int) -> dict[str, Any]:
    try:
        edges = _query(
            """
            SELECT p1.player_name AS passer, p2.player_name AS receiver, AVG(pne.pass_count) AS weight
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
        if not edges:
            return _fallback_cohesion()
        weights = [float(e["weight"] or 0) for e in edges]
        return {
            "kpi": {
                "cohesion_index": round(sum(weights) / (len(weights) * 10), 2),
                "network_density": round(min(1.0, len(edges) / 30), 2),
                "avg_degree": round((2 * len(edges)) / 11, 1),
                "clustering_coeff": round(min(1.0, sum(weights) / (len(weights) * 12)), 2),
            },
            "edges": [{"from": e["passer"], "to": e["receiver"], "weight": round(float(e["weight"] or 0), 1)} for e in edges],
            "source": "database",
        }
    except Exception:
        return _fallback_cohesion()


@app.get("/api/injury-risk")
def injury_risk(team_id: int) -> dict[str, Any]:
    try:
        rows = _query(
            """
            SELECT p.player_name, COALESCE(p.position, '-') AS position, COALESCE(pmf.minutes_last_30_days, 0) AS workload_30d,
                   COALESCE(pmf.days_since_last_injury, 365) AS days_since_last_injury,
                   COALESCE(CASE WHEN pmf.is_injured_next_30d THEN 0.75 ELSE 0.25 END, 0.25) AS risk_score
            FROM player_match_features pmf
            JOIN player_match_stats pms ON pms.stat_id = pmf.stat_id
            JOIN players p ON p.player_id = pmf.player_id
            WHERE pms.team_id = %s
            ORDER BY pmf.match_id DESC
            LIMIT 15
            """,
            (team_id,),
        )
        if not rows:
            return _fallback_injury_for_team(team_id)
        high = sum(1 for r in rows if float(r["risk_score"]) >= 0.67)
        med = sum(1 for r in rows if 0.4 <= float(r["risk_score"]) < 0.67)
        low = len(rows) - high - med
        return {"kpi": {"high": high, "medium": med, "low": low, "avg_score": round(sum(float(r["risk_score"]) for r in rows) / len(rows), 2)}, "players": rows, "source": "database"}
    except Exception:
        return _fallback_injury_for_team(team_id)


@app.get("/api/environment-impact")
def environment_impact(team_id: int) -> dict[str, Any]:
    try:
        points = _query(
            """
            SELECT w.temperature_c AS temp,
                   AVG((pms.xg * 20) + (pms.pass_accuracy * 0.5) + (pms.key_passes * 2)) AS perf
            FROM player_match_stats pms
            JOIN weather w ON w.weather_id = pms.weather_id
            WHERE pms.team_id = %s AND w.temperature_c IS NOT NULL
            GROUP BY w.temperature_c
            ORDER BY w.temperature_c
            """,
            (team_id,),
        )
        if not points:
            return _fallback_environment()
        return {"scatter": [{"x": round(float(p["temp"]), 1), "y": round(float(p["perf"] or 0), 1)} for p in points], "source": "database"}
    except Exception:
        return _fallback_environment()


@app.get("/api/win-probability")
def win_probability(team_id: int) -> dict[str, Any]:
    try:
        rows = _query(
            """
            SELECT AVG(CASE WHEN result='win' THEN 1 ELSE 0 END) AS win_rate,
                   AVG(CASE WHEN result='draw' THEN 1 ELSE 0 END) AS draw_rate,
                   AVG(CASE WHEN result='loss' THEN 1 ELSE 0 END) AS loss_rate
            FROM player_match_stats
            WHERE team_id = %s
            """,
            (team_id,),
        )
        rates = rows[0] if rows else {"win_rate": 0, "draw_rate": 0, "loss_rate": 0}
        win = round(float(rates["win_rate"] or 0) * 100, 1)
        draw = round(float(rates["draw_rate"] or 0) * 100, 1)
        loss = round(float(rates["loss_rate"] or 0) * 100, 1)
        minutes = list(range(91))
        series_win = [max(0, min(100, round(win + (m - 45) * 0.08, 1))) for m in minutes]
        series_draw = [max(0, min(100, round(draw - abs(m - 45) * 0.03, 1))) for m in minutes]
        series_loss = [round(max(0, 100 - series_win[i] - series_draw[i]), 1) for i in range(len(minutes))]
        return {"headline": {"win": win, "draw": draw, "loss": loss}, "timeline": {"labels": minutes, "win": series_win, "draw": series_draw, "loss": series_loss}, "source": "database"}
    except Exception:
        return _fallback_winprob(team_id)
