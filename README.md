# Soccer Analytics ML Pipeline

A data pipeline and machine learning system built on StatsBomb open data, Open-Meteo historical weather, and Transfermarkt injury records. It ingests match events, computes per-player statistics, links injury history, and trains five models covering player clustering, team cohesion, injury risk prediction, environmental impact, and win probability. A FastAPI backend exposes all five models through a live analytics dashboard.

---

## Quick Start

### Requirements

- Python 3.11+
- PostgreSQL 14+
- ~4 GB disk for raw data and DB

```bash
pip install -r requirements.txt
```

### 1. Configure paths

Edit `config/settings.py`:

```python
DB_DSN                    = "postgresql://user:pass@localhost:5432/soccer_db"
DATA_ROOT                 = "/path/to/open-data-master/data"
TRANSFERMARKT_CSV         = "/path/to/transfermarkt_injuries.csv"
TRANSFERMARKT_PLAYERS_CSV = "/path/to/transfermarkt_players.csv"
```

### 2. Create the database schema

Run once before anything else:

```bash
python init_db.py
```

### 3. Run the full pipeline

```bash
python main.py
```

This runs all four ingestion and labelling stages in order. To also train the ML models:

```bash
python main.py --train
```

### 4. Launch the dashboard

```bash
uvicorn api_server:app --host 0.0.0.0 --port 8000 --reload
```

Open **http://localhost:8000** in your browser.

### Other pipeline flags

```
--skip-ingest     Skip ingestion (steps 1–3), run only labels and optional training
--workers N       Set number of parallel worker processes (default: CPU count − 1)
```

### Running pipeline stages individually

```bash
python -m pipelines.ingest_statsbomb   # StatsBomb events, players, teams, pass networks
python -m pipelines.ingest_weather     # Historical weather from Open-Meteo
python -m pipelines.ingest_injuries    # Transfermarkt player and injury data
python -m pipelines.compute_labels     # Workload features and injury risk labels
```

---

## Diagnosing problems

Before clicking around the dashboard, check the health endpoint:

```
http://localhost:8000/api/health
```

It reports DB connectivity, row counts for every table, which ML artifact files are loaded, and the most recent exception per API endpoint. The dashboard also has a built-in **Debug** page (click the 🔧 icon in the sidebar) that surfaces the same information without opening browser DevTools.

Additional diagnostic endpoints:

| Endpoint | What it shows |
|---|---|
| `/api/health` | DB status, table counts, artifact load state, last errors |
| `/api/debug/artifacts` | Type and shape of every loaded `.pkl` / `.parquet` file |
| `/api/debug/db` | PostgreSQL version, psycopg2 version |

**Common issues:**

- `db_ok: false` — check `DB_DSN` in `config/settings.py` and confirm PostgreSQL is running.
- Tables showing `0` rows — run the ingestion pipeline first (`python main.py`).
- Artifacts showing `missing` — run training (`python main.py --train`). The dashboard falls back to DB queries and then demo data if artifacts are absent.
- Pages loading but showing demo data — check the `source` field returned by each API endpoint. `"fallback"` means neither DB nor artifacts produced real data.

---

## Data Sources

| Source | What it provides | How to get it |
|---|---|---|
| [StatsBomb Open Data](https://github.com/statsbomb/open-data) | Match events, lineups, player IDs | Clone the repo; point `DATA_ROOT` at the `data/` folder |
| [Open-Meteo Archive API](https://open-meteo.com/en/docs/historical-weather-api) | Historical daily weather per stadium | Fetched automatically at runtime |
| [Transfermarkt injuries (Kaggle)](https://www.kaggle.com/datasets/irrazional/transfermarkt-injuries) | Injury records with dates | Download CSV |
| [Transfermarkt players (Kaggle)](https://www.kaggle.com/datasets/davidcariboo/player-scores) | DOB, nationality, position, TM player IDs | Download `players.csv` |

### Competitions in scope

| Competition | Seasons |
|---|---|
| La Liga | 2015/16 – 2020/21 |
| UEFA Champions League | 2018/19 |
| FIFA World Cup | 2018, 2022 |

To change scope, edit the `COMPETITIONS` set in `config/settings.py`.

---

## Repository Structure

```
.
├── config/
│   └── settings.py                    # DB connection, file paths, competition scope
├── core/
│   ├── caches.py                      # In-memory team and player caches (batch DB writes)
│   └── utils.py                       # norm_name() for accent-stripped name matching
├── extract/
│   └── statsbomb_local.py             # Read StatsBomb JSON files from disk
├── load/
│   └── postgres.py                    # DB write helpers (upsert_match, insert_stats, etc.)
├── transform/
│   ├── features.py                    # Vectorised per-player stat aggregation, xa extraction
│   └── schema.py                      # StatsBomb nested-dict field accessors
├── pipelines/
│   ├── ingest_statsbomb.py            # Parallel match ingestion (ProcessPoolExecutor)
│   ├── ingest_weather.py              # Concurrent weather fetch (ThreadPoolExecutor + retry)
│   ├── ingest_injuries.py             # Transfermarkt player and injury matching
│   ├── compute_labels.py              # Workload features and injury risk labels
│   └── ingest_pass_network.py         # Backfill tool for pass edges only
├── models/
│   ├── model1_player_clustering.py    # KMeans / DBSCAN / GMM player archetypes
│   ├── model2_team_cohesion.py        # Pass-network graph metrics + regression
│   ├── model3_injury_risk.py          # XGBoost / RF binary classifier
│   ├── model4_environment.py          # Weather + venue regression (GBR)
│   └── model5_win_probability.py      # Win / draw / loss multiclass classifier
├── utils/
│   ├── aggregate.py                   # Match-level and batch aggregation utilities
│   ├── validate_data.py               # StatsBomb event validation helpers
│   └── validate_schema.py             # Live DB schema vs expected DDL checker
├── front-end/
│   ├── index.html                     # App shell (sidebar, topbar, page containers)
│   ├── css/
│   │   ├── variables.css              # Design tokens: colors, fonts, spacing
│   │   ├── reset.css                  # Browser resets, utility classes, animations
│   │   ├── layout.css                 # Sidebar, topbar, content area, grid helpers
│   │   ├── components.css             # Cards, badges, tables, progress bars, tooltips
│   │   └── pages.css                  # Page-specific styles (dashboard, player, etc.)
│   ├── js/
│   │   ├── charts.js                  # Chart.js wrappers for all five chart types
│   │   ├── passNetwork.js             # SVG force-directed pass network graph
│   │   ├── navigation.js              # Page switching with onNavigate callback hook
│   │   ├── api.js                     # Fetch wrappers for all API endpoints
│   │   ├── main.js                    # Bootstrap, render functions, error display
│   │   └── pageLoader.js              # Async HTML injection with script re-execution
│   └── pages/
│       ├── dashboard.html             # KPI grid, performance trend, squad status
│       ├── player.html                # Player profile, radar chart, cluster grid
│       ├── cohesion.html              # Pass network visualization, centrality cards
│       ├── injury.html                # Risk table, SHAP factors, injury history chart
│       ├── env.html                   # Temperature scatter, weather condition summary
│       ├── winprob.html               # Probability banner, timeline chart
│       └── debug.html                 # System diagnostics (DB, artifacts, console log)
├── api_server.py                      # FastAPI backend — all dashboard API endpoints
├── schema.sql                         # Full DDL (run via init_db.py)
├── init_db.py                         # One-time schema creation and verification
├── main.py                            # Pipeline orchestrator
└── requirements.txt
```

---

## Database Schema

ID naming convention used throughout:

| Prefix | Meaning |
|---|---|
| `*_id` | Internal surrogate primary key (SERIAL, generated by this DB) |
| `sb_*_id` | StatsBomb source identifier |
| `tm_*_id` | Transfermarkt source identifier |

### Tables

**`teams`** — one row per team  
**`players`** — one row per player; `sb_player_id` and `tm_player_id` are nullable and filled in as data is matched  
**`stadiums`** — one row per stadium; latitude/longitude backfilled by the weather pipeline  
**`matches`** — one row per match; references `teams` for home/away and `stadiums` for coordinates  
**`weather`** — one row per match; joined on `match_id`  
**`injuries`** — one row per injury record from Transfermarkt  
**`player_match_stats`** — one row per player × match; the central fact table used by all models  
**`player_match_features`** — one row per player × match; computed ML columns (workload, injury label) kept separate from raw stats  
**`pass_network_edges`** — aggregated passer → receiver counts per match and team  
**`match_minute_snapshots`** — cumulative in-game stats per team per minute; used by the Model 5 in-game sub-model  

---

## Pipeline Stages

### Stage 1 — StatsBomb ingestion (`ingest_statsbomb.py`)

Reads every in-scope match JSON in parallel using `ProcessPoolExecutor`. Workers compute all player stats, pass network edges, starting positions, and minute-by-minute snapshots from raw events. The main process batches DB writes, committing every 50 matches per competition.

Produces rows in: `teams`, `players`, `stadiums`, `matches`, `player_match_stats`, `pass_network_edges`, `match_minute_snapshots`

### Stage 2 — Weather ingestion (`ingest_weather.py`)

Fetches one Open-Meteo API call per match using `ThreadPoolExecutor`. Retries up to 4 times with exponential backoff on rate limits (HTTP 429) or transient errors. Derives a `weather_condition` label (`clear`, `rain`, `heavy_rain`, `windy`, `cold`, `hot`) from numeric fields. Backfills stadium coordinates into the `stadiums` table and `weather_id` into `player_match_stats`.

Produces rows in: `weather`; updates `player_match_stats.weather_id`, `stadiums.stadium_lat/lng`

### Stage 3 — Injuries ingestion (`ingest_injuries.py`)

Two-pass process:

**Pass 1** matches Transfermarkt players to StatsBomb players and backfills `tm_player_id`, `date_of_birth`, `nationality`, and `position` onto the `players` table. Matching priority:

1. TM ID already linked in DB (reruns)
2. Exact `norm_name` match (accent-stripped, lowercase)
3. Token-subset match (handles middle names)
4. Blocking + RapidFuzz fuzzy match (handles transliteration variants)

**Pass 2** inserts injury rows using the `tm_player_id → player_id` map built in Pass 1.

Produces rows in: `injuries`; updates `players`

### Stage 4 — Compute labels (`compute_labels.py`)

Three SQL `UPDATE` passes over `player_match_features`:

1. **`matches_last_30_days` / `minutes_last_30_days`** — counts prior matches and minutes in the 30-day window before each match date, per player
2. **`days_since_last_injury`** — days since the player's most recent `return_date` before each match
3. **`is_injured_next_30d`** — set to `TRUE` where an injury record falls within 30 days after the match date (Model 3 target label)

---

## ML Models

| # | Model | Type | Target |
|---|---|---|---|
| 1 | Player Efficiency and Style Profiling | Clustering (KMeans / DBSCAN / GMM) | Player tactical archetype |
| 2 | Team Cohesion Analysis | Graph metrics + GBR regression | Goals scored |
| 3 | Injury Risk Prediction | Binary classification (XGBoost / RF / LR) | `is_injured_next_30d` |
| 4 | Environmental Impact Analysis | GBR regression (per target) | xG, pass accuracy, pressures |
| 5 | Win Probability Modeling | Multiclass classification (GBC / RF / LR) | Win / draw / loss |

Trained artifacts are saved to `artifacts/model{N}/` as `.pkl` and `.parquet` files.

---

## Dashboard

The FastAPI server (`api_server.py`) exposes a browser dashboard at `http://localhost:8000`. Each page corresponds to one ML model.

### Data source priority

Every endpoint tries sources in this order and falls back automatically:

1. **DB + ML model** — live database query fed into the trained artifact for inference
2. **DB only** — database query with a heuristic score when no artifact is loaded
3. **Artifact only** — parquet features file used directly when the DB is unreachable
4. **Demo fallback** — hardcoded plausible data so the UI is never blank

The data source badge in the topbar and the `source` field in every API response indicate which path was taken.

### API endpoints

| Endpoint | Description |
|---|---|
| `GET /api/options/teams` | List of teams for the selector dropdown |
| `GET /api/dashboard?team_id=N` | KPI summary and performance trend |
| `GET /api/player-efficiency?team_id=N` | Player stats, archetypes, radar chart data |
| `GET /api/team-cohesion?team_id=N` | Pass network edges and graph metrics |
| `GET /api/injury-risk?team_id=N` | Per-player injury risk scores |
| `GET /api/environment-impact?team_id=N` | Weather vs performance correlation |
| `GET /api/win-probability?team_id=N` | Pre-match and in-game win probability |
| `GET /api/health` | DB status, table counts, artifact state, last errors |
| `GET /api/debug/artifacts` | Detailed type and shape of every loaded artifact |
| `GET /api/debug/db` | PostgreSQL and psycopg2 version |

### Frontend architecture

The frontend is a vanilla JS single-page application with no build step required.

- **`pageLoader.js`** fetches each page's HTML template and injects it into the shell. It re-executes `<script>` tags after injection (browsers do not run scripts added via `innerHTML` — this is a spec requirement that the loader works around by cloning and re-appending each script element).
- **`navigation.js`** switches visible pages and fires an `onNavigate` callback so `main.js` can lazy-render pages on first visit, avoiding Chart.js sizing bugs on off-screen canvases.
- **`main.js`** awaits `pagesLoadedPromise` before initialising navigation or fetching data, eliminating a race condition where users could click nav items before page HTML existed in the DOM.
- All chart initialisations are wrapped in `requestAnimationFrame` so Canvas elements are measured at their rendered dimensions.
- API errors are surfaced as visible red banners in the UI rather than silently swallowed.

### Design system

The dashboard uses a "Precision Analytics" theme — dark industrial palette with a technical/data-forward aesthetic.

| Role | Font | Rationale |
|---|---|---|
| Headings, page title | Syne | Geometric, distinctive, reads as a data product |
| KPI values, numbers, labels | IBM Plex Mono | Monospaced digits don't shift width as values change |
| Body, UI copy | DM Sans | Clean, readable at small sizes |

CSS is split across five files loaded in dependency order: `variables.css` → `reset.css` → `layout.css` → `components.css` → `pages.css`. All colors, spacing, and typography are defined as CSS custom properties in `variables.css`. No preprocessor or build tool is needed.

---

## Notes

- Run `init_db.py` again if you change `schema.sql`. It uses `CREATE TABLE IF NOT EXISTS` and `CREATE INDEX IF NOT EXISTS` so it is safe to re-run, but column renames require manual `ALTER TABLE` migrations. Use `utils/validate_schema.py` to check whether the live DB matches the expected DDL.
- `ingest_pass_network.py` is a backfill-only tool. Pass edges are extracted during Stage 1 as part of the same event read. Only run this script if you need to populate edges for matches ingested before this was added.
- Weather is fetched for all matches that have no weather row. Re-running `ingest_weather.py` is safe and only fetches what is missing.
- `rapidfuzz` is required for fuzzy name matching in Stage 3. Without it, matching falls back to exact and token-subset only, which reduces injury linkage significantly.
- The `player_match_features` table must be populated by `compute_labels.py` before training Model 3 or using the injury risk endpoint with real data. If it is empty, the API falls back to a simpler heuristic score derived from `minutes_played` alone.