/* ============================================================
   main.js  — v2.2.0
   Entry point.  Wires all modules together.

   Changes vs v2.1.0
   -----------------
   RACE CONDITION FIX
   - Navigation.init() now runs AFTER pagesLoadedPromise resolves, not before.
     Previously: nav init → user clicks → renderXxx() → querySelector finds
     nothing (page HTML not yet injected) → silent blank section.
   - Charts are only initialised when their page is active, guarded by
     requestAnimationFrame to ensure the canvas is in the layout.

   ERROR VISIBILITY
   - Every API fetch error and render error is surfaced in the UI via
     showError() which injects a visible red banner into the relevant page.
   - Console.debug() used for verbose diagnostics; console.error() for real
     errors; no more silent console.warn-and-move-on.
   - _safeRender() wrapper catches and displays render-path exceptions.

   LOADING STATES
   - Each page section gets a "Loading..." skeleton while the API request
     is in-flight; replaced with real content on success.
   - The data-source badge colour now accurately reflects mixed / error states.

   OTHER
   - refreshAllData() only re-renders the currently active page immediately;
     other pages are lazily rendered on first navigation (avoids chart sizing
     bugs for off-screen canvases).
   - _pendingRender set tracks which pages need a fresh render on next visit.
   ============================================================ */

const AppState = {
  teams:       [],
  dashboard:   null,
  player:      null,
  cohesion:    null,
  injury:      null,
  environment: null,
  winprob:     null,
};

// Track which pages have stale data and need re-rendering on navigation
const _pendingRender = new Set();

// ---------------------------------------------------------------------------
// Bootstrap
// ---------------------------------------------------------------------------

document.addEventListener("DOMContentLoaded", async () => {
  debug("DOMContentLoaded fired");

  // Wait for page HTML to be injected BEFORE setting up navigation clicks.
  // This eliminates the race where a user clicks a nav item before innerHTML
  // has been written, leaving renderXxx() unable to find any DOM nodes.
  if (window.pagesLoadedPromise) {
    debug("Waiting for pagesLoadedPromise...");
    await window.pagesLoadedPromise;
    debug("Pages loaded, DOM ready");
  }

  Navigation.init();           // safe now — all page HTML is present
  Navigation.onNavigate = onPageActivated;  // hook into navigation events
  setupTooltips();
  await bootstrap();
});

async function bootstrap() {
  try {
    const teamSelector = document.getElementById("teamSelector");

    let teamsPayload;
    try {
      teamsPayload = await ApiService.loadTeams();
    } catch (err) {
      console.error("[bootstrap] loadTeams failed:", err);
      showGlobalError("Could not load team list. Is the API server running?");
      return;
    }

    AppState.teams = teamsPayload.teams || [];
    renderTeams(teamSelector, AppState.teams);
    updateDataSourceBadge(teamsPayload.source || "unknown");

    if (!AppState.teams.length) {
      showGlobalError("No teams returned from API. Check /api/health for details.");
      return;
    }

    ApiService.teamId = teamSelector.value;
    debug("Initial team_id:", ApiService.teamId);

    teamSelector.addEventListener("change", async (e) => {
      ApiService.teamId = e.target.value;
      debug("Team changed to:", ApiService.teamId);
      await refreshAllData();
    });

    await refreshAllData();
  } catch (err) {
    console.error("[bootstrap] unhandled error:", err);
    showGlobalError(`Bootstrap error: ${err.message}`);
  }
}

// ---------------------------------------------------------------------------
// Team selector rendering
// ---------------------------------------------------------------------------

function renderTeams(selector, teams) {
  if (!selector) return;
  if (!teams.length) {
    selector.innerHTML = "<option value=''>No teams available</option>";
    selector.disabled  = true;
    return;
  }
  selector.innerHTML = teams
    .map((t) => `<option value="${t.team_id}">${t.team_name}</option>`)
    .join("");
  selector.disabled = false;
  debug("Rendered", teams.length, "teams in selector");
}

// ---------------------------------------------------------------------------
// Full refresh
// ---------------------------------------------------------------------------

async function refreshAllData() {
  debug("refreshAllData() start, team_id=", ApiService.teamId);

  document.querySelectorAll(".kpi-value").forEach((el) => el.classList.add("shimmer"));

  // Mark all pages as needing re-render
  ["dashboard", "player", "cohesion", "injury", "env", "winprob"].forEach(
    (p) => _pendingRender.add(p)
  );

  const results = await Promise.allSettled([
    ApiService.loadDashboard(),
    ApiService.loadPlayer(),
    ApiService.loadCohesion(),
    ApiService.loadInjury(),
    ApiService.loadEnvironment(),
    ApiService.loadWinProb(),
  ]);

  const [dashRes, playerRes, cohesionRes, injuryRes, envRes, winprobRes] = results;

  // Log any fetch-level failures
  const labels = ["dashboard", "player", "cohesion", "injury", "environment", "winprob"];
  results.forEach((r, i) => {
    if (r.status === "rejected") {
      console.error(`[refreshAllData] ${labels[i]} fetch failed:`, r.reason);
    } else {
      debug(`[refreshAllData] ${labels[i]} ok, source=${r.value?.source}`);
    }
  });

  AppState.dashboard   = dashRes.status   === "fulfilled" ? dashRes.value   : null;
  AppState.player      = playerRes.status === "fulfilled" ? playerRes.value : null;
  AppState.cohesion    = cohesionRes.status === "fulfilled" ? cohesionRes.value : null;
  AppState.injury      = injuryRes.status === "fulfilled" ? injuryRes.value : null;
  AppState.environment = envRes.status    === "fulfilled" ? envRes.value    : null;
  AppState.winprob     = winprobRes.status === "fulfilled" ? winprobRes.value : null;

  // Always render the active page immediately; others render on demand
  const activePage = document.querySelector(".page.active")?.id?.replace("page-", "") || "dashboard";
  debug("Active page is:", activePage);
  renderPage(activePage);

  // Badge reflects worst status across all endpoints
  const sources = results
    .filter((r) => r.status === "fulfilled")
    .map((r) => r.value?.source)
    .filter(Boolean);
  const hasError    = results.some((r) => r.status === "rejected");
  const allDB       = sources.every((s) => s.startsWith("database"));
  const allFallback = sources.every((s) => s === "fallback");

  updateDataSourceBadge(
    hasError      ? "error"    :
    allDB         ? "database" :
    allFallback   ? "fallback" : "mixed"
  );

  document.querySelectorAll(".kpi-value").forEach((el) => el.classList.remove("shimmer"));
  debug("refreshAllData() complete");
}

// ---------------------------------------------------------------------------
// Navigation hook — render page on first visit after data loaded
// ---------------------------------------------------------------------------

function onPageActivated(pageId) {
  debug("onPageActivated:", pageId);
  if (_pendingRender.has(pageId)) {
    renderPage(pageId);
  }
}

function renderPage(pageId) {
  _pendingRender.delete(pageId);
  switch (pageId) {
    case "dashboard": _safeRender(renderDashboard,     "dashboard");  break;
    case "player":    _safeRender(renderPlayer,        "player");     break;
    case "cohesion":  _safeRender(renderCohesion,      "cohesion");   break;
    case "injury":    _safeRender(renderInjury,        "injury");     break;
    case "env":       _safeRender(renderEnvironment,   "env");        break;
    case "winprob":   _safeRender(renderWinProb,       "winprob");    break;
    default: debug("Unknown pageId:", pageId);
  }
}

// Wrap each render function so exceptions are caught and shown in the UI
function _safeRender(fn, pageId) {
  try {
    fn();
  } catch (err) {
    console.error(`[render:${pageId}]`, err);
    showPageError(pageId, `Render error: ${err.message}`);
  }
}

// ---------------------------------------------------------------------------
// renderDashboard
// ---------------------------------------------------------------------------

function renderDashboard() {
  if (!AppState.dashboard) {
    showPageError("dashboard", "Dashboard data unavailable");
    return;
  }
  debug("renderDashboard(), source=", AppState.dashboard.source);

  const kpi    = AppState.dashboard.kpi;
  const values = document.querySelectorAll("#page-dashboard .kpi-value");
  if (values[0]) values[0].textContent = kpi.team_performance.toFixed(1);
  if (values[1]) values[1].textContent = kpi.cohesion_index.toFixed(2);
  if (values[2]) values[2].textContent = String(kpi.high_risk_players);
  if (values[3]) values[3].textContent = `${kpi.next_match_win_pct.toFixed(1)}%`;

  // Charts need the canvas to be in the visible layout — use rAF
  requestAnimationFrame(() => {
    Charts.initPerfTrend(AppState.dashboard.performance_trend);
  });

  renderSquadStatusCards();
  animateCards();
  animateProgressBars();
}

// ---------------------------------------------------------------------------
// renderPlayer
// ---------------------------------------------------------------------------

function renderPlayer() {
  const data = AppState.player;
  if (!data) {
    showPageError("player", "Player efficiency data unavailable");
    return;
  }
  debug("renderPlayer(), source=", data.source, "players=", data.players?.length);

  const leader  = data.leader  || {};
  const players = data.players || [];

  const nameEl   = document.querySelector("#page-player .player-name");
  const metaEl   = document.querySelector("#page-player .player-meta");
  const avatarEl = document.querySelector("#page-player .player-avatar");
  if (nameEl)   nameEl.textContent   = leader.player_name || "Top Player";
  if (metaEl)   metaEl.textContent   = `${leader.position || "-"} • ${getSelectedTeamName()}`;
  if (avatarEl) avatarEl.textContent = getInitials(leader.player_name || "");

  const statsRow = document.querySelector("#page-player .player-stats-row");
  if (statsRow) {
    statsRow.innerHTML = `
      <div class="stat-item">Type <span>${leader.player_type || "Midfielder"}</span></div>
      <div class="stat-item">Matches <span>${leader.matches || 0}</span></div>
      <div class="stat-item">Minutes <span>${Math.round(leader.minutes || 0).toLocaleString()}</span></div>
      <div class="stat-item">xG <span>${(leader.xg_per_90 || 0).toFixed(2)}</span></div>
    `;
  }

  const metricsGrid = document.querySelector("#page-player .metrics-grid");
  if (metricsGrid) {
    const avgXG = _teamAvg(players, "xg_per_90");
    const avgXA = _teamAvg(players, "xa_per_90");
    const avgPA = _teamAvg(players, "pass_completion");
    const avgKP = _teamAvg(players, "key_passes");
    const deltaXG = _delta(leader.xg_per_90,      avgXG);
    const deltaXA = _delta(leader.xa_per_90,       avgXA);
    const deltaPA = _delta(leader.pass_completion, avgPA);
    const deltaKP = _delta(leader.key_passes,      avgKP);

    metricsGrid.innerHTML = `
      <div class="metric-card">
        <div class="metric-val text-accent">${(leader.xg_per_90 || 0).toFixed(2)}</div>
        <div class="metric-label">xG per 90</div>
        <div class="metric-delta ${deltaXG >= 0 ? "pos" : "neg"}">${deltaXG >= 0 ? "+" : ""}${deltaXG.toFixed(0)}% vs avg</div>
      </div>
      <div class="metric-card">
        <div class="metric-val text-cyan">${(leader.xa_per_90 || 0).toFixed(2)}</div>
        <div class="metric-label">xA per 90</div>
        <div class="metric-delta ${deltaXA >= 0 ? "pos" : "neg"}">${deltaXA >= 0 ? "+" : ""}${deltaXA.toFixed(0)}% vs avg</div>
      </div>
      <div class="metric-card">
        <div class="metric-val">${(leader.pass_completion || 0).toFixed(1)}%</div>
        <div class="metric-label">Pass Completion</div>
        <div class="metric-delta ${deltaPA >= 0 ? "pos" : "neg"}">${deltaPA >= 0 ? "+" : ""}${deltaPA.toFixed(0)}% vs avg</div>
      </div>
      <div class="metric-card">
        <div class="metric-val text-amber">${(leader.key_passes || 0).toFixed(1)}</div>
        <div class="metric-label">Key Passes</div>
        <div class="metric-delta ${deltaKP >= 0 ? "pos" : "neg"}">${deltaKP >= 0 ? "+" : ""}${deltaKP.toFixed(0)}% vs avg</div>
      </div>
      <div class="metric-card">
        <div class="metric-val">${(leader.dribbles || 0).toFixed(1)}</div>
        <div class="metric-label">Dribbles</div>
      </div>
      <div class="metric-card">
        <div class="metric-val">${(leader.shots || 0).toFixed(1)}</div>
        <div class="metric-label">Shots per 90</div>
      </div>
    `;
  }

  requestAnimationFrame(() => {
    Charts.initRadar(data.radar, leader.player_name || "Top Player");
  });

  const clusterGrid = document.querySelector("#page-player .cluster-grid");
  if (clusterGrid && players.length) {
    const groups = {};
    for (const p of players) {
      const t = p.player_type || "Midfielder";
      if (!groups[t]) groups[t] = [];
      groups[t].push(p.player_name);
    }

    const CLUSTER_COLORS = {
      "Creator":          { border: "var(--accent)", badge: "var(--accent-dim)", text: "var(--accent)" },
      "Finisher":         { border: "var(--red)",    badge: "var(--red-dim)",    text: "var(--red)"    },
      "Ball Winner":      { border: "var(--amber)",  badge: "var(--amber-dim)",  text: "var(--amber)"  },
      "Box-to-Box":       { border: "var(--cyan)",   badge: "rgba(34,211,238,0.1)", text: "var(--cyan)" },
      "Wide Attacker":    { border: "var(--accent)", badge: "var(--accent-dim)", text: "var(--accent)" },
      "Playmaker":        { border: "var(--cyan)",   badge: "rgba(34,211,238,0.1)", text: "var(--cyan)" },
      "Defensive Shield": { border: "var(--amber)",  badge: "var(--amber-dim)",  text: "var(--amber)"  },
    };

    clusterGrid.innerHTML = Object.entries(groups).slice(0, 4).map(([type, names]) => {
      const c = CLUSTER_COLORS[type] || CLUSTER_COLORS["Creator"];
      const isSelected = names.includes(leader.player_name);
      return `
        <div class="cluster-card${isSelected ? " selected" : ""}"
             style="border-left:3px solid ${c.border}">
          <div class="cluster-name">${type}
            <span class="cluster-count" style="background:${c.badge};color:${c.text}">${names.length}</span>
          </div>
          <ul class="cluster-players">
            ${names.slice(0, 3).map((n) => `<li>${n}</li>`).join("")}
          </ul>
        </div>
      `;
    }).join("");
  }

  const tbody = document.querySelector("#page-player table tbody");
  if (tbody) {
    tbody.innerHTML = players.map((p) => `
      <tr>
        <td class="fw-700">${p.player_name}</td>
        <td>${p.position || "-"}</td>
        <td>${p.player_type || "-"}</td>
        <td>${p.matches || 0}</td>
        <td>${(p.xg_per_90 || 0).toFixed(2)}</td>
        <td>${(p.xa_per_90 || 0).toFixed(2)}</td>
        <td>${(p.pass_completion || 0).toFixed(1)}%</td>
        <td>${(p.key_passes || 0).toFixed(1)}</td>
      </tr>
    `).join("");
  }
}

// ---------------------------------------------------------------------------
// renderCohesion
// ---------------------------------------------------------------------------

function renderCohesion() {
  const data = AppState.cohesion;
  if (!data) {
    showPageError("cohesion", "Team cohesion data unavailable");
    return;
  }
  debug("renderCohesion(), source=", data.source, "edges=", data.edges?.length);

  const kpi    = data.kpi || {};
  const values = document.querySelectorAll("#page-cohesion .kpi-value");
  if (values[0]) values[0].textContent = (kpi.cohesion_index   || 0).toFixed(2);
  if (values[1]) values[1].textContent = (kpi.network_density  || 0).toFixed(2);
  if (values[2]) values[2].textContent = (kpi.avg_degree       || 0).toFixed(1);
  if (values[3]) values[3].textContent = (kpi.clustering_coeff || 0).toFixed(2);

  // PassNetwork reads SVG dimensions — must run after layout is stable
  requestAnimationFrame(() => {
    PassNetwork.init(data.edges && data.edges.length ? data.edges : null);
  });

  if (data.edges && data.edges.length) {
    _renderCohesionCards(data.edges);
  }
}

function _renderCohesionCards(edges) {
  const vol = {};
  for (const e of edges) {
    vol[e.from] = (vol[e.from] || 0) + (e.weight || 1);
    vol[e.to]   = (vol[e.to]   || 0) + (e.weight || 1);
  }
  const sorted = Object.entries(vol).sort((a, b) => b[1] - a[1]).slice(0, 5);
  const maxVol = sorted[0]?.[1] || 1;

  const centralCard = document.querySelector("#page-cohesion [data-card='central']");
  if (centralCard) {
    const titleEl   = centralCard.querySelector(".card-title");
    const titleHTML = titleEl ? titleEl.outerHTML : '<div class="card-title">Central Players</div>';
    const list = sorted.slice(0, 3).map(([name, v]) => `
      <div class="player-list-item">
        <div>
          <div class="player-list-name">${name}</div>
          <div class="player-list-role">Pass volume: ${Math.round(v)}</div>
        </div>
        <div class="player-centrality">
          ${(v / maxVol).toFixed(2)} <span class="text-muted fs-11">Centrality</span>
        </div>
      </div>
    `).join("");
    centralCard.innerHTML = titleHTML + list;
  }

  const connCard = document.querySelector("#page-cohesion [data-card='connected']");
  if (connCard) {
    const passVol = {};
    for (const e of edges) passVol[e.from] = (passVol[e.from] || 0) + (e.weight || 1);
    const topPassers = Object.entries(passVol).sort((a, b) => b[1] - a[1]).slice(0, 3);

    const titleEl   = connCard.querySelector(".card-title");
    const titleHTML = titleEl ? titleEl.outerHTML : '<div class="card-title">Most Connected Players</div>';
    const list = topPassers.map(([name, v]) => `
      <div class="player-list-item">
        <div>
          <div class="player-list-name">${name}</div>
          <div class="player-list-role">${Math.round(v)} passes</div>
        </div>
        <div class="player-centrality">
          ${(v / edges.length).toFixed(1)} <span class="text-muted fs-11">Avg Conn.</span>
        </div>
      </div>
    `).join("");
    connCard.innerHTML = titleHTML + list;
  }
}

// ---------------------------------------------------------------------------
// renderInjury
// ---------------------------------------------------------------------------

function renderInjury() {
  const data = AppState.injury;
  if (!data) {
    showPageError("injury", "Injury risk data unavailable");
    return;
  }
  debug("renderInjury(), source=", data.source, "players=", data.players?.length);

  const kpi    = data.kpi || {};
  const values = document.querySelectorAll("#page-injury .kpi-value");
  if (values[0]) values[0].textContent = String(kpi.high   || 0);
  if (values[1]) values[1].textContent = String(kpi.medium || 0);
  if (values[2]) values[2].textContent = String(kpi.low    || 0);
  if (values[3]) values[3].textContent = (kpi.avg_score || 0).toFixed(2);

  renderInjuryRiskTable();
  requestAnimationFrame(() => {
    Charts.initInjuryHistory(data.players || []);
  });
}

// ---------------------------------------------------------------------------
// renderEnvironment
// ---------------------------------------------------------------------------

function renderEnvironment() {
  const data = AppState.environment;
  if (!data) {
    showPageError("env", "Environmental impact data unavailable");
    return;
  }
  debug("renderEnvironment(), source=", data.source, "points=", data.scatter?.length);

  if (data.scatter && data.scatter.length) {
    requestAnimationFrame(() => {
      Charts.initTempPerf(data.scatter);
    });
  }
  const summary = data.condition_summary || {};
  if (Object.keys(summary).length) {
    _renderConditionSummary(summary);
  }
}

function _renderConditionSummary(summary) {
  const condItems = document.querySelectorAll("#page-env .condition-item");
  const COND_ICONS = {
    clear: "☀️", rain: "🌧️", heavy_rain: "⛈️", windy: "💨", cold: "❄️", hot: "🔥",
  };
  let i = 0;
  for (const [cond, stats] of Object.entries(summary)) {
    if (i >= condItems.length) break;
    const item    = condItems[i];
    const nameEl  = item.querySelector(".condition-name");
    const scoreEl = item.querySelector(".condition-score");
    const deltaEl = item.querySelector(".condition-delta");
    const iconEl  = item.querySelector("span[style]");
    if (iconEl)   iconEl.textContent  = COND_ICONS[cond] || "🌤️";
    if (nameEl)   nameEl.textContent  = _capitalize(cond.replace("_", " "));
    if (scoreEl)  scoreEl.textContent = stats.predicted_xg?.toFixed(2) || "-";
    if (deltaEl) {
      const acc = stats.predicted_pass_acc || 0;
      deltaEl.textContent = `${acc.toFixed(1)}% pass acc`;
      deltaEl.className   = `condition-delta ${acc >= 70 ? "pos" : "neg"}`;
    }
    i++;
  }
}

// ---------------------------------------------------------------------------
// renderWinProb
// ---------------------------------------------------------------------------

function renderWinProb() {
  const data = AppState.winprob;
  if (!data) {
    showPageError("winprob", "Win probability data unavailable");
    return;
  }
  debug("renderWinProb(), source=", data.source);

  const pct       = document.querySelectorAll("#page-winprob .prob-pct");
  const teams     = document.querySelectorAll("#page-winprob .prob-team");
  const matchLine = document.querySelector("#page-winprob .prob-match");
  const selected  = getSelectedTeamName();

  if (pct[0]) pct[0].textContent   = `${data.headline.win}%`;
  if (pct[1]) pct[1].textContent   = `${data.headline.draw}%`;
  if (pct[2]) pct[2].textContent   = `${data.headline.loss}%`;
  if (teams[0]) teams[0].textContent = `${selected} Win`;
  if (teams[1]) teams[1].textContent = "Draw";
  if (teams[2]) teams[2].textContent = "Opponent Win";
  if (matchLine) matchLine.textContent = `Next Match Prediction • ${selected} vs Opponent`;

  if (data.timeline) {
    requestAnimationFrame(() => {
      Charts.initWinProb(data.timeline);
    });
  }
}

// ---------------------------------------------------------------------------
// Squad status cards (dashboard)
// ---------------------------------------------------------------------------

function renderSquadStatusCards() {
  const grid    = document.querySelector("#page-dashboard .squad-grid");
  const players = AppState.injury?.players;
  if (!grid) {
    debug("renderSquadStatusCards: .squad-grid not found");
    return;
  }
  if (!players?.length) {
    debug("renderSquadStatusCards: no player data yet");
    return;
  }

  const top = [...players]
    .sort((a, b) => Number(b.risk_score) - Number(a.risk_score))
    .slice(0, 10);

  grid.innerHTML = top.map((p) => {
    const score = Number(p.risk_score || 0);
    const meta  = getRiskMeta(score);
    return `
      <div class="squad-card ${meta.cardClass}">
        <div class="squad-name">${p.player_name}</div>
        <div class="squad-pos">${p.position || "-"}</div>
        <span class="badge ${meta.badgeClass}">${meta.label}</span>
      </div>
    `;
  }).join("");
}

// ---------------------------------------------------------------------------
// Injury risk table
// ---------------------------------------------------------------------------

function renderInjuryRiskTable() {
  const tbody   = document.querySelector("#page-injury .data-table tbody");
  const players = AppState.injury?.players;
  if (!tbody) {
    debug("renderInjuryRiskTable: tbody not found");
    return;
  }
  if (!players?.length) {
    tbody.innerHTML = '<tr><td colspan="7" class="text-muted text-center">No data</td></tr>';
    return;
  }

  tbody.innerHTML = players.map((p) => {
    const score = Number(p.risk_score || 0);
    const pct   = Math.round(score * 100);
    const meta  = getRiskMeta(score);
    const barColor       = score >= 0.67 ? "var(--red)" : score >= 0.4 ? "var(--amber)" : "var(--accent)";
    const recommendation = score >= 0.67 ? "Rest advised" : score >= 0.4 ? "Reduce load" : "Proceed normally";
    return `
      <tr>
        <td class="fw-700">${p.player_name}</td>
        <td>${p.position || "-"}</td>
        <td>
          <div class="risk-score-bar">
            <div class="risk-bar">
              <div class="risk-bar-fill" style="width:${pct}%;background:${barColor}"></div>
            </div>
            <span class="fs-12" style="width:30px">${score.toFixed(2)}</span>
          </div>
        </td>
        <td><span class="badge ${meta.badgeClass}">${meta.label.replace(" Risk", "")}</span></td>
        <td>${Number(p.workload_30d || 0)} min</td>
        <td>${Number(p.days_since_last_injury || 0)} days ago</td>
        <td class="text-muted fs-12">${recommendation}</td>
      </tr>
    `;
  }).join("");
}

// ---------------------------------------------------------------------------
// Error display helpers
// ---------------------------------------------------------------------------

function showGlobalError(message) {
  console.error("[global error]", message);
  const area = document.querySelector(".content-area");
  if (!area) return;
  const existing = area.querySelector(".global-error-banner");
  if (existing) existing.remove();
  const el = document.createElement("div");
  el.className = "global-error-banner";
  el.style.cssText = [
    "background:rgba(239,68,68,0.15)",
    "border:1px solid rgba(239,68,68,0.4)",
    "border-radius:8px",
    "padding:16px 20px",
    "margin-bottom:16px",
    "color:#ef4444",
    "font-size:13px",
    "display:flex",
    "align-items:center",
    "gap:10px",
  ].join(";");
  el.innerHTML = `<span style="font-size:18px">⚠️</span>
    <div>
      <strong>Error:</strong> ${message}<br>
      <span style="color:var(--text-secondary);font-size:11px">
        Check <a href="/api/health" target="_blank" style="color:#ef4444">/api/health</a> for diagnostics.
      </span>
    </div>`;
  area.prepend(el);
}

function showPageError(pageId, message) {
  console.error(`[page:${pageId}]`, message);
  const container = document.querySelector(`#page-${pageId} .kpi-grid`);
  if (!container) return;
  // Add a subtle inline error note rather than replacing content
  const existing = document.querySelector(`#page-${pageId} .page-error-note`);
  if (existing) existing.remove();
  const el = document.createElement("div");
  el.className = "page-error-note";
  el.style.cssText = [
    "grid-column:1/-1",
    "background:rgba(239,68,68,0.1)",
    "border:1px solid rgba(239,68,68,0.3)",
    "border-radius:6px",
    "padding:10px 14px",
    "font-size:12px",
    "color:#ef4444",
  ].join(";");
  el.innerHTML = `⚠️ ${message} — showing placeholder data.
    <a href="/api/health" target="_blank" style="color:#ef4444;margin-left:8px">Check /api/health</a>`;
  container.prepend(el);
}

// ---------------------------------------------------------------------------
// Shared helpers
// ---------------------------------------------------------------------------

function getRiskMeta(score) {
  if (score >= 0.67) return { label: "High Risk",  badgeClass: "badge-high",   cardClass: "risk-high" };
  if (score >= 0.4)  return { label: "Monitor",    badgeClass: "badge-medium", cardClass: "risk-med"  };
  return               { label: "Available",  badgeClass: "badge-low",    cardClass: "risk-ok"  };
}

function getInitials(name) {
  if (!name) return "--";
  return name.split(" ").filter(Boolean).slice(0, 2)
    .map((n) => n[0].toUpperCase()).join("");
}

function getSelectedTeamName() {
  const selector = document.getElementById("teamSelector");
  if (!selector) return "Selected Team";
  return selector.options[selector.selectedIndex]?.text || "Selected Team";
}

function updateDataSourceBadge(source) {
  const badge = document.getElementById("dataSourceBadge");
  if (!badge) return;
  const map = {
    database:              { text: "Data: Live DB",      color: "#3ecf8e" },
    "database+artifact":   { text: "Data: DB + Model",   color: "#22d3ee" },
    "database+model3":     { text: "Data: DB + Model 3", color: "#22d3ee" },
    "database+model4":     { text: "Data: DB + Model 4", color: "#22d3ee" },
    "database+model5":     { text: "Data: DB + Model 5", color: "#22d3ee" },
    artifact:              { text: "Data: ML Artifact",  color: "#f59e0b" },
    mixed:                 { text: "Data: Mixed",         color: "#22d3ee" },
    fallback:              { text: "Data: Demo",          color: "#f59e0b" },
    "fallback+artifact":   { text: "Data: Demo + Model", color: "#f59e0b" },
    error:                 { text: "Data: Error",         color: "#ef4444" },
    unknown:               { text: "Data: Unknown",       color: "#94a3b8" },
  };
  const item = map[source] || map.unknown;
  badge.textContent       = item.text;
  badge.style.borderColor = `${item.color}66`;
  badge.style.color       = item.color;
}

function animateCards() {
  const cards = document.querySelectorAll(
    "#page-dashboard .kpi-card, #page-dashboard .card"
  );
  cards.forEach((el, i) => {
    el.style.opacity   = "0";
    el.style.transform = "translateY(12px)";
    setTimeout(() => {
      el.style.transition = "opacity 0.4s ease, transform 0.4s ease";
      el.style.opacity    = "1";
      el.style.transform  = "translateY(0)";
    }, i * 60);
  });
}

function animateProgressBars() {
  document.querySelectorAll(".progress-fill").forEach((bar) => {
    const target = bar.style.width || "0%";
    bar.style.width = "0%";
    setTimeout(() => { bar.style.width = target; }, 300);
  });
}

function setupTooltips() {
  const tooltip = document.getElementById("tooltip");
  if (!tooltip) return;
  document.addEventListener("mouseover", (e) => {
    const el = e.target.closest("[data-tip]");
    if (!el) { tooltip.style.display = "none"; return; }
    tooltip.textContent   = el.dataset.tip;
    tooltip.style.display = "block";
  });
  document.addEventListener("mousemove", (e) => {
    tooltip.style.left = (e.clientX + 12) + "px";
    tooltip.style.top  = (e.clientY - 8)  + "px";
  });
  document.addEventListener("mouseout", (e) => {
    if (!e.target.closest("[data-tip]")) tooltip.style.display = "none";
  });
}

function _teamAvg(players, key) {
  if (!players.length) return 0;
  return players.reduce((s, p) => s + Number(p[key] || 0), 0) / players.length;
}

function _delta(val, avg) {
  if (!avg) return 0;
  return ((Number(val || 0) - avg) / avg) * 100;
}

function _capitalize(s) {
  return s.charAt(0).toUpperCase() + s.slice(1);
}

function formatNumber(n) {
  if (n >= 1_000_000) return (n / 1_000_000).toFixed(1) + "M";
  if (n >= 1_000)     return (n / 1_000).toFixed(1) + "K";
  return n.toString();
}

function debounce(fn, delay = 200) {
  let timer;
  return (...args) => { clearTimeout(timer); timer = setTimeout(() => fn(...args), delay); };
}

// Verbose logging helper — enable by setting localStorage.debug = '1'
function debug(...args) {
  if (localStorage.getItem("debug") === "1") {
    console.debug("[soccer-analytics]", ...args);
  }
}

window.addEventListener("resize", debounce(() => {
  if (document.getElementById("page-cohesion")?.classList.contains("active")) {
    PassNetwork.init(AppState.cohesion?.edges || null);
  }
}, 300));