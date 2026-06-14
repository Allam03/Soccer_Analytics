/* ============================================================
   main.js  — v2.3.0

   Changes vs v2.2.0
   -----------------
   RACE CONDITION FIX (squad status cards)
   - renderSquadStatusCards() is no longer called inside renderDashboard().
     Previously, dashboard data could resolve before injury data, leaving
     AppState.injury null when renderDashboard ran, resulting in a silently
     empty squad grid with only a debug log.
   - Squad cards now render in refreshAllData() only after both dashboard
     AND injury data have resolved, regardless of which arrived first.
     renderDashboard() still renders everything else (KPIs, trend chart)
     immediately when its data is available.
   ============================================================ */

const AppState = {
  teams:     [],
  dashboard: null,
  player:    null,
  cohesion:  null,
  xg:        null,
  winprob:   null,
};

const _pendingRender = new Set();

// ---------------------------------------------------------------------------
// Bootstrap
// ---------------------------------------------------------------------------

document.addEventListener("DOMContentLoaded", async () => {
  debug("DOMContentLoaded fired");

  if (window.pagesLoadedPromise) {
    debug("Waiting for pagesLoadedPromise...");
    await window.pagesLoadedPromise;
    debug("Pages loaded, DOM ready");
  }

  Navigation.init();
  Navigation.onNavigate = onPageActivated;
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

  ["dashboard", "player", "cohesion", "xg", "winprob"].forEach(
    (p) => _pendingRender.add(p)
  );

  // Wrap each call in Promise.resolve().then(fn) so a *synchronous* throw
  // (e.g. a stale cached api.js missing a method) becomes an isolated
  // rejection instead of aborting the whole refresh and blanking every panel.
  const results = await Promise.allSettled([
    () => ApiService.loadDashboard(),
    () => ApiService.loadPlayer(),
    () => ApiService.loadCohesion(),
    () => ApiService.loadXG(),
    () => ApiService.loadWinProb(),
    () => ApiService.loadShotMap(),
    () => ApiService.loadLeagueXG(),
    () => ApiService.loadMatches(),
  ].map((fn) => Promise.resolve().then(fn)));

  const [dashRes, playerRes, cohesionRes, xgRes, winprobRes,
         shotmapRes, leagueRes, matchesRes] = results;

  const labels = ["dashboard", "player", "cohesion", "xg", "winprob",
                  "shotmap", "leaguexg", "matches"];
  results.forEach((r, i) => {
    if (r.status === "rejected") {
      console.error(`[refreshAllData] ${labels[i]} fetch failed:`, r.reason);
    } else {
      debug(`[refreshAllData] ${labels[i]} ok, source=${r.value?.source}`);
    }
  });

  AppState.dashboard = dashRes.status     === "fulfilled" ? dashRes.value     : null;
  AppState.player    = playerRes.status   === "fulfilled" ? playerRes.value   : null;
  AppState.cohesion  = cohesionRes.status === "fulfilled" ? cohesionRes.value : null;
  AppState.xg        = xgRes.status       === "fulfilled" ? xgRes.value       : null;
  AppState.winprob   = winprobRes.status  === "fulfilled" ? winprobRes.value  : null;
  AppState.shotmap   = shotmapRes.status  === "fulfilled" ? shotmapRes.value  : null;
  AppState.leaguexg  = leagueRes.status   === "fulfilled" ? leagueRes.value   : null;
  AppState.matches   = matchesRes.status  === "fulfilled" ? matchesRes.value  : null;

  const activePage = document.querySelector(".page.active")?.id?.replace("page-", "") || "dashboard";
  debug("Active page is:", activePage);
  renderPage(activePage);

  // Finishing-leader cards on the dashboard depend on the xG data, which may
  // settle after the dashboard payload. Render them here once both are in.
  if (activePage === "dashboard") {
    renderFinishingLeaders();
  }

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
// Navigation hook
// ---------------------------------------------------------------------------

function onPageActivated(pageId) {
  debug("onPageActivated:", pageId);
  if (_pendingRender.has(pageId)) {
    renderPage(pageId);
    // Finishing-leader cards live on the dashboard and need the xG payload.
    // Render them here on a lazy first visit too.
    if (pageId === "dashboard") {
      renderFinishingLeaders();
    }
  }
}

function renderPage(pageId) {
  _pendingRender.delete(pageId);
  switch (pageId) {
    case "dashboard": _safeRender(renderDashboard, "dashboard"); break;
    case "player":    _safeRender(renderPlayer,    "player");    break;
    case "cohesion":  _safeRender(renderCohesion,  "cohesion");  break;
    case "xg":        _safeRender(renderXG,        "xg");        break;
    case "winprob":   _safeRender(renderWinProb,   "winprob");   break;
    default: debug("Unknown pageId:", pageId);
  }
}

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
  if (values[2]) {
    const d = Number(kpi.finishing_diff || 0);
    values[2].textContent = `${d >= 0 ? "+" : ""}${d.toFixed(1)}`;
  }
  if (values[3]) values[3].textContent = `${kpi.next_match_win_pct.toFixed(1)}%`;

  requestAnimationFrame(() => {
    Charts.initPerfTrend(AppState.dashboard.performance_trend);
  });

  // Finishing-leader cards are rendered in refreshAllData()/onPageActivated()
  // once the xG payload has resolved (it may arrive after the dashboard data).

  renderLeagueXG();

  animateCards();
  animateProgressBars();
}

// ---------------------------------------------------------------------------
// renderLeagueXG  (dashboard) — season table, points vs expected points
// ---------------------------------------------------------------------------

function renderLeagueXG() {
  const data  = AppState.leaguexg;
  const tbody = document.querySelector("#leagueXgTable tbody");
  if (!tbody) return;
  const teams = data?.teams || [];

  const titleEl = document.getElementById("leagueXgTitle");
  if (titleEl && data?.season) {
    titleEl.textContent = `Season xG Performance ${data.season} — points vs expected points`;
  }

  if (!teams.length) {
    tbody.innerHTML = '<tr><td colspan="11" class="text-muted text-center">No xG data</td></tr>';
    return;
  }

  const selected = getSelectedTeamName();
  tbody.innerHTML = teams.slice(0, 20).map((t, i) => {
    const d = Number(t.points_diff || 0);
    const color = d >= 0 ? "var(--accent)" : "var(--red)";
    const isSel = t.team_name === selected;
    return `
      <tr${isSel ? ' style="background:rgba(56,189,131,0.07)"' : ""}>
        <td class="text-muted">${i + 1}</td>
        <td class="fw-700">${t.team_name}</td>
        <td>${t.played}</td>
        <td>${t.goals_for}</td>
        <td class="text-muted">${t.xg_for}</td>
        <td>${t.goals_against}</td>
        <td class="text-muted">${t.xg_against}</td>
        <td>${t.xg_diff > 0 ? "+" : ""}${t.xg_diff}</td>
        <td class="fw-700">${t.points}</td>
        <td class="text-muted">${t.xpoints}</td>
        <td style="color:${color};font-weight:600">${d >= 0 ? "+" : ""}${d.toFixed(1)}</td>
      </tr>`;
  }).join("");
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
      const t = p.player_type || "Unclassified";
      if (!groups[t]) groups[t] = [];
      groups[t].push(p.player_name);
    }

    // Colour by the broad role implied by the data-driven archetype name.
    // Keyed on substrings so it survives label changes from Model 1.
    const ROLE_COLORS = [
      { match: /defender|defensive/i, c: { border: "var(--cyan)",   badge: "rgba(34,211,238,0.1)", text: "var(--cyan)"   } },
      { match: /midfield|winning/i,   c: { border: "var(--amber)",  badge: "var(--amber-dim)",     text: "var(--amber)"  } },
      { match: /winger|playmaker/i,   c: { border: "var(--accent)", badge: "var(--accent-dim)",    text: "var(--accent)" } },
      { match: /goalscorer|forward/i, c: { border: "var(--red)",    badge: "var(--red-dim)",       text: "var(--red)"    } },
    ];
    const colorFor = (type) =>
      (ROLE_COLORS.find((r) => r.match.test(type)) || ROLE_COLORS[2]).c;

    clusterGrid.innerHTML = Object.entries(groups).slice(0, 4).map(([type, names]) => {
      const c = colorFor(type);
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

  requestAnimationFrame(() => {
    PassNetwork.init(
      data.edges && data.edges.length ? data.edges : null,
      data.nodes && data.nodes.length ? data.nodes : null,
    );
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
// renderXG  (Shot Quality — from-scratch Expected Goals)
// ---------------------------------------------------------------------------

function renderXG() {
  const data = AppState.xg;
  if (!data) {
    showPageError("xg", "xG data unavailable");
    return;
  }
  debug("renderXG(), source=", data.source, "players=", data.players?.length);

  const kpi     = data.kpi || {};
  const players = data.players || [];

  const values = document.querySelectorAll("#page-xg .kpi-value");
  if (values[0]) values[0].textContent = Number(kpi.team_xg || 0).toFixed(1);
  if (values[1]) values[1].textContent = String(kpi.team_goals || 0);
  if (values[2]) {
    const d = Number(kpi.xg_diff || 0);
    values[2].textContent = `${d >= 0 ? "+" : ""}${d.toFixed(1)}`;
  }
  if (values[3]) values[3].textContent = String(kpi.shots || 0);

  requestAnimationFrame(() => {
    Charts.initXGChart(players.slice(0, 10));
    Charts.initShotMap(AppState.shotmap?.shots || []);
  });

  const tbody = document.querySelector("#page-xg table tbody");
  if (tbody) {
    if (!players.length) {
      tbody.innerHTML = '<tr><td colspan="6" class="text-muted text-center">No shot data</td></tr>';
    } else {
      tbody.innerHTML = players.map((p) => {
        const d = Number(p.xg_diff || 0);
        const color = d >= 0 ? "var(--accent)" : "var(--red)";
        return `
          <tr>
            <td class="fw-700">${p.player_name}</td>
            <td>${p.position || "-"}</td>
            <td>${p.shots || 0}</td>
            <td>${p.goals || 0}</td>
            <td>${Number(p.xg || 0).toFixed(2)}</td>
            <td style="color:${color};font-weight:600">${d >= 0 ? "+" : ""}${d.toFixed(2)}</td>
          </tr>`;
      }).join("");
    }
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

  _initMatchSelector();
}

// ---------------------------------------------------------------------------
// Match xG timeline (win-probability page) — selector + cumulative-xG chart
// ---------------------------------------------------------------------------

function _initMatchSelector() {
  const sel = document.getElementById("matchSelector");
  if (!sel) return;
  const matches = AppState.matches?.matches || [];

  if (!matches.length) {
    sel.innerHTML = '<option>No matches available</option>';
    return;
  }

  sel.innerHTML = matches
    .map((m) => `<option value="${m.match_id}">${m.label}</option>`)
    .join("");

  sel.onchange = () => _loadMatchTimeline(Number(sel.value));
  _loadMatchTimeline(Number(matches[0].match_id));
}

async function _loadMatchTimeline(matchId) {
  if (!matchId) return;
  try {
    const tl = await ApiService.loadMatchTimeline(matchId);
    requestAnimationFrame(() => Charts.initMatchTimeline(tl));
    const summary = document.getElementById("matchXgSummary");
    if (summary && tl) {
      summary.innerHTML =
        `<strong>${tl.home_name} ${tl.home_score}–${tl.away_score} ${tl.away_name}</strong> · ` +
        `xG ${Number(tl.home_xg).toFixed(2)}–${Number(tl.away_xg).toFixed(2)}. ` +
        `Diamonds mark actual goals; the steeper line created chances faster.`;
    }
  } catch (err) {
    console.error("[_loadMatchTimeline]", err);
  }
}

// ---------------------------------------------------------------------------
// Finishing leaders (dashboard) — Goals vs xG from the xG model
// ---------------------------------------------------------------------------

function renderFinishingLeaders() {
  const grid    = document.querySelector("#page-dashboard .squad-grid");
  const players = AppState.xg?.players;
  if (!grid) {
    debug("renderFinishingLeaders: .squad-grid not found");
    return;
  }
  if (!players?.length) {
    debug("renderFinishingLeaders: no xG data yet");
    return;
  }

  // Highest over-performers (Goals - xG) first.
  const top = [...players]
    .sort((a, b) => Number(b.xg_diff) - Number(a.xg_diff))
    .slice(0, 10);

  grid.innerHTML = top.map((p) => {
    const d = Number(p.xg_diff || 0);
    const cls = d >= 0 ? "risk-ok" : "risk-high";
    const badgeClass = d >= 0 ? "badge-low" : "badge-high";
    return `
      <div class="squad-card ${cls}">
        <div class="squad-name">${p.player_name}</div>
        <div class="squad-pos">${p.goals || 0}G / ${Number(p.xg || 0).toFixed(1)} xG</div>
        <span class="badge ${badgeClass}">${d >= 0 ? "+" : ""}${d.toFixed(1)}</span>
      </div>
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
    mixed:                 { text: "Data: Mixed",        color: "#22d3ee" },
    fallback:              { text: "Data: Demo",         color: "#f59e0b" },
    "fallback+artifact":   { text: "Data: Demo + Model", color: "#f59e0b" },
    error:                 { text: "Data: Error",        color: "#ef4444" },
    unknown:               { text: "Data: Unknown",      color: "#94a3b8" },
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

function debug(...args) {
  if (localStorage.getItem("debug") === "1") {
    console.debug("[soccer-analytics]", ...args);
  }
}

window.addEventListener("resize", debounce(() => {
  if (document.getElementById("page-cohesion")?.classList.contains("active")) {
    PassNetwork.init(AppState.cohesion?.edges || null, AppState.cohesion?.nodes || null);
  }
}, 300));
