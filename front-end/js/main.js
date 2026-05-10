/* ============================================================
   main.js
   نقطة الدخول الرئيسية للتطبيق
   بيربط كل الـ modules مع بعض ويعمل init عند تحميل الصفحة
   ============================================================ */

const AppState = {
  teams: [],
  dashboard: null,
  player: null,
  cohesion: null,
  injury: null,
  environment: null,
  winprob: null,
};

document.addEventListener('DOMContentLoaded', async () => {
  Navigation.init();
  setupTooltips();
  await bootstrap();
});

/* ============================================================
   Entrance Animation
   كل كرت بيظهر بعد تأخير صغير بشكل تتابعي (stagger)
   ============================================================ */
function animateCards() {
  const cards = document.querySelectorAll('#page-dashboard .kpi-card, #page-dashboard .card');
  cards.forEach((el, i) => {
    el.style.opacity   = '0';
    el.style.transform = 'translateY(12px)';
    setTimeout(() => {
      el.style.transition = 'opacity 0.4s ease, transform 0.4s ease';
      el.style.opacity    = '1';
      el.style.transform  = 'translateY(0)';
    }, i * 60);
  });
}

/* ============================================================
   Progress Bars Animation
   بتاخد كل .progress-fill وبتحرّكها من 0 للـ width المحدد
   ============================================================ */
function animateProgressBars() {
  // نحفظ الـ target width وبعدين نبدأ من 0
  document.querySelectorAll('.progress-fill').forEach(bar => {
    const target = bar.style.width || '0%';
    bar.style.width = '0%';

    // بعد 300ms نبدأ الـ animation
    setTimeout(() => {
      bar.style.width = target;
    }, 300);
  });
}

/* ============================================================
   Tooltips
   بيعرض tooltip مع أي element معاه data-tip attribute
   ============================================================ */
function setupTooltips() {
  const tooltip = document.getElementById('tooltip');
  if (!tooltip) return;

  document.addEventListener('mouseover', e => {
    const el = e.target.closest('[data-tip]');
    if (!el) { tooltip.style.display = 'none'; return; }

    tooltip.textContent = el.dataset.tip;
    tooltip.style.display = 'block';
  });

  document.addEventListener('mousemove', e => {
    tooltip.style.left = (e.clientX + 12) + 'px';
    tooltip.style.top  = (e.clientY - 8)  + 'px';
  });

  document.addEventListener('mouseout', e => {
    if (!e.target.closest('[data-tip]')) {
      tooltip.style.display = 'none';
    }
  });
}

/* ============================================================
   Team Change Handler
   لما المستخدم يغير الفريق — بنعمل simulate لتحديث البيانات
   في الـ production هنا بنعمل API call فعلي
   ============================================================ */
async function bootstrap() {
  try {
    if (window.pagesLoadedPromise) {
      await window.pagesLoadedPromise;
    }
    const teamSelector = document.getElementById('teamSelector');
    const teamsPayload = await ApiService.loadTeams();
    AppState.teams = teamsPayload.teams || [];
    renderTeams(teamSelector, AppState.teams);
    updateDataSourceBadge(teamsPayload.source || 'unknown');
    if (!AppState.teams.length) return;
    ApiService.teamId = teamSelector.value;
    teamSelector.addEventListener('change', async (e) => {
      ApiService.teamId = e.target.value;
      await refreshAllData();
    });
    await refreshAllData();
  } catch (err) {
    console.error(err);
  }
}

function renderTeams(selector, teams) {
  if (!selector) return;
  selector.innerHTML = teams.map((t) => `<option value="${t.team_id}">${t.team_name}</option>`).join('');
  if (!teams.length) {
    selector.innerHTML = '<option value="">No teams available</option>';
    selector.disabled = true;
    return;
  }
  selector.disabled = false;
}

async function refreshAllData() {
  try {
    document.querySelectorAll('.kpi-value').forEach((el) => el.classList.add('shimmer'));
    const [dashboard, player, cohesion, injury, environment, winprob] = await Promise.all([
      ApiService.loadDashboard(),
      ApiService.loadPlayer(),
      ApiService.loadCohesion(),
      ApiService.loadInjury(),
      ApiService.loadEnvironment(),
      ApiService.loadWinProb(),
    ]);
    AppState.dashboard = dashboard;
    AppState.player = player;
    AppState.cohesion = cohesion;
    AppState.injury = injury;
    AppState.environment = environment;
    AppState.winprob = winprob;
    renderDashboard();
    renderPlayer();
    renderCohesion();
    renderInjury();
    renderEnvironment();
    renderWinProb();

    const sources = [dashboard, player, cohesion, injury, environment, winprob]
      .map((p) => p?.source)
      .filter(Boolean);
    const allFallback = sources.length && sources.every((s) => s === 'fallback');
    const allDatabase = sources.length && sources.every((s) => s === 'database');
    updateDataSourceBadge(allDatabase ? 'database' : allFallback ? 'fallback' : 'mixed');
  } catch (err) {
    console.error('[refreshAllData]', err);
    updateDataSourceBadge('error');
  } finally {
    document.querySelectorAll('.kpi-value').forEach((el) => el.classList.remove('shimmer'));
  }
}

function renderDashboard() {
  if (!AppState.dashboard) return;
  const kpi = AppState.dashboard.kpi;
  const values = document.querySelectorAll('#page-dashboard .kpi-value');
  if (values[0]) values[0].textContent = kpi.team_performance.toFixed(1);
  if (values[1]) values[1].textContent = kpi.cohesion_index.toFixed(2);
  if (values[2]) values[2].textContent = String(kpi.high_risk_players);
  if (values[3]) values[3].textContent = `${kpi.next_match_win_pct.toFixed(1)}%`;
  Charts.initPerfTrend(AppState.dashboard.performance_trend);
  renderSquadStatusCards();
  animateCards();
  animateProgressBars();
}

function renderPlayer() {
  if (!AppState.player) return;
  const leader = AppState.player.leader;
  const nameEl = document.querySelector('#page-player .player-name');
  const metaEl = document.querySelector('#page-player .player-meta');
  const avatarEl = document.querySelector('#page-player .player-avatar');
  if (nameEl) nameEl.textContent = leader.player_name;
  if (metaEl) metaEl.textContent = `${leader.position || '-'} • ${getSelectedTeamName()}`;
  if (avatarEl) avatarEl.textContent = getInitials(leader.player_name);
  Charts.initRadar(AppState.player.radar, leader.player_name);
}

function renderCohesion() {
  if (!AppState.cohesion) return;
  const values = document.querySelectorAll('#page-cohesion .kpi-value');
  const kpi = AppState.cohesion.kpi;
  if (values[0]) values[0].textContent = kpi.cohesion_index.toFixed(2);
  if (values[1]) values[1].textContent = kpi.network_density.toFixed(2);
  if (values[2]) values[2].textContent = kpi.avg_degree.toFixed(1);
  if (values[3]) values[3].textContent = kpi.clustering_coeff.toFixed(2);
  PassNetwork.init(AppState.cohesion.edges);
}

function renderInjury() {
  if (!AppState.injury) return;
  const values = document.querySelectorAll('#page-injury .kpi-value');
  const kpi = AppState.injury.kpi;
  if (values[0]) values[0].textContent = String(kpi.high);
  if (values[1]) values[1].textContent = String(kpi.medium);
  if (values[2]) values[2].textContent = String(kpi.low);
  if (values[3]) values[3].textContent = kpi.avg_score.toFixed(2);
  renderInjuryRiskTable();
  Charts.initInjuryHistory(AppState.injury.players);
}

function renderEnvironment() {
  if (!AppState.environment) return;
  Charts.initTempPerf(AppState.environment.scatter);
}

function renderWinProb() {
  if (!AppState.winprob) return;
  const pct = document.querySelectorAll('#page-winprob .prob-pct');
  const teams = document.querySelectorAll('#page-winprob .prob-team');
  const matchLine = document.querySelector('#page-winprob .prob-match');
  const selected = getSelectedTeamName();
  if (pct[0]) pct[0].textContent = `${AppState.winprob.headline.win}%`;
  if (pct[1]) pct[1].textContent = `${AppState.winprob.headline.draw}%`;
  if (pct[2]) pct[2].textContent = `${AppState.winprob.headline.loss}%`;
  if (teams[0]) teams[0].textContent = `${selected} Win`;
  if (teams[1]) teams[1].textContent = 'Draw';
  if (teams[2]) teams[2].textContent = 'Opponent Win';
  if (matchLine) matchLine.textContent = `Next Match Prediction • ${selected} vs Opponent`;
  Charts.initWinProb(AppState.winprob.timeline);
}

function getRiskMeta(score) {
  if (score >= 0.67) {
    return { label: 'High Risk', badgeClass: 'badge-high', cardClass: 'risk-high' };
  }
  if (score >= 0.4) {
    return { label: 'Monitor', badgeClass: 'badge-medium', cardClass: 'risk-med' };
  }
  return { label: 'Available', badgeClass: 'badge-low', cardClass: 'risk-ok' };
}

function renderSquadStatusCards() {
  const grid = document.querySelector('#page-dashboard .squad-grid');
  const players = AppState.injury?.players;
  if (!grid || !players || !players.length) return;

  const top = [...players].sort((a, b) => Number(b.risk_score) - Number(a.risk_score)).slice(0, 10);
  grid.innerHTML = top
    .map((p) => {
      const score = Number(p.risk_score || 0);
      const meta = getRiskMeta(score);
      return `
        <div class="squad-card ${meta.cardClass}">
          <div class="squad-name">${p.player_name}</div>
          <div class="squad-pos">${p.position || '-'}</div>
          <span class="badge ${meta.badgeClass}">${meta.label}</span>
        </div>
      `;
    })
    .join('');
}

function renderInjuryRiskTable() {
  const tbody = document.querySelector('#page-injury .data-table tbody');
  const players = AppState.injury?.players;
  if (!tbody || !players || !players.length) return;

  tbody.innerHTML = players
    .map((p) => {
      const score = Number(p.risk_score || 0);
      const pct = Math.round(score * 100);
      const meta = getRiskMeta(score);
      const barColor = score >= 0.67 ? 'var(--red)' : score >= 0.4 ? 'var(--amber)' : 'var(--accent)';
      const recommendation = score >= 0.67 ? 'Rest advised' : score >= 0.4 ? 'Reduce load' : 'Proceed normally';
      return `
        <tr>
          <td class="fw-700">${p.player_name}</td>
          <td>${p.position || '-'}</td>
          <td>
            <div class="risk-score-bar">
              <div class="risk-bar">
                <div class="risk-bar-fill" style="width:${pct}%;background:${barColor}"></div>
              </div>
              <span class="fs-12" style="width:30px">${score.toFixed(2)}</span>
            </div>
          </td>
          <td><span class="badge ${meta.badgeClass}">${meta.label.replace(' Risk', '')}</span></td>
          <td>${Number(p.workload_30d || 0)} min</td>
          <td>${Number(p.days_since_last_injury || 0)} days ago</td>
          <td class="text-muted fs-12">${recommendation}</td>
        </tr>
      `;
    })
    .join('');
}

function getInitials(name) {
  if (!name) return '--';
  return name
    .split(' ')
    .filter(Boolean)
    .slice(0, 2)
    .map((n) => n[0].toUpperCase())
    .join('');
}

function getSelectedTeamName() {
  const selector = document.getElementById('teamSelector');
  if (!selector) return 'Selected Team';
  const option = selector.options[selector.selectedIndex];
  return option?.text || 'Selected Team';
}

function updateDataSourceBadge(source) {
  const badge = document.getElementById('dataSourceBadge');
  if (!badge) return;
  const map = {
    database: { text: 'Data: Live DB', color: '#3ecf8e' },
    fallback: { text: 'Data: Demo Fallback', color: '#f59e0b' },
    mixed: { text: 'Data: Mixed', color: '#22d3ee' },
    error: { text: 'Data: Error', color: '#ef4444' },
    unknown: { text: 'Data: Unknown', color: '#94a3b8' },
  };
  const item = map[source] || map.unknown;
  badge.textContent = item.text;
  badge.style.borderColor = `${item.color}66`;
  badge.style.color = item.color;
}

/* ============================================================
   Utility: formatNumber
   بيحول أرقام كبيرة لشكل مقروء: 1000 → 1K
   ============================================================ */
function formatNumber(n) {
  if (n >= 1_000_000) return (n / 1_000_000).toFixed(1) + 'M';
  if (n >= 1_000)     return (n / 1_000).toFixed(1) + 'K';
  return n.toString();
}

/* ============================================================
   Utility: debounce
   بنستخدمها لتحسين performance عند الـ resize أو الـ search
   ============================================================ */
function debounce(fn, delay = 200) {
  let timer;
  return (...args) => {
    clearTimeout(timer);
    timer = setTimeout(() => fn(...args), delay);
  };
}

/* ── Resize handler: نعيد رسم الـ SVG لو الـ window اتغير ── */
window.addEventListener('resize', debounce(() => {
  if (document.getElementById('page-cohesion')?.classList.contains('active')) {
    PassNetwork.init();
  }
}, 300));
