/* =============================================================================
   charts.js
   Chart.js configuration and chart constructors.
   All charts read from the CSS design tokens where possible.
   ============================================================================= */

// Global Chart.js defaults — monospace font for axis labels and data
Chart.defaults.color         = '#8896aa';
Chart.defaults.borderColor   = 'rgba(56, 189, 131, 0.07)';
Chart.defaults.font.family   = "'IBM Plex Mono', monospace";
Chart.defaults.font.size     = 11;

const C = {
  accent: '#38bd83',
  cyan:   '#06b6d4',
  amber:  '#f59e0b',
  red:    '#ef4444',
  purple: '#8b5cf6',
};

const gridColor = 'rgba(56, 189, 131, 0.06)';

function makeGradient(ctx, colorStart, colorEnd, height = 200) {
  const grad = ctx.createLinearGradient(0, 0, 0, height);
  grad.addColorStop(0, colorStart);
  grad.addColorStop(1, colorEnd);
  return grad;
}

function destroyChart(canvas) {
  const existing = Chart.getChart(canvas);
  if (existing) existing.destroy();
}

// Shared tooltip style applied to every chart
const tooltipDefaults = {
  backgroundColor:  '#111827',
  borderColor:      'rgba(56, 189, 131, 0.3)',
  borderWidth:      1,
  padding:          10,
  cornerRadius:     4,
  titleFont:        { family: "'IBM Plex Mono', monospace", size: 11 },
  bodyFont:         { family: "'IBM Plex Mono', monospace", size: 11 },
  titleColor:       '#8896aa',
  bodyColor:        '#e8edf5',
};

// Shared scale style
function makeScales(opts = {}) {
  return {
    x: {
      grid:   { color: gridColor },
      ticks:  { color: '#4a5568', font: { size: 10 } },
      ...opts.x,
    },
    y: {
      grid:   { color: gridColor },
      ticks:  { color: '#4a5568', font: { size: 10 } },
      ...opts.y,
    },
  };
}

const Charts = {

  // ── Performance trend (line) ──────────────────────────────────────────────
  initPerfTrend(payload) {
    const canvas = document.getElementById('perfTrendChart');
    if (!canvas || !payload) return;
    destroyChart(canvas);
    const ctx  = canvas.getContext('2d');
    const grad = makeGradient(ctx, 'rgba(56,189,131,0.2)', 'rgba(56,189,131,0)', 200);

    new Chart(ctx, {
      type: 'line',
      data: {
        labels:   payload.labels,
        datasets: [{
          label:              'Performance Score',
          data:               payload.values,
          borderColor:        C.accent,
          backgroundColor:    grad,
          borderWidth:        2,
          pointBackgroundColor: C.accent,
          pointRadius:        4,
          pointHoverRadius:   6,
          tension:            0.4,
          fill:               true,
        }],
      },
      options: {
        responsive:          true,
        maintainAspectRatio: false,
        plugins: {
          legend:  { display: false },
          tooltip: {
            ...tooltipDefaults,
            callbacks: { label: ctx => ` Score: ${ctx.raw}` },
          },
        },
        scales: makeScales({ y: { min: 60, max: 100 } }),
      },
    });
  },

  // ── Player radar ──────────────────────────────────────────────────────────
  initRadar(payload, playerName = 'Player') {
    const canvas = document.getElementById('radarChart');
    if (!canvas || !payload) return;
    destroyChart(canvas);

    new Chart(canvas.getContext('2d'), {
      type: 'radar',
      data: {
        labels:   payload.labels,
        datasets: [{
          label:                playerName,
          data:                 payload.values,
          borderColor:          C.accent,
          backgroundColor:      'rgba(56,189,131,0.12)',
          pointBackgroundColor: C.accent,
          pointHoverBackgroundColor: C.cyan,
          borderWidth:          2,
          pointRadius:          3,
        }],
      },
      options: {
        responsive:          true,
        maintainAspectRatio: false,
        scales: {
          r: {
            min:        0,
            max:        100,
            grid:       { color: 'rgba(56,189,131,0.10)' },
            angleLines: { color: 'rgba(56,189,131,0.08)' },
            ticks: {
              stepSize:       25,
              color:          '#4a5568',
              font:           { size: 9 },
              backdropColor:  'transparent',
            },
            pointLabels: {
              color: '#8896aa',
              font:  { size: 11, family: "'IBM Plex Mono', monospace" },
            },
          },
        },
        plugins: {
          legend:  { display: false },
          tooltip: tooltipDefaults,
        },
      },
    });
  },

  // ── Injury history bar ────────────────────────────────────────────────────
  initInjuryHistory(players) {
    const canvas = document.getElementById('injuryHistChart');
    if (!canvas || !players || !players.length) return;
    destroyChart(canvas);

    const data   = players.slice(0, 12).map(p => Math.round((p.risk_score || 0) * 20));
    const labels = players.slice(0, 12).map(p => p.player_name.split(' ').slice(-1)[0]);
    const colors = data.map(v =>
      v >= 13 ? C.red :
      v >= 8  ? C.amber : C.accent
    );

    new Chart(canvas.getContext('2d'), {
      type: 'bar',
      data: {
        labels,
        datasets: [{
          data:                 data,
          backgroundColor:      colors,
          borderRadius:         3,
          borderSkipped:        false,
          hoverBackgroundColor: C.cyan,
        }],
      },
      options: {
        responsive:          true,
        maintainAspectRatio: false,
        plugins: {
          legend:  { display: false },
          tooltip: {
            ...tooltipDefaults,
            callbacks: { label: ctx => ` Risk score: ${(ctx.raw / 20).toFixed(2)}` },
          },
        },
        scales: makeScales({ x: { grid: { display: false } } }),
      },
    });
  },

  // ── Temperature vs performance scatter ────────────────────────────────────
  initTempPerf(points) {
    const canvas = document.getElementById('tempPerfChart');
    if (!canvas || !points || !points.length) return;
    destroyChart(canvas);

    const sorted = [...points].sort((a, b) => a.x - b.x);

    new Chart(canvas.getContext('2d'), {
      data: {
        datasets: [
          {
            type:            'scatter',
            label:           'Matches',
            data:            sorted,
            backgroundColor: C.cyan,
            pointRadius:     5,
            pointHoverRadius:7,
          },
          {
            type:            'line',
            label:           'Trend',
            data:            sorted,
            borderColor:     'rgba(6,182,212,0.4)',
            backgroundColor: 'transparent',
            borderWidth:     1.5,
            pointRadius:     0,
            tension:         0.4,
          },
        ],
      },
      options: {
        responsive:          true,
        maintainAspectRatio: false,
        plugins: {
          legend:  { display: false },
          tooltip: {
            ...tooltipDefaults,
            callbacks: {
              label: ctx => ` ${ctx.dataset.label === 'Matches' ? `Temp: ${ctx.parsed.x}°C  Score: ${ctx.parsed.y}` : ''}`,
            },
          },
        },
        scales: makeScales({
          x: {
            type:  'linear',
            title: { display: true, text: 'Temperature (°C)', color: '#4a5568', font: { size: 10 } },
          },
          y: {
            min:   60,
            max:   100,
            title: { display: true, text: 'Performance', color: '#4a5568', font: { size: 10 } },
          },
        }),
      },
    });
  },

  // ── Win probability timeline ──────────────────────────────────────────────
  initWinProb(payload) {
    const canvas = document.getElementById('winProbChart');
    if (!canvas || !payload) return;
    destroyChart(canvas);

    new Chart(canvas.getContext('2d'), {
      type: 'line',
      data: {
        labels:   payload.labels,
        datasets: [
          {
            label:           'Win',
            data:            payload.win,
            borderColor:     C.accent,
            backgroundColor: 'rgba(56,189,131,0.05)',
            borderWidth:     2,
            tension:         0.4,
            fill:            false,
            pointRadius:     0,
            pointHoverRadius:4,
          },
          {
            label:       'Draw',
            data:        payload.draw,
            borderColor: C.amber,
            borderWidth: 1.5,
            tension:     0.4,
            fill:        false,
            pointRadius: 0,
            borderDash:  [4, 4],
          },
          {
            label:       'Loss',
            data:        payload.loss,
            borderColor: C.red,
            borderWidth: 1.5,
            tension:     0.4,
            fill:        false,
            pointRadius: 0,
          },
        ],
      },
      options: {
        responsive:          true,
        maintainAspectRatio: false,
        interaction:         { mode: 'index', intersect: false },
        plugins: {
          legend: {
            position: 'bottom',
            labels: {
              usePointStyle: true,
              color:         '#8896aa',
              font:          { size: 11, family: "'IBM Plex Mono', monospace" },
              padding:       16,
            },
          },
          tooltip: {
            ...tooltipDefaults,
            callbacks: {
              title: ctx  => `Minute ${ctx[0].label}`,
              label: ctx  => ` ${ctx.dataset.label}: ${ctx.raw}%`,
            },
          },
        },
        scales: makeScales({
          x: {
            title:   { display: true, text: 'Match Minute', color: '#4a5568', font: { size: 10 } },
            ticks:   { maxTicksLimit: 10 },
          },
          y: {
            min:   0,
            max:   100,
            title: { display: true, text: 'Probability (%)', color: '#4a5568', font: { size: 10 } },
          },
        }),
      },
    });
  },

  // ── Goals vs xG (grouped bars) ────────────────────────────────────────────
  initXGChart(players) {
    const canvas = document.getElementById('xgChart');
    if (!canvas || !players || !players.length) return;
    destroyChart(canvas);

    const labels = players.map(p => p.player_name.split(' ').slice(-1)[0]);
    new Chart(canvas.getContext('2d'), {
      type: 'bar',
      data: {
        labels,
        datasets: [
          {
            label:           'Goals',
            data:            players.map(p => p.goals || 0),
            backgroundColor: C.accent,
            borderRadius:    3,
            borderSkipped:   false,
          },
          {
            label:           'xG',
            data:            players.map(p => Number(p.xg || 0)),
            backgroundColor: C.cyan,
            borderRadius:    3,
            borderSkipped:   false,
          },
        ],
      },
      options: {
        responsive:          true,
        maintainAspectRatio: false,
        plugins: {
          legend: {
            position: 'bottom',
            labels: {
              usePointStyle: true,
              color:         '#8896aa',
              font:          { size: 11, family: "'IBM Plex Mono', monospace" },
              padding:       16,
            },
          },
          tooltip: {
            ...tooltipDefaults,
            callbacks: { label: ctx => ` ${ctx.dataset.label}: ${Number(ctx.raw).toFixed(2)}` },
          },
        },
        scales: makeScales({ x: { grid: { display: false } } }),
      },
    });
  },
};