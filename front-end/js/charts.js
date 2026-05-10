Chart.defaults.color = '#94a3b8';
Chart.defaults.borderColor = 'rgba(99,225,180,0.08)';
Chart.defaults.font.family = 'DM Sans';
Chart.defaults.font.size = 12;

const C = { accent: '#3ecf8e', cyan: '#22d3ee', amber: '#f59e0b', red: '#ef4444' };
const gridColor = 'rgba(99,225,180,0.06)';

function makeGradient(ctx, colorStart, colorEnd, height = 200) {
  const grad = ctx.createLinearGradient(0, 0, 0, height);
  grad.addColorStop(0, colorStart);
  grad.addColorStop(1, colorEnd);
  return grad;
}

function destroyChart(canvas) {
  const current = Chart.getChart(canvas);
  if (current) current.destroy();
}

const Charts = {
  initPerfTrend(payload) {
    const canvas = document.getElementById('perfTrendChart');
    if (!canvas || !payload) return;
    destroyChart(canvas);
    const ctx = canvas.getContext('2d');
    const grad = makeGradient(ctx, 'rgba(62,207,142,0.25)', 'rgba(62,207,142,0)', 200);

    new Chart(ctx, {
      type: 'line',
      data: {
        labels: payload.labels,
        datasets: [{
          label: 'Performance Score',
          data: payload.values,
          borderColor:     C.accent,
          backgroundColor: grad,
          borderWidth:     2.5,
          pointBackgroundColor: C.accent,
          pointRadius:     4,
          pointHoverRadius:6,
          tension: 0.4,
          fill: true,
        }],
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
          legend: { display: false },
          tooltip: {
            backgroundColor: '#151e2d',
            borderColor: 'rgba(62,207,142,0.3)',
            borderWidth: 1,
            callbacks: {
              label: ctx => ` Score: ${ctx.raw}`,
            },
          },
        },
        scales: {
          x: { grid: { color: gridColor } },
          y: {
            min: 70, max: 100,
            grid: { color: gridColor },
          },
        },
      },
    });
  },

  initRadar(payload, playerName = 'Top Player') {
    const canvas = document.getElementById('radarChart');
    if (!canvas || !payload) return;
    destroyChart(canvas);

    new Chart(canvas.getContext('2d'), {
      type: 'radar',
      data: {
        labels: payload.labels,
        datasets: [{
          label: playerName,
          data: payload.values,
          borderColor:     C.accent,
          backgroundColor: 'rgba(62,207,142,0.15)',
          pointBackgroundColor: C.accent,
          pointHoverBackgroundColor: C.cyan,
          borderWidth: 2,
        }],
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        scales: {
          r: {
            min: 0,
            max: 100,
            grid:        { color: 'rgba(99,225,180,0.10)' },
            angleLines:  { color: 'rgba(99,225,180,0.08)' },
            ticks: {
              stepSize: 25,
              color: '#4b5563',
              font: { size: 10 },
              backdropColor: 'transparent',
            },
            pointLabels: {
              color: '#94a3b8',
              font: { size: 12, weight: '600' },
            },
          },
        },
        plugins: { legend: { display: false } },
      },
    });
  },

  initInjuryHistory(players) {
    const canvas = document.getElementById('injuryHistChart');
    if (!canvas || !players) return;
    destroyChart(canvas);

    const data = players.slice(0, 12).map((p) => Math.round((p.risk_score || 0) * 20));
    const labels = players.slice(0, 12).map((p) => p.player_name.split(' ').slice(-1)[0]);
    const colors = data.map(v =>
      v >= 13 ? C.red :
      v >= 10 ? C.amber : '#e55c3a'
    );

    new Chart(canvas.getContext('2d'), {
      type: 'bar',
      data: {
        labels,
        datasets: [{
          data,
          backgroundColor:      colors,
          borderRadius:         3,
          borderSkipped:        false,
          hoverBackgroundColor: C.amber,
        }],
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: { legend: { display: false } },
        scales: {
          x: { grid: { display: false } },
          y: {
            grid:    { color: gridColor },
            ticks:   { stepSize: 5 },
          },
        },
      },
    });
  },

  initTempPerf(points) {
    const canvas = document.getElementById('tempPerfChart');
    if (!canvas || !points) return;
    destroyChart(canvas);

    const sorted = [...points].sort((a, b) => a.x - b.x);

    new Chart(canvas.getContext('2d'), {
      data: {
        datasets: [
          // النقاط
          {
            type: 'scatter',
            label: 'Matches',
            data: sorted,
            backgroundColor: C.cyan,
            pointRadius: 6,
            pointHoverRadius: 8,
          },
          // خط الـ trend
          {
            type: 'line',
            label: 'Trend',
            data: sorted,
            borderColor:     'rgba(34,211,238,0.5)',
            backgroundColor: 'transparent',
            borderWidth:     2,
            pointRadius:     0,
            tension:         0.4,
          },
        ],
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: { legend: { display: false } },
        scales: {
          x: {
            type:  'linear',
            title: { display: true, text: 'Temperature (°C)', color: '#94a3b8' },
            grid:  { color: gridColor },
          },
          y: {
            title: { display: true, text: 'Performance Score', color: '#94a3b8' },
            grid:  { color: gridColor },
            min: 60, max: 100,
          },
        },
      },
    });
  },

  initWinProb(payload) {
    const canvas = document.getElementById('winProbChart');
    if (!canvas || !payload) return;
    destroyChart(canvas);

    new Chart(canvas.getContext('2d'), {
      type: 'line',
      data: {
        labels: payload.labels,
        datasets: [
          {
            label: 'Win Probability',
            data: payload.win,
            borderColor:     C.accent,
            backgroundColor: 'rgba(62,207,142,0.06)',
            borderWidth:     2.5,
            tension:         0.4,
            fill:            false,
            pointRadius:     0,
            pointHoverRadius:4,
          },
          {
            label: 'Draw Probability',
            data: payload.draw,
            borderColor:     C.amber,
            backgroundColor: 'transparent',
            borderWidth:     1.5,
            tension:         0.4,
            fill:            false,
            pointRadius:     0,
            borderDash:      [4, 4],
          },
          {
            label: 'Loss Probability',
            data: payload.loss,
            borderColor:     C.red,
            backgroundColor: 'transparent',
            borderWidth:     1.5,
            tension:         0.4,
            fill:            false,
            pointRadius:     0,
          },
        ],
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        interaction: { mode: 'index', intersect: false },
        plugins: {
          legend: {
            position: 'bottom',
            labels: {
              usePointStyle: true,
              color: '#94a3b8',
              font: { size: 11 },
            },
          },
          tooltip: {
            backgroundColor: '#151e2d',
            borderColor: 'rgba(62,207,142,0.3)',
            borderWidth: 1,
            callbacks: {
              title: ctx => `Minute ${ctx[0].label}`,
              label: ctx => ` ${ctx.dataset.label}: ${ctx.raw}%`,
            },
          },
        },
        scales: {
          x: {
            title: { display: true, text: 'Match Minute', color: '#94a3b8' },
            grid:  { color: gridColor },
            ticks: { maxTicksLimit: 10 },
          },
          y: {
            min:   0,
            max:   100,
            title: { display: true, text: 'Probability (%)', color: '#94a3b8' },
            grid:  { color: gridColor },
          },
        },
      },
    });
  },
};
