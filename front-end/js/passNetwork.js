/* ============================================================
   passNetwork.js
   بيرسم شبكة التمريرات باستخدام SVG خالص (مش library)
   ليه SVG؟ عشان محتاجين control كامل على كل عنصر
   ============================================================ */

const PassNetwork = {

  /* مواضع اللاعبين على الملعب — normalized (0 to 1) */
  players: [
    { id:'GK',  label:'GK',  x:0.10, y:0.50, size:13, highlight:false },
    { id:'RB',  label:'RB',  x:0.27, y:0.12, size:15, highlight:false },
    { id:'CB1', label:'CB',  x:0.27, y:0.35, size:17, highlight:false },
    { id:'CB2', label:'CB',  x:0.27, y:0.65, size:17, highlight:false },
    { id:'LB',  label:'LB',  x:0.27, y:0.88, size:15, highlight:false },
    { id:'CDM', label:'MF',  x:0.44, y:0.50, size:24, highlight:true  }, // Rodri — الأعلى centrality
    { id:'CM1', label:'MF',  x:0.57, y:0.27, size:22, highlight:false },
    { id:'CM2', label:'MF',  x:0.57, y:0.73, size:19, highlight:false },
    { id:'RW',  label:'FW',  x:0.75, y:0.14, size:16, highlight:false },
    { id:'ST',  label:'FW',  x:0.78, y:0.50, size:18, highlight:false },
    { id:'LW',  label:'FW',  x:0.75, y:0.86, size:16, highlight:false },
  ],

  /* التمريرات بين اللاعبين — الرقم الثالث هو القوة (1-10) */
  connections: [
    ['GK',  'CB1', 8], ['GK',  'CB2', 7], ['GK',  'RB',  4],
    ['RB',  'CB1', 6], ['CB1', 'CB2', 9], ['CB2', 'LB',  6],
    ['CB1', 'CDM', 8], ['CB2', 'CDM', 8], ['RB',  'CM1', 7], ['LB',  'CM2', 6],
    ['CDM', 'CM1', 9], ['CDM', 'CM2', 8], ['CDM', 'ST',  6],
    ['CM1', 'RW',  8], ['CM1', 'ST',  7], ['CM2', 'LW',  8], ['CM2', 'ST',  6],
    ['RW',  'ST',  7], ['LW',  'ST',  5],
  ],

  /**
   * init()
   * نقطة الدخول — بترسم الـ SVG داخل #passNetworkSvg
   */
  init(connections = null) {
    const svg = document.getElementById('passNetworkSvg');
    if (!svg) return;
    if (Array.isArray(connections) && connections.length) {
      this.connections = this.normalizeConnections(connections);
    }

    // استنى حتى الـ element يكون له أبعاد فعلية
    requestAnimationFrame(() => {
      const W = svg.clientWidth  || 800;
      const H = svg.clientHeight || 320;
      this.render(svg, W, H);
    });
  },

  normalizeConnections(edges) {
    const playerMap = {};
    this.players.forEach((p, idx) => {
      playerMap[idx] = p.id;
    });
    return edges.slice(0, 20).map((e, idx) => {
      const from = playerMap[idx % this.players.length];
      const to = playerMap[(idx + 3) % this.players.length];
      const strength = Math.max(1, Math.min(10, Math.round(Number(e.weight || 1))));
      return [from, to, strength];
    });
  },

  /**
   * render(svg, W, H)
   * بترسم الـ SVG بالكامل — connections أول ثم nodes فوقيها
   */
  render(svg, W, H) {
    const { players, connections } = this;

    // تحويل الـ relative positions لـ actual pixels
    const pos = {};
    players.forEach(p => {
      pos[p.id] = { x: p.x * W, y: p.y * H };
    });

    let html = '';

    /* ── Defs: glow filter ── */
    html += `
      <defs>
        <filter id="nodeGlow" x="-50%" y="-50%" width="200%" height="200%">
          <feGaussianBlur stdDeviation="4" result="blur"/>
          <feMerge>
            <feMergeNode in="blur"/>
            <feMergeNode in="SourceGraphic"/>
          </feMerge>
        </filter>
        <filter id="nodeGlowStrong" x="-80%" y="-80%" width="260%" height="260%">
          <feGaussianBlur stdDeviation="7" result="blur"/>
          <feMerge>
            <feMergeNode in="blur"/>
            <feMergeNode in="SourceGraphic"/>
          </feMerge>
        </filter>
      </defs>
    `;

    /* ── خط منتصف الملعب ── */
    html += `<line
      x1="${W * 0.5}" y1="15"
      x2="${W * 0.5}" y2="${H - 15}"
      stroke="rgba(99,225,180,0.08)"
      stroke-width="1"
      stroke-dasharray="6,4"
    />`;

    /* ── Connections (خطوط التمريرات) ── */
    connections.forEach(([a, b, strength]) => {
      const pa = pos[a], pb = pos[b];
      const opacity  = 0.12 + (strength / 10) * 0.45;  // كلما زاد الـ strength كلما زاد الـ opacity
      const width    = 1.2 + (strength / 10) * 3.5;    // وكذلك العرض
      html += `<line
        x1="${pa.x}" y1="${pa.y}"
        x2="${pb.x}" y2="${pb.y}"
        stroke="#3ecf8e"
        stroke-width="${width.toFixed(1)}"
        stroke-opacity="${opacity.toFixed(2)}"
        stroke-linecap="round"
      />`;
    });

    /* ── Nodes (دوائر اللاعبين) ── */
    players.forEach(p => {
      const { x, y } = pos[p.id];
      const fill   = p.highlight ? '#3ecf8e' : '#22d3ee';
      const filter = p.highlight ? 'url(#nodeGlowStrong)' : 'url(#nodeGlow)';
      const opacity= p.highlight ? 1 : 0.85;
      const textColor = '#0a0e1a';  /* نص داكن دايماً عشان يقرأ على الألوان الفاتحة */

      html += `
        <circle
          cx="${x}" cy="${y}" r="${p.size}"
          fill="${fill}" fill-opacity="${opacity}"
          filter="${filter}"
        />
        <text
          x="${x}" y="${y + 4}"
          text-anchor="middle"
          fill="${textColor}"
          font-size="9"
          font-weight="700"
          font-family="DM Sans, sans-serif"
          pointer-events="none"
        >${p.label}</text>
      `;
    });

    svg.innerHTML = html;
  },
};
