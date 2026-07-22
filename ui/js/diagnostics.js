// ════════════════════════════════════════════════════════════════
// DIAGNOSTICS TAB
// ════════════════════════════════════════════════════════════════

(function() {
'use strict';

// ── Rolling data store ─────────────────────────────────────────
let diagWindow = 120;       // frames of history
let diagPaused = false;
let diagPolling = false;
let diagTimer = null;

const diagSeries = {
  fps:    [],   // float[]
  ms:     [],   // float[]
};

// ── Tiny canvas sparkline renderer ────────────────────────────
const DIAG_COLORS = {
  fps:    '#34d399',  // green
  ms:     '#6c8ef5',  // blue
  grid:   'rgba(255,255,255,0.04)',
};

function diagGetCSSVar(name) {
  return getComputedStyle(document.documentElement).getPropertyValue(name).trim();
}

function diagDrawLine(canvas, data, color, opts = {}) {
  const dpr  = window.devicePixelRatio || 1;
  const W    = canvas.clientWidth  || canvas.width  / dpr;
  const H    = canvas.clientHeight || canvas.height / dpr;
  canvas.width  = W * dpr;
  canvas.height = H * dpr;
  const ctx = canvas.getContext('2d');
  ctx.scale(dpr, dpr);

  const bg = diagGetCSSVar('--bg2') || '#1a1c1f';
  ctx.clearRect(0, 0, W, H);

  if (!data || data.length < 2) {
    ctx.fillStyle = diagGetCSSVar('--muted2') || '#5f636c';
    ctx.font = '11px system-ui';
    ctx.textAlign = 'center';
    ctx.fillText('waiting for data…', W / 2, H / 2 + 4);
    return;
  }

  const window_ = opts.window || diagWindow;
  const slice   = data.slice(-window_);
  const min     = opts.min !== undefined ? opts.min : Math.min(...slice);
  const max_    = opts.max !== undefined ? opts.max : Math.max(...slice);
  const range   = Math.max(max_ - min, 1);

  // Grid lines
  ctx.strokeStyle = DIAG_COLORS.grid;
  ctx.lineWidth = 1;
  for (let i = 1; i < 4; i++) {
    const y = H - (i / 4) * H;
    ctx.beginPath(); ctx.moveTo(0, y); ctx.lineTo(W, y); ctx.stroke();
  }

  // Area fill
  const grad = ctx.createLinearGradient(0, 0, 0, H);
  grad.addColorStop(0, color + '55');
  grad.addColorStop(1, color + '08');
  ctx.beginPath();
  ctx.moveTo(0, H);
  slice.forEach((v, i) => {
    const x = (i / (slice.length - 1)) * W;
    const y = H - ((v - min) / range) * (H * 0.85) - H * 0.05;
    if (i === 0) ctx.lineTo(x, y); else ctx.lineTo(x, y);
  });
  ctx.lineTo(W, H);
  ctx.closePath();
  ctx.fillStyle = grad;
  ctx.fill();

  // Line
  ctx.beginPath();
  ctx.strokeStyle = color;
  ctx.lineWidth = 1.5;
  ctx.lineJoin = 'round';
  slice.forEach((v, i) => {
    const x = (i / (slice.length - 1)) * W;
    const y = H - ((v - min) / range) * (H * 0.85) - H * 0.05;
    i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
  });
  ctx.stroke();

  // Min/max labels
  ctx.fillStyle = diagGetCSSVar('--muted2') || '#5f636c';
  ctx.font = '9px system-ui';
  ctx.textAlign = 'right';
  ctx.fillText(max_.toFixed(1), W - 2, 10);
  ctx.fillText(min.toFixed(1), W - 2, H - 3);
}

// ── DOM refs ───────────────────────────────────────────────────
let _diagDomReady = false;
let _elFpsChart, _elMsChart, _elNodeBars, _elSplitCompute, _elSplitOverhead,
    _elComputeMs, _elOverheadMs, _elStatusPill, _elPauseBtn;

function diagBindDom() {
  if (_diagDomReady) return;
  _elFpsChart       = document.getElementById('diag-fps-chart');
  _elMsChart        = document.getElementById('diag-ms-chart');
  _elNodeBars       = document.getElementById('diag-node-bars');
  _elSplitCompute   = document.getElementById('diag-split-compute');
  _elSplitOverhead  = document.getElementById('diag-split-overhead');
  _elComputeMs      = document.getElementById('diag-compute-ms');
  _elOverheadMs     = document.getElementById('diag-overhead-ms');
  _elStatusPill     = document.getElementById('diag-status-pill');
  _elPauseBtn       = document.getElementById('diag-pause-btn');
  _diagDomReady = true;
}

// ── Update readout helpers ─────────────────────────────────────
function diagEl(id, txt) {
  const el = document.getElementById(id);
  if (el) el.textContent = txt;
}

// ── Render one diagnostics snapshot ───────────────────────────
function diagRender(d) {
  diagBindDom();

  const running = d.running;
  _elStatusPill.textContent = running ? 'live' : 'stopped';
  _elStatusPill.className   = 'diag-status-pill' + (running ? ' live' : '');

  if (!running) return;

  // Push to rolling series
  diagSeries.fps.push(d.fps   || 0);
  diagSeries.ms.push( d.cook_ms || 0);

  // Trim to window
  const wl = diagWindow;
  if (diagSeries.fps.length > wl) diagSeries.fps.splice(0, diagSeries.fps.length - wl);
  if (diagSeries.ms.length  > wl) diagSeries.ms.splice( 0, diagSeries.ms.length  - wl);

  // Numeric readouts
  diagEl('dkv-fps',    (d.fps || 0).toFixed(1));
  diagEl('dkv-ms',     (d.cook_ms || 0).toFixed(1));
  const msArr = diagSeries.ms;
  const avg   = msArr.length ? (msArr.reduce((a,b) => a+b, 0) / msArr.length) : 0;
  const mn    = msArr.length ? Math.min(...msArr) : 0;
  const mx    = msArr.length ? Math.max(...msArr) : 0;
  diagEl('dkv-avg',    avg.toFixed(1));
  diagEl('dkv-minmax', mn.toFixed(0) + ' / ' + mx.toFixed(0));
  diagEl('dkv-frame',  d.frame || 0);
  diagEl('dkv-nodes',  (d.active_nodes || 0) + ' / ' + (d.active_edges || 0) + ' edges');
  diagEl('dkv-res',    (d.canvas_w || '?') + '×' + (d.canvas_h || '?'));
  diagEl('dkv-cache',  d.sim_cache_entries != null ? d.sim_cache_entries : '–');

  // Charts
  diagDrawLine(_elFpsChart, diagSeries.fps, DIAG_COLORS.fps, { min: 0 });
  diagDrawLine(_elMsChart,  diagSeries.ms,  DIAG_COLORS.ms,  { min: 0 });

  // Per-node timing bars
  const timings  = d.node_timings  || {};
  const names    = d.node_names    || {};
  const totalMs  = Object.values(timings).reduce((a, b) => a + b, 0) || 1;
  const nodeIds  = Object.keys(timings).sort((a, b) => timings[b] - timings[a]);

  // Re-use or rebuild rows (keyed by node id)
  const existingRows = {};
  _elNodeBars.querySelectorAll('.diag-nbar-row').forEach(r => {
    existingRows[r.dataset.nid] = r;
  });
  const seenIds = new Set();
  nodeIds.forEach(nid => {
    seenIds.add(nid);
    const ms  = timings[nid];
    const pct = Math.min(100, (ms / totalMs) * 100);
    const lbl = names[nid] || nid;
    const isGpu = lbl.toLowerCase().startsWith('gpu') || (d.gpu_nodes && ms > 0);

    let row = existingRows[nid];
    if (!row) {
      row = document.createElement('div');
      row.className = 'diag-nbar-row';
      row.dataset.nid = nid;
      row.innerHTML = `
        <span class="diag-nbar-label" title="${lbl}">${lbl}</span>
        <div class="diag-nbar-track"><div class="diag-nbar-fill${isGpu ? ' gpu' : ''}"></div></div>
        <span class="diag-nbar-val">0ms</span>`;
      _elNodeBars.appendChild(row);
    }
    row.querySelector('.diag-nbar-label').textContent = lbl;
    row.querySelector('.diag-nbar-fill').style.width  = pct.toFixed(1) + '%';
    row.querySelector('.diag-nbar-val').textContent   = ms.toFixed(1) + 'ms';
  });
  // Remove stale rows
  Object.keys(existingRows).forEach(nid => {
    if (!seenIds.has(nid)) existingRows[nid].remove();
  });

  // Overhead split bar
  const computeMs  = d.node_compute_ms || 0;
  const overheadMs = d.overhead_ms     || 0;
  const totalExec  = computeMs + overheadMs || 1;
  const computePct  = (computeMs  / totalExec * 100).toFixed(1);
  const overheadPct = (overheadMs / totalExec * 100).toFixed(1);
  _elSplitCompute.style.width  = computePct  + '%';
  _elSplitOverhead.style.width = overheadPct + '%';
  _elSplitCompute.textContent  = computePct  > 15 ? 'Compute ' + computePct  + '%' : '';
  _elSplitOverhead.textContent = overheadPct > 15 ? 'Overhead ' + overheadPct + '%' : '';
  _elComputeMs.textContent  = 'Compute: '  + computeMs.toFixed(1)  + 'ms';
  _elOverheadMs.textContent = 'Overhead: ' + overheadMs.toFixed(1) + 'ms';

  // Cache & data-flow chips
  diagEl('dfc-hits',   d.total_cache_hits   || 0);
  diagEl('dfc-misses', d.total_cache_misses || 0);
  diagEl('dfc-inv',    d.last_invalidated   || 0);
  diagEl('dfc-gpu',    (d.gpu_nodes  || 0) + ' / ' + ((d.gpu_nodes || 0) + (d.cpu_nodes || 0)));
  diagEl('dfc-mem',    d.mem_edges   || 0);
  diagEl('dfc-disk',   d.disk_edges  || 0);

  // Incremental re-cook counters (Phase 6)
  if (d.nodes_cooked != null) diagEl('dfc-cooked',  d.nodes_cooked);
  if (d.nodes_skipped != null) diagEl('dfc-skipped', d.nodes_skipped);

  // Feed the p5 graph-flow visualization (#4b).
  gDiagFlowUpdate(d);
  // Feed the decorative graph FX overlay (#4c-safe).
  gOverlayFeed(d);
}

// ── #4b: p5 graph-flow visualization ────────────────────────────────────────
// A spatial view of the graph where each node glows by its cook time and data
// flows as particles along the wires. Lazy p5 instance; additive to Diagnostics.
let _gDiagFlowInst = null, _gDiagFlowData = { nodes: [], edges: [], maxMs: 1, t0: performance.now() };
function _gDiagFlowLoadP5() {
  if (window.p5) return Promise.resolve(window.p5);
  if (window.__p5FlowPromise) return window.__p5FlowPromise;
  window.__p5FlowPromise = new Promise((res, rej) => {
    const s = document.createElement('script');
    s.src = '/ui/vendor/p5.min.js'; s.async = true;
    s.onload = () => res(window.p5); s.onerror = () => rej(new Error('p5 load failed'));
    document.head.appendChild(s);
  });
  return window.__p5FlowPromise;
}
function gDiagFlowUpdate(d) {
  // Build node/edge geometry from the live graph + timings.
  const timings = d.node_timings || {};
  const names = d.node_names || {};
  const xs = gNodes.map(n => n.x || 0), ys = gNodes.map(n => n.y || 0);
  const minX = Math.min(...xs, 0), maxX = Math.max(...xs, 1);
  const minY = Math.min(...ys, 0), maxY = Math.max(...ys, 1);
  const nx = v => (v - minX) / Math.max(1, maxX - minX);
  const ny = v => (v - minY) / Math.max(1, maxY - minY);
  const nodes = gNodes.map(n => ({
    id: n.id, x: nx(n.x || 0), y: ny(n.y || 0),
    name: names[n.id] || gNodeDefs[n.method_id]?.name || n.method_id,
    ms: timings[n.id] || 0,
  }));
  const idset = new Set(nodes.map(n => n.id));
  const edges = gEdges.filter(e => idset.has(e.src_node) && idset.has(e.dst_node))
    .map(e => ({ a: e.src_node, b: e.dst_node }));
  _gDiagFlowData = {
    nodes, edges,
    maxMs: Math.max(1, ...nodes.map(n => n.ms)),
    t0: _gDiagFlowData.t0,
  };
  gDiagFlowEnsure();
}
async function gDiagFlowEnsure() {
  if (_gDiagFlowInst) return;
  const cont = document.getElementById('diag-flow');
  if (!cont) return;
  _gDiagFlowInst = 'loading';
  try { await _gDiagFlowLoadP5(); } catch { _gDiagFlowInst = null; return; }
  const P5 = window.p5;
  const sketch = (p) => {
    p.setup = () => {
      const w = cont.clientWidth || 600, h = cont.clientHeight || 260;
      p.createCanvas(w, h); p.pixelDensity(1);
    };
    p.windowResized = () => { p.resizeCanvas(cont.clientWidth || 600, cont.clientHeight || 260); };
    p.draw = () => {
      const D = _gDiagFlowData, W = p.width, H = p.height, pad = 34;
      p.clear(); p.background(11, 14, 20);
      const px = x => pad + x * (W - 2 * pad), py = y => pad + y * (H - 2 * pad);
      const byId = {}; D.nodes.forEach(n => byId[n.id] = n);
      const now = (performance.now() - D.t0) / 1000;
      // Edges + flowing particles.
      for (const e of D.edges) {
        const a = byId[e.a], b = byId[e.b]; if (!a || !b) continue;
        const ax = px(a.x), ay = py(a.y), bx = px(b.x), by = py(b.y);
        p.stroke(60, 80, 120, 150); p.strokeWeight(1.5); p.line(ax, ay, bx, by);
        const load = a.ms / D.maxMs;
        const cnt = 1 + Math.round(load * 4);
        for (let i = 0; i < cnt; i++) {
          const f = ((now * (0.35 + load) + i / cnt) % 1);
          const dotx = ax + (bx - ax) * f, doty = ay + (by - ay) * f;
          p.noStroke(); p.fill(90, 200, 255, 220);
          p.circle(dotx, doty, 3 + load * 3);
        }
      }
      // Nodes — glow by cook time.
      for (const n of D.nodes) {
        const x = px(n.x), y = py(n.y), load = n.ms / D.maxMs;
        const pulse = 1 + 0.12 * Math.sin(now * 4 + n.x * 6);
        const r = (13 + load * 16) * (load > 0 ? pulse : 1);
        p.noStroke();
        p.fill(40 + 180 * load, 120 + 60 * load, 255 - 120 * load, 60);
        p.circle(x, y, r * 2.2);
        p.fill(60 + 190 * load, 150 + 60 * load, 255 - 100 * load, 235);
        p.circle(x, y, r);
        p.fill(220); p.textSize(10); p.textAlign(p.CENTER, p.TOP);
        p.text(String(n.name).slice(0, 16), x, y + r + 3);
        if (n.ms > 0) { p.fill(150, 220, 255); p.textAlign(p.CENTER, p.BOTTOM); p.text(n.ms.toFixed(1) + 'ms', x, y - r - 2); }
      }
      if (!D.nodes.length) {
        p.fill(120); p.textAlign(p.CENTER, p.CENTER); p.textSize(12);
        p.text('Run a graph live to see it flow', W / 2, H / 2);
      }
    };
  };
  _gDiagFlowInst = new P5(sketch, cont);
}

// ── #4c-safe: decorative graph FX overlay ───────────────────────────────────
// A canvas layered over the DOM/SVG graph (pointer-events:none) that renders
// animated wires, node cook-time heat, active/error pulses, and a minimap. It
// shares the graph's screen space via gPortPos / node rects so it stays aligned
// through pan/zoom, and NEVER handles graph interaction. Default off.
let gOverlayEnabled = false;
let _gOvlTele = { timings: {}, errors: {}, names: {}, running: false, memEdges: 0, diskEdges: 0, ts: 0 };
let _gOvlRaf = null;
const _gOvlCanvas = document.getElementById('graph-overlay');
const _gOvlMini   = document.getElementById('graph-minimap');
const _gOvlMiniCv = document.getElementById('graph-minimap-canvas');
const NODE_W = 180, NODE_H = 92; // approx logical node box for the minimap

function gOverlayFeed(d) {
  // Telemetry from the live diagnostics feed (works for server-live graphs).
  _gOvlTele = {
    timings: d.node_timings || {}, errors: d.node_errors || {}, names: d.node_names || {},
    running: !!d.running, memEdges: d.mem_edges || 0, diskEdges: d.disk_edges || 0,
    edgeTransport: d.edge_transport || null,   // {"src->dst":"mem"|"disk"} — real per-edge transport
    fps: d.fps || 0, ts: performance.now(),
  };
  if (gOverlayEnabled && !_gOvlRaf) _gOvlStart();
}

function _gOvlResize() {
  if (!_gOvlCanvas) return;
  _gOvlReadPalette(); // pick up the active theme's colours
  const r = gCanvasWrap.getBoundingClientRect();
  const dpr = Math.min(2, window.devicePixelRatio || 1);
  _gOvlCanvas.width = Math.max(1, Math.round(r.width * dpr));
  _gOvlCanvas.height = Math.max(1, Math.round(r.height * dpr));
  const ctx = _gOvlCanvas.getContext('2d'); ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  if (_gOvlMiniCv) {
    const mr = _gOvlMini.getBoundingClientRect();
    _gOvlMiniCv.width = Math.round(mr.width * dpr); _gOvlMiniCv.height = Math.round(mr.height * dpr);
    _gOvlMiniCv.getContext('2d').setTransform(dpr, 0, 0, dpr, 0, 0);
  }
}

function _gOvlMaxMs() { const v = Object.values(_gOvlTele.timings); return Math.max(1, ...v); }

// ── "Meaningful motion" visual language ─────────────────────────────────────
// Every animated channel encodes live telemetry so the whole system state is
// readable at a glance (see the legend). Cohesive, limited palette.
let _gOvlPal = null;
// Node category → hue (subway-line style). Read instantly = node type.
const _OVL_CAT_COL = { gpu: [232,168,72], cpu: [96,165,250], '3d': [167,139,250], p5: [74,222,128] };
let _gOvlPrevMs = {};   // last frame's per-node cook time (cooking→cached edge detection)
let _gOvlTickAt = {};   // node id → time it last became cached (reuse-tick animation)
function _gOvlReadPalette() {
  const cs = getComputedStyle(document.documentElement);
  const hex = v => { const h = (cs.getPropertyValue(v) || '').trim().replace('#',''); const n = parseInt(h.length===3? h.replace(/(.)/g,'$1$1'):h, 16); return [(n>>16)&255,(n>>8)&255,n&255]; };
  _gOvlPal = { accent: hex('--accent'), live: hex('--live'), muted: hex('--muted'), text: hex('--text'),
               frost: [168, 190, 214], cat: _OVL_CAT_COL };
}
function _ovlLerp3(a, b, t) { return [a[0]+(b[0]-a[0])*t, a[1]+(b[1]-a[1])*t, a[2]+(b[2]-a[2])*t]; }
function _ovlRgba(c, a) { return `rgba(${c[0]|0},${c[1]|0},${c[2]|0},${a})`; }
function _ovlRoundRect(ctx, x, y, w, h, rad) {
  if (ctx.roundRect) { ctx.beginPath(); ctx.roundRect(x, y, w, h, rad); return; }
  ctx.beginPath(); ctx.moveTo(x+rad, y);
  ctx.arcTo(x+w, y, x+w, y+h, rad); ctx.arcTo(x+w, y+h, x, y+h, rad);
  ctx.arcTo(x, y+h, x, y, rad); ctx.arcTo(x, y, x+w, y, rad); ctx.closePath();
}
// Node category from the client node def (always available, no telemetry needed).
function _ovlCat(node) {
  const def = gNodeDefs[node.method_id] || {}, tags = def.tags || [], cat = def.category || '', mid = node.method_id;
  if (mid === '__p5sketch__' || tags.includes('p5')) return 'p5';
  if (cat === 'client_3d') return '3d';
  if (tags.includes('gpu') || cat === 'gpu_shaders' || (/^\d+$/.test(mid) && +mid >= 173 && +mid <= 219)) return 'gpu';
  return 'cpu';
}
// GPU/3D/p5 nodes use the in-memory ndarray contract (smooth stream); legacy CPU
// nodes move heavier buffers (chunky, disk-capable packets).
function _ovlIsMem(catKey) { return catKey !== 'cpu'; }
// Payload weight from the wire's port type → bytes/frame (packet size + density).
function _ovlPayload(ptype) {
  switch (ptype || 'image') {
    case 'image': case 'mask': return 1.0;
    case 'field': case 'colormap': return 0.58;
    case 'particles': return 0.5;
    case 'scalar': return 0.18;
    default: return 0.6;
  }
}
// Point on the flow bezier at parameter f∈[0,1].
function _ovlBez(p0, c0, c1, p1, f) {
  const u = 1 - f, a = u*u*u, b = 3*u*u*f, c = 3*u*f*f, d = f*f*f;
  return { x: a*p0.x + b*c0.x + c*c1.x + d*p1.x, y: a*p0.y + b*c0.y + c*c1.y + d*p1.y };
}

function _gOvlDraw() {
  if (!_gOvlCanvas || !gOverlayEnabled) return;
  if (!_gOvlPal) _gOvlReadPalette();
  const r = gCanvasWrap.getBoundingClientRect();
  // The graph wrap resizes when the live preview panel opens/closes; keep the
  // canvas buffer matched so the overlay stays pixel-aligned with the nodes.
  const dpr = Math.min(2, window.devicePixelRatio || 1);
  if (_gOvlCanvas.width !== Math.round(r.width * dpr) || _gOvlCanvas.height !== Math.round(r.height * dpr)) _gOvlResize();
  const ctx = _gOvlCanvas.getContext('2d');
  ctx.clearRect(0, 0, r.width, r.height);
  const now = performance.now() / 1000;
  const live = _gOvlTele.running && (performance.now() - _gOvlTele.ts < 1500);
  const maxMs = _gOvlMaxMs();
  const nodeById = new Map(gNodes.map(n => [n.id, n]));
  // GLOBAL TEMPO — the whole overlay's liveliness tracks fps (feel performance).
  // Floored so motion stays legible in a loop even on a slow graph.
  const fps = _gOvlTele.fps || 0;
  const tempo = Math.max(0.42, Math.min(1.2, 0.42 + fps / 55));
  const cookThresh = Math.max(0.6, maxMs * 0.03);

  // ── EDGES: data flow ──────────────────────────────────────────────────────
  ctx.lineCap = 'round';
  gEdges.forEach(edge => {
    const s = gNodesEl.querySelector(`.gport[data-nid="${edge.src_node}"][data-port="${edge.src_port}"][data-dir="output"]`);
    const d = gNodesEl.querySelector(`.gport[data-nid="${edge.dst_node}"][data-port="${edge.dst_port}"][data-dir="input"]`);
    if (!s || !d) return;
    const p0 = gPortPos(s), p1 = gPortPos(d);
    const load = Math.min(1, (_gOvlTele.timings[edge.src_node] || 0) / maxMs);
    if (!live || load <= 0.01) return;
    const srcNode = nodeById.get(edge.src_node);
    const catKey = srcNode ? _ovlCat(srcNode) : 'cpu';
    const col = _gOvlPal.cat[catKey];
    // Transport: prefer the REAL per-edge value from telemetry (mem/disk);
    // a present map with no entry for this edge = a value pass (mem). Fall back
    // to the category heuristic only when the field is absent (older frames).
    let mem;
    if (_gOvlTele.edgeTransport) {
      mem = (_gOvlTele.edgeTransport[edge.src_node + '->' + edge.dst_node] || 'mem') !== 'disk';
    } else {
      mem = _ovlIsMem(catKey);
    }
    const payload = _ovlPayload(s.dataset.ptype);
    const off = Math.max(24, Math.min(120, Math.abs(p1.x - p0.x) * 0.5));
    const c0 = { x: p0.x + off, y: p0.y }, c1 = { x: p1.x - off, y: p1.y };
    // faint guide so the path always reads
    ctx.beginPath(); ctx.moveTo(p0.x, p0.y); ctx.bezierCurveTo(c0.x, c0.y, c1.x, c1.y, p1.x, p1.y);
    ctx.setLineDash([]); ctx.lineWidth = 1; ctx.strokeStyle = _ovlRgba(col, 0.08 + 0.10 * load); ctx.stroke();
    const spd = (0.24 + load * 1.0) * tempo; // path-fractions/sec ∝ throughput
    ctx.setLineDash([]);
    if (mem) {
      // MEMORY: smooth continuous stream — many small dots; size/density ∝ bytes.
      const dotR = 1.1 + payload * 1.9;
      const gap = 0.10 - payload * 0.05;              // denser for bigger payloads
      const nDots = Math.max(4, Math.round(1 / gap));
      for (let i = 0; i < nDots; i++) {
        const f = ((now * spd + i / nDots) % 1);
        const pt = _ovlBez(p0, c0, c1, p1, f);
        ctx.beginPath(); ctx.arc(pt.x, pt.y, dotR, 0, 6.2832);
        ctx.fillStyle = _ovlRgba(col, 0.30 + 0.45 * load);
        ctx.fill();
      }
    } else {
      // DISK: discrete chunky packets — few large rounded blocks, stepped motion.
      const nPk = 3 + Math.round(payload * 2), steps = 8;
      const pw = 8 + payload * 7, ph = 4 + payload * 3;
      for (let i = 0; i < nPk; i++) {
        const raw = now * spd * 0.7 + i / nPk;
        const f = (Math.floor(raw * steps) / steps) % 1; // quantised → visibly chunky
        const pt = _ovlBez(p0, c0, c1, p1, f);
        _ovlRoundRect(ctx, pt.x - pw/2, pt.y - ph/2, pw, ph, 2);
        ctx.fillStyle = _ovlRgba(col, 0.5 + 0.35 * load); ctx.fill();
        ctx.lineWidth = 0.8; ctx.strokeStyle = _ovlRgba(_ovlLerp3(col, [255,255,255], 0.3), 0.5 + 0.3 * load); ctx.stroke();
      }
    }
  });

  // ── NODES: type, load, cooking vs cached, error ───────────────────────────
  for (const n of gNodes) {
    const el = document.getElementById('gnode-' + n.id);
    if (!el) continue;
    const nr = el.getBoundingClientRect();
    const x = nr.left - r.left, y = nr.top - r.top, w = nr.width, h = nr.height;
    const ms = _gOvlTele.timings[n.id] || 0, load = Math.min(1, ms / maxMs);
    const col = _gOvlPal.cat[_ovlCat(n)]; // rim HUE = category
    const cooking = live && ms > cookThresh;

    // cooking→cached transition → schedule a reuse tick (cache/skip event)
    if (live) {
      const prev = _gOvlPrevMs[n.id] || 0;
      if (prev > cookThresh && ms <= cookThresh) _gOvlTickAt[n.id] = now;
      _gOvlPrevMs[n.id] = ms;
    }

    if (cooking) {
      // GLOW intensity ∝ cook time, in the category hue.
      const pulse = 0.82 + 0.18 * Math.sin(now * (2.4 + load * 4) * tempo);
      ctx.save();
      ctx.shadowColor = _ovlRgba(col, 0.5); ctx.shadowBlur = 4 + load * 8;
      ctx.lineWidth = 1.0 + load * 1.3;
      ctx.strokeStyle = _ovlRgba(col, (0.26 + 0.30 * load) * pulse);
      _ovlRoundRect(ctx, x - 1.5, y - 1.5, w + 3, h + 3, 9); ctx.stroke();
      ctx.restore();
      // COOKING SHIMMER — a bright segment sweeping the rim (speed ∝ load·tempo).
      const perim = 2 * (w + h + 6);
      ctx.save();
      ctx.setLineDash([perim * 0.16, perim]);
      ctx.lineDashOffset = -((now * (55 + load * 150) * tempo) % perim);
      ctx.lineWidth = 1.7; ctx.strokeStyle = _ovlRgba(_ovlLerp3(col, [255,255,255], 0.25), 0.55 + 0.35 * load);
      _ovlRoundRect(ctx, x - 1.5, y - 1.5, w + 3, h + 3, 9); ctx.stroke();
      ctx.setLineDash([]); ctx.restore();
    } else if (live) {
      // CACHED / SKIPPED — static frosted dashed rim (calm, "frozen").
      ctx.save();
      ctx.setLineDash([1.5, 4]); ctx.lineWidth = 1;
      ctx.strokeStyle = _ovlRgba(_gOvlPal.frost, 0.22);
      _ovlRoundRect(ctx, x - 1.5, y - 1.5, w + 3, h + 3, 9); ctx.stroke();
      ctx.setLineDash([]); ctx.restore();
    }

    // CACHE REUSE TICK — a quick expanding ring when a node becomes cached.
    const tickAge = _gOvlTickAt[n.id] != null ? now - _gOvlTickAt[n.id] : 99;
    if (tickAge < 0.55) {
      const k = tickAge / 0.55, cx = x + w/2, cy = y + h/2;
      ctx.beginPath(); ctx.arc(cx, cy, (Math.min(w,h)*0.4) + k * 16, 0, 6.2832);
      ctx.strokeStyle = _ovlRgba(_gOvlPal.frost, (1 - k) * 0.5); ctx.lineWidth = 1.4; ctx.stroke();
    } else if (_gOvlTickAt[n.id] != null) { delete _gOvlTickAt[n.id]; }

    // ERROR — crisp red pulse ring (overrides).
    if (_gOvlTele.errors && _gOvlTele.errors[n.id]) {
      const a = 0.6 + 0.28 * Math.sin(now * 4.2);
      ctx.lineWidth = 1.6; ctx.strokeStyle = _ovlRgba(_gOvlPal.live, a);
      _ovlRoundRect(ctx, x - 2.5, y - 2.5, w + 5, h + 5, 10); ctx.stroke();
    }
  }

  _gOvlDrawMinimap(r);
}

function _gOvlDrawMinimap(wrapRect) {
  if (!_gOvlMiniCv) return;
  const ctx = _gOvlMiniCv.getContext('2d');
  const mr = _gOvlMini.getBoundingClientRect();
  ctx.clearRect(0, 0, mr.width, mr.height);
  if (!gNodes.length) return;
  let minX = Infinity, minY = Infinity, maxX = -Infinity, maxY = -Infinity;
  for (const n of gNodes) {
    minX = Math.min(minX, n.x); minY = Math.min(minY, n.y);
    maxX = Math.max(maxX, n.x + NODE_W); maxY = Math.max(maxY, n.y + NODE_H);
  }
  // include current viewport so the rect is always visible
  const vx0 = -gPanX / gCanvasScale, vy0 = -gPanY / gCanvasScale;
  const vx1 = (wrapRect.width - gPanX) / gCanvasScale, vy1 = (wrapRect.height - gPanY) / gCanvasScale;
  minX = Math.min(minX, vx0); minY = Math.min(minY, vy0);
  maxX = Math.max(maxX, vx1); maxY = Math.max(maxY, vy1);
  const pad = 6, spanX = Math.max(1, maxX - minX), spanY = Math.max(1, maxY - minY);
  const sc = Math.min((mr.width - 2 * pad) / spanX, (mr.height - 2 * pad) / spanY);
  const mx = x => pad + (x - minX) * sc, my = y => pad + (y - minY) * sc;
  _gOvlMini._map = { minX, minY, sc, pad }; // for click→pan
  if (!_gOvlPal) _gOvlReadPalette();
  const maxMs = _gOvlMaxMs(), live = _gOvlTele.running && (performance.now() - _gOvlTele.ts < 1500);
  // Node dots — category hue; active nodes brighten with cook time.
  for (const n of gNodes) {
    const load = live ? Math.min(1, (_gOvlTele.timings[n.id] || 0) / maxMs) : 0;
    const col = load > 0.02 ? _gOvlPal.cat[_ovlCat(n)] : _gOvlPal.muted;
    ctx.fillStyle = _ovlRgba(col, load > 0.02 ? 0.55 + 0.4 * load : 0.4);
    const w = Math.max(2.5, NODE_W * sc), h = Math.max(2, NODE_H * sc);
    _ovlRoundRect(ctx, mx(n.x), my(n.y), w, h, 1.5); ctx.fill();
  }
  // Thin viewport rectangle.
  ctx.strokeStyle = _ovlRgba(_gOvlPal.text, 0.5); ctx.lineWidth = 1;
  ctx.strokeRect(mx(vx0) + 0.5, my(vy0) + 0.5, (vx1 - vx0) * sc, (vy1 - vy0) * sc);
}

function _gOvlTick() {
  if (!gOverlayEnabled) { _gOvlRaf = null; return; }
  _gOvlDraw();
  const live = _gOvlTele.running && (performance.now() - _gOvlTele.ts < 1500);
  if (live) { _gOvlRaf = requestAnimationFrame(_gOvlTick); }
  else { _gOvlRaf = null; _gOvlDraw(); } // one calm static frame, then pause (saves fps)
}
function _gOvlStart() { if (!_gOvlRaf && gOverlayEnabled) _gOvlRaf = requestAnimationFrame(_gOvlTick); }

// Keep the overlay aligned when the user pans/zooms or moves nodes (idle redraw).
function gOverlayRefresh() { if (gOverlayEnabled) { _gOvlResize(); _gOvlDraw(); } }

function gOverlayToggle(force) {
  gOverlayEnabled = force !== undefined ? force : !gOverlayEnabled;
  try { localStorage.setItem('graph-overlay', gOverlayEnabled ? '1' : '0'); } catch {}
  _gOvlCanvas?.classList.toggle('on', gOverlayEnabled);
  if (_gOvlMini) _gOvlMini.style.display = gOverlayEnabled ? '' : 'none';
  const _lg = document.getElementById('graph-legend');
  if (_lg) {
    _lg.style.display = gOverlayEnabled ? '' : 'none';
    // Small screens: start collapsed so the legend never buries the graph.
    if (gOverlayEnabled) _lg.classList.toggle('collapsed', window.innerWidth <= 760);
  }
  for (const id of ['graph-overlay-btn', 'graph-overlay-btn-desk']) {
    document.getElementById(id)?.classList.toggle('active', gOverlayEnabled);
  }
  if (gOverlayEnabled) { _gOvlResize(); _gOvlDraw(); _gOvlStart(); }
  else if (_gOvlRaf) { cancelAnimationFrame(_gOvlRaf); _gOvlRaf = null; }
}
document.getElementById('graph-overlay-btn')?.addEventListener('click', () => gOverlayToggle());
document.getElementById('graph-overlay-btn-desk')?.addEventListener('click', () => gOverlayToggle());
window.addEventListener('resize', () => gOverlayRefresh());

// Legend collapse — its own widget; clicking the header folds it to the title.
(function () {
  const lg = document.getElementById('graph-legend');
  const head = lg && lg.querySelector('.glg-head');
  if (head) head.addEventListener('click', e => { e.stopPropagation(); lg.classList.toggle('collapsed'); });
})();

// Minimap interaction — its OWN widget (pointer-events:auto); isolated from the
// graph canvas so it never interferes with node drag / wiring / context menus.
(function () {
  if (!_gOvlMini) return;
  let dragging = false;
  const panTo = (ev) => {
    const map = _gOvlMini._map; if (!map) return;
    const mr = _gOvlMini.getBoundingClientRect();
    const lx = map.minX + (ev.clientX - mr.left - map.pad) / map.sc;
    const ly = map.minY + (ev.clientY - mr.top - map.pad) / map.sc;
    const wr = gCanvasWrap.getBoundingClientRect();
    gPanX = wr.width / 2 - lx * gCanvasScale;
    gPanY = wr.height / 2 - ly * gCanvasScale;
    gApplyPan(); gRedrawEdges(); gOverlayRefresh();
  };
  _gOvlMini.addEventListener('pointerdown', e => { e.stopPropagation(); dragging = true; _gOvlMini.setPointerCapture(e.pointerId); panTo(e); });
  _gOvlMini.addEventListener('pointermove', e => { if (dragging) { e.stopPropagation(); panTo(e); } });
  _gOvlMini.addEventListener('pointerup',   e => { dragging = false; try { _gOvlMini.releasePointerCapture(e.pointerId); } catch {} });
  _gOvlMini.addEventListener('contextmenu', e => e.preventDefault());
})();

// Exposed so the (earlier-defined) pan / edge-redraw code can keep the overlay
// aligned without any temporal-dead-zone risk.
window.gOverlayRefresh = gOverlayRefresh;

// Restore saved state (default off → nothing changes unless the user opts in).
try { if (localStorage.getItem('graph-overlay') === '1') gOverlayToggle(true); } catch {}

// ── Polling ────────────────────────────────────────────────────
async function diagPoll() {
  if (diagPaused) return;
  try {
    const r = await fetch('/api/graph/diagnostics');
    if (!r.ok) return;
    const d = await r.json();
    diagRender(d);
  } catch (_) {}
}

function diagEnsurePolling() {
  diagBindDom();
  if (diagPolling) return;
  diagPolling = true;
  const tick = () => {
    diagPoll();
    diagTimer = setTimeout(tick, 400);
  };
  tick();
}

// ── Controls ───────────────────────────────────────────────────
document.addEventListener('DOMContentLoaded', () => {
  const pauseBtn  = document.getElementById('diag-pause-btn');
  const clearBtn  = document.getElementById('diag-clear-btn');
  const winSelect = document.getElementById('diag-window-select');

  pauseBtn.addEventListener('click', () => {
    diagPaused = !diagPaused;
    pauseBtn.textContent = diagPaused ? 'Resume' : 'Pause';
    pauseBtn.classList.toggle('active', diagPaused);
  });

  clearBtn.addEventListener('click', () => {
    diagSeries.fps.length = 0;
    diagSeries.ms.length  = 0;
    diagBindDom();
    if (_elNodeBars) _elNodeBars.innerHTML = '';
    diagEl('dkv-fps', '–'); diagEl('dkv-ms', '–');
    diagEl('dkv-avg', '–'); diagEl('dkv-minmax', '–');
    if (_elFpsChart) diagDrawLine(_elFpsChart, [], DIAG_COLORS.fps);
    if (_elMsChart)  diagDrawLine(_elMsChart,  [], DIAG_COLORS.ms);
  });

  winSelect.addEventListener('change', () => {
    diagWindow = parseInt(winSelect.value, 10);
  });
});

// Resize charts when window resizes
window.addEventListener('resize', () => {
  if (!_diagDomReady) return;
  if (diagSeries.fps.length) diagDrawLine(_elFpsChart, diagSeries.fps, DIAG_COLORS.fps, { min: 0 });
  if (diagSeries.ms.length)  diagDrawLine(_elMsChart,  diagSeries.ms,  DIAG_COLORS.ms,  { min: 0 });
});

// Expose for tab-switch handler, WS feed, and external callers (e.g. Playwright)
window.diagEnsurePolling = diagEnsurePolling;
window.diagRender        = diagRender;
window.diagIsPaused      = () => diagPaused;

})(); // end IIFE
