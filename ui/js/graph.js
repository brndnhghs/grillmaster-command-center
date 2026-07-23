// ════════════════════════════════════════════════════════════════
// NODE GRAPH
// ════════════════════════════════════════════════════════════════

// ── Tab switching ──────────────────────────────────────────────
document.querySelectorAll('.tab-btn').forEach(btn => {
  btn.addEventListener('click', () => {
    document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
    btn.classList.add('active');
    const tab = btn.dataset.tab;
    document.getElementById('tab-methods').style.display = tab === 'methods' ? 'flex' : 'none';
    document.getElementById('tab-graph').style.display   = tab === 'graph'   ? 'flex' : 'none';
    document.getElementById('tab-diag').style.display    = tab === 'diag'    ? 'flex' : 'none';
    if (tab === 'graph') gLoadMethodPalette();
    if (tab === 'diag') { diagEnsurePolling(); gDiagFlowUpdate({}); }
  });
});

// ── Graph state ────────────────────────────────────────────────
let gNodeDefs       = {};
let gNodes          = [];
let gEdges          = [];
let gSelectedNode   = null;
let gSelectedEdge   = null;
let gSelectedNodes  = new Set();  // multi-select set
let gEdgeCounter    = 0;
let gPaletteLoaded  = false;
let gPalettes       = [];  // palette key list from /api/palettes
let gPendingEdge    = null;
let gPanX = 0, gPanY = 0;
let gCanvasScale = 1.0;
let gCanvasW = 768, gCanvasH = 512;
let gPanTouch = null, gPinchState = null, gMousePan = null, gTrimDragging = false;
let gAutoGen = false;
let gAutoGenJobId = null;
let gAutoGenAbort = null;  // AbortController for cancelling in-flight auto-gen
// nodeId → Set<portName> of input ports that currently have an incoming wire
let gConnectedPorts = new Map();

// ── Timeline clips (layered video compositor) ──────────────────
let tlClips = [];  // {id, name, seqName, startFrame, endFrame, lane, color, nodeId, paramKeyframes}
let tlClipIdCounter = 0;
let tlDragClip = null;  // {clip, lane, startX}

function getEventPos(e) {
  if (e.touches && e.touches.length) return { x: e.touches[0].clientX, y: e.touches[0].clientY };
  if (e.changedTouches && e.changedTouches.length) return { x: e.changedTouches[0].clientX, y: e.changedTouches[0].clientY };
  return { x: e.clientX, y: e.clientY };
}
const GRID = 24; // px — matches background dot spacing
function gApplyPan() {
  gNodesEl.style.transform = `translate(${gPanX}px,${gPanY}px) scale(${gCanvasScale})`;
  gNodesEl.style.transformOrigin = '0 0';
  const gs = GRID * gCanvasScale;
  gCanvasWrap.style.backgroundSize = `${gs}px ${gs}px`;
  gCanvasWrap.style.backgroundPosition = `${gPanX % gs}px ${gPanY % gs}px`;
  if (window.gOverlayRefresh) window.gOverlayRefresh(); // keep FX overlay aligned
}

// Short display labels for port names
function gPortLabel(name) {
  const m = { image_in: 'img', image: 'img', luminance: 'lum', field: 'fld', particles: 'ptcl' };
  return m[name] || name;
}

// Wire compatibility: IMAGE←FIELD is allowed; otherwise types must match
function gPortsCompatible(srcDot, dstDot) {
  if (!srcDot || !dstDot) return true;
  const st = srcDot.dataset.ptype, dt = dstDot.dataset.ptype;
  if (st === 'any' || dt === 'any') return true;
  if (st === dt) return true;
  if (st === 'field' && dt === 'image') return true;
  return false;
}

// Rebuild gConnectedPorts from current gEdges
function gUpdateConnectedPorts() {
  gConnectedPorts.clear();
  for (const edge of gEdges) {
    if (!gConnectedPorts.has(edge.dst_node)) gConnectedPorts.set(edge.dst_node, new Set());
    gConnectedPorts.get(edge.dst_node).add(edge.dst_port);
  }
}

// Grey-out / re-enable param inputs whose ports have active wires
function gRefreshParamOverrides(nodeId) {
  const node = gNodes.find(n => n.id === nodeId);
  if (!node) return;
  const def = gNodeDefs[node.method_id];
  if (!def) return;
  const wired = gConnectedPorts.get(nodeId) || new Set();
  const paramPorts = new Set(def.param_ports || []);
  gParamsForm.querySelectorAll('.param-row').forEach(row => {
    const ctrl = row.querySelector('.param-ctrl');
    if (!ctrl) return;
    const key = ctrl.id.replace('p_', '');
    if (!paramPorts.has(key)) return;
    const isWired = wired.has(key);
    ctrl.disabled = isWired;
    row.classList.toggle('overridden', isWired);
  });
}

// ── DOM refs ───────────────────────────────────────────────────
const gNodesEl      = document.getElementById('graph-nodes');
const gEdgesEl      = document.getElementById('graph-edges');
const gPendingEl    = document.getElementById('pending-edge');
const gCtxMenu      = document.getElementById('graph-ctx-menu');
const gWireTooltip  = document.getElementById('wire-tooltip');
const gNodeTooltip  = document.getElementById('node-tooltip');
let   gLastGraphJobId = null;

// ── Node error modal ─────────────────────────────────────
const nemModal   = document.getElementById('node-error-modal');
const nemName    = document.getElementById('nem-node-name');
const nemText    = document.getElementById('nem-error-text');
const nemClose   = document.getElementById('nem-close-btn');
nemClose.addEventListener('click', () => nemModal.classList.remove('open'));
nemModal.addEventListener('click', e => { if (e.target === nemModal) nemModal.classList.remove('open'); });
document.addEventListener('keydown', e => { if (e.key === 'Escape') nemModal.classList.remove('open'); });

// Click on a node header with error opens the error modal
gNodesEl.addEventListener('click', e => {
  const header = e.target.closest('.gnode-header');
  const nodeEl = header?.closest('.gnode');
  if (!nodeEl || !nodeEl.classList.contains('node-error') || !nodeEl.dataset.errorMsg) return;
  const nodeId = nodeEl.id.replace('gnode-', '');
  const node = gNodes.find(n => n.id === nodeId);
  nemName.textContent = node ? (node.name || node.method_id) : nodeId;
  nemText.textContent = nodeEl.dataset.errorMsg;
  nemModal.classList.add('open');
});

// ── Node header tooltip (description / error) ──────────────────
gNodesEl.addEventListener('mouseover', e => {
  const header = e.target.closest('.gnode-header');
  const nodeEl = header?.closest('.gnode');
  if (!nodeEl) { gNodeTooltip.style.display = 'none'; return; }
  let text = null;
  if (nodeEl.classList.contains('node-error') && nodeEl.dataset.errorMsg) {
    text = nodeEl.dataset.errorMsg.split('\n').slice(0, 3).join('\n');
  } else {
    const nodeId = nodeEl.id.replace('gnode-', '');
    const node = gNodes.find(n => n.id === nodeId);
    const def = node && gNodeDefs[node.method_id];
    if (def?.description) text = def.description;
  }
  if (text) {
    gNodeTooltip.textContent = text;
    gNodeTooltip.style.left = (e.clientX + 14) + 'px';
    gNodeTooltip.style.top  = (e.clientY - 8)  + 'px';
    gNodeTooltip.style.display = '';
  } else {
    gNodeTooltip.style.display = 'none';
  }
});
gNodesEl.addEventListener('mousemove', e => {
  if (gNodeTooltip.style.display !== 'none') {
    gNodeTooltip.style.left = (e.clientX + 14) + 'px';
    gNodeTooltip.style.top  = (e.clientY - 8)  + 'px';
  }
});
gNodesEl.addEventListener('mouseleave', () => { gNodeTooltip.style.display = 'none'; });
const gCanvasWrap   = document.getElementById('graph-canvas-wrap');
const gCanvasCol    = document.getElementById('graph-canvas-col');
// Workspace wraps the canvas + splitter + 3D dock; it, not the canvas, is the
// direct child of the column, so layout code must position against it.
const gWorkspace    = document.getElementById('graph-workspace');
const gVpdStage     = document.getElementById('vpd-stage');
const gOutputStrip  = document.getElementById('graph-output-strip');
const gParamsForm   = document.getElementById('graph-params-form');
const gParamsEmpty  = document.getElementById('graph-params-empty');
const gParamsPanel  = document.getElementById('graph-params-panel');
const gParamsHdr    = document.getElementById('gph-title');
const gPreviewPanel = document.getElementById('graph-preview-panel');
const gPreviewLatest= document.getElementById('graph-preview-latest');
const gPreviewLabel = document.getElementById('graph-preview-label');
const gPreviewToggle= document.getElementById('graph-preview-toggle');
const gBackdrop     = document.getElementById('graph-backdrop');
const gFsOverlay    = document.getElementById('graph-img-fullscreen');
const gFsImg        = document.getElementById('graph-fs-img');
const gFsVideo      = document.getElementById('graph-fs-video');
const gFsCanvas     = document.getElementById('graph-fs-canvas');
const gMainPreview  = document.getElementById('graph-main-preview');
const gGraphSidebar = document.getElementById('graph-sidebar');

// Mobile + desktop control pairs
const gRunBtn       = document.getElementById('graph-run-btn');
const gRunBtnDesk   = document.getElementById('graph-run-btn-desk');
const gClearBtn     = document.getElementById('graph-clear-btn');
const gClearBtnDesk = document.getElementById('graph-clear-btn-desk');
const gStatusEl     = document.getElementById('graph-status');
const gStatusDeskEl = document.getElementById('graph-status-desk');
const gPaletteBtn         = document.getElementById('graph-palette-btn');
const gPaletteClose       = document.getElementById('graph-palette-close');
const gParamsClose        = document.getElementById('graph-params-close');
const gGraphMobileBottomBar = document.getElementById('graph-mobile-bottom-bar');

// ── Shared helpers ─────────────────────────────────────────────
function gGetFrames() {
  const start = parseInt(document.getElementById('tl-start')?.value) || 0;
  const end   = parseInt(document.getElementById('tl-end')?.value)   || 24;
  return Math.max(1, end - start + 1);
}
function gUpdateFrameCount() {
  const el = document.getElementById('graph-frame-count');
  if (el) el.textContent = gGetFrames() + ' frames';
}
function gSetStatus(msg) {
  gStatusEl.textContent = msg;
  gStatusDeskEl.textContent = msg;
}
function gSetRunDisabled(v) {
  gRunBtn.disabled = v;
  gRunBtnDesk.disabled = v;
}

// ── Palette overlay (mobile) ───────────────────────────────────
function gOpenPalette() {
  gGraphSidebar.classList.add('gsidebar-open');
  gBackdrop.classList.add('visible');
  if (!gPaletteLoaded) gLoadMethodPalette();
}
function gClosePalette() {
  gGraphSidebar.classList.remove('gsidebar-open');
  if (!gParamsPanel.classList.contains('gparams-open')) gBackdrop.classList.remove('visible');
}
gPaletteBtn.addEventListener('click', gOpenPalette);
gPaletteClose.addEventListener('click', gClosePalette);

const gSidebarCollapseBtn = document.getElementById('graph-sidebar-collapse');
const gSidebarFloatBtn = document.getElementById('graph-sidebar-toggle-float');

function gSetSidebarCollapsed(collapsed) {
  gGraphSidebar.classList.toggle('gsidebar-collapsed', collapsed);
  gSidebarCollapseBtn.textContent = collapsed ? '›' : '‹';
  gSidebarCollapseBtn.title = collapsed ? 'Expand palette' : 'Collapse palette';
  gSidebarFloatBtn.style.display = collapsed ? 'block' : 'none';
}

gSidebarCollapseBtn.addEventListener('click', () => gSetSidebarCollapsed(!gGraphSidebar.classList.contains('gsidebar-collapsed')));
gSidebarFloatBtn.addEventListener('click', () => gSetSidebarCollapsed(false));

gSetSidebarCollapsed(true);

// ── Params bottom sheet (mobile) ───────────────────────────────
function gParamsSheetOpen() {
  gParamsPanel.classList.add('gparams-open');
  gBackdrop.classList.add('visible');
}
function gParamsSheetClose() {
  gParamsPanel.classList.remove('gparams-open');
  if (!gGraphSidebar.classList.contains('gsidebar-open')) gBackdrop.classList.remove('visible');
}
gParamsClose.addEventListener('click', () => {
  gParamsSheetClose();
  gShowNodeParams(null);
  gSelectedNode = null;
  gNodesEl.querySelectorAll('.gnode').forEach(el => el.classList.remove('selected'));
});

// Backdrop: close whichever overlay is open
gBackdrop.addEventListener('click', () => {
  gClosePalette();
  gParamsSheetClose();
});

// ── Preview panel ──────────────────────────────────────────────
let gPreviewExpanded = false;
try { gPreviewExpanded = localStorage.getItem('graph-preview-expanded') === 'true'; } catch {}

// Single live-preview img that gets its src swapped in-place each frame
let gLivePreviewImg = null;

function gGraphFrameUpdate(b64) {
  // Don't touch the main preview during rendering — timeline handles all display.
  // Just update the small preview-latest thumbnail.
  const src = 'data:image/jpeg;base64,' + b64;
  gPreviewLatest.src = src; gPreviewLatest.style.display = '';
}

function gGraphDoneSwap(type, relPath, seqName) {
  gPreviewLatest.style.display = 'none';
  gPreviewLabel.textContent = 'Timeline';

  if (seqName) {
    tlName.value = seqName;
    // Create a clip on the timeline
    const start = parseInt(tlStart.value) || 0;
    const end   = parseInt(tlEnd.value)   || 24;
    // Collect paramKeyframes from all nodes
    const allPkf = {};
    for (const node of gNodes) {
      if (node.paramKeyframes) {
        for (const [k, v] of Object.entries(node.paramKeyframes)) {
          if (!allPkf[k]) allPkf[k] = [];
          allPkf[k].push(...v);
        }
      }
    }
    const clip = tlAddClip(seqName, seqName, start, end, null, allPkf);
    tlSelectClip(clip.id);
    // Load frame 0
    tlLoadFrame(parseInt(tlStart.value) || 0);
  }
}

// Return {seqName, localFrame} for whichever clip covers `timelineFrame`,
// or null if no clip is there. Lowest lane wins when clips overlap.
function tlClipAtFrame(timelineFrame) {
  const candidates = tlClips
    .filter(c => timelineFrame >= c.startFrame && timelineFrame <= c.endFrame)
    .sort((a, b) => a.lane - b.lane);
  if (!candidates.length) return null;
  const clip = candidates[0];
  const srcLen = clip.srcLength || (clip.endFrame - clip.startFrame + 1);
  const trimIn = clip.trimIn || 0;
  const usableLen = Math.max(1, srcLen - trimIn);
  const raw = timelineFrame - clip.startFrame;
  const localFrame = trimIn + (((raw % usableLen) + usableLen) % usableLen);
  return { seqName: clip.seqName, localFrame, srcOffset: clip.srcOffset || 0 };
}

function tlClearPreview() {
  if (gLivePreviewImg) {
    gLivePreviewImg.style.opacity = '0';
  }
}

const _tlFrameCache = new Map();
// Fetches currently on the wire, keyed like the cache. A frame already in
// flight is awaited rather than requested a second time: the read-ahead below
// only lands in the cache once it *resolves*, so at playback speed the playhead
// routinely reaches frame N+1 while its read-ahead is still open, and the old
// code then issued a duplicate request for it — doubling load exactly when
// playback was already behind (measured 40 requests in a second that needed 24).
const _tlInflight = new Map(); // key → Promise<Blob|null>
// Bumped on every load. Responses arrive out of order, so one that comes back
// against a stale token belongs to a frame the playhead has already passed and
// must not be painted over a newer one. Pausing deliberately does *not* bump it:
// the frame being loaded when the user hits pause is the one they want to see.
let _tlFrameSeq = 0;

// Resolves to a Blob, or null if the frame is missing.
function _tlFetchFrame(seqName, fileFrame) {
  const key = `${seqName}:${fileFrame}`;
  const pending = _tlInflight.get(key);
  if (pending) return pending;
  const p = fetch(`/api/sequences/${encodeURIComponent(seqName)}/${fileFrame}`)
    .then(r => (r.ok ? r.blob() : null))
    .catch(() => null)
    .finally(() => { if (_tlInflight.get(key) === p) _tlInflight.delete(key); });
  _tlInflight.set(key, p);
  return p;
}

function _tlDisplayBlob(blob) {
  // Don't clobber the live canvas
  if (gLiveMode) return;
  const url = URL.createObjectURL(blob);
  if (!gLivePreviewImg) {
    gMainPreview.innerHTML = '';
    gLivePreviewImg = document.createElement('img');
    gLivePreviewImg.addEventListener('click', () => gOpenFullscreen());
    gMainPreview.appendChild(gLivePreviewImg);
    gMainPreview.classList.add('active');
    gPreviewShow();
  }
  if (gLivePreviewImg._tlUrl) URL.revokeObjectURL(gLivePreviewImg._tlUrl);
  gLivePreviewImg._tlUrl = url;
  gLivePreviewImg.src = url;
  gLivePreviewImg.style.display = '';
  gLivePreviewImg.style.opacity = '1';
}

function tlLoadFrame(timelineFrame) {
  const hit = tlClipAtFrame(timelineFrame);
  if (!hit) { tlClearPreview(); return; }
  const { seqName, localFrame, srcOffset } = hit;
  const fileFrame = localFrame + (srcOffset || 0);
  const cacheKey = `${seqName}:${fileFrame}`;

  const token = ++_tlFrameSeq;

  // Check cache first
  const cached = _tlFrameCache.get(cacheKey);
  if (cached) {
    _tlDisplayBlob(cached);
    _tlFrameCache.delete(cacheKey);
  } else {
    _tlFetchFrame(seqName, fileFrame).then(blob => {
      if (token !== _tlFrameSeq) return;  // the playhead has already moved on
      if (blob) { _tlDisplayBlob(blob); return; }
      if (!gLivePreviewImg) {
        gMainPreview.innerHTML = '<div style="padding:20px;color:var(--muted);font-size:12px;text-align:center;">No frame rendered yet</div>';
        gMainPreview.classList.add('active');
        gPreviewShow();
      }
    });
  }

  // Pre-fetch next frame
  const nextFileFrame = fileFrame + 1;
  const nextKey = `${seqName}:${nextFileFrame}`;
  if (!_tlFrameCache.has(nextKey)) {
    _tlFetchFrame(seqName, nextFileFrame)
      .then(blob => { if (blob) _tlFrameCache.set(nextKey, blob); });
  }

  // Clean old cache entries (keep max 5)
  while (_tlFrameCache.size > 5) {
    const firstKey = _tlFrameCache.keys().next().value;
    _tlFrameCache.delete(firstKey);
  }
}

function gPreviewShow() {
  gPreviewPanel.classList.add('pv-visible');
  if (isMobile()) {
    gPreviewPanel.classList.toggle('pv-expanded',  gPreviewExpanded);
    gPreviewPanel.classList.toggle('pv-collapsed', !gPreviewExpanded);
    gPreviewToggle.textContent = gPreviewExpanded ? '▼ collapse' : '▲ expand';
    gUpdateCanvasPad();
  }
}
function gPreviewHide() {
  gPreviewPanel.classList.remove('pv-visible', 'pv-expanded', 'pv-collapsed');
  gCanvasCol.classList.remove('has-preview-expanded', 'has-preview-collapsed');
}
function gUpdateCanvasPad() {
  if (!isMobile()) return;
  if (gPreviewPanel.classList.contains('mob-top')) {
    gCanvasCol.classList.remove('has-preview-expanded', 'has-preview-collapsed');
    return;
  }
  gCanvasCol.classList.toggle('has-preview-expanded',  gPreviewExpanded);
  gCanvasCol.classList.toggle('has-preview-collapsed', !gPreviewExpanded);
}

// ── Mobile graph layout: preview top, run bar bottom ──────────────────
function setGraphMobileLayout(mob) {
  if (mob) {
    gCanvasCol.insertBefore(gPreviewPanel, gWorkspace);
    gPreviewPanel.classList.add('mob-top');
    gGraphMobileBottomBar.append(gRunBtn, gClearBtn);
    gSidebarFloatBtn.style.display = 'none';
  } else {
    document.getElementById('graph-shell').after(gPreviewPanel);
    gPreviewPanel.classList.remove('mob-top');
    const paletteBtn = document.getElementById('graph-palette-btn');
    paletteBtn.after(gRunBtn, gClearBtn);
    if (gGraphSidebar.classList.contains('gsidebar-collapsed')) {
      gSidebarFloatBtn.style.display = 'block';
    }
  }
  gUpdateCanvasPad();
}
const mqMobileGraph = window.matchMedia('(max-width: 768px)');
setGraphMobileLayout(mqMobileGraph.matches);
mqMobileGraph.addEventListener('change', e => setGraphMobileLayout(e.matches));

document.getElementById('graph-preview-handle').addEventListener('click', () => {
  gPreviewExpanded = !gPreviewExpanded;
  try { localStorage.setItem('graph-preview-expanded', gPreviewExpanded); } catch {}
  gPreviewPanel.classList.toggle('pv-expanded',  gPreviewExpanded);
  gPreviewPanel.classList.toggle('pv-collapsed', !gPreviewExpanded);
  gPreviewToggle.textContent = gPreviewExpanded ? '▼ collapse' : '▲ expand';
  gUpdateCanvasPad();
});
gPreviewToggle.addEventListener('click', e => e.stopPropagation());

// ── Fullscreen ─────────────────────────────────────────────────
gFsOverlay.addEventListener('click', () => {
  gFsOverlay.classList.remove('visible');
  gFsVideo.pause(); gFsVideo.src = '';
  _gStopMirror(gFsCanvas);            // stop any live canvas mirror loop
});

// ── Load port types ────────────────────────────────────────────
async function gLoadPortTypes() {
  const res = await fetch('/api/port-types');
  window.gPortTypes = await res.json();
}

// ── Load method palette ────────────────────────────────────────
// ── Client-side (browser-GPU) node defs ─────────────────────────────────────
// Only nodes the server does not publish at all live here. The 3D family
// (__geometry__ … __scene_render__, __gltf__, __usd__) is defined once, in
// image_pipeline/core/threejs_nodes.py, and arrives via /api/node-defs — a
// second copy here would shadow the server's richer params (postfx, tone_map,
// env_preset) depending on which fetch resolved last.
const GCLIENT_NODE_DEFS = {
  '__p5sketch__': {
    method_id: '__p5sketch__',
    name: 'p5 Sketch',
    category: 'client_3d',
    clientExec: true,
    tags: ['p5', 'client', 'webgl', 'creative', 'fast'],
    inputs: { image_in: 'image' },
    outputs: { image: 'image', luminance: 'field' },
    param_ports: [],
    description: 'Client-side p5.js sketch (instance mode). Generator or filter.',
    version: 1, deprecated: false, start_frame: 0, end_frame: 0, prebake: 0,
    params: {
      sketch_code: {
        description: 'p5.js sketch — setup(p,g) & draw(p,g). g={width,height,time,frame,p1..p4,input(p5.Image in filter mode),WEBGL,P2D}',
        multiline: true,
        default:
`// p5.js instance-mode sketch. g carries animated params + input frame.
// Works as a GENERATOR, and as a FILTER when an image is wired to image_in
// (then g.input is the upstream frame as a canvas you can texture()/image()).
function setup(p, g) {
  p.createCanvas(g.width, g.height, g.WEBGL);
  p.noStroke();
}
function draw(p, g) {
  if (g.input) {                         // FILTER mode: show upstream frame
    p.push(); p.texture(g.input);
    p.translate(0, 0, -1); p.plane(g.width, g.height); p.pop();
  } else {
    p.background(8, 10, 20);             // GENERATOR mode
  }
  p.ambientLight(50);
  p.pointLight(255, 255, 255, 220, -220, 340);
  const rings = 6;
  for (let i = 0; i < rings; i++) {
    p.push();
    const a = g.time * (0.6 + 0.14 * i) + i;
    p.rotateY(a); p.rotateX(a * 0.5);
    p.translate(150, 0, 0);
    p.ambientMaterial(60 + 190 * g.p1, 120, 255 - 160 * g.p1);
    p.torus(34 + 26 * g.p2, 11);
    p.pop();
  }
}`,
      },
      p1:         { description: 'param 1 → g.p1', min: 0, max: 1, default: 0.5 },
      p2:         { description: 'param 2 → g.p2', min: 0, max: 1, default: 0.5 },
      p3:         { description: 'param 3 → g.p3', min: 0, max: 1, default: 0.5 },
      p4:         { description: 'param 4 → g.p4', min: 0, max: 1, default: 0.5 },
      time_scale: { description: 'animation speed → g.time', min: 0, max: 5, default: 1 },
    },
  },
};

// Lazy-loaded ES module handle (ui/js/client3d.js). Only imported when a client
// node is actually present, so three.js never bloats the initial page load.
let _gClient3D = null;
async function gClient3D() {
  if (!_gClient3D) _gClient3D = await import('/ui/js/client3d.js');
  return _gClient3D;
}
// Node ids the browser spine executes itself — mirrors CLIENT_RENDER_IDS in
// ui/js/client3d.js, minus the ids that also have a real server method
// (__custom_shader__, __blender_render__), which must keep routing to the
// server. Kept as a plain list so "is this graph client-rendered?" can be
// answered before it is worth importing three.js at all.
// The server's defs carry no clientExec flag of their own, so it is stamped on
// at fetch time — see gFetchNodeDefs.
const GCLIENT_EXEC_IDS = new Set([
  '__scene3d__', '__p5sketch__',
  '__geometry__', '__material__', '__mesh3d__', '__group3d__',
  '__light3d__', '__camera3d__', '__scene_render__', '__gltf__', '__usd__',
]);

// True when the current graph contains any client-rendered node, anywhere —
// including nodes parked off to the side that feed nothing. Use this only for
// cosmetic questions ("might this graph want 3D affordances?"). It is the wrong
// question for routing: see gGraphRunsOnClient.
function gGraphHasClientNode() {
  return gNodes.some(n => gNodeDefs[n.method_id]?.clientExec);
}

// ── Render routing: which engine owns this graph? ───────────────
// The two engines are not interchangeable — the browser spine has no method for
// most server nodes, and the server has none for the 3D family. So the choice
// has to follow the node that actually produces the output, not whatever else
// happens to be lying on the canvas.

/**
 * The node whose image is the graph's output. Mirrors _terminalId() in
 * client3d.js and _find_terminal() in core/graph.py: an explicit render flag
 * wins, otherwise the last sink (a node with no outgoing non-feedback edge).
 */
function gTerminalNode() {
  const flagged = gNodes.filter(n => n.render);
  if (flagged.length) return flagged[flagged.length - 1];
  const hasOut = new Set(gEdges.filter(e => !e.feedback).map(e => e.src_node));
  const sinks = gNodes.filter(n => !hasOut.has(n.id));
  // A dangling Geometry/Light must not outrank a Scene Render for the terminal.
  const imageSink = sinks.find(n => (gNodeDefs[n.method_id]?.outputs || {}).image);
  return imageSink || sinks[sinks.length - 1] || gNodes[gNodes.length - 1] || null;
}

/**
 * Ids of `nodeId` and everything upstream of it — the nodes that must actually
 * run to produce its output. Feedback edges are followed too: they carry last
 * frame's value, so their source still has to be cooked.
 */
function gAncestorIds(nodeId) {
  const seen = new Set();
  const stack = [nodeId];
  while (stack.length) {
    const id = stack.pop();
    if (!id || seen.has(id)) continue;
    seen.add(id);
    for (const e of gEdges) if (e.dst_node === id) stack.push(e.src_node);
  }
  return seen;
}

/**
 * True when the output actually depends on a client-rendered node.
 *
 * Deliberately narrower than gGraphHasClientNode: a 3D node dropped on the
 * canvas but wired to nothing does not make an otherwise-ordinary graph a 3D
 * graph, and routing it to the browser spine used to blank the render — the
 * spine has no method for most server nodes and silently blits black for them.
 */
function gGraphRunsOnClient() {
  const term = gTerminalNode();
  if (!term) return false;
  const live = gAncestorIds(term.id);
  return gNodes.some(n => live.has(n.id) && gNodeDefs[n.method_id]?.clientExec);
}

/**
 * Nodes + edges for a server run, with client-only nodes and their wiring
 * stripped out.
 *
 * The server has no method for the 3D family and aborts the entire job with
 * "Unknown method" the moment one is scheduled — and it prunes only by topo
 * position, not by ancestry, so a 3D node parked off to the side still gets
 * cooked. On this path such a node can never be an ancestor of the terminal
 * (gGraphRunsOnClient would have routed us to the browser instead), so
 * dropping it cannot change the output.
 */
function gServerGraphPayload(frame) {
  const drop = new Set(gNodes.filter(n => gNodeDefs[n.method_id]?.clientExec).map(n => n.id));
  return {
    nodes: gNodes.filter(n => !drop.has(n.id)).map(n => gSerializeNodeForApi(n, frame)),
    edges: gEdges.filter(e => !drop.has(e.src_node) && !drop.has(e.dst_node))
      .map(e => ({ src_node:e.src_node, src_port:e.src_port, dst_node:e.dst_node, dst_port:e.dst_port, feedback:e.feedback })),
  };
}

/**
 * The one way node defs enter the app. Every caller must go through this:
 * a bare `/api/node-defs` fetch yields defs with no clientExec flag, and a
 * 3D graph built on those routes to the server render path, which has no
 * method for `__geometry__` and fails with "Unknown method".
 */
async function gFetchNodeDefs() {
  const defs = await fetch('/api/node-defs').then(r => r.json());
  Object.assign(defs, GCLIENT_NODE_DEFS);  // client-only nodes the server never publishes
  for (const mid of GCLIENT_EXEC_IDS) if (defs[mid]) defs[mid].clientExec = true;
  return defs;
}

async function gLoadMethodPalette({ force = false } = {}) {
  // `force` is for the hot-reload push: without it the gPaletteLoaded guard
  // makes a node-defs-updated event a no-op and edited methods never appear.
  if (gPaletteLoaded && !force) return;
  gPaletteLoaded = true;
  try {
    gNodeDefs = await gFetchNodeDefs();
    gRenderPalette();
  } catch(e) {
    document.getElementById('graph-method-list').innerHTML =
      `<p style="padding:12px;color:var(--err);font-size:12px">Failed: ${e.message}</p>`;
  }
}

function gRenderPalette() {
  const methodGroups = {};
  for (const [mid, def] of Object.entries(gNodeDefs)) {
    (methodGroups[def.category || 'other'] = methodGroups[def.category || 'other'] || []).push(def);
  }
  let gPalState = {};
  try { gPalState = JSON.parse(localStorage.getItem('graph-palette-state') || '{}'); } catch {}
  let html = '';

  // Group presets section at top
  const presets = window._gGroupPresets || [];
  if (presets.length) {
    const isOpen = gPalState['__groups__'] !== undefined ? gPalState['__groups__'] : true;
    html += `<div class="gcat-group"><div class="gcat-header" data-cat="__groups__"><span class="gcat-chevron">${isOpen?'▼':'▶'}</span>⊞ Groups</div><div class="gcat-items"${isOpen?'':' style="display:none"'}>`;
    for (const g of presets) {
      html += `<div class="gmethod-item ggroup-preset-item" data-gname="${escHtml(g.name)}" draggable="true"><div class="gmname">⊞ ${escHtml(g.name)}</div></div>`;
    }
    html += `</div></div>`;
  }

  for (const [cat, items] of Object.entries(methodGroups).sort()) {
    const isOpen = gPalState[cat] !== undefined ? gPalState[cat] : true;
    html += `<div class="gcat-group">
      <div class="gcat-header" data-cat="${escHtml(cat)}">
        <span class="gcat-chevron">${isOpen ? '▼' : '▶'}</span>${escHtml(cat)}
      </div>
      <div class="gcat-items"${isOpen ? '' : ' style="display:none"'}>`;
    for (const def of items) {
      html += `<div class="gmethod-item" data-mid="${escHtml(def.method_id)}" draggable="true">
        <div class="gmid">#${escHtml(def.method_id)}</div>
        <div class="gmname">${escHtml(def.name)}</div>
      </div>`;
    }
    html += `</div></div>`;
  }
  const listEl = document.getElementById('graph-method-list');
  listEl.innerHTML = html;

  listEl.querySelectorAll('.gcat-header').forEach(el => {
    el.addEventListener('click', () => {
      const cat = el.dataset.cat;
      const itemsEl = el.nextElementSibling;
      const chevronEl = el.querySelector('.gcat-chevron');
      const nowOpen = itemsEl.style.display === 'none';
      itemsEl.style.display = nowOpen ? '' : 'none';
      chevronEl.textContent = nowOpen ? '▼' : '▶';
      gPalState[cat] = nowOpen;
      try { localStorage.setItem('graph-palette-state', JSON.stringify(gPalState)); } catch {}
    });
  });

  listEl.querySelectorAll('.gmethod-item:not(.ggroup-preset-item)').forEach(el => {
    el.addEventListener('dragstart', e => e.dataTransfer.setData('method_id', el.dataset.mid));
    el.addEventListener('click', () => {
      if (!isMobile()) return;
      const rect = gCanvasWrap.getBoundingClientRect();
      gAddNode(el.dataset.mid,
        (rect.width  / 2 - gPanX) / gCanvasScale - 80 + (Math.random() * 60 - 30),
        (rect.height / 2 - gPanY) / gCanvasScale - 40 + (Math.random() * 60 - 30));
      gClosePalette();
    });
  });

  listEl.querySelectorAll('.ggroup-preset-item').forEach(el => {
    el.addEventListener('dragstart', e => e.dataTransfer.setData('group_name', el.dataset.gname));
    el.addEventListener('click', async () => {
      const rect = gCanvasWrap.getBoundingClientRect();
      const x = (rect.width / 2 - gPanX) / gCanvasScale - 80;
      const y = (rect.height / 2 - gPanY) / gCanvasScale - 40;
      try {
        const d = await fetch(`/api/groups/${encodeURIComponent(el.dataset.gname)}`).then(r => r.json());
        gAddGroupNode(d, x, y);
        if (isMobile()) gClosePalette();
      } catch(err) { gShowToast('Error: '+err.message, true); }
    });
  });
}

// ── Canvas drop ────────────────────────────────────────────────
gCanvasWrap.addEventListener('dragover', e => e.preventDefault());
gCanvasWrap.addEventListener('drop', async e => {
  e.preventDefault();
  const rect = gCanvasWrap.getBoundingClientRect();
  const cx = (e.clientX - rect.left - gPanX) / gCanvasScale - 80;
  const cy = (e.clientY - rect.top  - gPanY) / gCanvasScale - 40;
  const mid = e.dataTransfer.getData('method_id');
  if (mid && gNodeDefs[mid]) { gAddNode(mid, cx, cy); return; }
  const gname = e.dataTransfer.getData('group_name');
  if (gname) {
    try {
      const d = await fetch(`/api/groups/${encodeURIComponent(gname)}`).then(r => r.json());
      gAddGroupNode(d, cx, cy);
    } catch(err) { gShowToast('Error: '+err.message, true); }
  }
});

// ── Canvas touch: 1-finger pan + 2-finger pinch-to-zoom ───────────────
gCanvasWrap.addEventListener('touchstart', e => {
  if (gTrimDragging) return;
  if (e.target.closest('.gnode') || e.target.classList.contains('gport')) return;
  e.preventDefault();
  if (e.touches.length === 2) {
    gPanTouch = null;
    const dx = e.touches[0].clientX - e.touches[1].clientX;
    const dy = e.touches[0].clientY - e.touches[1].clientY;
    const rect = gCanvasWrap.getBoundingClientRect();
    gPinchState = {
      dist0:  Math.hypot(dx, dy) || 1,
      scale0: gCanvasScale,
      pan0x:  gPanX, pan0y: gPanY,
      midX:   (e.touches[0].clientX + e.touches[1].clientX) / 2 - rect.left,
      midY:   (e.touches[0].clientY + e.touches[1].clientY) / 2 - rect.top,
    };
  } else if (e.touches.length === 1) {
    gPinchState = null;
    gPanTouch = { sx: e.touches[0].clientX - gPanX, sy: e.touches[0].clientY - gPanY };
  }
}, { passive: false });
gCanvasWrap.addEventListener('touchmove', e => {
  e.preventDefault();
  if (e.touches.length === 2 && gPinchState) {
    const dx = e.touches[0].clientX - e.touches[1].clientX;
    const dy = e.touches[0].clientY - e.touches[1].clientY;
    const newScale = Math.min(3.0, Math.max(0.2,
      gPinchState.scale0 * Math.hypot(dx, dy) / gPinchState.dist0));
    const mx = gPinchState.midX, my = gPinchState.midY;
    gCanvasScale = newScale;
    gPanX = mx - (mx - gPinchState.pan0x) * newScale / gPinchState.scale0;
    gPanY = my - (my - gPinchState.pan0y) * newScale / gPinchState.scale0;
    gApplyPan(); gRedrawEdges();
  } else if (e.touches.length === 1 && gPanTouch) {
    gPanX = e.touches[0].clientX - gPanTouch.sx;
    gPanY = e.touches[0].clientY - gPanTouch.sy;
    gApplyPan(); gRedrawEdges();
  }
}, { passive: false });
gCanvasWrap.addEventListener('touchend', () => {
  if (gPanTouch)   { gPanTouch   = null; gSave(); }
  if (gPinchState) { gPinchState = null; gSave(); }
});
gCanvasWrap.addEventListener('touchcancel', () => { gPanTouch = null; gPinchState = null; });

// ── Desktop: drag-to-pan + scroll-to-zoom ─────────────────────────────
gCanvasWrap.addEventListener('mousedown', e => {
  if (gTrimDragging) return;
  if (e.target.closest('.gnode') || e.target.classList.contains('gport')) return;
  if (e.button !== 0 || gPendingEdge) return;
  e.preventDefault();
  gMousePan = { sx: e.clientX - gPanX, sy: e.clientY - gPanY };
});
document.addEventListener('mousemove', e => {
  if (!gMousePan) return;
  gPanX = e.clientX - gMousePan.sx;
  gPanY = e.clientY - gMousePan.sy;
  gApplyPan(); gRedrawEdges();
});
document.addEventListener('mouseup', () => {
  if (gMousePan) { gMousePan = null; gSave(); }
});
gCanvasWrap.addEventListener('wheel', e => {
  e.preventDefault();
  const rect = gCanvasWrap.getBoundingClientRect();
  const mx = e.clientX - rect.left, my = e.clientY - rect.top;
  const newScale = Math.min(3.0, Math.max(0.2, gCanvasScale * (e.deltaY < 0 ? 1.1 : 0.9)));
  gPanX = mx - (mx - gPanX) * newScale / gCanvasScale;
  gPanY = my - (my - gPanY) * newScale / gCanvasScale;
  gCanvasScale = newScale;
  gApplyPan(); gRedrawEdges();
}, { passive: false });

// ── Add a node ─────────────────────────────────────────────────
let _gNodeIdSeq = 0;
function gAddNode(method_id, x, y) {
  const def = gNodeDefs[method_id];
  if (!def) return;
  // Timestamp + monotonic counter so IDs stay unique even when several nodes
  // are added within the same millisecond (e.g. loading a preset / template).
  const id = 'n' + Date.now().toString(36) + (_gNodeIdSeq++).toString(36);
  const node = { id, method_id, params: gDefaultParams(def), x, y, dirty: true };
  gNodes.push(node);
  gRenderNode(node);
  gPushApart(node);
  gPhysicsKick();
  gSave();
  return node;
}

// ── Dispatch render by node type ───────────────────────────────
function gRenderAnyNode(node) {
  if (node.type === 'group') gRenderGroupNode(node);
  else gRenderNode(node);
}

// ── Render a group node div ────────────────────────────────────
function gRenderGroupNode(node) {
  const el = document.createElement('div');
  el.className = 'gnode gnode-group';
  el.id = 'gnode-' + node.id;
  el.style.left = node.x + 'px';
  el.style.top  = node.y + 'px';
  if (node.render) el.classList.add('render-target');

  const header = document.createElement('div');
  header.className = 'gnode-header';
  header.innerHTML = `<button class="gnode-delete" title="Delete">✕</button>
    <span class="gnode-group-icon">⊞</span>
    <span class="gnode-title">${escHtml(node.name || 'Group')}</span>
    <button class="gnode-render${node.render ? ' active' : ''}" title="Set as render target">◎</button>`;
  el.appendChild(header);

  function _mkPort(name, type, dir) {
    const dot = document.createElement('div');
    dot.className = `gport pt-${type}`;
    Object.assign(dot.dataset, { nid: node.id, port: name, dir, ptype: type });
    return dot;
  }
  function _mkLabel(name) {
    const lbl = document.createElement('span');
    lbl.className = 'gport-label'; lbl.textContent = name;
    return lbl;
  }

  // Top row: primary image input (left) + image output (right)
  const imgIns  = (node.exposed_inputs  || []).filter(e => (e.port_type || 'image') === 'image');
  const imgOuts = (node.exposed_outputs || []).filter(e => (e.port_type || 'image') === 'image');
  const topPortsEl = document.createElement('div');
  topPortsEl.className = 'gnode-top-ports';
  { const row = document.createElement('div'); row.className = 'gnode-port-row input';
    if (imgIns[0]) { row.appendChild(_mkPort(imgIns[0].port, 'image', 'input')); row.appendChild(_mkLabel(imgIns[0].port)); }
    topPortsEl.appendChild(row); }
  { const row = document.createElement('div'); row.className = 'gnode-port-row output';
    if (imgOuts[0]) { row.appendChild(_mkLabel(imgOuts[0].port)); row.appendChild(_mkPort(imgOuts[0].port, 'image', 'output')); }
    topPortsEl.appendChild(row); }
  if (topPortsEl.querySelector('.gport')) el.appendChild(topPortsEl);

  // Bottom: non-image ports
  const portsEl = document.createElement('div');
  portsEl.className = 'gnode-ports';
  for (const ei of (node.exposed_inputs || [])) {
    if ((ei.port_type || 'image') === 'image') continue;
    const row = document.createElement('div'); row.className = 'gnode-port-row input';
    row.appendChild(_mkPort(ei.port, ei.port_type || 'scalar', 'input'));
    row.appendChild(_mkLabel(ei.port));
    portsEl.appendChild(row);
  }
  for (const eo of (node.exposed_outputs || [])) {
    if ((eo.port_type || 'image') === 'image') continue;
    const row = document.createElement('div'); row.className = 'gnode-port-row output';
    row.appendChild(_mkLabel(eo.port));
    row.appendChild(_mkPort(eo.port, eo.port_type || 'scalar', 'output'));
    portsEl.appendChild(row);
  }
  if (portsEl.children.length) el.appendChild(portsEl);
  gNodesEl.appendChild(el);

  header.querySelector('.gnode-delete').addEventListener('click', e => {
    e.stopPropagation(); gDeleteNode(node.id);
  });
  header.querySelector('.gnode-render').addEventListener('click', e => {
    e.stopPropagation();
    const wasActive = !!node.render;
    gNodes.forEach(n => { n.render = false; });
    document.querySelectorAll('.gnode-render').forEach(b => b.classList.remove('active'));
    document.querySelectorAll('.gnode.render-target').forEach(n => n.classList.remove('render-target'));
    node.render = !wasActive;
    if (node.render) { e.currentTarget.classList.add('active'); el.classList.add('render-target'); }
    gSave();
  });

  let _tapStart = null;
  el.addEventListener('mousedown', e => {
    if (e.target.classList.contains('gnode-delete') || e.target.classList.contains('gnode-render')) return;
    _tapStart = { x: e.clientX, y: e.clientY, t: Date.now() };
  });
  el.addEventListener('mouseup', e => {
    if (!_tapStart) return;
    const dx = Math.abs(e.clientX - _tapStart.x), dy = Math.abs(e.clientY - _tapStart.y);
    const dt = Date.now() - _tapStart.t;
    _tapStart = null;
    if (e.target.classList.contains('gnode-delete') || e.target.classList.contains('gnode-render')) return;
    if (dx < 8 && dy < 8 && dt < 300) {
      if (e.shiftKey) gToggleMultiSelect(node.id);
      else { gClearMultiSelect(); gSelectNode(node.id); }
    }
  });
  el.addEventListener('dblclick', e => { e.stopPropagation(); gOpenGroupModal(node); });
  el.addEventListener('touchstart', e => {
    const t = e.touches[0];
    _tapStart = { x: t.clientX, y: t.clientY, t: Date.now() };
  }, { passive: true });
  el.addEventListener('touchend', e => {
    if (!_tapStart) return;
    const t = e.changedTouches[0];
    const dx = Math.abs(t.clientX - _tapStart.x), dy = Math.abs(t.clientY - _tapStart.y);
    const dt = Date.now() - _tapStart.t;
    _tapStart = null;
    if (e.target.classList.contains('gnode-delete') || e.target.classList.contains('gnode-render')) return;
    if (dx < 8 && dy < 8 && dt < 300) gSelectNode(node.id);
  });

  gAttachNodeDrag(el, node);

  // Helpers: find best matching port dot, preferring type match
  function _nearestPort(dir, ptype) {
    const dots = [...el.querySelectorAll(`.gport[data-dir="${dir}"]`)];
    if (!dots.length) return null;
    if (ptype) {
      const typed = dots.find(d => d.dataset.ptype === ptype.toLowerCase());
      if (typed) return typed;
    }
    return dots[0];
  }

  function _startWireFrom(dot, e) {
    const pos = gPortPos(dot);
    gPendingEdge = { src_node: node.id, src_port: dot.dataset.port, x0: pos.x, y0: pos.y };
    gPendingEl.style.display = '';
    const r = gCanvasWrap.getBoundingClientRect();
    gUpdatePendingEdge(e.clientX - r.left, e.clientY - r.top);
  }

  function _acceptWireAt(dot, e) {
    e.stopPropagation();
    if (!gPendingEdge) return;
    if (gPendingEdge.reverse && dot.dataset.dir === 'output')
      gAddEdge(node.id, dot.dataset.port, gPendingEdge.dst_node, gPendingEdge.dst_port);
    else if (!gPendingEdge.reverse && dot.dataset.dir === 'input')
      gAddEdge(gPendingEdge.src_node, gPendingEdge.src_port, node.id, dot.dataset.port);
    gPendingEdge = null; gPendingEl.style.display = 'none';
  }

  // Port-row hit areas (entire row, not just the dot)
  el.querySelectorAll('.gnode-port-row').forEach(row => {
    const dir = row.classList.contains('output') ? 'output' : 'input';
    const dot = row.querySelector('.gport');
    if (!dot) return;
    row.style.cursor = dir === 'output' ? 'crosshair' : 'cell';
    row.addEventListener('mousedown', e => {
      if (e.button !== 0) return;
      if (e.target.closest('.gport')) return; // dot handles itself
      e.stopPropagation(); e.preventDefault();
      if (dir === 'output') _startWireFrom(dot, e);
    });
    row.addEventListener('mouseup', e => {
      if (e.target.closest('.gport')) return;
      _acceptWireAt(dot, e);
    });
  });

  // Large zone on node body: left half = accept input, right half = start output
  el.addEventListener('mousedown', e => {
    if (e.button !== 0) return;
    if (e.target.closest('.gnode-port-row, .gnode-header, .gport')) return;
    const nr = el.getBoundingClientRect();
    const isRight = (e.clientX - nr.left) > nr.width / 2;
    if (!isRight) return; // left half — let node drag handle it
    const ptype = gPendingEdge?.ptype;
    const dot = _nearestPort('output', ptype);
    if (!dot) return;
    e.stopPropagation(); e.preventDefault();
    _startWireFrom(dot, e);
  });
  el.addEventListener('mouseup', e => {
    if (!gPendingEdge) return;
    if (e.target.closest('.gnode-port-row, .gport')) return;
    const nr = el.getBoundingClientRect();
    const isLeft = (e.clientX - nr.left) <= nr.width / 2;
    if (!isLeft) return;
    const ptype = gPendingEdge?.src_ptype;
    const dot = _nearestPort('input', ptype);
    if (!dot) return;
    _acceptWireAt(dot, e);
  });

  // Dot-level listeners (still work for precision)
  el.querySelectorAll('.gport').forEach(dot => {
    dot.addEventListener('mousedown', e => {
      if (e.button !== 0) return;
      e.stopPropagation(); e.preventDefault();
      if (dot.dataset.dir === 'output') _startWireFrom(dot, e);
    });
    dot.addEventListener('mouseup', e => _acceptWireAt(dot, e));
    dot.addEventListener('touchstart', e => {
      e.stopPropagation(); e.preventDefault();
      if (dot.dataset.dir === 'output') _startWireFrom(dot, e.touches[0]);
    }, { passive: false });
  });
}

// ── Multi-select helpers ───────────────────────────────────────
function gToggleMultiSelect(id) {
  const el = document.getElementById('gnode-' + id);
  if (gSelectedNodes.has(id)) {
    gSelectedNodes.delete(id);
    if (el) el.classList.remove('selected-multi');
  } else {
    gSelectedNodes.add(id);
    if (el) el.classList.add('selected-multi');
  }
}
function gClearMultiSelect() {
  gSelectedNodes.clear();
  gNodesEl.querySelectorAll('.gnode.selected-multi').forEach(el => el.classList.remove('selected-multi'));
}

// ── Group selected nodes ───────────────────────────────────────
function gGroupSelectedNodes() {
  const selSet = gSelectedNodes.size > 1 ? new Set(gSelectedNodes) : (gSelectedNode ? new Set([gSelectedNode]) : new Set());
  if (selSet.size < 2) { gShowToast('Shift+click 2+ nodes to multi-select, then right-click → Group', true); return; }

  const name = prompt('Group name:', 'Group');
  if (name === null) return;

  const innerNodes = gNodes.filter(n => selSet.has(n.id));
  const cx = Math.round(innerNodes.reduce((s, n) => s + n.x, 0) / innerNodes.length);
  const cy = Math.round(innerNodes.reduce((s, n) => s + n.y, 0) / innerNodes.length);

  const innerEdges = [], boundaryIn = [], boundaryOut = [];
  for (const e of gEdges) {
    const si = selSet.has(e.src_node), di = selSet.has(e.dst_node);
    if (si && di) innerEdges.push(e);
    else if (!si && di) boundaryIn.push(e);
    else if (si && !di) boundaryOut.push(e);
  }

  // Infer port types from node defs
  function portType(nodeId, portName, dir) {
    const n = gNodes.find(x => x.id === nodeId);
    if (!n || n.type === 'group') return 'image';
    const def = gNodeDefs[n.method_id];
    if (!def) return 'image';
    const src = dir === 'output' ? def.outputs : def.inputs;
    return (src || {})[portName] || 'image';
  }
  const exposed_inputs  = boundaryIn.map(e => ({ port: e.dst_port, port_type: portType(e.dst_node, e.dst_port, 'input'),  inner_node: e.dst_node, inner_param: e.dst_port }));
  const exposed_outputs = boundaryOut.map(e => ({ port: e.src_port, port_type: portType(e.src_node, e.src_port, 'output'), inner_node: e.src_node }));

  const groupId = 'group_' + Date.now().toString(36);
  const groupNode = {
    id: groupId, type: 'group', name: name || 'Group',
    subgraph: {
      nodes: innerNodes.map(n => ({
        id: n.id, method_id: n.method_id, type: n.type, name: n.name,
        subgraph: n.subgraph, exposed_inputs: n.exposed_inputs, exposed_outputs: n.exposed_outputs,
        params: { ...(n.params || {}) }, animParams: { ...(n.animParams || {}) },
        x: Math.round(n.x - cx), y: Math.round(n.y - cy), render: false, dirty: true,
      })),
      edges: innerEdges.map(e => ({ src_node: e.src_node, src_port: e.src_port, dst_node: e.dst_node, dst_port: e.dst_port, feedback: !!e.feedback })),
    },
    exposed_inputs, exposed_outputs,
    x: cx, y: cy, dirty: true, render: false,
  };

  // Remove inner node DOM + data
  for (const n of innerNodes) { const el = document.getElementById('gnode-'+n.id); if (el) el.remove(); }
  gNodes = gNodes.filter(n => !selSet.has(n.id));
  gEdges = gEdges.filter(e => !(selSet.has(e.src_node) || selSet.has(e.dst_node)));

  // Add group + reconnect boundary wires
  gNodes.push(groupNode);
  for (const e of boundaryIn)  gEdges.push({ id: 'e'+(++gEdgeCounter), src_node: e.src_node, src_port: e.src_port, dst_node: groupId, dst_port: e.dst_port, feedback: false });
  for (const e of boundaryOut) gEdges.push({ id: 'e'+(++gEdgeCounter), src_node: groupId, src_port: e.src_port, dst_node: e.dst_node, dst_port: e.dst_port, feedback: false });

  gClearMultiSelect();
  gRenderGroupNode(groupNode);
  gUpdateConnectedPorts(); gRedrawEdges(); gSave();
  gShowToast(`Grouped ${innerNodes.length} nodes as "${groupNode.name}"`);
}

// ── Ungroup: expand a group node back onto the canvas ─────────
function gUngroup(groupId) {
  const groupNode = gNodes.find(n => n.id === groupId);
  if (!groupNode || groupNode.type !== 'group') return;

  const innerNodes   = (groupNode.subgraph?.nodes || []);
  const innerEdges   = (groupNode.subgraph?.edges || []);
  const expInputs    = groupNode.exposed_inputs  || [];
  const expOutputs   = groupNode.exposed_outputs || [];
  const outerIn      = gEdges.filter(e => e.dst_node === groupId);
  const outerOut     = gEdges.filter(e => e.src_node === groupId);

  gNodes = gNodes.filter(n => n.id !== groupId);
  gEdges = gEdges.filter(e => e.src_node !== groupId && e.dst_node !== groupId);
  const groupEl = document.getElementById('gnode-'+groupId);
  if (groupEl) groupEl.remove();

  const worldNodes = innerNodes.map(n => ({ ...n, x: (n.x||0) + groupNode.x, y: (n.y||0) + groupNode.y, dirty: true }));
  for (const n of worldNodes) gNodes.push(n);
  for (const e of innerEdges) gEdges.push({ id: 'e'+(++gEdgeCounter), src_node: e.src_node, src_port: e.src_port, dst_node: e.dst_node, dst_port: e.dst_port, feedback: !!e.feedback });

  // Re-attach boundary wires through exposed port mapping
  for (const outerEdge of outerIn) {
    const expIn = expInputs.find(ei => ei.port === outerEdge.dst_port);
    if (expIn) gEdges.push({ id: 'e'+(++gEdgeCounter), src_node: outerEdge.src_node, src_port: outerEdge.src_port, dst_node: expIn.inner_node, dst_port: expIn.inner_param || expIn.port, feedback: false });
  }
  for (const outerEdge of outerOut) {
    const expOut = expOutputs.find(eo => eo.port === outerEdge.src_port);
    if (expOut) gEdges.push({ id: 'e'+(++gEdgeCounter), src_node: expOut.inner_node, src_port: expOut.port, dst_node: outerEdge.dst_node, dst_port: outerEdge.dst_port, feedback: false });
  }

  for (const n of worldNodes) gRenderAnyNode(n);
  gUpdateConnectedPorts(); gRedrawEdges(); gSave();
  gShowToast('Ungrouped');
}

// ── Group expand modal ─────────────────────────────────────────
function gOpenGroupModal(node) {
  const modal    = document.getElementById('group-expand-modal');
  const titleEl  = document.getElementById('group-modal-title');
  const subEl    = document.getElementById('group-modal-subtitle');
  const nodesEl  = document.getElementById('group-modal-nodes');
  const innerCount = (node.subgraph?.nodes || []).length;
  titleEl.textContent = node.name || 'Group';
  subEl.textContent   = `${innerCount} node${innerCount===1?'':'s'} · ${(node.exposed_inputs||[]).length} in · ${(node.exposed_outputs||[]).length} out`;
  nodesEl.innerHTML   = (node.subgraph?.nodes || []).map(n => {
    const def = gNodeDefs[n.method_id];
    const label = n.type === 'group' ? `⊞ ${n.name || 'Group'}` : (def?.name || n.method_id || '?');
    return `<div class="gm-inner-item">${escHtml(label)}</div>`;
  }).join('');
  document.getElementById('gm-ungroup-btn').onclick = () => { modal.classList.remove('open'); gUngroup(node.id); };
  document.getElementById('gm-save-preset-btn').onclick = async () => {
    const pname = prompt('Save group preset as:', node.name || 'group');
    if (!pname) return;
    try {
      const r = await fetch('/api/groups/save', { method: 'POST', headers: {'Content-Type':'application/json'}, body: JSON.stringify({ name: pname, subgraph: node.subgraph, exposed_inputs: node.exposed_inputs, exposed_outputs: node.exposed_outputs }) });
      const d = await r.json();
      if (d.ok) { gShowToast(`Saved preset "${d.name}"`); gLoadGroupPresets(); }
      else gShowToast('Save failed', true);
    } catch(e) { gShowToast('Error: '+e.message, true); }
  };
  document.getElementById('gm-close-btn').onclick = () => modal.classList.remove('open');
  modal.classList.add('open');
}
document.getElementById('group-expand-modal').addEventListener('click', e => {
  if (e.target === document.getElementById('group-expand-modal'))
    document.getElementById('group-expand-modal').classList.remove('open');
});

// ── Add a group node from a preset definition ─────────────────
function gAddGroupNode(groupDef, x, y) {
  const id = 'group_' + Date.now().toString(36);
  const node = { id, type: 'group', name: groupDef.name, subgraph: groupDef.subgraph || {nodes:[],edges:[]}, exposed_inputs: groupDef.exposed_inputs || [], exposed_outputs: groupDef.exposed_outputs || [], x, y, dirty: true, render: false };
  gNodes.push(node);
  gRenderGroupNode(node);
  gPushApart(node);
  gPhysicsKick();
  gSave();
  return node;
}

// ── Load saved group presets ───────────────────────────────────
async function gLoadGroupPresets() {
  try {
    window._gGroupPresets = await fetch('/api/groups').then(r => r.json());
    gRenderPalette();
  } catch {}
}

// ── Test Node report loader ──────────────────────────────────────
async function gLoadTestNodeReport(nodeId) {
  const body = document.getElementById('tn-report-body');
  if (!body) return;
  try {
    const r = await fetch(`/api/test-node/report/${nodeId}`);
    const d = await r.json();
    if (!d.report) {
      body.innerHTML = '<span style="color:var(--muted)">Run the graph to generate a report.</span>';
      return;
    }
    const rep = d.report;
    let html = '';

    // Inputs section
    html += '<div style="margin-bottom:8px"><strong style="color:var(--accent)">Inputs</strong></div>';
    for (const [port, info] of Object.entries(rep.inputs)) {
      const status = info.connected ? '✓' : '✗';
      const color = info.connected ? 'var(--success)' : 'var(--err)';
      html += `<div style="margin:2px 0;font-size:10px"><span style="color:${color}">${status}</span> <strong>${port}</strong>`;
      if (info.connected) {
        if (info.error) {
          html += ` <span style="color:var(--err)">${escHtml(info.error)}</span>`;
        } else if (info.value !== undefined) {
          html += ` = <span style="color:var(--text)">${info.value}</span>`;
        } else if (info.shape) {
          html += ` <span style="color:var(--muted)">${info.shape.join('×')} ${info.dtype}</span>`;
          if (info.mean !== undefined) html += ` · μ=${info.mean}`;
          if (info.count !== undefined) html += ` · count=${info.count}`;
        }
      }
      html += '</div>';
    }

    // Outputs section
    html += '<div style="margin:8px 0 4px"><strong style="color:var(--success)">Outputs</strong></div>';
    for (const [port, info] of Object.entries(rep.outputs)) {
      html += `<div style="margin:2px 0;font-size:10px"><strong>${port}</strong>`;
      if (typeof info === 'object') {
        if (info.pattern) html += ` · ${info.pattern}`;
        if (info.shape) html += ` <span style="color:var(--muted)">${info.shape.join('×')}</span>`;
        if (info.mean !== undefined) html += ` · μ=${info.mean}`;
        if (info.count !== undefined) html += ` · count=${info.count}`;
      } else {
        html += ` = <span style="color:var(--text)">${info}</span>`;
      }
      html += '</div>';
    }

    body.innerHTML = html;
  } catch {
    body.innerHTML = '<span style="color:var(--err)">Could not load report</span>';
  }
}

// ── Serialize one node for API call ───────────────────────────
function gSerializeNodeForApi(n, frame) {
  if (n.type === 'group') {
    return { id: n.id, type: 'group', name: n.name, subgraph: n.subgraph, exposed_inputs: n.exposed_inputs || [], exposed_outputs: n.exposed_outputs || [], x: n.x, y: n.y, render: !!n.render, dirty: n.dirty !== false };
  }
  const params = frame !== undefined ? gGetAnimatedParams(n, frame) : n.params;
  return {
    id: n.id, method_id: n.method_id, params,
    x: n.x, y: n.y, render: !!n.render, dirty: n.dirty !== false,
    start_frame: n.start_frame || 0, end_frame: n.end_frame || 0,
    keyframes: n.keyframes || [],
    paramKeyframes: n.paramKeyframes || {},
    prebake: n.prebake || 0,
  };
}
function gDefaultParams(def) {
  const p = {};
  for (const [k, s] of Object.entries(def.params || {})) p[k] = s.default ?? null;
  return p;
}

// ── Render a node div ──────────────────────────────────────────
// Stable category → chip color (drives the .gnode-title::before swatch)
const _gCatColors = {};
function gCategoryColor(cat) {
  if (!cat) return '';
  if (!_gCatColors[cat]) {
    let h = 0;
    for (const ch of cat) h = (h * 31 + ch.charCodeAt(0)) >>> 0;
    _gCatColors[cat] = `hsl(${h % 360} 52% 60%)`;
  }
  return _gCatColors[cat];
}

// Category → header glyph (falls back to the plain colour chip when unknown)
const _gCatIcons = {
  gpu_shaders: '⚡', client_3d: '🧊', p5_sketches: '🎨', ml_models: '🧠',
  simulations: '🌀', cli_tools: '⌨️', io: '📁', channels: '🎛️',
};

function gRenderNode(node) {
  const def = gNodeDefs[node.method_id];
  if (!def) return;

  const el = document.createElement('div');
  el.className = 'gnode';
  el.id = 'gnode-' + node.id;
  el.style.left = node.x + 'px';
  el.style.top  = node.y + 'px';
  if (def.category) el.style.setProperty('--cat-c', gCategoryColor(def.category));

  if (node.render) el.classList.add('render-target');
  if (def.deprecated) el.classList.add('gnode-deprecated');

  const header = document.createElement('div');
  header.className = 'gnode-header';
  const catIco = _gCatIcons[def.category];
  if (catIco) el.classList.add('has-cat-ico');   // icon replaces the colour chip
  header.innerHTML = `<button class="gnode-delete" title="Delete">✕</button>
    ${catIco ? `<span class="gnode-cat-ico" title="${escHtml(def.category)}">${catIco}</span>` : ''}
    <span class="gnode-title">${escHtml(def.name)}</span>
    <button class="gnode-render${node.render ? ' active' : ''}" title="Set as render target">◎</button>`;
  el.appendChild(header);

  // ── Top ports: IMAGE input (left) + IMAGE output (right) ──────
  const topPortsEl = document.createElement('div');
  topPortsEl.className = 'gnode-top-ports';
  function _mkPort(name, type, dir) {
    const dot = document.createElement('div');
    dot.className = `gport pt-${type}`;
    Object.assign(dot.dataset, { nid: node.id, port: name, dir, ptype: type });
    return dot;
  }
  function _mkLabel(name) {
    const lbl = document.createElement('span');
    lbl.className = 'gport-label'; lbl.textContent = gPortLabel(name);
    return lbl;
  }
  // Left: image input
  { const row = document.createElement('div');
    row.className = 'gnode-port-row input';
    const imgIn = Object.entries(def.inputs || {}).find(([,t]) => t === 'image');
    if (imgIn) { row.appendChild(_mkPort(imgIn[0], imgIn[1], 'input')); row.appendChild(_mkLabel(imgIn[0])); }
    topPortsEl.appendChild(row); }
  // Right: image output
  { const row = document.createElement('div');
    row.className = 'gnode-port-row output';
    const imgOut = Object.entries(def.outputs || {}).find(([,t]) => t === 'image');
    if (imgOut) { row.appendChild(_mkLabel(imgOut[0])); row.appendChild(_mkPort(imgOut[0], imgOut[1], 'output')); }
    topPortsEl.appendChild(row); }
  if (topPortsEl.querySelector('.gport')) el.appendChild(topPortsEl);

  // ── Bottom ports: non-IMAGE inputs + non-IMAGE outputs ─────────
  // Rule 1 — passthrough: if an output's name is a prefix of an input name AND
  //   types match, they share a row (input left, output right).
  // Rule 2 — native: outputs with no matching input stack at the bottom.
  // 'luminance' is always native regardless of type matches.
  const portsEl = document.createElement('div');
  portsEl.className = 'gnode-ports';

  // Exclude only the primary image input (shown at top) from the bottom section.
  // Extra image inputs (e.g. image_a, image_b on Image Blend) stay here so they get DOM ports.
  const primaryImgIn = Object.entries(def.inputs || {}).find(([,t]) => t === 'image');
  const nonImgIn  = Object.entries(def.inputs  || {}).filter(([name, t]) =>
    t !== 'image' || (primaryImgIn && name !== primaryImgIn[0])
  );
  const nonImgOut = Object.entries(def.outputs || {}).filter(([,t]) => t !== 'image');

  // Build pairing: outName → inName (first input whose name starts with outName, same type)
  const pairedOut  = new Map();   // outName -> inName
  const usedInPort = new Set();
  for (const [outName, outType] of nonImgOut) {
    if (outName === 'luminance') continue;
    for (const [inName, inType] of nonImgIn) {
      if (!usedInPort.has(inName) && inType === outType && inName.startsWith(outName)) {
        pairedOut.set(outName, inName);
        usedInPort.add(inName);
        break;
      }
    }
  }
  const inToOut = new Map([...pairedOut.entries()].map(([o, i]) => [i, o]));

  // Render input rows — paired (both dots) or standalone (left dot only)
  for (const [name, type] of nonImgIn) {
    const row = document.createElement('div');
    const outName = inToOut.get(name);
    if (outName) {
      const outType = (def.outputs || {})[outName];
      row.className = 'gnode-port-row paired';
      row.appendChild(_mkPort(name, type, 'input'));
      row.appendChild(_mkLabel(name));
      row.appendChild(_mkLabel(outName));
      row.appendChild(_mkPort(outName, outType, 'output'));
    } else {
      row.className = 'gnode-port-row input';
      row.appendChild(_mkPort(name, type, 'input'));
      row.appendChild(_mkLabel(name));
    }
    portsEl.appendChild(row);
  }

  // Render native outputs (no matching input) — right-side only, stacked below
  for (const [name, type] of nonImgOut) {
    if (pairedOut.has(name)) continue;
    const row = document.createElement('div');
    row.className = 'gnode-port-row output';
    row.appendChild(_mkLabel(name));
    row.appendChild(_mkPort(name, type, 'output'));
    portsEl.appendChild(row);
  }

  if (portsEl.children.length) el.appendChild(portsEl);
  gNodesEl.appendChild(el);

  header.querySelector('.gnode-delete').addEventListener('click', e => {
    e.stopPropagation(); gDeleteNode(node.id);
  });
  header.querySelector('.gnode-render').addEventListener('click', e => {
    e.stopPropagation();
    const wasActive = !!node.render;
    // Mutually exclusive — clear all, then toggle this one
    gNodes.forEach(n => { n.render = false; });
    document.querySelectorAll('.gnode-render').forEach(b => b.classList.remove('active'));
    document.querySelectorAll('.gnode.render-target').forEach(n => n.classList.remove('render-target'));
    node.render = !wasActive;
    if (node.render) {
      e.currentTarget.classList.add('active');
      el.classList.add('render-target');
    }
    gSave();
  });
  // Tap-to-select: only open params on short tap, not after a drag
  const _isBtn = t => t.classList.contains('gport') || t.classList.contains('gnode-delete') || t.classList.contains('gnode-render');
  let _tapStart = null;
  el.addEventListener('mousedown', e => {
    if (_isBtn(e.target)) return;
    _tapStart = { x: e.clientX, y: e.clientY, t: Date.now() };
  });
  el.addEventListener('mouseup', e => {
    if (!_tapStart) return;
    const dx = Math.abs(e.clientX - _tapStart.x), dy = Math.abs(e.clientY - _tapStart.y);
    const dt = Date.now() - _tapStart.t;
    _tapStart = null;
    if (_isBtn(e.target)) return;
    if (dx < 8 && dy < 8 && dt < 300) {
      if (e.shiftKey) gToggleMultiSelect(node.id);
      else { gClearMultiSelect(); gSelectNode(node.id); }
    }
  });
  el.addEventListener('touchstart', e => {
    if (_isBtn(e.target)) return;
    const t = e.touches[0];
    _tapStart = { x: t.clientX, y: t.clientY, t: Date.now() };
  }, { passive: true });
  el.addEventListener('touchend', e => {
    if (!_tapStart) return;
    const t = e.changedTouches[0];
    const dx = Math.abs(t.clientX - _tapStart.x), dy = Math.abs(t.clientY - _tapStart.y);
    const dt = Date.now() - _tapStart.t;
    _tapStart = null;
    if (_isBtn(e.target)) return;
    if (dx < 8 && dy < 8 && dt < 300) gSelectNode(node.id);
  });
  gAttachNodeDrag(el, node);

  function _nearestPort2(dir, ptype) {
    const dots = [...el.querySelectorAll(`.gport[data-dir="${dir}"]`)];
    if (!dots.length) return null;
    if (ptype) { const t = dots.find(d => d.dataset.ptype === ptype.toLowerCase()); if (t) return t; }
    return dots[0];
  }
  function _startWireFrom2(dot, e) {
    const pos = gPortPos(dot);
    gPendingEdge = { src_node: node.id, src_port: dot.dataset.port, x0: pos.x, y0: pos.y };
    gPendingEl.style.display = '';
    const r = gCanvasWrap.getBoundingClientRect();
    gUpdatePendingEdge(e.clientX - r.left, e.clientY - r.top);
  }
  function _acceptWireAt2(dot, e) {
    e.stopPropagation();
    if (!gPendingEdge) return;
    if (gPendingEdge.reverse && dot.dataset.dir === 'output')
      gAddEdge(node.id, dot.dataset.port, gPendingEdge.dst_node, gPendingEdge.dst_port);
    else if (!gPendingEdge.reverse && dot.dataset.dir === 'input')
      gAddEdge(gPendingEdge.src_node, gPendingEdge.src_port, node.id, dot.dataset.port);
    gPendingEdge = null; gPendingEl.style.display = 'none';
  }

  el.querySelectorAll('.gnode-port-row').forEach(row => {
    const dir = row.classList.contains('output') ? 'output' : 'input';
    const dot = row.querySelector('.gport');
    if (!dot) return;
    row.style.cursor = dir === 'output' ? 'crosshair' : 'cell';
    row.addEventListener('mousedown', e => {
      if (e.button !== 0 || e.target.closest('.gport')) return;
      e.stopPropagation(); e.preventDefault();
      if (dir === 'output') _startWireFrom2(dot, e);
    });
    row.addEventListener('mouseup', e => {
      if (e.target.closest('.gport')) return;
      _acceptWireAt2(dot, e);
    });
  });

  el.addEventListener('mousedown', e => {
    if (e.button !== 0 || e.target.closest('.gnode-port-row, .gnode-header, .gport')) return;
    const nr = el.getBoundingClientRect();
    if ((e.clientX - nr.left) <= nr.width / 2) return;
    const dot = _nearestPort2('output');
    if (!dot) return;
    e.stopPropagation(); e.preventDefault();
    _startWireFrom2(dot, e);
  });
  el.addEventListener('mouseup', e => {
    if (!gPendingEdge || e.target.closest('.gnode-port-row, .gport')) return;
    const nr = el.getBoundingClientRect();
    if ((e.clientX - nr.left) > nr.width / 2) return;
    const dot = _nearestPort2('input');
    if (!dot) return;
    _acceptWireAt2(dot, e);
  });

  el.querySelectorAll('.gport').forEach(dot => {
    dot.addEventListener('mousedown', e => {
      if (e.button !== 0) return;
      e.stopPropagation(); e.preventDefault();
      if (dot.dataset.dir === 'output') _startWireFrom2(dot, e);
    });
    dot.addEventListener('mouseup', e => _acceptWireAt2(dot, e));
    dot.addEventListener('touchstart', e => {
      e.stopPropagation(); e.preventDefault();
      if (dot.dataset.dir === 'output') _startWireFrom2(dot, e.touches[0]);
    }, { passive: false });
  });
}

// ── Port position ──────────────────────────────────────────────
function gPortPos(dotEl) {
  const cr = gCanvasWrap.getBoundingClientRect();
  const dr = dotEl.getBoundingClientRect();
  return { x: dr.left + dr.width/2 - cr.left, y: dr.top + dr.height/2 - cr.top };
}
function gUpdatePendingEdge(x1, y1) {
  if (!gPendingEdge) return;
  const { x0, y0 } = gPendingEdge;
  gPendingEl.setAttribute('d', `M${x0},${y0} L${x1},${y1}`);
}
document.addEventListener('mousemove', e => {
  if (!gPendingEdge) return;
  const r = gCanvasWrap.getBoundingClientRect();
  gUpdatePendingEdge(e.clientX - r.left, e.clientY - r.top);
});
document.addEventListener('mouseup', () => {
  if (gPendingEdge) { gPendingEdge = null; gPendingEl.style.display = 'none'; }
});
document.addEventListener('touchmove', e => {
  if (!gPendingEdge) return;
  e.preventDefault();
  const t = e.touches[0];
  const r = gCanvasWrap.getBoundingClientRect();
  gUpdatePendingEdge(t.clientX - r.left, t.clientY - r.top);
}, { passive: false });
document.addEventListener('touchend', e => {
  if (!gPendingEdge) return;
  const t = e.changedTouches[0];
  // Direct hit first (elementFromPoint may return a child; walk up to be safe)
  let hit = document.elementFromPoint(t.clientX, t.clientY)?.closest('.gport');
  // Touch snap: if no direct hit, find nearest input port within 30px
  if (!hit || hit.dataset.dir !== 'input') {
    const TOUCH_SNAP_PX = 30;
    let best = null, bestDist = TOUCH_SNAP_PX;
    document.querySelectorAll('.gport[data-dir="input"]').forEach(p => {
      const rc = p.getBoundingClientRect();
      const cx = rc.left + rc.width / 2, cy = rc.top + rc.height / 2;
      const d = Math.hypot(t.clientX - cx, t.clientY - cy);
      if (d < bestDist) { bestDist = d; best = p; }
    });
    hit = best;
  }
  if (hit && hit.dataset.dir === 'input')
    gAddEdge(gPendingEdge.src_node, gPendingEdge.src_port, hit.dataset.nid, hit.dataset.port);
  gPendingEdge = null; gPendingEl.style.display = 'none';
});
document.addEventListener('touchcancel', () => {
  if (gPendingEdge) { gPendingEdge = null; gPendingEl.style.display = 'none'; }
});

// ── Node drag ──────────────────────────────────────────────────
function gAttachNodeDrag(el, node) {
  const header = el.querySelector('.gnode-header');
  function startDrag(e) {
    if (e.target.classList.contains('gnode-delete') || e.target.classList.contains('gnode-render')) return;
    e.preventDefault();
    node._pinned = true; node._vx = 0; node._vy = 0;
    const pos  = getEventPos(e);
    const rect0 = gCanvasWrap.getBoundingClientRect();
    const ox = (pos.x - rect0.left - gPanX) / gCanvasScale - node.x;
    const oy = (pos.y - rect0.top  - gPanY) / gCanvasScale - node.y;
    function onMove(e) {
      const p = getEventPos(e);
      const r = gCanvasWrap.getBoundingClientRect();
      node.x = (p.x - r.left - gPanX) / gCanvasScale - ox;
      node.y = (p.y - r.top  - gPanY) / gCanvasScale - oy;
      el.style.left = node.x + 'px'; el.style.top = node.y + 'px';
      gRedrawEdges();
    }
    function onEnd() {
      document.removeEventListener('mousemove', onMove);
      document.removeEventListener('mouseup', onEnd);
      document.removeEventListener('touchmove', onMove);
      document.removeEventListener('touchend', onEnd);
      document.removeEventListener('touchcancel', onEnd);
      node._pinned = false;
      gSave();
      gPhysicsKick();
    }
    document.addEventListener('mousemove', onMove);
    document.addEventListener('mouseup', onEnd);
    document.addEventListener('touchmove', onMove, { passive: false });
    document.addEventListener('touchend', onEnd);
    document.addEventListener('touchcancel', onEnd);
  }
  header.addEventListener('mousedown', startDrag);
  header.addEventListener('touchstart', startDrag, { passive: false });
}

// ── Push-apart & Physics ────────────────────────────────────────
const PHYSICS = {
  k_r: 40000, k_s_push: 0, k_s_pull: 0, rest: 125, damping: 0.80, k_c: 0.001,
  k_lane: 0.06, k_dir: 0.5, lane_width: 280, margin: 80, dt: 1.0,
};
let gPhysicsActive = false;
let gPhysicsRafId  = null;
let gDepths        = null;   // cached topological depths, invalidated on graph change

function _nodeBox(n) {
  const el = document.getElementById('gnode-' + n.id);
  return { w: el ? el.offsetWidth : 160, h: el ? el.offsetHeight : 120 };
}

// BFS longest-path (Kahn's topological sort) from source nodes (no incoming non-feedback edges).
function gComputeDepths() {
  const inDeg = new Map(gNodes.map(n => [n.id, 0]));
  const adj   = new Map(gNodes.map(n => [n.id, []]));
  for (const e of gEdges) {
    if (e.feedback) continue;
    inDeg.set(e.dst_node, (inDeg.get(e.dst_node) || 0) + 1);
    if (adj.has(e.src_node)) adj.get(e.src_node).push(e.dst_node);
  }
  const depths = new Map(gNodes.map(n => [n.id, 0]));
  const queue  = gNodes.filter(n => (inDeg.get(n.id) || 0) === 0).map(n => n.id);
  let i = 0;
  while (i < queue.length) {
    const id = queue[i++];
    for (const nid of (adj.get(id) || [])) {
      depths.set(nid, Math.max(depths.get(nid) || 0, (depths.get(id) || 0) + 1));
      inDeg.set(nid, (inDeg.get(nid) || 0) - 1);
      if ((inDeg.get(nid) || 0) === 0) queue.push(nid);
    }
  }
  // Render-target node is placed one lane past the deepest node
  let maxDepth = 0;
  for (const d of depths.values()) maxDepth = Math.max(maxDepth, d);
  const renderNode = gNodes.find(n => n.render);
  if (renderNode) depths.set(renderNode.id, maxDepth + 1);
  return depths;
}

function gPushApart(newNode, padding = 20) {
  const { w: nw, h: nh } = _nodeBox(newNode);
  for (const n of gNodes) {
    if (n.id === newNode.id) continue;
    const { w, h } = _nodeBox(n);
    const ox = Math.min(newNode.x + nw, n.x + w) - Math.max(newNode.x, n.x);
    const oy = Math.min(newNode.y + nh, n.y + h) - Math.max(newNode.y, n.y);
    if (ox <= 0 || oy <= 0) continue;
    const cx = (n.x + w / 2) - (newNode.x + nw / 2);
    const cy = (n.y + h / 2) - (newNode.y + nh / 2);
    const dist = Math.hypot(cx, cy) || 1;
    const push = Math.min(ox, oy) + padding;
    n.x += (cx / dist) * push;
    n.y += (cy / dist) * push;
    const el = document.getElementById('gnode-' + n.id);
    if (el) { el.style.left = n.x + 'px'; el.style.top = n.y + 'px'; }
  }
  if (gNodes.length > 1) gRedrawEdges();
}

function gPhysicsKick() {
  gDepths = null;  // invalidate depth cache on every graph change
  if (!gPhysicsActive || gPhysicsRafId) return;
  gPhysicsRafId = requestAnimationFrame(gPhysicsTick);
}

function gPhysicsBurst(ticks = 80) {
  // Run physics for a fixed number of ticks regardless of gPhysicsActive.
  // Used when combine nodes are auto-spawned so they settle immediately.
  gDepths = null;
  const wasActive = gPhysicsActive;
  gPhysicsActive = true;
  let t = 0;
  function tick() {
    gPhysicsTick();
    if (++t < ticks && gPhysicsRafId === null) {
      gPhysicsRafId = requestAnimationFrame(tick);
    } else if (t >= ticks) {
      if (!wasActive) {
        gPhysicsActive = false;
        gPhysicsRafId = null;
      }
    }
  }
  if (!gPhysicsRafId) gPhysicsRafId = requestAnimationFrame(tick);
}

function gPhysicsTick() {
  gPhysicsRafId = null;
  if (!gPhysicsActive || gNodes.length < 2) return;

  // Recompute depths once per simulation burst
  if (!gDepths) gDepths = gComputeDepths();

  const rect = gCanvasWrap.getBoundingClientRect();
  const ccx  = (rect.width  / 2 - gPanX) / gCanvasScale;
  const ccy  = (rect.height / 2 - gPanY) / gCanvasScale;

  // ── Pass 1: edge direction bias (per-edge, accumulate into force map) ──
  const fxBias = new Map(gNodes.map(n => [n.id, 0]));
  for (const edge of gEdges) {
    if (edge.feedback) continue;
    const src = gNodes.find(n => n.id === edge.src_node);
    const dst = gNodes.find(n => n.id === edge.dst_node);
    if (!src || !dst) continue;
    // Push src leftward and dst rightward; flip sign if dst is already left of src
    const edgeDirX = PHYSICS.k_dir * (dst.x < src.x ? -1 : 1);
    fxBias.set(src.id, (fxBias.get(src.id) || 0) - edgeDirX);
    fxBias.set(dst.id, (fxBias.get(dst.id) || 0) + edgeDirX);
  }

  // ── Pass 2: per-node forces ──
  for (const n of gNodes) {
    if (n._vx === undefined) { n._vx = 0; n._vy = 0; }
    if (n._pinned || n.render) { n._vx = 0; n._vy = 0; continue; }
    const { w, h } = _nodeBox(n);
    const ncx    = n.x + w / 2, ncy = n.y + h / 2;
    const nDepth = gDepths.get(n.id) ?? 0;
    let fx = fxBias.get(n.id) || 0;
    let fy = 0;

    // Node-node repulsion — 2.5× stronger between same-lane nodes (vertical spread)
    for (const m of gNodes) {
      if (m.id === n.id) continue;
      const { w: mw, h: mh } = _nodeBox(m);
      const dx = ncx - (m.x + mw / 2), dy = ncy - (m.y + mh / 2);
      const d  = Math.hypot(dx, dy) || 0.1;
      const ux = dx / d, uy = dy / d;

      // Soft repulsion — 1/d² with minimum-distance clamp to prevent singularity oscillation.
      const effD = Math.max(d, (w + mw) / 6);
      const f = PHYSICS.k_r / (effD * effD);
      fx += ux * f; fy += uy * f * 0.15;
    }

    // Edge spring attraction (radial, bidirectional)
    for (const edge of gEdges) {
      let other = null;
      if      (edge.src_node === n.id) other = gNodes.find(m => m.id === edge.dst_node);
      else if (edge.dst_node === n.id) other = gNodes.find(m => m.id === edge.src_node);
      if (!other) continue;
      const { w: ow, h: oh } = _nodeBox(other);
      const dx = (other.x + ow / 2) - ncx, dy = (other.y + oh / 2) - ncy;
      const d  = Math.hypot(dx, dy) || 1;
      const overlap = d - PHYSICS.rest;
      const f = overlap < 0 ? PHYSICS.k_s_push * overlap : PHYSICS.k_s_pull * overlap;
      fx += (dx / d) * f; fy += (dy / d) * f;
    }

    // Lane force — pull node's left edge toward its target x lane (horizontal only)
    const targetX = nDepth * PHYSICS.lane_width + PHYSICS.margin;
    fx += PHYSICS.k_lane * (targetX - n.x);

    // Weak vertical centering (don't apply horizontal — lane force owns that axis)
    fy += PHYSICS.k_c * (ccy - ncy);

    // Integrate — dampen vertical force to reduce excessive up/down drift
    n._vx = (n._vx + fx * PHYSICS.dt) * PHYSICS.damping;
    n._vy = (n._vy + fy * PHYSICS.dt) * PHYSICS.damping;
    n.x  += n._vx * PHYSICS.dt;
    n.y  += n._vy * PHYSICS.dt;
    const el = document.getElementById('gnode-' + n.id);
    if (el) { el.style.left = n.x + 'px'; el.style.top = n.y + 'px'; }
  }

  // ── Pass 3: hard overlap resolution (post-integration) ──
  // Directly separate any pair whose bounding boxes actually intersect.
  // Applied as position correction, not force — no oscillation.
  for (let iter = 0; iter < 3; iter++) {
    let anyOverlap = false;
    for (const n of gNodes) {
      if (n._pinned || n.render) continue;
      const { w: nw, h: nh } = _nodeBox(n);
      const ncx = n.x + nw / 2, ncy = n.y + nh / 2;
      for (const m of gNodes) {
        if (m.id === n.id) continue;
        const { w: mw, h: mh } = _nodeBox(m);
        const dx = ncx - (m.x + mw / 2), dy = ncy - (m.y + mh / 2);
        const gapX = Math.abs(dx) - (nw + mw) / 2;
        const gapY = Math.abs(dy) - (nh + mh) / 2;
        if (gapX >= 0 || gapY >= 0) continue;
        anyOverlap = true;
        // Push apart along the axis with the least overlap
        if (gapX < gapY) {
          const sign = dx > 0 ? 1 : -1;
          n.x += sign * (-gapX + 2);
          m.x -= sign * (-gapX + 2);
        } else {
          const sign = dy > 0 ? 1 : -1;
          n.y += sign * (-gapY + 2);
          m.y -= sign * (-gapY + 2);
        }
      }
    }
    if (!anyOverlap) break;
  }
  // Re-render positions after hard resolution
  for (const n of gNodes) {
    const el = document.getElementById('gnode-' + n.id);
    if (el) { el.style.left = n.x + 'px'; el.style.top = n.y + 'px'; }
  }

  gRedrawEdges();
  let maxV = 0;
  for (const n of gNodes) maxV = Math.max(maxV, Math.hypot(n._vx || 0, n._vy || 0));
  if (maxV > 0.5) {
    gPhysicsRafId = requestAnimationFrame(gPhysicsTick);
  } else {
    // Snap to grid when settled
    for (const n of gNodes) {
      n.x = Math.round(n.x / GRID) * GRID;
      n.y = Math.round(n.y / GRID) * GRID;
      const el = document.getElementById('gnode-' + n.id);
      if (el) { el.style.left = n.x + 'px'; el.style.top = n.y + 'px'; }
    }
    gRedrawEdges();
    gSave();
  }
}

function gPhysicsToggle() {
  gPhysicsActive = !gPhysicsActive;
  document.getElementById('graph-layout-btn')?.classList.toggle('active', gPhysicsActive);
  document.getElementById('graph-layout-btn-desk')?.classList.toggle('active', gPhysicsActive);
  if (gPhysicsActive) {
    gNodes.forEach(n => { n._vx = 0; n._vy = 0; });
    gPhysicsKick();
  } else {
    if (gPhysicsRafId) { cancelAnimationFrame(gPhysicsRafId); gPhysicsRafId = null; }
    gSave();
  }
}

// ── Edges ──────────────────────────────────────────────────────
function gAddEdge(src_node, src_port, dst_node, dst_port) {
  if (src_node === dst_node) return;
  const srcDot = gNodesEl.querySelector(`.gport[data-nid="${src_node}"][data-port="${src_port}"][data-dir="output"]`);
  const dstDot = gNodesEl.querySelector(`.gport[data-nid="${dst_node}"][data-port="${dst_port}"][data-dir="input"]`);
  if (!gPortsCompatible(srcDot, dstDot)) return;
  if (gEdges.find(e => e.src_node===src_node && e.src_port===src_port && e.dst_node===dst_node && e.dst_port===dst_port)) return;

  // Auto-spawn a merge node when a second wire lands on an occupied input port
  const conflict = gEdges.find(e => e.dst_node===dst_node && e.dst_port===dst_port);
  if (conflict) {
    const ptype = (srcDot?.dataset?.ptype || dstDot?.dataset?.ptype || '').toUpperCase();
    const mergeMap = {
      IMAGE:     { method: '137', out_port: 'image',     in_a: 'image_a',     in_b: 'image_b'     },
      SCALAR:    { method: '138', out_port: 'value',     in_a: 'value_a',     in_b: 'value_b'     },
      FIELD:     { method: '139', out_port: 'field',     in_a: 'field_a',     in_b: 'field_b'     },
      PARTICLES: { method: '140', out_port: 'particles', in_a: 'particles_a', in_b: 'particles_b' },
      // Client-side 3D: two objects into one Scene port → auto-group them.
      OBJECT3D:  { method: '__group3d__', out_port: 'object', in_a: 'object_a', in_b: 'object_b' },
    };
    const spec = mergeMap[ptype];
    if (!spec) { gSetStatus(`Cannot merge ${ptype || 'unknown'} wires`); return; }
    const nodeA = gNodes.find(n => n.id === conflict.src_node);
    const nodeB = gNodes.find(n => n.id === src_node);
    const mx = ((nodeA?.x || 0) + (nodeB?.x || 0)) / 2 + 140;
    const my = ((nodeA?.y || 0) + (nodeB?.y || 0)) / 2;
    const mergeNode = gAddNode(spec.method, mx, my);
    if (!mergeNode) { gSetStatus('Merge method not found — restart server'); return; }
    gEdges = gEdges.filter(e => e.id !== conflict.id);
    gEdges.push({ id: 'e'+(++gEdgeCounter), src_node: conflict.src_node, src_port: conflict.src_port, dst_node: mergeNode.id, dst_port: spec.in_a,     feedback: false });
    gEdges.push({ id: 'e'+(++gEdgeCounter), src_node,                    src_port,                    dst_node: mergeNode.id, dst_port: spec.in_b,     feedback: false });
    gEdges.push({ id: 'e'+(++gEdgeCounter), src_node: mergeNode.id,      src_port: spec.out_port,     dst_node,               dst_port,                feedback: false });
    gUpdateConnectedPorts();
    if (gSelectedNode === dst_node) gRefreshParamOverrides(dst_node);
    gRedrawEdges(); gSave(); gPhysicsBurst();
    return;
  }

  gEdges.push({ id: 'e'+(++gEdgeCounter), src_node, src_port, dst_node, dst_port, feedback: false });
  gUpdateConnectedPorts();
  if (gSelectedNode === dst_node) gRefreshParamOverrides(dst_node);
  gRedrawEdges(); gSave(); gPhysicsKick();
}
function gDeleteEdge(id) {
  const edge = gEdges.find(e => e.id === id);
  gEdges = gEdges.filter(e => e.id !== id);
  gUpdateConnectedPorts();
  if (edge && gSelectedNode === edge.dst_node) gRefreshParamOverrides(edge.dst_node);
  gRedrawEdges(); gSave(); gPhysicsKick();
}
function gDeleteNode(id) {
  gNodes = gNodes.filter(n => n.id !== id);
  gEdges = gEdges.filter(e => e.src_node !== id && e.dst_node !== id);
  const el = document.getElementById('gnode-' + id);
  if (el) el.remove();
  if (gSelectedNode === id) {
    gSelectedNode = null; gShowNodeParams(null);
    if (isMobile()) gParamsSheetClose();
  }
  // Drop it from the multi-select set too. Leaving a deleted id in there hands
  // every consumer of the selection (duplicate, copy, group) an id with no
  // node behind it — gGroupSelectedNodes silently under-groups, and the
  // clipboard's paste path resolves an empty fragment.
  gSelectedNodes.delete(id);
  gUpdateConnectedPorts();
  gRedrawEdges(); gSave(); gPhysicsKick();
}

// Keep the edge-id counter ahead of every edge already in the document.
// Shared by gLoad() and the history module's restore so the 'e<n>' id format
// is only understood in one place — a stale parse here silently yields 0 and
// makes the next wire reuse an existing edge id.
function gRecomputeEdgeCounter() {
  gEdgeCounter = Math.max(0, ...gEdges.map(e => parseInt(e.id?.slice(1)) || 0));
}

function gRedrawEdges() {
  gEdgesEl.innerHTML = '';
  for (const edge of gEdges) {
    const s = gNodesEl.querySelector(`.gport[data-nid="${edge.src_node}"][data-port="${edge.src_port}"][data-dir="output"]`);
    const d = gNodesEl.querySelector(`.gport[data-nid="${edge.dst_node}"][data-port="${edge.dst_port}"][data-dir="input"]`);
    if (!s || !d) continue;
    const p0 = gPortPos(s), p1 = gPortPos(d);
    const path = document.createElementNS('http://www.w3.org/2000/svg', 'path');
    path.setAttribute('d', `M${p0.x},${p0.y} L${p1.x},${p1.y}`);
    path.setAttribute('class', 'gedge' + (edge.feedback ? ' feedback' : ''));
    // Wire color = source port's payload type, from the server registry
    const ptSpec = (window.gPortTypes || {})[(s.dataset.ptype || '').toUpperCase()];
    if (ptSpec && ptSpec.color) path.style.stroke = ptSpec.color;
    path.dataset.eid = edge.id;
    path.addEventListener('contextmenu', ev => { ev.preventDefault(); gShowEdgeCtx(edge.id, ev.clientX, ev.clientY); });
    path.addEventListener('mousedown', ev => {
      if (ev.button !== 0) return;
      ev.stopPropagation(); ev.preventDefault();
      const r = gCanvasWrap.getBoundingClientRect();
      const mx = ev.clientX - r.left, my = ev.clientY - r.top;
      // Pick the closer end to detach
      const d0 = Math.hypot(mx - p0.x, my - p0.y);
      const d1 = Math.hypot(mx - p1.x, my - p1.y);
      gDeleteEdge(edge.id);
      if (d0 < d1) {
        // Detach from src — keep dst fixed, drag from dst back to a new src
        gPendingEdge = { dst_node: edge.dst_node, dst_port: edge.dst_port, x0: p1.x, y0: p1.y, reverse: true };
      } else {
        // Detach from dst — keep src fixed, drag from src to a new dst
        gPendingEdge = { src_node: edge.src_node, src_port: edge.src_port, x0: p0.x, y0: p0.y };
      }
      gPendingEl.style.display = '';
      gUpdatePendingEdge(mx, my);
    });
    path.addEventListener('mouseenter', async ev => {
      if (!gLastGraphJobId) return;
      try {
        const r = await fetch(`/api/graph/wire-payload/${gLastGraphJobId}/${edge.src_node}`);
        const d = await r.json();
        const lines = Object.entries(d.payload || {}).map(([k,v]) => `${k}  ${v}`).join('\n');
        gWireTooltip.textContent = lines || '(no payload)';
        gWireTooltip.style.left = (ev.clientX + 14) + 'px';
        gWireTooltip.style.top  = (ev.clientY - 8)  + 'px';
        gWireTooltip.style.display = '';
      } catch {}
    });
    path.addEventListener('mouseleave', () => { gWireTooltip.style.display = 'none'; });
    gEdgesEl.appendChild(path);
  }
  if (window.gOverlayRefresh) window.gOverlayRefresh(); // realign FX overlay on edge/node changes
}

// ── Select node ────────────────────────────────────────────────
function gSelectNode(id) {
  gNodesEl.querySelectorAll('.gnode').forEach(el => el.classList.remove('selected'));
  const el = document.getElementById('gnode-' + id);
  if (el) el.classList.add('selected');
  gSelectedNode = id;
  const node = gNodes.find(n => n.id === id);
  gShowNodeParams(node);
  renderTimelineRuler();   // Phase 5: refresh node KF lanes
  if (isMobile()) gParamsSheetOpen();
  // Sync selection into the 3D viewport editor when it's open.
  if (window._gEditor3D && window._gEditor3D.isOpen() && !window._gEditor3DSelecting)
    window._gEditor3D.selectNode(id);
}

function gShowNodeParams(node) {
  if (!node) {
    gParamsEmpty.style.display = ''; gParamsForm.style.display = 'none';
    if (gParamsHdr) gParamsHdr.textContent = 'Node Parameters';
    return;
  }
  // Group node: show info panel instead of param form
  if (node.type === 'group') {
    if (gParamsHdr) gParamsHdr.textContent = node.name || 'Group';
    gParamsEmpty.style.display = 'none'; gParamsForm.style.display = '';
    const innerCount = (node.subgraph?.nodes || []).length;
    gParamsForm.innerHTML = `<p style="font-size:12px;color:var(--muted);margin-bottom:12px">⊞ Group node<br>${innerCount} inner node${innerCount===1?'':'s'} · Double-click to expand</p>
      <button id="group-params-open-btn" style="width:100%;padding:8px;background:var(--bg3);border:1px solid var(--border);border-radius:7px;color:var(--text);cursor:pointer;font-size:13px;font-weight:600;">⊞ Open Group</button>`;
    document.getElementById('group-params-open-btn')?.addEventListener('click', () => gOpenGroupModal(node));
    return;
  }
  const def = gNodeDefs[node.method_id];
  if (gParamsHdr) gParamsHdr.textContent = def ? def.name : 'Parameters';
  gParamsEmpty.style.display = 'none'; gParamsForm.style.display = '';
  const entries = Object.entries(def.params || {});
  gParamsForm.innerHTML = entries.length
    ? entries.map(([k,s]) => renderParamField(k, s)).join('')
    : '<p style="font-size:12px;color:var(--muted)">No parameters.</p>';

  // Tag each param-row with its key and add a ⋮ flyout button.
  gParamsForm.querySelectorAll('.param-row').forEach(row => {
    if (row.dataset.param) return;
    const input = row.querySelector('[id^="p_"]');
    if (!input) return;
    const key = input.id.replace(/^p_/, '').replace(/_swatch$/, '');
    row.dataset.param = key;

    // Small kebab button to open the per-parameter flyout menu.
    const menuBtn = document.createElement('button');
    menuBtn.className = 'ndpm-btn';
    menuBtn.type = 'button';
    menuBtn.title = 'Parameter actions';
    menuBtn.textContent = '⋯';
    menuBtn.setAttribute('aria-label', 'Parameter actions');
    row.appendChild(menuBtn);
  });

  // ── Test Node: show report section ──────────────────────────────
  if (node.method_id === '__test__') {
    const reportDiv = document.createElement('div');
    reportDiv.id = 'test-node-report';
    reportDiv.style.cssText = 'margin-top:16px;border-top:1px solid var(--border);padding-top:12px';
    reportDiv.innerHTML = '<div class="section-label">Test Report</div><div id="tn-report-body" style="font-size:11px;color:var(--muted)">Run the graph to generate a report.</div>';
    gParamsForm.appendChild(reportDiv);
    gLoadTestNodeReport(node.id);
  }

  // ── Upload button for params that accept a file (spec.upload = accept list).
  // Uploads to /api/assets/upload (raw body) and writes the served URL back
  // into the param — used by the USD/GLTF model nodes.
  for (const [key, spec] of entries) {
    if (!spec || !spec.upload) continue;
    const el = document.getElementById(`p_${key}`);
    const row = el && el.closest('.param-row');
    if (!row) continue;
    const btn = document.createElement('button');
    btn.type = 'button';
    btn.className = 'asset-upload-btn';
    btn.textContent = '⬆ Upload file…';
    btn.style.cssText = 'margin-top:5px;width:100%;padding:6px;background:var(--bg3);'
      + 'border:1px solid var(--border);border-radius:6px;color:var(--text);cursor:pointer;font-size:12px;';
    const picker = document.createElement('input');
    picker.type = 'file';
    picker.accept = String(spec.upload);
    picker.style.display = 'none';
    picker.addEventListener('change', async () => {
      const f = picker.files && picker.files[0];
      if (!f) return;
      btn.textContent = '⬆ Uploading…'; btn.disabled = true;
      try {
        const r = await fetch('/api/assets/upload?name=' + encodeURIComponent(f.name),
                              { method: 'POST', body: f });
        if (!r.ok) throw new Error('HTTP ' + r.status);
        const j = await r.json();
        el.value = j.url;
        gUpdateNodeParam(node.id, key, j.url);
        gShowToast('Uploaded ' + j.name + ' (' + Math.round(j.bytes / 1024) + ' KB)');
        gDoAutoGen();
      } catch (e) {
        gShowToast('Upload failed: ' + (e && e.message || e), true);
      } finally {
        btn.textContent = '⬆ Upload file…'; btn.disabled = false;
        picker.value = '';
      }
    });
    btn.addEventListener('click', () => picker.click());
    row.appendChild(btn);
    row.appendChild(picker);
  }

  if (!node.animParams) node.animParams = {};
  for (const [key, spec] of entries) {
    const el = document.getElementById(`p_${key}`);
    if (!el) continue;
    const val = node.params[key] ?? spec.default;
    if (el.type === 'checkbox') el.checked = !!val; else el.value = val ?? '';
    const v = gParamsForm.querySelector(`#val_${key}`);
    if (v) v.textContent = formatVal(el.value);

    // Sync color swatch from text field value
    const swatch = document.getElementById(`p_${key}_swatch`);
    if (swatch) {
      const hex = _parseColorToHex(String(el.value));
      swatch.value = hex;
    }

    // Expr + anim toggles for numeric params
    if (el.type === 'number' || el.type === 'range') {
      const paramRow = el.closest('.param-row');
      if (paramRow) {
        gSetupExprToggle(key, spec, node, el);

        const animBtn = document.createElement('button');
        animBtn.className = 'anim-toggle' + (node.animParams[key] ? ' anim-on' : '');
        animBtn.title = 'Animate this param';
        animBtn.textContent = '~';

        const kfBtn = document.createElement('button');
        kfBtn.className = 'kf-toggle';
        kfBtn.title = 'Add keyframe for this param at current frame';
        kfBtn.textContent = '🔑';
        kfBtn.style.cssText = 'font-size:10px;padding:1px 4px;background:var(--bg3);border:1px solid var(--border);border-radius:3px;color:var(--text);cursor:pointer;margin-left:2px;';
        kfBtn.addEventListener('click', (e) => {
          e.stopPropagation();
          const frame = parseInt(document.getElementById('tl-frame')?.value) || 0;
          if (!node.paramKeyframes) node.paramKeyframes = {};
          if (!node.paramKeyframes[key]) node.paramKeyframes[key] = [];
          // Check if keyframe already exists at this frame for this param
          if (node.paramKeyframes[key].some(kf => kf.frame === frame)) {
            gShowToast('Keyframe already exists at frame ' + frame + ' for ' + key, true);
            return;
          }
          const curVal = parseFloat(el.value) || 0;
          node.paramKeyframes[key].push({ frame, value: curVal, easing: 'ease-in-out', handle_in: null, handle_out: null });
          node.paramKeyframes[key].sort((a, b) => a.frame - b.frame);
          gSave();
          renderPerParamKeyframes(node, entries);
          renderTimelineRuler();
          gShowToast('Keyframe added: ' + key + ' @ F' + frame + ' = ' + curVal.toFixed(2));
        });

        const animRow = document.createElement('div');
        animRow.className = 'anim-row' + (node.animParams[key] ? ' anim-visible' : '');

        const curAnim = node.animParams[key];
        const curVal = parseFloat(el.value) || 0;
        animRow.innerHTML =
          `<span>from</span>` +
          `<input type="number" step="any" class="anim-input anim-from-${key}" value="${curAnim ? curAnim.from : curVal}">` +
          `<span>→ to</span>` +
          `<input type="number" step="any" class="anim-input anim-to-${key}" value="${curAnim ? curAnim.to : curVal + 1}">`;

        animBtn.addEventListener('click', () => {
          if (node.animParams[key]) {
            delete node.animParams[key];
            animRow.classList.remove('anim-visible');
            animBtn.classList.remove('anim-on');
          } else {
            const fv = parseFloat(el.value) || 0;
            node.animParams[key] = { enabled: true, from: fv, to: fv + 1 };
            animRow.querySelector(`.anim-from-${key}`).value = node.animParams[key].from;
            animRow.querySelector(`.anim-to-${key}`).value   = node.animParams[key].to;
            animRow.classList.add('anim-visible');
            animBtn.classList.add('anim-on');
          }
          gSave();
        });

        animRow.querySelector(`.anim-from-${key}`).addEventListener('input', function() {
          if (!node.animParams[key]) node.animParams[key] = { enabled: true, from: 0, to: 1 };
          node.animParams[key].from = parseFloat(this.value) || 0;
          gSave();
        });
        animRow.querySelector(`.anim-to-${key}`).addEventListener('input', function() {
          if (!node.animParams[key]) node.animParams[key] = { enabled: true, from: 0, to: 1 };
          node.animParams[key].to = parseFloat(this.value) || 0;
          gSave();
        });

        paramRow.appendChild(animBtn);
        paramRow.appendChild(kfBtn);
        paramRow.insertAdjacentElement('afterend', animRow);
      }
    }
  }
  gParamsForm.querySelectorAll('input[type=range]').forEach(r => {
    const key = r.id.replace('p_',''), v = gParamsForm.querySelector(`#val_${key}`);
    r.addEventListener('input', () => { if (v) v.textContent = formatVal(r.value); gUpdateNodeParam(node.id, key, parseFloat(r.value)); gDoAutoGen(); });
  });
  gParamsForm.querySelectorAll('input.param-ctrl,select.param-ctrl').forEach(el => {
    const key = el.id.replace('p_','');
    el.addEventListener('change', () => {
      const val = el.type==='checkbox' ? el.checked : (el.type==='range'||el.type==='number') ? parseFloat(el.value) : el.value;
      gUpdateNodeParam(node.id, key, val);
      gDoAutoGen();
    });
  });
  // Wire GLSL editor Apply buttons
  gParamsForm.querySelectorAll('.glsl-apply-btn').forEach(btn => {
    const key = btn.id.replace('glsl-apply-', '');
    const ta  = document.getElementById(`p_${key}`);
    if (!ta) return;
    // Tab key → insert 2 spaces instead of moving focus
    ta.addEventListener('keydown', e => {
      if (e.key === 'Tab') {
        e.preventDefault();
        const s = ta.selectionStart, end = ta.selectionEnd;
        ta.value = ta.value.slice(0, s) + '  ' + ta.value.slice(end);
        ta.selectionStart = ta.selectionEnd = s + 2;
      }
    });
    btn.addEventListener('click', () => {
      gUpdateNodeParam(node.id, key, ta.value);
      if (gLiveMode) { gDoAutoGen(); } else { gDoRun(); }
    });
  });
  // Wire color swatch ↔ text field sync in graph node params
  gParamsForm.querySelectorAll('.color-swatch').forEach(swatch => {
    const textId = swatch.id.replace('_swatch', '');
    const textField = document.getElementById(textId);
    if (!textField) return;
    const key = textId.replace('p_', '');
    const defVal = textField.value;
    const isHexDefault = /^#/.test(defVal);
    const isFloatDefault = defVal.split(',').every(p => { const n = parseFloat(p.trim()); return !isNaN(n) && n <= 1; });
    // Swatch → text field + save
    swatch.addEventListener('input', () => {
      const hex = swatch.value;
      let newVal;
      if (isHexDefault) {
        newVal = hex;
      } else {
        const r = parseInt(hex.slice(1,3), 16);
        const g = parseInt(hex.slice(3,5), 16);
        const b = parseInt(hex.slice(5,7), 16);
        newVal = isFloatDefault ? `${(r/255).toFixed(3)},${(g/255).toFixed(3)},${(b/255).toFixed(3)}` : `${r},${g},${b}`;
      }
      textField.value = newVal;
      gUpdateNodeParam(node.id, key, newVal);
      gDoAutoGen();
    });
    // Text field → swatch + save on change
    textField.addEventListener('change', () => {
      const hex = _parseColorToHex(textField.value);
      swatch.value = hex;
      gUpdateNodeParam(node.id, key, textField.value);
      gDoAutoGen();
    });
  });
  gRefreshParamOverrides(node.id);

  // ── Per-node timing offset spinners ────────────────────────────
  const timingSection = document.createElement('div');
  timingSection.style.cssText = 'margin-top:12px;border-top:1px solid var(--border);padding-top:10px;';
  timingSection.innerHTML = `
    <div style="font-size:11px;font-weight:700;color:var(--accent);margin-bottom:6px;">⏱ Timing Offset</div>
    <div style="display:flex;gap:8px;align-items:center;flex-wrap:wrap;">
      <label style="font-size:10px;color:var(--muted);">Start</label>
      <input id="node-start-frame" type="number" min="0" value="${node.start_frame || 0}" style="width:48px;background:var(--bg2);border:1px solid var(--border);border-radius:3px;color:var(--text);padding:2px 4px;font-size:10px;">
      <label style="font-size:10px;color:var(--muted);">End</label>
      <input id="node-end-frame" type="number" min="0" value="${node.end_frame || 0}" style="width:48px;background:var(--bg2);border:1px solid var(--border);border-radius:3px;color:var(--text);padding:2px 4px;font-size:10px;">
      <span style="font-size:9px;color:var(--muted);">(0 = use global)</span>
    </div>
    <div style="display:flex;gap:8px;align-items:center;flex-wrap:wrap;margin-top:6px;">
      <label style="font-size:10px;color:var(--muted);">Prebake</label>
      <input id="node-prebake" type="number" min="0" max="300" value="${node.prebake || 0}" style="width:48px;background:var(--bg2);border:1px solid var(--border);border-radius:3px;color:var(--text);padding:2px 4px;font-size:10px;">
      <span style="font-size:9px;color:var(--muted);">frames (run sim ahead before first output frame)</span>
    </div>
  `;
  gParamsForm.appendChild(timingSection);

  document.getElementById('node-start-frame')?.addEventListener('change', function() {
    node.start_frame = parseInt(this.value) || 0;
    gSave();
  });
  document.getElementById('node-end-frame')?.addEventListener('change', function() {
    node.end_frame = parseInt(this.value) || 0;
    gSave();
  });
  document.getElementById('node-prebake')?.addEventListener('change', function() {
    node.prebake = parseInt(this.value) || 0;
    gSave();
  });

  // ── Per-param keyframe section ──────────────────────────────
  const kfSection = document.createElement('div');
  kfSection.style.cssText = 'margin-top:16px;border-top:1px solid var(--border);padding-top:12px;';
  kfSection.innerHTML = `
    <div style="font-size:11px;font-weight:700;color:var(--accent);margin-bottom:8px;">
      <span>🎬 Per-Param Keyframes</span>
    </div>
    <div id="kf-params-list"></div>
    <div id="kf-editor" style="display:none;margin-top:8px;padding:8px;background:var(--bg0);border-radius:6px;border:1px solid var(--border);font-size:11px;">
      <div style="display:flex;gap:6px;align-items:center;margin-bottom:6px;flex-wrap:wrap;">
        <label>Param</label>
        <span id="kf-edit-param-name" style="font-weight:600;color:var(--accent);font-size:11px;"></span>
        <label>Frame</label>
        <input id="kf-edit-frame" type="number" min="0" style="width:50px;background:var(--bg2);border:1px solid var(--border);border-radius:3px;color:var(--text);padding:2px 4px;">
        <label>Value</label>
        <input id="kf-edit-value" type="number" step="any" style="width:60px;background:var(--bg2);border:1px solid var(--border);border-radius:3px;color:var(--text);padding:2px 4px;">
        <label>Easing</label>
        <select id="kf-edit-easing" style="background:var(--bg2);border:1px solid var(--border);border-radius:3px;color:var(--text);padding:2px 4px;font-size:10px;">
          <option value="linear">Linear</option>
          <option value="ease">Ease</option>
          <option value="ease-in">Ease In</option>
          <option value="ease-out">Ease Out</option>
          <option value="ease-in-out" selected>Ease In Out</option>
          <option value="step">Step</option>
          <option value="bounce">Bounce</option>
          <option value="elastic">Elastic</option>
          <option value="cubic-bezier">Cubic Bézier</option>
        </select>
        <button id="kf-edit-save" style="font-size:10px;padding:2px 8px;background:var(--accent);border:none;border-radius:3px;color:#fff;cursor:pointer;">Save</button>
        <button id="kf-edit-delete" style="font-size:10px;padding:2px 8px;background:#7f1d1d;border:none;border-radius:3px;color:var(--err);cursor:pointer;">Delete</button>
      </div>
      <!-- Bézier handle editor (shown when easing is "cubic-bezier") -->
      <div id="kf-bezier-editor" style="display:none;margin-bottom:6px;padding:6px;background:var(--bg2);border-radius:4px;">
        <div style="font-size:10px;color:var(--muted);margin-bottom:4px;">Bézier control points:</div>
        <div style="display:flex;gap:8px;align-items:center;flex-wrap:wrap;">
          <label style="font-size:10px;color:var(--muted);">P1</label>
          <input id="kf-bezier-p1x" type="number" min="0" max="1" step="0.05" value="0.42" style="width:44px;background:var(--bg3);border:1px solid var(--border);border-radius:3px;color:var(--text);padding:1px 3px;font-size:10px;">
          <input id="kf-bezier-p1y" type="number" min="0" max="1" step="0.05" value="0.0" style="width:44px;background:var(--bg3);border:1px solid var(--border);border-radius:3px;color:var(--text);padding:1px 3px;font-size:10px;">
          <label style="font-size:10px;color:var(--muted);">P2</label>
          <input id="kf-bezier-p2x" type="number" min="0" max="1" step="0.05" value="0.58" style="width:44px;background:var(--bg3);border:1px solid var(--border);border-radius:3px;color:var(--text);padding:1px 3px;font-size:10px;">
          <input id="kf-bezier-p2y" type="number" min="0" max="1" step="0.05" value="1.0" style="width:44px;background:var(--bg3);border:1px solid var(--border);border-radius:3px;color:var(--text);padding:1px 3px;font-size:10px;">
          <canvas id="kf-bezier-preview" width="60" height="60" style="border:1px solid var(--border);border-radius:3px;background:#111;cursor:pointer;"></canvas>
        </div>
      </div>
    </div>
  `;
  gParamsForm.appendChild(kfSection);

  // ── Build per-param keyframe controls ──────────────────────
  if (!node.paramKeyframes) node.paramKeyframes = {};
  renderPerParamKeyframes(node, entries);

  // ── Wire up keyframe editor save/delete ─────────────────────
  document.getElementById('kf-edit-save')?.addEventListener('click', () => {
    const pname = document.getElementById('kf-edit-param-name').dataset.pname;
    const idx = parseInt(document.getElementById('kf-edit-param-name').dataset.idx);
    if (!pname || isNaN(idx)) return;
    const kfs = node.paramKeyframes[pname] || [];
    if (idx < 0 || idx >= kfs.length) return;
    const kf = kfs[idx];
    kf.frame = parseInt(document.getElementById('kf-edit-frame').value) || 0;
    kf.value = parseFloat(document.getElementById('kf-edit-value').value) || 0;
    kf.easing = document.getElementById('kf-edit-easing').value;
    if (kf.easing === 'cubic-bezier') {
      kf.handle_in = [parseFloat(p1x.value) || 0.42, parseFloat(p1y.value) || 0.0];
      kf.handle_out = [parseFloat(p2x.value) || 0.58, parseFloat(p2y.value) || 1.0];
    } else {
      kf.handle_in = null;
      kf.handle_out = null;
    }
    kfs.sort((a, b) => a.frame - b.frame);
    gSave();
    renderPerParamKeyframes(node, entries);
    renderTimelineRuler();
  });
  document.getElementById('kf-edit-delete')?.addEventListener('click', () => {
    const pname = document.getElementById('kf-edit-param-name').dataset.pname;
    const idx = parseInt(document.getElementById('kf-edit-param-name').dataset.idx);
    if (!pname || isNaN(idx)) return;
    const kfs = node.paramKeyframes[pname] || [];
    if (idx < 0 || idx >= kfs.length) return;
    kfs.splice(idx, 1);
    if (kfs.length === 0) delete node.paramKeyframes[pname];
    gSave();
    renderPerParamKeyframes(node, entries);
    renderTimelineRuler();
    document.getElementById('kf-editor').style.display = 'none';
  });
}

function renderPerParamKeyframes(node, entries) {
  const list = document.getElementById('kf-params-list');
  if (!list) return;
  const pkf = node.paramKeyframes || {};
  const hasAny = Object.values(pkf).some(kfs => kfs && kfs.length > 0);
  if (!hasAny) {
    list.innerHTML = '<div style="font-size:11px;color:var(--muted);padding:4px 0;">Click the 🔑 button next to any param to add a keyframe at the current frame.</div>';
    return;
  }
  let html = '';
  for (const [key, spec] of entries) {
    const kfs = pkf[key] || [];
    if (kfs.length === 0) continue;
    html += `<div style="margin-bottom:6px;padding:4px 6px;background:var(--bg2);border-radius:4px;border-left:2px solid var(--accent);">
      <div style="font-size:10px;font-weight:600;color:var(--accent);margin-bottom:3px;">${key}</div>
      <div style="display:flex;flex-wrap:wrap;gap:3px;">`;
    kfs.forEach((kf, i) => {
      const val = typeof kf.value === 'number' ? kf.value.toFixed(2) : kf.value;
      html += `<div class="kf-chip" data-pname="${key}" data-idx="${i}" style="display:inline-flex;align-items:center;gap:3px;padding:2px 5px;background:var(--bg3);border-radius:3px;font-size:10px;cursor:pointer;border:1px solid var(--border);">
        <span style="font-weight:600;">F${kf.frame}</span>
        <span style="color:var(--muted);">=${val}</span>
        <span style="color:var(--muted);font-size:8px;">${kf.easing || 'linear'}</span>
      </div>`;
    });
    html += `</div></div>`;
  }
  list.innerHTML = html;
  list.querySelectorAll('.kf-chip').forEach(el => {
    el.addEventListener('click', () => {
      const pname = el.dataset.pname;
      const idx = parseInt(el.dataset.idx);
      const kf = (node.paramKeyframes[pname] || [])[idx];
      if (!kf) return;
      openPerParamKfEditor(node, pname, idx, kf);
    });
  });
}

function openPerParamKfEditor(node, pname, idx, kf) {
  const editor = document.getElementById('kf-editor');
  if (!editor) return;
  editor.style.display = 'block';
  const nameEl = document.getElementById('kf-edit-param-name');
  nameEl.textContent = pname;
  nameEl.dataset.pname = pname;
  nameEl.dataset.idx = idx;
  document.getElementById('kf-edit-frame').value = kf.frame;
  document.getElementById('kf-edit-value').value = kf.value;
  document.getElementById('kf-edit-easing').value = kf.easing || 'ease-in-out';

  // ── Bézier handle editor ──────────────────────────────────────
  const bezierDiv = document.getElementById('kf-bezier-editor');
  const easingSelect = document.getElementById('kf-edit-easing');
  const p1x = document.getElementById('kf-bezier-p1x');
  const p1y = document.getElementById('kf-bezier-p1y');
  const p2x = document.getElementById('kf-bezier-p2x');
  const p2y = document.getElementById('kf-bezier-p2y');
  const canvas = document.getElementById('kf-bezier-preview');

  function showHideBezier() {
    bezierDiv.style.display = easingSelect.value === 'cubic-bezier' ? 'block' : 'none';
  }
  showHideBezier();
  easingSelect.onchange = showHideBezier;

  if (kf.handle_in) { p1x.value = kf.handle_in[0]; p1y.value = kf.handle_in[1]; }
  else { p1x.value = 0.42; p1y.value = 0.0; }
  if (kf.handle_out) { p2x.value = kf.handle_out[0]; p2y.value = kf.handle_out[1]; }
  else { p2x.value = 0.58; p2y.value = 1.0; }

  function drawBezierPreview() {
    const ctx = canvas.getContext('2d');
    const w = canvas.width, h = canvas.height;
    ctx.clearRect(0, 0, w, h);
    ctx.strokeStyle = '#333'; ctx.lineWidth = 0.5;
    for (let i = 0; i <= 4; i++) {
      const p = i * w / 4;
      ctx.beginPath(); ctx.moveTo(p, 0); ctx.lineTo(p, h); ctx.stroke();
      ctx.beginPath(); ctx.moveTo(0, p); ctx.lineTo(w, p); ctx.stroke();
    }
    const cx1 = parseFloat(p1x.value) || 0.42;
    const cy1 = 1.0 - (parseFloat(p1y.value) || 0.0);
    const cx2 = parseFloat(p2x.value) || 0.58;
    const cy2 = 1.0 - (parseFloat(p2y.value) || 1.0);
    ctx.strokeStyle = getComputedStyle(document.documentElement).getPropertyValue('--accent').trim() || '#64748b'; ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.moveTo(0, h);
    for (let t = 0; t <= 1; t += 0.02) {
      const mt = 1 - t;
      const x = 3*mt*mt*t*cx1 + 3*mt*t*t*cx2 + t*t*t;
      const y = 3*mt*mt*t*cy1 + 3*mt*t*t*cy2 + t*t*t;
      ctx.lineTo(x * w, y * h);
    }
    ctx.stroke();
    ctx.fillStyle = '#f59e0b';
    ctx.beginPath(); ctx.arc(cx1 * w, cy1 * h, 4, 0, Math.PI*2); ctx.fill();
    ctx.fillStyle = '#34d399';
    ctx.beginPath(); ctx.arc(cx2 * w, cy2 * h, 4, 0, Math.PI*2); ctx.fill();
    ctx.strokeStyle = '#555'; ctx.lineWidth = 1; ctx.setLineDash([2,2]);
    ctx.beginPath(); ctx.moveTo(0, h); ctx.lineTo(cx1 * w, cy1 * h); ctx.stroke();
    ctx.beginPath(); ctx.moveTo(w, 0); ctx.lineTo(cx2 * w, cy2 * h); ctx.stroke();
    ctx.setLineDash([]);
  }
  drawBezierPreview();
  [p1x, p1y, p2x, p2y].forEach(el => el.addEventListener('input', drawBezierPreview));
  canvas.addEventListener('click', (e) => {
    const rect = canvas.getBoundingClientRect();
    const x = (e.clientX - rect.left) / rect.width;
    const y = 1.0 - (e.clientY - rect.top) / rect.height;
    const d1 = Math.hypot(x - parseFloat(p1x.value), y - parseFloat(p1y.value));
    const d2 = Math.hypot(x - parseFloat(p2x.value), y - parseFloat(p2y.value));
    if (d1 < d2) { p1x.value = Math.round(x*100)/100; p1y.value = Math.round(y*100)/100; }
    else { p2x.value = Math.round(x*100)/100; p2y.value = Math.round(y*100)/100; }
    drawBezierPreview();
  });
}
function gUpdateNodeParam(nodeId, key, val) {
  const n = gNodes.find(n => n.id===nodeId); if (n) { n.params[key]=val; n.dirty=true; gSave(); }
  // Keep the 3D viewport editor in sync with slider edits.
  if (window._gEditor3D && window._gEditor3D.isOpen() && !window._gEditor3DWriting)
    window._gEditor3D.refresh();
}

// ── Context menus ──────────────────────────────────────────────
let gCtxTarget = null;
function gShowEdgeCtx(eid, x, y) {
  gSelectedEdge = eid; gCtxTarget = 'edge';
  document.getElementById('ctx-del-node').style.display = 'none';
  document.getElementById('ctx-del-edge').style.display = '';
  document.getElementById('ctx-feedback').style.display = '';
  const edge = gEdges.find(e => e.id===eid);
  document.getElementById('ctx-feedback').textContent = edge?.feedback ? 'Remove feedback' : 'Mark as feedback';
  gCtxMenu.style.cssText = `left:${x}px;top:${y}px;display:`;
}
document.getElementById('ctx-feedback').addEventListener('click', () => {
  const e = gEdges.find(e => e.id===gSelectedEdge);
  if (e) { e.feedback=!e.feedback; gRedrawEdges(); gSave(); }
  gCtxMenu.style.display='none';
});
document.getElementById('ctx-del-edge').addEventListener('click', () => { if (gSelectedEdge) gDeleteEdge(gSelectedEdge); gCtxMenu.style.display='none'; });
document.getElementById('ctx-del-node').addEventListener('click', () => { if (gSelectedNode) gDeleteNode(gSelectedNode); gCtxMenu.style.display='none'; });
gNodesEl.addEventListener('contextmenu', e => {
  const nodeEl = e.target.closest('.gnode');
  if (!nodeEl) return;
  e.preventDefault();
  const nodeId = nodeEl.id.replace('gnode-', '');
  gSelectNode(nodeId);
  const node = gNodes.find(n => n.id === nodeId);
  const isGroup = node?.type === 'group';
  const hasMultiSel = gSelectedNodes.size > 1;
  document.getElementById('ctx-del-node').style.display = '';
  document.getElementById('ctx-del-edge').style.display = 'none';
  document.getElementById('ctx-feedback').style.display = 'none';
  document.getElementById('ctx-group-sel').style.display = hasMultiSel ? '' : 'none';
  document.getElementById('ctx-ungroup').style.display = isGroup ? '' : 'none';
  gCtxMenu.style.left=e.clientX+'px'; gCtxMenu.style.top=e.clientY+'px'; gCtxMenu.style.display='';
});
document.getElementById('ctx-group-sel').addEventListener('click', () => { gCtxMenu.style.display='none'; gGroupSelectedNodes(); });
document.getElementById('ctx-ungroup').addEventListener('click', () => { gCtxMenu.style.display='none'; if (gSelectedNode) gUngroup(gSelectedNode); });
document.addEventListener('click', () => { gCtxMenu.style.display='none'; });

// ── Client-side expression evaluator (preview only) ───────────
function _evalExprClient(expr, frame, seed) {
  const t = frame / 100;
  try {
    const fn = new Function(
      'frame','seed','t','sin','cos','tan','pi','e','abs','floor','ceil',
      'round','sqrt','log','pow','min','max','noise',
      `return (${expr})`
    );
    const result = fn(
      frame, seed, t,
      Math.sin, Math.cos, Math.tan, Math.PI, Math.E, Math.abs,
      Math.floor, Math.ceil, Math.round, Math.sqrt, Math.log, Math.pow,
      Math.min, Math.max,
      x => Math.sin(x * 127.1 + 311.7) * 0.5 + 0.5
    );
    return typeof result === 'number' && isFinite(result) ? result : null;
  } catch(e) {
    return null;
  }
}

// ── Per-param expression toggle ────────────────────────────────
function gSetupExprToggle(key, spec, node, el) {
  const paramRow = el.closest('.param-row');
  if (!paramRow) return;

  const rangeWrap = el.type === 'range' ? el.closest('.param-range-wrap') : null;
  const targetEl  = rangeWrap || el;

  const exprInput = document.createElement('input');
  exprInput.type  = 'text';
  exprInput.className = 'param-input expr-input';
  exprInput.placeholder = 'sin(frame * 0.1)';
  exprInput.style.display = 'none';

  const exprBtn = document.createElement('button');
  exprBtn.className = 'expr-btn';
  exprBtn.title = 'Expression mode — variables: frame, seed, t; functions: sin, cos, pi, sqrt, …';
  exprBtn.textContent = 'ƒ';

  targetEl.insertAdjacentElement('afterend', exprInput);

  const isExprMode = typeof node.params[key] === 'string' && node.params[key] !== '';

  function activateExpr(formula) {
    exprInput.value = String(formula);
    targetEl.style.display = 'none';
    exprInput.style.display = '';
    exprBtn.classList.add('expr-on');
    node.params[key] = exprInput.value;
    node.dirty = true;
    _updateExprPreview();
  }

  function deactivateExpr() {
    const fv = parseFloat(exprInput.value);
    const newVal = isNaN(fv) ? (spec.default ?? 0) : fv;
    el.value = newVal;
    if (rangeWrap) {
      const vs = rangeWrap.querySelector(`#val_${key}`);
      if (vs) vs.textContent = formatVal(newVal);
    }
    exprInput.style.display = 'none';
    targetEl.style.display = '';
    exprBtn.classList.remove('expr-on');
    node.params[key] = newVal;
    node.dirty = true;
    gSave();
  }

  function _updateExprPreview() {
    const frame = parseInt(document.getElementById('tl-frame')?.value) || 0;
    const result = _evalExprClient(exprInput.value, frame, 42);
    exprInput.title = result !== null
      ? `≈ ${result.toFixed(4)} at frame ${frame}`
      : 'invalid expression';
  }

  if (isExprMode) {
    activateExpr(node.params[key]);
  }

  exprBtn.addEventListener('click', () => {
    if (exprBtn.classList.contains('expr-on')) {
      deactivateExpr();
    } else {
      activateExpr(String(el.value ?? spec.default ?? 0));
    }
    gSave();
  });

  exprInput.addEventListener('input', () => {
    node.params[key] = exprInput.value;
    node.dirty = true;
    _updateExprPreview();
    gSave();
    gDoAutoGen();
  });

  exprInput.addEventListener('blur', _updateExprPreview);

  paramRow.appendChild(exprBtn);
}

// ── Per-param linear animation ─────────────────────────────────
function gGetAnimatedParams(node, frame) {
  const start = parseInt(document.getElementById('tl-start').value) || 0;
  const end   = parseInt(document.getElementById('tl-end').value)   || 24;
  const t     = end > start ? (frame - start) / (end - start) : 0;
  const params = { ...node.params };
  if (node.animParams) {
    for (const [key, anim] of Object.entries(node.animParams)) {
      if (anim.enabled) {
        params[key] = anim.from + (anim.to - anim.from) * t;
      }
    }
  }
  return params;
}

// ── Client-side render display helper ──────────────────────────
// Mounts the three.js executor canvas in the main preview, replacing whatever
// server-fed <img>/<canvas> was there. Used by both client Run and client Live.
// Graphs with a 3D scene support the orbit viewport (#4a).
function gGraphHas3DScene() {
  return gNodes.some(n => n.method_id === '__scene3d__' || n.method_id === '__scene_render__');
}

function gMountClientCanvas(canvas) {
  if (canvas._mounted && canvas.parentNode === gMainPreview) return;
  gMainPreview.innerHTML = '';
  gLivePreviewImg = null; gLiveCanvas = null; gLiveImg = null;
  canvas.id = 'client3d-canvas';
  canvas.style.maxWidth = '100%';
  canvas.style.maxHeight = '100%';
  canvas.style.objectFit = 'contain';
  canvas.style.display = 'block';
  canvas.style.margin = 'auto';

  // ── Orbit viewport (#4a): drag to orbit, wheel to dolly — for 3D graphs.
  // A plain click (no drag) still opens the fullscreen viewer.
  let dragging = false, moved = false, lastX = 0, lastY = 0;
  canvas.style.touchAction = 'none';
  canvas.addEventListener('pointerdown', e => {
    if (!gGraphHas3DScene()) return;
    dragging = true; moved = false; lastX = e.clientX; lastY = e.clientY;
    canvas.setPointerCapture(e.pointerId);
    canvas.style.cursor = 'grabbing';
  });
  canvas.addEventListener('pointermove', e => {
    if (!dragging) return;
    const dx = e.clientX - lastX, dy = e.clientY - lastY;
    if (Math.abs(dx) + Math.abs(dy) > 2) moved = true;
    lastX = e.clientX; lastY = e.clientY;
    if (_gClient3D && _gClient3D.orbitRotate) _gClient3D.orbitRotate(dx * 0.008, -dy * 0.008);
    const badge = document.getElementById('orbit-badge'); if (badge) badge.textContent = '⟲ orbiting — dblclick to reset';
  });
  canvas.addEventListener('pointerup', e => {
    dragging = false; canvas.style.cursor = gGraphHas3DScene() ? 'grab' : '';
    try { canvas.releasePointerCapture(e.pointerId); } catch {}
  });
  canvas.addEventListener('wheel', e => {
    if (!gGraphHas3DScene() || !_gClient3D || !_gClient3D.orbitDolly) return;
    e.preventDefault();
    _gClient3D.orbitDolly(e.deltaY > 0 ? 1.08 : 0.926);
  }, { passive: false });
  canvas.addEventListener('dblclick', () => {
    if (_gClient3D && _gClient3D.orbitReset) _gClient3D.orbitReset();
    const badge = document.getElementById('orbit-badge'); if (badge) badge.textContent = '⟲ drag to orbit · scroll to zoom';
  });
  canvas.onclick = () => {
    if (moved) { moved = false; return; } // a drag, not a click — don't fullscreen
    gOpenFullscreen();
  };
  canvas.style.cursor = gGraphHas3DScene() ? 'grab' : '';

  canvas._mounted = true;
  gMainPreview.classList.add('active');
  gMainPreview.appendChild(canvas);
  // Orbit hint badge (only for 3D graphs).
  if (gGraphHas3DScene()) {
    const badge = document.createElement('div');
    badge.id = 'orbit-badge';
    badge.textContent = '⟲ drag to orbit · scroll to zoom';
    badge.style.cssText = 'position:absolute;left:10px;bottom:10px;z-index:5;'
      + 'font-size:11px;padding:4px 9px;border-radius:10px;pointer-events:none;'
      + 'background:rgba(20,24,34,.72);color:#9fb4d8;font-weight:600;backdrop-filter:blur(4px);';
    gMainPreview.appendChild(badge);
  }
  gPreviewShow();
}

// One-shot client render of the current graph at the timeline start frame.
async function gClientRunOnce() {
  const C = await gClient3D();
  const start = parseInt(document.getElementById('tl-start').value) || 0;
  const fps   = parseInt(document.getElementById('tl-fps')?.value) || 24;
  const nodes = gNodes.map(n => gSerializeNodeForApi(n));
  const edges = gEdges.map(e => ({ src_node:e.src_node, src_port:e.src_port, dst_node:e.dst_node, dst_port:e.dst_port, feedback:e.feedback }));
  const canvas = await C.renderFrame(nodes, edges, start, gCanvasW, gCanvasH, start / fps);
  gMountClientCanvas(canvas);
  gClientSurfaceErrors();
  const nodeErrs = C.getNodeErrors ? C.getNodeErrors() : {};
  const err = C.lastError() || Object.values(nodeErrs)[0];
  gSetStatus(err ? ('Client render · error (see node panel)') : `Client render · frame ${start} · ${gCanvasW}×${gCanvasH}`);
  if (err) console.warn('[client3d] error:', err);
}

// ── Run ────────────────────────────────────────────────────────
async function gDoRun() {
  if (!gNodes.length) { gSetStatus('No nodes.'); return; }
  // Client-rendered graphs (3D nodes) never hit the server render path.
  if (gGraphRunsOnClient()) {
    try { await gClientRunOnce(); } catch(e) { gSetStatus('Client render error: '+e.message); console.error(e); }
    return;
  }
  // Clear any previous node error states
  document.querySelectorAll('.node-error').forEach(el => {
    el.classList.remove('node-error');
    delete el.dataset.errorMsg;
  });
  const srcSet = new Set(gEdges.filter(e => !e.feedback).map(e => e.src_node));
  const hasRenderFlag = gNodes.some(n => n.render);
  if (!hasRenderFlag && !gNodes.some(n => !srcSet.has(n.id))) {
    gSetStatus('No output node — set a render flag (◎) or leave a node with no outgoing connections'); return;
  }
  gSetRunDisabled(true);
  gSetStatus('Running…');
  const tlStartVal = parseInt(document.getElementById('tl-start').value) || 0;
  const tlEndVal   = parseInt(document.getElementById('tl-end').value)   || 24;
  const totalFrames = Math.max(1, tlEndVal - tlStartVal + 1);
  const body = {
    ...gServerGraphPayload(tlStartVal),
    seed: 42, frames: totalFrames,
    frame: tlStartVal,
    width: gCanvasW, height: gCanvasH,
  };
  try {
    const res = await fetch('/api/graph/execute', { method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify(body) });
    gListenGraphSSE((await res.json()).job_id);
  } catch(e) { gSetStatus('Error: '+e.message); gSetRunDisabled(false); }
}
gRunBtn.addEventListener('click', gDoRun);
gRunBtnDesk.addEventListener('click', gDoRun);

// ── Auto-generate toggle ─────────────────────────────────────
const gAutoBtn = document.getElementById('graph-auto-btn');
const gAutoBtnDesk = document.getElementById('graph-auto-btn-desk');
function gToggleAutoGen() {
  gAutoGen = !gAutoGen;
  gAutoBtn.classList.toggle('active', gAutoGen);
  gAutoBtnDesk.classList.toggle('active', gAutoGen);
  gAutoBtn.textContent = gAutoGen ? '⏹ Auto' : '⏺ Auto';
  gAutoBtnDesk.textContent = gAutoGen ? '⏹ Auto' : '⏺ Auto';
  try { localStorage.setItem('graph-auto-gen', gAutoGen ? '1' : '0'); } catch {}
}
// Restore saved state
try { if (localStorage.getItem('graph-auto-gen') === '1') gToggleAutoGen(); } catch {}
gAutoBtn.addEventListener('click', gToggleAutoGen);
gAutoBtnDesk.addEventListener('click', gToggleAutoGen);

// ── 3D viewport editor (ui/js/editor3d.js) ─────────────────────
// Scene-editor view over the graph's 3D element nodes: orbit camera,
// click-select, transform gizmos. Gizmo edits write straight back into node
// params (single source of truth); slider edits flow back via gUpdateNodeParam.
window._gEditor3D = null;          // module handle while open
window._gEditor3DWriting = false;  // re-entrancy guard: gizmo → params → refresh
window._gEditor3DSelecting = false;

// The viewport docks beside the graph in #graph-workspace rather than taking
// over the output preview, so a 3D scene and a rendered frame can be on screen
// at once and live mode no longer has to be stopped to look at the scene.
// (gWorkspace / gVpdStage are declared with the other DOM refs above.)
async function gToggleEditor3D() {
  const btns = [document.getElementById('graph-edit3d-btn'),
                document.getElementById('graph-edit3d-btn-desk')].filter(Boolean);
  if (window._gEditor3D && window._gEditor3D.isOpen()) {
    window._gEditor3D.close();
    window._gEditor3D = null;
    btns.forEach(b => b.classList.remove('active'));
    gWorkspace.classList.remove('vpd-open');
    localStorage.setItem('vpd-open', '0');
    gSetStatus('3D viewport closed');
    return;
  }
  const ED = await import('/ui/js/editor3d.js');
  // Stop the browser-GPU live loop, not the server one. client3d.js owns its
  // own WebGLRenderer and RAF loop over the same 3D nodes the editor is about
  // to render, so running both means two GL contexts competing every frame.
  // The server live stream only blits JPEG frames, costs no GL context, and is
  // exactly what the dock exists to sit beside — so it keeps running.
  _gStopClientLive();
  // Show the dock before open() so the stage has real dimensions to size the
  // renderer against — measuring a display:none container yields 0×0.
  gWorkspace.classList.add('vpd-open');
  localStorage.setItem('vpd-open', '1');
  await ED.open({
    container: gVpdStage,
    getGraph: () => ({ nodes: gNodes, edges: gEdges }),
    onParamChange: (nodeId, patch) => {
      window._gEditor3DWriting = true;
      try {
        for (const [k, v] of Object.entries(patch)) gUpdateNodeParam(nodeId, k, v);
        gClientLiveRefresh();
      } finally {
        window._gEditor3DWriting = false;
      }
      // Refresh the param panel so sliders track the gizmo — debounced,
      // because objectChange fires per mousemove during a drag.
      clearTimeout(window._gEditor3DPanelT);
      window._gEditor3DPanelT = setTimeout(() => {
        if (gSelectedNode === nodeId) {
          const node = gNodes.find(n => n.id === nodeId);
          if (node) gShowNodeParams(node);
        }
      }, 150);
    },
    onSelectNode: (nodeId) => {
      window._gEditor3DSelecting = true;
      try { gSelectNode(nodeId); } finally { window._gEditor3DSelecting = false; }
    },
  });
  window._gEditor3D = ED;
  btns.forEach(b => b.classList.add('active'));
  gSetStatus('3D viewport — click to select · W/E/R gizmo modes · F frame');
}
document.getElementById('graph-edit3d-btn')?.addEventListener('click', gToggleEditor3D);
document.getElementById('graph-edit3d-btn-desk')?.addEventListener('click', gToggleEditor3D);
document.getElementById('vpd-close')?.addEventListener('click', gToggleEditor3D);

// ── Viewport splitter ──────────────────────────────────────────
// Width lives in a CSS var so the drag never touches layout classes, and
// editor3d's own ResizeObserver picks the new size up on its next frame.
(function () {
  const splitter = document.getElementById('viewport-splitter');
  if (!splitter) return;
  const MIN = 220, MIN_GRAPH = 260;

  const saved = parseInt(localStorage.getItem('vpd-w') || '', 10);
  if (Number.isFinite(saved)) gWorkspace.style.setProperty('--vpd-w', saved + 'px');

  splitter.addEventListener('pointerdown', (e) => {
    e.preventDefault();
    splitter.setPointerCapture(e.pointerId);
    splitter.classList.add('dragging');

    const onMove = (ev) => {
      const rect = gWorkspace.getBoundingClientRect();
      // Clamp against both edges so neither pane can be dragged out of
      // existence. The splitter sits between them, so its own width has to
      // come out of the budget or the graph ends up MIN_GRAPH minus 6px.
      const avail = rect.width - splitter.offsetWidth;
      const w = Math.round(Math.min(
        Math.max(rect.right - ev.clientX, MIN),
        avail - MIN_GRAPH));
      gWorkspace.style.setProperty('--vpd-w', w + 'px');
    };
    const onUp = (ev) => {
      splitter.releasePointerCapture(ev.pointerId);
      splitter.classList.remove('dragging');
      splitter.removeEventListener('pointermove', onMove);
      splitter.removeEventListener('pointerup', onUp);
      const cur = parseInt(gWorkspace.style.getPropertyValue('--vpd-w'), 10);
      if (Number.isFinite(cur)) localStorage.setItem('vpd-w', String(cur));
    };
    splitter.addEventListener('pointermove', onMove);
    splitter.addEventListener('pointerup', onUp);
  });
})();

// Restore the dock across reloads — a workspace pane the user opened should
// still be there next session, the way the palette and preview sizes are.
if (localStorage.getItem('vpd-open') === '1') {
  window.addEventListener('load', () => { gToggleEditor3D(); });
}

// ── Live mode (WebSocket primary / MJPEG fallback) ─────────────
const gLiveBtn = document.getElementById('graph-live-btn');
const gLiveBtnDesk = document.getElementById('graph-live-btn-desk');
let gLiveMode = false;
let gLiveImg = null;          // MJPEG fallback <img>
let gLiveCanvas = null;       // WS primary <canvas>
let _gLiveWs = null;          // active WebSocket
let _gLiveWsPingTimer = null; // keepalive interval
let _gLiveWsFallback = false; // true once WS failed → using MJPEG
let _gLiveStatTimer = null;   // poll timer (fallback only)

// ── Live frame metadata readout ────────────────────────────────
// Looked up per frame, not cached: gSetLive rebuilds the preview's innerHTML,
// which recreates #live-meta-readout — cached references would go stale and
// the readout would sit at "frame –" forever.
function _gUpdateLiveMeta(msg) {
  if (!msg) return;
  const _lmrFrame = document.getElementById('lmr-frame');
  const _lmrMs    = document.getElementById('lmr-ms');
  const _lmrFps   = document.getElementById('lmr-fps');
  const _lmrErr   = document.getElementById('lmr-err');
  const _lmrBars  = document.getElementById('lmr-bars');
  if (_lmrFrame) _lmrFrame.textContent = 'frame ' + msg.frame;
  if (_lmrMs)    _lmrMs.textContent    = msg.cook_ms + 'ms';
  if (_lmrFps)   _lmrFps.textContent   = msg.fps + ' fps';
  // Phase 5: sync timeline frame cursor to live WS frame
  if (gLiveMode && typeof msg.frame === 'number') {
    const tlFrameEl = document.getElementById('tl-frame');
    const tlEndEl   = document.getElementById('tl-end');
    if (tlFrameEl) {
      tlFrameEl.value = msg.frame;
      // Auto-extend timeline end so playhead stays on-screen
      if (tlEndEl && msg.frame > (parseInt(tlEndEl.value) || 24)) {
        tlEndEl.value = msg.frame + 24;
        renderTimelineRuler();
      } else {
        updatePlayhead();
      }
    }
  }
  if (_lmrErr) {
    const hasErr = msg.node_errors && Object.keys(msg.node_errors).length > 0;
    _lmrErr.style.display = hasErr ? '' : 'none';
  }
  // Surface GLSL compile errors into the selected node's params panel
  if (msg.node_errors && gSelectedNode) {
    const rawErr = msg.node_errors[gSelectedNode] || '';
    const errDiv = document.getElementById('glsl-err-glsl_code');
    if (errDiv) {
      // Strip Python traceback — keep only lines that look like GLSL errors
      const glslLines = rawErr.split('\n').filter(l =>
        /^\d+:\d+|error:|ERROR:|fragment|0\(/.test(l)
      );
      const errText = glslLines.length ? glslLines.join('\n') : (rawErr ? rawErr.split('\n').slice(-3).join('\n') : '');
      errDiv.textContent = errText;
      errDiv.style.display = errText ? '' : 'none';
    }
  }
  // Mini timing bars (one bar per node, height proportional to its share)
  if (_lmrBars && msg.node_timings) {
    const entries = Object.entries(msg.node_timings);
    const maxMs = entries.reduce((m, [,v]) => Math.max(m, v), 1);
    _lmrBars.innerHTML = '';
    entries.forEach(([nid, ms]) => {
      const bar = document.createElement('div');
      const pct = Math.max(4, Math.round((ms / maxMs) * 100));
      const isGpu = (msg.gpu_nodes || 0) > 0 && (msg.node_names||{})[nid]?.toLowerCase().startsWith('gpu');
      bar.className = 'lmr-bar' + (isGpu ? ' gpu' : '');
      bar.style.height = pct + '%';
      bar.title = ((msg.node_names||{})[nid] || nid) + ': ' + ms.toFixed(1) + 'ms';
      _lmrBars.appendChild(bar);
    });
  }
  // Per-node meters + heat glow on the graph canvas, edge-flow march
  gApplyNodeMetrics(msg);
  // Feed into Diagnostics tab — diagRender is exposed globally from the IIFE
  if (typeof window.diagRender === 'function') {
    const paused = typeof window.diagIsPaused === 'function' ? window.diagIsPaused() : false;
    if (!paused) window.diagRender({...msg, running: true, sim_cache_entries: 0});
  }
}

// ── Live node metrics on the canvas ────────────────────────────
// Each WS frame carries node_timings {id→ms}. Every node gets a meter strip
// (icon · cook ms · share-of-frame bar) coloured by its share of the frame,
// the hottest node breathes, cached/reused nodes frost over, and the wires
// march at a tempo derived from the measured fps.
const _gGraphSvg = document.getElementById('graph-svg');
// graph.py records a tiny "reuse cost" for cache-skipped nodes so node_timings
// stays complete — below this threshold the node didn't actually cook.
const _G_REUSE_MS = 0.08;
let _gMetricsLastApply = 0, _gMetricsLastFrame = 0;

function gApplyNodeMetrics(msg) {
  if (!msg.node_timings || !gNodes.length) return;
  _gMetricsLastFrame = performance.now();
  // Edge-flow tempo tracks fps (faster frames → faster march), clamped sane
  _gGraphSvg.classList.add('flowing');
  if (msg.fps > 0) {
    const dur = Math.min(2, Math.max(0.25, 18 / msg.fps));
    _gGraphSvg.style.setProperty('--flow-dur', dur.toFixed(2) + 's');
  }
  // Throttle the DOM writes — WS frames can arrive at 30+ fps
  if (_gMetricsLastFrame - _gMetricsLastApply < 120) return;
  _gMetricsLastApply = _gMetricsLastFrame;

  const t = msg.node_timings, errs = msg.node_errors || {};
  let total = 0, hotId = null, hotMs = -1;
  for (const [nid, ms] of Object.entries(t)) {
    total += ms;
    if (ms > hotMs) { hotMs = ms; hotId = nid; }
  }
  total = total || 1;

  for (const n of gNodes) {
    const el = document.getElementById('gnode-' + n.id);
    if (!el) continue;
    let meter = el.querySelector('.gnode-meter');
    if (!meter) {
      meter = document.createElement('div');
      meter.className = 'gnode-meter';
      meter.innerHTML = '<span class="gnm-ico"></span><span class="gnm-ms">–</span><div class="gnm-bar"><i></i></div>';
      el.appendChild(meter);
    }
    const ms     = t[n.id];
    const cooked = ms !== undefined && ms > _G_REUSE_MS;
    const cached = ms !== undefined && !cooked;
    const share  = cooked ? ms / total : 0;
    const err    = errs[n.id];
    el.classList.toggle('gnode-cached', cached && !err);
    el.classList.toggle('heat-cool', cooked && share < 0.18);
    el.classList.toggle('heat-warm', cooked && share >= 0.18 && share < 0.45);
    el.classList.toggle('heat-hot',  cooked && share >= 0.45);
    el.classList.toggle('gnode-hottest', cooked && n.id === hotId && hotMs >= 1);
    const def = gNodeDefs[n.method_id] || {};
    const isGpu = def.category === 'gpu_shaders'
      || ((msg.node_names || {})[n.id] || def.name || '').toLowerCase().startsWith('gpu');
    const ico = meter.querySelector('.gnm-ico');
    ico.textContent = err ? '⚠' : cached ? '❄' : isGpu ? '⚡' : '⏱';
    ico.title = err ? err.split('\n')[0]
      : cached ? 'cached — result reused this frame'
      : (isGpu ? 'GPU cook time' : 'cook time');
    meter.querySelector('.gnm-ms').textContent =
      ms === undefined ? '–' : cached ? 'cached'
      : ms >= 10 ? Math.round(ms) + 'ms' : ms.toFixed(1) + 'ms';
    meter.querySelector('.gnm-bar > i').style.width =
      cooked ? Math.max(3, Math.round(share * 100)) + '%' : '0%';
  }
}

// Stop the motion (edge march + hottest pulse) when frames stop arriving —
// covers live stop, WS drop, and render completion in one place. The last
// measured meter values stay visible on the nodes.
function gClearLiveMotion() {
  _gGraphSvg.classList.remove('flowing');
  gNodesEl.querySelectorAll('.gnode-hottest').forEach(el => el.classList.remove('gnode-hottest'));
}
setInterval(() => {
  if (_gGraphSvg.classList.contains('flowing')
      && performance.now() - _gMetricsLastFrame > 2000) gClearLiveMotion();
}, 1000);

// ── MJPEG status poll (fallback path only) ─────────────────────
async function _gPollLiveStats() {
  if (!gLiveMode || _gLiveWs) return;
  try {
    const r = await fetch('/api/graph/live/status');
    if (!r.ok) return;
    const d = await r.json();
    if (d.running) {
      const msg = `Live  ·  frame ${d.frame}  ·  ${d.cook_ms}ms  ·  ${d.fps} fps`
        + (d.fps_limit ? `  ·  limited ${d.target_fps}` : '');
      gSetStatus(msg);
    }
  } catch {}
}

// ── Live cook-rate limiter ─────────────────────────────────────
// Off, live cooks flat out (server: capped at 30fps; client GPU: every rAF).
// On, both pace their cooking to the timeline FPS field — a heavy graph stops
// burning the machine on frames nobody sees, and the preview runs at the
// tempo the sequence will export at.
const gFpsLimitEl   = document.getElementById('tl-fps-limit');
const gFpsLimitWrap = document.getElementById('tl-fps-limit-wrap');
function gLiveRate() {
  const fps = Math.max(1, parseInt(document.getElementById('tl-fps')?.value) || 24);
  return { fps, fps_limit: !!gFpsLimitEl?.checked };
}

function gLiveGraphBody(stop) {
  const { fps, fps_limit } = gLiveRate();
  return {
    ...gServerGraphPayload(),
    seed: 42, frames: stop ? 0 : 1,
    width: gCanvasW, height: gCanvasH,
    fps, fps_limit,
  };
}

// Push a rate change into whichever live path is running. Server-side this is
// a hot-swap (the loop keeps its executor and sim caches); client-side it just
// retunes the rAF loop. Off-air, there is nothing to do — the next start reads
// the field itself.
function gApplyLiveRate() {
  if (gFpsLimitWrap) gFpsLimitWrap.classList.toggle('on', !!gFpsLimitEl?.checked);
  if (!gLiveMode) return;
  if (_gClientLiveActive) { gClientApplyLiveRate(); return; }
  gLiveHotSwap();
}

function gClientApplyLiveRate() {
  if (!_gClient3D?.setLiveRate) return;
  const { fps, fps_limit } = gLiveRate();
  _gClient3D.setLiveRate({ fps, cookFps: fps_limit ? fps : 0 });
}

gFpsLimitEl?.addEventListener('change', () => {
  try { localStorage.setItem('live-fps-limit', gFpsLimitEl.checked ? '1' : ''); } catch {}
  gApplyLiveRate();
  gSetStatus(gFpsLimitEl.checked
    ? `Live cook limited to ${gLiveRate().fps} fps`
    : 'Live cook limiter off');
});
// 'change' (not 'input'): committing the field, not every keystroke, retunes
// the running loop — an in-flight POST per typed digit is pure waste.
document.getElementById('tl-fps')?.addEventListener('change', () => {
  if (gFpsLimitEl?.checked) gApplyLiveRate();
});
try {
  if (localStorage.getItem('live-fps-limit') === '1' && gFpsLimitEl) {
    gFpsLimitEl.checked = true;
    gFpsLimitWrap?.classList.add('on');
  }
} catch {}

// Push an edited graph into the already-running live loop. Transport
// (WS/MJPEG, canvas, preview chrome) is left completely alone — the server
// loop re-reads the graph every frame, so this is just a doc update. Only
// one swap is ever in flight: a newer edit aborts the older POST.
let _gLiveSwapAbort = null;
async function gLiveHotSwap() {
  if (!gLiveMode) return;
  if (_gClientLiveActive) { gClientLiveRefresh(); return; }
  if (_gLiveSwapAbort) _gLiveSwapAbort.abort();
  const ac = _gLiveSwapAbort = new AbortController();
  try {
    await fetch('/api/graph/live', {
      method: 'POST', headers: {'Content-Type': 'application/json'},
      body: JSON.stringify(gLiveGraphBody(false)),
      signal: ac.signal,
    });
  } catch (e) {
    if (e.name !== 'AbortError') gSetStatus('Live swap: ' + e.message);
  } finally {
    if (_gLiveSwapAbort === ac) _gLiveSwapAbort = null;
  }
}

// ── WebSocket live: open connection, draw frames, pass metadata ──
function _gOpenLiveWs() {
  _gLiveWsFallback = false;
  const proto = location.protocol === 'https:' ? 'wss:' : 'ws:';
  const ws = new WebSocket(`${proto}//${location.host}/api/live/ws`);

  // If no frame arrives within 3 s, fall back to MJPEG
  let _noFrameTimeout = setTimeout(() => {
    console.info('[live-ws] no frame in 3s — falling back to MJPEG');
    ws.close();
  }, 3000);

  ws.onmessage = async (evt) => {
    // Clear the no-frame watchdog on first message
    if (_noFrameTimeout) { clearTimeout(_noFrameTimeout); _noFrameTimeout = null; }

    let msg;
    try { msg = JSON.parse(evt.data); } catch { return; }
    if (!msg.img) return;

    // Draw JPEG onto the canvas
    if (gLiveCanvas) {
      try {
        const bin = atob(msg.img);
        const u8  = new Uint8Array(bin.length);
        for (let i = 0; i < bin.length; i++) u8[i] = bin.charCodeAt(i);
        const blob = new Blob([u8], { type: 'image/jpeg' });
        const bmp  = await createImageBitmap(blob);
        const ctx  = gLiveCanvas.getContext('2d');
        if (gLiveCanvas.width !== bmp.width || gLiveCanvas.height !== bmp.height) {
          gLiveCanvas.width  = bmp.width;
          gLiveCanvas.height = bmp.height;
        }
        ctx.drawImage(bmp, 0, 0);
        bmp.close();
      } catch (_) {}
    }

    // Update status bar. `fps` is cook throughput; when the limiter is on the
    // delivered rate is target_fps, so show both rather than a misleading one.
    gSetStatus(`Live · frame ${msg.frame} · ${msg.cook_ms}ms · ${msg.fps} fps`
      + (msg.fps_limit ? ` · limited ${msg.target_fps}` : ''));

    // Update live metadata readout overlay
    _gUpdateLiveMeta(msg);
  };

  ws.onerror = () => {};
  ws.onclose = () => {
    // Only clear shared state if this is still the current socket — a
    // superseded socket closing late must not orphan the live one's
    // references (that left _gTeardownLive unable to close it).
    if (_gLiveWs !== ws) return;
    clearInterval(_gLiveWsPingTimer); _gLiveWsPingTimer = null;
    _gLiveWs = null;
    // Fall back to MJPEG if live mode is still on
    if (gLiveMode && !_gLiveWsFallback) {
      _gLiveWsFallback = true;
      console.info('[live-ws] closed — switching to MJPEG fallback');
      _gStartMjpegFallback();
    }
  };

  // Keepalive ping every 15 s. Clear any previous timer first so a restart
  // can't leave an orphan interval pinging a dead socket forever.
  if (_gLiveWsPingTimer) clearInterval(_gLiveWsPingTimer);
  _gLiveWsPingTimer = setInterval(() => {
    if (ws.readyState === WebSocket.OPEN) ws.send('ping');
  }, 15000);

  _gLiveWs = ws;
}

function _gStartMjpegFallback() {
  // MJPEG: plain <img> pointing at the multipart stream
  if (gLiveImg) return; // already started
  gMainPreview.classList.add('active');
  gLiveImg = document.createElement('img');
  gLiveImg.src = '/api/live/stream?t=' + Date.now();
  gMainPreview.appendChild(gLiveImg);
  // Re-mount preview chrome that innerHTML wipes
  gPreviewShow();
  // Poll stats since WS is unavailable
  if (_gLiveStatTimer) clearInterval(_gLiveStatTimer);
  _gLiveStatTimer = setInterval(_gPollLiveStats, 500);
}

function _gTeardownLive() {
  // Client-side live loop (browser GPU)
  if (typeof _gStopClientLive === 'function') _gStopClientLive();
  // WS
  if (_gLiveWs) {
    _gLiveWsFallback = true; // prevent onclose from restarting MJPEG
    _gLiveWs.close();
    _gLiveWs = null;
  }
  if (_gLiveWsPingTimer) { clearInterval(_gLiveWsPingTimer); _gLiveWsPingTimer = null; }
  // MJPEG
  if (_gLiveStatTimer) { clearInterval(_gLiveStatTimer); _gLiveStatTimer = null; }
  if (gLiveImg) { gLiveImg.src = ''; gLiveImg.remove(); gLiveImg = null; }
  // Canvas
  if (gLiveCanvas) { gLiveCanvas.remove(); gLiveCanvas = null; }
  gMainPreview.classList.remove('ws-live');
  if (!gMainPreview.querySelector('img, video, canvas'))
    gMainPreview.classList.remove('active');
}

// ── Client-side live loop (browser GPU, for 3D graphs) ─────────
let _gClientLiveActive = false;
function _gClientGraphPayload() {
  return {
    nodes: gNodes.map(n => gSerializeNodeForApi(n)),
    edges: gEdges.map(e => ({ src_node:e.src_node, src_port:e.src_port, dst_node:e.dst_node, dst_port:e.dst_port, feedback:e.feedback })),
  };
}
async function _gStartClientLive() {
  const C = await gClient3D();
  const start = parseInt(document.getElementById('tl-start').value) || 0;
  const end   = parseInt(document.getElementById('tl-end').value)   || 24;
  const rate  = gLiveRate();
  const fps   = rate.fps;
  const { nodes, edges } = _gClientGraphPayload();
  const canvas = await C.startLive({
    nodes, edges, start, end, fps, width: gCanvasW, height: gCanvasH,
    cookFps: rate.fps_limit ? fps : 0,
    onStats: ({ frame, fps }) => {
      // Re-read the rate per stat tick — the limiter retunes mid-live.
      const lim = gLiveRate();
      gSetStatus(`Live (client GPU) · frame ${frame} · ${fps} fps`
        + (lim.fps_limit ? ` · limited ${lim.fps}` : ''));
      const fEl = document.getElementById('lmr-frame'); if (fEl) fEl.textContent = `frame ${frame}`;
      const pEl = document.getElementById('lmr-fps');   if (pEl) pEl.textContent = `${fps} fps`;
      gClientSurfaceErrors();
    },
  });
  gMountClientCanvas(canvas);
  _gClientLiveActive = true;
}
// Show client-side per-node errors (p5 compile/runtime, etc.) in the selected
// node's multiline-editor error panel, mirroring the server node_errors path.
function gClientSurfaceErrors() {
  if (!_gClient3D || !gSelectedNode) return;
  const errs = _gClient3D.getNodeErrors ? _gClient3D.getNodeErrors() : {};
  const node = gNodes.find(n => n.id === gSelectedNode);
  if (!node) return;
  const def = gNodeDefs[node.method_id];
  const mlKey = def && Object.entries(def.params || {})
    .find(([, s]) => s.multiline)?.[0];
  if (!mlKey) return;
  const errDiv = document.getElementById(`glsl-err-${mlKey}`);
  if (!errDiv) return;
  const msg = errs[gSelectedNode] || '';
  errDiv.textContent = msg;
  errDiv.style.display = msg ? '' : 'none';
}
function _gStopClientLive() {
  if (!_gClientLiveActive) return;
  _gClientLiveActive = false;
  if (_gClient3D) _gClient3D.stopLive();
}
// Push param/edge edits into a running client live loop without restarting.
function gClientLiveRefresh() {
  if (_gClientLiveActive && _gClient3D) {
    const { nodes, edges } = _gClientGraphPayload();
    _gClient3D.updateLiveGraph(nodes, edges);
  }
}
// Where the live preview is currently rendered — surfaced in Diagnostics.
function _gSetPreviewMode(mode) {
  window.gPreviewMode = mode;
  const el = document.getElementById('diag-preview-mode');
  if (el) {
    el.textContent = 'preview: ' + (mode === 'client' ? 'client GPU'
      : mode === 'server' ? 'server WS' : '—');
    el.className = (mode === 'client' || mode === 'server') ? mode : '';
  }
}
// True when the whole graph can be rendered by the browser spine (3D/p5/custom
// shader + any existing GPU shader node), and WebGL2 is available.
async function gGraphClientRenderable() {
  if (!('WebGL2RenderingContext' in window)) return false;
  if (!gNodes.length) return false;
  try {
    const C = await gClient3D();
    const { nodes } = _gClientGraphPayload();
    // Ensure the shader bundle (node_map) is loaded so client-GPU shims of CPU
    // nodes are recognized before the renderability check.
    if (C.prepare) await C.prepare(nodes);
    return C.graphClientRenderable(nodes);
  } catch { return false; }
}

// Single writer for live state: the button's lit/unlit look must never drift
// from gLiveMode, or the toggle strands itself in the "on" position and every
// later click re-runs the start path instead of stopping.
function _gSetLiveState(on) {
  gLiveMode = on;
  gLiveBtn.classList.toggle('active', on);
  gLiveBtnDesk.classList.toggle('active', on);
  document.getElementById('pvh-live-pill')?.classList.toggle('on', on);
  if (!on) gClearLiveMotion();
}

// Toggling live is async (renderability probe + a POST), so two overlapping
// calls could both run the start path and open two transports. Serialize
// them: each toggle waits for the previous one to settle.
let _gLiveToggleChain = Promise.resolve();
function gSetLive(on) {
  _gLiveToggleChain = _gLiveToggleChain
    .catch(() => {})
    .then(() => _gSetLiveImpl(on));
  return _gLiveToggleChain;
}

async function _gSetLiveImpl(on) {
  // Live preview runs on the browser GPU when the whole graph is client-
  // renderable (feature #1): 3D/p5/custom-shader nodes AND the existing GPU
  // shader nodes (via the shader parity layer). Otherwise the server /api/
  // graph/live + WS path (untouched) drives the preview. Run/export are
  // unaffected — the server stays authoritative for those.
  if (on && await gGraphClientRenderable()) {
    _gSetLiveState(true);
    try {
      await _gStartClientLive();
      _gSetPreviewMode('client');
      return;
    } catch (e) {
      console.warn('[live] client render failed', e);
      _gStopClientLive();
      // Graphs with client-only nodes cannot fall back to the server.
      if (gGraphRunsOnClient()) {
        gSetStatus('Client live error: ' + e.message);
        _gSetLiveState(false);
        return;
      }
      gSetStatus('Client render unavailable — using server preview');
      // Drop back to "off" before the server path so a failure there leaves
      // the button unlit rather than stuck lit.
      _gSetLiveState(false);
    }
  }
  if (!on && _gClientLiveActive) {
    _gStopClientLive();
    _gSetLiveState(false);
    _gSetPreviewMode('—');
    gSetStatus('Live stopped.');
    return;
  }

  try {
    // Always time out: an unresponsive server used to leave this fetch pending
    // forever, so the toggle never resolved and the button stayed lit.
    await fetch('/api/graph/live', {
      method: 'POST', headers: {'Content-Type': 'application/json'},
      body: JSON.stringify(gLiveGraphBody(!on)),
      signal: AbortSignal.timeout(15000),
    });
  } catch(e) {
    _gSetLiveState(false);
    _gSetPreviewMode('—');
    gSetStatus(e.name === 'TimeoutError'
      ? 'Live: server not responding (timed out)'
      : 'Live: ' + e.message);
    return;
  }

  _gSetLiveState(on);
  _gSetPreviewMode(on ? 'server' : '—');

  if (on) {
    // Close whatever transport is already attached before building a new
    // one. Starting while live (re-toggle, restart after an error) used to
    // leak the old socket, its ping timer and the MJPEG stream.
    _gTeardownLive();
    gMainPreview.innerHTML = '';
    gLivePreviewImg = null;
    gLiveCanvas = null; gLiveImg = null;

    const useWs = 'WebSocket' in window;
    if (useWs) {
      // Primary path: canvas + WebSocket
      gLiveCanvas = document.createElement('canvas');
      gLiveCanvas.id = 'live-ws-canvas';
      gLiveCanvas.onclick = () => gOpenFullscreen();
      gMainPreview.appendChild(gLiveCanvas);
      gMainPreview.classList.add('active', 'ws-live');
      gPreviewShow();
      gSetStatus('Live — connecting…');
      _gOpenLiveWs();
    } else {
      // Feature-detect fallback: browser has no WebSocket support
      _gLiveWsFallback = true;
      _gStartMjpegFallback();
      gSetStatus('Live — MJPEG (no WebSocket)…');
    }
    // Ensure the metadata readout is re-mounted (innerHTML wipe above removes it)
    const existingReadout = document.getElementById('live-meta-readout');
    if (!existingReadout) {
      const rd = document.createElement('div');
      rd.id = 'live-meta-readout';
      rd.innerHTML = `<div id="lmr-top">
        <span id="lmr-frame">frame –</span>
        <span id="lmr-ms">–ms</span>
        <span id="lmr-fps">– fps</span>
        <span id="lmr-err" style="display:none">⚠ error</span>
      </div><div id="lmr-bars"></div>`;
      gMainPreview.appendChild(rd);
    }
  } else {
    _gTeardownLive();
    gSetStatus('Live stopped.');
  }
}

function gToggleLive() {
  if (!gLiveMode && !gNodes.length) { gSetStatus('No nodes.'); return; }
  gSetLive(!gLiveMode);
}
gLiveBtn.addEventListener('click', gToggleLive);
gLiveBtnDesk.addEventListener('click', gToggleLive);

// ── Preview chrome: header actions (pop-out / PiP / fullscreen) ───────
// The buttons live in the always-visible #preview-header, so they survive the
// preview's innerHTML rebuilds; the MutationObserver only keeps the PiP
// button's visibility in sync with whether a <video> is currently mounted.
const gPvhPip = document.getElementById('pvh-pip');

function gSyncPreviewChrome() {
  const hasVideo = !!gMainPreview.querySelector('video');
  gPvhPip.style.display = (hasVideo && document.pictureInPictureEnabled) ? '' : 'none';
}
new MutationObserver(gSyncPreviewChrome).observe(gMainPreview, { childList: true });

// ── Live preview mirroring ───────────────────────────────────────
// The main preview swaps between a <canvas> (live: WS or client-GPU), a
// <video> (encoded clip) and an <img> (single rendered frame). To show the
// *moving* feed in a second surface (fullscreen overlay, pop-out window) we
// copy whatever element is currently mounted into a target <canvas> every
// animation frame, re-resolving the source each tick so mode switches follow
// automatically. This is why the old pop-out froze on one frame — it snapshotted
// once (or pointed a second <img> at a stream the client-GPU path never feeds).
function _gPreviewSource() {
  return gMainPreview.querySelector('canvas')
      || gMainPreview.querySelector('video')
      || gMainPreview.querySelector('img');
}
function _gSrcDims(el) {
  if (el.tagName === 'VIDEO') return [el.videoWidth, el.videoHeight];
  if (el.tagName === 'IMG')   return [el.naturalWidth, el.naturalHeight];
  return [el.width, el.height]; // canvas
}
const _gMirrors = new Map(); // target canvas → { raf, win }
// `rafWin` is the window whose requestAnimationFrame drives the loop. For a
// pop-out it must be the child window — the parent tab is backgrounded while the
// user watches the pop-out, and background-tab rAF is throttled to ~1fps.
function _gStartMirror(target, stopWhen, rafWin) {
  _gStopMirror(target);
  const win = rafWin || window;
  const ctx = target.getContext('2d');
  const rec = { raf: 0, win };
  _gMirrors.set(target, rec);
  const tick = () => {
    if (!_gMirrors.has(target)) return;
    if (stopWhen && stopWhen()) { _gStopMirror(target); return; }
    const src = _gPreviewSource();
    if (src) {
      const [sw, sh] = _gSrcDims(src);
      if (sw && sh) {
        if (target.width !== sw || target.height !== sh) { target.width = sw; target.height = sh; }
        try { ctx.drawImage(src, 0, 0, sw, sh); } catch (_) {}
      }
    }
    rec.raf = win.requestAnimationFrame(tick);
  };
  rec.raf = win.requestAnimationFrame(tick);
}
function _gStopMirror(target) {
  const rec = _gMirrors.get(target);
  if (rec) { try { rec.win.cancelAnimationFrame(rec.raf); } catch (_) {} _gMirrors.delete(target); }
}

// Fullscreen from the header button — same overlay the click-on-frame path uses.
// A live/animated canvas is mirrored continuously; a video or a static frame is
// shown directly.
function gOpenFullscreen() {
  const src = _gPreviewSource();
  if (!src) { gShowToast('Nothing to view yet', true); return; }
  gFsVideo.style.display = 'none';
  gFsImg.style.display = 'none';
  gFsCanvas.style.display = 'none';
  _gStopMirror(gFsCanvas);
  if (src.tagName === 'CANVAS') {
    gFsCanvas.style.display = '';
    _gStartMirror(gFsCanvas, () => !gFsOverlay.classList.contains('visible'));
  } else if (src.tagName === 'VIDEO' && (src.currentSrc || src.src)) {
    gFsVideo.src = src.currentSrc || src.src;
    gFsVideo.style.display = '';
  } else if (src.tagName === 'IMG' && src.src) {
    gFsImg.src = src.src;
    gFsImg.style.display = '';
  } else {
    gShowToast('Nothing to view yet', true); return;
  }
  gFsOverlay.classList.add('visible');
}

// ── Pop-out player ───────────────────────────────────────────────
// A second-window player that mirrors the live feed and re-hosts the full
// timeline control set. Controls proxy straight to the main-window elements
// (same origin, so the child document is built + wired from this realm), so
// every existing behaviour — scrubbing, play, render-sequence, live toggle,
// canvas size — is reused rather than reimplemented.
const _GPOPOUT_PRESETS = ['768x512', '512x512', '1024x576', '1280x720', '1920x1080'];
let _gPopoutWin = null;

function gPopOutViewer() {
  if (_gPopoutWin && !_gPopoutWin.closed) { _gPopoutWin.focus(); return; }
  const w = window.open('', 'grillmaster-player', 'width=900,height=680');
  if (!w) { gShowToast('Pop-up blocked by the browser', true); return; }
  _gPopoutWin = w;
  const d = w.document;
  d.title = 'Grillmaster — Player';
  d.body.style.cssText = 'margin:0;height:100vh;display:flex;flex-direction:column;background:#0b0b0d;color:#e8e8ea;font:13px/1.4 system-ui,sans-serif;overflow:hidden';

  // Stage: mirrored canvas
  const stage = d.createElement('div');
  stage.style.cssText = 'flex:1;min-height:0;display:flex;align-items:center;justify-content:center;background:#000;overflow:hidden';
  const canvas = d.createElement('canvas');
  canvas.style.cssText = 'max-width:100%;max-height:100%;object-fit:contain';
  stage.appendChild(canvas);
  d.body.appendChild(stage);

  // Control bar
  const bar = d.createElement('div');
  bar.style.cssText = 'flex:none;display:flex;flex-wrap:wrap;gap:10px 14px;align-items:center;padding:8px 12px;background:#151518;border-top:1px solid #2a2a30';
  d.body.appendChild(bar);

  const mkGroup = () => { const g = d.createElement('div'); g.style.cssText = 'display:flex;gap:5px;align-items:center'; bar.appendChild(g); return g; };
  const mkLabel = (t) => { const l = d.createElement('label'); l.textContent = t; l.style.cssText = 'color:#9a9aa2;font-size:11px'; return l; };
  const mkBtn = (t, title) => { const b = d.createElement('button'); b.textContent = t; if (title) b.title = title; b.style.cssText = 'background:#26262c;color:#e8e8ea;border:1px solid #34343c;border-radius:6px;padding:4px 9px;font-size:13px;cursor:pointer'; return b; };
  const mkNum = (val, wdt) => { const i = d.createElement('input'); i.type = 'number'; i.value = val; i.style.cssText = `width:${wdt}px;background:#0f0f12;color:#e8e8ea;border:1px solid #34343c;border-radius:5px;padding:3px 5px;font-size:12px`; return i; };

  // Frame + transport
  const gFrame = mkGroup();
  gFrame.appendChild(mkLabel('Frame'));
  const inFrame = mkNum(tlFrame.value, 54); gFrame.appendChild(inFrame);
  const btnPrev = mkBtn('◀', 'Previous frame');
  const btnPlay = mkBtn(tlPlay.textContent, 'Play / Pause');
  const btnNext = mkBtn('▶▶', 'Next frame');
  gFrame.append(btnPrev, btnPlay, btnNext);

  // Start / End / FPS
  const gRange = mkGroup();
  gRange.appendChild(mkLabel('Start')); const inStart = mkNum(tlStart.value, 48); gRange.appendChild(inStart);
  gRange.appendChild(mkLabel('End'));   const inEnd   = mkNum(tlEnd.value, 48);   gRange.appendChild(inEnd);
  gRange.appendChild(mkLabel('FPS'));   const inFps   = mkNum(tlFps.value, 40);   gRange.appendChild(inFps);
  // Live cook-rate limiter, mirrored from the main timeline bar. The pop-out
  // has no stylesheet, so pull the accent off the opener instead of baking in
  // a colour that ignores the active theme.
  const _accent = getComputedStyle(document.documentElement).getPropertyValue('--accent').trim() || '#e8833a';
  const chkLimit = d.createElement('input');
  chkLimit.type = 'checkbox';
  chkLimit.checked = !!gFpsLimitEl?.checked;
  chkLimit.style.cssText = `accent-color:${_accent};width:12px;height:12px;margin:0 0 0 6px;cursor:pointer`;
  const labLimit = mkLabel('⏱ limit');
  labLimit.title = 'Limit live cooking to the FPS above';
  labLimit.style.cursor = 'pointer';
  labLimit.addEventListener('click', () => { chkLimit.checked = !chkLimit.checked; fire(chkLimit, 'change'); });
  gRange.append(chkLimit, labLimit);

  // Name
  const gName = mkGroup();
  gName.appendChild(mkLabel('Name'));
  const inName = d.createElement('input'); inName.type = 'text'; inName.placeholder = 'my-sequence'; inName.value = tlName.value;
  inName.style.cssText = 'width:110px;background:#0f0f12;color:#e8e8ea;border:1px solid #34343c;border-radius:5px;padding:3px 6px;font-size:12px';
  gName.appendChild(inName);

  // Render sequence + live
  const gActions = mkGroup();
  const btnRender = mkBtn('🎬 Render Sequence', 'Render the frame range'); gActions.appendChild(btnRender);
  const btnLive = mkBtn('📺 Live', 'Toggle live preview'); gActions.appendChild(btnLive);

  // Resolution (presets + Custom)
  const gRes = mkGroup();
  gRes.appendChild(mkLabel('Size'));
  const selRes = d.createElement('select');
  selRes.style.cssText = 'background:#0f0f12;color:#e8e8ea;border:1px solid #34343c;border-radius:5px;padding:3px 5px;font-size:12px;cursor:pointer';
  for (const p of _GPOPOUT_PRESETS) { const o = d.createElement('option'); o.value = p; o.textContent = p.replace('x', '×'); selRes.appendChild(o); }
  const optCustom = d.createElement('option'); optCustom.value = 'custom'; optCustom.textContent = 'Custom…'; selRes.appendChild(optCustom);
  gRes.appendChild(selRes);
  const inW = mkNum(gCanvasW, 56); const inH = mkNum(gCanvasH, 56);
  const times = mkLabel('×');
  const wrapCustom = d.createElement('span'); wrapCustom.style.cssText = 'display:none;gap:4px;align-items:center'; wrapCustom.append(inW, times, inH);
  gRes.appendChild(wrapCustom);
  const syncResUI = () => {
    const key = `${gCanvasW}x${gCanvasH}`;
    if (_GPOPOUT_PRESETS.includes(key)) { selRes.value = key; wrapCustom.style.display = 'none'; }
    else { selRes.value = 'custom'; wrapCustom.style.display = 'inline-flex'; inW.value = gCanvasW; inH.value = gCanvasH; }
  };
  syncResUI();

  const btnFs = mkBtn('⛶', 'Fullscreen this window'); mkGroup().appendChild(btnFs);

  // ── Scrub bar (bottom strip) ──
  // Maps the Start…End range onto a draggable track; dragging writes the frame
  // straight through to the main window's #tl-frame, so scrubbing here drives
  // the same load-frame path as the main timeline.
  const scrub = d.createElement('div');
  scrub.style.cssText = 'flex:none;display:flex;align-items:center;gap:10px;padding:6px 12px 9px;background:#151518;border-top:1px solid #24242a';
  const track = d.createElement('div');
  track.style.cssText = 'position:relative;flex:1;height:16px;cursor:pointer;touch-action:none';
  const rail = d.createElement('div');
  rail.style.cssText = 'position:absolute;top:6px;left:0;right:0;height:4px;border-radius:2px;background:#2e2e36';
  const fill = d.createElement('div');
  fill.style.cssText = 'position:absolute;top:6px;left:0;width:0;height:4px;border-radius:2px;background:#e8833a';
  const knob = d.createElement('div');
  knob.style.cssText = 'position:absolute;top:1px;left:0;width:14px;height:14px;margin-left:-7px;border-radius:50%;background:#fff;box-shadow:0 1px 4px rgba(0,0,0,.6);pointer-events:none';
  track.append(rail, fill, knob);
  const readout = d.createElement('span');
  readout.style.cssText = 'flex:none;min-width:96px;text-align:right;color:#9a9aa2;font-size:11px;font-variant-numeric:tabular-nums';
  scrub.append(track, readout);
  d.body.appendChild(scrub);

  const scrubRange = () => {
    const s = parseInt(tlStart.value) || 0;
    const e = parseInt(tlEnd.value) || 24;
    return [s, Math.max(s, e)];
  };
  const paintScrub = () => {
    const [s, e] = scrubRange();
    const f = Math.min(e, Math.max(s, parseInt(tlFrame.value) || 0));
    const pct = e > s ? ((f - s) / (e - s)) * 100 : 0;
    fill.style.width = pct + '%';
    knob.style.left = pct + '%';
    readout.textContent = `${f} / ${e}`;
  };
  let scrubbing = false;
  const scrubTo = (clientX) => {
    const r = track.getBoundingClientRect();
    if (!r.width) return;
    const [s, e] = scrubRange();
    const ratio = Math.min(1, Math.max(0, (clientX - r.left) / r.width));
    const f = s + Math.round(ratio * (e - s));
    if (String(f) !== tlFrame.value) { tlFrame.value = f; tlFrame.dispatchEvent(new Event('input', { bubbles: true })); }
    paintScrub();
  };
  track.addEventListener('pointerdown', (ev) => {
    scrubbing = true;
    try { track.setPointerCapture(ev.pointerId); } catch (_) {}
    scrubTo(ev.clientX);
    ev.preventDefault();
  });
  track.addEventListener('pointermove', (ev) => { if (scrubbing) scrubTo(ev.clientX); });
  const endScrub = (ev) => {
    if (!scrubbing) return;
    scrubbing = false;
    try { track.releasePointerCapture(ev.pointerId); } catch (_) {}
  };
  track.addEventListener('pointerup', endScrub);
  track.addEventListener('pointercancel', endScrub);
  paintScrub();

  // ── Wire controls to the main window ──
  const fire = (el, type) => el.dispatchEvent(new Event(type, { bubbles: true }));
  const proxyNum = (input, target) => input.addEventListener('input', () => { target.value = input.value; fire(target, 'input'); fire(target, 'change'); });
  inFrame.addEventListener('input', () => { tlFrame.value = inFrame.value; fire(tlFrame, 'input'); });
  proxyNum(inStart, tlStart); proxyNum(inEnd, tlEnd); proxyNum(inFps, tlFps);
  chkLimit.addEventListener('change', () => {
    if (!gFpsLimitEl) return;
    gFpsLimitEl.checked = chkLimit.checked;
    fire(gFpsLimitEl, 'change');
  });
  inName.addEventListener('input', () => { tlName.value = inName.value; fire(tlName, 'input'); });
  btnPrev.addEventListener('click', () => tlPrev.click());
  btnPlay.addEventListener('click', () => tlPlay.click());
  btnNext.addEventListener('click', () => tlNext.click());
  btnRender.addEventListener('click', () => tlRenderBtn.click());
  btnLive.addEventListener('click', () => gToggleLive());
  btnFs.addEventListener('click', () => { (canvas.requestFullscreen || canvas.webkitRequestFullscreen).call(canvas); });
  const applyCustom = () => { const cw = parseInt(inW.value) || gCanvasW, ch = parseInt(inH.value) || gCanvasH; _gApplyCanvasSize(cw, ch); gSave(); };
  selRes.addEventListener('change', () => {
    if (selRes.value === 'custom') { wrapCustom.style.display = 'inline-flex'; inW.value = gCanvasW; inH.value = gCanvasH; return; }
    wrapCustom.style.display = 'none';
    const [cw, ch] = selRes.value.split('x').map(Number);
    _gApplyCanvasSize(cw, ch); gSave();
  });
  inW.addEventListener('change', applyCustom);
  inH.addEventListener('change', applyCustom);

  // ── Reflect main-window state back into the pop-out ──
  const sync = () => {
    if (w.closed) { clearInterval(syncTimer); _gStopMirror(canvas); if (_gPopoutWin === w) _gPopoutWin = null; return; }
    if (d.activeElement !== inFrame) inFrame.value = tlFrame.value;
    if (d.activeElement !== inStart) inStart.value = tlStart.value;
    if (d.activeElement !== inEnd)   inEnd.value = tlEnd.value;
    if (d.activeElement !== inFps)   inFps.value = tlFps.value;
    if (gFpsLimitEl) chkLimit.checked = gFpsLimitEl.checked;
    if (d.activeElement !== inName)  inName.value = tlName.value;
    btnPlay.textContent = tlPlay.textContent;
    btnLive.style.background = gLiveMode ? '#2b6' : '#26262c';
    btnLive.style.color = gLiveMode ? '#04120a' : '#e8e8ea';
    if (d.activeElement !== inW && d.activeElement !== inH) syncResUI();
    if (!scrubbing) paintScrub();
  };
  const syncTimer = setInterval(sync, 150);
  w.addEventListener('beforeunload', () => { clearInterval(syncTimer); _gStopMirror(canvas); if (_gPopoutWin === w) _gPopoutWin = null; });

  _gStartMirror(canvas, () => w.closed, w);
}

// ── Output backdrop ──────────────────────────────────────────────
// Swaps the node-graph's dot grid for a fitted mirror of the preview. It reuses
// the same mirror machinery as fullscreen and the pop-out, so it follows the
// preview through every source swap (live canvas / clip / rendered frame).
const gBdBtn    = document.getElementById('pvh-backdrop');
const gBdCanvas = document.getElementById('gbd-canvas');

function gSetBackdrop(on, quiet = false) {
  gCanvasWrap.classList.toggle('bd-on', on);
  gBdBtn.classList.toggle('on', on);
  gBdBtn.title = on ? 'Back to the grid background' : 'Use the output as the node-graph background';
  if (on) {
    // The loop re-resolves the source every tick, so a preview that only
    // appears later is picked up on its own — fine to enable on an empty one.
    _gStartMirror(gBdCanvas, () => !gCanvasWrap.classList.contains('bd-on'));
    if (!quiet && !_gPreviewSource()) gShowToast('No output yet — the backdrop fills in on the next render');
  } else {
    _gStopMirror(gBdCanvas);
  }
  try { localStorage.setItem('graph-output-backdrop', on ? '1' : ''); } catch {}
}
gBdBtn.addEventListener('click', () => gSetBackdrop(!gCanvasWrap.classList.contains('bd-on')));
// Restored quietly — on a cold load there is never a source to mirror yet.
try { if (localStorage.getItem('graph-output-backdrop') === '1') gSetBackdrop(true, true); } catch {}

document.getElementById('pvh-popout').addEventListener('click', gPopOutViewer);
document.getElementById('pvh-fullscreen').addEventListener('click', gOpenFullscreen);
gPvhPip.addEventListener('click', () => {
  const v = gMainPreview.querySelector('video');
  if (v) v.requestPictureInPicture().catch(err => gShowToast('PiP: ' + err.message, true));
});

// Collapse to header-only (persisted, desktop)
const gPvhCollapse = document.getElementById('pvh-collapse');
function gSetPreviewMin(min) {
  gPreviewPanel.classList.toggle('pv-min', min);
  gPvhCollapse.title = min ? 'Expand preview' : 'Collapse preview';
  try { localStorage.setItem('preview-min', min ? '1' : ''); } catch {}
}
gPvhCollapse.addEventListener('click', () => gSetPreviewMin(!gPreviewPanel.classList.contains('pv-min')));
try { if (localStorage.getItem('preview-min') === '1') gSetPreviewMin(true); } catch {}

// ── Resizable preview height (persisted) ──────────────────────
const gPvHandle = document.getElementById('preview-resize-handle');
(function () {
  const saved = parseInt(localStorage.getItem('preview-h') || '', 10);
  if (saved && window.innerWidth > 900) {
    document.documentElement.style.setProperty('--pv-h', saved + 'px');
    document.getElementById('graph-preview-panel').style.maxHeight = 'none';
  }
  let startY = 0, startH = 0, dragging = false;
  gPvHandle.addEventListener('pointerdown', e => {
    dragging = true; startY = e.clientY;
    startH = gMainPreview.getBoundingClientRect().height;
    gPvHandle.classList.add('dragging');
    gPvHandle.setPointerCapture(e.pointerId);
    document.getElementById('graph-preview-panel').style.maxHeight = 'none';
    e.preventDefault();
  });
  gPvHandle.addEventListener('pointermove', e => {
    if (!dragging) return;
    const h = Math.min(Math.round(window.innerHeight * 0.7),
                       Math.max(120, Math.round(startH + (e.clientY - startY))));
    document.documentElement.style.setProperty('--pv-h', h + 'px');
  });
  gPvHandle.addEventListener('pointerup', () => {
    if (!dragging) return;
    dragging = false; gPvHandle.classList.remove('dragging');
    const h = parseInt(getComputedStyle(gMainPreview).height, 10);
    if (h) try { localStorage.setItem('preview-h', String(h)); } catch {}
  });
  gPvHandle.addEventListener('dblclick', () => {
    document.documentElement.style.removeProperty('--pv-h');
    try { localStorage.removeItem('preview-h'); } catch {}
  });
})();

// ── Keyboard cheat-sheet ───────────────────────────────────────
const gHelpOverlay = document.getElementById('help-overlay');
function gToggleHelp(force) {
  gHelpOverlay.classList.toggle('visible', force);
}
document.getElementById('graph-help-btn')?.addEventListener('click', () => gToggleHelp());
document.getElementById('help-close-btn').addEventListener('click', () => gToggleHelp(false));
gHelpOverlay.addEventListener('click', e => { if (e.target === gHelpOverlay) gToggleHelp(false); });
document.addEventListener('keydown', e => {
  const tag = (document.activeElement?.tagName || '').toLowerCase();
  const isInput = tag === 'input' || tag === 'textarea' || tag === 'select';
  if (e.key === '?' && !isInput) { e.preventDefault(); gToggleHelp(); }
  if (e.key === 'Escape' && gHelpOverlay.classList.contains('visible')) gToggleHelp(false);
});

// ── Settings page ──────────────────────────────────────────────
// The theme picker and token editor live in ui/js/theme.js, which owns both
// the `data-theme` preset and the per-token overrides layered on top of it.

const gSettingsPage = document.getElementById('settings-page');
function gToggleSettings(force) {
  const show = force !== undefined ? force : !gSettingsPage.classList.contains('visible');
  gSettingsPage.classList.toggle('visible', show);
  if (show) {
    const tok = document.getElementById('settings-api-token');
    tok.value = localStorage.getItem('api-token') || '';
  }
}
document.getElementById('graph-settings-btn')?.addEventListener('click', () => gToggleSettings());
document.getElementById('graph-settings-btn-mob')?.addEventListener('click', () => gToggleSettings());
document.getElementById('settings-close-btn').addEventListener('click', () => gToggleSettings(false));
gSettingsPage.addEventListener('click', e => { if (e.target === gSettingsPage) gToggleSettings(false); });
document.addEventListener('keydown', e => {
  if (e.key === 'Escape' && gSettingsPage.classList.contains('visible')) gToggleSettings(false);
});

document.getElementById('settings-token-save').addEventListener('click', () => {
  const v = document.getElementById('settings-api-token').value.trim();
  try {
    if (v) localStorage.setItem('api-token', v);
    else localStorage.removeItem('api-token');
  } catch {}
  gShowToast(v ? 'Token saved — reload to apply' : 'Token cleared — reload to apply');
});

document.getElementById('settings-reset-layout').addEventListener('click', () => {
  try {
    ['preview-h', 'pane-widths', 'graph-preview-expanded', 'preview-min'].forEach(k => localStorage.removeItem(k));
  } catch {}
  document.documentElement.style.removeProperty('--pv-h');
  gPreviewPanel.classList.remove('pv-min');
  gShowToast('Panel sizes reset — reload to apply fully');
});

// ── Auto-generate: fire-and-cancel ───────────────────────────
let _autoGenTimer = null;
function gDoAutoGen() {
  // A running client-side live loop (3D/p5/custom shader OR a GPU-shader graph
  // under feature #1) hot-swaps on param edits — instant, no server round-trip.
  if (_gClientLiveActive) { gClientLiveRefresh(); return; }
  // Client-only graphs (3D/p5): one-shot browser render in auto mode.
  if (gGraphRunsOnClient()) {
    if (gAutoGen && gNodes.length) {
      clearTimeout(_autoGenTimer);
      _autoGenTimer = setTimeout(() => { gClientRunOnce().catch(e => console.error(e)); }, 80);
    }
    return;
  }
  if (gLiveMode) {
    // Param edits while live hot-swap the running loop instead of
    // firing one-shot executes. This must NOT go through gSetLive(true):
    // that re-runs the whole start path (new WS, rebuilt preview) and used
    // to stack a second server render per edit.
    clearTimeout(_autoGenTimer);
    _autoGenTimer = setTimeout(gLiveHotSwap, 150);
    return;
  }
  if (!gAutoGen) return;
  if (!gNodes.length) return;
  // Cancel any in-flight auto-gen
  if (gAutoGenAbort) { gAutoGenAbort.abort(); gAutoGenAbort = null; }
  clearTimeout(_autoGenTimer);
  // Debounce: wait 80ms after last param change before firing
  _autoGenTimer = setTimeout(() => {
    gAutoGenAbort = new AbortController();
    const body = {
      ...gServerGraphPayload(),
      seed: 42, frames: 1,
      width: gCanvasW, height: gCanvasH,
    };
    fetch('/api/graph/execute', {
      method:'POST', headers:{'Content-Type':'application/json'},
      body:JSON.stringify(body), signal: gAutoGenAbort.signal,
    }).then(r => r.json()).then(({job_id}) => {
      if (!job_id) return;
      gAutoGenJobId = job_id;
      gListenGraphSSE(job_id);
    }).catch(e => {
      if (e.name !== 'AbortError') gSetStatus('Auto-gen: ' + e.message);
    });
  }, 80);
}

function gListenGraphSSE(job_id) {
  gLivePreviewImg = null;
  gLastGraphJobId = job_id;
  const es = new EventSource(`/api/graph/jobs/${job_id}/stream`);
  es.addEventListener('progress', e => gSetStatus(JSON.parse(e.data).message));
  es.addEventListener('graph_frame', e => {
    const { data } = JSON.parse(e.data);
    gGraphFrameUpdate(data);
  });
  es.addEventListener('node-error', e => {
    const { nodeId, error } = JSON.parse(e.data);
    const el = document.getElementById('gnode-' + nodeId);
    if (el) { el.classList.add('node-error'); el.dataset.errorMsg = error; }
  });
  es.addEventListener('done', e => {
    es.close();
    const d = JSON.parse(e.data);
    gNodes.forEach(n => { n.dirty = false; });  // mark all clean after successful run
    gSetStatus(`Done — ${d.frames || 1} frame(s)`);
    gSetRunDisabled(false);
    // Set timeline name so scrubber can browse frames
    if (d.seq_name) {
      tlName.value = d.seq_name;
    }
    gGraphDoneSwap(d.type, d.rel_path, d.seq_name);
  });
  es.addEventListener('error', e => {
    es.close();
    try { gSetStatus('Error: '+JSON.parse(e.data).message); } catch { gSetStatus('Connection error.'); }
    gSetRunDisabled(false);
  });
}

// ════════════════════════════════════════════════════════════════
// TIMELINE UI
// ════════════════════════════════════════════════════════════════

// ── DOM refs ────────────────────────────────────────────────────
const tlFrame         = document.getElementById('tl-frame');
const tlStart         = document.getElementById('tl-start');
const tlEnd           = document.getElementById('tl-end');
const tlFps           = document.getElementById('tl-fps');
const tlName          = document.getElementById('tl-name');
const tlRenderBtn     = document.getElementById('tl-render-seq');
const tlProgressDiv   = document.getElementById('tl-progress');
const tlProgressBar   = document.getElementById('tl-progress-bar');
const tlProgressLabel = document.getElementById('tl-progress-label');
const tlCancel        = document.getElementById('tl-cancel');
const tlPrev          = document.getElementById('tl-prev');
const tlPlay          = document.getElementById('tl-play');
const tlNext          = document.getElementById('tl-next');

let tlSeqAbort  = null;
let tlPlaying   = false;
// Playback is driven off a wall-clock origin, not a per-tick accumulator — see
// _tlPlayTick for why. `_tlPlayClock` is null whenever playback is stopped.
let tlPlayRaf    = null;
let _tlPlayClock = null;  // { at, frame, fps } — wall-clock origin of playback
let _tlPlayShown = null;  // last frame this loop put on screen

// ── Timeline ruler rendering ─────────────────────────────────────
function renderTimelineRuler() {
  const header = document.getElementById('tl-ruler-header');
  const lanes  = document.getElementById('tl-lanes');
  if (!header || !lanes) return;

  const start = parseInt(document.getElementById('tl-start')?.value) || 0;
  const end   = parseInt(document.getElementById('tl-end')?.value) || 24;
  const total = Math.max(1, end - start + 1);
  const pixelWidth = total * 12; // 12px per frame

  // ── Ruler header (frame ticks) ──
  header.innerHTML = '';
  header.style.width = pixelWidth + 'px';
  for (let f = start; f <= end; f++) {
    const tick = document.createElement('div');
    tick.className = 'tl-ruler-tick' + (f % 10 === 0 ? ' major' : '');
    tick.style.width = '12px';
    if (f % 10 === 0) {
      tick.innerHTML = `<span class="tl-tick-label">${f}</span>`;
    }
    const line = document.createElement('div');
    line.className = 'tl-tick-line' + (f % 10 === 0 ? ' full' : '');
    line.style.left = '0';
    line.style.height = '100%';
    tick.appendChild(line);
    header.appendChild(tick);
  }

  // ── Clip lanes ──
  lanes.innerHTML = '';
  lanes.style.width = pixelWidth + 'px';

  if (tlClips.length === 0) {
    lanes.style.height = '20px';
    const empty = document.createElement('div');
    empty.className = 'tl-empty-lane';
    empty.style.width = pixelWidth + 'px';
    empty.textContent = 'Run the graph to create a clip on the timeline';
    lanes.appendChild(empty);
    // Phase 5: still render node KF lanes even with no clips
    _renderNodeKfLanes(lanes, start, end, total, pixelWidth);
    updatePlayhead();
    return;
  }

  // Determine max lane index
  const maxLane = Math.max(...tlClips.map(c => c.lane), 0);
  const laneHeight = 40; // px per lane

  // Set lanes container height
  lanes.style.height = ((maxLane + 1) * laneHeight) + 'px';

  for (let li = 0; li <= maxLane; li++) {
    const laneDiv = document.createElement('div');
    laneDiv.className = 'tl-clip-lane';
    laneDiv.style.cssText = `position:absolute;top:${li * laneHeight}px;left:0;width:${pixelWidth}px;height:${laneHeight}px;border-bottom:1px solid var(--border);`;

    const clipsInLane = tlClips.filter(c => c.lane === li);
    for (const clip of clipsInLane) {
      const relStart = Math.max(0, clip.startFrame - start);
      const relEnd = Math.min(total, clip.endFrame - start + 1);
      const barX = relStart * 12;
      const barW = Math.max(4, (relEnd - relStart) * 12);

      const bar = document.createElement('div');
      bar.className = 'tl-clip-bar' + (clip._selected ? ' selected' : '');
      bar.style.cssText = `position:absolute;top:4px;left:${barX}px;width:${barW}px;height:${laneHeight - 8}px;background:${clip.color || '#4a6cf7'};border-radius:4px;cursor:grab;display:flex;align-items:center;padding:0 14px;font-size:10px;color:#fff;overflow:hidden;white-space:nowrap;text-overflow:ellipsis;z-index:5;touch-action:none;`;
      bar.textContent = clip.name;
      bar.title = `${clip.name} (F${clip.startFrame}-F${clip.endFrame})`;
      bar.dataset.clipId = clip.id;

      // Trim handles — wider touch target, narrower visual
      const handleStyle = `position:absolute;top:0;width:14px;height:100%;cursor:ew-resize;z-index:10;display:flex;align-items:center;justify-content:center;`;
      const trimL = document.createElement('div');
      trimL.className = 'tl-trim-handle';
      trimL.style.cssText = handleStyle + 'left:0;border-radius:4px 0 0 4px;background:rgba(0,0,0,0.25);';
      trimL.innerHTML = `<svg width="3" height="12" viewBox="0 0 3 12"><rect x="0" y="0" width="1" height="12" fill="rgba(255,255,255,0.6)"/><rect x="2" y="0" width="1" height="12" fill="rgba(255,255,255,0.6)"/></svg>`;
      const trimR = document.createElement('div');
      trimR.className = 'tl-trim-handle';
      trimR.style.cssText = handleStyle + 'right:0;border-radius:0 4px 4px 0;background:rgba(0,0,0,0.25);';

      // ── Loop zone overlay (only when bar extends past usable source) ──
      const srcLen = clip.srcLength || (clip.endFrame - clip.startFrame + 1);
      const trimIn = clip.trimIn || 0;
      const usableLen = Math.max(1, srcLen - trimIn);
      const barLen = clip.endFrame - clip.startFrame + 1;
      const isLooped = !!clip.looped;
      const inLoop = isLooped && (barLen > usableLen);

      // Right handle: ↺ icon when clip is in loop mode, stripes when trim mode
      const rhSvg = `<svg width="3" height="12" viewBox="0 0 3 12"><rect x="0" y="0" width="1" height="12" fill="rgba(255,255,255,0.6)"/><rect x="2" y="0" width="1" height="12" fill="rgba(255,255,255,0.6)"/></svg>`;
      trimR.innerHTML = isLooped ? `<span style="font-size:9px;color:rgba(255,255,255,0.9)">↺</span>` : rhSvg;
      bar.appendChild(trimL);
      bar.appendChild(trimR);

      if (inLoop) {
        const loopPx = (barLen - usableLen) * 12;
        const loopZone = document.createElement('div');
        loopZone.className = 'tl-loop-zone';
        loopZone.style.left = `${usableLen * 12}px`;
        loopZone.style.width = `${loopPx}px`;
        bar.appendChild(loopZone);
        // GarageBand-style notch markers: thin vertical lines + top/bottom ears at each repetition boundary
        const iterPx = usableLen * 12;
        for (let x = iterPx; x < loopPx; x += iterPx) {
          if (x >= barW) break;
          const notch = document.createElement('div');
          notch.style.cssText = `position:absolute;top:0;left:${x}px;width:2px;height:100%;background:rgba(255,255,255,0.55);z-index:3;pointer-events:none;`;
          const earT = document.createElement('div');
          earT.style.cssText = `position:absolute;top:0;left:-2px;width:6px;height:4px;background:rgba(255,255,255,0.75);border-radius:0 0 3px 3px;pointer-events:none;`;
          const earB = document.createElement('div');
          earB.style.cssText = `position:absolute;bottom:0;left:-2px;width:6px;height:4px;background:rgba(255,255,255,0.75);border-radius:3px 3px 0 0;pointer-events:none;`;
          notch.appendChild(earT);
          notch.appendChild(earB);
          loopZone.appendChild(notch);
        }
      }

      // ── Single pointerdown on the bar: route to trim or drag ──
      bar.addEventListener('pointerdown', (e) => {
        e.stopPropagation();
        e.preventDefault();
        bar.setPointerCapture(e.pointerId);

        const handle = e.target.closest('.tl-trim-handle');
        if (handle) {
          // ── Trim / loop-extend mode ──
          const side = handle === trimL ? 'L' : 'R';
          const startX = e.clientX;
          const origStart = clip.startFrame;
          const origEnd = clip.endFrame;
          const origTrimIn = clip.trimIn || 0;
          document.body.style.cursor = 'ew-resize';
          gTrimDragging = true;

          // Ensure a live-update loop zone element exists in the bar
          if (!bar.querySelector('.tl-loop-zone')) {
            const lz = document.createElement('div');
            lz.className = 'tl-loop-zone';
            lz.style.display = 'none';
            bar.appendChild(lz);
          }

          // Helper: sync loop zone position/size + right handle icon
          const rhSvg = `<svg width="3" height="12" viewBox="0 0 3 12"><rect x="0" y="0" width="1" height="12" fill="rgba(255,255,255,0.6)"/><rect x="2" y="0" width="1" height="12" fill="rgba(255,255,255,0.6)"/></svg>`;
          function updateLoopLive() {
            const lz = bar.querySelector('.tl-loop-zone');
            const nowUsable = Math.max(1, srcLen - (clip.trimIn || 0));
            const nowBarLen = clip.endFrame - clip.startFrame + 1;
            const lp = clip.looped ? Math.max(0, nowBarLen - nowUsable) * 12 : 0;
            if (lz) {
              lz.style.left = `${nowUsable * 12}px`;
              lz.style.width = `${lp}px`;
              lz.style.display = lp > 0 ? '' : 'none';
            }
            trimR.innerHTML = clip.looped
              ? `<span style="font-size:9px;color:rgba(255,255,255,0.9)">↺</span>`
              : rhSvg;
          }

          const onMove = (pe) => {
            pe.preventDefault();
            const delta = Math.round((pe.clientX - startX) / 12);
            if (side === 'L') {
              // Left trim: startFrame + trimIn shift together; interior frames hold timeline position.
              // Clamps: can't expose before source start, can't go before frame 0, must keep ≥1 frame.
              const minDelta = Math.max(-origTrimIn, -origStart, -(origEnd - origStart - 1));
              const maxDelta = srcLen - 1 - origTrimIn;
              const d = Math.max(minDelta, Math.min(maxDelta, delta));
              clip.trimIn = origTrimIn + d;
              clip.startFrame = origStart + d;
              bar.style.left = (clip.startFrame - start) * 12 + 'px';
            } else if (clip.looped) {
              // Loop mode: extend freely; clamp at next clip in same lane or section end
              const sectionEnd = parseInt(document.getElementById('tl-end')?.value) || 24;
              const nextStart = tlClips
                .filter(c => c.id !== clip.id && c.lane === clip.lane && c.startFrame > clip.startFrame)
                .reduce((mn, c) => c.startFrame < mn ? c.startFrame : mn, sectionEnd + 1);
              clip.endFrame = Math.min(nextStart - 1, Math.max(origStart + 1, origEnd + delta));
            } else {
              // Trim mode: right edge cannot extend past source content end
              const maxEnd = origStart + usableLen - 1;
              clip.endFrame = Math.max(origStart, Math.min(maxEnd, origEnd + delta));
            }
            bar.style.width = Math.max(4, (clip.endFrame - clip.startFrame + 1) * 12) + 'px';
            bar.title = `${clip.name} (F${clip.startFrame}-F${clip.endFrame})`;
            updateLoopLive();
            tlLoadFrame(parseInt(tlFrame.value) || 0);
          };
          const onUp = () => {
            gTrimDragging = false;
            document.body.style.cursor = '';
            bar.removeEventListener('pointermove', onMove);
            bar.removeEventListener('pointerup', onUp);
            bar.removeEventListener('pointercancel', onUp);
            clip._origStart = clip.startFrame;
            clip._origEnd = clip.endFrame;
            renderTimelineRuler();
            tlSaveClips();
          };
          bar.addEventListener('pointermove', onMove);
          bar.addEventListener('pointerup', onUp);
          bar.addEventListener('pointercancel', onUp);
          return;
        }

        // ── Drag mode ──
        tlClips.forEach(c => { c._selected = (c.id === clip.id); });
        const origLane = clip.lane;
        let targetLane = origLane;
        let laneMode = false;
        let inCancel = false;
        let ghostEl = null, cancelEl = null;

        const dragStartX = e.clientX;
        const dragOrigStart = clip.startFrame;
        tlDragClip = { clip, lane: li, startX: e.clientX };

        const onMove = (pe) => {
          const dx = pe.clientX - dragStartX;
          const frameDelta = Math.round(dx / 12);
          const newStart = Math.max(0, dragOrigStart + frameDelta);
          const clipLen = clip._origEnd - clip._origStart;
          // Move bar horizontally via transform (no re-render)
          bar.style.left = (newStart - start) * 12 + 'px';

          const lanesRect = lanes.getBoundingClientRect();
          const relY = pe.clientY - lanesRect.top;
          const rawLane = Math.max(0, Math.floor(relY / laneHeight));
          const hoveredLane = tlNearestFreeLane(rawLane, clip.id);

          // Activate lane mode the first time cursor enters a different lane
          if (!laneMode && hoveredLane !== origLane) {
            laneMode = true;
            bar.style.opacity = '0.75';
            bar.style.cursor = 'grabbing';
            bar.style.boxShadow = '0 6px 24px rgba(0,0,0,0.55)';
            bar.style.zIndex = '50';

            ghostEl = document.createElement('div');
            ghostEl.style.cssText = `position:absolute;
              left:${barX}px;top:${hoveredLane * laneHeight + 4}px;
              width:${barW}px;height:${laneHeight - 8}px;
              border:1.5px dashed rgba(108,142,245,0.7);border-radius:4px;
              pointer-events:none;z-index:4;
              transition:top 0.1s ease;background:rgba(108,142,245,0.15);`;
            lanes.appendChild(ghostEl);

            cancelEl = document.createElement('div');
            cancelEl.style.cssText = `position:absolute;top:-26px;left:0;right:0;height:24px;
              background:rgba(239,68,68,0.08);border:1px dashed rgba(239,68,68,0.3);
              border-radius:4px;display:flex;align-items:center;justify-content:center;
              font-size:10px;color:rgba(239,68,68,0.55);pointer-events:none;z-index:20;
              transition:background 0.12s,color 0.12s;`;
            cancelEl.textContent = '↑ drag here to cancel';
            lanes.appendChild(cancelEl);
          }

          if (!laneMode) {
            // Preview frame under playhead as clip moves horizontally
            const playheadFrame = parseInt(tlFrame.value) || 0;
            const tempStart = newStart;
            const tempEnd = newStart + clipLen;
            if (playheadFrame >= tempStart && playheadFrame <= tempEnd) {
              tlLoadFrame(playheadFrame);
            } else {
              tlClearPreview();
            }
            return;
          }

          // Cancel zone: cursor above lanes area
          const nowCancel = relY < 0;
          if (nowCancel !== inCancel) {
            inCancel = nowCancel;
            bar.style.opacity = inCancel ? '0.3' : '0.75';
            bar.style.filter = inCancel ? 'grayscale(0.8)' : '';
            if (cancelEl) {
              cancelEl.style.background = inCancel ? 'rgba(239,68,68,0.28)' : 'rgba(239,68,68,0.08)';
              cancelEl.style.color = inCancel ? '#ef4444' : 'rgba(239,68,68,0.55)';
            }
          }

          if (!inCancel) {
            if (hoveredLane !== targetLane) targetLane = hoveredLane;
            if (ghostEl) {
              const gx = Math.max(0, newStart - start) * 12;
              const gw = Math.max(4, (clipLen + 1) * 12);
              ghostEl.style.top = (targetLane * laneHeight + 4) + 'px';
              ghostEl.style.left = gx + 'px';
              ghostEl.style.width = gw + 'px';
            }
            bar.style.top = (targetLane * laneHeight + 4) + 'px';
          } else {
            bar.style.top = Math.max(0, relY - laneHeight / 2) + 'px';
          }
        };

        const onUp = () => {
          if (ghostEl) { ghostEl.remove(); ghostEl = null; }
          if (cancelEl) { cancelEl.remove(); cancelEl = null; }
          const dx = (parseInt(bar.style.left) || barX) - barX;
          const frameDelta = Math.round(dx / 12);
          const trimmedDuration = clip.endFrame - clip.startFrame;
          clip.startFrame = Math.max(0, dragOrigStart + frameDelta);
          clip.endFrame = clip.startFrame + trimmedDuration;
          clip.lane = (laneMode && !inCancel) ? targetLane : origLane;
          renderTimelineRuler();
          bar.removeEventListener('pointermove', onMove);
          bar.removeEventListener('pointerup', onUp);
          bar.removeEventListener('pointercancel', onUp);
          tlDragClip = null;
          tlSaveClips();
        };

        bar.addEventListener('pointermove', onMove);
        bar.addEventListener('pointerup', onUp);
        bar.addEventListener('pointercancel', onUp);
      });

      // Click to select clip
      bar.addEventListener('click', (e) => {
        e.stopPropagation();
        tlSelectClip(clip.id);
      });

      laneDiv.appendChild(bar);
    }

    lanes.appendChild(laneDiv);
  }

  // ── Per-clip keyframe lanes (shown below clip bars for selected clip) ──
  const selectedClip = tlClips.find(c => c._selected);
  if (selectedClip) {
    renderClipKeyframeLanes(selectedClip, start, total, pixelWidth, lanes);
  }

  // ── Phase 5: Per-node param KF lanes for currently selected node ──
  _renderNodeKfLanes(lanes, start, end, total, pixelWidth);

  // Update playhead position
  updatePlayhead();
}

function renderClipKeyframeLanes(clip, start, total, pixelWidth, lanesContainer) {
  const pkf = clip.paramKeyframes || {};
  const paramNames = Object.keys(pkf);
  if (paramNames.length === 0) return;

  const baseY = (clip.lane + 1) * 40; // below the clip lane
  paramNames.forEach((pname, pi) => {
    const kfs = pkf[pname] || [];
    if (kfs.length === 0) return;

    const laneDiv = document.createElement('div');
    laneDiv.className = 'tl-kf-lane';
    laneDiv.style.cssText = `position:absolute;top:${baseY + pi * 20}px;left:0;width:${pixelWidth}px;height:20px;border-bottom:1px dashed var(--border);opacity:0.7;`;

    const label = document.createElement('div');
    label.style.cssText = `position:absolute;left:2px;top:1px;font-size:8px;color:var(--muted);white-space:nowrap;`;
    label.textContent = pname;
    laneDiv.appendChild(label);

    const sorted = [...kfs].sort((a, b) => a.frame - b.frame);
    for (const kf of sorted) {
      const relFrame = kf.frame - start;
      if (relFrame < 0 || relFrame > total) continue;
      const x = relFrame * 12 + 6;
      const diamond = document.createElement('div');
      diamond.className = 'tl-kf-diamond ' + (kf.easing || 'linear');
      diamond.style.cssText = `position:absolute;top:4px;left:${x}px;width:8px;height:8px;border-radius:50%;background:${clip.color || '#4a6cf7'};border:1px solid #fff;cursor:pointer;z-index:6;`;
      const val = typeof kf.value === 'number' ? kf.value.toFixed(2) : kf.value;
      diamond.title = `${pname} @ F${kf.frame} = ${val} (${kf.easing || 'linear'})`;
      diamond.dataset.clipId = clip.id;
      diamond.dataset.pname = pname;
      diamond.dataset.frame = kf.frame;
      diamond.addEventListener('click', (e) => {
        e.stopPropagation();
        document.getElementById('tl-frame').value = kf.frame;
        document.getElementById('tl-frame').dispatchEvent(new Event('input'));
      });
      laneDiv.appendChild(diamond);
    }
    lanesContainer.appendChild(laneDiv);
  });
  // Update container height so _renderNodeKfLanes can read it correctly
  const kfCount = Object.values(pkf).filter(v => v && v.length > 0).length;
  if (kfCount > 0) {
    const baseY = (clip.lane + 1) * 40;
    const needed = baseY + kfCount * 20;
    const cur = parseInt(lanesContainer.style.height) || 0;
    if (needed > cur) lanesContainer.style.height = needed + 'px';
  }
}

// ── Phase 5: JS easing utilities (mirror of core/easing.py) ──────────────

const _KF_EASING_COLORS = {
  'linear': '#888', 'ease': '#6c8ef5', 'ease-in': '#f59e0b',
  'ease-out': '#34d399', 'ease-in-out': '#6c8ef5',
  'step': '#f87171', 'bounce': '#a78bfa', 'elastic': '#f472b6', 'cubic-bezier': '#2dd4bf',
};

function _kfCubicBezier(t, p1x, p1y, p2x, p2y) {
  function sX(u) { return 3*(1-u)*(1-u)*u*p1x + 3*(1-u)*u*u*p2x + u*u*u; }
  function sY(u) { return 3*(1-u)*(1-u)*u*p1y + 3*(1-u)*u*u*p2y + u*u*u; }
  function dX(u) { return 3*(1-u)*(1-u)*p1x + 6*(1-u)*u*(p2x-p1x) + 3*u*u*(1-p2x); }
  let g = t;
  for (let i = 0; i < 8; i++) {
    const x = sX(g) - t, dx = dX(g);
    if (Math.abs(x) < 1e-7 || Math.abs(dx) < 1e-7) break;
    g = Math.max(0, Math.min(1, g - x/dx));
  }
  return sY(g);
}

function _kfApplyEasing(t, easing, handle_in, handle_out) {
  t = Math.max(0, Math.min(1, t));
  if (easing === 'step') return t < 1.0 ? 0.0 : 1.0;
  if (easing === 'bounce') {
    if (t < 1/2.75) return 7.5625*t*t;
    if (t < 2/2.75) { const u=t-1.5/2.75; return 7.5625*u*u+0.75; }
    if (t < 2.5/2.75) { const u=t-2.25/2.75; return 7.5625*u*u+0.9375; }
    const u=t-2.625/2.75; return 7.5625*u*u+0.984375;
  }
  if (easing === 'elastic') {
    if (t === 0 || t === 1) return t;
    return -Math.pow(2, 10*(t-1)) * Math.sin((t-1-0.075)*(2*Math.PI)/0.3);
  }
  const presets = { linear:[0,0,1,1], ease:[0.25,0.1,0.25,1], 'ease-in':[0.42,0,1,1], 'ease-out':[0,0,0.58,1], 'ease-in-out':[0.42,0,0.58,1] };
  let p = presets[easing];
  if (easing === 'cubic-bezier' && handle_in && handle_out) p = [handle_in[0], handle_in[1], handle_out[0], handle_out[1]];
  if (!p) return t;
  return _kfCubicBezier(t, p[0], p[1], p[2], p[3]);
}

function _kfEvaluateTrack(kfs, frame) {
  if (!kfs || !kfs.length) return null;
  const s = [...kfs].sort((a, b) => a.frame - b.frame);
  if (frame <= s[0].frame) return s[0].value;
  if (frame >= s[s.length-1].frame) return s[s.length-1].value;
  for (let i = 0; i < s.length - 1; i++) {
    const a = s[i], b = s[i+1];
    if (a.frame <= frame && frame < b.frame) {
      const w = b.frame - a.frame;
      if (!w) return b.value;
      const tE = _kfApplyEasing((frame - a.frame) / w, b.easing || 'linear', b.handle_in, b.handle_out);
      if (typeof a.value === 'number' && typeof b.value === 'number') return a.value + (b.value - a.value) * tE;
      return tE < 0.5 ? a.value : b.value;
    }
  }
  return null;
}

// ── Phase 5: Per-node KF lanes in timeline ────────────────────────────────

function _renderNodeKfLanes(lanes, start, end, total, pixelWidth) {
  if (!gSelectedNode) return;
  const node = gNodes.find(n => n.id === gSelectedNode);
  if (!node || !node.paramKeyframes) return;
  const pkf = node.paramKeyframes;
  const paramNames = Object.keys(pkf).filter(p => pkf[p] && pkf[p].length > 0);
  if (!paramNames.length) return;

  const LANE_H = 24, HEADER_H = 18;
  const curH = parseInt(lanes.style.height) || 0;
  const def = gNodeDefs[node.method_id];
  const nodeName = def ? def.name : node.id;

  // Section header
  const hdrDiv = document.createElement('div');
  hdrDiv.style.cssText = `position:absolute;top:${curH}px;left:0;width:${pixelWidth}px;height:${HEADER_H}px;border-top:2px solid var(--accent);background:#181818;z-index:4;display:flex;align-items:center;`;
  const hlbl = document.createElement('div');
  hlbl.className = 'tl-node-kf-header';
  hlbl.textContent = `⬤ ${nodeName}`;
  hdrDiv.appendChild(hlbl);
  lanes.appendChild(hdrDiv);

  const newH = curH + HEADER_H + paramNames.length * LANE_H;
  lanes.style.height = newH + 'px';

  paramNames.forEach((pname, pi) => {
    const kfs = [...(pkf[pname] || [])].sort((a, b) => a.frame - b.frame);
    const laneY = curH + HEADER_H + pi * LANE_H;

    const laneDiv = document.createElement('div');
    laneDiv.className = 'tl-lane';
    laneDiv.style.cssText = `position:absolute;top:${laneY}px;left:0;width:${pixelWidth}px;height:${LANE_H}px;display:flex;`;

    const label = document.createElement('div');
    label.className = 'tl-lane-label';
    label.style.cssText = `line-height:${LANE_H}px;height:${LANE_H}px;font-size:9px;`;
    label.title = pname;
    label.textContent = pname;
    laneDiv.appendChild(label);

    const track = document.createElement('div');
    track.className = 'tl-lane-track';
    track.style.cssText = `position:absolute;left:100px;right:0;top:0;height:${LANE_H}px;`;

    // Easing-curve SVG between adjacent KFs
    if (kfs.length > 1) {
      const vAll = kfs.map(k => typeof k.value === 'number' ? k.value : 0);
      const vMin = Math.min(...vAll), vMax = Math.max(...vAll);
      const vRange = Math.max(0.001, vMax - vMin);
      const svg = document.createElementNS('http://www.w3.org/2000/svg', 'svg');
      svg.style.cssText = 'position:absolute;left:0;top:0;width:100%;height:100%;pointer-events:none;overflow:visible;';
      for (let i = 0; i < kfs.length - 1; i++) {
        const kfA = kfs[i], kfB = kfs[i+1];
        const relA = kfA.frame - start, relB = kfB.frame - start;
        if (relB < 0 || relA > total) continue;
        const x1 = relA * 12 + 6, x2 = relB * 12 + 6;
        let pts = [];
        for (let s = 0; s <= 24; s++) {
          const t = s / 24;
          const tE = _kfApplyEasing(t, kfB.easing || 'linear', kfB.handle_in, kfB.handle_out);
          const vA = typeof kfA.value === 'number' ? kfA.value : 0;
          const vB = typeof kfB.value === 'number' ? kfB.value : 0;
          const v = vA + (vB - vA) * tE;
          const yNorm = (v - vMin) / vRange;
          const y = (LANE_H - 3) - yNorm * (LANE_H - 7);
          pts.push(`${(x1 + (x2-x1)*t).toFixed(1)},${y.toFixed(1)}`);
        }
        const path = document.createElementNS('http://www.w3.org/2000/svg', 'path');
        path.setAttribute('d', 'M ' + pts.join(' L '));
        path.setAttribute('fill', 'none');
        path.setAttribute('stroke', _KF_EASING_COLORS[kfB.easing || 'linear'] || '#6c8ef5');
        path.setAttribute('stroke-width', '1.5');
        path.setAttribute('opacity', '0.55');
        svg.appendChild(path);
      }
      track.appendChild(svg);
    }

    // Click-on-track → create KF
    track.addEventListener('pointerdown', (e) => {
      if (e.button !== 0 || e.target.closest('.tl-kf-diamond')) return;
      e.preventDefault();
      const r = track.getBoundingClientRect();
      const clickFrame = Math.max(start, Math.min(end, start + Math.round((e.clientX - r.left) / 12)));
      if (kfs.some(k => k.frame === clickFrame)) {
        document.getElementById('tl-frame').value = clickFrame;
        document.getElementById('tl-frame').dispatchEvent(new Event('input'));
        return;
      }
      // Value: interpolate existing track at cursor frame, else use node param default
      const cursorFrame = parseInt(document.getElementById('tl-frame')?.value) || 0;
      let curVal = kfs.length >= 2 ? (_kfEvaluateTrack(kfs, cursorFrame) ?? 0) :
                   kfs.length === 1 ? kfs[0].value :
                   (node.params[pname] ?? (def && def.params && def.params[pname] ? def.params[pname].default : 0));
      if (typeof curVal !== 'number') curVal = 0;
      if (!node.paramKeyframes[pname]) node.paramKeyframes[pname] = [];
      node.paramKeyframes[pname].push({ frame: clickFrame, value: parseFloat(curVal.toFixed(4)), easing: 'ease-in-out', handle_in: null, handle_out: null });
      node.paramKeyframes[pname].sort((a, b) => a.frame - b.frame);
      document.getElementById('tl-frame').value = clickFrame;
      document.getElementById('tl-frame').dispatchEvent(new Event('input'));
      gSave(); renderTimelineRuler();
      if (gSelectedNode === node.id && def) renderPerParamKeyframes(node, Object.entries(def.params || {}));
      gShowToast(`KF: ${pname} @ F${clickFrame} = ${curVal.toFixed(3)}`);
    });

    // KF diamonds: drag + click + right-click-delete
    kfs.forEach((kf, kfIdx) => {
      const relFrame = kf.frame - start;
      if (relFrame < 0 || relFrame > total) return;
      const diamond = document.createElement('div');
      diamond.className = 'tl-kf-diamond ' + (kf.easing || 'linear');
      diamond.style.left = (relFrame * 12 + 6) + 'px';
      const valStr = typeof kf.value === 'number' ? kf.value.toFixed(3) : kf.value;
      diamond.title = `${pname} @ F${kf.frame} = ${valStr} (${kf.easing || 'linear'})  |  drag to move, right-click to delete`;

      diamond.addEventListener('click', (e) => {
        e.stopPropagation();
        document.getElementById('tl-frame').value = kf.frame;
        document.getElementById('tl-frame').dispatchEvent(new Event('input'));
        if (gSelectedNode === node.id && def) openPerParamKfEditor(node, pname, kfIdx, kf);
      });

      diamond.addEventListener('contextmenu', (e) => {
        e.preventDefault(); e.stopPropagation();
        const arr = node.paramKeyframes[pname];
        if (!arr) return;
        const ai = arr.findIndex(k => k === kf);
        arr.splice(ai < 0 ? kfIdx : ai, 1);
        if (!arr.length) delete node.paramKeyframes[pname];
        gSave(); renderTimelineRuler();
        if (gSelectedNode === node.id && def) renderPerParamKeyframes(node, Object.entries(def.params || {}));
        gShowToast(`Deleted KF: ${pname} @ F${kf.frame}`, true);
      });

      // Drag to reposition
      let _dx0 = 0, _f0 = 0, _kfDragging = false;
      diamond.addEventListener('pointerdown', (e) => {
        if (e.button !== 0) return;
        e.stopPropagation(); e.preventDefault();
        diamond.setPointerCapture(e.pointerId);
        _dx0 = e.clientX; _f0 = kf.frame; _kfDragging = true;
        diamond.classList.add('selected');
        document.body.style.cursor = 'ew-resize';
      });
      diamond.addEventListener('pointermove', (e) => {
        if (!_kfDragging) return;
        const nf = Math.max(start, Math.min(end, _f0 + Math.round((e.clientX - _dx0) / 12)));
        if (nf !== kf.frame) {
          kf.frame = nf;
          diamond.style.left = ((nf - start) * 12 + 6) + 'px';
          document.getElementById('tl-frame').value = nf;
          updatePlayhead();
        }
      });
      const _kfUp = () => {
        if (!_kfDragging) return;
        _kfDragging = false; document.body.style.cursor = '';
        diamond.classList.remove('selected');
        node.paramKeyframes[pname].sort((a, b) => a.frame - b.frame);
        gSave(); renderTimelineRuler();
        if (gSelectedNode === node.id && def) renderPerParamKeyframes(node, Object.entries(def.params || {}));
      };
      diamond.addEventListener('pointerup', _kfUp);
      diamond.addEventListener('pointercancel', _kfUp);

      track.appendChild(diamond);
    });

    laneDiv.appendChild(track);
    lanes.appendChild(laneDiv);
  });
}

// Return the lowest lane index not occupied by any clip other than excludeId.
function tlNextFreeLane(excludeId = null) {
  const occupied = new Set(tlClips.filter(c => c.id !== excludeId).map(c => c.lane));
  let lane = 0;
  while (occupied.has(lane)) lane++;
  return lane;
}
function tlLaneOccupied(lane, excludeId = null) {
  return tlClips.some(c => c.id !== excludeId && c.lane === lane);
}
// Nearest free lane to `desired`, searching outward from it.
function tlNearestFreeLane(desired, excludeId = null) {
  if (!tlLaneOccupied(desired, excludeId)) return desired;
  for (let d = 1; d < 100; d++) {
    if (!tlLaneOccupied(desired + d, excludeId)) return desired + d;
    if (desired - d >= 0 && !tlLaneOccupied(desired - d, excludeId)) return desired - d;
  }
  return tlNextFreeLane(excludeId);
}

function tlSelectClip(clipId) {
  tlClips.forEach(c => { c._selected = (c.id === clipId); });
  renderTimelineRuler();
}

function tlDeleteClip(clipId) {
  tlClips = tlClips.filter(c => c.id !== clipId);
  tlSaveClips();
  renderTimelineRuler();
}

function tlSplitClipAtPlayhead(clipId) {
  const tf = parseInt(document.getElementById('tl-frame')?.value) || 0;
  const idx = tlClips.findIndex(c => c.id === clipId);
  if (idx === -1) return;
  const clip = tlClips[idx];
  const S = clip.startFrame, E = clip.endFrame;
  // Playhead must be strictly inside: left half needs ≥1 frame, right half needs ≥1 frame
  if (tf <= S || tf > E) return;

  const T = clip.trimIn || 0;
  const L = clip.srcLength || (E - S + 1);
  const usableLen = Math.max(1, L - T);
  // Source frame the playhead lands on — this becomes the right clip's trimIn
  const rightTrimIn = T + ((tf - S) % usableLen);

  const leftClip = {
    ...clip,
    id: 'clip_' + (++tlClipIdCounter),
    endFrame: tf - 1,
    _origStart: S, _origEnd: tf - 1,
    _selected: false,
  };
  const rightClip = {
    ...clip,
    id: 'clip_' + (++tlClipIdCounter),
    startFrame: tf,
    trimIn: rightTrimIn,
    _origStart: tf, _origEnd: E,
    _selected: false,
  };

  tlClips.splice(idx, 1, leftClip, rightClip);
  renderTimelineRuler();
  tlSaveClips();
}

// Delete key removes selected clip
document.addEventListener('keydown', (e) => {
  const tag = document.activeElement?.tagName;
  const isInput = tag === 'INPUT' || tag === 'TEXTAREA';

  if (e.key === ' ' && !isInput) {
    e.preventDefault();
    tlPlay.click();
    return;
  }

  if ((e.key === 'Delete' || e.key === 'Backspace') && !isInput) {
    const sel = tlClips.find(c => c._selected);
    if (!sel) return;
    e.preventDefault();
    tlDeleteClip(sel.id);
  }
});

function tlAddClip(name, seqName, startFrame, endFrame, nodeId, paramKeyframes) {
  const id = 'clip_' + (++tlClipIdCounter);
  const colors = ['#4a6cf7', '#f59e0b', '#34d399', '#ef4444', '#8b5cf6', '#ec4899', '#06b6d4', '#f97316'];
  const color = colors[tlClips.length % colors.length];
  // New clip goes at the top (lane 0); shift existing clips down by 1
  tlClips.forEach(c => { c.lane += 1; });
  const clip = {
    id, name, seqName, startFrame, endFrame, lane: 0, color,
    nodeId, paramKeyframes: JSON.parse(JSON.stringify(paramKeyframes || {})),
    srcLength: endFrame - startFrame + 1,
    trimIn: 0,
    looped: false,
    srcOffset: startFrame,
    _origStart: startFrame, _origEnd: endFrame,
  };
  tlClips.unshift(clip);
  tlSaveClips();
  renderTimelineRuler();
  return clip;
}

function tlSaveClips() {
  try {
    localStorage.setItem('tl-clips', JSON.stringify(tlClips.map(c => {
      const { _selected, _origStart, _origEnd, ...rest } = c;
      return rest;
    })));
  } catch {}
}

function tlLoadClips() {
  try {
    const data = localStorage.getItem('tl-clips');
    if (data) {
      tlClips = JSON.parse(data).map(c => ({
        srcLength: c.endFrame - c.startFrame + 1,
        ...c,
        looped: c.looped ?? false,
        _origStart: c.startFrame,
        _origEnd: c.endFrame,
      }));
      tlClipIdCounter = Math.max(...tlClips.map(c => parseInt(c.id.replace('clip_', '')) || 0), 0);
    }
  } catch {}
}

function updatePlayhead() {
  const playhead = document.getElementById('tl-playhead');
  if (!playhead) return;
  const start = parseInt(document.getElementById('tl-start')?.value) || 0;
  const frame = parseInt(document.getElementById('tl-frame')?.value) || 0;
  playhead.style.left = Math.max(0, (frame - start) * 12) + 'px';
}

// Re-render timeline ruler when frame range changes
document.getElementById('tl-start')?.addEventListener('input', () => { renderTimelineRuler(); gUpdateFrameCount(); });
document.getElementById('tl-end')?.addEventListener('input', () => { renderTimelineRuler(); gUpdateFrameCount(); });
document.getElementById('tl-fps')?.addEventListener('input', renderTimelineRuler);

// Update playhead on frame change
document.getElementById('tl-frame')?.addEventListener('input', updatePlayhead);

// ── Frame navigation ────────────────────────────────────────────
function tlStopPlay() {
  if (!tlPlaying) return;
  tlPlaying = false;
  tlPlay.textContent = '▶';
  _tlHaltPlayback();
}
tlPrev.addEventListener('click', () => {
  tlStopPlay();
  tlFrame.value = Math.max(0, (parseInt(tlFrame.value) || 0) - 1);
  tlFrame.dispatchEvent(new Event('input'));
});
tlNext.addEventListener('click', () => {
  tlStopPlay();
  tlFrame.value = (parseInt(tlFrame.value) || 0) + 1;
  tlFrame.dispatchEvent(new Event('input'));
});

// ── Playhead drag on timeline ruler header ──────────────────────
(function() {
  const wrap = document.getElementById('timeline-ruler-wrap');
  if (!wrap) return;
  let dragging = false;
  function frameFromEvent(e) {
    const ruler = document.getElementById('timeline-ruler');
    const rect = ruler.getBoundingClientRect();
    const x = ('touches' in e ? e.touches[0].clientX : e.clientX) - rect.left + wrap.scrollLeft;
    const start = parseInt(tlStart.value) || 0;
    const end   = parseInt(tlEnd.value)   || 24;
    return Math.max(start, Math.min(end, start + Math.round(x / 12)));
  }
  function onStart(e) {
    if (!e.target.closest('#tl-ruler-header') && !e.target.closest('#tl-playhead')) return;
    dragging = true;
    tlStopPlay();
    const f = frameFromEvent(e);
    tlFrame.value = f;
    updatePlayhead();
    tlLoadFrame(f);
  }
  function onMove(e) {
    if (!dragging) return;
    e.preventDefault();
    const f = frameFromEvent(e);
    tlFrame.value = f;
    updatePlayhead();
    tlLoadFrame(f);
  }
  function onEnd() { dragging = false; }
  wrap.addEventListener('mousedown', onStart);
  document.addEventListener('mousemove', onMove);
  document.addEventListener('mouseup', onEnd);
  wrap.addEventListener('touchstart', onStart, { passive: false });
  document.addEventListener('touchmove', onMove, { passive: false });
  document.addEventListener('touchend', onEnd);
})();

// ── Pinch zoom on timeline ────────────────────────────────────
(function() {
  const wrap = document.getElementById('timeline-ruler-wrap');
  if (!wrap) return;
  let lastDist = 0;
  wrap.addEventListener('touchstart', e => {
    if (e.touches.length === 2) {
      lastDist = Math.hypot(
        e.touches[0].clientX - e.touches[1].clientX,
        e.touches[0].clientY - e.touches[1].clientY
      );
    }
  }, { passive: true });
  wrap.addEventListener('touchmove', e => {
    if (e.touches.length !== 2) return;
    e.preventDefault();
    const dist = Math.hypot(
      e.touches[0].clientX - e.touches[1].clientX,
      e.touches[0].clientY - e.touches[1].clientY
    );
    const scale = dist / lastDist;
    lastDist = dist;
    const current = parseInt(tlEnd.value) - parseInt(tlStart.value) + 1;
    const newTotal = Math.max(4, Math.min(200, Math.round(current / scale)));
    const mid = (parseInt(tlStart.value) + parseInt(tlEnd.value)) / 2;
    const half = Math.round(newTotal / 2);
    tlStart.value = Math.max(0, Math.round(mid - half));
    tlEnd.value = tlStart.value + newTotal - 1;
    renderTimelineRuler();
    gUpdateFrameCount();
  }, { passive: false });
})();

// ── Timeline resize handle ────────────────────────────────────
(function() {
  const handle = document.getElementById('timeline-resize-handle');
  const wrap   = document.getElementById('timeline-ruler-wrap');
  if (!handle || !wrap) return;
  let dragging = false, startY = 0, startH = 0;
  function onStart(e) {
    dragging = true;
    startY = e.touches ? e.touches[0].clientY : e.clientY;
    startH = wrap.getBoundingClientRect().height;
    handle.classList.add('dragging');
    e.preventDefault();
  }
  function onMove(e) {
    if (!dragging) return;
    const y = e.touches ? e.touches[0].clientY : e.clientY;
    const newH = Math.max(44, Math.min(500, startH - (y - startY)));
    wrap.style.height = newH + 'px';
  }
  function onEnd() { dragging = false; handle.classList.remove('dragging'); }
  handle.addEventListener('mousedown', onStart);
  document.addEventListener('mousemove', onMove);
  document.addEventListener('mouseup', onEnd);
  handle.addEventListener('touchstart', onStart, { passive: false });
  document.addEventListener('touchmove', onMove, { passive: false });
  document.addEventListener('touchend', onEnd);
})();

// ── Playback clock ───────────────────────────────────────────────
// The frame position is derived from elapsed wall time rather than accumulated
// one tick at a time. The old loop was setInterval(Math.round(1000/fps)) with
// `next = cur + 1`, which drifted two ways and recovered from neither:
//
//   - The delay is rounded to whole milliseconds. At 24fps that is 42ms against
//     a true 41.667ms, so playback ran 0.8% slow — a measured 8.05ms of lag per
//     second of playing, growing without bound (283ms after 30s, ~2.4s after
//     five minutes). At 60fps the rounding is 17ms vs 16.667ms, three times
//     worse. Deriving the frame from `performance.now()` has no remainder to
//     lose, so the only error left is a sub-frame quantisation that never
//     accumulates.
//   - A tick that fired late still advanced exactly one frame, so any stall
//     (GC, a slow decode, a busy main thread) permanently stretched playback
//     instead of being caught up. Frames are now skipped to stay on the clock.
//
// requestAnimationFrame rather than a timer: it cannot outrun the display, and
// a backgrounded tab pauses cleanly instead of crawling — Chrome clamps
// setInterval to 1Hz when hidden, which made playback advance 28 frames in 26
// seconds and then resume that far behind.
function _tlPlayFrom(frame) {
  _tlPlayClock = { at: performance.now(), frame, fps: Math.max(1, parseFloat(tlFps.value) || 24) };
  _tlPlayShown = frame;
}

function _tlHaltPlayback() {
  if (tlPlayRaf) { cancelAnimationFrame(tlPlayRaf); tlPlayRaf = null; }
  _tlPlayClock = null;
  _tlPlayShown = null;
}

function _tlPlayTick() {
  if (!tlPlaying || !_tlPlayClock) { tlPlayRaf = null; return; }
  const st   = parseInt(tlStart.value) || 0;
  const end  = parseInt(tlEnd.value)   || 24;
  const span = Math.max(1, end - st + 1);
  const fps  = Math.max(1, parseFloat(tlFps.value) || 24);

  // Scrubbing or an fps change mid-playback re-bases the clock on the new
  // position instead of yanking the playhead back to where the old one points.
  const shown = parseInt(tlFrame.value);
  if (fps !== _tlPlayClock.fps || (!isNaN(shown) && shown !== _tlPlayShown)) {
    _tlPlayFrom(isNaN(shown) ? st : shown);
  }

  const advanced = Math.floor((performance.now() - _tlPlayClock.at) * fps / 1000);
  const next = st + ((((_tlPlayClock.frame - st + advanced) % span) + span) % span);
  if (next !== _tlPlayShown) {
    _tlPlayShown = next;
    tlFrame.value = next;
    updatePlayhead();
    tlLoadFrame(next);
  }
  tlPlayRaf = requestAnimationFrame(_tlPlayTick);
}

tlPlay.addEventListener('click', () => {
  if (tlPlaying) { tlStopPlay(); return; }
  tlPlaying = true;
  tlPlay.textContent = '⏸';
  _tlPlayFrom(parseInt(tlFrame.value) || 0);
  tlPlayRaf = requestAnimationFrame(_tlPlayTick);
});

// ── Frame scrubbing: preview rendered sequence frames ───────────
tlFrame.addEventListener('input', () => {
  const frame = parseInt(tlFrame.value);
  if (isNaN(frame)) return;
  tlLoadFrame(frame);
  // Backfill keyframe values into node params for the active clip
  const activeClip = tlClips.find(c => c._selected);
  if (activeClip && activeClip.paramKeyframes) {
    for (const [pname, kfs] of Object.entries(activeClip.paramKeyframes)) {
      if (!kfs || kfs.length === 0) continue;
      const sorted = [...kfs].sort((a, b) => a.frame - b.frame);
      // Find the interpolated value at this frame
      let val = null;
      if (frame <= sorted[0].frame) {
        val = sorted[0].value;
      } else if (frame >= sorted[sorted.length - 1].frame) {
        val = sorted[sorted.length - 1].value;
      } else {
        for (let i = 0; i < sorted.length - 1; i++) {
          if (sorted[i].frame <= frame && frame < sorted[i + 1].frame) {
            const window = sorted[i + 1].frame - sorted[i].frame;
            if (window <= 0) { val = sorted[i + 1].value; break; }
            const t = (frame - sorted[i].frame) / window;
            const a = sorted[i].value, b = sorted[i + 1].value;
            if (typeof a === 'number' && typeof b === 'number') {
              val = a + (b - a) * t;
            } else {
              val = t < 0.5 ? a : b;
            }
            break;
          }
        }
      }
      if (val !== null) {
        // Backfill into the node's params
        for (const node of gNodes) {
          if (node.params && pname in node.params) {
            node.params[pname] = val;
            // Update the UI slider if the node is selected
            const el = document.getElementById(`p_${pname}`);
            if (el) {
              el.value = val;
              const v = document.querySelector(`#val_${pname}`);
              if (v) v.textContent = formatVal(val);
            }
          }
        }
      }
    }
  }
});

// ── Render Sequence ─────────────────────────────────────────────
tlRenderBtn.addEventListener('click', tlDoRenderSequence);

tlCancel.addEventListener('click', () => {
  if (tlSeqAbort) { tlSeqAbort.abort(); tlSeqAbort = null; }
});

// Client-side export for 3D-containing graphs. Renders frames on the browser
// GPU and captures a WebM. The server export path (/api/graph/render-sequence)
// is untouched and still used for every 2D-only graph.
async function tlDoClientExport(start, end, fps, name, total) {
  tlRenderBtn.disabled        = true;
  tlProgressDiv.style.display = 'flex';
  tlProgressBar.style.width   = '0%';
  tlProgressLabel.textContent = `0 / ${total}`;
  const wasLive = _gClientLiveActive;
  try {
    const C = await gClient3D();
    const { nodes, edges } = _gClientGraphPayload();
    const blob = await C.exportWebM({
      nodes, edges, start, end, fps, width: gCanvasW, height: gCanvasH,
      onProgress: (i, t) => {
        tlProgressBar.style.width = `${Math.round(i / t * 100)}%`;
        tlProgressLabel.textContent = `${i} / ${t}`;
      },
    });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url; a.download = `${name}.webm`;
    document.body.appendChild(a); a.click(); a.remove();
    setTimeout(() => URL.revokeObjectURL(url), 10000);
    gShowToast(`Exported ${name}.webm (${total} frames, client GPU)`);
  } catch (e) {
    gShowToast('Client export failed: ' + e.message, true);
    console.error(e);
  } finally {
    tlRenderBtn.disabled = false;
    setTimeout(() => { tlProgressDiv.style.display = 'none'; }, 1200);
    if (wasLive) { try { await _gStartClientLive(); } catch {} }
  }
}

async function tlDoRenderSequence() {
  const start = parseInt(tlStart.value) || 0;
  const end   = parseInt(tlEnd.value)   || 24;
  const fps   = parseInt(tlFps.value)   || 24;
  const name  = tlName.value.trim()     || 'sequence';

  if (end <= start) { gShowToast('End must be greater than Start', true); return; }

  const total = end - start + 1;

  // 3D / client-rendered graphs export entirely in the browser.
  if (gGraphRunsOnClient()) { await tlDoClientExport(start, end, fps, name, total); return; }

  tlRenderBtn.disabled        = true;
  tlProgressDiv.style.display = 'flex';
  tlProgressBar.style.width   = '0%';
  tlProgressLabel.textContent = `0 / ${total}`;

  tlSeqAbort = new AbortController();

  // Same client-only strip as gServerGraphPayload — this path carries its own
  // node shape (animParams) so it cannot share the helper, only its rule.
  const tlDrop = new Set(gNodes.filter(n => gNodeDefs[n.method_id]?.clientExec).map(n => n.id));

  const body = {
    graph: {
      nodes: gNodes.filter(n => !tlDrop.has(n.id)).map(n => ({
        id: n.id, method_id: n.method_id, params: n.params,
        animParams: n.animParams || {},
        x: n.x, y: n.y, render: !!n.render, dirty: n.dirty !== false,
        start_frame: n.start_frame || 0, end_frame: n.end_frame || 0,
        keyframes: n.keyframes || [],
        paramKeyframes: n.paramKeyframes || {},
        prebake: n.prebake || 0,
      })),
      edges: gEdges.filter(e => !tlDrop.has(e.src_node) && !tlDrop.has(e.dst_node)).map(e => ({
        src_node: e.src_node, src_port: e.src_port,
        dst_node: e.dst_node, dst_port: e.dst_port, feedback: e.feedback,
      })),
      seed: 42,
    },
    start_frame: start,
    end_frame:   end,
    fps,
    output_name: name,
    width: gCanvasW, height: gCanvasH,
  };

  try {
    const res = await fetch('/api/graph/render-sequence', {
      method:  'POST',
      headers: { 'Content-Type': 'application/json' },
      body:    JSON.stringify(body),
      signal:  tlSeqAbort.signal,
    });
    if (!res.ok || !res.body) throw new Error(`Server error ${res.status}`);

    const reader  = res.body.getReader();
    const decoder = new TextDecoder();
    let buf = '';

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      buf += decoder.decode(value, { stream: true });
      const parts = buf.split('\n');
      buf = parts.pop(); // hold incomplete line

      let evType = '', evData = '';
      for (const line of parts) {
        if (line.startsWith('event:')) {
          evType = line.slice(6).trim();
        } else if (line.startsWith('data:')) {
          evData = line.slice(5).trim();
        } else if (line === '' && evType) {
          tlHandleSeqEvent(evType, evData, start, total);
          evType = ''; evData = '';
        }
      }
    }
  } catch (e) {
    if (e.name !== 'AbortError') gShowToast('Sequence error: ' + e.message, true);
  } finally {
    tlRenderBtn.disabled        = false;
    tlProgressDiv.style.display = 'none';
    tlSeqAbort = null;
  }
}

function tlHandleSeqEvent(type, data, startFrame, total) {
  try {
    if (type === 'frame-done') {
      const d    = JSON.parse(data);
      const done = ((d.frame ?? startFrame) - startFrame) + 1;
      tlProgressBar.style.width   = (done / total * 100) + '%';
      tlProgressLabel.textContent = `${done} / ${total}`;
      if (d.data) gGraphFrameUpdate(d.data);  // inline JPEG preview
    } else if (type === 'frame-error') {
      const d = JSON.parse(data);
      gShowToast(`Frame ${d.frame} failed: ${d.error}`, true);
    } else if (type === 'sequence-done') {
      const d = JSON.parse(data);
      tlRenderBtn.disabled        = false;
      tlProgressDiv.style.display = 'none';
      gShowToast(`🎬 Sequence complete — ${d.count ?? total} frames`, false);
      // Auto-encode to MP4 and trigger download
      setTimeout(() => tlDoEncodeAndDownload('mp4'), 500);
    }
  } catch {}
}

// ── Encode buttons ─────────────────────────────────────────────
async function tlDoEncode(format) {
  const name = tlName.value.trim() || 'sequence';
  const fps  = parseInt(tlFps.value) || 24;
  const btnId = format === 'mp4' ? 'tl-encode-mp4' : 'tl-encode-gif';
  const btn = document.getElementById(btnId);
  const status = document.getElementById('tl-encode-status');
  const origText = btn.textContent;
  btn.textContent = '⏳ Encoding…';
  btn.disabled = true;
  if (status) status.innerHTML = '';
  try {
    const res = await fetch(`/api/sequences/${encodeURIComponent(name)}/encode`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ fps, format }),
    });
    const data = await res.json();
    if (data.ok) {
      btn.textContent = origText;
      if (status) status.innerHTML = `<a href="${data.path}" download="${name}.${format}">⬇ Download ${format.toUpperCase()}</a>`;
    } else {
      gShowToast('Encode failed: ' + (data.error || 'unknown error'), true);
      btn.textContent = origText;
    }
  } catch (e) {
    gShowToast('Encode error: ' + e.message, true);
    btn.textContent = origText;
  } finally {
    btn.disabled = false;
  }
}
document.getElementById('tl-encode-mp4')?.addEventListener('click', () => tlDoEncode('mp4'));
document.getElementById('tl-encode-gif')?.addEventListener('click', () => tlDoEncode('gif'));

// ── Auto-encode + download after render sequence ──────────────
async function tlDoEncodeAndDownload(format) {
  const name = tlName.value.trim() || 'sequence';
  const fps  = parseInt(tlFps.value) || 24;
  try {
    const res = await fetch(`/api/sequences/${encodeURIComponent(name)}/encode`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ fps, format }),
    });
    const data = await res.json();
    if (data.ok) {
      // Trigger browser save dialog
      const a = document.createElement('a');
      a.href = data.path;
      a.download = `${name}.${format}`;
      document.body.appendChild(a);
      a.click();
      document.body.removeChild(a);
      gShowToast(`⬇ Downloading ${name}.${format}`, false);
    } else {
      gShowToast('Encode failed: ' + (data.error || 'unknown error'), true);
    }
  } catch (e) {
    gShowToast('Encode error: ' + e.message, true);
  }
}

// ── Clear ──────────────────────────────────────────────────────
function gDoClear() {
  if (gLiveMode) gSetLive(false);
  gNodes=[]; gEdges=[]; gSelectedNode=null; gSelectedEdge=null; gSelectedNodes.clear();
  gPanX=0; gPanY=0; gApplyPan();
  gNodesEl.innerHTML=''; gEdgesEl.innerHTML=''; gOutputStrip.innerHTML='';
  gLivePreviewImg = null; gMainPreview.innerHTML=''; gMainPreview.classList.remove('active');
  gPreviewHide(); gShowNodeParams(null);
  if (isMobile()) gParamsSheetClose();
  tlClips = []; localStorage.removeItem('tl-clips'); renderTimelineRuler();
  gSetStatus(''); gSave();
}
gClearBtn.addEventListener('click', gDoClear);
gClearBtnDesk.addEventListener('click', gDoClear);
document.getElementById('graph-layout-btn')?.addEventListener('click', gPhysicsToggle);
document.getElementById('graph-layout-btn-desk')?.addEventListener('click', gPhysicsToggle);

// ── Persistence ────────────────────────────────────────────────
function gSave() {
  localStorage.setItem('pipeline-graph', JSON.stringify({nodes:gNodes,edges:gEdges,panX:gPanX,panY:gPanY,scale:gCanvasScale,canvasW:gCanvasW,canvasH:gCanvasH}));
}
function _gApplyCanvasSize(w, h) {
  gCanvasW = w || 768; gCanvasH = h || 512;
  const key = `${gCanvasW}x${gCanvasH}`;
  for (const id of ['graph-canvas-preset', 'graph-canvas-preset-mob']) {
    const sel = document.getElementById(id);
    if (sel && [...sel.options].some(o => o.value === key)) sel.value = key;
  }
}
function gLoad() {
  try {
    const s = JSON.parse(localStorage.getItem('pipeline-graph')||'null');
    if (!s) return;
    gNodes = s.nodes || [];
    // Migrate: old image input port was named "image"; now it's "image_in"
    gEdges = (s.edges || []).map(e =>
      e.dst_port === 'image' ? { ...e, dst_port: 'image_in' } : e
    );
    gRecomputeEdgeCounter();
    gPanX = s.panX || 0; gPanY = s.panY || 0;
    gCanvasScale = s.scale || 1.0;
    _gApplyCanvasSize(s.canvasW, s.canvasH);
  } catch { gNodes=[]; gEdges=[]; }
}
async function gRestoreGraph() {
  if (!gNodes.length && !gEdges.length) return;
  if (!Object.keys(gNodeDefs).length) {
    gNodeDefs = await gFetchNodeDefs();
    gRenderPalette(); gPaletteLoaded=true;
  }
  gApplyPan();
  gNodes.forEach(gRenderAnyNode);
  gUpdateConnectedPorts();
  gRedrawEdges();
}

// ── Deep link: /?load=<saved-name> loads a library graph on boot ──
// Chained after gRestoreGraph so the localStorage restore can't re-render
// on top.
async function gDeepLinkLoad() {
  const name = new URLSearchParams(location.search).get('load');
  if (!name) return;
  history.replaceState(null, '', location.pathname); // don't re-load on refresh
  try {
    const r = await fetch(`/api/graph/saved/${encodeURIComponent(name)}`);
    if (!r.ok) throw new Error(`HTTP ${r.status}`);
    const saved = await r.json();
    // save_graph stores {graph:{nodes,edges}, name, saved_at}; gLoadGraph
    // reads .nodes/.edges at the top level, so unwrap .graph if present.
    await gLoadGraph(saved.graph || saved);
    gShowToast(`Loaded "${name}"`);
  } catch (e) { gShowToast(`Load failed: ${name} (${e.message})`, true); }
}

gLoad();
setTimeout(async () => { await gRestoreGraph(); await gDeepLinkLoad(); }, 100);

// ── Canvas size preset ────────────────────────────────────────────
_gApplyCanvasSize(gCanvasW, gCanvasH); // sync selects to restored state
for (const id of ['graph-canvas-preset', 'graph-canvas-preset-mob']) {
  const sel = document.getElementById(id);
  if (!sel) continue;
  sel.addEventListener('change', () => {
    const [w, h] = sel.value.split('x').map(Number);
    _gApplyCanvasSize(w, h); // updates both selects
    gSave();
  });
}

// ── Serialize graph to plain object ───────────────────────────
function gSerializeGraph() {
  return {
    version: 1,
    canvasW: gCanvasW,
    canvasH: gCanvasH,
    nodes: gNodes.map(n => {
      if (n.type === 'group') return {
        id: n.id, type: 'group', name: n.name,
        subgraph: n.subgraph,
        exposed_inputs:  n.exposed_inputs  || [],
        exposed_outputs: n.exposed_outputs || [],
        x: Math.round(n.x), y: Math.round(n.y),
        render: !!n.render,
      };
      return {
        id:             n.id,
        method_id:      n.method_id,
        params:         n.params || {},
        animParams:     n.animParams || {},
        paramKeyframes: n.paramKeyframes || {},
        start_frame:    n.start_frame || 0,
        end_frame:      n.end_frame   || 0,
        keyframes:      n.keyframes   || [],
        x:              Math.round(n.x),
        y:              Math.round(n.y),
        render:         !!n.render,
      };
    }),
    edges: gEdges.map(e => ({
      src_node: e.src_node,
      src_port: e.src_port,
      dst_node: e.dst_node,
      dst_port: e.dst_port,
      feedback: !!e.feedback,
    })),
  };
}

// ── Load a serialized graph onto the canvas ────────────────────
async function gLoadGraph(data) {
  // Clear canvas (without triggering gSave loop)
  gNodes = []; gEdges = []; gSelectedNode = null; gSelectedEdge = null;
  gSelectedNodes.clear();
  gPanX = 0; gPanY = 0; gApplyPan();
  gNodesEl.innerHTML = ''; gEdgesEl.innerHTML = ''; gOutputStrip.innerHTML = '';
  gLivePreviewImg = null; gMainPreview.innerHTML = ''; gMainPreview.classList.remove('active');
  gPreviewHide(); gShowNodeParams(null);
  if (isMobile()) gParamsSheetClose();
  gSetStatus('');

  // Restore canvas size if present in the saved graph
  if (data.canvasW && data.canvasH) _gApplyCanvasSize(data.canvasW, data.canvasH);

  // Ensure node defs are ready
  if (!Object.keys(gNodeDefs).length) {
    gNodeDefs = await gFetchNodeDefs();
    gRenderPalette(); gPaletteLoaded = true;
  }

  // Rebuild nodes with their saved IDs
  for (const n of (data.nodes || [])) {
    if (n.type === 'group') {
      const node = { id: n.id, type: 'group', name: n.name || 'Group', subgraph: n.subgraph || {nodes:[],edges:[]}, exposed_inputs: n.exposed_inputs || [], exposed_outputs: n.exposed_outputs || [], x: n.x, y: n.y, render: !!n.render, dirty: true };
      gNodes.push(node);
      gRenderGroupNode(node);
    } else {
      const def = gNodeDefs[n.method_id];
      if (!def) continue;
      const node = { id: n.id, method_id: n.method_id, params: n.params || gDefaultParams(def), animParams: n.animParams || {}, paramKeyframes: n.paramKeyframes || {}, start_frame: n.start_frame || 0, end_frame: n.end_frame || 0, keyframes: n.keyframes || [], x: n.x, y: n.y, render: !!n.render, dirty: true };
      gNodes.push(node);
      gRenderNode(node);
    }
  }

  // Rebuild edges
  gEdgeCounter = 0;
  for (const e of (data.edges || [])) {
    gEdges.push({
      id: 'e' + (++gEdgeCounter),
      src_node: e.src_node, src_port: e.src_port,
      dst_node: e.dst_node, dst_port: e.dst_port,
      feedback: !!e.feedback,
    });
  }

  gUpdateConnectedPorts();
  gRedrawEdges();
  gSave();
}

// ── Save graph to server ───────────────────────────────────────
async function gGraphSaveToServer() {
  const name = prompt('Save graph as:', 'my-graph');
  if (name === null) return;
  const graph = gSerializeGraph();
  try {
    const r = await fetch('/api/graph/save', {
      method:  'POST',
      headers: { 'Content-Type': 'application/json' },
      body:    JSON.stringify({ name: name.trim() || 'untitled', graph }),
    });
    const d = await r.json();
    if (d.ok) gShowToast(`Saved "${d.name}"`);
    else gShowToast('Save failed', true);
  } catch(e) { gShowToast('Save failed: ' + e.message, true); }
}

// ── Load graph list from server ────────────────────────────────
async function gGraphLoadFromServer() {
  let graphs;
  try { graphs = await fetch('/api/graph/saved').then(r => r.json()); }
  catch(e) { gShowToast('Could not reach server', true); return; }

  const list = document.getElementById('graph-load-list');
  if (!graphs.length) {
    list.innerHTML = '<div class="glm-empty">No saved graphs yet.</div>';
  } else {
    list.innerHTML = graphs.map(g =>
      `<div class="glm-item" data-gname="${escHtml(g.name)}">` +
      `<div class="glm-name">${escHtml(g.name)}</div>` +
      `<div class="glm-meta">${escHtml((g.saved_at || '').replace('T',' ').slice(0,16))}</div>` +
      `</div>`
    ).join('');
    list.querySelectorAll('.glm-item').forEach(el => {
      el.addEventListener('click', async () => {
        document.getElementById('graph-load-modal').classList.remove('visible');
        try {
          const data = await fetch(`/api/graph/saved/${encodeURIComponent(el.dataset.gname)}`).then(r => r.json());
          await gLoadGraph(data);
          gShowToast(`Loaded "${el.dataset.gname}"`);
        } catch(e) { gShowToast('Load failed: ' + e.message, true); }
      });
    });
  }
  document.getElementById('graph-load-modal').classList.add('visible');
}

// ── Toast helper ───────────────────────────────────────────────
function gShowToast(msg, isErr = false) {
  const t = document.getElementById('graph-toast');
  t.textContent = msg;
  t.style.background   = isErr ? '#2e1a1a' : '#1a2e1a';
  t.style.border       = `1px solid ${isErr ? 'var(--err)' : 'var(--success)'}`;
  t.style.color        = isErr ? 'var(--err)' : 'var(--success)';
  t.style.display      = '';
  clearTimeout(t._timer);
  t._timer = setTimeout(() => { t.style.display = 'none'; }, 2500);
}

// ── Wire save/load buttons ─────────────────────────────────────
document.getElementById('graph-save-btn')?.addEventListener('click', gGraphSaveToServer);
document.getElementById('graph-save-btn-desk')?.addEventListener('click', gGraphSaveToServer);
document.getElementById('graph-load-btn')?.addEventListener('click', gGraphLoadFromServer);
document.getElementById('graph-load-btn-desk')?.addEventListener('click', gGraphLoadFromServer);
document.getElementById('graph-load-close')?.addEventListener('click', () => {
  document.getElementById('graph-load-modal').classList.remove('visible');
});

// ═══════════════════════════════════════════════════════════════
// NODE PICKER  (Houdini-style Tab / right-click menu)
// ═══════════════════════════════════════════════════════════════

// ── Create picker DOM (runs synchronously, body already in DOM) ─
(function() {
  const el = document.createElement('div');
  el.id = 'node-picker';
  el.innerHTML =
    '<div class="np-search-wrap"><input type="text" id="node-picker-input" placeholder="Search nodes…" autocomplete="off" spellcheck="false"></div>' +
    '<div id="node-picker-list"></div>';
  document.body.appendChild(el);
})();

// ── State ──────────────────────────────────────────────────────
let npVisible   = false;
let npItems     = [];    // flat {def, cat} rows (no category separators)
let npHighlight = 0;
let npSpawnPos  = null;  // {canvasX, canvasY}

// ── Usage memory ───────────────────────────────────────────────
function npGetUsage() {
  try { return JSON.parse(localStorage.getItem('np-usage') || '{}'); } catch { return {}; }
}
function npBumpUsage(id) {
  const m = npGetUsage();
  m[id] = (m[id] || 0) + 1;
  try { localStorage.setItem('np-usage', JSON.stringify(m)); } catch {}
}

// ── Relevance score ────────────────────────────────────────────
function npScore(def, q) {
  if (!q) return 0;
  const name = def.name.toLowerCase();
  const cat  = (def.category || '').toLowerCase();
  if (name === q)          return 100;
  if (name.startsWith(q)) return 80;
  if (name.includes(q))   return 60;
  if (cat.includes(q))    return 40;
  return -1;
}

// ── Build display rows ─────────────────────────────────────────
function npBuildRows(query) {
  const usage = npGetUsage();
  const q     = query.trim().toLowerCase();
  const all   = Object.values(gNodeDefs);

  if (!q) {
    // Grouped by category; within each group usage-desc then name-asc
    const groups = {};
    for (const def of all) {
      const cat = def.category || 'other';
      (groups[cat] = groups[cat] || []).push(def);
    }
    const rows = [];
    for (const [cat, defs] of Object.entries(groups).sort()) {
      defs.sort((a, b) =>
        ((usage[b.method_id]||0) - (usage[a.method_id]||0)) ||
        a.name.localeCompare(b.name)
      );
      rows.push({ isCat: true, cat });
      for (const def of defs) rows.push({ isCat: false, def, cat });
    }
    return rows;
  }

  // Filtered: score → usage → name
  const scored = [];
  for (const def of all) {
    const s = npScore(def, q);
    if (s < 0) continue;
    scored.push({ def, cat: def.category || 'other', score: s, use: usage[def.method_id] || 0 });
  }
  scored.sort((a, b) =>
    (b.score - a.score) || (b.use - a.use) || a.def.name.localeCompare(b.def.name)
  );
  return scored.map(s => ({ isCat: false, def: s.def, cat: s.cat }));
}

// ── Render list ────────────────────────────────────────────────
function npRender(query) {
  const listEl = document.getElementById('node-picker-list');
  const rows   = npBuildRows(query);
  npItems      = rows.filter(r => !r.isCat);
  npHighlight  = 0;

  if (!npItems.length) {
    listEl.innerHTML = '<p style="padding:12px;font-size:12px;color:var(--muted)">No matches.</p>';
    return;
  }

  let html = '';
  for (const row of rows) {
    if (row.isCat) {
      html += `<div class="np-cat">${escHtml(row.cat)}</div>`;
    } else {
      const def = row.def;
      const deprCls = def.deprecated ? ' np-deprecated' : '';
      const deprTag = def.deprecated ? ` <span class="np-depr-tag">(deprecated)</span>` : '';
      const rawDesc = def.description || '';
      const descText = rawDesc.length > 70 ? rawDesc.slice(0, 70) + '…' : rawDesc;
      const descHtml = descText ? `<span class="np-desc">${escHtml(descText)}</span>` : '';
      html += `<div class="np-item${deprCls}" data-mid="${escHtml(def.method_id)}">` +
              `<div class="np-item-row"><span class="np-mid">#${escHtml(def.method_id)}</span>` +
              `<span class="np-name">${escHtml(def.name)}${deprTag}</span></div>` +
              descHtml + `</div>`;
    }
  }
  listEl.innerHTML = html;

  listEl.querySelectorAll('.np-item').forEach((el, i) => {
    el.addEventListener('mouseenter', () => { npHighlight = i; npMarkHighlight(); });
    el.addEventListener('click', () => npCommit(i));
  });
  npMarkHighlight();
}

function npMarkHighlight() {
  const els = document.getElementById('node-picker-list').querySelectorAll('.np-item');
  els.forEach((el, i) => el.classList.toggle('np-hl', i === npHighlight));
  if (els[npHighlight]) els[npHighlight].scrollIntoView({ block: 'nearest' });
}

// ── Open ───────────────────────────────────────────────────────
function npOpen(screenX, screenY, canvasX, canvasY) {
  if (!Object.keys(gNodeDefs).length) return; // palette not ready yet

  npSpawnPos = { canvasX, canvasY };
  npVisible  = true;

  const picker = document.getElementById('node-picker');
  const input  = document.getElementById('node-picker-input');
  input.value  = '';
  picker.style.display = 'flex';
  npRender('');

  // Clamp to viewport after one layout tick
  requestAnimationFrame(() => {
    const pw = picker.offsetWidth,  ph = picker.offsetHeight;
    const vw = window.innerWidth,   vh = window.innerHeight;
    let l = screenX, t = screenY;
    if (l + pw > vw - 8) l = vw - pw - 8;
    if (t + ph > vh - 8) t = vh - ph - 8;
    if (l < 8) l = 8;
    if (t < 8) t = 8;
    picker.style.left = l + 'px';
    picker.style.top  = t + 'px';
    input.focus();
  });
}

// ── Close ──────────────────────────────────────────────────────
function npClose() {
  npVisible = false;
  document.getElementById('node-picker').style.display = 'none';
  npSpawnPos = null;
}

// ── Commit (spawn node) ────────────────────────────────────────
function npCommit(idx) {
  if (idx < 0 || idx >= npItems.length || !npSpawnPos) return;
  const { def } = npItems[idx];
  npBumpUsage(def.method_id);
  gAddNode(def.method_id, npSpawnPos.canvasX, npSpawnPos.canvasY);
  npClose();
}

// ── Screen → canvas coords ─────────────────────────────────────
function npScreenToCanvas(sx, sy) {
  const rect = gCanvasWrap.getBoundingClientRect();
  return {
    canvasX: (sx - rect.left - gPanX) / gCanvasScale - 80,
    canvasY: (sy - rect.top  - gPanY) / gCanvasScale - 40,
  };
}

// ── Input: type to filter ──────────────────────────────────────
document.getElementById('node-picker-input').addEventListener('input', e => {
  npRender(e.target.value);
});

// ── Input: keyboard navigation ────────────────────────────────
document.getElementById('node-picker-input').addEventListener('keydown', e => {
  if (e.key === 'Escape')    { e.preventDefault(); npClose(); return; }
  if (e.key === 'ArrowDown') {
    e.preventDefault();
    npHighlight = Math.min(npHighlight + 1, npItems.length - 1);
    npMarkHighlight(); return;
  }
  if (e.key === 'ArrowUp') {
    e.preventDefault();
    npHighlight = Math.max(npHighlight - 1, 0);
    npMarkHighlight(); return;
  }
  if (e.key === 'Enter') { e.preventDefault(); npCommit(npHighlight); return; }
});

// ── Right-click on empty canvas → open picker ─────────────────
gCanvasWrap.addEventListener('contextmenu', e => {
  // Let node context menu handle right-clicks on nodes/ports
  if (e.target.closest('.gnode') || e.target.classList.contains('gport')) return;
  e.preventDefault();
  const pos = npScreenToCanvas(e.clientX, e.clientY);
  npOpen(e.clientX, e.clientY, pos.canvasX, pos.canvasY);
});

// ── Tab key → open picker (global, graph view only) ───────────
document.addEventListener('keydown', e => {
  if (e.key !== 'Tab') return;
  // If picker is open, just eat the Tab so focus doesn't wander
  if (npVisible) { e.preventDefault(); return; }
  // Don't steal Tab inside text inputs / selects (e.g. param panel)
  const tag = e.target.tagName;
  if (tag === 'INPUT' || tag === 'SELECT' || tag === 'TEXTAREA') return;
  e.preventDefault();
  const rect = gCanvasWrap.getBoundingClientRect();
  const cx = rect.left + rect.width  / 2;
  const cy = rect.top  + rect.height / 2;
  const pos = npScreenToCanvas(cx, cy);
  npOpen(cx, cy, pos.canvasX, pos.canvasY);
});

// ── Click outside → close (capture so it runs first) ─────────
document.addEventListener('click', e => {
  if (!npVisible) return;
  if (!document.getElementById('node-picker').contains(e.target)) npClose();
}, true);

// ── Mobile + button ────────────────────────────────────────────
document.getElementById('np-add-btn').addEventListener('click', () => {
  const rect = gCanvasWrap.getBoundingClientRect();
  const cx = rect.left + rect.width  / 2;
  const cy = rect.top  + rect.height / 2;
  const pos = npScreenToCanvas(cx, cy);
  npOpen(cx, cy, pos.canvasX, pos.canvasY);
});

// ── Hot-reload: listen for server-push node-def changes ───────
function gConnectEvents() {
    const es = new EventSource('/api/events');
    es.addEventListener('node-defs-updated', async () => {
        console.log('[hot-reload] node defs updated, refreshing...');
        await gLoadMethodPalette({ force: true });
        if (typeof gLoadPortTypes === 'function') await gLoadPortTypes();
        gShowToast('Methods updated');
        // Notify Node Doctor apply flow that hot-reload completed
        window.dispatchEvent(new CustomEvent('nd-hot-reload'));
    });
    es.onerror = () => {
        es.close();
        setTimeout(gConnectEvents, 3000);
    };
}
gConnectEvents();

// ── Per-parameter flyout menu (dropdown) ───────────────────────────
// Right-click (or click the ⋮ button) a parameter row to open a small
// menu. "Describe an issue…" routes straight into the Node Doctor chat,
// which keeps conversation history — so it can iterate as many times as
// you like and actually applies the rewrite. No complaint accumulation.
(function() {
  const menu = document.getElementById('param-flyout-menu');

  let ctxParam = null;   // param key last targeted
  let ctxRow   = null;   // param-row element last targeted

  function hideMenu() { menu.style.display = 'none'; ctxParam = null; ctxRow = null; }

  function showMenu(row, x, y) {
    ctxRow   = row;
    ctxParam = row.dataset.param;
    const titleEl = document.getElementById('ndpm-title');
    if (titleEl) titleEl.textContent = ctxParam ? `“${ctxParam}”` : 'Parameter';
    menu.style.display = 'block';
    const mw = menu.offsetWidth, mh = menu.offsetHeight;
    menu.style.left = Math.min(x, window.innerWidth  - mw - 4) + 'px';
    menu.style.top  = Math.min(y, window.innerHeight - mh - 4) + 'px';
  }

  // Open the menu on right-click of any parameter row.
  gParamsForm.addEventListener('contextmenu', e => {
    const row = e.target.closest('.param-row');
    if (!row || !row.dataset.param) return;   // let non-param right-clicks bubble
    e.preventDefault();
    e.stopPropagation();
    showMenu(row, e.clientX, e.clientY);
  });

  // Keep the graph-level context menu from also firing inside the panel.
  gParamsForm.addEventListener('contextmenu', e => e.stopPropagation(), true);

  // ⋮ button injected into each param row (built in gShowNodeParams).
  gParamsForm.addEventListener('click', e => {
    const btn = e.target.closest('.ndpm-btn');
    if (!btn) return;
    e.preventDefault();
    e.stopPropagation();
    const row = btn.closest('.param-row');
    if (row) {
      const r = btn.getBoundingClientRect();
      showMenu(row, r.left, r.bottom + 4);
    }
  });

  // Close on outside click / Escape.
  document.addEventListener('click', e => {
    if (menu.style.display === 'block' && !menu.contains(e.target) && !e.target.closest('.ndpm-btn'))
      hideMenu();
  });
  document.addEventListener('keydown', e => { if (e.key === 'Escape') hideMenu(); });

  function currentNode() { return gNodes.find(n => n.id === gSelectedNode); }

  document.getElementById('ndpm-issue').addEventListener('click', e => {
    e.stopPropagation();
    const node = currentNode();
    if (!node || !ctxParam) { hideMenu(); return; }
    const def  = gNodeDefs[node.method_id];
    const spec = (def && def.params && def.params[ctxParam]) || {};
    const prompt =
      `Issue with parameter "${ctxParam}" on node "${def ? def.name : node.method_id}".\n` +
      `Current value: ${JSON.stringify(node.params ? node.params[ctxParam] : undefined)}\n` +
      `Spec: ${JSON.stringify(spec)}\n\n` +
      `Describe what you want changed and rewrite the node to fix it. ` +
      `Keep the rest of the node intact and honor the node/dataflow hygiene contract. ` +
      `Output the full updated file.`;
    ndPrefill(prompt);
    hideMenu();
  });

  document.getElementById('ndpm-reset').addEventListener('click', e => {
    e.stopPropagation();
    const node = currentNode();
    if (!node || !ctxParam) { hideMenu(); return; }
    const def  = gNodeDefs[node.method_id];
    const spec = (def && def.params && def.params[ctxParam]) || {};
    if (!('default' in spec)) { gShowToast('No default for this parameter', true); hideMenu(); return; }
    const el = document.getElementById('p_' + ctxParam) || document.getElementById('p_' + ctxParam + '_swatch');
    if (el) {
      if (el.type === 'checkbox') el.checked = !!spec.default;
      else el.value = spec.default;
      el.dispatchEvent(new Event('change', { bubbles: true }));
    } else {
      gUpdateNodeParam(node.id, ctxParam, spec.default);
    }
    gShowToast(`Reset "${ctxParam}" to default`);
    hideMenu();
  });

  document.getElementById('ndpm-copy').addEventListener('click', e => {
    e.stopPropagation();
    if (ctxParam && navigator.clipboard) navigator.clipboard.writeText(ctxParam).then(
      () => gShowToast('Copied parameter name'),
      () => gShowToast('Copy failed', true)
    );
    hideMenu();
  });

  document.getElementById('ndpm-open').addEventListener('click', e => {
    e.stopPropagation();
    ndOpenPanel();
    if (ctxParam) ndPrefill(`Parameter "${ctxParam}" — what would you like to change?`);
    hideMenu();
  });
})();

// ── Load palette now (graph is the default view) ──────────────
gLoadMethodPalette();
gLoadPortTypes();
gLoadGroupPresets();

// ── NODE DOCTOR ────────────────────────────────────────────────
(function() {
  let ndMessages     = [];
  let ndPendingCode  = null;
  let ndBackupId     = null;
  let ndBusy         = false;

  const ndPanel      = document.getElementById('nd-panel');
  const ndToggleBtn  = document.getElementById('nd-toggle-btn');
  const ndCloseBtn   = document.getElementById('nd-close-btn');
  const ndMsgsEl     = document.getElementById('nd-messages');
  const ndInput      = document.getElementById('nd-input');
  const ndSendBtn    = document.getElementById('nd-send-btn');
  const ndApplyRow   = document.getElementById('nd-apply-row');
  const ndApplyBtn   = document.getElementById('nd-apply-btn');
  const ndUndoBtn    = document.getElementById('nd-undo-btn');

  // ── Pre-fill the chat input (used by the per-parameter flyout) ──
  // Appends to whatever is already typed, opens the panel, and focuses.
  function ndPrefill(text) {
    if (!text) return;
    if (!ndIsOpen()) ndOpenPanel();
    const cur = ndInput.value.trim();
    ndInput.value = cur ? cur + '\n\n' + text : text;
    ndInput.focus();
    // Move caret to the end.
    ndInput.selectionStart = ndInput.selectionEnd = ndInput.value.length;
  }
  window.ndPrefill = ndPrefill;   // exposed for the flyout menu handler

  function ndIsOpen() { return ndPanel.classList.contains('nd-open'); }

  function ndOpenPanel() {
    ndPanel.classList.add('nd-open');
    ndToggleBtn.classList.add('nd-active');
    ndInput.focus();
  }
  function ndClosePanel() {
    ndPanel.classList.remove('nd-open');
    ndToggleBtn.classList.remove('nd-active');
  }

  ndToggleBtn.addEventListener('click', () => ndIsOpen() ? ndClosePanel() : ndOpenPanel());
  ndCloseBtn.addEventListener('click', ndClosePanel);

  function ndAppend(role, text) {
    const el = document.createElement('div');
    el.className = `nd-msg nd-${role}`;
    el.textContent = text;
    ndMsgsEl.appendChild(el);
    ndMsgsEl.scrollTop = ndMsgsEl.scrollHeight;
    return el;
  }

  function ndExtractCode(text) {
    const m = text.match(/```python\s*([\s\S]*?)```/);
    return m ? m[1].trim() : null;
  }

  async function ndSend() {
    if (ndBusy) return;
    const text = ndInput.value.trim();
    if (!text) return;

    const node = gNodes.find(n => n.id === gSelectedNode);
    if (!node) return;
    const def = gNodeDefs[node.method_id];

    ndInput.value = '';
    ndBusy = true;
    ndSendBtn.disabled = true;

    ndMessages.push({ role: 'user', content: text });
    ndAppend('user', text);

    const thinkEl = ndAppend('bot', '');
    thinkEl.classList.add('nd-thinking');
    // Show spinner while waiting for first response chunk
    const spinner = document.createElement('span');
    spinner.className = 'nd-spinner';
    thinkEl.prepend(spinner);

    let botText = '';
    try {
      const resp = await fetch('/api/node-doctor/chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          method_id:   node.method_id,
          node_def:    def || {},
          node_params: node.params || {},
          messages:    ndMessages.slice(),
        }),
      });

      if (!resp.ok) throw new Error(`Server error ${resp.status}`);

      const reader = resp.body.getReader();
      const dec    = new TextDecoder();
      thinkEl.textContent = '';
      thinkEl.classList.remove('nd-thinking');
      // Add blinking cursor during streaming
      thinkEl.classList.add('nd-stream-cursor');

      let buf = '';
      while (true) {
        const { value, done } = await reader.read();
        if (done) break;
        buf += dec.decode(value, { stream: true });
        const lines = buf.split('\n');
        buf = lines.pop();
        for (const line of lines) {
          if (!line.startsWith('data: ')) continue;
          try {
            const d = JSON.parse(line.slice(6));
            if (d.done) break;
            if (d.text) {
              botText += d.text;
              thinkEl.textContent = botText;
              ndMsgsEl.scrollTop = ndMsgsEl.scrollHeight;
            }
          } catch { /* partial JSON */ }
        }
      }
      // Remove streaming cursor when done
      thinkEl.classList.remove('nd-stream-cursor');
    } catch (err) {
      thinkEl.classList.remove('nd-thinking');
      thinkEl.classList.remove('nd-stream-cursor');
      thinkEl.textContent = '⚠ ' + err.message;
    }

    ndMessages.push({ role: 'assistant', content: botText });

    const code = ndExtractCode(botText);
    if (code) {
      ndPendingCode = code;
      ndApplyRow.classList.add('nd-show');
      ndApplyBtn.disabled = false;
      ndUndoBtn.classList.remove('nd-show');
    }

    ndBusy = false;
    ndSendBtn.disabled = false;
  }

  ndSendBtn.addEventListener('click', ndSend);
  ndInput.addEventListener('keydown', e => {
    if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); ndSend(); }
  });

  ndApplyBtn.addEventListener('click', async () => {
    if (!ndPendingCode || ndBusy) return;
    const node = gNodes.find(n => n.id === gSelectedNode);
    if (!node) return;
    ndApplyBtn.disabled = true;
    // Show pending message with spinner
    const statusEl = ndAppend('bot', '');
    const spinner = document.createElement('span');
    spinner.className = 'nd-spinner';
    statusEl.prepend(spinner);
    statusEl.appendChild(document.createTextNode(' Applying & hot-reloading…'));
    try {
      const r = await fetch('/api/node-doctor/apply', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ method_id: node.method_id, source: ndPendingCode }),
      });
      const data = await r.json();
      if (data.ok) {
        ndBackupId = data.backup_id;
        ndUndoBtn.classList.add('nd-show');
        ndUndoBtn.disabled = false;
        // Wait for SSE confirmation or timeout
        const confirmed = await new Promise(resolve => {
          const timeout = setTimeout(() => resolve(false), 5000);
          const handler = () => { clearTimeout(timeout); resolve(true); };
          window.addEventListener('nd-hot-reload', handler, { once: true });
        });
        if (confirmed) {
          statusEl.textContent = '';
          statusEl.classList.remove('nd-thinking');
          statusEl.classList.add('nd-msg', 'nd-bot');
          statusEl.textContent = '✓ Hot-reload complete — method updated';
          // Flash the node green briefly
          const nodeEl = document.querySelector(`.gnode[data-id="${node.id}"]`);
          if (nodeEl) {
            nodeEl.style.transition = 'border-color 0.15s, box-shadow 0.15s';
            nodeEl.style.borderColor = '#34d399';
            nodeEl.style.boxShadow = '0 0 12px rgba(52,211,153,0.5)';
            setTimeout(() => {
              nodeEl.style.borderColor = '';
              nodeEl.style.boxShadow = '';
            }, 1200);
          }
        } else {
          statusEl.textContent = '';
          statusEl.classList.remove('nd-thinking');
          statusEl.classList.add('nd-msg', 'nd-bot');
          statusEl.textContent = '⚠ Applied, but hot-reload confirmation not received (server may need restart)';
          ndApplyBtn.disabled = false;
        }
      } else {
        statusEl.textContent = '';
        statusEl.classList.remove('nd-thinking');
        statusEl.classList.add('nd-msg', 'nd-bot');
        statusEl.textContent = '⚠ Apply failed: ' + (data.error || 'unknown');
        ndApplyBtn.disabled = false;
      }
    } catch (err) {
      statusEl.textContent = '';
      statusEl.classList.remove('nd-thinking');
      statusEl.classList.add('nd-msg', 'nd-bot');
      statusEl.textContent = '⚠ ' + err.message;
      ndApplyBtn.disabled = false;
    }
  });

  ndUndoBtn.addEventListener('click', async () => {
    if (!ndBackupId) return;
    ndUndoBtn.disabled = true;
    try {
      const r = await fetch(`/api/node-doctor/undo/${ndBackupId}`, { method: 'POST' });
      const data = await r.json();
      if (data.ok) {
        ndBackupId = null;
        ndUndoBtn.classList.remove('nd-show');
        ndApplyBtn.disabled = false;
        ndAppend('bot', '↩ Reverted to previous version.');
      } else {
        ndAppend('bot', '⚠ Undo failed: ' + (data.error || 'unknown'));
        ndUndoBtn.disabled = false;
      }
    } catch (err) {
      ndAppend('bot', '⚠ ' + err.message);
      ndUndoBtn.disabled = false;
    }
  });

  // Wire into node selection — show/hide the toggle button and reset state
  const _origGSelectNode = gSelectNode;
  gSelectNode = function(id) {
    _origGSelectNode(id);
    const node = gNodes.find(n => n.id === id);
    if (node) {
      ndToggleBtn.style.display = '';
    } else {
      ndToggleBtn.style.display = 'none';
      ndClosePanel();
    }
    // Reset conversation when switching nodes
    ndMessages    = [];
    ndPendingCode = null;
    ndMsgsEl.innerHTML = '';
    ndApplyRow.classList.remove('nd-show');
    ndUndoBtn.classList.remove('nd-show');
  };
})();

// ── Load saved clips on init ──────────────────────────────────
tlLoadClips();
renderTimelineRuler();
