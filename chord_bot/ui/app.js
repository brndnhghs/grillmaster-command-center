// Chord Bot — main application: state management, node ops, playback, init
'use strict';

import { S } from './state.js';
import { FUNC_COLOR, PX_PER_BEAT, NODE_W_H, NODE_H_H, TIMELINE_Y, HISTORY_LIMIT } from './config.js';
import { apiExecute, apiExportMidi, apiNodeDefs } from './api.js';
import { scheduleSequence, stopAudio } from './audio.js';
import { renderPreview, resizePreviewCanvases, initPreviewEvents } from './preview.js';
import { initRail, renderGraph, initRailEvents } from './rail.js';
import { initDrawer, openDrawer, renderParamDrawer } from './drawer.js';

// ── Utilities ──────────────────────────────────────────────────────────────────
export function genId()        { return 'n' + (S.nodeCounter++); }
export function isHoriz(type)  { return (S.nodeDefs[type]?.axis || 'horizontal') === 'horizontal'; }
export function getNodeMeta(t) { return S.nodeDefs[t] || { name: t, params: {}, axis: 'horizontal' }; }

export function defaultParams(type) {
  const p = {};
  for (const [k, s] of Object.entries(getNodeMeta(type).params || {}))
    p[k] = s.default !== undefined ? s.default : '';
  return p;
}

export function blockWidth(node) {
  if (node.type === 'subgraph') {
    const subH = (node.subNodes || []).filter(n => isHoriz(n.type));
    const total = subH.reduce((s, n) => s + (n.params?.duration || 4), 0);
    return Math.max(88, total * PX_PER_BEAT);
  }
  return Math.max(88, (node.params?.duration ?? 4) * PX_PER_BEAT);
}

export function getAugSummary(v) {
  const p = v.params;
  for (const k of ['amount','strength','type','style','mode','target']) {
    if (k in p) {
      const val = p[k];
      return typeof val === 'number' ? val.toFixed(2).replace(/\.?0+$/, '') : String(val).slice(0, 10);
    }
  }
  const first = Object.values(p)[0];
  if (first === undefined) return '';
  return typeof first === 'number' ? first.toFixed(2).replace(/\.?0+$/, '') : String(first).slice(0, 8);
}

function totalBeatsFromSeq(seq) { return seq.length > 0 ? seq[seq.length - 1].end_beat : 0; }

// ── Chain helpers ──────────────────────────────────────────────────────────────
export function getHChain() {
  const hNodes = S.nodes.filter(n => isHoriz(n.type));
  const hEdges = S.edges.filter(e => {
    const src = S.nodes.find(n => n.id === e.src_node);
    const dst = S.nodes.find(n => n.id === e.dst_node);
    return src && dst && isHoriz(src.type) && isHoriz(dst.type);
  });
  const hasIncoming = new Set(hEdges.map(e => e.dst_node));
  const chain = [], visited = new Set();
  for (const root of hNodes.filter(n => !hasIncoming.has(n.id))) {
    let cur = root;
    while (cur && !visited.has(cur.id)) {
      chain.push(cur); visited.add(cur.id);
      const ne = hEdges.find(e => e.src_node === cur.id);
      cur = ne ? S.nodes.find(n => n.id === ne.dst_node) : null;
    }
  }
  for (const n of hNodes) if (!visited.has(n.id)) chain.push(n);
  return chain;
}

export function getVChildren(hNodeId) {
  return S.edges.filter(e => e.src_node === hNodeId)
    .map(e => S.nodes.find(n => n.id === e.dst_node))
    .filter(n => n && !isHoriz(n.type));
}

export function setChainOrder(newHChain) {
  S.edges = S.edges.filter(e => {
    const s = S.nodes.find(n => n.id === e.src_node);
    const d = S.nodes.find(n => n.id === e.dst_node);
    return !(s && d && isHoriz(s.type) && isHoriz(d.type));
  });
  for (let i = 0; i < newHChain.length - 1; i++)
    S.edges.push({ src_node: newHChain[i].id, src_port: 'harmonic_out', dst_node: newHChain[i + 1].id, dst_port: 'harmonic_in' });
  newHChain.forEach((n, i) => { n.x = 60 + i * NODE_W_H; n.y = TIMELINE_Y; });
}

function snapAll() {
  const chain = getHChain();
  chain.forEach((n, i) => {
    n.x = 60 + i * NODE_W_H; n.y = TIMELINE_Y;
    S.edges.filter(e => e.src_node === n.id).forEach(e => {
      const v = S.nodes.find(x => x.id === e.dst_node);
      if (v && !isHoriz(v.type)) { v.x = n.x; v.y = n.y + NODE_H_H; }
    });
  });
}

function expandSubgraphs(nodeList, edgeList) {
  const result = []; let res = [...edgeList];
  for (const node of nodeList) {
    if (node.type !== 'subgraph') { result.push(node); continue; }
    const pfx  = node.id + '__';
    const subH = (node.subNodes || []).filter(n => isHoriz(n.type)).sort((a, b) => a.x - b.x);
    for (const sn of (node.subNodes || [])) result.push({ ...sn, id: pfx + sn.id });
    for (const se of (node.subEdges || [])) res.push({ ...se, src_node: pfx + se.src_node, dst_node: pfx + se.dst_node });
    const lId = subH.length ? pfx + subH[0].id : null;
    const rId = subH.length ? pfx + subH[subH.length - 1].id : null;
    res = res.map(e => {
      if (e.dst_node === node.id && lId) return { ...e, dst_node: lId };
      if (e.src_node === node.id && rId) return { ...e, src_node: rId };
      return e;
    }).filter(e => e.src_node !== node.id && e.dst_node !== node.id);
  }
  return { nodes: result, edges: res };
}

// ── History ────────────────────────────────────────────────────────────────────
function _snapshot() {
  return { nodes: JSON.parse(JSON.stringify(S.nodes)), edges: JSON.parse(JSON.stringify(S.edges)) };
}

export function pushHistory() {
  S._undoStack.push(_snapshot());
  if (S._undoStack.length > HISTORY_LIMIT) S._undoStack.shift();
  S._redoStack = [];
}

function _restoreSnapshot(snap) {
  S.nodes = snap.nodes; S.edges = snap.edges;
  S.nodeCounter = S.nodes.reduce((m, n) => Math.max(m, parseInt(n.id.replace(/\D/g, '')) + 1), 1);
  S.selectedNode = null; S.drawerAugIdx = -1;
  openDrawer(false); saveToLocal(); renderGraph(); renderPreview();
}

function undo() {
  if (!S._undoStack.length) return;
  S._redoStack.push(_snapshot());
  _restoreSnapshot(S._undoStack.pop());
}

function redo() {
  if (!S._redoStack.length) return;
  S._undoStack.push(_snapshot());
  _restoreSnapshot(S._redoStack.pop());
}

// ── Node management ────────────────────────────────────────────────────────────
export function addNodeAtPosition(type, afterId) {
  pushHistory();
  const node = { id: genId(), type, x: 60, y: TIMELINE_Y, params: defaultParams(type), paramKeyframes: {} };
  S.nodes.push(node);
  const chain = getHChain().filter(n => n.id !== node.id);
  if (afterId === '') setChainOrder([node, ...chain]);
  else {
    const idx = chain.findIndex(n => n.id === afterId);
    if (idx === -1) setChainOrder([...chain, node]);
    else { chain.splice(idx + 1, 0, node); setChainOrder(chain); }
  }
  S.selectedNode = node; S.drawerAugIdx = -1;
  renderParamDrawer(); openDrawer(true);
  saveToLocal(); renderGraph();
}

export function addAugmenter(type, parentHId) {
  if (!parentHId) return;
  pushHistory();
  const node = { id: genId(), type, x: 60, y: TIMELINE_Y, params: defaultParams(type), paramKeyframes: {} };
  S.nodes.push(node);
  let wire = parentHId, cur = parentHId;
  while (true) {
    const d = S.edges.find(e => e.src_node === cur && S.nodes.find(n => n.id === e.dst_node) && !isHoriz(S.nodes.find(n => n.id === e.dst_node)?.type));
    if (!d) break; wire = d.dst_node; cur = wire;
  }
  S.edges.push({ src_node: wire, src_port: 'harmonic_out', dst_node: node.id, dst_port: 'harmonic_in' });
  S.selectedNode = S.nodes.find(n => n.id === parentHId) || S.selectedNode;
  snapAll(); saveToLocal(); renderGraph();
  S.drawerAugIdx = getVChildren(parentHId).length - 1;
  renderParamDrawer(); openDrawer(true);
}

function addNode(type) {
  if (isHoriz(type)) {
    const chain = getHChain();
    addNodeAtPosition(type, chain.length > 0 ? chain[chain.length - 1].id : '');
  } else {
    const parentH = (S.selectedNode && isHoriz(S.selectedNode.type))
      ? S.selectedNode
      : S.nodes.filter(n => isHoriz(n.type)).slice(-1)[0];
    addAugmenter(type, parentH?.id);
  }
}

export function removeNode(id) {
  pushHistory();
  const isH = isHoriz(S.nodes.find(n => n.id === id)?.type);
  if (isH) {
    const chain = getHChain().filter(n => n.id !== id);
    S.nodes = S.nodes.filter(n => n.id !== id);
    S.edges = S.edges.filter(e => e.src_node !== id && e.dst_node !== id);
    setChainOrder(chain);
    if (S.selectedNode?.id === id) { S.selectedNode = null; S.drawerAugIdx = -1; openDrawer(false); }
  } else {
    const parentEdge = S.edges.find(e => e.dst_node === id);
    const parentH    = parentEdge ? S.nodes.find(n => n.id === parentEdge.src_node && isHoriz(n.type)) : null;
    S.nodes = S.nodes.filter(n => n.id !== id);
    S.edges = S.edges.filter(e => e.src_node !== id && e.dst_node !== id);
    if (S.selectedNode?.id === id) {
      S.selectedNode = parentH || null; S.drawerAugIdx = -1;
      if (S.selectedNode) { renderParamDrawer(); openDrawer(true); }
      else openDrawer(false);
    } else {
      S.drawerAugIdx = -1;
      if (S.selectedNode) renderParamDrawer();
    }
  }
  saveToLocal(); renderGraph();
}

// ── Node type picker ───────────────────────────────────────────────────────────
function closePickers() { document.querySelectorAll('.node-picker').forEach(p => p.remove()); }

export function showNodePicker(afterId, anchor, axis) {
  closePickers();
  const types = Object.entries(S.nodeDefs).filter(([, d]) => (d.axis || 'horizontal') === (axis === 'h' ? 'horizontal' : 'vertical'));
  const picker = document.createElement('div');
  picker.className = 'node-picker';
  picker.innerHTML = types.map(([t, d]) =>
    `<div class="picker-item" data-type="${t}"><span class="picker-dot" style="background:${axis === 'h' ? '#e67e22' : '#9b59b6'}"></span>${d.name || t}</div>`
  ).join('') || `<div class="picker-item" style="color:var(--dim)">No ${axis === 'h' ? 'chord' : 'augmenter'} types loaded</div>`;
  document.body.appendChild(picker);

  const r = anchor.getBoundingClientRect(), pw = 160;
  let left = r.left, top = r.bottom + 4;
  if (left + pw > window.innerWidth - 8) left = window.innerWidth - pw - 8;
  if (top + 240 > window.innerHeight) top = r.top - 240;
  picker.style.left = Math.max(4, left) + 'px';
  picker.style.top  = Math.max(4, top)  + 'px';

  picker.addEventListener('click', e => {
    const item = e.target.closest('[data-type]');
    if (item) {
      if (axis === 'h') addNodeAtPosition(item.dataset.type, afterId);
      else addAugmenter(item.dataset.type, afterId);
      closePickers();
    }
  });
  setTimeout(() => document.addEventListener('click', closePickers, { once: true }), 50);
}

// ── Execution & export ─────────────────────────────────────────────────────────
async function executeGraph() {
  if (!S.nodes.length) { setStatus('No nodes'); return; }
  setStatus('Executing…');
  document.getElementById('btnExec').disabled = true;
  try {
    const ex = expandSubgraphs(S.nodes, S.edges);
    const bpm = parseInt(document.getElementById('bpmIn').value) || 120;
    S.sequence   = await apiExecute(ex.nodes, ex.edges, bpm);
    S.totalBeats = totalBeatsFromSeq(S.sequence);
    S.playhead   = 0;
    setStatus(`${S.sequence.length} chords · ${S.totalBeats.toFixed(1)} beats`);
    renderGraph(); renderPreview();
  } catch (err) {
    setStatus('Error: ' + err.message);
  } finally {
    document.getElementById('btnExec').disabled = false;
  }
}

async function exportMidi() {
  if (!S.nodes.length) return;
  const bpm = parseInt(document.getElementById('bpmIn').value) || 120;
  try {
    const blob = await apiExportMidi(S.nodes, S.edges, bpm);
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a'); a.href = url; a.download = 'chord_bot.mid'; a.click();
    URL.revokeObjectURL(url);
  } catch (err) { setStatus('Export error: ' + err.message); }
}

// ── Project JSON ───────────────────────────────────────────────────────────────
function saveProjectJson() {
  const bpm = parseInt(document.getElementById('bpmIn').value) || 120;
  const project = { version: 1, bpm, nodes: S.nodes, edges: S.edges.map(e => ({ ...e })) };
  const blob = new Blob([JSON.stringify(project, null, 2)], { type: 'application/json' });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a'); a.href = url; a.download = 'chord_bot_project.json'; a.click();
  URL.revokeObjectURL(url);
}

function loadProjectJson(file) {
  const reader = new FileReader();
  reader.onload = e => {
    try {
      const project = JSON.parse(e.target.result);
      if (!Array.isArray(project.nodes)) throw new Error('Missing nodes array');
      S.nodes = project.nodes; S.edges = project.edges || [];
      if (project.bpm) document.getElementById('bpmIn').value = project.bpm;
      S.selectedNode = null; S.drawerAugIdx = -1;
      openDrawer(false); saveToLocal(); renderGraph(); renderPreview();
      setStatus(`Loaded ${S.nodes.length} node${S.nodes.length !== 1 ? 's' : ''}`);
    } catch (err) { setStatus('Load error: ' + err.message); }
  };
  reader.readAsText(file);
}

// ── Playback ───────────────────────────────────────────────────────────────────
function togglePlay() { if (S.isPlaying) stopPlay(); else startPlay(); }

function startPlay() {
  if (!S.sequence.length) { executeGraph().then(() => { if (S.sequence.length) startPlay(); }); return; }
  S.isPlaying = true; S.playStartTime = performance.now();
  S.playStartBeat = S.playhead >= S.totalBeats ? 0 : S.playhead;
  document.getElementById('btnPlay').textContent = '■ Stop';
  document.getElementById('btnPlay').classList.add('active');
  const bpm = parseInt(document.getElementById('bpmIn').value) || 120;
  scheduleSequence(S.sequence, bpm, S.playStartBeat);
  rafTick();
}

function stopPlay() {
  S.isPlaying = false;
  if (S.lastRaf) cancelAnimationFrame(S.lastRaf);
  document.getElementById('btnPlay').textContent = '▶ Play';
  document.getElementById('btnPlay').classList.remove('active');
  stopAudio(); renderGraph(); renderPreview();
}

function rafTick() {
  if (!S.isPlaying) return;
  const bpm = parseInt(document.getElementById('bpmIn').value) || 120;
  S.playhead = S.playStartBeat + (performance.now() - S.playStartTime) / 1000 * (bpm / 60);
  if (S.playhead >= S.totalBeats) { S.playhead = S.totalBeats; stopPlay(); return; }
  const cur = S.sequence.find(e => e.start_beat <= S.playhead && S.playhead < e.end_beat);
  if (cur) {
    const fn = cur.state.function, col = FUNC_COLOR[fn] || '#9b59b6';
    const badge = document.getElementById('fnBadge');
    badge.textContent  = fn === 'tonic' ? 'T' : fn === 'subdominant' ? 'S' : fn === 'dominant' ? 'D' : 'P';
    badge.style.background   = col + '33';
    badge.style.borderColor  = col;
    badge.style.color        = col;
  }
  renderGraph(); renderPreview();
  S.lastRaf = requestAnimationFrame(rafTick);
}

// ── Storage ────────────────────────────────────────────────────────────────────
export function saveToLocal() {
  try { localStorage.setItem('chord_bot_graph', JSON.stringify({ nodes: S.nodes, edges: S.edges })); } catch {}
}

function loadFromLocal() {
  try {
    const raw = localStorage.getItem('chord_bot_graph'); if (!raw) return false;
    const g = JSON.parse(raw); S.nodes = g.nodes || []; S.edges = g.edges || [];
    S.nodeCounter = S.nodes.reduce((m, n) => Math.max(m, parseInt(n.id.replace(/\D/g, '')) + 1), 1);
    snapAll(); return true;
  } catch { return false; }
}

function loadDefaultGraph() {
  S.nodes = [
    { id:'n1', type:'tonic',    x:60,  y:TIMELINE_Y, params:{ key:'C', mode:'major', duration:4, octave:4, velocity:80 }, paramKeyframes:{} },
    { id:'n2', type:'function', x:220, y:TIMELINE_Y, params:{ target:'subdominant', style:'jazz', strength:0.8, allow_substitutions:false, voice_lead:true, duration:4, octave:4, velocity:80, seed:0 }, paramKeyframes:{} },
    { id:'n3', type:'function', x:380, y:TIMELINE_Y, params:{ target:'dominant', style:'jazz', strength:0.9, allow_substitutions:true, voice_lead:true, duration:4, octave:4, velocity:80, seed:0 }, paramKeyframes:{} },
    { id:'n4', type:'cadence',  x:540, y:TIMELINE_Y, params:{ type:'authentic', strength:0.9, duration:4, octave:4, velocity:80, style:'jazz' }, paramKeyframes:{} },
    { id:'n5', type:'tension_shaper', x:235, y:TIMELINE_Y+NODE_H_H, params:{ amount:0.3, target_tension:-1, octave:4 }, paramKeyframes:{} },
  ];
  S.edges = [
    { src_node:'n1', src_port:'harmonic_out', dst_node:'n2', dst_port:'harmonic_in' },
    { src_node:'n2', src_port:'harmonic_out', dst_node:'n3', dst_port:'harmonic_in' },
    { src_node:'n3', src_port:'harmonic_out', dst_node:'n4', dst_port:'harmonic_in' },
    { src_node:'n2', src_port:'harmonic_out', dst_node:'n5', dst_port:'harmonic_in' },
  ];
  S.nodeCounter = 10; S.selectedNode = null; S.drawerAugIdx = -1;
  S.sequence = []; S.totalBeats = 0;
  openDrawer(false); saveToLocal(); renderGraph(); renderPreview();
  setStatus('Example loaded — press Execute or Enter');
}

function setStatus(msg) { document.getElementById('statusBar').textContent = msg; }

// ── Init ───────────────────────────────────────────────────────────────────────
async function init() {
  // Load node definitions
  try { S.nodeDefs = await apiNodeDefs(); }
  catch (e) { console.warn('No node defs:', e); }

  // Fetch tunnel URLs for cross-links
  try {
    const t = await (await fetch('/api/tunnel-url')).json();
    const chordEl = document.getElementById('tunnelLink');
    if (t.chord && t.chord.url) { chordEl.href = t.chord.url; chordEl.textContent = '🌐 ' + t.chord.url.replace(/^https?:\/\//, '').replace(/\.trycloudflare\.com$/, ''); }
    const pipeEl = document.getElementById('pipelineLink');
    if (t.pipeline && t.pipeline.url) { pipeEl.href = t.pipeline.url; }
  } catch (e) { /* no tunnel */ }

  // Wire module callbacks
  const drawerCb = { removeNode, pushHistory, saveToLocal, renderGraph, getNodeMeta, getVChildren, getAugSummary, isHoriz };
  const railCb   = { removeNode, showNodePicker, renderParamDrawer, openDrawer, pushHistory, saveToLocal, getHChain, getVChildren, setChainOrder, blockWidth, getNodeMeta, getAugSummary, isHoriz };
  initDrawer(drawerCb);
  initRail(railCb);

  initPreviewEvents();
  resizePreviewCanvases();
  new ResizeObserver(resizePreviewCanvases).observe(document.getElementById('previewCanvases'));

  initRailEvents();

  // Param drawer close — only via X button, Escape, or re-clicking the selected card.
  document.getElementById('paramDrawerClose').addEventListener('click', () => {
    S.selectedNode = null; S.drawerAugIdx = -1; openDrawer(false); renderGraph();
  });

  // + augmenter button
  document.getElementById('drawerAugAdd').addEventListener('click', e => {
    if (!S.selectedNode || !isHoriz(S.selectedNode.type)) return;
    showNodePicker(S.selectedNode.id, e.target, 'v');
  });

  // Load state
  if (!loadFromLocal()) loadDefaultGraph();
  else { renderGraph(); renderPreview(); }

  // Header buttons
  document.getElementById('btnExec').addEventListener('click', executeGraph);
  document.getElementById('btnPlay').addEventListener('click', togglePlay);
  document.getElementById('btnExport').addEventListener('click', exportMidi);
  document.getElementById('btnSaveJson').addEventListener('click', saveProjectJson);
  document.getElementById('btnLoadJson').addEventListener('click', () => document.getElementById('jsonFileIn').click());
  document.getElementById('jsonFileIn').addEventListener('change', e => {
    const f = e.target.files[0]; if (f) { loadProjectJson(f); e.target.value = ''; }
  });
  document.getElementById('btnAdd').addEventListener('click', e => {
    const chain = getHChain();
    showNodePicker(chain.length > 0 ? chain[chain.length - 1].id : '', e.target, 'h');
  });
  document.getElementById('btnClear').addEventListener('click', () => {
    S.nodes = []; S.edges = []; S.sequence = []; S.selectedNode = null; S.drawerAugIdx = -1; S.totalBeats = 0;
    openDrawer(false); saveToLocal(); renderGraph(); renderPreview();
  });
  document.getElementById('btnDefault').addEventListener('click', loadDefaultGraph);

  // Keyboard
  document.addEventListener('keydown', e => {
    if ((e.ctrlKey || e.metaKey) && e.key === 'z' && !e.shiftKey) { e.preventDefault(); undo(); return; }
    if ((e.ctrlKey || e.metaKey) && (e.key === 'y' || (e.key === 'z' && e.shiftKey))) { e.preventDefault(); redo(); return; }
    if (e.key === 'Escape') {
      closePickers();
      if (S.selectedNode) { S.selectedNode = null; S.drawerAugIdx = -1; openDrawer(false); renderGraph(); }
      return;
    }
    if (document.activeElement !== document.body) return;
    if (e.key === 'Delete' || e.key === 'Backspace') { if (S.selectedNode) removeNode(S.selectedNode.id); }
    if (e.key === 'Enter') executeGraph();
    if (e.key === ' ') { e.preventDefault(); togglePlay(); }
  });

  // Mobile tabs
  const mt = document.getElementById('mobTabs');
  if (mt) {
    function switchPanel(name) {
      mt.querySelectorAll('button').forEach(b => b.classList.remove('active'));
      mt.querySelector(`[data-panel="${name}"]`)?.classList.add('active');
      if (name === 'preview') requestAnimationFrame(() => { resizePreviewCanvases(); renderPreview(); });
    }
    mt.querySelectorAll('button').forEach(btn => btn.addEventListener('click', () => switchPanel(btn.dataset.panel)));
    if (window.innerWidth <= 640) switchPanel('timeline');
  }
}

window.addEventListener('DOMContentLoaded', init);
