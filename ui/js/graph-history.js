// ════════════════════════════════════════════════════════════════
// GRAPH HISTORY — undo / redo
// ════════════════════════════════════════════════════════════════
//
// WHY THIS SHAPE
// --------------
// Every graph mutation in graph.js already funnels through gSave() — 37 call
// sites covering add, delete, wire, unwire, drag, param edit, group, ungroup.
// Wrapping that one function captures history without touching any of them,
// so a new mutation added later is undoable for free instead of silently
// bypassing a command stack.
//
// Snapshots, not commands: the graph document is small (nodes + edges), and a
// snapshot cannot drift out of sync with the engine the way a hand-written
// inverse-op table does. gSave() already serialises the same payload.
//
// View state (pan / zoom / canvas size) is deliberately NOT part of a
// snapshot. Panning is not an edit, and undo should never yank the viewport.
// Because pan/zoom also call gSave(), those saves produce a byte-identical
// snapshot and are dropped by the dedupe below.

(function () {
  'use strict';

  const LIMIT       = 120;  // snapshots retained (~a few MB worst case)
  const COALESCE_MS = 450;  // gesture window: a drag or slider scrub = 1 entry

  let undoStack = [];   // states, oldest → newest; the LAST entry is "now"
  let redoStack = [];
  let applying  = false; // guard so our own restore doesn't re-record
  let lastAt    = 0;
  let lastSig   = null;

  // Structural fingerprint: node identities + wiring. Moving a node or
  // scrubbing a slider leaves this unchanged — that is exactly what lets a
  // continuous gesture collapse into one undo entry, while a wire or a
  // delete always starts a new one.
  function signature() {
    return gNodes.map(n => n.id).join(',') + '|' +
           gEdges.map(e => `${e.src_node}:${e.src_port}>${e.dst_node}:${e.dst_port}`)
                 .sort().join(',');
  }

  function snapshot() {
    return JSON.stringify({ nodes: gNodes, edges: gEdges });
  }

  function record() {
    if (applying) return;
    const snap = snapshot();
    const top  = undoStack[undoStack.length - 1];
    if (snap === top) return;          // no-op save (pan, zoom, re-save) — ignore

    const now = Date.now();
    const sig = signature();
    // Same wiring + still inside the gesture window → fold into the current
    // entry rather than stacking one undo step per pointermove.
    if (top && sig === lastSig && now - lastAt < COALESCE_MS && undoStack.length > 1) {
      undoStack[undoStack.length - 1] = snap;
    } else {
      undoStack.push(snap);
      if (undoStack.length > LIMIT) undoStack.shift();
    }
    lastSig = sig;
    lastAt  = now;
    redoStack.length = 0;              // new edit invalidates the redo branch
    refresh();
  }

  // Rebuild the document + DOM from a snapshot. Mirrors gDoClear()'s teardown
  // so no stale node divs or edge paths survive the swap.
  function apply(snapStr) {
    applying = true;
    try {
      const s = JSON.parse(snapStr);
      gNodes = s.nodes || [];
      gEdges = s.edges || [];
      // Keep the id counter ahead of anything restored, or the next wire
      // reuses an existing edge id.
      gEdgeCounter = Math.max(0, ...gEdges.map(e => parseInt(e.id?.slice(1)) || 0));

      gSelectedNode = null;
      gSelectedEdge = null;
      gSelectedNodes.clear();

      gNodesEl.innerHTML = '';
      gEdgesEl.innerHTML = '';
      gNodes.forEach(gRenderAnyNode);
      gUpdateConnectedPorts();
      gRedrawEdges();
      gShowNodeParams(null);
      if (isMobile()) gParamsSheetClose();

      gSave();                        // persist; applying=true keeps it silent
    } finally {
      applying = false;
    }
    refresh();
  }

  function undo() {
    if (undoStack.length < 2) { gShowToast('Nothing to undo'); return; }
    redoStack.push(undoStack.pop());
    apply(undoStack[undoStack.length - 1]);
    lastSig = null;                   // don't coalesce across an undo
    gShowToast('Undo');
  }

  function redo() {
    if (!redoStack.length) { gShowToast('Nothing to redo'); return; }
    const snap = redoStack.pop();
    undoStack.push(snap);
    apply(snap);
    lastSig = null;
    gShowToast('Redo');
  }

  // ── Toolbar buttons ────────────────────────────────────────────
  const btnUndo = document.getElementById('graph-undo-btn');
  const btnRedo = document.getElementById('graph-redo-btn');

  function refresh() {
    if (btnUndo) {
      btnUndo.disabled = undoStack.length < 2;
      btnUndo.title = `Undo${undoStack.length > 1 ? ` (${undoStack.length - 1})` : ''} — ⌘Z`;
    }
    if (btnRedo) {
      btnRedo.disabled = redoStack.length === 0;
      btnRedo.title = `Redo${redoStack.length ? ` (${redoStack.length})` : ''} — ⇧⌘Z`;
    }
  }

  btnUndo?.addEventListener('click', undo);
  btnRedo?.addEventListener('click', redo);

  // ── Keyboard ───────────────────────────────────────────────────
  // Bound on capture so the graph's own Delete/Space handlers can't swallow
  // the combo, but text fields still get their native undo.
  document.addEventListener('keydown', (e) => {
    if (!(e.metaKey || e.ctrlKey)) return;
    if (e.key.toLowerCase() !== 'z' && e.key.toLowerCase() !== 'y') return;

    const tag = document.activeElement?.tagName;
    if (tag === 'INPUT' || tag === 'TEXTAREA' || document.activeElement?.isContentEditable) return;
    if (document.getElementById('tab-graph')?.style.display === 'none') return;

    e.preventDefault();
    const isRedo = e.key.toLowerCase() === 'y' || e.shiftKey;
    isRedo ? redo() : undo();
  }, true);

  // ── Install ────────────────────────────────────────────────────
  // gSave is a function declaration, so it lives on the global object and
  // every existing `gSave()` call resolves through this wrapper.
  const saveOrig = window.gSave;
  window.gSave = function () {
    saveOrig.apply(this, arguments);
    record();
  };

  // Baseline: graph.js already ran gLoad() synchronously, so gNodes/gEdges
  // hold the restored document by the time this file executes.
  undoStack.push(snapshot());
  lastSig = signature();
  refresh();

  // Exposed for the clipboard module and for debugging.
  window.gHistory = { undo, redo, record, depth: () => undoStack.length - 1 };
})();
