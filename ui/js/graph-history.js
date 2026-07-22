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

  // `dirty` and the physics scratch fields (_pinned/_vx/_vy) are execution and
  // layout state, not document state. Carrying `dirty` in particular would
  // resurrect pre-run flags on undo and defeat the selective re-cook described
  // in DESIGN.md → "Dirty flags / selective recooking": the post-run
  // all-clean state is never saved, so every snapshot would pin dirty=true.
  // Stripping it also stops a bare flag flip from creating a phantom entry.
  function snapshot() {
    return JSON.stringify({
      nodes: gNodes.map(({ dirty, _pinned, _vx, _vy, ...keep }) => keep),
      edges: gEdges,
      // Timeline clips are part of the document a user expects Clear to undo.
      clips: tlClips.map(({ _selected, _origStart, _origEnd, ...keep }) => keep),
    });
  }

  // Snapshotting serialises the whole graph, and gSave() fires on every slider
  // `input` event. So the expensive part is deferred: a run of same-shape edits
  // (a scrub, a drag) settles into ONE trailing commit instead of one per tick.
  // A structural change commits immediately, or two quick node adds would
  // collapse into a single undo step.
  let pendingT = 0;

  function record() {
    if (applying) return;
    const sig = signature();
    if (sig !== lastSig) { flush(); commit(sig); return; }
    clearTimeout(pendingT);
    pendingT = setTimeout(() => { pendingT = 0; commit(sig); }, COALESCE_MS);
  }

  // Land any deferred snapshot now — undo/redo must not read a stale stack.
  // Caveat: the deferred state is only ever re-read from the live graph, so a
  // scrub followed within COALESCE_MS by a structural edit lands as ONE entry
  // covering both. Undo still returns to a coherent earlier state; it just
  // rewinds slightly further than a per-tick stack would.
  function flush() {
    if (!pendingT) return;
    clearTimeout(pendingT);
    pendingT = 0;
    commit(signature());
  }

  function commit(sig) {
    const snap = snapshot();
    if (snap === undoStack[undoStack.length - 1]) return;   // pan/zoom/re-save
    undoStack.push(snap);
    if (undoStack.length > LIMIT) undoStack.shift();
    lastSig = sig;
    redoStack.length = 0;              // new edit invalidates the redo branch
    refresh();
  }

  // Rebuild the document + DOM from a snapshot. Mirrors gDoClear()'s teardown
  // so no stale node divs or edge paths survive the swap.
  function apply(snapStr) {
    applying = true;
    try {
      const s = JSON.parse(snapStr);

      // Re-derive `dirty` instead of restoring it: a node only needs re-cooking
      // if this undo actually changed its params. Blanket-dirtying would force
      // a full re-cook of an untouched graph; blanket-clean would serve stale
      // cached output for the node that did change.
      const before = new Map(gNodes.map(n => [n.id, n]));
      gNodes = (s.nodes || []).map(n => {
        const old = before.get(n.id);
        const changed = !old || JSON.stringify(old.params) !== JSON.stringify(n.params);
        return { ...n, dirty: changed ? true : !!old.dirty };
      });
      gEdges = s.edges || [];
      gRecomputeEdgeCounter();

      // Clips ride along so Clear → undo restores the timeline too, not just
      // the nodes. Mirrors tlLoadClips()'s reconstruction of the derived fields.
      if (s.clips) {
        tlClips = s.clips.map(c => ({
          srcLength: c.endFrame - c.startFrame + 1,
          ...c,
          looped: c.looped ?? false,
          _origStart: c.startFrame,
          _origEnd: c.endFrame,
        }));
        tlClipIdCounter = Math.max(
          ...tlClips.map(c => parseInt(String(c.id).replace('clip_', '')) || 0), 0);
        tlSaveClips();
      }

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

      renderTimelineRuler();          // clip lanes + per-node KF lanes
      gSave();                        // persist; applying=true keeps it silent
    } finally {
      applying = false;
    }
    // Judge the next edit against what is actually on screen now.
    lastSig = signature();
    refresh();
  }

  function undo() {
    flush();                          // land any deferred snapshot first
    if (undoStack.length < 2) { gShowToast('Nothing to undo'); return; }
    redoStack.push(undoStack.pop());
    apply(undoStack[undoStack.length - 1]);
    gShowToast('Undo');
  }

  function redo() {
    flush();
    if (!redoStack.length) { gShowToast('Nothing to redo'); return; }
    const snap = redoStack.pop();
    undoStack.push(snap);
    apply(snap);
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
  // `flush` is public so a caller can force a deferred snapshot to land —
  // used by undo/redo above, and by tests that need a deterministic stack.
  window.gHistory = { undo, redo, record, flush, depth: () => undoStack.length - 1 };
})();
