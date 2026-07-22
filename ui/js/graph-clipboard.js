// ════════════════════════════════════════════════════════════════
// GRAPH CLIPBOARD — copy / cut / paste / duplicate
// ════════════════════════════════════════════════════════════════
//
// Selection copies as a self-contained fragment: the chosen nodes plus only
// the edges whose BOTH ends are inside the selection. A half-copied wire has
// no meaning once pasted, so dangling ends are dropped rather than pointed at
// whatever node happens to share an id later.
//
// Paste re-ids every node. Ids are the engine's object identity — reusing one
// would make the pasted copy alias the original in the executor and in the
// three.js object registry. New ids use the same 'n' + base36 scheme as
// gAddNode so nothing downstream can tell a pasted node from a placed one.
//
// Mutations go through gSave(), so undo/redo picks all of this up for free.

(function () {
  'use strict';

  const OFFSET = 36;   // canvas px: a paste lands visibly clear of its source
  let clipboard = null;

  let idSeq = 0;
  function freshId() {
    return 'n' + Date.now().toString(36) + (idSeq++).toString(36) + 'c';
  }

  // Ids currently selected: the multi-select set when it has entries,
  // otherwise the single selected node.
  function selectedIds() {
    if (gSelectedNodes.size) return [...gSelectedNodes];
    return gSelectedNode ? [gSelectedNode] : [];
  }

  function copy() {
    const ids = selectedIds();
    if (!ids.length) { gShowToast('Nothing selected', true); return false; }
    const set = new Set(ids);
    clipboard = {
      nodes: gNodes.filter(n => set.has(n.id)).map(n => JSON.parse(JSON.stringify(n))),
      edges: gEdges.filter(e => set.has(e.src_node) && set.has(e.dst_node))
                   .map(e => JSON.parse(JSON.stringify(e))),
    };
    gShowToast(`Copied ${clipboard.nodes.length} node${clipboard.nodes.length === 1 ? '' : 's'}`);
    return true;
  }

  function cut() {
    if (!copy()) return;
    const ids = selectedIds();
    gClearMultiSelect();
    // gDeleteNode drops attached edges and calls gSave() per node; the history
    // module's gesture window folds the whole cut into one undo entry.
    ids.forEach(gDeleteNode);
    gShowToast(`Cut ${ids.length} node${ids.length === 1 ? '' : 's'}`);
  }

  // Insert a fragment, remapping ids. Returns the new node ids.
  function insert(frag, dx, dy) {
    // gDeleteNode now prunes the multi-select set, so an all-stale selection
    // shouldn't reach here — but an empty fragment must still be a no-op
    // rather than an exception, since every caller ends by selecting made[0].
    if (!frag.nodes.length) return [];
    const remap = new Map();
    const made = [];

    for (const src of frag.nodes) {
      const node = JSON.parse(JSON.stringify(src));
      node.id = freshId();
      node.x = (node.x || 0) + dx;
      node.y = (node.y || 0) + dy;
      node.dirty = true;
      remap.set(src.id, node.id);
      gNodes.push(node);
      made.push(node);
    }
    for (const src of frag.edges) {
      gEdges.push({
        id: 'e' + (++gEdgeCounter),
        src_node: remap.get(src.src_node),
        src_port: src.src_port,
        dst_node: remap.get(src.dst_node),
        dst_port: src.dst_port,
        feedback: !!src.feedback,
      });
    }

    made.forEach(gRenderAnyNode);
    gUpdateConnectedPorts();
    gRedrawEdges();

    // Leave the pasted fragment selected — the usual next move is to drag it.
    gClearMultiSelect();
    if (made.length === 1) {
      gSelectNode(made[0].id);
    } else {
      made.forEach(n => gToggleMultiSelect(n.id));
      gSelectNode(made[0].id);
    }

    gSave();
    return made.map(n => n.id);
  }

  function paste() {
    if (!clipboard || !clipboard.nodes.length) { gShowToast('Clipboard is empty', true); return; }
    const made = insert(clipboard, OFFSET, OFFSET);
    // Stack repeated pastes instead of piling them all on one spot.
    clipboard = JSON.parse(JSON.stringify(clipboard));
    clipboard.nodes.forEach(n => { n.x += OFFSET; n.y += OFFSET; });
    gShowToast(`Pasted ${made.length} node${made.length === 1 ? '' : 's'}`);
  }

  function duplicate() {
    const ids = selectedIds();
    if (!ids.length) { gShowToast('Nothing selected', true); return; }
    const set = new Set(ids);
    const frag = {
      nodes: gNodes.filter(n => set.has(n.id)),
      edges: gEdges.filter(e => set.has(e.src_node) && set.has(e.dst_node)),
    };
    const made = insert(frag, OFFSET, OFFSET);
    gShowToast(`Duplicated ${made.length} node${made.length === 1 ? '' : 's'}`);
  }

  // ── Keyboard ───────────────────────────────────────────────────
  document.addEventListener('keydown', (e) => {
    if (!(e.metaKey || e.ctrlKey)) return;
    const key = e.key.toLowerCase();
    if (!['c', 'x', 'v', 'd'].includes(key)) return;

    // Never shadow the native clipboard while the user is in a text field,
    // and stay out of the way when the graph tab isn't the one on screen.
    const el = document.activeElement;
    const tag = el?.tagName;
    if (tag === 'INPUT' || tag === 'TEXTAREA' || el?.isContentEditable) return;
    if (document.getElementById('tab-graph')?.style.display === 'none') return;
    // A real text selection means the user meant to copy text, not nodes.
    if ((key === 'c' || key === 'x') && String(getSelection() || '').length) return;

    e.preventDefault();
    if (key === 'c') copy();
    else if (key === 'x') cut();
    else if (key === 'v') paste();
    else if (key === 'd') duplicate();
  }, true);

  window.gClipboard = { copy, cut, paste, duplicate, hasContent: () => !!clipboard?.nodes.length };
})();
