// Chord Bot — timeline rail (block rendering + drag/resize interactions)
'use strict';

import { S } from './state.js';
import { FUNC_COLOR, PX_PER_BEAT_RESIZE } from './config.js';

// Callbacks injected by app.js to break circular dep
let _cb = {};

/**
 * Call once from app.js init.
 * @param {{ removeNode, showNodePicker, renderParamDrawer, openDrawer,
 *            pushHistory, saveToLocal,
 *            getHChain, getVChildren, setChainOrder,
 *            blockWidth, getNodeMeta, getAugSummary, isHoriz }} callbacks
 */
export function initRail(callbacks) {
  _cb = callbacks;
}

// ── Block HTML builder ────────────────────────────────────────────────────────
function blockHTML(node) {
  const meta   = node.type === 'subgraph' ? { name: node.label || 'Group' } : _cb.getNodeMeta(node.type);
  const isSel  = S.selectedNode?.id === node.id;
  const se     = S.sequence.find(e => e.node_id === node.id);
  const fnCol  = se ? (FUNC_COLOR[se.state.function] || '#e67e22') : '#e67e22';
  const chord  = se?.state?.chord   || '';
  const numeral= se?.state?.numeral || '';

  // Currently playing node
  let curNodeId = null;
  if (S.sequence.length > 0 && S.totalBeats > 0) {
    const cur = S.sequence.find(e => e.start_beat <= S.playhead && S.playhead < e.end_beat)
             || (S.playhead >= S.totalBeats ? S.sequence[S.sequence.length - 1] : null);
    if (cur) curNodeId = cur.node_id;
  }
  const isPlay = curNodeId === node.id;
  const isSub  = node.type === 'subgraph';

  const cls = ['nc', isSel ? 'nc-sel' : '', isPlay ? 'nc-play' : '', isSub ? 'nc-sub' : ''].filter(Boolean).join(' ');
  const w   = _cb.blockWidth(node);

  const augs   = _cb.getVChildren(node.id);
  const dots   = augs.map(v => `<span class="nc-aug-dot" title="${(_cb.getNodeMeta(v.type).name || v.type)}"></span>`).join('');
  const augStr = augs.length ? augs.map(v => (_cb.getNodeMeta(v.type).name || v.type).slice(0, 6)).join(' ') : '';
  const label  = chord || (isSub ? node.label : meta.name || node.type);

  const tval = se?.state?.tension ?? -1;
  const tBar = tval >= 0
    ? `<div class="nc-tension" style="background:hsl(${Math.round(120 - tval * 120)},70%,45%)"></div>`
    : '';

  return `<div class="${cls}" data-id="${node.id}" style="width:${w}px;border-left-color:${fnCol}">
  ${tBar}
  <button class="nc-del" data-del="${node.id}">×</button>
  <div class="nc-resize" data-resize="${node.id}"></div>
  <div class="nc-top">
    <span class="nc-numeral">${numeral}</span>
    <span class="nc-type-label">${numeral ? '' : (meta.name || node.type)}</span>
  </div>
  <div class="nc-chord">${label}</div>
  <div class="nc-bottom">
    ${dots}
    ${augStr ? `<span class="nc-aug-label">${augStr}</span>` : ''}
  </div>
</div>`;
}

// ── Public render ─────────────────────────────────────────────────────────────
export function renderGraph() {
  const rail = document.getElementById('nodeRail');
  if (!rail) return;
  const chain = _cb.getHChain();

  if (chain.length === 0) {
    rail.innerHTML = '<div id="railEmpty"><span style="font-size:22px;opacity:.4">♩</span> Press <strong>＋ Add</strong> or click <strong>+</strong> to build a progression</div>';
    return;
  }

  const ins = afterId => `<button class="nc-ins" data-ins-after="${afterId}">+</button>`;
  let html = ins('');
  chain.forEach(n => { html += blockHTML(n) + ins(n.id); });
  rail.innerHTML = html;

  attachRailDrag(rail);
  attachRailResize(rail);
}

// ── Rail click events ─────────────────────────────────────────────────────────
export function initRailEvents() {
  const rail = document.getElementById('nodeRail');
  rail.addEventListener('click', e => {
    if (e.target.dataset.del) {
      e.stopPropagation();
      _cb.removeNode(e.target.dataset.del); return;
    }
    const ins = e.target.closest('.nc-ins');
    if (ins) { _cb.showNodePicker(ins.dataset.insAfter, ins, 'h'); return; }

    const card = e.target.closest('[data-id]');
    if (card) {
      const node = S.nodes.find(n => n.id === card.dataset.id);
      if (!node) return;
      if (S.selectedNode?.id === node.id) {
        S.selectedNode = null; S.drawerAugIdx = -1;
        _cb.openDrawer(false);
      } else {
        S.selectedNode = node; S.drawerAugIdx = -1;
        _cb.renderParamDrawer(); _cb.openDrawer(true);
      }
      renderGraph(); return;
    }

    if (e.target === rail) {
      S.selectedNode = null; S.drawerAugIdx = -1;
      _cb.openDrawer(false); renderGraph();
    }
  });
}

// ── Drag to reorder ───────────────────────────────────────────────────────────
function attachRailDrag(rail) {
  rail.querySelectorAll('.nc').forEach(card => {
    card.addEventListener('pointerdown', e => {
      if (e.target.dataset.del || e.target.closest('.nc-ins')) return;
      const dragId = card.dataset.id;
      let moved = false;
      const ox = e.clientX;

      function onMove(ev) {
        if (!moved && Math.abs(ev.clientX - ox) > 6) moved = true;
        if (!moved) return;
        card.classList.add('drag-ghost');
        const slots = [...rail.querySelectorAll('.nc-ins')];
        let best = null, bestD = Infinity;
        slots.forEach(s => {
          const r = s.getBoundingClientRect();
          const d = Math.abs(ev.clientX - (r.left + r.width / 2));
          if (d < bestD) { bestD = d; best = s; }
        });
        slots.forEach(s => s.classList.remove('drag-over'));
        if (best) best.classList.add('drag-over');
      }

      function onUp() {
        card.removeEventListener('pointermove', onMove);
        card.classList.remove('drag-ghost');
        const active = rail.querySelector('.nc-ins.drag-over');
        rail.querySelectorAll('.nc-ins').forEach(s => s.classList.remove('drag-over'));
        if (moved && active) {
          _cb.pushHistory();
          const afterId = active.dataset.insAfter;
          const chain = _cb.getHChain();
          const fi = chain.findIndex(n => n.id === dragId);
          if (fi > -1) {
            const [mv] = chain.splice(fi, 1);
            let ti = afterId === '' ? 0 : chain.findIndex(n => n.id === afterId) + 1;
            if (ti < 0) ti = chain.length;
            chain.splice(ti, 0, mv);
            _cb.setChainOrder(chain);
            _cb.saveToLocal();
          }
          renderGraph();
        }
      }

      card.addEventListener('pointermove', onMove);
      card.addEventListener('pointerup', onUp, { once: true });
    });
  });
}

// ── Right-edge resize ─────────────────────────────────────────────────────────
function attachRailResize(rail) {
  rail.querySelectorAll('[data-resize]').forEach(handle => {
    const nodeId = handle.dataset.resize;
    let startX = 0, startDur = 0, historyPushed = false;

    function onMove(e) {
      const dx   = e.clientX - startX;
      const node = S.nodes.find(n => n.id === nodeId);
      if (!node) return;
      const meta = _cb.getNodeMeta(node.type);
      const min  = meta?.params?.duration?.min ?? 0.25;
      const max  = meta?.params?.duration?.max ?? 32;
      node.params.duration = Math.max(min, Math.min(max,
        Math.round((startDur + dx / PX_PER_BEAT_RESIZE) * 4) / 4
      ));
      _cb.saveToLocal(); renderGraph();
    }

    function onUp() {
      document.removeEventListener('pointermove', onMove);
      document.removeEventListener('pointerup', onUp);
      handle.classList.remove('resizing');
    }

    handle.addEventListener('pointerdown', e => {
      e.stopPropagation(); e.preventDefault();
      if (!historyPushed) { _cb.pushHistory(); historyPushed = true; }
      const node = S.nodes.find(n => n.id === nodeId);
      if (!node) return;
      startX   = e.clientX;
      startDur = parseFloat(node.params.duration) || 4;
      handle.classList.add('resizing');
      handle.setPointerCapture(e.pointerId);
      document.addEventListener('pointermove', onMove);
      document.addEventListener('pointerup', onUp, { once: true });
      document.addEventListener('pointerup', () => { historyPushed = false; }, { once: true });
    });
  });
}
