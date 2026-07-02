// Chord Bot — parameter drawer (augmenter list + param editor)
'use strict';

import { S } from './state.js';

// Callbacks injected by app.js to avoid circular imports
let _cb = {};

/**
 * Call once from app.js init, passing functions that drawer needs from app.
 * @param {{ removeNode, pushHistory, saveToLocal, renderGraph,
 *            getNodeMeta, getVChildren, getAugSummary, isHoriz }} callbacks
 */
export function initDrawer(callbacks) {
  _cb = callbacks;
}

// ── Drawer open/close ──────────────────────────────────────────────────────────
const paramDrawer = document.getElementById('paramDrawer');

export function openDrawer(_open) {
  // Drawer is always visible — never collapses.
  paramDrawer.classList.add('open');
}

// ── Param rendering ────────────────────────────────────────────────────────────
const CHOICE_RE = /\(([^)]+(?:\/[^)]+)+)\)/;
function extractChoices(desc) {
  const m = CHOICE_RE.exec(desc || '');
  return m ? m[1].split('/').map(s => s.trim()).filter(Boolean) : null;
}

export function renderParams(node, meta, container) {
  let html = '';
  for (const [k, spec] of Object.entries(meta.params || {})) {
    const val = node.params[k] ?? spec.default ?? '';
    const choices = extractChoices(spec.description);
    html += `<div class="paramRow"><label>${k}</label>`;
    if (typeof spec.default === 'boolean') {
      html += `<input type="checkbox" data-param="${k}" ${val ? 'checked' : ''}>`;
    } else if (choices) {
      html += `<select data-param="${k}">${choices.map(c => `<option${c === String(val) ? ' selected' : ''}>${c}</option>`).join('')}</select>`;
    } else if (typeof spec.default === 'number' && spec.min !== undefined) {
      html += `<input type="range"  data-param="${k}"     min="${spec.min}" max="${spec.max}" step="${(spec.max - spec.min) / 100}" value="${val}">`;
      html += `<input type="number" data-param="${k}_num" min="${spec.min}" max="${spec.max}" step="any" class="numdisp" value="${val}">`;
    } else {
      html += `<input type="text" data-param="${k}" value="${String(val).replace(/"/g, '&quot;')}">`;
    }
    html += '</div>';
    if (spec.description) html += `<div class="paramHint">${spec.description.slice(0, 60)}</div>`;
  }
  container.innerHTML = html;

  container.querySelectorAll('[data-param]').forEach(el => {
    el.addEventListener('focus', () => { _cb.pushHistory(); }, { once: false });
    el.addEventListener(el.type === 'range' ? 'input' : 'change', () => {
      let pk = el.dataset.param;
      const isNum = pk.endsWith('_num');
      if (isNum) pk = pk.slice(0, -4);
      let v = el.type === 'checkbox' ? el.checked
        : (el.type === 'range' || el.type === 'number') ? parseFloat(el.value)
        : el.value;
      node.params[pk] = v;
      if (el.type === 'range') {
        const n = container.querySelector(`[data-param="${pk}_num"]`); if (n) n.value = v;
      } else if (isNum) {
        const r = container.querySelector(`[data-param="${pk}"]`); if (r && r.type === 'range') r.value = v;
      }
      _cb.saveToLocal(); _cb.renderGraph();
    });
  });
}

// ── Full drawer render ─────────────────────────────────────────────────────────
export function renderParamDrawer() {
  if (!S.selectedNode) {
    document.getElementById('paramDrawerTitle').textContent = 'No selection';
    document.getElementById('paramDrawerSub').textContent = '';
    document.getElementById('drawerAugsWrap').classList.remove('has-content');
    document.getElementById('drawerAugs').innerHTML = '';
    document.getElementById('paramScrollArea').innerHTML =
      '<div style="color:var(--dim);font-size:10px;padding:4px 0">Click a node to edit its parameters</div>';
    return;
  }

  const isH  = _cb.isHoriz(S.selectedNode.type);
  const meta  = S.selectedNode.type === 'subgraph'
    ? { name: S.selectedNode.label || 'Group', params: {} }
    : _cb.getNodeMeta(S.selectedNode.type);

  document.getElementById('paramDrawerTitle').textContent = meta.name || S.selectedNode.type;

  // Augmenter rows (H nodes only)
  const augs = isH ? _cb.getVChildren(S.selectedNode.id) : [];
  const augsWrap      = document.getElementById('drawerAugsWrap');
  const augsContainer = document.getElementById('drawerAugs');
  augsContainer.innerHTML = '';

  if (isH && augs.length > 0) {
    augsWrap.classList.add('has-content');
    augs.forEach((v, i) => {
      const vm  = _cb.getNodeMeta(v.type);
      const nm  = (vm.name || v.type).slice(0, 22);
      const sm  = _cb.getAugSummary(v);
      const row = document.createElement('div');
      row.className    = 'drawer-aug' + (S.drawerAugIdx === i ? ' active' : '');
      row.dataset.augId  = v.id;
      row.dataset.augIdx = i;
      row.innerHTML =
        `<span class="drawer-aug-name">${nm}</span>`
        + (sm ? `<span class="drawer-aug-val">${sm}</span>` : '')
        + `<button class="drawer-aug-del" data-del="${v.id}">×</button>`;
      row.addEventListener('click', e => {
        if (e.target.dataset.del) return;
        S.drawerAugIdx = S.drawerAugIdx === i ? -1 : i;
        renderParamDrawer();
      });
      row.querySelector('[data-del]').addEventListener('click', e => {
        e.stopPropagation();
        _cb.removeNode(v.id);
      });
      augsContainer.appendChild(row);
    });
  } else {
    augsWrap.classList.remove('has-content');
  }

  // Which node's params to show
  const showNode = (isH && S.drawerAugIdx >= 0 && augs[S.drawerAugIdx]) ? augs[S.drawerAugIdx] : S.selectedNode;
  const showMeta = showNode === S.selectedNode ? meta : _cb.getNodeMeta(showNode.type);

  document.getElementById('paramDrawerSub').textContent =
    showNode !== S.selectedNode ? `← ${meta.name || S.selectedNode.type}` : '';

  renderParams(showNode, showMeta, document.getElementById('paramScrollArea'));
}
