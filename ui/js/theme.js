// ════════════════════════════════════════════════════════════════
// THEME ENGINE — presets + live token editor
// ════════════════════════════════════════════════════════════════
//
// Two layers, deliberately separate:
//
//   1. PRESET  — a `data-theme` attribute on <html>. The values live in
//      editor.css so a preset costs nothing at runtime and cannot drift from
//      the stylesheet.
//   2. OVERRIDES — per-token inline properties on <html>, layered on top of
//      whichever preset is active. These are what the editor writes.
//
// Keeping them separate is what makes "tweak the accent, then try it against
// every preset" work: overrides are stored as a sparse patch, not a full
// snapshot, so switching preset keeps your edits and drops the rest.
//
// The pre-paint bootstrap in index.html applies BOTH before first paint —
// edit that snippet too if the storage keys ever change.

(function () {
  'use strict';

  const KEY_PRESET = 'gm-theme';
  const KEY_CUSTOM = 'gm-theme-custom';

  // ── Presets ────────────────────────────────────────────────────
  // `sw` drives the picker swatch: [surface, accent, text].
  const PRESETS = {
    ember:    { name: 'Ember',    desc: 'Charcoal + flame — the bold default', sw: ['#1b1715', '#f97316', '#f5efe8'] },
    flux:     { name: 'Flux',     desc: 'Cold cyan/violet, maximum contrast',  sw: ['#111725', '#22d3ee', '#eaf4ff'] },
    slate:    { name: 'Slate',    desc: 'Neutral gray, muted steel accent',    sw: ['#1a1c1f', '#64748b', '#e7e9ec'] },
    graphite: { name: 'Graphite', desc: 'Warm near-black, ember accent',       sw: ['#181816', '#c2622d', '#e8e6e1'] },
    midnight: { name: 'Midnight', desc: 'Deep blue, neon accent',              sw: ['#161a24', '#6c8ef5', '#e6ebf4'] },
  };
  const DEFAULT_PRESET = 'ember';
  // Slate is the bare :root block, so it is the one preset with no attribute.
  const BASE_PRESET = 'slate';

  // ── Token registry ─────────────────────────────────────────────
  // Drives the editor UI. `type` picks the control; order here is the order
  // shown. Only tokens listed here are user-editable — internal ones
  // (fonts, shadows) stay out so the editor can't produce an unusable UI.
  const GROUPS = [
    { label: 'Surfaces', tokens: [
      ['--bg0', 'Base'], ['--bg1', 'Panel'], ['--bg2', 'Raised'], ['--bg3', 'Overlay'],
      ['--border', 'Border'], ['--border-strong', 'Border strong'],
    ]},
    { label: 'Text', tokens: [
      ['--text', 'Primary'], ['--muted', 'Muted'], ['--muted2', 'Faint'],
    ]},
    { label: 'Accent', tokens: [
      ['--accent', 'Accent'], ['--accent2', 'Accent 2'],
    ]},
    { label: 'Semantic', tokens: [
      ['--success', 'Success'], ['--warn', 'Warning'], ['--err', 'Error'], ['--live', 'Live'],
    ]},
    { label: 'Graph', tokens: [
      ['--node-bg', 'Node body'], ['--node-head', 'Node header'],
      // Typed wires take their colour from the engine's port-type spec
      // (graph.js sets it inline); this token is the untyped fallback only.
      ['--node-border', 'Node border'], ['--wire', 'Wire (untyped)'], ['--grid-dot', 'Grid dot'],
    ]},
    { label: '3D viewport', tokens: [
      ['--viewport-bg', 'Stage'], ['--viewport-grid', 'Grid'],
    ]},
    { label: 'Shape', type: 'range', tokens: [
      ['--radius-s', 'Radius small', 0, 14],
      ['--radius-m', 'Radius medium', 0, 20],
      ['--radius-l', 'Radius large', 0, 28],
    ]},
  ];

  const COLOR_TOKENS = GROUPS.filter(g => g.type !== 'range')
                             .flatMap(g => g.tokens.map(t => t[0]));

  // ── State ──────────────────────────────────────────────────────
  const read = (k, fallback) => { try { return localStorage.getItem(k) ?? fallback; } catch { return fallback; } };
  let preset = read(KEY_PRESET, DEFAULT_PRESET);
  if (!PRESETS[preset]) preset = DEFAULT_PRESET;
  let custom = {};
  try { custom = JSON.parse(read(KEY_CUSTOM, '{}')) || {}; } catch { custom = {}; }

  const root = document.documentElement;
  // Declared up here because apply() runs at boot — before the editor is
  // built — and syncUI() reads it. A later `const` would be in its TDZ.
  const inputs = new Map();   // token -> {el, kind, hexEl|valEl}

  function persist() {
    try {
      localStorage.setItem(KEY_PRESET, preset);
      localStorage.setItem(KEY_CUSTOM, JSON.stringify(custom));
    } catch {}
  }

  function applyPreset() {
    if (preset === BASE_PRESET) delete root.dataset.theme;
    else root.dataset.theme = preset;
  }

  function applyOverrides() {
    // Clear first: a token dropped from `custom` must fall back to the preset,
    // and setProperty alone never removes anything.
    for (const t of COLOR_TOKENS) root.style.removeProperty(t);
    for (const g of GROUPS) if (g.type === 'range') for (const [t] of g.tokens) root.style.removeProperty(t);
    for (const [t, v] of Object.entries(custom)) root.style.setProperty(t, v);
  }

  function apply() {
    applyPreset(); applyOverrides(); persist(); syncUI();
    notify();
  }

  // Anything that can't read the cascade — WebGL scenes, canvas overlays —
  // listens for this instead of polling computed styles every frame.
  let notifyT = 0;
  function notify() {
    clearTimeout(notifyT);
    notifyT = setTimeout(() => {
      window.dispatchEvent(new CustomEvent('gm-theme-change', { detail: { preset } }));
    }, 16);
  }

  // Resolved value of a token as it would render with `custom` removed —
  // i.e. what the preset alone says. Used to seed the editor's inputs.
  function presetValue(token) {
    const saved = custom[token];
    if (saved !== undefined) root.style.removeProperty(token);
    const v = getComputedStyle(root).getPropertyValue(token).trim();
    if (saved !== undefined) root.style.setProperty(token, saved);
    return v;
  }

  function currentValue(token) {
    return (custom[token] ?? presetValue(token)).trim();
  }

  // ── Boot ───────────────────────────────────────────────────────
  apply();

  // ════════════════════════════════════════════════════════════════
  // Editor UI
  // ════════════════════════════════════════════════════════════════
  const picker = document.getElementById('theme-picker');
  const editor = document.getElementById('theme-editor');
  if (!picker) return;

  // ── Preset cards ───────────────────────────────────────────────
  for (const [id, t] of Object.entries(PRESETS)) {
    const card = document.createElement('button');
    card.className = 'theme-card';
    card.dataset.theme = id;
    card.innerHTML =
      `<span class="theme-swatches">${t.sw.map(c => `<i style="background:${c}"></i>`).join('')}</span>` +
      `<span class="theme-name">${t.name}</span>` +
      `<span class="theme-desc">${t.desc}</span>`;
    card.addEventListener('click', () => { preset = id; apply(); });
    picker.appendChild(card);
  }

  if (!editor) { syncUI(); return; }

  // ── Token controls ─────────────────────────────────────────────
  // Built once; values re-seeded by syncUI on every preset change.
  for (const g of GROUPS) {
    const sec = document.createElement('div');
    sec.className = 'tk-group';
    sec.innerHTML = `<div class="tk-group-label">${g.label}</div>`;
    const grid = document.createElement('div');
    grid.className = 'tk-grid';

    for (const spec of g.tokens) {
      const [token, label, min, max] = spec;
      const row = document.createElement('label');
      row.className = 'tk-row';

      if (g.type === 'range') {
        row.innerHTML =
          `<span class="tk-label">${label}</span>` +
          `<input type="range" min="${min}" max="${max}" step="1" class="tk-range">` +
          `<span class="tk-val"></span>`;
        const el = row.querySelector('input');
        el.addEventListener('input', () => {
          custom[token] = el.value + 'px';
          root.style.setProperty(token, custom[token]);
          row.querySelector('.tk-val').textContent = el.value + 'px';
          persist(); markDirty(); notify();
        });
        inputs.set(token, { el, kind: 'range', valEl: row.querySelector('.tk-val') });
      } else {
        row.innerHTML =
          `<span class="tk-label">${label}</span>` +
          `<input type="color" class="tk-color">` +
          `<code class="tk-hex"></code>`;
        const el = row.querySelector('input');
        el.addEventListener('input', () => {
          custom[token] = el.value;
          root.style.setProperty(token, el.value);
          row.querySelector('.tk-hex').textContent = el.value;
          persist(); markDirty(); notify();
        });
        inputs.set(token, { el, kind: 'color', hexEl: row.querySelector('.tk-hex') });
      }
      grid.appendChild(row);
    }
    sec.appendChild(grid);
    editor.appendChild(sec);
  }

  // ── Actions ────────────────────────────────────────────────────
  const actions = document.createElement('div');
  actions.className = 'tk-actions';
  actions.innerHTML =
    `<button id="tk-reset" class="g-autobtn">Reset to preset</button>` +
    `<button id="tk-export" class="g-autobtn">Export JSON</button>` +
    `<button id="tk-import" class="g-autobtn">Import JSON</button>` +
    `<span id="tk-dirty"></span>`;
  editor.appendChild(actions);

  function markDirty() {
    const n = Object.keys(custom).length;
    const el = document.getElementById('tk-dirty');
    if (el) el.textContent = n ? `${n} token${n === 1 ? '' : 's'} overridden` : '';
  }

  actions.querySelector('#tk-reset').addEventListener('click', () => {
    custom = {};
    apply();
    if (window.gShowToast) gShowToast('Theme reset to preset');
  });

  actions.querySelector('#tk-export').addEventListener('click', async () => {
    const payload = JSON.stringify({ preset, tokens: custom }, null, 2);
    try {
      await navigator.clipboard.writeText(payload);
      if (window.gShowToast) gShowToast('Theme JSON copied to clipboard');
    } catch {
      // Clipboard can be blocked (permissions, insecure origin) — fall back to
      // something the user can still copy by hand rather than failing silently.
      window.prompt('Theme JSON', payload);
    }
  });

  actions.querySelector('#tk-import').addEventListener('click', () => {
    const raw = window.prompt('Paste theme JSON');
    if (!raw) return;
    try {
      const parsed = JSON.parse(raw);
      const tokens = parsed.tokens || parsed;
      if (typeof tokens !== 'object' || Array.isArray(tokens)) throw new Error('expected an object of tokens');
      // Accept only known tokens so a pasted blob can't inject arbitrary CSS.
      const known = new Set([...COLOR_TOKENS,
        ...GROUPS.filter(g => g.type === 'range').flatMap(g => g.tokens.map(t => t[0]))]);
      const clean = {};
      for (const [k, v] of Object.entries(tokens)) {
        if (known.has(k) && typeof v === 'string' && /^[#a-zA-Z0-9(),.\s%-]+$/.test(v)) clean[k] = v;
      }
      if (parsed.preset && PRESETS[parsed.preset]) preset = parsed.preset;
      custom = clean;
      apply();
      if (window.gShowToast) gShowToast(`Imported ${Object.keys(clean).length} tokens`);
    } catch (e) {
      if (window.gShowToast) gShowToast('Import failed: ' + e.message, true);
    }
  });

  // ── Sync ───────────────────────────────────────────────────────
  function syncUI() {
    document.querySelectorAll('.theme-card').forEach(c =>
      c.classList.toggle('active', c.dataset.theme === preset));
    if (!inputs.size) return;
    for (const [token, rec] of inputs) {
      const v = currentValue(token);
      if (rec.kind === 'color') {
        // <input type=color> only accepts #rrggbb; skip anything else rather
        // than silently coercing a valid token to black.
        if (/^#[0-9a-fA-F]{6}$/.test(v)) rec.el.value = v;
        rec.hexEl.textContent = v;
      } else {
        const n = parseInt(v, 10);
        if (Number.isFinite(n)) { rec.el.value = n; rec.valEl.textContent = n + 'px'; }
      }
    }
    markDirty();
  }

  syncUI();

  // Public handle: lets the 3D viewport read themed colors (see editor3d.js).
  window.gTheme = {
    get: currentValue,
    preset: () => preset,
    apply: (id) => { if (PRESETS[id]) { preset = id; apply(); } },
  };
})();
