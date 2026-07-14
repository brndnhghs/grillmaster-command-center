// ── API auth ───────────────────────────────────────────────────────
// When the server runs with GRILLMASTER_API_TOKEN (tunneled setups),
// store the token in localStorage['api-token'] and every API call
// carries it. No-op when unset.
(() => {
  const t = localStorage.getItem('api-token');
  if (!t) return;
  const orig = window.fetch.bind(window);
  window.fetch = (url, opts = {}) => {
    opts.headers = Object.assign({}, opts.headers, { 'X-Api-Token': t });
    return orig(url, opts);
  };
})();

// ── Utils / helpers ────────────────────────────────────────────────
function isMobile() { return window.matchMedia('(max-width: 768px)').matches; }
function getCatState() {
  try { return JSON.parse(localStorage.getItem('cat-state') || '{}'); }
  catch { return {}; }
}
function saveCatState(s) { localStorage.setItem('cat-state', JSON.stringify(s)); }

// ── State ──────────────────────────────────────────────────────────
let allMethods = [];
let selectedMethod = null;
let currentJobId = null;
let timerInterval = null;
let jobStartTime = null;

// ── DOM refs ───────────────────────────────────────────────────────
const methodList    = document.getElementById('method-list');
const searchInput   = document.getElementById('search');
const methodTitle   = document.getElementById('method-title');
const methodMeta    = document.getElementById('method-meta');
const paramsEmpty   = document.getElementById('params-empty');
const paramsForm    = document.getElementById('params-form');
const generateBtn   = document.getElementById('generate-btn');
const stopBtn       = document.getElementById('stop-btn');
const autoBtn       = document.getElementById('auto-btn');
const statusBadge   = document.getElementById('status-badge');
const log           = document.getElementById('log');
const elapsedTimer  = document.getElementById('elapsed-timer');
const viewer        = document.getElementById('viewer');
const resultImg     = document.getElementById('result-img');
const resultVideo   = document.getElementById('result-video');
const downloadBtn   = document.getElementById('download-btn');

// ── Mobile sidebar toggle ──────────────────────────────────────────
const sidebarEl = document.getElementById('sidebar');
document.getElementById('sidebar-toggle').addEventListener('click', () => {
  sidebarEl.classList.toggle('sidebar-open');
});
// Auto-open method list when search is focused on mobile
document.getElementById('search').addEventListener('focus', () => {
  if (isMobile()) sidebarEl.classList.add('sidebar-open');
});

// ── Mobile sticky top bar: move viewer + btn-row into it on mobile ──
const mobileTopBar    = document.getElementById('mobile-top-bar');
const mobileBottomBar = document.getElementById('mobile-bottom-bar');
const outputBody      = document.getElementById('output-body');
const centerMain      = document.getElementById('center');
const btnRow          = document.getElementById('btn-row');
// Anchors for restoring to original positions
const viewerAnchor    = downloadBtn;   // viewer lives just before download-btn in output-body
function setMobileLayout(mob) {
  if (mob) {
    mobileTopBar.appendChild(viewer);
    mobileBottomBar.appendChild(btnRow);
  } else {
    outputBody.insertBefore(viewer, viewerAnchor);
    centerMain.appendChild(btnRow);
  }
}
const mqMobile = window.matchMedia('(max-width: 768px)');
setMobileLayout(mqMobile.matches);
mqMobile.addEventListener('change', e => setMobileLayout(e.matches));

// ── Fetch methods ──────────────────────────────────────────────────
async function loadMethods() {
  try {
    const [res, palRes] = await Promise.all([
      fetch('/api/methods'),
      fetch('/api/palettes'),
    ]);
    allMethods = await res.json();
    gPalettes  = await palRes.json();
    renderMethodList(allMethods);
  } catch (e) {
    methodList.innerHTML = `<p style="padding:16px 14px;color:var(--err)">Failed to load methods: ${e.message}</p>`;
  }
}

function renderMethodList(methods, isSearch = false) {
  if (!methods.length) {
    methodList.innerHTML = '<p style="padding:16px 14px;color:var(--muted)">No methods found.</p>';
    return;
  }

  // Group by category
  const groups = {};
  for (const m of methods) {
    (groups[m.category] = groups[m.category] || []).push(m);
  }

  const catState = getCatState();
  const mobile = isMobile();

  let html = '';
  for (const [cat, items] of Object.entries(groups).sort()) {
    // Search always expands; otherwise use stored state; default: all expanded
    const isOpen = isSearch ? true : (catState[cat] !== undefined ? catState[cat] : true);
    html += `<div class="category-group">
      <div class="category-header" data-cat="${escHtml(cat)}">
        <span class="cat-chevron">${isOpen ? '▼' : '▶'}</span>${escHtml(cat)}
      </div>
      <div class="category-items"${isOpen ? '' : ' style="display:none"'}>`;
    for (const m of items) {
      const tagHtml = m.tags.map(t => {
        const cls = t === 'gpu' ? 'tag tag-gpu' : 'tag';
        const txt = t === 'gpu' ? '⚡ GPU' : t;
        return `<span class="${cls}">${escHtml(txt)}</span>`;
      }).join('');
      html += `<div class="method-item" data-id="${escHtml(m.id)}">
        <div style="display:flex;gap:6px;align-items:baseline">
          <span class="mid">#${escHtml(m.id)}</span>
          <span class="mname">${escHtml(m.name)}</span>
        </div>
        <div class="tags">${tagHtml}</div>
      </div>`;
    }
    html += `</div></div>`;
  }
  methodList.innerHTML = html;

  // Method click handlers — auto-close sidebar on mobile after picking a method
  methodList.querySelectorAll('.method-item').forEach(el => {
    el.addEventListener('click', () => {
      selectMethod(el.dataset.id);
      if (isMobile()) sidebarEl.classList.remove('sidebar-open');
    });
  });

  // Category toggle handlers
  methodList.querySelectorAll('.category-header').forEach(el => {
    el.addEventListener('click', () => {
      const cat = el.dataset.cat;
      const itemsEl = el.nextElementSibling;
      const chevronEl = el.querySelector('.cat-chevron');
      const nowOpen = itemsEl.style.display === 'none';
      itemsEl.style.display = nowOpen ? '' : 'none';
      chevronEl.textContent = nowOpen ? '▼' : '▶';
      const state = getCatState();
      state[cat] = nowOpen;
      saveCatState(state);
    });
  });
}

// ── Search filter ──────────────────────────────────────────────────
searchInput.addEventListener('input', () => {
  const q = searchInput.value.trim().toLowerCase();
  if (!q) { renderMethodList(allMethods); return; }
  const filtered = allMethods.filter(m =>
    m.name.toLowerCase().includes(q) ||
    m.category.toLowerCase().includes(q) ||
    m.tags.some(t => t.toLowerCase().includes(q)) ||
    m.id.includes(q)
  );
  renderMethodList(filtered, true);
  // Re-highlight active
  if (selectedMethod) {
    const active = methodList.querySelector(`[data-id="${selectedMethod.id}"]`);
    if (active) active.classList.add('active');
  }
});

// ── Method selection ───────────────────────────────────────────────
function selectMethod(id) {
  selectedMethod = allMethods.find(m => m.id === id);
  if (!selectedMethod) return;

  // Highlight
  methodList.querySelectorAll('.method-item').forEach(el => el.classList.remove('active'));
  const active = methodList.querySelector(`[data-id="${id}"]`);
  if (active) active.classList.add('active');

  methodTitle.textContent = `#${selectedMethod.id} ${selectedMethod.name}`;
  methodMeta.textContent = `${selectedMethod.category}  ·  ${selectedMethod.tags.join(', ')}`;

  renderParams(selectedMethod);
  generateBtn.disabled = false;
}

// ── Param rendering ────────────────────────────────────────────────
function renderParams(method) {
  paramsEmpty.style.display = 'none';
  paramsForm.style.display = '';

  let html = '';

  // Seed
  html += paramNumber('seed', 'Seed', 'Random seed', 42, null, null, 1);

  // Method-specific params
  const paramEntries = Object.entries(method.params || {});
  if (paramEntries.length) {
    // Separate animation params from regular params
    const animKeys = new Set(['anim_mode', 'anim_speed', 'n_frames']);
    const animParams = [];
    const regularParams = [];
    for (const [key, spec] of paramEntries) {
      if (animKeys.has(key)) {
        animParams.push([key, spec]);
      } else {
        regularParams.push([key, spec]);
      }
    }
    // Render regular params first
    if (regularParams.length) {
      html += `<hr class="section-divider"><p class="section-label">Parameters</p>`;
      for (const [key, spec] of regularParams) {
        html += renderParamField(key, spec);
      }
    }
    // Render animation params under a dedicated heading
    if (animParams.length) {
      html += `<hr class="section-divider"><h2 class="section-label" style="font-size:14px;margin:8px 0 4px">Animation</h2>`;
      for (const [key, spec] of animParams) {
        html += renderParamField(key, spec);
      }
    }
  }

  // Animate toggle
  html += `<hr class="section-divider"><p class="section-label">Animation</p>`;
  html += `<div class="param-row">
    <div class="checkbox-row">
      <input type="checkbox" id="p_animate" class="param-ctrl">
      <label for="p_animate" style="font-size:13px;cursor:pointer">Animate</label>
    </div>
  </div>`;
  html += `<div id="anim-extra" style="display:none">
    ${paramNumber('fps', 'FPS', 'Frames per second', 24, 1, 60, 1)}
    ${paramNumber('duration', 'Duration (s)', 'Animation length in seconds', 3, 1, 30, 0.5)}
  </div>`;

  paramsForm.innerHTML = html;

  // Toggle animate extras
  document.getElementById('p_animate').addEventListener('change', e => {
    document.getElementById('anim-extra').style.display = e.target.checked ? '' : 'none';
  });

  // Wire up range sliders
  paramsForm.querySelectorAll('input[type=range]').forEach(range => {
    const valEl = paramsForm.querySelector(`#val_${range.id.replace('p_', '')}`);
    if (valEl) {
      range.addEventListener('input', () => {
        valEl.textContent = formatVal(range.value);
      });
    }
  });

  // Wire auto-generate on any param change (excluding animate checkbox)
  paramsForm.querySelectorAll('.param-ctrl').forEach(el => {
    if (el.id === 'p_animate') return;
    if (el.tagName === 'TEXTAREA') return; // GLSL editors use Apply button instead
    el.addEventListener('input', scheduleAutoGenerate);
    el.addEventListener('change', scheduleAutoGenerate);
  });

  // Wire GLSL editor Apply buttons in main params panel
  paramsForm.querySelectorAll('.glsl-apply-btn').forEach(btn => {
    const key = btn.id.replace('glsl-apply-', '');
    const ta  = document.getElementById(`p_${key}`);
    if (!ta) return;
    ta.addEventListener('keydown', e => {
      if (e.key === 'Tab') {
        e.preventDefault();
        const s = ta.selectionStart, end = ta.selectionEnd;
        ta.value = ta.value.slice(0, s) + '  ' + ta.value.slice(end);
        ta.selectionStart = ta.selectionEnd = s + 2;
      }
    });
    btn.addEventListener('click', () => generateBtn.click());
  });

  // Wire color picker swatch ↔ text field sync
  paramsForm.querySelectorAll('.color-swatch').forEach(swatch => {
    const textId = swatch.id.replace('_swatch', '');
    const textField = document.getElementById(textId);
    if (!textField) return;

    // Helper: sync swatch from text field value
    function syncSwatchFromText() {
      const hex = _parseColorToHex(textField.value);
      swatch.value = hex;
    }

    // Determine the expected format from the default value
    const defVal = textField.value;
    const isHexDefault = /^#/.test(defVal);
    const isFloatDefault = defVal.split(',').every(p => { const n = parseFloat(p.trim()); return !isNaN(n) && n <= 1; });

    // Initial sync: swatch reflects the text field's current value
    syncSwatchFromText();

    // Swatch → text field: convert hex to the expected format
    swatch.addEventListener('input', () => {
      const hex = swatch.value;
      if (isHexDefault) {
        textField.value = hex;
      } else {
        const r = parseInt(hex.slice(1,3), 16);
        const g = parseInt(hex.slice(3,5), 16);
        const b = parseInt(hex.slice(5,7), 16);
        if (isFloatDefault) {
          textField.value = `${(r/255).toFixed(3)},${(g/255).toFixed(3)},${(b/255).toFixed(3)}`;
        } else {
          textField.value = `${r},${g},${b}`;
        }
      }
    });
    // Text field → swatch: always update
    textField.addEventListener('input', syncSwatchFromText);
  });
}

function _isPaletteParam(key, spec) {
  if (spec.choices && spec.choices.length) return false;
  if (['palette', 'palette_name', 'colormap', 'color_palette'].includes(key)) return true;
  return /\bPALETTES\s+name\b/.test(spec.description || '');
}

function _isColorParam(key, spec) {
  if (spec.choices && spec.choices.length) return false;
  // Numbers and booleans are never color pickers regardless of name or description.
  if (typeof spec.default === 'number' || typeof spec.default === 'boolean') return false;
  const def = String(spec.default ?? '');
  const k = key.toLowerCase();
  const desc = (spec.description || '').toLowerCase();
  // Key directly names this as a color value (not merely related to color)
  if (k === 'color' || k.endsWith('_color') || k.startsWith('bg_') || k === 'edge_color') return true;
  // Default value already looks like a hex color
  if (_isHexColor(def)) return true;
  // Description explicitly describes a color format (not just "mentions color")
  if (desc.includes('hex color') || desc.includes('rgb tuple') || desc.includes('#rrggbb')) return true;
  return false;
}

function _isHexColor(s) {
  return /^#[0-9a-fA-F]{6}$/.test(s);
}

function _rgbToHex(r, g, b) {
  return '#' + [r,g,b].map(c => Math.max(0, Math.min(255, Math.round(c))).toString(16).padStart(2,'0')).join('');
}

function _parseColorToHex(s) {
  // Already hex
  if (_isHexColor(s)) return s;
  // RGB tuple string: "10,10,18" or "0.1,0.1,0.5"
  const parts = s.split(',').map(p => parseFloat(p.trim())).filter(n => !isNaN(n));
  if (parts.length >= 3) {
    if (parts[0] <= 1 && parts[1] <= 1 && parts[2] <= 1) {
      // Float [0,1] format
      return _rgbToHex(parts[0] * 255, parts[1] * 255, parts[2] * 255);
    }
    return _rgbToHex(parts[0], parts[1], parts[2]);
  }
  return '#000000';
}

function renderColorField(key, spec) {
  const def = spec.default || '#000000';
  const desc = spec.description || '';
  const hex = _parseColorToHex(String(def));
  return `<div class="param-row">
    <div class="param-label">
      <span class="param-name">${escHtml(key)}</span>
      <span class="param-desc">${escHtml(desc)}</span>
    </div>
    <div class="color-row">
      <input type="color" class="color-swatch param-ctrl" id="p_${escHtml(key)}_swatch" value="${hex}">
      <input type="text" class="param-input param-ctrl" id="p_${escHtml(key)}" value="${escHtml(String(def))}">
    </div>
  </div>`;
}

function renderParamField(key, spec) {
  const def = spec.default;
  const desc = spec.description || '';
  const hint = spec.hint || '';

  // Shared select renderer — defaults to current value; falls back to first option
  function makeSelect(optList) {
    const defStr = String(def ?? '');
    const hasMatch = optList.some(c => String(c) === defStr);
    const optHtml = optList.map((c, i) =>
      `<option value="${escHtml(c)}" ${(hasMatch ? String(c) === defStr : i === 0) ? 'selected' : ''}>${escHtml(c)}</option>`
    ).join('');
    return `<div class="param-row">
      <div class="param-label">
        <span class="param-name">${escHtml(key)}</span>
        <span class="param-desc">${escHtml(desc)}</span>
      </div>
      <select class="param-input param-ctrl" id="p_${escHtml(key)}">${optHtml}</select>
    </div>`;
  }

  // 0. Palette selector — any param named palette/palette_name/colormap/color_palette,
  //    or whose description contains the word "palette"
  if (gPalettes.length && _isPaletteParam(key, spec)) {
    const defStr = String(def ?? '');
    const opts = (defStr && !gPalettes.includes(defStr)) ? [defStr, ...gPalettes] : gPalettes;
    return makeSelect(opts);
  }

  // 0b. Color picker — any param whose key or description indicates a color value
  if (_isColorParam(key, spec)) {
    return renderColorField(key, spec);
  }

  // 1. Explicit choices array on the spec
  if (spec.choices && spec.choices.length) return makeSelect(spec.choices);

  // 2. Boolean
  if (typeof def === 'boolean') {
    return `<div class="param-row">
      <div class="param-label">
        <span class="param-name">${escHtml(key)}</span>
        <span class="param-desc">${escHtml(desc)}</span>
      </div>
      <div class="checkbox-row">
        <input type="checkbox" class="param-ctrl" id="p_${escHtml(key)}" ${def ? 'checked' : ''}>
        <label for="p_${escHtml(key)}" style="font-size:13px;cursor:pointer">${def ? 'Enabled' : 'Disabled'}</label>
      </div>
    </div>`;
  }

  // 3. Number — range slider or bare number input
  if (typeof def === 'number') {
    const isFloat = !Number.isInteger(def) || (spec.min !== undefined && !Number.isInteger(spec.min));
    return paramNumber(key, key, desc, def, spec.min, spec.max, isFloat ? 0.01 : 1);
  }

  // 4. Detect option list from description/hint text (backend injects choices for most;
  //    these patterns catch any stragglers not yet enriched server-side)
  const searchText = desc + ' ' + hint;
  // 4a. Parenthesized list: (word/word...) or (word, word...) — require ≥3 items
  const parenMatch = searchText.match(/\((\w[\w-]*(?:[,\/]\s*\w[\w-]*){2,})\)/);
  if (parenMatch) {
    return makeSelect(parenMatch[1].split(/[,\/]/).map(s => s.trim()).filter(Boolean));
  }
  // 4b. Bare slash run: word/word/word (no surrounding parens required)
  const slashMatch = searchText.match(/(\w[\w-]*(?:\/\w[\w-]*)+)/);
  if (slashMatch) {
    return makeSelect(slashMatch[1].split('/'));
  }
  // 4c. Colon + comma list: 'label: a, b, c' — require ≥3 items
  const colonMatch = searchText.match(/:\s*([\w-]+(?:,\s*[\w-]+){2,})\s*$/);
  if (colonMatch) {
    return makeSelect(colonMatch[1].split(',').map(s => s.trim()).filter(Boolean));
  }

  // 5a. GLSL code editor — multiline flag or glsl_code key
  if (spec.multiline || key === 'glsl_code') {
    const safeVal = escHtml(String(def ?? ''));
    return `<div class="param-row" id="pr_${escHtml(key)}">
      <div class="param-label">
        <span class="param-name">${escHtml(key)}</span>
        <span class="param-desc">${escHtml(desc)}</span>
      </div>
      <textarea class="glsl-editor param-ctrl" id="p_${escHtml(key)}" spellcheck="false">${safeVal}</textarea>
      <button class="glsl-apply-btn" id="glsl-apply-${escHtml(key)}">${key === 'sketch_code' ? 'Apply Sketch' : 'Apply Shader'}</button>
      <div class="glsl-error" id="glsl-err-${escHtml(key)}"></div>
    </div>`;
  }

  // 5. String fallback — free text
  return `<div class="param-row">
    <div class="param-label">
      <span class="param-name">${escHtml(key)}</span>
      <span class="param-desc">${escHtml(desc)}</span>
    </div>
    <input type="text" class="param-input param-ctrl" id="p_${escHtml(key)}" value="${escHtml(String(def ?? ''))}">
  </div>`;
}

function paramNumber(key, label, desc, defVal, minVal, maxVal, step) {
  const hasRange = minVal !== null && minVal !== undefined && maxVal !== null && maxVal !== undefined;
  if (hasRange) {
    return `<div class="param-row">
      <div class="param-label">
        <span class="param-name">${escHtml(label)}</span>
        <span class="param-desc">${escHtml(desc)}</span>
      </div>
      <div class="param-range-wrap">
        <input type="range" class="param-ctrl" id="p_${escHtml(key)}"
               min="${minVal}" max="${maxVal}" step="${step}" value="${defVal}">
        <span class="param-range-val" id="val_${escHtml(key)}">${formatVal(defVal)}</span>
      </div>
    </div>`;
  }
  return `<div class="param-row">
    <div class="param-label">
      <span class="param-name">${escHtml(label)}</span>
      <span class="param-desc">${escHtml(desc)}</span>
    </div>
    <input type="number" class="param-input param-ctrl" id="p_${escHtml(key)}"
           step="${step}" value="${defVal}">
  </div>`;
}

// ── Collect form values ────────────────────────────────────────────
function collectParams() {
  const result = {};
  if (!selectedMethod) return result;
  for (const [key, spec] of Object.entries(selectedMethod.params || {})) {
    const el = document.getElementById(`p_${key}`);
    if (!el) continue;
    result[key] = coerce(el, spec);
  }
  return result;
}

function coerce(el, spec) {
  const def = spec ? spec.default : null;
  if (el.type === 'checkbox') return el.checked;
  if (el.type === 'range' || el.type === 'number') {
    const v = parseFloat(el.value);
    if (Number.isInteger(def) && !el.step?.includes('.')) return Math.round(v);
    return v;
  }
  return el.value;
}

// ── Auto-generate ──────────────────────────────────────────────────
let autoMode = false;
let autoTimer = null;

autoBtn.addEventListener('click', () => {
  autoMode = !autoMode;
  autoBtn.classList.toggle('active', autoMode);
});

function scheduleAutoGenerate() {
  if (!autoMode || generateBtn.disabled) return;
  const animEl = document.getElementById('p_animate');
  if (animEl && animEl.checked) return;
  clearTimeout(autoTimer);
  autoTimer = setTimeout(() => {
    if (!autoMode || generateBtn.disabled) return;
    const animEl2 = document.getElementById('p_animate');
    if (animEl2 && animEl2.checked) return;
    generateBtn.click();
  }, 600);
}

// ── Running state ──────────────────────────────────────────────────
function setRunning(running) {
  generateBtn.style.display = running ? 'none' : '';
  stopBtn.style.display     = running ? ''     : 'none';
  generateBtn.disabled = running;
}

// ── Generate ───────────────────────────────────────────────────────
generateBtn.addEventListener('click', async () => {
  if (!selectedMethod) return;

  const seedEl = document.getElementById('p_seed');
  const animEl = document.getElementById('p_animate');
  const fpsEl  = document.getElementById('p_fps');
  const durEl  = document.getElementById('p_duration');

  const body = {
    method_id: selectedMethod.id,
    seed: seedEl ? parseInt(seedEl.value, 10) || 42 : 42,
    params: collectParams(),
    animate: animEl ? animEl.checked : false,
    fps:  fpsEl  ? parseInt(fpsEl.value, 10) || 24 : 24,
    duration: durEl ? parseFloat(durEl.value) || 3 : 3,
  };

  // Reset UI
  clearLog();
  setStatus('running');
  setRunning(true);
  viewer.style.display = 'none';
  resultImg.style.display = 'none';
  resultVideo.style.display = 'none';
  downloadBtn.style.display = 'none';
  startTimer();

  try {
    const res = await fetch('/api/generate', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
    });
    const { job_id } = await res.json();
    currentJobId = job_id;
    listenSSE(job_id);
  } catch (e) {
    appendLog(`Error: ${e.message}`, 'err');
    setStatus('error');
    setRunning(false);
    stopTimer();
  }
});

// ── Stop / cancel ──────────────────────────────────────────────────
stopBtn.addEventListener('click', async () => {
  if (!currentJobId) return;
  stopBtn.disabled = true;
  stopBtn.textContent = 'Stopping…';
  await fetch(`/api/jobs/${currentJobId}`, { method: 'DELETE' }).catch(() => {});
  // SSE will receive event: error / "Cancelled" and finish cleanup
});

// ── SSE listener ───────────────────────────────────────────────────
function listenSSE(jobId) {
  const es = new EventSource(`/api/jobs/${jobId}/stream`);

  es.addEventListener('progress', e => {
    const { message, elapsed } = JSON.parse(e.data);
    const cls = message.startsWith('  ✓') ? 'ok' : message.startsWith('  ✗') ? 'err' : '';
    appendLog(message, cls);
    elapsedTimer.textContent = `${elapsed}s`;
  });

  es.addEventListener('frame', e => {
    // Decode into an offscreen image first; swap visible src only when ready
    // to prevent blank flash between frames.
    const tmp = new Image();
    tmp.onload = () => {
      viewer.style.display = 'block';
      resultImg.style.display = '';
      resultVideo.style.display = 'none';
      resultImg.src = tmp.src;
    };
    tmp.src = 'data:image/jpeg;base64,' + e.data;
  });

  es.addEventListener('done', e => {
    const { output_path, type } = JSON.parse(e.data);
    es.close();
    stopTimer();
    setStatus('done');
    setRunning(false);
    stopBtn.disabled = false;
    stopBtn.textContent = 'Stop';
    showResult(jobId, output_path, type);
    appendLog('Done.', 'ok');
  });

  es.addEventListener('error', e => {
    es.close();
    stopTimer();
    setRunning(false);
    stopBtn.disabled = false;
    stopBtn.textContent = 'Stop';
    try {
      const { message } = JSON.parse(e.data);
      const wasCancelled = message === 'Cancelled';
      setStatus(wasCancelled ? 'idle' : 'error');
      appendLog(wasCancelled ? 'Cancelled.' : `Error: ${message}`,
                wasCancelled ? ''           : 'err');
    } catch {
      setStatus('error');
      appendLog('Connection error or job failed.', 'err');
    }
  });
}

// ── Result display ─────────────────────────────────────────────────
function showResult(jobId, outputPath, type) {
  const url = `/api/jobs/${jobId}/result`;
  viewer.style.display = 'block';
  downloadBtn.style.display = 'block';
  downloadBtn.href = url;
  downloadBtn.download = outputPath.split('/').pop();

  if (type === 'video') {
    resultVideo.src = url;
    resultVideo.style.display = '';
    resultImg.style.display = 'none';
  } else {
    // Decode offscreen; keep previous frame visible until final image is ready.
    const tmp = new Image();
    tmp.onload = () => {
      resultImg.src = tmp.src;
      resultImg.style.display = '';
      resultVideo.style.display = 'none';
    };
    tmp.src = url + '?t=' + Date.now();
  }
}

// ── Keyboard shortcuts ─────────────────────────────────────────────
document.addEventListener('keydown', e => {
  if (e.target.tagName === 'INPUT' || e.target.tagName === 'SELECT' || e.target.tagName === 'TEXTAREA') return;
  if (e.code === 'Space') {
    e.preventDefault();
    if (resultVideo.style.display !== 'none' && resultVideo.src) {
      if (resultVideo.paused) resultVideo.play();
      else resultVideo.pause();
    }
  }
});
function startTimer() {
  jobStartTime = Date.now();
  clearInterval(timerInterval);
  timerInterval = setInterval(() => {
    elapsedTimer.textContent = `${((Date.now() - jobStartTime) / 1000).toFixed(1)}s`;
  }, 200);
}
function stopTimer() { clearInterval(timerInterval); }

// ── Log helpers ────────────────────────────────────────────────────
function appendLog(msg, cls = '') {
  const line = document.createElement('div');
  line.className = 'log-line' + (cls ? ' ' + cls : '');
  line.textContent = msg;
  log.appendChild(line);
  log.scrollTop = log.scrollHeight;
}
function clearLog() { log.innerHTML = ''; elapsedTimer.textContent = '–'; }

// ── Status badge ───────────────────────────────────────────────────
function setStatus(state) {
  statusBadge.className = 'badge-' + state;
  statusBadge.textContent = state;
}

// ── Utils ──────────────────────────────────────────────────────────
function escHtml(s) {
  return String(s)
    .replace(/&/g,'&amp;').replace(/</g,'&lt;')
    .replace(/>/g,'&gt;').replace(/"/g,'&quot;');
}
function formatVal(v) {
  const n = parseFloat(v);
  return Number.isInteger(n) ? String(n) : n.toFixed(2);
}

// ── Resizable panes ────────────────────────────────────────────────
const sidebar      = document.getElementById('sidebar');
const outputPanel  = document.getElementById('output-panel');
const dividerLeft  = document.getElementById('divider-left');
const dividerRight = document.getElementById('divider-right');

const PANE_DEFAULTS = { sidebar: 260, output: 420 };
const PANE_MIN      = { sidebar: 150, output: 200 };
const PANE_MAX      = { sidebar: 600, output: 700 };

function loadPaneWidths() {
  try { return JSON.parse(localStorage.getItem('pane-widths') || '{}'); }
  catch { return {}; }
}

function savePaneWidths() {
  localStorage.setItem('pane-widths', JSON.stringify({
    sidebar: parseInt(sidebar.style.width, 10),
    output:  parseInt(outputPanel.style.width, 10),
  }));
}

function applyPaneWidths(saved) {
  sidebar.style.width     = (saved.sidebar || PANE_DEFAULTS.sidebar) + 'px';
  outputPanel.style.width = (saved.output  || PANE_DEFAULTS.output)  + 'px';
}

if (!isMobile()) applyPaneWidths(loadPaneWidths());

function attachDivider(divider, getWidth, setWidth, min, max) {
  divider.addEventListener('mousedown', e => {
    e.preventDefault();
    const startX = e.clientX;
    const startW = getWidth();
    divider.classList.add('dragging');
    document.body.style.cursor      = 'col-resize';
    document.body.style.userSelect  = 'none';

    function onMove(e) {
      const w = Math.max(min, Math.min(max, startW + (e.clientX - startX)));
      setWidth(w);
    }
    function onUp() {
      divider.classList.remove('dragging');
      document.body.style.cursor     = '';
      document.body.style.userSelect = '';
      savePaneWidths();
      document.removeEventListener('mousemove', onMove);
      document.removeEventListener('mouseup',   onUp);
    }
    document.addEventListener('mousemove', onMove);
    document.addEventListener('mouseup',   onUp);
  });
}

attachDivider(
  dividerLeft,
  () => parseInt(sidebar.style.width, 10),
  w  => sidebar.style.width = w + 'px',
  PANE_MIN.sidebar, PANE_MAX.sidebar
);
attachDivider(
  dividerRight,
  () => parseInt(outputPanel.style.width, 10),
  w  => outputPanel.style.width = w + 'px',
  PANE_MIN.output, PANE_MAX.output
);

// ── Init ───────────────────────────────────────────────────────────
loadMethods();

