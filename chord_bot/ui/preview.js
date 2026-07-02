// Chord Bot — canvas preview (piano roll + tension graph) + resize/zoom events
'use strict';

import { S } from './state.js';
import {
  FUNC_COLOR, FUNC_BG, VOICE_COLORS, IS_BLACK,
  PIANO_KEY_W, NOTE_LO, NOTE_HI,
} from './config.js';

// ── Canvas refs ────────────────────────────────────────────────────────────────
const pianoCanvas   = document.getElementById('pianoCanvas');
const pCtx          = pianoCanvas.getContext('2d');
const tensionCanvas = document.getElementById('tensionCanvas');
const tCtx          = tensionCanvas.getContext('2d');

// ── Internal helpers ───────────────────────────────────────────────────────────
function noteToY(midi, hi, noteH) { return (hi - midi) * noteH; }

// ── Public: resize & render ────────────────────────────────────────────────────
export function resizePreviewCanvases() {
  const wrap = document.getElementById('previewCanvases');
  const bW = wrap.clientWidth, bH = wrap.clientHeight;
  pianoCanvas.width   = Math.max(bW, Math.round(bW * S.pianoZoom));
  pianoCanvas.height  = Math.max(bH, Math.round(bH * S.pianoZoom));
  tensionCanvas.width  = Math.max(bW, Math.round(bW * S.tensionZoom));
  tensionCanvas.height = Math.max(bH, Math.round(bH * S.tensionZoom));
  renderPreview();
}

export function renderPianoRoll() {
  const W = pianoCanvas.width, H = pianoCanvas.height;
  const nR = NOTE_HI - NOTE_LO + 1, nH = H / nR, wW = W - PIANO_KEY_W;
  const { sequence, totalBeats, playhead } = S;

  pCtx.fillStyle = '#0a0a10'; pCtx.fillRect(0, 0, W, H);

  // Background keys
  for (let m = NOTE_LO; m <= NOTE_HI; m++) {
    const pc = m % 12, y = noteToY(m, NOTE_HI, nH);
    pCtx.fillStyle = IS_BLACK[pc] ? '#0e0e18' : '#131320';
    pCtx.fillRect(PIANO_KEY_W, y, wW, nH);
    if (pc === 0) {
      pCtx.strokeStyle = '#2a2a44'; pCtx.lineWidth = 0.5;
      pCtx.beginPath(); pCtx.moveTo(PIANO_KEY_W, y); pCtx.lineTo(W, y); pCtx.stroke();
    }
  }

  if (totalBeats > 0) {
    // Beat grid
    pCtx.strokeStyle = '#1e1e34'; pCtx.lineWidth = 0.5;
    for (let b = 0; b <= totalBeats; b++) {
      const x = PIANO_KEY_W + (b / totalBeats) * wW;
      pCtx.beginPath(); pCtx.moveTo(x, 0); pCtx.lineTo(x, H); pCtx.stroke();
    }

    // Sequence entries
    for (let i = 0; i < sequence.length; i++) {
      const { state: s, start_beat: sb, end_beat: eb } = sequence[i];
      if (s.velocity === 0) continue;
      const sx = PIANO_KEY_W + (sb / totalBeats) * wW;
      const ex = PIANO_KEY_W + (eb / totalBeats) * wW;
      const bw = ex - sx;
      const fC = FUNC_COLOR[s.function] || '#9b59b6';

      pCtx.fillStyle = FUNC_BG[s.function] || '#0d0a1a'; pCtx.fillRect(sx, 0, bw, H);
      pCtx.fillStyle = fC + 'aa'; pCtx.fillRect(sx, 0, 1.5, H);

      (s.voices || []).forEach((m, vi) => {
        if (m < NOTE_LO || m > NOTE_HI) return;
        const ny = noteToY(m, NOTE_HI, nH);
        pCtx.fillStyle = VOICE_COLORS[vi % VOICE_COLORS.length] + 'cc';
        pCtx.fillRect(sx + 1, ny + 0.5, bw - 2, nH - 1);
      });

      const bs = s.bass_note;
      if (bs && bs >= NOTE_LO && bs <= NOTE_HI) {
        const by = noteToY(bs, NOTE_HI, nH);
        pCtx.fillStyle = '#ffffffaa';
        pCtx.fillRect(sx + 1, by + 0.5, bw - 2, Math.max(1.5, nH - 1));
      }

      // Voice-leading lines
      if (i < sequence.length - 1) {
        const nxt = sequence[i + 1];
        (s.voices || []).forEach((m1, vi) => {
          const m2 = (nxt.state.voices || [])[vi];
          if (!m2 || m1 < NOTE_LO || m1 > NOTE_HI || m2 < NOTE_LO || m2 > NOTE_HI) return;
          const y1 = noteToY(m1, NOTE_HI, nH) + nH / 2;
          const y2 = noteToY(m2, NOTE_HI, nH) + nH / 2;
          pCtx.save();
          pCtx.strokeStyle = VOICE_COLORS[vi % VOICE_COLORS.length] + '88'; pCtx.lineWidth = 1.5;
          pCtx.beginPath(); pCtx.moveTo(ex, y1); pCtx.lineTo(ex + Math.min(bw * 0.3, 20), y2); pCtx.stroke();
          pCtx.restore();
        });
      }

      const fs = Math.max(8, Math.min(11, bw / 5));
      pCtx.fillStyle = fC + 'ee'; pCtx.font = `bold ${fs}px 'SF Mono',monospace`;
      if (s.numeral) pCtx.fillText(s.numeral, sx + 3, 11);
      pCtx.fillStyle = '#aaaacc'; pCtx.font = `${Math.max(7, fs - 2)}px 'SF Mono',monospace`;
      pCtx.fillText(s.chord || '', sx + 3, 11 + fs);
    }
  }

  // Piano key column
  for (let m = NOTE_LO; m <= NOTE_HI; m++) {
    const pc = m % 12, y = noteToY(m, NOTE_HI, nH);
    pCtx.fillStyle = IS_BLACK[pc] ? '#111' : '#2a2a2a';
    pCtx.fillRect(0, y, PIANO_KEY_W - 1, nH);
    pCtx.strokeStyle = '#333'; pCtx.lineWidth = 0.5;
    pCtx.strokeRect(0, y, PIANO_KEY_W - 1, nH);
    if (pc === 0) {
      pCtx.fillStyle = '#666';
      pCtx.font = `${Math.max(7, Math.min(9, nH * 0.85))}px 'SF Mono',monospace`;
      pCtx.fillText(`C${Math.floor(m / 12) - 1}`, 2, y + nH - 1);
    }
  }

  // Playhead
  if (totalBeats > 0) {
    const px = PIANO_KEY_W + (playhead / totalBeats) * wW;
    pCtx.save(); pCtx.strokeStyle = 'rgba(255,80,80,.8)'; pCtx.lineWidth = 1.5;
    pCtx.beginPath(); pCtx.moveTo(px, 0); pCtx.lineTo(px, H); pCtx.stroke(); pCtx.restore();
  }
}

export function renderTension() {
  const W = tensionCanvas.width, H = tensionCanvas.height;
  const { sequence, totalBeats, playhead } = S;

  tCtx.fillStyle = '#0a0a10'; tCtx.fillRect(0, 0, W, H);

  if (sequence.length < 2 || !totalBeats) {
    tCtx.fillStyle = '#2a2a44'; tCtx.font = '11px SF Mono,monospace';
    tCtx.fillText('Execute graph to see tension', 20, H / 2);
    return;
  }

  // Horizontal grid lines
  [0, .25, .5, .75, 1].forEach(t => {
    const y = H - t * H;
    tCtx.strokeStyle = '#1e1e34'; tCtx.lineWidth = 0.5;
    tCtx.beginPath(); tCtx.moveTo(0, y); tCtx.lineTo(W, y); tCtx.stroke();
    tCtx.fillStyle = '#3a3a60'; tCtx.font = '9px SF Mono,monospace';
    tCtx.fillText(t.toFixed(2), 2, y - 2);
  });

  const pts = sequence.map(e => ({
    x: (e.start_beat / totalBeats) * W,
    y: H - e.state.tension * H,
  }));
  pts.push({ x: (sequence[sequence.length - 1].end_beat / totalBeats) * W, y: pts[pts.length - 1].y });

  // Filled area
  const grad = tCtx.createLinearGradient(0, 0, 0, H);
  grad.addColorStop(0, '#e74c3c88'); grad.addColorStop(1, '#e67e2222');
  tCtx.fillStyle = grad;
  tCtx.beginPath(); tCtx.moveTo(pts[0].x, H);
  pts.forEach(p => tCtx.lineTo(p.x, p.y));
  tCtx.lineTo(pts[pts.length - 1].x, H); tCtx.closePath(); tCtx.fill();

  // Line
  tCtx.strokeStyle = '#e74c3c'; tCtx.lineWidth = 2;
  tCtx.beginPath(); pts.forEach((p, i) => i ? tCtx.lineTo(p.x, p.y) : tCtx.moveTo(p.x, p.y));
  tCtx.stroke();

  // Dots
  sequence.forEach(e => {
    const x = (e.start_beat / totalBeats) * W, y = H - e.state.tension * H;
    tCtx.fillStyle = FUNC_COLOR[e.state.function] || '#9b59b6';
    tCtx.beginPath(); tCtx.arc(x, y, 4, 0, Math.PI * 2); tCtx.fill();
  });

  // Playhead
  const px = (playhead / totalBeats) * W;
  tCtx.strokeStyle = 'rgba(255,80,80,.8)'; tCtx.lineWidth = 1.5;
  tCtx.beginPath(); tCtx.moveTo(px, 0); tCtx.lineTo(px, H); tCtx.stroke();

  tCtx.fillStyle = '#5050a0'; tCtx.font = '9px SF Mono,monospace';
  tCtx.fillText('tension', 2, 11);
}

export function renderPreview() {
  if (S.previewTab === 'piano') renderPianoRoll(); else renderTension();
}

// ── Event wiring ───────────────────────────────────────────────────────────────
export function initPreviewEvents() {
  const previewEl     = document.getElementById('preview');
  const resizeHandle  = document.getElementById('previewResizeHandle');
  const wrap          = document.getElementById('previewCanvases');

  // Vertical resize handle
  resizeHandle.addEventListener('mousedown', e => {
    S.previewResizing = true;
    S.previewResizeStartY = e.clientY;
    S.previewResizeStartH = previewEl.clientHeight;
    resizeHandle.classList.add('active');
    e.preventDefault();
  });
  document.addEventListener('mousemove', e => {
    if (!S.previewResizing) return;
    previewEl.style.height =
      Math.max(80, Math.min(700, S.previewResizeStartH + (S.previewResizeStartY - e.clientY))) + 'px';
    resizePreviewCanvases();
  });
  document.addEventListener('mouseup', () => {
    if (S.previewResizing) { S.previewResizing = false; resizeHandle.classList.remove('active'); }
  });

  // Piano roll zoom
  pianoCanvas.addEventListener('wheel', e => {
    e.preventDefault();
    const f = e.deltaY < 0 ? 1.15 : 1 / 1.15, old = S.pianoZoom;
    S.pianoZoom = Math.max(0.5, Math.min(10, S.pianoZoom * f));
    const sc = S.pianoZoom / old;
    wrap.scrollLeft = (wrap.scrollLeft + e.offsetX) * sc - e.offsetX;
    wrap.scrollTop  = (wrap.scrollTop  + e.offsetY) * sc - e.offsetY;
    resizePreviewCanvases();
  }, { passive: false });
  pianoCanvas.addEventListener('touchstart', e => {
    if (e.touches.length === 2)
      S.pianoPinchDist = Math.hypot(e.touches[0].clientX - e.touches[1].clientX, e.touches[0].clientY - e.touches[1].clientY);
  }, { passive: true });
  pianoCanvas.addEventListener('touchmove', e => {
    if (e.touches.length !== 2) return;
    e.preventDefault();
    const d = Math.hypot(e.touches[0].clientX - e.touches[1].clientX, e.touches[0].clientY - e.touches[1].clientY);
    if (S.pianoPinchDist > 0) { S.pianoZoom = Math.max(0.5, Math.min(10, S.pianoZoom * d / S.pianoPinchDist)); resizePreviewCanvases(); }
    S.pianoPinchDist = d;
  }, { passive: false });
  pianoCanvas.addEventListener('touchend', () => { S.pianoPinchDist = 0; }, { passive: true });

  // Tension zoom
  tensionCanvas.addEventListener('wheel', e => {
    e.preventDefault();
    const f = e.deltaY < 0 ? 1.15 : 1 / 1.15, old = S.tensionZoom;
    S.tensionZoom = Math.max(0.5, Math.min(10, S.tensionZoom * f));
    const sc = S.tensionZoom / old;
    wrap.scrollLeft = (wrap.scrollLeft + e.offsetX) * sc - e.offsetX;
    wrap.scrollTop  = (wrap.scrollTop  + e.offsetY) * sc - e.offsetY;
    resizePreviewCanvases();
  }, { passive: false });
  tensionCanvas.addEventListener('touchstart', e => {
    if (e.touches.length === 2)
      S.tensionPinchDist = Math.hypot(e.touches[0].clientX - e.touches[1].clientX, e.touches[0].clientY - e.touches[1].clientY);
  }, { passive: true });
  tensionCanvas.addEventListener('touchmove', e => {
    if (e.touches.length !== 2) return;
    e.preventDefault();
    const d = Math.hypot(e.touches[0].clientX - e.touches[1].clientX, e.touches[0].clientY - e.touches[1].clientY);
    if (S.tensionPinchDist > 0) { S.tensionZoom = Math.max(0.5, Math.min(10, S.tensionZoom * d / S.tensionPinchDist)); resizePreviewCanvases(); }
    S.tensionPinchDist = d;
  }, { passive: false });
  tensionCanvas.addEventListener('touchend', () => { S.tensionPinchDist = 0; }, { passive: true });

  // Preview tab switching
  document.querySelectorAll('.tab').forEach(tab => {
    tab.addEventListener('click', () => {
      document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
      tab.classList.add('active');
      S.previewTab = tab.dataset.tab;
      pianoCanvas.classList.toggle('visible', S.previewTab === 'piano');
      tensionCanvas.classList.toggle('visible', S.previewTab === 'tension');
      renderPreview();
    });
  });
}
