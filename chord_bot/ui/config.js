// Chord Bot — shared constants (no dependencies)
'use strict';

export const BASE = (() => {
  const p = location.pathname;
  return p.endsWith('/') ? p : p.substring(0, p.lastIndexOf('/') + 1);
})();

export const FUNC_COLOR = {
  tonic: '#27ae60', subdominant: '#3498db', dominant: '#e74c3c', 'pre-dominant': '#f39c12',
};
export const FUNC_BG = {
  tonic: '#081a0e', subdominant: '#08111a', dominant: '#1a0808', 'pre-dominant': '#1a1208',
};
export const VOICE_COLORS = ['#3498db','#27ae60','#e67e22','#9b59b6','#e74c3c','#1abc9c','#f39c12'];
export const NOTE_NAMES   = ['C','C#','D','D#','E','F','F#','G','G#','A','A#','B'];
export const IS_BLACK     = [0,1,0,1,0,0,1,0,1,0,1,0];

export const PX_PER_BEAT       = 28;
export const PX_PER_BEAT_RESIZE = 28;
export const NODE_W_H   = 160;
export const NODE_H_H   = 64;
export const NODE_W_V   = 130;
export const NODE_H_V   = 54;
export const TIMELINE_Y = 140;
export const HISTORY_LIMIT = 50;
export const PIANO_KEY_W = 32;
export const NOTE_LO = 30;
export const NOTE_HI = 88;
