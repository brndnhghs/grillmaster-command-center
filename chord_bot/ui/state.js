// Chord Bot — shared mutable state
// All modules import this single object and mutate its properties.
// ES modules guarantee they all share the same reference.
'use strict';

export const S = {
  // Graph data
  nodes: [],
  edges: [],
  sequence: [],

  // Selection / drawer
  selectedNode: null,
  drawerAugIdx: -1,

  // History
  _undoStack: [],
  _redoStack: [],

  // Node ID counter
  nodeCounter: 1,

  // Node type registry (loaded from /api/node-defs at init)
  nodeDefs: {},

  // Playback
  playhead: 0,
  isPlaying: false,
  lastRaf: null,
  playStartTime: null,
  playStartBeat: 0,
  totalBeats: 0,

  // Preview canvases
  previewTab: 'piano',
  pianoZoom: 1,
  tensionZoom: 1,
  pianoPinchDist: 0,
  tensionPinchDist: 0,
  previewResizing: false,
  previewResizeStartY: 0,
  previewResizeStartH: 0,

  // Audio
  audioCtx: null,
  audioSched: null,
};
