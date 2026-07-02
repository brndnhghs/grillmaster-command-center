// Chord Bot — backend API calls (no DOM, no state mutations)
'use strict';

import { BASE } from './config.js';

export function serializeNode(n) {
  const b = { id: n.id, type: n.type, x: n.x, y: n.y, params: n.params, paramKeyframes: n.paramKeyframes || {} };
  if (n.type === 'subgraph') Object.assign(b, { subNodes: n.subNodes, subEdges: n.subEdges, label: n.label });
  return b;
}

/** Execute graph, returns list of sequence entry dicts. */
export async function apiExecute(nodeList, edgeList, tempo = 120) {
  const resp = await fetch(BASE + 'api/graph/execute', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ nodes: nodeList.map(serializeNode), edges: edgeList, tempo }),
  });
  if (!resp.ok) {
    const e = await resp.json().catch(() => ({}));
    throw new Error(e.detail || resp.statusText);
  }
  return resp.json();
}

/** Export graph as MIDI, returns a Blob. */
export async function apiExportMidi(nodeList, edgeList, tempo = 120) {
  const resp = await fetch(BASE + 'api/graph/export-midi', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ nodes: nodeList.map(serializeNode), edges: edgeList, tempo }),
  });
  if (!resp.ok) throw new Error('MIDI export failed');
  return resp.blob();
}

/** Fetch node type definitions from the registry. */
export async function apiNodeDefs() {
  const r = await fetch(BASE + 'api/node-defs');
  if (!r.ok) throw new Error('Failed to load node defs');
  return r.json();
}
