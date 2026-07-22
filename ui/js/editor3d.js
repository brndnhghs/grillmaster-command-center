/**
 * editor3d.js — Interactive 3D viewport editor for the node graph's 3D elements.
 *
 * WHAT THIS IS
 * ------------
 * A scene-editor view over the SAME graph the node editor edits. It builds an
 * editable three.js scene from the graph's 3D element nodes (__mesh3d__,
 * __gltf__, __usd__, __light3d__, __camera3d__), lets the user orbit around it
 * and drag transform gizmos (translate / rotate / scale), and writes every
 * gizmo edit straight back into the node params (pos_x/rot_y/scale, …) via the
 * host's onParamChange callback. The graph document stays the single source of
 * truth: slider edits in the param panel move objects here, gizmo drags move
 * sliders there, and the live render (client3d.js) picks the same values up.
 *
 * The editor deliberately shows ALL 3D element nodes — wired into a Scene
 * Render or not — because an editor is a stage, not a render. Rendering
 * fidelity (per-scene backgrounds, image pipelines) stays client3d.js's job.
 *
 * API: open({container, getGraph, onParamChange, onSelectNode}), close(),
 *      isOpen(), refresh(), setMode('translate'|'rotate'|'scale'),
 *      selectNode(nodeId), frameSelection().
 */

import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { TransformControls } from 'three/addons/controls/TransformControls.js';
import { loadUsdModel } from '/ui/js/client3d.js';

const DEG = Math.PI / 180;
function num(v, d) { const n = parseFloat(v); return Number.isFinite(n) ? n : d; }
function pv(node, key, d) { return num((node.params || {})[key], d); }
function hex(v, fallback) {
  const c = new THREE.Color();
  try { c.set(v || fallback); } catch { c.set(fallback); }
  return c;
}

// Node types the editor can place, and which params a gizmo drag writes.
const EDITABLE = {
  '__mesh3d__': { pos: true, rot: true, scale: true },
  '__gltf__':   { pos: true, rot: true, scale: true },
  '__usd__':    { pos: true, rot: true, scale: true },
  '__light3d__': { pos: true, rot: false, scale: false },
  '__camera3d__': { pos: true, rot: false, scale: false },
};

let E = null; // editor state, null when closed

// Resolve a CSS custom property to a colour three.js can consume. The GL side
// has no access to the cascade, so every themed scene colour goes through here
// and is re-read whenever the theme changes.
function themeColor(token, fallback) {
  const v = getComputedStyle(document.documentElement).getPropertyValue(token).trim();
  return v || fallback;
}

// Re-tint the live scene in place — cheaper and less disruptive than tearing
// the viewport down and rebuilding it just because a colour moved.
window.addEventListener('gm-theme-change', () => {
  if (!E) return;
  E.scene.background = new THREE.Color(themeColor('--viewport-bg', '#0d111c'));
  if (E.grid) {
    const c = new THREE.Color(themeColor('--viewport-grid', '#2e3950'));
    E.grid.material.color.set(c);
  }
});

// ── Geometry / material builders (mirror client3d semantics) ────────────────
function paramGeometry(shape, size, detail) {
  const s = size, d = detail;
  const seg = (lo, hi) => Math.round(lo + (hi - lo) * d);
  switch (shape) {
    case 'sphere':       return new THREE.SphereGeometry(0.75 * s, seg(8, 64), seg(6, 48));
    case 'torus':        return new THREE.TorusGeometry(0.55 * s, 0.22 * s, seg(8, 32), seg(16, 96));
    case 'torusknot':    return new THREE.TorusKnotGeometry(0.5 * s, 0.18 * s, seg(32, 200), seg(8, 32));
    case 'cone':         return new THREE.ConeGeometry(0.7 * s, 1.3 * s, seg(8, 64));
    case 'cylinder':     return new THREE.CylinderGeometry(0.6 * s, 0.6 * s, 1.3 * s, seg(8, 64));
    case 'icosahedron':  return new THREE.IcosahedronGeometry(0.85 * s, Math.round(d * 3));
    case 'dodecahedron': return new THREE.DodecahedronGeometry(0.85 * s, Math.round(d * 3));
    case 'plane':        return new THREE.PlaneGeometry(1.6 * s, 1.6 * s, seg(1, 32), seg(1, 32));
    case 'box':
    default:             return new THREE.BoxGeometry(s, s, s, seg(1, 8), seg(1, 8), seg(1, 8));
  }
}

function wiredSource(edges, nodesById, dstId, port) {
  const e = edges.find(x => !x.feedback && x.dst_node === dstId && x.dst_port === port);
  return e ? nodesById.get(e.src_node) : null;
}

// Lazy GLTFLoader (same vendored addon client3d uses).
let _gltfP = null;
function gltfLoader() {
  if (!_gltfP) _gltfP = import('three/addons/loaders/GLTFLoader.js').then(m => m.GLTFLoader);
  return _gltfP;
}

// ── Per-node editor object builders ─────────────────────────────────────────
function buildMeshEntry(node, nodesById, edges) {
  const geoNode = wiredSource(edges, nodesById, node.id, 'geometry');
  const matNode = wiredSource(edges, nodesById, node.id, 'material');
  const shape = geoNode ? String((geoNode.params || {}).shape ?? 'torusknot') : 'box';
  const gsize = geoNode ? pv(geoNode, 'size', 1) : 1;
  const gdet = geoNode ? pv(geoNode, 'detail', 0.5) : 0.5;
  const geoKey = `${shape}|${gsize}|${gdet}`;
  return {
    kind: 'mesh', geoKey,
    make() {
      const mesh = new THREE.Mesh(paramGeometry(shape, gsize, gdet), new THREE.MeshStandardMaterial());
      mesh.userData.nodeId = node.id;
      return mesh;
    },
    update(obj) {
      const m = obj.material;
      if (matNode) {
        m.color = hex((matNode.params || {}).color, '#4a9eff');
        m.metalness = pv(matNode, 'metalness', 0.4);
        m.roughness = pv(matNode, 'roughness', 0.35);
        m.emissive = hex((matNode.params || {}).emissive, '#000000');
        m.emissiveIntensity = pv(matNode, 'emissive_intensity', 1);
      } else {
        m.color = hex(null, '#4a9eff');
      }
    },
  };
}

function buildModelEntry(node, type) {
  const url = String((node.params || {}).url || '');
  return {
    kind: 'model', geoKey: `${type}|${url}`,
    make() {
      const holder = new THREE.Group();
      holder.userData.nodeId = node.id;
      // Placeholder while loading (or when no URL yet).
      const ph = new THREE.Mesh(
        new THREE.BoxGeometry(0.6, 0.6, 0.6),
        new THREE.MeshStandardMaterial({ color: 0x667799, wireframe: true }));
      holder.add(ph);
      if (url) {
        const done = (model) => {
          if (!E || !holder.parent) return;   // editor closed / node removed
          holder.remove(ph);
          holder.add(model);
        };
        const fail = (e) => console.warn(`[editor3d] ${type} load failed:`, e && e.message || e);
        if (type === '__usd__') {
          loadUsdModel(url).then(done).catch(fail);
        } else {
          gltfLoader().then(L => new Promise((res, rej) =>
            new L().load(url, g => res(g.scene || g.scenes?.[0]), undefined, rej)))
            .then(done).catch(fail);
        }
      }
      return holder;
    },
    update() {},
  };
}

function buildLightEntry(node) {
  const ltype = String((node.params || {}).type || 'point');
  return {
    kind: 'light', geoKey: `light|${ltype}`,
    make() {
      const holder = new THREE.Group();
      holder.userData.nodeId = node.id;
      const light = ltype === 'directional' ? new THREE.DirectionalLight()
        : ltype === 'spot' ? new THREE.SpotLight() : new THREE.PointLight();
      // Visible, pickable handle — lights have no surface to raycast.
      const handle = new THREE.Mesh(
        new THREE.SphereGeometry(0.14, 16, 12),
        new THREE.MeshBasicMaterial({ color: 0xffee88 }));
      holder.add(light); holder.add(handle);
      holder.userData.light = light; holder.userData.handle = handle;
      return holder;
    },
    update(obj) {
      const L = obj.userData.light;
      L.color = hex((node.params || {}).color, '#ffffff');
      L.intensity = pv(node, 'intensity', 60);
      obj.userData.handle.material.color.copy(L.color);
    },
  };
}

function buildCameraEntry(node) {
  return {
    kind: 'camera', geoKey: 'camera',
    make() {
      const holder = new THREE.Group();
      holder.userData.nodeId = node.id;
      // A small frustum-ish handle so the camera is visible & pickable.
      const body = new THREE.Mesh(
        new THREE.BoxGeometry(0.28, 0.2, 0.36),
        new THREE.MeshStandardMaterial({ color: 0x88aaff }));
      const lens = new THREE.Mesh(
        new THREE.ConeGeometry(0.14, 0.24, 18),
        new THREE.MeshStandardMaterial({ color: 0x3355cc }));
      lens.rotation.x = -Math.PI / 2;
      lens.position.z = -0.28;
      holder.add(body); holder.add(lens);
      return holder;
    },
    update(obj) {
      // Aim the handle at the camera's look-at target.
      obj.lookAt(pv(node, 'look_x', 0), pv(node, 'look_y', 0), pv(node, 'look_z', 0));
    },
  };
}

function entryFor(node, nodesById, edges) {
  switch (node.method_id) {
    case '__mesh3d__':   return buildMeshEntry(node, nodesById, edges);
    case '__gltf__':     return buildModelEntry(node, '__gltf__');
    case '__usd__':      return buildModelEntry(node, '__usd__');
    case '__light3d__':  return buildLightEntry(node);
    case '__camera3d__': return buildCameraEntry(node);
    default: return null;
  }
}

function applyTransformFromParams(node, obj) {
  const caps = EDITABLE[node.method_id];
  if (caps.pos) obj.position.set(pv(node, 'pos_x', defaultPos(node, 'x')),
                                 pv(node, 'pos_y', defaultPos(node, 'y')),
                                 pv(node, 'pos_z', defaultPos(node, 'z')));
  if (caps.rot) obj.rotation.set(pv(node, 'rot_x', 0) * DEG,
                                 pv(node, 'rot_y', 0) * DEG,
                                 pv(node, 'rot_z', 0) * DEG);
  if (caps.scale) { const s = pv(node, 'scale', 1); obj.scale.set(s, s, s); }
}

// Lights default at (3,4,5), cameras at (0,0,4) — mirror the node-def defaults.
function defaultPos(node, axis) {
  if (node.method_id === '__light3d__') return { x: 3, y: 4, z: 5 }[axis];
  if (node.method_id === '__camera3d__') return { x: 0, y: 0, z: 4 }[axis];
  return 0;
}

// ── Gizmo → params writeback ────────────────────────────────────────────────
function writeBack(node, obj) {
  const caps = EDITABLE[node.method_id];
  const r3 = v => Math.round(v * 1000) / 1000;
  const patch = {};
  if (caps.pos) {
    patch.pos_x = r3(obj.position.x); patch.pos_y = r3(obj.position.y); patch.pos_z = r3(obj.position.z);
  }
  if (caps.rot) {
    patch.rot_x = r3(obj.rotation.x / DEG); patch.rot_y = r3(obj.rotation.y / DEG); patch.rot_z = r3(obj.rotation.z / DEG);
  }
  if (caps.scale) {
    // Uniform-scale contract: take the dominant axis and re-uniform the object.
    const s = r3(Math.max(obj.scale.x, obj.scale.y, obj.scale.z));
    patch.scale = s;
    obj.scale.set(s, s, s);
  }
  E.opts.onParamChange(node.id, patch);
}

// ── Toolbar overlay ─────────────────────────────────────────────────────────
function makeToolbar(container) {
  const bar = document.createElement('div');
  // Themed via CSS custom properties so the viewport chrome follows the
  // active theme instead of staying frozen at one hardcoded blue.
  bar.style.cssText = 'position:absolute;top:10px;left:10px;z-index:6;display:flex;gap:6px;'
    + 'background:color-mix(in srgb, var(--bg1) 82%, transparent);'
    + 'padding:6px 8px;border-radius:var(--radius-l);backdrop-filter:blur(4px);'
    + 'border:1px solid var(--border);font-size:12px;align-items:center;';
  const mkBtn = (label, title, onClick) => {
    const b = document.createElement('button');
    b.textContent = label; b.title = title;
    b.style.cssText = 'padding:4px 10px;background:var(--bg3);border:1px solid var(--border-strong);'
      + 'border-radius:var(--radius-m);color:var(--muted);cursor:pointer;font-size:12px;font-weight:600;';
    b.addEventListener('click', onClick);
    bar.appendChild(b);
    return b;
  };
  const modes = {
    translate: mkBtn('Move', 'Translate (W)', () => setMode('translate')),
    rotate:    mkBtn('Rotate', 'Rotate (E)', () => setMode('rotate')),
    scale:     mkBtn('Scale', 'Scale (R)', () => setMode('scale')),
  };
  mkBtn('Frame', 'Frame selection (F)', () => frameSelection());
  const hint = document.createElement('span');
  hint.textContent = 'click: select · drag: orbit · ⇧drag: pan · wheel: zoom';
  hint.style.cssText = 'color:var(--muted2);margin-left:6px;';
  bar.appendChild(hint);
  container.appendChild(bar);
  return { bar, modes };
}

function highlightMode(mode) {
  if (!E) return;
  for (const [m, btn] of Object.entries(E.toolbar.modes)) {
    btn.style.background = m === mode ? 'var(--accent)' : 'var(--bg3)';
    btn.style.color = m === mode ? 'var(--bg0)' : 'var(--muted)';
    btn.style.borderColor = m === mode ? 'var(--accent)' : 'var(--border-strong)';
  }
}

// ── Public API ──────────────────────────────────────────────────────────────

export function isOpen() { return !!E; }

export async function open(opts) {
  close();
  const container = opts.container;
  const w = Math.max(64, container.clientWidth || 800);
  const h = Math.max(64, container.clientHeight || 500);

  const canvas = document.createElement('canvas');
  canvas.style.cssText = 'width:100%;height:100%;display:block;outline:none;';
  canvas.tabIndex = 0;
  container.appendChild(canvas);

  // preserveDrawingBuffer: canvas stays readable between frames (page
  // screenshots, toDataURL export) — same choice client3d.js makes.
  const renderer = new THREE.WebGLRenderer({ canvas, antialias: true, preserveDrawingBuffer: true });
  renderer.setPixelRatio(Math.min(2, window.devicePixelRatio || 1));
  renderer.setSize(w, h, false);

  const scene = new THREE.Scene();
  scene.background = new THREE.Color(themeColor('--viewport-bg', '#0d111c'));
  scene.add(new THREE.AmbientLight(0xffffff, 0.45));
  const key = new THREE.DirectionalLight(0xffffff, 1.2);
  key.position.set(5, 8, 6);
  scene.add(key);

  // GL colours can't reference CSS vars, so they are sampled from the computed
  // tokens here and re-sampled on gm-theme-change (see the listener below).
  const gridC = themeColor('--viewport-grid', '#2e3950');
  const grid = new THREE.GridHelper(20, 40, gridC, gridC);
  grid.material.opacity = 0.8;
  grid.material.transparent = true;
  scene.add(grid);
  const axes = new THREE.AxesHelper(1.2);
  scene.add(axes);

  const camera = new THREE.PerspectiveCamera(50, w / h, 0.01, 500);
  camera.position.set(4.5, 3.2, 5.5);

  const orbit = new OrbitControls(camera, canvas);
  orbit.enableDamping = true;
  orbit.dampingFactor = 0.08;

  const gizmo = new TransformControls(camera, canvas);
  gizmo.addEventListener('dragging-changed', e => {
    orbit.enabled = !e.value;
    if (E) E.dragging = e.value;
    if (!e.value && E && E.selected) {
      const node = E.nodeById.get(E.selected);
      const obj = E.objects.get(E.selected)?.obj;
      if (node && obj) writeBack(node, obj);   // final commit on release
    }
  });
  gizmo.addEventListener('objectChange', () => {
    if (!E || !E.selected) return;
    const node = E.nodeById.get(E.selected);
    const obj = E.objects.get(E.selected)?.obj;
    if (node && obj) writeBack(node, obj);     // live while dragging
  });
  // r185: TransformControls extends Controls, not Object3D — it is no longer
  // addable itself. Its visual root comes from getHelper(), and that helper is
  // what has to be removed on teardown.
  const gizmoHelper = gizmo.getHelper();
  scene.add(gizmoHelper);

  const raycaster = new THREE.Raycaster();

  E = {
    opts, container, canvas, renderer, scene, camera, orbit, gizmo, gizmoHelper, raycaster, grid,
    objects: new Map(),   // nodeId -> {obj, entryKey}
    nodeById: new Map(),
    selected: null, dragging: false, raf: 0,
    toolbar: makeToolbar(container),
    mode: 'translate',
  };
  highlightMode('translate');

  // ── Selection ──
  let downXY = null;
  canvas.addEventListener('pointerdown', e => { downXY = [e.clientX, e.clientY]; });
  canvas.addEventListener('pointerup', e => {
    if (!E || E.dragging || !downXY) return;
    const moved = Math.abs(e.clientX - downXY[0]) + Math.abs(e.clientY - downXY[1]);
    downXY = null;
    if (moved > 4) return;                     // orbit drag, not a click
    const rect = canvas.getBoundingClientRect();
    const ndc = new THREE.Vector2(
      ((e.clientX - rect.left) / rect.width) * 2 - 1,
      -((e.clientY - rect.top) / rect.height) * 2 + 1);
    E.raycaster.setFromCamera(ndc, E.camera);
    const pickables = [...E.objects.values()].map(o => o.obj);
    const hits = E.raycaster.intersectObjects(pickables, true);
    let nodeId = null;
    for (const hit of hits) {
      let o = hit.object;
      while (o && !o.userData.nodeId) o = o.parent;
      if (o) { nodeId = o.userData.nodeId; break; }
    }
    selectNode(nodeId);
    if (nodeId && E.opts.onSelectNode) E.opts.onSelectNode(nodeId);
  });

  // ── Keyboard ──
  E.onKey = (e) => {
    if (!E) return;
    if (e.key === 'w' || e.key === 'W') setMode('translate');
    else if (e.key === 'e' || e.key === 'E') setMode('rotate');
    else if (e.key === 'r' || e.key === 'R') setMode('scale');
    else if (e.key === 'f' || e.key === 'F') frameSelection();
    else if (e.key === 'Escape') selectNode(null);
  };
  canvas.addEventListener('keydown', E.onKey);

  // ── Resize ──
  E.ro = new ResizeObserver(() => {
    if (!E) return;
    const cw = Math.max(64, container.clientWidth), ch = Math.max(64, container.clientHeight);
    E.renderer.setSize(cw, ch, false);
    E.camera.aspect = cw / ch;
    E.camera.updateProjectionMatrix();
  });
  E.ro.observe(container);

  refresh();

  // ── Render loop ──
  const tick = () => {
    if (!E) return;
    E.orbit.update();
    // Param → transform sync each frame (cheap), except the object being dragged.
    for (const [nodeId, rec] of E.objects) {
      if (E.dragging && nodeId === E.selected) continue;
      const node = E.nodeById.get(nodeId);
      if (node) applyTransformFromParams(node, rec.obj);
    }
    E.renderer.render(E.scene, E.camera);
    E.raf = requestAnimationFrame(tick);
  };
  E.raf = requestAnimationFrame(tick);
  canvas.focus();
  return canvas;
}

/** Rebuild/update editor objects from the current graph (structure or params). */
export function refresh() {
  if (!E) return;
  const { nodes, edges } = E.opts.getGraph();
  E.nodeById = new Map(nodes.map(n => [n.id, n]));
  const editable = nodes.filter(n => EDITABLE[n.method_id]);
  const liveIds = new Set(editable.map(n => n.id));

  // Remove deleted nodes' objects.
  for (const [id, rec] of [...E.objects]) {
    if (!liveIds.has(id)) {
      if (E.selected === id) selectNode(null);
      E.scene.remove(rec.obj);
      E.objects.delete(id);
    }
  }

  // Add/rebuild changed objects.
  const byId = E.nodeById;
  for (const node of editable) {
    const entry = entryFor(node, byId, edges);
    if (!entry) continue;
    let rec = E.objects.get(node.id);
    if (!rec || rec.entryKey !== entry.geoKey) {
      if (rec) { E.scene.remove(rec.obj); if (E.selected === node.id) E.gizmo.detach(); }
      const obj = entry.make();
      rec = { obj, entryKey: entry.geoKey, entry };
      E.objects.set(node.id, rec);
      E.scene.add(obj);
      if (E.selected === node.id) E.gizmo.attach(obj);
    }
    rec.entry = entry;
    entry.update(rec.obj);
    if (!(E.dragging && E.selected === node.id)) applyTransformFromParams(node, rec.obj);
  }

  // Scene dressing from a Scene Render node, if present.
  const sceneNode = nodes.find(n => n.method_id === '__scene_render__' || n.method_id === '__scene3d__');
  if (sceneNode) E.scene.background = hex((sceneNode.params || {}).bg_color, '#0d111c');
}

export function selectNode(nodeId) {
  if (!E) return;
  E.selected = nodeId || null;
  if (nodeId && E.objects.has(nodeId)) {
    const node = E.nodeById.get(nodeId);
    E.gizmo.attach(E.objects.get(nodeId).obj);
    // Constrain the gizmo to what this node type can write back.
    const caps = EDITABLE[node.method_id] || {};
    if (E.mode === 'rotate' && !caps.rot) setMode('translate');
    if (E.mode === 'scale' && !caps.scale) setMode('translate');
  } else {
    E.gizmo.detach();
  }
}

export function setMode(mode) {
  if (!E) return;
  const caps = E.selected ? (EDITABLE[E.nodeById.get(E.selected)?.method_id] || {}) : {};
  if (mode === 'rotate' && E.selected && !caps.rot) return;
  if (mode === 'scale' && E.selected && !caps.scale) return;
  E.mode = mode;
  E.gizmo.setMode(mode);
  highlightMode(mode);
}

export function frameSelection() {
  if (!E) return;
  const targets = E.selected && E.objects.has(E.selected)
    ? [E.objects.get(E.selected).obj]
    : [...E.objects.values()].map(o => o.obj);
  if (!targets.length) return;
  const box = new THREE.Box3();
  for (const t of targets) box.expandByObject(t);
  if (box.isEmpty()) return;
  const center = box.getCenter(new THREE.Vector3());
  const size = box.getSize(new THREE.Vector3()).length() || 2;
  E.orbit.target.copy(center);
  const dir = E.camera.position.clone().sub(E.orbit.target).normalize();
  E.camera.position.copy(center.clone().add(dir.multiplyScalar(Math.max(2, size * 1.4))));
  E.orbit.update();
}

export function close() {
  if (!E) return;
  cancelAnimationFrame(E.raf);
  E.ro && E.ro.disconnect();
  E.gizmo.detach();
  if (E.gizmoHelper) E.scene.remove(E.gizmoHelper);
  E.gizmo.dispose && E.gizmo.dispose();
  E.orbit.dispose();
  for (const rec of E.objects.values()) E.scene.remove(rec.obj);
  E.renderer.dispose();
  E.toolbar.bar.remove();
  E.canvas.remove();
  E = null;
}
