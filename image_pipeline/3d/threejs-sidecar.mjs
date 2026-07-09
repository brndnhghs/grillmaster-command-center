/**
 * threejs-sidecar.js — Headless three.js render server for the Grillmaster pipeline.
 *
 * Spawned alongside image_pipeline.server on an adjacent port (default 7862).
 * Receives graph JSON via HTTP, builds the three.js scene, renders it headlessly
 * with gl (ANGLE/Metal on M1), and returns raw PNG bytes.
 *
 * Node.js 22 + gl 8 + three.js r160 (vendored).
 *
 * API:
 *   POST /render  { nodes, edges, width, height, frame }
 *     → 200 image/png (raw bytes)
 *
 *   GET /health → 200 { ok: true, version: three_revision }
 */

import { createRequire } from 'module';
import http from 'http';
import * as THREE from '../../ui/vendor/three.module.js';

const require = createRequire(import.meta.url);
const makeContext = require('gl');

const PORT = parseInt(process.env.THREEJS_PORT || '7862', 10);

// ── Polyfill OffscreenCanvas for Node (three.js r160 uses it) ─────────────
globalThis.OffscreenCanvas = class {
  constructor(w, h) {
    this.width = w;
    this.height = h;
    this._gl = makeContext(w, h, { preserveDrawingBuffer: true });
    this._listeners = {};
    this.style = {};
  }
  addEventListener(type, fn) {
    if (!this._listeners[type]) this._listeners[type] = [];
    this._listeners[type].push(fn);
  }
  removeEventListener(type, fn) { /* noop */ }
  getContext(type, attrs) { return type === 'webgl2' ? null : this._gl; }
  get clientWidth() { return this.width; }
  get clientHeight() { return this.height; }
  getBoundingClientRect() { return { left: 0, top: 0, width: this.width, height: this.height }; }
};

// ── Scene builder ─────────────────────────────────────────────────────────

const SHAPE_MAP = {
  box:       THREE.BoxGeometry,
  sphere:    THREE.SphereGeometry,
  torus:     THREE.TorusGeometry,
  torusknot: THREE.TorusKnotGeometry,
  cone:      THREE.ConeGeometry,
  cylinder:  THREE.CylinderGeometry,
  icosahedron: THREE.IcosahedronGeometry,
  dodecahedron: THREE.DodecahedronGeometry,
  plane:     THREE.PlaneGeometry,
};

const LIGHT_TYPE_MAP = {
  point:      THREE.PointLight,
  directional: THREE.DirectionalLight,
  spot:       THREE.SpotLight,
};

/**
 * Build a three.js scene from a Grillmaster graph.
 * Returns { scene, camera } — the caller renders with its own renderer.
 */
function buildScene(nodes, edges, frame = 0) {
  const scene = new THREE.Scene();
  scene.background = new THREE.Color(0x0a0e18); // default, overridable

  // Collect outputs keyed by node id
  const outputs = {};

  // Topo sort: find leaf (Scene Render) first, then walk backwards
  const nodeMap = {};
  for (const n of nodes) nodeMap[n.id] = n;

  const edgeMap = {};
  for (const e of edges) {
    if (!edgeMap[e.target]) edgeMap[e.target] = {};
    edgeMap[e.target][e.targetPort] = { sourceId: e.source, sourcePort: e.sourcePort };
  }

  // Resolve a param with optional keyframe at given frame
  function resolveParam(node, paramName, defaultValue) {
    const p = node.params?.[paramName];
    if (p === undefined || p === null) return defaultValue;
    if (typeof p === 'object' && p !== null) {
      // Keyframed: pick the nearest keyframe <= frame
      const kfs = p.keyframes || [];
      if (kfs.length === 0) return p.default !== undefined ? p.default : defaultValue;
      let val = kfs[0].value;
      for (const kf of kfs) {
        if (kf.frame <= frame) val = kf.value;
        else break;
      }
      return val;
    }
    return p;
  }

  // Build inputs recursively
  function buildNode(node) {
    if (outputs[node.id]) return outputs[node.id];
    const mid = node.method_id;

    // Collect wired inputs
    const wired = {};
    const incoming = edgeMap[node.id] || {};
    for (const [port, conn] of Object.entries(incoming)) {
      const srcNode = nodeMap[conn.sourceId];
      if (srcNode) wired[port] = buildNode(srcNode);
    }

    let result = null;

    switch (mid) {

      case '__geometry__': {
        const shape = resolveParam(node, 'shape', 'torusknot');
        const size = resolveParam(node, 'size', 1);
        const detail = resolveParam(node, 'detail', 0.5);
        const seg = Math.round(12 + detail * 52);
        const Ctor = SHAPE_MAP[shape] || THREE.TorusKnotGeometry;
        result = new Ctor(size * 0.8, shape === 'torusknot' || shape === 'torus' ? size * 0.3 : size * 0.5, seg, Math.round(seg / 2));
        break;
      }

      case '__material__': {
        const color = resolveParam(node, 'color', '#4a9eff');
        const metalness = resolveParam(node, 'metalness', 0.4);
        const roughness = resolveParam(node, 'roughness', 0.35);
        const emissive = resolveParam(node, 'emissive', '#000000');
        const emissive_intensity = resolveParam(node, 'emissive_intensity', 1);
        const flat_shading = resolveParam(node, 'flat_shading', 0);
        result = new THREE.MeshStandardMaterial({
          color: new THREE.Color(color),
          metalness,
          roughness,
          emissive: new THREE.Color(emissive),
          emissiveIntensity: emissive_intensity,
          flatShading: !!flat_shading,
        });
        break;
      }

      case '__mesh3d__': {
        const geo = wired.geometry || new THREE.BoxGeometry(0.5, 0.5, 0.5);
        const mat = wired.material || new THREE.MeshStandardMaterial({ color: 0x4a9eff });
        const mesh = new THREE.Mesh(geo, mat);
        mesh.position.set(
          resolveParam(node, 'pos_x', 0),
          resolveParam(node, 'pos_y', 0),
          resolveParam(node, 'pos_z', 0)
        );
        const rx = resolveParam(node, 'rot_x', 0) * Math.PI / 180;
        const ry = resolveParam(node, 'rot_y', 0) * Math.PI / 180;
        const rz = resolveParam(node, 'rot_z', 0) * Math.PI / 180;
        mesh.rotation.set(rx, ry + resolveParam(node, 'spin_speed', 0) * frame / 60, rz);
        mesh.scale.setScalar(resolveParam(node, 'scale', 1));
        result = mesh;
        break;
      }

      case '__group3d__': {
        const group = new THREE.Group();
        // wired.object_a, wired.object_b may come from auto-inserted group
        // or from multiple OBJECT3D edges in the scene render
        for (const [port, obj] of Object.entries(wired)) {
          if (obj && typeof obj === 'object' && obj.isObject3D) {
            group.add(obj);
          }
        }
        result = group;
        break;
      }

      case '__light3d__': {
        const type = resolveParam(node, 'type', 'point');
        const intensity = resolveParam(node, 'intensity', 60);
        const color = resolveParam(node, 'color', '#ffffff');
        const LightCtor = LIGHT_TYPE_MAP[type] || THREE.PointLight;
        const light = new LightCtor(new THREE.Color(color), intensity);
        light.position.set(
          resolveParam(node, 'pos_x', 3),
          resolveParam(node, 'pos_y', 4),
          resolveParam(node, 'pos_z', 5)
        );
        result = light;
        break;
      }

      case '__camera3d__': {
        const camera = new THREE.PerspectiveCamera(
          resolveParam(node, 'fov', 50), 1, 0.1, 100
        );
        camera.position.set(
          resolveParam(node, 'pos_x', 0),
          resolveParam(node, 'pos_y', 0),
          resolveParam(node, 'pos_z', 4)
        );
        camera.lookAt(
          resolveParam(node, 'look_x', 0),
          resolveParam(node, 'look_y', 0),
          resolveParam(node, 'look_z', 0)
        );
        result = camera;
        break;
      }

      case '__scene_render__':
      case '__scene3d__': {
        // Assemble objects, lights, and camera
        let mainCamera = null;
        const ambient = resolveParam(node, 'ambient', 0.35);
        const bgColor = resolveParam(node, 'bg_color', '#0a0e18');
        scene.background = new THREE.Color(bgColor);

        // Add ambient light
        scene.add(new THREE.AmbientLight(0x404060, ambient));

        for (const [port, obj] of Object.entries(wired)) {
          if (!obj) continue;
          if (port === 'camera') {
            mainCamera = obj;
          } else if (port === 'object' || port === 'object_a' || port === 'object_b') {
            if (obj.isObject3D) scene.add(obj);
          } else if (port === 'light') {
            if (obj.isLight) scene.add(obj);
          }
        }

        if (!mainCamera) {
          mainCamera = new THREE.PerspectiveCamera(50, 1, 0.1, 100);
          mainCamera.position.set(0, 0, 4);
          mainCamera.lookAt(0, 0, 0);
        }
        result = { scene: this || scene, camera: mainCamera };
        break;
      }

      case '__gltf__': {
        // GLTFLoader is async — we can't load it in a sync build.
        // Return a placeholder mesh for now; async path TBD.
        const scale = resolveParam(node, 'scale', 1);
        const placeholder = new THREE.Mesh(
          new THREE.BoxGeometry(0.5, 0.5, 0.5),
          new THREE.MeshStandardMaterial({ color: 0x888888, wireframe: true })
        );
        placeholder.position.y = 0.25;
        placeholder.scale.setScalar(scale);
        const spin = resolveParam(node, 'spin_speed', 0);
        if (spin) placeholder.rotation.y = spin * frame / 60;
        result = placeholder;
        break;
      }

      default:
        // Unknown node — skip silently
        result = null;
    }

    outputs[node.id] = result;
    return result;
  }

  // Find the render terminal node
  const terminal = nodes.find(n =>
    n.method_id === '__scene_render__' || n.method_id === '__scene3d__'
  );

  if (!terminal) {
    // No 3D terminal — just return empty scene
    return { scene, camera: new THREE.PerspectiveCamera(50, 1, 0.1, 100) };
  }

  const terminalResult = buildNode(terminal);
  if (terminalResult && terminalResult.camera) {
    // The __scene_render__'s returned scene scene is already populated
    return terminalResult;
  }

  return { scene, camera: new THREE.PerspectiveCamera(50, 1, 0.1, 100) };
}

// ── Render function ────────────────────────────────────────────────────────

function renderSceneToPng(graphNodes, graphEdges, width, height, frame) {
  const canvas = new OffscreenCanvas(width || 512, height || 512);
  const renderer = new THREE.WebGLRenderer({
    canvas,
    antialias: true,
    preserveDrawingBuffer: true,
  });
  renderer.setSize(width || 512, height || 512);
  renderer.setPixelRatio(1);

  const { scene, camera } = buildScene(graphNodes, graphEdges, frame || 0);

  // Fix camera aspect ratio to match render size
  if (camera.aspect) {
    camera.aspect = (width || 512) / (height || 512);
    camera.updateProjectionMatrix();
  }

  renderer.render(scene, camera);

  // Read pixels
  const w = width || 512;
  const h = height || 512;
  const pixels = new Uint8Array(w * h * 4);
  const gl = canvas._gl;
  gl.readPixels(0, 0, w, h, gl.RGBA, gl.UNSIGNED_BYTE, pixels);

  // Flip Y (WebGL origin is bottom-left, PNG top-left)
  const flipped = new Uint8Array(w * h * 4);
  for (let y = 0; y < h; y++) {
    const srcOff = y * w * 4;
    const dstOff = (h - 1 - y) * w * 4;
    flipped.set(pixels.subarray(srcOff, srcOff + w * 4), dstOff);
  }

  return { pixels: flipped, width: w, height: h };
}

// ── HTTP Server ────────────────────────────────────────────────────────────

const server = http.createServer((req, res) => {
  // CORS
  res.setHeader('Access-Control-Allow-Origin', '*');
  res.setHeader('Access-Control-Allow-Methods', 'GET, POST, OPTIONS');
  res.setHeader('Access-Control-Allow-Headers', 'Content-Type');

  if (req.method === 'OPTIONS') {
    res.writeHead(204);
    res.end();
    return;
  }

  if (req.method === 'GET' && req.url === '/health') {
    res.writeHead(200, { 'Content-Type': 'application/json' });
    res.end(JSON.stringify({ ok: true, version: `three.js r${THREE.REVISION}`, port: PORT }));
    return;
  }

  if (req.method === 'POST' && req.url === '/render') {
    let body = '';
    req.on('data', chunk => body += chunk);
    req.on('end', async () => {
      try {
        const data = JSON.parse(body);
        const { nodes = [], edges = [], width = 512, height = 512, frame = 0 } = data;

        if (!nodes.length) {
          res.writeHead(400, { 'Content-Type': 'application/json' });
          res.end(JSON.stringify({ error: 'No nodes in graph' }));
          return;
        }

        const start = Date.now();
        const { pixels, width: w, height: h } = renderSceneToPng(nodes, edges, width, height, frame);
        const renderMs = Date.now() - start;

        // Convert raw RGBA to PNG using a minimal png encoder
        const png = encodePNG(pixels, w, h);

        res.writeHead(200, {
          'Content-Type': 'image/png',
          'X-Render-Ms': String(renderMs),
          'Content-Length': String(png.length),
        });
        res.end(png);
      } catch (e) {
        res.writeHead(500, { 'Content-Type': 'application/json' });
        res.end(JSON.stringify({ error: e.message, stack: e.stack?.substring(0, 300) }));
      }
    });
    return;
  }

  res.writeHead(404);
  res.end('Not found');
});

// ── Minimal PNG encoder (no dependencies beyond Node stdlib) ───────────────

function encodePNG(pixels, width, height) {
  // Raw RGBA pixels (flipped top-left), encode as PNG with zlib deflate
  const zlib = require('zlib');

  // Build IDAT rows: filter byte (0=no filter) + RGB bytes
  const raw = Buffer.alloc(height * (1 + width * 3));
  for (let y = 0; y < height; y++) {
    const rowOff = y * (1 + width * 3);
    raw[rowOff] = 0; // filter byte
    for (let x = 0; x < width; x++) {
      const pxOff = (y * width + x) * 4;
      const destOff = rowOff + 1 + x * 3;
      raw[destOff] = pixels[pxOff];       // R
      raw[destOff + 1] = pixels[pxOff + 1]; // G
      raw[destOff + 2] = pixels[pxOff + 2]; // B
    }
  }

  const deflated = zlib.deflateSync(raw, { level: 6 });

  // PNG signature
  const sig = Buffer.from([137, 80, 78, 71, 13, 10, 26, 10]);

  // IHDR chunk
  const ihdrData = Buffer.alloc(13);
  ihdrData.writeUInt32BE(width, 0);
  ihdrData.writeUInt32BE(height, 4);
  ihdrData[8] = 8;  // bit depth
  ihdrData[9] = 2;  // color type: RGB
  ihdrData[10] = 0; // compression
  ihdrData[11] = 0; // filter
  ihdrData[12] = 0; // interlace
  const ihdr = makeChunk('IHDR', ihdrData);

  // IDAT chunk
  const idat = makeChunk('IDAT', deflated);

  // IEND chunk
  const iend = makeChunk('IEND', Buffer.alloc(0));

  return Buffer.concat([sig, ihdr, idat, iend]);
}

function makeChunk(type, data) {
  const len = Buffer.alloc(4);
  len.writeUInt32BE(data.length, 0);
  const typeB = Buffer.from(type, 'ascii');
  const crcData = Buffer.concat([typeB, data]);
  const crc = crc32(crcData);
  const crcB = Buffer.alloc(4);
  crcB.writeUInt32BE(crc, 0);
  return Buffer.concat([len, typeB, data, crcB]);
}

function crc32(buf) {
  let crc = 0xFFFFFFFF;
  for (let i = 0; i < buf.length; i++) {
    crc ^= buf[i];
    for (let j = 0; j < 8; j++) {
      crc = (crc >>> 1) ^ (crc & 1 ? 0xEDB88320 : 0);
    }
  }
  return (crc ^ 0xFFFFFFFF) >>> 0;
}

// ── Start ──────────────────────────────────────────────────────────────────

server.listen(PORT, '127.0.0.1', () => {
  console.log(`three.js sidecar ready on http://127.0.0.1:${PORT}`);
  console.log(`three.js r${THREE.REVISION}, WebGL1 via gl (ANGLE/Metal)`);
});
