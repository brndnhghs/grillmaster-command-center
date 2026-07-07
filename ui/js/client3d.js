/**
 * client3d.js — Client-side render executor spine (three.js).
 *
 * WHY THIS EXISTS
 * ---------------
 * The 2D pipeline renders server-side (Python/moderngl) and streams JPEG frames
 * to the browser. That is untouched. This module is an ADDITIVE, browser-GPU
 * render path used ONLY for graphs that contain a client-side node (currently
 * the "3D Scene" node). The browser is the render target for 3D — no per-frame
 * GPU→CPU→network round-trip.
 *
 * THE SPINE
 * ---------
 * A tiny pass-based graph executor. Each supported node type has a renderer
 * function registered in NODE_RENDERERS. Execution:
 *   1. topo-sort the client subgraph
 *   2. for each node, gather input textures from upstream edges
 *   3. run its renderer into a per-node WebGLRenderTarget
 *   4. blit the terminal node's target to the visible canvas
 *
 * Everything stays inside ONE THREE.WebGLRenderer / one WebGL2 context, so a
 * 3D scene texture flows straight into a GLSL filter pass with zero readback.
 * Adding another client GPU node later = add one renderer fn. This is the
 * shared foundation for a future full client-side WebGL preview.
 *
 * The GLSL filter renderer runs the EXISTING "Custom GLSL Shader" node's code
 * (the `glsl_code` param) as a fullscreen-quad pass — the same shader body the
 * user writes runs server-side (moderngl, #version 330) and here (WebGL2,
 * GLSL ES 3.00). Only the injected prologue differs.
 */

import * as THREE from '/ui/vendor/three.module.js';

export const CLIENT_NODE_IDS = ['__scene3d__', '__p5sketch__'];

// ── Lazy p5.js loader (UMD global, injected only when a p5 node is present) ──
let _p5LoadPromise = null;
function loadP5() {
  if (window.p5) return Promise.resolve(window.p5);
  if (_p5LoadPromise) return _p5LoadPromise;
  _p5LoadPromise = new Promise((resolve, reject) => {
    const s = document.createElement('script');
    s.src = '/ui/vendor/p5.min.js';
    s.async = true;
    s.onload = () => resolve(window.p5);
    s.onerror = () => { _p5LoadPromise = null; reject(new Error('failed to load p5.js')); };
    document.head.appendChild(s);
  });
  return _p5LoadPromise;
}

// ── Shader parity bundle (WebGL2 sources for existing GPU shader nodes) ──────
// Fetched once from /api/shader-sources; lets the browser render the server's
// GPU shader nodes client-side for the live preview, from the SAME GLSL source.
let _shaderBundle = null;
let _shaderBundlePromise = null;
function loadShaderBundle() {
  if (_shaderBundle) return Promise.resolve(_shaderBundle);
  if (_shaderBundlePromise) return _shaderBundlePromise;
  _shaderBundlePromise = fetch('/api/shader-sources')
    .then(r => r.json())
    .then(b => { _shaderBundle = b; return b; })
    .catch(e => { _shaderBundlePromise = null; throw e; });
  return _shaderBundlePromise;
}
/** True if the graph node is an existing GPU shader node the client can render. */
function _isGpuShaderNode(mid) {
  return !!(_shaderBundle && _shaderBundle.node_map && _shaderBundle.node_map[mid]);
}
/** Heuristic (pre-bundle): numeric GPU shader ids 173–219. */
function _looksLikeGpuShader(mid) {
  return /^\d+$/.test(mid) && +mid >= 173 && +mid <= 219;
}

/** Preload any heavy libs a graph needs before its first synchronous execute(). */
export async function prepare(nodes) {
  const jobs = [];
  if (nodes.some(n => n.method_id === '__p5sketch__')) jobs.push(loadP5());
  if (nodes.some(n => _looksLikeGpuShader(n.method_id))) jobs.push(loadShaderBundle());
  await Promise.all(jobs);
}

/** Can every node in this graph be rendered by the client spine? */
export function graphClientRenderable(nodes) {
  if (!nodes.length) return false;
  const CLIENT = new Set(['__scene3d__', '__p5sketch__', '__custom_shader__']);
  return nodes.every(n => CLIENT.has(n.method_id) ||
    _isGpuShaderNode(n.method_id) || _looksLikeGpuShader(n.method_id));
}

// ── Easing — a faithful port of server core/easing.py apply_easing ──────────
// Client and server MUST agree so keyframed params look identical whether a
// frame is rendered in-browser (3D path) or server-side. Guarded by the
// keyframe-parity test in image_pipeline/tests/test_client3d.py.
const _EASE_PRESETS = {
  'linear':      [0.0,  0.0, 1.0,  1.0],
  'ease':        [0.25, 0.1, 0.25, 1.0],
  'ease-in':     [0.42, 0.0, 1.0,  1.0],
  'ease-out':    [0.0,  0.0, 0.58, 1.0],
  'ease-in-out': [0.42, 0.0, 0.58, 1.0],
};
function _cubicBezier(t, p1x, p1y, p2x, p2y) {
  const cx = tt => 3 * (1 - tt) ** 2 * tt * p1x + 3 * (1 - tt) * tt * tt * p2x + tt ** 3;
  const cy = tt => 3 * (1 - tt) ** 2 * tt * p1y + 3 * (1 - tt) * tt * tt * p2y + tt ** 3;
  const dx = tt => 3 * (1 - tt) ** 2 * p1x + 6 * (1 - tt) * tt * (p2x - p1x) + 3 * tt * tt * (1 - p2x);
  let g = t;
  for (let i = 0; i < 8; i++) {
    const x = cx(g) - t;
    if (Math.abs(x) < 1e-7) break;
    const d = dx(g);
    if (Math.abs(d) < 1e-7) break;
    g = Math.min(1, Math.max(0, g - x / d));
  }
  return cy(g);
}
function _bounce(t) {
  if (t < 1 / 2.75)       return 7.5625 * t * t;
  else if (t < 2 / 2.75)  { t -= 1.5 / 2.75;  return 7.5625 * t * t + 0.75; }
  else if (t < 2.5 / 2.75){ t -= 2.25 / 2.75; return 7.5625 * t * t + 0.9375; }
  else                    { t -= 2.625 / 2.75; return 7.5625 * t * t + 0.984375; }
}
function _elastic(t) {
  if (t === 0 || t === 1) return t;
  return -Math.pow(2, 10 * (t - 1)) * Math.sin((t - 1 - 0.075) * (2 * Math.PI) / 0.3);
}
function easeApply(t, easing) {
  t = Math.min(1, Math.max(0, t));
  if (easing === 'step')    return t < 1 ? 0 : 1;
  if (easing === 'bounce')  return _bounce(t);
  if (easing === 'elastic') return _elastic(t);
  const p = _EASE_PRESETS[easing];
  if (p) return _cubicBezier(t, p[0], p[1], p[2], p[3]);
  return t; // linear fallback
}
function sampleTrack(kfs, frame) {
  if (!kfs || !kfs.length) return undefined;
  if (frame <= kfs[0].frame) return kfs[0].value;
  if (frame >= kfs[kfs.length - 1].frame) return kfs[kfs.length - 1].value;
  for (let i = 0; i < kfs.length - 1; i++) {
    const a = kfs[i], b = kfs[i + 1];
    if (a.frame <= frame && frame < b.frame) {
      const w = b.frame - a.frame;
      if (w <= 0) return b.value;
      const t = (frame - a.frame) / w;
      const te = easeApply(t, b.easing || 'linear');
      if (typeof a.value === 'number' && typeof b.value === 'number')
        return a.value + (b.value - a.value) * te;
      return te < 0.5 ? a.value : b.value;
    }
  }
  return undefined;
}
/** Merge static params with keyframe-sampled values at `frame`. */
export function animatedParams(node, frame) {
  const p = { ...(node.params || {}) };
  const pkf = node.paramKeyframes || {};
  for (const [k, kfs] of Object.entries(pkf)) {
    const v = sampleTrack(kfs, frame);
    if (v !== undefined) p[k] = v;
  }
  return p;
}

// ── Small helpers ───────────────────────────────────────────────────────────
function num(v, d) { const n = parseFloat(v); return Number.isFinite(n) ? n : d; }
function hexToColor(hex, fallback) {
  const c = new THREE.Color();
  try { c.set(hex || fallback); } catch { c.set(fallback); }
  return c;
}
const DEG = Math.PI / 180;

// ── GLSL prologue for client-side filter passes (GLSL ES 3.00) ──────────────
// Provides the same names the server prologue does: v_uv, u_resolution, u_time,
// u_params, u_texture, f_color, plus rot/hash21/noise/fbm. `glslVersion:GLSL3`
// makes three prepend `#version 300 es`, so we start at `precision`.
const FILTER_VERT = `
precision highp float;
in vec3 position;
out vec2 v_uv;
void main() {
  v_uv = position.xy * 0.5 + 0.5;
  gl_Position = vec4(position.xy, 0.0, 1.0);
}`;

const FILTER_FRAG_PROLOGUE = `
precision highp float;
in vec2 v_uv;
out vec4 f_color;
uniform vec2  u_resolution;
uniform float u_time;
uniform vec4  u_params;
uniform sampler2D u_texture;
mat2 rot(float a){ float c=cos(a), s=sin(a); return mat2(c,-s,s,c); }
float hash21(vec2 p){ p=fract(p*vec2(123.34,456.21)); p+=dot(p,p+45.32); return fract(p.x*p.y); }
float noise(vec2 p){ vec2 i=floor(p), f=fract(p);
  float a=hash21(i), b=hash21(i+vec2(1,0)), c=hash21(i+vec2(0,1)), d=hash21(i+vec2(1,1));
  vec2 u=f*f*(3.0-2.0*f); return mix(mix(a,b,u.x), mix(c,d,u.x), u.y); }
float fbm(vec2 p){ float v=0.0, a=0.5; for(int i=0;i<5;i++){ v+=a*noise(p); p*=2.0; a*=0.5; } return v; }
// ---- user shader body follows ----
`;

// ─────────────────────────────────────────────────────────────────────────────
// The executor
// ─────────────────────────────────────────────────────────────────────────────
class ClientExecutor {
  constructor(width, height) {
    this.width = width;
    this.height = height;
    this.canvas = document.createElement('canvas');
    this.canvas.width = width;
    this.canvas.height = height;
    this.renderer = new THREE.WebGLRenderer({
      canvas: this.canvas,
      antialias: true,
      preserveDrawingBuffer: true, // needed for readback / captureStream
    });
    this.renderer.setPixelRatio(1);
    this.renderer.setSize(width, height, false);

    // Per-node output render targets, keyed by node id.
    this._rts = new Map();
    // Filter material cache, keyed by GLSL source.
    this._filterMats = new Map();
    // Persistent 3D scene resources.
    this._three = null;
    // Fullscreen-quad rig shared by all filter/blit passes.
    this._quadScene = new THREE.Scene();
    this._quadCam = new THREE.OrthographicCamera(-1, 1, 1, -1, 0, 1);
    this._quadMesh = new THREE.Mesh(new THREE.PlaneGeometry(2, 2), null);
    this._quadScene.add(this._quadMesh);
    // 1x1 black fallback texture for filters with no wired input.
    const blackData = new Uint8Array([0, 0, 0, 255]);
    this._blackTex = new THREE.DataTexture(blackData, 1, 1, THREE.RGBAFormat);
    this._blackTex.needsUpdate = true;
    // GPU shader node materials (parity layer): webgl2 fragment -> RawShaderMaterial.
    this._gpuMats = new Map();
    // Temp RT for the two-pass convention bake (flip Y + swap R/B to match server).
    this._convRT = null;
    // Per-node p5 instances: id -> { code, inst, container, globals, tex, errBox }.
    this._p5 = new Map();
    // Per-node error strings (p5 compile/runtime, GLSL, …) surfaced to the UI.
    this._nodeErrors = {};
    // Reused scratch for reading a spine RT back into a 2D canvas (filter input).
    this._p5InputCanvas = null; this._p5InputCtx = null;
    this._p5ReadBuf = null; this._p5InputImg = null;
    // Passthrough blit material.
    this._blitMat = new THREE.RawShaderMaterial({
      glslVersion: THREE.GLSL3,
      uniforms: { u_texture: { value: this._blackTex } },
      vertexShader: FILTER_VERT,
      fragmentShader: `
precision highp float;
in vec2 v_uv;
out vec4 f_color;
uniform sampler2D u_texture;
void main(){ f_color = texture(u_texture, v_uv); }`,
    });
    // Convention blit: flip Y + swap R/B, so a client GPU-shader render matches
    // the server's authoritative output (render_shader reads FBO bottom-up as BGR).
    this._convMat = new THREE.RawShaderMaterial({
      glslVersion: THREE.GLSL3,
      uniforms: { u_texture: { value: this._blackTex } },
      vertexShader: FILTER_VERT,
      fragmentShader: `
precision highp float;
in vec2 v_uv;
out vec4 f_color;
uniform sampler2D u_texture;
void main(){ f_color = texture(u_texture, vec2(v_uv.x, 1.0 - v_uv.y)).bgra; }`,
    });
    this.lastCompileError = null;
  }

  resize(width, height) {
    if (width === this.width && height === this.height) return;
    this.width = width; this.height = height;
    this.canvas.width = width; this.canvas.height = height;
    this.renderer.setSize(width, height, false);
    for (const rt of this._rts.values()) rt.setSize(width, height);
  }

  _rtFor(id) {
    let rt = this._rts.get(id);
    if (!rt) {
      rt = new THREE.WebGLRenderTarget(this.width, this.height, {
        minFilter: THREE.LinearFilter,
        magFilter: THREE.LinearFilter,
        depthBuffer: true,
      });
      this._rts.set(id, rt);
    }
    return rt;
  }

  // ── 3D Scene node ──────────────────────────────────────────────────────────
  _ensureThree() {
    if (this._three) return this._three;
    const scene = new THREE.Scene();
    const camera = new THREE.PerspectiveCamera(50, this.width / this.height, 0.01, 100);
    const ambient = new THREE.AmbientLight(0xffffff, 0.3);
    const light = new THREE.PointLight(0xffffff, 1.0, 0, 0);
    scene.add(ambient);
    scene.add(light);
    const material = new THREE.MeshStandardMaterial({ color: 0x4a9eff, metalness: 0.3, roughness: 0.4 });
    const mesh = new THREE.Mesh(new THREE.BoxGeometry(1, 1, 1), material);
    scene.add(mesh);
    this._three = { scene, camera, ambient, light, mesh, material, geomType: 'box' };
    return this._three;
  }

  _geometryFor(type) {
    switch (type) {
      case 'sphere':      return new THREE.SphereGeometry(0.75, 48, 32);
      case 'torus':       return new THREE.TorusGeometry(0.55, 0.22, 24, 64);
      case 'cone':        return new THREE.ConeGeometry(0.7, 1.3, 48);
      case 'cylinder':    return new THREE.CylinderGeometry(0.6, 0.6, 1.3, 48);
      case 'icosahedron': return new THREE.IcosahedronGeometry(0.85, 0);
      case 'torusknot':   return new THREE.TorusKnotGeometry(0.5, 0.18, 128, 20);
      case 'box':
      default:            return new THREE.BoxGeometry(1, 1, 1);
    }
  }

  renderScene3D(node, params, time, targetRT) {
    const t = this._ensureThree();

    // Geometry (rebuild only when the type changes).
    const gtype = String(params.geometry || 'box');
    if (gtype !== t.geomType) {
      t.mesh.geometry.dispose();
      t.mesh.geometry = this._geometryFor(gtype);
      t.geomType = gtype;
    }

    // Camera.
    const fov = num(params.fov, 50);
    if (t.camera.fov !== fov) { t.camera.fov = fov; }
    t.camera.aspect = this.width / this.height;
    t.camera.position.set(num(params.cam_x, 0), num(params.cam_y, 0), num(params.cam_z, 4));
    t.camera.lookAt(0, 0, 0);
    t.camera.updateProjectionMatrix();

    // Object transform. spin_speed adds time-based Y rotation for easy live motion.
    const spin = num(params.spin_speed, 0) * time;
    t.mesh.rotation.set(
      num(params.obj_rx, 0) * DEG,
      num(params.obj_ry, 0) * DEG + spin,
      num(params.obj_rz, 0) * DEG,
    );
    const s = num(params.scale, 1);
    t.mesh.scale.set(s, s, s);

    // Material.
    t.material.color = hexToColor(params.mat_color, '#4a9eff');
    t.material.metalness = num(params.metalness, 0.3);
    t.material.roughness = num(params.roughness, 0.4);

    // Lights.
    t.light.position.set(num(params.light_x, 3), num(params.light_y, 4), num(params.light_z, 5));
    t.light.color = hexToColor(params.light_color, '#ffffff');
    t.ambient.intensity = num(params.ambient, 0.3);

    // Background.
    t.scene.background = hexToColor(params.bg_color, '#101014');

    this.renderer.setRenderTarget(targetRT);
    this.renderer.clear();
    this.renderer.render(t.scene, t.camera);
    this.renderer.setRenderTarget(null);
  }

  // ── GLSL filter node (runs Custom GLSL Shader code) ─────────────────────────
  _filterMaterial(glsl) {
    let mat = this._filterMats.get(glsl);
    if (mat) return mat;
    mat = new THREE.RawShaderMaterial({
      glslVersion: THREE.GLSL3,
      uniforms: {
        u_resolution: { value: new THREE.Vector2(this.width, this.height) },
        u_time: { value: 0 },
        u_params: { value: new THREE.Vector4(0.5, 0.5, 0.5, 0.5) },
        u_texture: { value: this._blackTex },
      },
      vertexShader: FILTER_VERT,
      fragmentShader: FILTER_FRAG_PROLOGUE + '\n' + glsl,
    });
    this._filterMats.set(glsl, mat);
    return mat;
  }

  renderGlslFilter(node, params, time, inputTex, targetRT) {
    const glsl = params.glsl_code || 'void main(){ f_color = texture(u_texture, v_uv); }';
    const mat = this._filterMaterial(glsl);
    mat.uniforms.u_resolution.value.set(this.width, this.height);
    mat.uniforms.u_time.value = time * num(params.time_scale, 1);
    mat.uniforms.u_params.value.set(
      num(params.p1, 0.5), num(params.p2, 0.5), num(params.p3, 0.5), num(params.p4, 0.5),
    );
    mat.uniforms.u_texture.value = inputTex || this._blackTex;

    this._quadMesh.material = mat;
    this.renderer.setRenderTarget(targetRT);
    this.renderer.clear();
    this.renderer.render(this._quadScene, this._quadCam);
    this.renderer.setRenderTarget(null);

    // Surface GLSL compile errors (WebGL logs them; three throws on use).
    const prog = this.renderer.info.programs?.find(p => p.cacheKey && p.diagnostics?.programLog);
    if (prog?.diagnostics?.programLog) this.lastCompileError = prog.diagnostics.programLog;
  }

  // ── Existing GPU shader node (parity layer) ─────────────────────────────────
  // Renders one of the server's GPU shader nodes client-side from its WebGL2
  // fragment. Two passes: (1) the parity fragment into a temp RT, (2) a
  // convention blit (flip Y + swap R/B) into the node RT so the client output
  // matches the server's authoritative render pixel-for-pixel.
  _gpuMaterial(fragment) {
    let mat = this._gpuMats.get(fragment);
    if (mat) return mat;
    const bundle = _shaderBundle;
    const vert = (bundle && bundle.vertex ? bundle.vertex : ('#version 300 es\n' + FILTER_VERT))
      .replace(/^#version 300 es\n?/, '');
    mat = new THREE.RawShaderMaterial({
      glslVersion: THREE.GLSL3,
      uniforms: {
        u_resolution: { value: new THREE.Vector2(this.width, this.height) },
        u_time: { value: 0 },
        u_params: { value: new THREE.Vector4(0.5, 0.5, 0.5, 0.5) },
        u_texture: { value: this._blackTex },
      },
      vertexShader: vert,
      fragmentShader: fragment.replace(/^#version 300 es\n?/, ''),
    });
    this._gpuMats.set(fragment, mat);
    return mat;
  }

  renderGpuShader(node, params, time, inputTex, targetRT) {
    const entry = _shaderBundle && _shaderBundle.node_map[node.method_id];
    const info = entry && _shaderBundle.shaders[entry.shader];
    if (!info) { this._clearRT(targetRT); return; }
    const mat = this._gpuMaterial(info.fragment);
    mat.uniforms.u_resolution.value.set(this.width, this.height);
    mat.uniforms.u_time.value = time * num(params.time_scale, 1);
    if (entry.type === 'filter') {
      // Server filter param mapping: u_params = (strength, p2, 0.5, 0.5).
      mat.uniforms.u_params.value.set(num(params.strength, 0.5), num(params.p2, 0.5), 0.5, 0.5);
      mat.uniforms.u_texture.value = inputTex || this._blackTex;
    } else {
      mat.uniforms.u_params.value.set(
        num(params.p1, 0.5), num(params.p2, 0.5), num(params.p3, 0.5), num(params.p4, 0.5));
      mat.uniforms.u_texture.value = inputTex || this._blackTex;
    }

    // Pass 1: parity fragment → temp RT.
    if (!this._convRT) this._convRT = new THREE.WebGLRenderTarget(this.width, this.height,
      { minFilter: THREE.LinearFilter, magFilter: THREE.LinearFilter });
    if (this._convRT.width !== this.width || this._convRT.height !== this.height)
      this._convRT.setSize(this.width, this.height);
    this._quadMesh.material = mat;
    this.renderer.setRenderTarget(this._convRT);
    this.renderer.clear();
    this.renderer.render(this._quadScene, this._quadCam);

    // Pass 2: convention blit (flip Y + swap R/B) → node RT.
    this._convMat.uniforms.u_texture.value = this._convRT.texture;
    this._quadMesh.material = this._convMat;
    this.renderer.setRenderTarget(targetRT);
    this.renderer.clear();
    this.renderer.render(this._quadScene, this._quadCam);
    this.renderer.setRenderTarget(null);

    const prog = this.renderer.info.programs?.find(p => p.diagnostics?.programLog);
    if (prog?.diagnostics?.programLog) this._nodeErrors[node.id] = prog.diagnostics.programLog;
  }

  // ── p5.js sketch node ───────────────────────────────────────────────────────
  _clearRT(rt) {
    this.renderer.setRenderTarget(rt);
    this.renderer.setClearColor(0x000000, 1);
    this.renderer.clear();
    this.renderer.setRenderTarget(null);
  }

  // Read a spine render target back into a 2D canvas the p5 sketch can consume
  // (filter mode). This is the ONE readback in the p5 path — the OUTPUT handoff
  // (CanvasTexture) stays readback-free. Generator sketches do no readback.
  _readRTToInputCanvas(rt) {
    const w = this.width, h = this.height;
    if (!this._p5InputCanvas) {
      this._p5InputCanvas = document.createElement('canvas');
      this._p5InputCtx = this._p5InputCanvas.getContext('2d');
    }
    if (this._p5InputCanvas.width !== w || this._p5InputCanvas.height !== h) {
      this._p5InputCanvas.width = w; this._p5InputCanvas.height = h;
      this._p5InputImg = null;
    }
    if (!this._p5ReadBuf || this._p5ReadBuf.length !== w * h * 4)
      this._p5ReadBuf = new Uint8Array(w * h * 4);
    this.renderer.readRenderTargetPixels(rt, 0, 0, w, h, this._p5ReadBuf);
    if (!this._p5InputImg) this._p5InputImg = this._p5InputCtx.createImageData(w, h);
    // WebGL readback is bottom-up; flip rows into the ImageData.
    const d = this._p5InputImg.data, rw = w * 4, buf = this._p5ReadBuf;
    for (let y = 0; y < h; y++) {
      const src = (h - 1 - y) * rw;
      d.set(buf.subarray(src, src + rw), y * rw);
    }
    this._p5InputCtx.putImageData(this._p5InputImg, 0, 0);
    return this._p5InputCanvas;
  }

  _buildP5(id, code) {
    const P5 = window.p5;
    const globals = {
      width: this.width, height: this.height, time: 0, frame: 0,
      p1: 0.5, p2: 0.5, p3: 0.5, p4: 0.5, input: null,
      WEBGL: P5.prototype.WEBGL, P2D: P5.prototype.P2D,
    };
    const container = document.createElement('div');
    container.style.cssText = 'position:absolute;left:-99999px;top:0;width:1px;height:1px;overflow:hidden;';
    document.body.appendChild(container);
    const errBox = { error: null };

    // Compile the user code into {setup, draw}. Function declarations named
    // `setup`/`draw` inside the body are hoisted locals we return by name, so
    // nothing leaks to window. They take (p, g) — see the default sketch.
    let resolved;
    try {
      const factory = new Function(
        '"use strict";\n' + code +
        '\n;return {setup:(typeof setup!=="undefined")?setup:null,' +
        ' draw:(typeof draw!=="undefined")?draw:null};');
      resolved = factory();
    } catch (e) {
      errBox.error = 'compile: ' + (e && e.message || e);
      return { code, inst: null, container, globals, tex: null, errBox };
    }

    const sketch = (p) => {
      p.setup = () => {
        try {
          p.pixelDensity(1);
          if (resolved.setup) resolved.setup(p, globals);
          if (!p.canvas) p.createCanvas(globals.width, globals.height);
          p.noLoop(); // we drive redraw() per spine frame for determinism
        } catch (e) { errBox.error = 'setup: ' + (e && e.message || e); try { p.noLoop(); } catch {} }
      };
      p.draw = () => {
        if (!resolved.draw) return;
        try { resolved.draw(p, globals); errBox.error = null; }
        catch (e) { errBox.error = 'draw: ' + (e && e.message || e) +
          (e && e.stack ? ' @ ' + e.stack.split('\n').slice(1, 3).join(' | ') : ''); }
      };
    };
    const inst = new P5(sketch, container);
    return { code, inst, container, globals, tex: null, errBox };
  }

  _destroyP5(id) {
    const st = this._p5.get(id);
    if (!st) return;
    try { st.inst && st.inst.remove(); } catch {}
    if (st.container && st.container.parentNode) st.container.parentNode.removeChild(st.container);
    if (st.tex) st.tex.dispose();
    this._p5.delete(id);
    delete this._nodeErrors[id];
  }

  renderP5(node, params, time, frame, inputCanvas, targetRT) {
    if (!window.p5) { this._clearRT(targetRT); return; } // still loading
    const code = String(params.sketch_code || '');
    let st = this._p5.get(node.id);
    if (!st || st.code !== code) {          // new node or edited code → rebuild
      if (st) this._destroyP5(node.id);
      st = this._buildP5(node.id, code);
      this._p5.set(node.id, st);
    }
    if (st.errBox.error || !st.inst) {
      this._nodeErrors[node.id] = st.errBox.error || 'p5 build failed';
      this._clearRT(targetRT);
      return;
    }

    // Feed animated globals to the sketch.
    const g = st.globals;
    const inst = st.inst;
    g.width = this.width; g.height = this.height;
    g.time = time * num(params.time_scale, 1);
    g.frame = frame;
    g.p1 = num(params.p1, 0.5); g.p2 = num(params.p2, 0.5);
    g.p3 = num(params.p3, 0.5); g.p4 = num(params.p4, 0.5);
    // Filter mode: hand the sketch a p5.Image (p5 texture()/image() need a
    // p5.Image, not a raw canvas). Built from the readback ImageData.
    if (inputCanvas && this._p5InputImg) {
      const w = this.width, h = this.height;
      if (!st.inputImg || st.inputImg.width !== w || st.inputImg.height !== h)
        st.inputImg = inst.createImage(w, h);
      st.inputImg.loadPixels();
      st.inputImg.pixels.set(this._p5InputImg.data);
      st.inputImg.updatePixels();
      g.input = st.inputImg;
    } else {
      g.input = null;
    }

    try {
      if (inst.width !== this.width || inst.height !== this.height)
        inst.resizeCanvas(this.width, this.height, true);
      inst.redraw(); // runs the user draw once
    } catch (e) {
      this._nodeErrors[node.id] = 'redraw: ' + (e && e.message || e) +
        (e && e.stack ? ' @ ' + e.stack.split('\n').slice(1, 4).join(' | ') : '');
      this._clearRT(targetRT); return;
    }
    if (st.errBox.error) { this._nodeErrors[node.id] = st.errBox.error; this._clearRT(targetRT); return; }
    delete this._nodeErrors[node.id];

    // Hand the p5 canvas to the spine as a texture (texImage2D upload — no
    // readPixels). CanvasTexture.flipY (default) yields upright orientation.
    if (!st.tex || st.tex.image !== inst.canvas) {
      if (st.tex) st.tex.dispose();
      st.tex = new THREE.CanvasTexture(inst.canvas);
    }
    st.tex.needsUpdate = true;
    this._blitMat.uniforms.u_texture.value = st.tex;
    this._quadMesh.material = this._blitMat;
    this.renderer.setRenderTarget(targetRT);
    this.renderer.clear();
    this.renderer.render(this._quadScene, this._quadCam);
    this.renderer.setRenderTarget(null);
  }

  // ── Graph execution ─────────────────────────────────────────────────────────
  _topoSort(nodes, edges) {
    const byId = new Map(nodes.map(n => [n.id, n]));
    const indeg = new Map(nodes.map(n => [n.id, 0]));
    const adj = new Map(nodes.map(n => [n.id, []]));
    for (const e of edges) {
      if (e.feedback) continue;
      if (!byId.has(e.src_node) || !byId.has(e.dst_node)) continue;
      adj.get(e.src_node).push(e.dst_node);
      indeg.set(e.dst_node, indeg.get(e.dst_node) + 1);
    }
    const q = [], out = [];
    for (const [id, d] of indeg) if (d === 0) q.push(id);
    while (q.length) {
      const id = q.shift();
      out.push(id);
      for (const nxt of adj.get(id)) {
        indeg.set(nxt, indeg.get(nxt) - 1);
        if (indeg.get(nxt) === 0) q.push(nxt);
      }
    }
    // Any leftover (cycles) appended in original order.
    for (const n of nodes) if (!out.includes(n.id)) out.push(n.id);
    return out.map(id => byId.get(id)).filter(Boolean);
  }

  _terminalId(nodes, edges) {
    const flagged = nodes.find(n => n.render);
    if (flagged) return flagged.id;
    const hasOut = new Set(edges.filter(e => !e.feedback).map(e => e.src_node));
    const sink = nodes.find(n => !hasOut.has(n.id));
    return sink ? sink.id : (nodes.length ? nodes[nodes.length - 1].id : null);
  }

  /**
   * Execute the graph at `frame`/`time` and blit the terminal to the canvas.
   * Returns the terminal node id (or null).
   */
  execute(nodes, edges, frame, time) {
    if (!nodes.length) return null;
    const order = this._topoSort(nodes, edges);
    const outputs = new Map(); // node id -> RenderTarget

    for (const node of order) {
      const params = animatedParams(node, frame);
      const rt = this._rtFor(node.id);
      if (node.method_id === '__scene3d__') {
        this.renderScene3D(node, params, time, rt);
      } else if (node.method_id === '__custom_shader__') {
        // First wired image input.
        const inEdge = edges.find(e => !e.feedback && e.dst_node === node.id &&
          (e.dst_port === 'image_in' || e.dst_port === 'image'));
        const inTex = inEdge && outputs.has(inEdge.src_node)
          ? outputs.get(inEdge.src_node).texture : null;
        this.renderGlslFilter(node, params, time, inTex, rt);
      } else if (node.method_id === '__p5sketch__') {
        // Filter mode: read the upstream RT into a 2D canvas the sketch reads
        // as g.input. Generator mode (no wired input): g.input stays null.
        const inEdge = edges.find(e => !e.feedback && e.dst_node === node.id &&
          (e.dst_port === 'image_in' || e.dst_port === 'image'));
        const inputCanvas = inEdge && outputs.has(inEdge.src_node)
          ? this._readRTToInputCanvas(outputs.get(inEdge.src_node)) : null;
        this.renderP5(node, params, time, frame, inputCanvas, rt);
      } else if (_isGpuShaderNode(node.method_id)) {
        // Existing GPU shader node rendered client-side via the parity layer.
        const inEdge = edges.find(e => !e.feedback && e.dst_node === node.id &&
          (e.dst_port === 'image_in' || e.dst_port === 'image'));
        const inTex = inEdge && outputs.has(inEdge.src_node)
          ? outputs.get(inEdge.src_node).texture : null;
        this.renderGpuShader(node, params, time, inTex, rt);
      } else {
        // Unsupported client node — passthrough its first input (or black).
        const inEdge = edges.find(e => !e.feedback && e.dst_node === node.id);
        const inTex = inEdge && outputs.has(inEdge.src_node)
          ? outputs.get(inEdge.src_node).texture : this._blackTex;
        this._blitMat.uniforms.u_texture.value = inTex;
        this._quadMesh.material = this._blitMat;
        this.renderer.setRenderTarget(rt);
        this.renderer.clear();
        this.renderer.render(this._quadScene, this._quadCam);
        this.renderer.setRenderTarget(null);
      }
      outputs.set(node.id, rt);
    }

    const termId = this._terminalId(nodes, edges);
    const termRT = termId && outputs.get(termId);
    if (termRT) {
      this._blitMat.uniforms.u_texture.value = termRT.texture;
      this._quadMesh.material = this._blitMat;
      this.renderer.setRenderTarget(null);
      this.renderer.clear();
      this.renderer.render(this._quadScene, this._quadCam);
    }

    // Garbage-collect p5 instances whose node was removed from the graph.
    if (this._p5.size) {
      const live = new Set(nodes.map(n => n.id));
      for (const id of [...this._p5.keys()]) if (!live.has(id)) this._destroyP5(id);
    }
    return termId;
  }

  /** Read the current canvas back as a PNG blob (for export / thumbnails). */
  async readbackPNG() {
    return await new Promise(res => this.canvas.toBlob(res, 'image/png'));
  }

  dispose() {
    for (const id of [...this._p5.keys()]) this._destroyP5(id);
    for (const rt of this._rts.values()) rt.dispose();
    this._rts.clear();
    for (const m of this._filterMats.values()) m.dispose();
    this._filterMats.clear();
    for (const m of this._gpuMats.values()) m.dispose();
    this._gpuMats.clear();
    if (this._convRT) { this._convRT.dispose(); this._convRT = null; }
    this._convMat.dispose();
    if (this._three) {
      this._three.mesh.geometry.dispose();
      this._three.material.dispose();
    }
    this._quadMesh.geometry.dispose();
    this._blitMat.dispose();
    this._blackTex.dispose();
    this.renderer.dispose();
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// Public controller — one live loop + one-shot render + export.
// ─────────────────────────────────────────────────────────────────────────────
let _executor = null;
let _rafId = null;
let _live = null; // { nodes, edges, start, end, fps, onStats }

function _ensureExecutor(width, height) {
  if (!_executor) _executor = new ClientExecutor(width, height);
  else _executor.resize(width, height);
  return _executor;
}

/** One-shot render of `frame` into the executor canvas. Returns the canvas. */
export async function renderFrame(nodes, edges, frame, width, height, timeSeconds) {
  await prepare(nodes);
  const ex = _ensureExecutor(width, height);
  const time = timeSeconds !== undefined ? timeSeconds : frame / 24;
  ex.execute(nodes, edges, frame, time);
  return ex.canvas;
}

/**
 * Start the client-side live loop. Advances frame start..end at `fps`, looping,
 * driving u_time from wall clock. `onStats({frame, fps})` fires ~4x/sec.
 * Returns the executor canvas (caller mounts it).
 */
export async function startLive({ nodes, edges, start, end, fps, width, height, onStats }) {
  stopLive();
  await prepare(nodes);
  const ex = _ensureExecutor(width, height);
  _live = { nodes, edges, start, end, fps: fps || 24, onStats };

  const t0 = performance.now();
  let frames = 0, lastStat = t0, statFrames = 0;

  const tick = () => {
    if (!_live) return;
    const now = performance.now();
    const elapsed = (now - t0) / 1000;
    const span = Math.max(1, _live.end - _live.start);
    const frame = _live.start + Math.floor(elapsed * _live.fps) % (span + 1);
    ex.execute(_live.nodes, _live.edges, frame, elapsed);
    frames++; statFrames++;

    if (now - lastStat >= 250) {
      const fpsNow = statFrames / ((now - lastStat) / 1000);
      if (_live.onStats) _live.onStats({ frame, fps: Math.round(fpsNow * 10) / 10 });
      lastStat = now; statFrames = 0;
    }
    _rafId = requestAnimationFrame(tick);
  };
  _rafId = requestAnimationFrame(tick);
  return ex.canvas;
}

/** Update the graph the live loop renders (params/edges changed) without restart. */
export function updateLiveGraph(nodes, edges) {
  if (_live) { _live.nodes = nodes; _live.edges = edges; prepare(nodes); }
}

export function stopLive() {
  if (_rafId) { cancelAnimationFrame(_rafId); _rafId = null; }
  _live = null;
}

export function isLive() { return !!_live; }

/**
 * Client-side export: deterministically render frames start..end and capture the
 * canvas to a WebM via MediaRecorder (frame-accurate via requestFrame). Resolves
 * to a Blob. `onProgress(i, total)` fires per frame. This does NOT touch the
 * server export path — 3D-containing graphs export entirely client-side.
 */
export async function exportWebM({ nodes, edges, start, end, fps, width, height, onProgress }) {
  stopLive();
  await prepare(nodes);
  const ex = _ensureExecutor(width, height);
  const total = end - start + 1;
  const stream = ex.canvas.captureStream(0);
  const track = stream.getVideoTracks()[0];
  const mime = MediaRecorder.isTypeSupported('video/webm;codecs=vp9')
    ? 'video/webm;codecs=vp9' : 'video/webm';
  const chunks = [];
  const rec = new MediaRecorder(stream, { mimeType: mime, videoBitsPerSecond: 12_000_000 });
  rec.ondataavailable = e => { if (e.data.size) chunks.push(e.data); };
  const done = new Promise(res => { rec.onstop = () => res(new Blob(chunks, { type: 'video/webm' })); });
  rec.start();

  const frameDurMs = 1000 / (fps || 24);
  for (let i = 0; i < total; i++) {
    const frame = start + i;
    ex.execute(nodes, edges, frame, frame / (fps || 24));
    if (track.requestFrame) track.requestFrame();
    if (onProgress) onProgress(i + 1, total);
    // Give the recorder time to sample each frame deterministically.
    await new Promise(r => setTimeout(r, Math.max(frameDurMs, 16)));
  }
  rec.stop();
  return await done;
}

/** Tear down all GPU resources. */
export function disposeAll() {
  stopLive();
  if (_executor) { _executor.dispose(); _executor = null; }
}

/** Expose the last GLSL compile error (or null). */
export function lastError() { return _executor ? _executor.lastCompileError : null; }

/** Per-node error strings (p5 compile/runtime, etc.), keyed by node id. */
export function getNodeErrors() { return _executor ? { ..._executor._nodeErrors } : {}; }
