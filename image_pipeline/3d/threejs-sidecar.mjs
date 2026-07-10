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
 *
 * Quality notes (2026-07-09 upgrade):
 *   - Procedural PMREM studio environment gives metals/roughness something to
 *     reflect, so PBR materials no longer render near-black.
 *   - ACES Filmic tone mapping + sRGB output color space give a correct,
 *     filmic exposure curve instead of clipped linear values.
 *   - Optional shadow-casting key light + ground catcher for grounded renders.
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
    // three.js r160's WebGLRenderer.dispose() calls context.cancelAnimationFrame.
    // The `gl` package's context lacks request/cancelAnimationFrame, so add no-ops.
    if (!this._gl.cancelAnimationFrame) {
      this._gl.requestAnimationFrame = () => 0;
      this._gl.cancelAnimationFrame = () => {};
    }
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

// Keep a reference to the active WebGLRenderer so we can dispose between renders.
let _activeRenderer = null;

// Status of the procedural environment map for the most recent render.
let _resDebugEnv = 'n/a';

// ── Maps ─────────────────────────────────────────────────────────────────

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

const TONE_MAP_MAP = {
  none:      THREE.NoToneMapping,
  linear:    THREE.LinearToneMapping,
  reinhard:  THREE.ReinhardToneMapping,
  cineon:    THREE.CineonToneMapping,
  aces:      THREE.ACESFilmicToneMapping,
  agx:       THREE.AgXToneMapping,
  neutral:   THREE.NeutralToneMapping,
};

/**
 * Build a small procedural "studio" environment map and feed it through
 * PMREMGenerator. This gives PBR materials (especially metals) something to
 * reflect, which is what makes the scene read as lit instead of near-black.
 *
 * Uses only core three.js (no addons) so it works with the vendored r160 build.
 *
 * @param {THREE.WebGLRenderer} renderer
 * @param {'studio'|'warm'|'cool'|'none'} preset
 * @param {number} intensity multiplier on env intensity
 * @returns {THREE.Texture|null}
 */
function buildEnvironment(renderer, preset, intensity) {
  if (preset === 'none') return null;

  // A tiny scene: a dark room with a few emissive "softbox" planes.
  const envScene = new THREE.Scene();
  envScene.background = new THREE.Color(0x101218);

  // Base ambient gradient via hemisphere-ish fills.
  const base = new THREE.Mesh(
    new THREE.SphereGeometry(50, 24, 16),
    new THREE.MeshBasicMaterial({ side: THREE.BackSide, color: 0x0b0e14 })
  );
  envScene.add(base);

  const tone = preset === 'warm'
    ? { a: 0xffd9a0, b: 0xfff2e0, c: 0x3a2e22 }
    : preset === 'cool'
      ? { a: 0xbfd8ff, b: 0xe8f1ff, c: 0x223044 }
      : { a: 0xffffff, b: 0xdfe8ff, c: 0x2a3142 }; // studio

  // A few large soft emitter panels arranged around the sphere.
  const panelDefs = [
    { pos: [0, 18, 0],  size: [22, 10], color: tone.b, power: 3.0 },
    { pos: [-16, 6, 8], size: [12, 16], color: tone.a, power: 1.6 },
    { pos: [16, 2, -6], size: [12, 14], color: tone.b, power: 1.2 },
    { pos: [0, -8, 14], size: [16, 8],  color: tone.c, power: 0.7 },
  ];
  for (const p of panelDefs) {
    const mat = new THREE.MeshBasicMaterial({ color: new THREE.Color(p.color).multiplyScalar(p.power) });
    const panel = new THREE.Mesh(new THREE.PlaneGeometry(p.size[0], p.size[1]), mat);
    panel.position.set(...p.pos);
    panel.lookAt(0, 0, 0);
    envScene.add(panel);
  }

  const pmrem = new THREE.PMREMGenerator(renderer);
  pmrem.compileEquirectangularShader();
  let envRT;
  try {
    envRT = pmrem.fromScene(envScene, 0.04);
  } catch (e) {
    return null;
  }
  // NOTE: do NOT call pmrem.dispose() — on the headless `gl` backend it
  // invalidates the returned PMREM render target, leaving scene.environment a
  // dead (black) texture. The PMREM render target is small and short-lived;
  // it is GC'd with the renderer.
  return envRT ? envRT.texture : null;
}

// ── Scene builder ─────────────────────────────────────────────────────────

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
          envMapIntensity: resolveParam(node, 'env_intensity', 1.0),
        });
        break;
      }

      case '__mesh3d__': {
        const geo = wired.geometry || new THREE.BoxGeometry(0.5, 0.5, 0.5);
        const mat = wired.material || new THREE.MeshStandardMaterial({ color: 0x4a9eff });
        const mesh = new THREE.Mesh(geo, mat);
        mesh.castShadow = true;
        mesh.receiveShadow = true;
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
        if (type === 'directional' || type === 'spot') {
          light.castShadow = true;
          light.shadow.mapSize.set(1024, 1024);
          light.shadow.camera.near = 0.5;
          light.shadow.camera.far = 50;
          if (light.shadow.camera.left !== undefined) {
            light.shadow.camera.left = -8;
            light.shadow.camera.right = 8;
            light.shadow.camera.top = 8;
            light.shadow.camera.bottom = -8;
          }
          light.shadow.bias = -0.0005;
        }
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
        const bgMode = resolveParam(node, 'bg_mode', 'color');
        const exposure = resolveParam(node, 'exposure', 1.0);
        const toneMap = resolveParam(node, 'tone_map', 'aces');
        const envPreset = resolveParam(node, 'env_preset', 'studio');
        const envIntensity = resolveParam(node, 'env_intensity', 1.0);
        const shadows = resolveParam(node, 'shadows', 0);

        const bg = new THREE.Color(bgColor);
        if (bgMode === 'transparent') {
          scene.background = null;
        } else {
          scene.background = bg;
        }

        // Tone mapping + sRGB output color space (the big quality fix).
        const toneEnum = TONE_MAP_MAP[toneMap] ?? THREE.ACESFilmicToneMapping;
        scene._gmToneMapping = toneEnum;
        scene._gmExposure = exposure;

        // Procedural PMREM studio environment — gives PBR materials something
        // to reflect. Best-effort: if PMREM fails on the headless backend the
        // scene still renders via the explicit light rig below.
        if (_activeRenderer && envPreset !== 'none') {
          try {
            const envTex = buildEnvironment(_activeRenderer, envPreset, envIntensity);
            if (envTex) {
              scene.environment = envTex;
              scene.environmentIntensity = envIntensity;
              _resDebugEnv = `applied:${envPreset}@${envIntensity}`;
            } else {
              _resDebugEnv = `build-failed:${envPreset}`;
            }
          } catch (e) {
            _resDebugEnv = `error:${e.message?.substring(0, 60)}`;
          }
        } else {
          _resDebugEnv = 'disabled';
        }

        // Ambient light
        scene.add(new THREE.AmbientLight(0x404060, ambient));

        // Default 3-point light rig. A single point light leaves metals nearly
        // black (nothing to reflect / weak specular); the rig guarantees PBR
        // surfaces are reliably lit and read as three-dimensional. Users can
        // still wire their own __light3d__ nodes (added after this).
        const rig = resolveParam(node, 'lighting', 1.0);
        if (rig > 0) {
          const key = new THREE.DirectionalLight(0xffffff, 2.4 * rig);
          key.position.set(4, 6, 5);
          const fill = new THREE.DirectionalLight(0xbfd4ff, 0.9 * rig);
          fill.position.set(-6, 1, 3);
          const rim = new THREE.DirectionalLight(0xffe6c0, 1.3 * rig);
          rim.position.set(-2, 3, -6);
          if (shadows) {
            key.castShadow = true;
            key.shadow.mapSize.set(1024, 1024);
            key.shadow.camera.near = 0.5;
            key.shadow.camera.far = 40;
            key.shadow.camera.left = -8; key.shadow.camera.right = 8;
            key.shadow.camera.top = 8; key.shadow.camera.bottom = -8;
            key.shadow.bias = -0.0005;
          }
          scene.add(key, fill, rim);
        }

        // Optional ground shadow catcher.
        if (shadows) {
          const ground = new THREE.Mesh(
            new THREE.PlaneGeometry(40, 40),
            new THREE.ShadowMaterial({ opacity: 0.35 })
          );
          ground.rotation.x = -Math.PI / 2;
          ground.position.y = -1.6;
          ground.receiveShadow = true;
          scene.add(ground);
        }

        // Post-processing stack parameters (Route 3). These are read by the
        // render dispatcher; all neutral defaults ⇒ direct (unchanged) render.
        scene._gmPostFX = {
          bloom:           resolveParam(node, 'bloom', 0),
          bloom_threshold: resolveParam(node, 'bloom_threshold', 0.8),
          bloom_knee:      resolveParam(node, 'bloom_knee', 0.2),
          bloom_intensity: resolveParam(node, 'bloom_intensity', 0.6),
          bloom_radius:    resolveParam(node, 'bloom_radius', 1.0),
          bloom_passes:    resolveParam(node, 'bloom_passes', 4),
          brightness:      resolveParam(node, 'fx_brightness', 1.0),
          contrast:        resolveParam(node, 'fx_contrast', 1.0),
          saturation:      resolveParam(node, 'fx_saturation', 1.0),
          vignette:        resolveParam(node, 'vignette', 0),
          vignette_radius: resolveParam(node, 'vignette_radius', 0.85),
          vignette_softness: resolveParam(node, 'vignette_softness', 0.5),
          fxaa:            resolveParam(node, 'fxaa', 0),
        };

        for (const [port, obj] of Object.entries(wired)) {
          if (!obj) continue;
          if (port === 'camera') {
            mainCamera = obj;
          } else if (port === 'object' || port === 'object_a' || port === 'object_b') {
            if (obj.isObject3D) {
              scene.add(obj);
              obj.traverse(o => { if (o.isMesh) { o.castShadow = true; o.receiveShadow = true; } });
            }
          } else if (port === 'light') {
            if (obj.isLight) {
              obj.castShadow = shadows ? true : obj.castShadow;
              scene.add(obj);
            }
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

      case '__gltf__':
      case '__usd__': {
        // Model loaders are async — we can't load them in this sync build
        // path. Return a placeholder mesh; the browser client is the
        // authoritative renderer for model nodes.
        const scale = resolveParam(node, 'scale', 1);
        const placeholder = new THREE.Mesh(
          new THREE.BoxGeometry(0.5, 0.5, 0.5),
          new THREE.MeshStandardMaterial({ color: 0x888888, wireframe: true })
        );
        placeholder.position.set(
          resolveParam(node, 'pos_x', 0),
          resolveParam(node, 'pos_y', 0),
          resolveParam(node, 'pos_z', 0)
        );
        placeholder.scale.setScalar(scale);
        const d = Math.PI / 180;
        placeholder.rotation.set(
          resolveParam(node, 'rot_x', 0) * d,
          resolveParam(node, 'rot_y', 0) * d
            + resolveParam(node, 'spin_speed', 0) * frame / 60,
          resolveParam(node, 'rot_z', 0) * d
        );
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

// ── Post-processing (core three.js only — no addons vendored) ──────────────
//
// The headless `gl` backend exposes WebGL1, and the vendored three.module.js (r160)
// ships WITHOUT the postprocessing addons (EffectComposer/UnrealBloomPass/OutputPass),
// so we build the stack by hand with a fullscreen-quad pipeline rendered into
// RGBA8 render targets (no float-texture extension required):
//
//   scene ─▶ sceneRT (sRGB-encoded, tone-mapped)
//          ├─ bloom:  bright-pass ─▶ separable Gaussian blur (ping-pong)
//          └─ composite: scene + bloom*intensity + grade (brightness/contrast/
//                        saturation) + vignette  ─▶ compositeRT
//   compositeRT ─▶ FXAA pass ─▶ canvas
//
// Everything is gated behind the scene node's post-FX params. When none are
// engaged (all defaults) the renderer takes the byte-identical default direct
// path, so this is purely additive.

const _fsScene = new THREE.Scene();
const _fsCam = new THREE.OrthographicCamera(-1, 1, 1, -1, 0, 1);
const _fsGeo = new THREE.PlaneGeometry(2, 2);
const _fsMesh = new THREE.Mesh(_fsGeo, new THREE.MeshBasicMaterial());
_fsScene.add(_fsMesh);

const _FS_VS = `
varying vec2 vUv;
void main() {
  vUv = uv;
  gl_Position = vec4(position.xy, 0.0, 1.0);
}
`;

function _fsMat(frag, uniforms) {
  return new THREE.ShaderMaterial({
    vertexShader: _FS_VS,
    fragmentShader: frag,
    uniforms,
    depthTest: false,
    depthWrite: false,
  });
}

function _fsRender(renderer, target, material) {
  _fsMesh.material = material;
  renderer.setRenderTarget(target);
  renderer.render(_fsScene, _fsCam);
}

const _BLOOM_BRIGHT = `
uniform sampler2D tDiffuse;
uniform float threshold;
uniform float knee;
varying vec2 vUv;
void main() {
  vec4 c = texture2D(tDiffuse, vUv);
  float l = dot(c.rgb, vec3(0.2126, 0.7152, 0.0722));
  float w = smoothstep(threshold, threshold + max(knee, 0.0001), l);
  gl_FragColor = vec4(c.rgb * w, c.a);
}
`;

const _BLUR = `
uniform sampler2D tDiffuse;
uniform vec2 uDir;
uniform float uRadius;
varying vec2 vUv;
void main() {
  vec2 texel = uDir * uRadius;
  vec4 sum = texture2D(tDiffuse, vUv) * 0.227027;
  sum += texture2D(tDiffuse, vUv + texel * 1.3846) * 0.316216;
  sum += texture2D(tDiffuse, vUv - texel * 1.3846) * 0.316216;
  sum += texture2D(tDiffuse, vUv + texel * 3.2307) * 0.070270;
  sum += texture2D(tDiffuse, vUv - texel * 3.2307) * 0.070270;
  gl_FragColor = sum;
}
`;

const _COMPOSITE = `
uniform sampler2D tDiffuse;
uniform sampler2D tBloom;
uniform float hasBloom;
uniform float bloomIntensity;
uniform float brightness;
uniform float contrast;
uniform float saturation;
uniform float vignette;
uniform float vignette_radius;
uniform float vignette_softness;
varying vec2 vUv;
void main() {
  vec4 base = texture2D(tDiffuse, vUv);
  vec3 col = base.rgb;
  if (hasBloom > 0.5) {
    col += texture2D(tBloom, vUv).rgb * bloomIntensity;
  }
  col *= brightness;
  col = (col - 0.5) * contrast + 0.5;
  float l = dot(col, vec3(0.2126, 0.7152, 0.0722));
  col = mix(vec3(l), col, saturation);
  if (vignette > 0.0) {
    float dist = length(vUv - 0.5) * 1.41421356;
    float v = smoothstep(vignette_radius, vignette_radius - vignette_softness, dist);
    col *= mix(1.0, v, vignette);
  }
  gl_FragColor = vec4(clamp(col, 0.0, 1.0), base.a);
}
`;

// Compact FXAA 3.11-style edge anti-alias (works in WebGL1 GLSL).
const _FXAA = `
uniform sampler2D tDiffuse;
uniform vec2 uResolution;
varying vec2 vUv;
float luma(vec3 c) { return dot(c, vec3(0.299, 0.587, 0.114)); }
void main() {
  vec2 texel = 1.0 / uResolution;
  vec3 nw = texture2D(tDiffuse, vUv + texel * vec2(-1.0, -1.0)).rgb;
  vec3 ne = texture2D(tDiffuse, vUv + texel * vec2( 1.0, -1.0)).rgb;
  vec3 sw = texture2D(tDiffuse, vUv + texel * vec2(-1.0,  1.0)).rgb;
  vec3 se = texture2D(tDiffuse, vUv + texel * vec2( 1.0,  1.0)).rgb;
  vec3 m  = texture2D(tDiffuse, vUv).rgb;
  float lNW = luma(nw), lNE = luma(ne), lSW = luma(sw), lSE = luma(se), lM = luma(m);
  float lMin = min(lM, min(min(lNW, lNE), min(lSW, lSE)));
  float lMax = max(lM, max(max(lNW, lNE), max(lSW, lSE)));
  vec2 dir;
  dir.x = -((lNW + lNE) - (lSW + lSE));
  dir.y =  ((lNW + lSW) - (lNE + lSE));
  float dirReduce = max((lNW + lNE + lSW + lSE) * 0.03125, 0.0078125);
  float rcpDirMin = 1.0 / (min(abs(dir.x), abs(dir.y)) + dirReduce);
  dir = min(vec2(8.0), max(vec2(-8.0), dir * rcpDirMin)) * texel;
  vec3 rgbA = 0.5 * (
    texture2D(tDiffuse, vUv + dir * (1.0 / 3.0 - 0.5)).rgb +
    texture2D(tDiffuse, vUv + dir * (2.0 / 3.0 - 0.5)).rgb);
  vec3 rgbB = rgbA * 0.5 + 0.25 * (
    texture2D(tDiffuse, vUv + dir * -0.5).rgb +
    texture2D(tDiffuse, vUv + dir * 0.5).rgb);
  float lB = luma(rgbB);
  float a = texture2D(tDiffuse, vUv).a;
  if (lB < lMin || lB > lMax) gl_FragColor = vec4(rgbA, a);
  else gl_FragColor = vec4(rgbB, a);
}
`;

function renderWithPostFX(renderer, scene, camera, w, h, fx) {
  const rtOpts = {
    minFilter: THREE.LinearFilter,
    magFilter: THREE.LinearFilter,
    format: THREE.RGBAFormat,
    type: THREE.UnsignedByteType,
    stencilBuffer: false,
  };

  const sceneRT = new THREE.WebGLRenderTarget(w, h, { ...rtOpts, depthBuffer: true });
  // Store already tone-mapped + sRGB-encoded color (matches what the default
  // direct path writes to the canvas) so the post passes operate in display
  // space — no double tone-mapping / color-management surprises.
  sceneRT.texture.colorSpace = THREE.SRGBColorSpace;
  renderer.setRenderTarget(sceneRT);
  renderer.render(scene, camera);

  let bloomTex = null;
  let bloomRTs = null;
  if (fx.bloom > 0) {
    const a = new THREE.WebGLRenderTarget(w, h, { ...rtOpts, depthBuffer: false });
    const b = new THREE.WebGLRenderTarget(w, h, { ...rtOpts, depthBuffer: false });
    const bright = _fsMat(_BLOOM_BRIGHT, {
      tDiffuse: { value: sceneRT.texture },
      threshold: { value: fx.bloom_threshold },
      knee: { value: fx.bloom_knee },
    });
    _fsRender(renderer, a, bright);
    const blur = _fsMat(_BLUR, {
      tDiffuse: { value: null },
      uDir: { value: new THREE.Vector2() },
      uRadius: { value: fx.bloom_radius },
    });
    let src = a, dst = b;
    const passes = Math.max(1, Math.min(16, fx.bloom_passes | 0));
    for (let i = 0; i < passes; i++) {
      const horiz = (i % 2 === 0);
      blur.uniforms.tDiffuse.value = src.texture;
      blur.uniforms.uDir.value.set(horiz ? 1.0 / w : 0, horiz ? 0 : 1.0 / h);
      _fsRender(renderer, dst, blur);
      let t = src; src = dst; dst = t;
      blur.uniforms.tDiffuse.value = src.texture;
      blur.uniforms.uDir.value.set(horiz ? 0 : 1.0 / w, horiz ? 1.0 / h : 0);
      _fsRender(renderer, dst, blur);
      t = src; src = dst; dst = t;
    }
    bloomTex = src.texture;
    bloomRTs = [a, b];
  }

  const compositeTarget = fx.fxaa > 0
    ? new THREE.WebGLRenderTarget(w, h, { ...rtOpts, depthBuffer: false })
    : null;

  const comp = _fsMat(_COMPOSITE, {
    tDiffuse: { value: sceneRT.texture },
    tBloom: { value: bloomTex },
    hasBloom: { value: bloomTex ? 1 : 0 },
    bloomIntensity: { value: fx.bloom },
    brightness: { value: fx.brightness },
    contrast: { value: fx.contrast },
    saturation: { value: fx.saturation },
    vignette: { value: fx.vignette },
    vignette_radius: { value: fx.vignette_radius },
    vignette_softness: { value: fx.vignette_softness },
  });
  _fsRender(renderer, compositeTarget, comp);

  if (fx.fxaa > 0) {
    const fxaa = _fsMat(_FXAA, {
      tDiffuse: { value: compositeTarget.texture },
      uResolution: { value: new THREE.Vector2(w, h) },
    });
    _fsRender(renderer, null, fxaa);
  } else if (compositeTarget) {
    // blit composite RT → canvas (FXAA off but composite went through a RT)
    const blit = _fsMat(`uniform sampler2D tDiffuse; varying vec2 vUv;
      void main(){ gl_FragColor = texture2D(tDiffuse, vUv); }`,
      { tDiffuse: { value: compositeTarget.texture } });
    _fsRender(renderer, null, blit);
  }

  sceneRT.dispose();
  if (bloomRTs) bloomRTs.forEach(rt => rt.dispose());
  if (compositeTarget) compositeTarget.dispose();
  renderer.setRenderTarget(null);
}

// ── Render function ─────────────────────────────────────────────────────────

function renderSceneToPng(graphNodes, graphEdges, width, height, frame) {
  const w = width || 512;
  const h = height || 512;
  _resDebugEnv = 'n/a';

  const canvas = new OffscreenCanvas(w, h);
  const renderer = new THREE.WebGLRenderer({
    canvas,
    antialias: true,
    preserveDrawingBuffer: true,
    alpha: true,
  });
  renderer.setSize(w, h);
  renderer.setPixelRatio(1);
  renderer.outputColorSpace = THREE.SRGBColorSpace;
  renderer.shadowMap.enabled = true;
  renderer.shadowMap.type = THREE.PCFSoftShadowMap;
  _activeRenderer = renderer;

  const { scene, camera } = buildScene(graphNodes, graphEdges, frame || 0);
  if (scene._gmToneMapping !== undefined) {
    renderer.toneMapping = scene._gmToneMapping;
    renderer.toneMappingExposure = scene._gmExposure ?? 1.0;
  } else {
    renderer.toneMapping = THREE.ACESFilmicToneMapping;
    renderer.toneMappingExposure = 1.0;
  }

  // Fix camera aspect ratio to match render size
  if (camera.aspect) {
    camera.aspect = w / h;
    camera.updateProjectionMatrix();
  }

  // Post-processing stack (Route 3 improvement). The default path is taken
  // when every post-FX param is at its neutral default — that branch is the
  // unchanged direct render, so this feature is purely additive. Any
  // non-default value engages the render-target pipeline.
  const fx = scene._gmPostFX || {};
  const fxEngaged = !!(fx.bloom || fx.vignette || fx.fxaa ||
      fx.brightness !== 1 || fx.contrast !== 1 || fx.saturation !== 1);

  if (fxEngaged) {
    renderWithPostFX(renderer, scene, camera, w, h, fx);
  } else {
    renderer.render(scene, camera);
  }

  // Read pixels
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

  // NOTE: we intentionally do NOT call renderer.dispose() — three.js r160's
  // dispose() invokes context.cancelAnimationFrame, but the headless `gl`
  // context exposes no animation-frame methods (and no-opping them doesn't
  // help because the renderer's internal context ref is null here). The
  // process is short-lived per request, so the per-request context is GC'd.

  return { pixels: flipped, width: w, height: h, envApplied: _resDebugEnv };
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
        const { pixels, width: w, height: h, envApplied } = renderSceneToPng(nodes, edges, width, height, frame);
        const renderMs = Date.now() - start;

        // Convert raw RGBA to PNG using a minimal png encoder
        const png = encodePNG(pixels, w, h);

        res.writeHead(200, {
          'Content-Type': 'image/png',
          'X-Render-Ms': String(renderMs),
          'X-Env-Applied': String(envApplied),
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
  // Raw RGBA pixels (flipped top-left), encode as PNG with zlib deflate.
  // NOTE: we write a true RGBA PNG (color type 6) — not just RGB — so the
  // alpha channel the renderer captured (e.g. for bg_mode:'transparent') is
  // preserved in the exported file instead of being silently dropped. The
  // renderer is created with alpha:true and readPixels reads gl.RGBA, so
  // `pixels` already carries alpha; dropping it made transparent renders
  // export as opaque-on-black, which is wrong for every compositing use.
  const zlib = require('zlib');

  // Build IDAT rows: filter byte (0=no filter) + RGBA bytes
  const raw = Buffer.alloc(height * (1 + width * 4));
  for (let y = 0; y < height; y++) {
    const rowOff = y * (1 + width * 4);
    raw[rowOff] = 0; // filter byte
    for (let x = 0; x < width; x++) {
      const pxOff = (y * width + x) * 4;
      const destOff = rowOff + 1 + x * 4;
      raw[destOff] = pixels[pxOff];       // R
      raw[destOff + 1] = pixels[pxOff + 1]; // G
      raw[destOff + 2] = pixels[pxOff + 2]; // B
      raw[destOff + 3] = pixels[pxOff + 3]; // A (preserve transparency)
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
  ihdrData[9] = 6;  // color type: RGBA
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
