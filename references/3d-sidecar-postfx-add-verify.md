# 3D Sidecar — Post-Processing Add & Verify Recipe

Authoritative recipe for **adding a new post-processing (PostFX) pass** to the
headless three.js sidecar (`image_pipeline/3d/threejs-sidecar.mjs`) and
**proving it works headlessly** before commit.

> Status (2026-07-13): the full PostFX stack is ALREADY BUILT and
> committed — bloom (bright-pass + separable Gaussian ping-pong), grade
> (brightness/contrast/saturation), vignette, FXAA, radial chromatic
> aberration, and film grain (blue-noise / IGN-dithered). Every pass is
> engaged via `scene._gmPostFX` params resolved in `buildScene` (the
> `__scene_render__` / `__scene3d__` case) and applied by `renderWithPostFX`.
> The neutral-default contract holds: when every PostFX param is at its
> default, the renderer takes the byte-identical direct path
> (`renderer.render(scene, camera)`), so this feature is purely additive.
>
> This document therefore serves two purposes:
> 1. The **verify recipe** used to lock the existing stack (run it after
>    any edit to the sidecar to prove you didn't silently break a pass).
> 2. The **add recipe** for the NEXT pass (e.g. a depth-of-field or
>    lens-distortion pass) — follow the same engine contract.

## Engine contract (load-bearing)

The `gl` headless backend exposes **WebGL1**. The vendored `three.module.js`
(r160) ships **WITHOUT** the postprocessing addons
(`EffectComposer` / `UnrealBloomPass` / `OutputPass`). So every pass is
built by hand as a **fullscreen-quad pipeline** rendered into
`THREE.WebGLRenderTarget`s (RGBA8, no float-texture extension required):

```
scene ─▶ sceneRT (sRGB-encoded, tone-mapped)
         ├─ bloom:   bright-pass ─▶ separable Gaussian blur (ping-pong)
         └─ composite: scene + bloom*intensity + grade + vignette ─▶ compositeRT
compositeRT ─▶ FXAA pass ─▶ canvas
                             (chromatic aberration + grain run as
                              extra RT passes before the FXAA blit)
```

Key invariants:
- **Additive only.** Do NOT change the direct (no-PostFX) path. The
  engagement branch is `fxEngaged = !!(fx.bloom || fx.vignette || ...)`.
  All neutral defaults ⇒ `fxEngaged === false` ⇒ direct render, byte-identical.
- **No addons.** Reuse the `_fsScene` / `_fsCam` / `_fsMesh` / `_fsMat` /
  `_fsRender` helpers. Each pass is a `ShaderMaterial` with one of the
  existing fragment shaders (`_BLOOM_BRIGHT`, `_BLUR`, `_COMPOSITE`,
  `_FXAA`, `_CHROMATIC`, `_GRAIN`).
- **Dispose RTs.** Every render target created in `renderWithPostFX` is
  `.dispose()`'d at the end. The `sceneRT.texture.colorSpace` is set to
  `SRGBColorSpace` so the post passes operate in display space (no double
  tone-mapping).
- **Don't `renderer.dispose()`.** On the headless `gl` backend,
  `WebGLRenderer.dispose()` calls `context.cancelAnimationFrame`, which the
  `gl` context lacks — it would null the renderer's internal context. The
  per-request renderer is GC'd with the short-lived process.

## Add recipe (next pass)

1. **Add the params** to BOTH scene-node cases in `buildScene`
   (`__scene_render__` and `__scene3d__`) inside `scene._gmPostFX`:
   ```js
   my_pass:        resolveParam(node, 'my_pass', 0),
   my_pass_strength: resolveParam(node, 'my_pass_strength', 1.0),
   ```
2. **Declare the same params** on the server-side node defs
   (`_THREEJS_3D_NODE_DEFS["__scene_render__"]` / `["__scene3d__"]`
   in `image_pipeline/core/graph.py`) so the UI surfaces them and the
   parity test (`test_server_3d_nodes_exposed_in_node_defs`) keeps passing.
3. **Write the GLSL** as a new `_MY_PASS` template string near the other
   fragment shaders (core three.js GLSL — `texture2D`, `gl_FragColor`).
4. **Wire it** in `renderWithPostFX`: create RT + material, render into it
   when `fx.my_pass > 0`, chain `finalTex` through it, dispose at end.
5. **Add the engagement flag** to `fxEngaged` so a non-default value
   routes through the RT pipeline.

## Verify recipe (headless — no browser needed)

The live sidecar runs as a separate Node.js process on `THREEJS_PORT`
(default `:7862`). Talk to it straight over HTTP — no TestClient, no
browser. The lock-in tests live in
`image_pipeline/tests/test_3d_sidecar_render.py` and are **skipped unless the
sidecar is reachable** (`skipif(not _sidecar_reachable())`), so the cron run
detects `3D sidecar: LIVE` and locks them in automatically.

### Manual smoke (one pass)

```bash
cd ~/Documents/GitHub/grillmaster-command-center

# Render the SAME wire graph with the pass off vs on, diff the PNGs.
# _build_graph() in the test file is the canonical fully-wired graph
# (geometry→material→mesh, mesh→scene, light, camera).

# Start the sidecar if it isn't up (separate process, adjacent port):
THREEJS_PORT=7862 node image_pipeline/3d/threejs-sidecar.mjs &
sleep 3

# The pass must NOT be a no-op: off vs on PNGs must differ.
# e.g. bloom: assert mean(abs(on - off)) > 0.02
#      chromatic: same; grain: same.
# And the neutral default (all params 0 / 1.0) must equal the
# direct render (the additive contract).
```

### Assertion set the committed tests lock in

| Test | What it proves |
|---|---|
| `test_wired_3d_graph_renders_nonblank_png` | fully-wired graph → valid RGBA PNG, `std > 0.02` (not empty) |
| `test_spin_advances_render_per_frame` | `spin_speed > 0` advances frames (Δ > 0.005) |
| `test_bloom_postfx_changes_render` | bloom engagement is NOT a no-op (Δ > 0.02) |
| `test_chromatic_aberration_changes_render` | chromatic pass is NOT a no-op (Δ > 0.02) |
| `test_transparent_bg_preserves_alpha` | `bg_mode='transparent'` keeps true RGBA (fg α≈1, bg α≈0) |
| `test_multi_object_scene_renders_both` | `object_a`/`object_b` both render |
| `test_grain_postfx_changes_render` | grain is NOT a no-op (Δ > 0.02) |
| `test_grain_strength_is_live` | grain `amount` drives output (strong Δ > weak Δ) |
| `test_server_sidecar_node_parity` | `_THREEJS_3D_NODE_DEFS` ⇔ sidecar `case` arms stay in sync |
| `test_server_3d_nodes_exposed_in_node_defs` | PostFX params mirrored on both scene nodes |

### Run it

```bash
cd ~/Documents/GitHub/grillmaster-command-center
THREEJS_PORT=7862 node image_pipeline/3d/threejs-sidecar.mjs &  # if not already up
env -u PYTHONPATH .venv/bin/python -m pytest \
  image_pipeline/tests/test_3d_sidecar_render.py -q -p no:cacheprovider
```

All tests must pass. If a pass becomes a no-op (Δ≈0) or the
additive contract breaks (neutral default no longer byte-identical to the direct
render), the test catches it — do NOT commit until green.

## Pitfalls (learned)

- **`gl` backend is WebGL1** — no `texture()` (use `texture2D`),
  no `out` variables (use `gl_FragColor`), no float render targets.
- **Don't dispose the PMREM env texture** in `buildEnvironment` — on the
  headless backend it invalidates the returned render target, leaving
  `scene.environment` a dead (black) texture. The PMREM RT is GC'd with
  the renderer.
- **Module-scope helper state**: `buildScene` reads `_activeRenderer`
  (set in `renderSceneToPng` before the build call) for the PMREM pass.
  If you refactor the render entry, keep `_activeRenderer` assigned before
  `buildScene` runs or the env map silently fails.
- **Changed-pixel fraction for spin**: a mean-abs-diff of ~0 is a
  FALSE NEGATIVE for rotation (silhouette can shift without changing the
  mean). The committed `test_spin_advances_render_per_frame` uses mean-abs
  Δ > 0.005 which is adequate for a torus-knot at the test's 60°/frame
  rate; if you switch the default shape to a symmetry angle (e.g. a cube
  at exactly 90°), use a non-symmetry angle instead.
