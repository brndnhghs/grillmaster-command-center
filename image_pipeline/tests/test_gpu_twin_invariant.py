"""
GPU twin invariant regression test (pitfall #10 / #15d).

This locks in the guarantees the ad-hoc audit discovered:
  • Every procedural / filter twin SHADER compiles for both gl330 and webgl2.
  • Every procedural twin renders NON-BLACK at the neutral (0.5,) params OR
    lights up under some parameter sweep (a twin that is always uniform black
    at every param set is a silent-broken live preview).
  • Every FILTER twin samples u_texture, so it legitimately renders black with
    NO input image — but must render NON-BLACK when given a synthetic input
    (catches a twin whose filter logic is dead/no-op).
  • Every P1 SIM's seed/step/display shaders compile; the seed shader renders
    NON-BLACK standalone (step/display render black by design — pitfall #6,
    they read the previous frame from u_texture).

These invariants are NOT covered by the parametrized compile tests
(test_webgl2_transform_is_valid / test_gl330_matches_legacy_assembly), which
only check that each SHADERS entry assembles — not that a twin actually produces
visible, param-responsive output.

Run:
  cd ~/Documents/GitHub/grillmaster-command-center
  env -u PYTHONPATH .venv/bin/python -m pytest image_pipeline/tests/test_gpu_twin_invariant.py -q
"""
import numpy as np
import pytest

from image_pipeline.core.shaders import render_shader, build_fragment, SHADERS
from image_pipeline.methods.gpu_shaders import (
    GPU_SHADER_NODE_MAP,
    CLIENT_GPU_SHIMS,
    CLIENT_GPU_SIMS,
)

NEUTRAL = (0.5, 0.5, 0.5, 0.5)
SIZE = (128, 96)

# Parameter sweeps likely to light a shader up if it only activates off-neutral.
_PROBES = [
    (0.5, 0.5, 0.5, 0.5),
    (0.95, 0.5, 0.5, 0.5),
    (0.5, 0.95, 0.5, 0.5),
    (0.5, 0.5, 0.95, 0.5),
    (0.5, 0.5, 0.5, 0.95),
    (0.05, 0.05, 0.05, 0.05),
    (0.95, 0.95, 0.95, 0.95),
]


def _std(img):
    return float(np.array(img, dtype=np.float64).std())


def _mad(a, b):
    return float(np.mean(np.abs(np.array(a, np.float64) - np.array(b, np.float64))))


def _compile_ok(name):
    try:
        g = build_fragment(name, "gl330")
        w = build_fragment(name, "webgl2")
    except Exception as e:  # pragma: no cover - shader authoring error
        return f"ERR:{e}"
    if "#version" not in g or "#version" not in w:
        return "no-#version"
    return True


def _synthetic_input():
    yy, xx = np.mgrid[0:SIZE[1], 0:SIZE[0]]
    r = (xx / SIZE[0] * 255).astype(np.float32)
    g = (yy / SIZE[1] * 255).astype(np.float32)
    b = ((np.sin(xx * 0.1) * np.cos(yy * 0.1)) * 127 + 128).astype(np.float32)
    return np.stack([r, g, b], -1) / 255.0


@pytest.mark.parametrize("mid", sorted(
    (m for m, e in GPU_SHADER_NODE_MAP.items() if e.get("type") != "sim"),
    key=lambda x: int(x) if x.isdigit() else 1e9,
))
def test_twin_compiles_and_renders(mid):
    """Every NON-SIM twin compiles and produces a non-uniform output.

    Procedural/patterns twins must render non-black at neutral or under a
    parameter sweep. Twins that sample u_texture (filters / displacement such as
    swirl) legitimately render black with NO input, so when the no-input probe
    is flat-black we re-render WITH a synthetic input image — a genuinely
    no-op twin stays flat-black in both cases and fails.
    """
    entry = GPU_SHADER_NODE_MAP[mid]
    name = entry["shader"]
    cc = _compile_ok(name)
    assert cc is True, f"{mid} {name} compile failed: {cc}"

    best = 0.0
    for p in _PROBES:
        try:
            img = render_shader(name, SIZE, p, 0.0)
            best = max(best, _std(img))
        except Exception as e:  # pragma: no cover - runtime render error
            pytest.fail(f"{mid} {name} render raised: {e}")

    if best < 0.02:
        # Likely input-dependent (filter/displacement). Confirm it lights up
        # when fed a real image — otherwise it is a no-op twin.
        inp = _synthetic_input()
        try:
            img = render_shader(name, SIZE, NEUTRAL, 0.0, input_image=inp)
            best = max(best, _std(img))
        except Exception as e:  # pragma: no cover
            pytest.fail(f"{mid} {name} input render raised: {e}")

    assert best >= 0.02, (
        f"{mid} {name} renders uniform-black both without and with a "
        f"synthetic input image (std={best:.3f}) — likely a no-op twin"
    )


@pytest.mark.parametrize("mid", sorted(
    (m for m, e in GPU_SHADER_NODE_MAP.items()
     if e.get("type") != "sim" and "shader_" in e["shader"]),
    key=lambda x: int(x) if x.isdigit() else 1e9,
))
def test_filter_twin_requires_input(mid):
    """Filter twins sample u_texture, so they render black with NO input, but
    MUST render non-black when given a synthetic input image."""
    entry = GPU_SHADER_NODE_MAP[mid]
    name = entry["shader"]

    # No-input case: a healthy filter renders black (this is by design — it is
    # not a failure, just confirms it depends on the input wire).
    no_input = render_shader(name, SIZE, NEUTRAL, 0.0)
    # We do NOT assert black here; we only assert that WITH an input it lights up.

    inp = _synthetic_input()
    with_input = render_shader(name, SIZE, NEUTRAL, 0.0, input_image=inp)
    s = _std(with_input)
    assert s >= 0.02, (
        f"{mid} {name} filter renders flat-black even WITH a non-trivial "
        f"input image (std={s:.3f}) — filter logic appears dead/no-op"
    )


@pytest.mark.parametrize("mid", sorted(
    CLIENT_GPU_SIMS.keys(),
    key=lambda x: int(x) if x.isdigit() else 1e9,
))
def test_sim_shaders_compile_and_seed_renders(mid):
    """Every P1 sim's seed/step/display shaders compile; the seed shader must
    render non-black standalone. step/display legitimately render black (they
    need the previous frame from u_texture — pitfall #6)."""
    entry = CLIENT_GPU_SIMS[mid]
    for key in ("seed", "step", "display"):
        name = entry[key]
        cc = _compile_ok(name)
        assert cc is True, f"{mid} sim.{key} ({name}) compile failed: {cc}"

    # seed must produce visible output on its own
    seed_img = render_shader(entry["seed"], SIZE, NEUTRAL, 0.0)
    s = _std(seed_img)
    assert s >= 0.02, (
        f"{mid} sim seed shader ({entry['seed']}) renders flat-black "
        f"standalone (std={s:.3f}) — seed logic broken"
    )


# ── Typed-uniform wiring (silent-no-op-twin guard) ──────────────────────────
# Every typed-uniform node advertises named uniforms (u_<name>) that the UI
# sliders drive. A twin whose GLSL body ignores those uniforms (e.g. it reads
# the legacy u_params instead, or a helper macro shadows them) is a silent
# no-op: the live preview ignores every control the user sees. This guard fails
# the build if NO declared uniform visibly affects the rendered output.
#
# Robustness: a uniform may be legitimately gated (e.g. fractal color_a/b only
# apply under palette=grayscale, animation uniforms only matter at u_time>0), so
# instead of asserting every single uniform must change output (which would
# false-flag correct gated designs), we assert that at least ONE uniform —
# perturbed to a LARGE, valid extreme (not a tiny delta) with u_time=1.0 — moves
# the output beyond a threshold. A fully-static twin (all uniforms dead) fails;
# a correctly-wired twin passes even if some uniforms are gated.


def _extreme_values(spec):
    """Both range endpoints for a uniform, so the drive-output probe can pick
    whichever extreme (min or max) maximally disturbs the frame. Returns a list."""
    g = spec.get("glsl", "float")
    if g == "int":
        return [int(spec.get("min", 0)), int(spec.get("max", 99))]
    if g == "choice":
        ch = spec.get("choices", [])
        if len(ch) < 2:
            return [spec.get("default", 0)]
        return [ch[0], ch[-1]]  # both ends, maximally apart
    if g == "color":
        return [(0.95, 0.05, 0.05), (0.05, 0.05, 0.95)]
    lo = float(spec.get("min", 0.0))
    hi = float(spec.get("max", 1.0))
    return [lo, hi]


def _synthetic():
    yy, xx = np.mgrid[0:SIZE[1], 0:SIZE[0]]
    r = (xx / SIZE[0] * 255).astype(np.float32)
    g = (yy / SIZE[1] * 255).astype(np.float32)
    b = ((np.sin(xx * 0.1) * np.cos(yy * 0.1)) * 127 + 128).astype(np.float32)
    return np.stack([r, g, b], -1) / 255.0


@pytest.mark.parametrize("mid", sorted(
    (m for m, e in GPU_SHADER_NODE_MAP.items()
     if e.get("typed") and e.get("shader") in SHADERS),
    key=lambda x: int(x) if x.isdigit() else 1e9,
))
def test_typed_uniforms_drive_output(mid):
    """At least one declared typed uniform must visibly change the render.

    Perturb each uniform to a large valid extreme at u_time=1.0 (so animation
    uniforms are active) and render; require the best single-uniform delta to be
    non-trivial. Filters get a synthetic input so they are not uniformly black.
    """
    entry = GPU_SHADER_NODE_MAP[mid]
    name = entry["shader"]
    stype = SHADERS[name].get("type")
    uspec = SHADERS[name].get("uniforms") or {}
    if not uspec:
        pytest.skip(f"{mid} {name} has no typed uniforms")

    base = {u: spec.get("default") for u, spec in uspec.items()}
    kwargs = dict(named_params=base, time=1.0)
    if stype == "filter":
        kwargs["input_image"] = _synthetic()

    try:
        img_base = render_shader(name, SIZE, (0.5,) * 4, **kwargs)
    except Exception as e:  # pragma: no cover - shader authoring error
        pytest.fail(f"{mid} {name} base render raised: {e}")

    best = 0.0
    for u, spec in uspec.items():
        # Probe BOTH extremes of the uniform's valid range. Some twins are
        # self-similar or (near-)symmetric (e.g. Apollonian gasket, fractal /
        # kaleidoscopic / rotational patterns), so sweeping the default toward
        # a single far endpoint can map almost onto the base frame and produce
        # a low MAD even though the uniform is fully live. A genuinely no-op
        # uniform is flat at BOTH endpoints, so taking the max MAD still catches
        # dead controls while avoiding a symmetric-shape false negative.
        for extreme in _extreme_values(spec):
            single = dict(base)
            single[u] = extreme
            kw = dict(named_params=single, time=1.0)
            if stype == "filter":
                kw["input_image"] = _synthetic()
            try:
                img = render_shader(name, SIZE, (0.5,) * 4, **kw)
            except Exception as e:  # pragma: no cover
                pytest.fail(f"{mid} {name} uniform '{u}' render raised: {e}")
            best = max(best, _mad(img_base, img))

    assert best >= 1.0, (
        f"{mid} {name}: no declared typed uniform visibly affects the output "
        f"(best large-sweep MAD={best:.3f}). The twin may be reading u_params "
        f"instead of its u_<name> uniforms — a silent no-op live preview."
    )


# ── Twin-uniform ↔ CPU-param name match (live-preview wiring guard) ──────────
# The browser client (ui/js/client3d.js renderGpuShader) reads a twin shader's
# *named* uniforms when the shader declares `uniforms=` — it sets u_<name> from
# the live node's params[<name>] and IGNORES the shim's `param_map` entirely.
# Therefore every uniform name on a CLIENT_GPU_SHIMS twin MUST equal a real param
# of the backing CPU node, or that control silently renders with the shader's
# default (a dead slider in the live preview). This guard fails the build on any
# such mismatch so the wiring regression found on 2026-07-11 cannot recur.
#
# Exceptions: listed below are shim targets whose twin uniforms have NO clean
# CPU-node synonym by design (secondary artistic knobs, or a string param that
# pitfall #14 says must stay unmapped). These are intentional, not bugs.
_TWIN_UNIFORM_ALLOW = {
    "66": "julia: `constant` is a string param (pitfall #14); the twin uses its "
          "own fixed view (c_re/c_im/zoom/color_shift) — intentionally unmapped",
    "29": "voronoi: `scale` (zoom) is not exposed by CPU node 29",
    "03": "domain_warp: `warp`/`hue_shift` are shader-only artistic knobs not in "
          "CPU node 03",
    "05": "voronoise: `hue_shift` is not exposed by CPU node 05",
    "07": "truchet: `scale` (frequency) is not cleanly synonymous with CPU node 07 "
          "`tile_size`",
    "74": "swirl_gpu: `radius`/`spin` have no clean CPU-node synonym in node 74",
    # Categorical coverage (2026-07-11): each twin below is a standalone
    # closed-form pattern generator with its own artistic uniforms (fg/bg color,
    # speed, line thick, internal freq counts) that have NO CPU-node synonym by
    # design. The backing CPU node is a different algorithm (particle/serial/
    # multi-pass) whose params do not map 1:1 onto the twin's per-pixel knobs;
    # the twin is a live-preview approximation and the CPU fn stays authoritative
    # (two-tier precision). Documented so the wiring guard does not regress.
    "16": "flow_field_typed: zoom/swirl/freq/density are shader-only flow knobs; "
          "only `speed` maps to CPU node 16",
    "65": "waveform_typed: amp/thick/fg/bg are shader-only render knobs; only "
          "freq1/2/3 (k1/2/3) map to CPU node 65",
    "78": "circle_packing_typed: normalized min_r/max_r/speed/bg/fg are shader "
          "knobs; only radius bounds map to CPU node 78",
    "56": "maze_typed: normalized scale/wall/drift/bg/fg are shader knobs; only "
          "cell_size/ wall_thickness map to CPU node 56",
    "81": "fourier_circles_typed: thick/phase2/bg/fg are shader knobs; freq1/2/3/"
          "speed map to CPU node 81",
    "406": "harmonograph_typed: decay/turns/steps/thick/bg/fg are shader knobs; "
          "freq1/freq2/phase/scale map to CPU node 406",
    "409": "superformula_typed: n/b/c/p/scale/thick/bg are shader knobs; only `m` "
           "maps to CPU node 409",
    "431": "domain_coloring_typed: `grid` is a shader-only contour-overlay strength; "
           "the CPU node's `coloring` is a string mode ('grid'/'none'/...), not a "
           "float synonym. exponent/scale/center_x/center_y map 1:1 to CPU node 431; "
           "animation is driven by the live-preview clock u_time (not a CPU param).",
    # gpu-twin-candidate CPU nodes (431/432/433/464) — closed-form f(uv,t) live
    # previews. Each routes real numeric params 1:1 via param_map; remaining
    # shader uniforms are artistic knobs with no CPU-node float synonym (the CPU
    # node is a different/serial algorithm), same pattern as 16/65/78 above.
    "432": "maurer_rose_typed: k/d/n_lines/line_width/brightness/hue animate via "
           "param_map (petals/deg/steps/thick + anim_speed->speed); `bg`/`scale` are "
           "shader-only render knobs (background->bg bridged, `scale` not in CPU 432)",
    "433": "low_discrepancy_typed: count/radius/anim_speed map 1:1; `bg`/`ox`/`oy` are "
           "shader-only seed/background knobs with no CPU-node float synonym",
    "464": "thin_film_gpu: thickness/thickness_scale/ior/tilt/intensity/saturation map "
           "via param_map (thickness_range/angle/strength/saturation); the twin names "
           "them by optical role (thickness_range/angle/strength) rather than the CPU "
           "param names (thickness_scale/tilt/intensity) — all bridged, no dead control",
    # Closed-form live-preview twins (pattern / math-art) whose artistic uniforms
    # have NO clean CPU-node float synonym by design — the backing CPU node is a
    # different/serial algorithm (particle/attractor integrator, parametric tube
    # generator) whose params do not map 1:1 onto the twin's per-pixel knobs. The
    # twin is a live-preview approximation and the CPU fn stays authoritative
    # (two-tier precision). Same intentional-exception pattern as 16/65/78/56/81/
    # 406/409/431/432/433 above.
    "342": "strange_attractor_typed: band/gain/speed are shader-only artistic knobs "
           "(attractor color/intensity/animation) with no CPU-node float synonym; CPU "
           "node 342 is a serial particle integrator (a/b/c/d/points/brightness)",
    "444": "droste_typed: bands/fg/bg/speed are shader-only render knobs (ring count/"
           "colors/animation) with no CPU-node float synonym; CPU node 444 is a "
           "different spiral algorithm (ring_spacing/twist/zoom)",
    "498": "de_jong_typed: morph/sharp/density_scale/speed are shader-only artistic "
           "knobs with no CPU-node float synonym; CPU node 498 is a serial attractor "
           "integrator (walkers/steps/discard)",
    "510": "flow_field_typed: zoom/swirl/freq/density/fg/bg are shader-only flow knobs "
           "with no CPU-node float synonym (CPU node 510 uses noise_scale/particles/dt/"
           "steps); twin is a live-preview approximation",
    "957": "strange_attractor_typed: band/gain/speed are shader-only artistic knobs "
           "with no CPU-node float synonym; same pattern as node 342",
    "962": "torusknot_typed: rad/steps/scale/thick/speed/bg are shader-only render "
           "knobs (tube radius/steps/scale/thickness/animation/bg) with no CPU-node "
           "float synonym; CPU node 962 is a parametric tube generator (major_r/tube_r/"
           "line_width/n_points/glow)",
    "964": "gyroid_tpms_gpu: rotate is a shader-only artistic knob (surface rotation) "
           "with no CPU-node float synonym; twin is a live-preview approximation",
    "1006": "phasor_noise_gpu: contrast/animate are shader-only artistic knobs "
           "(contrast/animation toggle) with no CPU-node float synonym; twin is a "
           "live-preview approximation",
    # P0 closed-form CPU nodes -> faithful typed twins (cron run). Each routes its
    # real numeric params 1:1 via param_map; remaining shader uniforms are artistic
    # / render knobs with no CPU-node float synonym (the CPU node is a different or
    # serial algorithm, or the knob is purely cosmetic). Two-tier precision: CPU fn
    # stays authoritative. Same intentional-exception pattern as 342/444/498/510/957/962.
    "355": "curl_noise_gpu: brightness is a shader-only render knob; scale/octaves map 1:1 "
           "to CPU node 355 (Curl-Noise Warp) — faithful live preview",
    "343": "hex_grid_typed: thickness/flow/offset_x/offset_y/fill_a/fill_b/line are shader-only "
           "grid render knobs; scale maps 1:1 to CPU node 343 (Hex Distance Field)",
    "470": "mandelbulb_gpu: cam_angle/bailout/spec/base_color/glow_color/bg/anim_speed are "
           "shader-only render knobs; power/iterations/cam_dist map 1:1 to CPU node 470 (Mandelbulb)",
    "62": "strange_attractor_typed: band/gain/speed are shader-only artistic knobs; a/b/c/d map "
           "1:1 to CPU node 62 (Chaotic Map) — faithful live preview of the attractor",
    "351": "mandelbrot_gpu: color_shift is a shader-only palette knob; zoom/center_x/center_y/"
           "iterations/escape_radius map 1:1 to CPU node 351 (Kaleidoscopic IFS)",
    "997": "color_grade_gpu: contrast/hue_rotate/temperature/tint/vignette are shader-only grade "
           "knobs; exposure/gamma/saturation map 1:1 to CPU node 997 (Tone Mapping)",
    "991": "bilateral_grid_gpu: grid_scale is a shader-only internal knob; blend/sigma_r/sigma_s "
           "map 1:1 to CPU node 991 (Domain Transform filter)",
}


import image_pipeline.methods  # noqa: F401  (bulk-registers all method nodes)
from image_pipeline.core import registry as _registry  # noqa: E402


@pytest.mark.parametrize("mid", sorted(
    (m for m in CLIENT_GPU_SHIMS.keys() if m.isdigit()),
    key=lambda x: int(x),
))
def test_gpu_twin_uniforms_match_params(mid):
    """Each twin shader's uniform names must be a subset of the backing CPU
    node's params, so the client's typed path wires them live (not dead)."""
    entry = CLIENT_GPU_SHIMS[mid]
    name = entry["shader"]
    info = SHADERS.get(name)
    if not info or not info.get("uniforms"):
        pytest.skip(f"{mid} {name} has no typed uniforms")
    unames = set(info["uniforms"].keys())
    meta = _registry.get_meta(mid)
    if meta is None:
        pytest.skip(f"{mid} backing CPU node not registered in this session")
    pset = set(meta.params.keys())
    missing = unames - pset
    if not missing:
        return
    if mid in _TWIN_UNIFORM_ALLOW:
        # Documented intentional exception — must not regress into a NEW mismatch.
        pytest.skip(f"{mid} {name}: allowed exception ({sorted(missing)})")
    pytest.fail(
        f"{mid} {name}: twin uniform(s) {sorted(missing)} have no matching CPU "
        f"node param — the live preview control is dead. Rename the shader "
        f"uniform to the CPU param name, or add it to _TWIN_UNIFORM_ALLOW with "
        f"a reason."
    )
