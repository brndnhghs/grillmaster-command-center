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


def _extreme_value(spec):
    """A large, clearly-visible value for a uniform (not a tiny delta)."""
    g = spec.get("glsl", "float")
    if g == "int":
        return int(spec.get("max", 99))
    if g == "choice":
        ch = spec.get("choices", [])
        if len(ch) < 2:
            return spec.get("default", 0)
        return ch[-1]  # maximally different from the default (usually index 0)
    if g == "color":
        return (0.95, 0.05, 0.05)
    lo = float(spec.get("min", 0.0))
    hi = float(spec.get("max", 1.0))
    d = float(spec.get("default", (lo + hi) / 2))
    # pick the extreme farther from the default so the visual change is large
    return hi if abs(hi - d) >= abs(d - lo) else lo


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
        single = dict(base)
        single[u] = _extreme_value(spec)
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
