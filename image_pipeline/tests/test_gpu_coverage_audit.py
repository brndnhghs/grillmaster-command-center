"""GPU coverage audit — locks the categorical-coverage + typed-port contracts.

This test is the regression backstop for the GPU-First Build Plan. It does NOT
re-render shaders (that is covered by test_gpu_twin_invariant / test_shader_parity);
instead it enforces the *structural* guarantees the build must keep:

1. Typed-uniform port contract (GPU-First guardrail #2): every variable a typed
   GPU shader declares in its ``uniforms=`` spec MUST be exposed as a real,
   editable node param (so the UI shows a control for it and the client resolver
   can fill the uniform). A silent drop here means a control exists in the shader
   but no slider in the UI — a real bug the live preview would expose only at
   runtime, in the browser.

2. Categorical coverage: every real CPU category (minus documented-deferred
   ones) has at least one GPU source-of-truth mirror. The build's contract is
   "coverage by category, not by cherry-picked id".

3. Simulation deferral is exhaustive: the only ``simulations`` CPU nodes WITHOUT a
   GPU mirror are the ones on the documented deferred list (Arch-A stateful sims
   that need browser WebGL2 ping-pong parity, which cannot be verified
   headlessly). `channels` / `compositing` are likewise a documented,
   known gap (per-pixel transforms not yet ported). If a *non-deferred*
   sim or category silently loses its mirror, the build has regressed.

Run (headless, no browser needed):
  cd ~/Documents/GitHub/grillmaster-command-center
  env -u PYTHONPATH .venv/bin/python -m pytest image_pipeline/tests/test_gpu_coverage_audit.py -q
"""
from image_pipeline.core.registry import get_all, get_meta
from image_pipeline.core.shaders import SHADERS
from image_pipeline.methods.gpu_shaders import GPU_SHADER_NODE_MAP

# Stable count guard — bump when the GPU mirror grows (one logical chunk per run).
EXPECTED_MAP_ENTRIES = 201

# Simulations-category CPU nodes that are intentionally NOT GPU-mirrored yet.
# These are Architecture-A stateful sims (discrete CA, agent/particle systems,
# PDE/instability fields) whose honest GPU twin is a ping-pong sim needing
# browser WebGL2 parity — out of scope for headless verification. If you port
# one, remove it from this set AND add the appropriate CLIENT_GPU_SIMS entry.
DEFERRED_SIM_IDS = set(
    "20 34 35 36 55 79 83 84 86 88 89 90 92 94 97 98 99 101 102 103 "
    "106 107 109 110 111 112 113 114 116 117 123 129 130 131 134 "
    "136 145 147 149 151 152 156 158 159 161 167".split()
)

# CPU categories that MUST have a GPU mirror (coverage by category).
# patterns / fractals / filters / math_art / codegen are fully mirrored by the
# typed-uniform nodes (ids 220-300) + P0 shims. `simulations` is mirrored
# only for the closed-form subset; the rest are deferred ping-pong sims.
MIRRORED_CATEGORIES = {
    "patterns", "fractals", "filters", "math_art", "codegen", "simulations",
    "compositing",
}
# `channels` is intentionally NOT mirrored — its nodes are pure SCALAR control
# signals (LFO / envelope / math / logic), not per-pixel image operations, so a
# GPU pixel-shader twin does not apply. `compositing` gained its first GPU
# mirror (P0.7: __image_to_mask__ luminance mask twin), so it moved up to
# MIRRORED_CATEGORIES. Treat `channels` like the deferred sims: a documented gap.
DEFERRED_CATEGORIES = {"channels"}


def _norm(mid: str) -> str:
    return mid.lstrip("0")


def test_gpu_shadow_map_count_guard():
    """The GPU mirror must stay at the pinned size until a run grows it."""
    assert len(GPU_SHADER_NODE_MAP) == EXPECTED_MAP_ENTRIES, (
        f"GPU_SHADER_NODE_MAP grew/shrank to {len(GPU_SHADER_NODE_MAP)}; "
        f"bump EXPECTED_MAP_ENTRIES only after a real mapping change."
    )


def test_typed_uniforms_exposed_as_params():
    """GPU-First guardrail #2: every declared typed uniform is a node param.

    A typed shader advertises named variables in ``SHADERS[name]['uniforms']``.
    The factory turns each into a node param, but a typo or a renamed uniform
    would silently drop the control. This fails the build if any declared
    uniform is missing from the served node-def params.
    """
    import image_pipeline.methods  # noqa: F401 — ensure registration

    breaches = []
    for mid, entry in GPU_SHADER_NODE_MAP.items():
        if not entry.get("typed"):
            continue
        info = SHADERS.get(entry["shader"])
        if not info:
            breaches.append((mid, entry["shader"], "<shader not registered>"))
            continue
        uspec = info.get("uniforms") or {}
        meta = get_meta(str(mid))
        present = set(meta.params.keys()) if meta and meta.params else set()
        missing = [u for u in uspec if u not in present]
        if missing:
            breaches.append((mid, entry["shader"], missing))

    assert not breaches, (
        "Typed GPU shaders declare uniforms that are NOT exposed as node params "
        "(UI control would be missing / client uniform unfilled):\n"
        + "\n".join(
            f"  node {m} ({sh}): missing {miss}" for m, sh, miss in breaches
        )
    )


def test_mirrored_categories_have_coverage():
    """Coverage by category: every real CPU category (minus deferred) is mirrored."""
    import image_pipeline.methods  # noqa: F401 — ensure registration
    from image_pipeline.methods.gpu_shaders import CLIENT_GPU_SHIMS

    allm = get_all()

    # Categories touched by the GPU mirror.
    mirrored = set()
    for mid in CLIENT_GPU_SHIMS:
        m = allm.get(str(mid).zfill(2)) or allm.get(str(mid))
        if m:
            mirrored.add(m.category)
    for mid, entry in GPU_SHADER_NODE_MAP.items():
        if entry.get("typed"):
            m = allm.get(str(mid).zfill(2)) or allm.get(str(mid))
            if m:
                mirrored.add(m.category)

    missing = [c for c in MIRRORED_CATEGORIES if c not in mirrored]
    assert not missing, (
        "GPU mirror covers NO node in category(ies): " + ", ".join(missing)
    )


def test_deferred_categories_stable():
    """A deferred category must NOT gain a GPU mirror without being reclassified."""
    import image_pipeline.methods  # noqa: F401 — ensure registration
    from image_pipeline.methods.gpu_shaders import CLIENT_GPU_SHIMS

    allm = get_all()
    mirrored = set()
    for mid in CLIENT_GPU_SHIMS:
        m = allm.get(str(mid).zfill(2)) or allm.get(str(mid))
        if m:
            mirrored.add(m.category)
    for mid, entry in GPU_SHADER_NODE_MAP.items():
        if entry.get("typed"):
            m = allm.get(str(mid).zfill(2)) or allm.get(str(mid))
            if m:
                mirrored.add(m.category)

    prematurely = [c for c in DEFERRED_CATEGORIES if c in mirrored]
    assert not prematurely, (
        "Category(ies) in DEFERRED_CATEGORIES now HAVE a GPU mirror; "
        "move them into MIRRORED_CATEGORIES and port a node: "
        + ", ".join(premature)
    )


def test_sim_deferral_is_exhaustive():
    """The only simulations nodes without a GPU mirror are the deferred ones."""
    import image_pipeline.methods  # noqa: F401 — ensure registration

    allm = get_all()
    sims = {mid: m for mid, m in allm.items() if m.category == "simulations"}
    missing_from_map = [
        mid for mid in sims
        if mid not in GPU_SHADER_NODE_MAP and _norm(mid) not in GPU_SHADER_NODE_MAP
    ]
    not_deferred = [mid for mid in missing_from_map if mid not in DEFERRED_SIM_IDS]

    assert not not_deferred, (
        "Simulation node(s) are missing a GPU mirror but are NOT on the deferred "
        "list — either port them (add a CLIENT_GPU_SIMS entry) or add them to "
        "DEFERRED_SIM_IDS with a reason:\n"
        + "\n".join(f"  {mid} {sims[mid].name}" for mid in not_deferred)
    )
    # Sanity: the deferred set should track the real gap (no stale ids).
    assert len(missing_from_map) == len(DEFERRED_SIM_IDS), (
        f"deferred set size ({len(DEFERRED_SIM_IDS)}) != actual sim gap "
        f"({len(missing_from_map)}); reconcile DEFERRED_SIM_IDS."
    )
