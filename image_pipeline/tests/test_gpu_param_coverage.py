"""GPU variable-exposure coverage contract.

Every numeric-range slider on a node that has a client-GPU live-preview twin
MUST be either:

  * routed through the twin's ``param_map`` (so the live-preview slider is live),
    OR
  * explicitly listed in ``GPU_PREVIEW_DROP_ALLOW`` with a justification.

This codifies the GPU-First "no silent param drops" contract (no hidden GLSL
constants, no dead live-preview sliders). Without this guard, a twin edit that
drops a uniform from ``param_map`` fails SILENTLY — the live preview slider
freezes at the shader default and the user sees no error.

Auto-justified (never require a list entry):
  * ``time`` / ``anim_speed`` — timeline-driven by the graph ``_timeline``.
  * choice/string params — the client only resolves numeric uniforms (pitfall #14).
  * params with no numeric min/max — not slider-exposed.
"""
import image_pipeline.methods  # noqa: F401 — trigger registration
from image_pipeline.core.registry import get_meta
from image_pipeline.methods.gpu_shaders import (
    CLIENT_GPU_SHIMS,
    CLIENT_GPU_SIMS,
    GPU_PREVIEW_DROP_ALLOW,
    GPU_SHADER_NODE_MAP,
    is_param_justified_drop,
)

# Slider params the graph timeline drives directly (not a static uniform).
TIMELINE_DRIVEN = {"time", "anim_speed"}


def _numeric_range_params(mid: str) -> list[str]:
    """Node params that are real numeric-range sliders (choice/no-range excluded)."""
    meta = get_meta(str(mid).zfill(2))
    if not meta or not meta.params:
        return []
    out = []
    for name, spec in meta.params.items():
        if name in TIMELINE_DRIVEN:
            continue
        if spec.get("choices"):
            continue
        mn, mx = spec.get("min"), spec.get("max")
        if mn is None or mx is None:
            continue
        out.append(name)
    return out


def test_gpu_coverage_no_silent_numeric_drops():
    """Every numeric node slider must be mapped to its twin or explicitly allowed."""
    entries = {}
    entries.update(CLIENT_GPU_SHIMS)
    entries.update(CLIENT_GPU_SIMS)

    silent = []
    for mid, entry in entries.items():
        mapped = set((entry.get("param_map") or {}).keys())
        for p in _numeric_range_params(mid):
            if p in mapped:
                continue
            if is_param_justified_drop(mid, p):
                continue
            silent.append((mid, p))
    assert not silent, (
        "numeric node sliders with no twin uniform and no contract-allowed drop "
        "(silent dead live-preview slider):\n"
        + "\n".join(f"  node {m}: '{p}'" for m, p in silent)
        + "\nFix: add the uniform to the twin's `uniforms=` + `param_map`, OR add "
          "an entry to GPU_PREVIEW_DROP_ALLOW with a justification."
    )


def test_gpu_coverage_drop_list_only_known_params():
    """GPU_PREVIEW_DROP_ALLOW must not reference params that no longer exist."""
    stray = []
    for mid, drops in GPU_PREVIEW_DROP_ALLOW.items():
        meta = get_meta(str(mid).zfill(2))
        real = set(meta.params.keys()) if meta and meta.params else set()
        for p in drops:
            if p not in real:
                stray.append((mid, p))
    assert not stray, (
        "GPU_PREVIEW_DROP_ALLOW references non-existent params (stale allow-list):\n"
        + "\n".join(f"  node {m}: '{p}'" for m, p in stray)
    )


def test_gpu_coverage_drop_list_distinct_from_mapped():
    """A param cannot be both mapped to a twin AND listed as a dropped param."""
    conflict = []
    for mid, entry in {**CLIENT_GPU_SHIMS, **CLIENT_GPU_SIMS}.items():
        mapped = set((entry.get("param_map") or {}).keys())
        for p in mapped:
            if is_param_justified_drop(mid, p):
                conflict.append((mid, p))
    assert not conflict, (
        "params both mapped and listed as dropped (remove the allow-list entry):\n"
        + "\n".join(f"  node {m}: '{p}'" for m, p in conflict)
    )


def test_gpu_coverage_drop_list_not_mislabelled_as_uncovered():
    """GPU_PREVIEW_DROP_ALLOW must not describe a COVERED node as 'lacking a GPU
    slot' / 'explicit drop'. A node with an active shim/sim in
    GPU_SHADER_NODE_MAP IS ported; mislabelling it as uncovered is factually
    wrong and previously misled an audit into thinking ported nodes were
    unported. The allow-list only documents params not wired to the twin.
    """
    covered = set(GPU_SHADER_NODE_MAP.keys())
    bad = []
    for mid, drops in GPU_PREVIEW_DROP_ALLOW.items():
        if mid not in covered:
            continue
        for reason in drops.values():
            # Only the false claim "lacks a GPU slot" is forbidden on a COVERED
            # node. A legitimate per-param justification may still say "explicit
            # drop" (e.g. node 326 explains its twin renders at canvas res) - that
            # is accurate and must NOT be flagged.
            if "lacks a GPU slot" in reason:
                bad.append((mid, reason))
    assert not bad, (
        "GPU_PREVIEW_DROP_ALLOW mislabels a COVERED node as uncovered:\n"
        + "\n".join(f"  node {m}: {r}" for m, r in bad)
    )
