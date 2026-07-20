"""Regression guard for the GPU-First "coverage by category" mandate.

The GPU-First build plan requires a GPU live-preview source (a CLIENT_GPU_SHIMS
or CLIENT_GPU_SIMS entry) for EVERY node *category* that produces visual output
(IMAGE / FIELD), not just cherry-picked ids. This test locks that invariant in:
if a future run adds a new node to an applicable category (e.g. a fresh
``patterns`` or ``fractals`` node) without wiring a GPU twin, the category
regresses to zero GPU coverage and this test fails — surfacing the gap instead
of letting it silently ship.

Categories that are legitimately GPU-free (by design, per
references/gpu-category-reclassification.md and the 3D-sidecar / ML-utility
architecture) are excluded from the assertion:

  * ``channels``       — SCALAR/MASK-only extraction, no visual GPU twin needed
  * ``cli_tools``      — CPU-only utility nodes (no live preview)
  * ``client_3d``      — rendered by the separate three.js sidecar, not the GPU
                        shader twin path
  * ``io``             — input/output plumbing (load/save), not generative
  * ``ml_models``      — static CLIP/SAM utility nodes (CPU-only, models absent)
  * ``p5_sketches``    — single sketch node, not part of the twin family
  * ``system``         — timeline / graph-system control nodes
"""

import pytest

from image_pipeline.core.graph import get_all_node_defs
from image_pipeline.methods.gpu_shaders import GPU_SHADER_NODE_MAP

# Categories that are intentionally out of GPU-twin scope.
GPU_EXEMPT_CATEGORIES = {
    "channels",
    "cli_tools",
    "client_3d",
    "io",
    "ml_models",
    "p5_sketches",
    "system",
}


def _category_gpu_coverage():
    """Return (cat -> (n_methods, n_gpu_sourced)) for every category."""
    defs = get_all_node_defs()
    gpu_ids = set(GPU_SHADER_NODE_MAP.keys())
    cov = {}
    for mid, d in defs.items():
        cat = d.get("category", "?")
        entry = cov.setdefault(cat, [0, 0])
        entry[0] += 1
        if mid in gpu_ids:
            entry[1] += 1
    return cov


def test_gpu_coverage_by_category():
    cov = _category_gpu_coverage()
    failures = []
    for cat, (n_methods, n_gpu) in sorted(cov.items()):
        if cat in GPU_EXEMPT_CATEGORIES:
            continue  # out of GPU scope by design
        if n_gpu == 0:
            failures.append(f"{cat}: {n_methods} methods, 0 GPU-sourced")
    assert not failures, (
        "Categories with visual output but NO GPU twin (regression):\n  "
        + "\n  ".join(failures)
    )


def test_gpu_exempt_categories_have_no_twin():
    """Sanity: the exempt set really is GPU-free (guards the list above)."""
    cov = _category_gpu_coverage()
    leaky = [
        cat for cat in GPU_EXEMPT_CATEGORIES
        if cat in cov and cov[cat][1] > 0
    ]
    assert not leaky, f"Exempt categories that unexpectedly gained GPU twins: {leaky}"
