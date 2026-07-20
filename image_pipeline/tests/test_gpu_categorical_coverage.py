"""GPU categorical-coverage regression guard (Route 0 — Leverage tier).

The GPU-First build plan mandates "a GPU source for EVERY node category"
(patterns, fractals, math_art, filters, simulations, …): every CPU node must
be addressable as a GPU live-preview source via a shim, a ping-pong sim, or a
standalone typed node. This guard locks that investment in place so a future
refactor that silently drops a category's GPU coverage fails CI instead of
shipping a regression.

Purely additive — no behaviour change, no rendering, runs in milliseconds.
"""
import image_pipeline.methods  # noqa: F401  (registers all methods)
from image_pipeline.server import get_node_defs
from image_pipeline.methods.gpu_shaders import CLIENT_GPU_SHIMS, CLIENT_GPU_SIMS


def _coverage_by_category(node_defs):
    """Return {category: # CPU nodes of that category with a GPU source}."""
    gpu_covered = set(CLIENT_GPU_SHIMS) | set(CLIENT_GPU_SIMS)
    cov = {}
    for mid, d in node_defs.items():
        if mid.startswith("__"):
            continue
        cat = d.get("category")
        if cat is None:
            continue
        if mid in gpu_covered:
            cov[cat] = cov.get(cat, 0) + 1
    return cov


# Floors reflect the mature GPU-First surface as of this commit. They are
# deliberately slightly below the current live counts so a legitimate, small
# trim does not fail CI, but a major/accidental coverage drop does.
CATEGORY_FLOORS = {
    "patterns": 30,
    "fractals": 8,
    "math_art": 10,
    "filters": 25,
    "simulations": 40,
}


def test_every_generative_category_has_gpu_coverage():
    node_defs = get_node_defs()
    cov = _coverage_by_category(node_defs)
    missing = [c for c in CATEGORY_FLOORS if c not in cov or cov[c] == 0]
    assert not missing, (
        f"Generative categories with NO GPU source: {missing}. "
        "Every CPU category must retain at least one GPU twin (shim/sim/typed)."
    )


def test_categorical_gpu_coverage_floors_held():
    node_defs = get_node_defs()
    cov = _coverage_by_category(node_defs)
    violations = {
        c: cov.get(c, 0)
        for c, floor in CATEGORY_FLOORS.items()
        if cov.get(c, 0) < floor
    }
    assert not violations, (
        f"GPU coverage dropped below floor for: {violations}. "
        f"Floors={CATEGORY_FLOORS}. Restore the dropped twins or raise the floor "
        "with an explicit commit if the drop is intentional."
    )


def test_gpu_coverage_matrix_prints():
    # Informational: surfaces the live coverage matrix in the test log so a
    # human reviewing CI output can see the categorical state at a glance.
    node_defs = get_node_defs()
    cov = _coverage_by_category(node_defs)
    total_cpu = sum(
        1 for mid, d in node_defs.items()
        if not mid.startswith("__") and d.get("category") in CATEGORY_FLOORS
    )
    covered_cpu = sum(cov.get(c, 0) for c in CATEGORY_FLOORS)
    print("\nGPU categorical coverage:")
    for c in sorted(CATEGORY_FLOORS):
        print(f"  {c:14} {cov.get(c, 0):3} CPU nodes with GPU source")
    print(f"  {'TOTAL':14} {covered_cpu:3} / {total_cpu} generative CPU nodes covered")
    print(f"  shims={len(CLIENT_GPU_SHIMS)} sims={len(CLIENT_GPU_SIMS)}")
    assert covered_cpu > 0
