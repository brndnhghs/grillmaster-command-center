"""Headless verification of the shootout structural-mutation guarantee.

Background
----------
A prior evolution batch had bred offspring collapse to near-clones of the
parent: because ``mutate()`` only drew *parameter-jitter* ops most of the
time, ``repair_graph``'s terminal-reachability prune stripped the tiny
topology edits and the gallery showed identical-looking clips.

The fix (evolve.py) forces at least one STRUCTURAL op (node swap /
insert filter / add branch / add driver / rewire / remove node) into every
breeding attempt, so bred children genuinely change graph topology.

These tests prove, with no network / running server:
  * every bred (non-gentle) child carries >= 1 structural op, and
  * every bred child has graph_distance > 0 from the parent (real
    topology change), and
  * gentle mode (param-jitter only) still yields a divergent child.

Defs come from the in-process node registry, so this is fully hermetic.
"""
import random

import pytest

from image_pipeline.shootout.config import DEFAULT_CONFIG
from image_pipeline.shootout.evolve import (
    graph_distance,
    mutate,
    _MUTATION_OPS,
    _op_param_jitter,
)
from image_pipeline.shootout.generator import (
    build_gene_pool,
    random_genome,
)

# Structural ops = every mutation op except pure parameter jitter.
STRUCTURAL_OPS = {
    op.__name__.lstrip("_") for op, _ in _MUTATION_OPS
    if op is not _op_param_jitter
}


@pytest.fixture(scope="module")
def pool():
    return build_gene_pool(DEFAULT_CONFIG)


@pytest.fixture(scope="module")
def parent(pool):
    # Seed 1 is known-light (≈9 nodes) so the hermetic test stays
    # inside a few hundred MB instead of triggering a heavy motif-composition
    # branch that some seeds hit.
    rng = random.Random(1)
    return random_genome(pool, DEFAULT_CONFIG, rng, "random")


def _breed(pool, parent, seed):
    return mutate(parent, pool, DEFAULT_CONFIG, random.Random(seed), 1, gentle=False)


def test_bred_children_carry_structural_op(pool, parent):
    """Every breeding attempt must include >= 1 structural op."""
    n_struct = 0
    total = 0
    for seed in range(6):
        child = _breed(pool, parent, 5000 + seed)
        if child is None:
            continue
        total += 1
        ops = child["deviation"]["ops"]
        assert isinstance(ops, list) and ops, "deviation.ops must list applied ops"
        if any(o in STRUCTURAL_OPS for o in ops):
            n_struct += 1
    assert total >= 1, "expected at least one render-ready bred child"
    assert n_struct == total, (
        f"{n_struct}/{total} bred children carried a structural op — "
        "mutate() must force >=1 structural op per breeding attempt"
    )


def test_bred_children_change_topology(pool, parent):
    """Every bred child must differ topologically from the parent."""
    for seed in range(6):
        child = _breed(pool, parent, 7000 + seed)
        if child is None:
            continue
        div = graph_distance(parent, child, pool)
        assert div > 0.0, "bred child must differ topologically from the parent"
        assert child["deviation"].get("divergence", 0) > 0


def test_gentle_mutation_still_diverges(pool, parent):
    """Gentle mode uses param-jitter only (no structural guarantee) but the
    child must still be divergent (params pushed)."""
    diverged = 0
    for seed in range(6):
        child = mutate(
            parent, pool, DEFAULT_CONFIG, random.Random(9000 + seed), 1, gentle=True
        )
        if child is None:
            continue
        if child["deviation"].get("divergence", 0) > 0:
            diverged += 1
    assert diverged >= 1
