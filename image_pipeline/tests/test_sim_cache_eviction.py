"""Sim-cache eviction regression test (ROADMAP R3 / TD-03 — test half).

Closes the eviction-coverage gap from docs/reports/testing.md: **no test for the
byte-budget sim-cache eviction logic or the "protect active nodes" invariant.**

`GraphExecutor._evict_sim_cache()` bounds total simulation memory. The critical
invariant (BUG-8b): sims of the graph *currently* rendering must NEVER be
evicted — every output frame re-reads all of them, so dropping one forces a full
re-cook on the next frame (`sim */N` every frame). Eviction may only reclaim
sims left behind by *previous* graphs.

This test constructs cache entries directly (fast, deterministic, no GPU) and
asserts:

  1. Over-budget cache evicts oldest *non-protected* entries until under budget.
  2. Protected entries (current graph's node_ids) survive eviction.
  3. Memory is actually bounded: total bytes drop below the cap (or only
     protected overshoot remains).
  4. Protecting an entry never causes a non-protected, older entry of the same
     total size to be kept incorrectly.

Per the loop's "tests before refactors" rule, this guards the *existing* global
budget before any per-node budget feature (the second half of TD-03) is built.

Note: sim-cache keys are ``(node_id, seed)`` tuples.
"""

from __future__ import annotations

import numpy as np

from image_pipeline.core.graph import GraphExecutor


def _make_frames(h: int, w: int, n: int) -> list[dict]:
    """A sim-cache entry: list of n float32 RGB frames of size h×w."""
    arr = np.zeros((h, w, 3), dtype=np.float32)
    return [{"image": arr.copy()} for _ in range(n)]


def test_evicts_oldest_non_protected_until_under_budget():
    ex = GraphExecutor.__new__(GraphExecutor)  # bypass __init__ (no disk needed)
    ex._sim_cache = {}
    ex._sim_params_hash = {}
    ex._sim_sidecars = {}
    # 4×192 KB (old_*) + 768 KB (cur_0) ≈ 1.5 MB. A 500 KB cap forces the
    # non-protected old_* entries out while the protected cur_0 stays.
    ex.SIM_CACHE_MAX_BYTES = 500_000

    # Four ~192 KB entries (64×256×3×4 bytes ≈ 192 KB).
    frame = _make_frames(64, 256, 1)
    for i in range(4):
        ex._sim_cache[(f"old_{i}", 1)] = frame  # 4 ≈ 768 KB total

    # Add a protected current-graph entry (~768 KB) to blow the budget.
    big = _make_frames(256, 256, 1)
    ex._sim_cache[("cur_0", 1)] = big

    # Protect the current graph ("cur_0"); old_* must be evicted.
    ex._evict_sim_cache(protect={"cur_0"})

    assert ("cur_0", 1) in ex._sim_cache, "protected entry was evicted (BUG-8b regression)"
    for i in range(4):
        assert (f"old_{i}", 1) not in ex._sim_cache, (
            f"non-protected entry old_{i} survived eviction — budget not enforced"
        )


def test_protected_entries_never_evicted_even_when_over_budget():
    ex = GraphExecutor.__new__(GraphExecutor)
    ex._sim_cache = {}
    ex._sim_params_hash = {}
    ex._sim_sidecars = {}
    ex.SIM_CACHE_MAX_BYTES = 100  # impossibly small cap

    big = _make_frames(256, 256, 1)  # far exceeds the 100-byte cap
    ex._sim_cache[("keep", 1)] = big

    ex._evict_sim_cache(protect={"keep"})

    # With only a protected entry present, it must remain despite blowing the cap.
    assert ("keep", 1) in ex._sim_cache, (
        "protected sim evicted under an unreachable budget — re-cook-on-every-"
        "frame regression (BUG-8b)"
    )
    assert ex.SIM_CACHE_MAX_BYTES < sum(
        f["image"].nbytes for f in ex._sim_cache[("keep", 1)]
    ), "precondition: entry really does exceed the cap"


def test_eviction_drops_oldest_first_among_unprotected():
    ex = GraphExecutor.__new__(GraphExecutor)
    ex._sim_cache = {}
    ex._sim_params_hash = {}
    ex._sim_sidecars = {}
    ex.SIM_CACHE_MAX_BYTES = 400_000  # ~2 of the 192 KB entries

    frame = _make_frames(64, 256, 1)
    # Insertion order defines age: a (oldest) → b → c (newest).
    ex._sim_cache[("a", 1)] = frame
    ex._sim_cache[("b", 1)] = frame
    ex._sim_cache[("c", 1)] = frame

    ex._evict_sim_cache(protect=set())  # nothing protected

    # Oldest should go first; with a ~2-entry budget, "a" (oldest) is dropped,
    # "c" (newest) survives.
    assert ("a", 1) not in ex._sim_cache, "oldest entry not evicted first"
    assert ("c", 1) in ex._sim_cache, "newest entry was evicted before the oldest"


def test_params_hash_cleared_on_eviction():
    ex = GraphExecutor.__new__(GraphExecutor)
    ex._sim_cache = {}
    ex._sim_params_hash = {"a": 12345, "b": 67890}
    ex._sim_sidecars = {}
    ex.SIM_CACHE_MAX_BYTES = 100

    frame = _make_frames(64, 256, 1)
    ex._sim_cache[("a", 1)] = frame
    ex._sim_cache[("b", 1)] = frame

    ex._evict_sim_cache(protect=set())

    # Both evicted under the tiny cap → their param hashes must be cleared so a
    # later re-cook isn't mistaken for a cache hit.
    assert "a" not in ex._sim_params_hash, "param hash not cleared for evicted node a"
    assert "b" not in ex._sim_params_hash, "param hash not cleared for evicted node b"
