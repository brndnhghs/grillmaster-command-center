"""Per-node sim-cache byte budget regression test (ROADMAP R3-feature / TD-03).

Closes the *feature* half of TD-03: a single simulation's cached frames must be
bounded to SIM_CACHE_NODE_MAX_BYTES. An over-sized sim is subsampled (keep every
Nth frame) to fit — preserving full-duration playback coverage at lower temporal
resolution — instead of letting one node monopolise the whole cache or dropping
mid-sequence frames. Under-budget sims store in full (non-destructive default).

Uses tiny synthetic frames + a tiny cap so the subsample path is exercised
fast and deterministically, mirroring test_sim_cache_eviction.py's strategy.
"""

from __future__ import annotations

import numpy as np

from image_pipeline.core.graph import GraphExecutor, _sim_entry_bytes


def _mk_frames(n: int, wh: int = 8):
    """Synthetic sim frames: list[dict{"image": ndarray}]."""
    return [{"image": np.zeros((wh, wh, 3), dtype=np.float32)} for _ in range(n)]


def test_under_budget_sim_stored_in_full():
    ex = GraphExecutor.__new__(GraphExecutor)
    ex._sim_cache = {}
    ex._sim_params_hash = {}
    # tiny cap so any non-trivial sim is "over budget"
    ex.SIM_CACHE_NODE_MAX_BYTES = 10_000_000

    frames = _mk_frames(4, wh=8)  # 4 * 8*8*3*4 = 3072 bytes << cap
    ex._store_sim("n1", 1, frames, protect=set())

    stored = ex._sim_cache[("n1", 1)]
    assert len(stored) == 4, "under-budget sim must store all frames unchanged"


def test_oversized_sim_is_stride_subsampled_to_fit():
    ex = GraphExecutor.__new__(GraphExecutor)
    ex._sim_cache = {}
    ex._sim_params_hash = {}
    # cap = ~2 frames worth → even-spread sample preserves BOTH endpoints
    frame_bytes = _sim_entry_bytes(_mk_frames(1, wh=8))
    ex.SIM_CACHE_NODE_MAX_BYTES = int(frame_bytes * 2)  # 2 frames

    frames = _mk_frames(10, wh=8)
    ex._store_sim("big", 7, frames, protect=set())

    stored = ex._sim_cache[("big", 7)]
    assert 1 <= len(stored) < 10, f"expected subsample, got {len(stored)} frames"
    # even-spread sampling preserves BOTH endpoints (full-duration coverage)
    assert stored[0] is frames[0], "subsample must preserve first frame"
    assert stored[-1] is frames[-1], "subsample must preserve final frame (full duration)"
    assert _sim_entry_bytes(stored) <= ex.SIM_CACHE_NODE_MAX_BYTES, "still over budget"


def test_subsample_is_deterministic_and_covers_span():
    ex = GraphExecutor.__new__(GraphExecutor)
    ex._sim_cache = {}
    ex._sim_params_hash = {}
    frame_bytes = _sim_entry_bytes(_mk_frames(1, wh=8))
    ex.SIM_CACHE_NODE_MAX_BYTES = int(frame_bytes * 2)  # ~2 frames worth

    frames = _mk_frames(20, wh=8)
    ex._store_sim("span", 3, frames, protect=set())
    stored = ex._sim_cache[("span", 3)]

    # cap = ~2 frames worth → even-spread sample of 2 frames: endpoints 0 and 19
    assert len(stored) == 2, stored
    assert stored[0] is frames[0] and stored[1] is frames[19]


def test_empty_frames_noop():
    ex = GraphExecutor.__new__(GraphExecutor)
    ex._sim_cache = {}
    ex._sim_params_hash = {}
    ex.SIM_CACHE_NODE_MAX_BYTES = 1_400_000_000
    ex._store_sim("empty", 1, [], protect=set())
    assert ("empty", 1) not in ex._sim_cache


def test_store_sim_runs_global_eviction_afterwards():
    """_store_sim must still enforce the GLOBAL budget (BUG-8a guard)."""
    ex = GraphExecutor.__new__(GraphExecutor)
    ex._sim_cache = {}
    ex._sim_params_hash = {}
    ex.SIM_CACHE_MAX_BYTES = 2_000  # very small global cap
    ex.SIM_CACHE_NODE_MAX_BYTES = 1_400_000_000  # node cap high (no subsample)

    # Two under-node-budget sims whose combined size exceeds the global cap.
    a = _mk_frames(1, wh=16)  # 16*16*3*4 = 3072 bytes each
    ex._store_sim("node_a", 1, a, protect={"node_b"})  # protect b so it survives
    ex._store_sim("node_b", 1, a, protect={"node_b"})
    # global eviction should drop non-protected node_a, keep protected node_b
    assert ("node_b", 1) in ex._sim_cache
    assert ("node_a", 1) not in ex._sim_cache
