"""Regression test for node-quality #15 (field_defaulted_to_image).

Architecture A simulations write sidecars (field/particles/mask) to disk
during the sim cook. The executor must carry those sidecars through the
sim-cache so that cache-hit frames serve the typed ports correctly instead
of defaulting `field` to the RGB image.

This exercises the real gray_scott (id 155) sim, which calls
`write_field(out_dir, U)` (gray_scott.py:438), on both the render path
(`in_memory=False`, the default where #15 was originally observed) and the
live path (`in_memory=True`). In both modes the Arch-A cook path installs
only `set_job_context` (not the sidecar sink), so `write_field` falls through
to disk and the executor recovers the sidecar at cook time.
"""
import shutil
from pathlib import Path

import numpy as np
import pytest

import image_pipeline.methods  # noqa: F401 — trigger @method registration
from image_pipeline.core.graph import GraphExecutor
from image_pipeline.core.utils import set_canvas


@pytest.fixture(autouse=True)
def _pin_canvas():
    """Pin a small canvas so the sim stays fast and frames don't degenerate."""
    set_canvas(256, 256)
    yield


def _cleanup(path: Path):
    if path.exists():
        shutil.rmtree(path, ignore_errors=True)


def _make_nodes(n_frames: int) -> list:
    return [{
        "id": "n1",
        "method_id": "155",  # gray_scott — writes a field sidecar
        "params": {"n_frames": n_frames, "anim_mode": "spots"},
        "dirty": True,
    }]


def test_arch_a_sidecar_survives_cache_hit_render():
    """Arch-A sim field sidecar must survive a cache-hit frame (render path)."""
    out = Path("/tmp/test_sim_sidecar_cache")
    _cleanup(out)
    nodes = _make_nodes(100)
    # Render path (default in_memory=False, audit_to_disk=True) — where #15 hit.
    ex = GraphExecutor(out, fps=24)

    # Frame 0 — cooks the sim, caches frames + sidecars.
    ex.execute(nodes, [], 42, frame=0, frames=100)
    key = ("n1", 42)
    assert key in ex._sim_cache, "Sim frames should be cached after cook"
    assert key in ex._sim_sidecars, "Sidecars should be cached after cook"
    assert "field" in ex._sim_sidecars[key], "field sidecar missing after cook"
    assert ex._sim_sidecars[key]["field"].ndim == 2, "field must be 2D (H,W)"

    # Frame 3 — cache hit. Sidecar must still be present and correct.
    result, _, _ = ex.execute(nodes, [], 42, frame=3, frames=100)
    out1 = result.get("n1", {})
    field = out1.get("field")
    image = out1.get("image")
    assert field is not None, "Cache-hit frame must carry the field sidecar"
    assert field.ndim == 2, "field must be 2D (H,W), not the RGB image fallback"
    # The bug: field defaulted to the RGB image (3D). Prove it's the real field.
    assert image is not None, "Cache-hit frame must carry the image"
    assert field.ndim != image.ndim, "field must not be the image (bug #15)"
    assert np.array_equal(field, ex._sim_sidecars[key]["field"]), \
        "cache-hit field must equal the cooked sidecar"

    _cleanup(out)


def test_arch_a_sidecar_survives_cache_hit_live():
    """Arch-A sim field sidecar must survive a cache-hit frame (live path)."""
    out = Path("/tmp/test_sim_sidecar_cache_live")
    _cleanup(out)
    nodes = _make_nodes(100)
    # Live path: in_memory=True, audit_to_disk=False.
    ex = GraphExecutor(out, fps=24, in_memory=True, audit_to_disk=False)

    ex.execute(nodes, [], 42, frame=0, frames=100)
    key = ("n1", 42)
    assert key in ex._sim_sidecars, "Sidecars should be cached after cook (live)"
    assert "field" in ex._sim_sidecars[key], "field sidecar missing after cook (live)"

    result, _, _ = ex.execute(nodes, [], 42, frame=3, frames=100)
    out1 = result.get("n1", {})
    field = out1.get("field")
    image = out1.get("image")
    assert field is not None, "Cache-hit frame must carry the field sidecar (live)"
    assert field.ndim == 2, "field must be 2D (H,W) in live mode"
    assert image is not None
    assert field.ndim != image.ndim, "field must not be the image (bug #15, live)"
    assert np.array_equal(field, ex._sim_sidecars[key]["field"])

    _cleanup(out)


def test_sidecar_eviction_parity():
    """_evict_sim_cache must clear _sim_sidecars alongside _sim_cache."""
    ex = GraphExecutor(Path("/tmp/test_sim_sidecar_evict"), fps=24)
    # Force the eviction loop to actually run (budget 0 → everything unprotected pops).
    # setattr bypasses the Literal[1500000000] class-attribute type.
    setattr(ex, "SIM_CACHE_MAX_BYTES", 0)
    ex._sim_cache[("n1", 42)] = [np.zeros((4, 4, 3), dtype=np.float32)]
    ex._sim_sidecars[("n1", 42)] = {"field": np.zeros((4, 4), dtype=np.float32)}
    ex._sim_params_hash["n1"] = 1

    ex._evict_sim_cache(protect=None)

    assert ("n1", 42) not in ex._sim_cache, "_sim_cache not evicted"
    assert ("n1", 42) not in ex._sim_sidecars, "_sim_sidecars not evicted (parity gap)"
    assert "n1" not in ex._sim_params_hash, "_sim_params_hash not evicted"
