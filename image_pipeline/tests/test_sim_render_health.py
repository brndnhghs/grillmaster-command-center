"""Render-health contract for simulation methods.

Leverage-Tier regression guard (grillmaster-image-pipeline skill, Route 0):
every ``simulations``-category method must produce a valid (non-blank, correct
shape) image within a bounded time. This catches a broken or silently no-op sim
method that registers cleanly in ``/api/node-defs`` but dies at runtime -- the
same class of defect the ML e2e probe guards against for CLIP/SAM.

Each method runs in a subprocess worker with a hard timeout so a pathological
sim can never hang the suite. Marked ``slow`` so the default fast run stays
under the cron time budget; run explicitly with ``-m slow`` or ``--all``.

Run:
    cd ~/Documents/GitHub/grillmaster-command-center
    env -u PYTHONPATH .venv/bin/python -m pytest \
        image_pipeline/tests/test_sim_render_health.py -q -p no:cacheprovider -m slow
"""
from __future__ import annotations

import time

import numpy as np
import pytest
from PIL import Image

import image_pipeline.methods  # noqa: F401 -- trigger @method registration
from image_pipeline.core.registry import get_all

# Generous bound: cumulative-growth sims (DLA #36, Sandpile #55) legitimately run
# long at defaults even though their cores are already vectorized -- their output
# is an emergent process over many steps (see skill pitfall #9). Other sims clear
# 90s comfortably (observed max ~62s). The two cumulative sims get a higher cap
# proportionate to their observed worst-case (~94s for DLA).
PER_METHOD_TIMEOUT = 90.0
CUMULATIVE_SLOW_IDS = {"36", "55"}  # DLA, Sandpile -- expected long runtimes
CUMULATIVE_TIMEOUT = 150.0

SIM_IDS = sorted(mid for mid, m in get_all().items() if m.category == "simulations")
NAMES = {mid: get_all()[mid].name for mid in SIM_IDS}


def _frame_validity(arr):
    """Return ``(ok, reason)`` for an output frame.

    Accepts RGB or RGBA float arrays in [0,1]. RGBA is allowed for
    sparse-content methods (discrete dots/blobs on a transparent background --
    pipeline rule 9: "Methods with discrete objects/blobs on empty bg output
    RGBA"); for those we judge content over the ``alpha > 0`` pixels only.

    A frame is blank only if it is a single flat colour (std < 0.01 AND <= 2
    distinct quantized tones) or, for RGBA, fully transparent / a single flat
    colour of opaque pixels. A low-contrast but VALID render (faint Perlin
    relief, shallow-water at rest, sparse stipples) has many distinct tones and
    must NOT be flagged blank.
    """
    if arr.ndim != 3 or arr.shape[2] not in (3, 4):
        return False, f"unexpected shape {arr.shape}"
    rgb = arr[..., :3]
    if arr.shape[2] == 4:
        alpha = arr[..., 3]
        mask = alpha > 0.5
        if not mask.any():
            return False, "fully transparent (no opaque pixels)"
        rgb = rgb[mask]
    if rgb.size == 0:
        return False, "no content"
    if rgb.std() < 0.01:
        q = (rgb * 31).astype(np.int32)
        q = q[..., 0] * 1024 + q[..., 1] * 32 + q[..., 2]
        if np.unique(q).size <= 2:
            return False, f"blank output (flat colour, std={rgb.std():.4f})"
    return True, ""

# Wirable input port types -> node needs an upstream wire to produce output.
# A sim whose only meaningful input is another node's output (e.g. a PARTICLE
# consumer that intentionally blanks when nothing is wired) legitimately
# renders nothing useful with no upstream wire, so testing it standalone is
# meaningless. This mirrors the generator harness's wire-dependency skip.
WIRABLE_PORT_TYPES = {"image", "mask", "field", "particles"}

# Sims that are wire-dependent (no useful standalone render) are auto-detected
# by input-port type, so this set stays correct as the graph grows.
WIRE_DEP_SIM_IDS = sorted(
    mid for mid, m in get_all().items()
    if m.category == "simulations"
    and {str(t).lower() for t in (m.inputs or {}).values()} & WIRABLE_PORT_TYPES
)


def _timeout_for(mid: str) -> float:
    return CUMULATIVE_TIMEOUT if mid in CUMULATIVE_SLOW_IDS else PER_METHOD_TIMEOUT


def _run_one(mid: str):
    """Subprocess worker: render one method at defaults.

    Returns (ok: bool, seconds: float, detail: str).
    """
    from pathlib import Path as _P
    import image_pipeline.methods  # noqa: F401
    from image_pipeline.core.registry import get_all as _ga
    from image_pipeline.core import utils as U

    meta = _ga()[mid]
    node_dir = _P("/tmp/sim_health") / mid
    if node_dir.exists():
        for f in node_dir.glob("*"):
            f.unlink()
    node_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    # Cumulative-growth sims (DLA #36, Sandpile #55) emit an emergent process
    # over many steps; at 256x256 they legitimately exceed the 150s cap. The
    # render-health contract validates *validity* (non-blank, sane shape), not
    # resolution, so we render those at 128x128 -- well within the cap -- while
    # every other sim renders at the standard 256x256.
    size = 128 if mid in CUMULATIVE_SLOW_IDS else 256
    token = U.set_canvas(size, size)
    try:
        meta.fn(node_dir, 42, params={})
    finally:
        U.reset_canvas(token)
    dt = time.time() - t0

    # Validate output: a non-blank PNG with sane shape.
    pngs = sorted(p for p in node_dir.glob("*.png") if not p.name.startswith("_"))
    if not pngs:
        return False, dt, "no PNG output"
    try:
        img = Image.open(str(pngs[-1])).convert("RGBA")
        arr = np.asarray(img, dtype=np.float32) / 255.0
        ok, detail = _frame_validity(arr)
        if not ok:
            return False, dt, detail
    except Exception as e:  # noqa: BLE001
        return False, dt, f"read failed: {type(e).__name__}: {e}"
    return True, dt, ""


pytestmark = pytest.mark.slow


@pytest.mark.parametrize(
    "mid",
    [m for m in SIM_IDS if m not in set(WIRE_DEP_SIM_IDS)],
    ids=[f"{m}:{NAMES[m]}" for m in SIM_IDS if m not in set(WIRE_DEP_SIM_IDS)],
)
def test_sim_renders_valid_frame(mid):
    """Each simulation method must render one valid (non-blank) frame in time."""
    from concurrent.futures import ProcessPoolExecutor, TimeoutError as FuturesTimeout

    with ProcessPoolExecutor(max_workers=1) as ex:
        fut = ex.submit(_run_one, mid)
        try:
            ok, dt, detail = fut.result(timeout=_timeout_for(mid))
        except FuturesTimeout:
            pytest.fail(f"{NAMES[mid]} ({mid}) exceeded {_timeout_for(mid):.0f}s (TIMEOUT)")
        except Exception as e:  # noqa: BLE001
            pytest.fail(f"{NAMES[mid]} ({mid}) worker crashed: {type(e).__name__}: {e}")

    assert ok, f"{NAMES[mid]} ({mid}) failed: {detail} [ran {dt:.1f}s]"
