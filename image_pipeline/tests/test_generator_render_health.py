"""Render-health contract for self-contained GENERATOR methods (Leverage Tier).

Companion to ``test_sim_render_health.py`` (which covers ``simulations``). This
guard locks the OTHER half of the pipeline's image-output surface: the
generator-category methods (``patterns`` / ``fractals`` / ``filters`` /
``math_art``) that produce a still frame from params alone.

Why it matters: a generator that registers cleanly in ``/api/node-defs`` but
dies or goes blank at default params would otherwise ship silently -- the same
class of defect the sim contract guards against for ``simulations``. With this
test, every generator must produce a valid (non-blank, correct-shape) image at
its declared defaults.

Faithful read-back (mirrors ``core/graph.py`` execute()):
    The executor accepts a method's RETURN value OR a PNG on disk. So a method
    that returns ``{"image": arr}`` is just as valid as one that calls
    ``save()``. We therefore treat BOTH as a successful render -- we do NOT
    require a disk PNG (that would false-fail node 05 "Procedural Noise", which
    returns its image and lets the executor write the file). We only fail on a
    blank/None/missing image or a hard crash.

Wire-dependent nodes are excluded: methods whose declared input ports include a
wirable type (IMAGE / MASK / FIELD / PARTICLES) legitimately render nothing
useful with no upstream wire, so testing them with empty inputs is meaningless.
The 4 known wire-dependent generator nodes (43, 48, 73, __transform__) are
auto-detected by input-port type, so the skip list stays correct as the graph
grows.

Each method runs in a subprocess worker with a hard timeout (generators are fast
at 160x160, but a pathological one must never hang the suite). Marked ``slow`` so
the default fast run stays under the cron time budget; run explicitly with
``-m slow`` or ``--all``.

Run:
    cd ~/Documents/GitHub/grillmaster-command-center
    env -u PYTHONPATH .venv/bin/python -m pytest \\
        image_pipeline/tests/test_generator_render_health.py -q -p no:cacheprovider -m slow
"""
from __future__ import annotations

import time

import numpy as np
import pytest
from PIL import Image

import image_pipeline.methods  # noqa: F401 -- trigger @method registration
from image_pipeline.core.registry import get_all

# Generator categories covered by this contract (simulations has its own test).
GENERATOR_CATS = {"patterns", "fractals", "filters", "math_art"}
# Wirable input port types -> node needs an upstream wire to produce output.
WIRABLE_PORT_TYPES = {"image", "mask", "field", "particles"}

# Generous bound: generators at 160x160 clear a few seconds; a pathological one
# must never hang the suite. 30s is ample with margin.
PER_METHOD_TIMEOUT = 30.0

GEN_IDS = []
for mid, m in get_all().items():
    if m.category not in GENERATOR_CATS:
        continue
    in_types = {str(t).lower() for t in (m.inputs or {}).values()}
    if in_types & WIRABLE_PORT_TYPES:
        continue  # wire-dependent -- excluded
    GEN_IDS.append(mid)
GEN_IDS.sort(key=lambda x: (get_all()[x].category, x))
NAMES = {mid: get_all()[mid].name for mid in GEN_IDS}


def _timeout_for(mid: str) -> float:
    return PER_METHOD_TIMEOUT


def _readback(node_dir, raw_result):
    """Executor-faithful output extraction: return-dict image OR disk PNG.

    Returns (ok: bool, detail: str, arr_or_None).
    """
    arr = None
    # 1) return-dict / ndarray (graph.py:1255-1264)
    if isinstance(raw_result, dict):
        arr = raw_result.get("image")
    elif isinstance(raw_result, np.ndarray):
        arr = raw_result
    elif hasattr(raw_result, "mode") and hasattr(raw_result, "size"):
        try:
            arr = np.asarray(raw_result, dtype=np.float32) / 255.0
        except Exception:  # noqa: BLE001
            arr = None
    # 2) disk PNG fallback (graph.py:1266-1273 / 1304-1308)
    if arr is None:
        pngs = sorted(p for p in node_dir.glob("*.png") if not p.name.startswith("_"))
        if pngs:
            try:
                img = Image.open(str(pngs[-1])).convert("RGB")
                arr = np.asarray(img, dtype=np.float32) / 255.0
            except Exception:  # noqa: BLE001
                return False, "PNG read failed", None
    if arr is None:
        return False, "no image (return dict or PNG)", None
    arr = np.asarray(arr, dtype=np.float32)
    if arr.ndim != 3 or arr.shape[2] != 3:
        return False, f"unexpected shape {arr.shape}", None
    if arr.std() < 0.01:
        return False, f"blank output (std={arr.std():.4f})", None
    return True, "", arr


def _run_one(mid: str):
    """Subprocess worker: render one generator method at defaults."""
    from pathlib import Path as _P

    import image_pipeline.methods  # noqa: F401
    from image_pipeline.core.registry import get_all as _ga
    from image_pipeline.core import utils as U

    meta = _ga()[mid]
    node_dir = _P("/tmp/gen_health") / mid
    if node_dir.exists():
        for f in node_dir.glob("*"):
            f.unlink()
    node_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    token = U.set_canvas(160, 160)
    try:
        raw = meta.fn(node_dir, 42, params={})
    finally:
        U.reset_canvas(token)
    dt = time.time() - t0

    ok, detail, _ = _readback(node_dir, raw)
    return ok, dt, detail


pytestmark = pytest.mark.slow


@pytest.mark.parametrize("mid", GEN_IDS, ids=[f"{m}:{NAMES[m]}" for m in GEN_IDS])
def test_generator_renders_valid_frame(mid):
    """Each self-contained generator must render one valid (non-blank) frame."""
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
