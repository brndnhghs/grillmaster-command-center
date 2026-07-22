"""Route 8 — driver-path headless verification.

Proves that CHOP driver nodes (Counter/Ramp/Beats/Envelope/Counter) advance
every frame and that a driver's SCALAR output actually modulates a target
node's rendered pixels (temporal variance above the liveness floor).

Background: the GraphExecutor injects a per-frame ``_timeline`` (with
``global_frame``) into every node but does NOT inject an integer ``frame`` for
CHOP generators. The frame-based drivers previously read ``frame=0`` forever
and produced a constant output, which froze every driver-driven graph and got
it culled as static by the liveness gate. They now derive the live
frame from ``_timeline.global_frame``.

Marked ``slow`` (imports the full method registry and renders frames); it is
excluded from the default ``-m "not slow"`` run.
"""
import tempfile
from pathlib import Path

import pytest

import image_pipeline.methods  # noqa: F401  (registers all @method nodes)
from image_pipeline.core.registry import get_all, get_meta
from image_pipeline.core.graph import GraphExecutor


class FakeTL:
    def __init__(self, gf):
        self.global_frame = gf
        self.total_frames = 48


FRAME_BASED = ("__counter__", "__ramp__", "__beats__", "__envelope__")


@pytest.mark.slow
def test_frame_based_drivers_advance_per_frame():
    """Counter/Ramp/Beats/Envelope must vary when given an injected Timeline."""
    for mid in FRAME_BASED:
        meta = get_meta(mid)
        assert meta is not None, f"{mid} not registered"
        vals = []
        for f in range(24):
            out = meta.fn(Path(tempfile.mkdtemp()), 42, params={"_timeline": FakeTL(f)})
            key = "value" if "value" in out else next(iter(out))
            vals.append(float(out[key]))
        spread = max(vals) - min(vals)
        assert spread > 1e-3, f"{mid} did not vary across frames (spread={spread})"


@pytest.mark.slow
def test_driver_modulation_reaches_pixels():
    """An LFO/Counter SCALAR wire must modulate a target's rendered clip.

    Reproduces the liveness check: render a driver -> target graph
    over frames and assert the clip's temporal variance clears the floor.
    The target keeps ALL its default params so the driven param stays present
    in node.params (the executor only injects a SCALAR wire when
    edge.dst_port is in node.params).
    """
    import numpy as np

    FAST_CATS = {"patterns", "filters", "fractals", "math_art", "compositing"}
    candidates = []
    for mid, meta in get_all().items():
        if mid.startswith("__"):
            continue
        if (meta.category or "") not in FAST_CATS:
            continue
        for pname, spec in (meta.params or {}).items():
            if not isinstance(spec, dict):
                continue
            if "min" in spec or "max" in spec:
                continue
            d = spec.get("default")
            if isinstance(d, (int, float)) and not isinstance(d, bool):
                candidates.append((mid, pname))
                break

    assert candidates, "no wireable-param target found"

    FLOOR = 3e-3
    out_dir = Path(tempfile.mkdtemp(prefix="driver_int_"))
    best = None
    passed = None
    for driver_mid in ("__lfo__", "__counter__"):
        drv_params = {"waveform": "sine", "min": 0.0, "max": 1.0, "rate": 0.6} if driver_mid == "__lfo__" else {}
        for tgt_mid, tgt_param in candidates[:6]:
            tgt_meta = get_meta(tgt_mid)
            tgt_params = {k: (v.get("default") if isinstance(v, dict) else v)
                          for k, v in (tgt_meta.params or {}).items()}
            tgt_params["anim_mode"] = "none"
            nodes = [
                {"id": "0", "method_id": driver_mid, "params": drv_params},
                {"id": "1", "method_id": tgt_mid, "params": tgt_params},
            ]
            edges = [{"src_node": "0", "src_port": "value", "dst_node": "1", "dst_port": tgt_param}]
            try:
                ex = GraphExecutor(out_dir, fps=24)
                lum = []
                for fr in range(12):
                    res, _term, errs = ex.execute(nodes=nodes, edges=edges, seed=7, frame=fr, frames=24)
                    if errs:
                        break
                    img = (res.get("1", {}) or {}).get("image")
                    if img is None:
                        break
                    arr = np.array(img) if not isinstance(img, np.ndarray) else img
                    lum.append(arr.astype(np.float32).reshape(-1).mean())
                if len(lum) < 12:
                    continue
                tvar = float(np.std(lum))
                if best is None or tvar > best[0]:
                    best = (tvar, driver_mid, tgt_mid, tgt_param)
                if tvar > FLOOR and passed is None:
                    passed = (tvar, driver_mid, tgt_mid, tgt_param)
                    break
            except Exception:
                continue
        if passed is not None:
            break

    assert best is not None, "no driver->target wiring produced any output"
    assert passed is not None, (
        f"driver modulation did not reach pixels "
        f"(best tvar={best[0]:.4f} for {best[1]}->{best[2]}.{best[3]})"
    )
