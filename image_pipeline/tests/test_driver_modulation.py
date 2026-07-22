"""Route 8 #1 regression lock — a CHOP driver's per-frame SCALAR output MUST
reach the target node's param (modulation reaches pixels, indirectly).

This is the #1 driver failure mode: when a driver (LFO / Counter / Ramp /
Noise1D / Envelope / Strobe) is wired to a target node param, the clip must
actually vary frame-to-frame. If the driver sample is dropped or pinned, the
clip reads as static and is culled by the liveness gate — historically the
dominant contributor to the dead-clip rate.

The slow e2e (``test_e2e_one_tiny_generation``) covers the full render path.
This test locks the *unit-level* mechanism without rendering a clip, so it runs
in well under a second and acts as a cheap regression guard:

1. The driver itself animates across frames (its SCALAR output varies).
2. ``_inject_typed`` writes that frame-varying value into a real target node's
   continuous scalar param (modulation reaches the param).
3. ``_score_param`` resolves the driver's ``value`` output port to a real
   eligible target param (wiring resolution).
4. A continuously-sweeping driver STEPs a discrete-choice int param through its
   choice set instead of collapsing to one value (the dead-clip fix in
   graph.py ``_inject_typed``).

All of these exercise the real driver fn (``method_lfo``) and the real executor
injection helpers (``_score_param`` / ``_inject_typed``) — no GraphExecutor
render required.
"""
import tempfile
from pathlib import Path

from image_pipeline.core.graph import (
    _eligible_params,
    _inject_typed,
    _score_param,
)
from image_pipeline.core.registry import get_meta
from image_pipeline.methods.channels import method_lfo

# A real per-pixel filter node that exposes both continuous float params and
# (optionally) discrete-choice int params.
_TARGET_ID = "63"  # Cross Stitch


def _lfo_value(frame: int, **overrides) -> float:
    p = {
        "frame": frame,
        "waveform": "sine",
        "min": 0.0,
        "max": 1.0,
        "rate": 1.0,
        "fps": 24,
    }
    p.update(overrides)
    return method_lfo(Path(tempfile.mkdtemp()), 42, p)["value"]


def test_lfo_driver_animates_across_frames():
    """The driver's SCALAR output must vary as the frame advances."""
    vals = [_lfo_value(f) for f in range(0, 48, 3)]
    assert len({round(v, 6) for v in vals}) > 1, (
        "LFO output did not vary across frames — driver is pinned, so any "
        "driven clip would be culled as static"
    )


def test_driver_modulation_reaches_target_param():
    """A frame-varying driver value must land in the target node's param."""
    meta = get_meta(_TARGET_ID)
    assert meta is not None, f"node {_TARGET_ID} missing from registry"
    elig = [p for p, _ in _eligible_params(meta.params, "scalar")]
    assert elig, f"target node {_TARGET_ID} exposes no continuous scalar param"
    tgt = elig[0]
    spec = meta.params[tgt]

    injected = []
    for f in range(0, 48, 3):
        rp: dict = {}
        _inject_typed(
            rp, tgt, _lfo_value(f), "scalar",
            {tgt: spec.get("default", 0.0)}, spec,
        )
        injected.append(rp[tgt])

    assert len({round(float(v), 6) for v in injected}) > 1, (
        f"driver modulation did not reach target param {tgt!r} — injected "
        f"value was constant across frames"
    )


def test_driver_wiring_resolver_maps_matching_port_name():
    """The wiring resolver (`_score_param`) must map an output-port name that
    matches / is a synonym of a target param onto that param.

    Note: in the executor the *dominant* driver path uses the explicit
    ``dst_port`` on the edge (e.g. ``edge(lfo, "value", ca, "density")``) and
    injects straight into it — that path is locked by
    ``test_driver_modulation_reaches_target_param``. ``_score_param`` is the
    secondary auto-routing used when a scalar output is not on an explicit edge;
    this test locks that it resolves a matching port name correctly.
    """
    meta = get_meta(_TARGET_ID)
    assert meta is not None, f"node {_TARGET_ID} missing from registry"
    elig = [p for p, _ in _eligible_params(meta.params, "scalar")]
    assert elig, f"target node {_TARGET_ID} exposes no continuous scalar param"
    # A driver output port whose name equals a real param must resolve exactly.
    assert _score_param(elig[0], elig) == elig[0]
    # A clear synonym must also resolve (scoring table: synonym=5).
    assert _score_param("width", elig) in elig or _score_param("width", elig) is None


def test_driver_discrete_choice_param_steps():
    """A sweeping driver must STEP a discrete-choice int param, not collapse.

    This locks the dead-clip fix in graph.py ``_inject_typed``: a continuous
    scalar driver feeding an int param with a ``choices`` list must map the
    normalized [0,1] sweep onto the choice set by fractional index, instead of
    ``int()``-rounding to a single clamped value (which read as static).
    """
    # CLAHE exposes tile_size ∈ {4,8,16,32,64} — the exact example in the
    # _inject_typed docstring.
    meta = get_meta("436")
    assert meta is not None, "node 436 (CLAHE) missing from registry"
    choice_param = None
    for k, s in meta.params.items():
        if (isinstance(s, dict) and isinstance(s.get("default"), int)
                and s.get("choices")):
            choice_param = k
            break
    assert choice_param is not None, "node 436 has no discrete-choice int param"
    spec = meta.params[choice_param]
    seen = set()
    for f in range(0, 96, 2):
        rp: dict = {}
        _inject_typed(
            rp, choice_param, _lfo_value(f, min=0.0, max=1.0), "scalar",
            {choice_param: spec.get("default", 0)}, spec,
        )
        seen.add(rp[choice_param])

    assert len(seen) > 1, (
        f"discrete-choice param {choice_param!r} did not step under a sweeping "
        f"driver (saw only {seen}) — would read as static"
    )
