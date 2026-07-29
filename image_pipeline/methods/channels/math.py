"""CHOP-like channel generator nodes.
Auto-imported by channels/__init__.py.
"""
from __future__ import annotations
import math
import random
from pathlib import Path
import numpy as np
from ...core.registry import method
from ...core.utils import seed_all

# ── Concrete op list (used by both @method choices and the body) ────────

_OPS = [
    # ── Arithmetic ──────────────────────────────────────────────────────
    "add", "sub", "mul", "div", "mod", "pow",
    "min", "max", "average", "sum", "product", "difference",
    # ── Unary ────────────────────────────────────────────────────────────
    "abs", "round", "floor", "ceil", "negate", "reciprocal", "sign",
    # ── Range operations ─────────────────────────────────────────────────
    "normalize", "fit", "scale", "bias", "gain", "offset",
    "wrap", "fold", "mirror", "quantize", "snap",
    # ── Response shaping ─────────────────────────────────────────────────
    "gamma", "exponential", "logarithmic", "smoothstep", "smootherstep",
    "ease_in", "ease_out", "ease_inout",
    # ── Range-map (same inputs as map_range) ─────────────────────────────
    "map_range", "clamp",
    # ── Logic-as-math ────────────────────────────────────────────────────
    "threshold", "range_gate", "dead_zone", "soft_threshold",
    # ── Multi-stream ─────────────────────────────────────────────────────
    "weighted_avg", "crossfade", "distance", "magnitude", "dot_product",
]

# ── Per-operation layout: which params and inputs to show for each op ──
# When an op is absent from this map, ALL params and ALL inputs are shown.
_OP_LAYOUTS: dict[str, dict] = {
    # Binary arithmetic ops: show both inputs
    **{op: {"show_params": ["a_default", "b_default"], "show_inputs": ["a", "b"]}
       for op in ["add", "sub", "mul", "div", "mod", "pow", "min", "max",
                  "average", "sum", "product", "difference", "distance",
                  "magnitude", "dot_product"]},
    # Unary ops: only input A
    **{op: {"show_params": ["a_default"], "show_inputs": ["a"]}
       for op in ["abs", "round", "floor", "ceil", "negate", "reciprocal", "sign",
                  "smoothstep", "smootherstep", "wrap", "fold"]},
    # Range ops
    "normalize":      {"show_params": ["a_default", "map_src_min", "map_src_max"], "show_inputs": ["a"]},
    "fit":            {"show_params": ["a_default", "b_default", "map_src_min", "map_src_max",
                                       "map_dst_min", "map_dst_max"], "show_inputs": ["a", "b"]},
    "scale":          {"show_params": ["a_default", "scale_factor"], "show_inputs": ["a"]},
    "bias":           {"show_params": ["a_default", "bias_amount"], "show_inputs": ["a"]},
    "offset":         {"show_params": ["a_default", "bias_amount"], "show_inputs": ["a"]},
    "gain":           {"show_params": ["a_default", "gain_amount"], "show_inputs": ["a"]},
    "mirror":         {"show_params": ["a_default", "bias_amount"], "show_inputs": ["a"]},
    "quantize":       {"show_params": ["a_default", "quantize_step"], "show_inputs": ["a"]},
    "snap":           {"show_params": ["a_default", "snap_targets"], "show_inputs": ["a"]},
    # Response shaping
    "gamma":          {"show_params": ["a_default", "curve_power"], "show_inputs": ["a"]},
    "exponential":    {"show_params": ["a_default", "exp_rate"], "show_inputs": ["a"]},
    "logarithmic":    {"show_params": ["a_default", "log_base"], "show_inputs": ["a"]},
    "ease_in":        {"show_params": ["a_default", "curve_power"], "show_inputs": ["a"]},
    "ease_out":       {"show_params": ["a_default", "curve_power"], "show_inputs": ["a"]},
    "ease_inout":     {"show_params": ["a_default", "curve_power"], "show_inputs": ["a"]},
    # Range-map
    "map_range":      {"show_params": ["a_default", "b_default", "map_src_min", "map_src_max",
                                       "map_dst_min", "map_dst_max"], "show_inputs": ["a", "b"]},
    "clamp":          {"show_params": ["a_default", "clamp_min", "clamp_max"], "show_inputs": ["a"]},
    # Logic-as-math
    "threshold":      {"show_params": ["a_default", "threshold_val", "on_value", "off_value"], "show_inputs": ["a"]},
    "range_gate":     {"show_params": ["a_default", "threshold_lo", "threshold_hi", "on_value", "off_value"], "show_inputs": ["a"]},
    "dead_zone":      {"show_params": ["a_default", "threshold_lo", "threshold_hi"], "show_inputs": ["a"]},
    "soft_threshold": {"show_params": ["a_default", "threshold_val", "softness"], "show_inputs": ["a"]},
    # Multi-stream
    "weighted_avg":   {"show_params": ["a_default", "b_default", "weight_a", "weight_b"], "show_inputs": ["a", "b"]},
    "crossfade":      {"show_params": ["a_default", "b_default", "crossfade_mix"], "show_inputs": ["a", "b"]},
}


@method(id="__math__", name="Math", category="channels",
        tags=["chop", "math", "operator"],
        inputs={"a": "SCALAR", "b": "SCALAR"},
        outputs={"value": "SCALAR"},
        op_layouts=_OP_LAYOUTS,
        runtime={
            "value": {
                "type": "numeric",
                "label": "Value",
                "observable": True,
            }
        },
        signal={
            "a": "numeric",
            "b": "numeric",
            "value": "output",
        },
        params={
            "operation": {
                "description": "math operation",
                "choices": _OPS,
                "default": "add",
            },
            "a_default": {"description": "default value for input A when not wired", "default": 0.0},
            "b_default": {"description": "default value for input B when not wired", "default": 1.0},
            # ── Range-map params (also used by normalize / fit) ──────────
            "map_src_min": {"description": "map_range/fit: source range min", "default": 0.0},
            "map_src_max": {"description": "map_range/fit: source range max", "default": 1.0},
            "map_dst_min": {"description": "map_range/fit: destination range min", "default": 0.0},
            "map_dst_max": {"description": "map_range/fit: destination range max", "default": 1.0},
            "clamp_min": {"description": "clamp: minimum value", "default": 0.0},
            "clamp_max": {"description": "clamp: maximum value", "default": 1.0},
            # ── Scale / bias / gain / offset ─────────────────────────────
            "scale_factor": {"description": "scale: multiplier", "default": 1.0},
            "bias_amount": {"description": "bias/offset: additive constant", "default": 0.0},
            "gain_amount": {"description": "gain: response-shape amount", "default": 0.5},
            # ── Quantize / snap ──────────────────────────────────────────
            "quantize_step": {"description": "quantize: step size", "default": 0.1},
            "snap_targets": {"description": "snap: comma-separated snap values e.g. 0,0.5,1", "default": "0,0.5,1"},
            # ── Response shaping ─────────────────────────────────────────
            "curve_power": {"description": "gamma/ease: curve exponent", "default": 2.0},
            "exp_rate": {"description": "exponential: growth rate", "default": 2.0},
            "log_base": {"description": "logarithmic: base", "default": 2.0},
            # ── Logic-as-math ────────────────────────────────────────────
            "threshold_val": {"description": "threshold/gate: threshold level", "default": 0.5},
            "threshold_lo": {"description": "range_gate/dead_zone: low bound", "default": 0.3},
            "threshold_hi": {"description": "range_gate/dead_zone: high bound", "default": 0.7},
            "softness": {"description": "soft_threshold: rolloff width", "default": 0.1},
            "on_value": {"description": "threshold: output when condition met", "default": 1.0},
            "off_value": {"description": "threshold: output when condition not met", "default": 0.0},
            # ── Multi-stream ─────────────────────────────────────────────
            "weight_a": {"description": "weighted_avg: weight for input A", "default": 0.5},
            "weight_b": {"description": "weighted_avg: weight for input B", "default": 0.5},
            "crossfade_mix": {"description": "crossfade: blend 0=A … 1=B", "default": 0.5},
        },
        is_time_varying=False,
        description=(
            "Polymorphic math operator — 45+ operations including arithmetic, "
            "range ops, response shaping, logic-as-math, and multi-stream blends."
        ),
    )
def method_math(out_dir: Path, seed: int, params=None):
    """Math operations on two SCALAR inputs with 45+ modes.

    Accepts wired SCALAR inputs A and B, with fallback defaults.
    Many unary ops (abs, floor, sqrt, log, etc.) operate on input A only.

    Outputs:
        value (SCALAR): result of the operation
    """
    if params is None:
        params = {}
    seed_all(seed)

    op = params.get("operation", "add")
    a_raw = params.get("a")
    b_raw = params.get("b")

    # Fallback defaults (used when the port is unwired)
    a = float(a_raw) if a_raw is not None else float(params.get("a_default", 0.0))
    b = float(b_raw) if b_raw is not None else float(params.get("b_default", 1.0))

    # ── Helper functions ────────────────────────────────────────────────

    def _smoothstep(t: float) -> float:
        t = max(0.0, min(1.0, t))
        return t * t * (3.0 - 2.0 * t)

    def _smootherstep(t: float) -> float:
        t = max(0.0, min(1.0, t))
        return t * t * t * (t * (t * 6.0 - 15.0) + 10.0)

    def _wrap(v: float) -> float:
        span = 1.0
        if span <= 0:
            return v
        return v - span * math.floor(v / span)

    def _fold(v: float, center: float = 0.5) -> float:
        """Fold mirrors each time it crosses a boundary at distance `center` from edges."""
        half = center
        if half <= 0:
            return v
        # Reflect around multiples of half
        ratio = v / half
        if int(ratio) % 2 == 0:
            return v - half * math.floor(ratio)
        else:
            return half - (v - half * math.floor(ratio))

    # ── Delegate ────────────────────────────────────────────────────────

    if op == "add":
        val = a + b
    elif op == "sub":
        val = a - b
    elif op == "mul":
        val = a * b
    elif op == "div":
        val = a / b if b != 0 else 0.0
    elif op == "mod":
        val = a % b if b != 0 else 0.0
    elif op == "pow":
        val = a ** b
    elif op == "min":
        val = min(a, b)
    elif op == "max":
        val = max(a, b)
    elif op == "average":
        val = (a + b) * 0.5
    elif op == "sum":
        val = a + b
    elif op == "product":
        val = a * b
    elif op == "difference":
        val = abs(a - b)

    # ── Unary ───────────────────────────────────────────────────────────
    elif op == "abs":
        val = abs(a)
    elif op == "round":
        val = round(a)
    elif op == "floor":
        val = math.floor(a)
    elif op == "ceil":
        val = math.ceil(a)
    elif op == "negate":
        val = -a
    elif op == "reciprocal":
        val = 1.0 / a if a != 0 else 0.0
    elif op == "sign":
        val = float((a > 0) - (a < 0))  # 1.0 / -1.0 / 0.0

    # ── Range operations ────────────────────────────────────────────────
    elif op == "normalize":
        src_min = float(params.get("map_src_min", 0.0))
        src_max = float(params.get("map_src_max", 1.0))
        if src_max > src_min:
            val = (a - src_min) / (src_max - src_min)
        else:
            val = 0.0
    elif op == "fit":
        src_min = float(params.get("map_src_min", 0.0))
        src_max = float(params.get("map_src_max", 1.0))
        dst_min = float(params.get("map_dst_min", 0.0))
        dst_max = float(params.get("map_dst_max", 1.0))
        if src_max != src_min:
            norm = (a - src_min) / (src_max - src_min)
        else:
            norm = 0.0
        val = dst_min + norm * (dst_max - dst_min)
    elif op == "scale":
        factor = float(params.get("scale_factor", 1.0))
        val = a * factor
    elif op == "bias" or op == "offset":
        amount = float(params.get("bias_amount", 0.0))
        val = a + amount
    elif op == "gain":
        amount = float(params.get("gain_amount", 0.5))
        # Classic audio gain: shape the slope at different levels
        amount = max(0.01, min(0.99, amount))
        if a <= 0.5:
            val = (a / 2.0) / (0.5 / amount) if amount > 0 else 0.0
        else:
            val = 1.0 - ((1.0 - a) / 2.0) / (0.5 / (1.0 - amount)) if amount < 1 else 1.0
        val = max(0.0, min(1.0, val))
    elif op == "wrap":
        val = _wrap(a)
    elif op == "fold":
        val = _fold(a)
    elif op == "mirror":
        # Reflect around a center point
        center = float(params.get("bias_amount", 0.5))
        val = center + (center - a) if a > center else a
    elif op == "quantize":
        step = max(1e-6, float(params.get("quantize_step", 0.1)))
        val = round(a / step) * step
    elif op == "snap":
        raw = params.get("snap_targets", "0,0.5,1")
        if isinstance(raw, str):
            targets = [float(x.strip()) for x in raw.split(",") if x.strip()]
        else:
            targets = [float(x) for x in raw] if isinstance(raw, (list, tuple)) else [0.0, 0.5, 1.0]
        if not targets:
            val = a
        else:
            val = min(targets, key=lambda t: abs(a - t))

    # ── Response shaping ────────────────────────────────────────────────
    elif op == "gamma":
        g = float(params.get("curve_power", 2.0))
        t = max(0.0, min(1.0, a))
        val = t ** g
    elif op == "exponential":
        rate = float(params.get("exp_rate", 2.0))
        t = max(0.0, min(1.0, a))
        if rate > 0:
            val = (math.exp(t * rate) - 1.0) / (math.exp(rate) - 1.0)
        else:
            val = t
    elif op == "logarithmic":
        base = float(params.get("log_base", 2.0))
        t = max(1e-10, min(1.0, a))
        if base > 1:
            val = math.log(t * (base - 1.0) + 1.0) / math.log(base)
        else:
            val = t
    elif op == "smoothstep":
        val = _smoothstep(a)
    elif op == "smootherstep":
        val = _smootherstep(a)
    elif op == "ease_in":
        p = float(params.get("curve_power", 2.0))
        t = max(0.0, min(1.0, a))
        val = t ** p
    elif op == "ease_out":
        p = float(params.get("curve_power", 2.0))
        t = max(0.0, min(1.0, a))
        val = 1.0 - (1.0 - t) ** p
    elif op == "ease_inout":
        p = float(params.get("curve_power", 2.0))
        t = max(0.0, min(1.0, a))
        if t < 0.5:
            val = (2.0 * t) ** p / 2.0
        else:
            val = 1.0 - (2.0 * (1.0 - t)) ** p / 2.0

    # ── Range-map ───────────────────────────────────────────────────────
    elif op == "map_range":
        src_min = float(params.get("map_src_min", 0.0))
        src_max = float(params.get("map_src_max", 1.0))
        dst_min = float(params.get("map_dst_min", 0.0))
        dst_max = float(params.get("map_dst_max", 1.0))
        if src_max != src_min:
            norm = (a - src_min) / (src_max - src_min)
        else:
            norm = 0.0
        val = dst_min + norm * (dst_max - dst_min)
    elif op == "clamp":
        cmin = float(params.get("clamp_min", 0.0))
        cmax = float(params.get("clamp_max", 1.0))
        val = max(cmin, min(cmax, a))

    # ── Logic-as-math ───────────────────────────────────────────────────
    elif op == "threshold":
        thresh = float(params.get("threshold_val", 0.5))
        on_v = float(params.get("on_value", 1.0))
        off_v = float(params.get("off_value", 0.0))
        val = on_v if a >= thresh else off_v
    elif op == "range_gate":
        lo = float(params.get("threshold_lo", 0.3))
        hi = float(params.get("threshold_hi", 0.7))
        on_v = float(params.get("on_value", 1.0))
        off_v = float(params.get("off_value", 0.0))
        val = on_v if lo <= a <= hi else off_v
    elif op == "dead_zone":
        lo = float(params.get("threshold_lo", 0.3))
        hi = float(params.get("threshold_hi", 0.7))
        # Output 0 in dead zone, pass through otherwise
        if lo <= a <= hi:
            val = 0.0
        elif a < lo:
            val = a  # or could remap
        else:
            val = a
    elif op == "soft_threshold":
        thresh = float(params.get("threshold_val", 0.5))
        softness = float(params.get("softness", 0.1))
        if softness <= 0:
            val = a if a >= thresh else 0.0
        else:
            # Smooth roll-on around threshold
            t = (a - thresh) / softness
            t = max(0.0, min(1.0, t))
            val = a * _smoothstep(t)

    # ── Multi-stream ────────────────────────────────────────────────────
    elif op == "weighted_avg":
        wa = float(params.get("weight_a", 0.5))
        wb = float(params.get("weight_b", 0.5))
        total = wa + wb
        if total == 0:
            total = 1.0
        val = (a * wa + b * wb) / total
    elif op == "crossfade":
        mix = max(0.0, min(1.0, float(params.get("crossfade_mix", 0.5))))
        val = a * (1.0 - mix) + b * mix
    elif op == "distance":
        val = abs(a - b)
    elif op == "magnitude":
        val = math.sqrt(a * a + b * b)
    elif op == "dot_product":
        val = a * b
    else:
        val = 0.0

    return {"value": float(val)}
