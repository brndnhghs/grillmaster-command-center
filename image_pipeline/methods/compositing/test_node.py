"""Test Node — graph-level connection tester.

Outputs known test patterns for every data type (IMAGE, FIELD, PARTICLES, MASK, SCALAR)
so you can wire it to any node and verify connections pass data correctly.

When wired as an input, it reads the incoming data and reports what it received
(type, shape, value range, sample values) so you can see if a node is actually
producing valid output on its ports.

Supports animation modes for testing temporal continuity and FIELD-driven params.

Usage:
  1. Drop a Test Node into the graph
  2. Wire its outputs to the node you want to test
  3. Wire the node's outputs back to the Test Node's inputs
  4. Run the graph — the Test Node writes a detailed report to its output dir
  5. Click the Test Node to see the report in the params panel
"""
from __future__ import annotations
import json
import math
from pathlib import Path

import numpy as np

from ...core.registry import method
from ...core.utils import W, H
from ...core.animation import capture_frame


@method(
    id="__test__",
    name="Test Node",
    category="compositing",
    tags=["test", "debug", "diagnostic"],
    inputs={
        "image_in": "IMAGE",
        "scalar_in": "SCALAR",
        "field_in": "FIELD",
        "particles_in": "PARTICLES",
        "mask_in": "MASK",
        "anim_speed": "SCALAR",
    },
    outputs={
        "image": "IMAGE",
        "luminance": "FIELD",
        "field": "FIELD",
        "particles": "PARTICLES",
        "mask": "MASK",
        "test_scalar": "SCALAR",
    },
    params={
        "test_pattern": {
            "description": "output image pattern",
            "default": "color_bars",
            "choices": [
                "color_bars", "checkerboard", "gradient_h", "gradient_v",
                "white", "black", "noise", "color_ramp",
            ],
        },
        "field_pattern": {
            "description": "output field pattern",
            "default": "sine_gradient",
            "choices": ["sine_gradient", "gaussian", "checkerboard", "constant"],
        },
        "particle_count": {
            "description": "number of test particles to emit",
            "min": 0, "max": 10000, "default": 100,
        },
        "mask_pattern": {
            "description": "output mask pattern",
            "default": "circle",
            "choices": ["circle", "checkerboard", "gradient", "solid"],
        },
        "scalar_value": {
            "description": "test scalar output value",
            "default": 1.0,
        },
        "scalar_value_b": {
            "description": "second test scalar output",
            "default": 25.0,
        },
        "anim_mode": {
            "description": "animation mode for testing temporal continuity",
            "choices": ["none", "pattern_morph", "scalar_sweep", "field_rotate"],
            "default": "none",
        },
        "anim_speed": {
            "description": "animation speed multiplier (can be driven by SCALAR)",
            "min": 0.0, "max": 5.0, "default": 0.5,
        },
    },
)
def method_test_node(out_dir: Path, seed: int, params=None):
    """Test Node — graph-level connection tester.

    Generates test patterns for every data type and reports what inputs
    were received. Supports animation modes for testing temporal continuity.

    Returns:
        dict with "image", "field", "particles", "mask", "test_scalar", "test_scalar_b"
    """
    if params is None:
        params = {}

    t = float(params.get("time", 0.0))
    anim_mode = params.get("anim_mode", "none")
    anim_speed_field = params.get("_field_anim_speed")
    if anim_speed_field is not None:
        anim_speed = float(np.mean(anim_speed_field))
    else:
        anim_speed = float(params.get("anim_speed", 0.5))

    # ── Read wired inputs ──────────────────────────────────────────
    report = {"inputs": {}, "outputs": {}}

    # Check for wired image input (new contract: _input_image)
    input_img = params.get("_input_image")
    if input_img is not None:
        report["inputs"]["image_in"] = {
            "connected": True,
            "shape": list(input_img.shape),
            "dtype": str(input_img.dtype),
            "mean": round(float(np.mean(input_img)), 4),
            "min": round(float(np.min(input_img)), 4),
            "max": round(float(np.max(input_img)), 4),
            "std": round(float(np.std(input_img)), 4),
        }
    else:
        report["inputs"]["image_in"] = {"connected": False}

    # Check for wired scalar input
    scalar_val = params.get("scalar_in")
    if scalar_val is not None:
        report["inputs"]["scalar_in"] = {
            "connected": True,
            "value": float(scalar_val),
            "type": type(scalar_val).__name__,
        }
    else:
        report["inputs"]["scalar_in"] = {"connected": False}

    # Check for wired field input (new contract: _field_field_in)
    field_val = params.get("_field_field_in")
    if field_val is not None:
        report["inputs"]["field_in"] = {
            "connected": True,
            "shape": list(field_val.shape),
            "dtype": str(field_val.dtype),
            "mean": round(float(np.mean(field_val)), 4),
            "min": round(float(np.min(field_val)), 4),
            "max": round(float(np.max(field_val)), 4),
        }
    else:
        report["inputs"]["field_in"] = {"connected": False}

    # Check for wired particles input
    particles_val = params.get("particles_in")
    if particles_val is not None:
        try:
            parr = np.asarray(particles_val)
            report["inputs"]["particles_in"] = {
                "connected": True,
                "shape": list(parr.shape),
                "dtype": str(parr.dtype),
                "count": int(parr.shape[0]) if parr.ndim > 0 else 0,
            }
        except Exception as e:
            report["inputs"]["particles_in"] = {"connected": True, "error": str(e)[:200]}
    else:
        report["inputs"]["particles_in"] = {"connected": False}

    # Check for wired mask input
    mask_val = params.get("mask_in")
    if mask_val is not None:
        try:
            marr = np.asarray(mask_val)
            report["inputs"]["mask_in"] = {
                "connected": True,
                "shape": list(marr.shape),
                "dtype": str(marr.dtype),
                "mean": round(float(np.mean(marr)), 4),
                "min": round(float(np.min(marr)), 4),
                "max": round(float(np.max(marr)), 4),
            }
        except Exception as e:
            report["inputs"]["mask_in"] = {"connected": True, "error": str(e)[:200]}
    else:
        report["inputs"]["mask_in"] = {"connected": False}

    # ── Animation: effective params ──
    effective_pattern = params.get("test_pattern", "color_bars")
    effective_field = params.get("field_pattern", "sine_gradient")
    effective_scalar = float(params.get("scalar_value", 1.0))
    scalar_b = float(params.get("scalar_value_b", 25.0))
    morph_fade = 0.0
    pattern_b = effective_pattern
    field_b = effective_field

    if anim_mode == "pattern_morph":
        choices = ["color_bars", "checkerboard", "gradient_h", "gradient_v", "noise", "color_ramp"]
        n = len(choices)
        raw_idx = (t / (2 * math.pi)) * n * anim_speed
        idx_a = int(raw_idx) % n
        idx_b = (idx_a + 1) % n
        morph_fade = raw_idx - int(raw_idx)
        effective_pattern = choices[idx_a]
        pattern_b = choices[idx_b]

    elif anim_mode == "scalar_sweep":
        # Lerp between scalar_value and scalar_value_b on test_scalar output
        t_norm = 0.5 + 0.5 * math.sin(t * anim_speed)
        effective_scalar = effective_scalar + (scalar_b - effective_scalar) * t_norm

    elif anim_mode == "field_rotate":
        fchoices = ["sine_gradient", "gaussian", "checkerboard", "constant"]
        n = len(fchoices)
        raw_idx = (t / (2 * math.pi)) * n * anim_speed
        idx_a = int(raw_idx) % n
        idx_b = (idx_a + 1) % n
        morph_fade = raw_idx - int(raw_idx)
        effective_field = fchoices[idx_a]
        field_b = fchoices[idx_b]

    # ── Generate output image ─────────────────────────────────────
    img_arr = _make_image_pattern(effective_pattern, seed)
    if anim_mode == "pattern_morph" and morph_fade > 0.0:
        img_b = _make_image_pattern(pattern_b, seed + 1)
        img_arr = (1.0 - morph_fade) * img_arr + morph_fade * img_b

    report["outputs"]["image"] = {
        "pattern": effective_pattern,
        "shape": list(img_arr.shape),
        "mean": round(float(np.mean(img_arr)), 4),
    }

    # ── Generate output field ─────────────────────────────────────
    field_arr = _make_field_pattern(effective_field, seed)
    if anim_mode == "field_rotate" and morph_fade > 0.0:
        field_b_arr = _make_field_pattern(field_b, seed + 1)
        field_arr = (1.0 - morph_fade) * field_arr + morph_fade * field_b_arr

    report["outputs"]["field"] = {
        "pattern": effective_field,
        "shape": list(field_arr.shape),
        "mean": round(float(np.mean(field_arr)), 4),
        "min": round(float(np.min(field_arr)), 4),
        "max": round(float(np.max(field_arr)), 4),
    }

    # ── Generate output particles ──────────────────────────────────
    pcount = int(params.get("particle_count", 100))
    particles = _make_particles(pcount, seed)

    report["outputs"]["particles"] = {
        "count": pcount,
        "shape": list(particles.shape),
        "dtype": str(particles.dtype),
    }

    # ── Generate output mask ──────────────────────────────────────
    mask_pattern = params.get("mask_pattern", "circle")
    mask_arr = _make_mask_pattern(mask_pattern, seed)

    report["outputs"]["mask"] = {
        "pattern": mask_pattern,
        "shape": list(mask_arr.shape),
        "mean": round(float(np.mean(mask_arr)), 4),
    }

    # ── Write report ─────────────────────────────────────────────
    (out_dir / "test_report.json").write_text(json.dumps(report, indent=2))

    # ── Live preview ──────────────────────────────────────────────
    capture_frame("__test__", img_arr)

    # ── Return dict matching declared outputs ─────────────────────
    return {
        "image": img_arr,
        "field": field_arr,
        "particles": particles,
        "mask": mask_arr,
        "test_scalar": effective_scalar,
    }


def _make_image_pattern(pattern: str, seed: int) -> np.ndarray:
    """Generate a test image pattern."""
    rng = np.random.RandomState(seed)
    arr = np.zeros((H, W, 3), dtype=np.float32)

    if pattern == "color_bars":
        bars = 8
        bw = W // bars
        colors = [
            (1.0, 1.0, 1.0), (1.0, 1.0, 0.0), (0.0, 1.0, 1.0), (0.0, 1.0, 0.0),
            (1.0, 0.0, 1.0), (1.0, 0.0, 0.0), (0.0, 0.0, 1.0), (0.0, 0.0, 0.0),
        ]
        for i, (r, g, b) in enumerate(colors):
            arr[:, i * bw : (i + 1) * bw] = [r, g, b]
    elif pattern == "checkerboard":
        cells = 8
        cw, ch = W // cells, H // cells
        for y in range(cells):
            for x in range(cells):
                val = 1.0 if (x + y) % 2 == 0 else 0.0
                arr[y * ch : (y + 1) * ch, x * cw : (x + 1) * cw] = val
    elif pattern == "gradient_h":
        arr[:, :, 0] = np.linspace(0, 1, W)[np.newaxis, :]
        arr[:, :, 1] = np.linspace(0, 1, W)[np.newaxis, :]
        arr[:, :, 2] = np.linspace(0, 1, W)[np.newaxis, :]
    elif pattern == "gradient_v":
        arr[:, :, 0] = np.linspace(0, 1, H)[:, np.newaxis]
        arr[:, :, 1] = np.linspace(0, 1, H)[:, np.newaxis]
        arr[:, :, 2] = np.linspace(0, 1, H)[:, np.newaxis]
    elif pattern == "white":
        arr.fill(1.0)
    elif pattern == "black":
        arr.fill(0.0)
    elif pattern == "noise":
        arr = rng.rand(H, W, 3).astype(np.float32)
    elif pattern == "color_ramp":
        for c in range(3):
            arr[:, :, c] = np.linspace(0, 1, W)[np.newaxis, :] * (c + 1) / 3.0

    return arr


def _make_field_pattern(pattern: str, seed: int) -> np.ndarray:
    """Generate a test field array."""
    rng = np.random.RandomState(seed)
    if pattern == "sine_gradient":
        x = np.linspace(0, 4 * np.pi, W)
        y = np.linspace(0, 4 * np.pi, H)
        xx, yy = np.meshgrid(x, y)
        return np.sin(xx) * np.cos(yy * 0.7).astype(np.float32)
    elif pattern == "gaussian":
        cx, cy = W / 2, H / 2
        x = np.arange(W)
        y = np.arange(H)
        xx, yy = np.meshgrid(x, y)
        return np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * 80**2)).astype(np.float32)
    elif pattern == "checkerboard":
        cells = 8
        cw, ch = W // cells, H // cells
        arr = np.zeros((H, W), dtype=np.float32)
        for y in range(cells):
            for x in range(cells):
                val = 1.0 if (x + y) % 2 == 0 else -1.0
                arr[y * ch : (y + 1) * ch, x * cw : (x + 1) * cw] = val
        return arr
    elif pattern == "constant":
        return np.full((H, W), 0.5, dtype=np.float32)
    return np.zeros((H, W), dtype=np.float32)


def _make_particles(count: int, seed: int) -> np.ndarray:
    """Generate test particles: [x, y, vx, vy]."""
    rng = np.random.RandomState(seed)
    particles = np.zeros((count, 4), dtype=np.float32)
    particles[:, 0] = rng.rand(count).astype(np.float32) * W
    particles[:, 1] = rng.rand(count).astype(np.float32) * H
    particles[:, 2] = (rng.rand(count).astype(np.float32) - 0.5) * 4
    particles[:, 3] = (rng.rand(count).astype(np.float32) - 0.5) * 4
    return particles


def _make_mask_pattern(pattern: str, seed: int) -> np.ndarray:
    """Generate a test mask."""
    rng = np.random.RandomState(seed)
    mask = np.zeros((H, W), dtype=np.float32)
    if pattern == "circle":
        cx, cy, r = W / 2, H / 2, min(W, H) / 3
        x = np.arange(W)
        y = np.arange(H)
        xx, yy = np.meshgrid(x, y)
        mask = ((xx - cx) ** 2 + (yy - cy) ** 2 <= r**2).astype(np.float32)
    elif pattern == "checkerboard":
        cells = 8
        cw, ch = W // cells, H // cells
        for y in range(cells):
            for x in range(cells):
                val = 1.0 if (x + y) % 2 == 0 else 0.0
                mask[y * ch : (y + 1) * ch, x * cw : (x + 1) * cw] = val
    elif pattern == "gradient":
        mask = np.linspace(0, 1, W)[np.newaxis, :].repeat(H, axis=0).astype(np.float32)
    elif pattern == "solid":
        mask.fill(0.5)
    return mask
