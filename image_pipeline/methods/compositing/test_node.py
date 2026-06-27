"""Test Node — graph-level connection tester.

Outputs known test patterns for every data type (IMAGE, SCALAR, FIELD, PARTICLES, MASK)
so you can wire it to any node and verify connections pass data correctly.

When wired as an input, it reads the incoming data and reports what it received
(type, shape, value range, sample values) so you can see if a node is actually
producing valid output on its ports.

Usage:
  1. Drop a Test Node into the graph
  2. Wire its outputs to the node you want to test
  3. Wire the node's outputs back to the Test Node's inputs
  4. Run the graph — the Test Node writes a detailed report to its output dir
  5. Click the Test Node to see the report in the params panel
"""

from __future__ import annotations
import json
from pathlib import Path

import numpy as np
from PIL import Image

from ...core.registry import method
from ...core.utils import save, mn, write_scalars, write_field, W, H


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
    },
    outputs={
        "image": "IMAGE",
        "luminance": "SCALAR",
        "field": "FIELD",
        "particles": "PARTICLES",
        "mask": "MASK",
        "test_scalar": "SCALAR",
        "test_scalar_b": "SCALAR",
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
            "min": 0,
            "max": 10000,
            "default": 100,
        },
        "mask_pattern": {
            "description": "output mask pattern",
            "default": "circle",
            "choices": ["circle", "checkerboard", "gradient", "solid"],
        },
        "scalar_value": {
            "description": "test scalar output value",
            "min": -1000,
            "max": 1000,
            "default": 42.0,
        },
        "scalar_value_b": {
            "description": "second test scalar output",
            "min": -1000,
            "max": 1000,
            "default": 3.14159,
        },
    },
)
def method_test_node(out_dir: Path, seed: int, params=None):
    if params is None:
        params = {}

    # ── Read wired inputs ──────────────────────────────────────────
    report = {
        "inputs": {},
        "outputs": {},
    }

    # Check for wired image input
    input_image_path = params.get("input_image", "")
    if input_image_path:
        try:
            img = Image.open(input_image_path).convert("RGB")
            arr = np.array(img, dtype=np.float32) / 255.0
            report["inputs"]["image_in"] = {
                "connected": True,
                "shape": list(arr.shape),
                "dtype": str(arr.dtype),
                "mean": round(float(np.mean(arr)), 4),
                "min": round(float(np.min(arr)), 4),
                "max": round(float(np.max(arr)), 4),
                "std": round(float(np.std(arr)), 4),
            }
        except Exception as e:
            report["inputs"]["image_in"] = {"connected": True, "error": str(e)[:200]}
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

    # Check for wired field input
    field_a_path = params.get("field_a_path", "")
    if field_a_path:
        try:
            farr = np.load(field_a_path)
            report["inputs"]["field_in"] = {
                "connected": True,
                "shape": list(farr.shape),
                "dtype": str(farr.dtype),
                "mean": round(float(np.mean(farr)), 4),
                "min": round(float(np.min(farr)), 4),
                "max": round(float(np.max(farr)), 4),
            }
        except Exception as e:
            report["inputs"]["field_in"] = {"connected": True, "error": str(e)[:200]}
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

    # ── Generate output image ─────────────────────────────────────
    pattern = params.get("test_pattern", "color_bars")
    img_arr = _make_image_pattern(pattern, seed)
    save(img_arr, mn(999, "test-node"), out_dir)

    report["outputs"]["image"] = {
        "pattern": pattern,
        "shape": list(img_arr.shape),
        "mean": round(float(np.mean(img_arr)), 4),
    }

    # ── Generate output field ─────────────────────────────────────
    field_pattern = params.get("field_pattern", "sine_gradient")
    field_arr = _make_field_pattern(field_pattern, seed)
    write_field(out_dir, field_arr)

    report["outputs"]["field"] = {
        "pattern": field_pattern,
        "shape": list(field_arr.shape),
        "mean": round(float(np.mean(field_arr)), 4),
        "min": round(float(np.min(field_arr)), 4),
        "max": round(float(np.max(field_arr)), 4),
    }

    # ── Generate output particles ──────────────────────────────────
    pcount = int(params.get("particle_count", 100))
    particles = _make_particles(pcount, seed)
    np.save(str(out_dir / "particles.npy"), particles)

    report["outputs"]["particles"] = {
        "count": pcount,
        "shape": list(particles.shape),
        "dtype": str(particles.dtype),
    }

    # ── Generate output mask ──────────────────────────────────────
    mask_pattern = params.get("mask_pattern", "circle")
    mask_arr = _make_mask_pattern(mask_pattern, seed)
    np.save(str(out_dir / "mask.npy"), mask_arr)

    report["outputs"]["mask"] = {
        "pattern": mask_pattern,
        "shape": list(mask_arr.shape),
        "mean": round(float(np.mean(mask_arr)), 4),
    }

    # ── Output scalars ─────────────────────────────────────────────
    scalar_val = float(params.get("scalar_value", 42.0))
    scalar_b = float(params.get("scalar_value_b", 3.14159))
    write_scalars(out_dir, test_scalar=scalar_val, test_scalar_b=scalar_b)

    report["outputs"]["test_scalar"] = scalar_val
    report["outputs"]["test_scalar_b"] = scalar_b

    # ── Write report ─────────────────────────────────────────────
    (out_dir / "test_report.json").write_text(json.dumps(report, indent=2))


def _make_image_pattern(pattern: str, seed: int) -> np.ndarray:
    """Generate a test image pattern."""
    rng = np.random.RandomState(seed)
    arr = np.zeros((H, W, 3), dtype=np.float32)

    if pattern == "color_bars":
        # 8 vertical color bars
        bars = 8
        bw = W // bars
        colors = [
            (1.0, 1.0, 1.0),  # white
            (1.0, 1.0, 0.0),  # yellow
            (0.0, 1.0, 1.0),  # cyan
            (0.0, 1.0, 0.0),  # green
            (1.0, 0.0, 1.0),  # magenta
            (1.0, 0.0, 0.0),  # red
            (0.0, 0.0, 1.0),  # blue
            (0.0, 0.0, 0.0),  # black
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
    particles[:, 0] = rng.rand(count).astype(np.float32) * W  # x
    particles[:, 1] = rng.rand(count).astype(np.float32) * H  # y
    particles[:, 2] = (rng.rand(count).astype(np.float32) - 0.5) * 4  # vx
    particles[:, 3] = (rng.rand(count).astype(np.float32) - 0.5) * 4  # vy
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
