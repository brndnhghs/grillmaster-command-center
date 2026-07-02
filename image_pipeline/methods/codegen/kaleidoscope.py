"""Code-gen method — auto-split from codegen.py"""
from __future__ import annotations
import math
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from ...core.registry import method
from ...core.utils import get_font, W, H
from ...core.animation import capture_frame

# ────────────────────────────────────────────────────────────────────────────
# #12 — Kaleidoscope
# ────────────────────────────────────────────────────────────────────────────

@method(
    id="12",
    name="Kaleidoscope",
    category="codegen",
    tags=["kaleidoscope", "reflection", "symmetry"],
    inputs={
        "image_in": "IMAGE",
        "segments": "SCALAR",
        "rotation": "SCALAR",
        "wobble_x": "SCALAR",
        "wobble_y": "SCALAR",
        "zoom": "SCALAR",
        "wedge_offset": "SCALAR",
        "color_shift": "SCALAR",
        "pattern_select": "SCALAR",
    },
    outputs={"image": "IMAGE", "luminance": "FIELD"},
    params={
        "pattern": {
            "description": "kaleidoscope base pattern",
            "choices": ["radial", "spiral", "hexagonal", "mandala"],
            "default": "radial",
        },
        "segments": {
            "description": "number of reflective segments. Wire Counter.value here.",
            "default": 6,
        },
        "source": {
            "description": "texture source for the wedge (ignored when image_in is wired)",
            "choices": ["random", "gradient", "noise"],
            "default": "random",
        },
        "rotation": {
            "description": "SCALAR-driven base rotation. Wire LFO.value here.",
            "default": 0.0,
        },
        "wobble_x": {
            "description": "SCALAR-driven horizontal wobble. Wire LFO.value here.",
            "default": 0.0,
        },
        "wobble_y": {
            "description": "SCALAR-driven vertical wobble. Wire LFO.value here.",
            "default": 0.0,
        },
        "zoom": {
            "description": "SCALAR-driven zoom factor. Wire LFO.value here.",
            "default": 1.0,
        },
        "wedge_offset": {
            "description": "SCALAR-driven wedge rotation offset. Wire LFO.value here.",
            "default": 0.0,
        },
        "color_shift": {
            "description": "SCALAR-driven hue/color shift. Wire LFO.value here.",
            "default": 0.0,
        },
        "pattern_select": {
            "description": "SCALAR-driven pattern index (0-1 maps to radial/spiral/hexagonal/mandala). Wire Counter.value here.",
            "default": -1.0,
        },
    },
)
def method_kaleidoscope(out_dir: Path, seed: int, params=None):
    """Render kaleidoscopic reflection patterns using cv2.remap.

    Architecture B (stateless, one call = one frame). Animation is driven
    by wired SCALAR inputs instead of internal anim_mode logic.

    When image_in is wired, the input image is used as the wedge texture
    instead of the internally generated source (random/gradient/noise).

    Wire channel nodes to drive params:
      LFO.value → rotation      (rotation, replaces rotation mode)
      LFO.value → wobble_x      (horizontal wobble, replaces wobble mode)
      LFO.value → wobble_y      (vertical wobble, replaces wobble mode)
      LFO.value → zoom          (zoom pulse, replaces pulse_zoom mode)
      LFO.value → wedge_offset  (wedge dance, replaces wedge_dance mode)
      LFO.value → color_shift   (color wash, replaces color_wash mode)
      Counter.value → segments  (segment count sweep)
      Counter.value → pattern_select (pattern cycling)
    """
    if params is None:
        params = {}

    # Snapshot canvas size once per call
    w, h = int(W), int(H)

    try:
        import cv2
        _has_cv2 = True
    except ImportError:
        _has_cv2 = False

    if not _has_cv2:
        pil_img = Image.new("RGB", (w, h), (30, 10, 10))
        draw = ImageDraw.Draw(pil_img)
        font = get_font(20)
        draw.text((w // 2 - 100, h // 2 - 10), "cv2 library missing", fill=(200, 50, 50), font=font)
        img = np.array(pil_img).astype(np.float32) / 255.0
        capture_frame("12", img)
        return {"image": img}

    # ── Read SCALAR inputs ──
    segments_override = params.get("segments")
    if segments_override is not None:
        segments = max(3, min(16, int(segments_override)))
    else:
        segments = max(3, min(16, int(params.get("segments", 6))))

    rotation_override = params.get("rotation")
    effective_rotation = float(rotation_override) if rotation_override is not None else float(params.get("rotation", 0.0))

    wobble_x_override = params.get("wobble_x")
    effective_wobble_x = float(wobble_x_override) if wobble_x_override is not None else float(params.get("wobble_x", 0.0))

    wobble_y_override = params.get("wobble_y")
    effective_wobble_y = float(wobble_y_override) if wobble_y_override is not None else float(params.get("wobble_y", 0.0))

    zoom_override = params.get("zoom")
    effective_zoom = float(zoom_override) if zoom_override is not None else float(params.get("zoom", 1.0))

    wedge_offset_override = params.get("wedge_offset")
    effective_wedge_offset = float(wedge_offset_override) if wedge_offset_override is not None else float(params.get("wedge_offset", 0.0))

    color_shift_override = params.get("color_shift")
    effective_color_shift = float(color_shift_override) if color_shift_override is not None else float(params.get("color_shift", 0.0))

    pattern_select_override = params.get("pattern_select")
    if pattern_select_override is not None:
        pidx = int(float(pattern_select_override) * 4) % 4
        pattern = ["radial", "spiral", "hexagonal", "mandala"][pidx]
    else:
        pattern = params.get("pattern", "radial")

    source = params.get("source", "random")

    # ── Read upstream image (optional) ──
    input_img = params.get("_input_image")

    # ── Render helper ──
    def _render_frame(pat: str, seg: int, src: str, rot: float,
                      wobble_x: float = 0.0, wobble_y: float = 0.0,
                      zoom: float = 1.0,
                      wedge_rot_offset: float = 0.0,
                      color_shift: float = 0.0) -> np.ndarray:
        """Render a full kaleidoscope frame. Returns H×W×3 float32 [0,1] array."""
        wedge_size = max(w, h)
        base = np.zeros((wedge_size, wedge_size, 3), dtype=np.float32)

        cx_ws = wedge_size / 2.0 + wobble_x * wedge_size * 0.15
        cy_ws = wedge_size / 2.0 + wobble_y * wedge_size * 0.15
        xs = (np.arange(wedge_size, dtype=np.float32) - cx_ws) / (cx_ws * zoom)
        ys = (np.arange(wedge_size, dtype=np.float32) - cy_ws) / (cy_ws * zoom)
        xv, yv = np.meshgrid(xs, ys)
        r = np.sqrt(xv ** 2 + yv ** 2)
        theta = np.arctan2(yv, xv)

        if input_img is not None:
            # Use upstream image as wedge texture
            src_pil = Image.fromarray((np.clip(input_img, 0, 1) * 255).astype(np.uint8))
            src_resized = src_pil.resize((wedge_size, wedge_size), Image.LANCZOS)
            base = np.array(src_resized, dtype=np.float32) / 255.0
        elif src == "random":
            rng_skip = np.random.default_rng(seed % 100000)
            noise_layer = rng_skip.random((wedge_size, wedge_size)).astype(np.float32)
            base[:, :, 0] = noise_layer * 0.3 + rng_skip.random((wedge_size, wedge_size)).astype(np.float32) * 0.7
            base[:, :, 1] = np.roll(base[:, :, 0], 2, axis=0)
            base[:, :, 2] = np.roll(base[:, :, 0], -2, axis=1)
        elif src == "gradient":
            t_grad = r * 0.5
            hue_offset = color_shift
            r_ch = 0.5 + 0.5 * np.sin(t_grad * 3.0 + hue_offset)
            g_ch = 0.5 + 0.5 * np.cos(t_grad * 2.7 + 1.0 + hue_offset)
            b_ch = 0.5 + 0.5 * np.sin(t_grad * 3.3 + 2.0 + hue_offset)
            base = np.stack([r_ch, g_ch, b_ch], axis=-1)
        elif src == "noise":
            rng_skip = np.random.default_rng(seed % 100000 + 1)
            n = rng_skip.standard_normal((wedge_size, wedge_size)).astype(np.float32)
            n = (n - n.min()) / (n.max() - n.min() + 1e-8)
            hue_offset = color_shift
            base[:, :, 0] = (n + math.sin(hue_offset)) * 0.5
            base[:, :, 1] = (np.roll(n, 3, axis=0) + math.sin(hue_offset + 2.094)) * 0.5
            base[:, :, 2] = (np.roll(n, -3, axis=1) + math.sin(hue_offset + 4.189)) * 0.5

        # ── Apply pattern modulation ──
        if pat == "radial":
            band_t = r * seg * 2.0
            mod = (band_t - np.floor(band_t))
            base = base * mod[:, :, np.newaxis]
        elif pat == "spiral":
            spiral_t = r * seg * 2.0 + theta * 3.0
            mod = (spiral_t - np.floor(spiral_t))
            base = base * mod[:, :, np.newaxis]
        elif pat == "hexagonal":
            hx = xv * seg * 0.5
            hy = yv * seg * 0.5
            hex_r = np.sqrt(hx ** 2 + hy ** 2)
            hex_t = hex_r * 4.0
            mod = (hex_t - np.floor(hex_t))
            base = base * mod[:, :, np.newaxis]
        elif pat == "mandala":
            n_petals = seg * 2
            petal_angle = theta * n_petals
            mandala_mod = 0.5 + 0.5 * np.cos(petal_angle + r * 5.0)
            base = base * mandala_mod[:, :, np.newaxis]

        # ── Build polar reflection map using cv2.remap ──
        out_xs = np.arange(w, dtype=np.float32)
        out_ys = np.arange(h, dtype=np.float32)
        oxv, oyv = np.meshgrid(out_xs, out_ys)

        ocx = w / 2.0
        ocy = h / 2.0
        dx = oxv - ocx
        dy = oyv - ocy

        out_r = np.sqrt(dx ** 2 + dy ** 2)
        out_theta = np.arctan2(dy, dx)

        rot_rad = math.radians(rot)
        out_theta += rot_rad

        # ── Wedge dance: alternating wedge offsets ──
        if wedge_rot_offset != 0.0:
            wedge_angle = math.pi / seg
            wedge_idx = np.floor(out_theta / wedge_angle).astype(int) % seg
            for w_idx in range(seg):
                mask = wedge_idx == w_idx
                out_theta[mask] += wedge_rot_offset * math.sin(float(w_idx) * 1.8)

        wedge_angle = math.pi / seg
        folded_theta = np.abs(out_theta % (2.0 * wedge_angle) - wedge_angle)

        src_x = cx_ws + out_r * np.cos(folded_theta) / zoom
        src_y = cy_ws + out_r * np.sin(folded_theta) / zoom

        src_x = np.clip(src_x, 0, wedge_size - 1)
        src_y = np.clip(src_y, 0, wedge_size - 1)

        map_x = src_x.astype(np.float32)
        map_y = src_y.astype(np.float32)
        result = cv2.remap(base, map_x, map_y, interpolation=cv2.INTER_LINEAR,
                           borderMode=cv2.BORDER_WRAP)
        return np.clip(result, 0.0, 1.0)

    # ── Render ──
    img = _render_frame(
        pattern, segments, source, effective_rotation,
        wobble_x=effective_wobble_x, wobble_y=effective_wobble_y,
        zoom=effective_zoom, wedge_rot_offset=effective_wedge_offset,
        color_shift=effective_color_shift,
    )

    capture_frame("12", img)
    return {"image": img}

