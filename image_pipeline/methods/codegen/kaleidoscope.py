"""Code-gen method — auto-split from codegen.py"""
from __future__ import annotations
import math
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from ...core.registry import method
from ...core.utils import save, norm, mn, seed_all, get_font, W, H
from ...core.animation import capture_frame

# ────────────────────────────────────────────────────────────────────────────
# #12 — Kaleidoscope
# ────────────────────────────────────────────────────────────────────────────

@method(
    id="12",
    name="Kaleidoscope",
    category="codegen",
    tags=["kaleidoscope", "fast", "animation", "reflection"],
    params={
        "pattern": {
            "description": "kaleidoscope base pattern",
            "choices": ["radial", "spiral", "hexagonal", "mandala"],
            "default": "radial",
        },
        "segments": {
            "description": "number of reflective segments",
            "min": 3,
            "max": 16,
            "default": 6,
        },
        "source": {
            "description": "texture source for the wedge",
            "choices": ["random", "gradient", "noise"],
            "default": "random",
        },
        "rotation": {
            "description": "base rotation in degrees",
            "min": 0,
            "max": 360,
            "default": 0,
        },
        "time": {
            "description": "animation time (0-6.28)",
            "min": 0.0,
            "max": 6.28,
            "default": 0.0,
        },
        "anim_mode": {
            "description": "kaleidoscope animation mode",
            "choices": ["none", "rotation", "wobble", "pulse_zoom", "wedge_dance", "petal_breathe", "color_wash"],
            "default": "none",
        },
        "anim_speed": {
            "description": "animation speed multiplier",
            "min": 0.1,
            "max": 3.0,
            "default": 0.25,
        },
    },
)
def method_kaleidoscope(out_dir: Path, seed: int, params=None):
    """Render kaleidoscopic reflection patterns using cv2.remap."""
    if params is None:
        params = {}
    t = float(params.get("time", 0.0))
    seed_all(seed)

    try:
        import cv2
        _has_cv2 = True
    except ImportError:
        _has_cv2 = False

    if not _has_cv2:
        # Fallback: render a placeholder
        pil_img = Image.new("RGB", (W, H), (30, 10, 10))
        draw = ImageDraw.Draw(pil_img)
        font = get_font(20)
        draw.text((W // 2 - 100, H // 2 - 10), "cv2 library missing", fill=(200, 50, 50), font=font)
        img = np.array(pil_img).astype(np.float32) / 255.0
        capture_frame("12", img)
        save(img, mn(12, "Kaleidoscope"), out_dir)
        return

    pattern = params.get("pattern", "radial")
    segments = int(params.get("segments", 6))
    source = params.get("source", "random")
    rotation = float(params.get("rotation", 0.0))
    anim_mode = params.get("anim_mode", "none")
    anim_speed = float(params.get("anim_speed", 0.25))

    # ── Render helper that builds a complete kaleidoscope frame ──

    def _render_frame(pat: str, seg: int, src: str, rot: float,
                      wobble_x: float = 0.0, wobble_y: float = 0.0,
                      zoom: float = 1.0,
                      wedge_rot_offset: list = None,
                      color_shift: float = 0.0) -> np.ndarray:
        """Render a full kaleidoscope frame. Returns H×W×3 float32 [0,1] array."""
        wedge_size = max(W, H)
        base = np.zeros((wedge_size, wedge_size, 3), dtype=np.float32)

        cx_ws = wedge_size / 2.0 + wobble_x * wedge_size * 0.15
        cy_ws = wedge_size / 2.0 + wobble_y * wedge_size * 0.15
        xs = (np.arange(wedge_size, dtype=np.float32) - cx_ws) / (cx_ws * zoom)
        ys = (np.arange(wedge_size, dtype=np.float32) - cy_ws) / (cy_ws * zoom)
        xv, yv = np.meshgrid(xs, ys)
        r = np.sqrt(xv ** 2 + yv ** 2)
        theta = np.arctan2(yv, xv)

        if src == "random":
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
        out_xs = np.arange(W, dtype=np.float32)
        out_ys = np.arange(H, dtype=np.float32)
        oxv, oyv = np.meshgrid(out_xs, out_ys)

        ocx = W / 2.0
        ocy = H / 2.0
        dx = oxv - ocx
        dy = oyv - ocy

        out_r = np.sqrt(dx ** 2 + dy ** 2)
        out_theta = np.arctan2(dy, dx)

        rot_rad = math.radians(rot)
        out_theta += rot_rad

        # ── Wedge dance: alternating wedge offsets ──
        if wedge_rot_offset is not None and len(wedge_rot_offset) >= seg:
            wedge_angle = math.pi / seg
            wedge_idx = np.floor(out_theta / wedge_angle).astype(int) % seg
            for w_idx in range(seg):
                mask = wedge_idx == w_idx
                out_theta[mask] += wedge_rot_offset[w_idx]

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

    # ── Compute effective parameters based on anim_mode ──
    effective_rotation = rotation
    effective_wobble_x = 0.0
    effective_wobble_y = 0.0
    effective_zoom = 1.0
    effective_wedge_rot = None
    effective_color_shift = 0.0

    if anim_mode == "rotation":
        effective_rotation = (rotation + t * 30.0 * anim_speed) % 360.0

    elif anim_mode == "wobble":
        effective_rotation = rotation
        w_angle = t * anim_speed * 1.3
        effective_wobble_x = math.sin(w_angle)
        effective_wobble_y = math.cos(w_angle * 0.7)

    elif anim_mode == "pulse_zoom":
        effective_rotation = rotation
        effective_zoom = 0.6 + 0.4 * (0.5 + 0.5 * math.sin(t * 1.5 * anim_speed))

    elif anim_mode == "wedge_dance":
        effective_rotation = rotation
        offsets = []
        for w in range(segments):
            off = math.sin(t * anim_speed * 1.2 + w * 1.8) * 0.3 * (math.pi / segments)
            offsets.append(off)
        effective_wedge_rot = offsets

    elif anim_mode == "petal_breathe":
        effective_rotation = rotation
        breathe = 0.3 + 0.7 * (0.5 + 0.5 * math.sin(t * anim_speed * 1.1))
        # We modulate pattern intensity by scaling the modulation amplitude
        # This is applied via the pattern — effect visible in mandala/radial
        # We'll use effective_zoom as a proxy for breathe amplitude
        effective_zoom = breathe

    elif anim_mode == "color_wash":
        effective_rotation = rotation
        effective_color_shift = t * 1.5 * anim_speed

    # ── Render ──
    img = _render_frame(
        pattern, segments, source, effective_rotation,
        wobble_x=effective_wobble_x, wobble_y=effective_wobble_y,
        zoom=effective_zoom, wedge_rot_offset=effective_wedge_rot,
        color_shift=effective_color_shift,
    )

    capture_frame("12", img)
    save(img, mn(12, "Kaleidoscope"), out_dir)

