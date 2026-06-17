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
            "choices": ["none", "rotation", "pattern_morph", "segment_morph", "source_morph"],
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

    def _render_frame(pat: str, seg: int, src: str, rot: float) -> np.ndarray:
        """Render a full kaleidoscope frame. Returns H×W×3 float32 [0,1] array."""
        wedge_size = max(W, H)
        base = np.zeros((wedge_size, wedge_size, 3), dtype=np.float32)

        cx_ws = wedge_size / 2.0
        cy_ws = wedge_size / 2.0
        xs = (np.arange(wedge_size, dtype=np.float32) - cx_ws) / cx_ws
        ys = (np.arange(wedge_size, dtype=np.float32) - cy_ws) / cy_ws
        xv, yv = np.meshgrid(xs, ys)
        r = np.sqrt(xv ** 2 + yv ** 2)
        theta = np.arctan2(yv, xv)

        if src == "random":
            rng_skip = np.random.default_rng(seed % 100000)
            noise_layer = rng_skip.random((wedge_size, wedge_size)).astype(np.float32)
            for c in range(3):
                base[:, :, c] = noise_layer * 0.3 + rng_skip.random((wedge_size, wedge_size)).astype(np.float32) * 0.7

        elif src == "gradient":
            t_grad = r * 0.5
            r_ch = 0.5 + 0.5 * np.sin(t_grad * 3.0)
            g_ch = 0.5 + 0.5 * np.cos(t_grad * 2.7 + 1.0)
            b_ch = 0.5 + 0.5 * np.sin(t_grad * 3.3 + 2.0)
            base = np.stack([r_ch, g_ch, b_ch], axis=-1)

        elif src == "noise":
            rng_skip = np.random.default_rng(seed % 100000 + 1)
            n = rng_skip.standard_normal((wedge_size, wedge_size)).astype(np.float32)
            n = (n - n.min()) / (n.max() - n.min() + 1e-8)
            base[:, :, 0] = n
            base[:, :, 1] = np.roll(n, 3, axis=0)
            base[:, :, 2] = np.roll(n, -3, axis=1)

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

        wedge_angle = math.pi / seg
        folded_theta = np.abs(out_theta % (2.0 * wedge_angle) - wedge_angle)

        src_x = cx_ws + out_r * np.cos(folded_theta)
        src_y = cy_ws + out_r * np.sin(folded_theta)

        src_x = np.clip(src_x, 0, wedge_size - 1)
        src_y = np.clip(src_y, 0, wedge_size - 1)

        map_x = src_x.astype(np.float32)
        map_y = src_y.astype(np.float32)
        result = cv2.remap(base, map_x, map_y, interpolation=cv2.INTER_LINEAR,
                           borderMode=cv2.BORDER_WRAP)
        return np.clip(result, 0.0, 1.0)

    # ── Compute effective parameters based on anim_mode ──
    effective_pattern = pattern
    effective_segments = segments
    effective_source = source
    effective_rotation = rotation
    morph_fade = 0.0
    pattern_b = pattern
    source_b = source
    segments_b = segments

    if anim_mode == "rotation":
        effective_rotation = (rotation + t * 30.0 * anim_speed) % 360.0

    elif anim_mode == "pattern_morph":
        pattern_cycle = ["radial", "spiral", "hexagonal", "mandala"]
        n_pats = len(pattern_cycle)
        raw_idx = (t / (2 * math.pi)) * n_pats * anim_speed
        idx_a = int(raw_idx) % n_pats
        idx_b = (idx_a + 1) % n_pats
        morph_fade = raw_idx - int(raw_idx)
        effective_pattern = pattern_cycle[idx_a]
        pattern_b = pattern_cycle[idx_b]

    elif anim_mode == "segment_morph":
        seg_range = list(range(3, 17))
        n_segs = len(seg_range)
        raw_idx = (t / (2 * math.pi)) * n_segs * anim_speed
        idx_a = int(raw_idx) % n_segs
        idx_b = (idx_a + 1) % n_segs
        morph_fade = raw_idx - int(raw_idx)
        effective_segments = seg_range[idx_a]
        segments_b = seg_range[idx_b]

    elif anim_mode == "source_morph":
        source_cycle = ["random", "gradient", "noise"]
        n_srcs = len(source_cycle)
        raw_idx = (t / (2 * math.pi)) * n_srcs * anim_speed
        idx_a = int(raw_idx) % n_srcs
        idx_b = (idx_a + 1) % n_srcs
        morph_fade = raw_idx - int(raw_idx)
        effective_source = source_cycle[idx_a]
        source_b = source_cycle[idx_b]

    # ── Render frame(s) and blend if morphing ──
    if anim_mode in ("pattern_morph", "segment_morph") and morph_fade > 0:
        img_a = _render_frame(effective_pattern, effective_segments,
                              effective_source, effective_rotation)
        img_b = _render_frame(pattern_b, segments_b, source_b, effective_rotation)
        img = (1.0 - morph_fade) * img_a + morph_fade * img_b
    elif anim_mode == "source_morph" and morph_fade > 0:
        img_a = _render_frame(effective_pattern, effective_segments,
                              effective_source, effective_rotation)
        img_b = _render_frame(effective_pattern, effective_segments,
                              source_b, effective_rotation)
        img = (1.0 - morph_fade) * img_a + morph_fade * img_b
    else:
        img = _render_frame(effective_pattern, effective_segments,
                            effective_source, effective_rotation)

    capture_frame("12", img)
    save(img, mn(12, "Kaleidoscope"), out_dir)

