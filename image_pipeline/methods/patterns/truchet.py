from __future__ import annotations
import math
import random
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from ...core.registry import method
from ...core.utils import save, norm, mn, seed_all, W, H
from ...core.animation import capture_frame
from ...core.utils import PALETTES

@method(id="07", name="Truchet Tiles", category="patterns",
description="Truchet Tiles — patterns node.",
         tags=["classic", "tiling", "fast", "expanded", "animation"],
         params={
    "tile_type": {"description": "tile pattern (arcs/diagonals/crosses/chevrons/circles/quadrants/spirals/hexagons/rings/weave)", "default": "arcs"},
    "tile_size": {"description": "tile size in pixels", "min": 20, "max": 200, "default": 40},
    "colormode": {"description": "color mode (random/palette/gradient/heatmap/spectral/fire/ice/dual_layer)", "default": "random"},
    "palette": {"description": "color palette name", "default": "vapor"},
    "line_width": {"description": "line/arc width", "min": 1, "max": 20, "default": 3},
    "gap": {"description": "mortar gap between tiles", "min": 0, "max": 10, "default": 0},
    "rotation_noise": {"description": "per-tile rotation randomness (0=none, 1=max)", "min": 0.0, "max": 1.0, "default": 0.0},
    "color_variation": {"description": "per-tile color variation (0=none, 1=max)", "min": 0.0, "max": 1.0, "default": 0.3},
    "bg_color": {"description": "background color (dark/light/transparent/gradient)", "default": "dark"},
    "anim_mode": {"description": "animation mode: none, size_wave, gap_pulse, arc_rotation, color_osc, arc_stretch, weave_pulse", "default": "none"},
    "anim_speed": {"description": "animation speed multiplier", "min": 0.1, "max": 3.0, "default": 0.25},})
def method_truchet(out_dir: Path, seed: int, params=None):
    """Render Truchet tiling patterns with multiple tile types and color modes.

    Truchet tiles are square tiles with patterns that tile seamlessly
    when rotated. Supports arcs, diagonals, crosses, spirals, and more.
    """
    try:
        if params is None:
            params = {}
        t = float(params.get("time", 0.0))  # pipeline already gives 0→2π
        seed_all(seed)  # fixed seed — animate via continuous param oscillation
        rng = random.Random(seed)

        tile_type = params.get("tile_type", "arcs")
        tile_size = int(params.get("tile_size", 40))
        cmode = params.get("colormode", "random")
        pal_name = params.get("palette", "vapor")
        lw = int(params.get("line_width", 3))
        gap = int(params.get("gap", 0))
        rot_noise = float(params.get("rotation_noise", 0.0))
        color_var = float(params.get("color_variation", 0.3))
        bg_style = params.get("bg_color", "dark")
        anim_mode = params.get("anim_mode", "none")
        anim_speed = float(params.get("anim_speed", 0.25))

        # ── Matplotlib import (with fallback) ──
        try:
            from matplotlib import cm
            _has_mpl = True
        except ImportError:
            _has_mpl = False

        from ...core.utils import PALETTES

        # ── Animation: operate on tile parameters ──
        effective_tile_type = tile_type
        effective_gap = gap

        # Per-tile continuous parameters (gated by mode)
        def _tile_angle(tx, ty, idx, base_rot=0):
            """Arc start angle in degrees. Continuous rotation for arc-based tiles."""
            if anim_mode == "arc_rotation":
                return (t * 120 * anim_speed + idx * 23.0) % 360
            return base_rot

        def _tile_rot_selection(tx, ty, idx):
            """Discrete rotation (0-3) for symmetric tiles. Uses position hash for determinism."""
            if anim_mode in ("arc_rotation", "color_osc", "arc_stretch", "weave_pulse"):
                return idx % 4  # deterministic from position
            return 0

        def _tile_hue_offset(tx, ty, idx):
            """Per-tile color index oscillation or deterministic hue."""
            if anim_mode == "color_osc":
                # Deterministic per-tile hue drift — use sin of position + time
                h = int(128 + 127 * math.sin(tx * 0.05 + ty * 0.03 + t * 1.8 * anim_speed))
                s = int(155 + 100 * math.cos(tx * 0.04 - ty * 0.06 + t * 1.5 * anim_speed))
                v = int(150 + 100 * math.sin(tx * 0.06 + ty * 0.07 + t * 2.0 * anim_speed))
                return (h, s, v)
            return None

        def _tile_arc_span(tx, ty, idx):
            """Arc angular span in degrees. Used by arc_stretch mode."""
            if anim_mode == "arc_stretch":
                return 45 + 90 * (0.5 + 0.5 * math.sin(t * 1.2 * anim_speed + idx * 0.5))
            return 90

        def _tile_weave_width(tx, ty, idx):
            """Weave arm width as fraction of tile size. Used by weave_pulse mode."""
            if anim_mode == "weave_pulse":
                return 0.2 + 0.3 * (0.5 + 0.5 * math.sin(t * 1.2 * anim_speed + idx * 0.7))
            return 0.333

        if anim_mode == "gap_pulse":
            # Breathing gap — continuous float for step, int only for grid count
            breathing = gap + 4.0 + 4.0 * math.sin(t * 1.5 * anim_speed)
            effective_gap = breathing  # keep as float, step = tile_size + float = valid

        # ── Background ──
        if bg_style == "dark":
            img = Image.new("RGB", (W, H), (10, 10, 18))
        elif bg_style == "light":
            img = Image.new("RGB", (W, H), (240, 240, 235))
        elif bg_style == "transparent":
            img = Image.new("RGBA", (W, H), (0, 0, 0, 0))
        elif bg_style == "gradient":
            img = Image.new("RGB", (W, H), (10, 10, 18))
            for y in range(H):
                t_bg = y / H
                col = (int(10 + 30 * t_bg), int(10 + 20 * t_bg), int(18 + 40 * t_bg))
                for x in range(W):
                    img.putpixel((x, y), col)
        else:
            img = Image.new("RGB", (W, H), (10, 10, 18))

        draw = ImageDraw.Draw(img)

        # ── Color helpers ──
        pal = PALETTES.get(pal_name, PALETTES["vapor"])
        pal_arr = np.array(pal, dtype=np.uint8)

        def _tile_color(tx, ty, idx):
            if cmode == "random":
                r = random.randint(40, 255)
                g = random.randint(30, 220)
                b = random.randint(50, 200)
                if color_var > 0:
                    v = int(30 * color_var)
                    r = max(0, min(255, r + random.randint(-v, v)))
                    g = max(0, min(255, g + random.randint(-v, v)))
                    b = max(0, min(255, b + random.randint(-v, v)))
                return (r, g, b)
            elif cmode == "palette":
                ci = idx % len(pal)
                if color_var > 0 and random.random() < color_var:
                    ci = (ci + random.randint(-1, 1)) % len(pal)
                return tuple(int(c) for c in pal_arr[ci])
            elif cmode in ("gradient", "heatmap", "spectral", "fire", "ice", "dual_layer"):
                # Use position-based color
                gx = tx / W
                gy = ty / H
                val = (gx + gy) * 0.5
                if cmode == "gradient":
                    r = int(50 + 200 * val)
                    g = int(30 + 150 * (1 - val))
                    b = int(80 + 100 * val)
                elif cmode == "heatmap":
                    if _has_mpl:
                        c = cm.inferno(val)
                        r, g, b = int(c[0]*255), int(c[1]*255), int(c[2]*255)
                    else:
                        r, g, b = int(50 + 200 * val), int(30 + 150 * (1 - val)), int(80 + 100 * val)
                elif cmode == "spectral":
                    if _has_mpl:
                        c = cm.nipy_spectral(val)
                        r, g, b = int(c[0]*255), int(c[1]*255), int(c[2]*255)
                    else:
                        r, g, b = int(50 + 200 * val), int(30 + 150 * (1 - val)), int(80 + 100 * val)
                elif cmode == "fire":
                    r = int(255 * val)
                    g = int(100 * val)
                    b = int(30 * val)
                elif cmode == "ice":
                    r = int(30 * val)
                    g = int(100 * val)
                    b = int(200 * val)
                elif cmode == "dual_layer":
                    if _has_mpl:
                        c = cm.viridis(val) if val < 0.5 else cm.inferno(val)
                        r, g, b = int(c[0]*255), int(c[1]*255), int(c[2]*255)
                    else:
                        r, g, b = int(50 + 200 * val), int(30 + 150 * (1 - val)), int(80 + 100 * val)
                return (r, g, b)
            return (200, 150, 100)

        # ── Draw tiles ──
        step = float(tile_size + effective_gap)  # may be float from gap_pulse
        cols = W // int(step) + 1
        rows = H // int(step) + 1

        for ry in range(rows):
            for rx in range(cols):
                tx = rx * step
                ty = ry * step
                idx = ry * cols + rx

                # Per-tile size modulation: wave propagates across grid
                size_mod = 1.0
                if anim_mode == "size_wave":
                    px = rx / max(1, cols)
                    py = ry / max(1, rows)
                    size_mod = 0.7 + 0.3 * math.sin(t * 0.5 * anim_speed + px * 4 + py * 3)
                ts = max(2, int(tile_size * size_mod))

                # Per-tile animation parameters (gated by mode)
                angle_off = _tile_angle(tx, ty, idx)
                rot = _tile_rot_selection(tx, ty, idx)
                hue_off = _tile_hue_offset(tx, ty, idx)
                arc_span = _tile_arc_span(tx, ty, idx)
                weave_arm = _tile_weave_width(tx, ty, idx)

                if anim_mode == "color_osc" and isinstance(hue_off, tuple):
                    color = hue_off  # Use deterministic HSV directly
                else:
                    color = _tile_color(tx, ty, idx + (hue_off if isinstance(hue_off, int) else 0))

                if effective_tile_type == "arcs":
                    # Continuous arc rotation — arc span animated by arc_stretch
                    draw.arc([tx, ty, tx + ts, ty + ts], angle_off, arc_span + angle_off, fill=color, width=lw)
                    draw.arc([tx, ty, tx + ts, ty + ts], 180 + angle_off, 180 + arc_span + angle_off, fill=color, width=lw)
                elif effective_tile_type == "diagonals":
                    # Diagonal lines
                    if rot % 2 == 0:
                        draw.line([tx, ty, tx + ts, ty + ts], fill=color, width=lw)
                    else:
                        draw.line([tx + ts, ty, tx, ty + ts], fill=color, width=lw)

                elif effective_tile_type == "crosses":
                    # Cross/plus
                    cx, cy = tx + ts // 2, ty + ts // 2
                    arm = ts // 3
                    draw.line([cx - arm, cy, cx + arm, cy], fill=color, width=lw)
                    draw.line([cx, cy - arm, cx, cy + arm], fill=color, width=lw)

                elif effective_tile_type == "chevrons":
                    # Chevron/V shapes
                    if rot % 2 == 0:
                        draw.line([tx, ty + ts, tx + ts // 2, ty], fill=color, width=lw)
                        draw.line([tx + ts // 2, ty, tx + ts, ty + ts], fill=color, width=lw)
                    else:
                        draw.line([tx, ty, tx + ts // 2, ty + ts], fill=color, width=lw)
                        draw.line([tx + ts // 2, ty + ts, tx + ts, ty], fill=color, width=lw)

                elif effective_tile_type == "circles":
                    # Quarter circles
                    r = ts // 2
                    if rot == 0:
                        draw.pieslice([tx, ty, tx + ts, ty + ts], 0, 90, fill=color)
                    elif rot == 1:
                        draw.pieslice([tx, ty, tx + ts, ty + ts], 90, 180, fill=color)
                    elif rot == 2:
                        draw.pieslice([tx, ty, tx + ts, ty + ts], 180, 270, fill=color)
                    else:
                        draw.pieslice([tx, ty, tx + ts, ty + ts], 270, 360, fill=color)

                elif effective_tile_type == "quadrants":
                    # Split into 4 colored quadrants
                    cx, cy = tx + ts // 2, ty + ts // 2
                    colors = [
                        _tile_color(tx, ty, idx * 4 + 0),
                        _tile_color(tx, ty, idx * 4 + 1),
                        _tile_color(tx, ty, idx * 4 + 2),
                        _tile_color(tx, ty, idx * 4 + 3),
                    ]
                    draw.pieslice([tx, ty, tx + ts, ty + ts], 0, 90, fill=colors[0])
                    draw.pieslice([tx, ty, tx + ts, ty + ts], 90, 180, fill=colors[1])
                    draw.pieslice([tx, ty, tx + ts, ty + ts], 180, 270, fill=colors[2])
                    draw.pieslice([tx, ty, tx + ts, ty + ts], 270, 360, fill=colors[3])

                elif effective_tile_type == "spirals":
                    # Spiral arcs
                    cx, cy = tx + ts // 2, ty + ts // 2
                    r = ts // 2
                    for i in range(4):
                        a1 = i * 90 + rot * 45
                        a2 = a1 + 60
                        draw.arc([cx - r, cy - r, cx + r, cy + r], a1, a2, fill=color, width=lw)
                        r = r // 2

                elif effective_tile_type == "hexagons":
                    # Hexagon tile
                    cx, cy = tx + ts // 2, ty + ts // 2
                    r = ts // 2
                    pts = []
                    for i in range(6):
                        a = math.pi / 3 * i + math.pi / 6 + rot * math.pi / 6
                        pts.append((cx + r * math.cos(a), cy + r * math.sin(a)))
                    draw.polygon(pts, outline=color, width=lw)

                elif effective_tile_type == "rings":
                    # Concentric rings
                    cx, cy = tx + ts // 2, ty + ts // 2
                    for ri in range(3):
                        rr = ts // 2 - ri * (ts // 6)
                        draw.ellipse([cx - rr, cy - rr, cx + rr, cy + rr], outline=color, width=max(1, lw - ri))

                elif effective_tile_type == "weave":
                    # Weave/interlace pattern — arm width animated by weave_pulse
                    cx, cy = tx + ts // 2, ty + ts // 2
                    arm = max(2, int(ts * weave_arm))
                    # Horizontal
                    draw.rectangle([tx, cy - arm, tx + ts, cy + arm], fill=color)
                    # Vertical (alternating)
                    if rot % 2 == 0:
                        draw.rectangle([cx - arm, ty, cx + arm, cy - arm], fill=color)
                        draw.rectangle([cx - arm, cy + arm, cx + arm, ty + ts], fill=color)
                    else:
                        draw.rectangle([cx - arm, ty, cx + arm, ty + ts], fill=color)

        # ── Convert RGBA to RGB if needed ──
        if img.mode == "RGBA":
            bg = Image.new("RGB", img.size, (10, 10, 18))
            bg.paste(img, mask=img.split()[3])
            img = bg

        capture_frame("07", np.array(img).astype(np.float32) / 255.0)
        save(img, mn(7, "Truchet Tiles"), out_dir)
    except Exception as exc:
        fallback = np.full((H, W, 3), 128, dtype=np.uint8)
        save(fallback, mn(7, 'Truchet Tiles'), out_dir)
        print(f'[method_07] ERROR: {exc}')
        return fallback


