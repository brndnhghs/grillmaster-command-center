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

@method(id="06", name="Wallpaper Group", category="patterns",
description="Wallpaper Group — patterns node.",
        tags=["classic", "tiling", "fast", "animated", "expanded"],
        params={
    "group": {"description": "crystallographic symmetry group",
              "default": "p1",
              "choices": ["p1", "p2", "pm", "pg", "cm", "pmm", "pmg", "pgg",
                          "cmm", "p4", "p4m", "p4g", "p3", "p3m1", "p31m", "p6", "p6m",
                          "escher", "islamic", "penrose", "truchet"]},
    "motif": {"description": "tile shape motif",
              "default": "diamond",
              "choices": ["diamond", "triangle", "hexagon", "star", "cross",
                          "spiral", "wave", "scales", "escher_bird", "escher_fish",
                          "islamic_star", "arabesque", "penrose_kite", "penrose_dart",
                          "truchet_arc", "truchet_line", "truchet_circle"]},
    "palette": {"description": "color palette name from PALETTES", "default": "none"},
    "tile_size": {"description": "base tile size in px", "min": 20, "max": 300, "default": 80},
    "gap": {"description": "mortar/gap width between tiles", "min": 0, "max": 20, "default": 1},
    "rotation_noise": {"description": "per-tile rotation randomness (degrees)", "min": 0, "max": 90, "default": 0},
    "color_variation": {"description": "per-tile color variation (0=uniform, 1=max)", "min": 0.0, "max": 1.0, "default": 0.5},
    "scale_variation": {"description": "per-tile scale jitter", "min": 0.0, "max": 0.5, "default": 0.0},
    "penrose_generations": {"description": "Penrose inflation iterations", "min": 2, "max": 8, "default": 4},
    "star_rays": {"description": "star polygon rays (for star/islamic motifs)", "min": 4, "max": 16, "default": 8},
    "anim_mode": {"description": "animation mode: none, rotation_wave, scale_spiral, color_drift, position_wave, mosaic_shuffle, breathe_wave, vortex_spin, color_wave, deform, echo, glow, orbit", "default": "none"},
    "anim_speed": {"description": "animation speed multiplier", "min": 0.1, "max": 3.0, "default": 0.25},})
def method_wallpaper(out_dir: Path, seed: int, params=None):
    """
    Complete 2D tiling system with 17 crystallographic wallpaper symmetry groups,
    Escher-style interlocking tessellations, Islamic geometric star patterns,
    Penrose aperiodic tiling, and expanded Truchet tile variants.
    21 groups/modes × 16 motifs = 336+ unique pattern combinations.
    """
    if params is None:
        params = {}
    t = float(params.get("time", 0.0))
    seed_all(seed)  # fixed seed — animate via continuous param oscillation
    rng = random.Random(seed)

    group = params.get("group", "p1")
    motif = params.get("motif", "diamond")
    pal = params.get("palette", "none")
    tile_size = int(params.get("tile_size", 80))
    gap = int(params.get("gap", 1))
    rotation_noise = float(params.get("rotation_noise", 0))
    color_variation = float(params.get("color_variation", 0.5))
    scale_variation = float(params.get("scale_variation", 0.0))
    penrose_gens = int(params.get("penrose_generations", 4))
    star_rays = int(params.get("star_rays", 8))
    anim_speed = float(params.get("anim_speed", 0.25))
    anim_mode = params.get("anim_mode", "none")

    from ...core.utils import PALETTES, quantize_to_palette

    # ── Animation ──
    effective_motif = motif
    effective_group = group
    effective_color_var = color_variation
    global_rot = 0.0

    def _tile_angle_offset(x, y):
        """Per-tile rotation offset in degrees. 0 for none."""
        if anim_mode == "rotation_wave":
            return 60.0 * math.sin(x * 0.06 + y * 0.04 + t * 1.5 * anim_speed)
        return 0.0

    def _tile_pos_dx(x, y):
        if anim_mode == "position_wave":
            return 12.0 * math.sin(y * 0.03 + t * 1.8 * anim_speed)
        return 0.0

    def _tile_pos_dy(x, y):
        if anim_mode == "position_wave":
            return 12.0 * math.cos(x * 0.03 + t * 1.8 * anim_speed)
        return 0.0

    def _tile_scale(x, y):
        if anim_mode == "scale_spiral":
            d = math.sqrt(x * x + y * y) * 0.015
            return 0.5 + 0.5 * (0.5 + 0.5 * math.sin(d - t * 1.5 * anim_speed))
        return 1.0

    def _tile_gap_offset(x, y):
        """Effective gap per tile — 0 for none."""
        if anim_mode == "breathe_wave":
            wave = 0.5 + 0.5 * math.sin(x * 0.04 + y * 0.03 + t * 1.2 * anim_speed)
            return wave * 8.0  # 0-8px gap oscillates per tile
        return 0.0

    def _vortex_rotation(x, y, base_angle):
        """Add vortex rotation for vortex_spin mode."""
        if anim_mode == "vortex_spin":
            d = math.sqrt(x * x + y * y) * 0.008
            return base_angle + d * 360.0 * (0.5 + 0.5 * math.sin(t * 0.8 * anim_speed))
        return base_angle

    def _tile_motif_index(x, y, n_motifs, idx_map):
        """Per-tile motif index for mosaic_shuffle mode — oscillates tile by tile."""
        if anim_mode == "mosaic_shuffle":
            phase = math.sin(x * 0.12 + y * 0.08) * math.pi
            # Each tile independently toggles between two motifs based on time
            flip = 0.5 + 0.5 * math.sin(t * 1.5 * anim_speed + phase)
            return idx_map[int(flip * (len(idx_map) - 1))] if idx_map else 0
        return -1

    _shuffle_motif = None  # populated by mosaic_shuffle in _draw_motif

    img = Image.new("RGB", (W, H), (10, 10, 18))
    draw = ImageDraw.Draw(img)

    # ── Helper: random per-tile color ───────────────────────────────────
    def _tile_color(variation=1.0):
        var = variation * effective_color_var
        r = int(rng.randint(30, 255) * (1 - var * 0.5) + 128 * var * 0.5)
        g = int(rng.randint(30, 220) * (1 - var * 0.5) + 128 * var * 0.5)
        b = int(rng.randint(50, 200) * (1 - var * 0.5) + 128 * var * 0.5)
        return (r, g, b)

    def _drift_color(x, y):
        """Deterministic color from position + time. Used in color_drift mode."""
        h = math.sin(x * 0.07 + y * 0.05 + t * 0.8 * anim_speed) * 127 + 128
        s = math.cos(x * 0.04 - y * 0.06 + t * 0.9 * anim_speed) * 100 + 155
        v = math.sin(x * 0.05 + y * 0.08 - t * 0.7 * anim_speed) * 50 + 150
        return (int(h), int(s), int(v))

    def _inv_color(c):
        return (255 - c[0], 255 - c[1], 255 - c[2])

    # ── Motif drawing functions ─────────────────────────────────────────

    # Deform parameter: 0-1 controls internal geometry (inner radius, arm width,
    # spiral turns, etc). Used by deform mode.

    def _motif_diamond(d, cx, cy, sz, color, angle=0, deform=0.5):
        hsz = sz / 2
        pts = [(cx, cy - hsz), (cx + hsz, cy), (cx, cy + hsz), (cx - hsz, cy)]
        if angle:
            import math as m2
            c, s = m2.cos(m2.radians(angle)), m2.sin(m2.radians(angle))
            pts = [(cx + (x-cx)*c - (y-cy)*s, cy + (x-cx)*s + (y-cy)*c) for x,y in pts]
        d.polygon(pts, fill=color, outline=(10,10,18))
        inner_sz = max(2, sz * (0.1 + 0.4 * deform))
        d.ellipse([cx - inner_sz, cy - inner_sz, cx + inner_sz, cy + inner_sz],
                  fill=_inv_color(color), outline=None)

    def _motif_triangle(d, cx, cy, sz, color, angle=0, deform=0.5):
        h = sz * (3**0.5) / 2 * (1.0 - 0.3 * deform)  # deform = height variation
        pts = [(cx, cy - h/2), (cx + sz/2, cy + h/2), (cx - sz/2, cy + h/2)]
        if angle:
            import math as m2
            c, s = m2.cos(m2.radians(angle)), m2.sin(m2.radians(angle))
            pts = [(cx + (x-cx)*c - (y-cy)*s, cy + (x-cx)*s + (y-cy)*c) for x,y in pts]
        d.polygon(pts, fill=color, outline=(10,10,18))

    def _motif_hexagon(d, cx, cy, sz, color, angle=0, deform=0.5):
        pts = []
        import math as m2
        # deform stretches hexagon into an elongated shape
        x_scale = 1.0 + 0.3 * deform
        y_scale = 1.0 - 0.2 * deform
        for i in range(6):
            a = m2.radians(60 * i - 30 + angle)
            pts.append((cx + sz/2 * m2.cos(a) * x_scale, cy + sz/2 * m2.sin(a) * y_scale))
        d.polygon(pts, fill=color, outline=(10,10,18))

    def _motif_star(d, cx, cy, sz, color, angle=0, deform=0.5):
        n = star_rays
        import math as m2
        outer = sz / 2
        inner = outer * (0.2 + 0.5 * deform)  # deform = inner radius (sharp→rounded)
        pts = []
        for i in range(n * 2):
            a = m2.radians(360 * i / (n * 2) - 90 + angle)
            r = outer if i % 2 == 0 else inner
            pts.append((cx + r * m2.cos(a), cy + r * m2.sin(a)))
        d.polygon(pts, fill=color, outline=(10,10,18))

    def _motif_cross(d, cx, cy, sz, color, angle=0, deform=0.5):
        arm_w = sz * (0.1 + 0.2 * deform)  # deform = arm width (thin→thick)
        arm_l = sz * 0.4
        pts = [
            (cx - arm_w/2, cy - arm_l),
            (cx + arm_w/2, cy - arm_l),
            (cx + arm_w/2, cy - arm_w/2),
            (cx + arm_l, cy - arm_w/2),
            (cx + arm_l, cy + arm_w/2),
            (cx + arm_w/2, cy + arm_w/2),
            (cx + arm_w/2, cy + arm_l),
            (cx - arm_w/2, cy + arm_l),
            (cx - arm_w/2, cy + arm_w/2),
            (cx - arm_l, cy + arm_w/2),
            (cx - arm_l, cy - arm_w/2),
            (cx - arm_w/2, cy - arm_w/2),
        ]
        if angle:
            import math as m2
            c, s = m2.cos(m2.radians(angle)), m2.sin(m2.radians(angle))
            pts = [(cx + (x-cx)*c - (y-cy)*s, cy + (x-cx)*s + (y-cy)*c) for x,y in pts]
        d.polygon(pts, fill=color, outline=(10,10,18))

    def _motif_spiral(d, cx, cy, sz, color, angle=0, deform=0.5):
        import math as m2
        turns = int(2 + 4 * deform)  # deform = number of turns (2→6)
        pts = [(cx, cy)]
        for ti in range(int(sz * turns)):
            a = m2.radians(ti * 10 + angle)
            r = ti / (sz * turns / (sz/2))
            pts.append((cx + r * m2.cos(a), cy + r * m2.sin(a)))
        if len(pts) > 2:
            d.line(pts, fill=color, width=max(1, int(sz/30)))

    def _motif_wave(d, cx, cy, sz, color, angle=0, deform=0.5):
        import math as m2
        hsz = sz / 2
        amp_mod = 0.2 + 0.4 * deform  # deform = wave amplitude
        pts = []
        for x in range(int(-hsz), int(hsz)):
            y = m2.sin((x + hsz) / sz * 4 * m2.pi + m2.radians(angle)) * hsz * amp_mod
            pts.append((cx + x, cy + y))
        if len(pts) > 2:
            d.line(pts, fill=color, width=max(1, int(sz/20)))
        pts2 = []
        for x in range(int(-hsz), int(hsz)):
            y = m2.sin((x + hsz) / sz * 4 * m2.pi + m2.radians(angle) + m2.pi) * hsz * amp_mod
            pts2.append((cx + x, cy + y + hsz * amp_mod))
        if len(pts2) > 2:
            d.line(pts2, fill=_inv_color(color), width=max(1, int(sz/25)))

    def _motif_scales(d, cx, cy, sz, color, angle=0, deform=0.5):
        r = sz * (0.25 + 0.2 * deform)  # deform = scale size
        for ox, oy in [(0, 0), (r*0.6, -r*0.6), (-r*0.6, -r*0.6)]:
            d.ellipse([cx + ox - r, cy + oy - r, cx + ox + r, cy + oy + r],
                      fill=None, outline=color, width=max(1, int(sz/50)))

    def _motif_escher_bird(d, cx, cy, sz, color, angle=0, deform=0.5):
        import math as m2
        hsz = sz / 2
        pts = []
        for i in range(20):
            a = m2.radians(i * 18 + angle)
            r = hsz * (0.6 + 0.4 * m2.sin(i * 3 + m2.radians(angle)))
            pts.append((cx + r * m2.cos(a), cy + r * m2.sin(a)))
        d.polygon(pts, fill=color, outline=(10,10,18))
        d.ellipse([cx - hsz*0.12, cy - hsz*0.15, cx + hsz*0.12, cy + hsz*0.05],
                  fill=(10,10,18), outline=None)

    def _motif_escher_fish(d, cx, cy, sz, color, angle=0, deform=0.5):
        import math as m2
        hsz = sz / 2
        pts = []
        for i in range(24):
            a = m2.radians(i * 15 + angle)
            r = hsz * (0.5 + 0.5 * m2.sin(i * 2 + m2.radians(angle + 30)))
            pts.append((cx + r * m2.cos(a), cy + r * m2.sin(a)))
        d.polygon(pts, fill=color, outline=(10,10,18))
        tail_sz = hsz * 0.3
        d.polygon([(cx - hsz*0.6, cy), (cx - hsz, cy - tail_sz), (cx - hsz, cy + tail_sz)],
                  fill=color, outline=(10,10,18))

    def _motif_islamic_star(d, cx, cy, sz, color, angle=0, deform=0.5):
        n = star_rays
        import math as m2
        outer = sz * 0.45
        inner = outer * (0.15 + 0.5 * deform)  # deform = inner radius
        pts = []
        for i in range(n * 2):
            a = m2.radians(360 * i / (n * 2) + angle)
            r = outer if i % 2 == 0 else inner
            pts.append((cx + r * m2.cos(a), cy + r * m2.sin(a)))
        d.polygon(pts, fill=color, outline=(10,10,18))
        ring_r = outer * 1.3
        dot_r = sz * 0.06
        for i in range(n):
            a = m2.radians(360 * i / n + angle)
            dx, dy = cx + ring_r * m2.cos(a), cy + ring_r * m2.sin(a)
            d.ellipse([dx - dot_r, dy - dot_r, dx + dot_r, dy + dot_r],
                      fill=_inv_color(color), outline=None)

    def _motif_arabesque(d, cx, cy, sz, color, angle=0, deform=0.5):
        import math as m2
        hsz = sz / 2
        d.line([(cx, cy - hsz*0.6), (cx, cy + hsz*0.6)],
               fill=color, width=max(1, int(sz/30)))
        for side in [-1, 1]:
            for y_off in [-hsz*0.3, 0, hsz*0.3]:
                lx = cx + side * hsz * 0.25
                ly = cy + y_off
                pts = [(cx, ly),
                       (cx + side * hsz*0.15, ly - hsz*0.1),
                       (lx, ly - hsz*0.15),
                       (cx + side * hsz*0.1, ly)]
                d.line(pts, fill=color, width=max(1, int(sz/40)))
        d.ellipse([cx - hsz*0.08, cy - hsz*0.08, cx + hsz*0.08, cy + hsz*0.08],
                  fill=_inv_color(color), outline=None)

    def _motif_penrose_kite(d, cx, cy, sz, color, angle=0, deform=0.5):
        import math as m2
        phi = (1 + 5**0.5) / 2
        a = sz * 0.3
        b = a * phi
        pts = [(cx, cy - b), (cx + a, cy), (cx, cy + b), (cx - a, cy)]
        if angle:
            c, s = m2.cos(m2.radians(angle)), m2.sin(m2.radians(angle))
            pts = [(cx + (x-cx)*c - (y-cy)*s, cy + (x-cx)*s + (y-cy)*c) for x,y in pts]
        d.polygon(pts, fill=color, outline=(10,10,18))

    def _motif_penrose_dart(d, cx, cy, sz, color, angle=0, deform=0.5):
        import math as m2
        phi = (1 + 5**0.5) / 2
        a = sz * 0.3
        b = a * phi
        pts = [(cx, cy - b), (cx + a*0.5, cy), (cx, cy + b*0.4), (cx - a*0.5, cy)]
        if angle:
            c, s = m2.cos(m2.radians(angle)), m2.sin(m2.radians(angle))
            pts = [(cx + (x-cx)*c - (y-cy)*s, cy + (x-cx)*s + (y-cy)*c) for x,y in pts]
        d.polygon(pts, fill=color, outline=(10,10,18))

    def _motif_truchet_arc(d, cx, cy, sz, color, angle=0, deform=0.5):
        hsz = sz / 2
        opts = rng.randint(0, 3)
        corners = [(cx - hsz, cy - hsz), (cx + hsz, cy - hsz),
                   (cx + hsz, cy + hsz), (cx - hsz, cy + hsz)]
        for j, (sx, sy) in enumerate(corners):
            if j in (opts, (opts + 1) % 4):
                d.arc([sx, sy, sx + hsz, sy + hsz],
                      90 * j, 90 * (j + 1), fill=color, width=max(1, int(sz/20)))

    def _motif_truchet_line(d, cx, cy, sz, color, angle=0, deform=0.5):
        hsz = sz / 2
        opts = rng.randint(0, 3)
        if opts == 0:
            d.line([(cx - hsz, cy), (cx, cy - hsz)], fill=color, width=max(1, int(sz/20)))
            d.line([(cx + hsz, cy), (cx, cy + hsz)], fill=color, width=max(1, int(sz/20)))
        elif opts == 1:
            d.line([(cx, cy - hsz), (cx + hsz, cy)], fill=color, width=max(1, int(sz/20)))
            d.line([(cx, cy + hsz), (cx - hsz, cy)], fill=color, width=max(1, int(sz/20)))
        elif opts == 2:
            d.line([(cx - hsz, cy - hsz), (cx + hsz, cy + hsz)], fill=color, width=max(1, int(sz/20)))
        else:
            d.line([(cx + hsz, cy - hsz), (cx - hsz, cy + hsz)], fill=color, width=max(1, int(sz/20)))

    def _motif_truchet_circle(d, cx, cy, sz, color, angle=0, deform=0.5):
        r = sz * rng.uniform(0.2, 0.45)
        d.ellipse([cx - r, cy - r, cx + r, cy + r],
                  fill=None, outline=color, width=max(1, int(sz/25)))
        dr = r * 0.3
        d.ellipse([cx - dr, cy - dr, cx + dr, cy + dr], fill=color, outline=None)

    MOTIF_FN = {
        "diamond": _motif_diamond, "triangle": _motif_triangle,
        "hexagon": _motif_hexagon, "star": _motif_star,
        "cross": _motif_cross, "spiral": _motif_spiral,
        "wave": _motif_wave, "scales": _motif_scales,
        "escher_bird": _motif_escher_bird, "escher_fish": _motif_escher_fish,
        "islamic_star": _motif_islamic_star, "arabesque": _motif_arabesque,
        "penrose_kite": _motif_penrose_kite, "penrose_dart": _motif_penrose_dart,
        "truchet_arc": _motif_truchet_arc, "truchet_line": _motif_truchet_line,
        "truchet_circle": _motif_truchet_circle,
    }

    MOTIF_FN_KEYS = list(MOTIF_FN.keys())
    _motif_cycle = ["diamond", "hexagon", "star", "cross", "spiral", "scales"]

    def _draw_motif(cx, cy, sz, angle=0):
        # Determine motif function (may change per tile in mosaic_shuffle)
        fn_key = effective_motif
        if anim_mode == "mosaic_shuffle":
            phase = math.sin(cx * 0.12 + cy * 0.08) * math.pi
            flip = 0.5 + 0.5 * math.sin(t * 1.5 * anim_speed + phase)
            idx = int(flip * (len(_motif_cycle) - 1))
            fn_key = _motif_cycle[idx]
        fn = MOTIF_FN.get(fn_key, _motif_diamond)

        # Deform parameter (0-1): controls motif-specific internal geometry
        if anim_mode == "deform":
            deform = 0.5 + 0.5 * math.sin(cx * 0.04 + cy * 0.03 + t * 1.2 * anim_speed)
        else:
            deform = 0.5

        # Color
        if anim_mode == "color_drift":
            c = _drift_color(cx, cy)
        elif anim_mode == "color_wave":
            h = int(128 + 127 * math.sin(cx * 0.05 + cy * 0.03 + t * 1.8 * anim_speed))
            s = int(155 + 100 * math.cos(cx * 0.04 - cy * 0.06 + t * 1.5 * anim_speed))
            v = int(150 + 100 * math.sin(cx * 0.06 + cy * 0.07 + t * 2.0 * anim_speed))
            c = (h, s, v)
        else:
            c = _tile_color(color_variation)
            if pal and pal != "none" and pal in PALETTES:
                pal_colors = PALETTES[pal]
                if pal_colors:
                    c = rng.choice(pal_colors)

        # Rotation
        a = angle + rng.uniform(-rotation_noise, rotation_noise) if rotation_noise > 0 else angle
        a += _tile_angle_offset(cx, cy)

        # Vortex spin
        if anim_mode == "vortex_spin":
            d = math.sqrt(cx * cx + cy * cy) * 0.008
            a += d * 360.0 * (0.5 + 0.5 * math.sin(t * 0.8 * anim_speed))

        # Scale
        sv = _tile_scale(cx, cy) * (1 + rng.uniform(-scale_variation, scale_variation) if scale_variation > 0 else 1)
        gap_off = _tile_gap_offset(cx, cy)
        sv = sv * max(0.1, 1.0 - gap_off / (sz + 1))

        # Position with orbit mode
        px = cx + _tile_pos_dx(cx, cy)
        py = cy + _tile_pos_dy(cx, cy)
        if anim_mode == "orbit":
            orbit_r = 8.0
            orbit_a = math.atan2(cy, cx) + t * 1.5 * anim_speed
            px += orbit_r * math.cos(orbit_a)
            py += orbit_r * math.sin(orbit_a)

        # Echo mode: draw ghost copies trailing behind
        if anim_mode == "echo":
            for ei in range(3):
                fade = 0.15 + 0.2 * (1.0 - ei / 3.0)
                ghost = (int(c[0] * fade), int(c[1] * fade), int(c[2] * fade))
                dt = t * 1.5 * anim_speed + ei * 1.2
                ex = px + 15 * math.cos(dt + ei * 2.0)
                ey = py + 15 * math.sin(dt + ei * 1.5)
                fn(draw, ex, ey, sz * sv * (0.5 + 0.2 * ei), ghost, angle=a - 10 * ei, deform=deform)

        # Glow mode: draw concentric outline rings that pulse
        if anim_mode == "glow":
            glow_mod = 0.5 + 0.5 * math.sin(t * 1.5 * anim_speed + math.sqrt(cx*cx+cy*cy) * 0.03)
            for gi in range(3):
                gs = sz * sv * (1.0 + (gi + 1) * 0.25 * glow_mod)
                gf = int(40 + 60 * (1.0 - gi / 3.0))
                gcol = (min(255, c[0] + gf), min(255, c[1] + gf), min(255, c[2] + gf))
                fn(draw, px, py, gs, gcol, angle=a, deform=deform)

        # Main motif
        fn(draw, px, py, sz * sv, c, angle=a, deform=deform)

    # ── Penrose tiling ──────────────────────────────────────────────────
    if effective_group == "penrose":
        cx, cy = W // 2, H // 2
        max_r = max(W, H) * 0.7
        import math as m2
        n_petals = 10
        for i in range(n_petals):
            a = m2.radians(360 * i / n_petals)
            for j in range(3):
                r = max_r * (j + 1) / 4
                px = cx + r * m2.cos(a + j * 0.3)
                py = cy + r * m2.sin(a + j * 0.3)
                sz = max(tile_size // 2, 20) - j * 5
                c = _tile_color(color_variation)
                if "kite" in motif:
                    _motif_penrose_kite(draw, px, py, sz, c, angle=m2.degrees(a))
                else:
                    _motif_penrose_dart(draw, px, py, sz, c, angle=m2.degrees(a))
        capture_frame("06", np.array(img).astype(np.float32) / 255.0)
        save(img, mn(6, "wallpaper-group"), out_dir)
        return

    # ── Truchet tiling ──────────────────────────────────────────────────
    if effective_group == "truchet":
        for ty in range(0, H + tile_size, tile_size):
            for tx in range(0, W + tile_size, tile_size):
                _draw_motif(tx, ty, tile_size - gap)
        capture_frame("06", np.array(img).astype(np.float32) / 255.0)
        save(img, mn(6, "wallpaper-group"), out_dir)
        return

    # ── Grid helpers ────────────────────────────────────────────────────

    def _rect_grid(spacing_x, spacing_y, offset_x=0, offset_y=0):
        for ty in range(-tile_size, H + spacing_y, spacing_y):
            for tx in range(-tile_size, W + spacing_x, spacing_x):
                yield tx + offset_x, ty + offset_y

    def _hex_grid():
        w = tile_size * 0.866
        for ty in range(-tile_size, H + tile_size * 2, int(tile_size * 1.5)):
            for tx in range(-tile_size, W + tile_size * 2, int(w * 2)):
                yield int(tx), int(ty)
                yield int(tx + w), int(ty + tile_size * 0.75)

    def _tri_grid():
        for ty in range(-tile_size, H + tile_size, tile_size):
            row_off = tile_size // 2 if (ty // tile_size) % 2 else 0
            for tx in range(-tile_size + row_off, W + tile_size, tile_size):
                yield tx, ty

    import math as m2

    # Rectangular groups
    if effective_group in ("p1", "p2", "pm", "pg", "pmm", "pmg", "pgg", "cmm"):
        spacing = tile_size + gap
        for tx, ty in _rect_grid(spacing, spacing):
            c_angle = 0
            if effective_group == "p2":
                c_angle = 180 * (rng.randint(0, 1))
            elif effective_group == "pm":
                row = ty // spacing
                c_angle = 180 * (row % 2)
            elif effective_group == "pg":
                gl = spacing // 2 * ((ty // spacing) % 2)
                tx += gl
                row = ty // spacing
                c_angle = 180 * (row % 2)
            elif effective_group == "pmm":
                c_angle = 90 * (rng.randint(0, 3))
            elif effective_group == "pmg":
                c_angle = 90 + 180 * (rng.randint(0, 1))
            elif effective_group == "pgg":
                c_angle = 90 * (rng.randint(0, 3))
            elif effective_group == "cmm":
                row = ty // spacing
                if row % 2:
                    tx += spacing // 2
                c_angle = 180 * (row % 2)
            sz = min(tile_size - gap, int(spacing * 0.8))
            _draw_motif(tx, ty, sz, angle=c_angle + global_rot)

    # Square groups
    elif effective_group in ("p4", "p4m", "p4g"):
        spacing = tile_size + gap
        for tx, ty in _rect_grid(spacing, spacing):
            c_angle = 90 * (rng.randint(0, 3))
            if effective_group == "p4m":
                row, col = ty // spacing, tx // spacing
                c_angle = (row % 2) * 180 + (col % 2) * 90
            elif effective_group == "p4g":
                row, col = ty // spacing, tx // spacing
                c_angle = 45 + 90 * ((row + col) % 4)
            sz = min(tile_size - gap, spacing - gap)
            _draw_motif(tx, ty, sz, angle=c_angle + global_rot)

    # Triangular groups
    elif effective_group in ("p3", "p3m1", "p31m"):
        spacing = tile_size + gap
        for tx, ty in _tri_grid():
            c_angle = 120 * (rng.randint(0, 2))
            sz = min(tile_size - gap, spacing - gap)
            if motif == "triangle":
                row = ty // tile_size
                col = tx // tile_size
                c_angle = 180 * ((row + col) % 2) if effective_group == "p3m1" else 0
                sz = int(sz * 1.1)
            _draw_motif(tx, ty, sz, angle=c_angle + global_rot)

    # Hexagonal groups
    elif effective_group in ("p6", "p6m"):
        for tx, ty in _hex_grid():
            c_angle = 60 * (rng.randint(0, 5))
            sz = min(tile_size - gap, int(tile_size * 0.7))
            if effective_group == "p6m":
                row = ty // tile_size
                c_angle = 30 + 60 * (row % 6)
            _draw_motif(tx, ty, sz, angle=c_angle + global_rot)

    # Escher interlocking
    elif effective_group == "escher":
        spacing = tile_size + gap
        for tx, ty in _rect_grid(spacing, spacing):
            row = ty // spacing
            if row % 2:
                tx += spacing // 2
            sz = min(tile_size - gap, spacing - gap)
            _draw_motif(tx, ty, sz, angle=0 + global_rot)

    # Islamic geometric
    elif effective_group == "islamic":
        spacing = tile_size + gap
        for tx, ty in _rect_grid(spacing, spacing):
            sz = min(int(tile_size * 0.9), spacing - gap)
            _draw_motif(tx, ty, sz, angle=45 + global_rot)

    # Fallback: p1
    else:
        spacing = tile_size + gap
        for tx, ty in _rect_grid(spacing, spacing):
            sz = min(tile_size - gap, spacing - gap)
            _draw_motif(tx, ty, sz, angle=0 + global_rot)

    capture_frame("06", np.array(img).astype(np.float32) / 255.0)
    save(img, mn(6, "wallpaper-group"), out_dir)


