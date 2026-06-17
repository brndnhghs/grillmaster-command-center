"""
Pattern methods — Truchet, Quasicrystal, Moiré, Worley, Wallpaper, etc.
"""
from __future__ import annotations
import math
import random
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from ..core.registry import method
from ..core.utils import save, norm, mn, seed_all, W, H
from ..core.animation import capture_frame
from ..core.utils import PALETTES


def _render_wave_preview(result, h, w, r2=2.0):
    """Preview helper for animated captures — render waves overlay."""
    r = norm(result)
    return np.stack([r * 0.5 + 0.2, r * 0.3 + 0.1, r * 0.6 + 0.2], axis=-1).clip(0, 1)


@method(id="07", name="Truchet Tiles", category="patterns",
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
    "anim_mode": {"description": "animation mode: none, tile_morph, size_wave, gap_pulse", "default": "none"},
    "anim_speed": {"description": "animation speed multiplier", "min": 0.1, "max": 3.0, "default": 0.25},
    "time": {"description": "animation time (0.0-6.28 for phase shift)", "min": 0.0, "max": 6.28, "default": 0.0},
})
def method_truchet(out_dir: Path, seed: int, params=None):
    """Render Truchet tiling patterns with multiple tile types and color modes.

    Truchet tiles are square tiles with patterns that tile seamlessly
    when rotated. Supports arcs, diagonals, crosses, spirals, and more.
    """
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

    from ..core.utils import PALETTES

    # ── Animation: operate on tile grid parameters ──
    effective_tile_type = tile_type
    effective_tile_size = tile_size
    effective_gap = gap
    if anim_mode == "tile_morph":
        # Cycle through tile patterns (~1 full cycle over 3s)
        tile_cycle = ["arcs", "diagonals", "crosses", "chevrons", "circles", "spirals", "rings", "weave"]
        raw_idx = t * 0.48 * anim_speed * len(tile_cycle)
        effective_tile_type = tile_cycle[int(raw_idx) % len(tile_cycle)]
    elif anim_mode == "size_wave":
        # Per-tile size oscillation — wave propagates across grid
        pass  # handled per-tile via continuous size_mod
    elif anim_mode == "gap_pulse":
        # Breathing gap oscillation
        effective_gap = max(0, gap + int(4 * (1.0 + math.sin(t * 0.3 * anim_speed))))

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
    step = effective_tile_size + effective_gap
    cols = W // step + 1
    rows = H // step + 1

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
            ts = max(2, int(effective_tile_size * size_mod))

            # Per-tile rotation and color drift
            angle_off = (t * 360 * anim_speed + idx * 22.5) % 360
            rot = int((t * 4 * anim_speed + idx * 1.618) * 0.5) % 4
            if rot_noise > 0 and random.random() < rot_noise:
                rot = random.randint(0, 3)
                angle_off = rot * 90.0

            # Per-tile hue drift — cycles through palette continuously
            hue_offset = int(6 * math.sin(t * 2 * anim_speed + idx * 0.7))

            color = _tile_color(tx, ty, idx + hue_offset)

            if effective_tile_type == "arcs":
                # Continuous arc rotation — smooth, no discrete jumps
                draw.arc([tx, ty, tx + ts, ty + ts], angle_off, 90 + angle_off, fill=color, width=lw)
                draw.arc([tx, ty, tx + ts, ty + ts], 180 + angle_off, 270 + angle_off, fill=color, width=lw)
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
                # Weave/interlace pattern
                cx, cy = tx + ts // 2, ty + ts // 2
                arm = ts // 3
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


@method(id="02", name="Quasicrystal", category="patterns",
         tags=["classic", "wave", "fast", "expanded", "animation"],
         params={
    "waves": {"description": "number of wave planes", "min": 2, "max": 50, "default": 8},
    "lattice": {"description": "lattice symmetry (penrose/octagonal/dodecagonal/decagonal/tetragonal/hexagon/triangular/quasi/radial/custom)", "default": "penrose"},
    "wave_fn": {"description": "wave function (sin/triangle/square/sawtooth/gabor/gaussian/pulse)", "default": "sin"},
    "colormode": {"description": "color mode (grayscale/palette/heatmap/spectral/fire/ice/plasma/dual_layer)", "default": "heatmap"},
    "frequency": {"description": "wave frequency scale", "min": 0.005, "max": 0.5, "default": 0.05},
    "amplitude": {"description": "wave amplitude", "min": 0.1, "max": 2.0, "default": 1.0},
    "modulation": {"description": "space modulation (none/radial/gaussian/spiral/vortex)", "default": "none"},
    "mod_strength": {"description": "modulation strength", "min": 0.0, "max": 1.0, "default": 0.3},
    "palette": {"description": "color palette name (PALETTES keys)", "default": "vapor"},
    "rotation": {"description": "global rotation offset (radians)", "min": 0.0, "max": 6.2832, "default": 0.0},
    "anim_mode": {"description": "animation mode: none, plane_rotate, freq_sweep, counter_rotate, multi_plane_freq, wave_count_sweep", "default": "none"},
    "anim_speed": {"description": "animation speed multiplier", "min": 0.1, "max": 3.0, "default": 1.0},
    "time": {"description": "animation time (0.0-6.28 for drift)", "min": 0.0, "max": 6.28, "default": 0.0},
})
def method_quasicrystal(out_dir: Path, seed: int, params=None):
    """Render quasicrystal diffraction patterns via wave-plane superposition.

    Generates non-repeating but deterministic patterns by summing wave
    planes at rational/irrational angle relationships. Supports multiple
    lattice symmetries, wave functions, and color modes.
    """
    if params is None:
        params = {}
    t = float(params.get("time", 0.0))
    seed_all(seed)
    yy, xx = np.mgrid[:H, :W].astype(np.float32)
    cx, cy = W / 2.0, H / 2.0

    # ── Params ──
    n_waves = int(params.get("waves", 8))
    lattice = params.get("lattice", "penrose")
    wave_fn = params.get("wave_fn", "sin")
    cmode = params.get("colormode", "heatmap")
    freq = float(params.get("frequency", 0.05))
    amp = float(params.get("amplitude", 1.0))
    mod_type = params.get("modulation", "none")
    mod_str = float(params.get("mod_strength", 0.3))
    pal_name = params.get("palette", "vapor")
    rot = float(params.get("rotation", 0.0))
    anim_mode = params.get("anim_mode", "none")
    anim_speed = float(params.get("anim_speed", 1.0))

    # ── Matplotlib colormap import (with fallback) ──
    try:
        from matplotlib import cm
        _has_mpl = True
    except ImportError:
        _has_mpl = False

    # ── Animation: operate on wave plane parameters ──
    effective_freq = freq
    effective_rot = rot
    if anim_mode == "plane_rotate":
        # All wave plane angles rotate uniformly — the diffraction pattern spins
        effective_rot = rot + t * 0.5 * anim_speed
    elif anim_mode == "freq_sweep":
        # Frequency sweeps up and down — interference fringes zoom in/out
        effective_freq = freq * (0.3 + 0.7 * abs(math.sin(t * 0.3 * anim_speed)))
    elif anim_mode == "counter_rotate":
        # Half the waves rotate forward, half backward — shearing/interference motion
        pass  # applied per-wave in the field builder
    elif anim_mode == "multi_plane_freq":
        # Each wave plane's frequency oscillates out of phase — ripples cross at different rates
        pass  # applied per-wave in the field builder
    elif anim_mode == "wave_count_sweep":
        # Number of wave planes sweeps up and down — complexity of the pattern changes
        n_waves = max(2, int(n_waves * (0.3 + 0.7 * abs(math.sin(t * 0.2 * anim_speed)))))

    # ── Generate wave-plane data (deterministic — pre-computed once) ──
    rng = np.random.default_rng(seed)
    max_waves = n_waves  # n_waves may shrink due to wave_count_sweep, generate enough
    if anim_mode == "wave_count_sweep":
        max_waves = int(params.get("waves", 8))  # original value before potential reduction

    def _lattice_angles(n, sym):
        angles = []
        if sym == "penrose":        # 5-fold - golden ratio based
            phi = math.pi * (1 + math.sqrt(5)) / 2
            for i in range(n):
                angles.append((i * 2 * math.pi / phi) % (2 * math.pi))
        elif sym == "octagonal":    # 8-fold
            for i in range(n):
                angles.append(i * math.pi / 4)
        elif sym == "dodecagonal":  # 12-fold
            for i in range(n):
                angles.append(i * math.pi / 6)
        elif sym == "decagonal":    # 10-fold
            for i in range(n):
                angles.append(i * 2 * math.pi / 10 + rng.uniform(0, 0.01))
        elif sym == "tetragonal":   # 4-fold
            for i in range(n):
                angles.append(i * math.pi / 2 + rng.uniform(0, 0.3))
        elif sym == "hexagon":      # 6-fold
            for i in range(n):
                angles.append(i * math.pi / 3)
        elif sym == "triangular":   # 3-fold
            for i in range(n):
                angles.append(i * 2 * math.pi / 3 + rng.uniform(0, 0.2))
        elif sym == "quasi":        # quasi-random uniform
            angles = list(rng.uniform(0, 2 * math.pi, n))
        elif sym == "radial":        # converging
            for i in range(n):
                angles.append(i * 2 * math.pi / n)
        elif sym == "custom":
            base = rng.uniform(0, 2 * math.pi, n // 2 + 1)
            dither = rng.uniform(0, 0.5, n // 2 + 1)
            angles = list(base) + [a + d for a, d in zip(base[:n - n // 2], dither[:n - n // 2])]
        else:
            angles = [rng.uniform(0, 2 * math.pi) for _ in range(n)]
        return angles[:n]

    base_thetas = _lattice_angles(max_waves, lattice)
    base_thetas = [(a + rot) % (2 * math.pi) for a in base_thetas]
    base_phases = [rng.uniform(0, 2 * math.pi) for _ in range(max_waves)]
    base_freqs = [freq * (0.5 + rng.random()) for _ in range(max_waves)]

    # ── Build wave field from pre-computed per-wave data ──
    xc = xx - cx
    yc = yy - cy

    def _build_field(thetas_a, phases_a, freqs_a, wfn, t_phase, t_rot, n):
        """Build raw wave field. Uses pre-computed theta/phase/freq arrays.
        t_phase: time-based phase shift applied to all waves.
        t_rot: optional per-wave rotation offset [n] or scalar.
        Returns (H,W) float32."""
        if np.ndim(t_rot) == 0:
            t_rot_arr = np.full(n, t_rot)
        else:
            t_rot_arr = np.asarray(t_rot, dtype=np.float32)[:n]
        fld = np.zeros((H, W), dtype=np.float32)
        for i in range(n):
            theta = (thetas_a[i] + t_rot_arr[i]) % (2 * math.pi)
            ph = phases_a[i] + t_phase
            f = freqs_a[i]
            proj = xc * math.cos(theta) + yc * math.sin(theta)
            raw = proj * f + ph
            if wfn == "sin":
                w = np.sin(raw)
            elif wfn == "triangle":
                w = 2 * np.abs(2 * (raw / (2 * math.pi) - np.floor(raw / (2 * math.pi) + 0.5))) - 1
            elif wfn == "square":
                w = np.where(np.sin(raw) >= 0, 1.0, -1.0)
            elif wfn == "sawtooth":
                w = 2 * (raw / (2 * math.pi) - np.floor(raw / (2 * math.pi) + 0.5))
            elif wfn == "gabor":
                gauss = np.exp(-0.5 * (proj * f * 0.5) ** 2)
                w = np.sin(raw) * gauss
            elif wfn == "gaussian":
                w = np.exp(-0.5 * (np.sin(raw) * 2) ** 2)
            elif wfn == "pulse":
                w = np.where(np.abs(np.sin(raw)) > 0.95, 1.0, -0.5)
            else:
                w = np.sin(raw)
            fld += w * amp
        return fld

    t_phase = t * 0.3 * anim_speed
    t_rot = 0.0
    effective_freqs = list(base_freqs)

    if anim_mode == "plane_rotate":
        t_rot = t * 0.5 * anim_speed
    elif anim_mode == "freq_sweep":
        # Scale all wave frequencies uniformly — ripples zoom in/out coherently
        ratio = effective_freq / freq if freq > 0 else 1.0
        for i in range(n_waves):
            effective_freqs[i] = base_freqs[i] * ratio
    elif anim_mode == "counter_rotate":
        # Each wave plane rotates at different speed based on its angle index
        n = n_waves
        t_rot_per = np.empty(n, dtype=np.float32)
        for i in range(n):
            base_angle = base_thetas[i] % (2 * math.pi)
            # Even-indexed waves rotate forward, odd-indexed backward
            sign = 1.0 if i % 2 == 0 else -1.0
            # Speed varies by angle quadrant to create complex interference
            speed = 0.3 + 0.5 * abs(math.sin(base_angle))
            t_rot_per[i] = sign * speed * t * anim_speed
        t_rot = t_rot_per
    elif anim_mode == "multi_plane_freq":
        # Each wave plane's frequency oscillates independently
        for i in range(n_waves):
            offset = i * 0.7  # phase offset per wave
            osc = 0.5 + 0.5 * math.sin(t * 0.25 * anim_speed + offset)
            effective_freqs[i] = base_freqs[i] * (0.5 + osc)

    result = _build_field(base_thetas, base_phases, effective_freqs,
                          wave_fn, t_phase, t_rot, n_waves)

    if mod_type != "none":
        r = np.sqrt(xc ** 2 + yc ** 2)
        max_r = np.sqrt(cx ** 2 + cy ** 2)
        r_norm = r / max_r
        if mod_type == "radial":
            mask = np.exp(-0.5 * (r_norm * 3) ** 2) * (1 - mod_str) + mod_str
        elif mod_type == "gaussian":
            sigma = 0.4 * (1 - mod_str * 0.5)
            mask = np.exp(-0.5 * (r_norm / sigma) ** 2)
        elif mod_type == "spiral":
            theta_r = np.arctan2(yc, xc) + r_norm * 4 * math.pi * mod_str
            mask = 0.5 + 0.5 * np.sin(theta_r)
        elif mod_type == "vortex":
            theta_r = np.arctan2(yc, xc) * 3
            mask = 0.5 + 0.5 * np.sin(r_norm * 8 * math.pi * mod_str + theta_r)
        else:
            mask = 1.0
        result = result * mask

    # ── Normalize ──
    result = norm(result)

    # ── Color ──
    if cmode == "grayscale":
        rgb = np.stack([result, result, result], axis=-1)
    elif cmode == "palette":
        pal = PALETTES.get(pal_name, PALETTES["vapor"])
        idx = (result * (len(pal) - 1)).astype(np.int32)
        pal_arr = np.array(pal, dtype=np.float32) / 255.0
        rgb = pal_arr[idx]
    elif cmode == "heatmap":
        if _has_mpl:
            rgb = cm.inferno(result)[:, :, :3]
        else:
            rgb = np.stack([result, result, result], axis=-1)
    elif cmode == "spectral":
        if _has_mpl:
            rgb = cm.nipy_spectral(result)[:, :, :3]
        else:
            rgb = np.stack([result, result, result], axis=-1)
    elif cmode == "fire":
        r2 = np.clip(result * 1.5, 0, 1)
        rgb = np.stack([r2, result * 0.6, result * 0.2], axis=-1)
    elif cmode == "ice":
        rgb = np.stack([result * 0.2, result * 0.5, 0.5 + result * 0.5], axis=-1)
    elif cmode == "plasma":
        if _has_mpl:
            rgb = cm.plasma(result)[:, :, :3]
        else:
            rgb = np.stack([result, result, result], axis=-1)
    elif cmode == "dual_layer":
        if _has_mpl:
            hi = result > 0.5
            lo = result <= 0.5
            base = np.zeros((H, W, 3), dtype=np.float32)
            base[lo] = cm.viridis(result[lo] * 2)[:, :3]
            base[hi] = cm.inferno((result[hi] - 0.5) * 2)[:, :3]
            rgb = base
        else:
            rgb = np.stack([result, result, result], axis=-1)
    else:
        rgb = np.stack([result, result, result], axis=-1)

    rgb = np.clip(rgb, 0, 1).astype(np.float32)
    capture_frame("02", rgb)
    save(rgb, mn(2, "quasicrystal"), out_dir)


@method(id="03", name="Moiré", category="patterns",
         tags=["classic", "wave", "fast", "expanded", "animation"],
         params={
    "grids": {"description": "number of overlaid grids/layers", "min": 2, "max": 12, "default": 3},
    "pattern": {"description": "pattern type (radial/linear/concentric/spiral/wave/honeycomb/hexagon/triangle/circle_grid/fractal/checkerboard/star/bullseye)", "default": "linear"},
    "operation": {"description": "blend operation (multiply/min/add/max/difference/xor/divide/average/overlay/screen/exclusion/negation/luminosity)", "default": "multiply"},
    "colormode": {"description": "color mode (grayscale/rainbow/heatmap/palette/spectral/fire/ice/dual_layer)", "default": "rainbow"},
    "palette": {"description": "color palette name", "default": "vapor"},
    "frequency": {"description": "base frequency", "min": 0.005, "max": 0.5, "default": 0.06},
    "freq_variation": {"description": "frequency variation between layers", "min": 0.0, "max": 1.0, "default": 0.3},
    "rotation": {"description": "rotation between layers (radians)", "min": 0.0, "max": 3.1416, "default": 0.15},
    "offset_mode": {"description": "offset between layers (none/linear/radial/random)", "default": "linear"},
    "amplitude": {"description": "pattern contrast/amplitude", "min": 0.1, "max": 2.0, "default": 1.0},
    "thickness": {"description": "line thickness multiplier", "min": 0.2, "max": 5.0, "default": 1.0},
    "wobble": {"description": "wobble distortion of grid lines", "min": 0.0, "max": 3.0, "default": 0.0},
    "anim_mode": {"description": "animation mode: none, layer_rotate, op_morph, pattern_morph", "default": "none"},
    "anim_speed": {"description": "animation speed multiplier", "min": 0.1, "max": 3.0, "default": 0.25},
    "time": {"description": "animation time (0.0-6.28 for phase drift)", "min": 0.0, "max": 6.28, "default": 0.0},
})
def method_moire(out_dir: Path, seed: int, params=None):
    """Render Moiré interference patterns by overlaying transformed grids.

    Combines multiple grid layers with different operations to produce
    interference patterns. Supports radial, linear, spiral, and more
    pattern geometries with rotation, frequency variation, and wobble.
    """
    if params is None:
        params = {}
    t = float(params.get("time", 0.0))
    seed_all(seed)  # seed is fixed — animation from continuous time params only
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[:H, :W].astype(np.float32)
    cx, cy = W / 2.0, H / 2.0
    xc = xx - cx
    yc = yy - cy

    n_grids = int(params.get("grids", 3))
    pattern = params.get("pattern", "linear")
    operation = params.get("operation", "multiply")
    cmode = params.get("colormode", "rainbow")
    pal_name = params.get("palette", "vapor")
    freq = float(params.get("frequency", 0.06))
    freq_var = float(params.get("freq_variation", 0.3))
    rot = float(params.get("rotation", 0.15))
    offset_mode = params.get("offset_mode", "linear")
    amp = float(params.get("amplitude", 1.0))
    thick = float(params.get("thickness", 1.0))
    wobble = float(params.get("wobble", 0.0))
    anim_mode = params.get("anim_mode", "none")
    anim_speed = float(params.get("anim_speed", 1.0))
    from ..core.utils import PALETTES

    # ── Matplotlib colormap import (with fallback) ──
    try:
        from matplotlib import cm
        _has_mpl = True
    except ImportError:
        _has_mpl = False

    # ── Animation: operate on layer parameters, scaled for 0→2π time range ──
    effective_pattern = pattern
    effective_operation = operation
    effective_rot = rot
    # Cross-fade state for smooth morph transitions
    effective_next_op = operation
    effective_op_fade = 0.0
    effective_next_pattern = pattern
    effective_pat_fade = 0.0

    if anim_mode == "layer_rotate":
        # Smooth bounce: sin² avoids the cusp of abs(sin)
        effective_rot = rot * (0.5 + 0.5 * math.sin(t * 1.5 * anim_speed) ** 2)
    elif anim_mode == "op_morph":
        # Slowly cycle through blend operations (~1 full cycle) with cross-fade
        ops_list = ["multiply", "add", "difference", "screen", "overlay", "exclusion",
                    "divide", "average", "xor", "negation", "luminosity", "min", "max"]
        n_ops = len(ops_list)
        raw_idx = t * 0.95 * anim_speed
        idx_a = int(raw_idx) % n_ops
        idx_b = (idx_a + 1) % n_ops
        fade = raw_idx - int(raw_idx)
        effective_operation = ops_list[idx_a]
        effective_next_op = ops_list[idx_b]
        effective_op_fade = fade
    elif anim_mode == "pattern_morph":
        # Slowly cycle through pattern types (~1 full cycle) with cross-fade
        pts_list = ["linear", "radial", "spiral", "wave", "honeycomb", "star"]
        n_pts = len(pts_list)
        raw_idx = t * 0.95 * anim_speed
        idx_a = int(raw_idx) % n_pts
        idx_b = (idx_a + 1) % n_pts
        fade = raw_idx - int(raw_idx)
        effective_pattern = pts_list[idx_a]
        effective_next_pattern = pts_list[idx_b]
        effective_pat_fade = fade

    r = np.sqrt(xc ** 2 + yc ** 2)
    max_r = np.sqrt(cx ** 2 + cy ** 2) or 1
    theta = np.arctan2(yc, xc)

    def _grid_pattern(x, y, fx, fy, phase, pat, thick_scale=1.0):
        """Generate a single grid layer for the given pattern type."""
        if pat == "radial":
            rr = r * fx + phase
            return np.sin(rr) * thick_scale
        elif pat == "linear":
            val = (x * fx + y * fy + phase)
            return np.sin(val) * thick_scale
        elif pat == "concentric":
            rr = r * fx + phase
            return np.sin(rr) * thick_scale
        elif pat == "spiral":
            sp = r * fx + theta * fy * 3 + phase
            return np.sin(sp) * thick_scale
        elif pat == "wave":
            wx = x * fx + phase
            wy = y * fy + phase
            return np.sin(wx) * np.cos(wy) * thick_scale
        elif pat == "honeycomb":
            hx = x * fx + phase
            hy = y * fy * 0.866 + phase
            d = np.sin(hx)
            d2 = np.sin(hx * 0.5 + hy * 0.866)
            d3 = np.sin(hx * 0.5 - hy * 0.866)
            return (d + d2 + d3) / 3 * thick_scale
        elif pat == "hexagon":
            hx = x * fx + phase
            hy = y * fy + phase
            return np.sin(hx) * np.sin(hy) * thick_scale
        elif pat == "triangle":
            tx = x * fx + phase
            ty = y * fy + phase
            return (np.sin(tx) + np.sin(tx * 0.5 + ty * 0.866) + np.sin(tx * 0.5 - ty * 0.866)) / 3 * thick_scale
        elif pat == "circle_grid":
            rr = r * fx + phase
            ang = theta * 6 + phase * 0.5
            return np.sin(rr) * np.cos(ang) * thick_scale
        elif pat == "fractal":
            rr = r * fx + phase
            ang = theta * 3 + phase * 0.3
            return (np.sin(rr + np.sin(ang * 3 + rr * 0.2))) * thick_scale
        elif pat == "checkerboard":
            return np.sin(x * fx + phase) * np.sin(y * fy + phase) * thick_scale
        elif pat == "star":
            ang = theta * 5 + phase * 0.5
            rr = r * fx + phase
            return np.sin(rr) * np.cos(ang) * thick_scale
        elif pat == "bullseye":
            rr = r * fx + phase
            return np.sin(rr * math.pi) * thick_scale
        return np.sin(x * fx + y * fy + phase) * thick_scale

    def _blend_layers(layers_in, op):
        """Blend a list of normalized [-1,1] layers using the given operation. Returns raw result."""
        if len(layers_in) == 0:
            return np.zeros((H, W), dtype=np.float32)
        if len(layers_in) == 1:
            return layers_in[0]
        if op == "multiply":
            return np.prod(np.stack(layers_in, axis=-1), axis=-1)
        elif op == "min":
            return np.min(np.stack(layers_in, axis=-1), axis=-1)
        elif op == "max":
            return np.max(np.stack(layers_in, axis=-1), axis=-1)
        elif op == "add":
            return np.sum(np.stack(layers_in, axis=-1), axis=-1)
        elif op == "difference":
            return np.abs(layers_in[0] - layers_in[1] if len(layers_in) >= 2 else layers_in[0])
        elif op == "xor":
            return np.bitwise_xor(
                ((layers_in[0] + 1) * 127.5).astype(np.int32),
                ((layers_in[1] + 1) * 127.5).astype(np.int32)
            ).astype(np.float32) / 255.0 * 2 - 1
        elif op == "divide":
            return layers_in[0] / (np.abs(layers_in[1]) + 0.1) if len(layers_in) >= 2 else layers_in[0]
        elif op == "average":
            return np.mean(np.stack(layers_in, axis=-1), axis=-1)
        elif op == "overlay":
            a = (layers_in[0] + 1) / 2
            b = (layers_in[1] + 1) / 2 if len(layers_in) >= 2 else a
            r = np.where(a < 0.5, 2 * a * b, 1 - 2 * (1 - a) * (1 - b))
            return r * 2 - 1
        elif op == "screen":
            a = (layers_in[0] + 1) / 2
            b = (layers_in[1] + 1) / 2 if len(layers_in) >= 2 else a
            r = 1 - (1 - a) * (1 - b)
            return r * 2 - 1
        elif op == "exclusion":
            a = (layers_in[0] + 1) / 2
            b = (layers_in[1] + 1) / 2 if len(layers_in) >= 2 else a
            r = a + b - 2 * a * b
            return r * 2 - 1
        elif op == "negation":
            a = (layers_in[0] + 1) / 2
            b = (layers_in[1] + 1) / 2 if len(layers_in) >= 2 else a
            r = 1 - np.abs(a + b - 1)
            return r * 2 - 1
        elif op == "luminosity":
            return layers_in[0] * (1 - amp * 0.3) + layers_in[1] * amp * 0.3 if len(layers_in) >= 2 else layers_in[0]
        return layers_in[0]

    # ── Pre-compute per-layer random data (deterministic) ──
    layer_data = []
    for i in range(n_grids):
        fi = freq * (1.0 + freq_var * rng.uniform(-1, 1))
        fxi = fi * (0.5 + rng.uniform(0, 1))
        fyi = fi * (0.5 + rng.uniform(0, 1))
        base_phase = rng.uniform(0, 2 * math.pi)
        angle_i = i * rot + rng.uniform(-0.02, 0.02)
        cos_a = math.cos(angle_i)
        sin_a = math.sin(angle_i)
        # Offset
        if offset_mode == "linear":
            ox = i * 5.0
            oy = i * 5.0
        elif offset_mode == "radial":
            ox = math.cos(angle_i) * i * 8.0
            oy = math.sin(angle_i) * i * 8.0
        elif offset_mode == "random":
            ox = rng.uniform(-20, 20)
            oy = rng.uniform(-20, 20)
        else:
            ox, oy = 0.0, 0.0
        layer_data.append((fxi, fyi, base_phase, cos_a, sin_a, ox, oy))

    # ── Build layers ──
    def _build_layers(pat, rot_val):
        """Build layers for a given pattern and rotation value. Returns list of normalized [-1,1] arrays."""
        out = []
        for i, (fxi, fyi, base_phase, cos_a, sin_a, ox, oy) in enumerate(layer_data):
            # Continuous frequency oscillation per layer — wired to anim_speed
            freq_mod = 1.0 + 0.25 * math.sin(t * 0.8 * anim_speed + i * 0.7)
            fi_mod = fxi * freq_mod
            fy_mod = fyi * freq_mod
            phase_i = base_phase + t * (i + 1) * 0.15 * anim_speed

            # Rotate coordinates
            angle_i = i * rot_val
            ca = math.cos(angle_i)
            sa = math.sin(angle_i)
            rx = xc * ca - yc * sa + ox
            ry = xc * sa + yc * ca + oy

            # Wobble — wired to anim_speed
            if wobble > 0:
                wx = np.sin(ry * 0.1 + t * anim_speed) * wobble
                wy = np.cos(rx * 0.1 + t * 1.3 * anim_speed) * wobble
                rx = rx + wx
                ry = ry + wy

            layer = _grid_pattern(rx, ry, fi_mod, fy_mod, phase_i, pat, thick)
            out.append(layer)
        # Normalize layers to [-1, 1]
        for i in range(len(out)):
            lmin, lmax = out[i].min(), out[i].max()
            if lmax - lmin > 1e-8:
                out[i] = 2 * (out[i] - lmin) / (lmax - lmin) - 1.0
        return out

    layers = _build_layers(effective_pattern, effective_rot)
    result = _blend_layers(layers, effective_operation)
    result = norm(result)

    # ── Cross-fade for op_morph: blend layers with both operations ──
    if anim_mode == "op_morph" and effective_op_fade > 0.0:
        result_b = _blend_layers(layers, effective_next_op)
        result_b = norm(result_b)
        result = result * (1.0 - effective_op_fade) + result_b * effective_op_fade

    # ── Cross-fade for pattern_morph: rebuild layers with next pattern ──
    if anim_mode == "pattern_morph" and effective_pat_fade > 0.0:
        layers_b = _build_layers(effective_next_pattern, effective_rot)
        result_b = _blend_layers(layers_b, effective_operation)
        result_b = norm(result_b)
        result = result * (1.0 - effective_pat_fade) + result_b * effective_pat_fade

    # ── Color ──
    if cmode == "grayscale":
        rgb = np.stack([result, result, result], axis=-1)
    elif cmode == "rainbow":
        hue = result * 2 * math.pi
        rgb = np.stack([
            np.sin(hue) * 0.5 + 0.5,
            np.sin(hue + 2.094) * 0.5 + 0.5,
            np.sin(hue + 4.189) * 0.5 + 0.5
        ], axis=-1)
    elif cmode == "heatmap":
        if _has_mpl:
            rgb = cm.inferno(result)[:, :, :3]
        else:
            rgb = np.stack([result, result, result], axis=-1)
    elif cmode == "palette":
        pal = PALETTES.get(pal_name, PALETTES["vapor"])
        idx = (result * (len(pal) - 1)).astype(np.int32)
        pal_arr = np.array(pal, dtype=np.float32) / 255.0
        rgb = pal_arr[idx]
    elif cmode == "spectral":
        if _has_mpl:
            rgb = cm.nipy_spectral(result)[:, :, :3]
        else:
            rgb = np.stack([result, result, result], axis=-1)
    elif cmode == "fire":
        r2 = np.clip(result * 1.5, 0, 1)
        rgb = np.stack([r2, result * 0.6, result * 0.2], axis=-1)
    elif cmode == "ice":
        rgb = np.stack([result * 0.2, result * 0.5, 0.5 + result * 0.5], axis=-1)
    elif cmode == "dual_layer":
        if _has_mpl:
            hi = result > 0.5
            lo = result <= 0.5
            base = np.zeros((H, W, 3), dtype=np.float32)
            base[lo] = cm.viridis(result[lo] * 2)[:, :3]
            base[hi] = cm.inferno((result[hi] - 0.5) * 2)[:, :3]
            rgb = base
        else:
            rgb = np.stack([result, result, result], axis=-1)
    else:
        rgb = np.stack([result, result, result], axis=-1)

    rgb = np.clip(rgb, 0, 1).astype(np.float32)
    capture_frame("03", rgb)
    save(rgb, mn(3, "moire"), out_dir)


@method(id="04", name="Worley Noise", category="patterns",
         tags=["classic", "cellular", "fast", "expanded", "animation"],
         params={
    "points": {"description": "number of feature points", "min": 5, "max": 500, "default": 60},
    "distance": {"description": "distance metric (euclidean/manhattan/minkowski/chebyshev/angular)", "default": "euclidean"},
    "feature": {"description": "feature index for each pixel (F1=closest, F2=2nd closest, Fn=nth)", "min": 1, "max": 4, "default": 1},
    "colormode": {"description": "color mode (grayscale/palette/heatmap/spectral/fire/ice/dual_layer/flat_shaded/crackle)", "default": "heatmap"},
    "palette": {"description": "color palette name", "default": "vapor"},
    "jitter": {"description": "point position jitter (0=grid, 1=full random)", "min": 0.0, "max": 1.0, "default": 1.0},
    "tile_size": {"description": "spatial hash tile size for grid acceleration", "min": 16, "max": 128, "default": 64},
    "fractal": {"description": "fractal Worley layers (1=off, 2-4=layered FBM)", "min": 1, "max": 4, "default": 1},
    "fractal_gain": {"description": "amplitude scaling per fractal layer", "min": 0.1, "max": 1.0, "default": 0.5},
    "cell_border": {"description": "cell edge highlight width (0=off)", "min": 0, "max": 20, "default": 0},
    "anim_mode": {"description": "animation mode: none, point_drift, metric_morph, feature_sweep", "default": "none"},
    "anim_speed": {"description": "animation speed multiplier", "min": 0.1, "max": 3.0, "default": 0.5},
    "time": {"description": "animation time for point drift", "min": 0.0, "max": 6.28, "default": 0.0},
})
def method_worley_noise(out_dir: Path, seed: int, params=None):
    """Render Worley (Voronoi cell) noise with GPU-free vectorized KD-tree.

    Generates cellular textures based on distance to the nearest N feature
    points. Supports multiple distance metrics, fractal layering, and
    animation via point drift.
    """
    if params is None:
        params = {}
    t = float(params.get("time", 0.0))
    anim_mode = params.get("anim_mode", "none")
    anim_speed = float(params.get("anim_speed", 0.5))
    seed_all(seed)
    rng = np.random.default_rng(seed)
    n_points = int(params.get("points", 60))
    dist_metric = params.get("distance", "euclidean")
    feature_idx = int(params.get("feature", 1))
    cmode = params.get("colormode", "heatmap")
    pal_name = params.get("palette", "vapor")
    jitter = float(params.get("jitter", 1.0))
    tile_size = int(params.get("tile_size", 64))
    fractal_layers = int(params.get("fractal", 1))
    fractal_gain = float(params.get("fractal_gain", 0.5))
    cell_border = int(params.get("cell_border", 0))

    yy, xx = np.mgrid[:H, :W].astype(np.float32)

    # ── Matplotlib/scipy import (with fallback) ──
    try:
        from matplotlib import cm
        _has_mpl = True
    except ImportError:
        _has_mpl = False
    try:
        from scipy.ndimage import sobel
        _has_scipy = True
    except ImportError:
        _has_scipy = False

    # ── Animation: operate on distance metric, feature index, or point drift ──
    effective_metric = dist_metric
    effective_feature = feature_idx
    effective_drift = t * anim_speed
    if anim_mode == "metric_morph":
        metric_cycle = ["euclidean", "manhattan", "chebyshev", "minkowski", "angular"]
        idx = int((t / (2 * math.pi)) * len(metric_cycle) * anim_speed) % len(metric_cycle)
        effective_metric = metric_cycle[idx]
    elif anim_mode == "feature_sweep":
        effective_feature = max(1, min(4, int(1 + 3 * (0.5 + 0.5 * math.sin(t * anim_speed)))))
    # point_drift uses effective_drift directly in _generate_points

    def _generate_points(n, jit, drift):
        """Generate feature points with optional jitter and time drift."""
        if jit < 0.01:
            # Regular grid layout
            side = int(math.ceil(math.sqrt(n)))
            gx, gy = np.meshgrid(np.linspace(0, W, side, endpoint=False),
                                  np.linspace(0, H, side, endpoint=False))
            pts = np.stack([gx.ravel(), gy.ravel()], axis=-1).astype(np.float32)
            # Add tiny jitter to avoid exact grid artifacts
            pts += rng.uniform(-1, 1, pts.shape).astype(np.float32)
            return pts[:n]
        else:
            pts = rng.random((n, 2)).astype(np.float32)
            pts[:, 0] *= W
            pts[:, 1] *= H
            # Time drift
            if drift != 0:
                angle = rng.uniform(0, 2 * math.pi, n)
                drift_dist = rng.uniform(0, 15, n).astype(np.float32)
                pts[:, 0] += np.cos(angle) * drift_dist * drift * 0.1
                pts[:, 1] += np.sin(angle) * drift_dist * drift * 0.1
                pts[:, 0] = np.clip(pts[:, 0], 0, W - 1)
                pts[:, 1] = np.clip(pts[:, 1], 0, H - 1)
            return pts

    def _distance_matrix(pts, xs, ys, metric):
        """Compute distance from each point to each pixel."""
        dx = xs[np.newaxis, :, :] - pts[:, np.newaxis, np.newaxis, 0]  # (n, H, W)
        dy = ys[np.newaxis, :, :] - pts[:, np.newaxis, np.newaxis, 1]
        if metric == "euclidean":
            return np.sqrt(dx ** 2 + dy ** 2)
        elif metric == "manhattan":
            return np.abs(dx) + np.abs(dy)
        elif metric == "chebyshev":
            return np.maximum(np.abs(dx), np.abs(dy))
        elif metric == "minkowski":
            p = 3.0
            return (np.abs(dx) ** p + np.abs(dy) ** p) ** (1.0 / p)
        elif metric == "angular":
            # Angle from point to pixel (for star-burst effects)
            return np.arctan2(dy, dx) % (2 * math.pi)
        return np.sqrt(dx ** 2 + dy ** 2)

    # ── Build result ──
    if fractal_layers > 1:
        result = np.zeros((H, W), dtype=np.float32)
        total_amp = 0.0
        for layer in range(fractal_layers):
            n_layer = max(5, n_points // (layer + 1))
            scale = 1.0 / (layer + 1)
            jit_layer = max(0.1, jitter * (1.0 - layer * 0.2))
            pts = _generate_points(n_layer, jit_layer, effective_drift * (layer + 1))
            dist = _distance_matrix(pts, xx, yy, effective_metric)
            # Get the k-th nearest distance where k = effective_feature
            sorted_dist = np.sort(dist, axis=0)
            k_idx = min(effective_feature, sorted_dist.shape[0] - 1)
            layer_val = sorted_dist[k_idx]
            amp = fractal_gain ** layer
            result += norm(layer_val) * amp * scale
            total_amp += amp * scale
        result = norm(result / total_amp)
    else:
        pts = _generate_points(n_points, jitter, effective_drift)
        dist = _distance_matrix(pts, xx, yy, effective_metric)
        sorted_dist = np.sort(dist, axis=0)
        k_idx = min(effective_feature, sorted_dist.shape[0] - 1)
        raw = sorted_dist[k_idx]
        result = norm(raw)

    # ── Cell borders ──
    if cell_border > 0 and _has_scipy:
        # Detect cells by finding the nearest point's index
        dist = _distance_matrix(_generate_points(n_points, jitter, 0), xx, yy, effective_metric)
        nearest = np.argmin(dist, axis=0)
        # Sobel edge detect
        edge_x = sobel(nearest.astype(np.float32), axis=1)
        edge_y = sobel(nearest.astype(np.float32), axis=0)
        edges = np.sqrt(edge_x ** 2 + edge_y ** 2) > 0.01
        # Darken result at edges
        result = np.where(edges, result * (1.0 - min(cell_border / 20.0, 0.8)), result)

    # ── Color ──
    if cmode == "grayscale":
        rgb = np.stack([result, result, result], axis=-1)
    elif cmode == "palette":
        pal = PALETTES.get(pal_name, PALETTES["vapor"])
        idx = (result * (len(pal) - 1)).astype(np.int32)
        pal_arr = np.array(pal, dtype=np.float32) / 255.0
        rgb = pal_arr[idx]
    elif cmode == "heatmap":
        if _has_mpl:
            rgb = cm.inferno(result)[:, :, :3]
        else:
            rgb = np.stack([result, result, result], axis=-1)
    elif cmode == "spectral":
        if _has_mpl:
            rgb = cm.nipy_spectral(result)[:, :, :3]
        else:
            rgb = np.stack([result, result, result], axis=-1)
    elif cmode == "fire":
        r2 = np.clip(result * 1.5, 0, 1)
        rgb = np.stack([r2, result * 0.6, result * 0.2], axis=-1)
    elif cmode == "ice":
        rgb = np.stack([result * 0.2, result * 0.5, 0.5 + result * 0.5], axis=-1)
    elif cmode == "dual_layer":
        if _has_mpl:
            hi = result > 0.5
            lo = result <= 0.5
            base = np.zeros((H, W, 3), dtype=np.float32)
            base[lo] = cm.viridis(result[lo] * 2)[:, :3]
            base[hi] = cm.inferno((result[hi] - 0.5) * 2)[:, :3]
            rgb = base
        else:
            rgb = np.stack([result, result, result], axis=-1)
    elif cmode == "flat_shaded":
        if _has_mpl and _has_scipy:
            base = cm.magma(result)[:, :, :3]
            gx = sobel(result, axis=1)
            gy = sobel(result, axis=0)
            light = np.clip((gx * 0.5 + gy * 0.5 + 0.5) * 0.8 + 0.2, 0, 1)
            rgb = base * np.stack([light, light, light], axis=-1)
            rgb = np.clip(rgb, 0, 1)
        else:
            rgb = np.stack([result, result, result], axis=-1)
    elif cmode == "crackle":
        # Lightning-like crackle effect using multi-F1/F2 difference
        pts = _generate_points(n_points, jitter, effective_drift)
        dist = _distance_matrix(pts, xx, yy, effective_metric)
        sorted_dist = np.sort(dist, axis=0)
        f1 = sorted_dist[0]
        f2 = sorted_dist[min(1, sorted_dist.shape[0] - 1)]
        crackle = np.clip((f2 - f1) * 5, 0, 1)
        rgb = np.stack([crackle, crackle, crackle], axis=-1)
    else:
        rgb = np.stack([result, result, result], axis=-1)

    rgb = np.clip(rgb, 0, 1).astype(np.float32)
    capture_frame("04", rgb)
    save(rgb, mn(4, "worley-noise"), out_dir)


@method(id="06", name="Wallpaper Group", category="patterns",
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
    "anim_mode": {"description": "animation mode: none, motif_morph, group_morph", "default": "none"},
    "anim_speed": {"description": "animation speed multiplier", "min": 0.1, "max": 3.0, "default": 0.25},
    "time": {"description": "animation time (drives ripple wave + rotation)", "min": 0.0, "max": 6.28, "default": 0.0},
})
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

    from ..core.utils import PALETTES, quantize_to_palette

    # ── Animation: morph motif or group ──
    effective_motif = motif
    effective_group = group
    if anim_mode == "motif_morph":
        motif_cycle = ["diamond", "triangle", "hexagon", "star", "cross", "spiral", "wave", "scales"]
        idx = int((t / (2 * math.pi)) * len(motif_cycle) * anim_speed) % len(motif_cycle)
        effective_motif = motif_cycle[idx]
    elif anim_mode == "group_morph":
        group_cycle = ["p1", "p2", "pm", "pg", "pmm", "pmg", "pgg", "cmm", "p4", "p4m", "p4g", "p3", "p3m1", "p31m", "p6", "p6m"]
        idx = int((t / (2 * math.pi)) * len(group_cycle) * anim_speed) % len(group_cycle)
        effective_group = group_cycle[idx]

    img = Image.new("RGB", (W, H), (10, 10, 18))
    draw = ImageDraw.Draw(img)

    # ── Helper: random per-tile color ───────────────────────────────────
    def _tile_color(variation=1.0):
        r = int(rng.randint(30, 255) * (1 - variation * 0.5) + 128 * variation * 0.5)
        g = int(rng.randint(30, 220) * (1 - variation * 0.5) + 128 * variation * 0.5)
        b = int(rng.randint(50, 200) * (1 - variation * 0.5) + 128 * variation * 0.5)
        return (r, g, b)

    def _inv_color(c):
        return (255 - c[0], 255 - c[1], 255 - c[2])

    # Time-driven scale wave — tiles pulse in a ripple across the grid
    scale_wave = 0.7 + 0.6 * abs(math.sin(t * 0.5))  # stronger breathing

    # ── Animation: continuous rotation sweep ──
    global_rot = t * 120 * anim_speed  # degrees of continuous rotation over 3s

    # Per-tile phase offset computed in _draw_motif

    # ── Motif drawing functions ─────────────────────────────────────────

    def _motif_diamond(d, cx, cy, sz, color, angle=0):
        hsz = sz / 2
        pts = [(cx, cy - hsz), (cx + hsz, cy), (cx, cy + hsz), (cx - hsz, cy)]
        if angle:
            import math as m2
            c, s = m2.cos(m2.radians(angle)), m2.sin(m2.radians(angle))
            pts = [(cx + (x-cx)*c - (y-cy)*s, cy + (x-cx)*s + (y-cy)*c) for x,y in pts]
        d.polygon(pts, fill=color, outline=(10,10,18))
        inner_sz = max(2, sz * 0.3)
        d.ellipse([cx - inner_sz, cy - inner_sz, cx + inner_sz, cy + inner_sz],
                  fill=_inv_color(color), outline=None)

    def _motif_triangle(d, cx, cy, sz, color, angle=0):
        h = sz * (3**0.5) / 2
        pts = [(cx, cy - h/2), (cx + sz/2, cy + h/2), (cx - sz/2, cy + h/2)]
        if angle:
            import math as m2
            c, s = m2.cos(m2.radians(angle)), m2.sin(m2.radians(angle))
            pts = [(cx + (x-cx)*c - (y-cy)*s, cy + (x-cx)*s + (y-cy)*c) for x,y in pts]
        d.polygon(pts, fill=color, outline=(10,10,18))

    def _motif_hexagon(d, cx, cy, sz, color, angle=0):
        pts = []
        import math as m2
        for i in range(6):
            a = m2.radians(60 * i - 30 + angle)
            pts.append((cx + sz/2 * m2.cos(a), cy + sz/2 * m2.sin(a)))
        d.polygon(pts, fill=color, outline=(10,10,18))

    def _motif_star(d, cx, cy, sz, color, angle=0):
        n = star_rays
        import math as m2
        outer = sz / 2
        inner = outer * 0.4
        pts = []
        for i in range(n * 2):
            a = m2.radians(360 * i / (n * 2) - 90 + angle)
            r = outer if i % 2 == 0 else inner
            pts.append((cx + r * m2.cos(a), cy + r * m2.sin(a)))
        d.polygon(pts, fill=color, outline=(10,10,18))

    def _motif_cross(d, cx, cy, sz, color, angle=0):
        arm_w = sz * 0.2
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

    def _motif_spiral(d, cx, cy, sz, color, angle=0):
        import math as m2
        turns = 3
        pts = [(cx, cy)]
        for ti in range(int(sz * turns)):
            a = m2.radians(ti * 10 + angle)
            r = ti / (sz * turns / (sz/2))
            pts.append((cx + r * m2.cos(a), cy + r * m2.sin(a)))
        if len(pts) > 2:
            d.line(pts, fill=color, width=max(1, int(sz/30)))

    def _motif_wave(d, cx, cy, sz, color, angle=0):
        import math as m2
        hsz = sz / 2
        pts = []
        for x in range(int(-hsz), int(hsz)):
            y = m2.sin((x + hsz) / sz * 4 * m2.pi + m2.radians(angle)) * hsz * 0.3
            pts.append((cx + x, cy + y))
        if len(pts) > 2:
            d.line(pts, fill=color, width=max(1, int(sz/20)))
        pts2 = []
        for x in range(int(-hsz), int(hsz)):
            y = m2.sin((x + hsz) / sz * 4 * m2.pi + m2.radians(angle) + m2.pi) * hsz * 0.3
            pts2.append((cx + x, cy + y + hsz * 0.3))
        if len(pts2) > 2:
            d.line(pts2, fill=_inv_color(color), width=max(1, int(sz/25)))

    def _motif_scales(d, cx, cy, sz, color, angle=0):
        r = sz * 0.35
        for ox, oy in [(0, 0), (r*0.6, -r*0.6), (-r*0.6, -r*0.6)]:
            d.ellipse([cx + ox - r, cy + oy - r, cx + ox + r, cy + oy + r],
                      fill=None, outline=color, width=max(1, int(sz/50)))

    def _motif_escher_bird(d, cx, cy, sz, color, angle=0):
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

    def _motif_escher_fish(d, cx, cy, sz, color, angle=0):
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

    def _motif_islamic_star(d, cx, cy, sz, color, angle=0):
        n = star_rays
        import math as m2
        outer = sz * 0.45
        inner = outer * 0.35
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

    def _motif_arabesque(d, cx, cy, sz, color, angle=0):
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

    def _motif_penrose_kite(d, cx, cy, sz, color, angle=0):
        import math as m2
        phi = (1 + 5**0.5) / 2
        a = sz * 0.3
        b = a * phi
        pts = [(cx, cy - b), (cx + a, cy), (cx, cy + b), (cx - a, cy)]
        if angle:
            c, s = m2.cos(m2.radians(angle)), m2.sin(m2.radians(angle))
            pts = [(cx + (x-cx)*c - (y-cy)*s, cy + (x-cx)*s + (y-cy)*c) for x,y in pts]
        d.polygon(pts, fill=color, outline=(10,10,18))

    def _motif_penrose_dart(d, cx, cy, sz, color, angle=0):
        import math as m2
        phi = (1 + 5**0.5) / 2
        a = sz * 0.3
        b = a * phi
        pts = [(cx, cy - b), (cx + a*0.5, cy), (cx, cy + b*0.4), (cx - a*0.5, cy)]
        if angle:
            c, s = m2.cos(m2.radians(angle)), m2.sin(m2.radians(angle))
            pts = [(cx + (x-cx)*c - (y-cy)*s, cy + (x-cx)*s + (y-cy)*c) for x,y in pts]
        d.polygon(pts, fill=color, outline=(10,10,18))

    def _motif_truchet_arc(d, cx, cy, sz, color, angle=0):
        hsz = sz / 2
        opts = rng.randint(0, 3)
        corners = [(cx - hsz, cy - hsz), (cx + hsz, cy - hsz),
                   (cx + hsz, cy + hsz), (cx - hsz, cy + hsz)]
        for j, (sx, sy) in enumerate(corners):
            if j in (opts, (opts + 1) % 4):
                d.arc([sx, sy, sx + hsz, sy + hsz],
                      90 * j, 90 * (j + 1), fill=color, width=max(1, int(sz/20)))

    def _motif_truchet_line(d, cx, cy, sz, color, angle=0):
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

    def _motif_truchet_circle(d, cx, cy, sz, color, angle=0):
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

    def _draw_motif(cx, cy, sz, angle=0):
        fn = MOTIF_FN.get(effective_motif, _motif_diamond)
        c = _tile_color(color_variation)
        if pal and pal != "none" and pal in PALETTES:
            pal_colors = PALETTES[pal]
            if pal_colors:
                c = rng.choice(pal_colors)
        a = angle + rng.uniform(-rotation_noise, rotation_noise) if rotation_noise > 0 else angle
        # Per-tile scale wave: ripple propagates across the grid
        tile_phase = (cx * 0.02 + cy * 0.03 + t * 0.8)
        ripple = 0.7 + 0.3 * (0.5 + 0.5 * math.sin(tile_phase))
        sv = ripple * (1 + rng.uniform(-scale_variation, scale_variation) if scale_variation > 0 else 1)
        fn(draw, cx, cy, sz * sv, c, angle=a)

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


@method(id="08", name="Phyllotaxis", category="patterns",
        tags=["classic", "nature", "fast", "animated", "expanded"],
        params={
    "points": {"description": "number of points", "min": 100, "max": 50000, "default": 4000},
    "angle": {"description": "divergence angle in degrees (137.508=golden)", "min": 1, "max": 360, "default": 137.508},
    "spiral_type": {"description": "spiral arrangement",
                    "default": "classic",
                    "choices": ["classic", "sunflower", "alternating", "double", "custom"]},
    "point_shape": {"description": "point shape",
                    "default": "circle",
                    "choices": ["circle", "square", "diamond", "petal", "ring", "star"]},
    "point_size_min": {"description": "minimum point radius", "min": 1, "max": 20, "default": 1},
    "point_size_max": {"description": "maximum point radius", "min": 1, "max": 30, "default": 4},
    "fade": {"description": "fade opacity toward edges (0=off, 1=full)", "min": 0.0, "max": 1.0, "default": 0.0},
    "palette": {"description": "color palette name from PALETTES", "default": "none"},
    "radius_scale": {"description": "spread factor (compact=smaller)", "min": 0.5, "max": 10.0, "default": 6.0},
    "rotation": {"description": "global rotation in degrees", "min": 0, "max": 360, "default": 0},
    "center_x": {"description": "center X offset (-1 to 1, 0=center)", "min": -1.0, "max": 1.0, "default": 0.0},
    "center_y": {"description": "center Y offset (-1 to 1, 0=center)", "min": -1.0, "max": 1.0, "default": 0.0},
    "petal_angle": {"description": "rotate each petal toward center (degrees)", "min": 0, "max": 90, "default": 0},
    "anim_mode": {"description": "animation mode: none, spiral_morph, shape_morph", "default": "none"},
    "anim_speed": {"description": "animation speed multiplier", "min": 0.1, "max": 3.0, "default": 0.25},
    "time": {"description": "animation time (drives rotation + seed)", "min": 0.0, "max": 6.28, "default": 0.0},
})
def method_phyllotaxis(out_dir: Path, seed: int, params=None):
    """
    Phyllotaxis spiral pattern generator with multiple spiral types,
    point shapes, color palettes, fade, and animation support.
    """
    if params is None:
        params = {}
    t = float(params.get("time", 0.0))
    seed_all(seed)  # fixed seed — no random calls needed, animation via continuous rotation

    n_points = int(params.get("points", 4000))
    angle_deg = float(params.get("angle", 137.508))
    spiral_type = params.get("spiral_type", "classic")
    point_shape = params.get("point_shape", "circle")
    psize_min = float(params.get("point_size_min", 1))
    psize_max = float(params.get("point_size_max", 4))
    fade = float(params.get("fade", 0.0))
    pal = params.get("palette", "none")
    radius_scale = float(params.get("radius_scale", 6.0))
    rotation = float(params.get("rotation", 0))
    cx_off = float(params.get("center_x", 0.0))
    cy_off = float(params.get("center_y", 0.0))
    petal_angle = float(params.get("petal_angle", 0))
    anim_mode = params.get("anim_mode", "none")
    anim_speed = float(params.get("anim_speed", 0.25))

    from ..core.utils import PALETTES, quantize_to_palette

    # ── Animation: morph spiral type and point shape with cross-fade ──
    effective_spiral_type = spiral_type
    effective_point_shape = point_shape
    effective_next_spiral_type = spiral_type
    effective_next_point_shape = point_shape
    effective_morph_fade = 0.0
    if anim_mode == "spiral_morph":
        spiral_cycle = ["classic", "sunflower", "alternating", "double"]
        n_s = len(spiral_cycle)
        raw_idx = t * 0.4 * anim_speed * n_s
        idx_a = int(raw_idx) % n_s
        idx_b = (idx_a + 1) % n_s
        effective_spiral_type = spiral_cycle[idx_a]
        effective_next_spiral_type = spiral_cycle[idx_b]
        effective_morph_fade = raw_idx - int(raw_idx)
    elif anim_mode == "shape_morph":
        shape_cycle = ["circle", "square", "diamond", "petal", "ring", "star"]
        n_sh = len(shape_cycle)
        raw_idx = t * 0.4 * anim_speed * n_sh
        idx_a = int(raw_idx) % n_sh
        idx_b = (idx_a + 1) % n_sh
        effective_point_shape = shape_cycle[idx_a]
        effective_next_point_shape = shape_cycle[idx_b]
        effective_morph_fade = raw_idx - int(raw_idx)

    img = Image.new("RGB", (W, H), (10, 10, 18))
    cx = W // 2 + int(cx_off * W * 0.4)
    cy = H // 2 + int(cy_off * H * 0.4)
    golden_angle = 137.508

    # ── Pre-compute palette colors if needed ────────────────────────────
    pal_colors = None
    if pal and pal != "none" and pal in PALETTES:
        pal_colors = PALETTES[pal]

    # ── Point drawing functions ──────────────────────────────────────────

    def _draw_circle(d, x, y, r, color, alpha=1.0):
        if alpha < 0.05:
            return
        d.ellipse([x - r, y - r, x + r, y + r], fill=color)

    def _draw_square(d, x, y, r, color, alpha=1.0):
        if alpha < 0.05:
            return
        d.rectangle([x - r, y - r, x + r, y + r], fill=color)

    def _draw_diamond(d, x, y, r, color, alpha=1.0):
        if alpha < 0.05:
            return
        d.polygon([(x, y - r), (x + r, y), (x, y + r), (x - r, y)], fill=color)

    def _draw_petal(d, x, y, r, color, alpha=1.0, petal_rot=0):
        if alpha < 0.05:
            return
        import math as m2
        # 5-petal flower
        for i in range(5):
            a = m2.radians(72 * i + petal_rot)
            px = x + r * 0.6 * m2.cos(a)
            py = y + r * 0.6 * m2.sin(a)
            d.ellipse([px - r * 0.4, py - r * 0.4, px + r * 0.4, py + r * 0.4],
                      fill=color)

    def _draw_ring(d, x, y, r, color, alpha=1.0):
        if alpha < 0.05:
            return
        d.ellipse([x - r, y - r, x + r, y + r], fill=None, outline=color, width=max(1, int(r * 0.3)))

    def _draw_star(d, x, y, r, color, alpha=1.0):
        if alpha < 0.05:
            return
        import math as m2
        pts = []
        for i in range(10):
            a = m2.radians(36 * i - 90)
            rad = r if i % 2 == 0 else r * 0.4
            pts.append((x + rad * m2.cos(a), y + rad * m2.sin(a)))
        d.polygon(pts, fill=color)

    SHAPE_FN = {
        "circle": _draw_circle, "square": _draw_square, "diamond": _draw_diamond,
        "petal": _draw_petal, "ring": _draw_ring, "star": _draw_star,
    }

    # ── Render spiral into an image ──────────────────────────────────────
    def _render_spiral(target_img, spiral_type_val, point_shape_val):
        """Draw a phyllotaxis spiral with given spiral type and point shape into target_img."""
        d = ImageDraw.Draw(target_img)
        base_a = golden_angle
        if spiral_type_val == "sunflower":
            base_a = 99.5
        elif spiral_type_val == "custom":
            base_a = angle_deg
        fn = SHAPE_FN.get(point_shape_val, _draw_circle)
        max_r = max(W, H) * 0.5
        rot_rad = math.radians(rotation + t * 30)
        for n in range(n_points):
            if spiral_type_val == "alternating":
                an = n * math.radians(base_a + (10 if n % 2 == 0 else -10))
            elif spiral_type_val == "double":
                an = n * math.radians(base_a)
                if n % 2 == 0:
                    an += math.radians(180)
            else:
                an = n * math.radians(base_a)
            an += rot_rad
            rr = radius_scale * math.sqrt(n)
            if rr > max_r:
                break
            xx = cx + rr * math.cos(an)
            yy = cy + rr * math.sin(an)
            if 0 <= xx < W and 0 <= yy < H:
                sz = psize_max - (psize_max - psize_min) * (rr / max_r)
                sz = max(psize_min, min(psize_max, sz))
                if pal_colors:
                    ci = n % len(pal_colors)
                    c = pal_colors[ci]
                else:
                    rc = int(180 + 75 * math.sin(an * 3))
                    gc = int(100 + 100 * math.cos(an * 2 + n * 0.01))
                    bc = int(200 + 55 * math.sin(an * 5))
                    c = (rc, gc, bc)
                alpha = 1.0
                if fade > 0:
                    alpha = 1.0 - (rr / max_r) * fade
                    if alpha < 0.05:
                        continue
                if point_shape_val == "petal":
                    pa = math.degrees(math.atan2(cy - yy, cx - xx)) + petal_angle
                    _draw_petal(d, xx, yy, sz, c, alpha, petal_rot=pa)
                else:
                    fn(d, xx, yy, sz, c, alpha)

    # ── Render A (and B if morphing), then blend ──
    img = Image.new("RGB", (W, H), (10, 10, 18))
    _render_spiral(img, effective_spiral_type, effective_point_shape)

    if effective_morph_fade > 0.0:
        img_b = Image.new("RGB", (W, H), (10, 10, 18))
        if anim_mode == "spiral_morph":
            _render_spiral(img_b, effective_next_spiral_type, effective_point_shape)
        else:
            _render_spiral(img_b, effective_spiral_type, effective_next_point_shape)
        img = Image.blend(img, img_b, effective_morph_fade)

    capture_frame("08", np.array(img, dtype=np.float32) / 255.0)
    save(img, mn(8, "phyllotaxis"), out_dir)


# ── method 05: Procedural Noise Generator (v2, fully honed) ────────────

@method(id="05", name="Procedural Noise", category="patterns",
        tags=["classic", "noise", "generative", "animated", "expanded"],
        params={
    "noise_type": {"description": "type of noise",
                    "default": "perlin",
                    "choices": ["perlin", "value", "simplex", "curl",
                                "cloud", "marble", "plasma", "cell",
                                "wood", "terrain", "spot", "ring", "brick"]},
    "style": {"description": "rendering style", "default": "normal",
              "choices": ["normal", "colormap", "posterize", "edge"]},
    "palette": {"description": "color palette (PALETTES name or matplotlib cmap)", "default": "none"},
    "domain_warp": {"description": "domain warp strength (0=none)", "min": 0.0, "max": 10.0, "default": 0.0},
    "warp_mode": {"description": "domain warp style: normal or warped (Inigo Quilez 3-level)",
                  "default": "normal", "choices": ["normal", "warped"]},
    "scale": {"description": "noise frequency scale", "min": 0.1, "max": 10.0, "default": 2.0},
    "octaves": {"description": "number of octaves", "min": 1, "max": 12, "default": 4},
    "lacunarity": {"description": "frequency multiplier per octave", "min": 1.0, "max": 4.0, "default": 2.0},
    "gain": {"description": "amplitude multiplier per octave", "min": 0.1, "max": 1.0, "default": 0.5},
    "time": {"description": "animation time/phase (0..2pi for smooth evolution)", "min": 0.0, "max": 6.28, "default": 0.0},
    "anim_mode": {"description": "animation mode: none, type_morph", "default": "none"},
    "anim_speed": {"description": "animation speed multiplier", "min": 0.1, "max": 3.0, "default": 0.25},
    "cell_points": {"description": "Voronoi cell count (nx*ny, for cell noise)", "min": 20, "max": 500, "default": 96},
    "cell_borders": {"description": "Voronoi cell border thickness (0=off)", "min": 0, "max": 20, "default": 0},
    "cell_colors": {"description": "Voronoi cell colorized (vs grayscale)", "default": False},
    "ring_wobble": {"description": "wood ring wobble distortion (0=perfect circles)", "min": 0.0, "max": 5.0, "default": 2.0},
    "ring_count": {"description": "wood ring count", "min": 5, "max": 100, "default": 30},
    "water_level": {"description": "terrain water level cutoff (0=no water, 1=all water)", "min": 0.0, "max": 1.0, "default": 0.0},
    "erosion": {"description": "terrain simulated erosion strength (0=off)", "min": 0.0, "max": 1.0, "default": 0.0},
})
def method_noise(out_dir: Path, seed: int, params=None):
    """
    Multi-type procedural 2D noise generator with FBM styles, domain warping,
    palette mapping, Voronoi with colored cells, terrain with water/erosion,
    wood with wobble, and 13 noise types.
    """
    if params is None:
        params = {}

    seed_all(seed)

    noise_type = params.get("noise_type", "perlin")
    style = params.get("style", "normal")
    pal = params.get("palette", "none")
    domain_warp = float(params.get("domain_warp", 0.0))
    warp_mode = params.get("warp_mode", "normal")
    scale = float(params.get("scale", 2.0))
    octaves = int(params.get("octaves", 4))
    lacunarity = float(params.get("lacunarity", 2.0))
    gain = float(params.get("gain", 0.5))
    t = float(params.get("time", 0.0))
    anim_mode = params.get("anim_mode", "none")
    anim_speed = float(params.get("anim_speed", 0.25))
    cell_points = int(params.get("cell_points", 96))
    cell_borders = int(params.get("cell_borders", 0))
    cell_colors = bool(params.get("cell_colors", False))
    ring_wobble = float(params.get("ring_wobble", 2.0))
    ring_count = float(params.get("ring_count", 30))
    water_level = float(params.get("water_level", 0.0))
    erosion_strength = float(params.get("erosion", 0.0))

    # ── Matplotlib/scipy import (with fallback) ──
    try:
        import matplotlib.cm as cm
        _has_mpl = True
    except ImportError:
        _has_mpl = False
    try:
        from scipy.ndimage import sobel, gaussian_filter
        _has_scipy = True
    except ImportError:
        _has_scipy = False

    from ..core.utils import PALETTES, quantize_to_palette

    # ── Matplotlib colormap registry ────────────────────────────────────
    _COLORMAP_NAMES = {"viridis", "plasma", "inferno", "magma", "cividis",
                       "twilight", "turbo", "rainbow", "ocean", "terrain",
                       "gist_earth", "gist_ncar", "gist_stern", "gist_heat",
                       "hot", "cool", "copper", "bone", "gray", "pink",
                       "spring", "summer", "autumn", "winter", "RdYlBu",
                       "Spectral", "RdGy", "RdBu", "PiYG", "PRGn", "PuOr",
                       "BrBG", "flag", "prism", "hsv", "jet"}

    def _is_matplotlib_cmap(name: str) -> bool:
        """Check if a palette name is a matplotlib colormap."""
        return name.lower() in _COLORMAP_NAMES

    def _apply_colormap(no: np.ndarray, cmap_name: str) -> np.ndarray:
        """Apply a matplotlib colormap to normalized noise."""
        if not _has_mpl:
            return np.stack([no, no, no], axis=-1)
        try:
            cmap = cm.get_cmap(cmap_name)
            return cmap(no)[:, :, :3].astype(np.float32)
        except Exception:
            return np.stack([no, no, no], axis=-1)

    # ── Lattice noise primitives ────────────────────────────────────────

    from functools import lru_cache

    @lru_cache(maxsize=None)
    def _grad_table(seed_val: int, table_size: int = 256) -> np.ndarray:
        rng = np.random.RandomState(seed_val & 0xFFFFFFFF)
        angles = rng.uniform(0, 2 * np.pi, (table_size, table_size))
        return np.stack([np.cos(angles), np.sin(angles)], axis=-1)  # (T,T,2)

    def _grad_at(x: np.ndarray, y: np.ndarray, table: np.ndarray) -> np.ndarray:
        """Bilinear sample gradient from table at (x,y) normalized coordinates."""
        T = table.shape[0]
        ix = (x % 1).astype(np.int64) % T
        iy = (y % 1).astype(np.int64) % T
        fx = x - np.floor(x)
        fy = y - np.floor(y)
        sx = fx * fx * (3 - 2 * fx)
        sy = fy * fy * (3 - 2 * fy)
        tl = table[iy, ix]
        tr = table[iy, (ix+1)%T]
        bl = table[(iy+1)%T, ix]
        br = table[(iy+1)%T, (ix+1)%T]
        top = tl + (tr - tl) * sx[..., None]
        bot = bl + (br - bl) * sx[..., None]
        return top + (bot - top) * sy[..., None]

    def _lattice_gradient_noise(x: np.ndarray, y: np.ndarray, table: np.ndarray) -> np.ndarray:
        """Gradient noise (Perlin-style) sampled at (x,y)."""
        g = _grad_at(x, y, table)
        dx = x - np.floor(x)
        dy = y - np.floor(y)
        return g[:, :, 0] * dx + g[:, :, 1] * dy

    def _value_noise(x: np.ndarray, y: np.ndarray, table_size: int = 256, seed_val: int = 0) -> np.ndarray:
        """Value noise — bicubically interpolated random lattice."""
        rng = np.random.RandomState(seed_val & 0xFFFFFFFF)
        vals = rng.uniform(-1, 1, (table_size, table_size))
        ix = (x % 1).astype(np.int64) % table_size
        iy = (y % 1).astype(np.int64) % table_size
        fx = x - np.floor(x)
        fy = y - np.floor(y)
        sx = fx * fx * (3 - 2 * fx)
        sy = fy * fy * (3 - 2 * fy)
        tl = vals[iy, ix]
        tr = vals[iy, (ix+1)%table_size]
        bl = vals[(iy+1)%table_size, ix]
        br = vals[(iy+1)%table_size, (ix+1)%table_size]
        return (tl + (tr - tl) * sx + ((bl + (br - bl) * sx) - (tl + (tr - tl) * sx)) * sy)

    # ── Simplex-like noise (gradient on triangular lattice) ─────────────

    def _simplex_noise(x: np.ndarray, y: np.ndarray, table: np.ndarray) -> np.ndarray:
        """Simplified 2D simplex-like gradient noise using a skewed coordinate grid."""
        # Skew factors for triangular lattice
        F2 = 0.5 * (np.sqrt(3) - 1)
        G2 = (3 - np.sqrt(3)) / 6
        s = (x + y) * F2
        i = np.floor(x + s).astype(np.int64)
        j = np.floor(y + s).astype(np.int64)
        t_simplex = (i + j) * G2
        x0 = x - (i - t_simplex)
        y0 = y - (j - t_simplex)
        i1 = np.where(x0 > y0, 1, 0)
        j1 = np.where(x0 > y0, 0, 1)
        x1, y1 = x0 - i1 + G2, y0 - j1 + G2
        x2, y2 = x0 - 1 + 2 * G2, y0 - 1 + 2 * G2
        T = table.shape[0]
        gi = i.astype(np.int64) % T
        gj = j.astype(np.int64) % T
        g0 = table[gj, gi]
        g1 = table[(gj + j1) % T, (gi + i1) % T]
        g2 = table[(gj + 1) % T, (gi + 1) % T]
        t0 = 0.5 - x0 * x0 - y0 * y0
        t1 = 0.5 - x1 * x1 - y1 * y1
        t2 = 0.5 - x2 * x2 - y2 * y2
        t0 = np.where(t0 > 0, t0 * t0 * t0 * (g0[:, :, 0] * x0 + g0[:, :, 1] * y0), 0)
        t1 = np.where(t1 > 0, t1 * t1 * t1 * (g1[:, :, 0] * x1 + g1[:, :, 1] * y1), 0)
        t2 = np.where(t2 > 0, t2 * t2 * t2 * (g2[:, :, 0] * x2 + g2[:, :, 1] * y2), 0)
        return (t0 + t1 + t2) * 70.0

    # ── Curl noise ──────────────────────────────────────────────────────

    def _curl_noise(x: np.ndarray, y: np.ndarray, table: np.ndarray) -> np.ndarray:
        """Curl of a potential field — produces flow-like vector patterns.
        Returns scalar field: magnitude of the curl."""
        eps = 0.01
        # Central differences for partial derivatives
        fx = _lattice_gradient_noise(x + eps, y, table)
        fy = _lattice_gradient_noise(x - eps, y, table)
        gx = _lattice_gradient_noise(x, y + eps, table)
        gy = _lattice_gradient_noise(x, y - eps, table)
        dfdx = (fx - fy) / (2 * eps)
        dfdy = (gx - gy) / (2 * eps)
        # Curl magnitude: |dF/dx * (-dy) + dF/dy * dx| simplified
        return np.sqrt(dfdx**2 + dfdy**2) * (2 * np.pi)

    # ── Octave combiners ────────────────────────────────────────────────

    def _fbm(x: np.ndarray, y: np.ndarray, octaves: int, lacunarity: float, gain: float,
             base_amp: float = 1.0, base_freq: float = 1.0,
             use_gradient: bool = True,
             use_simplex: bool = False,
             use_curl: bool = False) -> np.ndarray:
        """Fractional Brownian Motion — sum of noise octaves."""
        result = np.zeros_like(x)
        amp = base_amp
        freq = base_freq
        if use_curl:
            table = _grad_table(seed * 55555 & 0xFFFFFFFF, 256)
            for _ in range(octaves):
                layer = _curl_noise(x * freq, y * freq, table)
                result += layer * amp
                amp *= gain
                freq *= lacunarity
        elif use_simplex:
            table = _grad_table(seed * 31337 & 0xFFFFFFFF, 256)
            for _ in range(octaves):
                layer = _simplex_noise(x * freq, y * freq, table)
                result += layer * amp
                amp *= gain
                freq *= lacunarity
        else:
            table = _grad_table(seed * 31337 & 0xFFFFFFFF, 256) if use_gradient else None
            for _ in range(octaves):
                if use_gradient:
                    layer = _lattice_gradient_noise(x * freq, y * freq, table)
                else:
                    layer = _value_noise(x * freq, y * freq, seed_val=seed)
                result += layer * amp
                amp *= gain
                freq *= lacunarity
        return result

    def _turbulence(x: np.ndarray, y: np.ndarray, octaves: int, lacunarity: float, gain: float) -> np.ndarray:
        """Absolute value FBM — produces 'fire' / plasma patterns."""
        result = np.zeros_like(x)
        amp = 1.0
        freq = 1.0
        table = _grad_table(seed * 31337 & 0xFFFFFFFF, 256)
        for _ in range(octaves):
            layer = _lattice_gradient_noise(x * freq, y * freq, table)
            result += np.abs(layer) * amp
            amp *= gain
            freq *= lacunarity
        return result

    def _ridged(x: np.ndarray, y: np.ndarray, octaves: int, lacunarity: float, gain: float) -> np.ndarray:
        """Ridged multifractal — sharp ridge-like features (terrain)."""
        result = np.zeros_like(x)
        amp = 1.0
        freq = 1.0
        table = _grad_table(seed * 31337 & 0xFFFFFFFF, 256)
        for _ in range(octaves):
            layer = _lattice_gradient_noise(x * freq, y * freq, table)
            result += (1.0 - np.abs(layer)) * amp
            amp *= gain
            freq *= lacunarity
        return result

    def _billow(x: np.ndarray, y: np.ndarray, octaves: int, lacunarity: float, gain: float) -> np.ndarray:
        """Billow — abs noise with bias (cloud-like)."""
        result = np.zeros_like(x)
        amp = 1.0
        freq = 1.0
        table = _grad_table(seed * 31337 & 0xFFFFFFFF, 256)
        for _ in range(octaves):
            layer = _lattice_gradient_noise(x * freq, y * freq, table)
            result += (np.abs(layer) * 2.0 - 1.0) * amp
            amp *= gain
            freq *= lacunarity
        return result

    # ── Domain warping ──────────────────────────────────────────────────

    def _warp_normal(x: np.ndarray, y: np.ndarray, strength: float) -> tuple[np.ndarray, np.ndarray]:
        """Standard warp: displace (x,y) by low-frequency noise."""
        if strength <= 0:
            return x, y
        table = _grad_table(seed * 55555 & 0xFFFFFFFF, 256)
        warp_scale = 2.0
        dx = _lattice_gradient_noise(x * warp_scale + t, y * warp_scale + t, table)
        dy = _lattice_gradient_noise(x * warp_scale + 10.3, y * warp_scale + 10.3, table)
        w2 = 4.0
        table2 = _grad_table(seed * 77777 & 0xFFFFFFFF, 256)
        ddx = _lattice_gradient_noise(x * w2 + t * 0.5, y * w2 + t * 0.5, table2)
        ddy = _lattice_gradient_noise(x * w2 + 50.0, y * w2 + 50.0, table2)
        return x + dx * strength * 0.1 + ddx * strength * 0.05, \
               y + dy * strength * 0.1 + ddy * strength * 0.05

    def _warp_warped(x: np.ndarray, y: np.ndarray, strength: float) -> tuple[np.ndarray, np.ndarray]:
        """Inigo Quilez-style 3-level domain warping: noise(warp(warp(x,y))).
        Produces dreamlike, organic, ink-in-water textures."""
        if strength <= 0:
            return x, y
        table = _grad_table(seed * 55555 & 0xFFFFFFFF, 256)
        # Level 1: coarse warp
        qx = _lattice_gradient_noise(x + t, y + t, table)
        qy = _lattice_gradient_noise(x + 3.7 + t, y + 3.7 + t, table)
        # Level 2: medium warp
        rx = _lattice_gradient_noise(x + qx * strength * 0.5 + t * 0.7,
                                      y + qy * strength * 0.5 + t * 0.7, table)
        ry = _lattice_gradient_noise(x + qx * strength * 0.5 + 2.4,
                                      y + qy * strength * 0.5 + 2.4, table)
        # Level 3: fine warp — this is what gets sampled
        return x + rx * strength * 0.2, y + ry * strength * 0.2

    # ── Coordinate setup ───────────────────────────────────────────────

    yy, xx = np.mgrid[:H, :W].astype(np.float32)
    xx = xx / W * scale
    yy = yy / H * scale

    # Domain warping
    if domain_warp > 0:
        if warp_mode == "warped":
            xx, yy = _warp_warped(xx, yy, strength=domain_warp)
        else:
            xx, yy = _warp_normal(xx, yy, strength=domain_warp)

    # ── Noise generation ───────────────────────────────────────────────

    # Animation: morph noise types with cross-fade between adjacent types
    effective_noise_type = noise_type
    effective_next_noise_type = noise_type
    effective_morph_fade = 0.0
    if anim_mode == "type_morph":
        type_cycle = ["perlin", "cloud", "marble", "plasma", "wood", "terrain", "spot", "ring", "brick"]
        n_types = len(type_cycle)
        raw_idx = t * 0.4 * anim_speed * n_types
        idx_a = int(raw_idx) % n_types
        idx_b = (idx_a + 1) % n_types
        effective_noise_type = type_cycle[idx_a]
        effective_next_noise_type = type_cycle[idx_b]
        effective_morph_fade = raw_idx - int(raw_idx)

    # Add time offset to coordinates — 3 pixel drift over full animation (visible)
    tx = xx + t * 0.5
    ty = yy + t * 0.5
    if effective_noise_type == "perlin":
        no = _fbm(tx, ty, octaves, lacunarity, gain, use_gradient=True)
    elif effective_noise_type == "value":
        no = _fbm(tx, ty, octaves, lacunarity, gain, use_gradient=False)
    elif effective_noise_type == "simplex":
        no = _fbm(tx, ty, octaves, lacunarity, gain, use_simplex=True)
    elif effective_noise_type == "curl":
        no = _fbm(tx, ty, octaves, lacunarity, gain, use_curl=True)
    elif effective_noise_type == "cloud":
        no = _billow(tx, ty, octaves, lacunarity, gain * 1.5)
    elif effective_noise_type == "marble":
        base = _fbm(tx, ty, octaves, lacunarity, gain, use_gradient=True)
        no = np.sin(base * 3 + tx * 0.5 + np.pi * 2 + t)
    elif effective_noise_type == "plasma":
        no = _turbulence(tx, ty, octaves, lacunarity, gain * 0.7)

    elif effective_noise_type == "wood":
        cx, cy = 0.5, 0.5
        r = np.sqrt((tx - cx) ** 2 + (ty - cy) ** 2)
        theta = np.arctan2(ty - cy, tx - cx)
        # Ring wobble via noise
        table = _grad_table(seed * 77777 & 0xFFFFFFFF, 256)
        wobble = _lattice_gradient_noise(theta * 0.5 + r * 2.0 + t * 0.3,
                                         r * 3.0 + theta * 0.3, table) * ring_wobble * 0.15
        grain = _lattice_gradient_noise(r * 5.0 + theta * 0.5, theta * 2.0, table) * 0.1
        no = np.sin(r * ring_count * 0.5 + wobble + grain + t * 0.2)

    elif effective_noise_type == "terrain":
        grain = _fbm(tx, ty, octaves, lacunarity, gain * 1.2, use_gradient=True)
        bump = _ridged(tx, ty, min(octaves, 6), lacunarity * 1.1, gain * 0.8)
        no = grain * 0.5 + bump * 0.5
        # Erosion (thermal diffusion-like smoothing on steep slopes)
        if erosion_strength > 0.01:
            for _ in range(int(erosion_strength * 5)):
                dx = np.gradient(no, axis=1)
                dy = np.gradient(no, axis=0)
                slope = np.sqrt(dx**2 + dy**2)
                mask = slope > 0.1
                # Diffuse down slope
                laplacian = np.gradient(np.gradient(no, axis=1), axis=1) + \
                            np.gradient(np.gradient(no, axis=0), axis=0)
                no = np.where(mask, no + laplacian * erosion_strength * 0.02, no)
        # Water level cutoff
        if water_level > 0:
            no = np.where(no < water_level, water_level * 0.5, no)

    elif effective_noise_type == "cell" or effective_noise_type == "voronoi":
        rng = np.random.RandomState(seed * 12345 & 0xFFFFFFFF)
        n_points = cell_points
        # Distribute points with jitter on a grid for even coverage
        grid_nx = int(np.sqrt(n_points * W / H))
        grid_ny = max(2, n_points // grid_nx)
        pts = []
        for gy in range(grid_ny):
            for gx in range(grid_nx):
                bx = gx / grid_nx
                by = gy / grid_ny
                jx = rng.uniform(-0.3, 0.3)
                jy = rng.uniform(-0.3, 0.3)
                pts.append(((bx + jx / grid_nx) * W, (by + jy / grid_ny) * H))
        pts = np.array(pts[:n_points], dtype=np.float32)

        # Compute F1 and F2 distances + which point is closest
        dists = np.full((H, W), 1e10, dtype=np.float32)
        dists2 = np.full((H, W), 1e10, dtype=np.float32)
        closest_idx = np.zeros((H, W), dtype=np.int32)
        px = tx * W / scale
        py = ty * H / scale
        for pi, p in enumerate(pts):
            d = np.sqrt((px - p[0]) ** 2 + (py - p[1]) ** 2)
            better = d < dists
            # Shift current F1 to F2 where we found a closer point
            dists2 = np.where(better, dists, dists2)
            dists = np.minimum(dists, d)
            closest_idx = np.where(better, pi, closest_idx)
        # Also fill F2 from remaining points
        for pi, p in enumerate(pts):
            d = np.sqrt((px - p[0]) ** 2 + (py - p[1]) ** 2)
            mask = (d > dists) & (d < dists2)
            if np.any(mask):
                dists2[mask] = d[mask]

        if cell_borders > 0 and _has_scipy:
            # Edge detection on closest_idx gives cell borders
            edge_x = sobel(closest_idx.astype(np.float32), axis=1)
            edge_y = sobel(closest_idx.astype(np.float32), axis=0)
            border = norm(np.abs(edge_x) + np.abs(edge_y))
            border = (border > 0.01).astype(np.float32)

        if cell_colors:
            # Assign each cell a color from the palette or random gradient
            rng2 = np.random.RandomState(seed * 99999 & 0xFFFFFFFF)
            cell_hues = rng2.uniform(0, 1, n_points)
            cell_sats = rng2.uniform(0.6, 1.0, n_points)
            cell_vals = rng2.uniform(0.5, 1.0, n_points)
            # HSV to RGB per pixel
            h = cell_hues[closest_idx]
            s = cell_sats[closest_idx]
            v = cell_vals[closest_idx]
            # Vectorized HSV→RGB
            hi = (h * 6).astype(np.int32) % 6
            f = h * 6 - hi.astype(np.float32)
            p = v * (1 - s)
            q = v * (1 - s * f)
            tt = v * (1 - s * (1 - f))
            result = np.zeros((H, W, 3), dtype=np.float32)
            for ch, (r0, g0, b0) in enumerate([(v, tt, p), (q, v, p), (p, v, tt),
                                               (p, q, v), (tt, p, v), (v, p, q)]):
                mask = hi == ch
                result[mask] = np.stack([r0[mask], g0[mask], b0[mask]], axis=-1)
            no = norm(dists)
        else:
            no = norm(dists)

        if cell_borders > 0:
            border_3ch = np.stack([border] * 3, axis=-1)
            if cell_colors:
                result = result * (1 - border[..., None] * 0.7) + border_3ch * 0.3
            else:
                no = np.where(border > 0.5, 1.0, no * 0.7)
                result = np.stack([no, no, no], axis=-1)

    elif effective_noise_type == "spot":
        # Binary dot patterns — thresholded noise with size variation
        table = _grad_table(seed * 44444 & 0xFFFFFFFF, 256)
        base = _fbm(tx, ty, 3, 2.0, 0.6, use_gradient=True, base_freq=0.5)
        # Dot placement as a 2D grid of sigmoid bumps
        spacing = 8.0 / max(1.0, scale * 0.5)
        sx = np.sin(tx * spacing * np.pi) * np.cos(ty * spacing * np.pi)
        sy = np.sin(ty * spacing * np.pi) * np.cos(tx * spacing * np.pi + 0.5)
        dots = sx * sy
        # Size varies with noise
        size_mod = norm(base) * 0.5 + 0.5
        threshold = 0.7 - size_mod * 0.4
        no = norm(dots)
        no = np.where(no > threshold, 1.0, 0.0)
        # Soften edges
        if _has_scipy:
            no = gaussian_filter(no, sigma=0.5).clip(0, 1)

    elif effective_noise_type == "ring":
        # Concentric rings with noise distortion
        cx, cy = 0.5, 0.5
        r = np.sqrt((tx - cx) ** 2 + (ty - cy) ** 2) * 40.0 / max(1.0, scale)
        table = _grad_table(seed * 88888 & 0xFFFFFFFF, 256)
        ripple = _lattice_gradient_noise(r * 0.1 + t, tx * 0.2 + t * 0.5, table) * 0.3
        grain = _lattice_gradient_noise(tx * 0.5, ty * 0.5, table) * 0.15
        no = np.sin(r + ripple + grain + t) * 0.5 + 0.5
        # Add radial fade
        no = no * (1 - r / 60.0).clip(0, 1)

    elif effective_noise_type == "brick":
        # Tiling brick/checkerboard noise texture
        brick_w = 6.0 / max(0.5, scale)
        brick_h = 3.0 / max(0.5, scale)
        # Offset every other row
        bx = tx * brick_w
        by = ty * brick_h
        row_parity = np.floor(by).astype(np.int32) % 2
        bx = bx + row_parity * 0.5
        ix = (bx % 1).astype(np.float32)
        iy = (by % 1).astype(np.float32)
        # Mortar: thin gaps between bricks
        mortar_x = (ix < 0.08).astype(np.float32)
        mortar_y = (iy < 0.08).astype(np.float32)
        mortar = np.maximum(mortar_x, mortar_y)
        # Brick color variation from noise
        table = _grad_table(seed * 66666 & 0xFFFFFFFF, 256)
        brick_var = _lattice_gradient_noise(tx * 0.3, ty * 0.3, table) * 0.3
        no = np.where(mortar > 0.5, 0.15, 0.5 + brick_var)
        no = norm(no)

    else:
        no = _fbm(tx, ty, octaves, lacunarity, gain, use_gradient=True)

    # ── Cross-fade to next noise type (smooth morph transitions) ──
    if effective_morph_fade > 0.0:
        # Render the second noise type
        if effective_next_noise_type == "perlin":
            no2 = _fbm(tx, ty, octaves, lacunarity, gain, use_gradient=True)
        elif effective_next_noise_type == "value":
            no2 = _fbm(tx, ty, octaves, lacunarity, gain, use_gradient=False)
        elif effective_next_noise_type == "simplex":
            no2 = _fbm(tx, ty, octaves, lacunarity, gain, use_simplex=True)
        elif effective_next_noise_type == "curl":
            no2 = _fbm(tx, ty, octaves, lacunarity, gain, use_curl=True)
        elif effective_next_noise_type == "cloud":
            no2 = _billow(tx, ty, octaves, lacunarity, gain * 1.5)
        elif effective_next_noise_type == "marble":
            base2 = _fbm(tx, ty, octaves, lacunarity, gain, use_gradient=True)
            no2 = np.sin(base2 * 3 + tx * 0.5 + np.pi * 2 + t)
        elif effective_next_noise_type == "plasma":
            no2 = _turbulence(tx, ty, octaves, lacunarity, gain * 0.7)
        elif effective_next_noise_type == "wood":
            cx2, cy2 = 0.5, 0.5
            r2 = np.sqrt((tx - cx2) ** 2 + (ty - cy2) ** 2)
            theta2 = np.arctan2(ty - cy2, tx - cx2)
            table2 = _grad_table(seed * 77777 & 0xFFFFFFFF, 256)
            wobble2 = _lattice_gradient_noise(theta2 * 0.5 + r2 * 2.0 + t * 0.3, r2 * 3.0 + theta2 * 0.3, table2) * ring_wobble * 0.15
            grain2 = _lattice_gradient_noise(r2 * 5.0 + theta2 * 0.5, theta2 * 2.0, table2) * 0.1
            no2 = np.sin(r2 * ring_count * 0.5 + wobble2 + grain2 + t * 0.2)
        elif effective_next_noise_type == "terrain":
            no2 = _fbm(tx, ty, octaves, lacunarity, gain * 1.2, use_gradient=True) * 0.5 + _ridged(tx, ty, min(octaves, 6), lacunarity * 1.1, gain * 0.8) * 0.5
        elif effective_next_noise_type in ("cell", "voronoi"):
            # Use same cell noise for both — cells change too drastically to cross-fade
            no2 = no
        elif effective_next_noise_type == "spot":
            table2 = _grad_table(seed * 44444 & 0xFFFFFFFF, 256)
            base2 = _fbm(tx, ty, 3, 2.0, 0.6, use_gradient=True, base_freq=0.5)
            spacing2 = 8.0 / max(1.0, scale * 0.5)
            sx2 = np.sin(tx * spacing2 * np.pi) * np.cos(ty * spacing2 * np.pi)
            sy2 = np.sin(ty * spacing2 * np.pi) * np.cos(tx * spacing2 * np.pi + 0.5)
            dots2 = sx2 * sy2
            size_mod2 = norm(base2) * 0.5 + 0.5
            threshold2 = 0.7 - size_mod2 * 0.4
            no2 = norm(dots2)
            no2 = np.where(no2 > threshold2, 1.0, 0.0)
        elif effective_next_noise_type == "ring":
            cx2, cy2 = 0.5, 0.5
            r2 = np.sqrt((tx - cx2) ** 2 + (ty - cy2) ** 2) * 40.0 / max(1.0, scale)
            table2 = _grad_table(seed * 88888 & 0xFFFFFFFF, 256)
            ripple2 = _lattice_gradient_noise(r2 * 0.1 + t, tx * 0.2 + t * 0.5, table2) * 0.3
            grain2 = _lattice_gradient_noise(tx * 0.5, ty * 0.5, table2) * 0.15
            no2 = np.sin(r2 + ripple2 + grain2 + t) * 0.5 + 0.5
            no2 = no2 * (1 - r2 / 60.0).clip(0, 1)
        elif effective_next_noise_type == "brick":
            brick_w2 = 6.0 / max(0.5, scale)
            brick_h2 = 3.0 / max(0.5, scale)
            bx2 = tx * brick_w2
            by2 = ty * brick_h2
            row_parity2 = np.floor(by2).astype(np.int32) % 2
            bx2 = bx2 + row_parity2 * 0.5
            ix2 = (bx2 % 1).astype(np.float32)
            iy2 = (by2 % 1).astype(np.float32)
            mortar_x2 = (ix2 < 0.08).astype(np.float32)
            mortar_y2 = (iy2 < 0.08).astype(np.float32)
            mortar2 = np.maximum(mortar_x2, mortar_y2)
            table2 = _grad_table(seed * 66666 & 0xFFFFFFFF, 256)
            brick_var2 = _lattice_gradient_noise(tx * 0.3, ty * 0.3, table2) * 0.3
            no2 = np.where(mortar2 > 0.5, 0.15, 0.5 + brick_var2)
            no2 = norm(no2)
        else:
            no2 = _fbm(tx, ty, octaves, lacunarity, gain, use_gradient=True)
        # Lerp
        no = no * (1.0 - effective_morph_fade) + no2 * effective_morph_fade

    # ── Post-generation styling ─────────────────────────────────────────

    if effective_noise_type != "cell" or not cell_colors:
        no = norm(no)
        # Actually normalize only if not already a color result
        if style == "colormap" and _is_matplotlib_cmap(pal) and pal != "none":
            result = _apply_colormap(no, pal)
        elif style == "colormap":
            result = _apply_colormap(no, "jet" if pal == "none" else pal)
        elif style == "posterize":
            result = np.stack([no, no, no], axis=-1)
            pal_name = pal if pal in PALETTES else "grayscale"
            result = quantize_to_palette(result, pal_name)
        elif style == "edge":
            d = np.gradient(no)
            edge = norm(np.abs(d[0]) + np.abs(d[1]))
            result = np.stack([edge, edge, edge], axis=-1)
        elif pal and pal != "none":
            if _is_matplotlib_cmap(pal):
                result = _apply_colormap(no, pal)
            else:
                result = np.stack([no, no, no], axis=-1)
                result = quantize_to_palette(result, pal)
        else:
            # Default: vibrant colorshift
            cx, cy = W // 2, H // 2
            r = np.sqrt((np.arange(H)[:, None] - cy) ** 2 + (np.arange(W)[None, :] - cx) ** 2)
            theta = np.arctan2(np.arange(H)[:, None] - cy, np.arange(W)[None, :] - cx)
            colored = np.stack([
                np.sin(no * 6 + theta * 0.5) * 0.5 + 0.5,
                np.sin(no * 8 + r * 0.02 + 2.0) * 0.5 + 0.5,
                np.sin(no * 10 + r * 0.03 + 4.0) * 0.5 + 0.5,
            ], axis=-1)
            result = norm(colored)

    save((result * 255).astype(np.uint8) if result.dtype.kind == 'f' else result,
         mn(5, "procedural-noise"), out_dir)
    capture_frame("05", result)


# ── method 09: DELETED — superseded by --filter system (228 effects) ──