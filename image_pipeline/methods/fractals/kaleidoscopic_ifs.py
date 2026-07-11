from __future__ import annotations
import math
from pathlib import Path

import numpy as np

from ...core.registry import method
from ...core.utils import (
    save, norm, mn, seed_all, W, H, PALETTES,
    write_field, write_scalars, write_mask,
)
from ...core.animation import capture_frame


# ── Vectorised KIFS folds (operate on complex ndarrays) ──────────────────────

def _box_fold(z: np.ndarray, s: float) -> np.ndarray:
    """Per-axis abs fold: z -> 2s - z when |z| > s (Mandelbox-style)."""
    zr = z.real
    zi = z.imag
    zr = np.where(zr > s, 2.0 * s - zr, zr)
    zr = np.where(zr < -s, -2.0 * s - zr, zr)
    zi = np.where(zi > s, 2.0 * s - zi, zi)
    zi = np.where(zi < -s, -2.0 * s - zi, zi)
    return zr + 1j * zi


def _rot_fold(z: np.ndarray, n: int, extra_rot: float = 0.0) -> np.ndarray:
    """Kaleidoscopic wedge fold: mirror-fold angle into a 2pi/n wedge."""
    a = np.angle(z) + extra_rot
    r = np.abs(z)
    period = 2.0 * math.pi / max(2, int(n))
    a = np.mod(a, period)
    a = np.where(a > period * 0.5, period - a, a)  # mirror
    return r * np.exp(1j * a)


# ═══════════════════════════════════════════════════════════════════════════

@method(
    id="351",
    name="Kaleidoscopic IFS",
    category="fractals",
    tags=["kifs", "fractal", "kaleidoscope", "symmetric", "escape-time", "expanded", "animation"],
    params={
        "system": {
            "description": "KIFS fold set (box / kaleidoscopic / inversion)",
            "choices": ["box", "kaleidoscopic", "inversion"],
            "default": "kaleidoscopic",
        },
        "iterations": {
            "description": "escape-time iteration count",
            "min": 4, "max": 40, "default": 18,
        },
        "scale": {
            "description": "KIFS scale / expansion factor",
            "min": 1.5, "max": 3.5, "default": 2.5,
        },
        "box_size": {
            "description": "box-fold half-width",
            "min": 0.1, "max": 2.0, "default": 1.0,
        },
        "folds": {
            "description": "rotational wedge fold count (kaleidoscope symmetry)",
            "min": 2, "max": 12, "default": 6,
        },
        "fold_rot": {
            "description": "wedge rotation offset (radians)",
            "min": 0.0, "max": 6.2832, "default": 0.4,
        },
        "c_real": {
            "description": "IFS constant real part",
            "min": -2.0, "max": 2.0, "default": -1.1,
        },
        "c_imag": {
            "description": "IFS constant imaginary part",
            "min": -2.0, "max": 2.0, "default": 0.5,
        },
        "escape_radius": {
            "description": "divergence threshold",
            "min": 2.0, "max": 30.0, "default": 4.0,
        },
        "center_x": {"description": "view center x", "min": -2.0, "max": 2.0, "default": 0.0},
        "center_y": {"description": "view center y", "min": -2.0, "max": 2.0, "default": 0.0},
        "zoom": {"description": "view zoom (>1 zooms in)", "min": 0.3, "max": 4.0, "default": 1.0},
        "color_mode": {
            "description": "coloring (escape_time / palette / angle / orbit_trap)",
            "choices": ["escape_time", "palette", "angle", "orbit_trap"],
            "default": "escape_time",
        },
        "palette_name": {"description": "palette name (palette/escape modes)", "default": "vapor"},
        "anim_mode": {
            "description": "animation mode (none/spin/pulse_scale/morph/color_cycle)",
            "choices": ["none", "spin", "pulse_scale", "morph", "color_cycle"],
            "default": "none",
        },
        "anim_speed": {"description": "animation speed multiplier", "min": 0.1, "max": 5.0, "default": 1.0},
    },
)
def method_kaleidoscopic_ifs(out_dir: Path, seed: int, params=None):
    """2D Kaleidoscopic IFS (KIFS) escape-time fractal.

    Implements the kaleidoscopic iterated-function-system technique
    (a.k.a. the "Kali set" family) made popular by modern fractal-mandala
    art: a point z is repeatedly transformed by a sequence of *folds* —
    an axis-aligned abs/box fold and a rotational wedge (mirror) fold —
    interleaved with a scale + offset (KIFS). Points that exceed the
    escape radius are colored by their smooth iteration count; points that
    stay bounded trace the dense, symmetric attractor. The technique is the
    2D ancestor of the Mandelbox / Mandelbulb and of Apollonian (circle-
    inversion) gasket fractals.

    Three fold systems are offered:
      box           — box-fold + scale + constant (Mandelbox-style)
      kaleidoscopic — box-fold + rotational wedge-fold + scale (classic KIFS)
      inversion     — circle inversion (z -> z - 1/z) + scale (Apollonian/Kleinian limit set)

    Architecture B: the image is recomputed each frame from a fixed seed;
    anim_mode drives per-frame re-calls (Architecture B re-call path). The
    CPU path is the authoritative export.

    Params:
        system:        fold set (box / kaleidoscopic / inversion)
        iterations:    escape-time iteration count (4-40)
        scale:         KIFS scale / expansion factor (1.5-3.5)
        box_size:      box-fold half-width (0.1-2.0)
        folds:         rotational wedge fold count (2-12)
        fold_rot:      wedge rotation offset in radians (0-6.28)
        c_real/c_imag: IFS constant offset
        escape_radius: divergence threshold (2-30)
        center_x/y:    view pan
        zoom:          view zoom
        color_mode:    escape_time / palette / angle / orbit_trap
        palette_name:  palette for palette/escape modes
        time:          animation clock (0-6.28)
        anim_mode:     none / spin / pulse_scale / morph / color_cycle
        anim_speed:    animation speed (0.1-5.0)
    """
    if params is None:
        params = {}
    seed_all(seed)
    rng = np.random.default_rng(seed)

    system = str(params.get("system", "kaleidoscopic"))
    iterations = int(params.get("iterations", 18))
    iterations = max(4, min(40, iterations))
    scale = float(params.get("scale", 2.5))
    box_size = float(params.get("box_size", 1.0))
    folds = int(params.get("folds", 6))
    folds = max(2, min(12, folds))
    fold_rot = float(params.get("fold_rot", 0.4))
    c_real = float(params.get("c_real", -1.1))
    c_imag = float(params.get("c_imag", 0.5))
    escape_radius = float(params.get("escape_radius", 10.0))
    center_x = float(params.get("center_x", 0.0))
    center_y = float(params.get("center_y", 0.0))
    zoom = float(params.get("zoom", 1.0))
    color_mode = str(params.get("color_mode", "escape_time"))
    pal_name = str(params.get("palette_name", "vapor"))
    anim_mode = str(params.get("anim_mode", "none"))
    anim_speed = float(params.get("anim_speed", 1.0))

    t = float(params.get("time", 0.0))
    if anim_mode == "none":
        t = 0.0
    _t = t * anim_speed

    # ── Animation modulation (Architecture B: per-frame re-call) ──
    spin_ang = 0.0
    if anim_mode == "spin":
        spin_ang = _t * 0.3
    elif anim_mode == "pulse_scale":
        scale = scale * (0.8 + 0.2 * (0.5 + 0.5 * math.sin(_t * 0.4)))
    elif anim_mode == "morph":
        box_size = box_size * (0.6 + 0.4 * (0.5 + 0.5 * math.sin(_t * 0.3)))
        fold_rot = (fold_rot + 0.5 * math.sin(_t * 0.25)) % (2 * math.pi)
    elif anim_mode == "color_cycle":
        pass  # handled in coloring below

    # ── Build the complex plane ──
    view = 1.5
    aspect = H / float(W)
    x0 = center_x - view / zoom
    x1 = center_x + view / zoom
    y0 = center_y - view / zoom * aspect
    y1 = center_y + view / zoom * aspect
    xs = np.linspace(x0, x1, int(W), dtype=np.float64)
    ys = np.linspace(y0, y1, int(H), dtype=np.float64)
    xg, yg = np.meshgrid(xs, ys)
    z = xg + 1j * yg
    if spin_ang != 0.0:
        z = z * np.exp(1j * spin_ang)

    offset = complex(c_real, c_imag)
    c = offset

    # ── Escape-time KIFS iteration ──
    hh, ww = z.shape
    iter_count = np.zeros((hh, ww), dtype=np.float32)
    trap = np.full((hh, ww), np.inf, dtype=np.float32)
    abs_z = np.ones((hh, ww), dtype=np.float32)
    escaped = np.zeros((hh, ww), dtype=bool)

    for i in range(iterations):
        if system == "box":
            z = _box_fold(z, box_size)
            z = z * scale + c
        elif system == "inversion":
            z = z - 1.0 / (z + 1e-9)
            z = z * scale
        else:  # kaleidoscopic
            z = _box_fold(z, box_size)
            z = _rot_fold(z, folds, fold_rot)
            z = z * scale - c

        mag = np.abs(z)
        trap = np.minimum(trap, mag)
        newly = (~escaped) & (mag > escape_radius)
        if np.any(newly):
            iter_count[newly] = float(i)
            abs_z[newly] = mag[newly]
            escaped |= newly
        if np.all(escaped):
            break

    # pixels that never escaped: bounded attractor (interior)
    interior = ~escaped
    iter_count[interior] = float(iterations)

    # ── Smooth (fractional) iteration count for the escaped exterior ──
    with np.errstate(divide="ignore", invalid="ignore"):
        nu = np.log(np.log(abs_z + 1e-30)) / math.log(2.0)
    smooth = np.where(escaped, iter_count - nu, float(iterations))
    value = smooth / float(iterations)  # ~0..1

    # Orbit-trap field (the KIFS/Kaliset image): bounded pixels trace the
    # attractor; escaped pixels are colored by escape time. Both are dense, so
    # the render is never simply black.
    trap_n = norm(np.log1p(trap))          # 0..1, dense everywhere
    hue_off = float(rng.random() * 6.2832)  # seed-dependent phase

    # ── Coloring ──
    if color_mode == "palette":
        pal = np.array(PALETTES.get(pal_name, PALETTES.get("vapor", [(0, 0, 0), (255, 255, 255)])),
                       dtype=np.float32)
        idx = (trap_n * (len(pal) - 1)).astype(np.int32)
        result = pal[idx].reshape(hh, ww, 3) / 255.0
        ext = norm(value)
        result[escaped] = pal[(ext * (len(pal) - 1)).astype(np.int32)].reshape(hh, ww, 3)[escaped] / 255.0
    elif color_mode == "angle":
        ang = np.angle(z)
        base = (ang / (2 * math.pi) + 0.5 + hue_off / 6.2832) % 1.0
        result = np.stack([
            np.sin(base * 6.2832) * 0.5 + 0.5,
            np.sin(base * 6.2832 + 2.094) * 0.5 + 0.5,
            np.sin(base * 6.2832 + 4.189) * 0.5 + 0.5,
        ], axis=-1)
    elif color_mode == "orbit_trap":
        result = np.stack([
            np.sin(trap_n * 3.0 + hue_off) * 0.5 + 0.5,
            np.sin(trap_n * 3.0 * 0.75 + 2.0 + hue_off) * 0.5 + 0.5,
            np.sin(trap_n * 3.0 * 0.5 + 4.0 + hue_off) * 0.5 + 0.5,
        ], axis=-1)
    else:  # escape_time — escape exterior by smooth iteration, interior by trap
        d = norm(value)
        ext = np.stack([
            np.sin(d * 6.2832 * 2.0 + hue_off) * 0.5 + 0.5,
            np.sin(d * 6.2832 * 1.5 + 2.0 + hue_off) * 0.5 + 0.5,
            np.sin(d * 6.2832 + 4.0 + hue_off) * 0.5 + 0.5,
        ], axis=-1)
        int_ = np.stack([
            np.sin(trap_n * 4.0 + hue_off) * 0.5 + 0.5,
            np.sin(trap_n * 3.0 + 2.0 + hue_off) * 0.5 + 0.5,
            np.sin(trap_n * 2.0 + 4.0 + hue_off) * 0.5 + 0.5,
        ], axis=-1)
        result = np.where(escaped[:, :, None], ext, int_)

    if anim_mode == "color_cycle":
        shift = (math.sin(_t * 0.5) * 0.5 + 0.5)
        result = np.roll(result, int(shift * 64), axis=(0, 1))

    result = np.clip(result, 0.0, 1.0).astype(np.float32)

    # ── Sidecar outputs (Rules #4/#5/#10) ──
    write_field(out_dir, value.astype(np.float32))
    mask = interior.astype(np.float32)  # bounded attractor region
    write_mask(out_dir, mask)
    write_scalars(
        out_dir,
        mean_escape_iter=float(np.mean(iter_count)),
        escape_fraction=float(np.mean(escaped.astype(np.float32))),
        hue_offset=hue_off,
    )

    capture_frame("351", result)
    save(result, mn(351, "Kaleidoscopic IFS"), out_dir)
    return result
