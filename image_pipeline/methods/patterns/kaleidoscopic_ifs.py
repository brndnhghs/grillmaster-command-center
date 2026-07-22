from __future__ import annotations
import math

import numpy as np
from PIL import Image

from ...core.registry import method
from ...core.utils import save, norm, mn, seed_all, write_field, write_scalars, W, H, wired_source_lum
from ...core.animation import capture_frame
from image_pipeline.core.spatial import sparam


@method(id='402', name='Kaleidoscopic IFS (Pattern)', category='patterns', tags=['fractal', 'escape-time', 'kaleidoscopic', 'animation', 'color_intrinsic', 'gpu-twin'], params={'iterations': {'description': 'number of fold iterations', 'min': 4, 'max': 24, 'default': 14}, 'scale': {'description': 'fold scale (negative = classic KIFS detail)', 'min': -3.0, 'max': 1.0, 'default': -2.0}, 'fold_angle': {'description': 'rotation between folds (radians)', 'min': 0.0, 'max': 6.2832, 'default': 1.0}, 'offset_x': {"spatial": True, 'description': 'fold offset x', 'min': -3.0, 'max': 3.0, 'default': 1.0}, 'offset_y': {"spatial": True, 'description': 'fold offset y', 'min': -3.0, 'max': 3.0, 'default': 1.0}, 'symmetry': {'description': 'kaleidoscopic symmetry order (2..8)', 'min': 2, 'max': 8, 'default': 6}, 'escape_radius': {'description': 'orbit-escape radius', 'min': 2.0, 'max': 40.0, 'default': 12.0}, 'colormode': {'description': 'color mode (orbit / bands / neon)', 'default': 'orbit'}, 'color_shift': {"spatial": True, 'description': 'palette color offset', 'min': 0.0, 'max': 1.0, 'default': 0.5}, 'anim_mode': {'description': 'animation mode (none/rotate/spin/pulse/zoom)', 'choices': ['none', 'rotate', 'spin', 'pulse', 'zoom'], 'default': 'none'}, 'anim_speed': {'description': 'animation speed multiplier', 'min': 0.1, 'max': 3.0, 'default': 1.0}, 'source': {'description': "wired upstream image's luminance", 'choices': ['none', 'input_image'], 'default': 'none'}}, inputs={'image_in': 'IMAGE'})
def method_kaleidoscopic_ifs(out_dir, seed: int, params=None):
    """Kaleidoscopic IFS fractal (iteration-fold escape-time).

    A KIFS (Kaleidoscopic Iterated Function System) repeatedly applies a
    sequence of folds (mirror + n-fold wedge reflection), rotation, scale and
    offset to each point in the plane. The mirror + wedge fold is what produces
    the characteristic snowflake / kaleidoscopic symmetry; a negative scale is
    what generates the classic self-similar detail (Knighty / Syntopia, ~2010).

    Coloring uses an orbit trap (min distance of the orbit to the origin) or
    smooth iteration bands; the interior (non-escaping region) is filled dark.
    Animated modes perturb the fold angle / scale / view over the animation
    clock so the structure visibly morphs.
    """
    try:
        if params is None:
            params = {}
        t = float(params.get("time", 0.0))
        seed_all(seed)
        rng = np.random.default_rng(seed)

        iters = int(params.get("iterations", 14))
        scale = float(params.get("scale", -2.0))
        fold_angle = float(params.get("fold_angle", 1.0))
        ox = sparam(params, "offset_x", 1.0)
        oy = sparam(params, "offset_y", 1.0)
        sym = max(2, min(8, int(round(float(params.get("symmetry", 6))))))
        escape_r = float(params.get("escape_radius", 12.0))
        colormode = params.get("colormode", "orbit")
        color_shift = sparam(params, "color_shift", 0.5)
        anim_mode = params.get("anim_mode", "none")
        anim_speed = float(params.get("anim_speed", 1.0))

        # ── Animation: perturb fold parameters via the animation clock ──
        if anim_mode == "none":
            _t = 0.0
        else:
            _t = t * anim_speed
        if anim_mode == "rotate":
            fold_angle = fold_angle + _t * 0.6
        elif anim_mode == "pulse":
            scale = scale + 0.4 * math.sin(_t)
        elif anim_mode == "spin":
            # handled below as an initial-point rotation
            pass
        elif anim_mode == "zoom":
            # handled below as a domain-span change
            pass

        # ── Initial point field: map pixels to a small complex-plane region ──
        yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
        # Wired image as a domain-warp source (luminance distorts the pattern grid)
        _src_lum = wired_source_lum(params, xx.shape[1], xx.shape[0])
        if _src_lum is not None:
            xx = xx + (_src_lum - 0.5) * 15.0
            yy = yy + (_src_lum - 0.5) * 15.0

        zx = (xx - W / 2.0) / (W / 2.0) * 3.0
        zy = (yy - H / 2.0) / (H / 2.0) * 3.0
        if anim_mode == "spin":
            ca, sa = math.cos(_t), math.sin(_t)
            nzx = zx * ca - zy * sa
            nzy = zx * sa + zy * ca
            zx, zy = nzx, nzy
        if anim_mode == "zoom":
            span = 1.0 + 0.35 * (0.5 + 0.5 * math.sin(_t))
            zx = zx * span
            zy = zy * span

        # ── KIFS iteration (vectorized over the whole image) ──
        ca = math.cos(fold_angle)
        sa = math.sin(fold_angle)
        trap = np.full(zx.shape, 1e9, dtype=np.float32)
        trap_line = np.full(zx.shape, 1e9, dtype=np.float32)
        count = np.zeros(zx.shape, dtype=np.float32)
        esc = np.zeros(zx.shape, dtype=bool)
        r2_esc = escape_r * escape_r
        inv_sqrt2 = 1.0 / math.sqrt(2.0)
        wedge = math.pi / float(sym)

        for i in range(iters):
            # 1) Kaleidoscopic fold: mirror across axes, then n-fold wedge reflect
            zx = np.abs(zx)
            ang = np.arctan2(zy, zx)
            rad = np.sqrt(zx * zx + zy * zy)
            ang = np.mod(ang, 2.0 * wedge)
            ang = np.where(ang > wedge, 2.0 * wedge - ang, ang)
            zx = rad * np.cos(ang)
            zy = rad * np.sin(ang)
            # 2) Rotation between folds
            nzx = zx * ca - zy * sa
            nzy = zx * sa + zy * ca
            zx, zy = nzx, nzy
            # 3) Scale + offset
            zx = zx * scale + ox
            zy = zy * scale + oy
            # Orbit traps
            r2 = zx * zx + zy * zy
            trap = np.minimum(trap, np.sqrt(r2))
            trap_line = np.minimum(trap_line, np.abs(zy - zx) * inv_sqrt2)
            newly = r2 > r2_esc
            count = np.where(newly & ~esc, float(i + 1), count)
            esc = esc | newly

        r2 = zx * zx + zy * zy
        interior = ~esc

        # ── Color ──
        if colormode == "bands":
            v = np.where(esc, count / float(max(iters, 1)), 0.0)
        elif colormode == "neon":
            v = np.where(esc, trap_line / max(escape_r, 1.0), 0.0)
        else:  # orbit
            v = np.where(esc, trap / max(escape_r, 1.0), 0.0)
        v = np.clip(v + color_shift, 0.0, 1.0)

        # Cosine palette matching the GLSL fractal_palette helper.
        r = 0.5 + 0.5 * np.cos(6.28318 * (1.0 * v + 0.0))
        g = 0.5 + 0.5 * np.cos(6.28318 * (0.75 * v + 2.0 / 6.28318))
        b = 0.5 + 0.5 * np.cos(6.28318 * (0.5 * v + 4.0 / 6.28318))
        rgb = np.stack([r, g, b], axis=-1).astype(np.float32)
        # Interior fill: dark navy so the fractal body reads.
        rgb[interior] = np.array([0.02, 0.02, 0.06], dtype=np.float32)

        rgb = np.clip(rgb, 0.0, 1.0).astype(np.float32)
        capture_frame("402", rgb)
        save(rgb, mn(402, "kaleidoscopic_ifs"), out_dir)

        # ── Rule 4/5: write scalars + the escape-time field ──
        field = np.where(esc, count / float(max(iters, 1)), 0.0).astype(np.float32)
        write_field(out_dir, field)
        interior_frac = float(interior.mean())
        write_scalars(out_dir, symmetry=float(sym), scale=scale,
                      escape_fraction=1.0 - interior_frac,
                      mean_iter=float(count[esc].mean()) if esc.any() else 0.0)

        return rgb
    except Exception as exc:
        fallback = np.full((H, W, 3), 18, dtype=np.uint8)
        save(fallback, mn(402, "Kaleidoscopic IFS"), out_dir)
        print(f"[method_402] ERROR: {exc}")
        return fallback
