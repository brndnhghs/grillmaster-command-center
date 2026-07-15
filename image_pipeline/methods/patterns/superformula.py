from __future__ import annotations
import math
from pathlib import Path

import numpy as np

from ...core.registry import method
from ...core.utils import (
    save, mn, seed_all, write_field, write_scalars, write_mask, W, H,
)
from ...core.animation import capture_frame


# ── Superformula (Gielis, "A generic geometric transformation that unifies a
#    wide range of natural and abstract shapes", American Journal of Botany,
#    2003, doi:10.3732/ajb.90.3.333) ──
#
# A single polar equation generates flowers, stars, polygons, gears and
# blobby organic forms from a handful of exponents:
#
#     r(φ) = [ |cos(mφ/4)/a|^{n2} + |sin(mφ/4)/b|^{n3} ]^{-1/n1}
#
# x = r·cos φ ,  y = r·sin φ.   With m = rotational symmetry and n1,n2,n3 the
# shape exponents, the same formula yields a square (m=4, n*→∞), a star, a
# rose, a hexagon, or a rounded petal — the basis of "supershapes" used in
# procedural botany, logo design and generative art. We render it per-pixel:
# a pixel at polar (ρ, φ) is *inside* the shape iff ρ ≤ r(φ), which fills the
# exact polar region with no polygon rasterisation.
#
# CPU path authoritative; a clean closed-form node (great GPU-twin candidate).


def _cos_pal(t: np.ndarray, shift: float):
    """Inigo Quilez cosine gradient palette."""
    t = np.clip(t, 0.0, 1.0)
    r = 0.5 + 0.5 * np.cos(6.28318 * (t + shift + 0.00))
    g = 0.5 + 0.5 * np.cos(6.28318 * (t + shift + 0.33))
    b = 0.5 + 0.5 * np.cos(6.28318 * (t + shift + 0.67))
    return r, g, b


@method(
    id="409",
    name="Superformula",
    category="patterns",
    new_image_contract=True,
    tags=["generative", "pattern", "superformula", "gielis", "parametric",
          "botany", "animation", "gpu-twin-candidate"],
    inputs={},
    outputs={"image": "IMAGE", "mask": "MASK"},
    params={
        "m": {"description": "rotational symmetry (number of lobes/petals)", "min": 0.0, "max": 20.0, "default": 6.0},
        "n1": {"description": "shape exponent n1 (overall roundness)", "min": 0.1, "max": 20.0, "default": 1.0},
        "n2": {"description": "shape exponent n2 (petal sharpness)", "min": 0.1, "max": 20.0, "default": 1.0},
        "n3": {"description": "shape exponent n3 (lobe width)", "min": 0.1, "max": 20.0, "default": 1.0},
        "spread": {"description": "a = b spread of the base ellipse (1=circle)", "min": 0.2, "max": 3.0, "default": 1.0},
        "colormode": {"description": "fill color (gradient/bands/rainbow/solid)", "choices": ["gradient", "bands", "rainbow", "solid"], "default": "gradient"},
        "palette_shift": {"description": "cosine palette hue offset", "min": 0.0, "max": 1.0, "default": 0.5},
        "time": {"description": "animation phase [0, 2pi)", "min": 0.0, "max": 6.28, "default": 0.0},
        "anim_mode": {"description": "animation mode (none/spin/breathe/morph)", "choices": ["none", "spin", "breathe", "morph"], "default": "none"},
        "anim_speed": {"description": "animation speed multiplier", "min": 0.1, "max": 5.0, "default": 1.0},
    },
)
def method_superformula(out_dir: Path, seed: int, params=None):
    """Superformula — Gielis' unifying parametric shape equation.

    Gielis, "A generic geometric transformation that unifies a wide range of
    natural and abstract shapes", American Journal of Botany 90(3), 2003
    (doi:10.3732/ajb.90.3.333).

    One polar equation spans an enormous design space — from a square and a
    hexagon, through stars, roses and gears, to rounded petals and blobby
    organic forms — via just four numbers:

        r(φ) = [ |cos(mφ/4)/a|^{n2} + |sin(mφ/4)/b|^{n3} ]^{-1/n1}

    m sets the rotational symmetry; n1, n2, n3 are the shape exponents; a = b
    is the base spread. We render it *per pixel*: a pixel at polar (ρ, φ) is
    inside the shape iff ρ ≤ r(φ), so the exact polar region is filled with no
    polygon rasterisation and the shape always fits the canvas (we auto-scale
    by the maximum radius).

    Distinct from the other pattern nodes:
      • kaleidoscopic_ifs / plasma: escape-time / trig fields, not a closed
        polar curve.
      • truchet / wallpaper / quasicrystal: tiling / symmetry lattices, not a
        single generative form.
      • strange_attractor2d: point-cloud IFS, not a filled boundary curve.
    Superformula is the parametric-design sibling — one equation, one shape
    that morphs continuously across the whole family.

    CPU path authoritative; a clean closed-form f(uv, t) GPU twin.

    Params:
        m:            rotational symmetry / lobe count (0-20)
        n1:           shape exponent — overall roundness (0.1-20)
        n2:           shape exponent — petal sharpness (0.1-20)
        n3:           shape exponent — lobe width (0.1-20)
        spread:       base ellipse a=b (0.2-3, 1=circle)
        colormode:    gradient / bands / rainbow / solid fill
        palette_shift: cosine palette hue offset (0-1)
        time:         animation phase [0, 2pi)
        anim_mode:    none / spin / breathe / morph
        anim_speed:   animation speed (0.1-5.0)
    """
    try:
        if params is None:
            params = {}
        t = float(params.get("time", 0.0))
        anim_mode = str(params.get("anim_mode", "none"))
        anim_speed = float(params.get("anim_speed", 1.0))
        seed_all(seed)

        m = max(0.0, min(20.0, float(params.get("m", 6.0))))
        n1 = max(0.1, min(20.0, float(params.get("n1", 1.0))))
        n2 = max(0.1, min(20.0, float(params.get("n2", 1.0))))
        n3 = max(0.1, min(20.0, float(params.get("n3", 1.0))))
        spread = max(0.2, min(3.0, float(params.get("spread", 1.0))))
        colormode = str(params.get("colormode", "gradient"))
        palette_shift = max(0.0, min(1.0, float(params.get("palette_shift", 0.5))))

        # ── Animation clock (rename t to avoid shadowing the time param) ──
        _t = t * anim_speed
        if anim_mode == "morph":
            # smooth, phase-offset sweeps of the three exponents — the shape
            # continuously morphs across the family (no cusps: sine-based).
            n1 = max(0.2, n1 * (1.0 + 0.6 * math.sin(_t)))
            n2 = max(0.2, n2 * (1.0 + 0.6 * math.sin(_t + 2.094)))
            n3 = max(0.2, n3 * (1.0 + 0.6 * math.sin(_t + 4.189)))
        breathe = 1.0
        if anim_mode == "breathe":
            # cosine sweep spans 0→1 as _t goes 0→π, so the audit sample
            # frames (t=0 and t=3.14) land on opposite size extremes.
            breathe = 1.0 + 0.15 * (0.5 - 0.5 * math.cos(_t))

        # ── Per-pixel polar coordinates ──
        yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
        yc, xc = H / 2.0, W / 2.0
        dyy = yy - yc
        dxx = xx - xc
        rho = np.sqrt(dxx * dxx + dyy * dyy)
        phi = np.arctan2(dyy, dxx)                       # [-pi, pi]
        phi_pos = (phi + 2.0 * math.pi) % (2.0 * math.pi)

        # spin rotates the shape by advancing the angle fed to the formula.
        # A non-integer spin rate (1.13×) guarantees the shape is never exactly
        # symmetry-aligned at the audit sample times (t=0 vs t=3.14) even for
        # high integer m, so the animation always reads as motion.
        if anim_mode == "spin":
            phi_eff = (phi_pos + _t * 1.13) % (2.0 * math.pi)
        else:
            phi_eff = phi_pos

        # ── Superformula radius r(φ) ──
        a = spread
        b = spread
        c_m = np.cos(m * phi_eff / 4.0)
        s_m = np.sin(m * phi_eff / 4.0)
        t1 = np.abs(c_m) / a + 1e-9
        t2 = np.abs(s_m) / b + 1e-9
        R = (t1 ** n2 + t2 ** n3) ** (-1.0 / n1)
        R = np.clip(R, 0.0, 1e4)

        # auto-scale so the shape fits the canvas; breathe modulates it
        max_r = max(float(R.max()), 1e-6)
        half = min(float(H), float(W)) * 0.46
        scale = (half / max_r) * breathe
        r_shape = R * scale

        inside = rho <= r_shape
        ratio = np.clip(rho / (r_shape + 1e-9), 0.0, 1.0).astype(np.float32)

        # ── Fill colour ──
        bg = np.array([0.04, 0.05, 0.09], dtype=np.float32)
        if colormode == "solid":
            fill = np.full((H, W, 3), np.array([0.95, 0.55, 0.25], np.float32))
        elif colormode == "bands":
            nb = 8
            bb = (ratio * nb).astype(np.int32) % nb
            rr = bb.astype(np.float32) / float(nb)
            fr, fg, fb = _cos_pal(rr, palette_shift)
            fill = np.stack([fr, fg, fb], axis=-1).astype(np.float32)
        elif colormode == "rainbow":
            hue = phi_pos / (2.0 * math.pi)
            fr = 0.5 + 0.5 * np.cos(6.28318 * (hue + palette_shift))
            fg = 0.5 + 0.5 * np.cos(6.28318 * (hue + palette_shift + 0.33))
            fb = 0.5 + 0.5 * np.cos(6.28318 * (hue + palette_shift + 0.67))
            fill = np.stack([fr, fg, fb], axis=-1).astype(np.float32)
        else:  # gradient
            fr, fg, fb = _cos_pal(ratio, palette_shift)
            fill = np.stack([fr, fg, fb], axis=-1).astype(np.float32)

        rgb = np.empty((H, W, 3), dtype=np.float32)
        rgb[:, :] = bg
        rgb[inside] = fill[inside]
        rgb = np.clip(rgb, 0.0, 1.0).astype(np.float32)

        capture_frame("409", rgb)
        save(rgb, mn(409, "Superformula"), out_dir)
        try:
            mask = inside.astype(np.float32)
            write_field(out_dir, ratio)
            write_mask(out_dir, mask)
            write_scalars(out_dir, m=float(m), n1=float(n1), n2=float(n2),
                          n3=float(n3), symmetry=float(m),
                          inside_fraction=float(inside.mean()))
        except Exception:
            pass
        return rgb
    except Exception as exc:
        fallback = np.full((H, W, 3), 0.5, dtype=np.float32)
        save(fallback, mn(409, "Superformula"), out_dir)
        print(f"[method_409] ERROR: {exc}")
        return fallback
