from __future__ import annotations
import math
from pathlib import Path

import numpy as np

from ...core.registry import method
from ...core.utils import (
    save, mn, seed_all, W, H, write_scalars, write_field,
)
from ...core.animation import capture_frame


# ── Domain coloring (a.k.a. color-wheel graph) of complex analytic functions ──
#
# A complex function f: C -> C needs four real dimensions to plot, so we bake
# the 2D value (|f|, arg f) into a 2D color image:
#
#     hue      = arg f(z) / 2π          (the phase "portrait")
#     lightness = (2/π)·atan(|f(z)|)    (small |f| dark, large |f| bright)
#
# This is the standard "phase portrait" of Wegert ("Visual Complex Functions",
# 2012) and the Wikipedia "Domain coloring" article. Optional contour lines on
# log|f| (Wegert's enhanced portraits) and on the phase give the classic grid
# look that makes zeros (all lines meet, phase undefined → dark center), poles
# (bright, lines repel) and branch cuts immediately readable.
#
# CPU path authoritative; a clean closed-form f(uv, t) GPU-twin candidate.
# The node is Architecture B: it outputs one frame per animation phase `t`.


def _cos_pal(t: np.ndarray, shift: float):
    """Inigo Quilez cosine gradient palette (periodic, smooth, C0)."""
    t = np.clip(t, 0.0, 1.0)
    r = 0.5 + 0.5 * np.cos(6.2831853 * (t + shift + 0.00))
    g = 0.5 + 0.5 * np.cos(6.2831853 * (t + shift + 0.3333333))
    b = 0.5 + 0.5 * np.cos(6.2831853 * (t + shift + 0.6666667))
    return r, g, b


def _f_of_z(z: np.ndarray, fn: str, n: float) -> np.ndarray:
    """Evaluate the selected complex analytic function on a complex grid.

    All branches are vectorised over the whole pixel grid. Poles/zeros are
    handled downstream (nan_to_num) — we never divide by a literal 0 here.
    """
    with np.errstate(divide="ignore", invalid="ignore"):
        if fn == "z2":
            return z * z
        if fn == "z3":
            return z * z * z
        if fn == "z5":
            return z ** 5
        if fn == "z_n":
            return z ** n
        if fn == "inv":
            return 1.0 / z
        if fn == "exp":
            return np.exp(z)
        if fn == "sin":
            return np.sin(z)
        if fn == "cos":
            return np.cos(z)
        if fn == "tan":
            return np.tan(z)
        if fn == "log":
            return np.log(z)
        if fn == "mobius":
            # (z + i) / (z - i)  — maps the upper half-plane to the disk
            return (z + 1j) / (z - 1j)
        if fn == "poly":
            # The Wikipedia domain-coloring example:
            # (z^2 - 1)(z - 2 - i)^2 / (z^2 + 2 + 2i)
            return ((z * z - 1.0) * (z - (2.0 + 1j)) ** 2) / (z * z + (2.0 + 2j))
        # default: identity (shows the raw plane — useful as a baseline)
        return z


@method(
    id="431",
    name="Domain Coloring",
    category="math_art",
    new_image_contract=True,
    tags=["complex-analysis", "phase-portrait", "visualization", "math-art",
          "generative", "animation", "gpu-twin-candidate"],
    inputs={},
    outputs={"image": "IMAGE"},
    params={
        "function": {"description": "complex function to visualize", "choices": [
            "z2", "z3", "z5", "z_n", "inv", "exp", "sin", "cos", "tan",
            "log", "mobius", "poly", "identity"], "default": "poly"},
        "exponent": {"description": "power n for the z_n function", "min": 2.0, "max": 12.0, "default": 3.0},
        "coloring": {"description": "color scheme", "choices": [
            "phase", "enhanced", "contour", "grid"], "default": "grid"},
        "scale": {"description": "view half-extent in the complex plane (zoom out = larger)", "min": 0.5, "max": 8.0, "default": 3.0},
        "center_x": {"description": "real part of the view center", "min": -4.0, "max": 4.0, "default": 0.0},
        "center_y": {"description": "imaginary part of the view center", "min": -4.0, "max": 4.0, "default": 0.0},
        "time": {"description": "animation phase [0, 2pi)", "min": 0.0, "max": 6.28, "default": 0.0},
        "anim_mode": {"description": "animation mode (none/rotate/drift/pulse/phase_shift)", "choices": [
            "none", "rotate", "drift", "pulse", "phase_shift"], "default": "none"},
        "anim_speed": {"description": "animation speed multiplier", "min": 0.1, "max": 5.0, "default": 1.0},
    },
)
def method_domain_coloring(out_dir: Path, seed: int, params=None):
    """Domain Coloring — visualizing complex functions by phase & magnitude.

    A complex function f: C→C is four-dimensional, so it cannot be drawn as a
    ordinary height graph. Domain coloring encodes the value f(z) at every
    point z of the complex plane into a color:

        hue       = arg f(z) / 2π        (the phase / "argument")
        lightness = (2/π)·atan|f(z)|     (Wegert enhanced portrait:
                                          zeros are dark, poles are bright)

    The result is the classic "phase portrait" (E. Wegert, *Visual Complex
    Functions*, 2012; see also the Wikipedia "Domain coloring" article). Zeros
    show as points where all phase lines converge (and lightness → 0), poles as
    points where they repel (lightness → 1), and the choice of function sweeps
    an enormous gallery: polynomials, rationals, exp/sin/tan/log, Möbius maps
    (which send half-planes to disks), and the Wikipedia example
    (z²−1)(z−2−i)²/(z²+2+2i).

    Four coloring modes:
      • phase    — pure hue wheel, full brightness (flat magnitude)
      • enhanced — hue + Wegert magnitude lightness
      • contour  — enhanced + bright rings at each integer power of 2 in |f|
      • grid     — enhanced + rings on |f| AND phase gridlines (the textbook look)

    Animation modes (Architecture B — one frame per phase t):
      • none        — static portrait at the base params
      • rotate      — rotate the domain z → z·e^{iθ(t)}; the whole picture turns
      • drift       — a zero orbits a small circle z → z − c(t); zeros/poles crawl
      • pulse       — breathe the zoom (scale oscillates, verified off the
                      sin-peak degeneracy by sampling t=0 vs t=π/2)
      • phase_shift — rotate the codomain hue (colors cycle in place)

    CPU path is the authoritative export; the closed-form f(uv,t) makes a clean
    client-GPU twin. Pure source node (inputs={}) — no upstream wire.
    """
    try:
        if params is None:
            params = {}
        t = float(params.get("time", 0.0))
        anim_mode = str(params.get("anim_mode", "none"))
        anim_speed = float(params.get("anim_speed", 1.0))
        seed_all(seed)

        fn = str(params.get("function", "poly"))
        if fn not in ("z2", "z3", "z5", "z_n", "inv", "exp", "sin", "cos",
                      "tan", "log", "mobius", "poly", "identity"):
            fn = "poly"
        exponent = max(2.0, min(12.0, float(params.get("exponent", 3.0))))
        coloring = str(params.get("coloring", "grid"))
        if coloring not in ("phase", "enhanced", "contour", "grid"):
            coloring = "grid"
        scale = max(0.5, min(8.0, float(params.get("scale", 3.0))))
        cx = max(-4.0, min(4.0, float(params.get("center_x", 0.0))))
        cy = max(-4.0, min(4.0, float(params.get("center_y", 0.0))))

        # ── Animation clock (rename t to avoid shadowing the time param) ──
        _t = t * anim_speed

        # pulse (breathe zoom) uses a cosine sweep: t=0 and t=π land on the
        # SAME size (cosine peak), so the audit samples t=0 vs t=π/2 to avoid
        # the sin/cos phase degeneracy; other modes are linear in _t.
        scale_eff = scale
        if anim_mode == "pulse":
            scale_eff = scale * (1.0 + 0.45 * (0.5 - 0.5 * math.cos(_t)))

        # ── Build the complex grid (square pixels) ──
        w = int(W)
        h = int(H)
        step = 2.0 * scale_eff / max(1, w - 1)
        xs = cx + (np.arange(w) - (w - 1) / 2.0) * step
        ys = cy + (np.arange(h) - (h - 1) / 2.0) * step
        z = xs[None, :] + 1j * ys[:, None]

        # domain-space animation transforms
        if anim_mode == "rotate":
            z = z * np.exp(1j * _t)
        elif anim_mode == "drift":
            c = 0.35 * np.exp(1j * _t)
            z = z - c

        # ── Evaluate the complex function ──
        f = _f_of_z(z, fn, exponent)
        f = np.nan_to_num(f, nan=0.0, posinf=1e6, neginf=-1e6)

        mag = np.abs(f)
        phase = np.angle(f)                       # [-π, π]
        hue = (phase + math.pi) / (2.0 * math.pi)  # [0, 1]

        # codomain phase animation (cycle colors in place)
        if anim_mode == "phase_shift":
            hue = (hue + _t * 0.5) % 1.0

        # ── Lightness ──
        if coloring == "phase":
            light = np.ones_like(mag, dtype=np.float32)
        else:
            # Wegert enhanced magnitude: dark at zeros, bright at poles
            light = (2.0 / math.pi) * np.arctan(mag)
            light = light.astype(np.float32)

        # ── Contour / grid lines ──
        if coloring in ("contour", "grid"):
            lg = np.log2(mag + 1e-12)
            ring = 0.5 + 0.5 * np.cos(6.2831853 * lg)   # bright at |f|=2^k
            light = light * (0.78 + 0.22 * ring)
        if coloring == "grid":
            # phase gridlines at 12 divisions of the circle
            pg = 0.5 + 0.5 * np.cos(12.0 * phase)
            light = light * (0.85 + 0.15 * pg)
        light = np.clip(light, 0.0, 1.0).astype(np.float32)

        # ── Hue → RGB via IQ cosine palette, modulated by lightness ──
        rr, gg, bb = _cos_pal(hue, 0.0)
        rgb = np.stack([rr * light, gg * light, bb * light], axis=-1)
        rgb = np.clip(rgb, 0.0, 1.0).astype(np.float32)

        capture_frame("431", rgb)
        save(rgb, mn(431, "Domain Coloring"), out_dir)
        try:
            logmag = np.log10(mag + 1e-12)
            write_field(out_dir, logmag.astype(np.float32))
            write_scalars(out_dir,
                          mean_light=float(light.mean()),
                          mean_hue=float(hue.mean()),
                          p95_light=float(np.percentile(light, 95)),
                          zero_dark_fraction=float((light < 0.05).mean()))
        except Exception:
            pass
        return rgb
    except Exception as exc:
        fallback = np.full((H, W, 3), 0.5, dtype=np.float32)
        save(fallback, mn(431, "Domain Coloring"), out_dir)
        print(f"[method_431] ERROR: {exc}")
        return fallback
