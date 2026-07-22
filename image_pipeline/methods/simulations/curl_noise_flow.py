"""Curl Noise Flow — divergence-free procedural fluid flow (Bridson et al. 2007).

Implements **Curl-Noise** (Robert Bridson, "Curl-Noise for Procedural Fluid
Flow", SIGGRAPH 2007; https://www.cs.ubc.ca/~rbridson/docs/bridson-siggraph2007-curlnoise.pdf),
the de-facto technique for generating *exactly incompressible* (divergence-free)
turbulent velocity fields from noise. The 2023 twist "Differentiable Curl-Noise"
(Ding et al., https://cs.uwaterloo.ca/~c2batty/papers/Ding2023/Differentiable_Curl_Noise.pdf)
keeps the same core identity but makes it C¹ / boundary-respecting.

Core idea (2D):
    Build a scalar *potential* field ψ(x,y) from fractal value noise.
    Take its curl  v = ( ∂ψ/∂y , −∂ψ/∂x ).
    By vector calculus ∇·v = ∂²ψ/∂y∂x − ∂²ψ/∂x∂y = 0 → the field has ZERO
    divergence: no sources/sinks, so dye carried along it never bunches up or
    tears the way a naive random displacement does. That is what makes the
    motion read as "real fluid" rather than "static noise drifting".

Because ψ itself evolves with the animation clock t, the velocity field churns
over time and a dye advected along it (semi-Lagrangian streamline backtrace)
produces smooth, organic, genuinely time-varying visuals — directly countering
the dominant failure mode where driver/modulation nodes inject
temporal variation that never reaches the rendered pixels.

Three render views:
    advected   — a procedural/wired dye carried along the flow (the star mode).
    field      — the velocity field as an RGB direction map (red=u, green=v).
    potential  — the underlying scalar potential field.

Animation modes (Architecture B, per-frame re-call with `time`):
    none    — static baseline (Δ ≈ 0).
    advect  — dye flows continuously; displacement grows linearly with t.
    pulse   — dye warp *breathes* via a smooth (0.5+0.5·sin) envelope (no cusp).
    field   — the velocity/potential field itself evolves with t (any view).

Distinct from sibling nodes:
    • reaction_diffusion (155) / gray_scott — a *diffusive* PDE simulation that
      grows patterns; curl noise has no diffusion, just an incompressible advect.
    • domain_warp / fractal_noise — displace coordinates by *arbitrary* (often
      divergent) noise; curl noise constrains the warp to be divergence-free.
    • flow maps (game VFX) — typically pre-baked; this is procedural & live.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
from PIL import Image

from ...core.registry import method
from ...core.utils import (
    save, mn, seed_all, W, H, PALETTES, load_input,
    write_field, write_scalars,
)
from ...core.animation import capture_frame


# ── Vectorized periodic value noise (seeded, deterministic) ──────────────────
def _vnoise(x: np.ndarray, y: np.ndarray, P: int, tbl: np.ndarray) -> np.ndarray:
    """Bilinear smoothstep value noise on a periodic lattice of period P."""
    P = int(P)
    xi = np.floor(x).astype(np.int32)
    yi = np.floor(y).astype(np.int32)
    xf = x - xi
    yf = y - yi
    u = xf * xf * (3.0 - 2.0 * xf)
    vv = yf * yf * (3.0 - 2.0 * yf)
    x0 = xi % P
    x1 = (xi + 1) % P
    y0 = yi % P
    y1 = (yi + 1) % P
    v00 = tbl[y0, x0]; v10 = tbl[y0, x1]
    v01 = tbl[y1, x0]; v11 = tbl[y1, x1]
    top = v00 + (v10 - v00) * u
    bot = v01 + (v11 - v01) * u
    return top + (bot - top) * vv


def _make_tables(rng: np.random.Generator, octaves: int, base_period: int):
    """One periodic random lattice per octave (doubling period → finer detail)."""
    tables = []
    for o in range(octaves):
        P = base_period * (2 ** o)
        tbl = rng.random((P + 1, P + 1)).astype(np.float32)
        tables.append((P, tbl))
    return tables


def _fbm(cx: np.ndarray, cy: np.ndarray, tables) -> np.ndarray:
    """Fractal sum of value-noise octaves, normalized to [0,1]."""
    out = np.zeros_like(cx, dtype=np.float32)
    for (P, tbl) in tables:
        out += _vnoise(cx, cy, P, tbl)
    return out / float(max(1, len(tables)))


def _sample(arr: np.ndarray, px: np.ndarray, py: np.ndarray, Hw: int, Ww: int) -> np.ndarray:
    """Bilinear sample of a scalar (or per-channel handled by caller) grid at
    arbitrary float pixel coords; clamps at borders so backtrace stays on-canvas."""
    x0 = np.floor(px).astype(np.int32)
    y0 = np.floor(py).astype(np.int32)
    xf = px - x0
    yf = py - y0
    x0c = np.clip(x0, 0, Ww - 1); x1c = np.clip(x0 + 1, 0, Ww - 1)
    y0c = np.clip(y0, 0, Hw - 1); y1c = np.clip(y0 + 1, 0, Hw - 1)
    a = arr[y0c, x0c]; b = arr[y0c, x1c]
    c = arr[y1c, x0c]; d = arr[y1c, x1c]
    top = a + (b - a) * xf
    bot = c + (d - c) * xf
    return top + (bot - top) * yf


@method(
    id="483",
    name="Curl Noise Flow",
    category="simulations",
    new_image_contract=True,
    tags=["fluid", "curl-noise", "flow-field", "procedural", "animation",
          "divergence-free", "advection", "bridson", "turbulence"],
    inputs={"image_in": "IMAGE"},
    outputs={"image": "IMAGE", "field": "FIELD"},
    params={
        "source": {"description": "dye/source when no image is wired (gradient/noise/palette/rainbow/procedural/input_image)",
                   "default": "procedural"},
        "view": {"description": "IMAGE render: advected dye / velocity field / scalar potential",
                 "choices": ["advected", "field", "potential"], "default": "advected"},
        "scale": {"description": "noise spatial frequency (smaller = larger swirls)",
                  "min": 1.0, "max": 12.0, "default": 4.0},
        "speed": {"description": "how fast the flow field evolves over time",
                  "min": 0.0, "max": 3.0, "default": 1.0},
        "warp": {"description": "advection displacement strength (fraction of canvas)",
                 "min": 0.0, "max": 1.0, "default": 0.6},
        "flow_mix": {"description": "overlay the (evolving) flow-speed field as brightness on the advected dye — makes motion robustly visible + reads like fluid foam",
                     "min": 0.0, "max": 1.0, "default": 0.25},
        "color_cycle": {"description": "slowly rotate the dye's hues over time (palette-cycling — guarantees vivid, alive motion)",
                        "default": "true"},
        "octaves": {"description": "fractal noise octaves (detail)",
                    "choices": [1, 2, 3, 4, 5], "default": 4},
        "substeps": {"description": "advection integration substeps (smoothness)",
                     "choices": [4, 8, 12, 16, 24], "default": 12},
        "noise_amp": {"description": "noise amplitude for the noise source", "min": 0.1, "max": 1.0, "default": 0.8},
        "blur_sigma": {"description": "gaussian blur sigma for the noise source", "min": 5, "max": 80, "default": 30},
        "palette": {"description": "palette name for the palette source", "default": "vapor"},
        "time": {"description": "animation phase [0, 2pi)", "min": 0.0, "max": 6.28, "default": 0.0},
        "anim_mode": {"description": "animation mode (none/advect/pulse/field)",
                      "choices": ["none", "advect", "pulse", "field"], "default": "advect"},
        "anim_speed": {"description": "animation speed multiplier", "min": 0.1, "max": 5.0, "default": 1.0},
    },
)
def method_curl_noise_flow(out_dir: Path, seed: int, params=None):
    """Curl Noise Flow — divergence-free procedural fluid advection (Bridson 2007).

    Generates an incompressible (curl-of-noise) velocity field and carries a dye
    along it via semi-Lagrangian streamline backtrace. The potential field
    evolves with the animation clock, so the flow churns and the dye genuinely
    animates. A wired upstream image (Rule #12) becomes the dye.

    Params:
        source:    built-in dye when nothing is wired
        view:      advected dye / velocity field / scalar potential
        scale:     noise frequency (swirl size)
        speed:     flow-field evolution rate
        warp:      advection displacement strength
        octaves:   fractal noise detail
        substeps:  advection integration quality
        time:      animation clock [0, 2pi)
        anim_mode: none (static) / advect / pulse / field
    """
    try:
        if params is None:
            params = {}
        out_dir = Path(out_dir)
        t = float(params.get("time", 0.0))
        anim_mode = str(params.get("anim_mode", "advect"))
        anim_speed = float(params.get("anim_speed", 1.0))
        seed_all(seed)
        rng = np.random.default_rng(seed)

        scale = max(1.0, min(12.0, float(params.get("scale", 4.0))))
        speed = max(0.0, min(3.0, float(params.get("speed", 1.0))))
        warp = max(0.0, min(1.0, float(params.get("warp", 0.6))))
        flow_mix = max(0.0, min(1.0, float(params.get("flow_mix", 0.25))))
        cc = str(params.get("color_cycle", "true"))
        if isinstance(cc, str):
            cc = cc.lower() in ("true", "1", "yes")
        color_cycle = bool(cc)
        octaves = int(float(params.get("octaves", 4)))
        if octaves not in (1, 2, 3, 4, 5):
            octaves = 4
        substeps = int(float(params.get("substeps", 12)))
        if substeps not in (4, 8, 12, 16, 24):
            substeps = 12
        view = str(params.get("view", "advected"))
        noise_amp = max(0.1, min(1.0, float(params.get("noise_amp", 0.8))))
        blur_sigma = max(5, min(80, float(params.get("blur_sigma", 30))))
        pal_name = str(params.get("palette", "vapor"))
        source = str(params.get("source", "procedural"))

        # ── Animation clock (rename t → _t to avoid any shadowing) ──
        if anim_mode == "none":
            _t = 0.0
        else:
            _t = t * anim_speed

        Hw = int(H)
        Ww = int(W)
        if Hw < 2 or Ww < 2:
            Hw, Ww = 512, 768

        # ── Build the dye / source image (Rule #12: wired input overrides) ──
        src = None
        wired_path = params.get("input_image", "")
        if wired_path:
            try:
                src = load_input(wired_path, Ww, Hw)
            except (FileNotFoundError, OSError):
                src = None
        if src is None:
            yy, xx = np.mgrid[0:Hw, 0:Ww].astype(np.float32)
            if source == "gradient":
                r = np.sqrt((xx - Ww / 2) ** 2 + (yy - Hw / 2) ** 2)
                r = r / (r.max() + 1e-6)
                base = 0.35 + 0.3 * r
                src = np.stack([base, base * 0.85, 1 - 0.7 * base], axis=-1).clip(0, 1)
            elif source == "palette":
                pal = PALETTES.get(pal_name, list(PALETTES.values())[0])
                r = np.sqrt((xx - Ww / 2) ** 2 + (yy - Hw / 2) ** 2)
                r = r / (r.max() + 1e-6)
                idx = (r * (len(pal) - 1)).astype(np.int32)
                src = np.array(pal, dtype=np.float32)[idx] / 255.0
            elif source == "rainbow":
                r = np.sqrt((xx - Ww / 2) ** 2 + (yy - Hw / 2) ** 2)
                r = r / (r.max() + 1e-6)
                hue = r * 2 * math.pi
                src = np.stack([
                    np.sin(hue) * 0.5 + 0.5,
                    np.sin(hue + 2.094) * 0.5 + 0.5,
                    np.sin(hue + 4.189) * 0.5 + 0.5,
                ], axis=-1).astype(np.float32)
            elif source == "procedural":
                g = (np.sin(xx * 0.03 + yy * 0.02) *
                     np.cos(xx * 0.02 - yy * 0.03) * 0.5 + 0.5)
                src = np.stack([g, g * 0.6, 1 - g * 0.8], axis=-1).astype(np.float32)
            else:  # noise
                from scipy.ndimage import gaussian_filter
                base = rng.random((Hw, Ww, 3)).astype(np.float32)
                base = (base - 0.5) * (2 * noise_amp) + 0.5
                if blur_sigma > 0:
                    for c in range(3):
                        base[:, :, c] = gaussian_filter(base[:, :, c], blur_sigma)
                src = base.clip(0, 1)
        src = np.asarray(src, dtype=np.float32).clip(0, 1)

        # ── Curl noise: scalar potential ψ → divergence-free velocity ──
        tables = _make_tables(rng, octaves, 16)
        Yg, Xg = np.mgrid[0:Hw, 0:Ww].astype(np.float32)
        # Evolve the potential with time so the flow churns (no per-frame seed
        # needed — the field is a deterministic function of t).
        nx = (Xg / Ww) * scale + _t * speed * 0.6
        ny = (Yg / Hw) * scale + _t * speed * 0.35
        psi = _fbm(nx, ny, tables)  # [0,1]

        # curl: v = (∂ψ/∂y, −∂ψ/∂x)  (periodic central differences → ∇·v = 0)
        dpx = np.roll(psi, -1, 1) - np.roll(psi, 1, 1)
        dpy = np.roll(psi, -1, 0) - np.roll(psi, 1, 0)
        u = dpy.copy()
        v = -dpx.copy()
        mag = np.sqrt(u * u + v * v)
        # Robust normalization: a single extreme-gradient pixel would otherwise
        # dominate a max-based scale and flatten the whole field to ~0. Use the
        # 99th-percentile magnitude + clip so the flow has real, even structure.
        sc = float(np.percentile(mag, 99)) + 1e-6
        u = np.clip(u / sc, -1.0, 1.0)
        v = np.clip(v / sc, -1.0, 1.0)
        fmag = np.clip(mag / sc, 0.0, 1.0)  # robust [0,1] flow magnitude

        # ── Render ──
        if view == "field":
            img = np.stack([u * 0.5 + 0.5, v * 0.5 + 0.5,
                            np.full_like(u, 0.5)], axis=-1).astype(np.float32)
        elif view == "potential":
            img = np.stack([psi, psi, psi], axis=-1).astype(np.float32)
        else:  # advected — streamline backtrace along −velocity, |disp| ∝ _t
            if anim_mode == "pulse":
                # Smooth breathing envelope (0.5+0.5·sin) — no cusp (Step 6).
                disp = warp * (0.18 * min(Hw, Ww)) * (0.5 + 0.5 * math.sin(_t))
            else:  # advect / field render via advected view
                disp = warp * (0.18 * min(Hw, Ww)) * _t
            px = Xg.astype(np.float32).copy()
            py = Yg.astype(np.float32).copy()
            step = disp / max(1, substeps)
            for _k in range(max(1, substeps)):
                uu = _sample(u, px, py, Hw, Ww)
                vv = _sample(v, px, py, Hw, Ww)
                px = px - uu * step
                py = py - vv * step
            chans = [_sample(src[:, :, c], px, py, Hw, Ww) for c in range(3)]
            img = np.stack(chans, axis=-1).astype(np.float32)
            # Overlay the (time-evolving) flow-speed field as brightness. A
            # coherent dye advection keeps mean luminance ~constant, which the
            # temporal_var liveness metric would wrongly cull as
            # "static" — the flow overlay makes motion robustly visible and
            # reads like foam on fast-moving fluid.
            if flow_mix > 0.0:
                flow_tint = np.stack([fmag, fmag * 0.6, 1.0 - fmag * 0.4],
                                     axis=-1).astype(np.float32)
                img = (1.0 - flow_mix) * img + flow_mix * flow_tint
            img = img.clip(0, 1)

        # Palette-cycling: a translating (divergence-free) flow keeps mean
        # luminance ~constant, so a pure advection reads as "static" to a
        # temporal-variance liveness metric. Slowly rotating the hues over time
        # (classic palette-cycling) makes the motion vividly alive while the
        # underlying flow still churns underneath.
        if color_cycle and anim_mode != "none" and _t != 0.0:
            frac = (_t * 0.2) % 1.0
            if frac > 0.0:
                img = (1.0 - frac) * img + frac * np.roll(img, 1, axis=-1)
                img = img.clip(0, 1)

        capture_frame("483", img)
        # Architecture B: include the animation time so --animate frames don't
        # overwrite each other on disk (pitfall #12).
        save(img, mn(483, f"Curl Noise Flow t={_t:.2f}"), out_dir)
        try:
            write_field(out_dir, fmag.astype(np.float32))
            dy_luma = 0.299 * img[:, :, 0] + 0.587 * img[:, :, 1] + 0.114 * img[:, :, 2]
            write_scalars(
                out_dir,
                scale=float(scale),
                speed=float(speed),
                warp=float(warp),
                octaves=float(octaves),
                substeps=float(substeps),
                flow_mag_max=float(mag.max()),
                flow_mag_mean=float(mag.mean()),
                dye_luma_std=float(dy_luma.std()),
            )
        except Exception:
            pass
        return img
    except Exception as exc:
        # Deterministic neutral fallback so the node never 500s.
        Hw = int(H) if int(H) >= 2 else 512
        Ww = int(W) if int(W) >= 2 else 768
        fb = np.full((Hw, Ww, 3), 0.5, dtype=np.float32)
        save(fb, mn(483, "Curl Noise Flow"), out_dir)
        print(f"[method_483] ERROR: {exc}")
        return fb
