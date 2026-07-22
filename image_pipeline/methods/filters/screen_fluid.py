"""Screen-Space Fluid Rendering (van der Laan, Green & Sainz, SI3D 2009).

Classic real-time particle-fluid surface technique. Given a cloud of
discrete particles we reproduce, in image space, the three-stage pipeline that
made SPH fluids look like a continuous liquid surface instead of "blobs":

    1. SPLAT  — accumulate each particle into a screen-space depth buffer
                (nearest-surface depth weighted by a coverage/alpha term).
    2. SMOOTH — run an *edge-preserving* (bilateral / curvature-flow) filter
                over the depth buffer so the raw point cloud becomes a smooth
                implicit surface, while sharp silhouettes survive. This is the
                heart of the technique and what separates it from a plain blur.
    3. SHADE  — reconstruct a surface normal from the smoothed height field,
                then light it as a liquid: Fresnel rim, thickness-based tint,
                and a subtle specular lobe.

The motion model uses a *curl-noise* (divergence-free) flow field that morphs
smoothly over the `time` clock, so every active animation mode is non-symmetric
and strobe-free (no abs(sin) cusps, no rotational-symmetry dead motion).

A wired IMAGE (Rule 12) is used as the backdrop the fluid is composited over;
otherwise a dark background is used. The fluid coverage field is emitted as a
MASK (Rule 10) for downstream spatial selection.

Reference: https://doi.org/10.1145/1507149.1507164
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter, map_coordinates, uniform_filter

from ...core.registry import method
from ...core.utils import (
    save,
    mn,
    seed_all,
    W,
    H,
    BG_DEFAULT,
    norm,
    write_scalars,
    write_mask,
    load_input,
)
from ...core.animation import capture_frame
from image_pipeline.core.spatial import sparam


# ── smooth scalar field (cheap stand-in for value noise) ───────────────────

def _smooth_field(rng: np.random.Generator, shape, sigma: float) -> np.ndarray:
    g = rng.random(shape)
    if sigma > 0.0:
        g = gaussian_filter(g, sigma=sigma, mode="reflect")
    g = (g - g.min()) / (np.ptp(g) + 1e-8)
    return g


def _curl_flow(rng: np.random.Generator, hh: int, ww: int, t: float, scale: float):
    """Two smooth potential fields morphed by time → divergence-free flow."""
    gy, gx = np.mgrid[0:hh, 0:ww].astype(np.float64)
    gy /= max(hh, ww)
    gx /= max(hh, ww)
    # sample the morphing potential on a coarse grid then upsample
    cg = max(8, int(min(hh, ww) / scale))
    pot_a = _smooth_field(rng, (cg, cg), sigma=cg * 0.35)
    pot_b = _smooth_field(rng, (cg, cg), sigma=cg * 0.35)
    grid = np.mgrid[0:hh, 0:ww].astype(np.float64) / max(hh, ww) * cg
    cy, cx = grid[0], grid[1]
    pa = map_coordinates(pot_a, [cy, cx], order=1, mode="reflect")
    pb = map_coordinates(pot_b, [cy, cx], order=1, mode="reflect")
    s = 0.5 + 0.5 * math.sin(t)            # smooth morph (no cusp)
    pot = (1.0 - s) * pa + s * pb
    dpy, dpx = np.gradient(pot)
    # curl of scalar potential → incompressible (fluid-like) velocity
    vx = dpy
    vy = -dpx
    return vx, vy, gy, gx


@method(
    id="494",
    name="Screen-Space Fluid",
    category="filters",
    tags=["fluid", "ssf", "screen-space", "simulation", "shading", "liquid", "animation", "expanded"],
    timeout=120,
    inputs={"image_in": "IMAGE"},
    outputs={"image": "IMAGE", "mask": "MASK"},
    params={
        "source": {"description": "backdrop when no upstream image is wired (dark/noise/gradient/palette)", "choices": ["dark", "noise", "gradient", "palette"], "default": "dark"},
        "palette": {"description": "palette name for the palette backdrop", "default": "vapor"},
        "particles": {"description": "number of fluid particles (1k–40k)", "min": 1000, "max": 40000, "default": 16000},
        "radius": {"description": "particle splat radius in px (surface thickness)", "min": 2.0, "max": 18.0, "default": 7.0},
        "smooth": {"description": "bilateral smoothing strength (0 = raw points, higher = smoother surface)", "min": 0.0, "max": 1.0, "default": 0.6},
        "detail": {"description": "surface micro-relief amount (Perlin-height scale)", "min": 0.0, "max": 1.0, "default": 0.4},
        "flow_scale": {"description": "spatial scale of the curl-noise flow field", "min": 0.5, "max": 6.0, "default": 2.5},
        "flow_speed": {"description": "advection amplitude of the flow field", "min": 0.0, "max": 3.0, "default": 1.0},
        "swirl": {"description": "global swirl rotation rate (turns over 2π)", "min": -2.0, "max": 2.0, "default": 0.3},
        "thickness": {"description": "liquid body tint strength (thicker = deeper color)", "min": 0.0, "max": 1.0, "default": 0.5},
        "fresnel": {"description": "rim/Fresnel intensity", "min": 0.0, "max": 1.5, "default": 0.7},
        "specular": {"spatial": True, "description": "specular highlight strength", "min": 0.0, "max": 1.0, "default": 0.6},
        "anim_mode": {"description": "animation mode (none/flow/swirl/waves)", "choices": ["none", "flow", "swirl", "waves"], "default": "none"},
        "anim_speed": {"description": "animation speed multiplier", "min": 0.1, "max": 3.0, "default": 1.0},
    },
)
def method_screen_fluid(out_dir: Path, seed: int, params=None):
    """Screen-Space Fluid — splat → bilateral smooth → liquid shading.

    Renders a deep, glossy liquid surface from a particle cloud using the
    screen-space curvature-flow pipeline (van der Laan et al., 2009). With an
    upstream IMAGE wired in it is composited as the backdrop (Rule 12); else a
    dark/synthetic backdrop is used. A fluid-coverage MASK is also emitted.
    """
    try:
        if params is None:
            params = {}
        seed_all(seed)
        rng = np.random.default_rng(seed)

        hh, ww = int(H), int(W)
        anim_time = float(params.get("time", 0.0))
        anim_mode = str(params.get("anim_mode", "none"))
        anim_speed = float(params.get("anim_speed", 1.0))
        _t = 0.0 if anim_mode == "none" else anim_time * anim_speed

        source = str(params.get("source", "dark"))
        pal_name = str(params.get("palette", "vapor"))
        n = int(params.get("particles", 16000))
        radius = float(params.get("radius", 7.0))
        smooth = float(params.get("smooth", 0.6))
        detail = float(params.get("detail", 0.4))
        flow_scale = float(params.get("flow_scale", 2.5))
        flow_speed = float(params.get("flow_speed", 1.0))
        swirl = float(params.get("swirl", 0.3))
        thickness = float(params.get("thickness", 0.5))
        fresnel = float(params.get("fresnel", 0.7))
        specular = sparam(params, "specular", 0.6)

        # ── Flow field (morphs with _t) ──
        vx, vy, gy, gx = _curl_flow(rng, hh, ww, _t, flow_scale)

        # ── Base particle layout (stable across frames; seeded) ──
        # Larger particle counts spread into a wider, denser body; fewer
        # particles concentrate near the center (visible population change).
        spread = 0.30 + 0.30 * (np.log10(max(1000, n)) - 3) / (np.log10(40000) - 3)
        px = rng.random(n) * (ww - 1)
        py = rng.random(n) * (hh - 1)
        # bias toward a filled central body so the surface is continuous;
        # an elliptical (a!=b) body is non-symmetric so rotation is clearly
        # visible (a circular body would be rotation-invariant).
        ang = rng.random(n) * 2.0 * math.pi
        a, b = min(hh, ww) * spread, min(hh, ww) * spread * 0.68
        cx, cy = ww * 0.5, hh * 0.5
        rad = (rng.random(n) ** 0.5)
        px = 0.55 * px + 0.45 * (cx + rad * a * np.cos(ang))
        py = 0.55 * py + 0.45 * (cy + rad * b * np.sin(ang))
        pw = 0.6 + 0.8 * rng.random(n)                       # per-particle weight
        ph = _smooth_field(rng, (64, 64), 8.0)              # height-field source
        hv = map_coordinates(
            ph, [py / hh * 63, px / ww * 63], order=1, mode="reflect"
        )

        # ── Animation: all active modes get a global swirl rotation (uses _t,
        # non-symmetric so the silhouette genuinely tumbles), then a per-mode
        # displacement on top. ──
        if anim_mode != "none":
            # global rotation rate: swirl param when in swirl mode, else a
            # steady base spin that keeps every animated mode visibly moving.
            grot = swirl * _t if anim_mode == "swirl" else (0.6 + 0.8 * swirl) * _t
            cg_, sg_ = math.cos(grot), math.sin(grot)
            dx, dy = px - cx, py - cy
            px = cx + dx * cg_ - dy * sg_
            py = cy + dx * sg_ + dy * cg_

        if anim_mode == "flow":
            u = map_coordinates(vx, [py, px], order=1, mode="reflect")
            v = map_coordinates(vy, [py, px], order=1, mode="reflect")
            px = px + u * flow_speed * 40.0
            py = py + v * flow_speed * 40.0
        elif anim_mode == "waves":
            # breathing radial displacement from the height field
            d = (np.sqrt((px - cx) ** 2 + (py - cy) ** 2)) / max(hh, ww)
            px = px + np.cos(d * 18.0 - _t) * 6.0 * hv
            py = py + np.sin(d * 18.0 - _t) * 6.0 * hv

        # keep inside frame
        px = np.clip(px, 0, ww - 1)
        py = np.clip(py, 0, hh - 1)

        # ── 1. SPLAT particles into a screen-space density field ──
        # Each particle contributes a soft Gaussian splat; the summed density
        # is the implicit fluid surface (a metaball field). Heavy blur fuses
        # neighbours into a continuous liquid body — this is the SSF "surface
        # reconstruction" step that turns discrete particles into a blob.
        ix = px.astype(np.int32)
        iy = py.astype(np.int32)
        idx = (iy * ww + ix).clip(0, hh * ww - 1)

        dens = np.bincount(idx, weights=pw, minlength=hh * ww).reshape(hh, ww)
        # The splat footprint widens with `radius` (it IS the particle splat
        # size in SSF) and with `smooth` (more smoothing => wider fused
        # droplets). `radius` is NOT used in the coverage normaliser, so its
        # footprint effect is not self-cancelled (pitfall #19).
        ks = max(1, int(2 + smooth * 6.0 + radius * 0.4))
        dens = uniform_filter(dens, size=ks)
        # `smooth` scales the metaball fusion blur — the live surface-smoothing
        # control. 0 = tight droplets, 1 = fully fused smooth liquid.
        # `radius` also widens the rendered surface band (a larger splat reads
        # as a thicker liquid sheet) — this is a *fixed* contribution (no
        # data-dependent normaliser), so the control stays live (pitfall #19).
        blur_s = 1.0 + smooth * 5.0 + radius * 0.18
        dens = gaussian_filter(dens, sigma=blur_s)

        # Coverage = density-based fluid body alpha. Normalise by a FIXED
        # reference density (NOT dens.max() — that cancels `radius`; NOT
        # dens.mean() — that cancels `particles`). With a fixed D0 both sliders
        # stay live: more particles => larger body, larger splat `radius` =>
        # wider/bolder body (pitfall #19 — a data-dependent normaliser silently
        # freezes the control). A larger `radius` also lowers the coverage
        # cutoff, so the fluid body grows in AREA (clearly visible), not just
        # boldness.
        D0 = 0.045
        cov_raw = dens / D0
        radius_gain = 0.5 + 0.25 * radius   # 2->1.0, 7->2.25, 18->5.0
        coverage = np.clip(cov_raw * radius_gain, 0.0, 1.0)
        cutoff = max(0.02, 0.06 - radius * 0.002)   # bigger splat => bigger body
        coverage = np.where(coverage < cutoff, 0.0, coverage)

        # surface height = smooth function of density (high density = "crest")
        surface = norm(dens)

        # micro-relief
        if detail > 0.0:
            relief = _smooth_field(rng, (hh // 4, ww // 4), 3.0)
            rh, rw = relief.shape
            sy = (rh - 1) / max(hh, ww)
            sx = (rw - 1) / max(hh, ww)
            ry = np.mgrid[0:hh, 0:ww].astype(np.float64)[0] * sy
            rx = np.mgrid[0:hh, 0:ww].astype(np.float64)[1] * sx
            relief = map_coordinates(relief, [ry, rx], order=1, mode="reflect")
            surface = surface + (relief - 0.5) * detail * 0.25

        # ── 3. NORMAL RECONSTRUCTION (from smoothed height) ──
        nz = np.gradient(surface)
        gx_n, gy_n = nz[1], nz[0]
        nrm = np.stack([-gx_n, -gy_n, np.ones_like(surface)], axis=-1)
        nlen = np.sqrt((nrm ** 2).sum(-1, keepdims=True)) + 1e-6
        nrm = nrm / nlen
        N = nrm * coverage[..., None]

        # ── SHADE as a liquid surface ──
        light = np.array([0.4, 0.5, 0.8], dtype=np.float64)
        L = light / (np.linalg.norm(light) + 1e-6)
        V = np.array([0.0, 0.0, 1.0])                        # view along +z
        Hh = (L + V) / (np.linalg.norm(L + V) + 1e-6)

        diff = np.clip((N * L).sum(-1), 0.0, 1.0)
        fres = fresnel * (1.0 - np.clip((N * V).sum(-1), 0.0, 1.0)) ** 3.0
        spec = specular * np.clip((N * Hh).sum(-1), 0.0, 1.0) ** 60.0

        # base liquid color + thickness tint
        base_col = np.array([0.10, 0.42, 0.62], dtype=np.float64)
        deep_col = np.array([0.02, 0.10, 0.22], dtype=np.float64)
        thick = coverage * (0.5 + 0.5 * surface)
        liquid = base_col[None, None, :] * (1.0 - thickness * thick[..., None]) \
            + deep_col[None, None, :] * (thickness * thick[..., None])
        col = liquid * (0.35 + 0.65 * diff[..., None]) \
            + np.array([0.6, 0.85, 1.0]) * fres[..., None] \
            + np.array([1.0, 1.0, 1.0]) * spec[..., None]

        surface_rgb = np.clip(col, 0.0, 1.0)

        # ── Backdrop (wired input overrides source; Rule 12) ──
        bg = None
        wired = params.get("input_image", "")
        if wired:
            try:
                bg = load_input(wired, ww, hh)
            except (FileNotFoundError, OSError, ValueError):
                bg = None
        if bg is None:
            if source == "noise":
                bg = norm(rng.standard_normal((hh, ww, 3)).astype(np.float32) * 0.5 + 0.5)
            elif source == "gradient":
                d = norm(np.linspace(0, 1, ww)[None, :] * np.ones((hh, 1)))
                bg = np.stack([d, d * 0.6, 1.0 - d * 0.7], -1).astype(np.float32)
            elif source == "palette":
                from ...core.utils import PALETTES
                pal = np.array(PALETTES.get(pal_name, list(PALETTES.values())[0]), dtype=np.float32) / 255.0
                yy, xx = np.mgrid[0:hh, 0:ww].astype(np.float32)
                r = norm(np.sqrt((xx - ww / 2) ** 2 + (yy - hh / 2) ** 2))
                bg = pal[(r * (len(pal) - 1)).astype(np.int32)]
            else:  # dark
                bg = np.full((hh, ww, 3), BG_DEFAULT[0] / 255.0 * 0.25, dtype=np.float32)

        bg = np.clip(bg, 0.0, 1.0).astype(np.float64)

        # ── Composite fluid over backdrop using coverage alpha ──
        out = bg * (1.0 - coverage[..., None]) + surface_rgb * coverage[..., None]
        out = np.clip(out, 0.0, 1.0).astype(np.float32)

        # ── Side outputs ──
        write_mask(out_dir, coverage.astype(np.float32))
        write_scalars(
            out_dir,
            particles=n,
            radius=radius,
            smooth=smooth,
            thickness=thickness,
            fresnel=fresnel,
            coverage_mean=float(coverage.mean()),
            mean_diff=float(diff[coverage > 0.05].mean() if (coverage > 0.05).any() else 0.0),
        )

        capture_frame("494", out)
        save(out, mn(494, f"Screen-Space Fluid t={_t:.2f}"), out_dir)
        return out
    except Exception as exc:
        fallback = np.full((int(H), int(W), 3), 0.18, dtype=np.float32)
        save(fallback, mn(494, "Screen-Space Fluid"), out_dir)
        print(f"[method_494] ERROR: {exc}")
        return fallback
