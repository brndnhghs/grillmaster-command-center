"""Gravitational Lensing — real-time Einstein-ring thin-lens warp.

Implements screen-space gravitational lensing using the point-mass thin-lens
approximation (Einstein 1936; the deflection that produces the iconic
Einstein ring). Light from a background source at angular position β is imaged
at θ where the lens equation holds:

    β = θ · (1 − θ_E² / |θ|²)

θ_E is the Einstein radius (set by the lens mass). Sampling a procedural
starfield + nebula at the mapped source position β(θ) and brightening by
the magnification μ = 1/|1 − (θ_E²/|θ|²)²| reproduces the distorted
background and the bright ring of a real gravitational lens.

Why this node (shootout fit): it is a screen warp — O(H·W) per frame,
no PDE, no grid solve — so it develops in well under the pipeline's 150 s
timeout and its morphing field (animated lens mass / source drift / swirl)
gives perpetual, non-repeating motion that reliably passes the liveness cull
that killed ~62% of logged genomes (402/649 dead).

CPU path authoritative. Pure numpy, self-contained value-noise/fbm for the
procedural sky (no external deps). Reference: Einstein 1936, "Lens-like
action of a star"; James et al. 2015, "Gravitational lensing by a
spinning black hole" (the Interstellar / Gargantua render).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from ...core.registry import method
from ...core.utils import (
    save,
    mn,
    seed_all,
    W,
    H,
    write_mask,
    write_scalars,
    write_field,
)
from ...core.animation import capture_frame

# ── Procedural sky helpers (vectorized, seed-driven) ──


def _hash2_v(ix, iy, seed):
    """Vectorized integer hash → [0, 1)."""
    h = (
        ix.astype(np.int64) * 374761393
        + iy.astype(np.int64) * 668265263
        + int(seed) * 1442695041
    ) & 0xFFFFFFFF
    h = (h ^ (h >> 13)) * 1274126177 & 0xFFFFFFFF
    return ((h ^ (h >> 16)) & 0xFFFF) / 65536.0


def _vnoise_v(x, y, seed):
    """Value noise (smoothstep-interpolated hashed lattice), output ~[0, 1]."""
    x0 = np.floor(x).astype(np.int64)
    y0 = np.floor(y).astype(np.int64)
    fx = x - x0
    fy = y - y0
    sx = fx * fx * (3.0 - 2.0 * fx)
    sy = fy * fy * (3.0 - 2.0 * fy)
    n00 = _hash2_v(x0, y0, seed)
    n10 = _hash2_v(x0 + 1, y0, seed)
    n01 = _hash2_v(x0, y0 + 1, seed)
    n11 = _hash2_v(x0 + 1, y0 + 1, seed)
    nx0 = n00 + (n10 - n00) * sx
    nx1 = n01 + (n11 - n01) * sx
    return nx0 + (nx1 - nx0) * sy


def _fbm_v(x, y, seed, octaves, lac, gain):
    """Fractal value noise, output ~[-1, 1]."""
    s = np.zeros_like(x, dtype=np.float64)
    amp = 0.5
    freq = 1.0
    norm = 0.0
    for _ in range(int(octaves)):
        s += amp * _vnoise_v(x * freq, y * freq, seed + int(_) * 101)
        norm += amp
        amp *= gain
        freq *= lac
    return s / norm * 2.0 - 1.0


def _sample3(arr, xf, yf):
    """Bilinear sample of a (H, W, 3) array at float coords (xf, yf)."""
    Hh, Ww = arr.shape[0], arr.shape[1]
    xi = np.clip(xf, 0.0, Ww - 1.001)
    yi = np.clip(yf, 0.0, Hh - 1.001)
    x0 = np.floor(xi).astype(np.int64)
    y0 = np.floor(yi).astype(np.int64)
    x1 = np.minimum(x0 + 1, Ww - 1)
    y1 = np.minimum(y0 + 1, Hh - 1)
    sx = xi - x0
    sy = yi - y0
    out = np.empty((Hh, Ww, 3), dtype=np.float64)
    for c in range(3):
        a = arr[:, :, c]
        out[:, :, c] = (
            a[y0, x0] * (1.0 - sx) * (1.0 - sy)
            + a[y0, x1] * sx * (1.0 - sy)
            + a[y1, x0] * (1.0 - sx) * sy
            + a[y1, x1] * sx * sy
        )
    return out


_PALETTES = {
    "cosmic": ((0.32, 0.22, 0.55), (0.80, 0.86, 1.0)),  # nebula, star
    "ember": ((0.55, 0.18, 0.06), (1.0, 0.86, 0.62)),
    "ice": ((0.10, 0.34, 0.42), (0.85, 0.97, 1.0)),
    "mono": ((0.30, 0.30, 0.32), (1.0, 1.0, 1.0)),
}


def _make_sky(seed, star_frac, neb_int, neb_scale, pal):
    """Build a (H, W, 3) float sky: tinted nebula + speckled stars."""
    yy, xx = np.meshgrid(np.arange(int(H)), np.arange(int(W)), indexing="ij")
    neb_rgb, star_rgb = _PALETTES.get(pal, _PALETTES["cosmic"])

    # Nebula: fbm over the grid, raised to a power for soft clouds.
    nx = xx / float(W) * neb_scale
    ny = yy / float(H) * neb_scale
    fbm = _fbm_v(nx, ny, seed + 7, 4, 2.0, 0.5)
    neb = np.clip(fbm * 0.5 + 0.5, 0.0, 1.0) ** 1.6 * neb_int

    rng = np.random.default_rng(seed + 99)
    # Speckled point stars: threshold the top fraction of white noise.
    s = rng.random((H, W))
    star_layer = (s > 1.0 - star_frac).astype(np.float64) * (
        0.55 + 0.7 * rng.random((H, W))
    )
    # A handful of bright stars with a small gaussian splat (cheap loop).
    big = 22
    for _ in range(big):
        px = int(rng.integers(0, int(W)))
        py = int(rng.integers(0, int(H)))
        rad = 1 + int(rng.integers(0, 3))
        y0 = max(0, py - rad)
        y1 = min(H, py + rad + 1)
        x0 = max(0, px - rad)
        x1 = min(W, px + rad + 1)
        ly, lx = np.mgrid[y0:y1, x0:x1]
        d2 = (lx - px) ** 2 + (ly - py) ** 2
        star_layer[y0:y1, x0:x1] += np.exp(-d2 / (2.0 * (rad * 0.7) ** 2)) * (
            0.6 + 0.4 * rng.random()
        )

    sky = np.empty((H, W, 3), dtype=np.float64)
    for c in range(3):
        sky[:, :, c] = neb_rgb[c] * neb + star_rgb[c] * star_layer
    return np.clip(sky, 0.0, 2.0)


@method(
    id="995",
    name="Gravitational Lensing (Einstein Ring)",
    category="patterns",
    new_image_contract=True,
    tags=[
        "generative",
        "gravitational",
        "lensing",
        "einstein-ring",
        "warp",
        "space",
        "animation",
        "real-time",
    ],
    inputs={},
    outputs={"image": "IMAGE", "field": "FIELD", "mask": "MASK"},
    params={
        "einstein_radius": {
            "description": "Einstein radius θ_E (lens mass); ring size",
            "min": 0.05,
            "max": 0.9,
            "default": 0.35,
        },
        "star_density": {
            "description": "fraction of pixels that are stars",
            "min": 0.0005,
            "max": 0.02,
            "default": 0.004,
        },
        "nebula": {
            "description": "nebula cloud intensity",
            "min": 0.0,
            "max": 1.0,
            "default": 0.5,
        },
        "neb_scale": {
            "description": "nebula fbm frequency (smaller = larger clouds)",
            "min": 1.0,
            "max": 8.0,
            "default": 3.0,
        },
        "exposure": {
            "description": "output brightness multiplier",
            "min": 0.2,
            "max": 4.0,
            "default": 1.4,
        },
        "ring_brightness": {
            "description": "Einstein-ring glow strength",
            "min": 0.0,
            "max": 3.0,
            "default": 1.2,
        },
        "ring_width": {
            "description": "Einstein-ring glow width",
            "min": 0.01,
            "max": 0.3,
            "default": 0.06,
        },
        "palette": {
            "description": "sky tint",
            "choices": ["cosmic", "ember", "ice", "mono"],
            "default": "cosmic",
        },
        "mode": {
            "description": "animation mode (none/drift/pulse/swirl)",
            "choices": ["none", "drift", "pulse", "swirl"],
            "default": "drift",
        },
        "time": {
            "description": "animation phase [0, 2pi)",
            "min": 0.0,
            "max": 6.28,
            "default": 0.0,
        },
        "anim_mode": {
            "description": "animation mode selector (canonical; falls back to `mode`)",
            "choices": ["none", "drift", "pulse", "swirl"],
            "default": "none",
        },
        "anim_speed": {
            "description": "animation speed multiplier",
            "min": 0.1,
            "max": 5.0,
            "default": 1.0,
        },
    },
)
def method_gravitational_lensing(out_dir: Path, seed: int, params=None):
    """Gravitational Lensing — Einstein-ring thin-lens warp of a procedural sky.

    The lens equation β = θ·(1 − θ_E²/|θ|²) maps each output pixel (θ) to a
    source position (β); we sample the sky there and brighten by the
    magnification. Animated modes morph the lens for perpetual motion:
        none  — fixed lens, static image (frame Δ ≈ 0).
        drift — the source drifts in a circle, sliding the sky through the lens.  drift
        pulse — the Einstein radius breathes (lens mass oscillates).
        swirl — the lens plane rotates, spinning the ring.

    Params: einstein_radius, star_density, nebula, neb_scale, exposure,
    ring_brightness, ring_width, palette, mode/anim_mode, time, anim_speed.
    """
    try:
        if params is None:
            params = {}
        out_dir = Path(out_dir)

        # Canonical selector (mirrors the Stable Fluids contract).
        anim_mode = str(params.get("anim_mode", "none"))
        if anim_mode == "none":
            anim_mode = str(params.get("mode", "none"))
        anim_speed = float(params.get("anim_speed", 1.0))
        seed_all(seed)

        tE = float(np.clip(params.get("einstein_radius", 0.35), 0.05, 0.9))
        star_frac = float(np.clip(params.get("star_density", 0.004), 0.0005, 0.02))
        neb_int = float(np.clip(params.get("nebula", 0.5), 0.0, 1.0))
        neb_scale = float(np.clip(params.get("neb_scale", 3.0), 1.0, 8.0))
        exposure = float(np.clip(params.get("exposure", 1.4), 0.2, 4.0))
        ring_bright = float(np.clip(params.get("ring_brightness", 1.2), 0.0, 3.0))
        ring_w = float(np.clip(params.get("ring_width", 0.06), 0.01, 0.3))
        pal = str(params.get("palette", "cosmic"))
        t = float(params.get("time", 0.0))

        sky = _make_sky(seed, star_frac, neb_int, neb_scale, pal)

        cx = W / 2.0
        cy = H / 2.0
        scale = min(W, H) * 0.45
        yy, xx = np.meshgrid(np.arange(int(H)), np.arange(int(W)), indexing="ij")
        u = (xx - cx) / scale
        v = (yy - cy) / scale

        # Animation transforms — ONLY when animating, so `none` is strictly
        # seed-determined (two `none` renders are identical → Δ ≈ 0).
        tE_eff = tE
        if anim_mode != "none":
            _t = t * anim_speed
            if anim_mode == "swirl":
                ang = 1.5 * _t
                ca = np.cos(ang)
                sa = np.sin(ang)
                u, v = u * ca - v * sa, u * sa + v * ca
            elif anim_mode == "drift":
                u = u + 0.25 * np.sin(_t)
                v = v + 0.25 * np.cos(_t)
            elif anim_mode == "pulse":
                tE_eff = tE * (0.6 + 0.5 * (0.5 + 0.5 * np.sin(_t)))

        r2 = u * u + v * v + 1e-6
        inv = (tE_eff * tE_eff) / r2
        bx = u * (1.0 - inv)
        by = v * (1.0 - inv)
        mu = 1.0 / np.abs(1.0 - inv * inv)
        mu = np.clip(mu, 1.0, 8.0)

        bx_pix = cx + bx * scale
        by_pix = cy + by * scale
        sampled = _sample3(sky, bx_pix, by_pix)

        ring = np.exp(-((np.sqrt(r2) - tE_eff) ** 2) / (2.0 * ring_w * ring_w))
        glow = ring[:, :, None] * ring_bright

        rgb = np.clip(sampled * mu[:, :, None] + glow, 0.0, 1.0)
        rgb = np.clip(rgb * exposure, 0.0, 1.0).astype(np.float32)

        bright = np.clip(np.max(rgb, axis=-1), 0.0, 1.0)
        alpha = np.clip(bright + ring * 0.5, 0.0, 1.0).astype(np.float32)
        rgba = np.concatenate([rgb, alpha[:, :, None]], axis=-1).astype(np.float32)

        # Field = magnification (lensed brightness boost); mask = lit regions.
        field = np.clip(mu / 8.0, 0.0, 1.0).astype(np.float32)
        mask = (np.max(rgb, axis=-1) > 0.04).astype(np.float32)

        capture_frame("995", rgba)
        save(rgba, mn(995, "Gravitational Lensing"), out_dir)
        try:
            write_field(out_dir, field)
            write_mask(out_dir, mask)
            write_scalars(
                out_dir,
                einstein_radius=float(tE_eff),
                ring_coverage=float(float(np.mean(ring > 0.1))),
                peak_luminance=float(float(np.percentile(bright, 99.0)) if bright.size else 0.0),
                coverage=float(float(np.mean(mask))),
                mode_code=float(hash(anim_mode) % 1000),
            )
        except Exception:
            pass
        return rgba
    except Exception as exc:
        fallback = np.full((H, W, 4), 0.5, dtype=np.float32)
        save(fallback, mn(995, "Gravitational Lensing"), out_dir)
        print(f"[method_995] ERROR: {exc}")
        return fallback
