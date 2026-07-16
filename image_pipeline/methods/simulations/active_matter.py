"""Active Matter — self-propelled particles & Motility-Induced Phase Separation.

Implements a minimal Active Brownian Particle (ABP) / Vicsek-style active-matter
model. Every particle is a self-propelled "micro-swimmer": it moves at a constant
speed v0 along a heading angle theta, while theta slowly aligns with the average
heading of its neighbours (polar alignment) plus angular noise.

The interesting, non-equilibrium behaviour — *Motility-Induced Phase Separation*
(MIPS) — emerges when alignment is *density-dependent*: in dense regions particles
align more strongly, so local clusters acquire a coherent net velocity and
coalesce into macroscopic, persistently churning "flocks"/"bubbles" of active
matter, even though there is NO attractive force whatsoever. This is the canonical
mechanism behind the collective motion of bacteria, colloidal rollers, and
cell monolayers.

References:
  * Marchetti, Joanny, Ramaswamy et al., "Hydrodynamics of soft active matter",
    Rev. Mod. Phys. 85, 1143 (2013). https://arxiv.org/abs/1207.2929
  * Palacci, Cottin-Bizonne, Ybert & Bocquet, "Sedimentation and effective
    temperature of active colloidal suspensions", Phys. Rev. Lett. 105, 088304
    (2010) / "Brownian dynamics of active colloids", J. ... (2013).
  * Vicsek, Czirók, Ben-Jacob, Cohen & Shochet, "Novel type of phase transition
    in a system of self-driven particles", Phys. Rev. Lett. 75, 1226 (1995).

No GPU twin yet; this is the authoritative CPU sim. Architecture A: an internal
simulation loop with periodic ``capture_frame`` calls.
"""

from __future__ import annotations

import math
import random
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from ...core.registry import method
from ...core.utils import (
    save, mn, seed_all, W, H, BG_DEFAULT, PALETTES,
    write_scalars, write_field, write_particles,
)
from ...core.animation import capture_frame


@method(
    id="915",
    name="Active Matter (MIPS)",
    category="simulations",
    new_image_contract=True,
    tags=["active-matter", "mips", "abp", "vicsek", "self-propelled",
          "non-equilibrium", "collective-motion", "simulation", "expanded"],
    inputs={},
    outputs={"image": "IMAGE", "luminance": "SCALAR", "density_field": "FIELD",
             "particles": "PARTICLES"},
    params={
        "particles": {"description": "number of self-propelled agents", "min": 50, "max": 2500, "default": 700},
        "motility": {"description": "self-propulsion speed v0 (px/step)", "min": 0.3, "max": 6.0, "default": 2.0},
        "interaction_radius": {"description": "neighbour alignment radius (px)", "min": 4, "max": 64, "default": 18},
        "alignment": {"description": "base polar-alignment rate (0=noiseless random walk, 1=instant align)", "min": 0.0, "max": 1.0, "default": 0.25},
        "noise": {"description": "angular noise sigma (radians)", "min": 0.0, "max": 2.0, "default": 0.35},
        "dense_boost": {"description": "extra alignment in dense regions (the MIPS driver)", "min": 0.0, "max": 1.0, "default": 0.45},
        "dense_threshold": {"description": "neighbour count above which extra alignment kicks in", "min": 1, "max": 24, "default": 6},
        "frames": {"description": "simulation steps", "min": 100, "max": 1200, "default": 420},
        "color_mode": {"description": "particle colouring", "choices": ["density", "heading", "speed", "constant"], "default": "density"},
        "palette": {"description": "colour palette name", "default": "viridis"},
        "trail_fade": {"description": "motion-trail persistence (higher = longer trails)", "min": 0.02, "max": 0.35, "default": 0.09},
        "point_size": {"description": "particle dot radius (px)", "min": 1, "max": 5, "default": 2},
        "anim_mode": {"description": "animation mode", "choices": ["none", "motility_pulse", "noise_sweep", "alignment_burst", "swirl", "polarity_drift"], "default": "none"},
        "anim_speed": {"description": "animation speed multiplier", "min": 0.1, "max": 5.0, "default": 1.0},
    },
)
def method_active_matter(out_dir: Path, seed: int, params=None):
    """Simulate self-propelled active particles and their MIPS phase separation.

    Args:
        out_dir: Output directory for the generated image.
        seed: Random seed for deterministic output.
        params: Dict with keys: particles, motility, interaction_radius,
            alignment, noise, dense_boost, dense_threshold, frames, color_mode,
            palette, trail_fade, point_size, anim_mode, anim_speed.
    """
    if params is None:
        params = {}
    anim_time = float(params.get("time", 0.0))
    anim_mode = params.get("anim_mode", "none")
    anim_speed = float(params.get("anim_speed", 1.0))
    seed_all(seed)
    rng = random.Random(seed)

    # ── Params ──
    n = int(params.get("particles", 700))
    v0 = float(params.get("motility", 2.0))
    R = float(params.get("interaction_radius", 18))
    align_base = float(params.get("alignment", 0.25))
    noise_sigma = float(params.get("noise", 0.35))
    dense_boost = float(params.get("dense_boost", 0.45))
    dense_threshold = int(params.get("dense_threshold", 6))
    frames = int(params.get("frames", 420))
    color_mode = params.get("color_mode", "density")
    palette_name = params.get("palette", "viridis")
    trail_fade = float(params.get("trail_fade", 0.09))
    point_size = max(1, int(params.get("point_size", 2)))

    # ── Animation mode flags ──
    _anim_motility_pulse = anim_mode == "motility_pulse"
    _anim_noise_sweep = anim_mode == "noise_sweep"
    _anim_alignment_burst = anim_mode == "alignment_burst"
    _anim_swirl = anim_mode == "swirl"
    _anim_polarity_drift = anim_mode == "polarity_drift"
    _anim_base_v0 = v0
    _anim_base_align = align_base
    _anim_base_noise = noise_sigma
    _anim_base_dense_boost = dense_boost
    _anim_base_dense_threshold = dense_threshold

    # ── Palette ──
    pal = PALETTES.get(palette_name, [(180, 120, 60), (240, 200, 120), (120, 200, 220), (80, 120, 220)])
    n_pal = len(pal)
    if n_pal == 0:
        pal = [(200, 200, 200)]
        n_pal = 1

    # ── Particle init ──
    # x, y, theta.
    px = np.array([rng.uniform(0, W) for _ in range(n)], dtype=np.float64)
    py = np.array([rng.uniform(0, H) for _ in range(n)], dtype=np.float64)
    pth = np.array([rng.uniform(0, 2 * math.pi) for _ in range(n)], dtype=np.float64)
    density = np.zeros(n, dtype=np.float32)

    # ── Spatial hash grid ──
    cell = max(R, 4.0)
    ncx = max(1, int(W / cell) + 1)
    ncy = max(1, int(H / cell) + 1)

    def build_grid():
        g = [[] for _ in range(ncx * ncy)]
        for i in range(n):
            cx = int(px[i] / cell)
            cy = int(py[i] / cell)
            if 0 <= cx < ncx and 0 <= cy < ncy:
                g[cy * ncx + cx].append(i)
        return g

    # ── Density field (coarse grid, written as FIELD) ──
    fld_res = 96
    cx_f = fld_res / W
    cy_f = fld_res / H

    def color_for(i):
        if color_mode == "heading":
            a = pth[i]
            h = (a + math.pi) / (2 * math.pi)
            return _hsv(h, 0.85, 1.0)
        if color_mode == "speed":
            return pal[min(n_pal - 1, int((v0 / 6.0) * (n_pal - 1)))]
        if color_mode == "constant":
            return pal[i % n_pal]
        # density
        dn = min(1.0, density[i] / max(2.0, dense_threshold * 1.6))
        idx = int(dn * (n_pal - 1))
        return pal[min(idx, n_pal - 1)]

    def _hsv(h, s, v):
        hi = int(h * 6) % 6
        f = h * 6 - hi
        p = v * (1 - s)
        q = v * (1 - f * s)
        t = v * (1 - (1 - f) * s)
        vals = [(v, t, p), (q, v, p), (p, v, t), (p, q, v), (t, p, v), (v, p, q)][hi]
        return (int(vals[0] * 255), int(vals[1] * 255), int(vals[2] * 255))

    # ── Image state (with motion-trail fade) ──
    img = Image.new("RGB", (W, H), BG_DEFAULT)
    drw = ImageDraw.Draw(img)
    bg_arr = np.array(BG_DEFAULT, dtype=np.float32)

    order_vals = []
    dens_vals = []

    # ══════════════════════════════════════════════════════
    #  SIMULATION LOOP
    # ══════════════════════════════════════════════════════
    for frame in range(frames):
        _t = anim_time * anim_speed + (frame / max(1, frames)) * 4 * math.pi * anim_speed

        # ── Per-frame animation modulation ──
        if _anim_motility_pulse:
            v0 = _anim_base_v0 * (0.4 + 0.6 * (0.5 + 0.5 * math.sin(_t * 0.3)))
        else:
            v0 = _anim_base_v0
        if _anim_noise_sweep:
            noise_sigma = _anim_base_noise * (0.2 + 1.8 * (0.5 + 0.5 * math.sin(_t * 0.25)))
        else:
            noise_sigma = _anim_base_noise
        if _anim_alignment_burst:
            align_base = _anim_base_align * (0.15 + 0.85 * (0.5 + 0.5 * math.sin(_t * 0.22)))
        else:
            align_base = _anim_base_align
        if _anim_polarity_drift:
            # Drift the global polarity (base alignment) so the flock's
            # coherence breathes — low alignment = turbulent gas, high = solid
            # flocks. Strong, visible modulation.
            align_base = _anim_base_align * (0.15 + 1.6 * (0.5 + 0.5 * math.sin(_t * 0.22)))
            dense_boost = _anim_base_dense_boost
        else:
            align_base = _anim_base_align
            dense_boost = _anim_base_dense_boost
        if _anim_swirl:
            swirl_bias = 0.6 * math.sin(_t * 0.15)
        else:
            swirl_bias = 0.0

        # ── Neighbour grid ──
        grid = build_grid()
        R2 = R * R
        nbr_count = np.zeros(n, dtype=np.int32)

        for i in range(n):
            cx = int(px[i] / cell)
            cy = int(py[i] / cell)
            sum_cos = 0.0
            sum_sin = 0.0
            cnt = 0
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    nx, ny = cx + dx, cy + dy
                    if 0 <= nx < ncx and 0 <= ny < ncy:
                        for j in grid[ny * ncx + nx]:
                            if j == i:
                                continue
                            ddx = px[j] - px[i]
                            ddy = py[j] - py[i]
                            if ddx * ddx + ddy * ddy <= R2:
                                sum_cos += math.cos(pth[j])
                                sum_sin += math.sin(pth[j])
                                cnt += 1
            nbr_count[i] = cnt
            density[i] = float(cnt)
            # Polar alignment toward neighbour average heading.
            if cnt > 0:
                target = math.atan2(sum_sin, sum_cos)
            else:
                target = pth[i]
            # Density-dependent alignment (the MIPS driver): more neighbours ->
            # stronger alignment, so dense clusters lock into coherent motion.
            extra = dense_boost if cnt >= dense_threshold else 0.0
            rate = min(1.0, align_base + extra)
            # Move heading toward target.
            dth = (target - pth[i] + math.pi) % (2 * math.pi) - math.pi
            pth[i] += rate * dth + rng.gauss(0.0, noise_sigma)
            # Swirl bias (global rotation of headings).
            pth[i] += swirl_bias * 0.02

        # ── Position update ──
        px += v0 * np.cos(pth)
        py += v0 * np.sin(pth)
        # Toroidal wrap keeps the system closed & cheap (no accumulation).
        px %= W
        py %= H

        # ── Order parameter (global polarization magnitude) ──
        mean_cos = float(np.mean(np.cos(pth)))
        mean_sin = float(np.mean(np.sin(pth)))
        order_vals.append(math.hypot(mean_cos, mean_sin))
        dens_vals.append(float(np.mean(nbr_count)))

        # ── Motion-trail fade ──
        arr = np.array(img, dtype=np.float32)
        arr = bg_arr[None, None, :] * trail_fade + arr * (1.0 - trail_fade)
        img = Image.fromarray(arr.astype(np.uint8))
        drw = ImageDraw.Draw(img)

        # ── Render particles ──
        si = point_size
        for i in range(n):
            col = color_for(i)
            x = int(px[i]); y = int(py[i])
            if si <= 1:
                img.putpixel((x, y), col)
            else:
                drw.ellipse((x - si, y - si, x + si, y + si), fill=col)

        # ── Capture every 3rd frame ──
        if frame % 3 == 0:
            capture_frame("915", np.array(img, dtype=np.float32) / 255.0)

    # ── Density field (coarse) ──
    fld = np.zeros((fld_res, fld_res), dtype=np.float32)
    ix = np.clip((px * cx_f).astype(np.int32), 0, fld_res - 1)
    iy = np.clip((py * cy_f).astype(np.int32), 0, fld_res - 1)
    np.add.at(fld, (iy, ix), 1.0)
    fmax = fld.max()
    if fmax > 0:
        fld /= fmax

    # ── Outputs ──
    _pos = np.stack([px, py], axis=1).astype(np.float32)
    _vel = np.stack([np.cos(pth) * _anim_base_v0, np.sin(pth) * _anim_base_v0], axis=1).astype(np.float32)
    write_scalars(out_dir,
                  luminance=float(np.mean(np.array(img, dtype=np.float32) / 255.0)),
                  order_param=float(np.mean(order_vals)),
                  mean_density=float(np.mean(dens_vals)))
    write_field(out_dir, fld)
    write_particles(out_dir, np.concatenate([_pos, _vel], axis=1))

    try:
        save(img, mn(915, "Active Matter (MIPS)"), out_dir)
    except Exception:
        img.save(str(out_dir / (mn(915, "Active Matter (MIPS)") + ".png")))
