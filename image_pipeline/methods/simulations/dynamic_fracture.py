"""
#145 — Dynamic Fracture / Crack-Branching Network

Bresenham line-driven crack propagation. Crack tips advance pixel by
pixel, guided by a rotating stress field. When stress at a tip exceeds
a threshold, the tip branches — creating two daughter cracks at slightly
different angles. No PDE-based damage diffusion; cracks are thin
(1-3px) binary traces that never saturate the canvas.

Physics:
  - Stress field guides tip direction via cone sampling
  - Tip momentum gives smooth curves
  - Branching triggered by high stress × heterogeneity
  - Periodic nucleation at high-stress hot spots away from existing cracks
  - Coverage capped at ~25% to prevent saturation

Animation modes:
  evolve:  rotating stress → branching tree fracture network
  radial:  outward radial cracks from center
  impact:  central impact with stress wave
  quench:  many cracks nucleate simultaneously

Render styles:
  damage:  bright cracks on dark textured background
  stress:  stress intensity as grayscale with crack overlay
  combined: stress in hue, cracks in luminance
"""
from __future__ import annotations
import math
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter

from ...core.registry import method
from ...core.utils import save, mn, seed_all, W, H
from ...core.animation import capture_frame


def _render_damage(crack: np.ndarray, stress: np.ndarray,
                   sh: int, sw: int) -> np.ndarray:
    """Bright crack lines on a dark, stress-textured background.

    Sharp crack cores (pure white) with a soft glow halo from Gaussian blur.
    """
    # Background: stress provides subtle texture (dark range 3–40)
    s_max = max(stress.max(), 0.01)
    bg = np.clip(stress / s_max * 35.0 + 5.0, 0, 45).astype(np.uint8)

    # Blur crack mask for glow halo
    crack_u8 = (crack * 255).astype(np.uint8)
    crack_img = Image.fromarray(crack_u8, mode='L')
    glow = np.array(
        crack_img.filter(ImageFilter.GaussianBlur(radius=1.5)),
        dtype=np.float32,
    ) / 255.0

    # Boost glow: brighter, sharper falloff
    glow = np.clip(glow * 1.8 - 0.1, 0.0, 1.0)

    # Core: crack pixels themselves are fully bright
    core = crack.astype(np.float32)

    # Combine: core (pure white) + glow halo + background
    intensity = np.maximum(core, glow)
    r = np.clip(bg.astype(np.float32) + intensity * 255.0, 0, 255).astype(np.uint8)
    g = np.clip(bg.astype(np.float32) + intensity * 255.0, 0, 255).astype(np.uint8)
    b = np.clip(bg.astype(np.float32) + intensity * 220.0, 0, 255).astype(np.uint8)
    return np.stack([r, g, b], axis=-1)


def _render_stress(stress_conc: np.ndarray,
                   sh: int, sw: int) -> np.ndarray:
    """Stress field as grayscale."""
    s = np.clip(stress_conc / max(stress_conc.max(), 0.01) * 255,
                0, 255).astype(np.uint8)
    return np.stack([s] * 3, axis=-1)


def _render_combined(crack: np.ndarray, stress: np.ndarray,
                     sh: int, sw: int) -> Image.Image:
    """Stress as hue, cracks as bright white overlay."""
    s_norm = stress / max(stress.max(), 0.001)
    hue = np.clip(s_norm * 0.5, 0, 0.66)
    sat = np.clip(s_norm, 0.2, 1.0)
    val = np.clip(s_norm * 0.7 + 0.1, 0, 1)

    hh, ss, vv = hue.ravel(), sat.ravel(), val.ravel()
    hi = np.floor(hh * 6).astype(np.int32) % 6
    f = hh * 6 - np.floor(hh * 6)
    p = vv * (1 - ss)
    q = vv * (1 - f * ss)
    t = vv * (1 - (1 - f) * ss)

    rgb = np.zeros((len(hh), 3), dtype=np.float64)
    for i in range(6):
        mask = hi == i
        if i == 0:
            rgb[mask] = np.column_stack([vv[mask], t[mask], p[mask]])
        elif i == 1:
            rgb[mask] = np.column_stack([q[mask], vv[mask], p[mask]])
        elif i == 2:
            rgb[mask] = np.column_stack([p[mask], vv[mask], t[mask]])
        elif i == 3:
            rgb[mask] = np.column_stack([p[mask], q[mask], vv[mask]])
        elif i == 4:
            rgb[mask] = np.column_stack([t[mask], p[mask], vv[mask]])
        elif i == 5:
            rgb[mask] = np.column_stack([vv[mask], p[mask], q[mask]])

    base = rgb.reshape(sh, sw, 3).astype(np.float32)

    # Overlay cracks as bright white on top
    crack_u8 = (crack * 255).astype(np.uint8)
    crack_img = Image.fromarray(crack_u8, mode='L')
    glow = np.array(
        crack_img.filter(ImageFilter.GaussianBlur(radius=1.2)),
        dtype=np.float32,
    ) / 255.0
    glow = np.clip(glow * 2.0 - 0.1, 0.0, 1.0)
    intensity = np.maximum(crack.astype(np.float32), glow)
    for c in range(3):
        base[:, :, c] = np.clip(base[:, :, c] + intensity * 0.8, 0, 1)

    return Image.fromarray((base * 255).astype(np.uint8), mode="RGB")


def _smooth_noise(sh: int, sw: int, scale: int,
                  rng: np.random.Generator) -> np.ndarray:
    """Multi-octave smooth noise for heterogeneity."""
    ch, cw = max(4, sh // scale), max(4, sw // scale)
    coarse = rng.random((ch, cw)) * 2 - 1
    cimg = Image.fromarray(((coarse + 1) * 127.5).astype(np.uint8), mode="L")
    up = np.array(cimg.resize((sw, sh), Image.BILINEAR), dtype=np.float64) / 127.5 - 1.0
    return up / scale


@method(
    id="145",
    name="Dynamic Fracture Network",
    category="simulations",
    tags=["animation", "fracture", "cracks", "branching",
           "mechanical", "instability"],
    timeout=300,
    params={
        "anim_mode": {
            "description": "fracture evolution mode",
            "choices": ["evolve", "radial", "impact", "quench"],
            "default": "evolve",
        },
        "render_style": {
            "description": "visualization style",
            "choices": ["damage", "stress", "combined"],
            "default": "damage",
        },
        "stress_rate": {
            "description": "applied stress magnitude",
            "min": 0.3, "max": 5.0, "default": 2.0,
        },
        "strength": {
            "description": "material strength (0.3-2.0, higher=stronger)",
            "min": 0.2, "max": 3.0, "default": 0.6,
        },
        "heterogeneity": {
            "description": "material heterogeneity (0.0-1.0)",
            "min": 0.0, "max": 1.0, "default": 0.5,
        },
        "n_frames": {
            "description": "simulation frames",
            "min": 100, "max": 600, "default": 300,
        },
    },
)
def method_fracture(out_dir: Path, seed: int, params=None):
    """Dynamic fracture via Bresenham-guided crack-branching tree growth.

    Crack tips advance pixel-by-pixel, guided by the local stress field
    gradient. When stress at a tip exceeds the material's strength, the
    tip branches into two. The crack mask is binary (thin lines, 1-3px
    wide) — no continuous damage diffusion, so saturation never occurs.
    Coverage is additionally capped at 25%.

    Anim modes:
      evolve:  rotating stress field → branching network
      radial:  outward radial cracks from center
      impact:  central impact with stress wave pulse
      quench:  many cracks nucleate simultaneously

    Render styles:
      damage:   bright cracks on dark stress-textured background
      stress:   stress intensity (white=high)
      combined: stress in hue, cracks in luminance
    """
    if params is None:
        params = {}

    anim_mode = str(params.get("anim_mode", "evolve"))
    render_style = str(params.get("render_style", "damage"))
    sigma_0 = float(params.get("stress_rate", 2.0))
    strength = float(params.get("strength", 0.6))
    het = float(params.get("heterogeneity", 0.5))
    n_frames = int(params.get("n_frames", 300))

    rng = np.random.default_rng(seed)
    seed_all(seed)

    grid_div = 2
    sh, sw = H // grid_div, W // grid_div
    fh, fw = H, W
    cy, cx = sh / 2, sw / 2
    yy, xx = np.ogrid[:sh, :sw]

    # ── Material heterogeneity field ──
    het_field = np.zeros((sh, sw), dtype=np.float64)
    for sc in [3, 6, 12, 24]:
        het_field += _smooth_noise(sh, sw, sc, rng)
    het_field = het_field / max(abs(het_field).max(), 0.01) * het

    # ── Multi-octave stress modifier for directional guidance ──
    stress_mod = np.zeros((sh, sw), dtype=np.float64)
    for sc in [8, 16, 32]:
        stress_mod += _smooth_noise(sh, sw, sc, rng)
    stress_mod = stress_mod / max(abs(stress_mod).max(), 0.01) * 0.25

    # ── State ──
    crack = np.zeros((sh, sw), dtype=bool)  # Binary crack trace (thin lines)

    # Crack tips: each is [y, x, angle_rad, age_frames, tip_strength]
    tips: list[list] = []

    # Derived parameters
    step_len = max(1, int(1.0 + sigma_0 * 0.6))        # pixels per frame
    branch_threshold = 0.04 + 0.08 * (2.0 / max(strength, 0.3))
    max_coverage = 0.25  # stop growth at 25% coverage
    nucleation_interval = max(8, 30 - int(sigma_0 * 5))

    # ── Initial seeds ──
    if anim_mode == "evolve":
        for _ in range(12):
            a = rng.uniform(0, 2 * math.pi)
            y = int(cy + rng.integers(-sh // 4, sh // 4))
            x = int(cx + rng.integers(-sw // 4, sw // 4))
            y = max(3, min(sh - 3, y))
            x = max(3, min(sw - 3, x))
            crack[y, x] = True
            tips.append([float(y), float(x), a, 0, 1.0])
    elif anim_mode == "radial":
        for angle in np.linspace(0, 2 * math.pi, 24, endpoint=False):
            a = angle + rng.uniform(-0.12, 0.12)
            sx = int(cx + 2 * math.cos(a))
            sy = int(cy + 2 * math.sin(a))
            if 0 <= sy < sh and 0 <= sx < sw:
                crack[sy, sx] = True
                tips.append([float(sy), float(sx), a, 0, 1.0])
        crack[int(cy), int(cx)] = True
    elif anim_mode == "impact":
        for angle in np.linspace(0, 2 * math.pi, 16, endpoint=False):
            a = angle + rng.uniform(-0.15, 0.15)
            sx = int(cx + 2 * math.cos(a))
            sy = int(cy + 2 * math.sin(a))
            if 0 <= sy < sh and 0 <= sx < sw:
                crack[sy, sx] = True
                tips.append([float(sy), float(sx), a, 0, 1.0])
        crack[int(cy), int(cx)] = True
    else:  # quench
        n_init = max(20, int(sh * sw * 0.0005))
        for _ in range(n_init):
            y = rng.integers(5, sh - 5)
            x = rng.integers(5, sw - 5)
            if not crack[y, x]:
                a = rng.uniform(0, 2 * math.pi)
                crack[y, x] = True
                tips.append([float(y), float(x), a, 0, 0.7 + 0.3 * rng.random()])

    print(f"  Dynamic Fracture | {anim_mode} σ₀={sigma_0:.1f} "
          f"str={strength:.2f} het={het:.2f} grid={sh}×{sw} "
          f"step={step_len} n_tips={len(tips)}")

    # Precompute neighbor offsets for stress concentration
    neigh_offsets = [(-1, -1), (-1, 0), (-1, 1), (0, -1),
                     (0, 1), (1, -1), (1, 0), (1, 1)]

    # ── Main simulation loop ──
    for frame in range(n_frames):
        t_total = frame * 0.015

        # ── Applied stress field ──
        if anim_mode == "evolve":
            theta = t_total * 0.5 + frame * 0.002
            gx, gy = math.cos(theta), math.sin(theta)
            stress = sigma_0 * (
                (xx / sw - 0.5) * gx + (yy / sh - 0.5) * gy + 1.0
            ) * 0.5
            stress *= (1.0 + 0.3 * math.sin(frame * 0.01))
        elif anim_mode == "radial":
            rr = np.sqrt((yy - cy)**2 + (xx - cx)**2)
            stress = sigma_0 * np.exp(-rr / (sh * 0.2))
        elif anim_mode == "impact":
            rr = np.sqrt((yy - cy)**2 + (xx - cx)**2)
            wave = np.sin(rr * 0.15 - t_total * 3.0) * np.exp(-rr / (sh * 0.3))
            stress = sigma_0 * np.exp(-rr / (sh * 0.25)) + sigma_0 * wave * 0.6
        else:
            stress = np.full((sh, sw), sigma_0)

        # Apply stress modifier for directional texture
        stress = stress * (1.0 + stress_mod)

        # ── Stress concentration at crack flanks ──
        stress_conc = stress.copy()
        K_tip = 3.0
        for dy, dx in neigh_offsets:
            rolled = np.roll(stress, (dy, dx), (0, 1))
            stress_conc = np.maximum(stress_conc,
                                     np.where(crack, rolled * K_tip, 0))

        # Also amplify stress at undamaged neighbors of cracks
        for dy, dx in neigh_offsets:
            neighbor_crack = np.roll(crack, (dy, dx), (0, 1))
            mask = neighbor_crack & ~crack
            stress_conc[mask] = np.maximum(stress_conc[mask],
                                           stress[mask] * K_tip * 0.8)

        # ── Grow tips ──
        new_tips = []
        crack_frac = crack.mean()

        for tip_data in tips:
            tip_y, tip_x, angle, age, tip_strength = tip_data
            iy, ix = int(round(tip_y)), int(round(tip_x))

            # Validate tip: must be in bounds and on a crack pixel
            if not (0 <= iy < sh and 0 <= ix < sw):
                continue
            if not crack[iy, ix]:
                continue
            # Stop if too much coverage
            if crack_frac > max_coverage:
                continue

            # ── Sample stress cone to find best growth direction ──
            n_samples = 9
            cone_half = 0.5 + 0.3 * het  # wider cone for more heterogeneous
            sample_angles = np.linspace(angle - cone_half,
                                        angle + cone_half, n_samples)
            # Add noise proportional to heterogeneity
            noise_scale = 0.35 * het
            sample_angles += rng.uniform(-noise_scale, noise_scale, n_samples)

            look_ahead = step_len * 2 + 1  # look a bit ahead
            best_val = -1e9
            best_angle = angle

            for sa in sample_angles:
                ny = iy + int(round(look_ahead * math.sin(sa)))
                nx = ix + int(round(look_ahead * math.cos(sa)))
                if 0 <= ny < sh and 0 <= nx < sw and not crack[ny, nx]:
                    val = stress_conc[ny, nx]
                    # Momentum bonus: prefer continuing in current direction
                    val += 0.25 * stress_conc[iy, ix] * abs(math.cos(sa - angle))
                    if val > best_val:
                        best_val = val
                        best_angle = sa

            # ── Advance the tip (draw Bresenham pixels) ──
            moved = False
            new_y, new_x = float(iy), float(ix)
            for _ in range(step_len):
                dy_i = int(round(math.sin(best_angle)))
                dx_i = int(round(math.cos(best_angle)))
                ny = int(round(new_y)) + dy_i
                nx = int(round(new_x)) + dx_i
                if 0 <= ny < sh and 0 <= nx < sw and not crack[ny, nx]:
                    crack[ny, nx] = True
                    new_y, new_x = float(ny), float(nx)
                    moved = True
                else:
                    break

            if not moved:
                continue  # tip is stuck

            # Smooth angle: momentum
            angle_diff = (best_angle - angle + math.pi) % (2 * math.pi) - math.pi
            new_angle = angle + angle_diff * 0.55

            # ── Branching ──
            # Higher stress at tip + older age + heterogeneity → more branching
            local_stress = stress_conc[int(round(new_y)), int(round(new_x))]
            branch_prob = branch_threshold * (
                0.5 + 0.5 * local_stress / max(stress_conc.max(), 0.01)
            ) * (1.0 + 0.5 * het) * min(1.0, age / 10.0)

            should_branch = rng.random() < branch_prob and age > 2

            if should_branch:
                branch_angle = 0.25 + 0.2 * rng.random()
                # Left branch
                new_tips.append([new_y, new_x,
                                 new_angle + branch_angle,
                                 0, tip_strength * 0.85])
                # Right branch
                new_tips.append([new_y, new_x,
                                 new_angle - branch_angle,
                                 0, tip_strength * 0.85])
            else:
                new_tips.append([new_y, new_x, new_angle,
                                 age + 1, tip_strength])

        tips = new_tips

        # ── Nucleate new cracks at high-stress hot spots ──
        # Use absolute stress threshold to avoid mean-drift issues
        nuc_hotspots = (stress_conc > sigma_0 * 0.35) & ~crack
        has_nuc_candidates = nuc_hotspots.any()

        # Emergency nucleation: when tips are critically low, force new seeds
        emergency_nuc = (len(tips) < 3 and has_nuc_candidates
                         and frame < n_frames * 0.92
                         and crack_frac < max_coverage * 0.7)

        # Periodic nucleation standard
        periodic_nuc = (frame % max(3, nucleation_interval // 2) == 0
                        and frame < n_frames * 0.85
                        and crack_frac < max_coverage * 0.7)

        if (periodic_nuc or emergency_nuc) and has_nuc_candidates:
            # Dilate crack mask to prefer areas farther from existing cracks
            crack_u8 = (crack * 255).astype(np.uint8)
            crack_img_pil = Image.fromarray(crack_u8, mode='L')
            dilated = np.array(
                crack_img_pil.filter(ImageFilter.MaxFilter(7)),
                dtype=bool,
            ) if not emergency_nuc else np.zeros((sh, sw), dtype=bool)

            candidates = nuc_hotspots & ~dilated
            # Fallback: if no candidates with dilation, try without
            if not candidates.any():
                candidates = nuc_hotspots
            if candidates.any():
                ys, xs = np.where(candidates)
                weights = stress_conc[ys, xs]
                w_sum = weights.sum()
                if w_sum > 0:
                    weights = weights / w_sum
                    n_nuc = max(1, min(3 + int(sigma_0 * 0.5), len(ys)))
                    for _ in range(n_nuc):
                        idx = rng.choice(len(ys), p=weights)
                        ny, nx = int(ys[idx]), int(xs[idx])

                        # Orient the new crack along the local stress gradient
                        ny1 = max(0, min(sh - 1, ny + 2))
                        ny0 = max(0, min(sh - 1, ny - 2))
                        nx1 = max(0, min(sw - 1, nx + 2))
                        nx0 = max(0, min(sw - 1, nx - 2))
                        dy = ny1 - ny0
                        dx = nx1 - nx0
                        if dy > 0 or dx > 0:
                            grad_y = (stress_conc[ny1, nx] - stress_conc[ny0, nx]) / max(dy, 1)
                            grad_x = (stress_conc[ny, nx1] - stress_conc[ny, nx0]) / max(dx, 1)
                            init_angle = math.atan2(grad_y, grad_x)
                        else:
                            init_angle = rng.uniform(0, 2 * math.pi)

                        crack[ny, nx] = True
                        # Also seed the next pixel in the growth direction
                        ny2 = int(round(ny + math.sin(init_angle)))
                        nx2 = int(round(nx + math.cos(init_angle)))
                        if 0 <= ny2 < sh and 0 <= nx2 < sw and not crack[ny2, nx2]:
                            crack[ny2, nx2] = True

                        tips.append([float(ny), float(nx), init_angle,
                                     0, 0.5 + 0.3 * rng.random()])

        # ── Render ──
        if render_style == "damage":
            canvas_np = _render_damage(crack, stress_conc, sh, sw)
            canvas = Image.fromarray(canvas_np, mode="RGB")
        elif render_style == "stress":
            canvas_np = _render_stress(stress_conc, sh, sw)
            canvas = Image.fromarray(canvas_np, mode="RGB")
        else:
            canvas = _render_combined(crack, stress_conc, sh, sw)

        canvas = canvas.resize((fw, fh), Image.BILINEAR)
        canvas_np = np.array(canvas, dtype=np.uint8)
        save(canvas_np, mn(145, "Dynamic Fracture"), out_dir)
        capture_frame("145", canvas_np)

        if frame % 60 == 0:
            frac = crack.mean() * 100
            print(f"  {frame}/{n_frames} tips={len(tips)} "
                  f"crack={frac:.1f}% D_mean={crack.mean():.4f}")

    frac = crack.mean() * 100
    print(f"  ✓ {n_frames} frames | fractured area={frac:.1f}%")
