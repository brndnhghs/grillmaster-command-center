"""Physarum transport network — Jeff Jones (2010) agent + trail model.

Slime-mold (Physarum polycephalum) foraging simulation after Sage Jenson's
GPU implementation (cargocollective.com/sagejenson/physarum), which follows
Jones' "Characteristics of pattern formation and evolution in approximations
of Physarum transport networks."

Two coupled layers, each tick:
  • AGENT layer  — many particles, each with a position + heading and three
                   sensors (front, front-left, front-right).
  • TRAIL layer  — a 2D intensity grid (like an image) the agents read from
                   and write to.

Per-tick sub-steps (the Jones "six steps"):
  1. SENSE   — each agent samples the trail map at its 3 sensors.
  2. ROTATE  — steer toward the strongest sensor (turn left / right / straight).
  3. MOVE    — step forward by `step_size`; wrap toroidally.
  4. DEPOSIT — add `deposit_amount` to the trail at the agent's cell.
  5. DIFFUSE — 3x3 mean blur of the whole trail map.
  6. DECAY   — multiply the trail by `decay` so old paths fade.

The emergent result is the classic Physarum vein / network / blob patterns.
Collision detection (one-particle-per-cell) is intentionally omitted, as in
Jenson's implementation, which yields richer patterns and removes sequential
dependence. Pace is one operation per frame; vary `n_frames` to evolve.

Architecture A: internal tick loop with `capture_frame()` per visible frame.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
from PIL import Image

from ...core.registry import method
from ...core.utils import (
    save, mn, seed_all, W, H, write_scalars, write_field, write_particles,
    PALETTES,
)
from ...core.animation import capture_frame


def _box_blur3x3(trail: np.ndarray) -> np.ndarray:
    """Separable 3x3 mean blur (pure numpy, no scipy dependency)."""
    # horizontal pass (mode='same' via edge padding)
    k = np.array([1.0, 1.0, 1.0]) / 3.0
    row = np.apply_along_axis(
        lambda r: np.convolve(r, k, mode="same"), axis=1, arr=trail
    )
    # vertical pass
    col = np.apply_along_axis(
        lambda c: np.convolve(c, k, mode="same"), axis=0, arr=row
    )
    return col


def _intensity_to_color(trail: np.ndarray, colormode: str, pal_name: str,
                        bg_light: bool) -> np.ndarray:
    """Map normalized trail intensity -> RGB image."""
    mx = trail.max()
    norm = trail / mx if mx > 1e-6 else trail
    norm = np.clip(norm, 0.0, 1.0)
    if colormode == "mono":
        if bg_light:
            img = np.ones((H, W, 3), dtype=np.float64)
            img *= (1.0 - norm)[..., None]
            # light bg, dark veins -> invert so veins are dark
            img = (1.0 - norm)[..., None] * np.array([0.05, 0.05, 0.08]) + \
                  norm[..., None] * np.array([0.95, 0.95, 0.92])
        else:
            img = norm[..., None] * np.array([0.95, 0.92, 0.85])
        return np.clip(img, 0.0, 1.0)
    # palette mode: sample a ramp by intensity
    pal = PALETTES.get(pal_name)
    if not pal:
        pal = PALETTES.get("amber")
    pal_arr = np.array(pal, dtype=np.float64) / 255.0
    # build a smooth ramp across the palette
    idx = np.clip(norm * (len(pal_arr) - 1), 0, len(pal_arr) - 1).astype(np.int64)
    ramp = pal_arr[idx]  # (H,W,3)
    if bg_light:
        bg = np.array([0.95, 0.95, 0.92])
    else:
        bg = np.array([0.02, 0.02, 0.04])
    # blend veins over background by intensity
    img = norm[..., None] * ramp + (1.0 - norm[..., None]) * bg
    return np.clip(img, 0.0, 1.0)


@method(
    id="530",
    name="Physarum Transport Network",
    category="simulations",
    tags=["physarum", "slime-mold", "agents", "trail-map", "foraging", "animation"],
    inputs={},
    outputs={"image": "IMAGE", "field": "FIELD", "particles": "PARTICLES", "luminance": "SCALAR"},
    params={
        "agents": {"description": "number of slime agents", "min": 500, "max": 40000, "default": 12000},
        "spawn": {"description": "agent initial placement", "choices": ["random", "ring", "center"], "default": "random"},
        "sensor_dist": {"description": "how far ahead sensors look (px)", "min": 1.0, "max": 40.0, "default": 9.0},
        "sensor_angle": {"description": "left/right sensor angle (radians)", "min": 0.05, "max": 1.8, "default": 0.5},
        "rotation_angle": {"description": "turn step per tick (radians)", "min": 0.02, "max": 1.2, "default": 0.4},
        "step_size": {"description": "move distance per tick (px)", "min": 0.2, "max": 6.0, "default": 1.0},
        "deposit_amount": {"description": "trail deposited per agent", "min": 0.05, "max": 5.0, "default": 1.0},
        "decay": {"description": "trail multiplicative decay (higher = persists)", "min": 0.80, "max": 0.99, "default": 0.93},
        "diffuse": {"description": "diffusion blend with 3x3 mean (0=none,1=full)", "min": 0.0, "max": 1.0, "default": 0.6},
        "colormode": {"description": "vein color: mono (b/w) or palette", "choices": ["mono", "palette"], "default": "palette"},
        "palette": {"description": "palette name when colormode=palette", "default": "amber"},
        "bg_style": {"description": "background (dark/light)", "choices": ["dark", "light"], "default": "dark"},
        "n_frames": {"description": "simulation ticks", "min": 30, "max": 600, "default": 220},
    },
)
def method_physarum(out_dir: Path, seed: int, params=None):
    try:
        if params is None:
            params = {}
        agents = int(params.get("agents", 12000))
        spawn = str(params.get("spawn", "random"))
        sensor_dist = float(params.get("sensor_dist", 9.0))
        sensor_angle = float(params.get("sensor_angle", 0.5))
        rotation_angle = float(params.get("rotation_angle", 0.4))
        step_size = float(params.get("step_size", 1.0))
        deposit_amount = float(params.get("deposit_amount", 1.0))
        decay = float(params.get("decay", 0.93))
        diffuse = float(params.get("diffuse", 0.6))
        colormode = str(params.get("colormode", "palette"))
        pal_name = str(params.get("palette", "amber"))
        bg_style = str(params.get("bg_style", "dark"))
        n_frames = int(params.get("n_frames", 220))

        seed_all(seed)
        rng = np.random.default_rng(seed)

        bg_light = bg_style == "light"

        # ── Trail map ──
        trail = np.zeros((H, W), dtype=np.float64)

        # ── Agent initialization ──
        agents = max(1, min(agents, 40000))
        if spawn == "ring":
            cx, cy = W / 2.0, H / 2.0
            r = min(W, H) * 0.35
            a = rng.uniform(0, 2 * math.pi, size=agents)
            pos = np.stack([cx + r * np.cos(a), cy + r * np.sin(a)], axis=-1)
            heading = a + math.pi  # point inward
        elif spawn == "center":
            cx, cy = W / 2.0, H / 2.0
            pos = np.stack([
                cx + rng.normal(0, min(W, H) * 0.02, size=agents),
                cy + rng.normal(0, min(W, H) * 0.02, size=agents),
            ], axis=-1)
            heading = rng.uniform(0, 2 * math.pi, size=agents)
        else:  # random
            pos = rng.uniform(0, [W - 1, H - 1], size=(agents, 2))
            heading = rng.uniform(0, 2 * math.pi, size=agents)

        pos = pos.astype(np.float64)
        heading = heading.astype(np.float64)

        def _sample_trail(px, py):
            """Bilinear sample of the trail map at float pixel coords."""
            fx = np.clip(px, 0, W - 1)
            fy = np.clip(py, 0, H - 1)
            x0 = fx.astype(np.int64); y0 = fy.astype(np.int64)
            x1 = np.minimum(x0 + 1, W - 1); y1 = np.minimum(y0 + 1, H - 1)
            tx = fx - x0; ty = fy - y0
            w00 = (1 - tx) * (1 - ty); w10 = tx * (1 - ty)
            w01 = (1 - tx) * ty; w11 = tx * ty
            return (trail[y0, x0] * w00 + trail[y0, x1] * w10 +
                    trail[y1, x0] * w01 + trail[y1, x1] * w11)

        img = None
        for frame in range(n_frames):
            # ── 1. SENSE ──
            ca = np.cos(heading); sa = np.sin(heading)
            # front
            fx = pos[:, 0] + ca * sensor_dist
            fy = pos[:, 1] + sa * sensor_dist
            # front-left
            la = heading - sensor_angle
            lx = pos[:, 0] + np.cos(la) * sensor_dist
            ly = pos[:, 1] + np.sin(la) * sensor_dist
            # front-right
            ra = heading + sensor_angle
            rx = pos[:, 0] + np.cos(ra) * sensor_dist
            ry = pos[:, 1] + np.sin(ra) * sensor_dist
            s_f = _sample_trail(fx, fy)
            s_l = _sample_trail(lx, ly)
            s_r = _sample_trail(rx, ry)

            # ── 2. ROTATE ──
            turn = np.zeros(agents, dtype=np.float64)
            left_mask = (s_l > s_f) & (s_l >= s_r)
            right_mask = (s_r > s_f) & (s_r >= s_l)
            turn[left_mask] = -rotation_angle
            turn[right_mask] = rotation_angle
            # tie / straight -> no turn
            heading = heading + turn

            # ── 3. MOVE (toroidal wrap) ──
            pos[:, 0] += np.cos(heading) * step_size
            pos[:, 1] += np.sin(heading) * step_size
            pos[:, 0] = np.mod(pos[:, 0], W - 1)
            pos[:, 1] = np.mod(pos[:, 1], H - 1)

            # ── 4. DEPOSIT ──
            ix = np.clip(pos[:, 0].astype(np.int64), 0, W - 1)
            iy = np.clip(pos[:, 1].astype(np.int64), 0, H - 1)
            np.add.at(trail, (iy, ix), deposit_amount)

            # ── 5. DIFFUSE (3x3 mean) ──
            if diffuse > 0.0:
                blurred = _box_blur3x3(trail)
                trail = (1.0 - diffuse) * trail + diffuse * blurred

            # ── 6. DECAY ──
            trail *= decay

            # ── render ──
            img = _intensity_to_color(trail, colormode, pal_name, bg_light)
            pout = np.stack([pos[:, 0], pos[:, 1],
                             np.cos(heading), np.sin(heading)], axis=-1).astype(np.float32)
            write_particles(out_dir, pout)
            capture_frame("530", img.astype(np.float32))

        if img is None:
            img = _intensity_to_color(trail, colormode, pal_name, bg_light)
        img = img.astype(np.float32)

        write_field(out_dir, trail.astype(np.float32))
        write_scalars(out_dir,
                      mean_trail=float(trail.mean()),
                      max_trail=float(trail.max()),
                      n_agents=float(agents))
        save(img, mn(530, "Physarum"), out_dir)
        return img
    except Exception as exc:
        fallback = np.full((H, W, 3), 128, dtype=np.uint8)
        save(fallback, mn(530, "Physarum"), out_dir)
        print(f"[method_530] ERROR: {exc}")
        return fallback
