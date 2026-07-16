"""Ant Colony Optimization — stigmergic trail networks (Dorigo & Stützle, 1996/2004).

Agents (ants) leave two pheromone fields as they move:
  * searching ants deposit a "home" pheromone (H) along their path,
  * ants carrying food deposit a "food" pheromone (F) on the way back.
Each ant senses the gradient of the pheromone it is currently *following*
(F when searching, H when returning home) through three forward sensors and
steers up-gradient, so trails self-reinforce into lacy networks between the
nest and the food sources. Both fields diffuse and evaporate every step, which
keeps the animation smooth (no discrete-time strobing).

Architecture A: internal substep loop with capture_frame() per visible frame.
The ant/field state evolves across the loop, so the clip animates even when the
method is called once; the animation phase (time) only offsets mode-specific
parameters (food-source orbit / deposit breathing).
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
from PIL import Image

from ...core.registry import method
from ...core.utils import (
    save, mn, seed_all, W, H, write_scalars, write_field, write_particles,
)
from ...core.animation import capture_frame

WARM = np.array([1.0, 0.5, 0.12], dtype=np.float64)   # food pheromone colour
COOL = np.array([0.12, 0.7, 1.0], dtype=np.float64)   # home pheromone colour


def _diffuse(f: np.ndarray, k: float) -> np.ndarray:
    """Cheap 4-neighbour diffusion (toroidal), blended by k in [0,1]."""
    lap = (np.roll(f, 1, 0) + np.roll(f, -1, 0)
           + np.roll(f, 1, 1) + np.roll(f, -1, 1)) * 0.25
    return f * (1.0 - k) + lap * k


@method(
    id="974",
    name="Ant Colony",
    category="simulations",
    tags=["ant-colony", "stigmergy", "pheromone", "agents", "foraging", "animation", "color_intrinsic"],
    inputs={},
    outputs={"image": "IMAGE", "field": "FIELD", "particles": "PARTICLES", "luminance": "SCALAR"},
    params={
        "ants": {"description": "number of foraging ants", "min": 200, "max": 4000, "default": 1500},
        "food_sources": {"description": "number of food sources", "min": 1, "max": 10, "default": 4},
        "deposit": {"description": "pheromone deposited per step", "min": 0.1, "max": 2.0, "default": 0.7},
        "evaporate": {"description": "per-step pheromone retention (lower = faster fade)", "min": 0.90, "max": 0.999, "default": 0.985},
        "speed": {"description": "ant step length (px)", "min": 0.5, "max": 4.0, "default": 1.8},
        "sensor_angle": {"description": "ant sensor spread (rad)", "min": 0.2, "max": 1.6, "default": 0.7},
        "sensor_dist": {"description": "ant sensor reach (px)", "min": 2.0, "max": 30.0, "default": 10.0},
        "random_turn": {"description": "random wander strength (rad)", "min": 0.0, "max": 1.2, "default": 0.35},
        "n_frames": {"description": "simulation frames (visible)", "min": 60, "max": 400, "default": 140},
        "anim_mode": {"description": "animation mode", "choices": ["none", "forage", "drift", "pulse"], "default": "forage"},
        "anim_speed": {"description": "animation speed multiplier", "min": 0.1, "max": 3.0, "default": 1.0},
        "time": {"description": "animation phase [0, 2pi)", "min": 0.0, "max": 6.28, "default": 0.0},
    },
)
def method_ant_colony(out_dir: Path, seed: int, params=None):
    try:
        if params is None:
            params = {}
        t = float(params.get("time", 0.0))
        anim_mode = str(params.get("anim_mode", "forage"))
        anim_speed = float(params.get("anim_speed", 1.0))
        is_anim = anim_mode != "none" or t > 0.01

        n_ants = int(params.get("ants", 1500))
        n_food = int(params.get("food_sources", 4))
        deposit = float(params.get("deposit", 0.7))
        evaporate = float(params.get("evaporate", 0.985))
        speed = float(params.get("speed", 1.8))
        sensor_angle = float(params.get("sensor_angle", 0.7))
        sensor_dist = float(params.get("sensor_dist", 10.0))
        random_turn = float(params.get("random_turn", 0.35))
        n_frames = int(params.get("n_frames", 140))

        seed_all(seed)
        rng = np.random.default_rng(seed)

        _t = t * anim_speed if is_anim else 0.0

        # ── Simulation grid (coarse, upscaled to canvas) ──
        gw = max(32, W // 6)
        gh = max(24, H // 6)
        F = np.zeros((gh, gw), dtype=np.float64)   # food pheromone
        Hh = np.zeros((gh, gw), dtype=np.float64)  # home pheromone

        # ── Nest (canvas centre) and food sources ──
        nest = np.array([W / 2.0, H / 2.0], dtype=np.float64)
        nest_r = max(6.0, min(W, H) * 0.02)
        food_base = rng.uniform(0.12, 0.88, size=(n_food, 2))
        food_base[:, 0] *= W
        food_base[:, 1] *= H
        food_r = max(5.0, min(W, H) * 0.015)

        # ── Ants ──
        pos = rng.uniform(0, 1, size=(n_ants, 2)).astype(np.float64)
        pos[:, 0] *= (W - 1)
        pos[:, 1] *= (H - 1)
        heading = rng.uniform(0, 2 * math.pi, size=n_ants)
        carrying = np.zeros(n_ants, dtype=bool)

        def _to_grid(p):
            gx = np.clip((p[:, 0] / W * gw).astype(np.int64), 0, gw - 1)
            gy = np.clip((p[:, 1] / H * gh).astype(np.int64), 0, gh - 1)
            return gx, gy

        img = None
        for frame in range(n_frames):
            # per-clip phase (drives mode-specific evolution)
            phase = frame / max(1, n_frames - 1) * (2.0 * math.pi)

            # mode-specific modulation
            if anim_mode == "drift":
                # Food sources orbit the nest. Per-source radii vary (0.55→1.0×)
                # so the ring is NOT rotationally symmetric — an even source
                # count aliases under a 180° rotation and `time` looks dead at
                # t=π (frame-Δ probe caught Δ≈0.004). Varying radii keeps the
                # field genuinely time-dependent at every phase.
                orbit = phase + _t
                ang = np.linspace(0, 2 * math.pi, n_food, endpoint=False) + orbit
                radii = min(W, H) * 0.32 * np.linspace(0.55, 1.0, n_food)
                food = nest + radii[:, None] * np.stack([np.cos(ang), np.sin(ang)], axis=-1)
                evap = evaporate
                dep = deposit
            elif anim_mode == "pulse":
                b = 0.5 + 0.5 * math.sin(phase * 2.0 + _t)
                food = food_base
                evap = evaporate * (1.0 - 0.03 * b)
                dep = deposit * (0.5 + 1.1 * b)
            else:  # forage (and none)
                food = food_base
                evap = evaporate
                dep = deposit

            # ── Sense & steer ──
            # Each ant reads the pheromone it is currently following:
            # searching ants follow the food field F, returning ants follow
            # the home field Hh. Sample both and pick per-ant (carrying flag).
            ang3 = heading[:, None] + np.array([-1.0, 0.0, 1.0])[None, :] * sensor_angle
            sx = pos[:, 0:1] + np.cos(ang3) * sensor_dist
            sy = pos[:, 1:2] + np.sin(ang3) * sensor_dist
            gxs = np.clip((sx / W * gw).astype(np.int64), 0, gw - 1)
            gys = np.clip((sy / H * gh).astype(np.int64), 0, gh - 1)
            sF = F[gys, gxs]   # (N,3) food pheromone at the three sensors
            sH = Hh[gys, gxs]  # (N,3) home pheromone at the three sensors
            s = np.where(carrying[:, None], sH, sF)  # each ant reads what it follows
            sL, sC, sR = s[:, 0], s[:, 1], s[:, 2]
            dl = sL > sR
            dr = sR > sL
            heading = (heading
                       - (sensor_angle * 0.5) * dl.astype(np.float64)
                       + (sensor_angle * 0.5) * dr.astype(np.float64))
            heading = heading + rng.normal(0.0, random_turn, n_ants)

            # ── Move ──
            pos[:, 0] += np.cos(heading) * speed
            pos[:, 1] += np.sin(heading) * speed
            out_x = (pos[:, 0] < 0) | (pos[:, 0] > W - 1)
            out_y = (pos[:, 1] < 0) | (pos[:, 1] > H - 1)
            heading[out_x] = math.pi - heading[out_x]
            heading[out_y] = -heading[out_y]
            pos[:, 0] = np.clip(pos[:, 0], 0, W - 1)
            pos[:, 1] = np.clip(pos[:, 1], 0, H - 1)

            # ── State transitions ──
            d_nest = np.hypot(pos[:, 0] - nest[0], pos[:, 1] - nest[1])
            d_food = np.min(np.hypot(pos[:, 0:1] - food[:, 0], pos[:, 1:2] - food[:, 1]), axis=1)
            pick = (~carrying) & (d_food < food_r)
            carrying = np.where(pick, True, carrying)
            heading = np.where(pick, heading + math.pi, heading)
            drop = carrying & (d_nest < nest_r)
            carrying = np.where(drop, False, carrying)
            heading = np.where(drop, heading + math.pi, heading)

            # ── Deposit ──
            gx, gy = _to_grid(pos)
            np.add.at(Hh, (gy[~carrying], gx[~carrying]), dep)
            np.add.at(F, (gy[carrying], gx[carrying]), dep)

            # ── Diffuse + evaporate ──
            F = _diffuse(F, 0.14) * evap
            Hh = _diffuse(Hh, 0.14) * evap

            # ── Render ──
            # Ceil division so the upscaled grid always covers the full canvas;
            # the [:H, :W] slice then crops cleanly (floor division left it 510
            # rows tall vs the 512-row marker mask → boolean-index mismatch).
            sy_up = max(1, math.ceil(H / gh))
            sx_up = max(1, math.ceil(W / gw))
            F_up = np.repeat(np.repeat(F, sy_up, axis=0), sx_up, axis=1)[:H, :W]
            H_up = np.repeat(np.repeat(Hh, sy_up, axis=0), sx_up, axis=1)[:H, :W]
            rgb = np.clip(F_up[:, :, None] * WARM + H_up[:, :, None] * COOL, 0.0, 1.0)
            # nest + food markers for legibility
            yy, xx = np.ogrid[0:H, 0:W]
            nm = (yy - int(nest[1])) ** 2 + (xx - int(nest[0])) ** 2 <= max(3, int(nest_r)) ** 2
            rgb[nm] = np.array([0.95, 0.95, 0.98])
            for fx, fy in food:
                fm = (yy - int(fy)) ** 2 + (xx - int(fx)) ** 2 <= max(3, int(food_r)) ** 2
                rgb[fm] = np.array([1.0, 0.85, 0.2])
            rgb = rgb.astype(np.float64)

            # particles sidecar (x, y, vx, vy)
            pout = np.stack([pos[:, 0], pos[:, 1], np.cos(heading), np.sin(heading)], axis=-1).astype(np.float32)
            write_particles(out_dir, pout)

            if is_anim:
                capture_frame("974", rgb.astype(np.float32))

        # final frame is the last rendered rgb
        img = rgb.astype(np.float32)
        write_field(out_dir, (F + Hh).astype(np.float32))
        write_scalars(
            out_dir,
            n_ants=float(n_ants),
            n_food=float(n_food),
            mean_food_pher=float(F.mean()),
            mean_home_pher=float(Hh.mean()),
            carrying_frac=float(carrying.mean()),
        )
        save(img, mn(974, f"Ant Colony t={_t:.2f}"), out_dir)
        return img
    except Exception as exc:  # Rule 1: PNG in every code path
        fallback = np.full((H, W, 3), 128, dtype=np.uint8)
        save(fallback, mn(974, "Ant Colony"), out_dir)
        print(f"[method_974] ERROR: {exc}")
        return fallback
