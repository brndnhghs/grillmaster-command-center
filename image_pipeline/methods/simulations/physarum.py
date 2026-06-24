"""#86 — Physarum Slime Mold
Jeff Jones 2010 Physarum transport networks simulation.
Agent-based slime mold that creates organic vein networks
through chemoattractant sensing, movement, and trail diffusion.
"""
from __future__ import annotations
import math
from pathlib import Path

import numpy as np
import cv2

from ...core.registry import method
from ...core.utils import save, norm, mn, seed_all, BG_DEFAULT, W, H, write_field, write_particles
from ...core.animation import capture_frame


# ─── Colormap rendering (global ceiling) ───────────────────────────────────

def _render_trail(trail: np.ndarray, colormap_name: str, ceiling: float = None) -> np.ndarray:
    """Render log-scaled trail map to float32 [0,1] RGB.

    When ceiling is provided, uses global normalization — early frames
    are dark (trail ≪ ceiling), later frames brighten as the network
    matures. This makes emergence visible. When ceiling is None,
    falls back to per-frame min/max (for single-frame renders).
    """
    log_trail = np.log1p(trail)

    if ceiling is not None and ceiling > 0:
        normalized = np.clip(log_trail / ceiling, 0.0, 1.0)
    else:
        vmax = log_trail.max()
        if vmax > 0:
            normalized = log_trail / vmax
        else:
            normalized = np.zeros_like(log_trail)

    img = np.zeros((H, W, 4), dtype=np.float32)

    if colormap_name == "blues":
        img[:, :, 0] = normalized * 0.35
        img[:, :, 1] = normalized * 0.75
        img[:, :, 2] = 0.04 + normalized * 0.96
    elif colormap_name == "plasma":
        t = normalized
        img[:, :, 0] = np.clip(t * 2.0, 0, 1)
        img[:, :, 1] = np.clip(t * 2.0 - 0.5, 0, 1) * 0.9
        img[:, :, 2] = np.clip(1.0 - t * 1.5, 0, 1)
    elif colormap_name == "neon":
        img[:, :, 0] = normalized * 0.25
        img[:, :, 1] = 0.04 + normalized * 0.96
        img[:, :, 2] = normalized * 0.45
    elif colormap_name == "amber":
        img[:, :, 0] = 0.04 + normalized * 0.96
        img[:, :, 1] = normalized * 0.65
        img[:, :, 2] = normalized * 0.12

    # Alpha: trail intensity → transparency
    img[:, :, 3] = np.clip(normalized * 1.2, 0, 1)  # semi-transparent at low intensity, opaque at high

    return np.clip(img, 0.0, 1.0)


# ─── Jeff Jones 2010 Physarum transport networks ───────────────────────────

@method(id="86", name="Physarum Slime Mold", category="simulations",
         tags=["physarum", "slime-mold", "agents", "organic", "animation"],
         timeout=300,
         params={
             "num_agents": {"description": "number of slime mold agents",
                            "min": 500, "max": 5000, "default": 2000},
             "sa": {"description": "sensor angle (radians)",
                    "min": 0.1, "max": 1.5, "default": 0.7},
             "sd": {"description": "sensor distance (pixels)",
                    "min": 2, "max": 30, "default": 9},
             "ra": {"description": "rotation angle (radians)",
                    "min": 0.1, "max": 1.5, "default": 0.3},
             "md": {"description": "move distance (pixels)",
                    "min": 1, "max": 5, "default": 1},
             "deposit": {"description": "trail deposit amount",
                         "min": 1, "max": 50, "default": 5},
             "decay": {"description": "trail decay factor",
                       "min": 0.5, "max": 0.99, "default": 0.85},
             "blur_sigma": {"description": "diffusion blur sigma",
                            "min": 0.5, "max": 5.0, "default": 1.0},
             "n_frames": {"description": "total animation frames",
                          "min": 50, "max": 300, "default": 150},
             "colormap": {"description": "color scheme",
                          "choices": ["blues", "plasma", "neon", "amber"],
                          "default": "blues"},"anim_mode": {"description": "animation mode",
                           "choices": ["none", "evolve"],
                           "default": "none"},
             "anim_speed": {"description": "animation speed multiplier",
                            "min": 0.1, "max": 5.0, "default": 1.0},
         },
         outputs={"image": "IMAGE", "luminance": "SCALAR", "field": "FIELD", "particles": "PARTICLES"})
def method_physarum(out_dir: Path, seed: int, params=None):
    """Simulate Physarum polycephalum slime mold transport networks.

    Implements Jeff Jones' 2010 agent-based model. Agents sense
    chemoattractant trail at three forward positions, turn toward the
    strongest signal, move forward, and deposit trail. The trail map
    diffuses and decays each frame.

    Animation (evolve mode): the network emerges organically over
    ~150 frames — starts dark with no trail, brightens as agents build
    the transport network. No parameter modulation — pure growth.

    Static (none mode): renders the mature network after all frames.

    Args:
        out_dir: Output directory.
        seed: Random seed.
        params: Parameter overrides dict.
    """
    if params is None:
        params = {}

    t = float(params.get("time", 0.0))
    anim_speed = float(params.get("anim_speed", 1.0))
    anim_mode = params.get("anim_mode", "none")

    seed_all(seed)
    rng = np.random.default_rng(seed)

    num_agents = int(params.get("num_agents", 2000))
    sa = float(params.get("sa", 0.7))
    sd = float(params.get("sd", 9))
    ra = float(params.get("ra", 0.3))
    md = float(params.get("md", 1))
    deposit = float(params.get("deposit", 5))
    decay = float(params.get("decay", 0.85))
    blur_sigma = float(params.get("blur_sigma", 1.0))
    n_frames = int(params.get("n_frames", 150))
    colormap_name = params.get("colormap", "blues")

    _t = t * anim_speed

    # ── Initialize agents ──
    positions = rng.random((num_agents, 2)) * np.array([W, H], dtype=np.float32)
    headings = rng.random(num_agents, dtype=np.float32) * 2.0 * math.pi

    # ── Trail map ──
    trail = np.zeros((H, W), dtype=np.float32)

    # ── Global ceiling for emergence rendering ──
    # Trail max converges to ~log1p(8) after 144 frames at default params.
    # Use 0.001 factor so ceiling ≈ 2.4 — final frame reaches ~90% brightness,
    # early frames are proportionally dimmer (visible growth).
    ceiling = max(np.log1p(num_agents * deposit * 0.001), 0.1)

    # ── Kernel for blur (constant, not varying per frame) ──
    ksize = max(3, int(blur_sigma * 6.0 + 1.0))
    if ksize % 2 == 0:
        ksize += 1

    # ── Simulation loop ──
    for frame in range(n_frames):
        # Per-frame _t for seed variation between animations
        _frame_seed = seed + int((t + frame / max(1, n_frames)) * anim_speed * 10000)

        # ── SENSE → DECIDE → MOVE → DEPOSIT ──
        for i in range(num_agents):
            px, py = positions[i]
            heading = headings[i]

            cx = px + math.cos(heading) * sd
            cy = py + math.sin(heading) * sd
            lx = px + math.cos(heading - sa) * sd
            ly = py + math.sin(heading - sa) * sd
            rx = px + math.cos(heading + sa) * sd
            ry = py + math.sin(heading + sa) * sd

            cx_w = int(cx) % W; cy_w = int(cy) % H
            lx_w = int(lx) % W; ly_w = int(ly) % H
            rx_w = int(rx) % W; ry_w = int(ry) % H

            c_val = trail[cy_w, cx_w]
            l_val = trail[ly_w, lx_w]
            r_val = trail[ry_w, rx_w]

            if c_val > l_val and c_val > r_val:
                pass
            elif l_val > c_val and l_val > r_val:
                headings[i] -= ra
            elif r_val > c_val and r_val > l_val:
                headings[i] += ra
            elif c_val < l_val and c_val < r_val:
                headings[i] += ra if rng.random() > 0.5 else -ra

            px += math.cos(headings[i]) * md
            py += math.sin(headings[i]) * md
            px %= W; py %= H
            positions[i] = (px, py)

            ix = int(px) % W; iy = int(py) % H
            trail[iy, ix] += deposit

        # ── Diffuse + decay ──
        trail = cv2.GaussianBlur(trail, (ksize, ksize), blur_sigma)
        trail *= decay

        # ── Capture (every frame for smooth animation) ──
        img = _render_trail(trail, colormap_name, ceiling=ceiling)
        capture_frame("86", img)

    # ── Final render and save (single-frame: use per-frame normalization) ──
    final_img = _render_trail(trail, colormap_name, ceiling=None)
    capture_frame("86", final_img)
    write_field(out_dir, trail)
    _vx = np.cos(headings).astype(np.float32)
    _vy = np.sin(headings).astype(np.float32)
    write_particles(out_dir, np.stack([positions[:, 0], positions[:, 1], _vx, _vy], axis=1))
    save(final_img, mn(86, "Physarum"), out_dir)
    return final_img
