"""
#120 — LV 3-Species Food Web (Prey + Specialist + Generalist)

Three-species spatial Lotka-Volterra with intraguild predation.
Prey (green), specialist predator (red), generalist predator (blue).
The specialist only eats prey; the generalist eats both prey AND the
specialist — creating a food web with dominance fronts, shifting
territories, and richer spatiotemporal dynamics.

Physics: ∂u/∂t = αu - β₁uv₁ - β₂uv₂ + Du∇²u   (prey)
         ∂v₁/∂t = δ₁uv₁ - γ₁v₁ - εv₁v₂ + Dv₁∇²v₁ (specialist)
         ∂v₂/∂t = δ₂uv₂ + εv₁v₂ - γ₂v₂ + Dv₂∇²v₂ (generalist)

Architecture A — single-call internal simulation with capture_frame().

Animation modes:
- evolve: uniform noise → multi-species front dynamics
- wave: prey-seeded wave front, species stratify behind
- mosaic: isolated clusters of each species, watch them compete
"""
from __future__ import annotations
import math
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter

from ...core.registry import method
from ...core.utils import save, mn, seed_all, W, H, write_field
from ...core.animation import capture_frame


def _load_image_seed_3sp(path: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load an image and return edge-seeded 3-species fields + modulation mask.

    Edge strength maps to prey (u), complement maps to specialist (v1),
    and luminance maps to generalist (v2). This creates habitat partitioning
    that drives visible territorial front dynamics.

    Returns:
        u_seed:    prey (high at edges)
        v1_seed:   specialist (high in flat interiors)
        v2_seed:   generalist (high in bright areas)
        modulate:  grayscale brightness [0,1] for continuous forcing
    """
    img = Image.open(str(path)).convert("RGB").resize((W, H), Image.LANCZOS)
    arr = np.array(img, dtype=np.float64) / 255.0

    # Luminance
    lum = 0.299 * arr[:, :, 0] + 0.587 * arr[:, :, 1] + 0.114 * arr[:, :, 2]

    # Gradient magnitude — edge strength
    gy = np.roll(lum, -1, 0) - np.roll(lum, 1, 0)
    gx = np.roll(lum, -1, 1) - np.roll(lum, 1, 1)
    edges = np.sqrt(gx**2 + gy**2)
    edges = np.clip(edges / max(edges.max(), 0.01), 0, 1)

    # Habitat partitioning: prey at edges, specialist in interiors,
    # generalist in bright areas
    u_seed = edges * 1.5 + 0.2
    v1_seed = (1.0 - edges) * 1.2 + 0.1
    v2_seed = lum * 1.2 + 0.1

    return u_seed, v1_seed, v2_seed, lum


# ── Constants ──

DT = 0.15
SUBSTEPS = 2
ALPHA = 1.2       # prey birth rate
BETA_PREY_SP = 0.6   # predation by specialist
BETA_PREY_GN = 0.4   # predation by generalist
DELTA_SP = 0.5    # specialist growth from prey
DELTA_GN = 0.4    # generalist growth from prey
GAMMA_SP = 0.7    # specialist death rate
GAMMA_GN = 0.5    # generalist death rate
EPSILON = 0.3     # intraguild predation (generalist eats specialist)
DU = 0.12         # prey diffusion
DV1 = 0.25        # specialist diffusion
DV2 = 0.35        # generalist diffusion (fastest — wide ranging)


def _laplacian_5pt(field: np.ndarray) -> np.ndarray:
    """5-point Laplacian stencil (pure NumPy, periodic)."""
    return (np.roll(field, 1, 0) + np.roll(field, -1, 0) +
            np.roll(field, 1, 1) + np.roll(field, -1, 1) - 4 * field)


def _render_3species(u: np.ndarray, v1: np.ndarray,
                     v2: np.ndarray) -> Image.Image:
    """Render 3-species: green=prey, red=specialist, blue=generalist."""
    u_disp = np.clip(u / max(u.max(), 0.01), 0, 1)
    v1_disp = np.clip(v1 / max(v1.max(), 0.01), 0, 1)
    v2_disp = np.clip(v2 / max(v2.max(), 0.01), 0, 1)

    h, w = u.shape
    arr = np.zeros((h, w, 3), dtype=np.uint8)

    # Prey = green channel
    arr[:, :, 1] = (u_disp * 180).astype(np.uint8)

    # Specialist = red channel
    arr[:, :, 0] = (v1_disp * 200).astype(np.uint8)

    # Generalist = blue channel
    arr[:, :, 2] = (v2_disp * 220).astype(np.uint8)

    # Interfaces: overlap areas get mixed colors
    # Prey+specialist → yellow (R+G)
    ps = np.sqrt(u_disp * v1_disp)
    arr[:, :, 0] = np.clip(arr[:, :, 0] + (ps * 80).astype(np.uint8), 0, 255)
    arr[:, :, 1] = np.clip(arr[:, :, 1] + (ps * 80).astype(np.uint8), 0, 255)

    # Specialist+generalist → magenta (R+B)
    sg = np.sqrt(v1_disp * v2_disp)
    arr[:, :, 0] = np.clip(arr[:, :, 0] + (sg * 55).astype(np.uint8), 0, 255)
    arr[:, :, 2] = np.clip(arr[:, :, 2] + (sg * 55).astype(np.uint8), 0, 255)

    # All three → white
    all3 = np.cbrt(u_disp * v1_disp * v2_disp)
    arr = np.clip(arr + (all3 * 80)[:, :, np.newaxis].astype(np.uint8), 0, 255)

    return Image.fromarray(arr, mode="RGB")


def _render_species_channel(u: np.ndarray, v1: np.ndarray,
                            v2: np.ndarray, channel: str) -> Image.Image:
    """Render single species in its color against dark bg."""
    h, w = u.shape
    arr = np.zeros((h, w, 3), dtype=np.uint8)
    if channel == "prey":
        val = np.clip(u / max(u.max(), 0.01), 0, 1)
        arr[:, :, 1] = (val * 255).astype(np.uint8)
    elif channel == "specialist":
        val = np.clip(v1 / max(v1.max(), 0.01), 0, 1)
        arr[:, :, 0] = (val * 255).astype(np.uint8)
    elif channel == "generalist":
        val = np.clip(v2 / max(v2.max(), 0.01), 0, 1)
        arr[:, :, 2] = (val * 255).astype(np.uint8)
    return Image.fromarray(arr, mode="RGB")


@method(
    id="120",
    name="LV 3-Species Food Web",
    description="LV 3-Species Food Web — simulations node.",
    category="simulations",
    tags=["physics", "reaction-diffusion", "ecological", "multi-species", "expanded"],
    timeout=180,
    outputs={"image": "IMAGE", "luminance": "SCALAR", "field": "FIELD"},
    params={
        "dt": {"description": "timestep",
               "min": 0.01, "max": 1.0, "default": 0.15},
        "n_frames": {"description": "simulation frames",
                     "min": 50, "max": 600, "default": 300},
        "noise_amp": {"description": "initial noise amplitude",
                      "min": 0.01, "max": 0.5, "default": 0.15},
        "render_style": {"description": "render style",
                         "choices": ["composite", "prey", "specialist", "generalist"],
                         "default": "composite"},"anim_mode": {"description": "animation / initial condition mode",
                      "choices": ["none", "evolve", "wave", "mosaic"],
                      "default": "none"},
        "anim_speed": {"description": "animation speed multiplier",
                       "min": 0.1, "max": 5.0, "default": 1.0},
        "input_image": {"description": "path to image for seeding initial fields (R→specialist, G→prey, B→generalist)",
                        "default": ""},
    }
)
def method_lv_3species(out_dir: Path, seed: int, params=None):
    """LV 3-Species Food Web — multi-colored predator-prey front dynamics.

    Three-species Lotka-Volterra with intraguild predation: prey (green),
    specialist predator (red, eats only prey), and generalist predator
    (blue, eats both prey and the specialist). Produces shifting territories,
    dominance fronts, and rich multi-colored wave dynamics.

    Physics:
        ∂u/∂t  = αu - β₁uv₁ - β₂uv₂ + Du∇²u   (prey)
        ∂v₁/∂t = δ₁uv₁ - γ₁v₁ - εv₁v₂ + Dv₁∇²v₁ (specialist)
        ∂v₂/∂t = δ₂uv₂ + εv₁v₂ - γ₂v₂ + Dv₂∇²v₂ (generalist)

    Where ε is intraguild predation.

    Animation modes:
        none: static snapshot
        evolve: uniform noise → species differentiation → front dynamics
        wave: prey-seeded wave front, species stratify behind
        mosaic: isolated clusters of each species, territorial competition

    Architecture A — internal simulation loop with capture_frame().
    """
    # ── Params ──
    if params is None:
        params = {}
    t = float(params.get("time", 0.0))
    anim_mode = str(params.get("anim_mode", "none"))
    anim_speed = float(params.get("anim_speed", 1.0))

    dt = float(params.get("dt", DT))
    n_frames = int(params.get("n_frames", 300))
    noise_amp = float(params.get("noise_amp", 0.15))
    render_style = str(params.get("render_style", "composite"))
    input_image = str(params.get("input_image", ""))

    seed_all(seed)
    rng = np.random.default_rng(seed)

    evolve_modes = {"evolve", "wave", "mosaic"}
    is_evolve = anim_mode in evolve_modes or t > 0.01

    substeps = SUBSTEPS

    # ── Initialize fields ──
    if input_image and Path(input_image).exists():
        # Edge-seeded: edges→prey, interiors→specialist, bright→generalist
        u_seed, v1_seed, v2_seed, mod_mask = _load_image_seed_3sp(input_image)
        u = u_seed.astype(np.float64)
        v1 = v1_seed.astype(np.float64)
        v2 = v2_seed.astype(np.float64)
        u  += noise_amp * 0.3 * rng.normal(0, 1, (H, W))
        v1 += noise_amp * 0.3 * rng.normal(0, 1, (H, W))
        v2 += noise_amp * 0.3 * rng.normal(0, 1, (H, W))

    elif anim_mode == "wave":
        # Wave front: prey sweep from left, species stratify behind
        u = np.zeros((H, W), dtype=np.float64)
        v1 = np.zeros((H, W), dtype=np.float64)
        v2 = np.zeros((H, W), dtype=np.float64)

        # Prey wave at left edge
        xx = np.arange(W)[np.newaxis, :]
        yy = np.arange(H)[:, np.newaxis]
        wave_front = 0.5 * (1 - np.tanh((xx - 20) / 8))
        u += wave_front * 0.8

        # Add some noise
        u += noise_amp * 0.3 * rng.uniform(-1, 1, (H, W))
        # Specialist seed behind prey
        v1[:, 10:40] = 0.3 * rng.uniform(0.5, 1.0, (H, 30))
        # Generalist seed further back
        v2[:, :20] = 0.2 * rng.uniform(0.5, 1.0, (H, 20))

    elif anim_mode == "mosaic":
        # Isolated clusters of each species
        u = noise_amp * rng.uniform(0, 1, (H, W)).astype(np.float64)
        v1 = noise_amp * rng.uniform(0, 1, (H, W)).astype(np.float64)
        v2 = noise_amp * rng.uniform(0, 1, (H, W)).astype(np.float64)

        # Place species clusters
        for species, amp, radius in [(0, 0.8, 20), (1, 0.8, 25), (2, 0.8, 15)]:
            for _ in range(12):
                sx = int(rng.uniform(40, W - 40))
                sy = int(rng.uniform(40, H - 40))
                yy, xx_grid = np.ogrid[:H, :W]
                dist = np.sqrt((xx_grid - sx)**2 + (yy - sy)**2)
                gauss = amp * np.exp(-dist**2 / (radius**2))
                if species == 0:
                    u += gauss
                elif species == 1:
                    v1 += gauss
                else:
                    v2 += gauss

    else:
        # evolve: uniform noise, all three species intermingled
        # Smooth spatial heterogeneities for natural patches
        raw = rng.normal(0, 1, (H, W))
        kx = np.fft.fftfreq(W) * 2.0 * math.pi
        ky = np.fft.fftfreq(H) * 2.0 * math.pi
        k2 = kx[np.newaxis, :]**2 + ky[:, np.newaxis]**2
        filt = np.exp(-k2 * 12.0)
        smooth = np.real(np.fft.ifft2(np.fft.fft2(raw) * filt))
        smooth = smooth / max(np.std(smooth), 0.01)

        u = (0.5 + 0.3 * smooth).clip(0.05, 1.5).astype(np.float64)
        v1 = (0.3 + 0.25 * smooth).clip(0.0, 1.0).astype(np.float64)
        v2 = (0.2 + 0.2 * smooth).clip(0.0, 1.0).astype(np.float64)

    img = None

    # ══════════════════════════════════════════
    #  SIMULATION LOOP
    # ══════════════════════════════════════════
    for frame in range(n_frames):
        # Continuous image modulation: brightness → local prey birth rate
        if input_image and Path(input_image).exists():
            mod = 0.5 + 0.5 * mod_mask  # [0.5, 1.0] range
            cur_alpha = ALPHA * mod
            # Slight specialist boost in darker areas
            cur_beta_sp = BETA_PREY_SP * (0.7 + 0.3 * (1.0 - mod_mask))
        else:
            cur_alpha = ALPHA
            cur_beta_sp = BETA_PREY_SP

        for _ in range(substeps):
            lap_u = _laplacian_5pt(u)
            lap_v1 = _laplacian_5pt(v1)
            lap_v2 = _laplacian_5pt(v2)

            u  += dt * (cur_alpha * u - cur_beta_sp * u * v1
                        - BETA_PREY_GN * u * v2 + DU * lap_u)
            v1 += dt * (DELTA_SP * u * v1 - GAMMA_SP * v1
                        - EPSILON * v1 * v2 + DV1 * lap_v1)
            v2 += dt * (DELTA_GN * u * v2 + EPSILON * v1 * v2
                        - GAMMA_GN * v2 + DV2 * lap_v2)

            u = np.clip(u, 0, None)
            v1 = np.clip(v1, 0, None)
            v2 = np.clip(v2, 0, None)

        # ── Render ──
        if render_style == "prey":
            canvas = _render_species_channel(u, v1, v2, "prey")
        elif render_style == "specialist":
            canvas = _render_species_channel(u, v1, v2, "specialist")
        elif render_style == "generalist":
            canvas = _render_species_channel(u, v1, v2, "generalist")
        else:
            canvas = _render_3species(u, v1, v2)

        if frame % 3 == 0:
            canvas = canvas.filter(ImageFilter.GaussianBlur(radius=0.4))

        img = canvas

        if is_evolve:
            capture_frame("120", np.array(img, dtype=np.float32) / 255.0)

    # ── Final ──
    if img is None:
        from PIL import Image as PILImage
        img = PILImage.new("RGB", (W, H), (5, 5, 18))

    capture_frame("120", np.array(img, dtype=np.float32) / 255.0)
    write_field(out_dir, u.astype(np.float32))
    save(img, mn(120, "LV 3-Species Food Web"), out_dir)
    return img
