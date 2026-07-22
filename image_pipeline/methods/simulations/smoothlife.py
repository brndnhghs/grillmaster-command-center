"""#560 — SmoothLife (continuous Game of Life)

Stephan Rafler, "Generalization of Conway's Game of Life to a continuous
domain — SmoothLife" (2011). A continuous-valued cellular automaton: each
cell holds a state in [0,1] (an "aliveness" field) and is born/survised by
comparing two floating-point neighbourhood averages (inner/outer rings) to
smooth birth/survival intervals. Produces organic, amoeba-like growth that
is structurally alive at every frame.

Reference: https://arxiv.org/abs/1111.1567

Architecture B (stateless, one call = one frame). The simulation evolves
the fixed state to the generation dictated by the animation clock `t`
(time-reveal), so a static frame is still meaningful and the live path
plays the whole emergent history. The birth/survival/death thresholds are
SCALAR-wireable so a wired driver (LFO, noise, ramp) can morph the organism
live — closing the driver→pixel loop that the diagnostics showed
was the dominant dead-node hotspot.
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
from scipy.ndimage import uniform_filter

from ...core.registry import method
from ...core.animation import capture_frame
from ...core.utils import (
    save, mn, W, H, write_field, write_scalars, seed_all, BG_DEFAULT,
)


# ─── SmoothLife core (Rafler 2011) ───────────────────────────────────────────

INNER_R = 6.0   # outer radius of the inner (filling) neighbourhood
OUTER_R = 12.0  # outer radius of the outer (emptying) neighbourhood
EPS = 0.10      # smooth-transition width (b / ε)

# Canonical "SmoothLifeL" parameters (Rafler). These are the defaults; each
# is overridable live via SCALAR wiring.
B1, B2 = 0.278, 0.365   # birth interval (outer avg)
D1, D2 = 0.267, 0.445   # death interval (outer avg)
S1, S2 = 0.100, 0.380   # survival interval (outer avg)


def _ring_filter(state: np.ndarray, r_out: float, r_in: float) -> np.ndarray:
    """Smoothed neighbourhood average over an annulus (r_in .. r_out).

    Implemented as the difference of two disc averages (uniform_filter with a
    diameter ~ 2*r), which is O(N) regardless of radius — cheap even at large
    canvas. Periodic boundary gives toroidal continuity (no edge artefacts).
    """
    d_out = max(1, int(round(2.0 * r_out)))
    d_in = max(1, int(round(2.0 * r_in)))
    outer = uniform_filter(state, size=d_out, mode="wrap")
    inner = uniform_filter(state, size=d_in, mode="wrap")
    # annulus average = (disc_outer - disc_inner) / (area_outer - area_inner)
    a_out = math.pi * r_out * r_out
    a_in = math.pi * r_in * r_in
    return (outer * a_out - inner * a_in) / max(a_out - a_in, 1e-6)


def _smooth_step(x: np.ndarray) -> np.ndarray:
    x = np.clip(x, 0.0, 1.0)
    return x * x * (3.0 - 2.0 * x)


def _smooth_transition(val: np.ndarray, lo: float, hi: float, eps: float) -> np.ndarray:
    """Smooth 0→1 step centred on [lo, hi] with width ϵ (Rafler's s_equation)."""
    if hi <= lo:
        return np.zeros_like(val)
    a = _smooth_step((val - lo) / eps)
    b = _smooth_step((val - hi) / eps)
    return a * (1.0 - b)


def _smoothlife_step(
    state: np.ndarray,
    b1: float, b2: float, d1: float, d2: float, s1: float, s2: float,
    eps: float,
) -> np.ndarray:
    """One SmoothLife update (continuous birth / survival / death)."""
    n_outer = _ring_filter(state, OUTER_R, INNER_R)
    n_inner = _ring_filter(state, INNER_R, 0.0)

    # Transition functions (Rafler):
    #   b = living → born   (outer avg in birth band)
    #   s = living → survive (outer avg in survival band)
    #   d = living → die    (1 - (outer avg in death band))
    b = _smooth_transition(n_outer, b1, b2, eps)
    s = _smooth_transition(n_outer, s1, s2, eps)
    d = 1.0 - _smooth_transition(n_outer, d1, d2, eps)

    # New state: survival of the living + birth of the empty.
    new_state = state * (s + (1.0 - s) * d) + (1.0 - state) * b * n_inner
    return np.clip(new_state, 0.0, 1.0)


# ─── Initialization ──────────────────────────────────────────────────────────

def _random_init(h: int, w: int, density: float, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    state = rng.random((h, w)) < density
    return state.astype(np.float32)


def _disc_init(h: int, w: int, seed: int, radius_frac: float = 0.20) -> np.ndarray:
    # A few soft blobs to give organic, non-grid origins.
    rng = np.random.default_rng(seed)
    state = np.zeros((h, w), dtype=np.float32)
    n = 5
    cx, cy = w / 2.0, h / 2.0
    for _ in range(n):
        ox = (rng.random() - 0.5) * w * 0.5
        oy = (rng.random() - 0.5) * h * 0.5
        r = max(3.0, radius_frac * min(w, h) * rng.random())
        yy, xx = np.mgrid[0:h, 0:w]
        d2 = (xx - (cx + ox)) ** 2 + (yy - (cy + oy)) ** 2
        blob = np.exp(-d2 / (2.0 * r * r))
        state = np.clip(state + blob, 0.0, 1.0)
    return state


# ─── Rendering ───────────────────────────────────────────────────────────────

def _render(state: np.ndarray, color_mode: str, hue_shift: float) -> np.ndarray:
    img = np.zeros((H, W, 3), dtype=np.float32)
    s = state
    if color_mode == "heat":
        img[:, :, 0] = np.clip(s * 2.0, 0, 1)
        img[:, :, 1] = np.clip(s * 2.0 - 1.0, 0, 1)
        img[:, :, 2] = np.clip(s * 3.0 - 2.0, 0, 1)
    elif color_mode == "rainbow":
        hv = (s * 0.6 + hue_shift) % 1.0
        img[:, :, 0] = 0.5 + 0.5 * np.sin(hv * 2 * np.pi)
        img[:, :, 1] = 0.5 + 0.5 * np.sin(hv * 2 * np.pi + 2.094)
        img[:, :, 2] = 0.5 + 0.5 * np.sin(hv * 2 * np.pi + 4.189)
    elif color_mode == "cyan":
        img[:, :, 1] = s
        img[:, :, 2] = np.clip(s * 0.7 + 0.0, 0, 1)
    elif color_mode == "amber":
        img[:, :, 0] = np.clip(s * 1.2, 0, 1)
        img[:, :, 1] = np.clip(s * 0.8, 0, 1)
        img[:, :, 2] = s * 0.15
    else:  # mono
        img[:, :, :] = s[:, :, np.newaxis]
    return np.clip(img, 0.0, 1.0)


# ─── The Method (Architecture B — stateless, one call = one frame) ───────────

@method(id="560", name="SmoothLife (Continuous Life)", category="simulations",
        tags=["smoothlife", "continuous", "cellular", "life", "animation", "organic"],
        inputs={
            "death": "SCALAR",
            "birth": "SCALAR",
            "survive": "SCALAR",
            "hue_shift": "SCALAR",
        },
        outputs={"image": "IMAGE", "luminance": "FIELD"},
        params={
            "init": {"description": "initial pattern",
                     "choices": ["random", "blobs"], "default": "blobs"},
            "density": {"description": "initial fill density (random init)", "min": 0.02, "max": 0.50, "default": 0.10},
            "color_mode": {"description": "coloring scheme",
                           "choices": ["mono", "heat", "rainbow", "cyan", "amber"], "default": "mono"},
            "generations": {"description": "number of SmoothLife steps across the full timeline", "min": 20, "max": 300, "default": 140},
            "eps": {"description": "smooth-transition width b/ε (smaller = sharper edges)", "min": 0.02, "max": 0.30, "default": 0.10},
            "death": {"description": "outer-average death band CENTRE (SCALAR-wireable). Live driver can morph the organism.", "min": 0.20, "max": 0.60, "default": 0.356},
            "birth": {"description": "outer-average birth band CENTRE (SCALAR-wireable).", "min": 0.20, "max": 0.60, "default": 0.322},
            "survive": {"description": "outer-average survival band CENTRE (SCALAR-wireable).", "min": 0.10, "max": 0.50, "default": 0.240},
            "hue_shift": {"description": "hue shift for rainbow color mode (0-1)", "min": 0.0, "max": 1.0, "default": 0.0},
        })
def method_560_smoothlife(out_dir: Path, seed: int, params=None):
    """Continuous Game-of-Life (SmoothLife, Rafler 2011).

    Architecture B (stateless, one call = one frame). The simulation runs to
    the generation dictated by the animation clock `t` (time-reveal), so the
    live path plays the full emergent history and a still frame is meaningful.
    The birth/survival/death thresholds are SCALAR-wireable so a driver can
    evolve the organism live.
    """
    if params is None:
        params = {}
    t = float(params.get("time", 0.0))

    # ── SCALAR-driven params (override UI params when wired) ──
    death_ov = params.get("death")
    birth_ov = params.get("birth")
    survive_ov = params.get("survive")
    hue_ov = params.get("hue_shift")

    # Each SCALAR is normalized [0,1] by the graph executor; map to band centre.
    death_c = float(death_ov) if death_ov is not None else float(params.get("death", 0.356))
    birth_c = float(birth_ov) if birth_ov is not None else float(params.get("birth", 0.322))
    survive_c = float(survive_ov) if survive_ov is not None else float(params.get("survive", 0.240))
    hue_shift = float(hue_ov) if hue_ov is not None else float(params.get("hue_shift", 0.0))

    color_mode = params.get("color_mode", "mono")
    init = params.get("init", "blobs")
    density = float(params.get("density", 0.10))
    max_gens = int(params.get("generations", 140))
    eps = float(params.get("eps", 0.10))

    # Birth/survival/death bands (each centred on its SCALAR-driven value).
    b1, b2 = birth_c - 0.045, birth_c + 0.045
    d1, d2 = death_c - 0.089, death_c + 0.089
    s1, s2 = survive_c - 0.070, survive_c + 0.070

    # Freeze seed so the animation is purely clock/driver driven (no seed drift).
    seed = seed & 0xFFFF0000
    seed_all(seed)
    rng = np.random.default_rng(seed)

    # ── Initialize ──
    if init == "random":
        state = _random_init(H, W, density, seed)
    else:
        state = _disc_init(H, W, seed)

    # ── Evolve to the generation dictated by the clock (time-reveal) ──
    frac = (t / (2.0 * math.pi)) % 1.0
    n_total = int(frac * max_gens)
    for _ in range(n_total):
        state = _smoothlife_step(state, b1, b2, d1, d2, s1, s2, eps)

    # ── Render + emit ──
    img = _render(state, color_mode, hue_shift)
    mask = state  # alive/unborn selection (float32 [0,1])
    lum = state  # FIELD = the aliveness field itself

    capture_frame("560", img)
    save(img, mn(560, f"SmoothLife t={t:.2f}"), out_dir)
    write_field(out_dir, lum)                       # Rule 5
    write_scalars(out_dir, generation=n_total, alive_fraction=float(state.mean()),
                  birth_c=birth_c, death_c=death_c, survive_c=survive_c)  # Rule 4
    return {"image": img, "luminance": lum, "mask": mask}
