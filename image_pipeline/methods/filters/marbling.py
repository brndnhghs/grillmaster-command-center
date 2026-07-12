"""Computer-Generated Marbling (suminagashi / Ebru).

A vectorized, closed-form implementation of the mathematical marbling model
of Aubrey Jaffer (2006, "Mathematical Marbling") built on the earlier
conformal-map formulation of Lu, Jaffer & co. (SIGGRAPH 2004, "Real-Time
Digital Sumi-e").

Core idea: marbling is a *conformal deformation* of a 2D color field.
Tines (rakes) dragged across the bath are modeled as logarithmic-potential
comb maps; ink drops are concentric circles injected into the field *before*
the tine pass, so the tine stretch turns the rings into the classic
feathered marbled veins. "Ragged" tines (Jaffer's organic augmentation)
modulate the comb strength per column with a smooth hart noise so the
streaks are not mechanically uniform.

Because the deformation is fully differentiable/closed-form, the same code
path renders a still (anim_mode="none") and an Architecture-B animation:
the tine comb translates with the animation phase `_t`, sweeping the veins
across the bath.

References
  - Jaffer, "Mathematical Marbling", 2006.
  - Lu, Jaffer, et al., "Real-Time Digital Sumi-e", SIGGRAPH 2004.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image
from scipy.ndimage import map_coordinates

from ...core.registry import method
from ...core.utils import (
    H,
    W,
    load_input,
    mn,
    save,
    seed_all,
    write_field,
    write_scalars,
)
from ...core.animation import capture_frame

# ── Palettes (RGB 0-255) ──────────────────────────────────────────
_PALETTES: dict[str, list[tuple[int, int, int]]] = {
    "jewel": [
        (20, 24, 82), (38, 70, 160), (0, 140, 150),
        (232, 90, 60), (244, 196, 48), (255, 255, 255),
    ],
    "pastel": [
        (244, 234, 222), (222, 196, 214), (180, 210, 222),
        (196, 222, 188), (246, 214, 168), (126, 154, 184),
    ],
    "mono": [
        (10, 14, 38), (24, 52, 92), (40, 96, 130),
        (120, 170, 186), (224, 232, 238), (255, 255, 255),
    ],
    "sunset": [
        (28, 12, 48), (92, 26, 92), (196, 54, 96),
        (240, 110, 70), (250, 186, 96), (252, 240, 196),
    ],
}


def _hsv_to_rgb(h: float, s: float, v: float) -> tuple[int, int, int]:
    h = h % 1.0
    i = int(h * 6.0)
    f = h * 6.0 - i
    p = v * (1.0 - s)
    q = v * (1.0 - f * s)
    t = v * (1.0 - (1.0 - f) * s)
    r, g, b = [
        (v, t, p), (q, v, p), (p, v, t),
        (p, q, v), (t, p, v), (v, p, q),
    ][i % 6]
    return (int(r * 255), int(g * 255), int(b * 255))


def _pick_palette(name: str, rng: np.random.Generator) -> list[tuple[int, int, int]]:
    if name == "random":
        return [
            _hsv_to_rgb(rng.random(), 0.45 + 0.45 * rng.random(), 0.7 + 0.3 * rng.random())
            for _ in range(6)
        ]
    return list(_PALETTES.get(name, _PALETTES["jewel"]))


def _build_background(bg_mode: str, palette: list, rng: np.random.Generator) -> np.ndarray:
    """Return an (H,W,3) uint8 base color field."""
    if bg_mode == "solid":
        base = palette[0]
        return np.full((H, W, 3), base, dtype=np.uint8)
    # vertical gradient between two palette colors
    top = np.asarray(palette[-1], dtype=np.float32)
    bot = np.asarray(palette[0], dtype=np.float32)
    yy = np.linspace(0.0, 1.0, H, dtype=np.float32)[:, None]
    grad = top[None, None, :] * (1.0 - yy) + bot[None, None, :] * yy
    return grad.astype(np.uint8)


def _inject_drops(field: np.ndarray, n_drops: int, rings: int,
                  ring_gap: float, palette: list, rng: np.random.Generator) -> None:
    """Inject concentric ink rings at jittered-drop centers (rings drawn large→small)."""
    h, w = field.shape[:2]
    cols = np.arange(w, dtype=np.float32)[None, :]
    rows = np.arange(h, dtype=np.float32)[:, None]
    # jittered grid of drop centers
    n_side = max(1, int(np.ceil(np.sqrt(n_drops))))
    xs = np.linspace(w * 0.12, w * 0.88, n_side)
    ys = np.linspace(h * 0.12, h * 0.88, n_side)
    centers = []
    for gy in ys:
        for gx in xs:
            if len(centers) >= n_drops:
                break
            jx = gx + rng.uniform(-w * 0.06, w * 0.06)
            jy = gy + rng.uniform(-h * 0.06, h * 0.06)
            centers.append((jx, jy))
    for ci, (cx, cy) in enumerate(centers):
        dist = np.sqrt((cols - cx) ** 2 + (rows - cy) ** 2)
        # draw rings from outer to inner so inner sits on top
        for k in range(rings, 0, -1):
            r_out = k * ring_gap
            r_in = max(0.0, r_out - ring_gap * 0.62)
            color = palette[(k + ci) % len(palette)]
            mask = (dist <= r_out) & (dist > r_in)
            field[mask] = color


def _tine_warp(field: np.ndarray, n_tines: int, strength: float,
                spacing: float, ragged: float, t_offset: float,
                rng: np.random.Generator):
    """Apply the logarithmic-potential comb map. Returns (warped, disp_mag)."""
    h, w = field.shape[:2]
    yy, xx = np.meshgrid(np.arange(h, dtype=np.float32),
                          np.arange(w, dtype=np.float32), indexing="ij")
    offs = (np.arange(n_tines, dtype=np.float32) - (n_tines - 1) * 0.5) * spacing + t_offset
    K = strength * spacing * 0.02
    dx = np.zeros((h, w), dtype=np.float32)
    dy = np.zeros((h, w), dtype=np.float32)
    eps = 1e-3 * (w + h)
    for tx in offs:
        d = (tx - xx) ** 2 + yy ** 2 + eps
        dx += (tx - xx) / d
        dy += yy / d
    # ragged: smooth per-column modulation of the comb strength AND
    # per-column phase wobble of each tine (organic, non-mechanical streaks)
    col = np.arange(w, dtype=np.float32)
    if ragged > 1e-3:
        amp = np.sin(col * 0.045 + rng.uniform(0, 6.28)) \
            + 0.5 * np.sin(col * 0.013 + rng.uniform(0, 6.28))
        mod = 1.0 + ragged * 0.35 * amp
        dx *= mod[None, :]
        # phase wobble: shift each tine's center column-wise
        phase = ragged * 6.0 * np.sin(col * 0.03 + rng.uniform(0, 6.28))
        dx += phase[None, :] * 0.01 * spacing
    dx *= K
    dy *= K
    xw = np.clip(xx + dx, 0.0, w - 1.0)
    yw = np.clip(yy + dy, 0.0, h - 1.0)
    out = np.empty((h, w, 3), dtype=np.float32)
    for c in range(3):
        out[..., c] = map_coordinates(
            field[..., c].astype(np.float32), [yw, xw], order=1, mode="reflect"
        )
    disp = np.hypot(dx, dy)
    return out.clip(0, 255).astype(np.uint8), disp


@method(
    id="437",
    name="Marbling (Sumi-e)",
    category="filters",
    tags=["marbling", "suminagashi", "ebru", "deformation", "npr", "expanded"],
    timeout=120,
    inputs={"image_in": "IMAGE"},
    outputs={"image": "IMAGE", "field": "FIELD"},
    params={
        "bg_mode": {
            "description": "base bath color (gradient, solid, or use a wired input image as the bath)",
            "choices": ["gradient", "solid", "input"],
            "default": "gradient",
        },
        "ink_palette": {
            "description": "ink color family for the dropped rings",
            "choices": ["jewel", "pastel", "mono", "sunset", "random"],
            "default": "jewel",
        },
        "n_drops": {
            "description": "number of concentric-ring drop centers",
            "min": 1, "max": 6, "default": 2,
        },
        "rings": {
            "description": "concentric ink rings per drop center",
            "min": 3, "max": 28, "default": 12,
        },
        "ring_gap": {
            "description": "spacing between rings (pixels)",
            "min": 5, "max": 40, "default": 16,
        },
        "n_tines": {
            "description": "rake tines (comb teeth) dragged across the bath",
            "min": 2, "max": 48, "default": 18,
        },
        "tine_strength": {
            "description": "comb displacement strength (higher = longer veins)",
            "min": 0.2, "max": 6.0, "default": 2.0,
        },
        "tine_spacing": {
            "description": "distance between tines (pixels)",
            "min": 8, "max": 60, "default": 28,
        },
        "ragged": {
            "description": "ragged-tine organic edge modulation (0=mechanical, 1=feathered)",
            "min": 0.0, "max": 1.0, "default": 0.5,
        },
        "anim_speed": {
            "description": "animation speed multiplier (comb sweep per phase)",
            "min": 0.1, "max": 3.0, "default": 1.0,
        },
    },
)
def method_marbling(out_dir: Path, seed: int, params=None):
    """Computer-Generated Marbling — conformal-map ink deformation.

    Drops concentric ink rings into a bath, then rakes them with a
    logarithmic-potential tine comb to produce the classic feathered
    suminagashi / ebru veining. Accepts an optional wired IMAGE as the
    bath (the color field that gets deformed); otherwise builds a
    gradient/solid bath from the chosen ink palette.

    Args:
        out_dir: Output directory
        seed: Random seed
        params: Parameter overrides dict
    """
    if params is None:
        params = {}
    bg_mode = str(params.get("bg_mode", "gradient"))
    ink_palette = str(params.get("ink_palette", "jewel"))
    n_drops = int(params.get("n_drops", 2))
    rings = int(params.get("rings", 12))
    ring_gap = float(params.get("ring_gap", 16))
    n_tines = int(params.get("n_tines", 18))
    tine_strength = float(params.get("tine_strength", 2.0))
    tine_spacing = float(params.get("tine_spacing", 28))
    ragged = float(params.get("ragged", 0.5))
    anim_speed = float(params.get("anim_speed", 1.0))
    t = float(params.get("time", 0.0))

    seed_all(seed)
    rng = np.random.default_rng(seed)
    _t = t * anim_speed

    # ── Build bath (color field) ──
    palette = _pick_palette(ink_palette, rng)
    wired = params.get("input_image", "")
    if bg_mode == "input" and wired:
        try:
            arr = load_input(wired, W, H)  # float32 [0,1], (H,W,3)
            field = (np.clip(arr, 0.0, 1.0) * 255).astype(np.uint8)
        except (FileNotFoundError, OSError, ValueError):
            field = _build_background("gradient", palette, rng)
    else:
        field = _build_background(bg_mode if bg_mode != "input" else "gradient", palette, rng)

    # ── Inject ink rings (before the tine pass) ──
    _inject_drops(field, n_drops, rings, ring_gap, palette, rng)

    # ── Tine comb sweep (animation phase translates the comb) ──
    t_offset = _t * tine_spacing * 0.5
    warped, disp = _tine_warp(
        field, n_tines, tine_strength, tine_spacing, ragged, t_offset, rng
    )

    # ── Emit ──
    result = warped  # full-coverage RGB
    save(result, mn(437, f"Marbling t={_t:.2f}"), out_dir)
    write_field(out_dir, disp.astype(np.float32))
    lum = float(np.mean(result) / 255.0)
    write_scalars(out_dir, luminance=lum, n_tines=float(n_tines),
                  ring_count=float(n_drops * rings))
    capture_frame("437", result)
    return result
