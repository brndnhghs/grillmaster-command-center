from __future__ import annotations

import math
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from ...core.registry import method
from ...core.utils import save, mn, seed_all, W, H, load_input, write_scalars
from ...core.animation import capture_frame


# Standard CMYK screen angles (degrees). Per-channel rotation is what gives
# the technique its signature rosette / moiré-free look — each ink is sampled
# on a grid rotated by a different angle so the dots never align.
_SCREEN_ANGLES = {
    "c": 15.0,
    "m": 75.0,
    "y": 0.0,
    "k": 45.0,
}


def _rgb_to_cmyk(rgb: np.ndarray) -> dict[str, np.ndarray]:
    """Vectorized RGB -> CMYK in [0,1]. rgb is (H,W,3) float32 in [0,1]."""
    r = rgb[..., 0].astype(np.float64)
    g = rgb[..., 1].astype(np.float64)
    b = rgb[..., 2].astype(np.float64)
    k = 1.0 - np.maximum(np.maximum(r, g), b)
    inv = 1.0 - k
    safe = np.where(inv > 1e-6, inv, 1.0)
    c = (1.0 - r - k) / safe
    m = (1.0 - g - k) / safe
    y = (1.0 - b - k) / safe
    c = np.clip(c, 0.0, 1.0)
    m = np.clip(m, 0.0, 1.0)
    y = np.clip(y, 0.0, 1.0)
    return {"c": c, "m": m, "y": y, "k": k}


def _halftone_channel(intensity: np.ndarray, spacing: float, max_dot: float,
                       angle_deg: float, tx: float = 0.0, ty: float = 0.0) -> np.ndarray:
    """Return per-pixel ink coverage [0,1] for one rotated dot screen.

    The image plane is rotated by ``angle_deg``; dots live on a square grid of
    period ``spacing`` in that rotated frame. Dot radius scales with
    sqrt(intensity) of the cell it belongs to (constant per cell -> true
    halftone), and a smoothstep gives anti-aliased edges. ``tx``/``ty`` shift
    the screen origin so an animation can pan the dots (translation breaks the
    lattice's rotational symmetry, which pure rotation at 180 deg does not).
    """
    Hh, Ww = intensity.shape
    yy, xx = np.mgrid[0:Hh, 0:Ww].astype(np.float64)
    theta = math.radians(angle_deg)
    cos_t, sin_t = math.cos(theta), math.sin(theta)
    # image -> screen (rotate sample point)
    s = xx * cos_t + yy * sin_t + tx
    t = -xx * sin_t + yy * cos_t + ty

    period = max(2.0, float(spacing))
    # cell center in rotated frame
    sc = np.round(s / period) * period
    tc = np.round(t / period) * period
    # cell center back in image space, then sample that pixel's intensity
    xc = (sc * cos_t - tc * sin_t)
    yc = (sc * sin_t + tc * cos_t)
    xi = np.clip(np.round(xc).astype(np.int64), 0, Ww - 1)
    yi = np.clip(np.round(yc).astype(np.int64), 0, Hh - 1)
    cell_i = intensity[yi, xi]

    d = np.sqrt((s - sc) ** 2 + (t - tc) ** 2)
    radius = np.sqrt(np.clip(cell_i, 0.0, 1.0)) * (period * 0.5) * max_dot
    aa = max(0.75, period * 0.12)
    # smoothstep: 1 inside the dot, 0 outside
    cov = 1.0 - np.clip((d - (radius - aa)) / (2.0 * aa), 0.0, 1.0)
    cov = np.clip(cov, 0.0, 1.0)
    return cov


def _procedural_source(Hh: int, Ww: int, seed: int) -> np.ndarray:
    """Colorful default source so a standalone render is non-black."""
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:Hh, 0:Ww].astype(np.float64)
    nx = xx / max(1, Ww - 1)
    ny = yy / max(1, Hh - 1)
    # smooth multi-blob color field
    cx = rng.uniform(0.15, 0.85, 4)
    cy = rng.uniform(0.15, 0.85, 4)
    r = rng.uniform(0.15, 0.4, 4)
    base = np.zeros((Hh, Ww, 3), dtype=np.float64)
    for ch in range(3):
        for i in range(4):
            d = ((nx - cx[i]) ** 2 + (ny - cy[i]) ** 2)
            base[..., ch] += np.exp(-d / (r[i] ** 2)) * rng.uniform(0.4, 1.0)
    # diagonal ramp adds structure
    ramp = (nx * 0.6 + ny * 0.4)
    base[..., 0] = base[..., 0] * 0.7 + ramp * 0.5
    base[..., 1] = base[..., 1] * 0.7 + (1 - ramp) * 0.5
    base[..., 2] = base[..., 2] * 0.7 + np.sin(nx * 6.0) * 0.25 + 0.5
    return np.clip(base, 0.0, 1.0).astype(np.float32)


@method(
    id="399",
    name="CMYK Halftone",
    new_image_contract=True,
    category="filters",
    tags=["halftone", "print", "cmyk", "dots", "classic", "expanded"],
    inputs={"image_in": "IMAGE"},
    outputs={"image": "IMAGE"},
    params={
        "spacing": {"description": "screen frequency / dot grid spacing (px)", "min": 2, "max": 40, "default": 8},
        "max_dot": {"description": "max dot diameter as fraction of spacing", "min": 0.3, "max": 1.4, "default": 1.0},
        "angle_offset": {"description": "rotate all screens by this many degrees", "min": -45.0, "max": 45.0, "default": 0.0},
        "ink_set": {"description": "which inks to print", "default": "cmyk", "choices": ["cmyk", "cmy", "gray", "rgb"]},
        "paper": {"description": "paper color", "default": "white", "choices": ["white", "cream", "black"]},
        "source": {"description": "source when no image is wired (input_image always wins)", "default": "procedural", "choices": ["procedural", "gradient", "checker"]},
        "anim_mode": {"description": "animation mode: none, screen_rotate, breathe", "default": "none"},
        "anim_speed": {"description": "animation speed multiplier", "min": 0.1, "max": 3.0, "default": 1.0},
        "time": {"description": "animation phase [0, 2pi)", "min": 0.0, "max": 6.28, "default": 0.0},
    },
)
def method_cmyk_halftone(out_dir: Path, seed: int, params=None):
    """Classic CMYK color-separation halftone (the print technique).

    The source image is split into C/M/Y/K channels; each channel is screened
    onto its own rotated dot grid (C 15 deg, M 75 deg, Y 0 deg, K 45 deg) and
    the resulting ink layers are recombined subtractively on paper. Per-channel
    rotation is what produces the characteristic rosette pattern and avoids
    visible moire.

    Architecture-B (closed-form per frame): the orchestrator re-calls this with
    an increasing ``time`` value; animation rotates the screens or breathes the
    dot size. A wired IMAGE input always overrides the source param.
    """
    try:
        if params is None:
            params = {}
        t = float(params.get("time", 0.0))
        seed_all(seed)

        spacing = float(params.get("spacing", 8))
        max_dot = float(params.get("max_dot", 1.0))
        angle_offset = float(params.get("angle_offset", 0.0))
        ink_set = params.get("ink_set", "cmyk")
        paper = params.get("paper", "white")
        src_mode = params.get("source", "procedural")
        anim_mode = params.get("anim_mode", "none")
        anim_speed = float(params.get("anim_speed", 1.0))

        Hh, Ww = int(H), int(W)

        # ── Wired input ALWAYS overrides source param (Architecture override) ──
        wired = params.get("input_image", "")
        if wired:
            try:
                src = load_input(wired, Ww, Hh)
            except (FileNotFoundError, OSError, ValueError):
                src = None
        else:
            src = None

        if src is None:
            if src_mode == "gradient":
                nx = np.linspace(0, 1, Ww, dtype=np.float32)[None, :, None]
                ny = np.linspace(0, 1, Hh, dtype=np.float32)[:, None, None]
                src = np.clip(np.stack([nx, ny * 0.8, (nx + ny) * 0.5], axis=-1), 0, 1).astype(np.float32)
            elif src_mode == "checker":
                yy, xx = np.mgrid[0:Hh, 0:Ww]
                c = ((xx // max(8, int(spacing))) + (yy // max(8, int(spacing)))) % 2
                src = np.stack([c, (c + 1) % 2, c * 0.5 + 0.25], axis=-1).astype(np.float32)
            else:
                src = _procedural_source(Hh, Ww, seed)

        src = np.clip(src[..., :3].astype(np.float32), 0.0, 1.0)

        # ── Animation wiring (Architecture B) ──
        # screen_rotate: pan the screens (translation breaks lattice symmetry,
        # unlike pure rotation which maps a square grid onto itself at 180deg).
        # breathe: scale dot size with cos (differs at t=0 vs t=π, unlike sin).
        sx = sy = 0.0
        if anim_mode == "screen_rotate":
            # Pan the screens by a NON-lattice-aligned fraction of the period
            # (0.37) so the grid never re-aligns between t=0 and t=π. Rotation
            # alone maps a square lattice onto itself, so translation is what
            # actually animates the rosette pattern.
            sx = math.cos(t * anim_speed) * spacing * 0.37
            sy = math.sin(t * anim_speed) * spacing * 0.37
        elif anim_mode == "breathe":
            _t = t * anim_speed
            max_dot = max_dot * (0.5 + 0.5 * math.cos(_t))
        else:
            _t = 0.0

        cmyk = _rgb_to_cmyk(src)
        c, m, y_ch, k = cmyk["c"], cmyk["m"], cmyk["y"], cmyk["k"]

        # paper background (full canvas, not a single pixel)
        if paper == "cream":
            base = np.array([0.96, 0.93, 0.84], dtype=np.float64)
        elif paper == "black":
            base = np.array([0.04, 0.04, 0.05], dtype=np.float64)
        else:
            base = np.array([1.0, 1.0, 1.0], dtype=np.float64)
        paper_rgb = np.ones((Hh, Ww, 3), dtype=np.float64) * base

        # ink absorption: each ink subtracts from the paper in its absorbed band
        # cyan absorbs RED, magenta absorbs GREEN, yellow absorbs BLUE, black all.
        def _screen(chan: str, intensity: np.ndarray) -> np.ndarray:
            ang = _SCREEN_ANGLES[chan] + angle_offset
            return _halftone_channel(intensity, spacing, max_dot, ang, tx=sx, ty=sy)

        # Compute per-channel coverages based on the chosen ink set.
        if ink_set == "rgb":
            # additive colored dots on black paper -> no black ink
            cov_c = _screen("c", c)
            cov_m = _screen("m", m)
            cov_y = _screen("y", y_ch)
            cov_k = np.zeros((Hh, Ww), dtype=np.float64)
            out = paper_rgb.copy() * 0.0 + 0.02
            out[..., 0] += cov_c * (1.0 - cov_m) * (1.0 - cov_y) * (1.0 - cov_k)
            out[..., 1] += cov_m * (1.0 - cov_c) * (1.0 - cov_y) * (1.0 - cov_k)
            out[..., 2] += cov_y * (1.0 - cov_c) * (1.0 - cov_m) * (1.0 - cov_k)
            out = np.clip(out, 0.0, 1.0)
        elif ink_set == "gray":
            # single black screen only
            cov_k = _screen("k", (1.0 - np.mean(src, axis=-1)))
            out = paper_rgb.copy()
            out *= (1.0 - cov_k[..., None])
        else:
            # "cmyk" or "cmy": subtractive inks on paper.
            # cyan absorbs RED, magenta absorbs GREEN, yellow absorbs BLUE,
            # black absorbs all bands.
            out = paper_rgb.copy()
            out[..., 0] *= (1.0 - _screen("c", c))     # cyan removes red
            out[..., 1] *= (1.0 - _screen("m", m))     # magenta removes green
            out[..., 2] *= (1.0 - _screen("y", y_ch))  # yellow removes blue
            if ink_set == "cmyk":
                out *= (1.0 - _screen("k", k))[..., None]   # black removes all

        rgb = np.clip(out, 0.0, 1.0).astype(np.float32)

        # ── scalar telemetry (Rule 4) ──
        write_scalars(out_dir, mean_cyan=float(c.mean()), mean_magenta=float(m.mean()),
                      mean_yellow=float(y_ch.mean()), mean_black=float(k.mean()))

        capture_frame("399", rgb)
        save(rgb, mn(399, "CMYK Halftone"), out_dir)
        return rgb
    except Exception as exc:
        fallback = np.full((int(H), int(W), 3), 128, dtype=np.uint8)
        save(fallback, mn(399, "CMYK Halftone"), out_dir)
        print(f"[method_399] ERROR: {exc}")
        return fallback
