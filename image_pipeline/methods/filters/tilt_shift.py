"""#515 — Tilt-Shift (selective-focus / miniature / fake-diorama)

Tilt-shift is the photographic post-processing trick that fakes a
miniature/diorama look by keeping a *band* of the image in sharp focus
while heavily blurring everything else — mimicking the shallow depth of
field of a macro photograph of a small model. The effect is driven by a
focus mask: a smooth ramp from "sharp" (inside the focus band) to "blurred"
(outside it). We blur the whole frame with a Gaussian and blend it with
the sharp source under that mask.

Two focus geometries are supported:
  * linear  — a horizontal sharp band (classic diorama look, horizon-based)
  * radial  — a sharp disc/ring centred on the frame (fake macro of a subject)

The mask, blur radius, and band position are all live controls, and the
animation modes (drift / breathe) sweep them over time, so the node is a
good cheap, always-moving recombination seed. A wired upstream
IMAGE overrides the procedural source.

Reference: the "tilt-shift miniature" effect is a standard computational-
photography post-process (see e.g. the Scheimpflug / shallow-DoF miniature
faking tutorials); this is the focus-mask formulation used by most
real-time implementations.
"""
from __future__ import annotations

from pathlib import Path

import math

import numpy as np
from scipy.ndimage import gaussian_filter

from ...core.registry import method
from ...core.utils import save, mn, seed_all, W, H, write_scalars, wired_source_rgb
from ...core.animation import capture_frame


# ══════════════════════════════════════════════════════════════════════════
#  Procedural tonal sources (used only when nothing is wired in)
# ══════════════════════════════════════════════════════════════════════════

def _gray2rgb(lum: np.ndarray) -> np.ndarray:
    return np.stack([lum, lum, lum], axis=-1).astype(np.float32)


def _perlin(H: int, W: int, rng: np.random.Generator) -> np.ndarray:
    field = np.zeros((H, W), dtype=np.float32)
    amp = 1.0
    tot = 0.0
    for o in range(4):
        sigma = max(2.0, 2.0 ** (o + 2))
        n = rng.random((H, W)).astype(np.float32)
        n = gaussian_filter(n, sigma=sigma)
        n = (n - n.min()) / (n.max() - n.min() + 1e-8)
        field += amp * n
        tot += amp
        amp *= 0.5
    return field / max(tot, 1e-8)


def _generate_source(source: str, H: int, W: int, rng: np.random.Generator) -> np.ndarray:
    if source == "gradient":
        yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
        r = np.sqrt((xx - W / 2) ** 2 + (yy - H / 2) ** 2)
        return _gray2rgb(r / max(r.max(), 1e-8))
    if source == "checker":
        yy, xx = np.mgrid[0:H, 0:W]
        c = ((xx // max(1, W // 8)) + (yy // max(1, H // 8))) % 2
        return _gray2rgb(c.astype(np.float32))
    if source == "noisy":
        yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
        rr = np.sqrt((xx - W / 2) ** 2 + (yy - H / 2) ** 2)
        r = rr / max(rr.max(), 1e-8)
        base = 0.5 + 0.5 * np.sin(r * 18.0) * np.cos(xx * 0.03)
        noise = rng.standard_normal((H, W)).astype(np.float32) * 0.18
        return _gray2rgb(np.clip(base + noise, 0.0, 1.0))
    # default: perlin
    return _gray2rgb(_perlin(H, W, rng))


# ══════════════════════════════════════════════════════════════════════════
#  Focus mask + blend
# ══════════════════════════════════════════════════════════════════════════

def _smoothstep(a: float, b: float, x: np.ndarray) -> np.ndarray:
    t = np.clip((x - a) / (b - a + 1e-8), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def _focus_mask(H: int, W: int, mode: str, center: float, width: float) -> np.ndarray:
    """Return (H,W) float mask in [0,1]: 1 = sharp (in focus), 0 = blurred."""
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    if mode == "radial":
        cy, cx = H / 2.0, W / 2.0
        d = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
        d = d / (np.hypot(H, W) / 2.0 + 1e-8)  # ~0..1
        band = np.abs(d - center)
    else:  # linear: band centred on a horizontal line at `center`
        fy = center * H
        band = np.abs(yy - fy) / H
    inner = max(width * 0.4, 1e-3)
    outer = max(width, inner + 1e-3)
    return 1.0 - _smoothstep(inner, outer, band)


def _render(src: np.ndarray, mode: str, center: float, width: float,
            blur_sigma: float) -> np.ndarray:
    Hn, Wn = src.shape[:2]
    blurred = np.empty_like(src)
    for c in range(src.shape[2]):
        blurred[..., c] = gaussian_filter(src[..., c], sigma=blur_sigma)
    mask = _focus_mask(Hn, Wn, mode, center, width)[..., None]
    return (src * mask + blurred * (1.0 - mask)).astype(np.float32)


# ══════════════════════════════════════════════════════════════════════════
#  Main method
# ══════════════════════════════════════════════════════════════════════════

@method(
    id="515",
    name="Tilt-Shift",
    category="filters",
    new_image_contract=True,
    tags=["post-processing", "tilt-shift", "miniature", "selective-focus", "dof", "photographic"],
    inputs={"image_in": "IMAGE"},
    outputs={"image": "IMAGE"},
    params={
        "source": {
            "description": "tonal source when nothing is wired (perlin/gradient/checker/noisy)",
            "default": "noisy",
        },
        "focus_mode": {
            "description": "focus geometry",
            "choices": ["linear", "radial"],
            "default": "linear",
        },
        "focus_center": {
            "description": "focus band centre (0-1): line height (linear) / disc radius (radial)",
            "min": 0.0,
            "max": 1.0,
            "default": 0.5,
        },
        "focus_width": {
            "description": "sharp band half-width (0-1)",
            "min": 0.02,
            "max": 1.0,
            "default": 0.18,
        },
        "blur_sigma": {
            "description": "defocus blur radius in px",
            "min": 0.5,
            "max": 30.0,
            "default": 8.0,
        },
        "anim_mode": {
            "description": "animation mode (none / drift / breathe)",
            "choices": ["none", "drift", "breathe"],
            "default": "none",
        },
        "anim_speed": {
            "description": "animation speed multiplier",
            "min": 0.1,
            "max": 5.0,
            "default": 1.0,
        },
        "time": {
            "description": "animation time in radians",
            "min": 0.0,
            "max": 6.2832,
            "default": 0.0,
        },
    },
)
def method_tilt_shift(out_dir: Path, seed: int, params=None):
    """Tilt-Shift — selective-focus miniature / fake-diorama post-process.

    Keeps a band of the image sharp while Gaussian-blurring the rest, faking
    the shallow depth of field of a macro photograph of a small model. The
    focus region is a smooth mask (linear band or radial disc) blended between
    the sharp source and its blurred copy.

    Args:
        out_dir: Output directory for the generated image.
        seed: Random seed for deterministic output.
        params: Dict with keys:
            source:       procedural tonality when unwired (perlin/gradient/checker/noisy)
            focus_mode:   linear (horizontal band) / radial (centred disc)
            focus_center: band centre (line height for linear, disc radius for radial)
            focus_width:  sharp band half-width (0-1)
            blur_sigma:   defocus blur radius in px
            anim_mode:    none / drift (band sweeps) / breathe (blur pulses)
            anim_speed:   animation speed multiplier
            time:         animation time (radians)
    """
    if params is None:
        params = {}

    seed_all(seed)
    rng = np.random.default_rng(seed)

    Hn, Wn = int(H), int(W)

    source = str(params.get("source", "noisy"))
    focus_mode = str(params.get("focus_mode", "linear"))
    focus_center = float(params.get("focus_center", 0.5))
    focus_width = float(params.get("focus_width", 0.18))
    blur_sigma = float(params.get("blur_sigma", 8.0))
    anim_mode = str(params.get("anim_mode", "none"))
    anim_speed = float(params.get("anim_speed", 1.0))
    # `time` is injected by the orchestrator as the animation phase (0..2π);
    # Architecture B: render ONE frame per call, single capture_frame().
    t = float(params.get("time", 0.0))
    _t = t * anim_speed

    # ── Source image (wired upstream overrides procedural generation) ──
    wired = wired_source_rgb(params, Hn, Wn)
    if wired is not None and wired.size > 0:
        src = np.asarray(wired, dtype=np.float32)[..., :3]
        src = src.reshape(Hn, Wn, 3)
    else:
        src = _generate_source(source, Hn, Wn, rng)
    src = src.clip(0.0, 1.0)

    # ── Animation modes modulate the focus band / blur radius over time ──
    if anim_mode == "drift":
        center_eff = 0.5 + 0.45 * math.sin(_t)          # sharp band sweeps vertically
        width_eff = focus_width
        blur_eff = blur_sigma
    elif anim_mode == "breathe":
        s = 0.5 + 0.5 * math.sin(_t)                      # 0..1 over a half-cycle
        center_eff = focus_center
        width_eff = focus_width * (0.7 + 0.6 * s)         # sharp band breathes
        blur_eff = blur_sigma * (1.0 + math.sin(_t))      # DoF pulse: 0×..2× base defocus
    else:  # none — static baseline
        center_eff = focus_center
        width_eff = focus_width
        blur_eff = blur_sigma

    result = _render(src, focus_mode, center_eff, width_eff, blur_eff)
    result = result.clip(0.0, 1.0).astype(np.float32)
    capture_frame("515", result)

    # Scalar readouts for the node graph sidecar
    mask = _focus_mask(Hn, Wn, focus_mode, focus_center, focus_width)
    write_scalars(out_dir, focus_fraction=float(mask.mean()), blur_sigma=blur_sigma)

    save(result, mn(515, f"Tilt-Shift {focus_mode}"), out_dir)
    return result
