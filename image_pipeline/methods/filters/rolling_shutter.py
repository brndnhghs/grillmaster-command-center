from __future__ import annotations
import math
from pathlib import Path

import numpy as np
from scipy.ndimage import uniform_filter

from ...core.registry import method
from ...core.utils import (
    save, norm, mn, seed_all, W, H, PALETTES, load_input, write_scalars,
)
from ...core.animation import capture_frame


@method(
    id="430",
    name="Rolling Shutter",
    category="filters",
    new_image_contract=True,
    tags=["distortion", "photography", "rolling-shutter", "jello", "warp", "expanded", "animation"],
    inputs={"image_in": "IMAGE"},
    outputs={"image": "IMAGE"},
    params={
        "source": {"description": "source when no image is wired (checkerboard/grid/gradient/noise/rainbow/palette)", "default": "checkerboard"},
        "skew": {"description": "translation shear amount (fraction of width the last scanned row is shifted)", "min": -1.0, "max": 1.0, "default": 0.3},
        "direction": {"description": "scan axis the shear runs along (horizontal/vertical)", "choices": ["horizontal", "vertical"], "default": "horizontal"},
        "wobble": {"description": "jello bend amplitude (rows bow by a sinusoid that grows down the scan)", "min": 0.0, "max": 0.6, "default": 0.15},
        "wobble_freq": {"description": "spatial frequency of the jello bend across the frame", "min": 0.5, "max": 10.0, "default": 3.0},
        "noise_amp": {"description": "noise amplitude for the noise source", "min": 0.1, "max": 1.0, "default": 0.6},
        "blur_sigma": {"description": "gaussian blur sigma for the noise source", "min": 5, "max": 80, "default": 30},
        "palette": {"description": "palette name for the palette source", "default": "vapor"},
        "anim_mode": {"description": "animation mode (none/pan/wobble/both)", "choices": ["none", "pan", "wobble", "both"], "default": "none"},
        "anim_speed": {"description": "animation speed multiplier", "min": 0.1, "max": 5.0, "default": 1.0},
    },
)
def method_rolling_shutter(out_dir: Path, seed: int, params=None):
    """Rolling Shutter — the CMOS scan-line artifact that shears a moving frame.

    A rolling-shutter sensor does not capture the whole frame at once; it scans
    top-to-bottom (or left-to-right), exposing each row at a slightly later
    instant. If the camera or the scene moves during that scan, row ``y`` is
    captured ``tau(y) = y/H`` of the way through the exposure and is therefore
    shifted relative to row 0. The result is the familiar "leaning" of vertical
    edges on a fast pan and the "jello"/"wobble" of a vibrating camera — the
    bend you see on propeller blades and guitar strings in phone video.

    This node reproduces it as a closed-form, per-row remap. For a horizontal
    scan the horizontal sample coordinate of output row ``y`` is shifted by

        dx(y) = skew * tau(y) * W

    (so the bottom row is displaced by ``skew`` widths while the top is untouched
    — a pure linear shear). A second, optional "jello" term bows each row by a
    sinusoid whose phase advances down the scan and scales with the row's
    horizontal position, modelling a camera that rotates/vibrates about an axis
    during the read-out:

        dy(y, x) = wobble * sin(2*pi*wobble_freq*tau(y) + phase) * (x/W - 0.5) * H

    Both terms are sampled with bilinear interpolation so the warp stays smooth
    (no aliased cusps). Animation modes drive the effect over the timeline:

        none  — static warp at the base skew/wobble (a still "phone photo")
        pan   — skew = base * sin(t): camera pans left/right, edges lean back and forth
        wobble— wobble = base * sin(t): the jello bend breathes in and out
        both  — pan and wobble combined

    A wired upstream image always overrides source generation (Rule #12). The
    CPU path is the authoritative export; a checkerboard default makes the shear
    obvious even with no input.
    """
    try:
        if params is None:
            params = {}
        anim_time = float(params.get("time", 0.0))
        anim_mode = str(params.get("anim_mode", "none"))
        anim_speed = float(params.get("anim_speed", 1.0))
        seed_all(seed)
        rng = np.random.default_rng(seed)

        source = str(params.get("source", "checkerboard"))
        skew = float(params.get("skew", 0.3))
        skew = max(-1.0, min(1.0, skew))
        direction = str(params.get("direction", "horizontal"))
        wobble = float(params.get("wobble", 0.15))
        wobble = max(0.0, min(0.6, wobble))
        wobble_freq = float(params.get("wobble_freq", 3.0))
        wobble_freq = max(0.5, min(10.0, wobble_freq))
        noise_amp = float(params.get("noise_amp", 0.6))
        blur_sigma = float(params.get("blur_sigma", 30))
        pal_name = str(params.get("palette", "vapor"))

        # ── Animation (rename t to avoid shadowing the time param) ──
        _t = anim_time * anim_speed
        skew_eff = skew
        wobble_eff = wobble
        phase = 0.0
        if anim_mode == "pan":
            skew_eff = skew * math.sin(_t)
            phase = 0.0
        elif anim_mode == "wobble":
            wobble_eff = wobble * math.sin(_t)
            phase = _t
        elif anim_mode == "both":
            skew_eff = skew * math.sin(_t)
            wobble_eff = wobble * math.sin(_t)
            phase = _t
        # else: none — static warp at base skew/wobble, phase fixed at 0

        # ── Resolve source image (float32 [0,1], H×W×3) ──
        # A wired upstream image always overrides source generation (Rule #12).
        src = None
        wired_path = params.get("input_image", "")
        if wired_path:
            try:
                src = load_input(wired_path, int(W), int(H))
            except (FileNotFoundError, OSError):
                src = None

        if src is None:
            if source == "gradient":
                yy, xx = np.mgrid[:H, :W].astype(np.float32)
                r = norm(np.sqrt((xx - W / 2) ** 2 + (yy - H / 2) ** 2))
                src = np.stack([r, r * 0.7, 1 - r], axis=-1).clip(0, 1)
            elif source == "rainbow":
                yy, xx = np.mgrid[:H, :W].astype(np.float32)
                r = norm(np.sqrt((xx - W / 2) ** 2 + (yy - H / 2) ** 2))
                hue = r * 2 * math.pi
                src = np.stack([
                    np.sin(hue) * 0.5 + 0.5,
                    np.sin(hue + 2.094) * 0.5 + 0.5,
                    np.sin(hue + 4.189) * 0.5 + 0.5,
                ], axis=-1).astype(np.float32)
            elif source == "palette":
                pal = PALETTES.get(pal_name, list(PALETTES.values())[0])
                yy, xx = np.mgrid[:H, :W].astype(np.float32)
                r = norm(np.sqrt((xx - W / 2) ** 2 + (yy - H / 2) ** 2))
                idx = (r * (len(pal) - 1)).astype(np.int32)
                src = np.array(pal, dtype=np.float32)[idx] / 255.0
            elif source == "noise":
                n = rng.standard_normal((H, W, 3)).astype(np.float32) * noise_amp + 0.5
                n = uniform_filter(n, size=max(3, int(blur_sigma)), mode="reflect")
                src = norm(n)
            elif source == "grid":
                cs = max(10, int(min(W, H) / 24))
                yy, xx = np.mgrid[:H, :W]
                g = ((xx % cs == 0) | (yy % cs == 0)).astype(np.float32)
                src = np.stack([1 - g, 1 - g, 1 - g], axis=-1).astype(np.float32)
            else:  # checkerboard (default — makes the shear obvious)
                cs = max(8, int(min(W, H) / 16))
                yy, xx = np.mgrid[:H, :W]
                chk = (((xx // cs) + (yy // cs)) % 2).astype(np.float32)
                src = np.stack([chk, chk, chk], axis=-1).astype(np.float32)

        src = np.clip(src, 0.0, 1.0).astype(np.float32)
        out = _warp(src, skew_eff, wobble_eff, direction, wobble_freq, phase)

        capture_frame("430", out)
        save(out, mn(430, "Rolling Shutter"), out_dir)
        try:
            write_scalars(out_dir, skew=float(skew_eff), wobble=float(wobble_eff),
                          wobble_freq=float(wobble_freq))
        except Exception:
            pass
        return out
    except Exception as exc:
        fallback = np.full((H, W, 3), 0.5, dtype=np.float32)
        save(fallback, mn(430, "Rolling Shutter"), out_dir)
        print(f"[method_430] ERROR: {exc}")
        return fallback


def _warp(src, skew_eff, wobble_eff, direction, wobble_freq, phase):
    """Bilinear per-row remap implementing the rolling-shutter shear + jello bend."""
    H, W, _ = src.shape
    out = np.empty_like(src)

    if direction == "vertical":
        # columns shift vertically by d(x); the scan runs left->right
        xx = np.arange(W, dtype=np.float64)
        tau = xx / max(1, W - 1)
        dy = skew_eff * tau * H                         # vertical shift (px)
        yn = np.arange(H, dtype=np.float64) / max(1, H - 1) - 0.5
        wave = np.sin(2 * math.pi * wobble_freq * tau[None, :] + phase)  # (1, W)
        dx = wobble_eff * wave * yn[:, None] * W        # (H, W) horizontal bow
        src_y = yn[:, None] * 0 + np.arange(H)[:, None] + dy[None, :]
        src_x = np.arange(W)[None, :] + dx
    else:
        # rows shift horizontally by d(y); the scan runs top->bottom (default)
        yy = np.arange(H, dtype=np.float64)
        tau = yy / max(1, H - 1)
        dx = skew_eff * tau * W                         # horizontal shift (px)
        xn = np.arange(W, dtype=np.float64) / max(1, W - 1) - 0.5
        wave = np.sin(2 * math.pi * wobble_freq * tau[:, None] + phase)  # (H, 1)
        dy = wobble_eff * wave * xn[None, :] * H        # (H, W) vertical bow
        src_x = np.arange(W)[None, :] + dx[:, None]
        src_y = np.arange(H)[:, None] + dy

    src_x = np.clip(src_x, 0.0, W - 1)
    src_y = np.clip(src_y, 0.0, H - 1)
    x0 = np.floor(src_x).astype(np.int32)
    x1 = np.minimum(x0 + 1, W - 1)
    fx = (src_x - x0).astype(np.float32)
    y0 = np.floor(src_y).astype(np.int32)
    y1 = np.minimum(y0 + 1, H - 1)
    fy = (src_y - y0).astype(np.float32)

    for c in range(3):
        v00 = src[y0, x0, c]
        v01 = src[y0, x1, c]
        v10 = src[y1, x0, c]
        v11 = src[y1, x1, c]
        top = v00 * (1.0 - fx) + v01 * fx
        bot = v10 * (1.0 - fx) + v11 * fx
        out[:, :, c] = top * (1.0 - fy) + bot * fy

    return np.clip(out, 0.0, 1.0).astype(np.float32)
