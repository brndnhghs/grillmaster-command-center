"""#525 — VHS Analog Tape Degradation (chroma smear + tracking + sync roll)

Faithful emulation of a consumer VHS / analog tape playback, reproducing the
signatures that make tape footage read as "analog":

  1. **Chroma smear + delay** -- VHS records chrominance at quarter
     resolution and the chroma "bleeds" horizontally with a delay line, so
     colour edges lag and fringe. We blur Cb/Cr with a wide *horizontal-only*
     Gaussian and shift them by slightly different delays (classic colour
     fringing). Reference: the SMPTE/Rec.601 chroma-subsampling model and
     the "chroma delay line" of VHS (see the VHS FAQ /archive.org docs).
  2. **Line jitter / tracking error** -- the tape head loses lock, so whole
     scanlines are displaced vertically by small per-row offsets (the wobble
     you see on a mistracked tape). Animated modes evolve these offsets.
  3. **Rolling sync bar + head-switching noise** -- a soft brightness
     band rolls down the frame (vertical hold drift) and the bottom few
     scanlines carry the characteristic head-switching "tear" noise.
  4. **Luma snow + dropout sparkle** -- tape hiss adds gaussian grain and
     sparse bright impulse dropouts.
  5. **Tape wobble / skew** -- the whole frame shears horizontally with a
     slow sine (the image bends as the tape stretches).

The effect is genuinely *structured* (chroma fringe + scanline jitter are
strong spatial features) and, in any animated mode, *strongly temporal*
(jitter evolves, the sync bar rolls, the skew breathes) -- so it survives
the liveness cull and is cheap O(W*H) (a couple of vectorised
remaps + separable blurs), never hitting the 150 s render-budget timeout.

Source: this is the authoritative CPU render/export path. When an upstream
image is wired in it is ALWAYS used (Rule #12); otherwise a vivid seeded
scene (color bars / night lights / gradient / checkerboard / noise) is
synthesised so the effect is obvious.

Animation (Architecture B, per-frame re-call): the source scene is built
once from the seed (stable across frames -- no strobing) while the chroma
delay, jitter, sync-roll and skew evolve, so only the tape artifacts move.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
from PIL import Image

from ...core.registry import method
from ...core.utils import (
    save, mn, seed_all, W, H, PALETTES, load_input, write_scalars,
)
from ...core.animation import capture_frame

import cv2
from image_pipeline.core.spatial import sparam


# ── Procedural self-contained source scene ──

def _make_scene(source: str, rng: np.random.Generator, pal_name: str) -> np.ndarray:
    """Return an HxWx3 float32 [0,1] scene so the VHS look is visible unwired."""
    Ww, Hh = int(W), int(H)
    if source == "color_bars":
        bars = np.array([
            [1.0, 1.0, 1.0], [1.0, 1.0, 0.0], [0.0, 1.0, 1.0],
            [0.0, 1.0, 0.0], [1.0, 0.0, 1.0], [1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0], [0.7, 0.7, 0.7],
        ], dtype=np.float32)
        n = len(bars)
        cw = max(1, Ww // n)
        canvas = np.zeros((Hh, Ww, 3), dtype=np.float32)
        for i, col in enumerate(bars):
            x0 = i * cw
            x1 = (i + 1) * cw if i < n - 1 else Ww
            canvas[:, x0:x1] = col
        canvas[int(Hh * 0.86):, :] *= 0.0
        canvas[int(Hh * 0.86):, :] += 0.12
        return canvas.astype(np.float32)
    if source == "gradient":
        xx = np.linspace(0, 1, Ww, dtype=np.float32)[None, :]
        yy = np.linspace(0, 1, Hh, dtype=np.float32)[:, None]
        g = (xx * 0.5 + yy * 0.5)
        return np.stack([g, g * 0.7, g * 0.4], axis=-1).astype(np.float32)
    if source == "checkerboard":
        c = 24
        xx = (np.arange(Ww) // c)
        yy = (np.arange(Hh) // c)
        ch = ((xx[None, :] + yy[:, None]) % 2).astype(np.float32)
        return np.stack([ch, ch * 0.6, ch * 0.9], axis=-1).astype(np.float32)
    if source == "noise":
        return rng.random((Hh, Ww, 3)).astype(np.float32)
    # night_lights (default): dark field with colourful point lights -> glow the
    # chroma smear + jitter clearly modulate.
    canvas = np.zeros((Hh, Ww, 3), dtype=np.float32)
    n_lights = int(rng.integers(70, 130))
    pal = PALETTES.get(pal_name, PALETTES["vapor"])
    pal_arr = np.array(pal, dtype=np.float32) / 255.0
    for _ in range(n_lights):
        x = int(rng.integers(0, Ww))
        y = int(rng.integers(0, Hh))
        rad = float(rng.uniform(3.0, 9.0))
        col = pal_arr[int(rng.integers(0, len(pal_arr)))]
        span = 24
        yyA, xxA = np.mgrid[max(0, y - span):min(Hh, y + span + 1),
                            max(0, x - span):min(Ww, x + span + 1)].astype(np.float32)
        d = np.hypot(xxA - x, yyA - y)
        glow = np.clip(1.0 - d / (rad * 3.0), 0, 1) ** 2
        core = np.clip(1.0 - d / rad, 0, 1) ** 1.5 * 1.5
        sy0 = max(0, y - span)
        sx0 = max(0, x - span)
        canvas[sy0:sy0 + glow.shape[0], sx0:sx0 + glow.shape[1]] += (
            col[None, None, :] * (glow + core)[:, :, None]
        )
    canvas += 0.04
    return np.clip(canvas, 0.0, 1.0).astype(np.float32)


# ── Colour-space helpers (manual RGB<->YCbCr, no BGR confusion) ──

def _rgb_to_ycbcr(rgb: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    y = 0.299 * r + 0.587 * g + 0.114 * b
    cb = -0.168736 * r - 0.331264 * g + 0.5 * b + 0.5
    cr = 0.5 * r - 0.418688 * g - 0.081312 * b + 0.5
    return y, cb, cr


def _ycbcr_to_rgb(y: np.ndarray, cb: np.ndarray, cr: np.ndarray) -> np.ndarray:
    r = y + 1.402 * (cr - 0.5)
    g = y - 0.344136 * (cb - 0.5) - 0.714136 * (cr - 0.5)
    b = y + 1.772 * (cb - 0.5)
    return np.stack([r, g, b], axis=-1).astype(np.float32)


def _gauss_h(ch: np.ndarray, sigma: float) -> np.ndarray:
    """Horizontal-only Gaussian blur (separable, ksize (kw,1))."""
    if sigma <= 0.0:
        return ch
    kw = max(3, 2 * int(3.0 * sigma) + 1)
    return cv2.GaussianBlur(ch, (kw, 1), sigmaX=sigma, sigmaY=0.0)


# ── Method ──

@method(
    id="527",
    name="VHS Tape",
    category="filters",
    new_image_contract=True,
    tags=["post-process", "vhs", "analog", "tape", "retro", "chroma",
          "tracking", "skew", "animation", "expanded"],
    inputs={"image_in": "IMAGE"},
    outputs={"image": "IMAGE", "luminance": "FIELD"},
    params={
        "source": {"description": "procedural scene when no image is wired (color_bars/night_lights/gradient/checkerboard/noise)",
                   "default": "color_bars"},
        "chroma_smear": {"description": "chroma horizontal blur / smear amount (0 = sharp colour)",
                         "min": 0.0, "max": 1.0, "default": 0.55},
        "chroma_shift": {"description": "chroma delay-line shift in px (colour fringing)",
                         "min": 0.0, "max": 24.0, "default": 8.0},
        "luma_noise": {"description": "tape snow / luma grain amount",
                       "min": 0.0, "max": 0.6, "default": 0.12},
        "line_jitter": {"description": "vertical scanline tracking jitter amount (px)",
                        "min": 0.0, "max": 1.0, "default": 0.45},
        "tracking": {"description": "head-switching noise band intensity (0 = none)",
                     "min": 0.0, "max": 1.0, "default": 0.5},
        "roll_speed": {"description": "vertical sync-roll / band animation rate",
                      "min": 0.0, "max": 3.0, "default": 1.0},
        "skew": {"description": "tape wobble / horizontal shear amount",
                  "min": 0.0, "max": 1.0, "default": 0.35},
        "saturation": {"spatial": True, "description": "chroma saturation boost (1 = unchanged)",
                       "min": 0.0, "max": 2.0, "default": 1.25},
        "contrast": {"description": "overall contrast",
                     "min": 0.3, "max": 2.0, "default": 1.10},
        "brightness": {"description": "output brightness gain",
                       "min": 0.4, "max": 2.0, "default": 1.05},
        "palette": {"description": "palette for the night_lights source", "default": "vapor"},
        "anim_mode": {"description": "none / tracking (jitter+tracking band) / roll (sync bar) / warp (skew shear) / flow (all)",
                      "choices": ["none", "tracking", "roll", "warp", "flow"], "default": "none"},
        "anim_speed": {"description": "animation speed multiplier",
                       "min": 0.1, "max": 5.0, "default": 1.0},
        "time": {"description": "animation phase [0, 2pi)",
                 "min": 0.0, "max": 6.28, "default": 0.0},
    },
)
def method_vhs(out_dir: Path, seed: int, params=None):
    """VHS Analog Tape Degradation -- chroma smear + tracking jitter + rolling sync + tape wobble (node 525).

    Emulates a consumer VHS playback: a horizontal chroma smear with a delay
    line (colour fringing), per-scanline tracking jitter, a soft rolling sync
    bar with head-switching noise, luma snow + dropout sparkles, and a slow
    tape-wobble shear.

    Architecture B -- per-frame re-call via ``time``; the source scene is
    built once from the seed (stable across frames) so only the tape artifacts
    evolve. The CPU path is authoritative.
    """
    try:
        if params is None:
            params = {}
        anim_time = float(params.get("time", 0.0))
        anim_mode = str(params.get("anim_mode", "none"))
        anim_speed = float(params.get("anim_speed", 1.0))
        seed_all(seed)
        rng = np.random.default_rng(seed)

        Ww, Hh = int(W), int(H)
        source = str(params.get("source", "color_bars"))
        chroma_smear = float(params.get("chroma_smear", 0.55))
        chroma_shift = float(params.get("chroma_shift", 8.0))
        luma_noise = float(params.get("luma_noise", 0.12))
        line_jitter = float(params.get("line_jitter", 0.45))
        tracking = float(params.get("tracking", 0.5))
        roll_speed = float(params.get("roll_speed", 1.0))
        skew = float(params.get("skew", 0.35))
        saturation = sparam(params, "saturation", 1.25)
        contrast = float(params.get("contrast", 1.10))
        brightness = float(params.get("brightness", 1.05))
        pal_name = str(params.get("palette", "vapor"))

        # ── Animation wiring (rename t; never shadow the time param) ──
        _t = anim_time * anim_speed
        animated = anim_mode != "none"
        jitter_phase = 0.0
        skew_phase = 0.0
        roll_pos = 0.85      # static band rest position for 'none'
        if animated:
            if anim_mode in ("tracking", "flow"):
                jitter_phase = _t
            if anim_mode in ("roll", "flow"):
                roll_pos = (_t * roll_speed * 0.15) % 1.0
            if anim_mode in ("warp", "flow"):
                skew_phase = _t

        # ── Resolve source image (Rule #12: wired image overrides) ──
        src = None
        wired_path = params.get("input_image", "")
        if wired_path:
            try:
                src = load_input(wired_path, Ww, Hh)
            except (FileNotFoundError, OSError):
                src = None
        if src is None:
            src = _make_scene(source, rng, pal_name)
        src = np.clip(src, 0.0, 1.0).astype(np.float32)

        # ── Chroma smear + delay line (YCbCr space) ──
        y, cb, cr = _rgb_to_ycbcr(src)
        cb = _gauss_h(cb, chroma_smear * 9.0)
        cr = _gauss_h(cr, chroma_smear * 9.0)
        # Delay line: Cb/Cr lag by slightly different pixel counts -> fringing.
        sh_b = int(round(chroma_shift))
        sh_r = int(round(chroma_shift * 0.6))
        if sh_b:
            cb = np.roll(cb, sh_b, axis=1)
        if sh_r:
            cr = np.roll(cr, sh_r, axis=1)
        # Saturation (operate around the 0.5 chroma centre).
        cb = np.clip(0.5 + (cb - 0.5) * saturation, 0.0, 1.0)
        cr = np.clip(0.5 + (cr - 0.5) * saturation, 0.0, 1.0)
        rgb = _ycbcr_to_rgb(y, cb, cr)
        rgb = np.clip(rgb, 0.0, 1.0)

        # ── Geometric remap: skew shear (x) + line jitter (y) ──
        yv = np.arange(Hh, dtype=np.float32)
        xv = np.arange(Ww, dtype=np.float32)
        # Per-row static jitter base (seed-stable) + optional temporal evolution.
        jb = rng.random(Hh) * 2.0 - 1.0
        jit_base = jb * line_jitter * 6.0
        if anim_mode in ("tracking", "flow"):
            jit_base = jit_base + line_jitter * 4.0 * np.sin(yv * 0.15 + jitter_phase)
        # Per-row offset applies to every column -> broadcast to (H,W), float32.
        map_y = np.broadcast_to((yv + jit_base)[:, None], (Hh, Ww)).astype(np.float32).copy()
        # Skew: horizontal shear that breathes with _t in warp/flow.
        skew_px = float(skew) * 18.0
        shear = (skew_px * np.sin(yv * 0.025 + skew_phase)).astype(np.float32)  # (H,)
        map_x = (xv[None, :] + shear[:, None]).astype(np.float32)              # (H,W)
        src_u8 = np.clip(rgb * 255.0, 0, 255).astype(np.uint8)
        warped = np.empty_like(src_u8)
        for c in range(3):
            warped[..., c] = cv2.remap(
                src_u8[..., c], map_x, map_y,
                cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE,
            )
        rgb = warped.astype(np.float32) / 255.0

        # ── Head-switching tracking band (rolls in roll/flow) ──
        if tracking > 0.0:
            band_h = 0.12
            band_rows = np.where(np.abs(yv / Hh - roll_pos) < band_h)[0]
            if band_rows.size:
                # Desaturate + heavy grain inside the band.
                lum = rgb[band_rows].mean(axis=-1, keepdims=True)
                mix = tracking * 0.7
                rgb[band_rows] = rgb[band_rows] * (1.0 - mix) + lum * mix
                gn = (rng.random(rgb[band_rows].shape).astype(np.float32) - 0.5) * tracking * 0.9
                rgb[band_rows] = np.clip(rgb[band_rows] + gn, 0.0, 1.0)
                # Horizontal tear: shift each band row by a random amount.
                for yb in band_rows.tolist():
                    rgb[yb] = np.roll(rgb[yb], int(rng.integers(-6, 7)), axis=0)

        # ── Luma snow + dropout sparkles ──
        if luma_noise > 0.0:
            snow = (rng.random((Hh, Ww, 3)).astype(np.float32) - 0.5) * luma_noise
            rgb = np.clip(rgb + snow, 0.0, 1.0)
            n_spark = int(luma_noise * 220.0)
            if n_spark > 0:
                sy = rng.integers(0, Hh, n_spark)
                sx = rng.integers(0, Ww, n_spark)
                rgb[sy, sx] = np.clip(
                    rgb[sy, sx] + rng.random((n_spark, 3)).astype(np.float32),
                    0.0, 1.0,
                )

        # ── Vertical-hold roll (classic VHS loss of vertical sync) ──
        # The whole picture scrolls; strongly temporal and authenticaly "rolls".
        if anim_mode in ("roll", "flow"):
            vscroll = int(roll_pos * Hh) % Hh
            if vscroll:
                rgb = np.roll(rgb, vscroll, axis=0)
        # ── Rolling sync bar (brightness ripple, rolls in roll/flow) ──
        if anim_mode in ("roll", "flow"):
            dy = (yv / Hh) - roll_pos
            band = np.exp(-(dy * dy) / (2.0 * 0.03 * 0.03)) * 0.10
            rgb = np.clip(rgb + band[:, None, None], 0.0, 1.0)

        # ── Contrast + brightness ──
        out = np.clip((rgb - 0.5) * contrast + 0.5, 0.0, 1.0)
        out = np.clip(out * brightness, 0.0, 1.0).astype(np.float32)

        write_scalars(out_dir, chroma_smear=float(chroma_smear),
                      line_jitter=float(line_jitter), tracking=float(tracking),
                      skew=float(skew))
        capture_frame("527", out)
        save(out, mn(527, f"VHS Tape t={_t:.2f}"), out_dir)
        return out
    except Exception as exc:
        fallback = np.full((int(H), int(W), 3), 0.5, dtype=np.float32)
        save(fallback, mn(527, "VHS Tape"), out_dir)
        print(f"[method_527] ERROR: {exc}")
        return fallback
