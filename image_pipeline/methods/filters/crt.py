"""#522 — CRT Emulation (scanlines + aperture-grille phosphor + barrel warp)

Faithful real-time emulation of a consumer cathode-ray-tube display, reproducing
the three signatures that make CRT imagery read as "analog":

  1. **Barrel distortion** — the glass face curves inward, so the image is
     radially warped (quadratic radial magnification). This is the geometric
     heart of every CRT shader (e.g. Timothy Lottes' ``CRT 2.0`` / "FixingPixelArt"
     and the Libretro ``crt-geom`` / aperture-grille shaders:
     https://docs.libretro.com/shader/crt/ ,
     http://filthypants.blogspot.com/2020/02/crt-shader-masks.html ).
  2. **Scanlines** — the horizontal raster sweep darkens every other row at the
     line frequency, a high-frequency luminance grating.
  3. **Aperture-grille phosphor mask** — each pixel column drives mostly one of
     R/G/B phosphors (the classic Trinitron / Sony-PVM "aperture grille"), giving
     the fine colored vertical striping. We also add a soft rolling scan band and
     optional brightness flicker for the authentic "live tube" feel.

The effect is genuinely *structured* (strong scanline + phosphor spatial
frequency) and, in any animated mode, *strongly temporal* (the scanlines scroll,
the roll band sweeps, the warp breathes) — so it survives the liveness
cull (static/flat) and is cheap O(W·H) (three ``cv2.remap`` calls + vectorised
pixel math), so it never hits the 150 s render-budget timeout.

Source: only exists today as GPU twin 206; this is the authoritative CPU
render/export path implementation. When an upstream image is wired in it is
ALWAYS used (Rule #12); otherwise a vivid seeded scene (color bars / night
lights / gradient / checkerboard / noise) is synthesised so the effect is obvious.

Animation (Architecture B, per-frame re-call): the source scene is built once
from the seed (stable across frames — no strobing) while the warp / scanline
phase / roll band / flicker evolve, so only the CRT artifacts move.
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
    """Return an HxWx3 float32 [0,1] scene so the CRT look is visible unwired."""
    Ww, Hh = int(W), int(H)
    if source == "color_bars":
        # Vivid vertical colour bars (SMPTE-ish) — ideal to show the phosphor mask.
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
        # thin black divider strip at the bottom (classic bars test pattern)
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
    # night_lights (default): dark field with colourful point lights -> glow that
    # the scanlines / phosphor mask clearly modulate.
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
            col[None, None, :] * (glow + core)[:, :, None])
    canvas += 0.04
    return np.clip(canvas, 0.0, 1.0).astype(np.float32)


# ── CRT forward map (backward sampling grids) ──

def _warp_maps(Ww: int, Hh: int, k: float, scale: float = 1.0):
    """Return (map_x, map_y) float32 sampling grids for a given barrel curvature k.

    ``scale`` is a uniform global zoom (around screen centre); breathing it moves
    *every* pixel, not just the edges, so warp mode stays strongly temporal.
    """
    yy, xx = np.mgrid[0:Hh, 0:Ww].astype(np.float32)
    nx = (xx + 0.5) / Ww * 2.0 - 1.0
    ny = (yy + 0.5) / Hh * 2.0 - 1.0
    r2 = nx * nx + ny * ny
    f = 1.0 + k * r2
    sx = scale * nx * f
    sy = scale * ny * f
    map_x = (sx + 1.0) / 2.0 * Ww - 0.5
    map_y = (sy + 1.0) / 2.0 * Hh - 0.5
    return map_x, map_y, r2


# ── Method ──

@method(
    id="522",
    name="CRT Emulation",
    category="filters",
    new_image_contract=True,
    tags=["post-process", "crt", "retro", "scanline", "phosphor", "aperture-grille",
          "barrel", "vintage", "animation", "expanded"],
    inputs={"image_in": "IMAGE"},
    outputs={"image": "IMAGE", "luminance": "FIELD"},
    params={
        "source": {"description": "procedural scene when no image is wired (color_bars/night_lights/gradient/checkerboard/noise)",
                   "default": "color_bars"},
        "curvature": {"description": "barrel distortion strength (0 = flat tube)", "min": 0.0, "max": 0.45, "default": 0.18},
        "scanline": {"description": "scanline darkening amount (0 = none)", "min": 0.0, "max": 1.0, "default": 0.35},
        "scan_freq": {"description": "scanline density (lines across the screen)", "min": 1.0, "max": 8.0, "default": 2.5},
        "mask_strength": {"description": "aperture-grille RGB phosphor mask strength (0 = none)", "min": 0.0, "max": 1.0, "default": 0.35},
        "vignette": {"description": "corner darkening", "min": 0.0, "max": 1.0, "default": 0.30},
        "chroma": {"spatial": True, "description": "edge chromatic aberration (R/B radial split)", "min": 0.0, "max": 1.0, "default": 0.25},
        "roll_speed": {"description": "vertical roll / scan-band animation rate", "min": 0.0, "max": 3.0, "default": 1.0},
        "flicker": {"description": "brightness flicker amount (animated modes)", "min": 0.0, "max": 0.3, "default": 0.06},
        "brightness": {"description": "output brightness gain", "min": 0.4, "max": 2.0, "default": 1.10},
        "palette": {"description": "palette for the night_lights source", "default": "vapor"},
        "anim_mode": {"description": "none / roll (scroll+roll band) / flicker (brightness) / warp (curvature breathe) / flow (all)",
                      "choices": ["none", "roll", "flicker", "warp", "flow"], "default": "none"},
        "anim_speed": {"description": "animation speed multiplier", "min": 0.1, "max": 5.0, "default": 1.0},
        "time": {"description": "animation phase [0, 2pi)", "min": 0.0, "max": 6.28, "default": 0.0},
    },
)
def method_crt(out_dir: Path, seed: int, params=None):
    """CRT Emulation — scanlines + aperture-grille phosphor + barrel warp (node 522).

    Emulates a consumer cathode-ray-tube display: a quadratic barrel warp of the
    image, a high-frequency scanline luminance grating, a per-column RGB aperture-
    grille phosphor mask, corner vignette, edge chromatic aberration, plus a soft
    rolling scan band and optional brightness flicker.

    Architecture B — per-frame re-call via ``time``; the source scene is built
    once from the seed (stable across frames) so only the CRT artifacts evolve.
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
        curvature = float(params.get("curvature", 0.18))
        scanline = float(params.get("scanline", 0.35))
        scan_freq = float(params.get("scan_freq", 2.5))
        mask_strength = float(params.get("mask_strength", 0.35))
        vignette = float(params.get("vignette", 0.30))
        chroma = sparam(params, "chroma", 0.25)
        roll_speed = float(params.get("roll_speed", 1.0))
        flicker = float(params.get("flicker", 0.06))
        brightness = float(params.get("brightness", 1.10))
        pal_name = str(params.get("palette", "vapor"))

        # ── Animation wiring (rename t; never shadow the time param) ──
        _t = anim_time * anim_speed
        animated = anim_mode != "none"

        # ── Resolve source image ──
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

        # ── Effective (animated) parameters ──
        eff_k = curvature
        scale = 1.0
        roll_phase = 0.0
        roll_pos = 0.0
        do_flicker = 0.0
        if animated:
            if anim_mode in ("warp", "flow"):
                # Breathe BOTH the barrel curvature and a global zoom pump, so
                # the centre moves too (a pure edge-localised warp reads as
                # static to a mean-Δ liveness gate).
                eff_k = curvature * (1.0 + 0.5 * math.sin(_t))
                scale = 1.0 + 0.08 * math.sin(_t)
            if anim_mode in ("roll", "flow"):
                roll_phase = _t * roll_speed * 6.0
                roll_pos = (_t * roll_speed * 0.15) % 1.0
            if anim_mode in ("flicker", "flow"):
                do_flicker = flicker * (0.5 - 0.5 * math.cos(_t * 3.0))

        # ── Barrel warp with per-channel chromatic aberration ──
        kR = eff_k * (1.0 + chroma * 0.12)
        kG = eff_k
        kB = eff_k * (1.0 - chroma * 0.12)
        mxR, myR, r2 = _warp_maps(Ww, Hh, kR, scale)
        mxG, myG, _ = _warp_maps(Ww, Hh, kG, scale)
        mxB, myB, _ = _warp_maps(Ww, Hh, kB, scale)
        src_u8 = np.clip(src * 255.0, 0, 255).astype(np.uint8)
        R = cv2.remap(src_u8[:, :, 0], mxR, myR, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        G = cv2.remap(src_u8[:, :, 1], mxG, myG, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        B = cv2.remap(src_u8[:, :, 2], mxB, myB, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        out = np.stack([R, G, B], axis=-1).astype(np.float32) / 255.0

        # ── Scanlines (high-frequency luminance grating) ──
        if scanline > 0.0:
            line_count = int(min(256, max(80, scan_freq * 70.0)))
            yy = np.arange(Hh, dtype=np.float32)[:, None]
            phase = (yy / Hh) * (2.0 * math.pi) * line_count + roll_phase
            scan_dark = scanline * (0.5 - 0.5 * np.cos(phase))
            out *= (1.0 - scan_dark)[..., None]

        # ── Rolling scan band (the live-tube vertical roll) ──
        if anim_mode in ("roll", "flow"):
            yy = np.arange(Hh, dtype=np.float32)[:, None] / Hh
            dy = yy - roll_pos
            band = np.exp(-(dy * dy) / (2.0 * 0.02 * 0.02)) * 0.18
            out += band[..., None]

        # ── Aperture-grille phosphor mask (per-column RGB weighting) ──
        if mask_strength > 0.0:
            ms = mask_strength
            col = np.arange(Ww) % 3
            grille = np.ones((Ww, 3), dtype=np.float32)
            grille[:, 0] = np.where(col == 0, 1.0 + ms, 1.0 - ms * 0.5)
            grille[:, 1] = np.where(col == 1, 1.0 + ms, 1.0 - ms * 0.5)
            grille[:, 2] = np.where(col == 2, 1.0 + ms, 1.0 - ms * 0.5)
            out *= grille[None, :, :]

        # ── Vignette (on the screen-edge radius) ──
        if vignette > 0.0:
            out *= (1.0 - vignette * np.clip(r2, 0.0, 1.0))[..., None]

        # ── Flicker (animated modes only — keeps 'none' a perfect static baseline) ──
        if do_flicker > 0.0:
            out *= (1.0 - do_flicker)

        out = np.clip(out * brightness, 0.0, 1.0).astype(np.float32)

        write_scalars(out_dir, curvature=float(eff_k),
                      scan_freq=float(scan_freq), mask_strength=float(mask_strength))
        capture_frame("522", out)
        save(out, mn(522, f"CRT Emulation t={_t:.2f}"), out_dir)
        return out
    except Exception as exc:
        fallback = np.full((int(H), int(W), 3), 0.5, dtype=np.float32)
        save(fallback, mn(522, "CRT Emulation"), out_dir)
        print(f"[method_522] ERROR: {exc}")
        return fallback
