from __future__ import annotations

import math
from pathlib import Path

import numpy as np

from ...core.registry import method
from ...core.utils import (save, norm, mn, seed_all, W, H, PALETTES, load_input)
from ...core.animation import capture_frame


# ── Separable box / Gaussian blur (pure numpy, O(n) via summed-area tables) ──
def _hblur(img: np.ndarray, r: int) -> np.ndarray:
    """Horizontal box blur of an (H, W, C) array using a cumulative sum."""
    if r <= 0:
        return img.astype(np.float32)
    H, Wc, C = img.shape
    cs = np.cumsum(img.astype(np.float64), axis=1)
    pad = np.zeros((H, 1, C), dtype=np.float64)
    cs = np.concatenate([pad, cs], axis=1)  # (H, W+1, C)
    idx1 = np.clip(np.arange(Wc) + r + 1, 0, Wc)
    idx0 = np.clip(np.arange(Wc) - r, 0, Wc)
    out = cs[:, idx1, :] - cs[:, idx0, :]
    denom = np.maximum((idx1 - idx0)[None, :, None], 1.0)
    return (out / denom).astype(np.float32)


def _vblur(img: np.ndarray, r: int) -> np.ndarray:
    """Vertical box blur of an (H, W, C) array using a cumulative sum."""
    if r <= 0:
        return img.astype(np.float32)
    H, Wc, C = img.shape
    cs = np.cumsum(img.astype(np.float64), axis=0)
    pad = np.zeros((1, Wc, C), dtype=np.float64)
    cs = np.concatenate([pad, cs], axis=0)  # (H+1, W, C)
    idx1 = np.clip(np.arange(H) + r + 1, 0, H)
    idx0 = np.clip(np.arange(H) - r, 0, H)
    out = cs[idx1, :, :] - cs[idx0, :, :]
    denom = np.maximum((idx1 - idx0)[:, None, None], 1.0)
    return (out / denom).astype(np.float32)


def _box_blur(img: np.ndarray, r: int) -> np.ndarray:
    return _vblur(_hblur(img, r), r)


def _gaussian_blur(img: np.ndarray, r: int, iters: int = 3) -> np.ndarray:
    """Approximate a Gaussian blur with `iters` successive box blurs."""
    out = img.astype(np.float32)
    for _ in range(max(1, iters)):
        out = _box_blur(out, max(1, int(round(r))))
    return out


# ── Bright-pass (HDR-style soft-knee threshold) ──
_LUMA = np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)


def _bright_pass(c: np.ndarray, threshold: float, softness: float) -> np.ndarray:
    """Keep only the bright parts of `c` with a soft (smooth) knee.

    Mirrors the standard real-time bloom prefilter: instead of a hard `max(c - t, 0)`,
    a smoothstep knee fades sub-threshold pixels in, eliminating the hard banding
    edge that a binary threshold produces.
    """
    lum = c @ _LUMA  # (H, W)
    knee = max(threshold * softness, 1e-3)
    f = np.clip((lum - (threshold - knee)) / (2.0 * knee), 0.0, 1.0)
    f = f * f  # smoothstep
    return (c * f[:, :, None]).astype(np.float32)


# ── Procedural sources (used when no image is wired in) ──
def _gen_source(source: str, rng: np.random.Generator, w: int, h: int,
                t_anim: float, noise_amp: float, pal_name: str) -> np.ndarray:
    """Generate a float32 [0,1] H×W×3 source image."""
    if source == "gradient":
        yy, xx = np.mgrid[:h, :w].astype(np.float32)
        g = norm(np.sqrt((xx - w / 2) ** 2 + (yy - h / 2) ** 2))
        return np.stack([g, g * 0.7, 1 - g], axis=-1).clip(0, 1)
    if source == "palette":
        pal = PALETTES.get(pal_name, list(PALETTES.values())[0])
        yy, xx = np.mgrid[:h, :w].astype(np.float32)
        r = norm(np.sqrt((xx - w / 2) ** 2 + (yy - h / 2) ** 2))
        idx = (r * (len(pal) - 1)).astype(np.int32)
        return np.array(pal, dtype=np.float32)[idx] / 255.0
    if source == "rainbow":
        yy, xx = np.mgrid[:h, :w].astype(np.float32)
        r = norm(np.sqrt((xx - w / 2) ** 2 + (yy - h / 2) ** 2))
        hue = r * 2 * math.pi
        return np.stack([
            np.sin(hue) * 0.5 + 0.5,
            np.sin(hue + 2.094) * 0.5 + 0.5,
            np.sin(hue + 4.189) * 0.5 + 0.5,
        ], axis=-1).astype(np.float32)
    if source == "procedural":
        # Bright Gaussian "light sources" on a black field — ideal bloom subject.
        img = np.zeros((h, w, 3), dtype=np.float32)
        n = 6 + int(rng.random() * 5)
        base_rad = min(w, h) * 0.05
        for _ in range(n):
            x = int(rng.random() * w)
            y = int(rng.random() * h)
            rad = max(6, int(base_rad * (0.5 + rng.random())))
            col = rng.random(3)
            yy, xx = np.ogrid[:h, :w]
            d2 = (xx - x) ** 2 + (yy - y) ** 2
            m = np.exp(-d2 / (2.0 * rad * rad))
            img += m[:, :, None] * col[None, None, :]
        # gentle drift so animation modes have something to act on
        img = np.roll(img, int(t_anim * 4) % w, axis=1)
        return np.clip(img, 0.0, 1.0).astype(np.float32)
    # noise / input_image fallback
    n = rng.standard_normal((h, w, 3)).astype(np.float32) * noise_amp + 0.5
    return norm(n)


def _render_bloom(src: np.ndarray, threshold: float, softness: float,
                  intensity: float, radius: int, streak: float,
                  iterations: int) -> np.ndarray:
    """Core bloom render: bright-pass -> anisotropic Gaussian -> additive composite."""
    h, w = src.shape[:2]
    bright = _bright_pass(src, threshold, softness)
    if streak > 1.0:
        # Anamorphic streak: blur much wider on X than Y (cinematic lens streaks).
        rh = max(1, int(round(radius * streak)))
        rv = max(1, int(round(radius)))
        glow = _hblur(bright, rh)
        glow = _vblur(glow, rv)
        for _ in range(max(0, iterations - 1)):
            glow = _hblur(glow, rh)
            glow = _vblur(glow, rv)
        glow = glow.astype(np.float32)
    else:
        glow = _gaussian_blur(bright, radius, iterations)
    out = src + glow * intensity
    return np.clip(out, 0.0, 1.0).astype(np.float32)


@method(
    id="408",
    name="Bloom / Glow",
    category="filters",
    new_image_contract=True,
    tags=["post-process", "glow", "bloom", "hdr", "anamorphic", "animation", "expanded"],
    inputs={"image_in": "IMAGE"},
    outputs={"image": "IMAGE"},
    params={
        "source": {"description": "source when no image is wired (noise/gradient/input_image/palette/rainbow/procedural)", "default": "procedural"},
        "threshold": {"description": "brightness cutoff for the bloom prefilter (0=everything glows, 1=almost nothing)", "min": 0.0, "max": 1.0, "default": 0.5},
        "softness": {"description": "soft-knee width as a fraction of threshold (smooth fade vs hard cut)", "min": 0.0, "max": 1.0, "default": 0.5},
        "intensity": {"description": "glow additive strength", "min": 0.0, "max": 3.0, "default": 1.2},
        "radius": {"description": "blur radius in px (glow spread)", "min": 1, "max": 40, "default": 12},
        "streak": {"description": "anamorphic streak anisotropy (1=round bloom, 8=wide cinematic streaks)", "min": 1.0, "max": 8.0, "default": 1.0},
        "iterations": {"description": "blur passes (wider, softer glow)", "min": 1, "max": 6, "default": 3},
        "noise_amp": {"description": "noise amplitude for generated sources", "min": 0.1, "max": 1.0, "default": 0.4},
        "palette": {"description": "palette name for palette source", "default": "vapor"},
        "anim_mode": {"description": "animation mode (none/pulse/threshold_sweep/streak_sweep)", "choices": ["none", "pulse", "threshold_sweep", "streak_sweep"], "default": "none"},
        "anim_speed": {"description": "animation speed multiplier", "min": 0.1, "max": 5.0, "default": 1.0},
        "time": {"description": "animation phase [0, 2pi)", "min": 0.0, "max": 6.28, "default": 0.0},
    },
)
def method_bloom(out_dir: Path, seed: int, params=None):
    """Bloom / Glow post-process with an optional anamorphic streak.

    Bloom is the canonical real-time HDR post-effect (Mitchell, "Bloom" in
    GPU Gems 1, Ch. 21, 2004; modernised as the soft-knee threshold in
    Unity's Post-Processing Stack and Unreal's Bloom). The pipeline:

        1. bright-pass  — keep only pixels above `threshold`, faded in by a
                          smoothstep "soft knee" so the cutoff has no hard edge;
        2. blur         — separable Gaussian (3 stacked box blurs ≈ Gaussian);
        3. composite    — additively blend the glow back over the source.

    A fresh twist: `streak > 1` turns the blur anisotropic (much wider on X
    than Y), producing the cinematic **anamorphic lens-streak** look popular in
    sci-fi / Unreal "Anamorphic Lens Flare" setups.

    The CPU path is authoritative. A wired upstream image always overrides
    source generation (Rule #12).

    Params:
        source:      generated source when unwired (noise/gradient/palette/rainbow/procedural)
        threshold:   brightness cutoff (0=all glows, 1=none)
        softness:    soft-knee width as fraction of threshold
        intensity:   additive glow strength (0-3)
        radius:      blur radius px (1-40)
        streak:      anamorphic anisotropy (1=round, 8=wide streaks)
        iterations:  blur passes (1-6)
        time:        animation clock [0, 2pi)
        anim_mode:   none / pulse / threshold_sweep / streak_sweep
        anim_speed:  animation speed (0.1-5.0)
    """
    try:
        if params is None:
            params = {}
        anim_time = float(params.get("time", 0.0))
        anim_mode = str(params.get("anim_mode", "none"))
        anim_speed = float(params.get("anim_speed", 1.0))
        seed_all(seed)
        rng = np.random.default_rng(seed)

        threshold = float(params.get("threshold", 0.6))
        softness = float(params.get("softness", 0.5))
        intensity = float(params.get("intensity", 1.2))
        radius = int(params.get("radius", 12))
        streak = float(params.get("streak", 1.0))
        iterations = int(params.get("iterations", 3))
        source = str(params.get("source", "procedural"))
        noise_amp = float(params.get("noise_amp", 0.4))
        pal_name = str(params.get("palette", "vapor"))

        # ── Animation (rename t so we never shadow the time param) ──
        _t = anim_time * anim_speed
        if anim_mode == "pulse":
            intensity = intensity * (1.1 + 0.9 * math.sin(_t * 0.4))  # 0.2x .. 2.0x swing
        elif anim_mode == "threshold_sweep":
            threshold = float(np.clip(threshold * (0.5 + 0.5 * math.sin(_t * 0.3)), 0.02, 0.98))
        elif anim_mode == "streak_sweep":
            streak = float(np.clip(1.0 + (streak - 1.0) * (0.5 + 0.5 * math.sin(_t * 0.3)), 1.0, 8.0))

        w, h = int(W), int(H)

        # ── Resolve source (wired input overrides generation) ──
        src = None
        wired_path = params.get("input_image", "")
        if wired_path:
            try:
                src = load_input(wired_path, w, h)
            except (FileNotFoundError, OSError):
                src = None
        if src is None and params.get("_input_image") is not None:
            src = np.asarray(params["_input_image"], dtype=np.float32)
        if src is None:
            src = _gen_source(source, rng, w, h, _t, noise_amp, pal_name)
        src = np.clip(src, 0.0, 1.0).astype(np.float32)

        result = _render_bloom(src, threshold, softness, intensity, radius, streak, iterations)

        capture_frame("408", result)
        save(result, mn(408, "Bloom / Glow"), out_dir)
        return result
    except Exception as exc:
        fallback = np.full((int(H), int(W), 3), 0.0, dtype=np.float32)
        save(fallback, mn(408, "Bloom / Glow"), out_dir)
        print(f"[method_408] ERROR: {exc}")
        return fallback
