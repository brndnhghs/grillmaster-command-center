from __future__ import annotations

import math

import numpy as np

from ...core.registry import method
from ...core.utils import (
    save, mn, seed_all, W, H, wired_source_rgb, write_scalars,
)
from ...core.animation import capture_frame

# ─────────────────────────────────────────────────────────────────────────────
# Tone Mapping  (node 997)
#
# Maps a high-dynamic-range (HDR) linear-ish image into the displayable
# [0,1] range. Implemented as a filter node so any upstream generator (fluids,
# reaction-diffusion, fractals, a wired image) can be compressed for display
# the way a real-time renderer would.
#
# Operators:
#   * agx        — AgX (Sobotka / Blender 4.0 default, 2023), derived from the
#                  Blender / Filament / three.js GLSL (dmnsgn/glsl-tone-map).
#                  A perceptually-neutral, film-negative-inspired curve that
#                  preserves hue better than ACES. Three looks: neutral,
#                  punchy, golden.
#   * aces       — ACES filmic approximation (Narkowicz, 2015 / ACES 1.0).
#   * reinhard   — photographic Reinhard (Reinhard et al. 2002), extended with
#                  a configurable white point.
#   * uncharted2 — Hable / Uncharted 2 "filmic" operator (Epic / COD: AW).
#   * log        — log(1+x) / log(1+white) — classic HDR display curve.
#   * gamma      — pure gamma compression (reference / debug).
#
# AgX matrices below are the column-major Blender matrices transcribed into
# row-major [i][j] form; they are applied as ``color @ M.T`` so the per-pixel
# product matches GLSL ``M * v`` exactly.
# ─────────────────────────────────────────────────────────────────────────────

_AGX_INSET = np.array([
    [0.856627153315983, 0.0951212405381588, 0.0482516061458583],
    [0.137318972929847, 0.761241990602591, 0.101439036467562],
    [0.11189821299995,  0.0767994186031903, 0.811302368396859],
], dtype=np.float64)

_AGX_OUTSET = np.array([
    [1.1271005818144368, -0.1413297634984383, -0.14132976349843826],
    [-0.11060664309660323, 1.157823702216272, -0.11060664309660294],
    [-0.016493938717834573, -0.016493938717834257, 1.2519364065950405],
], dtype=np.float64)

_AGX_MIN_EV = -12.47393
_AGX_MAX_EV = 4.026069

_LW = np.array([0.2126, 0.7152, 0.0722], dtype=np.float64)


def _srgb_encode(lin: np.ndarray) -> np.ndarray:
    """Linear [0,1] -> sRGB display (true piecewise transfer)."""
    lin = np.clip(lin, 0.0, 1.0)
    return np.where(lin <= 0.0031308,
                    12.92 * lin,
                    1.055 * np.power(lin, 1.0 / 2.4) - 0.055)


def _agx(color: np.ndarray, slope, offset, power) -> np.ndarray:
    """AgX forward transform; returns linear LDR in [0,1] (per channel)."""
    c = color @ _AGX_INSET.T
    c = np.maximum(c, 1e-10)
    c = np.clip(np.log2(c), _AGX_MIN_EV, _AGX_MAX_EV)
    c = (c - _AGX_MIN_EV) / (_AGX_MAX_EV - _AGX_MIN_EV)
    c = np.clip(c, 0.0, 1.0)
    x2 = c * c
    x4 = x2 * x2
    # AgX sigmoid approximation (mean error^2 ~ 3.7e-06)
    c = (15.5 * x4 * x2 - 40.14 * x4 * c + 31.96 * x4
         - 6.868 * x2 * c + 0.4298 * x2 + 0.1191 * c - 0.00232)
    c = np.power(np.clip(c * slope + offset, 0.0, None), power)
    # internal saturation fixed to 1.0; global saturation is applied later
    luma = c @ _LW
    c = luma[..., None] + (c - luma[..., None])
    c = c @ _AGX_OUTSET.T
    return c


def _aces(x: np.ndarray) -> np.ndarray:
    a, b, c, d, e = 2.51, 0.03, 2.43, 0.59, 0.14
    return np.clip((x * (a * x + b)) / (x * (c * x + d) + e), 0.0, 1.0)


def _reinhard(x: np.ndarray, white: float = 4.0) -> np.ndarray:
    w2 = white * white
    return (x * (1.0 + x / w2)) / (1.0 + x)


def _uncharted2(x: np.ndarray) -> np.ndarray:
    A, B, C, D, E, F = 0.15, 0.50, 0.10, 0.20, 0.02, 0.30
    f = lambda v: ((v * (A * v + C * B) + D * E) / (v * (A * v + B) + D * F)) - E / F
    white = 11.2
    norm = f(np.array([white], dtype=np.float64))[0]
    return f(x) / norm


def _log_op(x: np.ndarray, white: float = 8.0) -> np.ndarray:
    return np.log(1.0 + x) / np.log(1.0 + white)


def _gamma_op(x: np.ndarray, gamma: float = 1.0) -> np.ndarray:
    return np.power(np.clip(x, 0.0, None), 1.0 / max(gamma, 1e-3))


def _apply_sat(c: np.ndarray, sat: float) -> np.ndarray:
    luma = c @ _LW
    return luma[..., None] + sat * (c - luma[..., None])


def _hsv_to_rgb(h, s, v):
    """Vectorized HSV -> RGB (all arrays broadcastable to (...,))."""
    h = np.asarray(h, dtype=np.float64)
    i = np.floor(h * 6.0).astype(int) % 6
    f = h * 6.0 - np.floor(h * 6.0)
    p = v * (1.0 - s)
    q = v * (1.0 - s * f)
    t = v * (1.0 - s * (1.0 - f))
    r = np.choose(i, [v, t, p, p, q, v])
    g = np.choose(i, [q, v, v, t, p, p])
    b = np.choose(i, [p, p, q, v, v, t])
    return np.stack([r, g, b], axis=-1)


@method(
    id="997",
    name="Tone Mapping",
    category="filters",
    tags=["tone-mapping", "agx", "aces", "hdr", "exposure", "color", "post-process"],
    params={
        "operator": {
            "description": "tone mapping operator: agx (Blender 2023), aces, reinhard, uncharted2, log, gamma",
            "choices": ["agx", "aces", "reinhard", "uncharted2", "log", "gamma"],
            "default": "agx",
        },
        "agx_look": {
            "description": "AgX look (only used when operator=agx): neutral, punchy, golden",
            "choices": ["neutral", "punchy", "golden"],
            "default": "neutral",
        },
        "exposure": {
            "description": "exposure in EV stops applied before the tone curve (brightens/darkens highlights)",
            "min": -3.0, "max": 3.0, "default": 0.0,
        },
        "saturation": {
            "description": "output color saturation (1.0 = unchanged)",
            "min": 0.0, "max": 2.0, "default": 1.0,
        },
        "gamma": {
            "description": "output gamma — only used by the gamma operator",
            "min": 0.3, "max": 2.5, "default": 1.0,
        },
        "white": {
            "description": "highlight white point for reinhard/log operators",
            "min": 1.0, "max": 16.0, "default": 4.0,
        },
        "source": {
            "description": "image source: none (procedural HDR test scene) or input_image (wired upstream)",
            "choices": ["none", "input_image"], "default": "none",
        },
        "anim_mode": {
            "description": "animation: none (static), exposure_pulse (EV breathes), saturation_pulse (sat breathes)",
            "default": "none",
        },
        "anim_speed": {
            "description": "animation speed multiplier",
            "min": 0.1, "max": 3.0, "default": 1.0,
        },
        "time": {
            "description": "animation phase [0, 2pi) (system-injected)",
            "min": 0.0, "max": 6.28, "default": 0.0,
        },
    },
    inputs={"image_in": "IMAGE"},
    outputs={"image": "IMAGE"},
)
def method_tone_mapping(out_dir, seed: int, params=None):
    """Tone mapping (HDR -> display) with the AgX operator (Blender 4.0, 2023)
    plus ACES, Reinhard, Uncharted-2, log and gamma.

    With no wired image it renders a deterministic HDR test scene (a bright
    core plus a few specular hotspots reaching ~20x display white) so the
    highlight-compression behaviour of each operator is directly visible.
    With a wired IMAGE it tone-maps that image (Rule-#12 override), which is
    the primary use: drop it after any HDR-ish generator (fluids, RD,
    fractals) to get a properly displayed result.

    Closed-form per-pixel, fully vectorized (O(W*H)) — never hits the
    render-timeout cull, so it is safe for cheap-alive graphs.
    """
    try:
        if params is None:
            params = {}
        t = float(params.get("time", 0.0))
        anim_mode = params.get("anim_mode", "none")
        anim_speed = float(params.get("anim_speed", 1.0))
        _t = 0.0 if anim_mode == "none" else t * anim_speed

        operator = params.get("operator", "agx")
        agx_look = params.get("agx_look", "punchy")
        exposure_ev = float(params.get("exposure", 0.0))
        saturation = float(params.get("saturation", 1.0))
        gamma = float(np.clip(params.get("gamma", 1.0), 0.3, 2.5))
        white = float(np.clip(params.get("white", 4.0), 1.0, 16.0))

        # ── animation: breathe a scalar so active modes pass Δ>0.05 ──
        if anim_mode == "exposure_pulse":
            exposure_ev = exposure_ev + 2.0 * math.sin(_t)
        elif anim_mode == "saturation_pulse":
            saturation = float(np.clip(saturation + 1.0 * math.sin(_t), 0.0, 2.0))

        # ── source ──
        wired = wired_source_rgb(params, W, H)
        if wired is not None:
            src = wired.astype(np.float64)
        else:
            seed_all(seed)
            yy, xx = np.mgrid[0:H, 0:W].astype(np.float64)
            nx = xx / W - 0.5
            ny = yy / H - 0.5
            r = np.sqrt(nx * nx + ny * ny)
            # HDR brightness: dim surround (~0.1) up to a bright core (~2.7x
            # display white), plus a few specular hotspots that exceed 1.
            brightness = 0.10 + 2.6 * np.clip(1.0 - r * 1.5, 0.0, 1.0)
            for (cx, cy, amp) in [(0.30, 0.35, 14.0),
                                  (0.72, 0.62, 9.0),
                                  (0.55, 0.20, 6.0)]:
                d = np.sqrt((nx - cx + 0.5) ** 2 + (ny - cy + 0.5) ** 2)
                brightness = brightness + amp * np.exp(-(d * d) / (2 * 0.02))
            # chromatic tint (hue by angle) so colour operators + the
            # saturation control are meaningful on the default scene.
            ang = np.arctan2(ny, nx) / (2.0 * np.pi) + 0.5
            tint = _hsv_to_rgb(ang, 0.65, 1.0)
            src = tint * brightness[..., None]

        peak = float(src.max())
        src = src * (2.0 ** exposure_ev)

        # ── operator ──
        if operator == "agx":
            if agx_look == "neutral":
                slope, offset, power = 1.0, 0.0, 1.0
            elif agx_look == "golden":
                slope, offset, power = np.array([1.0, 0.9, 0.5]), 0.0, 0.8
            else:  # punchy
                slope, offset, power = 1.0, 0.0, 1.35
            color = _agx(src, slope, offset, power)
        elif operator == "aces":
            color = _aces(src)
        elif operator == "reinhard":
            color = _reinhard(src, white=white)
        elif operator == "uncharted2":
            color = _uncharted2(src)
        elif operator == "log":
            color = _log_op(src, white=white)
        else:  # gamma
            color = _gamma_op(src, gamma=gamma)

        color = np.clip(color, 0.0, None)
        color = _srgb_encode(color)
        # saturation is a display-space (sRGB) control — conventional for a
        # saturation knob and far more visually responsive than linear-space.
        color = _apply_sat(color, saturation)
        rgb = np.clip(color, 0.0, 1.0).astype(np.float32)

        capture_frame("997", rgb)
        save(rgb, mn(997, "Tone Mapping"), out_dir)
        try:
            write_scalars(out_dir, exposure_ev=float(exposure_ev),
                          peak_luminance=float(peak))
        except Exception:
            pass
        return rgb
    except Exception as exc:
        fallback = np.full((int(H), int(W), 3), 128, dtype=np.uint8)
        save(fallback, mn(997, "Tone Mapping"), out_dir)
        print(f"[method_997] ERROR: {exc}")
        return fallback
