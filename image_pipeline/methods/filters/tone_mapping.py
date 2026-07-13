from __future__ import annotations

import math
from pathlib import Path

import numpy as np

from ...core.registry import method
from ...core.utils import (save, norm, mn, seed_all, W, H, PALETTES, load_input)
from ...core.animation import capture_frame


# ── Tone mapping operators ────────────────────────────────────────────────
# Each takes a linear HDR-ish float image (values may exceed 1.0) and returns
# an LDR [0,1] image. Inputs are assumed already scaled by exposure.
def _reinhard(c: np.ndarray, key: float) -> np.ndarray:
    # Global Reinhard (2002): L / (1 + L), luminance keyed for brightness.
    # key only shifts input scale here; brightness control is via exposure.
    L = c * key
    return L / (1.0 + L)


def _reinhard_ext(c: np.ndarray, key: float, lum: float, white: float) -> np.ndarray:
    # Extended Reinhard (Drago 2003 local adaptation form):
    # out = (L * (1 + L/white^2)) / (1 + L) * exposure_bias
    # We expose `white` for shoulder control and `lum` for shoulder rolloff.
    L = c * key
    w2 = max(1e-3, white) ** 2
    return (L * (1.0 + L / w2)) / (1.0 + L) * lum


def _drago(c: np.ndarray, key: float, bias: float, max_lum: float) -> np.ndarray:
    # Drago et al. 2003 adaptive log mapping.
    eps = 1e-4
    L = np.maximum(c * key, eps)
    Lmax = max(float(L.max()), eps)
    # log bias term
    lb = math.log(bias) / math.log(0.5)
    num = np.log(1.0 + L) * lb / np.log(2.0 + 8.0 * (L / max_lum) ** lb)
    den = math.log(1.0 + Lmax) * lb / math.log(2.0 + 8.0)
    return np.clip(num / (den + eps), 0.0, 1.0)


def _aces_filmic(c: np.ndarray) -> np.ndarray:
    # Narkowicz ACES filmic approximation (RRT+ODT fit).
    a, b, cc, d, e = 2.51, 0.03, 2.43, 0.59, 0.14
    out = (c * (a * c + b)) / (c * (cc * c + d) + e)
    return np.clip(out, 0.0, 1.0)


def _uncharted2(c: np.ndarray, key: float) -> np.ndarray:
    # Hable / Uncharted 2 filmic curve (with exposure pre-scale + white point).
    A, B, C, D, E, F = 0.15, 0.50, 0.10, 0.20, 0.02, 0.30
    x = c * key

    def f(t: np.ndarray) -> np.ndarray:
        return ((t * (A * t + C * B) + D * E) /
                (t * (A * t + B) + D * F)) - E / F

    w = 11.2
    out = f(x) / f(np.array([w], dtype=np.float32))
    return np.clip(out, 0.0, 1.0)


def _srgb_encode(c: np.ndarray) -> np.ndarray:
    # sRGB OETF (linear [0,1] → sRGB display [0,1]).
    c = np.clip(c, 0.0, 1.0)
    return np.where(c <= 0.0031308, c * 12.92, 1.055 * (c ** (1.0 / 2.4)) - 0.055)


# ── AgX (Troy Sobotka / Filament, adopted by Blender 4.0, 2023) ──
# A scene-linear, matrix-based filmic tone mapper tuned for desaturation and
# pleasant highlight rolloff. Three matrices: linear→AgX space, the look matrix
# (optional grade), and AgX space→linear. The per-channel contrast approximation
# is Filament's polynomial fit. Output is sRGB-encoded (display-ready).
_AGX_M = np.array([
    [0.842479062853662, 0.0423282422610123, 0.0423756542741196],
    [0.0784335999999992, 0.878468636469662, 0.0784336],
    [0.079223745147764, 0.0791661274605434, 0.879142973793104],
], dtype=np.float32)
_AGX_M_INV = np.array([
    [1.19687900512017, -0.0528968517574562, -0.0529716355146663],
    [-0.0980208811401368, 1.15190312990417, -0.0980434501171241],
    [-0.0990297440797205, -0.0989611768448433, 1.15107367264116],
], dtype=np.float32)
_AGX_LOOKS = {
    "none": np.eye(3, dtype=np.float32),
    # "AgX Golden": warmer, slightly lifted
    "golden": np.array([
        [1.147220982, -0.042192627, -0.105131429],
        [0.014661568, 1.035746895, -0.049728253],
        [-0.001343032, 0.009162349, 0.919066668],
    ], dtype=np.float32),
    # "AgX Punchy": more saturation/contrast
    "punchy": np.array([
        [1.125599268, -0.038983671, -0.086685761],
        [0.010284708, 1.056097987, -0.066294175],
        [-0.004264267, 0.022689565, 0.871606535],
    ], dtype=np.float32),
}


def _agx(c: np.ndarray, look: str = "none") -> np.ndarray:
    """AgX filmic tone mapping. `c` is linear (exposure-pre-scaled), returns
    sRGB-encoded display values in [0,1]."""
    c = np.clip(np.asarray(c, dtype=np.float32), 0.0, None)
    x = c @ _AGX_M.T
    x2 = x * x
    x4 = x2 * x2
    # Filament contrast-approximation polynomial (per channel)
    out = (15.5 * x4 * x2 - 40.14 * x4 * x + 31.96 * x4
           - 6.868 * x2 * x + 0.4298 * x2 + 0.1191 * x - 0.00232)
    out = np.clip(out, 0.0, None)
    lm = _AGX_LOOKS.get(look, _AGX_LOOKS["none"])
    out = out @ lm.T
    out = np.clip(out, 0.0, None)
    out = out @ _AGX_M_INV.T
    out = np.clip(out, 0.0, None)
    return _srgb_encode(out)


_OPERATORS = {
    "reinhard": lambda c, p: _reinhard(c, p["key"]),
    "reinhard_ext": lambda c, p: _reinhard_ext(c, p["key"], p["lum"], p["white"]),
    "drago": lambda c, p: _drago(c, p["key"], p["bias"], p["max_lum"]),
    "aces": lambda c, p: _aces_filmic(c),
    "uncharted2": lambda c, p: _uncharted2(c, p["key"]),
    "agx": lambda c, p: _agx(c, p["look"]),
}


@method(id='428', name='Photographic Tone Mapping', category='filters', new_image_contract=True, tags=['tonemap', 'hdr', 'exposure', 'photographic', 'color', 'expanded', 'animation'], inputs={'image_in': 'IMAGE'}, outputs={'image': 'IMAGE'}, params={'source': {'description': 'source (noise/gradient/input_image/palette/rainbow/procedural)', 'default': 'gradient'}, 'operator': {'description': 'tone mapping operator (reinhard/reinhard_ext/drago/aces/uncharted2/agx)', 'choices': ['reinhard', 'reinhard_ext', 'drago', 'aces', 'uncharted2', 'agx'], 'default': 'aces'}, 'exposure': {'description': 'exposure multiplier applied before the operator (EV, log2-ish)', 'min': 0.1, 'max': 8.0, 'default': 1.6}, 'gamma': {'description': 'output gamma (display encoding; 1.0=linear, 2.2=approx sRGB)', 'min': 0.6, 'max': 3.0, 'default': 2.2}, 'saturation': {'description': 'color saturation boost after tonemap (0=gray, 1.5=punchy)', 'min': 0.0, 'max': 2.0, 'default': 1.15}, 'look': {'description': 'AgX look grade (none/golden/punchy) — only used when operator=agx', 'choices': ['none', 'golden', 'punchy'], 'default': 'none'}, 'key': {'description': 'Reinhard family brightness key (operator scale)', 'min': 0.3, 'max': 3.0, 'default': 1.0}, 'white': {'description': 'shoulder white point for reinhard_ext (high=brighter highlights)', 'min': 1.0, 'max': 16.0, 'default': 8.0}, 'lum': {'description': 'shoulder luminance rolloff for reinhard_ext', 'min': 0.5, 'max': 1.5, 'default': 1.0}, 'bias': {'description': 'Drago log bias (lower=brighter midtones)', 'min': 0.3, 'max': 0.9, 'default': 0.85}, 'noise_amp': {'description': 'noise amplitude for generated sources', 'min': 0.1, 'max': 1.0, 'default': 0.8}, 'blur_sigma': {'description': 'gaussian blur sigma for noise source', 'min': 5, 'max': 80, 'default': 30}, 'palette': {'description': 'palette name for palette source', 'default': 'vapor'}, 'anim_mode': {'description': 'animation mode (none/exposure_sweep/operator_cycle/gamma_pulse)', 'choices': ['none', 'exposure_sweep', 'operator_cycle', 'gamma_pulse'], 'default': 'none'}, 'anim_speed': {'description': 'animation speed multiplier', 'min': 0.1, 'max': 5.0, 'default': 1.0}})
def method_tone_mapping(out_dir: Path, seed: int, params=None):
    """Photographic Tone Mapping — map HDR-ish radiance to a displayable LDR image.

    Implements the classic photographic tone-mapping operators that turn a
    high-dynamic-range luminance field into a perceptually-tuned low-dynamic-range
    picture (Reinhard 2002; Drago et al. 2003; the ACES filmic fit of Narkowicz
    2016; the Uncharted-2 / Hable filmic curve; and AgX — Troy Sobotka's
    scene-linear matrix filmic operator adopted by Blender 4.0 in 2023). These
    are the "look" curves behind every HDR photo and real-time renderer — and
    they also make a great stylizing pass on ordinary LDR images.

    Pipeline: build/resolve a source field → scale by exposure → apply the chosen
    operator (per-channel, which is the standard real-time approximation; for the
    luminance-based Reinhard variants this is equivalent to luminance mapping for
    our generated tonal sources) → gamma-encode → saturation adjust. The CPU path
    is the authoritative export.

    Params:
        source:    generated source type (noise/gradient/input_image/palette/rainbow/procedural)
        operator:  reinhard / reinhard_ext / drago / aces / uncharted2 / agx
        look:      AgX look grade (none/golden/punchy); only used when operator=agx
        exposure:  pre-operator EV multiplier (0.1-8, default 1.6)
        gamma:     output display gamma (0.6-3.0, default 2.2)
        saturation: post-tonemap saturation (0-2, default 1.15)
        key:       Reinhard family brightness key
        white:     reinhard_ext shoulder white point
        lum:       reinhard_ext shoulder luminance rolloff
        bias:      Drago log bias
        time:      animation clock (0-6.28)
        anim_mode: none / exposure_sweep / operator_cycle / gamma_pulse
    """
    try:
        if params is None:
            params = {}
        anim_time = float(params.get("time", 0.0))
        anim_mode = str(params.get("anim_mode", "none"))
        anim_speed = float(params.get("anim_speed", 1.0))
        seed_all(seed)
        rng = np.random.default_rng(seed)

        operator = str(params.get("operator", "aces"))
        if operator not in _OPERATORS:
            operator = "aces"
        exposure = float(params.get("exposure", 1.6))
        gamma = float(params.get("gamma", 2.2))
        saturation = float(params.get("saturation", 1.15))
        look = str(params.get("look", "none"))
        if look not in _AGX_LOOKS:
            look = "none"
        key = float(params.get("key", 1.0))
        white = float(params.get("white", 8.0))
        lum = float(params.get("lum", 1.0))
        bias = float(params.get("bias", 0.85))
        noise_amp = float(params.get("noise_amp", 0.8))
        blur_sigma = float(params.get("blur_sigma", 30))
        pal_name = str(params.get("palette", "vapor"))
        source = str(params.get("source", "gradient"))

        # ── Animation (rename t to avoid shadowing the time param) ──
        _t = anim_time * anim_speed
        if anim_mode == "exposure_sweep":
            # smooth 0.3 → 6.0 oscillation, no cusp
            exposure = 0.3 + 5.7 * (0.5 + 0.5 * math.sin(_t * 0.3))
        elif anim_mode == "gamma_pulse":
            gamma = 1.0 + 2.0 * (0.5 + 0.5 * math.sin(_t * 0.4))
        # operator_cycle handled below (uses _t); "none" = static

        # ── Resolve source image (float32 [0,1], H×W×3) ──
        # A wired upstream image always overrides source generation (Rule #12).
        src = None
        wired_path = params.get("input_image", "")
        if wired_path:
            try:
                src = load_input(wired_path, int(W), int(H))
            except (FileNotFoundError, OSError):
                src = None
        if src is None and params.get("_input_image") is not None:
            src = np.asarray(params["_input_image"], dtype=np.float32)

        if src is None:
            if source == "gradient":
                yy, xx = np.mgrid[:H, :W].astype(np.float32)
                r = np.sqrt((xx - W / 2) ** 2 + (yy - H / 2) ** 2)
                g = norm(r)
                src = np.stack([g, g * 0.7, 1 - g], axis=-1).clip(0, 1)
            elif source == "palette":
                pal = PALETTES.get(pal_name, list(PALETTES.values())[0])
                yy, xx = np.mgrid[:H, :W].astype(np.float32)
                r = norm(np.sqrt((xx - W / 2) ** 2 + (yy - H / 2) ** 2))
                idx = (r * (len(pal) - 1)).astype(np.int32)
                src = np.array(pal, dtype=np.float32)[idx] / 255.0
            elif source == "rainbow":
                yy, xx = np.mgrid[:H, :W].astype(np.float32)
                r = norm(np.sqrt((xx - W / 2) ** 2 + (yy - H / 2) ** 2))
                hue = r * 2 * math.pi
                src = np.stack([
                    np.sin(hue) * 0.5 + 0.5,
                    np.sin(hue + 2.094) * 0.5 + 0.5,
                    np.sin(hue + 4.189) * 0.5 + 0.5,
                ], axis=-1).astype(np.float32)
            elif source == "procedural":
                yy, xx = np.mgrid[:H, :W].astype(np.float32)
                g = (np.sin(xx * 0.03 + yy * 0.02 + _t * 0.5) *
                     np.cos(xx * 0.02 - yy * 0.03 + _t * 0.3) * 0.5 + 0.5)
                src = np.stack([g, g * 0.6, 1 - g * 0.8], axis=-1).astype(np.float32)
            else:  # noise
                from scipy.ndimage import gaussian_filter
                base = rng.random((H, W, 3)).astype(np.float32)
                base = (base - 0.5) * (2 * noise_amp) + 0.5
                if blur_sigma > 0:
                    for c in range(3):
                        base[:, :, c] = gaussian_filter(base[:, :, c], blur_sigma)
                src = base.clip(0, 1)

        src = np.asarray(src, dtype=np.float32).clip(0, 1)
        img = src * float(exposure)

        # ── Operator cycle (animation): rotate through the 5 operators over time ──
        op_name = operator
        if anim_mode == "operator_cycle":
            names = list(_OPERATORS.keys())
            op_name = names[int((_t / (2 * math.pi)) * len(names)) % len(names)]

        op_params = {"key": key, "lum": lum, "white": white, "bias": bias,
                     "max_lum": float(np.maximum(img, 1e-4).max()),
                     "look": look}
        mapped = _OPERATORS[op_name](img, op_params)

        # ── Per-channel Reinhard variants are luminance-equivalent for our
        # tonal sources; keep channels then gamma + saturation. AgX already
        # returns sRGB-encoded display values, so skip the gamma power for it. ──
        out = np.clip(mapped, 0.0, 1.0)
        if op_name != "agx":
            out = np.power(out, 1.0 / max(0.01, gamma))

        # Saturation: lerp around luminance
        if saturation != 1.0:
            lum_out = out.mean(axis=-1, keepdims=True)
            out = np.clip(lum_out + (out - lum_out) * saturation, 0.0, 1.0)

        result = (np.clip(out, 0.0, 1.0) * 255.0).astype(np.uint8)
        save(result, mn(428, f"Tone Map {op_name}"), out_dir)
    except Exception:
        fallback = (np.ones((int(H), int(W), 3), dtype=np.uint8) * 128)
        save(fallback, mn(428, "Tone Map"), out_dir)
        raise
