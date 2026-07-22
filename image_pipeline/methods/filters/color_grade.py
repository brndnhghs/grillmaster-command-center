"""Color Grading in the OKLab perceptual color space (Ottosson, 2020).

Color grading is the compositing primitive that is conspicuously absent from
the pipeline: every brightness/contrast/saturation/hue/temperature adjustment
should happen in a *perceptually uniform* space so that, e.g., saturation
changes do not also shift hue, and contrast changes do not crush the same way
across shadows and highlights. Classic sRGB-space RGB ops fail all of these.

We use OKLab (Björn Ottosson, 2020 — https://bottosson.github.io/posts/oklab/),
a modern, lightweight perceptual color space with excellent hue/lightness
uniformity and trivial linear↔OKLab matrices (no white-point or large-matrix
machinery like CIELAB). The grade is applied to an upstream WIRED image (Rule
12: a wired image always overrides the synthetic `source`) as:

    L' = ((L * 2^exposure - 0.5) * contrast + 0.5) ^ (1/gamma)     # lightness
    (a, b) rotated by hue_rotate, scaled by saturation              # chroma/hue
    (a, b) += temperature/tint offset                              # white balance
    L' *= vignette(r)                                              # radial falloff
    invert:  L' = 1 - L',  (a, b) = -(a, b)

Because the operations live in OKLab, "saturation +0.5" genuinely only changes
chroma, and "hue_rotate 90°" rotates hue without the brightness wobble you get
in HSV/HSL. The node also emits the perceptual-lightness FIELD (the OKLab L
channel) for downstream use.

Animation modes modulate the grading parameters over the `time` clock so the
grade itself can breathe / cycle — a cheap, always-moving seed that fights the
"static/flat" dead-rate.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np

from ...core.registry import method
from ...core.utils import (
    save,
    mn,
    seed_all,
    W,
    H,
    PALETTES,
    norm,
    write_scalars,
    write_field,
    load_input,
)
from ...core.animation import capture_frame


# ── sRGB <-> OKLab (Ottosson 2020) ──────────────────────────────────────────

def _srgb_to_linear(c: np.ndarray) -> np.ndarray:
    c = c.astype(np.float64)
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def _linear_to_srgb(c: np.ndarray) -> np.ndarray:
    c = np.clip(c, 0.0, 1.0).astype(np.float64)
    return np.where(c <= 0.0031308, c * 12.92, 1.055 * c ** (1 / 2.4) - 0.055)


def _rgb_to_oklab(rgb: np.ndarray):
    """rgb: (H,W,3) float [0,1] -> (L, a, b) each (H,W) float64."""
    lin = _srgb_to_linear(rgb)
    r, g, b = lin[..., 0], lin[..., 1], lin[..., 2]
    l = 0.4122214708 * r + 0.5363325363 * g + 0.0514459929 * b
    m = 0.2119034982 * r + 0.6806995451 * g + 0.1073969566 * b
    s = 0.0883024619 * r + 0.2817188376 * g + 0.6299787005 * b
    l_ = np.cbrt(l)
    m_ = np.cbrt(m)
    s_ = np.cbrt(s)
    L = 0.2104542553 * l_ + 0.7936177850 * m_ - 0.0040720468 * s_
    A = 1.9779984951 * l_ - 2.4285922050 * m_ + 0.4505937099 * s_
    B = 0.0259040371 * l_ + 0.7827717662 * m_ - 0.8086757660 * s_
    return L, A, B


def _oklab_to_rgb(L: np.ndarray, A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Inverse: (L, a, b) -> rgb (H,W,3) float [0,1]."""
    l_ = L + 0.3963377774 * A + 0.2158037573 * B
    m_ = L - 0.1055613458 * A - 0.0638541728 * B
    s_ = L - 0.0894841775 * A - 1.2914855480 * B
    l = l_ ** 3
    m = m_ ** 3
    s = s_ ** 3
    r = 4.0767416621 * l - 3.3077115913 * m + 0.2309699292 * s
    g = -1.2684380046 * l + 2.6097574011 * m - 0.3413193965 * s
    bb = -0.0041960863 * l - 0.7034186147 * m + 1.7076147010 * s
    lin = np.stack([r, g, bb], axis=-1)
    return _linear_to_srgb(lin)


# ── Synthetic source (standalone generation) ────────────────────────────────

def _source_rgb(source, hh, ww, rng, _t, pal_name):
    yy, xx = np.mgrid[0:hh, 0:ww].astype(np.float32)
    cx, cy = ww * 0.5, hh * 0.5
    nx = (xx - cx) / max(hh, ww)
    ny = (yy - cy) / max(hh, ww)

    if source == "gradient":
        ang = 0.3 + 0.2 * math.sin(_t)
        d = 0.5 + 0.5 * (nx * math.cos(ang) + ny * math.sin(ang))
        img = np.stack([d, d * 0.7, 1.0 - d], axis=-1)
    elif source == "palette":
        pal = PALETTES.get(pal_name, list(PALETTES.values())[0])
        r = norm(np.sqrt(nx * nx + ny * ny))
        idx = (r * (len(pal) - 1)).astype(np.int32)
        img = np.array(pal, dtype=np.float32)[idx] / 255.0
    elif source == "rainbow":
        hue = norm(np.sqrt(nx * nx + ny * ny)) * 2 * math.pi
        img = np.stack([
            np.sin(hue) * 0.5 + 0.5,
            np.sin(hue + 2.094) * 0.5 + 0.5,
            np.sin(hue + 4.189) * 0.5 + 0.5,
        ], axis=-1)
    elif source == "procedural":
        g = np.sin(xx * 0.03 + yy * 0.02) * np.cos(xx * 0.02 - yy * 0.03) * 0.5 + 0.5
        img = np.stack([g, g * 0.6, 1.0 - g * 0.8], axis=-1)
    else:  # noise
        n = rng.standard_normal((hh, ww, 3)).astype(np.float32) * 0.35 + 0.5
        img = norm(n)
    return np.clip(img, 0.0, 1.0).astype(np.float32)


@method(
    id="493",
    name="Color Grading (OKLab)",
    category="filters",
    tags=["color", "grading", "oklab", "perceptual", "adjust", "tone", "animation", "expanded"],
    timeout=120,
    inputs={"image_in": "IMAGE"},
    outputs={"image": "IMAGE", "field": "FIELD"},
    params={
        "source": {"description": "standalone source when no upstream image is wired (noise/gradient/palette/rainbow/procedural/input_image)", "choices": ["noise", "gradient", "palette", "rainbow", "procedural"], "default": "noise"},
        "palette": {"description": "palette name for the palette source", "default": "vapor"},
        "exposure": {"description": "exposure in stops — scales perceptual lightness by 2^exposure", "min": -3.0, "max": 3.0, "default": 0.0},
        "contrast": {"description": "contrast around mid-grey (1.0 = unchanged)", "min": 0.2, "max": 3.0, "default": 1.0},
        "gamma": {"description": "gamma / tone curve (1.0 = linear, >1 lifts shadows)", "min": 0.3, "max": 3.0, "default": 1.0},
        "saturation": {"description": "chroma multiplier in OKLab (1.0 = unchanged)", "min": 0.0, "max": 3.0, "default": 1.0},
        "hue_rotate": {"description": "hue rotation in degrees", "min": -180.0, "max": 180.0, "default": 0.0},
        "temperature": {"description": "white balance: warm (+) / cool (-)", "min": -1.0, "max": 1.0, "default": 0.0},
        "tint": {"description": "green (-) / magenta (+) balance", "min": -1.0, "max": 1.0, "default": 0.0},
        "vignette": {"description": "radial lightness falloff strength (0 = none)", "min": 0.0, "max": 1.0, "default": 0.0},
        "invert": {"description": "invert luminance and chroma", "choices": ["no", "yes"], "default": "no"},
        "anim_mode": {"description": "animation mode (none/exposure_sweep/hue_cycle/breathe/vignette_pulse/temperature_drift)", "choices": ["none", "exposure_sweep", "hue_cycle", "breathe", "vignette_pulse", "temperature_drift"], "default": "none"},
        "anim_speed": {"description": "animation speed multiplier", "min": 0.1, "max": 3.0, "default": 1.0},
    },
)
def method_color_grade(out_dir: Path, seed: int, params=None):
    """Color Grading (OKLab) — perceptually-uniform brightness / contrast /
    saturation / hue / white-balance / vignette.

    Unlike naive RGB-space grading, every operation runs in OKLab so that
    saturation changes stay hue-neutral and contrast stays even across tones.
    With an upstream IMAGE wired in, that image is graded (Rule 12); otherwise
    a synthetic `source` is generated from the seed (static in 'none' mode).
    """
    try:
        if params is None:
            params = {}
        seed_all(seed)
        rng = np.random.default_rng(seed)

        hh, ww = int(H), int(W)
        anim_time = float(params.get("time", 0.0))
        anim_mode = str(params.get("anim_mode", "none"))
        anim_speed = float(params.get("anim_speed", 1.0))
        _t = 0.0 if anim_mode == "none" else anim_time * anim_speed

        source = str(params.get("source", "noise"))
        pal_name = str(params.get("palette", "vapor"))
        exposure = float(params.get("exposure", 0.0))
        contrast = float(params.get("contrast", 1.0))
        gamma = float(params.get("gamma", 1.0))
        saturation = float(params.get("saturation", 1.0))
        hue_rotate = float(params.get("hue_rotate", 0.0))
        temperature = float(params.get("temperature", 0.0))
        tint = float(params.get("tint", 0.0))
        vignette = float(params.get("vignette", 0.0))
        invert = str(params.get("invert", "no")) == "yes"

        # ── Animation: modulate grade params off the clock (smooth, no abs(sin) cusps) ──
        s = 0.5 + 0.5 * math.sin(_t)
        if anim_mode == "exposure_sweep":
            exposure = -2.0 + 4.0 * s
        elif anim_mode == "hue_cycle":
            hue_rotate = 180.0 * math.sin(_t)
        elif anim_mode == "breathe":
            contrast = 0.6 + 0.9 * s
            saturation = 0.4 + 1.3 * s
        elif anim_mode == "vignette_pulse":
            # full 0 -> 1 -> 0 swing (cos-based: peak at t=pi)
            vignette = 0.5 - 0.5 * math.cos(_t)
        elif anim_mode == "temperature_drift":
            temperature = math.sin(_t)
            tint = 0.5 * math.sin(_t * 0.5 + 1.0)

        # ── Resolve source image ──
        img = None
        wired = params.get("input_image", "")
        if wired:
            try:
                img = load_input(wired, ww, hh)
            except (FileNotFoundError, OSError, ValueError):
                img = None
        if img is None:
            img = _source_rgb(source, hh, ww, rng, _t, pal_name)
        img = np.clip(img, 0.0, 1.0).astype(np.float64)

        # ── To OKLab ──
        L, A, B = _rgb_to_oklab(img)

        # ── Grade in OKLab ──
        L = L * (2.0 ** exposure)            # exposure (stops)
        L = 0.5 + contrast * (L - 0.5)       # contrast around mid-grey
        L = np.power(np.clip(L, 1e-6, 1e6), 1.0 / gamma)  # gamma / tone

        if hue_rotate != 0.0:
            ang = math.radians(hue_rotate)
            ca, sa = math.cos(ang), math.sin(ang)
            na = ca * A - sa * B
            nb = sa * A + ca * B
            A, B = na, nb

        A = A * saturation
        B = B * saturation

        # white balance: warm = +yellow(+B)/+red(+A), cool = -B
        A = A + 0.04 * temperature + (-0.03) * tint
        B = B + 0.05 * temperature + 0.04 * tint

        if invert:
            L = 1.0 - L
            A = -A
            B = -B

        if vignette > 0.0:
            yy, xx = np.mgrid[0:hh, 0:ww]
            ny = (yy - hh / 2) / max(hh, ww)
            nx = (xx - ww / 2) / max(hh, ww)
            r2 = (nx * nx + ny * ny).astype(np.float64)
            vig = np.clip(1.0 - vignette * r2, 0.0, 1.0)
            L = L * vig

        # ── Back to sRGB ──
        out = _oklab_to_rgb(L, A, B)
        out = np.clip(out, 0.0, 1.0).astype(np.float32)

        # ── Side outputs (Rule 5: perceptual-lightness field) ──
        write_field(out_dir, np.clip(L, 0.0, 1.0).astype(np.float32))
        write_scalars(
            out_dir,
            exposure=float(exposure),
            contrast=float(contrast),
            gamma=float(gamma),
            saturation=float(saturation),
            hue_rotate=float(hue_rotate),
            temperature=float(temperature),
            tint=float(tint),
            vignette=float(vignette),
            mean_lightness=float(np.clip(L, 0, 1).mean()),
        )

        capture_frame("493", out)
        save(out, mn(493, f"Color Grading (OKLab) t={_t:.2f}"), out_dir)
        return out
    except Exception as exc:
        fallback = np.full((int(H), int(W), 3), 0.5, dtype=np.float32)
        save(fallback, mn(493, "Color Grading (OKLab)"), out_dir)
        print(f"[method_493] ERROR: {exc}")
        return fallback
