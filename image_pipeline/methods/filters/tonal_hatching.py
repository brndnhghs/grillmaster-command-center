from __future__ import annotations
import math
import random
from pathlib import Path

import numpy as np
from PIL import Image

from ...core.registry import method
from ...core.utils import save, norm, mn, seed_all, W, H, PALETTES, load_input
from ...core.animation import capture_frame
from image_pipeline.core.spatial import sparam

try:
    import cv2
    _has_cv2 = True
except ImportError:
    _has_cv2 = False


@method(
    id="339",
    name="Tonal Hatching",
    category="filters",
    new_image_contract=True,
    tags=["pen-and-ink", "hatching", "stippling", "abstraction", "expanded", "animation"],
    inputs={"image_in": "IMAGE"},
    outputs={"image": "IMAGE"},
    params={
        "source": {"description": "luminance source (noise/gradient/input_image/palette/rainbow/procedural)", "default": "noise"},
        "paper": {"description": "paper tone (light/dark)", "choices": ["light", "dark"], "default": "light"},
        "ink_tone": {"description": "ink color (black/sepia/indigo/iron_gall)", "choices": ["black", "sepia", "indigo", "iron_gall"], "default": "black"},
        "spacing": {"description": "line spacing in px (smaller = denser hatching)", "min": 4, "max": 24, "default": 10},
        "line_width": {"description": "line thickness in px", "min": 0.5, "max": 5.0, "default": 1.5},
        "layers": {"description": "number of crosshatch layers (1-4)", "min": 1, "max": 4, "default": 3},
        "angle": {"description": "base hatch orientation in degrees", "min": 0, "max": 180, "default": 45},
        "contrast": {"spatial": True, "description": "luminance gamma (higher = more ink in midtones)", "min": 0.3, "max": 3.0, "default": 1.0},
        "source_blur": {"description": "pre-smooth sigma before hatching", "min": 0.0, "max": 30.0, "default": 6.0},
        "noise_amp": {"description": "noise amplitude for generated sources", "min": 0.1, "max": 1.0, "default": 0.35},
        "blur_sigma": {"description": "gaussian blur sigma for noise source", "min": 5, "max": 80, "default": 30},
        "palette": {"description": "palette name for palette source", "default": "vapor"},
        "anim_mode": {"description": "animation mode (none/flow/weave/breathe)", "choices": ["none", "flow", "weave", "breathe"], "default": "none"},
        "anim_speed": {"description": "animation speed multiplier", "min": 0.1, "max": 5.0, "default": 1.0},
        "time": {"description": "animation time in radians", "min": 0.0, "max": 6.2832, "default": 0.0},
    },
)
def method_hatching(out_dir: Path, seed: int, params=None):
    """Tonal Hatching — pen-and-ink illustration via crosshatched line screening.

    Implements the classic non-photorealistic rendering (NPR) technique from
    Winkenbach & Salesin, "Computer-Generated Pen-and-Ink Illustration" (SIGGRAPH
    1994). The input luminance L is mapped to an *ink coverage fraction* f = 1 - L.
    As f grows, progressively more hatch layers are screened on top of each other,
    each layer rotated by 180/layers degrees, exactly like an artist stacking
    parallel, then cross, then diagonal strokes to deepen shadow. The darkest
    tones collapse to solid ink. The result is the familiar pen-and-ink look:
    white paper, dark strokes that thicken and crosshatch into shadow.

    This CPU path is the authoritative export. The geometry is fully
    vectorized (rotated-stripe masking) and deterministic in the seed.

    Params:
        source:      luminance source (noise/gradient/input_image/palette/rainbow/procedural)
        paper:       paper tone (light/dark)
        ink_tone:    ink color (black/sepia/indigo/iron_gall)
        spacing:     line spacing in px (4-24, default 10)
        line_width:  line thickness in px (0.5-5, default 1.5)
        layers:      number of crosshatch layers (1-4, default 3)
        angle:       base hatch orientation in degrees (0-180, default 45)
        contrast:    luminance gamma (0.3-3, default 1)
        source_blur: pre-smooth sigma before hatching (0-30, default 6)
        noise_amp:   noise amplitude for generated sources (0.1-1, default 0.35)
        blur_sigma:  gaussian blur sigma for noise source (5-80, default 30)
        palette:     palette name for palette source
        anim_mode:   animation mode (none/flow/weave/breathe)
        anim_speed:  animation speed multiplier (0.1-5, default 1)
        time:        animation time in radians (0-6.28, default 0)
    """
    if params is None:
        params = {}
    seed_all(seed)
    rng = np.random.default_rng(seed)

    source = str(params.get("source", "noise"))
    paper = str(params.get("paper", "light"))
    ink_tone = str(params.get("ink_tone", "black"))
    spacing = float(params.get("spacing", 10))
    line_width = float(params.get("line_width", 1.5))
    n_layers = int(params.get("layers", 3))
    angle = float(params.get("angle", 45))
    contrast = sparam(params, "contrast", 1.0)
    source_blur = float(params.get("source_blur", 6.0))
    noise_amp = float(params.get("noise_amp", 0.35))
    blur_sigma = float(params.get("blur_sigma", 30))
    pal_name = str(params.get("palette", "vapor"))
    anim_mode = str(params.get("anim_mode", "none"))
    anim_speed = float(params.get("anim_speed", 1.0))
    t = float(params.get("time", 0.0)) * anim_speed

    # ── Animation: modulate geometry (every mode is smooth, no cusps) ──
    spacing_eff = spacing
    angle_eff = angle
    lw_eff = line_width
    contrast_eff = contrast
    if anim_mode == "flow":
        spacing_eff = spacing * (1.0 + 0.3 * math.sin(t))
        angle_eff = angle + 10.0 * math.sin(t * 0.5)
    elif anim_mode == "weave":
        angle_eff = angle + 30.0 * math.sin(t * 0.7)
    elif anim_mode == "breathe":
        lw_eff = line_width * (0.5 + 0.5 * math.sin(t))
        contrast_eff = contrast * (1.0 + 0.3 * math.sin(t * 0.8))

    spacing_eff = max(3.0, spacing_eff)
    lw_eff = min(spacing_eff - 1.0, max(0.5, lw_eff))
    n_layers = max(1, min(4, n_layers))

    # ── Build luminance source ──
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    wired = params.get("_input_image")
    if wired is not None:
        arr = np.asarray(wired, dtype=np.float32)
        if arr.ndim == 3 and arr.shape[-1] >= 3:
            lum = arr[..., :3].mean(axis=-1)
        else:
            lum = np.asarray(arr, dtype=np.float32).reshape(H, W)
        lum = lum.clip(0.0, 1.0)
    elif source == "gradient":
        r = np.sqrt((xx - W / 2) ** 2 + (yy - H / 2) ** 2)
        g = norm(r)
        lum = g
    elif source == "palette":
        pal = PALETTES.get(pal_name, PALETTES.get("vapor", [(20, 20, 20), (235, 235, 235)]))
        pal_arr = np.array(pal, dtype=np.float32) / 255.0
        if _has_cv2:
            n = rng.standard_normal((H, W, 3)).astype(np.float32) * noise_amp + 0.5
            n = cv2.GaussianBlur(n, (0, 0), sigmaX=blur_sigma, sigmaY=blur_sigma)
        else:
            n = rng.standard_normal((H, W, 1)).astype(np.float32) * noise_amp + 0.5
        n = norm(n[..., 0]) if n.ndim == 3 else norm(n)
        idx = (n * (len(pal_arr) - 1)).astype(np.int32)
        lum = pal_arr[idx].mean(axis=-1)
    elif source == "rainbow":
        r = np.sqrt((xx - W / 2) ** 2 + (yy - H / 2) ** 2)
        hue = norm(r) * 2 * math.pi
        rr = np.sin(hue) * 0.5 + 0.5
        gg = np.sin(hue + 2.094) * 0.5 + 0.5
        bb = np.sin(hue + 4.189) * 0.5 + 0.5
        lum = (rr * 0.299 + gg * 0.587 + bb * 0.114)
    elif source == "procedural":
        g = np.sin(xx * 0.03 + yy * 0.02) * np.cos(xx * 0.02 - yy * 0.03) * 0.5 + 0.5
        lum = g.astype(np.float32)
    else:  # noise
        if _has_cv2:
            n = rng.standard_normal((H, W, 3)).astype(np.float32) * noise_amp + 0.5
            n = cv2.GaussianBlur(n, (0, 0), sigmaX=blur_sigma, sigmaY=blur_sigma)
            n = norm(n)
            lum = n.mean(axis=-1)
        else:
            n = rng.standard_normal((H, W)).astype(np.float32) * noise_amp + 0.5
            lum = norm(n)

    # Pre-smooth the luminance field (pen-and-ink reads smoothed tone)
    if source_blur > 0.0 and _has_cv2:
        lum = cv2.GaussianBlur(lum, (0, 0), sigmaX=source_blur, sigmaY=source_blur)
        lum = norm(lum)

    # Contrast / gamma on luminance (ink fraction = 1 - L)
    lum = np.clip(lum, 0.0, 1.0) ** contrast_eff
    f = (1.0 - lum).astype(np.float32)  # ink coverage fraction 0..1

    # ── Screen the hatch layers (Winkenbach & Salesin screening) ──
    a = math.radians(angle_eff)
    ca, sa = math.cos(a), math.sin(a)
    # rotate coords so stripes run along the rotated x-axis
    xr = xx * ca - yy * sa
    yr = xx * sa + yy * ca

    ink = np.zeros((H, W), dtype=bool)
    for i in range(n_layers):
        thr = (i + 1) / (n_layers + 1.0)  # f threshold at which this layer switches on
        layer_ang = angle_eff + i * (180.0 / n_layers)
        la = math.radians(layer_ang)
        lca, lsa = math.cos(la), math.sin(la)
        lxr = xx * lca - yy * lsa
        lyr = xx * lsa + yy * lca
        phase = np.mod(lyr + 0.5, spacing_eff)
        line = phase < lw_eff
        ink |= (f > thr) & line
    # Darkest tones collapse to solid ink
    ink |= f > 0.9

    # ── Composite ink over paper ──
    ink_f = ink.astype(np.float32)
    if paper == "dark":
        paper_color = np.array([0.05, 0.05, 0.06], dtype=np.float32)
    else:
        paper_color = np.array([0.97, 0.96, 0.92], dtype=np.float32)

    ink_colors = {
        "black": np.array([0.09, 0.09, 0.11], dtype=np.float32),
        "sepia": np.array([0.28, 0.17, 0.09], dtype=np.float32),
        "indigo": np.array([0.12, 0.12, 0.24], dtype=np.float32),
        "iron_gall": np.array([0.15, 0.16, 0.13], dtype=np.float32),
    }
    ink_color = ink_colors.get(ink_tone, ink_colors["black"])

    result = (paper_color[None, None, :] * (1.0 - ink_f[:, :, None])
              + ink_color[None, None, :] * ink_f[:, :, None]).astype(np.float32)

    # Scalar readout for the node graph sidecar
    from ...core.utils import write_scalars
    write_scalars(out_dir, ink_coverage=float(ink_f.mean()), layers=float(n_layers))

    result = np.clip(result, 0.0, 1.0).astype(np.float32)
    capture_frame("339", result)
    save(result, mn(339, f"Tonal Hatching a={angle_eff:.1f}"), out_dir)
