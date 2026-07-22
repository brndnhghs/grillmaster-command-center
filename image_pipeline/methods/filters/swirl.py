from __future__ import annotations
import math
import random
from io import BytesIO
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageOps

from ...core.registry import method
from ...core.utils import save, norm, mn, seed_all, BG_DEFAULT, W, H, PALETTES, quantize_to_palette, load_input
from ...core.animation import capture_frame
from image_pipeline.core.spatial import sparam

try:
    import cv2
    _has_cv2 = True
except ImportError:
    _has_cv2 = False

@method(
    id="74",
    name="Swirl Displacement",
    category="filters",
    new_image_contract=True,
    tags=["warp", "fast", "expanded", "animation"],
    params={
        "displacement": {"description": "displacement type (swirl/pinch/bulge/twist/ripple/fisheye/wave/kaleidoscope/spiralize)", "default": "swirl"},
        "source": {"description": "source (noise/gradient/input_image/palette/rainbow/procedural)", "default": "noise"},
        "colormode": {"description": "color mode (source/palette/heatmap/spectral/fire/ice/dual_layer)", "default": "source"},
        "palette": {"description": "color palette name", "default": "vapor"},
        "strength": {"spatial": True, "description": "displacement strength", "min": 0.0, "max": 0.5, "default": 0.01},
        "blur_sigma": {"description": "gaussian blur sigma for noise source", "min": 1, "max": 50, "default": 15},
        "noise_amp": {"description": "noise amplitude", "min": 0.1, "max": 1.0, "default": 0.3},
        "frequency": {"description": "spatial frequency for wave/ripple", "min": 0.01, "max": 0.5, "default": 0.05},
        "amplitude": {"description": "wave amplitude for displacement", "min": 1.0, "max": 100.0, "default": 20.0},
        "rotation": {"spatial": True, "description": "global rotation offset", "min": 0.0, "max": 6.2832, "default": 0.0},
        "zoom": {"description": "zoom factor for kaleidoscope", "min": 0.5, "max": 5.0, "default": 1.0},
        "segments": {"description": "symmetry segments for kaleidoscope", "min": 2, "max": 32, "default": 6},
        "anim_mode": {"description": "animation mode (none/morph/speed_pulse/rotation_cycle)", "choices": ["none", "morph", "speed_pulse", "rotation_cycle"], "default": "none"},
        "anim_speed": {"description": "animation speed multiplier", "min": 0.0, "max": 5.0, "default": 1.0}}
)
def method_swirl(out_dir: Path, seed: int, params=None):
    """Swirl Displacement — geometric image remapping with 9 displacement types and animation.

    Applies geometric remapping to a source image using polar/coordinate
    transforms. Supports animated morphing between displacement types,
    speed pulsing, and rotation cycling.

    Parameters:
        displacement (str): Displacement type (swirl/pinch/bulge/twist/ripple/fisheye/wave/kaleidoscope/spiralize)
        source (str): Source image type (noise/gradient/input_image/palette/rainbow/procedural)
        colormode (str): Color mode (source/palette/heatmap/spectral/fire/ice/dual_layer)
        palette (str): Color palette name
        strength (float): Displacement strength (0-0.5, default 0.01)
        blur_sigma (float): Gaussian blur sigma for noise source (1-50, default 15)
        noise_amp (float): Noise amplitude (0.1-1.0, default 0.3)
        frequency (float): Spatial frequency for wave/ripple (0.01-0.5, default 0.05)
        amplitude (float): Wave amplitude for displacement (1-100, default 20)
        rotation (float): Global rotation offset (0-2pi, default 0)
        zoom (float): Zoom factor for kaleidoscope (0.5-5.0, default 1.0)
        segments (int): Symmetry segments for kaleidoscope (2-32, default 6)
        anim_mode (str): Animation mode (none/morph/speed_pulse/rotation_cycle)
        anim_speed (float): Animation speed multiplier (0-5, default 1.0)
        time (float): Animation time in radians (0-2pi, default 0.0)
    """
    if params is None:
        params = {}
    import cv2
    seed_all(seed)
    rng = np.random.default_rng(seed)

    disp_type = params.get("displacement", "swirl")
    source = params.get("source", "noise")
    cmode = params.get("colormode", "source")
    pal_name = params.get("palette", "vapor")
    strength = sparam(params, "strength", 0.01)
    blur_sigma = float(params.get("blur_sigma", 15))
    noise_amp = float(params.get("noise_amp", 0.3))
    freq = float(params.get("frequency", 0.05))
    amp = float(params.get("amplitude", 20.0))
    rot = sparam(params, "rotation", 0.0)
    zoom = float(params.get("zoom", 1.0))
    segs = int(params.get("segments", 6))
    anim_mode = params.get("anim_mode", "none")
    anim_speed = float(params.get("anim_speed", 1.0))
    anim_time = float(params.get("time", 0.0))
    t = anim_time * anim_speed
    from ...core.utils import PALETTES

    # Animation: morph between displacement types
    if anim_mode == "morph":
        disp_types = ["swirl", "pinch", "bulge", "twist", "ripple", "fisheye", "wave", "kaleidoscope", "spiralize"]
        idx = int(t * 0.3) % len(disp_types)
        disp_type = disp_types[idx]

    # Animation: modulate strength
    if anim_mode == "speed_pulse":
        strength = strength * (0.5 + 0.5 * math.sin(t * 0.5))

    # Animation: modulate rotation
    if anim_mode == "rotation_cycle":
        rot = rot + t * 0.3

    # ── Generate source image ──
    def _make_source():
        _inp = params.get("_input_image")
        if _inp is not None:
            return _inp
        elif source == "noise":
            noise = rng.standard_normal((H, W, 3)).astype(np.float32) * noise_amp + 0.5
            noise = cv2.GaussianBlur(noise, (0, 0), sigmaX=blur_sigma, sigmaY=blur_sigma)
            noise = norm(noise)
            return noise
        elif source == "gradient":
            yy, xx = np.mgrid[:H, :W].astype(np.float32)
            r = np.sqrt((xx - W/2)**2 + (yy - H/2)**2)
            g = norm(r)
            return np.stack([g, g * 0.7, 1 - g], axis=-1).clip(0, 1)
        elif source == "palette":
            pal = PALETTES.get(pal_name, PALETTES["vapor"])
            yy, xx = np.mgrid[:H, :W].astype(np.float32)
            r = np.sqrt((xx - W/2)**2 + (yy - H/2)**2)
            g = norm(r)
            idx = (g * (len(pal) - 1)).astype(np.int32)
            pal_arr = np.array(pal, dtype=np.float32) / 255.0
            return pal_arr[idx]
        elif source == "rainbow":
            yy, xx = np.mgrid[:H, :W].astype(np.float32)
            r = np.sqrt((xx - W/2)**2 + (yy - H/2)**2)
            g = norm(r)
            hue = g * 2 * math.pi
            return np.stack([
                np.sin(hue) * 0.5 + 0.5,
                np.sin(hue + 2.094) * 0.5 + 0.5,
                np.sin(hue + 4.189) * 0.5 + 0.5
            ], axis=-1).astype(np.float32)
        elif source == "procedural":
            yy, xx = np.mgrid[:H, :W].astype(np.float32)
            g = np.sin(xx * 0.03 + yy * 0.02 + t * 0.5) * \
                np.cos(xx * 0.02 - yy * 0.03 + t * 0.3) * 0.5 + 0.5
            return np.stack([g, g * 0.6, 1 - g * 0.8], axis=-1).astype(np.float32)
        else:
            noise = rng.standard_normal((H, W, 3)).astype(np.float32) * noise_amp + 0.5
            noise = cv2.GaussianBlur(noise, (0, 0), sigmaX=blur_sigma, sigmaY=blur_sigma)
            return norm(noise)

    # ── Build remap ──
    yy, xx = np.mgrid[:H, :W].astype(np.float32)
    cx, cy = W / 2.0, H / 2.0
    yd = yy - cy
    xd = xx - cx
    r = np.sqrt(xd**2 + yd**2)
    theta = np.arctan2(yd, xd) + rot
    max_r = np.sqrt(cx**2 + cy**2)

    if disp_type == "swirl":
        new_theta = theta + r * strength * (1.0 + 0.3 * math.sin(t))
        src_x = cx + r * np.cos(new_theta)
        src_y = cy + r * np.sin(new_theta)

    elif disp_type == "pinch":
        factor = 1.0 / (1.0 + strength * 20.0 * (1.0 - r / max_r))
        src_x = cx + xd * factor
        src_y = cy + yd * factor

    elif disp_type == "bulge":
        factor = 1.0 + strength * 30.0 * (1.0 - r / max_r)
        src_x = cx + xd * factor
        src_y = cy + yd * factor

    elif disp_type == "twist":
        angle_strength = strength * 10.0 * (1.0 - r / max_r) * (1.0 + 0.2 * math.sin(t))
        new_theta = theta + angle_strength
        src_x = cx + r * np.cos(new_theta)
        src_y = cy + r * np.sin(new_theta)

    elif disp_type == "ripple":
        # Concentric sine wave displacement
        ripple = np.sin(r * freq * 10.0 + t) * amp * strength * 20.0
        src_x = xx + xd / (r + 1e-6) * ripple
        src_y = yy + yd / (r + 1e-6) * ripple

    elif disp_type == "fisheye":
        factor = r / (max_r + 1e-6)
        new_r = r * (1.0 + factor * strength * 10.0)
        new_r = np.clip(new_r, 0, max_r * 1.5)
        src_x = cx + new_r * np.cos(theta)
        src_y = cy + new_r * np.sin(theta)

    elif disp_type == "wave":
        # Sine wave displacement in both axes
        wave_x = np.sin(yy * freq * 5.0 + t) * amp * strength * 10.0
        wave_y = np.cos(xx * freq * 5.0 + t * 0.7) * amp * strength * 10.0
        src_x = xx + wave_x
        src_y = yy + wave_y

    elif disp_type == "kaleidoscope":
        # Fold into N wedges
        angle_per_seg = 2 * math.pi / segs
        folded = theta % angle_per_seg
        # Mirror within each wedge
        folded = np.where(folded > angle_per_seg / 2, angle_per_seg - folded, folded)
        new_theta = folded + math.floor(segs / 2) * angle_per_seg
        new_r = r * zoom
        src_x = cx + new_r * np.cos(new_theta + rot + t * 0.1)
        src_y = cy + new_r * np.sin(new_theta + rot + t * 0.1)

    elif disp_type == "spiralize":
        # Logarithmic spiral
        spiral_theta = theta + r * strength * 5.0 * (1.0 + 0.2 * math.sin(t))
        new_r = r * (1.0 + strength * 2.0 * np.sin(theta * 3 + t))
        src_x = cx + new_r * np.cos(spiral_theta)
        src_y = cy + new_r * np.sin(spiral_theta)

    else:
        src_x = cx + r * np.cos(theta)
        src_y = cy + r * np.sin(theta)

    # Clamp to valid range
    src_x = np.clip(src_x, 0, W - 1).astype(np.float32)
    src_y = np.clip(src_y, 0, H - 1).astype(np.float32)

    # ── Sample ──
    src_img = _make_source()
    result = cv2.remap(src_img, src_x, src_y, cv2.INTER_LINEAR)

    # ── Color mode post-processing ──
    if cmode == "palette":
        pal = PALETTES.get(pal_name, PALETTES["vapor"])
        gray = np.mean(result, axis=-1)
        idx = (norm(gray) * (len(pal) - 1)).astype(np.int32)
        pal_arr = np.array(pal, dtype=np.float32) / 255.0
        result = pal_arr[idx]
    elif cmode == "heatmap":
        from matplotlib import cm
        gray = np.mean(result, axis=-1)
        result = cm.inferno(norm(gray))[:, :, :3].astype(np.float32)
    elif cmode == "spectral":
        from matplotlib import cm
        gray = np.mean(result, axis=-1)
        result = cm.nipy_spectral(norm(gray))[:, :, :3].astype(np.float32)
    elif cmode == "fire":
        gray = norm(np.mean(result, axis=-1))
        result = np.stack([np.clip(gray * 1.5, 0, 1), gray * 0.6, gray * 0.2], axis=-1).astype(np.float32)
    elif cmode == "ice":
        gray = norm(np.mean(result, axis=-1))
        result = np.stack([gray * 0.2, gray * 0.5, 0.5 + gray * 0.5], axis=-1).astype(np.float32)
    elif cmode == "dual_layer":
        from matplotlib import cm
        gray = norm(np.mean(result, axis=-1))
        hi = gray > 0.5
        lo = gray <= 0.5
        base = np.zeros((H, W, 3), dtype=np.float32)
        base[lo] = cm.viridis(gray[lo] * 2)[:, :3]
        base[hi] = cm.inferno((gray[hi] - 0.5) * 2)[:, :3]
        result = base.astype(np.float32)

    result = np.clip(result, 0, 1).astype(np.float32)
    capture_frame("74", result)
    save(result, mn(74, "Swirl Displacement"), out_dir)


