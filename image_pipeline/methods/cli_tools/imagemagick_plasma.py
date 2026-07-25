from __future__ import annotations
import math
import subprocess
from pathlib import Path

import numpy as np
from PIL import Image

from ...core.registry import method
from ...core.utils import save, mn, seed_all, W, H
from ...core.animation import capture_frame


@method(id="44", name="ImageMagick Plasma", category="cli_tools", tags=["imagemagick", "expanded"],
        params={
            "plasma_type": {"description": "plasma type for convert", "default": "fractal"},
            "oil_paint": {"description": "oil paint effect radius", "min": 0, "max": 20, "default": 3},
            "blur": {"description": "Gaussian blur radius", "default": "0x1"},
            "min_bytes": {"description": "minimum output file size to accept", "min": 100, "max": 100000, "default": 1000},
            "anim_mode": {"description": "animation mode", "choices": ["none", "plasma_pulse", "blur_cycle", "tile_cycle", "seed_morph", "oil_shock"], "default": "none"},
            "anim_speed": {"description": "animation speed multiplier", "min": 0.1, "max": 5.0, "default": 1.0},
        })
def method_gmic_plasma(out_dir: Path, seed: int, params=None):
    """Generate a fractal plasma image using ImageMagick convert, with PIL fallback.

    Uses ImageMagick's `convert` CLI to generate a fractal plasma image with
    optional oil paint and blur effects. Falls back to a PIL-generated noise
    image if convert is unavailable.

    Args:
        out_dir: Output directory for the generated image.
        seed: Random seed for deterministic output.
        params: Dict with keys:
            plasma_type: plasma type for convert (fractal/tile)
            oil_paint: oil paint effect radius (0-20)
            blur: Gaussian blur radius (e.g. \"0x1\")
            min_bytes: minimum output file size to accept (100-100000)
            time: animation time in radians (0-6.28)
            anim_mode: animation mode (none/plasma_pulse/blur_cycle/tile_cycle/seed_morph/oil_shock)
            anim_speed: animation speed multiplier (0.1-5.0)
    """
    if params is None:
        params = {}
    anim_time = float(params.get("time", 0.0))
    anim_mode = params.get("anim_mode", "none")
    anim_speed = float(params.get("anim_speed", 1.0))
    seed_all(seed)
    rng = np.random.default_rng(seed)

    plasma_type = params.get("plasma_type", "fractal")
    oil_paint = int(params.get("oil_paint", 3))
    blur = params.get("blur", "0x1")
    min_bytes = int(params.get("min_bytes", 1000))

    # ── Animation ──
    t = anim_time * anim_speed
    if anim_mode == "plasma_pulse":
        oil_paint = max(0, int(oil_paint * (0.5 + 0.5 * math.sin(t * 0.5))))
    elif anim_mode == "blur_cycle":
        blur_val = 0.5 + 2.0 * (0.5 + 0.5 * math.sin(t * 0.3))
        blur = f"0x{blur_val:.1f}"
    elif anim_mode == "tile_cycle":
        plasma_type = "tile" if int(t * 0.3) % 2 == 0 else "fractal"
    elif anim_mode == "seed_morph":
        # Per-frame animation seed: drives both ImageMagick's -seed and
        # creates a fundamentally different plasma each frame. Seed value
        # also modulates paint and blur for combined visual evolution.
        frame_seed = seed + int(t * 10000)
        oil_paint = max(0, int(oil_paint * (0.3 + 0.7 * (0.5 + 0.5 * math.sin(t * 0.2)))))
        blur_val = 0.5 + 3.0 * (0.5 + 0.5 * math.sin(t * 0.15))
        blur = f"0x{blur_val:.1f}"
    elif anim_mode == "oil_shock":
        # Multi-param shockwave: seed, paint, and blur all oscillate at
        # different frequencies for a continuously evolving plasma morph
        frame_seed = seed + int(t * 7919)
        oil_paint = max(0, int(oil_paint * (0.1 + 0.9 * (0.5 + 0.5 * math.sin(t * 0.35)))))
        blur_val = 0.5 + 4.0 * (0.5 + 0.5 * math.sin(t * 0.21))
        blur = f"0x{blur_val:.1f}"

    # IMv7 compatibility: use `magick` with `-paint` instead of deprecated `convert` + `-oil-paint`
    _im_cmd = ["magick", "-size", f"{W}x{H}"]
    if anim_mode in ("seed_morph", "oil_shock"):
        _im_cmd += ["-seed", str(frame_seed)]
    _im_cmd += [f"plasma:{plasma_type}", "-paint", str(oil_paint), "-blur", blur]

    # ── Check for magick/convert binary ──
    have_convert = False
    try:
        subprocess.run(["magick", "--version"], capture_output=True, timeout=5)
        have_convert = True
    except (FileNotFoundError, subprocess.TimeoutExpired):
        try:
            subprocess.run(["convert", "--version"], capture_output=True, timeout=5)
            have_convert = True
            _im_cmd[0] = "convert"
            # convert uses -oil-paint instead of -paint
            _im_cmd = [c if c != "-paint" else "-oil-paint" for c in _im_cmd]
        except (FileNotFoundError, subprocess.TimeoutExpired):
            have_convert = False

    if not have_convert:
        # PIL fallback with per-frame seed for animation
        _frame_seed = seed + int(anim_time * 10000) if anim_mode != "none" else seed
        rng = np.random.default_rng(_frame_seed)
        fallback = rng.random((H, W, 3)).astype(np.float32) * 0.1 + 0.05
        capture_frame("46", fallback)
        save(fallback, mn(46, "ImageMagick Plasma"), out_dir)
        return

    # ── Render via magick ──
    outpath = str(out_dir / mn(46, "ImageMagick Plasma"))
    try:
        subprocess.run(
            _im_cmd + [outpath],
            capture_output=True, text=True, timeout=30,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass

    if (out_dir / mn(46, "ImageMagick Plasma")).exists() and (out_dir / mn(46, "ImageMagick Plasma")).stat().st_size > min_bytes:
        capture_frame("46", out_dir / mn(46, "ImageMagick Plasma"))
        print(f"  ✓ {mn(46, 'ImageMagick Plasma')}  ({(out_dir / mn(46, 'ImageMagick Plasma')).stat().st_size // 1024} KB)")
    else:
        _frame_seed = seed + int(anim_time * 10000) if anim_mode != "none" else seed
        rng = np.random.default_rng(_frame_seed)
        fallback = rng.random((H, W, 3)).astype(np.float32) * 0.1 + 0.05
        capture_frame("46", fallback)
        save(fallback, mn(46, "ImageMagick Plasma"), out_dir)
