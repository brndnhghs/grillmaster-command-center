from __future__ import annotations

import math
from pathlib import Path

import numpy as np
from scipy.ndimage import map_coordinates, gaussian_filter

from ...core.registry import method
from ...core.utils import (
    save, norm, mn, seed_all, W, H, PALETTES, load_input, write_scalars,
)
from ...core.animation import capture_frame


@method(
    id='486',
    name='Radial & Spin Blur',
    category='filters',
    new_image_contract=True,
    tags=['motion', 'blur', 'zoom', 'spin', 'radial', 'rotational', 'optical', 'expanded', 'animation'],
    inputs={'image_in': 'IMAGE'},
    outputs={'image': 'IMAGE'},
    params={
        'source': {'description': 'source (noise/gradient/input_image/palette/rainbow/procedural)', 'default': 'noise'},
        'blur_type': {'description': 'motion kernel geometry (radial zoom / rotational spin)', 'choices': ['radial', 'rotational'], 'default': 'radial'},
        'length': {'description': 'blur strength in px (max displacement at the edge)', 'min': 0, 'max': 64, 'default': 14},
        'center_x': {'description': 'blur center x (0-1)', 'min': 0.0, 'max': 1.0, 'default': 0.5},
        'center_y': {'description': 'blur center y (0-1)', 'min': 0.0, 'max': 1.0, 'default': 0.5},
        'noise_amp': {'description': 'noise amplitude for generated sources', 'min': 0.1, 'max': 1.0, 'default': 0.35},
        'blur_sigma': {'description': 'gaussian blur sigma for noise source', 'min': 5, 'max': 80, 'default': 30},
        'palette': {'description': 'palette name for palette source', 'default': 'vapor'},
        'anim_mode': {'description': 'animation mode (none/zoom_pulse/spin_sweep/orbit)', 'choices': ['none', 'zoom_pulse', 'spin_sweep', 'orbit'], 'default': 'none'},
        'anim_speed': {'description': 'animation speed multiplier', 'min': 0.1, 'max': 5.0, 'default': 1.0},
    },
)
def method_radial_spin_blur(out_dir: Path, seed: int, params=None):
    """Radial & Spin Blur — zoom and rotation motion-blur kernels.

    A motion blur is a low-pass filter along a motion path (Heitz, Hill &
    Nehab, "A Low-Pass Filter for Real-Time Rendering of Multilayer Motion
    Blur", SIGGRAPH 2019). The common GPU ``Motion Blur`` node only does the
    *linear* (directional) case; this CPU node covers the two other standard
    screen-space motion kernels, which are absent from the pipeline:

    * ``radial``   — samples interpolated toward/away from a pivot (dolly /
      zoom blur). Classic for hyperspace / warp and "rush-in" shots.
    * ``rotational`` — samples rotated about a pivot (spin / turntable blur).

    Every geometry's edge displacement is bounded by ``length`` (px), so cost
    stays O(samples x pixels) and the two modes are directly comparable. The
    CPU path is the authoritative export. Animation offers a zoom breathe, a
    continuous spin sweep, and a pivot-orbit (the blur origin circles the
    frame). All animation is smooth (no cusps / no parameter cancellation).

    Params:
        source:     generated source type when no upstream image is wired
        blur_type:  radial / rotational
        length:     blur strength in px (edge displacement)
        center_x/y: blur pivot (0-1)
        time:       animation clock (0-6.28)
        anim_mode:  none / zoom_pulse / spin_sweep / orbit
        anim_speed: animation speed (0.1-5.0)
    """
    try:
        if params is None:
            params = {}
        anim_time = float(params.get("time", 0.0))
        anim_mode = str(params.get("anim_mode", "none"))
        anim_speed = float(params.get("anim_speed", 1.0))
        seed_all(seed)
        rng = np.random.default_rng(seed)

        blur_type = str(params.get("blur_type", "radial"))
        length = float(params.get("length", 14.0))
        cx = float(params.get("center_x", 0.5))
        cy = float(params.get("center_y", 0.5))
        noise_amp = float(params.get("noise_amp", 0.35))
        blur_sigma = float(params.get("blur_sigma", 30.0))
        pal_name = str(params.get("palette", "vapor"))
        source = str(params.get("source", "noise"))

        # ── Animation (rename t to avoid shadowing the time param) ──
        _t = anim_time * anim_speed
        length_eff = length
        spin0 = 0.0          # base rotation offset (rotational)
        ocx, ocy = cx, cy    # possibly orbited pivot
        if anim_mode == "zoom_pulse":
            # smooth breathe 0.5+0.5*sin; ranges ~0.12x..1.0x of length
            length_eff = max(2.0, length * (0.12 + 0.88 * (0.5 + 0.5 * math.sin(_t * 0.4))))
        elif anim_mode == "spin_sweep":
            # continuous rotation sweep — linear in _t, no cusps
            spin0 = _t * 0.6
        elif anim_mode == "orbit":
            # pivot circles the frame; affects both radial & rotational origin
            R = 0.28
            ocx = 0.5 + R * math.cos(_t * 0.8)
            ocy = 0.5 + R * math.sin(_t * 0.8)

        # ── Resolve source image (float32 [0,1], H×W×3) ──
        # A wired upstream image always overrides source generation (Rule #12).
        src = None
        wired_path = params.get("input_image", "")
        if wired_path:
            try:
                src = load_input(wired_path, int(W), int(H))
            except (FileNotFoundError, OSError):
                src = None
        if src is not None and src.ndim == 2:
            src = np.stack([src, src, src], axis=-1)
        if src is None:
            if source == "gradient":
                yy, xx = np.mgrid[:H, :W].astype(np.float32)
                r = norm(np.sqrt((xx - W / 2) ** 2 + (yy - H / 2) ** 2))
                src = np.stack([r, r * 0.7, 1 - r], axis=-1).clip(0, 1)
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
                g = np.sin(xx * 0.03 + yy * 0.02 + _t * 0.5) * \
                    np.cos(xx * 0.02 - yy * 0.03 + _t * 0.3) * 0.5 + 0.5
                src = np.stack([g, g * 0.6, 1 - g * 0.8], axis=-1).astype(np.float32)
            else:  # noise / input_image fallback
                n = rng.standard_normal((H, W, 3)).astype(np.float32) * noise_amp + 0.5
                n = gaussian_filter(n, sigma=(blur_sigma, blur_sigma, 0), mode='reflect')
                src = norm(n)

        src = np.clip(src, 0.0, 1.0).astype(np.float32)

        # ── Radial / Spin blur core ──
        result = _radial_spin_blur(src, blur_type, length_eff, ocx, ocy, spin0)

        result = np.clip(result, 0.0, 1.0).astype(np.float32)

        # Rule 4/13: record the effective (possibly animated) parameters
        write_scalars(out_dir, length_eff=float(length_eff), spin_eff=float(spin0))

        capture_frame("486", result)
        save(result, mn(486, f"Radial & Spin Blur t={_t:.2f}"), out_dir)
        return result
    except Exception as exc:
        fallback = np.full((H, W, 3), 0.5, dtype=np.float32)
        save(fallback, mn(486, "Radial & Spin Blur"), out_dir)
        print(f"[method_486] ERROR: {exc}")
        return fallback


def _radial_spin_blur(src: np.ndarray, blur_type: str, length: float,
                       cx: float, cy: float, spin0: float) -> np.ndarray:
    """Average ``n`` samples of ``src`` laid out along a radial or rotational
    path about the pivot ``(cx, cy)``. Edge displacement is bounded by
    ``length`` pixels, so cost stays O(n x pixels)."""
    H, W, _ = src.shape
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)

    cx_p = cx * W
    cy_p = cy * H
    maxr = math.hypot(max(cx_p, W - cx_p), max(cy_p, H - cy_p))
    maxr = max(maxr, 1.0)

    if length <= 0.0:
        return src.astype(np.float32)

    n = int(length) + 1
    n = max(2, min(n, 32))  # bound cost / memory

    accum = np.zeros((H, W, 3), dtype=np.float32)

    for i in range(n):
        f = (i / (n - 1)) - 0.5  # -0.5 .. 0.5
        if blur_type == "radial":
            z = (length / maxr) * f  # edge displacement == length*f px
            coords = np.empty((2, H, W), dtype=np.float32)
            coords[0] = cy_p + (yy - cy_p) * (1.0 - z)
            coords[1] = cx_p + (xx - cx_p) * (1.0 - z)
        else:  # rotational
            ang = (length / maxr) * f + spin0  # tangential edge disp == length*f px
            ca, sa = math.cos(ang), math.sin(ang)
            dy = yy - cy_p
            dx = xx - cx_p
            coords = np.empty((2, H, W), dtype=np.float32)
            coords[0] = cy_p + dx * sa + dy * ca
            coords[1] = cx_p + dx * ca - dy * sa

        for c in range(3):
            accum[:, :, c] += map_coordinates(
                src[:, :, c], coords, order=1, mode='reflect'
            )

    accum /= n
    return accum.astype(np.float32)
