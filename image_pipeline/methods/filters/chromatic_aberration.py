from __future__ import annotations
import math
from pathlib import Path

import numpy as np
from scipy.ndimage import map_coordinates

from ...core.registry import method
from ...core.utils import (save, norm, mn, seed_all, W, H, PALETTES, load_input)
from ...core.animation import capture_frame

try:
    import cv2
    _has_cv2 = True
except ImportError:
    _has_cv2 = False


@method(
    id="417",
    name="Chromatic Aberration",
    category="filters",
    new_image_contract=True,
    tags=["post-process", "lens", "aberration", "color-fringe", "fast", "expanded", "animation"],
    inputs={"image_in": "IMAGE"},
    outputs={"image": "IMAGE", "luminance": "FIELD"},
    params={
        "source": {"description": "source when no image is wired (input_image/gradient/rainbow/palette/noise/checker/procedural)", "default": "gradient"},
        "amount": {"description": "max lateral RGB split in px at the frame edge (fringe strength)", "min": 0, "max": 60, "default": 20},
        "curve": {"description": "radial falloff exponent of the split (r^k); 1=linear, 2=physical lateral CA", "min": 1.0, "max": 4.0, "default": 2.0},
        "barrel": {"description": "barrel(+)/pincushion(-) radial lens distortion coefficient", "min": -0.4, "max": 0.4, "default": 0.0},
        "vignette": {"description": "vignette darkening at the frame edge (0=off, 1=strong)", "min": 0.0, "max": 1.0, "default": 0.0},
        "center_drift": {"description": "orbit radius of the aberration center for spin mode (0-1 of half-width)", "min": 0.0, "max": 1.0, "default": 0.4},
        "noise_amp": {"description": "noise amplitude for generated sources", "min": 0.1, "max": 1.0, "default": 0.5},
        "blur_sigma": {"description": "gaussian blur sigma for noise source", "min": 5, "max": 80, "default": 30},
        "palette": {"description": "palette name for palette source", "default": "vapor"},
        "anim_mode": {"description": "animation mode (none/pulse/breathe/spin)", "choices": ["none", "pulse", "breathe", "spin"], "default": "none"},
        "anim_speed": {"description": "animation speed multiplier", "min": 0.1, "max": 5.0, "default": 1.0},
    },
)
def method_chromatic_aberration(out_dir: Path, seed: int, params=None):
    """Chromatic Aberration — radial RGB channel separation (lateral CA post-process).

    Chromatic aberration is the failure of a lens to focus all wavelengths to
    the same point. In real-time rendering it is reproduced as a *lateral*
    (transverse) aberration: the R, G and B samples are taken at slightly
    different radial distances from the optical center, so colored fringes
    appear wherever the image contains high-frequency radial structure
    (edges, corners, fine grids). It is a standard entry in every real-time
    post-processing stack (Unity's "Chromatic Aberration" and Unreal's
    "Chromatic Aberration" lens effects), and a physically-motivated model
    makes the split grow with radius as r^k (k=2 reproduces the geometric
    lateral-CA falloff).

    The same displacement field also drives an optional barrel/pincushion
    distortion and a vignette, and three animation modes (pulse / breathe /
    spin) evolve the effect over time.

    A wired upstream image (image_in) ALWAYS overrides source generation
    (Rule #12). When unwired, a generated demo pattern is used so the node is
    self-contained.

    Params:
        source:      generated source type when nothing is wired
        amount:      max RGB split in px at the frame edge
        curve:       radial falloff exponent (r^k)
        barrel:      barrel(+)/pincushion(-) distortion
        vignette:    edge darkening
        center_drift: spin-mode orbit radius
        time:        animation clock (0-6.28)
        anim_mode:   none / pulse / breathe / spin
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

        amount = float(params.get("amount", 14))
        amount = max(0.0, min(60.0, amount))
        curve = float(params.get("curve", 2.0))
        curve = max(1.0, min(4.0, curve))
        barrel = float(params.get("barrel", 0.0))
        barrel = max(-0.4, min(0.4, barrel))
        vignette = float(params.get("vignette", 0.0))
        vignette = max(0.0, min(1.0, vignette))
        center_drift = float(params.get("center_drift", 0.4))
        center_drift = max(0.0, min(1.0, center_drift))
        noise_amp = float(params.get("noise_amp", 0.5))
        blur_sigma = float(params.get("blur_sigma", 30))
        pal_name = str(params.get("palette", "vapor"))
        source = str(params.get("source", "gradient"))

        # ── Animation (rename t to avoid shadowing the time param) ──
        _t = anim_time * anim_speed
        if anim_mode == "pulse":
            # Wide fringe-strength swing (0.1..1.0x) — combined with a center
            # orbit below so the whole fringe pattern visibly travels.
            amount = amount * (0.1 + 0.9 * (0.5 + 0.5 * math.sin(_t * 0.5)))
        elif anim_mode == "breathe":
            barrel = barrel + 0.25 * math.sin(_t * 0.5)
        # Center orbit (spin / pulse) applied after cx,cy are computed.

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
            if source == "rainbow":
                yy, xx = np.mgrid[:H, :W].astype(np.float32)
                r = norm(np.sqrt((xx - W / 2) ** 2 + (yy - H / 2) ** 2))
                hue = r * 2 * math.pi
                src = np.stack([
                    np.sin(hue) * 0.5 + 0.5,
                    np.sin(hue + 2.094) * 0.5 + 0.5,
                    np.sin(hue + 4.189) * 0.5 + 0.5,
                ], axis=-1).astype(np.float32)
            elif source == "palette":
                pal = PALETTES.get(pal_name, list(PALETTES.values())[0])
                yy, xx = np.mgrid[:H, :W].astype(np.float32)
                r = norm(np.sqrt((xx - W / 2) ** 2 + (yy - H / 2) ** 2))
                idx = (r * (len(pal) - 1)).astype(np.int32)
                src = np.array(pal, dtype=np.float32)[idx] / 255.0
            elif source == "noise":
                n = rng.standard_normal((H, W, 3)).astype(np.float32) * noise_amp + 0.5
                if _has_cv2:
                    n = cv2.GaussianBlur(n, (0, 0), sigmaX=blur_sigma, sigmaY=blur_sigma)
                src = norm(n)
            elif source == "checker":
                # High-frequency grid — makes the colored fringe obvious.
                cx = int(W / 16)
                cy = int(H / 16)
                grid = ((np.arange(W) // cx)[:, None] + (np.arange(H) // cy)[None, :]) % 2
                g3 = grid[:, :, None].astype(np.float32)
                src = np.concatenate([g3, g3 * 0.4, (1 - g3) * 0.4 + 0.2 * g3], axis=-1)
            elif source == "procedural":
                yy, xx = np.mgrid[:H, :W].astype(np.float32)
                g = np.sin(xx * 0.03 + yy * 0.02 + _t * 0.5) * \
                    np.cos(xx * 0.02 - yy * 0.03 + _t * 0.3) * 0.5 + 0.5
                src = np.stack([g, g * 0.6, 1 - g * 0.8], axis=-1).astype(np.float32)
            else:  # gradient (default)
                yy, xx = np.mgrid[:H, :W].astype(np.float32)
                r = norm(np.sqrt((xx - W / 2) ** 2 + (yy - H / 2) ** 2))
                src = np.stack([r, r * 0.7, 1 - r], axis=-1).clip(0, 1)

        src = np.clip(src, 0.0, 1.0).astype(np.float32)

        # ── Coordinate field ──
        # ── Animation-driven geometry (structural motion → clear Δ on detailed sources) ──
        zoom = 1.0
        rot_ang = 0.0
        if anim_mode == "pulse":
            zoom = 1.0 + 0.18 * math.sin(_t * 0.5)
        elif anim_mode == "spin":
            rot_ang = 0.25 * math.sin(_t * 0.5)

        cx = (W - 1) / 2.0
        cy = (H - 1) / 2.0
        if anim_mode == "spin":
            cx += math.cos(_t * 0.5) * center_drift * (W * 0.25)
            cy += math.sin(_t * 0.5) * center_drift * (H * 0.25)

        xs = np.arange(W, dtype=np.float32)
        ys = np.arange(H, dtype=np.float32)
        xv, yv = np.meshgrid(xs, ys)
        dx = xv - cx
        dy = yv - cy
        # Rotate the sampling frame (spin mode) so the whole image turns.
        if rot_ang != 0.0:
            ca_ = math.cos(rot_ang)
            sa_ = math.sin(rot_ang)
            dx, dy = dx * ca_ - dy * sa_, dx * sa_ + dy * ca_
        r = np.sqrt(dx * dx + dy * dy)
        rmax = max(r.max(), 1.0)
        ux = np.where(r > 0, dx / r, 0.0)
        uy = np.where(r > 0, dy / r, 0.0)
        rn = r / rmax
        # barrel distortion applied along the radial direction
        r_barrel = r * (1.0 + barrel * rn * rn)
        # per-channel lateral split (R outward, G neutral, B inward)
        signs = (1.0, 0.0, -1.0)
        result = np.zeros((H, W, 3), dtype=np.float32)
        for c, s in enumerate(signs):
            rs = (r_barrel + s * amount * np.power(rn, curve)) * zoom
            sx = cx + ux * rs
            sy = cy + uy * rs
            sampled = map_coordinates(
                src[:, :, c],
                np.stack([sy.ravel(), sx.ravel()]),
                order=1,
                mode="mirror",
            ).reshape(H, W)
            result[:, :, c] = sampled

        # ── Vignette ──
        if vignette > 0.0:
            v = (1.0 - vignette * rn * rn)
            result = result * v[:, :, None]

        result = np.clip(result, 0.0, 1.0).astype(np.float32)

        capture_frame("417", result)
        save(result, mn(417, "Chromatic Aberration"), out_dir)
    except Exception as exc:
        fallback = np.full((H, W, 3), 0.5, dtype=np.float32)
        save(fallback, mn(417, "Chromatic Aberration"), out_dir)
        print(f"[method_417] ERROR: {exc}")
        return fallback
