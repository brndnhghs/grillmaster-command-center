from __future__ import annotations

import math
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.ndimage import map_coordinates

from ...core.registry import method
from ...core.utils import (
    save,
    mn,
    seed_all,
    W,
    H,
    PALETTES,
    load_input,
    write_scalars,
)
from ...core.animation import capture_frame


def _procedural_source(kind: str, _t: float, rng: np.random.Generator) -> np.ndarray:
    """Build a default RGB source so the Droste spiral is visible standalone."""
    yy, xx = np.mgrid[:H, :W].astype(np.float32)
    fx = (xx - W / 2) / (max(W, H) / 2.0)
    fy = (yy - H / 2) / (max(W, H) / 2.0)
    if kind == "gradient":
        g = np.clip(0.5 + 0.5 * np.sin(6.0 * (fx + fy) + _t * 0.3), 0, 1)
        return np.stack([g, np.clip(g + 0.2, 0, 1), 1 - g], axis=-1).astype(np.float32)
    if kind == "rainbow":
        hue = (np.arctan2(fy, fx) / (2 * np.pi) + 0.5 + _t * 0.02) % 1.0
        r = np.clip(abs(hue * 6 - 3) - 1, 0, 1)
        g = np.clip(2 - abs(hue * 6 - 2), 0, 1)
        b = np.clip(2 - abs(hue * 6 - 4), 0, 1)
        return np.stack([r, g, b], axis=-1).astype(np.float32)
    # checker (default) — highest-frequency, most Droste-legible
    sq = (np.floor(fx * 10) + np.floor(fy * 10)) % 2
    base = np.stack([sq, sq * 0.6 + 0.2, 1 - sq * 0.7], axis=-1).astype(np.float32)
    return base


@method(
    id="444",
    name="Droste Spiral",
    category="filters",
    tags=["droste", "conformal", "log-polar", "spiral", "escher", "remap", "npr", "animation"],
    inputs={"image_in": "IMAGE"},
    outputs={"image": "IMAGE"},
    params={
        "source": {
            "description": "source image (input_image/wired, or a generated default)",
            "choices": ["input_image", "checker", "gradient", "rainbow"],
            "default": "checker",
        },
        "twist": {
            "description": "rotation coupling in log-polar space (0=concentric rings, >0=Escher spiral)",
            "min": 0.0,
            "max": 1.2,
            "default": 0.35,
        },
        "ring_spacing": {
            "description": "period of log-radius (smaller = more, tighter rings)",
            "min": 0.15,
            "max": 1.2,
            "default": 0.55,
        },
        "zoom": {
            "description": "radial zoom factor (scales distance before the log map)",
            "min": 0.3,
            "max": 3.0,
            "default": 1.0,
        },
        "anim_mode": {
            "description": "animation mode (none/spiral_twist/zoom_pulse/rotate)",
            "choices": ["none", "spiral_twist", "zoom_pulse", "rotate"],
            "default": "none",
        },
        "anim_speed": {
            "description": "animation speed multiplier",
            "min": 0.1,
            "max": 5.0,
            "default": 1.0,
        },
    },
)
def method_droste(out_dir: Path, seed: int, params=None):
    """Droste conformal (log-polar) spiral remap — infinite self-similar tiling.

    The core technique is the complex logarithm: mapping each output pixel to
    centered polar coordinates (r, theta) and taking its complex log
    w = ln(r) + i*theta. This conformal map turns rotation+scaling into a pure
    translation. Rotating w in (ln r, theta) space by ``twist`` couples radial
    scaling to angular rotation, which is exactly what produces the Droste /
    Escher "Print Gallery" spiral: zooming inward spirals the image rather than
    repeating it concentrically. The transformed coordinates are wrapped into
    [0,1) and used to sample the source picture, tiling it infinitely.

    Animation modulates the twist, zoom, or angular pan — all smooth (offset
    sine, no cusps), so the spiral breathes / zooms / rotates continuously.

    References:
      - Lenstra, de Smit et al. 2003, "The mathematical structure of Escher's
        Print Gallery", Notices of the AMS  (analysis of the Droste effect).
      - Preshing 2012, "The Droste Effect in WebGL".

    Args:
        out_dir: Output directory.
        seed: Deterministic seed.
        params: source, twist, ring_spacing, zoom, anim_mode, anim_speed, time.
    """
    try:
        seed_all(seed)
        rng = np.random.default_rng(seed)
        p = params or {}
        _t = float(p.get("time", 0.0)) * float(p.get("anim_speed", 1.0))
        anim_mode = p.get("anim_mode", "none")

        src_kind = p.get("source", "checker")
        twist = float(p.get("twist", 0.35))
        ring_spacing = float(p.get("ring_spacing", 0.55))
        zoom = float(p.get("zoom", 1.0))

        # ── Animation (smooth, offset-sine to avoid cusps) ──
        ang_pan = 0.0
        if anim_mode == "spiral_twist":
            twist = twist + 0.45 * (0.5 + 0.5 * math.sin(_t * 0.5))
        elif anim_mode == "zoom_pulse":
            zoom = zoom * math.exp(0.6 * (0.5 + 0.5 * math.sin(_t * 0.5)) - 0.5)
        elif anim_mode == "rotate":
            ang_pan = _t * 0.4

        # ── Build source image ──
        wired_path = p.get("input_image", "")
        wired_arr = p.get("_input_image", None)
        src = None
        if wired_arr is not None and isinstance(wired_arr, np.ndarray):
            src = wired_arr.astype(np.float32)
        elif wired_path:
            try:
                src = load_input(wired_path, W, H)
            except (FileNotFoundError, OSError):
                src = None
        if src is None:
            # Rule 12: only fall back to generation when no wire is present
            src = _procedural_source(src_kind if src_kind != "input_image" else "checker", _t, rng)

        src = np.clip(src, 0.0, 1.0).astype(np.float32)

        # ── Centered, aspect-correct coords ──
        yy, xx = np.mgrid[:H, :W].astype(np.float32)
        fx = (xx - W / 2) / (max(W, H) / 2.0)
        fy = (yy - H / 2) / (max(W, H) / 2.0)
        r = np.sqrt(fx * fx + fy * fy) + 1e-6
        theta = np.arctan2(fy, fx) + ang_pan

        # Complex log: A = ln(r), B = theta  (conformal rotation+scale -> translation)
        A = np.log(r * zoom)
        B = theta

        # Rotate (A,B) in log-polar space by `twist` -> couples scale to angle
        ct, st = math.cos(twist), math.sin(twist)
        Ap = A * ct - B * st
        Bp = A * st + B * ct

        # Wrap into [0,1) texture coordinates (infinite tiling of the source)
        u = np.fmod(Ap / ring_spacing + 0.5, 1.0)
        u = np.where(u < 0, u + 1.0, u)
        v = np.fmod(Bp / (2.0 * math.pi) + 0.5, 1.0)
        v = np.where(v < 0, v + 1.0, v)

        # Sample source at wrapped coords (row=vertical, col=horizontal)
        row_coords = (v * (H - 1)).astype(np.float64)
        col_coords = (u * (W - 1)).astype(np.float64)
        coords = np.stack([row_coords.ravel(), col_coords.ravel()])
        out = np.zeros((H, W, 3), dtype=np.float32)
        for c in range(3):
            out[..., c] = map_coordinates(
                src[..., c], coords, order=1, mode="wrap"
            ).reshape(H, W)

        out = np.clip(out, 0.0, 1.0).astype(np.float32)

        # Report interesting scalars (Rule 4)
        write_scalars(
            out_dir,
            twist=round(float(twist), 4),
            ring_spacing=round(ring_spacing, 4),
            zoom=round(float(zoom), 4),
        )

        capture_frame("444", out)
        save(out, mn(444, "Droste Spiral"), out_dir)
        return out
    except Exception as exc:  # Rule 1: PNG in every code path
        fallback = np.full((H, W, 3), 128, dtype=np.float32)
        save(fallback, mn(444, "Droste Spiral"), out_dir)
        print(f"[method_444] ERROR: {exc}")
        return fallback
