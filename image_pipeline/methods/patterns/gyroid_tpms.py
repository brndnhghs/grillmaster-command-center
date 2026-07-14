from __future__ import annotations

import math

import numpy as np

from ...core.registry import method
from ...core.utils import (
    save, mn, seed_all, W, H, PALETTES,
    write_field, write_mask, write_scalars,
)
from ...core.animation import capture_frame


def _tpms_field(surface: str, x: np.ndarray, y: np.ndarray, z: float) -> np.ndarray:
    """Evaluate a triply-periodic minimal-surface implicit field at slice z.

    Closed-form nodal approximations of classic minimal surfaces (Schoen 1970,
    Schwarz 1890). Each returns a scalar field whose zero level set is the
    surface. Fully vectorized over the pixel grid.
    """
    sx, cx = np.sin(x), np.cos(x)
    sy, cy = np.sin(y), np.cos(y)
    sz, cz = math.sin(z), math.cos(z)
    if surface == "schwarz_p":
        return cx + cy + cz
    if surface == "diamond":
        # Schwarz D
        return sx * sy * sz + sx * cy * cz + cx * sy * cz + cx * cy * sz
    if surface == "neovius":
        return 3.0 * (cx + cy + cz) + 4.0 * cx * cy * cz
    if surface == "iwp":
        # I-WP (Schoen)
        return 2.0 * (cx * cy + cy * cz + cz * cx) - (np.cos(2 * x) + np.cos(2 * y) + math.cos(2 * z))
    # default: gyroid (Schoen)
    return sx * cy + sy * cz + sz * cx


@method(
    id='964', name='Gyroid TPMS', category='patterns',
    tags=['procedural', 'tpms', 'gyroid', 'minimal-surface', 'implicit', 'sdf', 'animation'],
    params={
        'surface': {'description': 'minimal surface (gyroid/schwarz_p/diamond/neovius/iwp)', 'default': 'gyroid'},
        'freq': {'description': 'spatial frequency (number of cells across the canvas)', 'min': 1.0, 'max': 16.0, 'default': 5.0},
        'level': {'description': 'iso-level of the surface (shifts the shell inward/outward)', 'min': -1.5, 'max': 1.5, 'default': 0.0},
        'thickness': {'description': 'shell half-thickness of the surface band', 'min': 0.02, 'max': 0.8, 'default': 0.22},
        'warp': {'description': 'domain-warp strength (organic distortion of the lattice)', 'min': 0.0, 'max': 1.5, 'default': 0.0},
        'contrast': {'description': 'final tone contrast', 'min': 0.5, 'max': 3.0, 'default': 1.2},
        'colormode': {'description': 'color mapping (grayscale/rainbow/inferno/viridis/palette/fire/ice)', 'default': 'inferno'},
        'palette': {'description': 'palette name for palette mode', 'default': 'vapor'},
        'shell': {'description': 'render mode: field (smooth signed field) or shell (surface band)', 'choices': ['field', 'shell'], 'default': 'shell'},
        'anim_mode': {'description': 'animation mode: none, slice, phase, warp, rotate', 'default': 'none'},
        'anim_speed': {'description': 'animation speed multiplier', 'min': 0.1, 'max': 3.0, 'default': 1.0},
        'time': {'description': 'animation phase [0, 2pi)', 'min': 0.0, 'max': 6.28, 'default': 0.0},
    },
    inputs={},
    outputs={'image': 'IMAGE', 'field': 'FIELD', 'mask': 'MASK'},
)
def method_gyroid_tpms(out_dir, seed: int, params=None):
    """Gyroid / triply-periodic minimal-surface field generator.

    Evaluates a closed-form nodal approximation of a classic minimal surface
    (gyroid, Schwarz P/D, Neovius, I-WP) over the pixel plane. The third
    coordinate ``z`` is the slice plane, animated in ``slice`` mode so the
    2D cross-section morphs continuously as the plane sweeps through the 3D
    volume. The zero (or ``level``) iso-set is rendered as a shell band, or
    the smooth signed field is shown directly.

    Purely closed-form per frame (Architecture B): the orchestrator re-calls
    with an increasing ``time`` value. O(W*H) vectorized, sub-ms per frame ->
    timeout-immune high-liveness generator.
    """
    try:
        if params is None:
            params = {}
        t = float(params.get("time", 0.0))
        seed_all(seed)
        _ = np.random.default_rng(seed)

        surface = params.get("surface", "gyroid")
        freq = float(params.get("freq", 5.0))
        level = float(params.get("level", 0.0))
        thickness = float(params.get("thickness", 0.22))
        warp = float(params.get("warp", 0.0))
        contrast = float(params.get("contrast", 1.2))
        cmode = params.get("colormode", "inferno")
        pal_name = params.get("palette", "vapor")
        shell_mode = params.get("shell", "shell")
        anim_mode = params.get("anim_mode", "none")
        anim_speed = float(params.get("anim_speed", 1.0))

        _t = 0.0 if anim_mode == "none" else t * anim_speed

        # Normalized coordinates -> lattice space scaled by frequency.
        yy, xx = np.mgrid[0:H, 0:W].astype(np.float64)
        nx = (xx / W - 0.5) * 2.0 * math.pi * freq
        ny = (yy / H - 0.5) * 2.0 * math.pi * freq

        # Optional lattice rotation (rotate mode).
        if anim_mode == "rotate":
            ang = _t * 0.5
            ca, sa = math.cos(ang), math.sin(ang)
            rx = nx * ca - ny * sa
            ry = nx * sa + ny * ca
            nx, ny = rx, ry

        # Domain warp (static base + animated component in warp mode).
        wstr = warp
        if anim_mode == "warp":
            wstr = warp + 0.6  # ensure motion even at warp=0
        if wstr > 0.0:
            wt = _t if anim_mode == "warp" else 0.0
            nx = nx + wstr * np.sin(ny * 0.5 + wt)
            ny = ny + wstr * np.cos(nx * 0.5 + wt * 0.8)

        # Slice plane z: swept in slice mode, phase-offset in phase mode.
        if anim_mode == "slice":
            z = _t
        elif anim_mode == "phase":
            z = _t
        else:
            z = 0.0

        g = _tpms_field(surface, nx, ny, z)
        # Normalize field to a stable range (surfaces have amplitude ~[-3,3]).
        gmax = 3.0 if surface in ("neovius", "schwarz_p", "iwp") else 1.5
        gn = g / gmax  # roughly [-1, 1]

        if shell_mode == "shell":
            # Distance from the iso-level, folded into a bright shell band.
            d = np.abs(gn - level / gmax)
            val = np.clip(1.0 - d / max(thickness, 1e-6), 0.0, 1.0)
        else:
            # Smooth signed field mapped to [0,1].
            val = np.clip((gn - level / gmax) * 0.5 + 0.5, 0.0, 1.0)

        val = np.clip(0.5 + (val - 0.5) * contrast, 0.0, 1.0)

        # ── Colour mapping ──
        if cmode == "grayscale":
            rgb = np.stack([val, val, val], axis=-1)
        elif cmode == "rainbow":
            hue = val * 2 * math.pi
            rgb = np.stack([
                np.sin(hue) * 0.5 + 0.5,
                np.sin(hue + 2.094) * 0.5 + 0.5,
                np.sin(hue + 4.189) * 0.5 + 0.5,
            ], axis=-1)
        elif cmode == "palette":
            pal = PALETTES.get(pal_name, PALETTES["vapor"])
            idx = (val * (len(pal) - 1)).astype(np.int32)
            rgb = np.array(pal, dtype=np.float32)[idx] / 255.0
        elif cmode == "inferno":
            try:
                from matplotlib import cm
                rgb = cm.inferno(val)[:, :, :3]
            except ImportError:
                rgb = np.stack([val ** 1.4, val ** 0.6 * (1 - val) * 2 + val * 0.2, val ** 0.3 * 0.5], axis=-1)
        elif cmode == "viridis":
            try:
                from matplotlib import cm
                rgb = cm.viridis(val)[:, :, :3]
            except ImportError:
                rgb = np.stack([val * 0.3, val ** 0.5 * 0.8, (1 - val) * 0.4 + val * 0.6], axis=-1)
        elif cmode == "fire":
            rgb = np.stack([np.clip(val * 1.5, 0, 1), val * 0.6, val * 0.2], axis=-1)
        elif cmode == "ice":
            rgb = np.stack([val * 0.2, val * 0.5, 0.5 + val * 0.5], axis=-1)
        else:
            rgb = np.stack([val, val, val], axis=-1)

        rgb = np.clip(rgb, 0.0, 1.0).astype(np.float32)

        # ── Sidecar outputs (Rule 5 / Rule 10) ──
        field = gn.astype(np.float32)
        mask = np.clip(val, 0.0, 1.0).astype(np.float32)
        write_field(out_dir, field)
        write_mask(out_dir, mask)
        write_scalars(out_dir, mean_field=float(gn.mean()), shell_frac=float((mask > 0.5).mean()))

        capture_frame("964", rgb)
        save(rgb, mn(964, "Gyroid TPMS"), out_dir)
        return rgb
    except Exception as exc:
        fallback = np.full((H, W, 3), 128, dtype=np.uint8)
        save(fallback, mn(964, "Gyroid TPMS"), out_dir)
        print(f"[method_964] ERROR: {exc}")
        return fallback
