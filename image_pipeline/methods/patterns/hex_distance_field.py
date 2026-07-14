from __future__ import annotations

import math

import numpy as np
from PIL import Image

from ...core.registry import method
from ...core.utils import (
    save, mn, seed_all, W, H, PALETTES, wired_source_rgb,
    write_field, write_mask, write_scalars,
)
from ...core.animation import capture_frame


# ── Vectorized signed value noise (deterministic, seed-stable) ──
def _hash_corner(ix: np.ndarray, iy: np.ndarray, seed: int) -> np.ndarray:
    """Integer lattice hash -> float in [0,1). Vectorized, platform-stable."""
    ix = ix.astype(np.uint64)
    iy = iy.astype(np.uint64)
    n = (ix * np.uint64(73856093)) ^ (iy * np.uint64(19349663)) ^ (np.uint64(seed) * np.uint64(83492791))
    n = (n ^ (n >> np.uint64(13))) * np.uint64(1274126177)
    n = n ^ (n >> np.uint64(16))
    return (n & np.uint64(0x7FFFFFFF)).astype(np.float64) / 2147483647.0


def _value_noise(x: np.ndarray, y: np.ndarray, seed: int) -> np.ndarray:
    """Smooth value noise in [-1, 1] via bilerp + smoothstep (IQ-style)."""
    xi = np.floor(x).astype(np.int64)
    yi = np.floor(y).astype(np.int64)
    xf = x - xi
    yf = y - yi
    u = xf * xf * (3.0 - 2.0 * xf)
    v = yf * yf * (3.0 - 2.0 * yf)
    h00 = _hash_corner(xi, yi, seed)
    h10 = _hash_corner(xi + 1, yi, seed)
    h01 = _hash_corner(xi, yi + 1, seed)
    h11 = _hash_corner(xi + 1, yi + 1, seed)
    a = h00 + (h10 - h00) * u
    b = h01 + (h11 - h01) * u
    return (a + (b - a) * v) * 2.0 - 1.0


def _fbm(x: np.ndarray, y: np.ndarray, seed: int, octaves: int,
         lacunarity: float, gain: float) -> np.ndarray:
    """Fractional Brownian motion in [-1, 1]."""
    amp = 1.0
    freq = 1.0
    total = np.zeros_like(x, dtype=np.float64)
    norm = 0.0
    for o in range(octaves):
        total += amp * _value_noise(x * freq, y * freq, seed + o * 101)
        norm += amp
        amp *= gain
        freq *= lacunarity
    return total / norm if norm > 0 else total


def _sd_hexagon(px: np.ndarray, py: np.ndarray, r: float) -> np.ndarray:
    """IQ flat-top hexagon signed distance (negative inside). Vectorized.

    r is the centre-to-vertex radius. Returns <0 inside, >0 outside.
    """
    k1 = -0.8660254037844386  # -sqrt(3)/2
    k2 = 0.5
    k3 = 0.5773502691896258   # 1/sqrt(3)
    ax = np.abs(px)
    ay = np.abs(py)
    # dot(k.xy, p) = k1*ax + k2*ay
    d = k1 * ax + k2 * ay
    m = 2.0 * np.minimum(d, 0.0)
    qx = ax + m * k1
    qy = ay + m * k2
    # clamp to the hexagon extent
    cx = np.clip(qx, -k3 * r, k3 * r)
    qx = qx - cx
    qy = qy - r
    return np.sqrt(qx * qx + qy * qy) * np.sign(qy)


@method(
    id='343', name='Hex Distance Field', category='patterns',
    tags=['procedural', 'hexagonal', 'voronoi', 'distance-field', 'sampling-lattice', 'animation'],
    params={
        'scale': {'description': 'hex cell size (centre-to-vertex), in pixels', 'min': 6.0, 'max': 80.0, 'default': 28.0},
        'jitter': {'description': 'organic per-cell centre displacement (0 = perfect lattice)', 'min': 0.0, 'max': 1.0, 'default': 0.0},
        'octaves': {'description': 'FBM octaves for procedural cell colour', 'min': 1, 'max': 6, 'default': 4},
        'contrast': {'description': 'final tone contrast', 'min': 0.5, 'max': 3.0, 'default': 1.1},
        'colormode': {'description': 'color mapping (grayscale/rainbow/inferno/viridis/palette/fire/ice)', 'default': 'inferno'},
        'palette': {'description': 'palette name for palette mode', 'default': 'vapor'},
        'edge_width': {'description': 'cell-border thickness as a fraction of cell size (0 = none)', 'min': 0.0, 'max': 0.4, 'default': 0.12},
        'source': {'description': 'cell colour source', 'choices': ['procedural', 'input_image'], 'default': 'procedural'},
        'anim_mode': {'description': 'animation mode: none, flow, rotate, pulse', 'default': 'none'},
        'anim_speed': {'description': 'animation speed multiplier', 'min': 0.1, 'max': 3.0, 'default': 1.0},
        'time': {'description': 'animation phase [0, 2pi)', 'min': 0.0, 'max': 6.28, 'default': 0.0},
    },
    inputs={'image_in': 'IMAGE'},
    outputs={'image': 'IMAGE', 'field': 'FIELD', 'mask': 'MASK'},
)
def method_hex_distance_field(out_dir, seed: int, params=None):
    """Hexagonal Voronoi mosaic + hex distance field.

    A hexagonal sampling lattice has one higher degree of symmetry than a
    square grid, giving better isotropy and more accurate image processing
    (He 2005, "Hexagonal Structure for Intelligent Vision"; Rives/Ouwerkerk
    hexagonal image-processing framework). Each pixel is mapped to its
    containing flat-top hex cell via axial-coordinate cube rounding; the cell
    is flat-coloured from a procedural FBM (or a wired image), and the signed
    distance to the cell boundary (IQ ``sdHexagon``) is exported as a FIELD
    and as a MASK of interior strength.

    Purely closed-form per frame (Architecture B): the orchestrator re-calls
    with an increasing ``time`` value.
    """
    try:
        if params is None:
            params = {}
        t = float(params.get("time", 0.0))
        seed_all(seed)
        _ = np.random.default_rng(seed)

        scale = float(params.get("scale", 28.0))
        jitter = float(params.get("jitter", 0.0))
        octaves = int(params.get("octaves", 4))
        contrast = float(params.get("contrast", 1.1))
        cmode = params.get("colormode", "inferno")
        pal_name = params.get("palette", "vapor")
        edge_width = float(params.get("edge_width", 0.12))
        src = params.get("source", "procedural")
        anim_mode = params.get("anim_mode", "none")
        anim_speed = float(params.get("anim_speed", 1.0))

        if anim_mode == "none":
            _t = 0.0
        else:
            _t = t * anim_speed

        # Effective cell size (pulse breathes the lattice).
        s = scale * (0.7 + 0.3 * (0.5 + 0.5 * math.sin(_t))) if anim_mode == "pulse" else scale

        yy, xx = np.mgrid[0:H, 0:W].astype(np.float64)
        cx0, cy0 = W / 2.0, H / 2.0

        # Optional lattice rotation (rotate mode) about the canvas centre.
        if anim_mode == "rotate":
            ang = _t * 0.5
            dx = xx - cx0
            dy = yy - cy0
            xx = cx0 + dx * math.cos(ang) - dy * math.sin(ang)
            yy = cy0 + dx * math.sin(ang) + dy * math.cos(ang)

        X = xx - cx0
        Y = yy - cy0
        sqrt3 = math.sqrt(3.0)

        # Pixel -> fractional axial coordinates (flat-top lattice).
        qf = (2.0 / 3.0) * X / s
        rf = Y / (sqrt3 * s) - qf / 2.0

        # Cube rounding -> nearest hex centre index.
        xc, yc, zc = qf, -qf - rf, rf
        rx = np.round(xc)
        ry = np.round(yc)
        rz = np.round(zc)
        dxd = np.abs(rx - xc)
        dyd = np.abs(ry - yc)
        dzd = np.abs(rz - zc)
        # fix the largest-diff component so x+y+z == 0
        fix = (dxd > dyd) & (dxd > dzd)
        rx = np.where(fix, -ry - rz, rx)
        fix = (~fix) & (dyd > dzd)
        ry = np.where(fix, -rx - rz, ry)
        rz = np.where((~fix) & (dyd <= dzd), -rx - ry, rz)
        q = rx.astype(np.int64)
        r = rz.astype(np.int64)

        # Per-cell organic jitter (stable per cell via hashed noise).
        if jitter > 0.0:
            jx = _value_noise(q.astype(np.float64) * 0.6, r.astype(np.float64) * 0.6, seed + 11)
            jy = _value_noise(q.astype(np.float64) * 0.6 + 4.2, r.astype(np.float64) * 0.6 + 1.7, seed + 12)
            jx = jx * jitter * s
            jy = jy * jitter * s
        else:
            jx = 0.0
            jy = 0.0

        # Hex centre pixel position.
        ccx = 1.5 * s * q + cx0 + jx
        ccy = sqrt3 * s * (r + q / 2.0) + cy0 + jy

        # ── Cell colour ──
        cell_rgb = np.zeros((int(H), int(W), 3), dtype=np.float32)
        if src == "input_image":
            wired = wired_source_rgb(params, int(W), int(H))
            if wired is not None:
                # Sample the wired image at each cell centre (nearest pixel).
                sx = np.clip(np.round(ccy).astype(np.int64), 0, int(H) - 1)
                sy = np.clip(np.round(ccx).astype(np.int64), 0, int(W) - 1)
                cell_rgb = wired[sx, sy]  # (H,W,3)
            else:
                # Fall through to procedural if nothing is wired.
                src = "procedural"

        if src != "input_image":
            # Procedural colour: FBM sampled at the cell's WORLD position
            # (so the colour field is world-locked and visibly rotates/pans with
            # the lattice in rotate/flow modes), evolving over time.
            flow = _t * 0.25 if anim_mode == "flow" else 0.0
            nq = ccx * 0.02 + flow
            nr = ccy * 0.02 + flow * 0.7
            val = _fbm(nq, nr, seed + 5, octaves, 2.0, 0.5)
            val = (val + 1.0) * 0.5
            val = np.clip(0.5 + (val - 0.5) * contrast, 0.0, 1.0)

            if cmode == "grayscale":
                cell_rgb = np.stack([val, val, val], axis=-1)
            elif cmode == "rainbow":
                hue = val * 2 * math.pi
                cell_rgb = np.stack([
                    np.sin(hue) * 0.5 + 0.5,
                    np.sin(hue + 2.094) * 0.5 + 0.5,
                    np.sin(hue + 4.189) * 0.5 + 0.5,
                ], axis=-1)
            elif cmode == "palette":
                pal = PALETTES.get(pal_name, PALETTES["vapor"])
                idx = (val * (len(pal) - 1)).astype(np.int32)
                cell_rgb = np.array(pal, dtype=np.float32)[idx] / 255.0
            elif cmode == "inferno":
                try:
                    from matplotlib import cm
                    cell_rgb = cm.inferno(val)[:, :, :3]
                except ImportError:
                    cell_rgb = np.stack([val ** 1.4, val ** 0.6 * (1 - val) * 2 + val * 0.2, val ** 0.3 * 0.5], axis=-1)
            elif cmode == "viridis":
                try:
                    from matplotlib import cm
                    cell_rgb = cm.viridis(val)[:, :, :3]
                except ImportError:
                    cell_rgb = np.stack([val * 0.3, val ** 0.5 * 0.8, (1 - val) * 0.4 + val * 0.6], axis=-1)
            elif cmode == "fire":
                cell_rgb = np.stack([np.clip(val * 1.5, 0, 1), val * 0.6, val * 0.2], axis=-1)
            elif cmode == "ice":
                cell_rgb = np.stack([val * 0.2, val * 0.5, 0.5 + val * 0.5], axis=-1)
            else:
                cell_rgb = np.stack([val, val, val], axis=-1)

        # ── Edge distance field (IQ sdHexagon) ──
        lx = X - (ccx - cx0)
        ly = Y - (ccy - cy0)
        sd = _sd_hexagon(lx, ly, s)
        edge_dist = np.clip(-sd, 0.0, None)  # 0 at boundary, ~s*0.866 at centre

        # Cell-border darkening.
        if edge_width > 0.0:
            thr = edge_width * s
            border = np.clip(1.0 - edge_dist / max(thr, 1e-6), 0.0, 1.0)
            cell_rgb = cell_rgb * (1.0 - 0.85 * border)[..., None]

        # Pulse brightness breathing (separate from lattice breathing).
        if anim_mode == "pulse":
            cell_rgb = cell_rgb * (0.6 + 0.4 * (0.5 + 0.5 * math.sin(_t)))

        rgb = np.clip(cell_rgb, 0.0, 1.0).astype(np.float32)

        # ── Sidecar outputs (Rule 5 / Rule 10) ──
        field = edge_dist.astype(np.float32)
        mask = np.clip(edge_dist / max(s, 1e-6), 0.0, 1.0).astype(np.float32)
        n_cells = int(round((W * H) / ((1.5 * math.sqrt(3)) * s * s)))
        write_field(out_dir, field)
        write_mask(out_dir, mask)
        write_scalars(out_dir, mean_edge=float(float(edge_dist.mean())), hex_count=float(n_cells))

        capture_frame("343", rgb)
        save(rgb, mn(343, "Hex Distance Field"), out_dir)
        return rgb
    except Exception as exc:
        fallback = np.full((H, W, 3), 128, dtype=np.uint8)
        save(fallback, mn(343, "Hex Distance Field"), out_dir)
        print(f"[method_343] ERROR: {exc}")
        return fallback
