from __future__ import annotations
import math
from pathlib import Path

import numpy as np

from ...core.registry import method
from ...core.utils import save, norm, mn, seed_all, write_field, W, H, PALETTES, wired_source_lum
from ...core.animation import capture_frame

_ERROR_IMG = np.full((H, W, 3), 128, dtype=np.uint8)


@method(id='25', name='Gabor Noise', category='patterns', new_image_contract=True, tags=['classic', 'noise', 'gabor', 'anisotropic', 'band-limited', 'generative', 'animated'], inputs={'phase': 'SCALAR', 'orientation': 'SCALAR', 'image_in': 'IMAGE'}, outputs={'image': 'IMAGE', 'luminance': 'FIELD'}, params={'frequency': {'description': 'Gabor kernel radial frequency (cycles per unit)', 'min': 0.5, 'max': 40.0, 'default': 12.0}, 'bandwidth': {'description': 'Gaussian envelope width (higher = wider, lower frequency spread)', 'min': 1.0, 'max': 12.0, 'default': 4.0}, 'orientation': {'description': 'kernel orientation in degrees (anisotropic mode)', 'min': 0.0, 'max': 360.0, 'default': 45.0}, 'aniso': {'description': 'anisotropy: anisotropic (fixed orientation) or isotropic (random per-impulse)', 'default': 'anisotropic', 'choices': ['anisotropic', 'isotropic']}, 'impulses': {'description': 'average impulses per cell (kernel density)', 'min': 4, 'max': 128, 'default': 32}, 'cells': {'description': 'sparse-convolution grid cells across the short axis', 'min': 4, 'max': 40, 'default': 12}, 'style': {'description': 'render style', 'default': 'grayscale', 'choices': ['grayscale', 'colormap', 'signed']}, 'palette': {'description': "matplotlib colormap for 'colormap' style", 'default': 'twilight'}, 'phase': {'description': 'temporal phase (animates impulse phases / orientation drift)', 'default': 0.0}, 'source': {'description': "wired upstream image's luminance", 'choices': ['none', 'input_image'], 'default': 'none'}}, is_time_varying=False)
def method_gabor_noise(out_dir: Path, seed: int, params=None):
    """Gabor noise — sparse convolution of randomly-placed Gabor kernels.

    Implements Lagae, Lefebvre, Drettakis & Dutre, "Procedural Noise using
    Sparse Gabor Convolution" (ACM SIGGRAPH 2009). Noise is built by summing
    band-limited Gabor kernels

        g(x, y) = exp(-pi * a^2 * (x^2 + y^2)) * cos(2*pi*F0 * (x*cos w + y*sin w) + phi)

    placed at Poisson-distributed impulse positions. Setting a single kernel
    orientation gives anisotropic (streaky, oriented) noise; randomizing the
    orientation per impulse gives isotropic noise with a controllable power
    spectrum. Kernel support is bounded (~1/a), so each pixel only needs the
    impulses in its 3x3 neighbourhood — exact, tileable, band-limited noise.
    """
    try:
        if params is None:
            params = {}
        seed_all(seed)

        frequency = float(params.get("frequency", 12.0))
        bandwidth = float(params.get("bandwidth", 4.0))
        orientation_deg = float(params.get("orientation", 45.0))
        aniso = params.get("aniso", "anisotropic")
        impulses = int(params.get("impulses", 32))
        cells = int(params.get("cells", 12))
        style = params.get("style", "grayscale")
        pal = params.get("palette", "twilight")

        # SCALAR-driven animation
        phase_override = params.get("phase")
        phase = float(phase_override) if phase_override is not None else 0.0
        orient_override = params.get("orientation")
        if orient_override is not None:
            orientation_deg = float(orient_override)

        # Gaussian envelope coefficient 'a' from bandwidth (kernel radius ~ 1/a in cell units)
        a = 1.0 / max(1e-3, bandwidth)
        # F0 in cell-space cycles: normalize frequency so it reads consistently
        F0 = frequency / 40.0 * 3.0
        # Kernel truncation radius (in cell units) where the Gaussian ~ 0
        radius = math.sqrt(-math.log(0.01) / math.pi) / a  # exp(-pi a^2 r^2) = 0.01

        # ── Sparse convolution grid ──
        cells_y = max(4, cells)
        cell_px = H / cells_y            # square cells in pixel space
        cells_x = int(math.ceil(W / cell_px)) + 1
        cells_y = cells_y + 1

        # Per-pixel coordinates in cell units
        yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
        # Wired image as a domain-warp source (luminance distorts the pattern grid)
        _src_lum = wired_source_lum(params, xx.shape[1], xx.shape[0])
        if _src_lum is not None:
            xx = xx + (_src_lum - 0.5) * 15.0
            yy = yy + (_src_lum - 0.5) * 15.0

        cx = xx / cell_px
        cy = yy / cell_px

        acc = np.zeros((H, W), dtype=np.float32)
        w0 = math.radians(orientation_deg) + phase * 0.3

        rng = np.random.default_rng(seed & 0x7FFFFFFF)

        # Precompute integer cell index per pixel
        icx = np.floor(cx).astype(np.int32)
        icy = np.floor(cy).astype(np.int32)

        # For each cell, generate a deterministic set of impulses (seeded by cell
        # coords) and splat kernels into pixels whose cell is within radius.
        cell_reach = int(math.ceil(radius)) + 1
        for gy in range(cells_y):
            for gx in range(cells_x):
                # Deterministic per-cell RNG so noise is stable/tileable
                cseed = (seed ^ (gx * 0x9E3779B1) ^ (gy * 0x85EBCA77)) & 0x7FFFFFFF
                crng = np.random.default_rng(cseed)
                n_imp = crng.poisson(impulses)
                if n_imp <= 0:
                    continue
                # Impulse positions within the cell (cell units)
                ix = gx + crng.random(n_imp).astype(np.float32)
                iy = gy + crng.random(n_imp).astype(np.float32)
                # Random weights, phases; orientation per aniso mode
                wgt = crng.uniform(-1.0, 1.0, n_imp).astype(np.float32)
                phi = crng.uniform(0, 2 * math.pi, n_imp).astype(np.float32) + phase
                if aniso == "isotropic":
                    omega = crng.uniform(0, 2 * math.pi, n_imp).astype(np.float32)
                else:
                    omega = np.full(n_imp, w0, dtype=np.float32)

                # Only touch pixels in cells near this one
                sel = (np.abs(icx - gx) <= cell_reach) & (np.abs(icy - gy) <= cell_reach)
                if not np.any(sel):
                    continue
                pcx = cx[sel]
                pcy = cy[sel]
                sub = np.zeros(pcx.shape[0], dtype=np.float32)
                for k in range(n_imp):
                    dx = pcx - ix[k]
                    dy = pcy - iy[k]
                    r2 = dx * dx + dy * dy
                    env = np.exp(-math.pi * a * a * r2)
                    proj = dx * math.cos(omega[k]) + dy * math.sin(omega[k])
                    sub += wgt[k] * env * np.cos(2 * math.pi * F0 * proj + phi[k])
                acc[sel] += sub

        # Normalize: divide by sqrt of impulse count for stable variance
        acc /= max(1.0, math.sqrt(impulses))
        field = acc.astype(np.float32)  # signed FIELD

        no = norm(acc)  # [0,1] for rendering

        # ── Render ──
        if style == "colormap":
            try:
                from matplotlib import cm
                rgb = cm.get_cmap(pal)(no)[:, :, :3].astype(np.float32)
            except Exception:
                rgb = np.stack([no, no, no], axis=-1)
        elif style == "signed":
            # blue negative, red positive around neutral gray
            s = np.clip(acc / (np.abs(acc).max() + 1e-6), -1, 1)
            pos = np.clip(s, 0, 1)
            neg = np.clip(-s, 0, 1)
            rgb = np.stack([0.5 + pos * 0.5, 0.5 - (pos + neg) * 0.3, 0.5 + neg * 0.5], axis=-1).astype(np.float32)
        else:
            rgb = np.stack([no, no, no], axis=-1).astype(np.float32)

        rgb = np.clip(rgb, 0, 1).astype(np.float32)

        capture_frame("25", rgb)
        try:
            write_field(out_dir, field)
        except Exception:
            pass
        save(rgb, mn(25, "gabor-noise"), out_dir)
        return {"image": rgb, "luminance": field}
    except Exception as exc:
        import traceback as _tb
        _tb.print_exc()
        print(f"[method_25] ERROR: {exc}")
        save(_ERROR_IMG, mn(25, "Gabor Noise"), out_dir)
        return {"image": _ERROR_IMG.astype(np.float32) / 255.0}
