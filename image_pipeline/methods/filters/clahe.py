"""CLAHE — Contrast-Limited Adaptive Histogram Equalization (Zuiderveld 1994).

Implements textbook **CLAHE** (Karel Zuiderveld, "Contrast Limited Adaptive
Histogram Equalization", in *Graphics Gems IV*, 1994) as a self-contained filter
node. CLAHE is the adaptive, artifact-controlled successor to ordinary
histogram equalization: instead of one global tone curve for the whole image,
it builds a *local* equalization curve for every tile of the canvas and
**bilinearly blends** the four nearest tile curves at each pixel — so local
contrast is boosted everywhere without the harsh "blocky" boundaries of naive
tiled equalization. The **contrast limit** clips each tile histogram at a
ceiling and redistributes the excess uniformly, which is what stops a few
blown-out pixels from washing out the whole tile (the classic AHE halo/block
artifact).

Core algorithm (per channel):
    1. Tile the canvas into `tile`×`tile` regions; build a 256-bin histogram
       for every tile (vectorized scatter).
    2. Clip each histogram at `clip_limit` (fraction of the per-tile mean bin
       count); redistribute the clipped excess evenly across all bins.
    3. Form the CDF of the clipped histogram → that tile's mapping function
       T_tile(v) ∈ [0,1].
    4. For each pixel, bilinearly interpolate T over the four surrounding tile
       centres, evaluated at the pixel's own intensity bin.

Because the blend is bilinear and the clip caps the gain, the result is smooth
and never produces the salt-and-pepper "blooming" of plain adaptive
equalization — which is exactly why CLAHE is the de-facto local-contrast
enhancer in medical imaging (CT/MRI), astrophotography, and every "local
contrast / clarity" slider in modern photo editors.

Reference URLs:
    • Zuiderveld 1994 (Graphics Gems IV, ch. "CLAHE"): https://doi.org/10.1016/B978-0-12-336156-1.50061-6
    • OpenCV equalizeAdapthist docs: https://docs.opencv.org/4.x/d5/daf/tutorial_py_histogram_equalization.html
    • Heidelberg CLAHE notes: https://www.cs.duke.edu/courses/spring03/cps296.1/handouts/spring03/CLAHE.pdf

Distinct from sibling nodes:
    • tone_mapping (428): maps HDR→LDR with a *global* filmic curve (Reinhard/
      ACES). CLAHE is a *local*, histogram-based contrast expander that works on
      ordinary LDR images and never clips highlights the way a global operator
      can.
    • wls_smoothing / l0_smoothing / guided_filter / rolling_guidance /
      domain_transform: these are *edge-preserving smoothers* (they REMOVE
      detail). CLAHE does the opposite — it *amplifies* local detail/contrast.

Architecture B (per-frame re-call with `time`):
    none         — static: Δ ≈ 0 (same enhancement every frame).
    amount_pulse — the amount of CLAHE applied pulses 0→1 with t, morphing the
                   image between original and locally-enhanced (Δ > 0.05, no cusp).
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
from PIL import Image

from ...core.registry import method
from ...core.utils import (
    save, mn, seed_all, W, H, PALETTES, load_input,
    write_field, write_scalars,
)
from ...core.animation import capture_frame


# ── CLAHE core (per single-channel float image) ─────────────────────────────
def _clahe_channel(channel: np.ndarray, tile: int, clip_frac: float) -> np.ndarray:
    """Contrast-limited adaptive histogram equalization for one H×W channel.

    `channel` is float32 in [0,1]. Returns the equalized channel in [0,1].
    Fully vectorized: per-tile histograms (binned into each tile's OWN min..max
    range so a smooth tile fills every bin and gets a full local-contrast
    stretch, instead of a sparse fixed-[0,1] histogram that collapses to a
    near-identity), the clip + redistribute step, and a bilinear blend of the
    four surrounding tile CDFs at every pixel.
    """
    Hc, Wc = channel.shape
    nbins = 256
    n_tx = max(1, Wc // tile)
    n_ty = max(1, Hc // tile)

    yy, xx = np.mgrid[0:Hc, 0:Wc]
    tx = np.minimum(xx // tile, n_tx - 1)
    ty = np.minimum(yy // tile, n_ty - 1)

    # Per-tile value range. Each tile's histogram is binned into its OWN
    # [min,max] so a smooth tile fills ALL bins → full local-contrast stretch.
    tmin = np.full((n_ty, n_tx), 1.0)
    tmax = np.full((n_ty, n_tx), 0.0)
    np.minimum.at(tmin, (ty.ravel(), tx.ravel()), channel.ravel())
    np.maximum.at(tmax, (ty.ravel(), tx.ravel()), channel.ravel())
    trange = np.maximum(tmax - tmin, 1e-3)

    # Per-pixel bin index within the pixel's OWN tile range (used for scatter).
    bin_local = np.clip(
        ((channel - tmin[ty, tx]) / trange[ty, tx] * (nbins - 1)).astype(np.int32),
        0, nbins - 1,
    )

    # Tile histogram via scatter-add (vectorized).
    hist = np.zeros((n_ty, n_tx, nbins), dtype=np.float64)
    np.add.at(hist, (ty.ravel(), tx.ravel(), bin_local.ravel()), 1.0)

    # Contrast limit = clip_frac × (average bin count per tile). 1 ≈ off (pure
    # adaptive equalization, strongest local-contrast boost); higher values clip
    # histogram spikes and suppress the blocking / over-enhancement artifacts.
    counts = hist.sum(axis=-1, keepdims=True)
    clip = clip_frac * (counts / nbins)
    excess = np.clip(hist - clip, 0.0, None).sum(axis=-1, keepdims=True)
    hist_c = np.minimum(hist, clip) + excess / nbins

    # CDF of the clipped histogram, normalized to [0,1] per tile.
    cdf = hist_c.cumsum(axis=-1)
    cdf_max = cdf[:, :, -1:]
    cdf = np.divide(cdf, cdf_max, out=np.zeros_like(cdf), where=cdf_max > 0.0)

    # Continuous tile coordinate centred on each tile (xx/tile - 0.5).
    xf = xx / tile - 0.5
    yf = yy / tile - 0.5
    tx0 = np.clip(np.floor(xf).astype(np.int32), 0, n_tx - 1)
    tx1 = np.clip(tx0 + 1, 0, n_tx - 1)
    ty0 = np.clip(np.floor(yf).astype(np.int32), 0, n_ty - 1)
    ty1 = np.clip(ty0 + 1, 0, n_ty - 1)
    wx1 = (xf - tx0).astype(np.float64)
    wy1 = (yf - ty0).astype(np.float64)
    wx0 = 1.0 - wx1
    wy0 = 1.0 - wy1

    # Bilinear blend of the 4 surrounding tile mapping functions. Each tile's
    # CDF is defined over THAT tile's [tmin,tmax] range, so a pixel's bin within
    # a neighbour tile is recomputed against the neighbour's own range.
    def _map_at(cy: np.ndarray, cx: np.ndarray) -> np.ndarray:
        vmin = tmin[cy, cx]
        vr = trange[cy, cx]
        bl = np.clip(((channel - vmin) / vr * (nbins - 1)).astype(np.int32),
                     0, nbins - 1)
        return cdf[cy, cx, bl]  # cy,cx,bl all (H,W) → (H,W)

    out = (wy0 * wx0 * _map_at(ty0, tx0)
           + wy0 * wx1 * _map_at(ty0, tx1)
           + wy1 * wx0 * _map_at(ty1, tx0)
           + wy1 * wx1 * _map_at(ty1, tx1))
    return out.astype(np.float32)


def _apply_clahe(img: np.ndarray, tile: int, clip_frac: float,
                 strength: float) -> tuple[np.ndarray, np.ndarray]:
    """Apply per-channel CLAHE to an RGB float image.

    Returns (equalized_rgb, gain_field) where gain_field is the per-pixel
    luminance boost (out_luma / in_luma) — useful as a FIELD / debug view.
    """
    eq = np.empty_like(img)
    for c in range(3):
        ch = img[:, :, c].astype(np.float32)
        eq_ch = _clahe_channel(ch, tile, clip_frac)
        # strength blends equalized ←→ original so the slider can be subtle.
        eq[:, :, c] = (1.0 - strength) * ch + strength * eq_ch
    eq = np.clip(eq, 0.0, 1.0)

    in_luma = (0.299 * img[:, :, 0] + 0.587 * img[:, :, 1] + 0.114 * img[:, :, 2])
    out_luma = (0.299 * eq[:, :, 0] + 0.587 * eq[:, :, 1] + 0.114 * eq[:, :, 2])
    gain = out_luma / (in_luma + 1e-4)
    return eq, gain.astype(np.float32)


@method(
    id="436",
    name="CLAHE",
    category="filters",
    new_image_contract=True,
    tags=["filter", "contrast", "clahe", "histogram", "adaptive", "local-contrast",
          "medical-imaging", "astrophotography", "enhancement", "animation"],
    inputs={"image_in": "IMAGE"},
    outputs={"image": "IMAGE", "field": "FIELD"},
    params={
        "source": {"description": "source when no image is wired (gradient/noise/palette/rainbow/procedural/input_image)",
                   "default": "procedural"},
        "tile_size": {"description": "CLAHE tile edge in px — smaller = more local contrast",
                      "choices": [4, 8, 16, 32, 64], "default": 8},
        "clip_limit": {"description": "contrast-limit multiple of the per-tile average bin count (1≈off/full equalization … higher=stronger spike limiting)",
                       "min": 1.0, "max": 8.0, "default": 4.0},
        "strength": {"description": "blend equalized ←→ original (0=off, 1=full CLAHE)",
                     "min": 0.0, "max": 1.0, "default": 1.0},
        "view": {"description": "IMAGE render: equalized result / local 'gain' (contrast boost) field",
                 "choices": ["equalized", "gain"], "default": "equalized"},
        "noise_amp": {"description": "noise amplitude for the noise source", "min": 0.1, "max": 1.0, "default": 0.8},
        "blur_sigma": {"description": "gaussian blur sigma for the noise source", "min": 5, "max": 80, "default": 30},
        "palette": {"description": "palette name for the palette source", "default": "vapor"},
        "time": {"description": "animation phase [0, 2pi)", "min": 0.0, "max": 6.28, "default": 0.0},
        "anim_mode": {"description": "animation mode (none/clip_sweep)",
                     "choices": ["none", "clip_sweep"], "default": "none"},
        "anim_speed": {"description": "animation speed multiplier", "min": 0.1, "max": 5.0, "default": 1.0},
    },
)
def method_clahe(out_dir: Path, seed: int, params=None):
    """CLAHE — Contrast-Limited Adaptive Histogram Equalization (Zuiderveld 1994).

    Boosts *local* contrast adaptively across the canvas while a contrast limit
    stops blown-out pixels from washing out whole tiles. Works as a standalone
    generator (built-in low-contrast source so the effect is visible) or as a
    filter on a wired upstream image (Rule #12: a wired `input_image` always
    overrides source generation).

    Params:
        source:     built-in source when nothing is wired (gradient/noise/palette/rainbow/procedural/input_image)
        tile_size:  CLAHE tile edge (px)
        clip_limit: contrast-limit multiple of the per-tile average bin count (1≈off, higher=stronger spike limiting)
        strength:   blend equalized ←→ original
        view:       equalized image / local gain (contrast-boost) field
        time:       animation clock [0, 2pi)
        anim_mode:  none (static) / clip_sweep (contrast limit breathes with t)
    """
    try:
        if params is None:
            params = {}
        out_dir = Path(out_dir)
        t = float(params.get("time", 0.0))
        anim_mode = str(params.get("anim_mode", "none"))
        anim_speed = float(params.get("anim_speed", 1.0))
        seed_all(seed)
        rng = np.random.default_rng(seed)

        tile = int(float(params.get("tile_size", 8)))
        if tile not in (4, 8, 16, 32, 64):
            tile = 8
        clip_limit = max(1.0, min(8.0, float(params.get("clip_limit", 4.0))))
        strength = max(0.0, min(1.0, float(params.get("strength", 1.0))))
        view = str(params.get("view", "equalized"))
        noise_amp = max(0.1, min(1.0, float(params.get("noise_amp", 0.8))))
        blur_sigma = max(5, min(80, float(params.get("blur_sigma", 30))))
        pal_name = str(params.get("palette", "vapor"))
        source = str(params.get("source", "gradient"))

        # ── Animation clock (rename t to avoid shadowing the time param) ──
        _t = t * anim_speed
        if anim_mode == "clip_sweep":
            # Sweep the contrast limit across its full 1..8 range so the
            # local-contrast strength breathes over time. Offset sine → no cusp.
            # Makes the UI-exposed `clip_sweep` mode actually drive the output
            # (the previous `amount_pulse` name never matched the schema choice).
            clip_limit = 1.0 + 7.0 * (0.5 + 0.5 * math.sin(_t * 0.4))

        # ── Resolve canvas size (robust if no canvas context is set) ──
        Hw = int(H)
        Ww = int(W)
        if Hw < 2 or Ww < 2:
            Hw, Ww = 512, 768

        # ── Build source image (float32 [0,1], H×W×3) ──
        # A wired upstream image always overrides source generation (Rule #12).
        src = None
        wired_path = params.get("input_image", "")
        if wired_path:
            try:
                src = load_input(wired_path, Ww, Hw)
            except (FileNotFoundError, OSError):
                src = None

        if src is None:
            if source == "gradient":
                yy, xx = np.mgrid[:Hw, :Ww].astype(np.float32)
                r = np.sqrt((xx - Ww / 2) ** 2 + (yy - Hw / 2) ** 2)
                r = r / (r.max() + 1e-6)
                # deliberately LOW local contrast so CLAHE has something to expand
                g = 0.35 + 0.3 * r
                src = np.stack([g, g * 0.85, 1 - 0.7 * g], axis=-1).clip(0, 1)
            elif source == "palette":
                pal = PALETTES.get(pal_name, list(PALETTES.values())[0])
                yy, xx = np.mgrid[:Hw, :Ww].astype(np.float32)
                r = np.sqrt((xx - Ww / 2) ** 2 + (yy - Hw / 2) ** 2)
                r = r / (r.max() + 1e-6)
                idx = (r * (len(pal) - 1)).astype(np.int32)
                src = np.array(pal, dtype=np.float32)[idx] / 255.0
            elif source == "rainbow":
                yy, xx = np.mgrid[:Hw, :Ww].astype(np.float32)
                r = np.sqrt((xx - Ww / 2) ** 2 + (yy - Hw / 2) ** 2)
                r = r / (r.max() + 1e-6)
                hue = r * 2 * math.pi
                src = np.stack([
                    np.sin(hue) * 0.5 + 0.5,
                    np.sin(hue + 2.094) * 0.5 + 0.5,
                    np.sin(hue + 4.189) * 0.5 + 0.5,
                ], axis=-1).astype(np.float32)
            elif source == "procedural":
                yy, xx = np.mgrid[:Hw, :Ww].astype(np.float32)
                g = (np.sin(xx * 0.03 + yy * 0.02) *
                     np.cos(xx * 0.02 - yy * 0.03) * 0.5 + 0.5)
                src = np.stack([g, g * 0.6, 1 - g * 0.8], axis=-1).astype(np.float32)
            else:  # noise
                from scipy.ndimage import gaussian_filter
                base = rng.random((Hw, Ww, 3)).astype(np.float32)
                base = (base - 0.5) * (2 * noise_amp) + 0.5
                if blur_sigma > 0:
                    for c in range(3):
                        base[:, :, c] = gaussian_filter(base[:, :, c], blur_sigma)
                src = base.clip(0, 1)

        src = np.asarray(src, dtype=np.float32).clip(0, 1)

        # ── Apply CLAHE ──
        eq, gain = _apply_clahe(src, tile, clip_limit, strength)

        # ── IMAGE output ──
        if view == "gain":
            # Visualise the per-pixel contrast boost: gain 1.0 → mid-gray,
            # >1 (boosted) → brighter, <1 (suppressed) → darker.
            gain_disp = np.clip(gain, 0.0, 2.0) / 2.0
            img = np.stack([gain_disp, gain_disp, gain_disp], axis=-1)
        else:
            img = eq

        capture_frame("436", img)
        # Architecture B: include the animation time so --animate frames don't
        # overwrite each other on disk (pitfall #12).
        save(img, mn(436, f"CLAHE t={_t:.2f}"), out_dir)
        try:
            write_field(out_dir, gain)
            out_luma = (0.299 * eq[:, :, 0] + 0.587 * eq[:, :, 1] + 0.114 * eq[:, :, 2])
            src_luma = (0.299 * src[:, :, 0] + 0.587 * src[:, :, 1] + 0.114 * src[:, :, 2])
            write_scalars(
                out_dir,
                tile_size=float(tile),
                clip_limit=float(clip_limit),
                strength=float(strength),
                out_luma_std=float(out_luma.std()),
                src_luma_std=float(src_luma.std()),
                contrast_gain=float(out_luma.std() / (src_luma.std() + 1e-6)),
            )
        except Exception:
            pass
        return img
    except Exception as exc:
        # Deterministic neutral fallback so the node never 500s.
        Hw = int(H) if int(H) >= 2 else 512
        Ww = int(W) if int(W) >= 2 else 768
        fb = np.full((Hw, Ww, 3), 0.5, dtype=np.float32)
        save(fb, mn(436, "CLAHE"), out_dir)
        print(f"[method_436] ERROR: {exc}")
        return fb
