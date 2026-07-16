from __future__ import annotations
import math
from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter

from ...core.registry import method
from ...core.utils import (
    save, norm, mn, seed_all, W, H, PALETTES, load_input, write_scalars,
)
from ...core.animation import capture_frame


# ── Domain Transform Edge-Aware Filtering (Gastal & Oliveira, SIGGRAPH 2011) ──
# O(N) edge-preserving smoothing that fakes a 2D bilateral filter with a pair of
# 1D recursive passes. The trick: remap each pixel into a "domain transform"
# coordinate that STRETCHES the distance across intensity edges, so a plain 1D
# exponential smoother (which only sees the stretched coordinate) automatically
# stops at boundaries. Per axis the spacing between neighbours is
#     d[i] = 1 + sigma_s * rg[i],   rg = |grad| / (sigma_r + |grad|)
# and the recursive weight is w[i] = exp(-d[i] / sigma_s). rg is ~1 on flat
# areas (compact spacing -> heavy smoothing) and -> 0 at edges (stretched
# spacing -> preserved). Two separable passes (H then V) repeated a few times
# approximate the full 2D filter in O(N) with no per-pixel neighbourhood.


def _dt_1d(line: list[float], w: list[float]) -> list[float]:
    """One axis of the Domain Transform recursive filter (Gastal & Oliveira).

    Forward + backward exponential smoother along a 1-D signal. ``w[i]`` is the
    edge-aware weight between sample ``i`` and its predecessor; values live in
    (0, 1) so the recurrence is numerically stable (no under/overflow).
    """
    n = len(line)
    if n == 0:
        return line
    out_f = line[:]
    prev = line[0]
    for i in range(1, n):
        wi = w[i]
        v = (line[i] + wi * prev) / (1.0 + wi)
        out_f[i] = v
        prev = v
    out_b = line[:]
    prev = line[n - 1]
    for i in range(n - 2, -1, -1):
        wi = w[i + 1]
        v = (line[i] + wi * prev) / (1.0 + wi)
        out_b[i] = v
        prev = v
    return [(out_f[i] + out_b[i]) * 0.5 for i in range(n)]


@method(
    id="991",
    name="Domain Transform",
    category="filters",
    new_image_contract=True,
    tags=["edge-preserving", "smoothing", "domain-transform", "bilateral", "npr", "expanded", "animation"],
    inputs={"image_in": "IMAGE"},
    outputs={"image": "IMAGE"},
    params={
        "source": {"description": "source (noise/gradient/input_image/palette/rainbow/procedural)", "default": "noise"},
        "sigma_s": {"description": "spatial smoothing scale (bigger = wider smooth regions)", "min": 1, "max": 20, "default": 8},
        "sigma_r": {"description": "range/edge sensitivity in [0,1] (smaller = sharper edges)", "min": 0.02, "max": 0.5, "default": 0.15},
        "iterations": {"description": "filtering passes (more = stronger edge sharpening)", "min": 1, "max": 4, "default": 3},
        "blend": {"description": "blend original source back in (0=pure filter, 1=original)", "min": 0.0, "max": 1.0, "default": 0.0},
        "noise_amp": {"description": "noise amplitude for generated sources", "min": 0.1, "max": 1.0, "default": 0.8},
        "blur_sigma": {"description": "gaussian blur sigma for noise source (keep low so edges survive to smooth)", "min": 5, "max": 80, "default": 14},
        "palette": {"description": "palette name for palette source", "default": "vapor"},
        "time": {"min": 0.0, "max": 6.28, "default": 0.0},
        "anim_mode": {"description": "animation mode (none/sigma_pulse/range_sweep/blend_sweep)", "choices": ["none", "sigma_pulse", "range_sweep", "blend_sweep"], "default": "none"},
        "anim_speed": {"description": "animation speed multiplier", "min": 0.1, "max": 5.0, "default": 1.0},
    },
)
def method_domain_transform(out_dir: Path, seed: int, params=None):
    """Domain Transform Edge-Aware Filter (Gastal & Oliveira, SIGGRAPH 2011).

    A real-time O(N) replacement for the bilateral filter. Instead of the
    expensive per-pixel spatial+range weighted average, it remaps the image into
    a "domain transform" that stretches distances across intensity edges, then
    runs a plain 1-D recursive exponential smoother along each axis. Because the
    smoother only sees the stretched coordinate, it naturally stops at edges:

        1. guidance = luminance;  gx, gy = |grad| along each axis
        2. rg = |grad| / (sigma_r + |grad|)        # ~1 flat, ->0 at edges
        3. spacing  d = 1 + sigma_s * rg           # stretched at edges
        4. weight   w = exp(-d / sigma_s)          # ~1 flat, small at edges
        5. recursive 1-D smoother (fwd+bwd) per row (w_x) and per col (w_y)
        6. repeat steps 5 a few times (iterations)

    Flat regions melt into smooth gradients (cartoon / HDR-detail look) while
    strong silhouettes survive untouched. CPU path is authoritative (pure numpy
    + scipy for source generation); no cv2 required.

    Params:
        source:     generated source type (noise/gradient/input_image/palette/rainbow/procedural)
        sigma_s:    spatial smoothing scale (1-20, default 8)
        sigma_r:    range/edge sensitivity (0.02-0.5, default 0.15)
        iterations: filtering passes (1-4, default 3)
        blend:      mix original source back in (0-1, default 0)
        noise_amp:  amplitude for generated sources (0.1-1.0)
        blur_sigma: blur sigma for noise source (5-80)
        palette:    palette name for palette source
        time:       animation clock (0-6.28)
        anim_mode:  none / sigma_pulse / range_sweep / blend_sweep
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

        source = str(params.get("source", "noise"))
        sigma_s = float(params.get("sigma_s", 12))
        sigma_s = max(1.0, min(50.0, sigma_s))
        sigma_r = float(params.get("sigma_r", 0.15))
        sigma_r = max(0.02, min(0.5, sigma_r))
        n_iter = int(params.get("iterations", 3))
        n_iter = max(1, min(4, n_iter))
        blend = float(params.get("blend", 0.0))
        blend = max(0.0, min(1.0, blend))
        noise_amp = float(params.get("noise_amp", 0.5))
        blur_sigma = float(params.get("blur_sigma", 30))
        pal_name = str(params.get("palette", "vapor"))

        # ── Animation (rename t to avoid shadowing the time param) ──
        _t = anim_time * anim_speed
        if anim_mode == "sigma_pulse":
            # Full sine period in [0,2pi]; sweep sigma_s across its FULL live
            # range [1,20] (params max). A smoothing filter's per-frame change is
            # intrinsically modest, so use the widest band. Test at true extremes
            # (t=0 vs t=3pi/2), not t=0 vs pi (both sit at sin=0).
            sigma_s = max(1.0, min(20.0, 1.0 + 19.0 * (0.5 + 0.5 * math.sin(_t))))
        elif anim_mode == "range_sweep":
            # Sweep edge sensitivity across its full live range [0.02, 0.5].
            # sigma_r only acts at edges (localized), so to make the sweep
            # clearly visible we also ride sigma_s with it: sharp edges (low
            # sigma_r) paired with light smoothing, soft edges (high sigma_r)
            # paired with heavy smoothing -> a large global smoothing change.
            sigma_r = max(0.02, min(0.5, 0.02 + 0.48 * (0.5 + 0.5 * math.sin(_t))))
            sigma_s = max(1.0, min(20.0, 4.0 + 16.0 * (0.5 + 0.5 * math.sin(_t))))
        elif anim_mode == "blend_sweep":
            # Dissolve between a lightly-smoothed and a heavily-smoothed version
            # of the source (both clearly distinct), which is far more visible
            # than filtered<->source for already-smooth sources.
            blend = 0.5 + 0.5 * math.sin(_t)  # full 0->1 sweep over one period
        # else: none -> static

        # ── Resolve source image (float32 [0,1], HxWx3) ──
        # A wired upstream image always overrides source generation (Rule #12).
        src = None
        wired_path = params.get("input_image", "")
        if wired_path:
            try:
                src = load_input(wired_path, int(W), int(H))
            except (FileNotFoundError, OSError):
                src = None

        if src is None:
            if source == "gradient":
                yy, xx = np.mgrid[:H, :W].astype(np.float32)
                r = norm(np.sqrt((xx - W / 2) ** 2 + (yy - H / 2) ** 2))
                src = np.stack([r, r * 0.6, 1 - r], axis=-1).clip(0, 1)
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
                # Seed-stable (no _t) so the `none` mode stays a true static baseline.
                yy, xx = np.mgrid[:H, :W].astype(np.float32)
                g = np.sin(xx * 0.03 + yy * 0.02) * \
                    np.cos(xx * 0.02 - yy * 0.03) * 0.5 + 0.5
                src = np.stack([g, g * 0.6, 1 - g * 0.8], axis=-1).astype(np.float32)
            else:  # noise / input_image fallback
                n = rng.standard_normal((H, W, 3)).astype(np.float32) * noise_amp + 0.5
                if blur_sigma >= 1.0:
                    n = gaussian_filter(n, sigma=blur_sigma, mode="reflect")
                src = norm(n)

        src = np.clip(src, 0.0, 1.0).astype(np.float32)
        Hh, Ww = src.shape[:2]

        def _filter(sig_s: float) -> np.ndarray:
            """Run the domain-transform edge-aware smoother at scale `sig_s`."""
            # Edge-aware recurrence coefficient a_i = exp(-d_i / sig_s), where the
            # domain-transform spacing d_i = 1 + sig_s * M_i and M_i = |grad|/(sigma_r
            # + |grad|) is the normalized gradient magnitude. The 1-D smoother below
            # consumes wi = a/(1-a), so a near 1 (flat region) -> large wi -> heavy
            # smoothing; a small (edge) -> small wi -> preserved. This is what makes
            # sigma_s / sigma_r actually bite instead of collapsing to a fixed blend.
            rg_x = gx / (sigma_r + gx)
            rg_y = gy / (sigma_r + gy)
            base = math.exp(-1.0 / sig_s)
            a_x = np.clip((base * np.exp(-rg_x)).astype(np.float32), 0.0, 0.9995)
            a_y = np.clip((base * np.exp(-rg_y)).astype(np.float32), 0.0, 0.9995)
            wx = (a_x / (1.0 - a_x)).astype(np.float32)  # smoother weight along x
            wy = (a_y / (1.0 - a_y)).astype(np.float32)  # smoother weight along y
            res = src.copy()
            for _it in range(n_iter):
                for r in range(Hh):
                    line = res[r, :, 0].tolist(); wl = wx[r, :].tolist()
                    res[r, :, 0] = _dt_1d(line, wl)
                    line = res[r, :, 1].tolist()
                    res[r, :, 1] = _dt_1d(line, wl)
                    line = res[r, :, 2].tolist()
                    res[r, :, 2] = _dt_1d(line, wl)
                for c in range(Ww):
                    line = res[:, c, 0].tolist(); wl = wy[:, c].tolist()
                    res[:, c, 0] = _dt_1d(line, wl)
                    line = res[:, c, 1].tolist()
                    res[:, c, 1] = _dt_1d(line, wl)
                    line = res[:, c, 2].tolist()
                    res[:, c, 2] = _dt_1d(line, wl)
            return np.clip(res, 0.0, 1.0).astype(np.float32)

        # ── Domain transform weights from luminance guidance ─
        lum = (0.299 * src[:, :, 0] + 0.587 * src[:, :, 1] + 0.114 * src[:, :, 2]).astype(np.float32)
        gy, gx = np.gradient(lum)  # gradient returns [d/dy, d/dx]
        gx = np.abs(gx).astype(np.float32)
        gy = np.abs(gy).astype(np.float32)

        out = _filter(sigma_s)

        if blend > 0.0 or anim_mode == "blend_sweep":
            # blend_sweep dissolves between the raw source and a heavily filtered
            # version (clearly distinct). The blend>0 OR blend_sweep guard ensures
            # the raw-source extreme (blend=0) is still produced, not skipped.
            if anim_mode == "blend_sweep":
                # Dissolve between the raw source and a *heavily* filtered version
                # (clearly distinct from source, unlike a lightly-smoothed one).
                heavy = _filter(20.0)
                out = (src * (1.0 - blend) + heavy * blend).astype(np.float32)
            else:
                out = (out * (1.0 - blend) + src * blend).astype(np.float32)
            out = np.clip(out, 0.0, 1.0).astype(np.float32)

        capture_frame("991", out)
        save(out, mn(991, f"Domain Transform t={_t:.2f}"), out_dir)
        try:
            write_scalars(out_dir, sigma_s=float(sigma_s), sigma_r=float(sigma_r),
                          iterations=float(n_iter), blend=float(blend))
        except Exception:
            pass
        return out
    except Exception as exc:
        fallback = np.full((int(H), int(W), 3), 0.93, dtype=np.float32)
        save(fallback, mn(991, "Domain Transform"), out_dir)
        print(f"[method_991] ERROR: {exc}")
        return fallback
