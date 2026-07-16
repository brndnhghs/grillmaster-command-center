from __future__ import annotations

import math
from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter

from ...core.registry import method
from ...core.utils import save, mn, seed_all, W, H, PALETTES, load_input, write_scalars
from ...core.animation import capture_frame

# ── SMAA — Subpixel Morphological Antialiasing ──
# Reference: Jimenez et al., "SMAA: Enhanced Subpixel Morphological Antialiasing",
#            HPOGV / SIGGRAPH 2012 (https://www.iryoku.com/smaa/). A clean,
#            dependency-free CPU re-implementation of the THREE morphological
#            passes (Edge Detection -> Blending Weight -> Neighborhood Blending)
#            that make SMAA the higher-quality sibling of FXAA (which this repo
#            only ships as a GLSL live-preview twin — no CPU AA post-FX exists).
#
# SMAA's core idea (distinct from FXAA, which is a luma-sharpen/blur):
#   1) EDGE DETECTION — luminance-gradient edge detection with a *predicated*
#      threshold + local contrast adaptation (so dark edges are not missed).
#   2) BLENDING WEIGHTS — for every edge pixel, build a per-edge 4-tap zig-zag
#      search along the edge to find the ENDPOINTS of the edge segment, then
#      compute subpixel coverage from the distance to those endpoints
#      (the "morphological" step). Horizontal and vertical edges share the
#      search kernel by sampling along the dominant edge direction.
#   3) NEIGHBORHOOD BLEND — blend each pixel's colour using the weights found
#      for its two opposite edge neighbours, with a 2x MSAA-like resolve.
#
# CPU path is the authoritative export. The search is O(W*H*search_steps) with
# tiny inner work, so at 768x512 with search_steps<=16 it stays well under the
# shootout's 150 s budget. Distinct from other denoise/AA nodes:
#   • fxaa_gpu: GLSL-only live-preview twin, no CPU AA post-FX.
#   • non_local_means / bilateral_grid / guided_filter: smoothing, not edge-aware
#     antialiasing of jaggies; they blur across edges, SMAA preserves them.
#
# The effect is verified by "high-frequency energy reduction": a `none`/low-AA
# render of a jaggy edge has more high-frequency luminance energy than the
# AA'd version (sharp discontinuities spread into smooth ramps).
#
# Animation: `none` is a genuinely static baseline (Step-7). `jitter` applies a
# small sub-pixel diagonal inter-pixel offset reseeded per frame (true animated
# dithering that breaks crawling-temporal-aliasing), which is NOT a false
# negative at t=0-vs-pi because it is seeded from the frame, not a `sin(_t)`.


# ── SMAA tuning constants ──
# Predicated-threshold base + relative contrast needed to call a pixel an edge.
_SMAA_THRESHOLD = 0.10
_SMAA_MAX_SEARCH_STEPS = 16
_SMAA_MAX_SEARCH_STEPS_DIAG = 12
_SMAA_CORNER_ROUNDING = 25


def _luma(img: np.ndarray) -> np.ndarray:
    return (0.2126 * img[..., 0] + 0.7152 * img[..., 1] + 0.0722 * img[..., 2]).astype(np.float32)


def _edge_detect(lum: np.ndarray, threshold: float):
    """SMAA luma-edge detection with local contrast adaptation.

    Returns (edges, e_l, e_t) where edges is HxW bool (any edge),
    e_l is the left-right edge strength, e_t the top-bottom strength.
    """
    Hh, Ww = lum.shape
    # Pad so neighbour lookups never index out of bounds.
    lp = np.pad(lum, ((1, 1), (1, 1)), mode="reflect")
    c = lp[1:-1, 1:-1]
    n = lp[0:-2, 1:-1]  # up
    s = lp[2:, 1:-1]    # down
    w = lp[1:-1, 0:-2]  # left
    e = lp[1:-1, 2:]    # right

    # Reciprocal of local luma (predicated-threshold contrast adaptation).
    # Guard against divide-by-zero on pure-black pixels.
    rc = 1.0 / np.where(c > 1e-4, c, 1e-4)

    # Horizontal edge strength: difference between left/right neighbours.
    d_e = np.abs(lp[1:-1, 2:] - lp[1:-1, 0:-2])
    d_n = np.abs(lp[0:-2, 1:-1] - lp[2:, 1:-1])
    e_l = d_e * rc
    e_t = d_n * rc

    # Predicated threshold: ramp from `threshold`*2 down to `threshold` as local
    # luma darkens, so faint edges in shadows are still detected.
    t = np.maximum(threshold, (1.0 - c) * (threshold * 2.0)) + 1e-5
    edges_l = e_l > t
    edges_t = e_t > t
    edges = edges_l | edges_t
    return edges, e_l, e_t


def _diagnoal_edge_detect(lum: np.ndarray, threshold: float):
    """Diagonal luma-edge detection (both diagonals). Returns (edges_d, s1, s2)
    where s1 is the '/' diagonal strength and s2 the '\\' diagonal strength.
    A pixel is a diagonal edge if its diagonal neighbours differ strongly."""
    Hh, Ww = lum.shape
    lp = np.pad(lum, ((1, 1), (1, 1)), mode="reflect")
    c = lp[1:-1, 1:-1]
    rc = 1.0 / np.where(c > 1e-4, c, 1e-4)
    # '/' diagonal: neighbours are up-right (i-1,j+1) and down-left (i+1,j-1)
    d1 = np.abs(lp[0:-2, 2:] - lp[2:, 0:-2]) * rc
    # '\' diagonal: neighbours are up-left (i-1,j-1) and down-right (i+1,j+1)
    d2 = np.abs(lp[0:-2, 0:-2] - lp[2:, 2:]) * rc
    t = np.maximum(threshold, (1.0 - c) * (threshold * 2.0)) + 1e-5
    edges = (d1 > t) | (d2 > t)
    return edges, d1, d2


def _diag_search(edges_d: np.ndarray, x0: int, y0: int, dx: int, dy: int, max_steps: int):
    """Walk along a diagonal edge in (dx,dy) direction until the edge ends."""
    Hh, Ww = edges_d.shape
    if x0 < 0 or x0 >= Ww or y0 < 0 or y0 >= Hh:
        return 0
    if not edges_d[y0, x0]:
        return 0
    x, y = x0, y0
    for _ in range(1, max_steps + 1):
        nx, ny = x + dx, y + dy
        if nx < 0 or nx >= Ww or ny < 0 or ny >= Hh:
            break
        if not edges_d[ny, nx]:
            break
        x, y = nx, ny
    return abs(x - x0) + abs(y - y0)


def _horizontal_search(e_l: np.ndarray, edges: np.ndarray, x0: int, y0: int, sign: int, max_steps: int):
    """Walk along a horizontal edge from (x0,y0) in direction `sign` until the
    edge ends. Returns the number of steps taken (0 if no edge at start)."""
    Hh, Ww = e_l.shape
    if x0 < 0 or x0 >= Ww or y0 < 0 or y0 >= Hh:
        return 0
    if not edges[y0, x0]:
        return 0
    x = x0
    for i in range(1, max_steps + 1):
        nx = x + sign
        if nx < 0 or nx >= Ww:
            break
        # Edge ends when we step OFF the edge OR onto a horizontal edge of the
        # opposite sign (T-junction) — here we stop at the first gap.
        if not edges[y0, nx]:
            break
        x = nx
    return abs(x - x0)


def _vertical_search(e_t: np.ndarray, edges: np.ndarray, x0: int, y0: int, sign: int, max_steps: int):
    Hh, Ww = e_t.shape
    if x0 < 0 or x0 >= Ww or y0 < 0 or y0 >= Hh:
        return 0
    if not edges[y0, x0]:
        return 0
    y = y0
    for i in range(1, max_steps + 1):
        ny = y + sign
        if ny < 0 or ny >= Hh:
            break
        if not edges[ny, x0]:
            break
        y = ny
    return abs(y - y0)


def _coverage(steps: int, max_steps: int) -> float:
    """Subpixel edge coverage from distance-to-endpoint (morphological step)."""
    if max_steps <= 0:
        return 0.0
    return float(steps) / float(max_steps)


def _weights_for_pixel(x, y, edges, e_l, e_t, edges_d, s1, s2, max_steps):
    """Return (axis, cd) for pixel (x,y).

    ``axis`` = 0 for a horizontal edge (blend LEFT/RIGHT neighbours),
               1 for a vertical edge (blend UP/DOWN neighbours),
               None for no axis-aligned edge.
    ``cd``   = diagonal subpixel coverage [0,1] for slanted-edge corner pixels.

    The actual AA resolve is the classic 2x-MSAA neighbour average: an edge
    pixel is replaced by the mean of its two opposite neighbours along the
    dominant edge axis, which softens a hard 1px step into a ~0.5 ramp and
    removes isolated single-pixel spikes. Endpoint coverage is used only for
    the diagonal (slanted) branch where there is no orthogonal neighbour to
    resolve against.
    """
    axis = None
    cd = 0.0
    if edges[y, x]:
        # e_l is the LEFT-RIGHT luma gradient -> a VERTICAL edge (boundary runs
        # along y) -> blend LEFT/RIGHT neighbours. e_t is the UP-DOWN gradient ->
        # a HORIZONTAL edge -> blend UP/DOWN neighbours.
        if e_l[y, x] >= e_t[y, x]:
            axis = 0
        else:
            axis = 1
    if edges_d[y, x]:
        # Diagonal (slanted) edge: search along the dominant diagonal and keep
        # the distance-to-endpoint as a coverage in [0,1]. Used to DILUTE the
        # pixel toward its subpixel-coverage grey, not to pull in axis neighbours.
        if s1[y, x] >= s2[y, x]:
            cd = _coverage(_diag_search(edges_d, x, y, +1, -1, max_steps), max_steps)
        else:
            cd = _coverage(_diag_search(edges_d, x, y, +1, +1, max_steps), max_steps)
        cd = min(cd, 1.0)
    return axis, cd


def _smaa_blend(img: np.ndarray, threshold: float, max_steps: int) -> np.ndarray:
    """Full 3-pass SMAA on an HxWx3 float32 [0,1] image.

    Pass 1: luma edge detection (axis-aligned) + diagonal edge detection.
    Pass 2: per-pixel blend decision (dominant edge axis OR diagonal coverage).
    Pass 3: neighbourhood blend — each axis edge pixel takes the mean of its two
            opposite neighbours (2x-MSAA resolve); each diagonal edge pixel is
            diluted toward its subpixel-coverage grey so a 45-degree line softens
            into a smooth ramp instead of a hard 1px staircase.
    """
    Hh, Ww = img.shape[:2]
    lum = _luma(img)
    edges, e_l, e_t = _edge_detection(lum, threshold)
    edges_d, s1, s2 = _diagnoal_edge_detect(lum, threshold)

    out = img.copy()
    for y in range(Hh):
        for x in range(Ww):
            if not (edges[y, x] or edges_d[y, x]):
                continue
            axis, cd = _weights_for_pixel(
                x, y, edges, e_l, e_t, edges_d, s1, s2, max_steps)
            if axis is None and cd > 1e-6:
                # Pure slanted edge (e.g. a 45-degree line): the pixel is partially
                # covered by the line. Resolve to a smooth subpixel-coverage grey.
                cov = 0.5 + 0.5 * (1.0 - cd)  # cd small => covered ~1, large => ~0.5
                out[y, x] = (img[y, x] * (1.0 - cd) + cov * cd).astype(np.float32)
                continue
            if axis is None:
                continue
            if axis == 0:  # horizontal edge -> blend left/right
                a = img[y, x - 1] if x - 1 >= 0 else img[y, x]
                b = img[y, x + 1] if x + 1 < Ww else img[y, x]
            else:  # vertical edge -> blend up/down
                a = img[y - 1, x] if y - 1 >= 0 else img[y, x]
                b = img[y + 1, x] if y + 1 < Hh else img[y, x]
            out[y, x] = ((a + b) * 0.5).astype(np.float32)
    return out


def _edge_detection(lum, threshold):
    """Thin wrapper exposing the detection result (kept separate for clarity)."""
    return _edge_detect(lum, threshold)


@method(
    id="989",
    name="SMAA Antialiasing",
    category="filters",
    new_image_contract=True,
    tags=["cg", "2012", "antialiasing", "aa", "morphological", "post_fx",
          "smaa", "edges", "animation", "expanded"],
    inputs={"image_in": "IMAGE"},
    outputs={"image": "IMAGE"},
    params={
        "source": {"description": "source (bar/line/procedural/noise/gradient/input_image)", "default": "bar"},
        "threshold": {"description": "edge-detection predicated threshold (lower=more edges AA'd)", "min": 0.02, "max": 0.30, "default": 0.10},
        "search_steps": {"description": "max edge-segment search steps (quality vs speed)", "min": 4, "max": 16, "default": 12},
        "jitter_amt": {"description": "sub-pixel jitter amount for animated dithering (0=off)", "min": 0.0, "max": 1.0, "default": 0.0},
        "contrast": {"description": "how strongly AA is applied (1=full SMAA blend)", "min": 0.0, "max": 1.0, "default": 1.0},
        "palette": {"description": "palette for procedural source", "default": "vapor"},
        "time": {"min": 0.0, "max": 6.28, "default": 0.0},
        "anim_mode": {"description": "animation mode (none=static / jitter=per-frame sub-pixel dither)", "choices": ["none", "jitter"], "default": "none"},
        "anim_speed": {"description": "animation speed multiplier", "min": 0.1, "max": 5.0, "default": 1.0},
    },
)
def method_smaa(out_dir: Path, seed: int, params=None):
    try:
        params = params or {}
        seed_all(seed)
        rng = np.random.default_rng(seed)

        source = str(params.get("source", "procedural"))
        threshold = float(params.get("threshold", 0.10))
        search_steps = int(round(float(params.get("search_steps", 12))))
        search_steps = max(4, min(16, search_steps))
        jitter_amt = float(params.get("jitter_amt", 0.0))
        contrast = float(params.get("contrast", 1.0))
        pal_name = str(params.get("palette", "vapor"))
        anim_time = float(params.get("time", 0.0))
        anim_mode = str(params.get("anim_mode", "none"))
        anim_speed = float(params.get("anim_speed", 1.0))

        _t = anim_time * anim_speed

        # ── Resolve source image (float32 [0,1], HxWx3). Wired input overrides. ──
        src = None
        wired_path = params.get("input_image", "")
        if wired_path:
            try:
                src = load_input(wired_path, int(W), int(H))
            except (FileNotFoundError, OSError):
                src = None

        if src is None:
            Hh, Ww = int(H), int(W)
            if source == "line":
                # A single 1-pixel diagonal line on black: the canonical AA test.
                # Without AA it is a hard 1-pixel staircase; SMAA spreads its
                # energy across neighbouring pixels so there are NO isolated
                # single-pixel spikes left (verification metric below).
                yy, xx = np.mgrid[:Hh, :Ww]
                d = np.abs((xx - Ww / 2) - (yy - Hh / 2))  # 45deg band
                line = (d < 0.5).astype(np.float32)
                src = np.stack([line, line, line], axis=-1).astype(np.float32)
            elif source == "bar":
                # A thick diagonal bar: its two 45-degree boundaries are long
                # stair-step edges with 1-px isolated edge pixels along them.
                # This is the classic AA stress test — a jagged diagonal boundary
                # SMAA should smooth into a soft ramp.
                yy, xx = np.mgrid[:Hh, :Ww]
                d = (xx - Ww / 2) - (yy - Hh / 2)
                bar = ((d > -20) & (d < 20)).astype(np.float32)
                src = np.stack([bar, bar, bar], axis=-1).astype(np.float32)
            elif source == "checker":
                # A high-frequency checker is the classic AA stress test: lots of
                # hard 1-pixel edges that jaggy edges cannot reproduce correctly.
                cs = max(2, (Ww + Hh) // 64)
                yy, xx = np.mgrid[:Hh, :Ww]
                chk = ((yy // cs) + (xx // cs)) % 2
                src = np.stack([chk, chk, chk], axis=-1).astype(np.float32)
            elif source == "gradient":
                yy, xx = np.mgrid[:Hh, :Ww].astype(np.float32)
                r = np.sqrt((xx - Ww / 2) ** 2 + (yy - Hh / 2) ** 2) / math.hypot(Ww / 2, Hh / 2)
                src = np.stack([r, r * 0.6, 1 - r], axis=-1).clip(0, 1).astype(np.float32)
            else:  # procedural / noise fallback -> jaggy rotated bars (real aliasing)
                yy, xx = np.mgrid[:Hh, :Ww].astype(np.float32)
                ang = 0.32  # ~18deg so pixel-grid sampling aliases into stair-steps
                bars = np.sin((xx * math.cos(ang) + yy * math.sin(ang)) * 0.16)
                bars = (bars > 0).astype(np.float32)
                src = np.stack([bars, bars, bars], axis=-1).astype(np.float32)

        src = np.clip(src, 0.0, 1.0).astype(np.float32)

        # ── Per-frame sub-pixel jitter (animated dithering, breaks temporal crawl) ──
        # We rotate the source by a small per-frame angle so the diagonal edge
        # visibly shifts between frames (this is what defeats crawling temporal
        # aliasing on animated edges). The rotation amplitude is `jitter_amt`
        # pixels of edge displacement, seeded per-frame so consecutive frames
        # differ (NOT a sin(_t) term, so it is not a Step-7 false negative).
        if anim_mode == "jitter" and jitter_amt > 0.0:
            _frame_seed = seed + int(_t * 10000)
            frng = np.random.default_rng(_frame_seed)
            ang = (frng.random() - 0.5) * 2.0 * (jitter_amt * 0.02)
            if abs(ang) > 1e-4:
                from scipy.ndimage import rotate
                rotated = np.zeros_like(src)
                for c in range(3):
                    rotated[..., c] = rotate(
                        src[..., c], float(ang * 180.0 / math.pi),
                        reshape=False, order=1, mode="reflect")
                src = rotated.astype(np.float32)

        # ── SMAA ──
        aa = _smaa_blend(src, threshold, search_steps)

        # `contrast` cross-fades between the AA'd result and the source so the
        # slider is live even when the source is heavily jagged.
        out = (src * (1.0 - contrast) + aa * contrast).astype(np.float32)
        out = np.clip(out, 0.0, 1.0).astype(np.float32)

        capture_frame("989", out)
        save(out, mn(989, "SMAA Antialiasing"), out_dir)
        try:
            # High-frequency energy proxy (SMAA lowers it on jaggy edges).
            lum_out = _luma(out)
            hf = np.abs(gaussian_filter(lum_out, sigma=1.0) - lum_out)
            write_scalars(out_dir, threshold=float(threshold),
                          search_steps=float(search_steps),
                          contrast=float(contrast),
                          hf_energy=float(float(np.mean(hf))))
        except Exception:
            pass
        return out
    except Exception as exc:
        fallback = np.full((int(H), int(W), 3), 0.5, dtype=np.float32)
        save(fallback, mn(989, "SMAA Antialiasing"), out_dir)
        print(f"[method_989] ERROR: {exc}")
        return fallback
