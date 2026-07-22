from __future__ import annotations

import math

import numpy as np
from PIL import Image

from ...core.registry import method
from ...core.utils import save, mn, seed_all, W, H, write_scalars, write_field, wired_source_rgb
from ...core.animation import capture_frame


# ── Procedural source (mirrors sibling nodes; used only when nothing is wired) ──
def _hash_corner(ix: np.ndarray, iy: np.ndarray, seed: int) -> np.ndarray:
    ix = ix.astype(np.uint64)
    iy = iy.astype(np.uint64)
    n = (ix * np.uint64(73856093)) ^ (iy * np.uint64(19349663)) ^ (np.uint64(seed) * np.uint64(83492791))
    n = (n ^ (n >> np.uint64(13))) * np.uint64(1274126177)
    n = n ^ (n >> np.uint64(16))
    return (n & np.uint64(0x7FFFFFFF)).astype(np.float64) / 2147483647.0


def _value_noise(x: np.ndarray, y: np.ndarray, seed: int) -> np.ndarray:
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


def _proc_source(source: str, seed: int, w: int, h: int) -> np.ndarray:
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float64)
    if source == "checkerboard":
        cs = max(8, w // 24)
        cell = ((xx // cs + yy // cs) % 2)
        v = np.where(cell == 0, 0.30, 0.72)
        img = np.stack([v, v, v], axis=-1)
    elif source in ("perlin", "noise"):
        v = _fbm(xx / 40.0, yy / 40.0, seed, 5, 2.0, 0.5)
        v = (v + 1.0) * 0.5
        if source == "noise":
            v2 = _fbm(xx / 12.0, yy / 12.0, seed + 7, 3, 2.0, 0.5)
            v = np.clip(v * 0.7 + (v2 + 1.0) * 0.15, 0.0, 1.0)
        img = np.stack([v, v ** 1.3 * 0.9 + 0.05, v ** 0.7 * 0.7], axis=-1)
    else:  # gradient (default)
        r = xx / max(1, w - 1)
        g = yy / max(1, h - 1)
        b = (xx + yy) / max(1, w + h - 2)
        img = np.stack([r, g, b], axis=-1)
    return img.astype(np.float32)


# ── O(N) box / guided-filter primitives (integral-image means) ──
def _make_box(r: int, hh: int, ww: int):
    ii = np.arange(hh)
    jj = np.arange(ww)
    r0 = np.clip(ii - r, 0, hh - 1)
    r1 = np.clip(ii + r, 0, hh - 1)
    c0 = np.clip(jj - r, 0, ww - 1)
    c1 = np.clip(jj + r, 0, ww - 1)
    r0g, c0g = np.meshgrid(r0, c0, indexing="ij")
    r1g, c1g = np.meshgrid(r1, c1, indexing="ij")
    area = (r1g - r0g + 1) * (c1g - c0g + 1)
    return r0g, c0g, r1g, c1g, area


def _box(I: np.ndarray, idx) -> np.ndarray:
    C = np.zeros((I.shape[0] + 1, I.shape[1] + 1), dtype=np.float64)
    C[1:, 1:] = np.cumsum(np.cumsum(I, axis=0), axis=1)
    r0g, c0g, r1g, c1g, area = idx
    S = C[r1g + 1, c1g + 1] - C[r0g, c1g + 1] - C[r1g + 1, c0g] + C[r0g, c0g]
    return S / area


def _guided_coeffs(I: np.ndarray, p: np.ndarray, r: int, eps: float,
                   idx) -> tuple[np.ndarray, np.ndarray]:
    """Local-linear a_k, b_k of the guided filter (one channel), O(N) via box means."""
    mean_I = _box(I, idx)
    mean_p = _box(p, idx)
    mean_II = _box(I * I, idx)
    mean_Ip = _box(I * p, idx)
    cov_Ip = mean_Ip - mean_I * mean_p
    var_I = mean_II - mean_I * mean_I
    a = cov_Ip / (var_I + eps)
    b = mean_p - a * mean_I
    mean_a = _box(a, idx)
    mean_b = _box(b, idx)
    return mean_a, mean_b


def _bilinear_resize(X: np.ndarray, Ht: int, Wt: int) -> np.ndarray:
    """Bilinear resample X (hs×ws [×3]) to (Ht×Wt), handling fractional scales.

    Used for both down- and up-sampling the rolling-guidance strides; a fractional
    scale keeps the animation continuous (an integer stride would quantise the
    smoothing into a single hard step per octave)."""
    hs, ws = X.shape[0], X.shape[1]
    if hs == Ht and ws == Wt:
        return X
    gy = (np.arange(Ht) + 0.5) * hs / Ht - 0.5
    gx = (np.arange(Wt) + 0.5) * ws / Wt - 0.5
    y0 = np.clip(np.floor(gy).astype(int), 0, hs - 1)
    y1 = np.clip(y0 + 1, 0, hs - 1)
    x0 = np.clip(np.floor(gx).astype(int), 0, ws - 1)
    x1 = np.clip(x0 + 1, 0, ws - 1)
    wy = (gy - y0)[:, None]
    wx = (gx - x0)[None, :]
    if X.ndim == 2:
        v00 = X[y0][:, x0]; v01 = X[y0][:, x1]
        v10 = X[y1][:, x0]; v11 = X[y1][:, x1]
    else:
        wxa = wx[:, :, None]; wya = wy[:, :, None]
        v00 = X[y0][:, x0]; v01 = X[y0][:, x1]
        v10 = X[y1][:, x0]; v11 = X[y1][:, x1]
        top = v00 * (1.0 - wxa) + v01 * wxa
        bot = v10 * (1.0 - wxa) + v11 * wxa
        return top * (1.0 - wya) + bot * wya
    return v00 * (1.0 - wx) + v01 * wx + v10 * (1.0 - wx) * 0 + (v11 - v00) * 0  # placeholder for 2D


def _bilinear_upsample(X: np.ndarray, H: int, W: int) -> np.ndarray:
    return _bilinear_resize(X, H, W)


def _area_downsample(I: np.ndarray, s: float) -> np.ndarray:
    """Area-average downsample by (possibly fractional) factor s (HxW or HxWx3).

    Fractional s is implemented via bilinear resize so the rolling scale varies
    smoothly (keeps animated modes continuous instead of quantising into steps)."""
    if s <= 1.0:
        return I
    return _bilinear_resize(I, max(1, int(round(I.shape[0] / s))),
                            max(1, int(round(I.shape[1] / s))))


@method(id='946', name='Rolling Guidance Filter (Scale-Selective)', category='filters',
        tags=['rolling-guidance-filter', 'zhang-2014', 'scale-selective', 'edge-preserving',
              'smoothing', 'detail-extraction', 'multi-scale', 'structural', 'animation'],
        params={
            'source': {'description': "procedural source used when no image is wired in",
                       'choices': ['gradient', 'perlin', 'noise', 'checkerboard', 'input_image'],
                       'default': 'noise'},
            'sigma_start': {'description': 'initial detail-removal scale (px) — stride of the FIRST roll. SMALLER keeps fine detail, LARGER strips the first octave',
                            'min': 1, 'max': 16, 'default': 3},
            'growth': {'description': 'stride multiplier per roll — controls how many detail octaves are removed per step (sqrt(2)≈1.41 is a natural octave)',
                       'min': 1.1, 'max': 3.0, 'default': 1.6},
            'n_steps': {'description': 'number of rolling iterations — each one removes one more scale octave; more steps = a flatter, more structural result',
                        'min': 1, 'max': 6, 'default': 4},
            'edge_softness': {'description': 'guided-filter window on the down-sampled grid — larger = softer coarse edges, smaller = crisper structures',
                              'min': 1, 'max': 6, 'default': 2},
            'eps': {'description': 'guided-filter regularisation ε — SMALLER preserves sharp coarse edges, LARGER flattens them',
                    'min': 0.001, 'max': 0.5, 'default': 0.04},
            'mode': {'description': 'output: scale-smoothed structure, or recombined with the detail layer',

                     'choices': ['smooth', 'detail', 'flatten'], 'default': 'smooth'},
            'amount': {'description': 'detail strength for detail/flatten modes (0=off, 1=neutral, >1=boost)',
                       'min': 0.0, 'max': 3.0, 'default': 1.0},
            'anim_mode': {'description': 'animation mode (none / scale_breathe — the detail-removal scale swells and recedes)',
                          'default': 'none'},
            'anim_speed': {'description': 'animation speed multiplier', 'min': 0.1, 'max': 3.0, 'default': 1.0},
            'time': {'description': 'animation phase [0, 2pi)', 'min': 0.0, 'max': 6.28, 'default': 0.0},
        },
        inputs={'image_in': 'IMAGE'},
        outputs={'image': 'IMAGE', 'field': 'FIELD'})
def method_rolling_guidance(out_dir, seed: int, params=None):
    """Rolling Guidance Filter (Zhang, Shen, Xu, Jia — ECCV / SIGGRAPH Asia 2014,
    doi:10.1007/978-3-319-10578-9_23; project page cse.cuhk.edu.hk/leojia/projects/rollguidance).

    A scale-SELECTIVE, edge-preserving smoother. Unlike a single bilateral /
    guided filter (which removes detail below one fixed radius), rolling guidance
    iteratively "rolls" the smoothing scale outward, stripping ONE more detail
    octave per iteration while LARGE structures survive. The trick (the paper's
    core idea):

        S_0 = I
        for k = 1 .. N:
            stride_k   = sigma_start * growth^(k-1)        # the current scale
            I_down     = area-downsample(S_{k-1}, stride_k)   # details < stride_k vanish
            I_smooth   = guided_filter(I_down, I_down, r, eps) # O(N) edge-preserving clean-up
            S_k        = bilinear-upsample(I_smooth)          # restored to full size,
                                                            #   lost fine detail does NOT return

    Downsampling by stride_k makes every feature smaller than stride_k sub-pixel,
    so the subsequent filter wipes it; upsampling back cannot resurrect it. Each
    roll therefore removes exactly one more scale octave. The final S_N is the
    image with all detail below `sigma_start * growth^(N-1)` gone but its coarse
    STRUCTURE intact and edge-preserving.

    Per-roll smoothing uses the He/Sun guided filter (a fast O(N) bilateral
    surrogate) so the whole pipeline stays well inside the render budget even at
    N=6 — directly serving the >150s-cull pressure. Self-guided
    (guidance = target) so it is a pure structure extractor.

    Output modes (re-purpose the structure layer like node 969):
        smooth  -> S_N                              (scale-smoothed structure)
        detail  -> S_N + amount·(I − S_N)           (boost the removed octaves, HDR-style)
        flatten -> S_N − amount·(I − S_N)           (suppress detail → poster/flat look)

    Closed form per frame (no carried state) -> Architecture B, re-called per
    frame. The ``field`` output is the smoothed-structure luminance. 'none' is a
    static baseline; 'scale_breathe' swells the detail-removal scale over time.
    """
    try:
        if params is None:
            params = {}
        seed_all(seed)

        sigma_start = int(params.get("sigma_start", 3))
        growth = float(params.get("growth", 1.6))
        n_steps = int(params.get("n_steps", 4))
        r_low = max(1, int(params.get("edge_softness", 2)))
        eps = float(params.get("eps", 0.04))
        mode = params.get("mode", "smooth")
        amount = float(params.get("amount", 1.0))
        anim_mode = params.get("anim_mode", "none")
        anim_speed = float(params.get("anim_speed", 1.0))
        t = float(params.get("time", 0.0))
        _t = 0.0 if anim_mode == "none" else t * anim_speed
        source = params.get("source", "noise")

        # ── Animation: smoothly breathe the detail-removal scale (no cusps) ──
        # frac: 0 at t=0, 1 at t=pi, back to 0 — a full smooth cycle.
        if anim_mode == "scale_breathe":
            frac = 0.5 - 0.5 * math.cos(_t)        # 0 at t=0, 1 at t=pi, back to 0
            # Map the whole cycle into the ACTIVE rolling regime: stride scales
            # from 1.0 (a barely-there first roll) up to 3× nominal and back.
            # Because every frame now actually performs a different number of
            # effective sub-pixel rolls, the structure visibly coarsens and
            # refines smoothly with no flat plateau.
            scale_mul = 1.0 + 2.0 * frac
        else:
            scale_mul = 1.0

        # ── Rule #12: a wired image always wins over the procedural source ──
        wired = wired_source_rgb(params, int(W), int(H))
        if wired is not None:
            rgb = wired.astype(np.float32)
        else:
            rgb = _proc_source(source, seed, int(W), int(H))

        rgb = rgb.astype(np.float64)
        hh, ww = rgb.shape[0], rgb.shape[1]
        max_stride = max(1, min(hh, ww) // 2)

        # ── Rolling guidance iterations ──
        S = rgb
        effective_scales = []
        for k in range(n_steps):
            # Fractional stride (NOT rounded) so the rolling scale varies
            # continuously between frames -> smooth animation. The downsample
            # tolerates fractional s; the guided-filter window stays an integer.
            stride = float(sigma_start) * (growth ** k) * scale_mul
            stride = min(max(stride, 1.0), float(max_stride))
            effective_scales.append(stride)
            if stride < 1.5:
                # Already down to nothing meaningful; remaining rolls are no-ops.
                continue
            I_down = _area_downsample(S, stride)
            dh, dw = I_down.shape[0], I_down.shape[1]
            if dh < 3 or dw < 3:
                # Grid too small to filter meaningfully — stop rolling.
                break
            idx = _make_box(min(r_low, dh // 2, dw // 2), dh, dw)
            a_up = np.empty_like(I_down)
            b_up = np.empty_like(I_down)
            for c in range(3):
                ac, bc = _guided_coeffs(I_down[..., c], I_down[..., c],
                                        min(r_low, dh // 2, dw // 2), eps, idx)
                a_up[..., c] = ac
                b_up[..., c] = bc
            I_smooth = np.clip(a_up * I_down + b_up, 0.0, 1.0)
            S = _bilinear_upsample(I_smooth, hh, ww).astype(np.float64)

        base = np.clip(S, 0.0, 1.0).astype(np.float64)

        if mode == "smooth":
            result = base
        elif mode == "detail":
            result = base + amount * (rgb - base)
        else:  # flatten
            result = base - amount * (rgb - base)

        result = np.clip(result, 0.0, 1.0).astype(np.float32)
        base_lum = (0.299 * base[..., 0] + 0.587 * base[..., 1] + 0.114 * base[..., 2]).astype(np.float32)
        detail_energy = float(np.mean(np.abs(rgb - base)))

        # ── Scalars (Rule #4) + Field (Rule #5) ──
        write_scalars(out_dir, sigma_start=float(sigma_start), growth=growth,
                      n_steps=float(n_steps), edge_softness=float(r_low), eps=eps,
                      scale_mul=scale_mul,
                      effective_max_stride=float(effective_scales[-1]) if effective_scales else 0.0,
                      detail_energy=detail_energy, mean_base=float(base_lum.mean()))
        write_field(out_dir, base_lum)

        capture_frame("946", result)
        save(result, mn(946, f"Rolling Guidance {mode} s0={sigma_start} g={growth:.2f} n={n_steps}"), out_dir)
        return result
    except Exception as exc:
        fallback = np.full((int(H), int(W), 3), 128, dtype=np.uint8)
        save(fallback, mn(946, "Rolling Guidance Filter"), out_dir)
        print(f"[method_946] ERROR: {exc}")
        return fallback
