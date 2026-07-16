"""Optical Flow (Horn-Schunck dense motion field).

Dense, per-pixel optical flow via the classic Horn & Schunck (1981)
variational formulation: a data term that enforces the brightness-
constancy constraint (Ix*u + Iy*v + It = 0) coupled with a global
smoothness term (the flow field is piecewise smooth). The Euler-Lagrange
fixpoint iteration solves for (u, v) in closed form each step:

    u <- u_bar - Ix*(Ix*u_bar + Iy*v_bar + It) / (alpha^2 + Ix^2 + Iy^2)
    v <- v_bar - Iy*(Ix*u_bar + Iy*v_bar + It) / (alpha^2 + Ix^2 + Iy^2)

where u_bar / v_bar are the locally-averaged flow (Laplacian smoothing)
and alpha is the smoothness weight (smaller alpha -> accepts more
fine-grained, less smooth flow).

Why this node exists in the pipeline:
  - It outputs a FIELD (per-pixel flow magnitude) and an IMAGE (the
    canonical Middlebury HSV flow visualization: hue = motion angle,
    value = motion magnitude). The magnitude field is a *structural
    liveness* signal — exactly the kind of motion the shootout's
    frame-mean `temporal_var_min` metric can miss (it culls contrast-only
    animation as "static"). A genome driving a parameter from this flow
    field carries genuine structural motion.
  - It is self-contained: it generates a moving textured scene (two frames
    offset in time) and recovers the motion, so it doubles as a liveness
    demonstrator and an optical-flow ground-truth check (wired content is
    moved by a known transform the algorithm should recover).

References
  - Horn, B.K.P. & Schunck, B.G., "Determining Optical Flow", AI 17, 1981.
  - Baker, Scharstein, Lewis, Roth, Black, Szeliski, "A Database and
    Evaluation Methodology for Optical Flow", IJCV 2011 (Middlebury; the
    HSV color-wheel visualization convention used here).
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
from scipy.ndimage import map_coordinates, rotate, uniform_filter

from ...core.registry import method
from ...core.utils import (
    save, mn, seed_all, W, H, load_input,
    write_scalars, write_field, norm,
)
from ...core.animation import capture_frame


# ── Motion models ────────────────────────────────────────────────────────
# Each warps a canonical grayscale texture T by a phase-dependent transform.
# Reference frame uses phase (p - delta), current frame uses phase p, so the
# recovered flow is the motion accumulated over the inter-frame gap.

def _warp(T: np.ndarray, kind: str, p: float, amp: float) -> np.ndarray:
    """Warp texture T by motion model `kind` at phase p (amplitude amp px)."""
    hh, ww = T.shape
    yy, xx = np.mgrid[0:hh, 0:ww].astype(np.float32)
    cy, cx = hh / 2.0, ww / 2.0
    if kind == "rotate":
        a = amp * math.radians(8.0) * math.sin(p)        # small rotation (rad)
        X = xx - cx
        Y = yy - cy
        sx = (X * math.cos(a) + Y * math.sin(a)) + cx
        sy = (-X * math.sin(a) + Y * math.cos(a)) + cy
    elif kind == "zoom":
        s = 1.0 + (amp / 200.0) * math.sin(p)            # scale about centre
        sx = (xx - cx) / s + cx
        sy = (yy - cy) / s + cy
    elif kind == "wave":
        # Travelling horizontal shear: an advancing sine wave of displacement.
        sx = xx - amp * np.sin(yy / max(1, hh) * 6.2831853 + p)
        sy = yy
    else:  # translate (default)
        sx = xx - amp * math.sin(p)
        sy = yy - amp * math.cos(p)
    out = map_coordinates(T, [sy, sx], order=1, mode="reflect")
    return np.clip(out, 0.0, 1.0).astype(np.float32)


# ── Texture generation (when nothing is wired in) ──────────────────────────

def _make_texture(kind: str, rng: np.random.Generator) -> np.ndarray:
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    if kind == "checkerboard":
        fx, fy = 0.06, 0.06
        c = (np.floor(xx * fx) + np.floor(yy * fy)) % 2
        return c.astype(np.float32)
    if kind == "dots":
        g = np.zeros((int(H), int(W)), np.float32)
        n = 26
        ys = np.linspace(int(H * 0.12), int(H * 0.88), n).astype(int)
        xs = np.linspace(int(W * 0.12), int(W * 0.88), n).astype(int)
        for iy in ys:
            for ix in xs:
                d = np.sqrt((xx - ix) ** 2 + (yy - iy) ** 2)
                g += np.exp(-(d ** 2) / (2 * 9.0 ** 2))
        return norm(g)
    if kind == "perlin":
        from scipy.ndimage import zoom
        small = rng.random((16, 16)).astype(np.float32)
        sm = small
        for _ in range(8):
            sm = uniform_filter(sm, size=3, mode="reflect")  # smooth blobby value noise
        z = zoom(sm, (int(H) / sm.shape[0], int(W) / sm.shape[1]), order=1)
        return norm(z[:int(H), :int(W)])
    if kind == "rings":
        r = np.sqrt((xx - W / 2) ** 2 + (yy - H / 2) ** 2)
        return (0.5 + 0.5 * np.sin(r * 0.08)).astype(np.float32)
    # gradient (fallback)
    r = norm(np.sqrt((xx - W / 2) ** 2 + (yy - H / 2) ** 2))
    return r.astype(np.float32)


def _hsv2rgb(hsv: np.ndarray) -> np.ndarray:
    """Vectorized HSV -> RGB (h,s,v in [0,1])."""
    h = hsv[..., 0]
    s = hsv[..., 1]
    v = hsv[..., 2]
    i = np.floor(h * 6.0)
    f = h * 6.0 - i
    p = v * (1.0 - s)
    q = v * (1.0 - f * s)
    t = v * (1.0 - (1.0 - f) * s)
    i = i.astype(int) % 6
    r = np.choose(i, [v, q, p, p, t, v])
    g = np.choose(i, [t, v, v, q, p, p])
    b = np.choose(i, [p, p, t, v, v, q])
    return np.stack([r, g, b], axis=-1).astype(np.float32)


def _hs_flow(I0: np.ndarray, I1: np.ndarray, alpha: float, iters: int):
    """Horn-Schunck dense flow. Returns (u, v) float32."""
    # Central-difference gradients (use current frame for spatial gradients).
    Ix = np.gradient(I1, axis=1)
    Iy = np.gradient(I1, axis=0)
    It = I1 - I0
    u = np.zeros_like(I1, dtype=np.float64)
    v = np.zeros_like(I1, dtype=np.float64)
    a2 = float(alpha) ** 2
    for _ in range(int(iters)):
        u_bar = uniform_filter(u, size=3, mode="reflect")
        v_bar = uniform_filter(v, size=3, mode="reflect")
        tmp = Ix * u_bar + Iy * v_bar + It
        denom = a2 + Ix * Ix + Iy * Iy
        denom = np.where(denom < 1e-6, 1e-6, denom)
        u = u_bar - Ix * tmp / denom
        v = v_bar - Iy * tmp / denom
    return u.astype(np.float32), v.astype(np.float32)


@method(
    id="977",
    name="Optical Flow (Horn-Schunck)",
    category="filters",
    tags=["optical-flow", "motion", "dense", "field", "liveness",
          "horn-schunck", "real-time", "analysis", "animation"],
    timeout=120,
    inputs={"image_in": "IMAGE"},
    outputs={"image": "IMAGE", "field": "FIELD"},
    params={
        "source": {
            "description": "texture source (input_image/checkerboard/dots/perlin/rings/gradient); a wired image always overrides",
            "default": "input_image",
            "choices": ["input_image", "checkerboard", "dots", "perlin", "rings", "gradient"],
        },
        "anim_mode": {
            "description": "motion model (none/translate/rotate/zoom/wave)",
            "choices": ["none", "translate", "rotate", "zoom", "wave"],
            "default": "translate",
        },
        "anim_speed": {
            "description": "animation speed multiplier",
            "min": 0.1, "max": 5.0, "default": 1.0,
        },
        "time": {
            "description": "animation phase [0, 2pi)",
            "min": 0.0, "max": 6.28, "default": 0.0,
        },
        "flow_amount": {
            "description": "motion amplitude in pixels (how far the scene moves)",
            "min": 0.0, "max": 40.0, "default": 12.0,
        },
        "delta": {
            "description": "inter-frame phase gap (larger = more motion per frame, stronger flow)",
            "min": 0.05, "max": 1.5, "default": 0.35,
        },
        "smoothness": {
            "description": "Horn-Schunck smoothness weight alpha (low = detailed/sharp flow, high = smooth)",
            "min": 1.0, "max": 80.0, "default": 20.0,
        },
        "iters": {
            "description": "fixpoint iterations",
            "min": 5, "max": 60, "default": 25,
        },
        "viz_scale": {
            "description": "flow-magnitude -> brightness scaling for the HSV visualization",
            "min": 0.2, "max": 4.0, "default": 1.0,
        },
    },
)
def method_optical_flow(out_dir: Path, seed: int, params=None):
    """Optical Flow (Horn-Schunck) — dense per-pixel motion field.

    Generates a moving textured scene (reference frame at phase t-delta,
    current frame at phase t), then recovers the dense (u, v) motion via the
    Horn-Schunck variational solver. Outputs:
      - image: Middlebury HSV flow visualization (hue=angle, value=magnitude)
      - field: per-pixel flow magnitude (the structural-liveness signal)

    A wired upstream IMAGE supplies the texture (Rule #12); otherwise a
    generated texture is used. In `none` mode the scene is static, so the
    recovered flow is ~0 (static baseline).

    Params: source, anim_mode, anim_speed, time, flow_amount, delta,
            smoothness, iters, viz_scale.
    """
    try:
        if params is None:
            params = {}
        anim_time = float(params.get("time", 0.0))
        anim_mode = str(params.get("anim_mode", "translate"))
        anim_speed = float(params.get("anim_speed", 1.0))
        seed_all(seed)
        rng = np.random.default_rng(seed)

        source = str(params.get("source", "input_image"))
        flow_amount = float(params.get("flow_amount", 12.0))
        flow_amount = max(0.0, min(40.0, flow_amount))
        delta = float(params.get("delta", 0.35))
        delta = max(0.05, min(1.5, delta))
        alpha = float(params.get("smoothness", 20.0))
        alpha = max(1.0, min(80.0, alpha))
        iters = int(params.get("iters", 25))
        iters = max(5, min(60, iters))
        viz_scale = float(params.get("viz_scale", 1.0))
        viz_scale = max(0.2, min(4.0, viz_scale))

        # ── Animation clock (rename t to avoid shadowing the time param) ──
        _t = anim_time * anim_speed

        # ── Build canonical texture T (grayscale, [0,1]) ──
        T = None
        wired_path = params.get("input_image", "")
        if wired_path:
            try:
                arr = load_input(wired_path, int(W), int(H))
                T = arr.mean(axis=-1).astype(np.float32)
            except (FileNotFoundError, OSError):
                T = None
        if T is None:
            gen = "checkerboard" if source == "input_image" else source
            T = _make_texture(gen, rng)

        # ── Two frames; motion disabled entirely in `none` mode ──
        if anim_mode == "none":
            I0 = T.copy()
            I1 = T.copy()
        else:
            I0 = _warp(T, anim_mode, _t - delta, flow_amount)
            I1 = _warp(T, anim_mode, _t, flow_amount)

        # ── Horn-Schunck dense optical flow ──
        u, v = _hs_flow(I0, I1, alpha, iters)
        mag = np.sqrt(u * u + v * v).astype(np.float32)

        # ── IMAGE: Middlebury HSV flow visualization ──
        ang = np.arctan2(v, u)                       # [-pi, pi]
        hue = (ang + math.pi) / (2.0 * math.pi)      # [0, 1]
        with np.errstate(divide="ignore", invalid="ignore"):
            val = np.clip(mag * viz_scale / (mag.max() + 1e-6), 0.0, 1.0)
        hsv = np.stack([hue, np.ones_like(hue), val], axis=-1)
        viz = _hsv2rgb(hsv)

        # ── FIELD: normalized flow magnitude [0,1] ──
        field = (mag / (mag.max() + 1e-6)).astype(np.float32)

        capture_frame("977", viz)
        save(viz, mn(977, f"Optical Flow t={_t:.2f}"), out_dir)
        try:
            write_scalars(
                out_dir,
                flow_amount=float(flow_amount),
                delta=float(delta),
                smoothness=float(alpha),
                iters=float(iters),
                mean_magnitude=float(mag.mean()),
                max_magnitude=float(mag.max()),
            )
            write_field(out_dir, field)
        except Exception:
            pass
        return viz
    except Exception as exc:
        fallback = np.full((int(H), int(W), 3), 0.0, dtype=np.float32)
        save(fallback, mn(977, "Optical Flow"), out_dir)
        print(f"[method_977] ERROR: {exc}")
        return fallback
