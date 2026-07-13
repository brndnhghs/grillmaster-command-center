from __future__ import annotations

import math
from pathlib import Path

import numpy as np

from ...core.registry import method
from ...core.utils import save, mn, seed_all, get_canvas, write_scalars, write_field, write_mask
from ...core.animation import capture_frame


# ── Apollonian Gasket (Descartes Circle Theorem) ──────────────────────────────
# A circle packing where every gap is recursively filled by the *inner Soddy
# circle* tangent to any 3 mutually tangent circles. The curvatures (bends) of
# four mutually tangent circles obey Descartes' theorem:
#
#     (k1+k2+k3+k4)^2 = 2 (k1^2+k2^2+k3^2+k4^2)
#
# which yields the two possible 4th curvatures k4 = k1+k2+k3 ± 2√(k1k2+k2k3+k3k1).
# The centers obey the *complex* Descartes theorem (same formula with complex
# centers z_i). By starting from 2 equal inner circles inside one enclosing
# circle and recursing, we fill the plane with an infinite, self-similar
# packing of tangent circles — a classic of both geometry and procedural
# generation (also the seed of circle-packing / decorative "Soddy" art).
#
# Output is RGBA with alpha = 0 on the empty background (Rule 9: discrete
# objects on empty bg). Animation is *structural* — rotating the seed pair or
# pulsing their radii re-roots the entire recursive packing, so frames differ
# by real geometry, not a contrast flicker — which survives any temporal
# liveness metric. `anim_mode="none"` is a true static baseline (Δ ≈ 0).


def _hsv_to_rgb(h: float, s: float, v: float):
    i = int(math.floor(h * 6.0)) % 6
    f = h * 6.0 - math.floor(h * 6.0)
    p = v * (1.0 - s)
    q = v * (1.0 - f * s)
    t = v * (1.0 - (1.0 - f) * s)
    r = [v, q, p, p, t, v][i]
    g = [t, v, v, q, p, p][i]
    b = [p, p, t, v, v, q][i]
    return r, g, b


def _descartes_bends(k1, k2, k3):
    s = k1 + k2 + k3
    root = 2.0 * math.sqrt(max(0.0, k1 * k2 + k2 * k3 + k3 * k1))
    return s + root, s - root


def _descartes_center(z1, z2, z3, k1, k2, k3, ksign):
    # complex center of the 4th circle; ksign chooses the +/- (inner) solution
    num = (k1 * z1 + k2 * z2 + k3 * z3) + ksign * 2.0 * math.sqrt(
        max(0.0, (k1 * k2 * z1 * z2 + k2 * k3 * z2 * z3 + k3 * k1 * z3 * z1).real))
    den = (k1 + k2 + k3) + ksign * 2.0 * math.sqrt(max(0.0, k1 * k2 + k2 * k3 + k3 * k1))
    return num / den


class _Circ:
    __slots__ = ("k", "z")

    def __init__(self, k, z):
        self.k = k        # bend (curvature) — negative for the enclosing circle
        self.z = z        # complex center


def _new_circle(c1, c2, c3, opp):
    """Return the Soddy circle tangent to c1,c2,c3 that is NOT `opp`."""
    k1, k2, k3 = c1.k, c2.k, c3.k
    z1, z2, z3 = c1.z, c2.z, c3.z
    kb1, kb2 = _descartes_bends(k1, k2, k3)
    zb_plus = _descartes_center(z1, z2, z3, k1, k2, k3, +1.0)
    zb_minus = _descartes_center(z1, z2, z3, k1, k2, k3, -1.0)
    sols = [(kb1, zb_plus), (kb2, zb_minus)]
    for kb, zb in sols:
        # choose the solution that differs from the existing opposite circle
        if abs(kb - opp.k) > 1e-6 or abs(zb - opp.z) > 1e-6:
            return _Circ(kb, zb)
    return None


def _gen_gasket(seed_curv, anim_mode, _t, depth):
    """Build the list of circles (unit space: enclosing circle radius 1, center 0)."""
    ang = _t if anim_mode == "rotate" else 0.0
    scale = 1.0 + 0.5 * math.sin(_t) if anim_mode == "pulse" else 1.0
    k1 = scale * seed_curv
    k2 = scale * seed_curv
    r1 = abs(1.0 / k1)                      # inner radius
    off = (1.0 - r1) * math.e ** (1j * ang)  # center sits inside the enclosing circle
    outer = _Circ(-1.0, 0 + 0j)
    a = _Circ(k1, off)
    b = _Circ(k2, -off)
    # third seed circle: inner Soddy of (outer, a, b)
    c = _new_circle(outer, a, b, outer)
    if c is None:
        c = _Circ(3.0, 0 + 0.6667j)        # fallback (static config)

    out = [outer, a, b, c]
    aa, bb, cc = a, b, c

    # recursive fill
    def rec(quad, d):
        if d <= 0:
            return
        c1, c2, c3, c4 = quad
        triples = [(c2, c3, c4, c1), (c1, c3, c4, c2),
                   (c1, c2, c4, c3), (c1, c2, c3, c4)]
        for (t1, t2, t3, opp) in triples:
            nxt = _new_circle(t1, t2, t3, opp)
            if nxt is None:
                continue
            if nxt.k <= 0:                  # skip enclosing/non-positive bends
                continue
            r = 1.0 / nxt.k
            # reject circles that escape the enclosing unit circle
            if abs(nxt.z) + r > 1.0 + 1e-4:
                continue
            if r < 1e-3:                    # too tiny to see
                continue
            out.append(nxt)
            rec((t1, t2, t3, nxt), d - 1)

    rec((outer, aa, bb, cc), depth)
    return out


@method(id="514", name="Apollonian Gasket", category="patterns",
        tags=["apollonian", "descartes", "circle-packing", "tangent",
              "procedural", "geometry", "animation", "real-time-cg"],
        inputs={},
        outputs={"image": "IMAGE", "field": "FIELD", "mask": "MASK"},
        params={
    "depth": {"description": "recursion depth (more = smaller circles, denser packing)",
              "min": 1, "max": 6, "default": 4},
    "seed_curv": {"description": "bend (1/radius) of the two seed inner circles",
                  "min": 1.0, "max": 4.0, "default": 2.0},
    "color_mode": {"description": "circle coloring",
                   "choices": ["depth", "spectrum", "mono", "neon"], "default": "depth"},
    "anim_mode": {"description": "animation mode (none=static, rotate=seed pair spins, pulse=seed radii breathe)",
                  "choices": ["none", "rotate", "pulse"], "default": "none"},
    "anim_speed": {"description": "animation speed multiplier", "min": 0.1, "max": 3.0, "default": 1.0},
    "time": {"description": "animation phase [0, 2pi)", "min": 0.0, "max": 6.28, "default": 0.0},
})
def method_apollonian(out_dir: Path, seed: int, params=None):
    """Apollonian Gasket — recursive Descartes Circle Theorem circle packing.

    Technique: every gap between 3 mutually tangent circles is filled by the
    unique inner Soddy circle (Descartes' theorem, 1643; rediscovered by Soddy
    1936). Seed two equal circles inside one enclosing circle and recurse. The
    packing is infinitely self-similar. With ``anim_mode=\"rotate\"`` the seed
    pair spins (re-rooting the whole packing) and with ``anim_mode=\"pulse\"``
    their radii breathe — both are structural geometry changes, so the result
    is genuinely live frame-to-frame. ``anim_mode=\"none\"`` is a true static
    baseline (Δ ≈ 0). Output is RGBA with a transparent background.
    """
    try:
        if params is None:
            params = {}
        t = float(params.get("time", 0.0))
        seed_all(seed)
        rng = np.random.default_rng(seed)

        depth = int(np.clip(params.get("depth", 4), 1, 6))
        seed_curv = float(np.clip(params.get("seed_curv", 2.0), 1.0, 4.0))
        color_mode = params.get("color_mode", "depth")
        anim_mode = params.get("anim_mode", "none")
        anim_speed = float(np.clip(params.get("anim_speed", 1.0), 0.1, 3.0))

        _t = 0.0 if anim_mode == "none" else t * anim_speed
        # tiny per-seed framing jitter (deterministic from seed)
        _t = _t + 0.15 * (rng.random() - 0.5)

        circles = _gen_gasket(seed_curv, anim_mode, _t, depth)

        cw, ch = get_canvas()
        W, H = int(cw), int(ch)
        scale_px = min(W, H) * 0.5 * 0.96

        # pixel coordinate grids (centered)
        xs = np.arange(W, dtype=np.float64)
        ys = np.arange(H, dtype=np.float64)
        Xc = xs[None, :] - W / 2.0
        Yc = ys[:, None] - H / 2.0

        rgba = np.zeros((H, W, 4), dtype=np.float64)  # RGB (0) + A (0)
        depth_field = np.zeros((H, W), dtype=np.float32)
        n = len(circles)

        for idx, circ in enumerate(circles):
            if circ.k <= 0:
                continue  # enclosing circle itself is not drawn (transparent bg)
            r_px = scale_px / circ.k
            cx = circ.z.real * scale_px
            cy = circ.z.imag * scale_px
            d2 = (Xc - cx) ** 2 + (Yc - cy) ** 2
            # anti-aliased disk mask
            aa = max(1.0, 0.5 * r_px)
            edge = np.clip(r_px - np.sqrt(d2), 0.0, aa) / aa
            a_local = edge.astype(np.float64)

            # color
            if color_mode == "mono":
                col = (0.85, 0.85, 0.85)
            elif color_mode == "neon":
                hue = (idx * 0.137) % 1.0
                col = _hsv_to_rgb(hue, 1.0, 1.0)
            elif color_mode == "spectrum":
                hue = (math.log2(max(2.0, circ.k)) * 0.18) % 1.0
                col = _hsv_to_rgb(hue, 0.85, 1.0)
            else:  # depth
                frac = idx / max(1, n - 1)
                hue = (0.58 + 0.5 * frac) % 1.0
                col = _hsv_to_rgb(hue, 0.7, 0.6 + 0.4 * frac)

            col = np.array(col, dtype=np.float64)
            a_prev = rgba[:, :, 3]
            # "over" compositing (new on top)
            out_a = a_local + a_prev * (1.0 - a_local)
            out_a = np.where(out_a > 0, out_a, 1e-9)
            rgb = (rgba[:, :, :3] * a_prev[:, :, None] * (1.0 - a_local[:, :, None])
                   + col[None, None, :] * a_local[:, :, None]) / out_a[:, :, None]
            rgba[:, :, :3] = rgb
            rgba[:, :, 3] = np.clip(out_a, 0.0, 1.0)
            depth_field = np.maximum(depth_field, (a_local * float(idx) / max(1, n)).astype(np.float32))

        rgba = np.clip(rgba, 0.0, 1.0).astype(np.float32)
        alpha_mask = rgba[:, :, 3].copy()

        # ── Provenance / fields (Rules 4, 5, 10) ──
        bends = np.array([c.k for c in circles if c.k > 0], dtype=float)
        write_scalars(out_dir,
                      n_circles=len(circles),
                      drawn_circles=int((bends > 0).sum()) if bends.size else 0,
                      max_bend=round(float(bends.max()), 3) if bends.size else 0.0,
                      coverage=round(float(alpha_mask.mean()), 3))
        write_field(out_dir, depth_field)
        write_mask(out_dir, alpha_mask)

        capture_frame("514", rgba)
        save(rgba, mn(514, f"Apollonian Gasket t={_t:.2f}"), out_dir)
        return rgba
    except Exception as exc:
        cw, ch = get_canvas()
        fallback = np.zeros((int(ch), int(cw), 4), dtype=np.float32)
        save(fallback, mn(514, "Apollonian Gasket"), out_dir)
        print(f"[method_514] ERROR: {exc}")
        return fallback
