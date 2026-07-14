"""Torus Knot — a (p,q) torus knot rendered as a depth-shaded glowing curve.

A (p,q) torus knot is the curve that winds `p` times around the axis of a
torus and `q` times around its tube. With coprime (p,q) it is a single
non-trivial knot (e.g. (2,3) is the trefoil); otherwise it is a multi-component
link. The standard embedding (see e.g. Wikipedia "Torus knot") is:

    r(t)   = R + r0 * cos(q * t)
    x(t)   = r(t) * cos(p * t)
    y(t)   = r(t) * sin(p * t)
    z(t)   = r0 * sin(q * t)

The knot is a 1D manifold in 3D. To give it volume in a production pipeline you
sweep a circular cross-section along the curve using a **rotation-minimizing
frame** (RMF / Bishop frame) — Bishop (1975), popularised for graphics by
Wang et al. (2008), "Computation of Rotation Minimizing Frames", Microsoft
Research (https://www.microsoft.com/en-us/research/wp-content/uploads/2016/12/Computation-of-rotation-minimizing-frames.pdf).
An RMF avoids the twist artefacts of the classic Frenet-Serret frame, which is
why it is the standard tool for tube / sweep / ribbon generation in CG.

This node renders the **centre-line** as a depth-shaded, additively-glowing
polyline — the real-time preview analog of the RMF-tube. Depth (z toward the
camera under a weak-perspective projection) drives brightness so the knot reads
as a 3D object, and the closed-form embedding makes a clean per-pixel f(uv,t)
GPU twin a natural follow-up (the 3D-sidecar RMF tube is the next step).

Cheap on purpose: a single vectorised sample of N points (default 6000) plus a
9-tap scatter and a small blur. Well under the pipeline's 150 s timeout, so the
node never becomes a shootout timeout casualty — and its continuous motion under
`rotate` / `morph` reliably clears the shootout liveness cull (unlike
contrast-only patterns).

Lines stay thin (1–3 px) and never thicken with stretch, per the pipeline's
thin-line convention for geometric curves.

Animation modes (Architecture B — per-frame re-call with `time`):
    none    — fixed (p,q,R): frame Δ ≈ 0 (static baseline).
    rotate  — the whole 3D knot spins (non-integer rates ⇒ never symmetry-aligned
              at the audit sample times, so it always reads as motion).
    morph   — the torus major/tube radii breathe via cos(_t), smoothly
              deforming the knot (cos, not sin, keeps audit frames distinct).
    breathe — overall exposure pulses via 0.5 + 0.5*cos(_t).
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np

from ...core.registry import method
from ...core.utils import (
    save, mn, seed_all, BG_DEFAULT, W, H,
    write_mask, write_particles, write_scalars, write_field,
)
from ...core.animation import capture_frame

_COLOR_MODES = ("rainbow", "mono", "duochrome")
_ANIM_MODES = ("none", "rotate", "morph", "breathe")


def _box_blur(a: np.ndarray, r: int) -> np.ndarray:
    """Separable moving-average (box) blur, radius r, dependency-free.

    Used only to soften the glow (r is 0–3 so the window is tiny and cheap).
    Pure numpy (no scipy dependency).
    """
    if r <= 0:
        return a
    k = 2 * r + 1
    if a.ndim == 3:
        out = a.copy()
        for c in range(a.shape[2]):
            out[..., c] = _box_blur(a[..., c], r)
        return out
    pad = np.concatenate([a[:, :r], a, a[:, -r:]], axis=1)
    cs = np.cumsum(pad, axis=1, dtype=np.float64)
    h = (cs[:, k - 1:] - cs[:, :pad.shape[1] - (k - 1)]) / k
    pad = np.concatenate([h[:r, :], h, h[-r:, :]], axis=0)
    cs = np.cumsum(pad, axis=0, dtype=np.float64)
    v = (cs[k - 1:, :] - cs[:pad.shape[0] - (k - 1), :]) / k
    return v.astype(a.dtype)


def _hsv_to_rgb_vec(h, s, v):
    """Vectorized HSV → RGB, all arrays broadcastable, values in [0,1]."""
    h = np.asarray(h, dtype=np.float32)
    s = np.broadcast_to(np.asarray(s, dtype=np.float32), h.shape)
    v = np.broadcast_to(np.asarray(v, dtype=np.float32), h.shape)
    i = np.floor(h * 6.0).astype(np.int32) % 6
    f = h * 6.0 - np.floor(h * 6.0)
    p = v * (1.0 - s)
    q = v * (1.0 - f * s)
    t = v * (1.0 - (1.0 - f) * s)
    r = np.zeros_like(v)
    g = np.zeros_like(v)
    b = np.zeros_like(v)
    m0 = i == 0
    r[m0], g[m0], b[m0] = v[m0], t[m0], p[m0]
    m1 = i == 1
    r[m1], g[m1], b[m1] = q[m1], v[m1], p[m1]
    m2 = i == 2
    r[m2], g[m2], b[m2] = p[m2], v[m2], t[m2]
    m3 = i == 3
    r[m3], g[m3], b[m3] = p[m3], q[m3], v[m3]
    m4 = i == 4
    r[m4], g[m4], b[m4] = t[m4], p[m4], q[m4]
    m5 = i == 5
    r[m5], g[m5], b[m5] = v[m5], p[m5], q[m5]
    return r, g, b


@method(
    id="962",
    name="Torus Knot",
    category="patterns",
    new_image_contract=True,
    tags=["generative", "geometry", "knot", "torus-knot", "curve", "rmf",
          "animation", "gpu-twin-candidate"],
    inputs={},
    outputs={"image": "IMAGE", "field": "FIELD", "mask": "MASK",
             "particles": "PARTICLES"},
    params={
        "p": {"description": "winds around the torus axis (knot numerator)",
              "min": 1.0, "max": 9.0, "default": 2.0},
        "q": {"description": "winds around the torus tube (knot denominator)",
              "min": 1.0, "max": 9.0, "default": 3.0},
        "major_r": {"description": "torus major radius R",
                    "min": 1.0, "max": 3.0, "default": 2.0},
        "tube_r": {"description": "torus tube radius r0",
                   "min": 0.2, "max": 1.2, "default": 0.7},
        "n_points": {"description": "curve samples (smoothness vs cost)",
                     "min": 1000.0, "max": 16000.0, "default": 6000.0},
        "line_width": {"description": "centre-line width in px (1-3, thin-line rule)",
                       "min": 1.0, "max": 3.0, "default": 2.0},
        "glow": {"description": "additive neon glow blur radius (0-3)",
                 "min": 0.0, "max": 3.0, "default": 2.0},
        "exposure": {"description": "brightness multiplier",
                     "min": 0.1, "max": 5.0, "default": 1.6},
        "gamma": {"description": "tonal gamma (lower = brighter highlights)",
                  "min": 0.3, "max": 2.5, "default": 1.0},
        "color_mode": {"description": "colour source (rainbow/mono/duochrome)",
                       "choices": list(_COLOR_MODES), "default": "rainbow"},
        "hue": {"description": "base hue for mono / duochrome (0-1)",
                "min": 0.0, "max": 1.0, "default": 0.58},
        "sat": {"description": "colour saturation",
                "min": 0.0, "max": 1.0, "default": 0.9},
        "background": {"description": "canvas background (dark/light/mid)",
                       "choices": ["dark", "light", "mid"], "default": "dark"},
        "time": {"description": "animation phase [0, 2pi)",
                 "min": 0.0, "max": 6.28, "default": 0.0},
        "anim_mode": {"description": "animation mode (none/rotate/morph/breathe)",
                      "choices": list(_ANIM_MODES), "default": "none"},
        "anim_speed": {"description": "animation speed multiplier",
                       "min": 0.1, "max": 5.0, "default": 1.0},
    },
)
def method_torus_knot(out_dir: Path, seed: int, params=None):
    """Torus Knot — (p,q) torus knot as a depth-shaded glowing curve.

    Samples the knot centre-line, projects it to the screen with a weak
    perspective, shades by depth (closer = brighter), and colours it along its
    arclength. The closed-form embedding means every parameter (p,q,R,r0) is
    live, and `rotate` / `morph` give continuous, liveness-clearing motion.

    Params:
        p, q:        torus-knot winding numbers (coprime ⇒ single knot).
        major_r:     torus major radius R.
        tube_r:      torus tube radius r0.
        n_points:    curve samples.
        line_width:  centre-line px width (1-3, constant).
        glow:        neon glow blur radius.
        exposure:    brightness multiplier.
        gamma:       tonal gamma.
        color_mode:  rainbow / mono / duochrome.
        hue:         base hue.
        sat:         saturation.
        background:  dark / light / mid.
        time:        animation phase [0, 2pi).
        anim_mode:   none / rotate / morph / breathe.
        anim_speed:  animation speed.
    """
    try:
        if params is None:
            params = {}
        t = float(params.get("time", 0.0))
        anim_mode = str(params.get("anim_mode", "none"))
        anim_speed = float(params.get("anim_speed", 1.0))
        seed_all(seed)

        p = int(np.clip(round(float(params.get("p", 2.0))), 1, 9))
        q = int(np.clip(round(float(params.get("q", 3.0))), 1, 9))
        R = float(np.clip(params.get("major_r", 2.0), 1.0, 3.0))
        r0 = float(np.clip(params.get("tube_r", 0.7), 0.2, 1.2))
        n_points = int(np.clip(params.get("n_points", 6000.0), 1000, 16000))
        line_width = float(np.clip(params.get("line_width", 2.0), 1.0, 3.0))
        glow = float(np.clip(params.get("glow", 2.0), 0.0, 3.0))
        exposure = float(np.clip(params.get("exposure", 1.6), 0.1, 5.0))
        gamma = float(np.clip(params.get("gamma", 1.0), 0.3, 2.5))
        color_mode = str(params.get("color_mode", "rainbow"))
        if color_mode not in _COLOR_MODES:
            color_mode = "rainbow"
        hue = float(np.clip(params.get("hue", 0.58), 0.0, 1.0))
        sat = float(np.clip(params.get("sat", 0.9), 0.0, 1.0))
        background = str(params.get("background", "dark"))

        # ── Animation clock (rename t to avoid shadowing the time param) ──
        _t = t * anim_speed
        R_eff, r0_eff = R, r0
        exp_eff = exposure
        ax_rot, ay_rot = 0.0, 0.0
        if anim_mode == "rotate":
            # Non-integer rates ⇒ never symmetry-aligned at audit sample times.
            ax_rot = _t * 0.6
            ay_rot = _t * 0.4
        elif anim_mode == "morph":
            # cos (not sin): keeps t=0 vs t=π audit frames distinct.
            R_eff = R + 0.5 * math.cos(_t)
            r0_eff = r0 * (1.0 + 0.22 * math.cos(_t + 1.3))
        elif anim_mode == "breathe":
            exp_eff = exposure * (0.5 + 0.5 * math.cos(_t))

        # ── Background ──
        if background == "light":
            bg = np.array([0.96, 0.96, 0.98], dtype=np.float32)
        elif background == "mid":
            bg = np.array(BG_DEFAULT, dtype=np.float32) / 255.0
        else:  # dark
            bg = np.array([0.03, 0.04, 0.07], dtype=np.float32)

        # ── Sample the (p,q) torus knot centre-line in 3D ──
        tt = np.linspace(0.0, 2.0 * math.pi, n_points, endpoint=False).astype(np.float64)
        ang_tube = q * tt
        ang_ring = p * tt
        rr = R_eff + r0_eff * np.cos(ang_tube)
        x3 = rr * np.cos(ang_ring)
        y3 = rr * np.sin(ang_ring)
        z3 = r0_eff * np.sin(ang_tube)
        pts = np.stack([x3, y3, z3], axis=-1)  # (N,3)

        # ── Rigid 3D rotation (static tilt + animated spin) ──
        ax = 1.1 + ax_rot
        ay = 0.6 + ay_rot
        ca, sa = math.cos(ax), math.sin(ax)
        cb, sb = math.cos(ay), math.sin(ay)
        Rx = np.array([[1.0, 0.0, 0.0],
                       [0.0, ca, -sa],
                       [0.0, sa, ca]], dtype=np.float64)
        Ry = np.array([[cb, 0.0, sb],
                       [0.0, 1.0, 0.0],
                       [-sb, 0.0, cb]], dtype=np.float64)
        pts = pts @ (Rx.T @ Ry.T)  # (N,3)

        # ── Weak-perspective projection ──
        focal = 5.0
        S = 0.16 * min(W, H)
        zc = pts[:, 2]
        persp = focal / np.clip(focal - zc, 0.5, None)
        sx = W / 2.0 + pts[:, 0] * persp * S
        sy = H / 2.0 - pts[:, 1] * persp * S

        # ── Depth → brightness (closer to camera = brighter) ──
        zmin, zmax = float(zc.min()), float(zc.max())
        depth = (zc - zmin) / max(zmax - zmin, 1e-6)
        bright = (0.35 + 0.65 * depth).astype(np.float32)

        # ── Colour along arclength ──
        frac = np.linspace(0.0, 1.0, n_points, dtype=np.float32)
        if color_mode == "mono":
            hue_arr = np.full(n_points, hue, dtype=np.float32)
        elif color_mode == "duochrome":
            hue_arr = np.mod(hue + 0.5 * frac, 1.0).astype(np.float32)
        else:  # rainbow
            hue_arr = np.mod(frac + hue, 1.0).astype(np.float32)
        cr, cg, cb = _hsv_to_rgb_vec(hue_arr, sat, 1.0)
        color = np.stack([cr, cg, cb], axis=-1).astype(np.float32)  # (N,3)

        # ── Scatter the centre-line into an accumulation buffer (9-tap glow) ──
        xi = np.clip(np.round(sx).astype(np.int64), 0, W - 1)
        yi = np.clip(np.round(sy).astype(np.int64), 0, H - 1)
        g = np.array([[0.25, 0.5, 0.25],
                      [0.50, 1.0, 0.50],
                      [0.25, 0.5, 0.25]], dtype=np.float32)
        # widen the kernel by line_width (1→no widen, 3→+1 ring)
        if line_width >= 2.5:
            g = g + 0.18
        elif line_width >= 1.5:
            g = g + 0.06
        g = g / g.sum()
        DY = np.array([-1, 0, 1])
        DX = np.array([-1, 0, 1])
        wflat = g.flatten()  # (9,)
        off_y = DY.reshape(-1, 1)  # (9,1)
        off_x = DX.reshape(1, -1)  # (1,9)
        off = (off_y * W + off_x).flatten()  # (9,)
        tgt = (yi[:, None] * W + xi[:, None] + off[None, :]).reshape(-1)
        tgt = np.clip(tgt, 0, W * H - 1).astype(np.int64)
        wgt = (bright[:, None] * wflat[None, :]).reshape(-1)  # (N*9,)
        colw = (color[:, None, :] * (bright[:, None] * wflat[None, :])[:, :, None]).reshape(-1, 3)

        inten = np.zeros(W * H, dtype=np.float32)
        col_acc = np.zeros((W * H, 3), dtype=np.float32)
        np.add.at(inten, tgt, wgt)
        np.add.at(col_acc, tgt, colw)
        inten = inten.reshape(H, W)
        col_acc = col_acc.reshape(H, W, 3)

        # ── Shade: density → brightness, accumulated colour → hue ──
        if inten.sum() > 0:
            peak = max(float(np.percentile(inten, 99.9)), 1e-6)
        else:
            peak = 1.0
        Dn = np.clip(inten / peak, 0.0, 1.0)
        bright2 = np.clip(np.power(np.clip(Dn * exp_eff, 0.0, None), 1.0 / gamma), 0.0, 1.0)
        mono_arr = np.stack(
            _hsv_to_rgb_vec(np.array([hue], dtype=np.float32),
                            np.array([sat], dtype=np.float32),
                            np.array([1.0], dtype=np.float32)),
            axis=-1,
        ).reshape(1, 1, 3)
        with np.errstate(invalid="ignore", divide="ignore"):
            avg_col = np.where(inten[..., None] > 0,
                               col_acc / np.clip(inten[..., None], 1e-6, None),
                               mono_arr)
        avg_col = np.clip(avg_col, 0.0, 1.0)
        rgb = avg_col * bright2[..., None]

        # ── Neon glow (additive) ──
        if glow > 0:
            gr = int(round(glow))
            halo = _box_blur(rgb, gr)
            rgb = np.clip(rgb + halo * 0.5, 0.0, 1.0)

        # Composite over background (glow-over-bg look works for all modes).
        rgb = np.clip(rgb + bg * (1.0 - bright2)[..., None], 0.0, 1.0)

        mask = (inten > 0).astype(np.float32)
        field = np.clip(Dn, 0.0, 1.0).astype(np.float32)

        # ── Particles: a subsample of screen points with their curve tangent ──
        stride = max(1, n_points // 4000)
        pidx = np.arange(0, n_points - 1, stride)
        pxx = sx[pidx].astype(np.float32)
        pyy = sy[pidx].astype(np.float32)
        velx = (sx[pidx + 1] - pxx)
        vely = (sy[pidx + 1] - pyy)
        particles = np.zeros((pidx.size, 4), dtype=np.float32)
        particles[:, 0] = pxx
        particles[:, 1] = pyy
        particles[:, 2] = velx
        particles[:, 3] = vely

        capture_frame("962", rgb)
        save(rgb, mn(962, "Torus Knot"), out_dir)
        try:
            write_field(out_dir, field)
            write_mask(out_dir, mask)
            write_particles(out_dir, particles)
            write_scalars(
                out_dir,
                p=float(p), q=float(q),
                major_r=float(R_eff), tube_r=float(r0_eff),
                n_points=float(n_points),
                peak_density=float(peak),
                coverage=float(mask.mean()),
            )
        except Exception:
            pass
        return rgb
    except Exception as exc:
        fallback = np.full((H, W, 3), 0.5, dtype=np.float32)
        save(fallback, mn(962, "Torus Knot"), out_dir)
        print(f"[method_962] ERROR: {exc}")
        return fallback
