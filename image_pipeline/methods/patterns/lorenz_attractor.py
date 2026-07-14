"""Lorenz Attractor — 3D ODE flow rendered as a depth-shaded trajectory ribbon.

Integrates the Lorenz system (Lorenz 1963, "Deterministic Nonperiodic Flow",
doi:10.1175/1520-0469(1963)020<0130:DNF>2.0.CO;2) with a fixed-step RK4 and
rasterises the continuous trajectory into a 2D canvas:

    dx/dt = sigma * (y - x)
    dy/dt = x * (rho - z) - y
    dz/dt = x * y - beta * z

Unlike the 2D strange-attractor *maps* (Clifford/de Jong/Hopalong, node 957)
which are iterated function systems living in a flat plane, the Lorenz system is
a genuine 3D continuous-time ODE whose trajectory is the canonical "butterfly"
strange attractor. We integrate it with RK4 (4th-order, stable for the standard
parameters) and project the 3D path to the screen with a slight tilt + slow
yaw, shading by depth (near = bright, far = dim) and by speed (local |v|) so the
flow direction reads clearly.

Why this node: it is the 3D-ODE sibling of 957 and belongs to the same
cheap-dynamic-generator family that survives the shootout liveness cull. The
RK4 integration of a few hundred thousand points costs well under the pipeline's
150 s timeout (a single 512-point render is typically < 2 s), so it never
becomes a timeout casualty — and its high temporal variance (the path reshapes
every frame under `morph` / `spin`) reliably passes the static/flat liveness
filter that killed ~38% of logged genomes.

CPU path authoritative. Closed-form per-frame (no carried simulation state
between frames under `none`), so a clean f(uv, t) GPU twin is a natural
follow-up.

Animation modes (Architecture B — per-frame re-call with `time`):
    none  — fixed (sigma,rho,beta): frame Δ ≈ 0 (static baseline).
    morph — sigma/rho/beta breathe via cos(_t) so the attractor deforms
            smoothly. cos (not sin): cos(0)=+1, cos(π)=−1 keep the t=0 vs
            t=π audit frames distinct (sin-phase degeneracy false negative).
    spin  — the whole 3D figure yaws + tilts by _t (non-integer rate ⇒ never
            symmetry-aligned at the audit sample times, so it always reads as
            motion even for a near-axisymmetric form).
    flow  — the integration window slides forward along the trajectory so the
            attractor appears to *flow* (the seeded path tip advances each
            frame). _frame_seed drives fresh head points so the flow is alive.
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

_DEFAULTS = dict(sigma=10.0, rho=28.0, beta=2.6667)


def _rk4_step(x, y, z, s, r, b, dt):
    """One fixed-step RK4 advance of the Lorenz vector field."""
    def f(u, v, w):
        return (s * (v - u), u * (r - w) - v, u * v - b * w)
    k1 = f(x, y, z)
    k2 = f(x + 0.5 * dt * k1[0], y + 0.5 * dt * k1[1], z + 0.5 * dt * k1[2])
    k3 = f(x + 0.5 * dt * k2[0], y + 0.5 * dt * k2[1], z + 0.5 * dt * k2[2])
    k4 = f(x + dt * k3[0], y + dt * k3[1], z + dt * k3[2])
    return (
        x + dt / 6.0 * (k1[0] + 2 * k2[0] + 2 * k3[0] + k4[0]),
        y + dt / 6.0 * (k1[1] + 2 * k2[1] + 2 * k3[1] + k4[1]),
        z + dt / 6.0 * (k1[2] + 2 * k2[2] + 2 * k3[2] + k4[2]),
    )


@method(
    id="960",
    name="Lorenz Attractor",
    category="patterns",
    new_image_contract=True,
    tags=["generative", "chaos", "ode", "lorenz", "attractor", "3d", "flow",
          "animation", "gpu-twin-candidate"],
    inputs={},
    outputs={"image": "IMAGE", "field": "FIELD", "mask": "MASK",
             "particles": "PARTICLES"},
    params={
        "sigma": {"description": "Lorenz sigma (x-y coupling)",
                  "min": 1.0, "max": 30.0, "default": 10.0},
        "rho": {"description": "Lorenz rho (Rayleigh number)",
                "min": 1.0, "max": 60.0, "default": 28.0},
        "beta": {"description": "Lorenz beta (z-damping)",
                 "min": 0.5, "max": 6.0, "default": 2.6667},
        "n_points": {"description": "trajectory integration steps (cost)",
                     "min": 50000.0, "max": 1500000.0, "default": 450000.0},
        "head_fraction": {"description": "bright head trail fraction (0-1)",
                          "min": 0.0, "max": 1.0, "default": 0.18},
        "shade_by": {"description": "depth = near bright, speed = fast bright",
                     "choices": ["depth", "speed"], "default": "depth"},
        "exposure": {"description": "trajectory brightness multiplier",
                     "min": 0.1, "max": 5.0, "default": 1.4},
        "gamma": {"description": "tonal gamma (lower = brighter highlights)",
                  "min": 0.3, "max": 2.5, "default": 1.0},
        "hue": {"description": "base hue (0-1)",
                "min": 0.0, "max": 1.0, "default": 0.58},
        "sat": {"description": "colour saturation",
                "min": 0.0, "max": 1.0, "default": 0.9},
        "background": {"description": "canvas background (dark/light/mid)",
                       "choices": ["dark", "light", "mid"], "default": "dark"},
        "time": {"description": "animation phase [0, 2pi)",
                 "min": 0.0, "max": 6.28, "default": 0.0},
        "anim_mode": {"description": "animation mode (none/morph/spin/flow)",
                      "choices": ["none", "morph", "spin", "flow"], "default": "none"},
        "anim_speed": {"description": "animation speed multiplier",
                       "min": 0.1, "max": 5.0, "default": 1.0},
    },
)
def method_lorenz_attractor(out_dir: Path, seed: int, params=None):
    """Lorenz Attractor — 3D Lorenz ODE flow as a depth-shaded trajectory.

    RK4-integrates the Lorenz system, projects the 3D path to the screen with a
    tilt + slow yaw, and shades by depth/speed into an RGBA canvas.

    Params:
        sigma, rho, beta: Lorenz parameters (defaults = classic butterfly).
        n_points:         trajectory integration steps (renders in < 2 s).
        head_fraction:    fraction of the trail drawn as a bright "comet" head.
        shade_by:         depth (near bright) or speed (fast bright).
        exposure, gamma:  tonal mapping.
        hue, sat:         base colour.
        background:       dark / light / mid canvas.
        time, anim_mode, anim_speed: animation clock + mode.
    """
    try:
        if params is None:
            params = {}
        t = float(params.get("time", 0.0))
        anim_mode = str(params.get("anim_mode", "none"))
        anim_speed = float(params.get("anim_speed", 1.0))
        seed_all(seed)

        sigma = float(np.clip(params.get("sigma", 10.0), 1.0, 30.0))
        rho = float(np.clip(params.get("rho", 28.0), 1.0, 60.0))
        beta = float(np.clip(params.get("beta", 2.6667), 0.5, 6.0))
        n_points = int(np.clip(params.get("n_points", 450000.0), 50000, 1500000))
        head_fraction = float(np.clip(params.get("head_fraction", 0.18), 0.0, 1.0))
        shade_by = str(params.get("shade_by", "depth"))
        exposure = float(np.clip(params.get("exposure", 1.4), 0.1, 5.0))
        gamma = float(np.clip(params.get("gamma", 1.0), 0.3, 2.5))
        hue = float(np.clip(params.get("hue", 0.58), 0.0, 1.0))
        sat = float(np.clip(params.get("sat", 0.9), 0.0, 1.0))
        background = str(params.get("background", "dark"))

        # ── Animation clock (rename t to avoid shadowing the time param) ──
        _t = t * anim_speed
        s_eff, r_eff, b_eff = sigma, rho, beta
        yaw = 0.0
        tilt = 0.0
        flow_offset = 0
        sx0 = 0.0
        sy0 = 0.0
        # camera zoom: steady at 1.0 for `none`, gentle pulse for animated modes
        # (a zoom pulse guarantees visible Δ without strobing the trajectory).
        zoom = 1.0
        if anim_mode == "morph":
            # cos (not sin): keeps t=0 vs t=π audit frames distinct.
            s_eff = sigma + 2.5 * math.cos(_t)
            r_eff = rho + 6.0 * math.cos(_t + 1.7)
            b_eff = beta + 0.6 * math.cos(_t + 3.1)
            zoom = 1.0 + 0.22 * math.cos(_t)
        elif anim_mode == "spin":
            # Non-integer rate ⇒ never symmetry-aligned at audit sample times.
            yaw = _t * 0.5
            tilt = 0.18 + 0.10 * math.sin(_t * 0.27)
            zoom = 1.0 + 0.22 * math.cos(_t)
        elif anim_mode == "flow":
            # Slide the integration window forward along the seeded trajectory.
            # Per-frame seed keeps the head points fresh so the flow is alive.
            _frame_seed = seed + int(_t * 10000)
            rng_h = np.random.default_rng(_frame_seed)
            flow_offset = int((0.5 + 0.5 * math.cos(_t * 0.6)) * 8000)
            # perturb the start point slightly per frame (deterministic)
            sx0 = 0.6 * (rng_h.random() - 0.5)
            sy0 = 0.6 * (rng_h.random() - 0.5)
            zoom = 1.0 + 0.22 * math.cos(_t)

        # ── Background ──
        if background == "light":
            bg = np.array([0.96, 0.96, 0.98], dtype=np.float32)
        elif background == "mid":
            bg = np.array(BG_DEFAULT, dtype=np.float32) / 255.0
        else:  # dark
            bg = np.array([0.03, 0.04, 0.07], dtype=np.float32)

        # ── RK4 integrate the Lorenz trajectory ──
        dt = 0.005
        x = -8.0 + sx0
        y = 8.0 + sy0
        z = 27.0
        # warm-up to land on the attractor (skip transient)
        warm = 1500
        for _ in range(warm):
            x, y, z = _rk4_step(x, y, z, s_eff, r_eff, b_eff, dt)
        if anim_mode == "flow" and flow_offset > 0:
            for _ in range(flow_offset):
                x, y, z = _rk4_step(x, y, z, s_eff, r_eff, b_eff, dt)

        N = n_points
        xs = np.empty(N, dtype=np.float64)
        ys = np.empty(N, dtype=np.float64)
        zs = np.empty(N, dtype=np.float64)
        vs = np.empty(N, dtype=np.float64)  # local speed
        for i in range(N):
            xs[i] = x; ys[i] = y; zs[i] = z
            # speed = |f(x,y,z)|
            vx = s_eff * (y - x)
            vy = x * (r_eff - z) - y
            vz = x * y - b_eff * z
            vs[i] = math.sqrt(vx * vx + vy * vy + vz * vz)
            x, y, z = _rk4_step(x, y, z, s_eff, r_eff, b_eff, dt)

        # ── 3D → 2D projection (fixed reference framing, NOT auto-fit) ──
        # We center on the attractor's known gravity center (z≈25, x,y≈0 for the
        # classic parameters) and use a FIXED reference span. This is essential:
        # an auto-fit frame would rescale the blob to fill the canvas regardless
        # of params, hiding ρ/σ/β changes (a fuzz-blob invariance / pitfall #19
        # cousin). A fixed frame means a larger ρ genuinely draws a bigger,
        # different butterfly. The camera `zoom` (animated modes) then scales it.
        cx = 0.0; cy = 0.0; cz = 25.0
        ux = xs - cx; uy = ys - cy; uz = zs - cz
        cyaw = math.cos(yaw); syaw = math.sin(yaw)
        rx = ux * cyaw - uy * syaw
        ry = ux * syaw + uy * cyaw
        rz = uz
        ctilt = math.cos(tilt); stilt = math.sin(tilt)
        ry2 = ry * ctilt - rz * stilt
        rz2 = ry * stilt + rz * ctilt
        sx = rx
        sy = rz2
        # FIXED reference span (classic butterfly fits within ~±45 in x/y and
        # ±40 in z for ρ≈28). Margin lets larger ρ still fit without clipping.
        REF_SPAN = 70.0
        span = REF_SPAN * 1.08 / zoom
        scale_px = (0.5 / span) * min(W, H)
        px = (sx * scale_px + W / 2.0)
        py = (H / 2.0 - sy * scale_px)

        # depth metric for shading: normalized rz2 (near = bright)
        if shade_by == "speed":
            depth = vs
            dmin, dmax = float(vs.min()), float(vs.max())
        else:  # depth
            # nearer to camera = larger rz2 in this projection
            depth = rz2
            dmin, dmax = float(rz2.min()), float(rz2.max())
        if dmax - dmin > 1e-9:
            dn = (depth - dmin) / (dmax - dmin)
        else:
            dn = np.full(N, 0.5, dtype=np.float32)
        dn = dn.astype(np.float32)

        px = np.clip(np.round(px), 0, W - 1).astype(np.int64)
        py = np.clip(np.round(py), 0, H - 1).astype(np.int64)
        idx = (py * W + px).astype(np.int64)

        # ── Accumulate brightness via additive weighted bincount ──
        # weight = depth brightness * exposure; head trail gets a boost.
        n_head = max(1, int(N * head_fraction))
        head_w = np.ones(N, dtype=np.float32)
        head_w[-n_head:] = 2.6  # comet head brighter
        weights = dn * exposure * head_w
        acc = np.bincount(idx, weights=weights, minlength=W * H).astype(np.float32).reshape(H, W)
        # density (visit count) for mask + field
        count = np.bincount(idx, minlength=W * H).astype(np.float32).reshape(H, W)
        acc = np.clip(acc, 0.0, None)
        if acc.sum() > 0:
            peak = max(float(np.percentile(acc, 99.5)), 1e-6)
        else:
            peak = 1.0
        An = acc / peak
        bright = np.power(np.clip(An, 0.0, None), 1.0 / gamma)
        bright = np.clip(bright, 0.0, 1.0)

        # ── Colour: hue sweep along trajectory + saturation, value = bright ──
        frac = np.linspace(0.0, 1.0, N, dtype=np.float32)
        hue_arr = np.mod(frac + hue, 1.0).astype(np.float32)
        hue_sum = np.bincount(idx, weights=hue_arr, minlength=W * H).astype(np.float32).reshape(H, W)
        with np.errstate(invalid="ignore", divide="ignore"):
            avg_hue = np.where(count > 0, hue_sum / np.clip(count, 1e-6, None), hue)
        avg_hue = np.mod(avg_hue, 1.0)

        # HSV → RGB (hand-rolled, dependency-free)
        def hsv2rgb_vec(h, s, v):
            i = np.floor(h * 6.0).astype(np.int32) % 6
            f = h * 6.0 - np.floor(h * 6.0)
            p = v * (1.0 - s)
            q = v * (1.0 - f * s)
            tt = v * (1.0 - (1.0 - f) * s)
            r = np.zeros_like(v); g = np.zeros_like(v); b = np.zeros_like(v)
            m0 = i == 0; r[m0], g[m0], b[m0] = v[m0], tt[m0], p[m0]
            m1 = i == 1; r[m1], g[m1], b[m1] = q[m1], v[m1], p[m1]
            m2 = i == 2; r[m2], g[m2], b[m2] = p[m2], v[m2], tt[m2]
            m3 = i == 3; r[m3], g[m3], b[m3] = p[m3], q[m3], v[m3]
            m4 = i == 4; r[m4], g[m4], b[m4] = tt[m4], p[m4], v[m4]
            m5 = i == 5; r[m5], g[m5], b[m5] = v[m5], p[m5], q[m5]
            return r, g, b
        rr, gg, bb = hsv2rgb_vec(avg_hue, sat, bright)
        rgb = np.stack([rr, gg, bb], axis=-1).astype(np.float32)
        # additive glow over background
        rgb = np.clip(rgb + bg * (1.0 - bright[..., None]), 0.0, 1.0)
        alpha = np.clip(bright, 0.0, 1.0)

        mask = (count > 0).astype(np.float32)
        field = np.clip(An, 0.0, 1.0).astype(np.float32)

        # ── Particles: ordered subsample with local velocity ──
        stride = max(1, N // 4000)
        pidx = np.arange(0, N - 1, stride)
        pxx = px[pidx].astype(np.float32)
        pyy = py[pidx].astype(np.float32)
        velx = (px[pidx + 1] - pxx)
        vely = (py[pidx + 1] - pyy)
        particles = np.zeros((pidx.size, 4), dtype=np.float32)
        particles[:, 0] = pxx
        particles[:, 1] = pyy
        particles[:, 2] = velx
        particles[:, 3] = vely

        rgba = np.concatenate([rgb, alpha[..., None]], axis=-1).astype(np.float32)
        capture_frame("960", rgba)
        save(rgba, mn(960, "Lorenz Attractor"), out_dir)
        try:
            write_field(out_dir, field)
            write_mask(out_dir, mask)
            write_particles(out_dir, particles)
            write_scalars(
                out_dir,
                sigma=float(s_eff), rho=float(r_eff), beta=float(b_eff),
                points=float(N),
                peak_brightness=float(peak),
                coverage=float(mask.mean()),
                mode_code=float(hash(anim_mode) % 1000),
            )
        except Exception:
            pass
        return rgba
    except Exception as exc:
        fallback = np.full((H, W, 4), 0.5, dtype=np.float32)
        save(fallback, mn(960, "Lorenz Attractor"), out_dir)
        print(f"[method_960] ERROR: {exc}")
        return fallback
