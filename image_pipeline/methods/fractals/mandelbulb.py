from __future__ import annotations

import math
from pathlib import Path

import numpy as np
from PIL import Image

from ...core.registry import method
from ...core.utils import (
    save, mn, seed_all, write_field, write_mask, write_scalars, W, H, PALETTES,
    wired_source_lum, wired_source_lum, wired_source_lum,
)
from ...core.animation import capture_frame
from image_pipeline.core.spatial import sparam


# ── Mandelbulb — 3D fractal raymarched in NumPy ──
# Daniel White & Paul Nylander (2009); popularized by Inigo Quilez's
# Shadertoy "Mandelbulb" (distance-estimator raymarching of the power-n
# spherical-angle fold). The map iterates the spherical-coordinate power:
#
#     z -> z^n + c,   z_{k+1} = (r^n) * ( sin(n*theta)cos(n*phi),
#                                       sin(n*theta)sin(n*phi),
#                                       cos(n*theta) ) + c
#
# with r=|z|, theta=acos(z.z/r), phi=atan2(z.y,z.x). The classic bulb is n=8.
# We march a pinhole camera through the analytic distance estimator
# d = 0.5*log(r)*r / dr  (dr = accumulated derivative), so the surface is the
# n=8 "bulb" — a genuinely 3D fractal (distinct from the 2D SDF primitive
# scene at node 412 and from the 2D Julia/Mandelbrot shells).
#
# Each frame is a pure function of (uv, time) -> Architecture B (re-called per
# frame by the orchestrator). Animated modes rotate the camera ('spin') or
# morph the exponent ('morph'); 'none' is time-static (Δ≈0). CPU path is the
# authoritative 2D export; the scene is rendered at a bounded internal
# resolution then upscaled, so per-frame cost stays safe for animation.

def _cos_pal(t: np.ndarray, shift: float):
    """Inigo Quilez cosine gradient palette (cheap, matplotlib-free)."""
    t = np.clip(t, 0.0, 1.0)
    r = 0.5 + 0.5 * np.cos(6.28318 * (t + shift + 0.00))
    g = 0.5 + 0.5 * np.cos(6.28318 * (t + shift + 0.33))
    b = 0.5 + 0.5 * np.cos(6.28318 * (t + shift + 0.67))
    return r, g, b


def _mandelbulb_de(pts: np.ndarray, power: float, iters: int):
    """Vectorized Mandelbulb distance estimator.

    pts: (N,3) float64 in world space. Returns (dist, trap, r_final) each (N,).
    dist is the analytic DE; trap is the min orbit radius (colouring cue);
    r_final is |z| at the final iteration. Uses the standard bailout/escape
    break (Quilez): once the orbit's radius exceeds `bailout` we stop updating
    that point, so `r_final` stays a sane value (~bailout) and the analytic DE
    0.5*log(r)*r/dr never overflows on escaped points.
    """
    bailout = 2.0
    z = pts.copy()
    dr = np.ones(pts.shape[0], dtype=np.float64)
    trap = np.full(pts.shape[0], 1e3, dtype=np.float64)
    escaped = np.zeros(pts.shape[0], dtype=bool)
    for _ in range(iters):
        x = z[:, 0]
        y = z[:, 1]
        w = z[:, 2]
        r = np.sqrt(x * x + y * y + w * w)
        trap = np.minimum(trap, r)
        just_escaped = (r > bailout) & (~escaped)
        escaped = escaped | just_escaped
        rs = np.maximum(r, 1e-12)
        theta = np.arccos(np.clip(w / rs, -1.0, 1.0))
        phi = np.arctan2(y, x)
        zr = r ** power
        dr_new = (rs ** (power - 1.0)) * power * dr + 1.0
        dr = np.where(escaped, dr, np.minimum(dr_new, 1e30))
        theta *= power
        phi *= power
        st = np.sin(theta)
        nx = zr * st * np.cos(phi)
        ny = zr * st * np.sin(phi)
        nw = zr * np.cos(theta)
        newz = np.stack([nx, ny, nw], axis=-1) + pts
        z = np.where(escaped[:, None], z, newz)
    r_final = np.sqrt((z * z).sum(axis=-1))
    dist = 0.5 * np.log(np.maximum(r_final, 1e-9)) * r_final / np.maximum(dr, 1e-9)
    dist = np.maximum(dist, 0.0)
    return dist, trap, r_final


def _camera(az: float, el: float, cam_dist: float):
    """Return (ro, fwd, rgt, upv) for a camera looking at the origin."""
    ro = np.array([
        cam_dist * math.cos(el) * math.sin(az),
        cam_dist * math.sin(el),
        cam_dist * math.cos(el) * math.cos(az),
    ], dtype=np.float64)
    fwd = -ro
    fwd /= np.linalg.norm(fwd)
    rgt = np.cross(np.array([0.0, 1.0, 0.0]), fwd)
    rgt /= np.linalg.norm(rgt)
    upv = np.cross(fwd, rgt)
    return ro, fwd, rgt, upv


def _render(RW: int, RH: int, az: float, el: float, cam_dist: float,
            power: float, iters: int, steps: int, warp=None):
    """Raymarch the bulb at (RW,RH). Returns (rgb (RH,RW,3), hit (RH,RW),
    depth (RH,RW), trap (RH,RW))."""
    ro, fwd, rgt, upv = _camera(az, el, cam_dist)
    xs = (np.arange(RW) + 0.5) / RW * 2.0 - 1.0
    ys = (np.arange(RH) + 0.5) / RH * 2.0 - 1.0
    ys = ys[::-1]
    X, Y = np.meshgrid(xs, ys)
    X = X * (RW / RH)
    rd = (fwd[None, None, :] * 1.6 + X[..., None] * rgt[None, None, :]
          + Y[..., None] * upv[None, None, :])
    rd /= np.linalg.norm(rd, axis=-1, keepdims=True)
    rd_f = rd.reshape(-1, 3)

    t = np.zeros(rd_f.shape[0], dtype=np.float64)
    hit = np.zeros(rd_f.shape[0], dtype=bool)
    trap_min = np.full(rd_f.shape[0], 1e3, dtype=np.float64)
    eps = 4e-4
    for _ in range(steps):
        p = ro[None, :] + rd_f * t[:, None]
        if warp is not None:
            p = p + warp.reshape(-1)[:, None]
        d, trap, r_final = _mandelbulb_de(p, power, iters)
        trap_min = np.minimum(trap_min, trap)
        new_hit = d < eps
        hit = hit | new_hit
        t = t + np.where(hit, 0.0, np.minimum(d, 0.25))
        if hit.all():
            break

    trap_min = trap_min.reshape(RH, RW)
    hit = hit.reshape(RH, RW)

    # Normals via central differences of the DE (hit pixels only).
    n = np.zeros((RH, RW, 3), dtype=np.float64)
    if hit.any():
        p_f = (ro[None, None, :] + rd * t.reshape(RH, RW)[..., None]).reshape(-1, 3)
        h = 1.5e-3
        for axis in range(3):
            e = np.zeros(3)
            e[axis] = h
            dpx, _, _ = _mandelbulb_de(p_f + e, power, iters)
            dmx, _, _ = _mandelbulb_de(p_f - e, power, iters)
            n[..., axis] = (dpx - dmx).reshape(RH, RW)
        nn = np.linalg.norm(n, axis=-1, keepdims=True)
        n = np.where(hit[..., None], n / (nn + 1e-9), 0.0)

    # Shading.
    light_dir = np.array([0.7, 0.65, -0.5], dtype=np.float64)
    light_dir /= np.linalg.norm(light_dir)
    dif = np.clip((n * light_dir).sum(axis=-1), 0.0, 1.0) * hit
    rim = np.clip(1.0 - np.clip((n * (-rd)).sum(axis=-1), 0.0, 1.0), 0.0, 1.0) ** 2.0
    amb = 0.22 + 0.18 * hit
    shade = amb + 0.95 * dif + 0.25 * rim * hit

    # Background gradient (subtle, by ray elevation).
    bg_t = np.clip(0.5 + 0.5 * rd[..., 1], 0.0, 1.0)
    bg = np.stack([0.04 + 0.06 * bg_t, 0.05 + 0.07 * bg_t, 0.09 + 0.12 * bg_t], axis=-1)

    # Base colour from the orbit trap, modulated by shade.
    base_t = np.clip(trap_min / 1.25, 0.0, 1.0)
    col = np.where(hit[..., None], base_t[..., None], 0.0)
    return shade, hit, col, bg, base_t


@method(id='470', name='Mandelbulb 3D Fractal', category='fractals', new_image_contract=True, tags=['mandelbulb', '3d-fractal', 'raymarch', 'distance-estimator', 'animation', 'quilez'], inputs={'image_in': 'IMAGE'}, outputs={'image': 'IMAGE', 'mask': 'MASK', 'field': 'FIELD'}, params={'source': {'description': "domain-warp the per-ray sample point from the wired image's luminance", 'choices': ['none', 'input_image'], 'default': 'none'}, 'warp_strength': {"spatial": True, 'description': 'domain-warp strength for the 3D sample points', 'min': 0.0, 'max': 2.0, 'default': 0.6}, 'power': {"spatial": True, 'description': 'bulb exponent (8 = classic; other powers morph the surface)', 'min': 2.0, 'max': 12.0, 'default': 8.0}, 'iterations': {'description': 'distance-estimator fold iterations (higher = sharper surface)', 'min': 2, 'max': 16, 'default': 6}, 'steps': {'description': 'raymarch steps (higher = fewer surface holes)', 'min': 16, 'max': 128, 'default': 56}, 'cam_dist': {'description': 'camera distance from the bulb', 'min': 1.6, 'max': 4.0, 'default': 2.7}, 'elevation': {"spatial": True, 'description': 'camera elevation angle (radians)', 'min': -1.3, 'max': 1.3, 'default': 0.35}, 'detail': {'description': 'internal render scale (caps per-frame cost)', 'min': 0.25, 'max': 1.0, 'default': 0.55}, 'colormode': {'description': 'colour map (rainbow/inferno/fire/ice/grayscale)', 'default': 'inferno'}, 'palette_shift': {'description': 'cosine palette hue offset', 'min': 0.0, 'max': 1.0, 'default': 0.5}, 'time': {'description': 'animation phase [0, 2pi)', 'min': 0.0, 'max': 6.28, 'default': 0.0}, 'anim_mode': {'description': 'animation mode (none/spin/morph)', 'choices': ['none', 'spin', 'morph'], 'default': 'none'}, 'anim_speed': {'description': 'animation speed multiplier', 'min': 0.1, 'max': 3.0, 'default': 1.0}})
def method_mandelbulb(out_dir: Path, seed: int, params=None):
    """Mandelbulb — 3D fractal raymarched with a distance estimator.

    Technique: Daniel White & Paul Nylander, "The Mandelbulb" (2009);
    distance-estimator raymarching popularized by Inigo Quilez
    (shadertoy.com/view/MfgXWu, "Mandelbulb"). The spherical power map

        z -> z^n + c,  z_{k+1} = r^n ( sin(nθ)cos(nφ), sin(nθ)sin(nφ),
                                      cos(nθ) ) + c

    (with r=|z|, θ=acos(z.z/r), φ=atan2(z.y,z.x)) produces a true 3D fractal
    whose n=8 "bulb" has the familiar bulbous, spiralled surface. We march a
    pinhole camera through the analytic DE  d = 0.5·log(r)·r / dr  and shade
    hits with a single directional light + rim term; colour comes from the
    orbit-trap radius.

    The scene is rendered at a bounded internal resolution (scaled by `detail`)
    then upscaled to the canvas, which keeps per-frame cost safe for animation.

    Params:
        power:        bulb exponent (2-12)
        iterations:   DE fold iterations
        steps:        raymarch steps
        cam_dist:     camera distance
        elevation:    camera elevation (radians)
        detail:       internal render scale
        colormode:    rainbow / inferno / fire / ice / grayscale
        palette_shift: cosine palette hue offset
        time:         animation phase
        anim_mode:    none (static) / spin (orbit camera) / morph (exponent)
        anim_speed:   animation speed
    """
    try:
        if params is None:
            params = {}
        t = float(params.get("time", 0.0))
        anim_mode = str(params.get("anim_mode", "none"))
        anim_speed = float(params.get("anim_speed", 1.0))
        seed_all(seed)

        power = sparam(params, "power", 8.0)
        iterations = int(max(2, min(16, float(params.get("iterations", 6)))))
        steps = int(max(16, min(128, float(params.get("steps", 56)))))
        cam_dist = float(params.get("cam_dist", 2.7))
        elevation0 = sparam(params, "elevation", 0.35)
        detail = max(0.25, min(1.0, float(params.get("detail", 0.55))))
        colormode = str(params.get("colormode", "inferno"))
        palette_shift = max(0.0, min(1.0, float(params.get("palette_shift", 0.5))))

        # ── Seed -> deterministic but non-degenerate view (avoids dead param) ──
        rng = np.random.default_rng(seed)
        el_jitter = (rng.random() - 0.5) * 0.5
        az0 = rng.random() * 6.2831853

        # ── Animation clock (rename to avoid shadowing the time param) ──
        _t = t * anim_speed if anim_mode != "none" else 0.0
        az = az0
        el = elevation0 + el_jitter
        p_eff = power
        if anim_mode == "spin":
            az = az0 + _t
            el = elevation0 + el_jitter + 0.25 * math.sin(_t * 0.5)
        elif anim_mode == "morph":
            p_eff = power * (1.0 + 0.18 * math.sin(_t))   # breathe the exponent

        # ── Bounded internal resolution ──
        aspect = max(0.5, W / max(1, H))
        anim_scale = 0.7 if anim_mode != "none" else 1.0
        RH = max(64, int(280.0 * detail * anim_scale))
        RW = max(64, int(RH * aspect))
        # hard pixel cap regardless of canvas size
        cap = 90000
        if RW * RH > cap:
            s = math.sqrt(cap / (RW * RH))
            RW, RH = max(64, int(RW * s)), max(64, int(RH * s))

        # ── Domain-warp from wired luminance (per-pixel 3D offset) ──
        bulb_warp = None
        if str(params.get("source", "none")) == "input_image":
            lum = wired_source_lum(params, RW, RH)
            if lum is not None:
                wst = sparam(params, "warp_strength", 0.6)
                bulb_warp = np.stack([(lum - 0.5) * wst,
                                      (lum - 0.5) * wst,
                                      (lum - 0.5) * wst], axis=-1).astype(np.float64)

        shade, hit, col, bg, base_t = _render(
            RW, RH, az, el, cam_dist, p_eff, iterations, steps, warp=bulb_warp)

        # ── Colour mapping ──
        if colormode == "grayscale":
            cbase = np.stack([base_t, base_t, base_t], axis=-1)
        elif colormode == "fire":
            cbase = np.stack([
                np.clip(base_t * 1.7, 0, 1),
                np.clip(base_t * base_t * 1.5, 0, 1),
                np.clip((1.0 - base_t) * 0.3 * base_t, 0, 1)], axis=-1)
        elif colormode == "ice":
            cbase = np.stack([
                np.clip(base_t * 0.3, 0, 1),
                np.clip(0.35 + base_t * 0.65, 0, 1),
                np.clip(0.5 + base_t * 0.5, 0, 1)], axis=-1)
        elif colormode in PALETTES:
            pal = np.array(PALETTES[colormode], dtype=np.float32) / 255.0
            idx = (np.clip(base_t, 0, 1) * (len(pal) - 1)).astype(np.int32)
            cbase = pal[idx]
        else:  # rainbow (cosine palette)
            fr, fg, fb = _cos_pal(base_t, palette_shift)
            cbase = np.stack([fr, fg, fb], axis=-1).astype(np.float32)

        lit = cbase * shade[..., None]
        rgb_small = np.where(hit[..., None], lit, bg).astype(np.float32)
        rgb_small = np.clip(rgb_small, 0.0, 1.0)

        # ── Upscale to canvas ──
        img = Image.fromarray((rgb_small * 255.0).astype(np.uint8), "RGB")
        img = img.resize((W, H), Image.BILINEAR)
        rgb = (np.asarray(img, dtype=np.float32) / 255.0).astype(np.float32)

        capture_frame("470", rgb)
        save(rgb, mn(470, "Mandelbulb 3D Fractal"), out_dir)
        try:
            mask = hit.astype(np.float32)
            # upscale mask to canvas (nearest) for spatial selection
            mimg = Image.fromarray((mask * 255.0).astype(np.uint8), "L")
            mimg = mimg.resize((W, H), Image.NEAREST)
            mask_u = (np.asarray(mimg, dtype=np.float32) / 255.0).astype(np.float32)
            # depth/iteration field: normalised trap
            fld = np.clip(base_t, 0.0, 1.0).astype(np.float32)
            fimg = Image.fromarray((fld * 255.0).astype(np.uint8), "L")
            fimg = fimg.resize((W, H), Image.BILINEAR)
            fld_u = (np.asarray(fimg, dtype=np.float32) / 255.0).astype(np.float32)
            write_mask(out_dir, mask_u)
            write_field(out_dir, fld_u)
            write_scalars(out_dir, power=float(p_eff), iterations=float(iterations),
                          steps=float(steps), camera_azimuth=float(az),
                          camera_elevation=float(el), cam_dist=cam_dist,
                          animated=int(anim_mode != "none"),
                          hit_fraction=float(mask.mean()))
        except Exception:
            pass
        return rgb
    except Exception as exc:
        fallback = np.full((H, W, 3), 0.06, dtype=np.float32)
        save(fallback, mn(470, "Mandelbulb 3D Fractal"), out_dir)
        print(f"[method_470] ERROR: {exc}")
        return fallback
