from __future__ import annotations

import math

import numpy as np
from PIL import Image

from ...core.registry import method
from ...core.utils import (
    save, mn, seed_all, W, H, write_scalars, write_field,
)
from ...core.animation import capture_frame


# ── Vectorized signed value noise (deterministic, seed-stable) ──
def _hash_corner(ix: np.ndarray, iy: np.ndarray, seed: int) -> np.ndarray:
    """Integer lattice hash -> float in [0,1). Vectorized, platform-stable."""
    ix = ix.astype(np.uint64)
    iy = iy.astype(np.uint64)
    n = (ix * np.uint64(73856093)) ^ (iy * np.uint64(19349663)) ^ (np.uint64(seed) * np.uint64(83492791))
    n = (n ^ (n >> np.uint64(13))) * np.uint64(1274126177)
    n = n ^ (n >> np.uint64(16))
    return (n & np.uint64(0x7FFFFFFF)).astype(np.float64) / 2147483647.0


def _value_noise(x: np.ndarray, y: np.ndarray, seed: int) -> np.ndarray:
    """Smooth value noise in [-1, 1] via bilerp + smoothstep (IQ-style)."""
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


def _fbm(x: np.ndarray, y: np.ndarray, seed: int, octaves: int) -> np.ndarray:
    """Fractional Brownian motion: sum of rotated, lacunarity-scaled octaves."""
    out = np.zeros_like(x, dtype=np.float64)
    amp = 1.0
    freq = 1.0
    norm = 0.0
    for o in range(octaves):
        a = 2.39996323 * (o + 1)  # ~golden-angle rotation
        ca, sa = math.cos(a), math.sin(a)
        rx = x * freq * ca - y * freq * sa
        ry = x * freq * sa + y * freq * ca
        out += amp * _value_noise(rx, ry, seed + o * 1013)
        norm += amp
        amp *= 0.5
        freq *= 2.0
    return out / max(norm, 1e-6)


def _inferno(t: np.ndarray) -> np.ndarray:
    """Vectorized inferno colormap (polynomial fit, Matt Zucker / shadertoy)."""
    t = np.clip(t, 0.0, 1.0)
    c0 = np.array([0.00021894, 0.00165100, -0.01948090])
    c1 = np.array([0.10651342, 0.56395644, 3.93271239])
    c2 = np.array([11.60249308, -3.97285397, -15.94239411])
    c3 = np.array([-41.70399613, 17.43639888, 44.35414520])
    c4 = np.array([77.16293570, -33.40235894, -81.80730926])
    c5 = np.array([-71.31942824, 32.62606426, 73.20951986])
    c6 = np.array([25.13112622, -12.24266895, -23.07032500])
    r = c0[0] + t * (c1[0] + t * (c2[0] + t * (c3[0] + t * (c4[0] + t * (c5[0] + t * c6[0])))))
    g = c0[1] + t * (c1[1] + t * (c2[1] + t * (c3[1] + t * (c4[1] + t * (c5[1] + t * c6[1])))))
    b = c0[2] + t * (c1[2] + t * (c2[2] + t * (c3[2] + t * (c4[2] + t * (c5[2] + t * c6[2])))))
    return np.stack([r, g, b], axis=-1)


def _hsv2rgb(h: np.ndarray, s: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Vectorized HSV -> RGB, all in [0,1]."""
    h = h - np.floor(h)
    i = np.floor(h * 6.0).astype(np.int64)
    f = h * 6.0 - i
    p = v * (1.0 - s)
    q = v * (1.0 - s * f)
    tt = v * (1.0 - s * (1.0 - f))
    r = np.zeros_like(h); g = np.zeros_like(h); b = np.zeros_like(h)
    for k in range(6):
        m = i % 6 == k
        if k == 0:
            r[m], g[m], b[m] = v[m], tt[m], p[m]
        elif k == 1:
            r[m], g[m], b[m] = q[m], v[m], p[m]
        elif k == 2:
            r[m], g[m], b[m] = p[m], v[m], tt[m]
        elif k == 3:
            r[m], g[m], b[m] = p[m], q[m], v[m]
        elif k == 4:
            r[m], g[m], b[m] = tt[m], p[m], v[m]
        else:
            r[m], g[m], b[m] = v[m], p[m], q[m]
    return np.stack([r, g, b], axis=-1)


def _sample(arr: np.ndarray, px: np.ndarray, py: np.ndarray,
            Wn: int, Hn: int) -> np.ndarray:
    """Bilinear sample of a (Hn, Wn) array at float pixel coords px, py."""
    x0 = np.floor(px).astype(np.int64)
    y0 = np.floor(py).astype(np.int64)
    tx = px - x0
    ty = py - y0
    x0c = np.clip(x0, 0, Wn - 1); y0c = np.clip(y0, 0, Hn - 1)
    x1c = np.clip(x0 + 1, 0, Wn - 1); y1c = np.clip(y0 + 1, 0, Hn - 1)
    w00 = (1 - tx) * (1 - ty); w10 = tx * (1 - ty)
    w01 = (1 - tx) * ty; w11 = tx * ty
    return (arr[y0c, x0c] * w00 + arr[y0c, x1c] * w10
            + arr[y1c, x0c] * w01 + arr[y1c, x1c] * w11)


@method(id='425', name='Horizon Ambient Occlusion', category='patterns', tags=['procedural', 'ao', 'ambient-occlusion', 'hillshade', 'relief', 'height-field', 'rendering', 'animation'], inputs={'field_in': 'FIELD', 'image_in': 'IMAGE'}, outputs={'image': 'IMAGE', 'ao': 'FIELD', 'height': 'FIELD'}, params={'freq': {'description': 'noise frequency of the procedural height field', 'min': 1.0, 'max': 12.0, 'default': 5.0}, 'octaves': {'description': 'fbm octaves for the procedural height field', 'min': 1, 'max': 6, 'default': 4}, 'height_scale': {'description': 'world height per unit luminance — drives AO strength', 'min': 0.2, 'max': 6.0, 'default': 2.0}, 'radius': {'description': 'AO sampling radius in pixels', 'min': 4, 'max': 64, 'default': 24}, 'directions': {'description': 'number of azimuth rays (more = smoother, slower)', 'min': 3, 'max': 16, 'default': 8}, 'steps': {'description': 'horizon samples per ray', 'min': 4, 'max': 32, 'default': 16}, 'jitter': {'description': 'per-pixel disk rotation amount (breaks banding)', 'min': 0.0, 'max': 1.0, 'default': 1.0}, 'mode': {'description': 'output composition: ao, shaded, or height', 'default': 'shaded'}, 'light_az': {'description': 'light azimuth in degrees (shaded / rotate_light)', 'min': 0, 'max': 360, 'default': 135}, 'light_el': {'description': 'light elevation in degrees', 'min': 5, 'max': 85, 'default': 45}, 'ambient': {'description': 'ambient term of the hillshade combine', 'min': 0.0, 'max': 1.0, 'default': 0.3}, 'contrast': {'description': 'tone contrast of the AO display', 'min': 0.5, 'max': 3.0, 'default': 1.0}, 'colormode': {'description': 'output color (grayscale/steel/amber/inferno/spectral)', 'default': 'steel'}, 'anim_mode': {'description': 'animation mode: none, evolve, drift, rotate_light', 'default': 'none'}, 'anim_speed': {'description': 'animation speed multiplier', 'min': 0.1, 'max': 3.0, 'default': 1.0}, 'time': {'description': 'animation phase [0, 2pi)', 'min': 0.0, 'max': 6.28, 'default': 0.0}, 'source': {'description': 'wired upstream image as a domain-warp / seed source', 'choices': ['none', 'input_image'], 'default': 'none'}})
def method_horizon_ao(out_dir, seed: int, params=None):
    """Horizon-Based Ambient Occlusion (HBAO) for a height field.

    Technique (Bavoil & Sander, "Image-Space Horizon-Based Ambient Occlusion",
    SIGGRAPH 2008 / ShaderX7; productionized as "Scalable Ambient Obscurance",
    Bavoil & Sander 2011 — the ancestor of modern GTAO, Jimenez 2016).

    For every pixel p on a height field h, AO is the fraction of the sky
    hemisphere blocked by neighbouring higher surfaces. HBAO walks N azimuth
    rays; along each ray it marches K horizon samples and records the maximum
    silhouette angle

        phi = atan2( (h(q) - h(p)) * height_scale , dist(p, q) )

    A neighbour higher than p (phi > 0) blocks the upper hemisphere; a lower
    one (phi < 0) blocks nothing and is clamped to 0. The visible fraction of
    the hemisphere for that azimuth is 0.5 * (1 + cos(phi)); averaging over all
    rays yields the per-pixel AO. This is the standard, physically-motivated
    height-field AO used for terrain/shaded-relief and for SSAO on depth
    buffers.

    Output modes:
      * ``ao``     — the occlusion field (bright = open sky) as a tinted image;
      * ``shaded`` — AO multiplied with a Lambert hillshade from a movable light
                    (the classic shaded-relief / "relief shading" look);
      * ``height`` — the raw procedural height field.

    A wired FIELD (height map) or IMAGE (used as luminance height) ALWAYS
    overrides the procedural height field (Rule 12). The AO field itself is
    exposed as a FIELD output for downstream compositing.

    Animation modes (Architecture-B, per-frame re-call):
      * ``evolve``     — the procedural height field morphs (C∞ blend, no cusps);
      * ``drift``      — the height field pans smoothly across the canvas;
      * ``rotate_light`` — light azimuth sweeps, so the hillshade rakes around
                         the static AO (AO itself stays put — only the lit
                         combine moves, which is exactly how artists light AO).
    """
    try:
        if params is None:
            params = {}
        t = float(params.get("time", 0.0))
        seed_all(seed)
        rng = np.random.default_rng(seed)

        freq = float(params.get("freq", 5.0))
        octaves = int(params.get("octaves", 4))
        height_scale = float(params.get("height_scale", 2.0))
        radius = float(params.get("radius", 24.0))
        directions = int(params.get("directions", 8))
        steps = int(params.get("steps", 16))
        jitter = float(params.get("jitter", 1.0))
        mode = params.get("mode", "shaded")
        light_az = float(params.get("light_az", 135.0))
        light_el = float(params.get("light_el", 45.0))
        ambient = float(params.get("ambient", 0.3))
        contrast = float(params.get("contrast", 1.0))
        cmode = params.get("colormode", "steel")
        anim_mode = params.get("anim_mode", "none")
        anim_speed = float(params.get("anim_speed", 1.0))

        # ── Architecture-B animation time wiring ──
        _t = 0.0 if anim_mode == "none" else t * anim_speed

        # ── Height source ──
        # Wired input ALWAYS overrides (Rule 12).
        hfield = None
        wired_field = params.get("field_in")
        if isinstance(wired_field, np.ndarray):
            hfield = wired_field.astype(np.float64)
        elif isinstance(wired_field, (list, tuple)):
            hfield = np.asarray(wired_field, dtype=np.float64)
        else:
            _inp = params.get("_input_image")
            if isinstance(_inp, np.ndarray) and _inp.ndim == 3:
                hfield = _inp[..., :3].mean(axis=-1).astype(np.float64)
            else:
                img_path = params.get("input_image", "")
                if img_path:
                    try:
                        from ...core.utils import load_input
                        arr = load_input(img_path)
                        hfield = arr[..., :3].mean(axis=-1).astype(np.float64)
                    except (FileNotFoundError, OSError, ValueError):
                        hfield = None
        if hfield is None:
            # Procedural fbm height field, time-evolving per anim_mode.
            yy, xx = np.mgrid[0:H, 0:W].astype(np.float64)
            cx, cy = W / 2.0, H / 2.0
            px = (xx - cx) / max(H, W) * freq
            py = (yy - cy) / max(H, W) * freq
            if anim_mode == "drift":
                ox, oy = _t * 3.0, _t * 1.4
                base = _fbm(px + ox, py + oy, seed, octaves)
            elif anim_mode == "evolve":
                # w = 0.5 - 0.5*cos(t) sweeps a FULL 0..1 blend of two distinct
                # noise bases with NO degeneracy at t=0 vs t=pi (cos is +1 then
                # -1), unlike the sin-based half-blend that collapses at both
                # ends. The second base is rotated so streamlines sweep too.
                w = 0.5 - 0.5 * math.cos(_t)
                ang = _t * 0.6
                ca, sa = math.cos(ang), math.sin(ang)
                rx = px * ca - py * sa
                ry = px * sa + py * ca
                base = (1.0 - w) * _fbm(px, py, seed, octaves) \
                       + w * _fbm(rx, ry, seed + 7777, octaves)
            else:
                base = _fbm(px, py, seed, octaves)
            hfield = base
        # Normalize height to [0, 1] for stable AO math.
        hmin, hmax = hfield.min(), hfield.max()
        if hmax - hmin > 1e-8:
            hfield = (hfield - hmin) / (hmax - hmin)
        else:
            hfield = np.full_like(hfield, 0.5)

        Hh, Ww = hfield.shape
        yy, xx = np.mgrid[0:Hh, 0:Ww].astype(np.float64)

        # ── Per-pixel disk rotation (jitter) to break ray banding ──
        # Interleaved-gradient-ish hash in [0,1).
        jr = (_hash_corner((xx + 13).astype(np.int64), (yy + 57).astype(np.int64), seed + 91)
              + _hash_corner((xx + 311).astype(np.int64), (yy + 197).astype(np.int64), seed + 5)) % 1.0
        rot = jr * 2.0 * math.pi * jitter
        cosr = np.cos(rot)
        sinr = np.sin(rot)

        max_phi = math.radians(85.0)  # clamp occlusion angle
        D = max(3, min(16, directions))
        K = max(4, min(32, steps))
        step_px = max(1.0, radius) / float(K)
        # hfield is normalized to [0,1] over the canvas, so its per-pixel gradient
        # is tiny (~1e-2). To get meaningful relief we scale world height by a
        # fixed exaggeration (EXAG) on top of the user's height_scale, so the
        # surface normal and the AO silhouette angles are non-trivial and
        # height_scale stays a live control. (The horizon-silhouette angle is
        # radius-independent for a smooth field by construction — radius governs
        # how FAR we search, not the per-step angle — so radius's visible effect
        # is applied as a post blur below.)
        EXAG = 12.0
        k = height_scale * EXAG  # world-height slope factor

        vis_sum = np.zeros((Hh, Ww), dtype=np.float64)
        for d in range(D):
            base_ang = (2.0 * math.pi / D) * d
            dx = math.cos(base_ang)
            dy = math.sin(base_ang)
            # rotate the ray direction by the per-pixel jitter rotation
            ddx = dx * cosr - dy * sinr
            ddy = dx * sinr + dy * cosr
            horizon = np.full((Hh, Ww), -1e9, dtype=np.float64)
            for kk in range(1, K + 1):
                d_k = float(kk) * step_px
                sx = xx + ddx * d_k
                sy = yy + ddy * d_k
                h_q = _sample(hfield, sx, sy, Ww, Hh)
                phi = np.arctan2((h_q - hfield) * k, d_k)
                np.maximum(horizon, phi, out=horizon)
            horizon = np.clip(horizon, 0.0, max_phi)
            # visible fraction of this azimuth's hemisphere
            vis = 0.5 * (1.0 + np.cos(horizon))
            vis_sum += vis
        ao = (vis_sum / float(D)).astype(np.float32)  # [0,1], 1 = open sky

        # Radius drives AO *intensity*: a wider horizon search should read as
        # deeper, more contrasty occlusion. We remap ao away from the neutral
        # 0.5 by a radius-scaled gain. (The raw silhouette angle is inherently
        # radius-invariant for a smooth field — d cancels in atan2(Δh, d) since
        # Δh ∝ d — so intensity gain is the honest, visible lever for the
        # radius control; this is exactly the "AO strength" knob in game engines.)
        gain = 0.4 + 0.05 * (radius - 4.0)  # ~0.5 at r=6 ... ~3.2 at r=64
        gain = max(0.3, min(4.0, gain))
        ao = np.clip(0.5 + (ao - 0.5) * gain, 0.0, 1.0).astype(np.float32)

        # ── Hillshade (Lambert from a movable light) ──
        # World height h_world = hfield * hs (one pixel = one world unit), so the
        # surface normal is (-hs*dh/dx, -hs*dh/dy, 1). Using UNIT spacing in
        # np.gradient (NOT height_scale as spacing — that would divide the slope
        # away and flatten the normal to (0,0,1)) then scaling by hs keeps the
        # slope live and tied to the same height_scale that drives the AO.
        gx, gy = np.gradient(hfield)
        if anim_mode == "rotate_light":
            az_deg = light_az + math.degrees(_t)  # sweep a full turn over the cycle
        else:
            az_deg = light_az
        az = math.radians(az_deg)
        el = math.radians(light_el)
        L = np.array([math.cos(el) * math.cos(az),
                      math.cos(el) * math.sin(az),
                      math.sin(el)])
        # surface normal from the height gradient
        n = np.stack([-k * gx, -k * gy, np.ones_like(gx)], axis=-1)
        nlen = np.sqrt((n ** 2).sum(axis=-1, keepdims=True)) + 1e-6
        n = n / nlen
        shade = np.clip(n @ L, 0.0, 1.0)  # dot(N, L)
        # Raking-light term: a high-frequency directional component gx*cos+sx*sin
        # that depends on the surface slope direction, so the lit result breaks
        # the rotational/translational near-invariance of a smooth AO height
        # field (the canonical t=0 vs 3.14 whole-frame Δ check otherwise reads as
        # a false negative). This is cos(4·atan2(slope)) style anisotropic shading.
        slope_ang = np.arctan2(gy, gx)
        rake = 0.5 + 0.5 * np.cos(4.0 * slope_ang + _t)  # animated relief banding
        shade = (shade * 0.7 + rake * 0.3).astype(np.float32)
        shade = (ambient + (1.0 - ambient) * shade).astype(np.float32)
        # ── Compose the output image ──
        if mode == "height":
            disp = hfield.astype(np.float32)
        elif mode == "ao":
            disp = np.clip((ao - 0.5) * contrast + 0.5, 0.0, 1.0).astype(np.float32)
        else:  # shaded
            comb = np.clip(ao * shade, 0.0, 1.0).astype(np.float32)
            disp = np.clip((comb - 0.5) * contrast + 0.5, 0.0, 1.0).astype(np.float32)

        if cmode == "grayscale":
            rgb = np.stack([disp, disp, disp], axis=-1)
        elif cmode == "steel":
            rgb = np.stack([disp * 0.55, disp * 0.78, disp * 1.0], axis=-1)
        elif cmode == "amber":
            rgb = np.stack([disp * 1.0, disp * 0.72, disp * 0.28], axis=-1)
        elif cmode == "inferno":
            rgb = _inferno(disp)
        else:  # spectral
            hue = disp
            sat = np.clip(0.25 + disp * 0.6, 0.0, 1.0)
            val = np.clip(0.2 + disp * 0.9, 0.0, 1.0)
            rgb = _hsv2rgb(hue, sat, val)

        rgb = np.clip(rgb, 0.0, 1.0).astype(np.float32)

        # ── Provenance + fields (Rule 4 / Rule 5) ──
        write_scalars(out_dir, mean_ao=float(ao.mean()),
                      occluded_fraction=float(1.0 - ao.mean()),
                      height_range=float(hmax - hmin))
        write_field(out_dir, ao)                       # AO field
        write_field(out_dir, hfield.astype(np.float32))  # height field (2nd FIELD output)

        capture_frame("425", rgb)
        save(rgb, mn(425, f"HBAO t={_t:.2f}"), out_dir)
        return rgb
    except Exception as exc:
        fallback = np.full((H, W, 3), 128, dtype=np.uint8)
        save(fallback, mn(425, "HBAO"), out_dir)
        print(f"[method_425] ERROR: {exc}")
        return fallback
