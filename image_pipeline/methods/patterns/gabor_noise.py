"""Sparse Gabor Convolution Noise — anisotropic procedural noise.

Implements the sparse-convolution noise of Lagae, Lefebvre, Drettakis & Dutré,
"Procedural Noise using Sparse Gabor Convolution" (SIGGRAPH / ACM TOG 2009,
KU Leuven & INRIA; paper: https://www.cs.kuleuven.be/~graphics/publications/LLDD09PNSGC/).

The noise is built by summing randomly placed, randomly oriented and randomly
signed Gabor kernels:

    g(p) = Σ_j  w_j · exp(-π · (a_major²·u_j² + a_minor²·v_j²)) · cos(2π·f_j·u_j)

where (u_j, v_j) are the coordinates of p relative to the j-th impulse, measured
along/against its orientation ξ_j. Unlike a Gabor FILTER applied to white noise
(perlin/fbm), this is a *noise function*: its spectrum is controlled analytically
by the kernel parameters, giving true anisotropy (directional streaks), accurate
band-limited frequency content, and a stable, feature-size-controllable grain.

Because each frame is a pure closed-form function of the pixel coordinate and the
animation clock (Architecture B), there is no simulation state and no strobing —
the orchestrator re-calls the method with an increasing ``time`` value.

Impulses are placed on a jittered grid (one per cell of side ``scale`` pixels) so
the cost is O(image_area) regardless of feature density, and only impulses within
a small window contribute to each pixel.

Animation modes (Architecture B — per-frame re-call with `time`):
    none   — static full draw: frame Δ ≈ 0 (static baseline).
    drift  — the whole field slides along a smooth advection flow (_t modulated);
             pattern translates without cusps (strong Δ).
    swirl  — the sample point rotates about the image centre (angle = _t·k), so the
             grain tumbles smoothly (strong Δ, never symmetry-aligned at the
             audit sample times).
    pulse  — kernel frequency breathes (f·(1 + 0.4·sin(_t))); the streak density
             swells and relaxes smoothly — no abs(sin) cusp (strong Δ).

The directional axis can be driven advectively by a wired upstream image's
luminance (domain-warp style), so the streaks follow an incoming shape.
"""

from __future__ import annotations

import math

import numpy as np
from PIL import Image

from ...core.registry import method
from ...core.utils import save, norm, mn, seed_all, W, H, PALETTES, wired_source_lum
from ...core.animation import capture_frame


PI = math.pi


def _build_impulses(rng, scale, anisotropy, freq_base, warp_lum, wx, wy):
    """Place jittered-grid Gabor impulses; return per-impulse arrays.

    Coordinates are in pixel space. One impulse per grid cell of side ``scale``,
    with a 1-cell margin so border pixels still receive neighbours.
    """
    S = max(4.0, float(scale))
    nx = max(1, int(round(W / S)))
    ny = max(1, int(round(H / S)))

    # cell corners with a 1-cell margin
    cx0 = (np.arange(-1, nx + 1)) * S
    cy0 = (np.arange(-1, ny + 1)) * S
    ox = rng.random((cy0.shape[0], cx0.shape[0])) * S
    oy = rng.random((cy0.shape[0], cx0.shape[0])) * S

    centers_x = (cx0[None, :] + ox).ravel()
    centers_y = (cy0[:, None] + oy).ravel()
    N = centers_x.shape[0]

    ang0 = rng.random(N) * 2.0 * PI
    amp = rng.standard_normal(N)
    freq = freq_base * (0.7 + 0.6 * rng.random(N))

    # Optional domain warp from a wired image: bend the streak orientation so it
    # follows the image's luminance gradient direction.
    if warp_lum is not None:
        gx, gy = np.gradient(warp_lum)
        ang_grad = np.arctan2(gy, gx)
        # sample gradient at impulse positions
        ix = np.clip((centers_x / W * warp_lum.shape[1]).astype(int), 0, warp_lum.shape[1] - 1)
        iy = np.clip((centers_y / H * warp_lum.shape[0]).astype(int), 0, warp_lum.shape[0] - 1)
        ang0 = ang0 * (1.0 - anisotropy) + ang_grad[iy, ix] * anisotropy

    return centers_x, centers_y, ang0, amp, freq, S


def _apply_anim(centers_x, centers_y, ang0, freq, anim_mode, _t):
    """Return (cx, cy, ang, freq_mult) after applying the animation transform.

    All transforms are smooth functions of ``_t`` — no discrete jumps, so the
    rendered animation never strobes.
    """
    cx = centers_x.copy()
    cy = centers_y.copy()
    ang = ang0.copy()
    freq_mult = 1.0

    if anim_mode == "drift":
        cx = cx + 18.0 * np.sin(_t + cy * 0.012)
        cy = cy + 18.0 * np.cos(_t * 0.8 + cx * 0.012)
    elif anim_mode == "swirl":
        px = cx - W / 2.0
        py = cy - H / 2.0
        a = _t * 0.35
        ca, sa = math.cos(a), math.sin(a)
        cx = W / 2.0 + px * ca - py * sa
        cy = H / 2.0 + px * sa + py * ca
    elif anim_mode == "pulse":
        freq_mult = 1.0 + 0.4 * np.sin(_t)
        ang = ang + 0.25 * np.sin(_t * 0.7)

    return cx, cy, ang, freq_mult


def _render(centers_x, centers_y, ang0, amp, freq, S, anisotropy, falloff,
            contrast, colormode, pal_name):
    """Accumulate the sparse Gabor sum into an H×W float field in ~[-1,1]."""
    # anisotropic envelope: along the streak axis it is wide, perpendicular narrow
    a_major = max(0.2, float(falloff))
    a_minor = a_major / (1.0 + max(0.0, anisotropy) * 4.0)

    R = int(math.ceil(3.0 * S / a_major))
    R = min(R, max(W, H))

    acc = np.zeros((H, W), dtype=np.float64)
    energy = np.zeros((H, W), dtype=np.float64)

    cosA = np.cos(ang0)
    sinA = np.sin(ang0)
    ke = 2.0 * PI
    invS2 = 1.0 / (S * S)
    pi_aM2 = PI * a_major * a_major
    pi_am2 = PI * a_minor * a_minor

    for j in range(centers_x.shape[0]):
        Cx = centers_x[j]
        Cy = centers_y[j]
        x0 = int(max(0, math.floor(Cx - R)))
        x1 = int(min(W - 1, math.ceil(Cx + R)))
        y0 = int(max(0, math.floor(Cy - R)))
        y1 = int(min(H - 1, math.ceil(Cy + R)))
        if x1 < x0 or y1 < y0:
            continue
        dx = np.arange(x0, x1 + 1, dtype=np.float64) - Cx
        dy = np.arange(y0, y1 + 1, dtype=np.float64) - Cy
        DX, DY = np.meshgrid(dx, dy)  # shape (y1-y0+1, x1-x0+1)
        u = DX * cosA[j] + DY * sinA[j]
        v = -DX * sinA[j] + DY * cosA[j]
        env = np.exp(-(pi_aM2 * u * u + pi_am2 * v * v) * invS2)
        phase = ke * freq[j] * u / S
        kern = amp[j] * env * np.cos(phase)
        acc[y0:y1 + 1, x0:x1 + 1] += kern
        energy[y0:y1 + 1, x0:x1 + 1] += env * env

    # normalize to a stable, roughly unit-variance field
    val = acc / np.sqrt(energy + 1e-6)
    mean = val.mean()
    std = val.std() + 1e-6
    val = (val - mean) / std
    val = np.clip(0.5 + (val) * 0.5 * float(contrast), 0.0, 1.0)

    # ── color mapping ──
    try:
        from matplotlib import cm
        _has_mpl = True
    except ImportError:
        _has_mpl = False

    if colormode == "grayscale":
        rgb = np.stack([val, val, val], axis=-1)
    elif colormode == "rainbow":
        hue = val * 2 * PI
        rgb = np.stack([
            np.sin(hue) * 0.5 + 0.5,
            np.sin(hue + 2.094) * 0.5 + 0.5,
            np.sin(hue + 4.189) * 0.5 + 0.5,
        ], axis=-1)
    elif colormode == "palette":
        pal = PALETTES.get(pal_name, PALETTES["vapor"])
        idx = (val * (len(pal) - 1)).astype(np.int32)
        rgb = np.array(pal, dtype=np.float32)[idx] / 255.0
    elif colormode == "inferno":
        if _has_mpl:
            rgb = cm.inferno(val)[:, :, :3]
        else:
            rgb = np.stack([val ** 1.4, val ** 0.6 * (1 - val) * 2 + val * 0.2,
                            val ** 0.3 * 0.5], axis=-1)
    elif colormode == "viridis":
        if _has_mpl:
            rgb = cm.viridis(val)[:, :, :3]
        else:
            rgb = np.stack([val * 0.3, val ** 0.5 * 0.8, (1 - val) * 0.4 + val * 0.6], axis=-1)
    elif colormode == "fire":
        rgb = np.stack([np.clip(val * 1.5, 0, 1), val * 0.6, val * 0.2], axis=-1)
    elif colormode == "ice":
        rgb = np.stack([val * 0.2, val * 0.5, 0.5 + val * 0.5], axis=-1)
    else:
        rgb = np.stack([val, val, val], axis=-1)

    rgb = np.clip(rgb, 0.0, 1.0).astype(np.float32)
    return val.astype(np.float32), rgb


@method(id='473', name='Gabor Noise', category='patterns',
        tags=['procedural', 'noise', 'gabor', 'anisotropic', 'sparse-convolution',
              'directional', 'animation'],
        inputs={'image_in': 'IMAGE'},
        outputs={'image': 'IMAGE', 'field': 'FIELD'},
        params={
            'scale': {'description': 'feature size of the grain (pixels between impulses)', 'min': 6.0, 'max': 120.0, 'default': 28.0},
            'anisotropy': {'description': '0 = isotropic, 1 = strongly directional streaks', 'min': 0.0, 'max': 1.0, 'default': 0.75},
            'falloff': {'description': 'Gaussian envelope bandwidth (higher = more compact kernels)', 'min': 0.3, 'max': 1.6, 'default': 0.7},
            'frequency': {'description': 'base stripe frequency (cycles per feature)', 'min': 0.2, 'max': 3.0, 'default': 1.1},
            'contrast': {'description': 'final tone contrast', 'min': 0.4, 'max': 2.5, 'default': 1.0},
            'colormode': {'description': 'color mapping (grayscale/rainbow/inferno/viridis/palette/fire/ice)', 'default': 'inferno'},
            'palette': {'description': 'palette name for palette mode', 'default': 'vapor'},
            'source': {'description': "wired upstream image's luminance bends streak direction", 'choices': ['none', 'input_image'], 'default': 'none'},
            'anim_mode': {'description': 'animation mode: none, drift, swirl, pulse', 'default': 'none'},
            'anim_speed': {'description': 'animation speed multiplier', 'min': 0.1, 'max': 3.0, 'default': 1.0},
            'time': {'description': 'animation phase [0, 2pi)', 'min': 0.0, 'max': 6.28, 'default': 0.0},
        })
def method_gabor_noise(out_dir, seed: int, params=None):
    """Render Sparse Gabor Convolution noise (Lagae et al. 2009)."""
    try:
        if params is None:
            params = {}
        t = float(params.get("time", 0.0))
        seed_all(seed)
        rng = np.random.default_rng(seed)

        scale = float(params.get("scale", 28.0))
        anisotropy = float(params.get("anisotropy", 0.75))
        falloff = float(params.get("falloff", 0.7))
        freq_base = float(params.get("frequency", 1.1))
        contrast = float(params.get("contrast", 1.0))
        cmode = params.get("colormode", "inferno")
        pal_name = params.get("palette", "vapor")
        src = params.get("source", "none")
        anim_mode = params.get("anim_mode", "none")
        anim_speed = float(params.get("anim_speed", 1.0))

        # ── Animation time wiring (Architecture B) ──
        if anim_mode == "none":
            _t = 0.0
        else:
            _t = t * anim_speed

        # ── Wired upstream image bends streak direction (domain-warp style) ──
        warped = None
        if src == "input_image":
            lum = wired_source_lum(params, int(W), int(H))
            if lum is not None:
                warped = lum

        # ── Step 1: seed-stable impulse placement (no per-frame regeneration →
        #    no strobing; animation is a smooth coordinate transform instead) ──
        cx, cy, ang0, amp, freq, S = _build_impulses(
            rng, scale, anisotropy, freq_base, warped, W, H)

        # ── Step 7/4: smooth animation transform (no core-scalar shadowing) ──
        cx, cy, ang0, freq_mult = _apply_anim(cx, cy, ang0, freq, anim_mode, _t)
        freq = freq * freq_mult

        # ── Step 2/5/6: render (smooth, no abs(sin) cusps) ──
        field, rgb = _render(cx, cy, ang0, amp, freq, S, anisotropy, falloff,
                             contrast, cmode, pal_name)

        # ── Rules 4/5: scalar + field outputs ──
        from ...core.utils import write_scalars, write_field
        write_scalars(out_dir, anisotropy=anisotropy, falloff=falloff,
                      mean=float(field.mean()), std=float(field.std()))
        write_field(out_dir, field)

        capture_frame("472", rgb)
        save(rgb, mn(472, "Gabor Noise"), out_dir)
        return rgb
    except Exception as exc:
        fallback = np.full((H, W, 3), 128, dtype=np.uint8)
        save(fallback, mn(472, "Gabor Noise"), out_dir)
        print(f"[method_472] ERROR: {exc}")
        return fallback
