"""Procedural Phasor Noise — highly-contrasted controllable oscillating patterns.

Implements the phasor noise of Tricard, Efremov, Zanni, Neyret, Martínez &
Lefebvre, "Procedural Phasor Noise" (ACM TOG / SIGGRAPH 2019, Univ. of Lyon /
INRIA; paper: https://hal.science/hal-02118508/file/ProceduralPhasorNoise.pdf,
project: http://thibaulttricard.fr/project_page/phasor_noise/phasor.html).

Phasor noise builds a stochastic *smooth phase field* — a phasor field — as a
sparse convolution of spatially-located Gaussians multiplied by unit complex
oscillations, then feeds the field's ARGUMENT into a periodic profile function
(sine / sawtooth / square).  The key difference from Gabor noise (which sums
real cosines and therefore has an amplitude that fluctuates, washing out
contrast) is that phasor noise sums COMPLEX phasors and extracts only the phase:

    Z(x)     = Σ_j  a_j(x) · exp( i · (2π f_j · u_j + ψ_j) )
    a_j(x)   = exp( -π · b² · |x - p_j|² )          (Gaussian window)
    u_j      = ⟨ x - p_j , dir_j ⟩                  (oriented coordinate)
    phase(x) = atan2( Im Z , Re Z )                 (the phasor field)
    out(x)   = profile( phase(x) + φ_global )

Because the output depends only on the *phase*, the oscillation has CONSTANT
amplitude everywhere — producing the crisp, unbroken stripes / fingerprints /
wood-grain the technique is known for, with local control of both periodicity
(f_j) and orientation (dir_j).

Each frame is a pure closed-form function of pixel coordinate + animation clock
(Architecture B): there is no simulation state and no strobing.  Animating the
global phase φ is essentially free (one add before the profile), so the node is
cheap to render AND genuinely time-varying — it dodges both the
150s-timeout cull and the static-liveness cull.

Animation modes (Architecture B — per-frame re-call with `time`):
    none   — static full draw: frame Δ ≈ 0 (static baseline).
    phase  — the global phase advances (φ = _t); every stripe glides smoothly
             perpendicular to its orientation (strong Δ, zero extra cost).
    drift  — impulses advect along a smooth flow so the field translates.
    swirl  — the impulse cloud rotates about the image centre (grain tumbles).

A wired upstream image's luminance gradient can bend the local streak
orientation (domain-warp style), so the phasor stripes follow an incoming shape.
"""

from __future__ import annotations

import math

import numpy as np

from ...core.registry import method
from ...core.utils import save, mn, seed_all, W, H, PALETTES, wired_source_lum
from ...core.animation import capture_frame

PI = math.pi


def _build_impulses(rng, scale, anisotropy, freq_base, warp_lum):
    """Place jittered-grid phasor impulses; return per-impulse arrays.

    One impulse per grid cell of side ``scale`` px, with a 1-cell margin so
    border pixels still receive neighbours. Each impulse carries an orientation,
    a per-impulse frequency and a random phase offset ψ_j.
    """
    S = max(4.0, float(scale))
    nx = max(1, int(round(W / S)))
    ny = max(1, int(round(H / S)))

    cx0 = (np.arange(-1, nx + 1)) * S
    cy0 = (np.arange(-1, ny + 1)) * S
    ox = rng.random((cy0.shape[0], cx0.shape[0])) * S
    oy = rng.random((cy0.shape[0], cx0.shape[0])) * S

    centers_x = (cx0[None, :] + ox).ravel()
    centers_y = (cy0[:, None] + oy).ravel()
    N = centers_x.shape[0]

    ang0 = rng.random(N) * 2.0 * PI
    psi = rng.random(N) * 2.0 * PI                       # random phase offset
    freq = freq_base * (0.75 + 0.5 * rng.random(N))

    # Optional domain warp: bend streak orientation toward the image's gradient.
    if warp_lum is not None:
        gx, gy = np.gradient(warp_lum)
        ang_grad = np.arctan2(gy, gx)
        ix = np.clip((centers_x / W * warp_lum.shape[1]).astype(int), 0, warp_lum.shape[1] - 1)
        iy = np.clip((centers_y / H * warp_lum.shape[0]).astype(int), 0, warp_lum.shape[0] - 1)
        ang0 = ang0 * (1.0 - anisotropy) + ang_grad[iy, ix] * anisotropy

    return centers_x, centers_y, ang0, psi, freq, S


def _apply_anim(centers_x, centers_y, ang0, anim_mode, _t):
    """Return (cx, cy, ang) after a smooth animation transform (no cusps)."""
    cx = centers_x.copy()
    cy = centers_y.copy()
    ang = ang0.copy()

    if anim_mode == "drift":
        cx = cx + 20.0 * np.sin(_t + cy * 0.010)
        cy = cy + 20.0 * np.cos(_t * 0.8 + cx * 0.010)
    elif anim_mode == "swirl":
        px = cx - W / 2.0
        py = cy - H / 2.0
        a = _t * 0.35
        ca, sa = math.cos(a), math.sin(a)
        cx = W / 2.0 + px * ca - py * sa
        cy = H / 2.0 + px * sa + py * ca
    return cx, cy, ang


def _render(centers_x, centers_y, ang0, psi, freq, S, anisotropy, falloff,
            profile_kind, sharpness, phase_shift, cmode, pal_name):
    """Accumulate the complex phasor sum, extract the phase, apply the profile."""
    a_major = max(0.2, float(falloff))
    a_minor = a_major / (1.0 + max(0.0, anisotropy) * 4.0)

    R = int(math.ceil(3.0 * S / a_major))
    R = min(R, max(W, H))

    acc = np.zeros((H, W), dtype=np.complex128)

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
        DX, DY = np.meshgrid(dx, dy)
        u = DX * cosA[j] + DY * sinA[j]
        v = -DX * sinA[j] + DY * cosA[j]
        env = np.exp(-(pi_aM2 * u * u + pi_am2 * v * v) * invS2)
        theta = ke * freq[j] * u / S + psi[j]
        # complex phasor: Gaussian-windowed unit oscillation
        acc[y0:y1 + 1, x0:x1 + 1] += env * (np.cos(theta) + 1j * np.sin(theta))

    # ── the phasor field: the ARGUMENT of the complex sum (constant amplitude) ──
    phase = np.angle(acc)                                # [-π, π]
    ph = phase + float(phase_shift)

    # ── periodic profile ──
    if profile_kind == "sine":
        val = 0.5 + 0.5 * np.sin(ph)
    elif profile_kind == "sawtooth":
        val = (ph / (2.0 * PI)) % 1.0
    elif profile_kind == "square":
        s = np.sin(ph)
        k = max(1.0, float(sharpness) * 12.0)
        val = 0.5 + 0.5 * np.tanh(k * s)                # smooth square (no aliasing)
    elif profile_kind == "triangle":
        saw = (ph / (2.0 * PI)) % 1.0
        val = 1.0 - np.abs(2.0 * saw - 1.0)
    else:
        val = 0.5 + 0.5 * np.sin(ph)

    # coherence mask: where the phasor magnitude is tiny the phase is undefined;
    # fade those (rare) regions toward mid-grey so they don't show hard noise.
    mag = np.abs(acc)
    coh = np.clip(mag / (mag.mean() + 1e-6), 0.0, 1.0)
    val = 0.5 + (val - 0.5) * coh
    val = np.clip(val, 0.0, 1.0).astype(np.float32)

    # ── color mapping ──
    try:
        from matplotlib import cm
        _has_mpl = True
    except ImportError:
        _has_mpl = False

    if cmode == "grayscale":
        rgb = np.stack([val, val, val], axis=-1)
    elif cmode == "rainbow":
        hue = val * 2 * PI
        rgb = np.stack([
            np.sin(hue) * 0.5 + 0.5,
            np.sin(hue + 2.094) * 0.5 + 0.5,
            np.sin(hue + 4.189) * 0.5 + 0.5,
        ], axis=-1)
    elif cmode == "palette":
        pal = PALETTES.get(pal_name, PALETTES["vapor"])
        idx = (val * (len(pal) - 1)).astype(np.int32)
        rgb = np.array(pal, dtype=np.float32)[idx] / 255.0
    elif cmode == "inferno" and _has_mpl:
        rgb = cm.inferno(val)[:, :, :3]
    elif cmode == "viridis" and _has_mpl:
        rgb = cm.viridis(val)[:, :, :3]
    elif cmode == "fire":
        rgb = np.stack([np.clip(val * 1.5, 0, 1), val * 0.6, val * 0.2], axis=-1)
    elif cmode == "ice":
        rgb = np.stack([val * 0.2, val * 0.5, 0.5 + val * 0.5], axis=-1)
    else:
        rgb = np.stack([val, val, val], axis=-1)

    rgb = np.clip(rgb, 0.0, 1.0).astype(np.float32)
    return val, rgb


@method(id='1006', name='Phasor Noise', category='patterns',
        tags=['procedural', 'noise', 'phasor', 'anisotropic', 'sparse-convolution',
              'stripes', 'directional', 'animation'],
        inputs={'image_in': 'IMAGE'},
        outputs={'image': 'IMAGE', 'field': 'FIELD'},
        params={
            'scale': {'description': 'feature size of the grain (pixels between impulses)', 'min': 6.0, 'max': 120.0, 'default': 32.0},
            'anisotropy': {'description': '0 = isotropic, 1 = strongly directional streaks', 'min': 0.0, 'max': 1.0, 'default': 0.8},
            'falloff': {'description': 'Gaussian envelope bandwidth (higher = more compact kernels)', 'min': 0.3, 'max': 1.6, 'default': 0.7},
            'frequency': {'description': 'base stripe frequency (cycles per feature)', 'min': 0.2, 'max': 3.0, 'default': 1.2},
            'profile': {'description': 'periodic profile fed by the phase (sine/sawtooth/square/triangle)', 'choices': ['sine', 'sawtooth', 'square', 'triangle'], 'default': 'sine'},
            'sharpness': {'description': 'edge sharpness for the square profile', 'min': 0.1, 'max': 2.0, 'default': 0.6},
            'colormode': {'description': 'color mapping (grayscale/rainbow/inferno/viridis/palette/fire/ice)', 'default': 'inferno'},
            'palette': {'description': 'palette name for palette mode', 'default': 'vapor'},
            'source': {'description': "wired upstream image's luminance bends streak direction", 'choices': ['none', 'input_image'], 'default': 'none'},
            'anim_mode': {'description': 'animation mode: none, phase, drift, swirl', 'choices': ['none', 'phase', 'drift', 'swirl'], 'default': 'none'},
            'anim_speed': {'description': 'animation speed multiplier', 'min': 0.1, 'max': 3.0, 'default': 1.0},
            'time': {'description': 'animation phase [0, 2pi)', 'min': 0.0, 'max': 6.28, 'default': 0.0},
        })
def method_phasor_noise(out_dir, seed: int, params=None):
    """Render Procedural Phasor Noise (Tricard et al. 2019)."""
    try:
        if params is None:
            params = {}
        t = float(params.get("time", 0.0))
        seed_all(seed)
        rng = np.random.default_rng(seed)

        scale = float(params.get("scale", 32.0))
        anisotropy = float(params.get("anisotropy", 0.8))
        falloff = float(params.get("falloff", 0.7))
        freq_base = float(params.get("frequency", 1.2))
        profile_kind = params.get("profile", "sine")
        sharpness = float(params.get("sharpness", 0.6))
        cmode = params.get("colormode", "inferno")
        pal_name = params.get("palette", "vapor")
        src = params.get("source", "none")
        anim_mode = params.get("anim_mode", "none")
        anim_speed = float(params.get("anim_speed", 1.0))

        # ── Animation time wiring (Architecture B) ──
        _t = 0.0 if anim_mode == "none" else t * anim_speed
        # 'phase' mode advances the global phase (nearly free, strong Δ).
        phase_shift = _t if anim_mode == "phase" else 0.0

        # ── Wired upstream image bends streak direction (domain-warp style) ──
        warped = None
        if src == "input_image":
            lum = wired_source_lum(params, int(W), int(H))
            if lum is not None:
                warped = lum

        cx, cy, ang0, psi, freq, S = _build_impulses(
            rng, scale, anisotropy, freq_base, warped)
        cx, cy, ang0 = _apply_anim(cx, cy, ang0, anim_mode, _t)

        field, rgb = _render(cx, cy, ang0, psi, freq, S, anisotropy, falloff,
                             profile_kind, sharpness, phase_shift, cmode, pal_name)

        from ...core.utils import write_scalars, write_field
        write_scalars(out_dir, anisotropy=anisotropy, frequency=freq_base,
                      mean=float(field.mean()), std=float(field.std()))
        write_field(out_dir, field)

        capture_frame("1006", rgb)
        save(rgb, mn(1006, "Phasor Noise"), out_dir)
        return rgb
    except Exception as exc:
        fallback = np.full((H, W, 3), 128, dtype=np.uint8)
        save(fallback, mn(1006, "Phasor Noise"), out_dir)
        print(f"[method_1006] ERROR: {exc}")
        return fallback
