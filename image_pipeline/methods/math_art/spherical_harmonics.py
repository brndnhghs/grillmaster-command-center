"""Animated Spherical Harmonics — 3D atomic orbital visualization."""
from __future__ import annotations
import math
import random
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter
try:
    from scipy.special import sph_harm_y  # scipy >= 1.15
except ImportError:
    # scipy < 1.15: sph_harm(m, n, theta, phi) — swapped arg order
    from scipy.special import sph_harm as _sph_harm
    def sph_harm_y(n, m, theta, phi):
        return _sph_harm(m, n, theta, phi)

from ...core.registry import method
from ...core.utils import save, norm, mn, seed_all, BG_DEFAULT, W, H
from ...core.animation import capture_frame


# ── Constants ──
DARK_BG = (5, 5, 25)
PI = math.pi
TAU = 2 * PI

# Colours for positive & negative phase lobes (used in non-phase modes)
POS_COLOR = np.array([60, 120, 255], dtype=np.float32)   # electric blue
NEG_COLOR = np.array([255, 140, 30], dtype=np.float32)   # fiery orange

# Surface sampling resolution
N_THETA = 100
N_PHI = 140


def _rot_y(points: np.ndarray, angle: float) -> np.ndarray:
    c, s = math.cos(angle), math.sin(angle)
    x = points[:, 0] * c + points[:, 2] * s
    z = -points[:, 0] * s + points[:, 2] * c
    out = points.copy()
    out[:, 0] = x; out[:, 2] = z
    return out


def _rot_x(points: np.ndarray, angle: float) -> np.ndarray:
    c, s = math.cos(angle), math.sin(angle)
    y = points[:, 1] * c - points[:, 2] * s
    z = points[:, 1] * s + points[:, 2] * c
    out = points.copy()
    out[:, 1] = y; out[:, 2] = z
    return out


def _hue_to_rgb(hue: float) -> tuple[int, int, int]:
    """Convert hue 0-1 to bright RGB (HSV with S=0.9, V=0.95)."""
    h = hue * 6.0
    i = int(h) % 6
    f = h - math.floor(h)
    q = 1.0 - f
    t = f
    s, v = 0.9, 0.95
    c = v * s
    x = c * (1.0 - abs(h % 2.0 - 1.0))
    m = v - c
    r, g, b = [(c, x, 0), (x, c, 0), (0, c, x), (0, x, c), (x, 0, c), (c, 0, x)][i]
    return (int((r + m) * 255), int((g + m) * 255), int((b + m) * 255))


def _make_surface(y_values: np.ndarray, amplitude: float
                  ) -> tuple:
    """Build 3D point clouds from spherical harmonic values on grid.

    Returns:
        pos_points, neg_points, pos_vals, neg_vals, pos_complex, neg_complex
    """
    theta = np.linspace(0, PI, N_THETA)
    phi = np.linspace(0, TAU, N_PHI)
    TH, PH = np.meshgrid(theta, phi, indexing='ij')

    y_mag = np.abs(y_values)
    re = np.real(y_values)
    y_complex_flat = y_values.ravel()

    pos_mask = re > 0.01
    neg_mask = re < -0.01

    r_pos = np.where(pos_mask, y_mag * amplitude, 0.0)
    r_neg = np.where(neg_mask, y_mag * amplitude, 0.0)

    sin_th = np.sin(TH.ravel())
    cos_th = np.cos(TH.ravel())
    cos_ph = np.cos(PH.ravel())
    sin_ph = np.sin(PH.ravel())

    pos_idx = np.where(pos_mask.ravel())[0]
    r_p = r_pos.ravel()[pos_idx]
    pos_points = np.column_stack([
        r_p * sin_th[pos_idx] * cos_ph[pos_idx],
        r_p * cos_th[pos_idx],
        r_p * sin_th[pos_idx] * sin_ph[pos_idx],
    ])
    pos_vals = y_mag.ravel()[pos_idx]
    pos_c = y_complex_flat[pos_idx]

    neg_idx = np.where(neg_mask.ravel())[0]
    r_n = r_neg.ravel()[neg_idx]
    neg_points = np.column_stack([
        r_n * sin_th[neg_idx] * cos_ph[neg_idx],
        r_n * cos_th[neg_idx],
        r_n * sin_th[neg_idx] * sin_ph[neg_idx],
    ])
    neg_vals = y_mag.ravel()[neg_idx]
    neg_c = y_complex_flat[neg_idx]

    return pos_points, neg_points, pos_vals, neg_vals, pos_c, neg_c


def _render_orbital(pos_points, neg_points, pos_vals, neg_vals,
                    pos_complex, neg_complex,
                    camera_rot_y: float, camera_rot_x: float,
                    scale: float, px: float, py: float,
                    color_mode: str, glow_strength: float = 1.0) -> Image.Image:
    """Render orbital with configurable coloring mode.

    color_mode:
      "bipolar"   — blue for Re>0, orange for Re<0 (classic)
      "phase"     — hue from complex argument arg(Y), brightness from |Y|
    """
    def _project(pts):
        if len(pts) == 0:
            return np.zeros((0, 3)), np.zeros(0)
        pp = _rot_y(_rot_x(pts.copy(), camera_rot_x), camera_rot_y)
        x = px + pp[:, 0] * scale
        y = py - pp[:, 1] * scale
        return np.column_stack([x, y, pp[:, 2]]), pp[:, 2]

    pos_proj, pos_z = _project(pos_points)
    neg_proj, neg_z = _project(neg_points)

    entries = []
    if len(pos_proj) > 0:
        for i in range(len(pos_proj)):
            entries.append((pos_z[i], 0, pos_proj[i, 0], pos_proj[i, 1],
                            pos_vals[i], pos_complex[i]))
    if len(neg_proj) > 0:
        for i in range(len(neg_proj)):
            entries.append((neg_z[i], 1, neg_proj[i, 0], neg_proj[i, 1],
                            neg_vals[i], neg_complex[i]))

    entries.sort(key=lambda p: -p[0])

    all_mags = np.array([e[4] for e in entries]) if entries else np.array([1.0])
    mag_max = max(all_mags.max(), 0.01)

    canvas = Image.new("RGB", (W, H), DARK_BG)
    drw = ImageDraw.Draw(canvas)

    for entry in entries:
        _, sign, sx, sy, mag, val_c = entry
        ix, iy = int(sx), int(sy)

        brightness = 0.3 + 0.7 * (mag / mag_max)

        if color_mode == "phase":
            # Color by complex argument: hue = arg(Y) mapped 0→1
            arg = math.atan2(val_c.imag, val_c.real)
            hue = (arg + PI) / TAU  # 0 to 1
            r, g, b = _hue_to_rgb(hue)
        else:
            # Bipolar: blue for positive, orange for negative
            if sign == 0:
                r, g, b = int(POS_COLOR[0]), int(POS_COLOR[1]), int(POS_COLOR[2])
            else:
                r, g, b = int(NEG_COLOR[0]), int(NEG_COLOR[1]), int(NEG_COLOR[2])

        cr = max(1, int(r * brightness * glow_strength))
        cg = max(1, int(g * brightness * glow_strength))
        cb = max(1, int(b * brightness * glow_strength))

        dot_r = max(1, int(1.5 + 4.0 * brightness))
        drw.ellipse((ix - dot_r, iy - dot_r, ix + dot_r, iy + dot_r),
                    fill=(cr, cg, cb))

    return canvas.filter(ImageFilter.GaussianBlur(radius=2.0))


def _smoothstep(t: float) -> float:
    return t * t * (3 - 2 * t)


# Waveform helper for twist profile — determines how ring speed varies with latitude
_twist_wave_type = "sine"
_twist_pulse_width = 0.15


def _wave(phase_rad: float) -> float:
    """Return amplitude in [-1, 1] for given phase in radians."""
    if _twist_wave_type == "sine":
        return math.sin(phase_rad)
    elif _twist_wave_type == "square":
        return 1.0 if math.sin(phase_rad) >= 0 else -1.0
    elif _twist_wave_type == "sawtooth":
        norm = phase_rad / TAU
        return 2.0 * (norm - math.floor(norm + 0.5))
    elif _twist_wave_type == "triangle":
        norm = phase_rad / TAU
        return 2.0 * abs(2.0 * (norm - math.floor(norm + 0.5))) - 1.0
    elif _twist_wave_type == "pulse":
        t_mod = (phase_rad % TAU) / TAU
        return 1.0 if t_mod < _twist_pulse_width else 0.0
    elif _twist_wave_type == "gaussian":
        sigma = _twist_pulse_width * TAU
        t_mod = min(phase_rad % TAU, (TAU - phase_rad) % TAU)
        return math.exp(-0.5 * (t_mod / sigma) ** 2) * 2.0 - 1.0
    elif _twist_wave_type == "chirp":
        chirp_phase = phase_rad * (1.0 + 1.5 * phase_rad / TAU)
        return math.sin(chirp_phase)
    elif _twist_wave_type == "noise":
        mod = (0.5 + 0.5 * math.sin(phase_rad * 0.7) *
               math.cos(phase_rad * 1.3) * math.sin(phase_rad * 2.1))
        return (0.5 + 0.5 * math.sin(phase_rad)) * mod * 2.0 - 1.0
    return math.sin(phase_rad)


@method(
    id="104",
    name="Spherical Harmonics",
    description="Spherical Harmonics — math-art node.",
    category="math_art",
    tags=["quantum", "3d", "glow", "expanded", "animation"],
    params={
        "max_l": {"description": "maximum angular momentum quantum number",
                  "min": 1, "max": 8, "default": 5},
        "amplitude": {"description": "orbital size scale",
                      "min": 0.5, "max": 3.0, "default": 1.5},
        "glow_strength": {"description": "glow intensity",
                          "min": 0.5, "max": 3.0, "default": 1.5},
        "anim_mode": {"description": "animation mode",
                       "choices": ["none", "morph", "spin", "breathe",
                                   "phase_color", "twist", "combined",
                                   "superposition"],
                       "default": "morph"},
        "twist_wave": {"description": "waveform for ring spin profile (sine=organic twist, square=banded, sawtooth=shear)",
                        "choices": ["sine", "square", "sawtooth", "triangle",
                                    "pulse", "gaussian", "chirp", "noise"],
                        "default": "sine"},
        "twist_drive": {"description": "how twist evolves — spatial=static waveform profile, oscillator=independent ring oscillators create propagating waves",
                        "choices": ["spatial", "oscillator"],
                        "default": "spatial"},
        "osc_spread": {"description": "frequency spread per ring (oscillator mode only; higher = more chaotic wave propagation)",
                       "min": 0.0, "max": 5.0, "default": 1.5},
        "twist_speed": {"description": "twist rotation speed multiplier",
                        "min": 0.1, "max": 5.0, "default": 1.0},
        "twist_amplitude": {"description": "twist intensity (rings per cycle)",
                            "min": 0.5, "max": 5.0, "default": 2.0},
        "n_frames": {"description": "simulation frames",
                     "min": 60, "max": 500, "default": 180},"anim_speed": {"description": "animation speed multiplier",
                       "min": 0.1, "max": 5.0, "default": 1.0},
    }
)
def method_spherical_harmonics(out_dir: Path, seed: int, params=None):
    """Animated Spherical Harmonics — 3D atomic orbital visualisation.

    Renders glowing 3D isosurfaces of spherical harmonics Y_l^m(θ,φ)
    anim_mode values:
      morph       — smooth transitions between adjacent (l,m) states
      spin        — rotate a single orbital's lobes in place
      breathe     — orbital pulses in size
      phase_color — color by complex phase (rainbow), static shape
      twist       — latitudinal rings spin independently (spatial or oscillator drive)
      combined    — morph + spin + breathe + twist + phase_color all at once
      superposition — blend 3+ harmonics with evolving weights

    Twist drive modes (twist_drive param):
      spatial     — static twist profile from waveform(θ), cumulative top→bottom
      oscillator  — each ring has an independent oscillator; twist waves propagate

    Architecture A — internal simulation loop with capture_frame().
    """
    if params is None:
        params = {}
    t = float(params.get("time", 0.0))
    anim_mode = str(params.get("anim_mode", "morph"))
    anim_speed = float(params.get("anim_speed", 1.0))

    max_l = int(params.get("max_l", 5))
    amplitude = float(params.get("amplitude", 1.5))
    glow_strength = float(params.get("glow_strength", 1.5))
    n_frames = int(params.get("n_frames", 180))

    twist_wave = str(params.get("twist_wave", "sine"))
    twist_drive = str(params.get("twist_drive", "spatial"))
    twist_speed = float(params.get("twist_speed", 1.0))
    twist_amplitude = float(params.get("twist_amplitude", 2.0))
    osc_spread = float(params.get("osc_spread", 1.5))

    seed_all(seed)
    rng = np.random.default_rng(seed)

    is_evolve = anim_mode != "none"
    if is_evolve and t > 0.01:
        n_frames = max(60, int(60 + t * anim_speed * 25))

    # ── Geometry ──
    base_scale = min(W, H) * 0.22
    cx, cy = W // 2, H // 2

    # ── Set global twist waveform params ──
    global _twist_wave_type, _twist_pulse_width
    _twist_wave_type = twist_wave

    # ── Precompute theta/phi grid (shared) ──
    theta = np.linspace(0, PI, N_THETA)
    phi = np.linspace(0, TAU, N_PHI)
    TH, PH = np.meshgrid(theta, phi, indexing='ij')
    TH_flat = TH.ravel()
    PH_flat = PH.ravel()

    # ── Precompute twist profile per θ ring (cumulative — twist accumulates top→bottom) ──
    theta_vals = np.linspace(0, PI, N_THETA)
    d_theta = 1.0 / N_THETA
    raw_profile = np.array([_wave(theta_norm * TAU)
                            for theta_norm in theta_vals / PI])
    # Cumulative sum: each ring's twist includes all rings above it
    # This guarantees bottom rings have most twist → visible shearing everywhere
    twist_profile = np.cumsum(raw_profile) * d_theta
    # Normalize so peak-to-peak spans roughly [-1, 1]
    twist_profile = twist_profile * 2.0 / max(np.ptp(twist_profile), 0.01)
    twist_profile_2d = twist_profile[:, np.newaxis]  # (N_THETA, 1) — broadcast over phi

    # ── Build orbital sequence ──
    orbital_sequence = [(l, m) for l in range(max_l + 1) for m in range(-l, l + 1)]

    # ── Precompute superposition basis (3 fixed harmonics for blending) ──
    super_mask = [(0, 0), (1, 0), (1, 1), (2, 0), (2, 2),
                  (3, 1), (3, 3), (4, 0), (4, 2), (4, 4)]
    super_basis = [sph_harm_y(l, m, TH_flat, PH_flat)
                   for l, m in super_mask[:min(len(super_mask), 5)]]

    img = None

    # ═══════════════════════════════════════════
    #  ANIMATION LOOP
    # ═══════════════════════════════════════════
    for frame in range(n_frames):
        progress = frame / max(1, n_frames - 1)  # 0→1
        _t = progress * TAU * anim_speed  # time in radians

        # ── 1. Compute Y values ──
        if anim_mode == "spin":
            # Static (l,m), phase rotation applied
            l0, m0 = max_l, max_l // 2  # interesting starting shape
            l0 = max(2, l0)
            m0 = max(1, l0 // 2)
            y = sph_harm_y(l0, m0, TH_flat, PH_flat)
            # Phase rotation: multiply by exp(i * _t)
            rot = complex(math.cos(_t), math.sin(_t))
            y = y * rot
            color_mode = "bipolar"

        elif anim_mode == "breathe":
            # Static (l,m), amplitude pulses
            l0, m0 = max_l, 0
            y = sph_harm_y(l0, m0, TH_flat, PH_flat)
            amp = 1.0 + 0.2 * math.sin(_t * 2.0)
            color_mode = "bipolar"

        elif anim_mode == "phase_color":
            # Static (l,m), colored by complex argument
            l0, m0 = max_l, max_l - 1
            y = sph_harm_y(l0, m0, TH_flat, PH_flat)
            # Slow phase drift to shift colors
            rot = complex(math.cos(_t * 0.3), math.sin(_t * 0.3))
            y = y * rot
            color_mode = "phase"

        elif anim_mode == "combined":
            # Morph + spin + breathe + twist + phase_color
            pos = progress * (len(orbital_sequence) - 1)
            idx_a = int(math.floor(pos))
            idx_b = min(idx_a + 1, len(orbital_sequence) - 1)
            blend = _smoothstep(pos - idx_a)
            l_a, m_a = orbital_sequence[idx_a]
            l_b, m_b = orbital_sequence[idx_b]
            y_a = sph_harm_y(l_a, m_a, TH_flat, PH_flat)
            y_b = sph_harm_y(l_b, m_b, TH_flat, PH_flat)
            y = (1 - blend) * y_a + blend * y_b
            # Phase rotation on blended value
            rot = complex(math.cos(_t * 0.5), math.sin(_t * 0.5))
            y = y * rot
            # Add twist — per-ring phase shift
            y_grid = y.reshape(N_THETA, N_PHI)
            phase_grid = twist_profile_2d * twist_amplitude * _t * 0.5
            y_grid = y_grid * np.exp(1j * phase_grid)
            y = y_grid.ravel()
            color_mode = "phase"

        elif anim_mode == "superposition":
            # Blend 3-5 basis harmonics with time-varying weights
            n_base = len(super_basis)
            weights = np.array([
                0.5 + 0.5 * math.sin(_t * 1.3 + i * 2.1)
                for i in range(n_base)
            ], dtype=np.float64)
            weights = np.maximum(weights, 0.0)
            weights /= weights.sum()
            y = sum(w * b for w, b in zip(weights, super_basis))
            # Gentle phase rotation
            rot = complex(math.cos(_t * 0.2), math.sin(_t * 0.2))
            y = y * rot
            color_mode = "phase"

        elif anim_mode == "twist":
            # Static (l,m), each θ ring spins independently
            l0, m0 = max_l, max(1, max_l - 1)
            y_base = sph_harm_y(l0, m0, TH_flat, PH_flat)
            y_grid = y_base.reshape(N_THETA, N_PHI)

            if twist_drive == "oscillator":
                # Each ring driven by an independent oscillator with its own frequency
                freqs = 1.0 + osc_spread * (np.arange(N_THETA) / N_THETA - 0.5)
                osc_phases = freqs * _t * twist_speed
                # Per-ring twist rate oscillates (large amplitude)
                twist_rates = twist_amplitude * np.sin(osc_phases)
                # Raw cumulative sum (no d_theta) — bottom rings get full accumulated twist
                twist_profile_dyn = np.cumsum(twist_rates)
                phase_grid = twist_profile_dyn[:, np.newaxis]
            else:
                # Spatial: waveform profile × time, scaled for strong shearing
                phase_grid = twist_profile_2d * twist_amplitude * _t * twist_speed * 5.0

            y_grid = y_grid * np.exp(1j * phase_grid)
            y = y_grid.ravel()
            color_mode = "phase"

        else:  # "morph" (default)
            pos = progress * (len(orbital_sequence) - 1)
            idx_a = int(math.floor(pos))
            idx_b = min(idx_a + 1, len(orbital_sequence) - 1)
            blend = _smoothstep(pos - idx_a)
            l_a, m_a = orbital_sequence[idx_a]
            l_b, m_b = orbital_sequence[idx_b]
            y_a = sph_harm_y(l_a, m_a, TH_flat, PH_flat)
            y_b = sph_harm_y(l_b, m_b, TH_flat, PH_flat)
            y = (1 - blend) * y_a + blend * y_b
            color_mode = "bipolar"

        # ── 2. Apply amplitude breathing (for modes that use it) ──
        if anim_mode == "breathe":
            amp_factor = 1.0 + 0.2 * math.sin(_t * 2.0) * anim_speed
        elif anim_mode == "combined":
            amp_factor = 1.0 + 0.15 * math.sin(_t * 1.5)
        else:
            amp_factor = 1.0

        effective_amp = amplitude * amp_factor

        # ── 3. Build surface ──
        pos_pts, neg_pts, pos_vals, neg_vals, pos_c, neg_c = \
            _make_surface(y, effective_amp)

        # ── 4. Camera orbit ──
        if anim_mode in ("spin", "phase_color", "breathe", "none"):
            cam_y = progress * TAU * 1.5
            cam_x = 0.4 + 0.3 * math.sin(progress * PI)
        elif anim_mode == "superposition":
            cam_y = _t * 0.5
            cam_x = 0.5 + 0.2 * math.sin(_t * 0.3)
        else:
            cam_y = progress * TAU * 1.5
            cam_x = 0.4 + 0.3 * math.sin(progress * PI)

        # ── 5. Render ──
        img = _render_orbital(pos_pts, neg_pts, pos_vals, neg_vals,
                              pos_c, neg_c,
                              cam_y, cam_x, base_scale * amplitude,
                              cx, cy, color_mode, glow_strength)

        if is_evolve:
            capture_frame("104", np.array(img, dtype=np.float32) / 255.0)

    if img is None:
        img = Image.new("RGB", (W, H), DARK_BG)
    capture_frame("104", np.array(img, dtype=np.float32) / 255.0)
    save(img, mn(104, "Spherical Harmonics"), out_dir)
    return img
