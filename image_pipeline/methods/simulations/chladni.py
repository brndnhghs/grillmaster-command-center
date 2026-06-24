"""#125 — Chladni Eigenmode Morphing

Continuous morphing between standing-wave eigenmodes of a vibrating 2D plate.
Each mode shape u_mn = sin(m·π·x/L) · sin(n·π·y/L) produces characteristic
nodal patterns (Chladni figures). Animation sweeps through (m,n) mode space,
rotates the plate, and breathes the modal amplitude.

Chladni figures are the nodal curves u(x,y)=0 of plate eigenfunctions.
By smoothly interpolating between modes and adding phase/rotation modulation,
the nodal structure breathes and morphs continuously — from simple single-line
nodes through increasingly intricate mandala-like geometric patterns.

Rendering: signed displacement field mapped to grayscale with sharp nodal
interfaces. Nodal lines (zero crossings) are emphasized as bright highlights.
Pipeline applies --recolor for palette coloring.

Architecture A — internal animation loop with capture_frame().
"""
from __future__ import annotations
import math
from pathlib import Path

import numpy as np
from PIL import Image

from ...core.registry import method
from ...core.utils import save, mn, seed_all, W, H, write_field
from ...core.animation import capture_frame


# ── Constants ──

N_FRAMES = 180
TAU = 2.0 * math.pi


# ── Chladni mode computation ──


def _chladni_mode(X: np.ndarray, Y: np.ndarray,
                  m: float, n: float,
                  phase_x: float = 0.0,
                  phase_y: float = 0.0,
                  rotation: float = 0.0) -> np.ndarray:
    """Compute a single 2D standing-wave eigenmode.

    u_mn(x,y) = sin(m·π·x̂ + φx) · sin(n·π·ŷ + φy)

    where x̂, ŷ are normalized to [-1, 1] and optionally rotated.

    Args:
        X, Y: Centered coordinate grids
        m, n: Mode numbers (fractional OK for morphing)
        phase_x, phase_y: Phase offsets for shimmer/sweep effects
        rotation: CCW rotation of the coordinate axes

    Returns:
        Displacement field (H, W)
    """
    # Normalize to [-1, 1] so modes cover the full canvas
    xn = X / (W / 2.0)
    yn = Y / (H / 2.0)

    # Apply rotation
    if abs(rotation) > 1e-8:
        c = math.cos(rotation)
        s = math.sin(rotation)
        xr = c * xn - s * yn
        yr = s * xn + c * yn
    else:
        xr, yr = xn, yn

    # Clamp to valid range to avoid wrap artifacts
    xr = np.clip(xr, -1.0, 1.0)
    yr = np.clip(yr, -1.0, 1.0)

    # Shift from [-1, 1] to [0, 2π] for the sine argument
    # This maps the plate edge to nodes (sin(0) = 0 at edge)
    u = np.sin(m * math.pi * (xr + 1.0) / 2.0 + phase_x) * \
        np.sin(n * math.pi * (yr + 1.0) / 2.0 + phase_y)

    return u


def _blend_modes(X: np.ndarray, Y: np.ndarray,
                 mode_params: list[tuple[float, float, float]],
                 rotation: float = 0.0) -> np.ndarray:
    """Blend multiple eigenmodes with weights.

    Args:
        mode_params: List of (m, n, weight) tuples
        rotation: Coordinate rotation angle

    Returns:
        Blended displacement field (H, W)
    """
    result = np.zeros_like(X, dtype=np.float64)
    for m, n, w in mode_params:
        result += w * _chladni_mode(X, Y, m, n, rotation=rotation)
    return result


def _render_chladni(field: np.ndarray, gain: float = 3.5, glow: float = 50.0) -> np.ndarray:
    """Render displacement field as uint8 grayscale.

    Centers the field (subtracts mean), normalizes to [-1, 1], then maps
    through a sharpened sigmoid that emphasizes zero crossing (nodal lines).
    Nodal lines get a gaussian-bell bright overlay for crisp interface highlighting.
    """
    # Center the field so both positive and negative displacements are visible
    f = field - np.mean(field)
    max_abs = max(abs(f.min()), abs(f.max()), 1e-10)
    f = f / max_abs

    # Sharp sigmoid: high gain for crisp transitions at nodes
    sigmoid = np.tanh(f * gain)

    # Map [-1, 1] → [0, 255]
    gray = ((sigmoid + 1.0) * 127.5).astype(np.float64)

    # Emphasize nodal lines: gaussian bell centered at f=0
    nodal = np.exp(-f * f * 8.0) * glow
    gray = np.clip(gray + nodal, 0, 255)

    return np.stack([gray] * 3, axis=-1).astype(np.uint8)


def _mode_sequence(n_frames: int,
                   m_start: float, n_start: float,
                   m_end: float, n_end: float,
                   loop_back: bool = True) -> list[tuple[float, float]]:
    """Smooth mode sweep with smoothstep easing.

    When loop_back=True, the sequence returns from end to start
    so the animation can cycle seamlessly.
    """
    seq = []
    total = n_frames * 2 if loop_back else n_frames
    for i in range(n_frames):
        t = i / max(n_frames - 1, 1)
        t_smooth = t * t * (3.0 - 2.0 * t)  # smoothstep
        if loop_back:
            # Go forward then backward (ping-pong)
            t_ping = t_smooth * 2.0
            if t_ping < 1.0:
                tp = t_ping
            else:
                tp = 2.0 - t_ping
            m = m_start + tp * (m_end - m_start)
            n = n_start + tp * (n_end - n_start)
        else:
            m = m_start + t_smooth * (m_end - m_start)
            n = n_start + t_smooth * (n_end - n_start)
        seq.append((m, n))
    return seq


# ── Method ──


@method(
    id="125",
    name="Chladni Eigenmode Morphing",
    category="simulations",
    tags=["simulation", "animation", "waves", "standing-waves", "modal",
          "geometric", "mandala", "nodal"],
    timeout=120,
    outputs={"image": "IMAGE", "luminance": "SCALAR", "field": "FIELD"},
    params={
        "n_frames": {
            "description": "number of animation frames",
            "min": 30, "max": 400, "default": 180},
        "m_start": {
            "description": "starting mode number X",
            "min": 0.5, "max": 12.0, "default": 1.0},
        "n_start": {
            "description": "starting mode number Y",
            "min": 0.5, "max": 12.0, "default": 1.0},
        "m_end": {
            "description": "ending mode number X",
            "min": 0.5, "max": 12.0, "default": 7.0},
        "n_end": {
            "description": "ending mode number Y",
            "min": 0.5, "max": 12.0, "default": 7.0},
        "phase_speed_x": {
            "description": "phase drift speed X (shimmer effect)",
            "min": 0.0, "max": 5.0, "default": 0.0},
        "phase_speed_y": {
            "description": "phase drift speed Y (shimmer effect)",
            "min": 0.0, "max": 5.0, "default": 0.0},
        "rotation_speed": {
            "description": "plate rotation speed (revs per full animation)",
            "min": 0.0, "max": 3.0, "default": 0.0},
        "breathe_amp": {
            "description": "amplitude breathing modulation (0=none)",
            "min": 0.0, "max": 1.0, "default": 0.0},
        "n_modes": {
            "description": "number of simultaneous modes to blend",
            "min": 1, "max": 5, "default": 1},
        "sigmoid_gain": {
            "description": "sigmoid sharpness for nodal contrast",
            "min": 1.0, "max": 8.0, "default": 3.5},
        "nodal_glow": {
            "description": "nodal line highlight brightness",
            "min": 0, "max": 120, "default": 50},
        "anim_mode": {
            "description": "animation mode",
            "choices": ["none", "morph", "spin", "breathe", "combined"],
            "default": "morph"},
        "anim_speed": {
            "description": "animation speed multiplier",
            "min": 0.1, "max": 5.0, "default": 1.0},
    }
)
def method_chladni(out_dir: Path, seed: int, params=None):
    """Chladni Eigenmode Morphing.

    Animates the nodal patterns of a vibrating 2D plate by sweeping
    through the eigenmode spectrum. Modes are standing-wave solutions
    to the Helmholtz equation: u_mn = sin(m·π·x̂)·sin(n·π·ŷ).

    Animation modes:
        none: Static single mode at (m_start, n_start)
        morph: Smooth ping-pong sweep from start to end mode numbers
        spin: Fixed mode with phase drift + coordinate rotation
        breathe: Amplitude pulsing without mode change
        combined: Morph + phase drift + rotation + breathing all together

    Output is pure grayscale displacement — pipeline applies --recolor.
    Nodal lines (zero crossings) are bright in grayscale for crisp
    interface coloring.

    Architecture A — internal animation loop with capture_frame().
    """
    # ── Params ──
    if params is None:
        params = {}
    anim_mode = str(params.get("anim_mode", "morph"))
    anim_speed = float(params.get("anim_speed", 1.0))

    n_frames = int(params.get("n_frames", N_FRAMES))
    m_start = float(params.get("m_start", 1.0))
    n_start = float(params.get("n_start", 1.0))
    m_end = float(params.get("m_end", 7.0))
    n_end = float(params.get("n_end", 7.0))
    phase_speed_x = float(params.get("phase_speed_x", 0.0))
    phase_speed_y = float(params.get("phase_speed_y", 0.0))
    rotation_speed = float(params.get("rotation_speed", 0.0))
    breathe_amp = float(params.get("breathe_amp", 0.0))
    n_modes = int(params.get("n_modes", 1))
    sigmoid_gain = float(params.get("sigmoid_gain", 3.5))
    nodal_glow = float(params.get("nodal_glow", 50.0))

    seed_all(seed)

    # ── Static grid ──
    yy, xx = np.mgrid[:H, :W]
    X = xx - W / 2.0
    Y = yy - H / 2.0

    is_evolve = anim_mode != "none"

    # Pre-compute morph sequence
    if anim_mode in ("morph", "combined"):
        mode_seq = _mode_sequence(n_frames, m_start, n_start, m_end, n_end, loop_back=True)
    else:
        mode_seq = [(m_start, n_start)] * n_frames

    # ══════════════════════════════════════════
    #  ANIMATION LOOP
    # ══════════════════════════════════════════

    last_img: np.ndarray | None = None
    last_field: np.ndarray | None = None

    for frame in range(n_frames):
        progress = frame / max(n_frames - 1, 1)
        _t = progress * TAU * anim_speed

        # ── Resolve mode params for this frame ──
        if anim_mode == "morph":
            m, n = mode_seq[frame]
            px = py = 0.0
            rot = 0.0
        elif anim_mode == "spin":
            m, n = m_start, n_start
            px = _t * phase_speed_x
            py = _t * phase_speed_y
            rot = _t * rotation_speed
        elif anim_mode == "breathe":
            m, n = m_start, n_start
            px = py = 0.0
            rot = 0.0
        elif anim_mode == "combined":
            m, n = mode_seq[frame]
            px = _t * phase_speed_x
            py = _t * phase_speed_y
            rot = _t * rotation_speed
        else:  # none
            m, n = m_start, n_start
            px = py = 0.0
            rot = 0.0

        # ── Compute field ──
        if n_modes <= 1:
            field = _chladni_mode(X, Y, m, n, phase_x=px, phase_y=py, rotation=rot)
        else:
            # Blend multiple adjacent modes for richer nodal structure
            mode_params = []
            for i in range(n_modes):
                mi = m + i * 0.4
                ni = n + i * 0.4
                w = 1.0 / (i + 1)
                f_i = _chladni_mode(X, Y, mi, ni, phase_x=px, phase_y=py, rotation=rot)
                mode_params.append(f_i * w)
            field = sum(mode_params)

        # ── Breathing ──
        if breathe_amp > 0 and anim_mode in ("breathe", "combined"):
            breathe = 1.0 + breathe_amp * 0.3 * math.sin(_t * 1.5 + math.sin(_t * 0.7))
            field *= breathe

        # ── Render ──
        img = _render_chladni(field, gain=sigmoid_gain, glow=nodal_glow)
        last_img = img
        last_field = field

        # ── Save + Capture ──
        if is_evolve:
            capture_frame("125", img.astype(np.float32) / 255.0)
            # Save every 30th frame + final frame for disk previews
            if frame % 30 == 0 or frame == n_frames - 1 or frame == 0:
                save(Image.fromarray(img, mode="RGB"),
                     mn(125, f"Chladni f{frame:04d}"),
                     out_dir)
        else:
            save(Image.fromarray(img, mode="RGB"), mn(125, "Chladni"), out_dir)

    # Return final frame
    if last_field is not None:
        write_field(out_dir, last_field.astype(np.float32))
    if last_img is not None:
        return Image.fromarray(last_img, mode="RGB")
    return Image.new("RGB", (W, H), (128, 128, 128))
