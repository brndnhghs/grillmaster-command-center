from __future__ import annotations

import math
from pathlib import Path

import numpy as np

from ...core.registry import method
from ...core.utils import (save, mn, seed_all, W, H, load_input)
from ...core.animation import capture_frame
from image_pipeline.core.spatial import sparam


# ── HSV -> RGB (full saturation/value helpers for ghost colour ramps) ──
def _hsv2rgb(h: float, s: float, v: float) -> tuple[float, float, float]:
    i = int(h * 6.0) % 6
    f = h * 6.0 - i
    p = v * (1.0 - s)
    q = v * (1.0 - f * s)
    t = v * (1.0 - (1.0 - f) * s)
    return {
        0: (v, t, p), 1: (q, v, p), 2: (p, v, t),
        3: (p, q, v), 4: (t, p, v), 5: (v, p, q),
    }[i]


def _add_disc(canvas: np.ndarray, xx: np.ndarray, yy: np.ndarray,
              x: float, y: float, r: float, color: tuple[float, float, float],
              chroma: float) -> None:
    """Additively splat a soft Gaussian disc with a per-channel chromatic offset."""
    d2 = (xx - x) ** 2 + (yy - y) ** 2
    for c, off in enumerate((-chroma, 0.0, chroma)):
        rr = max(1.0, r * (1.0 + off))
        g = np.exp(-d2 / (2.0 * rr * rr))
        canvas[..., c] += g * color[c]


def _render_flare(xx: np.ndarray, yy: np.ndarray, w: int, h: int,
                  lx: float, ly: float, rng: np.random.Generator,
                  num_ghosts: int, ghost_start: float, ghost_end: float,
                  ghost_size: float, halo_radius: float, halo_thickness: float,
                  halo_intensity: float, streak_len: float, streak_wid: float,
                  streak_intensity: float, chroma: float, hue_shift: float,
                  intensity: float) -> np.ndarray:
    """Build the additive flare (ghosts + halo + anamorphic streak) on black."""
    canvas = np.zeros((h, w, 3), dtype=np.float32)
    cx, cy = w / 2.0, h / 2.0
    vx, vy = cx - lx, cy - ly  # vector from light through screen centre

    diag = math.hypot(w, h)
    hue_base = rng.random()  # seed-driven global colour shift -> seed is live
    for i in range(max(1, num_ghosts)):
        frac = i / max(1, num_ghosts - 1)
        wt = ghost_start + frac * (ghost_end - ghost_start)
        # Per-ghost axial scatter (seed-driven) so different seeds differ visibly.
        scatter = (rng.random() - 0.5) * 0.12
        gx = lx + vx * (wt + scatter)
        gy = ly + vy * (wt + scatter)
        jr = 0.75 + 0.5 * rng.random()
        jh = (rng.random() - 0.5) * 0.12
        r = max(4.0, ghost_size * (0.35 + 0.65 * abs(wt)) * jr)
        hue = (frac * 0.8 + hue_shift + hue_base + jh) % 1.0
        col = _hsv2rgb(hue, 0.85, 1.0)
        b = 1.0 - abs(wt) * 0.35  # ghosts fade toward the extremes
        _add_disc(canvas, xx, yy, gx, gy, r,
                  (col[0] * b, col[1] * b, col[2] * b), chroma)

    # Halo: soft ring along the light->centre axis.
    hx, hy = cx + vx * 0.7, cy + vy * 0.7
    d = np.sqrt((xx - hx) ** 2 + (yy - hy) ** 2)
    R = max(2.0, halo_radius * diag)
    thick = max(1.0, halo_thickness * diag)
    ring = np.exp(-((d - R) ** 2) / (2.0 * thick * thick))
    hc = _hsv2rgb((0.55 + hue_shift) % 1.0, 0.7, 1.0)
    canvas[..., 0] += ring * hc[0] * halo_intensity
    canvas[..., 1] += ring * hc[1] * halo_intensity
    canvas[..., 2] += ring * hc[2] * halo_intensity

    # Anamorphic streak: horizontal bar through the light (cinematic blue/cyan).
    hx_ = xx - lx
    hy_ = yy - ly
    sl = max(2.0, streak_len * w * 0.5)
    sw = max(1.0, streak_wid * h)
    streak = np.exp(-(hx_ ** 2) / (2.0 * sl * sl)) * np.exp(-(hy_ ** 2) / (2.0 * sw * sw))
    sc = _hsv2rgb((0.58 + hue_shift) % 1.0, 0.6, 1.0)
    canvas[..., 0] += streak * sc[0] * streak_intensity
    canvas[..., 1] += streak * sc[1] * streak_intensity
    canvas[..., 2] += streak * sc[2] * streak_intensity

    return np.clip(canvas * intensity, 0.0, 1.0).astype(np.float32)


@method(
    id="434",
    name="Lens Flare",
    category="filters",
    new_image_contract=True,
    tags=["post-process", "flare", "lens", "anamorphic", "cinematic", "ghosts", "halo", "animation"],
    inputs={"image_in": "IMAGE"},
    outputs={"image": "IMAGE"},
    params={
        "source": {"description": "background when no image is wired (black/gradient)", "choices": ["black", "gradient"], "default": "black"},
        "light_x": {"spatial": True, "description": "light source X position, normalised [0,1] (off-screen <0 or >1 ok)", "min": -0.5, "max": 1.5, "default": 0.28},
        "light_y": {"spatial": True, "description": "light source Y position, normalised [0,1]", "min": -0.5, "max": 1.5, "default": 0.32},
        "num_ghosts": {"description": "number of reflected ghosts along the light->centre axis", "min": 1, "max": 14, "default": 8},
        "ghost_start": {"description": "near weight (negative = before the light)", "min": -1.5, "max": 0.5, "default": -0.6},
        "ghost_end": {"description": "far weight (past the centre)", "min": 0.5, "max": 2.5, "default": 1.4},
        "ghost_size": {"description": "base ghost radius in px", "min": 8, "max": 160, "default": 60},
        "halo_radius": {"description": "halo ring radius as fraction of the diagonal", "min": 0.05, "max": 1.0, "default": 0.45},
        "halo_thickness": {"description": "halo ring softness as fraction of the diagonal", "min": 0.01, "max": 0.4, "default": 0.06},
        "halo_intensity": {"description": "halo brightness", "min": 0.0, "max": 2.0, "default": 0.8},
        "streak_len": {"description": "anamorphic streak half-length as fraction of width", "min": 0.0, "max": 1.5, "default": 0.8},
        "streak_wid": {"description": "anamorphic streak vertical width as fraction of height", "min": 0.002, "max": 0.1, "default": 0.02},
        "streak_intensity": {"description": "anamorphic streak brightness", "min": 0.0, "max": 2.0, "default": 0.7},
        "chroma": {"description": "per-channel chromatic split of the ghosts (0=no fringing)", "min": 0.0, "max": 0.5, "default": 0.15},
        "hue_shift": {"description": "global hue rotation of the flare colours", "min": 0.0, "max": 1.0, "default": 0.0},
        "intensity": {"description": "overall flare brightness multiplier", "min": 0.0, "max": 3.0, "default": 1.0},
        "anim_mode": {"description": "animation mode (none/orbit/pulse)", "choices": ["none", "orbit", "pulse"], "default": "none"},
        "anim_speed": {"description": "animation speed multiplier", "min": 0.1, "max": 5.0, "default": 1.0},
        "time": {"description": "animation phase [0, 2pi)", "min": 0.0, "max": 6.28, "default": 0.0},
    },
)
def method_lens_flare(out_dir: Path, seed: int, params=None):
    """Cinematic lens flare — ghosts, halo ring, and an anamorphic streak.

    A real-time lens flare is the classic screen-space post-effect popularised
    in game engines (e.g. Unity's "Procedural Lens Flare" / the GPU Gems
    pseudo-flare). The flare is built additively from three parts:

        1. **ghosts** — soft discs reflected along the axis from the light
           through the screen centre, each tinted by a position-dependent hue
           (Spectral-Gaussian ghost ramp), with a per-channel chromatic split;
        2. **halo**  — a soft coloured ring at a fixed fraction along that axis;
        3. **anamorphic streak** — a wide, thin horizontal bar through the light
           (the signature JJ-Abrams blue streak).

    When an upstream image is wired in it is composited *under* the additive
    flare (Rule #12: the wired image always overrides the generated
    background). The CPU path is authoritative.

    Params:
        light_x/y:   light position (normalised; may sit off-screen)
        num_ghosts:  ghost count (1-14)
        ghost_size:  base ghost radius px
        halo_*:      ring radius / softness / brightness
        streak_*:    anamorphic bar length / width / brightness
        chroma:      RGB fringe amount on ghosts
        hue_shift:   global colour rotation
        intensity:   overall multiplier
        anim_mode:   none / orbit (light circles centre) / pulse (brightness)
        time:        animation clock [0, 2pi)
    """
    try:
        if params is None:
            params = {}
        anim_time = float(params.get("time", 0.0))
        anim_mode = str(params.get("anim_mode", "none"))
        anim_speed = float(params.get("anim_speed", 1.0))
        seed_all(seed)
        rng = np.random.default_rng(seed)

        w, h = int(W), int(H)
        yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)

        light_x = sparam(params, "light_x", 0.28)
        light_y = sparam(params, "light_y", 0.32)

        # ── Animation (rename t so we never shadow the time param) ──
        _t = anim_time * anim_speed
        if anim_mode == "orbit":
            ang = _t
            R = 0.35
            light_x = 0.5 + R * math.cos(ang)
            light_y = 0.5 + R * math.sin(ang)
        pulse = 1.0
        if anim_mode == "pulse":
            pulse = 0.55 + 0.45 * (0.5 + 0.5 * math.sin(_t))

        lx, ly = light_x * w, light_y * h

        num_ghosts = int(params.get("num_ghosts", 8))
        ghost_start = float(params.get("ghost_start", -0.6))
        ghost_end = float(params.get("ghost_end", 1.4))
        ghost_size = float(params.get("ghost_size", 60))
        halo_radius = float(params.get("halo_radius", 0.45))
        halo_thickness = float(params.get("halo_thickness", 0.06))
        halo_intensity = float(params.get("halo_intensity", 0.8)) * pulse
        streak_len = float(params.get("streak_len", 0.8))
        streak_wid = float(params.get("streak_wid", 0.02))
        streak_intensity = float(params.get("streak_intensity", 0.7)) * pulse
        chroma = float(params.get("chroma", 0.15))
        hue_shift = float(params.get("hue_shift", 0.0))
        intensity = float(params.get("intensity", 1.0))

        flare = _render_flare(
            xx, yy, w, h, lx, ly, rng, num_ghosts, ghost_start, ghost_end,
            ghost_size, halo_radius, halo_thickness, halo_intensity,
            streak_len, streak_wid, streak_intensity, chroma, hue_shift, intensity,
        )

        # ── Resolve background (wired image overrides generated source) ──
        bg = None
        wired_path = params.get("input_image", "")
        if wired_path:
            try:
                bg = load_input(wired_path, w, h)
            except (FileNotFoundError, OSError):
                bg = None
        if bg is None:
            src = str(params.get("source", "black"))
            if src == "gradient":
                r = np.sqrt((xx - w / 2) ** 2 + (yy - h / 2) ** 2)
                r = r / r.max()
                bg = np.stack([r, 1 - r, (xx / w)], -1).astype(np.float32)
            else:
                bg = np.zeros((h, w, 3), dtype=np.float32)

        result = np.clip(bg + flare, 0.0, 1.0).astype(np.float32)

        capture_frame("434", result)
        save(result, mn(434, "Lens Flare"), out_dir)
        return result
    except Exception as exc:
        fallback = np.zeros((int(H), int(W), 3), dtype=np.float32)
        save(fallback, mn(434, "Lens Flare"), out_dir)
        print(f"[method_434] ERROR: {exc}")
        return fallback
