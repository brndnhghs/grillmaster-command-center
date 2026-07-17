from __future__ import annotations

import math
from pathlib import Path

import numpy as np
from scipy.ndimage import map_coordinates

from ...core.registry import method
from ...core.utils import (save, mn, seed_all, W, H, load_input)
from ...core.animation import capture_frame


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


def _uv_grid(w: int, h: int) -> tuple[np.ndarray, np.ndarray]:
    """Top-left origin uv grids in [0,1] (x right, y down)."""
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    return xx / (w - 1), yy / (h - 1)


def _make_scene(w: int, h: int, src: str, light_x: float, light_y: float,
                sun_r: float, sun_i: float, sun_col: tuple[float, float, float]
                ) -> np.ndarray:
    """Build the source buffer (scene). When no upstream image is wired we
    generate a backdrop; an additive bright sun disc at the light position
    guarantees a strong emitter so the radial blur produces real rays."""
    if src == "gradient":
        xx, yy = _uv_grid(w, h)
        cx, cy = w * 0.5, h * 0.5
        r = np.sqrt((xx * w - cx) ** 2 + (yy * h - cy) ** 2)
        r /= r.max()
        buf = np.stack([r, 1.0 - r, xx], -1).astype(np.float32)
    else:
        buf = np.zeros((h, w, 3), dtype=np.float32)

    if sun_r > 0.5 and sun_i > 0.0:
        xx, yy = _uv_grid(w, h)
        lx, ly = light_x, light_y
        d2 = (xx - lx) ** 2 + (yy - ly) ** 2
        disc = np.exp(-d2 / (2.0 * (sun_r / max(w, h)) ** 2))
        for c in range(3):
            buf[..., c] += disc * sun_col[c] * sun_i
    return np.clip(buf, 0.0, 1.0).astype(np.float32)


def _scatter_rays(buf: np.ndarray, light_x: float, light_y: float,
                  density: float, decay: float, weight: float,
                  exposure: float, n_samples: int) -> np.ndarray:
    """Volumetric Light Scattering (Mitchell & McGuire, GPU Gems 3 Ch.13).

    Radially blur `buf` along the vector from every pixel toward the light
    position, accumulating with an exponentially decaying weight. Bright
    emitters near the light bleed into the surrounding pixels as shafts."""
    h, w = buf.shape[:2]
    ux, uy = _uv_grid(w, h)
    uv = np.stack([ux, uy], -1)  # (H,W,2) top-left uv

    delta = (uv - np.array([light_x, light_y], np.float32)) * (density / n_samples)

    accum = np.zeros((h, w, 3), dtype=np.float32)
    illum = 1.0
    for k in range(1, n_samples + 1):
        coord = uv - delta * k  # (H,W,2), approaches the light as k grows
        px = coord[..., 0] * (w - 1)
        py = coord[..., 1] * (h - 1)
        coords = np.stack([py, px], 0)  # map_coordinates wants (row, col)
        # Process each colour channel separately (map_coordinates needs one
        # coordinate pair per spatial axis, not per channel).
        s = np.empty((h, w, 3), dtype=np.float32)
        for c in range(3):
            s[..., c] = map_coordinates(buf[..., c], coords, order=1,
                                       mode="constant", cval=0.0)
        s = s * (illum * weight)
        accum += s
        illum *= decay

    return np.clip(accum * exposure, 0.0, 1.0).astype(np.float32)


@method(
    id="446",
    name="God Rays (Filter)",
    category="filters",
    new_image_contract=True,
    tags=["post-process", "god-rays", "volumetric", "crepuscular", "light-scattering",
          "screen-space", "animation"],
    inputs={"image_in": "IMAGE"},
    outputs={"image": "IMAGE"},
    params={
        "source": {"description": "backdrop when no image is wired (black/gradient)",
                   "choices": ["black", "gradient"], "default": "black"},
        "light_x": {"description": "light/sun X position, normalised [0,1] (off-screen ok)",
                    "min": -0.5, "max": 1.5, "default": 0.30},
        "light_y": {"description": "light/sun Y position, normalised [0,1] (off-screen ok)",
                    "min": -0.5, "max": 1.5, "default": 0.28},
        "density": {"description": "ray length / distortion strength (higher = longer shafts)",
                    "min": 0.1, "max": 1.5, "default": 0.92},
        "decay": {"description": "per-sample illumination decay (0.8=short, 0.99=long trails)",
                  "min": 0.80, "max": 0.99, "default": 0.95},
        "weight": {"description": "per-sample contribution weight",
                   "min": 0.1, "max": 1.0, "default": 0.5},
        "exposure": {"description": "final ray exposure multiplier",
                     "min": 0.1, "max": 1.5, "default": 0.6},
        "sun_radius": {"description": "bright emitter disc radius in px (0 = no injected sun)",
                       "min": 0.0, "max": 120.0, "default": 36.0},
        "sun_intensity": {"description": "bright emitter brightness",
                          "min": 0.0, "max": 3.0, "default": 1.6},
        "tint": {"description": "ray/sun colour tint hue [0,1] (0=red,0.33=green,0.66=blue)",
                 "min": 0.0, "max": 1.0, "default": 0.58},
        "intensity": {"description": "overall god-ray add strength over the scene",
                      "min": 0.0, "max": 3.0, "default": 1.0},
        "num_samples": {"description": "radial blur sample count (quality vs cost)",
                        "min": 16, "max": 128, "default": 64},
        "anim_mode": {"description": "animation mode (none/orbit/pulse)",
                      "choices": ["none", "orbit", "pulse"], "default": "none"},
        "anim_speed": {"description": "animation speed multiplier",
                       "min": 0.1, "max": 5.0, "default": 1.0},
        "time": {"description": "animation phase [0, 2pi)",
                 "min": 0.0, "max": 6.28, "default": 0.0},
    },
)
def method_god_rays(out_dir: Path, seed: int, params=None):
    """Volumetric Light Scattering — crepuscular god-rays (Mitchell & McGuire 2008).

    A screen-space post-process that produces the classic shafts of light
    streaming around occluders. The scene is radially blurred toward a light
    position; bright emitters near that position bleed outward as soft beams.

    The algorithm (GPU Gems 3, Ch.13 "Volumetric Light Scattering as a
    Post-Process"):
        1. Build a light buffer (the wired image, or a generated backdrop with
           an injected bright sun disc at the light position).
        2. For each output pixel, walk N steps from the pixel toward the light,
           sampling the buffer and accumulating it under an exponentially
           decaying weight (`illum *= decay`).
        3. Additively composite the accumulated shafts over the original scene.

    When an upstream image is wired in it is used as the light buffer AND as the
    scene (Rule #12: the wired image overrides the generated backdrop). The CPU
    path is authoritative.

    Params:
        light_x/y:    light/sun position (normalised; may sit off-screen)
        density:      ray distortion/length (the classic `Density / NUM_SAMPLES`)
        decay:        per-sample illumination decay (classic 0.95)
        weight:       per-sample contribution
        exposure:     final shaft exposure
        sun_radius/i: injected bright emitter so there is always a light source
        tint:         ray/sun colour hue
        intensity:    overall additive strength over the scene
        num_samples:  radial blur quality
        anim_mode:    none / orbit (light circles centre) / pulse (brightness)
        time:         animation clock [0, 2pi)
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

        light_x = float(params.get("light_x", 0.30))
        light_y = float(params.get("light_y", 0.28))

        # ── Animation (use _t so we never shadow the time param) ──
        _t = anim_time * anim_speed
        if anim_mode == "orbit":
            ang = _t
            R = 0.38
            light_x = 0.5 + R * math.cos(ang)
            light_y = 0.5 + R * math.sin(ang)
        pulse = 1.0
        if anim_mode == "pulse":
            # Smooth oscillation (no abs(sin) cusp).
            pulse = 0.55 + 0.45 * (0.5 + 0.5 * math.sin(_t))

        density = float(params.get("density", 0.92))
        decay = float(params.get("decay", 0.95))
        weight = float(params.get("weight", 0.5))
        exposure = float(params.get("exposure", 0.6)) * pulse
        sun_radius = float(params.get("sun_radius", 36.0))
        sun_intensity = float(params.get("sun_intensity", 1.6)) * pulse
        # Seed-live emitter hue (subtle nudge so different seeds differ).
        tint = float(params.get("tint", 0.58))
        seed_hue = (tint + (rng.random() - 0.5) * 0.08) % 1.0
        sun_col = _hsv2rgb(seed_hue, 0.35, 1.0)
        intensity = float(params.get("intensity", 1.0))
        num_samples = int(params.get("num_samples", 64))
        num_samples = max(16, min(128, num_samples))

        # ── Resolve scene/light buffer (wired image overrides generated) ──
        wired_path = params.get("input_image", "")
        scene = None
        if wired_path:
            try:
                scene = load_input(wired_path, w, h)
            except (FileNotFoundError, OSError):
                scene = None
        if scene is None:
            src = str(params.get("source", "black"))
            buf = _make_scene(w, h, src, light_x, light_y,
                              sun_radius, sun_intensity, sun_col)
            scene = buf
        else:
            # Wired image is the scene; still inject a sun disc so rays have a source.
            buf = scene.copy()
            if sun_radius > 0.5 and sun_intensity > 0.0:
                xx, yy = _uv_grid(w, h)
                d2 = (xx - light_x) ** 2 + (yy - light_y) ** 2
                disc = np.exp(-d2 / (2.0 * (sun_radius / max(w, h)) ** 2))
                for c in range(3):
                    buf[..., c] += disc * sun_col[c] * sun_intensity
                buf = np.clip(buf, 0.0, 1.0).astype(np.float32)

        rays = _scatter_rays(buf, light_x, light_y, density, decay,
                             weight, exposure, num_samples)
        result = np.clip(scene + rays * intensity, 0.0, 1.0).astype(np.float32)

        capture_frame("446", result)
        save(result, mn(446, "God Rays"), out_dir)
        return result
    except Exception as exc:
        fallback = np.zeros((int(H), int(W), 3), dtype=np.float32)
        save(fallback, mn(446, "God Rays"), out_dir)
        print(f"[method_446] ERROR: {exc}")
        return fallback
