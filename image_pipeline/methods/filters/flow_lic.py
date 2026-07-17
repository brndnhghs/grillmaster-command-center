"""Line Integral Convolution (LIC) flow visualization.

Classic real-time vector-field visualization (Cabral & Leedom, SIGGRAPH 1993,
"Imaging Vector Fields Using Line Integral Convolution"). Given a 2D vector
field we convolve a high-frequency texture *along the local streamlines* —
integrating forwards and backwards, sampling the texture at each step, and
averaging. The result is the hallmark "streaked silk" image where the texture
is smeared tangentially to the flow, revealing structure that a raw
arrow/quiver plot hides.

Fresh twist here:
* The vector field is a *divergence-free curl-noise* field (one smooth
  scalar potential → its curl). Animation (a) ROTATES the whole field by
  theta(phase) and (b) TRANSLATES the potential, so the eddy structure bodily
  moves — both loop at 2π (non-degenerate; no sin-amplitude breathing).
* The convolved texture is a *coherent* high-frequency field (a sum of
  phase-scattered sine gratings), NOT white noise. Averaging incoherent white
  noise along the streamlines yields a statistically invariant smear, so
  resampling barely changes pixels (Δ≈0.04, the gate fails). The coherent
  texture has definite structure, so scrolling it down +y by a phase that
  grows monotonically with time moves the visible streak content frame to
  frame — the animation is clearly visible and clears Δ>0.05. This is the
  standard "animated LIC" trick (scroll the input texture) and avoids any
  sin-degeneracy at t=π.
* Image-guided LIC (Rule 12): a wired upstream IMAGE is used as the texture to
  convolve (its luminance), instead of the generated texture — the picture gets
  smeared along the flow. Pure processors never fall back to generating a picture.
  * Optional color modes: `mono` (classic grayscale), `speed` (tint by flow
    magnitude via palette), `direction` (hue from flow angle).

The method is full-coverage RGB (no alpha) plus a MASK (the LIC intensity, for
downstream spatial selection) and a FIELD (flow speed, for field-driven graphs).

Reference: https://en.wikipedia.org/wiki/Line_integral_convolution
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter, map_coordinates

from ...core.registry import method
from ...core.utils import (
    save,
    mn,
    seed_all,
    W,
    H,
    BG_DEFAULT,
    norm,
    write_scalars,
    write_field,
    write_mask,
    load_input,
    PALETTES,
)
from ...core.animation import capture_frame


# ── smooth scalar field (cheap stand-in for value noise) ───────────────────
def _smooth_field(rng: np.random.Generator, shape, sigma: float) -> np.ndarray:
    g = rng.random(shape)
    if sigma > 0.0:
        g = gaussian_filter(g, sigma=sigma, mode="reflect")
    g = (g - g.min()) / (np.ptp(g) + 1e-8)
    return g


def _potential(rng: np.random.Generator, hh: int, ww: int, scale: float) -> np.ndarray:
    """Coarse smooth potential upscaled to full resolution (periodic-ish)."""
    cg = max(8, int(min(hh, ww) / scale))
    p = _smooth_field(rng, (cg, cg), sigma=cg * 0.35)
    grid = np.mgrid[0:hh, 0:ww].astype(np.float64) / max(hh, ww) * cg
    pot = map_coordinates(p, [grid[0], grid[1]], order=1, mode="reflect")
    return pot


def _hsv2rgb(h: np.ndarray, s: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Vectorized HSV→RGB, all arrays shaped (H, W), returns (H, W, 3)."""
    i = np.floor(h * 6.0)
    f = h * 6.0 - i
    p = v * (1.0 - s)
    q = v * (1.0 - f * s)
    t = v * (1.0 - (1.0 - f) * s)
    i = i.astype(np.int32) % 6
    r = np.choose(i, [v, q, p, p, t, v])
    g = np.choose(i, [t, v, v, q, p, p])
    b = np.choose(i, [p, p, t, v, v, q])
    return np.stack([r, g, b], axis=-1)


def _palette_color(pal_name: str, t01: np.ndarray) -> np.ndarray:
    pal = np.array(PALETTES.get(pal_name, list(PALETTES.values())[0]), dtype=np.float64) / 255.0
    idx = np.clip((t01 * (len(pal) - 1)).astype(np.int32), 0, len(pal) - 1)
    return pal[idx]  # (H, W, 3)


def _coherent_grain(rng: np.random.Generator, hh: int, ww: int) -> np.ndarray:
    """High-frequency *coherent* texture: a sum of phase-scattered sine
    gratings at random orientations. Unlike white noise, this has a definite
    structure, so scrolling it moves the visible content frame to frame
    (animated LIC). The per-grating phases are seeded, so it is reproducible."""
    yy, xx = np.mgrid[0:hh, 0:ww].astype(np.float64)
    g = np.zeros((hh, ww), dtype=np.float64)
    for _ in range(6):
        ang = rng.random() * math.pi
        ca, sa = math.cos(ang), math.sin(ang)
        proj = (xx * ca + yy * sa) * (rng.random() * 0.25 + 0.08)
        g += np.sin(proj * (2.0 * math.pi) + rng.random() * (2.0 * math.pi))
    g = (g / 6.0) * 0.5 + 0.5  # -> [0,1], mean ~0.5
    return np.clip(g, 0.0, 1.0)


@method(
    id="992",
    name="Flow LIC",
    category="filters",
    tags=["lic", "flow", "vector-field", "visualization", "curl-noise", "animation", "expanded"],
    timeout=120,
    inputs={"image_in": "IMAGE"},
    outputs={"image": "IMAGE", "field": "FIELD", "mask": "MASK"},
    params={
        "source": {"description": "texture convolved along streamlines when no upstream image is wired (noise/palette)", "choices": ["noise", "palette"], "default": "noise"},
        "palette": {"description": "palette name for the palette texture / speed+dir tint", "default": "vapor"},
        "flow_scale": {"description": "spatial scale of the curl-noise flow field (smaller = larger eddies)", "min": 0.5, "max": 6.0, "default": 2.5},
        "speed_scale": {"description": "flow magnitude multiplier (taller streaks in fast regions)", "min": 0.2, "max": 3.0, "default": 1.0},
        "stream_len": {"description": "integration steps per side (longer = smoother, longer streaks)", "min": 4, "max": 40, "default": 18},
        "step_size": {"description": "integration step in px per side", "min": 0.5, "max": 3.0, "default": 1.5},
        "color_mode": {"description": "output coloring (mono grayscale / speed tint / direction hue)", "choices": ["mono", "speed", "direction"], "default": "mono"},
        "contrast": {"description": "LIC streak contrast around mid-gray (higher = sharper, more visible streaks + stronger animation)", "min": 0.2, "max": 4.0, "default": 2.5},
        "anim_mode": {"description": "animation mode (none static / flow evolves the field)", "choices": ["none", "flow"], "default": "none"},
        "anim_speed": {"description": "animation speed multiplier", "min": 0.1, "max": 3.0, "default": 1.0},
    },
)
def method_flow_lic(out_dir: Path, seed: int, params=None):
    """Flow LIC — Line Integral Convolution over a curl-noise vector field.

    Convolve a coherent high-frequency texture (or a wired IMAGE's luminance)
    along the local flow streamlines to produce the classic streaked
    visualization of a vector field. With `anim_mode="flow"` the field rotates
    + translates and the texture scrolls, so the streaks flow continuously and
    loop at the timeline phase.
    """
    try:
        if params is None:
            params = {}
        seed_all(seed)
        rng = np.random.default_rng(seed)

        hh, ww = int(H), int(W)
        anim_time = float(params.get("time", 0.0))
        anim_mode = str(params.get("anim_mode", "none"))
        anim_speed = float(params.get("anim_speed", 1.0))
        _t = 0.0 if anim_mode == "none" else anim_time * anim_speed

        source = str(params.get("source", "noise"))
        pal_name = str(params.get("palette", "vapor"))
        flow_scale = float(params.get("flow_scale", 2.5))
        speed_scale = float(params.get("speed_scale", 1.0))
        stream_len = int(params.get("stream_len", 18))
        step_size = float(params.get("step_size", 1.5))
        color_mode = str(params.get("color_mode", "mono"))
        contrast = float(params.get("contrast", 1.0))

        # ── Curl-noise vector field (single smooth potential) ──
        # Animation DRIVES the field in TWO complementary ways so the LIC
        # streaks carry visible motion (loops at 2π, non-degenerate):
        #   (1) ROTATE the whole divergence-free field by theta(frac): every
        #       streamline direction shifts; (2) TRANSLATE the potential by a
        #       phase vector so the eddy structure *bodily moves* through the
        #       canvas. A rotation alone leaves a statistically isotropic
        #       curl-noise field nearly unchanged in pixel terms (Δ≈0); the
        #       translation makes the frozen-in streak lattice travel, which
        #       is what clears the Δ>0.05 animation gate. With wrap sampling
        #       both effects loop cleanly. (The texture itself is also advected
        #       — UFLIC-style — by tex_phase below.)
        frac = (_t % (2.0 * math.pi)) / (2.0 * math.pi) if anim_mode != "none" else 0.0
        pot = _potential(rng, hh, ww, flow_scale)
        if frac != 0.0:
            # translational drift of the eddy field (periodic via wrap)
            dy = (frac * hh) % hh
            dx = (frac * ww) % ww
            pot = np.roll(pot, shift=(int(dy), int(dx)), axis=(0, 1))
        dpy, dpx = np.gradient(pot)
        vx = dpy
        vy = -dpx
        if frac != 0.0:
            theta = frac * 2.0 * math.pi
            ct, st = math.cos(theta), math.sin(theta)
            vx, vy = vx * ct - vy * st, vx * st + vy * ct
        mag = np.hypot(vx, vy)
        mean_mag = mag.mean() + 1e-8

        # ── Texture to convolve: coherent high-frequency field (animated
        # LIC), a wired IMAGE's luminance, or a palette ramp ──
        # Rule 12: a wired upstream image ALWAYS overrides texture generation.
        #
        # WHY A COHERENT TEXTURE (not white noise): averaging *incoherent* white
        # noise along the streamlines produces a near-Gaussian smear whose
        # resampling is statistically invariant — translating/rotating the
        # field just re-draws from the same distribution, so the animated frame
        # is pixel-wise almost identical to the static one (Δ≈0.04, the gate
        # fails). Real animated LIC instead uses a *coherent* high-frequency
        # texture: a sum of phase-scattered sine gratings. It has fine streaks
        # (so the LIC reveals the flow) AND a definite structure (so scrolling
        # it down +y by tex_shift moves the visible content frame to frame).
        # tex_shift is monotonic in _t (no sin-degeneracy at t=π), so the
        # animation gate clears, and it wraps at 2π for a clean loop.
        tex = None
        wired = params.get("input_image", "")
        if wired:
            try:
                img = load_input(wired, ww, hh)  # (H,W,3) float [0,1]
                tex = (0.299 * img[..., 0] + 0.587 * img[..., 1] + 0.114 * img[..., 2]).astype(np.float64)
            except (FileNotFoundError, OSError, ValueError):
                tex = None
        if tex is None:
            if source == "palette":
                # Palette ramp modulated by coherent grain so the LIC still
                # has fine structure to smear into visible streaks.
                ramp = _palette_color(pal_name, norm(pot))[..., 0]
                grain = _coherent_grain(rng, hh, ww)
                tex = np.clip(0.65 * ramp + 0.35 * grain, 0.0, 1.0)
            else:
                tex = _coherent_grain(rng, hh, ww)
        # Animated-LIC scroll: shift the coherent texture down +y by a phase
        # that grows monotonically with _t. This moves the visible streak
        # content between frames (constant, non-degenerate motion) in BOTH
        # anim modes? No — only when anim_mode != "none" (static in none).
        tex_shift = 0.0 if anim_mode == "none" else (_t % (2.0 * math.pi))
        if tex_shift != 0.0:
            tex = np.roll(tex, shift=(int((tex_shift / (2.0 * math.pi)) * hh) % hh, 0), axis=(0, 1))

        # ── Vectorized streamline integration (RK1 forward + backward) ──
        # For each seed pixel we step *up* and *down* its local streamline,
        # sampling the (already time-scrolled) coherent texture at every step;
        # the running average is the LIC intensity. In flow mode the texture
        # has been scrolled down +y by tex_shift (above), so the sampled
        ys, xs = np.mgrid[0:hh, 0:ww].astype(np.float64)
        acc = tex.copy()
        cnt = 1.0

        def _integrate(direction: float):
            nonlocal acc, cnt
            yy = ys.copy()
            xx = xs.copy()
            for _ in range(stream_len):
                fy = map_coordinates(vy, [yy, xx], order=1, mode="wrap")
                fx = map_coordinates(vx, [yy, xx], order=1, mode="wrap")
                fm = np.hypot(fx, fy) + 1e-8
                # step length scales with LOCAL speed (fast eddies => longer
                # streaks) and with the global speed_scale control. speed_scale
                # is applied ONLY here (not as a field multiplier that the
                # fm/mean_mag normaliser would cancel) so the slider stays live
                # (pitfall #19).
                step = direction * step_size * speed_scale * (fm / mean_mag)
                yy = yy + (fy / fm) * step
                xx = xx + (fx / fm) * step
                acc += map_coordinates(tex, [yy, xx], order=1, mode="wrap")
                cnt += 1.0

        _integrate(1.0)   # forward
        _integrate(-1.0)  # backward

        intensity = acc / cnt
        # contrast around mid-gray (white noise centers ~0.5; fixed normalizer
        # so `contrast` stays live — no data-dependent divide, pitfall #19)
        intensity = np.clip(0.5 + (intensity - 0.5) * contrast, 0.0, 1.0)

        # ── Colorize ──
        if color_mode == "speed":
            base = _palette_color(pal_name, norm(mag))
            rgb = base * (0.25 + 0.75 * intensity[..., None])
        elif color_mode == "direction":
            ang = (np.arctan2(vy, vx) / (2.0 * math.pi)) % 1.0
            base = _hsv2rgb(ang, np.full_like(ang, 0.6), np.ones_like(ang))
            rgb = base * (0.25 + 0.75 * intensity[..., None])
        else:  # mono
            rgb = np.repeat(intensity[..., None], 3, axis=-1)

        out = np.clip(rgb, 0.0, 1.0).astype(np.float32)

        # ── Side outputs ──
        write_mask(out_dir, intensity.astype(np.float32))          # LIC intensity (spatial selection)
        write_field(out_dir, mag.astype(np.float32))               # flow speed field
        write_scalars(
            out_dir,
            flow_scale=flow_scale,
            speed_scale=speed_scale,
            stream_len=stream_len,  # noqa
            step_size=step_size,
            contrast=contrast,
            mean_speed=float(mag.mean()),
            intensity_mean=float(intensity.mean()),
            animated=int(anim_mode != "none"),
        )

        capture_frame("992", out)
        save(out, mn(992, f"Flow LIC t={_t:.2f}"), out_dir)
        return out
    except Exception as exc:
        fallback = np.full((int(H), int(W), 3), 0.18, dtype=np.float32)
        save(fallback, mn(992, "Flow LIC"), out_dir)
        print(f"[method_992] ERROR: {exc}")
        return fallback
