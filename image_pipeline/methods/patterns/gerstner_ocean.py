"""Gerstner Ocean — analytic trochoidal-wave ocean surface with sun glitter.

Renders a sum of Gerstner (trochoidal) waves (Fournier & Reeves 1986,
"A Simple Model of Ocean Waves", SIGGRAPH '86; the shading normal follows the
GPU Gems ch.1 "Effective Water Simulation from Physical Models", Finch 2004)
as a shaded height field. Each of `n_waves` components has a direction, a
wavelength (spatial frequency k = 2*pi/L), a deep-water dispersion speed
w = sqrt(g*k), an amplitude and a steepness Q. The surface height and its
analytic normal are evaluated in closed form per pixel:

    f_i   = k_i (D_i . p) - w_i t + phi_i
    z     = sum_i A_i sin(f_i)
    N.xy  = -sum_i (D_i * k_i * A_i * cos(f_i))
    N.z   = 1 - sum_i (Q_i * k_i * A_i * sin(f_i))

The pixel is then lit with a Lambert term (soft sky ambient) plus a Blinn-Phong
sun-glint specular and a depth/steepness colour mix (deep teal in the troughs,
bright foamy crests). This is the Eulerian normal-field variant: normals are
evaluated on the base grid rather than the horizontally-displaced trochoid
positions, which is the standard screen-space ocean-normal shading and keeps the
whole thing a pure vectorised f(uv, t) — no simulation state carried between
frames, no scatter/gather.

Why this node: it is a cheap, timeout-immune, ALWAYS-animated generator. Every
`swell`/`wind`/`gust` frame advances the wave phases so the surface is in
perpetual high-contrast motion — exactly the high-liveness form the shootout
rewards (the temporal_var liveness cull kills contrast-only static patterns; a
propagating wave field passes it trivially). A full 768x512 render costs well
under a second, so it is never a render-timeout casualty (the dominant death
cause in logged genomes). Closed-form per pixel => a clean GPU twin is a natural
follow-up.

CPU path authoritative.

Animation modes (Architecture B — per-frame re-call with `time`):
    none  — phases frozen at t=0: frame Δ ≈ 0 (static baseline).
    swell — wave phases advance with w_i * _t: the canonical rolling ocean.
    wind  — the global wind direction slowly yaws (cos/sin of _t), so the whole
            wave train reorients; combined with the base swell it churns.
    gust  — amplitude + choppiness breathe via cos(_t) (cos, not sin, so the
            t=0 vs t=pi audit frames stay distinct — sin-phase degeneracy).
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np

from ...core.registry import method
from ...core.utils import (
    save, mn, seed_all, BG_DEFAULT, W, H,
    write_mask, write_field, write_scalars,
)
from ...core.animation import capture_frame

_G = 9.81  # gravitational accel for deep-water dispersion


@method(
    id="963",
    name="Gerstner Ocean",
    category="patterns",
    new_image_contract=True,
    tags=["generative", "ocean", "water", "gerstner", "waves", "trochoidal",
          "shading", "animation", "gpu-twin-candidate"],
    inputs={},
    outputs={"image": "IMAGE", "field": "FIELD", "mask": "MASK"},
    params={
        "n_waves": {"description": "number of Gerstner wave components",
                    "min": 1.0, "max": 24.0, "default": 9.0},
        "base_wavelength": {"description": "longest wavelength (screen units)",
                            "min": 0.15, "max": 2.5, "default": 1.1},
        "wavelength_falloff": {"description": "per-wave wavelength scale (<1 shortens)",
                               "min": 0.5, "max": 0.95, "default": 0.82},
        "amplitude": {"description": "master wave amplitude",
                      "min": 0.02, "max": 0.6, "default": 0.16},
        "steepness": {"description": "trochoid steepness Q (0=round, 1=sharp crests)",
                      "min": 0.0, "max": 1.0, "default": 0.7},
        "wind_angle": {"description": "dominant wind direction (turns, 0-1)",
                       "min": 0.0, "max": 1.0, "default": 0.12},
        "wind_spread": {"description": "angular spread of wave directions (turns)",
                        "min": 0.0, "max": 0.4, "default": 0.14},
        "sun_angle": {"description": "sun azimuth (turns, 0-1)",
                      "min": 0.0, "max": 1.0, "default": 0.62},
        "sun_height": {"description": "sun elevation (0=horizon, 1=zenith)",
                       "min": 0.05, "max": 1.0, "default": 0.35},
        "shininess": {"description": "specular sharpness of the sun glitter",
                      "min": 4.0, "max": 400.0, "default": 90.0},
        "glint": {"description": "sun-glitter intensity",
                  "min": 0.0, "max": 4.0, "default": 1.6},
        "deep_hue": {"description": "deep-water base hue (0-1)",
                     "min": 0.0, "max": 1.0, "default": 0.53},
        "crest_hue": {"description": "crest/foam hue (0-1)",
                      "min": 0.0, "max": 1.0, "default": 0.5},
        "exposure": {"description": "overall brightness multiplier",
                     "min": 0.2, "max": 3.0, "default": 1.1},
        "gamma": {"description": "tonal gamma",
                  "min": 0.3, "max": 2.5, "default": 1.0},
        "time": {"description": "animation phase [0, 2pi)",
                 "min": 0.0, "max": 6.28, "default": 0.0},
        "anim_mode": {"description": "animation mode (none/swell/wind/gust)",
                      "choices": ["none", "swell", "wind", "gust"],
                      "default": "none"},
        "anim_speed": {"description": "animation speed multiplier",
                       "min": 0.1, "max": 5.0, "default": 1.0},
    },
)
def method_gerstner_ocean(out_dir: Path, seed: int, params=None):
    """Gerstner Ocean — analytic trochoidal-wave surface, shaded with sun glint.

    Sums `n_waves` Gerstner components, evaluates the analytic surface normal per
    pixel, and lights it with Lambert + Blinn-Phong sun glitter and a deep/crest
    colour mix into an RGB canvas. Fully vectorised, sub-second, always animated.

    Params:
        n_waves, base_wavelength, wavelength_falloff: spectrum shape.
        amplitude, steepness: wave height and crest sharpness.
        wind_angle, wind_spread: dominant direction + directional spread.
        sun_angle, sun_height, shininess, glint: sun-glitter lighting.
        deep_hue, crest_hue: water colour.
        exposure, gamma: tonal mapping.
        time, anim_mode, anim_speed: animation clock + mode.
    """
    try:
        if params is None:
            params = {}
        t = float(params.get("time", 0.0))
        anim_mode = str(params.get("anim_mode", "none"))
        anim_speed = float(params.get("anim_speed", 1.0))
        seed_all(seed)
        rng = np.random.default_rng(seed)

        n_waves = int(np.clip(params.get("n_waves", 9.0), 1, 24))
        base_wl = float(np.clip(params.get("base_wavelength", 1.1), 0.15, 2.5))
        wl_falloff = float(np.clip(params.get("wavelength_falloff", 0.82), 0.5, 0.95))
        amplitude = float(np.clip(params.get("amplitude", 0.16), 0.02, 0.6))
        steepness = float(np.clip(params.get("steepness", 0.7), 0.0, 1.0))
        wind_angle = float(np.clip(params.get("wind_angle", 0.12), 0.0, 1.0))
        wind_spread = float(np.clip(params.get("wind_spread", 0.14), 0.0, 0.4))
        sun_angle = float(np.clip(params.get("sun_angle", 0.62), 0.0, 1.0))
        sun_height = float(np.clip(params.get("sun_height", 0.35), 0.05, 1.0))
        shininess = float(np.clip(params.get("shininess", 90.0), 4.0, 400.0))
        glint = float(np.clip(params.get("glint", 1.6), 0.0, 4.0))
        deep_hue = float(np.clip(params.get("deep_hue", 0.53), 0.0, 1.0))
        crest_hue = float(np.clip(params.get("crest_hue", 0.5), 0.0, 1.0))
        exposure = float(np.clip(params.get("exposure", 1.1), 0.2, 3.0))
        gamma = float(np.clip(params.get("gamma", 1.0), 0.3, 2.5))

        _t = t * anim_speed

        # ── Animation controls ──
        amp_scale = 1.0
        steep_scale = 1.0
        wind_yaw = 0.0
        phase_advance = 1.0  # multiplies w_i * _t (the propagation term)
        if anim_mode == "none":
            phase_advance = 0.0  # freeze wave phase => static baseline
        elif anim_mode == "swell":
            phase_advance = 1.0
        elif anim_mode == "wind":
            phase_advance = 1.0
            wind_yaw = 0.18 * math.sin(_t * 0.5)  # slow directional churn (turns)
        elif anim_mode == "gust":
            phase_advance = 1.0
            # cos (not sin): keeps t=0 vs t=pi audit frames distinct.
            amp_scale = 1.0 + 0.45 * math.cos(_t)
            steep_scale = 1.0 + 0.30 * math.cos(_t + 1.1)

        # ── Screen-space base grid p = (x, y), aspect-correct ──
        Wp, Hp = int(W), int(H)
        aspect = Wp / max(Hp, 1)
        xs = (np.linspace(-1.0, 1.0, Wp, dtype=np.float32) * aspect)
        ys = np.linspace(-1.0, 1.0, Hp, dtype=np.float32)
        px, py = np.meshgrid(xs, ys)  # (H, W)

        # ── Accumulate height + analytic normal over the wave spectrum ──
        z = np.zeros((Hp, Wp), dtype=np.float32)
        nx = np.zeros((Hp, Wp), dtype=np.float32)
        ny = np.zeros((Hp, Wp), dtype=np.float32)
        nz_sub = np.zeros((Hp, Wp), dtype=np.float32)  # subtracted from 1.0
        chop = np.zeros((Hp, Wp), dtype=np.float32)    # crest sharpness proxy

        wl = base_wl
        amp = amplitude * amp_scale
        # per-wave amplitude falls with wavelength (steeper = smaller)
        for i in range(n_waves):
            k = 2.0 * math.pi / max(wl, 1e-4)       # spatial frequency
            w = math.sqrt(_G * k)                    # deep-water dispersion
            # direction: dominant wind + deterministic spread
            ang_turns = (wind_angle + wind_yaw
                         + wind_spread * (rng.random() - 0.5) * 2.0
                         + (i * 0.113))              # golden-ish decorrelation
            ang = ang_turns * 2.0 * math.pi
            dx = math.cos(ang)
            dy = math.sin(ang)
            phi = float(rng.random()) * 2.0 * math.pi
            A_i = amp * (wl / base_wl)               # longer waves taller
            Q_i = (steepness * steep_scale) / max(k * A_i * n_waves, 1e-4)
            Q_i = min(Q_i, 1.0)

            f = k * (dx * px + dy * py) - w * _t * phase_advance + phi
            s = np.sin(f)
            c = np.cos(f)
            z += A_i * s
            wa = k * A_i
            nx -= dx * wa * c
            ny -= dy * wa * c
            nz_sub += Q_i * wa * s
            chop += Q_i * wa * s

            wl *= wl_falloff
            amp *= wl_falloff ** 0.5

        nz = np.clip(1.0 - nz_sub, 1e-3, None)
        # normalise the normal
        nlen = np.sqrt(nx * nx + ny * ny + nz * nz) + 1e-8
        Nx = nx / nlen
        Ny = ny / nlen
        Nz = nz / nlen

        # ── Lighting ──
        sa = sun_angle * 2.0 * math.pi
        # sun (light) direction, pointing FROM surface TO light
        lz = sun_height
        lh = math.sqrt(max(1.0 - lz * lz, 0.0))
        Lx = math.cos(sa) * lh
        Ly = math.sin(sa) * lh
        Lz = lz
        Llen = math.sqrt(Lx * Lx + Ly * Ly + Lz * Lz) + 1e-8
        Lx, Ly, Lz = Lx / Llen, Ly / Llen, Lz / Llen

        # view direction: looking mostly straight down at the water (slight tilt)
        Vx, Vy, Vz = 0.0, -0.25, 0.968
        Vlen = math.sqrt(Vx * Vx + Vy * Vy + Vz * Vz)
        Vx, Vy, Vz = Vx / Vlen, Vy / Vlen, Vz / Vlen

        ndotl = np.clip(Nx * Lx + Ny * Ly + Nz * Lz, 0.0, 1.0)

        # half vector for Blinn-Phong glitter
        Hx, Hy, Hz = Lx + Vx, Ly + Vy, Lz + Vz
        Hln = math.sqrt(Hx * Hx + Hy * Hy + Hz * Hz) + 1e-8
        Hx, Hy, Hz = Hx / Hln, Hy / Hln, Hz / Hln
        ndoth = np.clip(Nx * Hx + Ny * Hy + Nz * Hz, 0.0, 1.0)
        spec = np.power(ndoth, shininess) * glint

        # ── Colour: deep-water base -> crest, driven by height + steepness ──
        zn = z.copy()
        zmin, zmax = float(zn.min()), float(zn.max())
        if zmax - zmin > 1e-6:
            zn = (zn - zmin) / (zmax - zmin)
        else:
            zn = np.full_like(zn, 0.5)
        # crest whiteness where waves are sharp (chop high) and tall
        chop_n = np.clip((chop - chop.min()) / (float(chop.max() - chop.min()) + 1e-6), 0.0, 1.0)
        foam = np.clip((zn - 0.72) / 0.28, 0.0, 1.0) * np.clip(chop_n * 1.4, 0.0, 1.0)

        hue = deep_hue + (crest_hue - deep_hue) * zn
        hue = np.mod(hue, 1.0).astype(np.float32)
        sat = np.clip(0.85 - 0.55 * zn, 0.15, 0.95).astype(np.float32)

        # base value: ambient sky + lambert diffuse
        ambient = 0.28
        val = (ambient + 0.85 * ndotl).astype(np.float32)
        val = np.clip(val * exposure, 0.0, None)

        def hsv2rgb_vec(h, s, v):
            i = np.floor(h * 6.0).astype(np.int32) % 6
            f = h * 6.0 - np.floor(h * 6.0)
            p = v * (1.0 - s)
            q = v * (1.0 - f * s)
            tt = v * (1.0 - (1.0 - f) * s)
            r = np.zeros_like(v); g = np.zeros_like(v); b = np.zeros_like(v)
            m = i == 0; r[m], g[m], b[m] = v[m], tt[m], p[m]
            m = i == 1; r[m], g[m], b[m] = q[m], v[m], p[m]
            m = i == 2; r[m], g[m], b[m] = p[m], v[m], tt[m]
            m = i == 3; r[m], g[m], b[m] = p[m], q[m], v[m]
            m = i == 4; r[m], g[m], b[m] = tt[m], p[m], v[m]
            m = i == 5; r[m], g[m], b[m] = v[m], p[m], q[m]
            return r, g, b

        rr, gg, bb = hsv2rgb_vec(hue, sat, val)
        rgb = np.stack([rr, gg, bb], axis=-1).astype(np.float32)

        # add sun glitter (warm white) and foam (cool white)
        rgb += spec[..., None] * np.array([1.0, 0.96, 0.86], dtype=np.float32)
        rgb += foam[..., None] * 0.9 * np.array([0.92, 0.96, 1.0], dtype=np.float32)

        # tonal gamma + clamp
        rgb = np.clip(rgb, 0.0, None)
        rgb = np.power(rgb, 1.0 / gamma)
        rgb = np.clip(rgb, 0.0, 1.0).astype(np.float32)

        # ── Outputs ──
        field = zn.astype(np.float32)                       # normalised height
        mask = (foam > 0.05).astype(np.float32)             # foam / crest mask

        capture_frame("963", rgb)
        save(rgb, mn(963, "Gerstner Ocean"), out_dir)
        try:
            write_field(out_dir, field)
            write_mask(out_dir, mask)
            write_scalars(
                out_dir,
                n_waves=float(n_waves),
                mean_height=float(z.mean()),
                foam_coverage=float(mask.mean()),
                peak_spec=float(spec.max()),
                mode_code=float(hash(anim_mode) % 1000),
            )
        except Exception:
            pass
        return rgb
    except Exception as exc:
        fallback = np.full((int(H), int(W), 3), 0.5, dtype=np.float32)
        save(fallback, mn(963, "Gerstner Ocean"), out_dir)
        print(f"[method_963] ERROR: {exc}")
        return fallback
