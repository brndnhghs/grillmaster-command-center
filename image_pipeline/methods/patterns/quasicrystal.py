from __future__ import annotations
import math
import random
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from ...core.registry import method
from ...core.utils import save, norm, mn, seed_all, W, H, write_field, wired_source_lum
from ...core.animation import capture_frame
from ...core.utils import PALETTES

@method(id='02', name='Quasicrystal', category='patterns', tags=['classic', 'wave', 'fast', 'expanded', 'animation'], outputs={'image': 'IMAGE', 'field': 'FIELD'}, params={'waves': {'description': 'number of wave planes', 'min': 2, 'max': 50, 'default': 8}, 'lattice': {'description': 'lattice symmetry (penrose/octagonal/dodecagonal/decagonal/tetragonal/hexagon/triangular/quasi/radial/custom)', 'default': 'penrose'}, 'wave_fn': {'description': 'wave function (sin/triangle/square/sawtooth/gabor/gaussian/pulse)', 'default': 'sin'}, 'colormode': {'description': 'color mode (grayscale/palette/heatmap/spectral/fire/ice/plasma/dual_layer)', 'default': 'heatmap'}, 'frequency': {'description': 'wave frequency scale', 'min': 0.005, 'max': 0.5, 'default': 0.05}, 'amplitude': {'description': 'wave amplitude', 'min': 0.1, 'max': 2.0, 'default': 1.0}, 'modulation': {'description': 'space modulation (none/radial/gaussian/spiral/vortex)', 'default': 'none'}, 'mod_strength': {'description': 'modulation strength', 'min': 0.0, 'max': 1.0, 'default': 0.3}, 'palette': {'description': 'color palette name (PALETTES keys)', 'default': 'vapor'}, 'rotation': {'description': 'global rotation offset (radians)', 'min': 0.0, 'max': 6.2832, 'default': 0.0}, 'anim_mode': {'description': 'animation mode: none, plane_rotate, freq_sweep, counter_rotate, multi_plane_freq, wave_count_sweep, lattice_cycle, wave_fn_cycle, mod_cycle, colormode_cycle, phase_drift', 'default': 'none'}, 'anim_speed': {'description': 'animation speed multiplier', 'min': 0.1, 'max': 3.0, 'default': 1.0}, 'source': {'description': "wired upstream image's luminance", 'choices': ['none', 'input_image'], 'default': 'none'}}, inputs={'image_in': 'IMAGE'})
def method_quasicrystal(out_dir: Path, seed: int, params=None):
    """Render quasicrystal diffraction patterns via wave-plane superposition.

    Generates non-repeating but deterministic patterns by summing wave
    planes at rational/irrational angle relationships. Supports multiple
    lattice symmetries, wave functions, and color modes.
    """
    try:
        if params is None:
            params = {}
        t = float(params.get("time", 0.0))
        seed_all(seed)
        yy, xx = np.mgrid[:H, :W].astype(np.float32)
        # Wired image as a domain-warp source (luminance distorts the pattern grid)
        _src_lum = wired_source_lum(params, xx.shape[1], xx.shape[0])
        if _src_lum is not None:
            xx = xx + (_src_lum - 0.5) * 15.0
            yy = yy + (_src_lum - 0.5) * 15.0

        cx, cy = W / 2.0, H / 2.0

        # ── Params ──
        n_waves = int(params.get("waves", 8))
        lattice = params.get("lattice", "penrose")
        wave_fn = params.get("wave_fn", "sin")
        cmode = params.get("colormode", "heatmap")
        freq = float(params.get("frequency", 0.05))
        amp = float(params.get("amplitude", 1.0))
        mod_type = params.get("modulation", "none")
        mod_str = float(params.get("mod_strength", 0.3))
        pal_name = params.get("palette", "vapor")
        rot = float(params.get("rotation", 0.0))
        anim_mode = params.get("anim_mode", "none")
        anim_speed = float(params.get("anim_speed", 1.0))

        # Freeze time when no animation mode is active
        if anim_mode == "none":
            t = 0.0
            anim_speed = 0.0

        # ── Matplotlib colormap import (with fallback) ──
        try:
            from matplotlib import cm
            _has_mpl = True
        except ImportError:
            _has_mpl = False

        # ── Pre-compute wave-plane data (before animation block so _lattice_angles is available) ──
        rng = np.random.default_rng(seed)

        def _lattice_angles(n, sym):
            angles = []
            if sym == "penrose":        # 5-fold - golden ratio based
                phi = math.pi * (1 + math.sqrt(5)) / 2
                for i in range(n):
                    angles.append((i * 2 * math.pi / phi) % (2 * math.pi))
            elif sym == "octagonal":    # 8-fold
                for i in range(n):
                    angles.append(i * math.pi / 4)
            elif sym == "dodecagonal":  # 12-fold
                for i in range(n):
                    angles.append(i * math.pi / 6)
            elif sym == "decagonal":    # 10-fold
                for i in range(n):
                    angles.append(i * 2 * math.pi / 10 + rng.uniform(0, 0.01))
            elif sym == "tetragonal":   # 4-fold
                for i in range(n):
                    angles.append(i * math.pi / 2 + rng.uniform(0, 0.3))
            elif sym == "hexagon":      # 6-fold
                for i in range(n):
                    angles.append(i * math.pi / 3)
            elif sym == "triangular":   # 3-fold
                for i in range(n):
                    angles.append(i * 2 * math.pi / 3 + rng.uniform(0, 0.2))
            elif sym == "quasi":        # quasi-random uniform
                angles = list(rng.uniform(0, 2 * math.pi, n))
            elif sym == "radial":        # converging
                for i in range(n):
                    angles.append(i * 2 * math.pi / n)
            elif sym == "custom":
                base = rng.uniform(0, 2 * math.pi, n // 2 + 1)
                dither = rng.uniform(0, 0.5, n // 2 + 1)
                angles = list(base) + [a + d for a, d in zip(base[:n - n // 2], dither[:n - n // 2])]
            else:
                angles = [rng.uniform(0, 2 * math.pi) for _ in range(n)]
            return angles[:n]

        # ── Animation: operate on wave plane parameters ──
        effective_freq = freq
        max_waves = int(params.get("waves", 8))  # original count before reduction
        if anim_mode == "plane_rotate":
            # All wave plane angles rotate uniformly — the diffraction pattern spins
            pass  # applied per-wave in field builder via t_rot
        elif anim_mode == "freq_sweep":
            # Frequency sweeps up and down — interference fringes zoom in/out
            effective_freq = freq * (0.3 + 0.7 * (0.5 + 0.5 * math.sin(t * 0.3 * anim_speed)))
        elif anim_mode == "counter_rotate":
            # Half the waves rotate forward, half backward — shearing/interference motion
            pass  # applied per-wave in the field builder
        elif anim_mode == "multi_plane_freq":
            # Each wave plane's frequency oscillates out of phase — ripples cross at different rates
            pass  # applied per-wave in the field builder
        elif anim_mode == "wave_count_sweep":
            # Number of wave planes sweeps up and down — complexity of the pattern changes
            n_waves = max(2, int(n_waves * (0.3 + 0.7 * (0.5 + 0.5 * math.sin(t * 0.2 * anim_speed)))))
        elif anim_mode == "lattice_cycle":
            # Cycle through all lattice symmetries
            lattices = ["penrose", "octagonal", "dodecagonal", "decagonal", "tetragonal", "hexagon", "triangular", "quasi", "radial"]
            lattice = lattices[int(t * 0.2 * anim_speed) % len(lattices)]
        elif anim_mode == "wave_fn_cycle":
            # Cycle through wave functions, each frame a new shape
            wave_fns = ["sin", "triangle", "square", "sawtooth", "gabor", "gaussian", "pulse"]
            wave_fn = wave_fns[int(t * 0.25 * anim_speed) % len(wave_fns)]
        elif anim_mode == "mod_cycle":
            # Cycle through spatial modulation patterns
            mods = ["none", "radial", "gaussian", "spiral", "vortex"]
            mod_type = mods[int(t * 0.18 * anim_speed) % len(mods)]
        elif anim_mode == "colormode_cycle":
            # Cycle through all color modes
            colormodes = ["grayscale", "heatmap", "spectral", "fire", "ice", "plasma", "dual_layer", "palette"]
            cmode = colormodes[int(t * 0.15 * anim_speed) % len(colormodes)]
        elif anim_mode == "phase_drift":
            # Each wave plane's phase drifts at different rates
            pass  # applied per-wave via t_phase array in field builder

        # ── Generate wave-plane data ──

        base_thetas = _lattice_angles(max_waves, lattice)
        base_thetas = [(a + rot) % (2 * math.pi) for a in base_thetas]
        base_phases = [rng.uniform(0, 2 * math.pi) for _ in range(max_waves)]
        base_freqs = [freq * (0.5 + rng.random()) for _ in range(max_waves)]

        # ── Build wave field from pre-computed per-wave data ──
        xc = xx - cx
        yc = yy - cy

        def _build_field(thetas_a, phases_a, freqs_a, wfn, t_phase, t_rot, n):
            """Build raw wave field. Uses pre-computed theta/phase/freq arrays.
            t_phase: time-based phase shift applied to all waves.
            t_rot: optional per-wave rotation offset [n] or scalar.
            Returns (H,W) float32."""
            if np.ndim(t_rot) == 0:
                t_rot_arr = np.full(n, t_rot)
            else:
                t_rot_arr = np.asarray(t_rot, dtype=np.float32)[:n]
            fld = np.zeros((H, W), dtype=np.float32)
            for i in range(n):
                theta = (thetas_a[i] + t_rot_arr[i]) % (2 * math.pi)
                ph = phases_a[i] + t_phase
                f = freqs_a[i]
                proj = xc * math.cos(theta) + yc * math.sin(theta)
                raw = proj * f + ph
                if wfn == "sin":
                    w = np.sin(raw)
                elif wfn == "triangle":
                    w = 2 * np.abs(2 * (raw / (2 * math.pi) - np.floor(raw / (2 * math.pi) + 0.5))) - 1
                elif wfn == "square":
                    w = np.where(np.sin(raw) >= 0, 1.0, -1.0)
                elif wfn == "sawtooth":
                    w = 2 * (raw / (2 * math.pi) - np.floor(raw / (2 * math.pi) + 0.5))
                elif wfn == "gabor":
                    gauss = np.exp(-0.5 * (proj * f * 0.5) ** 2)
                    w = np.sin(raw) * gauss
                elif wfn == "gaussian":
                    w = np.exp(-0.5 * (np.sin(raw) * 2) ** 2)
                elif wfn == "pulse":
                    w = np.where(np.abs(np.sin(raw)) > 0.95, 1.0, -0.5)
                else:
                    w = np.sin(raw)
                fld += w * amp
            return fld

        t_phase = t * 0.3 * anim_speed
        t_rot = 0.0
        effective_freqs = list(base_freqs)

        if anim_mode == "plane_rotate":
            t_rot = t * 0.5 * anim_speed
        elif anim_mode == "freq_sweep":
            # Scale all wave frequencies uniformly — ripples zoom in/out coherently
            ratio = effective_freq / freq if freq > 0 else 1.0
            for i in range(n_waves):
                effective_freqs[i] = base_freqs[i] * ratio
        elif anim_mode == "counter_rotate":
            # Each wave plane rotates at different speed based on its angle index
            n = n_waves
            t_rot_per = np.empty(n, dtype=np.float32)
            for i in range(n):
                base_angle = base_thetas[i] % (2 * math.pi)
                # Even-indexed waves rotate forward, odd-indexed backward
                sign = 1.0 if i % 2 == 0 else -1.0
                # Speed varies by angle quadrant to create complex interference
                speed = 0.3 + 0.5 * (0.5 + 0.5 * math.sin(base_angle))
                t_rot_per[i] = sign * speed * t * anim_speed
            t_rot = t_rot_per
        elif anim_mode == "multi_plane_freq":
            # Each wave plane's frequency oscillates independently
            for i in range(n_waves):
                offset = i * 0.7  # phase offset per wave
                osc = 0.5 + 0.5 * math.sin(t * 0.25 * anim_speed + offset)
                effective_freqs[i] = base_freqs[i] * (0.5 + osc)
        elif anim_mode == "phase_drift":
            # Each wave plane's phase drifts at different rates
            n = n_waves
            t_rot_per = np.empty(n, dtype=np.float32)
            for i in range(n):
                rate = 0.3 + 0.7 * (i / max(1, n))  # slower for early waves, faster for late
                t_rot_per[i] = rate * t * anim_speed
            t_rot = t_rot_per

        result = _build_field(base_thetas, base_phases, effective_freqs,
                              wave_fn, t_phase, t_rot, n_waves)

        if mod_type != "none":
            r = np.sqrt(xc ** 2 + yc ** 2)
            max_r = np.sqrt(cx ** 2 + cy ** 2)
            r_norm = r / max_r
            if mod_type == "radial":
                mask = np.exp(-0.5 * (r_norm * 3) ** 2) * (1 - mod_str) + mod_str
            elif mod_type == "gaussian":
                sigma = 0.4 * (1 - mod_str * 0.5)
                mask = np.exp(-0.5 * (r_norm / sigma) ** 2)
            elif mod_type == "spiral":
                theta_r = np.arctan2(yc, xc) + r_norm * 4 * math.pi * mod_str
                mask = 0.5 + 0.5 * np.sin(theta_r)
            elif mod_type == "vortex":
                theta_r = np.arctan2(yc, xc) * 3
                mask = 0.5 + 0.5 * np.sin(r_norm * 8 * math.pi * mod_str + theta_r)
            else:
                mask = 1.0
            result = result * mask

        # ── Normalize ──
        result = norm(result)
        write_field(out_dir, result)

        # ── Color ──
        if cmode == "grayscale":
            rgb = np.stack([result, result, result], axis=-1)
        elif cmode == "palette":
            pal = PALETTES.get(pal_name, PALETTES["vapor"])
            idx = (result * (len(pal) - 1)).astype(np.int32)
            pal_arr = np.array(pal, dtype=np.float32) / 255.0
            rgb = pal_arr[idx]
        elif cmode == "heatmap":
            if _has_mpl:
                rgb = cm.inferno(result)[:, :, :3]
            else:
                rgb = np.stack([result, result, result], axis=-1)
        elif cmode == "spectral":
            if _has_mpl:
                rgb = cm.nipy_spectral(result)[:, :, :3]
            else:
                rgb = np.stack([result, result, result], axis=-1)
        elif cmode == "fire":
            r2 = np.clip(result * 1.5, 0, 1)
            rgb = np.stack([r2, result * 0.6, result * 0.2], axis=-1)
        elif cmode == "ice":
            rgb = np.stack([result * 0.2, result * 0.5, 0.5 + result * 0.5], axis=-1)
        elif cmode == "plasma":
            if _has_mpl:
                rgb = cm.plasma(result)[:, :, :3]
            else:
                rgb = np.stack([result, result, result], axis=-1)
        elif cmode == "dual_layer":
            if _has_mpl:
                hi = result > 0.5
                lo = result <= 0.5
                base = np.zeros((H, W, 3), dtype=np.float32)
                base[lo] = cm.viridis(result[lo] * 2)[:, :3]
                base[hi] = cm.inferno((result[hi] - 0.5) * 2)[:, :3]
                rgb = base
            else:
                rgb = np.stack([result, result, result], axis=-1)
        else:
            rgb = np.stack([result, result, result], axis=-1)

        rgb = np.clip(rgb, 0, 1).astype(np.float32)
        capture_frame("02", rgb)
        save(rgb, mn(2, "quasicrystal"), out_dir)
        return rgb
    except Exception as exc:
        fallback = np.full((H, W, 3), 128, dtype=np.uint8)
        save(fallback, mn(2, 'Quasicrystal'), out_dir)
        print(f'[method_02] ERROR: {exc}')
        return fallback


