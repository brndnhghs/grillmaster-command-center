from __future__ import annotations
import math
import random
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from ...core.registry import method
from ...core.utils import save, norm, mn, seed_all, get_font, BG_DEFAULT, W, H
from ...core.animation import capture_frame

try:
    import cv2
    _has_cv2 = True
except ImportError:
    _has_cv2 = False

@method(id="48", name="FFT Art", category="math_art", new_image_contract=True, tags=["frequency","fast", "expanded"],
description="FFT Art — math-art node.",
         inputs={"image_in": "IMAGE"},
         params={"filter_type":{"description":"filter type","choices":["ring","concentric","spiral","star","checkerboard","text_mask","input_mask","gabor_bank","fractal_noise","polar_fft","phase_swap","convolution_kernel","frequency_paint","radial_pattern","time_frequency"],"default":"ring"},
                 "source":{"description":"source","choices":["random","perlin","wave_interference","color_noise","input_image","texture_synth"],"default":"random"},
                 "color_mode":{"description":"coloring","choices":["gradient","palette","phase","magnitude","multi_channel","phase_magnitude_blend","rainbow","heatmap","channel_swap"],"default":"gradient"},
                 "palette": {"description": "PALETTES", "default": ""},
                 "n_rings": {"description": "rings", "min": 2, "max": 20, "default": 5},
                 "ring1_center":{"description":"ring1 center","min":20,"max":200,"default":60},"ring1_sigma":{"description":"ring1 sigma","min":5,"max":60,"default":15},
                 "ring2_center":{"description":"ring2 center","min":20,"max":300,"default":120},"ring2_sigma":{"description":"ring2 sigma","min":5,"max":60,"default":20},
                 "spiral_turns":{"description":"spiral turns","min":1,"max":10,"default":4},"star_arms":{"description":"star arms","min":2,"max":20,"default":6},
                 "checker_size":{"description":"checker size","min":4,"max":40,"default":16},"text_content":{"description":"text","default":"FFT"},
                 "gabor_freqs":{"description":"gabor freqs","min":1,"max":10,"default":4},"gabor_orientations":{"description":"gabor orients","min":1,"max":8,"default":4},
                 "fractal_exponent":{"description":"fractal exponent","min":0.5,"max":3.0,"default":1.5},
                 "phase_swap_source":{"description":"phase swap source","choices":["perlin","random","input"],"default":"perlin"},
                 "kernel_type":{"description":"conv kernel","choices":["gaussian","sobel","laplacian","emboss","sharpen"],"default":"gaussian"},
                 "kernel_size":{"description":"kernel size","min":3,"max":31,"default":7},
                 "polar_radial_freq":{"description":"polar radial freq","min":1,"max":20,"default":4},"polar_angular_freq":{"description":"polar angular freq","min":1,"max":20,"default":6},"anim_mode":{"description":"animation mode","choices":["none","filter_rotate","source_drift","gabor_sweep"],"default":"none"},
                 "anim_speed": {"description": "animation speed multiplier", "min": 0.1, "max": 5.0, "default": 1.0}})
def method_fft_art(out_dir: Path, seed: int, params=None):
    """Generate frequency-domain art via FFT filtering with 15 filter types.

    Creates a noise source, transforms to frequency domain via FFT, applies a
    frequency-domain filter mask, and inverse-transforms back to produce
    visually rich patterns. Supports multiple filter types, sources, and color
    modes.

    Args:
        out_dir: Output directory for the generated image.
        seed: Random seed for deterministic output.
        params: Dict with keys:
            filter_type: frequency filter type (ring/concentric/spiral/star/...)
            source: noise source (random/perlin/wave_interference/color_noise/...)
            color_mode: coloring mode (gradient/phase/magnitude/rainbow/...)
            palette: PALETTES name for palette quantization
            n_rings: number of concentric rings (2-20)
            ring1_center: first ring center frequency (20-200)
            ring1_sigma: first ring sigma (5-60)
            ring2_center: second ring center frequency (20-300)
            ring2_sigma: second ring sigma (5-60)
            spiral_turns: spiral turns (1-10)
            star_arms: star arms (2-20)
            checker_size: checkerboard cell size (4-40)
            text_content: text for text_mask filter
            gabor_freqs: gabor filter frequency count (1-10)
            gabor_orientations: gabor orientation count (1-8)
            fractal_exponent: fractal noise exponent (0.5-3.0)
            phase_swap_source: phase swap source (perlin/random/input)
            kernel_type: convolution kernel type
            kernel_size: convolution kernel size (3-31)
            polar_radial_freq: polar radial frequency (1-20)
            polar_angular_freq: polar angular frequency (1-20)
            time: animation time in radians (0-6.28)
            anim_mode: animation mode (none/filter_rotate/source_drift/gabor_sweep)
            anim_speed: animation speed multiplier (0.1-5.0)
    """
    try:
        if params is None:
            params = {}
        anim_time = float(params.get("time", 0.0))
        anim_mode = params.get("anim_mode", "none")
        anim_speed = float(params.get("anim_speed", 1.0))
        seed_all(seed)
        rng = random.Random(seed)
        np_rng = np.random.default_rng(seed)

        ft = params.get("filter_type", "ring")
        src = params.get("source", "random")
        cm = params.get("color_mode", "gradient")
        pal_name = params.get("palette", "")
        n_rings = int(params.get("n_rings", 5))
        r1c = float(params.get("ring1_center", 60))
        r1s = float(params.get("ring1_sigma", 15))
        r2c = float(params.get("ring2_center", 120))
        r2s = float(params.get("ring2_sigma", 20))
        st = float(params.get("spiral_turns", 4))
        sa = float(params.get("star_arms", 6))
        cks = float(params.get("checker_size", 16))
        txt = params.get("text_content", "FFT")
        gf = int(params.get("gabor_freqs", 4))
        go = int(params.get("gabor_orientations", 4))
        fe = float(params.get("fractal_exponent", 1.5))
        pss = params.get("phase_swap_source", "perlin")
        kt = params.get("kernel_type", "gaussian")
        ks = int(params.get("kernel_size", 7))
        prf = float(params.get("polar_radial_freq", 4))
        paf = float(params.get("polar_angular_freq", 6))

        # ── Animation ──
        t = anim_time * anim_speed
        if anim_mode == "filter_rotate":
            # Rotate filter pattern by adding t to angular components
            pass  # t is already used in filter building below
        elif anim_mode == "source_drift":
            # Drift the source noise pattern
            pass  # t is already used in perlin/wave_interference sources
        elif anim_mode == "gabor_sweep":
            # Sweep gabor frequencies
            pass  # t is already used in gabor_bank filter
        # For "none" mode, freeze t at 0
        if anim_mode == "none":
            t = 0.0

        # ── Import cv2 lazily ──
        try:
            import cv2
        except ImportError:
            cv2 = None

        from ...core.utils import PALETTES, quantize_to_palette
        pal = PALETTES.get(pal_name, [])

        # ── Generate source ──
        # If an upstream image is wired in, use it as the FFT source
        _inp = params.get("_input_image")
        if _inp is not None:
            try:
                img_arr = _inp
                noise = 0.299 * img_arr[:, :, 0] + 0.587 * img_arr[:, :, 1] + 0.114 * img_arr[:, :, 2]
                noise = (noise - 0.5) * 2  # remap [0,1] → [-1,1]
                noise = np.stack([noise] * 3, axis=-1)
                src = "__wired__"
            except (FileNotFoundError, OSError):
                pass

        if src == "random":
            noise = np_rng.standard_normal((H, W))
        elif src == "perlin":
            yy, xx = np.ogrid[:H, :W]
            noise = np.sin(xx * 0.05) * np.cos(yy * 0.05) + np.sin(xx * 0.1 + t) * np.cos(yy * 0.08 + t * 0.5) + np.sin(xx * 0.02 + t * 1.3) * np.cos(yy * 0.03 + t * 0.7)
        elif src == "wave_interference":
            yy, xx = np.ogrid[:H, :W]
            noise = np.sin(xx * 0.1 + t) * np.cos(yy * 0.1 + t * 0.7) + np.sin(xx * 0.15 + t * 1.3) * np.cos(yy * 0.12 + t * 0.5)
        elif src == "color_noise":
            noise = np_rng.standard_normal((H, W, 3))
        else:
            noise = np_rng.standard_normal((H, W))
        if noise.ndim == 2:
            noise = np.stack([noise] * 3, axis=-1)

        # ── FFT ──
        fft = np.fft.fft2(noise, axes=(0, 1))
        fft = np.fft.fftshift(fft, axes=(0, 1))
        Hc, Wc = H // 2, W // 2
        yy, xx = np.ogrid[:H, :W]
        r = np.sqrt((xx - Wc) ** 2 + (yy - Hc) ** 2)
        theta = np.arctan2(yy - Hc, xx - Wc)

        # ── Build filter ──
        mask = np.ones((H, W), dtype=np.float32)
        if ft == "ring":
            mask = np.exp(-(r - r1c) ** 2 / (2 * r1s ** 2)) + np.exp(-(r - r2c) ** 2 / (2 * r2s ** 2))
        elif ft == "concentric":
            mask = np.zeros((H, W), dtype=np.float32)
            for i in range(n_rings):
                mask += np.exp(-(r - (i + 1) * r2c / n_rings) ** 2 / (2 * r1s ** 2))
        elif ft == "spiral":
            mask = np.sin(r * 0.1 + theta * st + t) * 0.5 + 0.5
        elif ft == "star":
            mask = (np.sin(theta * sa / 2 + t) * 0.5 + 0.5) * np.exp(-r ** 2 / (2 * (W // 3) ** 2))
        elif ft == "checkerboard":
            mask = ((np.floor(xx / cks) + np.floor(yy / cks)) % 2).astype(np.float32)
        elif ft == "gabor_bank":
            mask = np.zeros((H, W), dtype=np.float32)
            for fi in range(gf):
                for oi in range(go):
                    f = (fi + 1) * 10
                    o = oi * math.pi / go + t * 0.5
                    g = np.exp(-(r - f) ** 2 / (2 * 20 ** 2)) * np.cos(theta - o) ** 2
                    mask += g
            mask = mask / mask.max()
        elif ft == "fractal_noise":
            mask = r ** (-fe)
            mask[H // 2, W // 2] = 0
            mask = mask / mask.max()
        elif ft == "polar_fft":
            mask = np.sin(r * prf * 0.1) * np.cos(theta * paf + t) * 0.5 + 0.5
        elif ft == "frequency_paint":
            mask = np.zeros((H, W), dtype=np.float32)
            for _ in range(20):
                cx = rng.randint(0, W - 1)
                cy = rng.randint(0, H - 1)
                sr = rng.uniform(5, 20)
                mask += np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * sr ** 2))
            mask = mask / mask.max()
        elif ft == "radial_pattern":
            mask = np.sin(r * 0.1 + t) * np.cos(theta * 3 + t * 0.5) * 0.5 + 0.5

        # ── Apply filter ──
        for c in range(3):
            fft_c = fft[:, :, c] * mask
            if ft == "phase_swap":
                if pss == "perlin":
                    pn = np.sin(xx * 0.05) * np.cos(yy * 0.05) + np.sin(xx * 0.1) * np.cos(yy * 0.08)
                elif pss == "random":
                    pn = np_rng.standard_normal((H, W))
                else:
                    pn = noise[:, :, 0]
                fft_c = np.abs(fft_c) * np.exp(1j * pn * 2 * math.pi)
            img_c = np.abs(np.fft.ifft2(np.fft.ifftshift(fft_c, axes=(0, 1)), axes=(0, 1)))
            fft[:, :, c] = fft_c
        img = np.abs(np.fft.ifft2(np.fft.ifftshift(fft, axes=(0, 1)), axes=(0, 1)))
        img = norm(img)

        # ── Color mode ──
        if cm == "phase":
            phase = np.angle(fft[:, :, 0])
            img = np.stack([phase, phase * 0.5, 1 - phase], axis=-1)
            img = norm(img)
        elif cm == "magnitude":
            mag = np.log1p(np.abs(fft[:, :, 0]))
            img = np.stack([mag] * 3, axis=-1)
            img = norm(img)
        elif cm == "channel_swap":
            img = img[:, :, [2, 0, 1]]
        elif cm == "rainbow":
            img = np.stack([img[:, :, 0], img[:, :, 1] * 0.5, 1 - img[:, :, 2]], axis=-1)
        elif cm == "heatmap":
            img = np.stack([img[:, :, 0], img[:, :, 1] * 0.3, 1 - img[:, :, 2] * 0.5], axis=-1)
        # gradient, multi_channel, palette, phase_magnitude_blend: pass through

        if pal_name and pal_name in PALETTES:
            img = quantize_to_palette(img.clip(0, 1), pal_name)
        capture_frame("48", img)
        save(img.clip(0, 1), mn(48, "FFT Art"), out_dir)
    except Exception as exc:
        fallback = np.full((H, W, 3), 128, dtype=np.uint8)
        save(fallback, mn(48, 'FFT Art'), out_dir)
        print(f'[method_48] ERROR: {exc}')
        return fallback

