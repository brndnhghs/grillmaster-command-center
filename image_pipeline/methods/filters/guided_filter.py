from __future__ import annotations
import math
from pathlib import Path

import numpy as np
from scipy.ndimage import uniform_filter

from ...core.registry import method
from ...core.utils import (save, norm, mn, seed_all, W, H, PALETTES, load_input)
from ...core.animation import capture_frame


@method(
    id="335",
    name="Guided Image Filter",
    category="filters",
    new_image_contract=True,
    tags=["smoothing", "edge-preserving", "detail", "abstraction", "expanded", "animation"],
    inputs={},
    outputs={"image": "IMAGE", "mask": "MASK"},
    params={
        "source": {"description": "source (noise/gradient/input_image/palette/rainbow/procedural)", "default": "noise"},
        "radius": {"description": "guided-filter box radius in px (smoothing spatial extent)", "min": 1, "max": 30, "default": 8},
        "epsilon": {"description": "regularization: larger = flatter smoothing (1e-4..0.5)", "min": 0.0001, "max": 0.5, "default": 0.02},
        "guide": {"description": "guidance image (self = guide with full color, luminance = coherent grayscale guide)", "choices": ["self", "luminance"], "default": "luminance"},
        "mode": {"description": "output (smooth = edge-preserving filtered, detail = extracted residual, enhance = src + amount*residual)", "choices": ["smooth", "detail", "enhance"], "default": "smooth"},
        "amount": {"description": "detail/enhance strength (0-2)", "min": 0.0, "max": 2.0, "default": 1.0},
        "noise_amp": {"description": "noise amplitude for generated sources", "min": 0.1, "max": 1.0, "default": 0.6},
        "blur_sigma": {"description": "gaussian blur sigma for noise source", "min": 5, "max": 80, "default": 30},
        "palette": {"description": "palette name for palette source", "default": "vapor"},
        "anim_mode": {"description": "animation mode (none/radius_pulse/epsilon_pulse/mode_cycle)", "choices": ["none", "radius_pulse", "epsilon_pulse", "mode_cycle"], "default": "none"},
        "anim_speed": {"description": "animation speed multiplier", "min": 0.1, "max": 5.0, "default": 1.0},
    },
)
def method_guided_filter(out_dir: Path, seed: int, params=None):
    """Guided Image Filter — O(N) edge-preserving smoothing (He, Sun & Tang, ECCV 2010 / TPAMI 2013).

    Unlike the Kuwahara filter (which picks the lowest-variance sector of an
    oriented window), the guided filter models the output as a *local linear
    transform* of a guidance image I inside each window. Solving the
    regularization

        min  sum((a_k*I_i + b_k - p_i)^2 + eps*a_k^2)

    over a window k gives, per channel,

        a = cov(I,p) / (var(I) + eps)
        b = mean(p) - a*mean(I)

    and the output is the box-filtered (mean) of a*I + b. Because the model is
    linear in I, the filter exactly preserves edges in I (no gradient
    reversal) and has O(N) cost via two box-filter passes — the basis for
    structure-preserving smoothing, detail/structure separation, joint
    upsampling, and haze removal.

    CPU path is the authoritative export. The method is fully self-contained
    (scipy box filters, no cv2).

    Params:
        source:     generated source type (noise/gradient/input_image/palette/rainbow/procedural)
        radius:     box radius in px (1-30, default 8)
        epsilon:    regularization; larger flattens (1e-4..0.5, default 0.02)
        guide:      self (per-channel color guidance) or luminance (coherent grayscale guidance)
        mode:       smooth / detail (residual) / enhance (unsharp via guided residual)
        amount:     detail/enhance strength (0-2, default 1)
        noise_amp:  amplitude for generated sources (0.1-1.0)
        blur_sigma: blur sigma for noise source (5-80)
        palette:    palette name for palette source
        time:       animation clock (0-6.28)
        anim_mode:  none / radius_pulse / epsilon_pulse / mode_cycle
        anim_speed: animation speed (0.1-5.0)
    """
    try:
        if params is None:
            params = {}
        anim_time = float(params.get("time", 0.0))
        anim_mode = str(params.get("anim_mode", "none"))
        anim_speed = float(params.get("anim_speed", 1.0))
        seed_all(seed)
        rng = np.random.default_rng(seed)

        source = str(params.get("source", "noise"))
        radius = int(params.get("radius", 8))
        radius = max(1, min(30, radius))
        eps = float(params.get("epsilon", 0.02))
        eps = max(1e-4, min(0.5, eps))
        guide = str(params.get("guide", "luminance"))
        mode = str(params.get("mode", "smooth"))
        amount = float(params.get("amount", 1.0))
        noise_amp = float(params.get("noise_amp", 0.6))
        blur_sigma = float(params.get("blur_sigma", 30))
        pal_name = str(params.get("palette", "vapor"))

        # ── Animation (rename t to avoid shadowing the time param) ──
        _t = anim_time * anim_speed
        if anim_mode == "radius_pulse":
            # smooth oscillation, never below 1
            radius = max(1, int(radius * (0.4 + 0.6 * (0.5 + 0.5 * math.sin(_t * 0.3)))))
        elif anim_mode == "epsilon_pulse":
            eps = max(1e-4, eps * (0.3 + 0.7 * (0.5 + 0.5 * math.sin(_t * 0.4))))
        elif anim_mode == "mode_cycle":
            # smooth / detail / enhance cycle (intentional discrete content switch)
            idx = int((_t / 2.094)) % 3
            mode = ["smooth", "detail", "enhance"][idx]
        # else: none — static

        # ── Resolve source image (float32 [0,1], H×W×3) ──
        # A wired upstream image always overrides source generation (Rule #12).
        src = None
        wired_path = params.get("input_image", "")
        if wired_path:
            try:
                src = load_input(wired_path, int(W), int(H))
            except (FileNotFoundError, OSError):
                src = None

        if src is None:
            if source == "gradient":
                yy, xx = np.mgrid[:H, :W].astype(np.float32)
                r = np.sqrt((xx - W / 2) ** 2 + (yy - H / 2) ** 2)
                g = norm(r)
                src = np.stack([g, g * 0.7, 1 - g], axis=-1).clip(0, 1)
            elif source == "palette":
                pal = PALETTES.get(pal_name, list(PALETTES.values())[0])
                yy, xx = np.mgrid[:H, :W].astype(np.float32)
                r = norm(np.sqrt((xx - W / 2) ** 2 + (yy - H / 2) ** 2))
                idx = (r * (len(pal) - 1)).astype(np.int32)
                src = np.array(pal, dtype=np.float32)[idx] / 255.0
            elif source == "rainbow":
                yy, xx = np.mgrid[:H, :W].astype(np.float32)
                r = norm(np.sqrt((xx - W / 2) ** 2 + (yy - H / 2) ** 2))
                hue = r * 2 * math.pi
                src = np.stack([
                    np.sin(hue) * 0.5 + 0.5,
                    np.sin(hue + 2.094) * 0.5 + 0.5,
                    np.sin(hue + 4.189) * 0.5 + 0.5,
                ], axis=-1).astype(np.float32)
            elif source == "procedural":
                yy, xx = np.mgrid[:H, :W].astype(np.float32)
                g = np.sin(xx * 0.03 + yy * 0.02 + _t * 0.5) * \
                    np.cos(xx * 0.02 - yy * 0.03 + _t * 0.3) * 0.5 + 0.5
                src = np.stack([g, g * 0.6, 1 - g * 0.8], axis=-1).astype(np.float32)
            else:  # noise / input_image fallback
                n = rng.standard_normal((H, W, 3)).astype(np.float32) * noise_amp + 0.5
                n = uniform_filter(n, size=max(3, int(blur_sigma)), mode="reflect")
                src = norm(n)

        src = np.clip(src, 0.0, 1.0).astype(np.float32)

        # ── Guided filter core ──
        gray = (0.299 * src[:, :, 0] + 0.587 * src[:, :, 1] + 0.114 * src[:, :, 2]).astype(np.float32)
        if guide == "luminance":
            guide_img = gray  # single-channel coherent guidance for all channels
        else:
            guide_img = None   # self-guided: each channel guides its own output

        smoothed = np.empty_like(src)
        for c in range(3):
            p = src[:, :, c]
            g = guide_img if guide_img is not None else p
            smoothed[:, :, c] = _guided_filter_1d(g, p, radius, eps)

        # ── Compose output ──
        residual = src - smoothed
        detail_energy = float(np.mean(np.abs(residual)))
        if mode == "detail":
            out = np.clip(0.5 + amount * residual, 0.0, 1.0).astype(np.float32)
        elif mode == "enhance":
            out = np.clip(src + amount * residual, 0.0, 1.0).astype(np.float32)
        else:  # smooth
            out = np.clip(smoothed, 0.0, 1.0).astype(np.float32)

        # ── Mask: edge-strength map = magnitude of the detail residual ──
        mask = np.clip(np.mean(np.abs(residual), axis=-1), 0.0, 1.0).astype(np.float32)

        capture_frame("335", out)
        save(out, mn(335, "Guided Image Filter"), out_dir)
        try:
            from ...core.utils import write_scalars, write_mask
            write_scalars(out_dir, radius=float(radius), epsilon=float(eps),
                          detail_energy=detail_energy)
            write_mask(out_dir, mask)
        except Exception:
            pass
        return out
    except Exception as exc:
        fallback = np.full((H, W, 3), 0.5, dtype=np.float32)
        save(fallback, mn(335, "Guided Image Filter"), out_dir)
        print(f"[method_335] ERROR: {exc}")
        return fallback


def _guided_filter_1d(g: np.ndarray, p: np.ndarray, r: int, eps: float) -> np.ndarray:
    """Single-channel O(N) guided filter.

    g, p are (H, W) float32 in [0,1]; r is box radius, eps is regularization.
    Two box-filter (mean) passes give O(N) cost. Boundary handled by reflect.
    """
    if g is p:
        # self-guided: a = var / (var + eps), b = mean(p) - a*mean(g) == mean(p)*(1-a)
        mean_g = uniform_filter(g, size=2 * r + 1, mode="reflect")
        mean_gg = uniform_filter(g * g, size=2 * r + 1, mode="reflect")
        var_g = np.maximum(mean_gg - mean_g * mean_g, 0.0)
        mean_p = uniform_filter(p, size=2 * r + 1, mode="reflect")
        a = var_g / (var_g + eps)
        b = mean_p * (1.0 - a)
        mean_a = uniform_filter(a, size=2 * r + 1, mode="reflect")
        mean_b = uniform_filter(b, size=2 * r + 1, mode="reflect")
        return mean_a * g + mean_b
    else:
        mean_g = uniform_filter(g, size=2 * r + 1, mode="reflect")
        mean_p = uniform_filter(p, size=2 * r + 1, mode="reflect")
        mean_gp = uniform_filter(g * p, size=2 * r + 1, mode="reflect")
        mean_gg = uniform_filter(g * g, size=2 * r + 1, mode="reflect")
        cov_gp = mean_gp - mean_g * mean_p          # cov(I, p)
        var_g = np.maximum(mean_gg - mean_g * mean_g, 0.0)
        a = cov_gp / (var_g + eps)
        b = mean_p - a * mean_g
        mean_a = uniform_filter(a, size=2 * r + 1, mode="reflect")
        mean_b = uniform_filter(b, size=2 * r + 1, mode="reflect")
        return mean_a * g + mean_b
