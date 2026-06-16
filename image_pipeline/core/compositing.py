"""
53 blend modes + layout compositing.
Extracted and cleaned up from the monolithic pipeline.
"""
from __future__ import annotations
import math
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .registry import get_meta

# ── Helpers ────────────────────────────────────────────────────────────


def _norm(arr: np.ndarray) -> np.ndarray:
    lo, hi = arr.min(), arr.max()
    return (arr - lo) / (hi - lo + 1e-8)


def _safe_div(a: np.ndarray, b: np.ndarray, fallback=0.0) -> np.ndarray:
    out = np.zeros_like(a, dtype=np.float32)
    mask = b != 0
    out[mask] = a[mask] / b[mask]
    out[~mask] = fallback
    return out


def load_as_array(path: Path) -> np.ndarray:
    img = Image.open(str(path)).convert("RGB")
    return np.array(img, dtype=np.float32) / 255.0


def resize_to_target(img: np.ndarray, tw: int, th: int) -> np.ndarray:
    h, w = img.shape[:2]
    if w == tw and h == th:
        return img
    return (
        np.array(
            Image.fromarray((img * 255).astype(np.uint8)).resize((tw, th), Image.LANCZOS),
            dtype=np.float32,
        )
        / 255.0
    )


def save(arr: np.ndarray | Image.Image, path: Path):
    if isinstance(arr, np.ndarray):
        arr = (
            (arr.clip(0, 1) * 255).astype(np.uint8)
            if arr.max() <= 1
            else arr.clip(0, 255).astype(np.uint8)
        )
        img = Image.fromarray(arr)
    else:
        img = arr
    img.save(str(path))


# ── 53 Blend Modes ────────────────────────────────────────────────────


def blend_two(a: np.ndarray, b: np.ndarray, mode: str) -> np.ndarray:
    """Blend two float32 RGB images [0,1]. a = base, b = top."""

    # ── Normal ──
    if mode == "normal":
        return b.copy()
    elif mode == "dissolve":
        mask = np.random.random(a.shape) > 0.5
        return np.where(mask, b, a)

    # ── Photoshop Darken ──
    elif mode == "multiply":
        return a * b
    elif mode == "color-burn":
        return 1 - _safe_div(1 - a, b + 1e-8)
    elif mode == "linear-burn":
        return np.clip(a + b - 1, 0, 1)
    elif mode == "darken-only":
        return np.minimum(a, b)
    elif mode == "darker-color":
        la = 0.299 * a[:, :, 0] + 0.587 * a[:, :, 1] + 0.114 * a[:, :, 2]
        lb = 0.299 * b[:, :, 0] + 0.587 * b[:, :, 1] + 0.114 * b[:, :, 2]
        return np.where(lb[..., None] < la[..., None], b, a)

    # ── Photoshop Lighten ──
    elif mode == "screen":
        return 1 - (1 - a) * (1 - b)
    elif mode == "color-dodge":
        return _safe_div(a, 1 - b + 1e-8)
    elif mode in ("linear-dodge", "addition"):
        return np.clip(a + b, 0, 1)
    elif mode == "lighten-only":
        return np.maximum(a, b)
    elif mode == "lighter-color":
        la = 0.299 * a[:, :, 0] + 0.587 * a[:, :, 1] + 0.114 * a[:, :, 2]
        lb = 0.299 * b[:, :, 0] + 0.587 * b[:, :, 1] + 0.114 * b[:, :, 2]
        return np.where(lb[..., None] > la[..., None], b, a)

    # ── Contrast ──
    elif mode == "overlay":
        mask = a < 0.5
        return np.where(mask, 2 * a * b, 1 - 2 * (1 - a) * (1 - b))
    elif mode == "hard-light":
        mask = b < 0.5
        return np.where(mask, 2 * a * b, 1 - 2 * (1 - a) * (1 - b))
    elif mode == "soft-light-pegtop":
        return (1 - 2 * b) * a**2 + 2 * b * a
    elif mode == "soft-light-w3c":
        mask = b <= 0.5
        g = np.where(a <= 0.25, ((16 * a - 12) * a + 4) * a, np.sqrt(a))
        out = np.where(mask, a - (1 - 2 * b) * a * (1 - a), a + (2 * b - 1) * (g - a))
        return np.clip(out, 0, 1)
    elif mode == "vivid-light":
        mask = b > 0.5
        out = np.where(
            mask, 1 - _safe_div(1 - a, 2 * (b - 0.5) + 1e-8), _safe_div(a, 1 - 2 * b + 1e-8)
        )
        return np.clip(out, 0, 1)
    elif mode == "linear-light":
        return np.clip(a + 2 * b - 1, 0, 1)
    elif mode == "pin-light":
        mask = b > 0.5
        out = np.where(mask, np.maximum(a, 2 * (b - 0.5)), np.minimum(a, 2 * b))
        return np.clip(out, 0, 1)
    elif mode == "hard-mix":
        c = blend_two(a, b, "vivid-light")
        return (c > 0.5).astype(np.float32)
    elif mode == "hard-overlay":
        return np.clip(2 * blend_two(a, b, "overlay") - 0.5, 0, 1)

    # ── Comparative ──
    elif mode == "difference":
        return np.abs(a - b)
    elif mode == "exclusion":
        return 0.5 - 2 * (a - 0.5) * (b - 0.5)
    elif mode == "subtract":
        return np.clip(a - b, 0, 1)
    elif mode == "divide":
        return np.clip(_safe_div(a, b + 1e-8), 0, 1)

    # ── Krita/GIMP Exotics ──
    elif mode == "grain-extract":
        return np.clip(a - b + 0.5, 0, 1)
    elif mode == "grain-merge":
        return np.clip(a + b - 0.5, 0, 1)
    elif mode == "negation":
        return 1 - np.abs(1 - a - b)
    elif mode == "phoenix":
        return np.clip(np.minimum(a, b) - np.maximum(a, b) + 1, 0, 1)
    elif mode == "reflect":
        return np.clip(_safe_div(a**2, 1 - b + 1e-8), 0, 1)
    elif mode == "glow":
        return np.clip(_safe_div(b**2, 1 - a + 1e-8), 0, 1)
    elif mode == "freeze":
        return np.clip(a * (1 - np.sqrt(np.clip(1 - b, 0, 1))), 0, 1)
    elif mode == "heat":
        return np.clip(1 - (1 - a) * np.sqrt(b), 0, 1)
    elif mode == "stamp":
        return (a > 0.5).astype(np.float32) * b

    # ── Mathematical ──
    elif mode == "arithmetic":
        return (a + b) / 2
    elif mode == "geometric-mean":
        return np.sqrt(a * b + 1e-8)
    elif mode == "harmonic-mean":
        return _safe_div(2 * a * b, a + b + 1e-8)
    elif mode == "rms":
        return np.sqrt((a**2 + b**2) / 2)
    elif mode == "signed-difference":
        return np.clip(a - b + 0.5, 0, 1)
    elif mode == "soft-subtract":
        return np.clip((a - b + 1) / 2, 0, 1)
    elif mode == "cross-fade":
        xx, yy = np.meshgrid(
            np.linspace(0, 2 * np.pi, a.shape[1]), np.linspace(0, 2 * np.pi, a.shape[0])
        )
        t = (np.sin(xx * 3 + yy * 2) + 1) / 2
        return a * (1 - t[..., None]) + b * t[..., None]

    # ── Porter-Duff Alpha ──
    elif mode == "source-over":
        return b.copy()
    elif mode == "destination-over":
        return a.copy()
    elif mode in ("source-in", "destination-in"):
        return np.clip(b * a, 0, 1)
    elif mode == "source-out":
        return np.clip(b * (1 - a), 0, 1)
    elif mode == "destination-out":
        return np.clip(a * (1 - b), 0, 1)
    elif mode == "source-atop":
        return np.clip(b * a + a * (1 - b), 0, 1)
    elif mode == "destination-atop":
        return np.clip(a * b + b * (1 - a), 0, 1)
    elif mode == "xor":
        return np.clip(a * (1 - b) + b * (1 - a), 0, 1)
    elif mode in ("lighter", "linear-dodge-alt"):
        return np.clip(a + b, 0, 1)
    elif mode in ("darker", "linear-burn-alt"):
        return np.clip(a + b - 1, 0, 1)

    # ── Colour Space ──
    elif mode == "luminosity":
        luma_top = 0.299 * b[:, :, 0] + 0.587 * b[:, :, 1] + 0.114 * b[:, :, 2]
        luma_base = 0.299 * a[:, :, 0] + 0.587 * a[:, :, 1] + 0.114 * a[:, :, 2]
        ratio = luma_top / (luma_base + 1e-8)
        return np.clip(a * ratio[..., None], 0, 1)
    elif mode in ("hue", "color"):
        return np.clip(a * 0.3 + b * 0.7, 0, 1)

    # ── Creative ──
    elif mode == "heatmap":
        intensity = (a[:, :, 0] + a[:, :, 1] + a[:, :, 2]) / 3
        h = (1 - intensity) * 0.66
        h_6 = h * 6
        i = h_6.astype(np.int32) % 6
        f = h_6 - np.floor(h_6)
        v_val = b[..., :3] if b.shape[-1] >= 3 else b[..., None]
        v_val = v_val[:, :, 0] if v_val.ndim == 3 else v_val
        v_val = np.clip(v_val, 0, 1)
        p = v_val * (1 - 1)  # saturation = 1
        q = v_val * (1 - f)
        t = v_val * (1 - (1 - f))
        out = np.zeros(a.shape, dtype=np.float32)
        for ci in range(3):
            out[:, :, ci] = np.choose(i, [v_val, q, p, p, t, v_val])
        return np.clip(out, 0, 1)
    elif mode == "ascii-quantize":
        gray = (a[:, :, 0] + a[:, :, 1] + a[:, :, 2]) / 3
        quantized = np.floor(gray * 8) / 7
        return np.stack([quantized] * 3, axis=-1) * b
    elif mode == "edge-burn":
        gray_b = (b[:, :, 0] + b[:, :, 1] + b[:, :, 2]) / 3
        edges = np.abs(np.gradient(gray_b)[0]) + np.abs(np.gradient(gray_b)[1])
        edge_mask = (edges > 0.15).astype(np.float32)
        return np.clip(a * (1 - edge_mask[..., None] * 0.7), 0, 1)

    # ── Simple blenders ──
    elif mode in ("blend", "overlay-simple"):
        return (a + b) / 2
    elif mode in ("diff", "min"):
        return np.minimum(a, b)
    elif mode == "max":
        return np.maximum(a, b)

    else:
        raise ValueError(f"Unknown blend mode: {mode}")


# ── All modes listing ─────────────────────────────────────────────────

BLEND_MODES = [
    "normal",
    "dissolve",
    "multiply",
    "color-burn",
    "linear-burn",
    "darken-only",
    "darker-color",
    "screen",
    "color-dodge",
    "linear-dodge",
    "lighten-only",
    "lighter-color",
    "overlay",
    "hard-light",
    "soft-light-pegtop",
    "soft-light-w3c",
    "vivid-light",
    "linear-light",
    "pin-light",
    "hard-mix",
    "hard-overlay",
    "difference",
    "exclusion",
    "subtract",
    "divide",
    "grain-extract",
    "grain-merge",
    "negation",
    "phoenix",
    "reflect",
    "glow",
    "freeze",
    "heat",
    "stamp",
    "arithmetic",
    "geometric-mean",
    "harmonic-mean",
    "rms",
    "signed-difference",
    "soft-subtract",
    "cross-fade",
    "source-over",
    "destination-over",
    "source-in",
    "destination-in",
    "source-out",
    "destination-out",
    "source-atop",
    "destination-atop",
    "xor",
    "lighter",
    "darker",
    "luminosity",
    "hue",
    "color",
    "heatmap",
    "ascii-quantize",
    "edge-burn",
    "blend",
    "dif",
    "max",
]


# ── Layout modes ──────────────────────────────────────────────────────


def composite_images(
    paths: list[Path], mode: str, out: Path, cols: int = 3, label: str | None = None
):
    """Composite multiple images. Supports layout modes and pairwise blend."""
    imgs = [load_as_array(p) for p in paths]
    n = len(imgs)
    tw, th = imgs[0].shape[1], imgs[0].shape[0]
    imgs = [resize_to_target(i, tw, th) for i in imgs]

    if mode in ("hstack", "vstack", "grid", "mosaic"):
        if mode == "hstack":
            ims = [Image.fromarray((i * 255).astype(np.uint8)) for i in imgs]
            ws = [im.width for im in ims]
            mh = max(im.height for im in ims)
            c = Image.new("RGB", (sum(ws), mh), (10, 10, 18))
            x = 0
            for im in ims:
                c.paste(im, (x, (mh - im.height) // 2))
                x += im.width
            c.save(str(out))
        elif mode == "vstack":
            ims = [Image.fromarray((i * 255).astype(np.uint8)) for i in imgs]
            hs = [im.height for im in ims]
            mw = max(im.width for im in ims)
            c = Image.new("RGB", (mw, sum(hs)), (10, 10, 18))
            y = 0
            for im in ims:
                c.paste(im, ((mw - im.width) // 2, y))
                y += im.height
            c.save(str(out))
        elif mode in ("grid", "mosaic"):
            nc = min(cols, n)
            nr = math.ceil(n / nc)
            cw, ch = imgs[0].shape[1], imgs[0].shape[0]
            bdr = 4 if mode == "mosaic" else 2
            canvas = Image.new(
                "RGB", (nc * cw + (nc + 1) * bdr, nr * ch + (nr + 1) * bdr), (20, 20, 30)
            )
            draw = ImageDraw.Draw(canvas)
            try:
                font = ImageFont.truetype("/System/Library/Fonts/Menlo.ttc", 10)
            except OSError:
                font = ImageFont.load_default()
            for idx, img_arr in enumerate(imgs):
                row, col = idx // nc, idx % nc
                x = bdr + col * (cw + bdr)
                y = bdr + row * (ch + bdr)
                canvas.paste(Image.fromarray((img_arr * 255).astype(np.uint8)), (x, y))
                if mode == "mosaic":
                    draw.text((x + 4, y + 4), label or paths[idx].stem, fill=(180, 180, 200), font=font)
            canvas.save(str(out))
        print(f"  ✓ {out.name} ({mode})")
        return

    # Iterative pairwise blend
    result = imgs[0].copy()
    for i in range(1, n):
        result = blend_two(result, imgs[i], mode)
    save(result, out)
    size_kb = out.stat().st_size // 1024
    print(f"  ✓ {out.name}  ({size_kb} KB, {mode}, {n} images)")