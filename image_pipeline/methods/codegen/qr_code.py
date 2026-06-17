"""Code-gen method — auto-split from codegen.py"""
from __future__ import annotations
import math
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from ...core.registry import method
from ...core.utils import save, mn, seed_all, get_font, W, H
from ...core.animation import capture_frame


@method(id="09", name="QR Code", category="codegen",
         tags=["qr", "code", "fast", "animation", "expanded"],
         params={
    "time": {"description": "animation phase (0 to 2pi)", "min": 0.0, "max": 6.28, "default": 0.0},
    "content": {"description": "text content to encode as QR", "default": "HELLO"},
    "anim_mode": {"description": "QR animation mode", "choices": ["none", "rotate_pulse", "mask_morph", "ring_pulse", "color_sweep"], "default": "none"},
    "anim_speed": {"description": "animation speed multiplier", "min": 0.0, "max": 2.0, "default": 0.25},
})
def method_09_qr_code(out_dir: Path, seed: int, params=None):
    if params is None:
        params = {}
    t = float(params.get("time", 0.0))
    seed_all(seed)

    content = params.get("content", "HELLO")
    anim_mode = params.get("anim_mode", "none")
    anim_speed = float(params.get("anim_speed", 0.25))

    try:
        import qrcode
        _has_qrcode = True
    except ImportError:
        _has_qrcode = False

    if not _has_qrcode:
        img = Image.new("RGB", (W, H), (255, 255, 255))
        draw = ImageDraw.Draw(img)
        font = get_font(20)
        draw.text((W // 2 - 100, H // 2 - 10), "qrcode lib missing", fill=(200, 0, 0), font=font)
        arr = np.array(img).astype(np.float32) / 255.0
        capture_frame("09", arr)
        save(img, mn(9, "qr"), out_dir)
        return

    DEFAULT_BOX = 20
    BORDER = 4

    def _make_qr(mask_pattern=None, box_size=None):
        q = qrcode.QRCode(box_size=box_size or DEFAULT_BOX, border=BORDER, mask_pattern=mask_pattern)
        q.add_data(content)
        q.make()
        return q.make_image(fill_color="black", back_color="white").convert("RGB")

    def _draw_ring(draw_obj, center, radius, width=2, color=(40, 40, 40)):
        draw_obj.ellipse([center[0] - radius, center[1] - radius,
                          center[0] + radius, center[1] + radius],
                         outline=color, width=width)

    def _render_centered(qr_img, scale_factor=0.7, fallback_scale=None):
        qw, qh = qr_img.size
        base_scale = min(W * scale_factor / qw, H * scale_factor / qh) if fallback_scale is None else fallback_scale
        dw = max(1, int(qw * base_scale))
        dh = max(1, int(qh * base_scale))
        qr_img = qr_img.resize((dw, dh), Image.LANCZOS)
        img = Image.new("RGB", (W, H), (255, 255, 255))
        x = (W - dw) // 2
        y = (H - dh) // 2
        img.paste(qr_img, (x, y))
        return img, x, y, dw, dh

    # ── Ring animation — common across modes ──
    ring_pulse_t = t * anim_speed
    if anim_mode == "ring_pulse":
        ring_r_raw = W * (0.08 + 0.40 * (0.5 + 0.5 * math.sin(ring_pulse_t * 2.37)))
        ring_w_raw = 2.0 + 7.0 * (0.5 + 0.5 * math.sin(ring_pulse_t * 1.3))
        ring_r = int(ring_r_raw)
        ring_w = int(ring_w_raw)
        ring_c = (40, 40, 40)
        # Draw ring only (no QR)
        img = Image.new("RGB", (W, H), (255, 255, 255))
        draw = ImageDraw.Draw(img)
        _draw_ring(draw, (W // 2, H // 2), ring_r, ring_w, ring_c)
        # Crosshair decoration
        draw.line([(W // 2 - ring_r - 10, H // 2), (W // 2 + ring_r + 10, H // 2)],
                  fill=(200, 200, 200), width=1)
        draw.line([(W // 2, H // 2 - ring_r - 10), (W // 2, H // 2 + ring_r + 10)],
                  fill=(200, 200, 200), width=1)
        arr = np.array(img).astype(np.float32) / 255.0
        capture_frame("09", arr)
        save(img, mn(9, "qr-ring-pulse"), out_dir)
        return

    # ── Color sweep: QR fill color oscillates — modules change hue ──
    if anim_mode == "color_sweep":
        hue = int(128 + 127 * math.sin(t * 2.37 * anim_speed))  # 1→255, avoids 0 plateau
        qr_img = _make_qr()
        q = qrcode.QRCode(box_size=DEFAULT_BOX, border=BORDER, mask_pattern=None)
        q.add_data(content)
        q.make()
        qr_img = q.make_image(fill_color=(hue, 255 - hue, 128), back_color="white").convert("RGB")
        qw, qh = qr_img.size
        base_scale = min(W * 0.7 / qw, H * 0.7 / qh)
        dw = max(1, int(qw * base_scale))
        dh = max(1, int(qh * base_scale))
        qr_img = qr_img.resize((dw, dh), Image.LANCZOS)
        img = Image.new("RGB", (W, H), (255, 255, 255))
        x = (W - dw) // 2
        y = (H - dh) // 2
        img.paste(qr_img, (x, y))
        draw = ImageDraw.Draw(img)
        ring_r = dw // 2 + 8
        ring_w = max(1, int(2 + 2 * (0.5 + 0.5 * math.sin(t * 0.8 * anim_speed))))
        _draw_ring(draw, (W // 2, H // 2), ring_r, ring_w, color=(hue, 0, 80))
        arr = np.array(img).astype(np.float32) / 255.0
        capture_frame("09", arr)
        save(img, mn(9, "qr-color-sweep"), out_dir)
        return

    # ── Mask morph with cross-fade ──
    if anim_mode == "mask_morph":
        n_masks = 8
        raw_idx = (t / 6.28) * n_masks * anim_speed
        idx_a = int(raw_idx) % n_masks
        idx_b = (idx_a + 1) % n_masks
        fade = raw_idx - int(raw_idx)

        qr_a = _make_qr(mask_pattern=idx_a)
        img_a, _, _, _, _ = _render_centered(qr_a, scale_factor=0.7)
        qr_b = _make_qr(mask_pattern=idx_b)
        img_b, _, _, _, _ = _render_centered(qr_b, scale_factor=0.7)

        img = Image.blend(img_a, img_b, fade)
        draw = ImageDraw.Draw(img)
        qw, qh = qr_a.size
        base_scale = min(W * 0.7 / qw, H * 0.7 / qh)
        dw = int(qw * base_scale)
        ring_r = dw // 2 + 8
        ring_w = max(1, int(2 + 2 * (0.5 + 0.5 * math.sin(t * anim_speed))))
        _draw_ring(draw, (W // 2, H // 2), ring_r, ring_w)
        arr = np.array(img).astype(np.float32) / 255.0
        capture_frame("09", arr)
        save(img, mn(9, "qr-mask-morph"), out_dir)
        return

    # ── Rotate pulse (default for anim_mode not none) ──
    if anim_mode == "rotate_pulse":
        qr_img = _make_qr()
        qw, qh = qr_img.size
        base_scale = min(W * 0.55 / qw, H * 0.55 / qh)
        scale = base_scale * (0.85 + 0.15 * math.sin(t * 2 * anim_speed))
        dw = max(1, int(qw * scale))
        dh = max(1, int(qh * scale))
        qr_img = qr_img.resize((dw, dh), Image.LANCZOS)
        angle = t * (360.0 / 6.28) * anim_speed
        diag = int(math.sqrt(W * W + H * H)) + max(dw, dh)
        expanded = Image.new("RGB", (diag, diag), (255, 255, 255))
        px = (diag - dw) // 2
        py = (diag - dh) // 2
        expanded.paste(qr_img, (px, py))
        rotated = expanded.rotate(angle, center=(diag // 2, diag // 2),
                                  fillcolor=(255, 255, 255))
        cx = diag // 2
        cy = diag // 2
        crop_x = cx - W // 2
        crop_y = cy - H // 2
        img = rotated.crop((crop_x, crop_y, crop_x + W, crop_y + H))
        draw = ImageDraw.Draw(img)
        # Animated ring
        ring_r = W // 2 - 10 + int(5 * math.sin(t * anim_speed))
        ring_w = max(1, int(2 + 2 * (0.5 + 0.5 * math.sin(t * 1.3 * anim_speed))))
        _draw_ring(draw, (W // 2, H // 2), ring_r, ring_w)
        arr = np.array(img).astype(np.float32) / 255.0
        capture_frame("09", arr)
        save(img, mn(9, "qr-rotate-pulse"), out_dir)
        return

    # ── Static (anim_mode == "none") ──
    qr_img = _make_qr()
    img, _, _, _, _ = _render_centered(qr_img, scale_factor=0.7)
    arr = np.array(img).astype(np.float32) / 255.0
    capture_frame("09", arr)
    save(img, mn(9, "qr"), out_dir)