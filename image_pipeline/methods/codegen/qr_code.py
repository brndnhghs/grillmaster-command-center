"""Code-gen method — auto-split from codegen.py"""
from __future__ import annotations
import math
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from ...core.registry import method
from ...core.utils import save, mn, seed_all, get_font, W, H
from ...core.animation import capture_frame


@method(
    inputs={},id="09", name="QR Code", category="codegen",
         tags=["qr", "code", "fast", "animation", "expanded"],
         params={
    "content": {"content": True, "description": "text content to encode as QR", "default": "HELLO"},
    "anim_mode": {"description": "QR animation mode", "choices": ["none", "rotate_pulse", "mask_morph", "color_sweep"], "default": "none"},
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

    def _render_centered(qr_img, scale_factor=0.7):
        qw, qh = qr_img.size
        base_scale = min(W * scale_factor / qw, H * scale_factor / qh)
        dw = max(1, int(qw * base_scale))
        dh = max(1, int(qh * base_scale))
        qr_img = qr_img.resize((dw, dh), Image.LANCZOS)
        img = Image.new("RGB", (W, H), (255, 255, 255))
        x = (W - dw) // 2
        y = (H - dh) // 2
        img.paste(qr_img, (x, y))
        return img

    # ── Color sweep: QR fill color oscillates — modules change hue ──
    if anim_mode == "color_sweep":
        hue = int(128 + 126 * math.sin(t * 3.71 * anim_speed))
        q = qrcode.QRCode(box_size=DEFAULT_BOX, border=BORDER, mask_pattern=None)
        q.add_data(content)
        q.make()
        qr_img = q.make_image(fill_color=(hue, 255 - hue, 128), back_color="white").convert("RGB")
        img = _render_centered(qr_img)
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
        img_a = _render_centered(qr_a)
        qr_b = _make_qr(mask_pattern=idx_b)
        img_b = _render_centered(qr_b)

        img = Image.blend(img_a, img_b, fade)
        arr = np.array(img).astype(np.float32) / 255.0
        capture_frame("09", arr)
        save(img, mn(9, "qr-mask-morph"), out_dir)
        return

    # ── Rotate pulse ──
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
        arr = np.array(img).astype(np.float32) / 255.0
        capture_frame("09", arr)
        save(img, mn(9, "qr-rotate-pulse"), out_dir)
        return

    # ── Static (anim_mode == "none") ──
    qr_img = _make_qr()
    img = _render_centered(qr_img)
    arr = np.array(img).astype(np.float32) / 255.0
    capture_frame("09", arr)
    save(img, mn(9, "qr"), out_dir)