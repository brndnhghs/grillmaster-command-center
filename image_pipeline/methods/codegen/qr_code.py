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
    "anim_mode": {"description": "QR animation mode", "choices": ["none", "rotate_pulse", "mask_morph"], "default": "rotate_pulse"},
    "anim_speed": {"description": "animation speed multiplier", "min": 0.0, "max": 2.0, "default": 0.25},
})
def method_09_qr_code(out_dir: Path, seed: int, params=None):
    if params is None:
        params = {}
    t = float(params.get("time", 0.0))
    seed_all(seed)

    content = params.get("content", "HELLO")
    anim_mode = params.get("anim_mode", "rotate_pulse")
    anim_speed = float(params.get("anim_speed", 0.25))

    import qrcode

    BOX_SIZE = 20
    BORDER = 4

    def _make_qr(mask_pattern=None):
        q = qrcode.QRCode(box_size=BOX_SIZE, border=BORDER, mask_pattern=mask_pattern)
        q.add_data(content)
        q.make()
        return q.make_image(fill_color="black", back_color="white").convert("RGB")

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
        ring_r = W // 2 - 10
        ring_color = (40, 40, 40)
        draw.ellipse([W // 2 - ring_r, H // 2 - ring_r,
                      W // 2 + ring_r, H // 2 + ring_r],
                     outline=ring_color, width=2)
        arr = np.array(img).astype(np.float32) / 255.0
        capture_frame("09", arr)
        save(img, mn(9, "qr-rotate-pulse"), out_dir)

    elif anim_mode == "mask_morph":
        mask_idx = int((t / 6.28) * 8 * anim_speed) % 8
        q2 = qrcode.QRCode(box_size=BOX_SIZE, border=BORDER, mask_pattern=mask_idx)
        q2.add_data(content)
        q2.make()
        qr_img = q2.make_image(fill_color="black", back_color="white").convert("RGB")
        qw, qh = qr_img.size
        scale = min(W * 0.7 / qw, H * 0.7 / qh)
        dw = max(1, int(qw * scale))
        dh = max(1, int(qh * scale))
        qr_img = qr_img.resize((dw, dh), Image.LANCZOS)
        img = Image.new("RGB", (W, H), (255, 255, 255))
        x = (W - dw) // 2
        y = (H - dh) // 2
        img.paste(qr_img, (x, y))
        draw = ImageDraw.Draw(img)
        font = get_font(24)
        draw.text((20, 20), f"Mask {mask_idx}", fill=(60, 60, 60), font=font)
        arr = np.array(img).astype(np.float32) / 255.0
        capture_frame("09", arr)
        save(img, mn(9, f"qr-mask-{mask_idx}"), out_dir)

    else:
        qr_img = _make_qr()
        qw, qh = qr_img.size
        scale = min(W * 0.7 / qw, H * 0.7 / qh)
        dw = max(1, int(qw * scale))
        dh = max(1, int(qh * scale))
        qr_img = qr_img.resize((dw, dh), Image.LANCZOS)
        img = Image.new("RGB", (W, H), (255, 255, 255))
        x = (W - dw) // 2
        y = (H - dh) // 2
        img.paste(qr_img, (x, y))
        arr = np.array(img).astype(np.float32) / 255.0
        capture_frame("09", arr)
        save(img, mn(9, "qr"), out_dir)

