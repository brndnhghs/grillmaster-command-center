"""JPEG Encode — JPEG-compress an image at a configurable quality.

This is a filter node: it JPEG-encodes the input image, then decodes it
back, so downstream nodes see the actual compression artifacts.  The JPEG
file is also written to disk alongside the usual PNG output.

Useful for:
  • Previewing how a frame will look at different JPEG qualities.
  • Saving a JPEG copy at a specific quality for web/email/share.
  • Estimating the file size of a compressed frame.
  • Bottleneck diagnosis — insert between camera and render target to
    isolate encode cost from pipeline cost.

When ``quality=0`` the node is a no-op: the image passes through untouched
and a JPEG is NOT written (avoids compressing to garbage).
"""

from __future__ import annotations
from pathlib import Path

import numpy as np

from ...core.registry import method
from ...core.utils import save, mn, W, H


@method(
    id="__jpeg_encode__",
    name="JPEG Encode",
    category="io",
    tags=["io", "encode", "jpeg", "compress", "save", "export"],
    new_image_contract=True,
    inputs={"image_in": "IMAGE"},
    outputs={"image": "IMAGE"},
    params={
        "quality": {
            "description": "JPEG quality 1–100 (lower = smaller file, more artifacts). "
                           "0 = passthrough (no compression, no JPEG written).",
            "min": 0,
            "max": 100,
            "default": 85,
        },
    },
    description="JPEG-compress an image at configurable quality.  Shows compression "
                "artefacts downstream; saves a .jpg to disk.",
)
def method_jpeg_encode(out_dir: Path, seed: int, params=None):
    import cv2

    params = params or {}
    quality = int(params.get("quality", 85))

    # Load wired input (the executor provides it via params["input_image"])
    arr = None
    wired_path = params.get("input_image", "")
    if wired_path:
        try:
            from ...core.utils import load_input
            arr = load_input(wired_path, int(W), int(H))
        except Exception:
            arr = None

    if arr is None:
        # Fallback: dark frame with message
        from PIL import Image as _PIL, ImageDraw as _Draw
        from ...core.utils import get_font
        img = _PIL.new("RGB", (int(W), int(H)), (16, 8, 8))
        d = _Draw.Draw(img)
        font = get_font(max(10, int(H) // 20))
        txt = "JPEG Encode: no input"
        tw = d.textlength(txt, font=font)
        d.text(((int(W) - tw) // 2, int(H) // 2 - 10), txt,
               fill=(140, 120, 120), font=font)
        arr = np.array(img, dtype=np.float32) / 255.0

    # ── JPEG encode / decode round-trip ────────────────────────────
    if quality > 0:
        # float32 [0,1] → uint8 → JPEG bytes → decode → float32
        u8 = (arr.clip(0, 1) * 255).astype(np.uint8)
        ok, enc = cv2.imencode(
            ".jpg", cv2.cvtColor(u8, cv2.COLOR_RGB2BGR),
            [int(cv2.IMWRITE_JPEG_QUALITY), quality],
        )
        if ok:
            # Save JPEG to disk alongside the PNG
            jpeg_path = out_dir / (str(mn(0, "JPEG")) + ".jpg")
            jpeg_path.write_bytes(enc.tobytes())
            print(f"  ✓ JPEG@{quality}  ({len(enc) // 1024} KB)")

            # Decode back so downstream nodes see actual compression
            dec = cv2.imdecode(enc, cv2.IMREAD_COLOR)
            if dec is not None:
                arr = cv2.cvtColor(dec, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0

    # Normal PNG save (keeps the chain consistent)
    save(arr, mn(0, "JPEG Encode"), out_dir)
    return arr
