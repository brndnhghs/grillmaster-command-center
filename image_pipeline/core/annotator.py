"""
Output annotator — overlays method params + ranges onto generated images.

Used by the --demo flag. Reads the @method meta from registry and renders
the specification directly onto the image.
"""
from __future__ import annotations
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .registry import get_meta, get_all


def annotate_image(method_id: str, image_path: Path, out_path: Path | None = None) -> Path:
    """Overlay method param documentation onto a generated image.

    Layout
    ┌─────────────────────────────────┐
    │  Image content (method output)  │
    │                                 │
    │  ── overlay bar ──             │
    │  #07 Fractal (Mandelbrot)      │
    │  iterations: 50–300 (150)      │
    │  viewport: [-2,1]×[-1.5,1.5]   │
    │  colormap: sin-loop            │
    └─────────────────────────────────┘
    """
    meta = get_meta(method_id)
    if not meta:
        raise ValueError(f"Unknown method: {method_id}")

    img = Image.open(str(image_path)).convert("RGB")
    arr = np.array(img)
    h, w = arr.shape[:2]

    # ── Build overlay text ──
    lines = [f"#{meta.id}  {meta.name}", f"tags: {', '.join(meta.tags)}"]

    if meta.params:
        for pname, pinfo in meta.params.items():
            desc = pinfo.get("description", pname)
            default = pinfo.get("default", "?")
            if "min" in pinfo and "max" in pinfo:
                lines.append(f"  {desc}: {pinfo['min']}–{pinfo['max']}  (default {default})")
            else:
                lines.append(f"  {desc}: default {default}")
    else:
        lines.append("  (no tunable parameters)")

    # ── Render overlay ──
    overlay_h = 20 + len(lines) * 18
    canvas = Image.new("RGB", (w, h + overlay_h), (12, 12, 22))
    canvas.paste(img, (0, 0))

    draw = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.truetype("/System/Library/Fonts/Menlo.ttc", 13)
        small = ImageFont.truetype("/System/Library/Fonts/Menlo.ttc", 11)
    except OSError:
        font = ImageFont.load_default()
        small = font

    # Method title bar
    draw.rectangle([0, h, w, h + overlay_h], fill=(18, 18, 30))
    draw.line([(0, h), (w, h)], fill=(40, 40, 60), width=1)

    draw.text((12, h + 4), lines[0], fill=(180, 160, 120), font=font)
    draw.text((12, h + 22), lines[1], fill=(100, 100, 120), font=small)

    y_offset = h + overlay_h - 18 * len(lines[2:]) - 6
    for line in lines[2:]:
        draw.text((12, y_offset), line, fill=(140, 140, 170), font=small)
        y_offset += 18

    out = out_path or image_path
    canvas.save(str(out))
    return out


def annotate_batch(out_dir: Path, method_ids: list[str], suffix: str = "-demo") -> list[Path]:
    """Annotate all generated images for the given method IDs."""
    results = []
    for mid in method_ids:
        meta = get_meta(mid)
        if not meta:
            continue
        src = out_dir / meta.filename()
        if not src.exists():
            continue
        dst = out_dir / f"{meta.id}-{meta.name.lower().replace(' ','-')}{suffix}.png"
        annotate_image(mid, src, dst)
        results.append(dst)
        print(f"  ✓ demo {dst.name}")
    return results