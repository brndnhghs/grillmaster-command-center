"""Code-gen method - auto-split from codegen.py"""
from __future__ import annotations
import colorsys
import math
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from ...core.registry import method
from ...core.utils import save, mn, seed_all, W, H
from ...core.animation import capture_frame

# --- 30 SVG Vector ---

@method(id="30", name="SVG Vector", category="codegen",
description="SVG Vector — generative node.",
         tags=["vector", "svg", "geometric", "animation"],
         params={
             "pattern": {"description": "SVG pattern type", "choices": ["grid", "circles", "stars", "waves", "mandala"], "default": "grid"},
             "stroke_width": {"description": "stroke width", "min": 1, "max": 10, "default": 2},
             "fill": {"description": "fill shapes with color", "default": True},
             "anim_mode": {"description": "animation mode", "choices": ["none", "animate"], "default": "animate"},
             "anim_speed": {"description": "animation speed multiplier", "min": 0.1, "max": 5.0, "default": 1.0},
         })
def method_30_svg_vector(out_dir: Path, seed: int, params=None):
    """Render geometric SVG patterns using xml.etree.ElementTree.

    Generates both an SVG file and a PNG raster for each of 5 geometric
    patterns (grid, circles, stars, waves, mandala). All patterns animate
    via structural parameters (position, size, rotation, phase) modulated
    by the animation time.

    Args:
        out_dir: Output directory for the generated image.
        seed: Random seed for deterministic output.
        params: Dict with keys:
            pattern: SVG pattern type (grid/circles/stars/waves/mandala)
            stroke_width: stroke width in pixels (1-10)
            fill: fill shapes with color
            time: animation time in radians (0-6.28)
            anim_mode: animation mode (none/animate)
            anim_speed: animation speed multiplier (0.1-5.0)
    """
    if params is None:
        params = {}
    t = float(params.get("time", 0.0))
    anim_mode = params.get("anim_mode", "animate")
    anim_speed = float(params.get("anim_speed", 1.0))
    seed_all(seed)

    pattern = params.get("pattern", "grid")
    stroke_width = int(params.get("stroke_width", 2))
    fill_enabled = params.get("fill", True)

    if anim_mode == "none":
        t = 0.0
    else:
        t = t * anim_speed

    svg = ET.Element("svg", {
        "xmlns": "http://www.w3.org/2000/svg",
        "width": str(W), "height": str(H),
        "viewBox": f"0 0 {W} {H}",
    })
    ET.SubElement(svg, "rect", {"width": "100%", "height": "100%", "fill": "#0a0a12"})

    cx, cy = W / 2.0, H / 2.0

    def _add_elem(tag, **attrs):
        ET.SubElement(svg, tag, attrs)

    def _hsv_to_hex(h, s, v):
        r, g, b = colorsys.hsv_to_rgb(h % 1.0, s, v)
        return f"#{int(r*255):02x}{int(g*255):02x}{int(b*255):02x}"

    sw = str(stroke_width)
    rot_angle = t * 30.0

    if pattern == "grid":
        spacing = 40 + 20 * math.sin(t * 0.5)
        n_cols = int(W / spacing) + 2
        n_rows = int(H / spacing) + 2
        for r_idx in range(n_rows):
            for c_idx in range(n_cols):
                x = c_idx * spacing + (spacing * 0.5 * math.sin(t * 0.3 + r_idx * 0.5))
                y = r_idx * spacing + (spacing * 0.5 * math.cos(t * 0.4 + c_idx * 0.7))
                sz = spacing * 0.3 * (0.7 + 0.3 * math.sin(t + r_idx + c_idx))
                color = _hsv_to_hex((r_idx / max(1, n_rows) + t * 0.05) % 1.0, 0.7, 0.9)
                _add_elem("rect",
                    x=str(x), y=str(y), width=str(sz), height=str(sz),
                    stroke=color, stroke_width=sw,
                    fill=color if fill_enabled else "none",
                    transform=f"rotate({rot_angle + c_idx * 10} {x + sz / 2} {y + sz / 2})")
    elif pattern == "circles":
        n = 80
        for i in range(n):
            frac = i / n
            angle = frac * 2 * math.pi + t * 0.3
            radius = 50 + 350 * (0.5 + 0.5 * math.sin(frac * 3 + t * 0.5))
            x = cx + radius * math.cos(angle)
            y = cy + radius * math.sin(angle)
            r = 10 + 30 * (0.5 + 0.5 * math.sin(frac * 5 + t))
            color = _hsv_to_hex(frac + t * 0.02, 0.7, 0.9)
            _add_elem("circle",
                cx=str(x), cy=str(y), r=str(r),
                stroke=color, stroke_width=sw,
                fill=color if fill_enabled else "none",
                transform=f"rotate({rot_angle + i * 7} {x} {y})")
    elif pattern == "stars":
        n_stars = 40
        for i in range(n_stars):
            frac = i / n_stars
            angle = frac * 2 * math.pi + t * 0.2
            radius = 80 + 380 * (0.5 + 0.5 * math.sin(frac * 4 + t * 0.7))
            x = cx + radius * math.cos(angle)
            y = cy + radius * math.sin(angle)
            sz = 20 + 40 * (0.5 + 0.5 * math.sin(frac * 6 + t * 0.5))
            color = _hsv_to_hex(frac + t * 0.03, 0.8, 0.95)
            pts = []
            for k in range(10):
                r_k = sz if k % 2 == 0 else sz * 0.4
                a_k = k * math.pi / 5 + t * 0.1 + i * 0.3
                pts.append(f"{x + r_k * math.cos(a_k):.1f},{y + r_k * math.sin(a_k):.1f}")
            _add_elem("polygon",
                points=" ".join(pts),
                stroke=color, stroke_width=sw,
                fill=color if fill_enabled else "none")
    elif pattern == "waves":
        step = 8
        for row in range(0, H, step):
            pts = []
            for col in range(0, W, 4):
                y = row + 20 * math.sin(col * 0.02 + t * 1.5 + row * 0.01)
                pts.append(f"{col},{y:.1f}")
            hue = (row / H + t * 0.02) % 1.0
            color = _hsv_to_hex(hue, 0.6, 0.85)
            _add_elem("polyline", points=" ".join(pts), stroke=color, stroke_width=sw, fill="none")
        for col in range(0, W, step):
            pts = []
            for row in range(0, H, 4):
                x = col + 20 * math.sin(row * 0.02 + t * 1.2 + col * 0.01)
                pts.append(f"{x:.1f},{row}")
            hue = (col / W + 0.5 + t * 0.02) % 1.0
            color = _hsv_to_hex(hue, 0.6, 0.85)
            _add_elem("polyline", points=" ".join(pts), stroke=color, stroke_width=sw, fill="none")
    elif pattern == "mandala":
        n_rays = 24
        for i in range(n_rays):
            base_angle = i * 2 * math.pi / n_rays + t * 0.1
            for ring in range(1, 6):
                r1 = ring * 70 + 30 * math.sin(t * 0.5 + ring + i)
                r2 = r1 + 20 + 15 * math.sin(t * 0.7 + i * 0.5)
                hue = (ring / 6 + i / n_rays + t * 0.02) % 1.0
                color = _hsv_to_hex(hue, 0.8, 0.9)
                x1 = cx + r1 * math.cos(base_angle)
                y1 = cy + r1 * math.sin(base_angle)
                x2 = cx + r2 * math.cos(base_angle + 0.3)
                y2 = cy + r2 * math.sin(base_angle + 0.3)
                _add_elem("line", x1=str(x1), y1=str(y1), x2=str(x2), y2=str(y2),
                    stroke=color, stroke_width=sw)
            for ring in range(5):
                r = (ring + 1) * 70 + 30 * math.sin(t * 0.5 + ring)
                arc_pts = []
                for k in range(10):
                    a = base_angle + k * 2 * math.pi / (10 * n_rays) * 2
                    arc_pts.append(f"{cx + r * math.cos(a):.1f},{cy + r * math.sin(a):.1f}")
                hue = (ring / 6 + t * 0.03) % 1.0
                color = _hsv_to_hex(hue, 0.7, 0.85)
                _add_elem("polyline", points=" ".join(arc_pts), stroke=color, stroke_width=sw, fill="none")

    svg_str = '<?xml version="1.0" encoding="utf-8"?>\n' + ET.tostring(svg, encoding="unicode")
    svg_path = out_dir / mn(30, f"svg-vector-{pattern}")
    svg_path = svg_path.with_suffix(".svg")
    try:
        with open(svg_path, "w") as f:
            f.write(svg_str)
        print(f"  ✓ {svg_path.name}")
    except OSError as e:
        print(f"  ✗ SVG write failed: {e}")

    img = Image.new("RGB", (W, H), (10, 10, 18))
    draw = ImageDraw.Draw(img)

    if pattern == "grid":
        spacing = 40 + 20 * math.sin(t * 0.5)
        n_cols = int(W / spacing) + 2
        n_rows = int(H / spacing) + 2
        for r_idx in range(n_rows):
            for c_idx in range(n_cols):
                x = c_idx * spacing + (spacing * 0.5 * math.sin(t * 0.3 + r_idx * 0.5))
                y = r_idx * spacing + (spacing * 0.5 * math.cos(t * 0.4 + c_idx * 0.7))
                sz = spacing * 0.3 * (0.7 + 0.3 * math.sin(t + r_idx + c_idx))
                hue = (r_idx / max(1, n_rows) + t * 0.05) % 1.0
                col = tuple(int(c * 255) for c in colorsys.hsv_to_rgb(hue, 0.7, 0.9))
                rect = [x, y, x + sz, y + sz]
                if fill_enabled:
                    draw.rectangle(rect, fill=col, outline=col, width=stroke_width)
                else:
                    draw.rectangle(rect, outline=col, width=stroke_width)
    elif pattern == "circles":
        n = 80
        for i in range(n):
            frac = i / n
            angle = frac * 2 * math.pi + t * 0.3
            radius = 50 + 350 * (0.5 + 0.5 * math.sin(frac * 3 + t * 0.5))
            x = cx + radius * math.cos(angle)
            y = cy + radius * math.sin(angle)
            r = 10 + 30 * (0.5 + 0.5 * math.sin(frac * 5 + t))
            hue = (frac + t * 0.02) % 1.0
            col = tuple(int(c * 255) for c in colorsys.hsv_to_rgb(hue, 0.7, 0.9))
            bbox = [x - r, y - r, x + r, y + r]
            if fill_enabled:
                draw.ellipse(bbox, fill=col, outline=col, width=stroke_width)
            else:
                draw.ellipse(bbox, outline=col, width=stroke_width)
    elif pattern == "stars":
        n_stars = 40
        for i in range(n_stars):
            frac = i / n_stars
            angle = frac * 2 * math.pi + t * 0.2
            radius = 80 + 380 * (0.5 + 0.5 * math.sin(frac * 4 + t * 0.7))
            x = cx + radius * math.cos(angle)
            y = cy + radius * math.sin(angle)
            sz = 20 + 40 * (0.5 + 0.5 * math.sin(frac * 6 + t * 0.5))
            hue = (frac + t * 0.03) % 1.0
            col = tuple(int(c * 255) for c in colorsys.hsv_to_rgb(hue, 0.8, 0.95))
            pts = []
            for k in range(10):
                r_k = sz if k % 2 == 0 else sz * 0.4
                a_k = k * math.pi / 5 + t * 0.1 + i * 0.3
                pts.append((x + r_k * math.cos(a_k), y + r_k * math.sin(a_k)))
            if fill_enabled:
                draw.polygon(pts, fill=col, outline=col, width=stroke_width)
            else:
                draw.polygon(pts, outline=col, width=stroke_width)
    elif pattern == "waves":
        step = 8
        for row in range(0, H, step):
            pts = []
            for col in range(0, W, 4):
                y = row + 20 * math.sin(col * 0.02 + t * 1.5 + row * 0.01)
                pts.append((col, y))
            hue = (row / H + t * 0.02) % 1.0
            col = tuple(int(c * 255) for c in colorsys.hsv_to_rgb(hue, 0.6, 0.85))
            draw.line(pts, fill=col, width=stroke_width)
        for c in range(0, W, step):
            pts = []
            for r in range(0, H, 4):
                x = c + 20 * math.sin(r * 0.02 + t * 1.2 + c * 0.01)
                pts.append((x, r))
            hue = (c / W + 0.5 + t * 0.02) % 1.0
            col = tuple(int(c * 255) for c in colorsys.hsv_to_rgb(hue, 0.6, 0.85))
            draw.line(pts, fill=col, width=stroke_width)
    elif pattern == "mandala":
        n_rays = 24
        for i in range(n_rays):
            base_angle = i * 2 * math.pi / n_rays + t * 0.1
            for ring in range(1, 6):
                r1 = ring * 70 + 30 * math.sin(t * 0.5 + ring + i)
                r2 = r1 + 20 + 15 * math.sin(t * 0.7 + i * 0.5)
                hue = (ring / 6 + i / n_rays + t * 0.02) % 1.0
                col = tuple(int(c * 255) for c in colorsys.hsv_to_rgb(hue, 0.8, 0.9))
                x1 = cx + r1 * math.cos(base_angle)
                y1 = cy + r1 * math.sin(base_angle)
                x2 = cx + r2 * math.cos(base_angle + 0.3)
                y2 = cy + r2 * math.sin(base_angle + 0.3)
                draw.line([(x1, y1), (x2, y2)], fill=col, width=stroke_width)
            for ring in range(5):
                r = (ring + 1) * 70 + 30 * math.sin(t * 0.5 + ring)
                pts = []
                for k in range(10):
                    a = base_angle + k * 2 * math.pi / (10 * n_rays) * 2
                    pts.append((cx + r * math.cos(a), cy + r * math.sin(a)))
                hue = (ring / 6 + t * 0.03) % 1.0
                col = tuple(int(c * 255) for c in colorsys.hsv_to_rgb(hue, 0.7, 0.85))
                draw.line(pts, fill=col, width=stroke_width)

    arr = np.array(img).astype(np.float32) / 255.0
    capture_frame("30", arr)
    save(img, mn(30, f"svg-vector-{pattern}"), out_dir)

