"""Penrose aperiodic tiling (P3 rhombi) via Robinson-triangle deflation.

A Penrose tiling is a **non-periodic** (aperiodic) tiling of the plane by two
rhombus shapes — it has local five-fold / ten-fold order but never repeats
globally. It was discovered by Roger Penrose (1974) and is the most famous
example of a quasicrystal in 2D. The construction used here is the standard
**deflation / subdivision** of *Robinson triangles* (red acute 36°-72°-72° and
blue obtuse 108°-36°-36° isosceles triangles): start from a decagonal "sun" of
ten red triangles, then repeatedly subdivide each triangle into smaller
Robinson triangles. After enough generations, pairs of same-coloured triangles
form the thick and thin rhombi.

Reference (algorithm):
    https://preshing.com/20110831/penrose-tiling-explained  (Robinson-triangle
    deflation, the widely used complex-number formulation also shown on the
    Wikipedia "Penrose tiling" article).
Aperiodic tilings have seen a fresh wave of generative-art interest (2023-2025
ShaderToy / coding-channel pieces, and the 2025 arXiv "finite-state transducers
for substitution tilings" formalisation of exactly this Robinson deflation).

This is a CPU method; the 2D render/export path is authoritative and the output
is a full-coverage per-pixel colour field (RGB), with a tile-type FIELD (0/1)
emitted alongside.

Animation modes (Architecture B — per-frame re-call with `time`):
    none    — static full draw: frame Δ ≈ 0 (static baseline).
    rotate  — the whole tiling rotates about its centre (strong Δ).
    breathe — the tiling zooms (self-similar, so a continuous scale reads as a
              slow in/out pulse; smooth, no cusp).
    phase   — the palette hues rotate around the colour wheel (colour cycle).

NOTE on the audit: the seed "sun" has 10-fold rotational symmetry (every 36°),
so a rotation of exactly 180° maps the rasterised image onto itself. The
verification therefore samples *non-symmetry* angles (e.g. t=0 vs t=0.4 rad ≈
23°), never t=π, to avoid a false Δ≈0.
"""

from __future__ import annotations

import cmath
import math
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from ...core.registry import method
from ...core.utils import (
    save, mn, seed_all, W, H,
    write_field, write_scalars,
)
from ...core.animation import capture_frame

_PHI = (1.0 + math.sqrt(5.0)) / 2.0

# Palettes: (hueA, hueB, satA, satB, bg(rgb 0-255), edge(rgb 0-255))
_PALETTES = {
    "gold-blue": (0.12, 0.62, 0.85, 0.75, (12, 12, 18), (225, 205, 150)),
    "sunset":    (0.06, 0.78, 0.90, 0.70, (18, 12, 22), (255, 180, 140)),
    "emerald":   (0.38, 0.55, 0.80, 0.60, (10, 18, 16), (170, 230, 200)),
    "magma":     (0.05, 0.92, 0.95, 0.80, (15, 8, 12),  (255, 150, 90)),
    "ice":       (0.55, 0.60, 0.60, 0.80, (10, 14, 22), (190, 220, 245)),
    "violet":    (0.75, 0.83, 0.70, 0.85, (16, 12, 20), (220, 180, 255)),
}


def _hsv2rgb(h: float, s: float, v: float):
    h = h - math.floor(h)
    i = int(h * 6.0) % 6
    f = h * 6.0 - i
    p = v * (1.0 - s)
    q = v * (1.0 - f * s)
    t = v * (1.0 - (1.0 - f) * s)
    return {
        0: (v, t, p), 1: (q, v, p), 2: (p, v, t),
        3: (p, q, v), 4: (t, p, v), 5: (v, p, q),
    }[i]


def _initial_wheel():
    """Ten red (type-0) Robinson triangles arranged as a decagonal 'sun'."""
    tris = []
    for i in range(10):
        B = cmath.exp((2 * i - 1) * 1j * math.pi / 10.0)
        C = cmath.exp((2 * i + 1) * 1j * math.pi / 10.0)
        if i % 2 == 0:
            B, C = C, B
        tris.append((0, 0j, B, C))
    return tris


def _subdivide(tris):
    """One deflation step (preshing / Wikipedia Robinson-triangle rules)."""
    out = []
    for color, A, B, C in tris:
        if color == 0:                      # red acute triangle
            P = A + (B - A) / _PHI
            out += [(0, C, P, B), (1, P, C, A)]
        else:                               # blue obtuse triangle
            Q = B + (A - B) / _PHI
            R = B + (C - B) / _PHI
            out += [(1, R, C, A), (1, Q, R, B), (0, R, Q, A)]
    return out


def _build_triangles(subdivisions: int):
    tris = _initial_wheel()
    for _ in range(max(0, int(subdivisions))):
        tris = _subdivide(tris)
    return tris


@method(
    id="465",
    name="Penrose Tiling",
    category="patterns",
    tags=["penrose", "aperiodic", "tiling", "quasicrystal", "generative",
          "geometry", "robinson-triangle", "animation"],
    inputs={},
    outputs={"image": "IMAGE", "field": "FIELD"},
    params={
        "subdivisions": {"description": "deflation depth — sets tile size (higher = smaller, more numerous tiles)",
                         "min": 3, "max": 8, "default": 6},
        "palette": {"description": "colour scheme",
                    "choices": list(_PALETTES.keys()), "default": "gold-blue"},
        "coloring": {"description": "how tiles are coloured",
                     "choices": ["two-tone", "orientation", "radial"], "default": "two-tone"},
        "show_edges": {"description": "draw rhombus outlines",
                       "choices": ["off", "on"], "default": "off"},
        "time": {"description": "animation phase [0, 2pi)",
                 "min": 0.0, "max": 6.28, "default": 0.0},
        "anim_mode": {"description": "animation mode (none/rotate/breathe/phase)",
                      "choices": ["none", "rotate", "breathe", "phase"], "default": "none"},
        "anim_speed": {"description": "animation speed multiplier",
                       "min": 0.1, "max": 5.0, "default": 1.0},
    },
)
def method_penrose(out_dir: Path, seed: int, params=None):
    """Penrose P3 rhombus tiling — aperiodic quasicrystal via deflation.

    Builds the tiling by deflating a decagonal seed of Robinson triangles, then
    rasterises the two triangle types as the thick/thin rhombi. A base rotation
    seeded from ``seed`` lets every seed show a different patch of the
    (locally-isomorphic) infinite tiling. Four colourings and four animation
    modes are supported.

    Params:
        subdivisions: deflation depth (tile size)
        palette:      colour scheme
        coloring:     two-tone / orientation / radial
        show_edges:   draw rhombus outlines
        time:         animation phase [0, 2pi)
        anim_mode:    none / rotate / breathe / phase
        anim_speed:   animation speed (0.1-5.0)
    """
    try:
        if params is None:
            params = {}
        t = float(params.get("time", 0.0))
        anim_mode = str(params.get("anim_mode", "none"))
        anim_speed = float(params.get("anim_speed", 1.0))
        seed_all(seed)
        rng = np.random.default_rng(seed)

        w, h = int(W), int(H)
        cx, cy = w / 2.0, h / 2.0
        scale_fit = 0.95 * min(w, h) * 0.5

        subdivisions = int(max(3, min(8, float(params.get("subdivisions", 6)))))
        palette_name = str(params.get("palette", "gold-blue"))
        if palette_name not in _PALETTES:
            palette_name = "gold-blue"
        coloring = str(params.get("coloring", "two-tone"))
        show_edges = str(params.get("show_edges", "off")) == "on"

        hueA, hueB, satA, satB, bg, edge = _PALETTES[palette_name]
        edge_rgb = tuple(int(c) for c in edge)
        bg_rgb = tuple(int(c) for c in bg)

        # Base rotation from the seed so different seeds show different patches.
        base_rot = rng.random() * 2.0 * math.pi

        # ── Animation clock (rename t so we never shadow the time param) ──
        _t = t * anim_speed
        theta = base_rot + (_t if anim_mode == "rotate" else 0.0)
        scale = 1.0 + (0.18 * math.sin(_t) if anim_mode == "breathe" else 0.0)
        hue_shift = _t if anim_mode == "phase" else 0.0

        ca, sa = math.cos(theta), math.sin(theta)

        def to_screen(z: complex):
            x = (z.real * scale) * scale_fit
            y = (z.imag * scale) * scale_fit
            return (cx + x * ca - y * sa, cy + x * sa + y * ca)

        tris = _build_triangles(subdivisions)

        img = Image.new("RGB", (w, h), bg_rgb)
        fld = Image.new("L", (w, h), 128)
        draw = ImageDraw.Draw(img)
        fdraw = ImageDraw.Draw(fld)

        for color, A, B, C in tris:
            pa = to_screen(A)
            pb = to_screen(B)
            pc = to_screen(C)
            poly = [pa, pb, pc]

            if coloring == "orientation":
                ang = math.atan2((B.imag - A.imag), (B.real - A.real))
                hue = (ang / math.pi + 1.0) * 0.5
                fill = tuple(int(255 * c) for c in _hsv2rgb(hue + hue_shift, 0.8, 0.92))
            elif coloring == "radial":
                cen = (A + B + C) / 3.0
                d = min(1.0, abs(cen) / _PHI)
                fill = tuple(int(255 * c) for c in _hsv2rgb(d + hue_shift, 0.75, 0.90))
            else:  # two-tone by Robinson-triangle type
                if color == 0:
                    fill = tuple(int(255 * c) for c in _hsv2rgb(hueA + hue_shift, satA, 0.95))
                else:
                    fill = tuple(int(255 * c) for c in _hsv2rgb(hueB + hue_shift, satB, 0.80))

            draw.polygon(poly, fill=fill)
            fdraw.polygon(poly, fill=60 if color == 0 else 200)
            if show_edges:
                draw.line([pa, pb, pc, pa], fill=edge_rgb, width=1)

        rgb = np.array(img, dtype=np.float32) / 255.0
        field = np.array(fld, dtype=np.float32) / 255.0

        capture_frame("465", rgb)
        save(img, mn(465, "Penrose Tiling"), out_dir)
        try:
            # Tile-type FIELD (0.5 = background, ~0.23 = thin rhombus, ~0.78 = thick).
            write_field(out_dir, field)
            write_scalars(
                out_dir,
                tile_count=float(len(tris)),
                subdivisions=float(subdivisions),
            )
        except Exception:
            pass
        return rgb
    except Exception as exc:
        fallback = np.full((int(H), int(W), 3), 0.5, dtype=np.float32)
        save(fallback, mn(465, "Penrose Tiling"), out_dir)
        print(f"[method_465] ERROR: {exc}")
        return fallback
