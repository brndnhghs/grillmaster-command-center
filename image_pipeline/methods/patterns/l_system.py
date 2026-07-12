"""L-System (Lindenmayer system) — procedural generative geometry.

L-Systems were introduced by Aristid Lindenmayer (1968,
https://en.wikipedia.org/wiki/L-system) to model plant growth and have become a
cornerstone of procedural content generation: fractal plants, architectural
ornament, and — at "Everything Procedural" (2024,
https://www.reddit.com/r/proceduralgeneration/comments/1f5rgp3/) — modern game
vegetation and tilemap authoring. An L-System rewrites an axiom string with
production rules for a fixed number of iterations, then a turtle interprets the
result: ``F``/draw-symbols move forward drawing a segment, ``+``/``-`` turn, and
``[``/``]`` push/pop a branching state.

This node ships nine canonical presets (Koch snowflake/curve, Sierpinski
arrowhead, dragon curve, Barnsley plant/bush, fractal tree, sticks, Hilbert
curve), each with a distinct visual signature. It is a CPU method; the 2D
render/export path is authoritative and the output is a per-pixel line drawing
(vectorised via PIL at a supersampled resolution for AA). A clean closed-form
GPU twin is a possible follow-up, but the branching turtle state makes it a poor
fit for a stateless f(uv,t) shader, so it stays CPU-only for now.

Animation modes (Architecture B — per-frame re-call with `time`):
    none  — static full draw: frame Δ ≈ 0 (static baseline).
    grow  — segments are revealed in traversal order as _t sweeps 0→2π, so the
            structure "draws itself" (strong frame-to-frame Δ).
    spin  — the whole drawing rotates by a non-integer rate (_t * 0.6), so it is
            never symmetry-aligned at the audit sample times (always reads as
            motion, even for symmetric figures).
    sway  — a height-dependent bend about the base, driven by cos(_t) (wind).
            cos (not sin) keeps t=0 vs t=π distinct (sin-phase degeneracy).

Lines are kept thin (1–2 px, supersampled for AA) per the pipeline's
mechanical-line rendering convention — they do NOT thicken under any mode.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from ...core.registry import method
from ...core.utils import (
    save, mn, seed_all, BG_DEFAULT, W, H,
    write_mask, write_particles, write_scalars,
)
from ...core.animation import capture_frame

_SS = 2  # supersample factor for anti-aliased line rasterisation

# ── Canonical L-System presets ───────────────────────────────────────────────
# axiom      : starting string
# rules      : symbol → replacement (deterministic)
# angle      : turn angle in degrees (canonical)
# max_iter   : safety cap on iterations for this preset
# draw       : symbols that draw a forward segment
# heading0   : initial turtle heading (deg, math convention, +y is up)
PRESETS = {
    "koch_snowflake": dict(
        axiom="F++F++F", rules={"F": "F-F++F-F"}, angle=60.0, max_iter=5,
        draw={"F"}, heading0=90.0),
    "koch_curve": dict(
        axiom="F", rules={"F": "F+F-F-F+F"}, angle=90.0, max_iter=5,
        draw={"F"}, heading0=0.0),
    "sierpinski_arrowhead": dict(
        axiom="A", rules={"A": "B-A-B", "B": "A+B+A"}, angle=60.0, max_iter=7,
        draw={"A", "B"}, heading0=0.0),
    "dragon_curve": dict(
        axiom="FX", rules={"X": "X+YF+", "Y": "-FX-Y"}, angle=90.0, max_iter=12,
        draw={"F"}, heading0=0.0),
    "plant": dict(
        axiom="X", rules={"X": "F+[[X]-X]-F[-FX]+X", "F": "FF"}, angle=25.0,
        max_iter=6, draw={"F"}, heading0=90.0),
    "bush": dict(
        axiom="F", rules={"F": "FF-[-F+F+F]+[+F-F-F]"}, angle=22.5, max_iter=5,
        draw={"F"}, heading0=90.0),
    "tree": dict(
        axiom="X", rules={"X": "F[+X]F[-X]+X", "F": "FF"}, angle=20.0,
        max_iter=6, draw={"F"}, heading0=90.0),
    "sticks": dict(
        axiom="X", rules={"X": "F[+X][-X]FX", "F": "FF"}, angle=25.0,
        max_iter=6, draw={"F"}, heading0=90.0),
    "hilbert": dict(
        axiom="A", rules={"A": "+BF-AFA-FB+", "B": "-AF+BFB+FA-"}, angle=90.0,
        max_iter=6, draw={"F"}, heading0=0.0),
}


def _hsl_to_rgb(h: float, s: float, l: float):
    """HSL → RGB, all in [0,1]."""
    h = h % 1.0
    c = (1.0 - abs(2.0 * l - 1.0)) * s
    x = c * (1.0 - abs((h * 6.0) % 2.0 - 1.0))
    m = l - c / 2.0
    if h < 1.0 / 6.0:
        r, g, b = c, x, 0.0
    elif h < 2.0 / 6.0:
        r, g, b = x, c, 0.0
    elif h < 3.0 / 6.0:
        r, g, b = 0.0, c, x
    elif h < 4.0 / 6.0:
        r, g, b = 0.0, x, c
    elif h < 5.0 / 6.0:
        r, g, b = x, 0.0, c
    else:
        r, g, b = c, 0.0, x
    return r + m, g + m, b + m


def _expand(axiom: str, rules: dict, iters: int):
    """Rewrites the axiom `iters` times. Capped to avoid runaway length."""
    s = axiom
    for _ in range(iters):
        ns = []
        for ch in s:
            ns.append(rules.get(ch, ch))
        s = "".join(ns)
        if len(s) > 2_000_000:
            break
    return s


def _interpret(s: str, angle_deg: float, draw, heading0: float):
    """Turtle-interpret the string → list of (x0,y0,x1,y1,depth) in turtle space."""
    segs = []
    x = y = 0.0
    heading = math.radians(heading0)
    step = 1.0
    a = math.radians(angle_deg)
    stack = []
    depth = 0
    max_depth = 0
    for ch in s:
        if ch in draw:
            nx = x + step * math.cos(heading)
            ny = y + step * math.sin(heading)
            segs.append((x, y, nx, ny, depth))
            x, y = nx, ny
        elif ch in ("f", "G"):
            x += step * math.cos(heading)
            y += step * math.sin(heading)
        elif ch == "+":
            heading += a
        elif ch == "-":
            heading -= a
        elif ch == "[":
            stack.append((x, y, heading, depth))
            depth += 1
            if depth > max_depth:
                max_depth = depth
        elif ch == "]":
            if stack:
                x, y, heading, depth = stack.pop()
        # other symbols (X, Y, A-as-control, ...) are no-ops for drawing
    return segs, max_depth


@method(
    id="461",
    name="L-System",
    category="patterns",
    tags=["generative", "lsystem", "lindenmayer", "fractal", "plant",
          "procedural", "animation"],
    inputs={},
    outputs={"image": "IMAGE", "mask": "MASK", "particles": "PARTICLES"},
    params={
        "preset": {"description": "L-System preset (canonical axiom/rules/angle)",
                   "choices": list(PRESETS.keys()), "default": "plant"},
        "iterations": {"description": "production-rule iterations (detail)",
                       "min": 1.0, "max": 8.0, "default": 5.0},
        "angle_scale": {"description": "multiplier on the preset turn angle (1=canonical)",
                        "min": 0.3, "max": 2.0, "default": 1.0},
        "line_width": {"description": "stroke width in px (kept thin)",
                       "min": 1.0, "max": 4.0, "default": 1.5},
        "color_mode": {"description": "line colouring",
                       "choices": ["depth", "uniform", "rainbow"], "default": "depth"},
        "hue": {"description": "base line colour hue (HSL, 0-1)",
                "min": 0.0, "max": 1.0, "default": 0.33},
        "brightness": {"description": "line colour brightness multiplier",
                       "min": 0.3, "max": 1.4, "default": 1.0},
        "background": {"description": "canvas background",
                       "choices": ["dark", "light", "mid"], "default": "dark"},
        "time": {"description": "animation phase [0, 2pi)",
                 "min": 0.0, "max": 6.28, "default": 0.0},
        "anim_mode": {"description": "animation mode (none/grow/spin/sway)",
                      "choices": ["none", "grow", "spin", "sway"], "default": "none"},
        "anim_speed": {"description": "animation speed multiplier",
                       "min": 0.1, "max": 5.0, "default": 1.0},
    },
)
def method_lsystem(out_dir: Path, seed: int, params=None):
    """L-System — procedural fractal geometry via string-rewrite + turtle graphics.

    Nine canonical presets (Koch, Sierpinski arrowhead, dragon, Barnsley plant,
    bush, fractal tree, sticks, Hilbert). The turn angle is the preset's
    canonical value scaled by `angle_scale`, so 1.0 reproduces the textbook
    figure. Lines stay thin (1–2 px) per the mechanical-line convention.

    Params:
        preset:      which L-System
        iterations:  production-rule iterations (more ⇒ finer detail)
        angle_scale: turn-angle multiplier (1 = canonical)
        line_width:  stroke width (kept thin)
        color_mode:  depth (by branch depth) / uniform / rainbow (by path)
        hue:         base line hue
        brightness:  line brightness multiplier
        background:  dark / light / mid canvas
        time:        animation phase [0, 2pi)
        anim_mode:   none / grow / spin / sway
        anim_speed:  animation speed (0.1-5.0)
    """
    try:
        if params is None:
            params = {}
        t = float(params.get("time", 0.0))
        anim_mode = str(params.get("anim_mode", "none"))
        anim_speed = float(params.get("anim_speed", 1.0))
        seed_all(seed)

        preset_name = str(params.get("preset", "plant"))
        if preset_name not in PRESETS:
            preset_name = "plant"
        P = PRESETS[preset_name]

        iters = int(max(1.0, min(8.0, float(params.get("iterations", 5.0)))))
        iters = min(iters, P["max_iter"])
        angle_scale = float(params.get("angle_scale", 1.0))
        line_width = max(1.0, min(4.0, float(params.get("line_width", 1.5))))
        color_mode = str(params.get("color_mode", "depth"))
        hue = max(0.0, min(1.0, float(params.get("hue", 0.33))))
        brightness = max(0.3, min(1.4, float(params.get("brightness", 1.0))))
        background = str(params.get("background", "dark"))

        angle = P["angle"] * angle_scale

        # ── Animation clock (rename t to avoid shadowing the time param) ──
        _t = t * anim_speed

        # ── Build + interpret (independent of t for a fixed seed/preset) ──
        s = _expand(P["axiom"], P["rules"], iters)
        segs, max_depth = _interpret(s, angle, P["draw"], P["heading0"])
        n_seg = len(segs)

        # ── Background ──
        if background == "light":
            bg = np.array([0.96, 0.96, 0.98], dtype=np.float32)
        elif background == "mid":
            bg = np.array(BG_DEFAULT, dtype=np.float32) / 255.0
        else:  # dark
            bg = np.array([0.04, 0.05, 0.09], dtype=np.float32)
        bg_u8 = (bg * 255).astype(np.uint8).tolist()

        if n_seg == 0:
            img = Image.new("RGB", (W, H), tuple(bg_u8))
            capture_frame("461", np.asarray(img, dtype=np.float32) / 255.0)
            save(img, mn(461, "L-System"), out_dir)
            return np.asarray(img, dtype=np.float32) / 255.0

        # ── Bounding box in turtle space ──
        xs = [p for seg in segs for p in (seg[0], seg[2])]
        ys = [p for seg in segs for p in (seg[1], seg[3])]
        minx, maxx = min(xs), max(xs)
        miny, maxy = min(ys), max(ys)
        cx0 = (minx + maxx) / 2.0
        cy0 = (miny + maxy) / 2.0
        ext = max(maxx - minx, maxy - miny) or 1.0
        pivotx, pivoty = cx0, miny  # bottom-centre, for sway

        # ── Per-frame transforms in turtle space ──
        spin_a = 0.0
        sway_amp = 0.0
        if anim_mode == "spin":
            # Non-integer rate ⇒ never symmetry-aligned at audit sample times.
            spin_a = _t * 0.6
        elif anim_mode == "sway":
            # cos (not sin): cos(0)=+1, cos(π)=−1 keep audit frames distinct.
            sway_amp = 0.4 * math.cos(_t)

        def tf(px, py):
            if sway_amp != 0.0:
                h = (py - miny) / ext if ext > 0 else 0.0
                th = sway_amp * h
                dx = px - pivotx
                dy = py - pivoty
                px = pivotx + dx * math.cos(th) - dy * math.sin(th)
                py = pivoty + dx * math.sin(th) + dy * math.cos(th)
            if spin_a != 0.0:
                dx = px - cx0
                dy = py - cy0
                px = cx0 + dx * math.cos(spin_a) - dy * math.sin(spin_a)
                py = cy0 + dx * math.sin(spin_a) + dy * math.cos(spin_a)
            return px, py

        W2, H2 = W * _SS, H * _SS
        scale = (min(W2, H2) * 0.92) / ext

        def to_img(px, py):
            px, py = tf(px, py)
            sx = W2 / 2.0 + (px - cx0) * scale
            sy = H2 / 2.0 - (py - cy0) * scale
            return sx, sy

        # ── Growth: reveal first fraction of segments in traversal order ──
        if anim_mode == "grow":
            n_draw = int((_t / (2.0 * math.pi)) * n_seg)
            n_draw = max(0, min(n_seg, n_draw))
        else:
            n_draw = n_seg

        # ── Rasterise (image + coverage mask) at supersample, then AA-downscale ──
        img = Image.new("RGB", (W2, H2), tuple(bg_u8))
        dimg = ImageDraw.Draw(img)
        mask_img = Image.new("L", (W2, H2), 0)
        dmask = ImageDraw.Draw(mask_img)
        lw = max(1, int(round(line_width * _SS)))

        for i in range(n_draw):
            x0, y0, x1, y1, depth = segs[i]
            p0 = to_img(x0, y0)
            p1 = to_img(x1, y1)
            if color_mode == "uniform":
                col_h = hue
            elif color_mode == "rainbow":
                col_h = (i / n_seg) if n_seg else 0.0
            else:  # depth
                col_h = (hue + (depth / max(1, max_depth)) * 0.5) % 1.0
            r, g, b = _hsl_to_rgb(col_h, 0.9, 0.55)
            col = (int(max(0.0, min(1.0, r * brightness)) * 255.0),
                   int(max(0.0, min(1.0, g * brightness)) * 255.0),
                   int(max(0.0, min(1.0, b * brightness)) * 255.0))
            dimg.line([p0, p1], fill=col, width=lw)
            dmask.line([p0, p1], fill=255, width=lw)

        img = img.resize((W, H), Image.Resampling.LANCZOS)
        mask_img = mask_img.resize((W, H), Image.Resampling.LANCZOS)
        rgb = np.asarray(img, dtype=np.float32) / 255.0
        mask = np.asarray(mask_img, dtype=np.float32) / 255.0

        # ── Particles: branch tips (segments at maximum recursion depth) ──
        tips = [(segs[i][2], segs[i][3]) for i in range(n_draw)
                if segs[i][4] == max_depth]
        if tips:
            if len(tips) > 4000:
                step_k = len(tips) // 4000
                tips = tips[::step_k]
            particles = np.zeros((len(tips), 4), dtype=np.float32)
            for k, (tx, ty) in enumerate(tips):
                sx, sy = to_img(tx, ty)
                particles[k, 0] = sx / _SS
                particles[k, 1] = sy / _SS
        else:
            particles = np.zeros((0, 4), dtype=np.float32)

        capture_frame("461", rgb)
        save(img, mn(461, "L-System"), out_dir)
        try:
            write_mask(out_dir, mask)
            write_particles(out_dir, particles)
            write_scalars(
                out_dir,
                preset=float(hash(preset_name) & 0xFFFF),
                iterations=float(iters),
                segment_count=float(n_seg),
                max_depth=float(max_depth),
                coverage=float(mask.mean()),
            )
        except Exception:
            pass
        return rgb
    except Exception as exc:
        fallback = np.full((H, W, 3), 0.5, dtype=np.float32)
        save(fallback, mn(461, "L-System"), out_dir)
        print(f"[method_461] ERROR: {exc}")
        return fallback
