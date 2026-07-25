from __future__ import annotations
import math
import random
import shutil
import subprocess
from io import BytesIO
from pathlib import Path

import numpy as np
from PIL import Image

from ...core.registry import method
from ...core.utils import save, mn, seed_all, W, H
from ...core.animation import capture_frame


@method(id="45", name="Graphviz", category="cli_tools", tags=["graph", "expanded"],
        # SCALAR, not FIELD: every one of these becomes a Graphviz DOT attribute
        # or a loop bound — `range(use_n_nodes)`, `fontsize={use_font_size}`,
        # `len={use_edge_len}` — so they are irreducibly one number per render.
        # They were declared FIELD and then np.mean()'d on arrival, which made a
        # wired field a silent no-op. A SCALAR driver (LFO sweeping node_count)
        # still works and is the meaningful way to animate them.
        inputs={"image_in": "IMAGE",
                "anim_speed": "SCALAR",
                "edge_density": "SCALAR",
                "node_count": "SCALAR",
                "edge_len": "SCALAR",
                "node_font_size": "SCALAR"},
        outputs={"image": "IMAGE", "luminance": "FIELD"},
        params={
            "node_count": {"description": "number of graph nodes (structural — SCALAR only)", "min": 10, "max": 200, "default": 40},
            "edge_density": {"description": "number of random edges (node_count × multiplier; structural — SCALAR only)", "min": 1, "max": 10, "default": 2},
            "layout": {"description": "Graphviz layout engine (neato/dot/fdp/sfdp/twopi/circo)", "default": "neato"},
            "bg_color": {"description": "graph background hex color", "default": "#0a0a12"},
            "node_fill": {"description": "default node fill hex color", "default": "#2a2a32"},
            "node_font_color": {"description": "node label font hex color", "default": "#8a7a6a"},
            "node_border": {"description": "node border hex color", "default": "#4a4a5a"},
            "node_font_size": {"description": "node label font size (structural — SCALAR only)", "min": 4, "max": 24, "default": 8},
            "edge_color": {"description": "edge line hex color", "default": "#4a3a2a"},
            "edge_len": {"description": "edge length factor (structural — SCALAR only)", "min": 0.5, "max": 10.0, "default": 1.5},
            "dpi": {"description": "output DPI", "min": 36, "max": 300, "default": 72},
            "anim_mode": {"description": "animation mode", "choices": ["none", "edge_morph", "color_cycle",
                "layout_cycle", "node_drift", "font_pulse", "bg_cycle", "edge_len_morph"], "default": "none"},
            "anim_speed": {"description": "animation speed multiplier (can be driven by FIELD)", "min": 0.1, "max": 5.0, "default": 1.0},
        })
def method_graphviz(out_dir: Path, seed: int, params=None):
    """Generate a graph visualization using Graphviz dot.

    Creates a random graph with N nodes and random edges, renders it via
    the Graphviz `dot` CLI tool, and saves the result as a PNG. Falls back
    to a dark placeholder if dot is unavailable. 8 animation modes modulate
    edge density, node colors, layout engine, node count, font size, and
    background color.

    Returns:
        dict with "image" (H,W,3 float32 [0,1]) — luminance auto-computed
    """
    if params is None:
        params = {}
    anim_time = float(params.get("time", 0.0))
    anim_mode = params.get("anim_mode", "none")
    seed_all(seed)
    rng = random.Random(seed)

    # ── SCALAR-driven anim_speed ──
    anim_speed_override = params.get("anim_speed")
    if anim_speed_override is not None:
        anim_speed = float(anim_speed_override)
    else:
        anim_speed = float(params.get("anim_speed", 1.0))

    # These four are structural (see the inputs= note above): a SCALAR wire
    # lands straight in params, so no _field_ handling is needed or meaningful.
    n_nodes = int(params.get("node_count", 40))
    base_edge_density = int(params.get("edge_density", 2))

    layout = params.get("layout", "neato")
    bg_color = params.get("bg_color", "#0a0a12")
    node_fill = params.get("node_fill", "#2a2a32")
    node_font_color = params.get("node_font_color", "#8a7a6a")
    node_border = params.get("node_border", "#4a4a5a")
    base_font_size = int(params.get("node_font_size", 8))
    edge_color = params.get("edge_color", "#4a3a2a")
    base_edge_len = float(params.get("edge_len", 1.5))

    dpi = int(params.get("dpi", 72))

    # ── Per-frame time + seed ──
    t = anim_time * anim_speed
    if anim_mode == "none":
        t = 0.0
    _frame_seed = seed + int(t * 10000)
    _frng = random.Random(_frame_seed)

    # ── Animation modulation ──
    edge_density = base_edge_density
    hue_shift = 0.0
    use_layout = layout
    use_n_nodes = n_nodes
    use_font_size = base_font_size
    use_bg_color = bg_color
    use_edge_len = base_edge_len
    _layouts = ["neato", "dot", "fdp", "sfdp", "twopi", "circo"]

    if anim_mode == "edge_morph":
        frac = 0.3 + 0.7 * (0.5 + 0.5 * math.sin(t * 0.3))
        edge_density = max(1, round(base_edge_density * frac))
    elif anim_mode == "color_cycle":
        edge_density = base_edge_density
        hue_shift = (t * 0.1) % 1.0
    elif anim_mode == "layout_cycle":
        edge_density = base_edge_density
        idx = int(t * 0.2) % len(_layouts)
        use_layout = _layouts[idx]
    elif anim_mode == "node_drift":
        edge_density = base_edge_density
        frac = 0.5 + 0.5 * math.sin(t * 0.15)
        use_n_nodes = max(10, int(n_nodes * (0.5 + 0.5 * frac)))
    elif anim_mode == "font_pulse":
        edge_density = base_edge_density
        use_font_size = max(4, round(base_font_size * (0.6 + 0.8 * (0.5 + 0.5 * math.sin(t * 0.3)))))
    elif anim_mode == "bg_cycle":
        edge_density = base_edge_density
        hue = (t * 0.08) % 1.0
        r_c = int(40 * (0.5 + 0.5 * math.sin(hue * 2 * math.pi)))
        g_c = int(40 * (0.5 + 0.5 * math.sin(hue * 2 * math.pi + 2.094)))
        b_c = int(40 * (0.5 + 0.5 * math.sin(hue * 2 * math.pi + 4.189)))
        use_bg_color = f"#{r_c:02x}{g_c:02x}{b_c:02x}"
    elif anim_mode == "edge_len_morph":
        edge_density = base_edge_density
        use_edge_len = base_edge_len * (0.5 + 1.0 * (0.5 + 0.5 * math.sin(t * 0.25)))
        use_edge_len = max(0.5, min(10.0, use_edge_len))

    # ── Check for dot binary ──
    try:
        subprocess.run(["dot", "--version"], capture_output=True, timeout=5)
    except (FileNotFoundError, subprocess.TimeoutExpired):
        fallback = np.ones((H, W, 3), dtype=np.float32) * 0.05
        capture_frame("45", fallback)
        return {"image": fallback}

    # ── Build DOT graph ──
    dot_lines = [
        "graph G {",
        f"  layout={use_layout};",
        f'  bgcolor="{use_bg_color}";',
        f'  node [style=filled, fillcolor="{node_fill}", fontcolor="{node_font_color}", color="{node_border}", fontsize={use_font_size}];',
        f'  edge [color="{edge_color}", len={use_edge_len}];',
    ]
    for i in range(use_n_nodes):
        if anim_mode == "color_cycle":
            hue = (i / max(1, use_n_nodes) + hue_shift) % 1.0
            r_c = int(255 * (0.5 + 0.5 * math.sin(hue * 2 * math.pi)))
            g_c = int(255 * (0.5 + 0.5 * math.sin(hue * 2 * math.pi + 2.094)))
            b_c = int(255 * (0.5 + 0.5 * math.sin(hue * 2 * math.pi + 4.189)))
        else:
            r_c = _frng.randint(20, 60)
            g_c = _frng.randint(20, 50)
            b_c = _frng.randint(30, 60)
        dot_lines.append(f'  n{i} [fillcolor="#{r_c:02x}{g_c:02x}{b_c:02x}", label=""];')
    for _ in range(use_n_nodes * edge_density):
        a = _frng.randint(0, use_n_nodes - 1)
        b_node = _frng.randint(0, use_n_nodes - 1)
        if a != b_node:
            dot_lines.append(f"  n{a} -- n{b_node};")
    dot_lines.append("}")
    dot_content = "\n".join(dot_lines)

    # ── Render via dot ──
    try:
        result = subprocess.run(
            ["dot", "-Tpng", f"-Gsize={W / dpi},{H / dpi}", f"-Gdpi={dpi}"],
            input=dot_content.encode(), capture_output=True, timeout=30,
        )
        if result.returncode == 0:
            try:
                img = Image.open(BytesIO(result.stdout)).convert("RGB")
                img = img.resize((W, H), Image.LANCZOS)
                arr = np.array(img, dtype=np.float32) / 255.0
                capture_frame("45", arr)
                return {"image": arr}
            except Exception:
                pass
    except (FileNotFoundError, Exception):
        pass

    # ── Fallback ──
    fallback = np.ones((H, W, 3), dtype=np.float32) * 0.05
    capture_frame("45", fallback)
    return {"image": fallback}
