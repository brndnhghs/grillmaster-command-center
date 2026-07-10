"""
GPU shader nodes — individual @method wrappers for every GLSL shader.

Procedural shaders (IDs 173-197): generate imagery from scratch.
Filter shaders    (IDs 198-219): consume _input_image, return modified image.

The legacy combined method #82 is kept for backward-compatibility with
existing graphs that reference it by ID.

All methods are tagged "gpu" so the ⚡ badge renders in the palette.
Filter methods set new_image_contract=True — the executor skips _input.png
writes and they receive the upstream ndarray directly as params["_input_image"].
"""
from __future__ import annotations
from pathlib import Path

import numpy as np

from ..core.registry import method
from ..core.utils import save, seed_all, get_canvas
from ..core.animation import capture_frame
from ..core.shaders import render_shader, render_procedural, SHADERS

# ── Ordered shader lists (stable IDs) ────────────────────────────────
_PROC_SHADERS = [
    ("173", "mandelbrot",         "GPU Mandelbrot"),
    ("174", "julia",              "GPU Julia"),
    ("175", "plasma",             "GPU Plasma"),
    ("176", "domain_warp",        "GPU Domain Warp"),
    ("177", "voronoi",            "GPU Voronoi"),
    ("178", "voronoise",          "GPU Voronoise"),
    ("179", "ripples",            "GPU Ripples"),
    ("180", "cells",              "GPU Cells"),
    ("181", "bubble_chamber",     "GPU Bubble Chamber"),
    ("182", "stars",              "GPU Stars"),
    ("183", "lightning_fractal",  "GPU Lightning Fractal"),
    ("184", "spiral",             "GPU Spiral"),
    ("185", "dendritic",          "GPU Dendritic"),
    ("186", "barnsley",           "GPU Barnsley Fern"),
    ("187", "spectral",           "GPU Spectral"),
    ("188", "truchet",            "GPU Truchet"),
    ("189", "kaleidoscope_fractal","GPU Kaleidoscope Fractal"),
    ("190", "waves_3d",           "GPU Waves 3D"),
    ("191", "pixel_sort_gpu",     "GPU Pixel Sort"),
    ("192", "ocean",              "GPU Ocean"),
    ("193", "nebula_gpu",         "GPU Nebula"),
    ("194", "terrain",            "GPU Terrain"),
    ("195", "wood_grain_gpu",     "GPU Wood Grain"),
    ("196", "fire_gpu",           "GPU Fire"),
    ("197", "smoke_gpu",          "GPU Smoke"),
]

_FILT_SHADERS = [
    ("198", "shader_bloom",           "GPU Bloom"),
    ("199", "shader_emboss",          "GPU Emboss"),
    ("200", "shader_kaleidoscope",    "GPU Kaleidoscope"),
    ("201", "shader_water_ripple",    "GPU Water Ripple"),
    ("202", "shader_heat_shimmer",    "GPU Heat Shimmer"),
    ("203", "shader_pixelate_gpu",    "GPU Pixelate"),
    ("204", "shader_ink_bleed",       "GPU Ink Bleed"),
    ("205", "shader_halftone_gpu",    "GPU Halftone"),
    ("206", "shader_crt_gpu",         "GPU CRT"),
    ("207", "shader_hologram",        "GPU Hologram"),
    ("208", "shader_mosaic_gpu",      "GPU Mosaic"),
    ("209", "shader_edge_detect_gpu", "GPU Edge Detect"),
    ("210", "shader_warhol",          "GPU Warhol"),
    ("211", "shader_duotone_gpu",     "GPU Duotone"),
    ("212", "shader_rgb_split",       "GPU RGB Split"),
    ("213", "shader_caustics_gpu",    "GPU Caustics"),
    ("214", "shader_glitch_gpu",      "GPU Glitch"),
    ("215", "shader_posterize_gpu",   "GPU Posterize"),
    ("216", "shader_oil_gpu",         "GPU Oil Paint"),
    ("217", "shader_neon_gpu",        "GPU Neon Glow"),
    ("218", "shader_pencil_gpu",      "GPU Pencil"),
    ("219", "shader_motion_blur_gpu", "GPU Motion Blur"),
]

_PROC_PARAMS = {
    "p1": {"description": "shader param 1", "min": 0.0, "max": 1.0, "default": 0.5},
    "p2": {"description": "shader param 2", "min": 0.0, "max": 1.0, "default": 0.5},
    "p3": {"description": "shader param 3", "min": 0.0, "max": 1.0, "default": 0.5},
    "p4": {"description": "shader param 4", "min": 0.0, "max": 1.0, "default": 0.5},
    "time_scale": {"description": "animation speed", "min": 0.0, "max": 5.0, "default": 1.0},
}

_FILT_PARAMS = {
    "strength": {"description": "effect strength", "min": 0.0, "max": 1.0, "default": 0.5},
    "p2": {"description": "shader param 2", "min": 0.0, "max": 1.0, "default": 0.5},
    "time_scale": {"description": "animation speed", "min": 0.0, "max": 5.0, "default": 1.0},
}


# ── Factory: procedural ───────────────────────────────────────────────

def _make_proc(method_id: str, shader_name: str, method_name: str):
    @method(id=method_id, name=method_name, category="gpu_shaders",
            new_image_contract=True,
            tags=["gpu", "fast"],
            params=_PROC_PARAMS)
    def _fn(out_dir: Path, seed: int, params=None):
        if params is None:
            params = {}
        t = float(params.get("time", 0.0)) * float(params.get("time_scale", 1.0))
        p = tuple(float(params.get(f"p{i}", 0.5)) for i in range(1, 5))
        cw, ch = get_canvas()
        img = render_shader(shader_name, (cw, ch), p, t)
        arr = np.array(img, dtype=np.uint8)
        # Return dict: executor captures image directly; no disk write needed in live mode.
        # Disk mode: executor writes the output PNG at graph.py:891 when in_memory=False.
        return {"image": arr.astype(np.float32) / 255.0}

    _fn.__name__ = f"gpu_proc_{shader_name}"
    return _fn


# ── Factory: filter ───────────────────────────────────────────────────

def _make_filt(method_id: str, shader_name: str, method_name: str):
    @method(id=method_id, name=method_name, category="gpu_shaders",
            new_image_contract=True,
            inputs={"image_in": "IMAGE"},
            tags=["gpu", "fast"],
            params=_FILT_PARAMS)
    def _fn(out_dir: Path, seed: int, params=None):
        if params is None:
            params = {}
        inp = params.get("_input_image")  # float32 [0,1] or None
        t = float(params.get("time", 0.0)) * float(params.get("time_scale", 1.0))
        strength = float(params.get("strength", 0.5))
        p2 = float(params.get("p2", 0.5))
        p = (strength, p2, 0.5, 0.5)
        cw, ch = get_canvas()
        img = render_shader(shader_name, (cw, ch), p, t, inp)
        arr = np.array(img, dtype=np.uint8)
        return {"image": arr.astype(np.float32) / 255.0}

    _fn.__name__ = f"gpu_filt_{shader_name}"
    return _fn


# ── Typed-uniform shader nodes (ids 220+) ─────────────────────────────
# These shaders declare named, typed variables (core/shaders.py `uniforms=`)
# instead of the generic p1..p4 vec4. The factory turns every declared
# variable into:
#   • a real node param — slider (float/int), color picker (color, '#rrggbb'
#     default renders a swatch in the UI), or dropdown (choice), AND
#   • a wireable, data-typed SCALAR input port (float/int uniforms), so any
#     scalar output (LFO, luminance mean, counter, …) can drive the variable.
# Inputs/outputs are explicitly data-typed: filters take image_in: IMAGE;
# every node emits image: IMAGE + luminance: FIELD.

_TYPED_SHADER_NODES = [
    ("220", "gradient_gpu2",    "GPU Gradient"),
    ("221", "ascii_art_gpu",    "GPU ASCII Art"),
    ("222", "solid_color_gpu",  "GPU Solid Color"),
    ("223", "checker_gpu2",     "GPU Checkerboard"),
    ("224", "wave_pattern_gpu", "GPU Wave Pattern"),
    ("225", "fbm_noise_gpu",    "GPU FBM Noise"),
    # Categorical coverage expansion (2026-07-10): animated plasma, voronoi
    # cells, and the filter family — kaleidoscope / bloom / posterize / edge.
    ("226", "plasma_gpu2",        "GPU Plasma 2"),
    ("227", "voronoi_gpu2",       "GPU Voronoi 2"),
    ("228", "kaleidoscope_gpu",   "GPU Kaleidoscope"),
    ("229", "bloom_gpu",          "GPU Bloom"),
    ("230", "posterize_gpu",      "GPU Posterize"),
    ("231", "edge_gpu",           "GPU Edge Detect"),
    # Categorical coverage expansion (2026-07-10 pt.2): displacement, RGB split,
    # halftone screen, concentric rings, truchet tiling, pixelate/mosaic.
    ("232", "swirl_gpu",          "GPU Swirl"),
    ("233", "chromatic_gpu",      "GPU Chromatic Aberration"),
    ("234", "halftone_gpu",       "GPU Halftone"),
    ("235", "rings_gpu",          "GPU Rings"),
    ("236", "truchet_gpu",       "GPU Truchet"),
    ("237", "pixelate_gpu",       "GPU Pixelate"),
    # Categorical coverage pt.3 (2026-07-10): signature escape-time fractals
    # with NAMED typed controls (zoom/center/iterations/palette/colors) replacing
    # the opaque p1..p4 shims — Mandelbrot, Julia, Burning Ship, Newton,
    # Sierpinski, Lyapunov.
    ("238", "mandelbrot_typed",   "GPU Mandelbrot"),
    ("239", "julia_typed",        "GPU Julia"),
    ("240", "burning_ship_typed", "GPU Burning Ship"),
    ("241", "newton_typed",       "GPU Newton"),
    ("242", "sierpinski_typed",   "GPU Sierpinski"),
    ("243", "lyapunov_typed",     "GPU Lyapunov"),
    # Categorical coverage pt.4 (2026-07-11): per-pixel filter / color-grade
    # family with NAMED typed controls — box blur, unsharp sharpen, vignette,
    # luminance threshold, hue rotate, ordered (Bayer) dither.
    ("244", "box_blur_gpu",       "GPU Box Blur"),
    ("245", "sharpen_gpu",        "GPU Sharpen"),
    ("246", "vignette_gpu",       "GPU Vignette"),
    ("247", "threshold_gpu",      "GPU Threshold"),
    ("248", "hue_shift_gpu",      "GPU Hue Shift"),
    ("249", "dither_gpu",         "GPU Dither"),
    # Categorical coverage expansion (2026-07-11): closed-form field-eval twins
    # moire / chladni / dunes / quasicrystal / metaballs / nebula / wood / ripples.
    ("250", "moire_typed",         "GPU Moiré"),
    ("251", "chladni_typed",       "GPU Chladni"),
    ("252", "dunes_typed",         "GPU Dunes"),
    ("253", "quasicrystal_typed",  "GPU Quasicrystal"),
    ("254", "metaballs_typed",     "GPU Metaballs"),
    ("255", "nebula_typed",        "GPU Nebula"),
    ("256", "wood_grain_typed",   "GPU Wood Grain"),
    ("257", "ripples_typed",      "GPU Ripples"),
    # Categorical coverage pt.6 (2026-07-11): derivative-field filters that
    # derive a FIELD from the upstream image — Sobel magnitude / direction,
    # Laplacian, Scharr, normal map, gradient orientation flow, emboss. Single
    # image_in: IMAGE; every numeric variable is a wireable SCALAR port.
    ("258", "sobel_mag_typed",    "GPU Sobel Magnitude"),
    ("259", "sobel_dir_typed",    "GPU Sobel Direction"),
    ("260", "laplacian_typed",    "GPU Laplacian"),
    ("261", "scharr_typed",       "GPU Scharr"),
    ("262", "normal_map_typed",   "GPU Normal Map"),
    ("263", "gradient_orient_typed", "GPU Gradient Flow"),
    ("264", "emboss_typed",       "GPU Emboss"),
]

_TIME_SCALE_PARAM = {"description": "animation speed", "min": 0.0, "max": 5.0, "default": 1.0}


def _param_from_uniform(spec: dict) -> dict:
    """Node param spec from a typed uniform spec (same shape the UI expects)."""
    gtype = spec.get("glsl", "float")
    p: dict = {"description": spec.get("description", "")}
    if gtype == "choice":
        p["choices"] = list(spec.get("choices", []))
        p["default"] = spec.get("default", p["choices"][0] if p["choices"] else "")
    elif gtype == "color":
        p["default"] = spec.get("default", "#ffffff")
    else:  # float / int — slider with the uniform's declared range
        if "min" in spec:
            p["min"] = spec["min"]
        if "max" in spec:
            p["max"] = spec["max"]
        p["default"] = spec.get("default", 0)
    return p


def _make_typed(method_id: str, shader_name: str, method_name: str):
    info = SHADERS[shader_name]
    uspec: dict = info.get("uniforms") or {}
    is_filter = info["type"] == "filter"

    params = {uname: _param_from_uniform(spec) for uname, spec in uspec.items()}
    params["time_scale"] = dict(_TIME_SCALE_PARAM)

    inputs: dict[str, str] = {}
    if is_filter:
        inputs["image_in"] = "IMAGE"
    for uname, spec in uspec.items():
        if spec.get("glsl", "float") in ("float", "int"):
            inputs[uname] = "SCALAR"

    @method(id=method_id, name=method_name, category="gpu_shaders",
            new_image_contract=True,
            inputs=inputs,
            outputs={"image": "IMAGE", "luminance": "FIELD"},
            tags=["gpu", "fast", "typed-uniforms"],
            description=info.get("description", ""),
            params=params)
    def _fn(out_dir: Path, seed: int, params=None,
            _shader=shader_name, _uspec=uspec, _is_filter=is_filter):
        if params is None:
            params = {}
        t = float(params.get("time", 0.0)) * float(params.get("time_scale", 1.0))
        named = {u: params.get(u, spec.get("default"))
                 for u, spec in _uspec.items()}
        inp = params.get("_input_image") if _is_filter else None
        cw, ch = get_canvas()
        img = render_shader(_shader, (cw, ch), (0.5, 0.5, 0.5, 0.5), t, inp,
                            named_params=named)
        arr = np.array(img, dtype=np.uint8)
        return {"image": arr.astype(np.float32) / 255.0}

    _fn.__name__ = f"gpu_typed_{shader_name}"
    return _fn


# ── Register all shaders ──────────────────────────────────────────────

for _mid, _sname, _mname in _PROC_SHADERS:
    _make_proc(_mid, _sname, _mname)

for _mid, _sname, _mname in _FILT_SHADERS:
    _make_filt(_mid, _sname, _mname)

for _mid, _sname, _mname in _TYPED_SHADER_NODES:
    _make_typed(_mid, _sname, _mname)


# ── Node → shader map for client-side rendering (parity layer / feature #1) ──
# Lets the browser executor render these EXISTING server nodes client-side for
# the live preview, from the same GLSL source (see core/shaders.py). The server
# remains authoritative for one-shot Run and export.
GPU_SHADER_NODE_MAP: dict[str, dict] = {}
for _mid, _sname, _mname in _PROC_SHADERS:
    GPU_SHADER_NODE_MAP[_mid] = {"shader": _sname, "type": "procedural"}
for _mid, _sname, _mname in _FILT_SHADERS:
    GPU_SHADER_NODE_MAP[_mid] = {"shader": _sname, "type": "filter"}
for _mid, _sname, _mname in _TYPED_SHADER_NODES:
    # typed: client sets u_<name> uniforms from node params (no p1..p4).
    GPU_SHADER_NODE_MAP[_mid] = {"shader": _sname,
                                 "type": SHADERS[_sname]["type"], "typed": True}


# ── P0 client-GPU shims for EXISTING CPU nodes ───────────────────────────────
# Route a pre-existing CPU node's LIVE preview to a client-GPU parity shader
# (see core/shaders.py) without a new node and WITHOUT touching the node's CPU
# fn — the CPU numpy path stays the authoritative export (two-tier precision).
# `param_map` translates the node's real params to the shader's u_params slots
# (p1..p4). Merged into GPU_SHADER_NODE_MAP so the existing /api/shader-sources
# endpoint serves it; client3d.js renderGpuShader reads `param_map`.
CLIENT_GPU_SHIMS: dict[str, dict] = {
    "04": {"shader": "worley_gpu", "type": "procedural",
           "param_map": {"jitter": "p1", "fractal_gain": "p2"}},
    "02": {"shader": "quasicrystal_gpu", "type": "procedural",
           "param_map": {"frequency": "p1", "amplitude": "p2",
                         "rotation": "p3", "waves": "p4"}},
    # ── P0.1 patterns ──
    "03": {"shader": "domain_warp", "type": "procedural",
           "param_map": {"frequency": "p1", "rotation": "p3"}},
    "06": {"shader": "wallpaper_gpu", "type": "procedural",
           "param_map": {"scale_variation": "p1", "color_variation": "p2",
                         "rotation_noise": "p3"}},
    "07": {"shader": "truchet", "type": "procedural",
           "param_map": {"tile_size": "p1", "line_width": "p2"}},
    "08": {"shader": "phyllotaxis_gpu", "type": "procedural",
           "param_map": {"points": "p1", "angle": "p2", "radius_scale": "p3"}},
    "105": {"shader": "morph_grid_gpu", "type": "procedural",
            "param_map": {"warp_strength": "p1", "line_width": "p2"}},
    # ── P0.2 noise/cellular ──
    "05": {"shader": "voronoise", "type": "procedural",
           "param_map": {"scale": "p1", "octaves": "p2"}},
    "29": {"shader": "voronoi", "type": "procedural",
           "param_map": {"n_cells": "p1", "jitter": "p2"}},
    # ── P0.3 escape-time / deterministic fractals ──
    # 33 Fractal Explorer (mandelbrot default): zoom via p1, color_shift p2,
    # center p3/p4. Defaults (0.5) map to full view at center (-0.5, 0).
    "33": {"shader": "mandelbrot_gpu", "type": "procedural",
           "param_map": {"zoom": "p1", "color_shift": "p2",
                         "center_x": "p3", "center_y": "p4"}},
    # 51 Burning Ship: zoom p1, color_offset p2.
    "51": {"shader": "burning_ship_gpu", "type": "procedural",
           "param_map": {"zoom": "p1", "color_offset": "p2"}},
    # 52 Newton: color_speed p1, color_offset p2, zoom p3.
    "52": {"shader": "newton_gpu", "type": "procedural",
           "param_map": {"color_speed": "p1", "color_offset": "p2", "zoom": "p3"}},
    # 66 Julia Set: zoom p3 (default 0.5 = full view). c stays at the shader's
    # fixed famous Julia constant; constant is a string param and not mapped.
    "66": {"shader": "julia", "type": "procedural",
           "param_map": {"zoom": "p3"}},
    # 67 Sierpinski Carpet: depth p1, color_shift p2.
    "67": {"shader": "sierpinski_gpu", "type": "procedural",
           "param_map": {"depth": "p1", "color_shift": "p2"}},
    # 69 Lyapunov: r_min p1, r_max p2. color_mode/color_shift via p3/p4 if added.
    "69": {"shader": "lyapunov_gpu", "type": "procedural",
           "param_map": {"r_min": "p1", "r_max": "p2"}},
    # ── P0.4 per-pixel filters ──
    # 12 Kaleidoscope → existing GPU twin (200). segments p1.
    "12": {"shader": "shader_kaleidoscope", "type": "filter",
            "param_map": {"segments": "p1"}},
    # 17 Glitch Art → existing GPU twin (214). intensity p1.
    "17": {"shader": "shader_glitch_gpu", "type": "filter",
            "param_map": {"intensity": "p1"}},
    # 41 Oil Paint → existing GPU twin (216). radius p1.
    "41": {"shader": "shader_oil_gpu", "type": "filter",
            "param_map": {"radius": "p1"}},
    # 80 Pixel Mosaic → existing GPU twin (208). tile_size p1.
    "80": {"shader": "shader_mosaic_gpu", "type": "filter",
            "param_map": {"tile_size": "p1"}},
    # 42 Fake HDR → new twin. contrast p1, saturation p2, vignette p3, bloom p4.
    "42": {"shader": "hdr_gpu", "type": "filter",
            "param_map": {"contrast": "p1", "saturation": "p2",
                        "vignette": "p3", "bloom": "p4"}},
    # 63 Cross Stitch → new twin. thread_step p1, line_width p2.
    "63": {"shader": "cross_stitch_gpu", "type": "filter",
            "param_map": {"thread_step": "p1", "line_width": "p2"}},
    # 64 Edge Halftone → new twin. dot_spacing p1, dot_size p2.
    "64": {"shader": "edge_halftone_gpu", "type": "filter",
            "param_map": {"dot_spacing": "p1", "dot_size": "p2"}},
    # 74 Swirl Displacement → new twin. strength p1 (0.5 = none).
    "74": {"shader": "swirl_gpu", "type": "filter",
           "param_map": {"strength": "p1"}},
    # ── P0.5 LUT / color ──
    # 11 Gradient: cx/cy are already in [0,1] so they map cleanly onto the
    # twin's center params (0.5 = middle). `direction` (0-360°) and
    # `gradient_type` (choice) do NOT fit the 0.5-neutral u_params convention,
    # so they are left unmapped — the preview shows a default linear gradient
    # at the wired center. The CPU node stays authoritative for exact geometry.
    "11": {"shader": "gradient_gpu", "type": "procedural",
           "param_map": {"cx": "p2", "cy": "p3"}},
    # 10 Color Palette: only `hue_offset` (0-1), `saturation` and `value`
    # (-1 = auto, and the twin treats <=0 as auto) map cleanly. `n_colors` is
    # on a 2-32 count scale that doesn't match the twin's 0-1 ramp, so it is
    # left unmapped (preview uses the twin's default ~17 swatches).
    "10": {"shader": "palette_gpu", "type": "procedural",
           "param_map": {"hue_offset": "p3", "saturation": "p2", "value": "p4"}},
    # 39 Posterize → existing GPU twin (shader_posterize_gpu, P0.4). `n_colors`
    # (2-32 forward) is inverted vs the twin's levels = 16 - p1*14 convention,
    # so it is left unmapped (preview renders the twin's default ~9 levels).
    # `poster_method` is a choice and not mapped (pitfall #14).
    "39": {"shader": "shader_posterize_gpu", "type": "filter",
           "param_map": {}},
    # 77 False Color IR: `strength` (0-1) maps cleanly onto the twin's blend
    # factor. `color_scheme` is a choice string (pitfall #14) so it is left
    # unmapped; the preview defaults to the thermal ramp.
    "77": {"shader": "false_color_gpu", "type": "filter",
           "param_map": {"strength": "p1"}},
    # ── P0.6 field-eval ──
    # 125 Chladni: `m_start`/`n_start` map onto the twin's m/n mode slots
    # (0.5 -> 3.0 neutral); `rotation_speed` -> plate spin, `phase_speed_x` ->
    # shimmer. `m_end`/`n_end` are morph endpoints (used only in anim_mode !=
    # none) and are left unmapped — the live preview shows the start mode. The
    # twin is an exact closed-form preview of the per-pixel displacement field.
    "125": {"shader": "chladni_gpu", "type": "procedural",
            "param_map": {"m_start": "p1", "n_start": "p2",
                          "rotation_speed": "p3", "phase_speed_x": "p4"}},
    # 164 Moiré: `mode` (radial/linear/spiral/hex -> 0..3) maps onto p1,
    # `speed1` -> p2, `speed2` -> p3, `frequency` -> p4. `grid_div` is a choice
    # integer and the twin renders at full res, so it is left unmapped. The twin
    # is an exact parity preview (closed-form function of uv, t).
    "164": {"shader": "moire_gpu", "type": "procedural",
            "param_map": {"mode": "p1", "speed1": "p2",
                          "speed2": "p3", "frequency": "p4"}},
    # 172 Sand Dune Migration: `wind_strength` -> p1, `sediment_supply` -> p2.
    # `anim_mode`/`render_style` are choice strings (pitfall #14) so they are
    # left unmapped — the live preview renders the closed-form "evolve"+
    # hypsometric-height default. Exact parity preview (closed-form function of
    # uv, t); CPU numpy node stays the authoritative export.
    "172": {"shader": "dunes_gpu", "type": "procedural",
            "param_map": {"wind_strength": "p1", "sediment_supply": "p2"}},
    # ── P0.6 field-eval (closed-form f(uv,t) twins, same family as 125/164) ──
    # 53 Metaballs: isovalue -> p1, ball_speed -> p2. behavior/field_fn/style
    # are choice strings (pitfall #14) left unmapped; preview shows orbiting
    # soft-metaball field at the node's default isovalue. Exact parity preview.
    "53": {"shader": "metaballs_gpu", "type": "procedural",
           "param_map": {"isovalue": "p1", "ball_speed": "p2"}},
    # 43 Density Heatmap: sigma -> p1, colormap_shift -> p3. source/style/cmap
    # are choice strings (pitfall #14) left unmapped; preview is a drifting KDE
    # inferno field. Exact parity preview (no seeded point layout divergence).
    "43": {"shader": "heatmap_gpu", "type": "procedural",
           "param_map": {"sigma": "p1", "colormap_shift": "p3"}},
    # 57 Slit Scan: amplitude -> p1, frequency -> p2, slit_type -> p3. source/
    # waveform/style/color_mode are choice strings (pitfall #14) left unmapped;
    # preview is a procedural noise+rainbow displacement. Exact parity preview.
    "57": {"shader": "slitscan_gpu", "type": "procedural",
           "param_map": {"amplitude": "p1", "frequency": "p2", "slit_type": "p3"}},
}
GPU_SHADER_NODE_MAP.update(CLIENT_GPU_SHIMS)


# ── P1 client-GPU sim shims for EXISTING Arch-A simulation nodes ─────────────
# A reaction-diffusion node whose LIVE preview runs on a WebGL2 ping-pong pair of
# RGBA-float state textures (client3d.js owns the {a,b} pair + substep loop). The
# entry names the seed/step/display GLSL (core/shaders.py), the state channel
# count, substeps per rendered frame, and which events force a reseed. As with
# the P0 shims the CPU numpy path (methods/simulations/gray_scott.py) stays the
# authoritative export — two-tier precision, nothing here is rendered server-side.
CLIENT_GPU_SIMS: dict[str, dict] = {
    "155": {
        "type": "sim",
        "seed": "grayscott_seed",
        "step": "grayscott_step",
        "display": "grayscott_display",
        "state_channels": 2,          # U in .r, V in .g
        "substeps": 8,                # Euler steps per rendered frame (live pace)
        "reset_on": ["seed", "param", "loop", "resize"],
        "param_map": {"feed": "p1", "kill": "p2", "diff_u": "p3", "diff_v": "p4"},
    },
    # ── P1.1 textbook sims (reuse the proven ping-pong machinery) ──
    # 32 Reaction-Diffusion: same Gray-Scott engine, parametric preset.
    "32": {
        "type": "sim",
        "seed": "grayscott_seed",
        "step": "grayscott_step",
        "display": "grayscott_display",
        "state_channels": 2,
        "substeps": 8,
        "reset_on": ["seed", "param", "loop", "resize"],
        "param_map": {"feed_rate": "p1", "kill_rate": "p2",
                      "diff_u": "p3", "diff_v": "p4"},
    },
    # 18 / 58 Cellular Automata (Conway's Game of Life).
    "18": {
        "type": "sim",
        "seed": "ca_seed",
        "step": "ca_step",
        "display": "ca_display",
        "state_channels": 2,          # .r alive mask, .g age
        "substeps": 1,                # discrete CA: 1 step per rendered frame
        "reset_on": ["seed", "param", "loop", "resize"],
        "param_map": {"density": "p1"},
    },
    "58": {
        "type": "sim",
        "seed": "ca_seed",
        "step": "ca_step",
        "display": "ca_display",
        "state_channels": 2,
        "substeps": 1,
        "reset_on": ["seed", "param", "loop", "resize"],
        "param_map": {"density": "p1"},
    },
    # 91 BZ Oregonator: 2-var RD, Oregonator kinetics.
    "91": {
        "type": "sim",
        "seed": "bz_seed",
        "step": "bz_step",
        "display": "bz_display",
        "state_channels": 2,          # U in .r, V in .g
        "substeps": 20,               # small dt -> many steps/frame for live pace
        "reset_on": ["seed", "param", "loop", "resize"],
        "param_map": {"epsilon": "p1", "q": "p2", "f": "p3", "Du": "p4"},
    },
    # ── P1.2 RD family (same Laplacian/ping-pong, different reaction term) ──
    # 118 / 119 Lotka-Volterra RD: p1=alpha, p2=beta, p3=gamma, p4=delta.
    "118": {
        "type": "sim",
        "seed": "rd_seed", "step": "lv_step", "display": "rd_display_composite",
        "state_channels": 2, "substeps": 6,
        "reset_on": ["seed", "param", "loop", "resize"],
        "param_map": {"alpha": "p1", "beta": "p2", "gamma": "p3", "delta": "p4"},
    },
    "119": {
        "type": "sim",
        "seed": "rd_seed", "step": "lv_step", "display": "rd_display_composite",
        "state_channels": 2, "substeps": 6,
        "reset_on": ["seed", "param", "loop", "resize"],
        "param_map": {"alpha": "p1", "beta": "p2", "gamma": "p3", "delta": "p4"},
    },
    # 120 LV 3-species food web: U,V,W channels. Approx interaction strengths.
    "120": {
        "type": "sim",
        "seed": "lv3_seed", "step": "lv3_step", "display": "lv3_display",
        "state_channels": 3, "substeps": 4,
        "reset_on": ["seed", "param", "loop", "resize"],
        "param_map": {"noise_amp": "p1"},
    },
    # 121 LV anisotropic: isotropic approximation of the RD step for live preview.
    "121": {
        "type": "sim",
        "seed": "rd_seed", "step": "lv_step", "display": "rd_display_composite",
        "state_channels": 2, "substeps": 6,
        "reset_on": ["seed", "param", "loop", "resize"],
        "param_map": {"alpha": "p1", "beta": "p2"},
    },
    # 133 FitzHugh-Nagumo: p1=epsilon, p2=param_a, p3=param_b, p4=diff_u.
    "133": {
        "type": "sim",
        "seed": "rd_seed", "step": "fhn_step", "display": "rd_display_u",
        "state_channels": 2, "substeps": 8,
        "reset_on": ["seed", "param", "loop", "resize"],
        "param_map": {"epsilon": "p1", "param_a": "p2", "param_b": "p3", "diff_u": "p4"},
    },
    # 143 / 160 Bacterial colony: N nutrient (.r), C colony (.g).
    "143": {
        "type": "sim",
        "seed": "colony_seed", "step": "colony_step", "display": "colony_display",
        "state_channels": 2, "substeps": 4,
        "reset_on": ["seed", "param", "loop", "resize"],
        "param_map": {"growth_rate": "p1", "diff_c": "p2", "consumption": "p3", "death_rate": "p4"},
    },
    "160": {
        "type": "sim",
        "seed": "colony_seed", "step": "colony_step", "display": "colony_display",
        "state_channels": 2, "substeps": 4,
        "reset_on": ["seed", "param", "loop", "resize"],
        "param_map": {"growth_rate": "p1", "diff_c": "p2", "consumption": "p3", "death_rate": "p4"},
    },
    # 168 PM anisotropic RD: p1=b, p2=c, p3=bias (isotropic live approximation).
    "168": {
        "type": "sim",
        "seed": "rd_seed", "step": "turing_step", "display": "rd_display_u",
        "state_channels": 2, "substeps": 6,
        "reset_on": ["seed", "param", "loop", "resize"],
        "param_map": {"b": "p1", "c": "p2"},
    },
    # 169 Turing morphogenesis (Schnakenberg): p1=a, p2=b, p3=gamma, p4=Du.
    "169": {
        "type": "sim",
        "seed": "rd_seed", "step": "turing_step", "display": "rd_display_u",
        "state_channels": 2, "substeps": 6,
        "reset_on": ["seed", "param", "loop", "resize"],
        "param_map": {"a": "p1", "b": "p2", "gamma": "p3", "Du": "p4"},
    },
    # 146 AC + PM Diffusion: single scalar field, Allen-Cahn reaction +
    # Perona-Malik anisotropic diffusion. p1=alpha, p2=K, p3=bias, p4=dt.
    "146": {
        "type": "sim",
        "seed": "acpm_seed", "step": "acpm_step", "display": "acpm_display",
        "state_channels": 1, "substeps": 4,
        "reset_on": ["seed", "param", "loop", "resize"],
        "param_map": {"alpha": "p1", "K": "p2", "bias": "p3", "dt": "p4"},
    },
    # ── P1.3 wave-equation family (leapfrog u/v fields on RGBA-float ping-pong) ──
    # 100 Wave Equation: p1=speed, p2=damping, p3=source_frequency, p4=source_amplitude.
    "100": {
        "type": "sim",
        "seed": "wave_eq_seed", "step": "wave_eq_step", "display": "wave_eq_display",
        "state_channels": 2, "substeps": 4,
        "reset_on": ["seed", "param", "loop", "resize"],
        "param_map": {"wave_speed": "p1", "damping": "p2",
                      "source_frequency": "p3", "source_amplitude": "p4"},
    },
    # 144 Faraday Waves: p1=amplitude, p2=omega0, p3=damping, p4=capillary.
    "144": {
        "type": "sim",
        "seed": "faraday_seed", "step": "faraday_step", "display": "faraday_display",
        "state_channels": 2, "substeps": 4,
        "reset_on": ["seed", "param", "loop", "resize"],
        "param_map": {"amplitude": "p1", "omega0": "p2", "damping": "p3", "capillary": "p4"},
    },
    # 166 Parametric Oscillator Lattice (Oscillon): p1=epsilon, p2=omega0,
    # p3=damping, p4=diffusion.
    "166": {
        "type": "sim",
        "seed": "oscillon_seed", "step": "oscillon_step", "display": "oscillon_display",
        "state_channels": 2, "substeps": 4,
        "reset_on": ["seed", "param", "loop", "resize"],
        "param_map": {"epsilon": "p1", "omega0": "p2", "damping": "p3", "diffusion": "p4"},
    },
    # ── P1.3b — Fluid / surface-growth / lattice sim twins ──
    # 132 Shallow Water Waves: p1=gravity, p2=base_depth, p3=viscosity, p4=amplitude.
    "132": {
        "type": "sim",
        "seed": "sw_seed", "step": "sw_step", "display": "sw_display",
        "state_channels": 3, "substeps": 4,
        "reset_on": ["seed", "param", "loop", "resize"],
        "param_map": {"gravity": "p1", "base_depth": "p2", "nu": "p3", "amplitude": "p4"},
    },
    # 135 KPZ Surface Growth: p1=nu, p2=lambda, p3=noise_amplitude, p4=dt.
    "135": {
        "type": "sim",
        "seed": "kpz_seed", "step": "kpz_step", "display": "kpz_display",
        "state_channels": 2, "substeps": 6,
        "reset_on": ["seed", "param", "loop", "resize"],
        "param_map": {"nu": "p1", "lam": "p2", "noise_amplitude": "p3", "dt": "p4"},
    },
    # 150 FPU Chain Lattice: p1=k2, p2=k3, p3=k4, p4=dt.
    "150": {
        "type": "sim",
        "seed": "fpu_seed", "step": "fpu_step", "display": "fpu_display",
        "state_channels": 2, "substeps": 6,
        "reset_on": ["seed", "param", "loop", "resize"],
        "param_map": {"k2": "p1", "k3": "p2", "k4": "p3", "dt": "p4"},
    },
    # 95 Coupled Logistic: p1=r, p2=eps, p3=decay(trail). Magma colormap display.
    "95": {
        "type": "sim",
        "seed": "cml_seed", "step": "cml_step", "display": "cml95_display",
        "state_channels": 2, "substeps": 2,
        "reset_on": ["seed", "param", "loop", "resize"],
        "param_map": {"r": "p1", "eps": "p2"},
    },
    # 142 Coupled Map Lattice: p1=r, p2=epsilon, p3=decay(trail). Grayscale display.
    "142": {
        "type": "sim",
        "seed": "cml_seed", "step": "cml_step", "display": "cml142_display",
        "state_channels": 2, "substeps": 2,
        "reset_on": ["seed", "param", "loop", "resize"],
        "param_map": {"r": "p1", "epsilon": "p2", "decay": "p3"},
    },
    # ── P1.4 discrete CA / stat-mech twins (clean per-frame ping-pong CAs) ──
    # 87 Cyclic (RPS) CA: p1=n_states, p2=threshold. 1 step per frame.
    "87": {
        "type": "sim",
        "seed": "cyclic_ca_seed", "step": "cyclic_ca_step", "display": "cyclic_ca_display",
        "state_channels": 3, "substeps": 1,
        "reset_on": ["seed", "param", "loop", "resize"],
        "param_map": {"n_states": "p1", "threshold": "p2"},
    },
    # 96 Forest Fire: p1=p(growth), p2=f(lightning), p3=initial_trees. 1 step/frame.
    "96": {
        "type": "sim",
        "seed": "forest_fire_seed", "step": "forest_fire_step", "display": "forest_fire_display",
        "state_channels": 3, "substeps": 1,
        "reset_on": ["seed", "param", "loop", "resize"],
        "param_map": {"p": "p1", "f": "p2", "initial_trees": "p3"},
    },
    # 93 Ising Model (Glauber live approx of Wolff): p1=J, p2=T/Tc. Many Glauber
    # sweeps per frame for a smooth live magnetization wander; below Tc -> domains.
    "93": {
        "type": "sim",
        "seed": "ising_seed", "step": "ising_step", "display": "ising_display",
        "state_channels": 3, "substeps": 6,
        "reset_on": ["seed", "param", "loop", "resize"],
        "param_map": {"J": "p1", "T_max": "p2"},
    },
    # 153 Spatial Prisoner's Dilemma: binary coop/defect lattice, Fermi imitation.
    # p1=temptation T, p2=sucker_payoff S, p3=fermi_K. 1 step/frame; RNG carried in .b.
    "153": {
        "type": "sim",
        "seed": "spd125_seed", "step": "spd125_step", "display": "spd125_display",
        "state_channels": 3, "substeps": 2,
        "reset_on": ["seed", "param", "loop", "resize"],
        "param_map": {"temptation": "p1", "sucker_payoff": "p2", "fermi_K": "p3"},
    },
    # 154 Continuous Spatial PD (replicator dynamics): continuous field s∈[0,1]
    # PDE. p1=temptation, p2=reward, p3=sucker, p4=punishment. R=raw s, G=EMA trail.
    "154": {
        "type": "sim",
        "seed": "spd154_seed", "step": "spd154_step", "display": "spd154_display",
        "state_channels": 3, "substeps": 3,
        "reset_on": ["seed", "param", "loop", "resize"],
        "param_map": {"temptation": "p1", "reward": "p2", "sucker_payoff": "p3", "punishment": "p4"},
    },
    # P1.3 complex-field PDE — Complex Ginzburg-Landau (node 126). Complex field
    # A packed as R=Re(A), G=Im(A); explicit Euler + 5-pt Laplacian, toroidal.
    # alpha/beta/dt are numeric node params → map cleanly to p1..p3.
    "126": {
        "type": "sim",
        "seed": "cgl_seed", "step": "cgl_step", "display": "cgl_display",
        "state_channels": 2, "substeps": 4,
        "reset_on": ["seed", "param", "loop", "resize"],
        "param_map": {"alpha": "p1", "beta": "p2", "dt": "p3"},
    },
    # P1.3 complex-field PDE — Nonlinear Schrödinger (node 124). Same R/G complex
    # field packing as CGL. beta/g/dt are numeric node params → map cleanly to
    # p1..p3; trap_strength → p4 (harmonic confining potential for live preview).
    "124": {
        "type": "sim",
        "seed": "nls_seed", "step": "nls_step", "display": "nls_display",
        "state_channels": 2, "substeps": 4,
        "reset_on": ["seed", "param", "loop", "resize"],
        "param_map": {"beta": "p1", "g": "p2", "dt": "p3", "trap_strength": "p4"},
    },
    # P1.3 complex-field PDE — Gross-Pitaevskii (node 148). Same R/G complex
    # field packing as CGL/NLSE. g/stir_speed/alpha/stir_amp are all numeric
    # node params → map cleanly to p1..p4. Sim-time for the orbiting stirrer is
    # carried in the .b state channel (step shaders get u_time=0, pitfall #6b).
    "148": {
        "type": "sim",
        "seed": "gpe_seed", "step": "gpe_step", "display": "gpe_display",
        "state_channels": 3, "substeps": 3,
        "reset_on": ["seed", "param", "loop", "resize"],
        "param_map": {"g": "p1", "stir_speed": "p2", "alpha": "p3", "stir_amp": "p4"},
    },
}
GPU_SHADER_NODE_MAP.update(CLIENT_GPU_SIMS)


# ── Legacy combined method #82 (kept for backward compatibility) ──────

SHADER_NAMES = sorted([k for k, v in SHADERS.items() if v["type"] == "procedural"])


@method(
    id="82",
    name="GPU Procedural Shaders",
    category="ml_models",
    new_image_contract=True,
    tags=["gpu", "glsl", "fast", "expanded"],
    params={
        "shader": {
            "description": f"shader name: {', '.join(SHADER_NAMES)}",
            "default": "domain_warp",
        },
        "p1": {"description": "generic float param 1", "min": 0.0, "max": 1.0, "default": 0.5},
        "p2": {"description": "generic float param 2", "min": 0.0, "max": 1.0, "default": 0.5},
        "p3": {"description": "generic float param 3", "min": 0.0, "max": 1.0, "default": 0.5},
        "p4": {"description": "generic float param 4", "min": 0.0, "max": 1.0, "default": 0.5},
        "anim_mode": {"description": "animation mode", "choices": ["none", "animate"], "default": "none"},
        "anim_speed": {"description": "animation speed multiplier", "min": 0.0, "max": 5.0, "default": 1.0},
    },
)
def method_gpu_procedural(out_dir: Path, seed: int, params=None):
    """GPU Procedural Shaders — generate imagery from GLSL fragment shaders on the GPU."""
    if params is None:
        params = {}

    raw_time = float(params.get("time", 0.0))
    anim_speed = float(params.get("anim_speed", 1.0))
    t = raw_time * anim_speed
    seed_all(seed)

    shader_name = params.get("shader", "domain_warp")
    if shader_name not in SHADERS or SHADERS[shader_name]["type"] != "procedural":
        shader_name = "domain_warp"

    p = (
        float(params.get("p1", 0.5)),
        float(params.get("p2", 0.5)),
        float(params.get("p3", 0.5)),
        float(params.get("p4", 0.5)),
    )

    cw, ch = get_canvas()
    result = render_shader(shader_name, (cw, ch), p, t)
    arr = np.array(result, dtype=np.uint8)
    capture_frame("82", arr.astype(np.float32) / 255.0)
    save(arr, f"82_{shader_name}_{seed:04d}.png", out_dir)
    return out_dir / f"82_{shader_name}_{seed:04d}.png"
