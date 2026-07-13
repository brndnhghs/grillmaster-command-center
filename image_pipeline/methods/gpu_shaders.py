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
from ..core.shaders import render_shader, render_procedural, SHADERS, shader_uses_time

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
    ("412", "sdf_raymarch_gpu",   "GPU SDF Raymarch"),
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
            is_time_varying=shader_uses_time(shader_name),
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
            is_time_varying=shader_uses_time(shader_name),
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
    # Categorical coverage pt.7 (2026-07-11): closed-form pattern generators
    # with NAMED typed controls — spirograph, truchet maze, reaction waves,
    # hex grid, starfield, concentric rings.
    ("265", "spirograph_typed",    "GPU Spirograph"),
    ("266", "truchet_maze_typed",  "GPU Truchet Maze"),
    ("267", "reaction_waves_typed", "GPU Reaction Waves"),
    ("268", "hex_grid_typed",      "GPU Hex Grid"),
    ("269", "starfield_typed",     "GPU Starfield"),
    ("270", "concentric_rings_typed", "GPU Concentric Rings"),
    # Categorical coverage pt.8 (2026-07-11): closed-form math_art patterns
    # with NAMED typed controls — Ulam-spiral homage, hash maze, circle
    # packing, Fourier epicycles, summed waveform, Clifford strange-attractor.
    ("271", "ulam_spiral_typed",     "GPU Ulam Spiral"),
    ("272", "maze_typed",            "GPU Hash Maze"),
    ("273", "circle_packing_typed",  "GPU Circle Packing"),
    ("274", "fourier_circles_typed", "GPU Fourier Circles"),
    ("275", "waveform_typed",        "GPU Waveform"),
    ("276", "strange_attractor_typed", "GPU Strange Attractor"),
    # Categorical coverage pt.8 (2026): closed-form patterns with NAMED typed
    # controls — phyllotaxis dots, guilloché engraving, Lissajous trace, radial
    # wave interference, curl-noise flow field, kaleidoscopic petal bloom.
    ("277", "phyllotaxis_typed",   "GPU Phyllotaxis"),
    ("278", "guilloche_typed",     "GPU Guilloché"),
    ("279", "lissajous_typed",     "GPU Lissajous"),
    ("280", "interference_typed",  "GPU Wave Interference"),
    ("281", "flow_field_typed",    "GPU Flow Field"),
    ("282", "kaleido_bloom_typed", "GPU Kaleido Bloom"),
    # Categorical coverage pt.9 (2026-07-11): closed-form math_art patterns
    # with NAMED typed controls — superformula, harmonograph, Maurer rose,
    # magnetic dipole field, star polygon {n/k}, torus-knot ribbon.
    ("283", "superformula_typed",  "GPU Superformula"),
    ("284", "harmonograph_typed",  "GPU Harmonograph"),
    ("285", "maurer_rose_typed",   "GPU Maurer Rose"),
    ("286", "magnetic_typed",      "GPU Magnetic Field"),
    ("287", "star_polygon_typed",  "GPU Star Polygon"),
    ("288", "torusknot_typed",   "GPU Torus Knot"),
    # Categorical coverage pt.10 (2026-07-11): closed-form pattern nodes with
    # NAMED typed controls — infinite tunnel, vortex/galaxy field, woven fabric,
    # topographic contour map, cross-hatch engraving, domain-warped grid.
    ("289", "tunnel_typed",    "GPU Tunnel"),
    ("290", "vortex_typed",    "GPU Vortex"),
    ("291", "weave_typed",     "GPU Weave"),
    ("292", "contour_typed",   "GPU Contour Map"),
    ("293", "hatch_typed",     "GPU Cross-Hatch"),
    ("294", "gridwarp_typed",  "GPU Warp Grid"),
    # Categorical coverage pt.11 (2026-07-11): extended closed-form procedural
    # family with NAMED typed controls — domain-warped flow, animated caustics,
    # spectral prism, SDF scene, radial energy burst, iridescent bubble foam.
    ("295", "domainwarp_typed", "GPU Domain Warp"),
    ("296", "caustics_typed",   "GPU Caustics"),
    ("297", "prism_typed",      "GPU Spectral Prism"),
    ("298", "sdfscene_typed",   "GPU SDF Scene"),
    ("299", "burst_typed",      "GPU Energy Burst"),
    ("300", "foam_typed",   "GPU Bubble Foam"),
    # Categorical coverage pt.12 (2026-07-11): closed-form pattern node —
    # Gyroid / triply-periodic minimal-surface slice (animation by in-plane
    # spin + slice advance through the 3D field).
    ("301", "gyroid_typed", "GPU Gyroid"),
    # Categorical coverage pt.13 (2026-07-11): closed-form generative-art
    # patterns with NAMED typed controls — Schotter grid, Thue-Morse fractal,
    # crystal diffraction, Apollonian gasket, confocal parabola family,
    # Poincaré-disk hyperbolic tiling.
    ("302", "schotter_typed",    "GPU Schotter"),
    ("303", "thue_morse_typed",  "GPU Thue-Morse"),
    ("304", "crystal_typed",     "GPU Crystal Diffraction"),
    ("305", "apollonian_typed",  "GPU Apollonian Gasket"),
    ("306", "parabola_typed",    "GPU Parabola Family"),
    ("307", "hyperbolic_typed", "GPU Hyperbolic Tiling"),
    # Categorical coverage pt.14 (2026-07-12): real-time volumetric clouds —
    # screen-space fbm density raymarch with single-scatter sun lighting.
    ("308", "clouds_typed", "GPU Volumetric Clouds"),
    # Categorical coverage pt.15 (2026-07-12): closed-form procedural patterns
    # with NAMED typed controls — Droste log-spiral, Voronoi stained glass,
    # Op-Art sinusoidal band distortion. (309 free; 310-315 are CPU method ids.)
    ("316", "droste_typed",        "GPU Droste Spiral"),
    ("317", "stained_glass_typed", "GPU Stained Glass"),
    ("318", "opart_typed",         "GPU Op-Art Waves"),
    ("319", "aurora_typed",        "GPU Aurora Borealis"),
    # Categorical coverage pt.16 (2026-07-13): closed-form procedural — classic
    # Perlin-turbulence marble veining with domain warp (typed, node 320).
    ("320", "marble_typed",         "GPU Marble"),
    # Node 321: Smooth-min Metaballs — Quilez exponential smin union of
    # orbiting SDF spheres. Distinct from node 53 (sum-of-inverse-square
    # field): true SDF + smin so `blend` (k) controls edge softness.
    ("321", "smin_metaballs_gpu",   "GPU Smooth-min Metaballs"),
    # Node 322: Procedural Phasor Noise (Tricard 2019) — sum of complex Gabor
    # kernels; the ARGUMENT (phase) of the accumulated phasor field gives
    # intensity-decoupled oscillating ridges (fingerprint/wood-grain) with
    # locally controllable frequency + orientation. Renders the PHASE, not the
    # magnitude — distinct from any Perlin/Gabor magnitude node.
    ("322", "phasor_noise_gpu",      "GPU Phasor Noise"),
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
            is_time_varying=shader_uses_time(shader_name),
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
    # Typed-uniform shaders (those with a `uniforms=` spec) get named params +
    # wireable SCALAR ports; the rest keep the legacy generic-p1..p4 path.
    if SHADERS.get(_sname, {}).get("uniforms"):
        _make_typed(_mid, _sname, _mname)
    else:
        _make_proc(_mid, _sname, _mname)

for _mid, _sname, _mname in _FILT_SHADERS:
    if SHADERS.get(_sname, {}).get("uniforms"):
        _make_typed(_mid, _sname, _mname)
    else:
        _make_filt(_mid, _sname, _mname)

for _mid, _sname, _mname in _TYPED_SHADER_NODES:
    _make_typed(_mid, _sname, _mname)


# ── Node → shader map for client-side rendering (parity layer / feature #1) ──
# Lets the browser executor render these EXISTING server nodes client-side for
# the live preview, from the same GLSL source (see core/shaders.py). The server
# remains authoritative for one-shot Run and export.
GPU_SHADER_NODE_MAP: dict[str, dict] = {}
# 173-197 are registered as typed-uniform nodes (each shader declares named
# variables → real params + wireable SCALAR ports + IMAGE/FIELD outputs).
for _mid, _sname, _mname in _PROC_SHADERS:
    GPU_SHADER_NODE_MAP[_mid] = {"shader": _sname,
                                 "type": SHADERS[_sname]["type"], "typed": True}
for _mid, _sname, _mname in _FILT_SHADERS:
    GPU_SHADER_NODE_MAP[_mid] = {"shader": _sname, "type": "filter",
                                 "typed": bool(SHADERS.get(_sname, {}).get("uniforms"))}
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
    # Each legacy twin now carries `uniforms=` whose NAMES match the CPU node's
    # REAL numeric params (contract #6: client typed-branch fills u_<name> from
    # params[name], so the uniform name MUST equal the node param name or the
    # live-preview slider is dead). `warp_strength`/`source`/choice params are
    # string/domain-warp controls (pitfall #14) intentionally left unmapped —
    # the twin is the closed-form parity preview; CPU numpy node is authoritative.
    # 33 Fractal Explorer → mandelbrot_gpu. zoom/center_x/center_y/color_shift
    # already match node 33; `iterations` added to the twin so the iter slider
    # drives the preview (was frozen at MAXI=200).
    "33": {"shader": "mandelbrot_gpu", "type": "procedural", "typed": True,
           "param_map": {"zoom": "zoom", "center_x": "center_x",
                         "center_y": "center_y", "iterations": "iterations",
                         "color_shift": "color_shift"}},
    # 51 Burning Ship → burning_ship_gpu. color_speed/color_offset already match
    # node 51; `iterations` added so the iter slider is live (was frozen).
    "51": {"shader": "burning_ship_gpu", "type": "procedural", "typed": True,
           "param_map": {"color_speed": "color_speed", "color_offset": "color_offset",
                         "iterations": "iterations"}},
    # 52 Newton Fractal → newton_gpu. color_speed/color_offset match node 52;
    # `max_iter` added so the iter slider is live (was frozen at MAXI=60).
    "52": {"shader": "newton_gpu", "type": "procedural", "typed": True,
           "param_map": {"color_speed": "color_speed", "color_offset": "color_offset",
                         "max_iter": "max_iter"}},
    # 66 Julia Set → julia. The old shader had NO uniforms= spec, so its
    # live preview was FROZEN at the neutral zoom/center (Route #15 branch-1
    # silent-bypass). Rewritten to carry `iterations`/`escape_radius` matching
    # node 66's REAL numeric params; the Julia `constant` is a STRING param
    # (pitfall #14) and uses the twin's own famous constant for the preview.
    "66": {"shader": "julia", "type": "procedural", "typed": True,
           "param_map": {"iterations": "iterations", "escape_radius": "escape_radius"}},
    # 67 Sierpinski Carpet → sierpinski_gpu. `depth` matches node 67's REAL
    # numeric param depth (1-7). `fractal_type`/`color_mode` are choice strings
    # (pitfall #14) left unmapped.
    "67": {"shader": "sierpinski_gpu", "type": "procedural", "typed": True,
           "param_map": {"depth": "depth"}},
    # 69 Lyapunov Fractal → lyapunov_gpu. r_min/r_max match node 69; `r_max`
    # was read by the body but never declared in uniforms= (dead slider) — now
    # declared. `sequence`/`warmup`/`measure` are choice/int (pitfall #14).
    "69": {"shader": "lyapunov_gpu", "type": "procedural", "typed": True,
           "param_map": {"r_min": "r_min", "r_max": "r_max"}},
    # ── P0.4 per-pixel filters ──
    # 12 Kaleidoscope → existing GPU twin (200). segments p1.
    "12": {"shader": "shader_kaleidoscope", "type": "filter",
            "param_map": {"segments": "p1"}},
    # 17 Glitch Art → existing GPU twin (214). intensity p1.
    "17": {"shader": "shader_glitch_gpu", "type": "filter",
            "param_map": {"intensity": "p1"}},
    # 41 Oil Paint → existing GPU twin (216). radius drives u_radius named uniform
    # (twin is now typed, so the client sets u_radius directly, not via p1).
    "41": {"shader": "shader_oil_gpu", "type": "filter", "typed": True,
           "param_map": {"radius": "radius"}},
    # 80 Pixel Mosaic → existing GPU twin (208). tile_size p1.
    "80": {"shader": "shader_mosaic_gpu", "type": "filter",
            "param_map": {"tile_size": "p1"}},
    # 42 Fake HDR → new twin. contrast p1, saturation p2, vignette p3, bloom p4.
    # 42 Fake HDR → typed-uniform twin. contrast/saturation/vignette/bloom are
    # the node's REAL numeric params (node 42), now wired by name to the
    # shader's u_<name> uniforms (replacing the legacy p1..p4 contract so the
    # live preview actually tracks the sliders — contract #5).
    "42": {"shader": "hdr_gpu", "type": "filter", "typed": True,
            "param_map": {"contrast": "contrast", "saturation": "saturation",
                        "vignette": "vignette", "bloom": "bloom"}},
    # 63 Cross Stitch → typed-uniform twin. thread_step/line_width by name.
    "63": {"shader": "cross_stitch_gpu", "type": "filter", "typed": True,
            "param_map": {"thread_step": "thread_step", "line_width": "line_width"}},
    # 64 Edge Halftone → typed-uniform twin. dot_spacing/dot_size by name.
    "64": {"shader": "edge_halftone_gpu", "type": "filter", "typed": True,
            "param_map": {"dot_spacing": "dot_spacing", "dot_size": "dot_size"}},
    # 74 Swirl Displacement → new twin. strength p1 (0.5 = none).
    "74": {"shader": "swirl_gpu", "type": "filter",
           "param_map": {"strength": "p1"}},
    # 13 Dithering → new twin (Bayer-8 ordered path). `levels` p1 (2..8),
    # `contrast` p2. The CPU node's default `fs` and other error-diffusion
    # algorithms are serial scans that cannot be reproduced per-pixel, so the
    # twin renders the ORDERED (Bayer) approximation and the CPU fn stays
    # 13 Dithering → typed-uniform twin (Bayer ordered path). `levels`/`contrast`
    # are the node's REAL numeric params, now wired by name (contract #5).
    # `algorithm`/`palette`/`noise_type` are string choices (pitfall #14) and are
    # left unmapped. Gives `filters` another GPU mirror.
    "13": {"shader": "dither13_gpu", "type": "filter", "typed": True,
           "param_map": {"levels": "levels", "contrast": "contrast"}},
    # 350 FXAA Anti-Aliasing → twin. edge_threshold p1 (0.5 = neutral
    # medium strength; see pitfall #15 — 0.5 must not be a degenerate extreme).
    # Kept on the legacy p-path: FXAA is an edge-AA filter that is a genuine
    # no-op on smooth regions, so the named-uniform drive-output guard does not
    # apply; migration to typed-uniform is deferred until that guard is relaxed.
    "350": {"shader": "fxaa_gpu", "type": "filter",
            "param_map": {"edge_threshold": "p1"}},
    # 422 Palette Posterize → new twin (ordered-Bayer preview). levels p1,
    # 422 Palette Posterize → typed-uniform twin (ordered-Bayer preview).
    # `levels`/`dither_scale` are the node's REAL numeric params, wired by name
    # (contract #5). The CPU node does median-cut + Floyd-Steinberg + CIELAB
    # (serial/perceptual steps with no per-pixel GPU equivalent), so the twin
    # renders the ORDERED dither path and the CPU fn stays authoritative.
    # `use_lab`/`palette`/`dither` are string choices (pitfall #14), unmapped.
    "422": {"shader": "dither_palette_gpu", "type": "filter", "typed": True,
            "param_map": {"levels": "levels", "dither_scale": "dither_scale"}},
    # 339 Tonal Hatching → typed-uniform twin (tonal_hatching_gpu). The node's
    # REAL numeric params are mapped by name (contract #5); `paper`/`ink_tone`
    # are string choices left unmapped (pitfall #14) so the preview uses the
    # canonical light-paper / black-ink look. CPU fn stays authoritative for
    # every paper/ink palette + the flow/weave/breathe animation modes.
    "339": {"shader": "tonal_hatching_gpu", "type": "filter", "typed": True,
            "param_map": {"spacing": "spacing", "line_width": "line_width",
                          "layers": "layers", "angle": "angle",
                          "contrast": "contrast"}},
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
    # 77 False Color IR → typed-uniform twin. `strength` wired by name (contract
    # #5). `color_scheme` is a choice string (pitfall #14) so the preview locks to
    # the thermal ramp; CPU fn stays authoritative for all schemes.
    "77": {"shader": "false_color_gpu", "type": "filter", "typed": True,
           "param_map": {"strength": "strength"}},
    # ── P0.7 compositing (per-pixel utility) ──
    # __image_to_mask__: converts an IMAGE wire to a MASK. `mode` is a STRING
    # choice param (luminance/red/green/blue/alpha_from_white/invert_luminance)
    # so per pitfall #14 it is NOT mapped — the twin renders the DEFAULT
    # `luminance` extraction. The CPU fn stays authoritative for all six modes.
    # This gives the `compositing` category its first GPU source-of-truth mirror.
    "__image_to_mask__": {"shader": "image_to_mask_gpu", "type": "filter",
                          "param_map": {}},
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
    # 312 Water Caustics: scale->p1, caustic_gain->p2, sharpen->p3, anim_speed->p4.
    # colormode/anim_mode are choice strings (pitfall 14) left unmapped; the twin
    # renders the default ocean colormap in flow-mode. Closed-form f(uv,t) ->
    # exact parity preview. CPU numpy node stays authoritative for export.
    "312": {"shader": "caustics_gpu", "type": "procedural",
            "param_map": {"scale": "p1", "caustic_gain": "p2",
                          "sharpen": "p3", "anim_speed": "p4"}},
    # 68 Anisotropic Kuwahara → typed-uniform twin. `radius`/`anisotropy` wired
    # by name (contract #5); presmooth/blend are not mapped — the twin renders
    # the default strength. CPU numpy node stays authoritative for export.
    "68": {"shader": "anisotropic_kuwahara_gpu", "type": "filter", "typed": True,
           "param_map": {"radius": "radius", "anisotropy": "anisotropy"}},
    # 311 Domain Warping: scale -> p1, warp_strength -> p2, contrast -> p3,
    # octaves -> p4. colormode/warp_levels/anim_mode are choice strings
    # (pitfall #14) left unmapped; preview shows IQ inferno marbling at the
    # node's default scale/warp/contrast. Closed-form f(uv,t) -> exact parity
    # preview; CPU numpy node stays authoritative for export.
    "311": {"shader": "domain_warp_gpu", "type": "procedural",
            "param_map": {"scale": "p1", "warp_strength": "p2",
                          "contrast": "p3", "octaves": "p4"}},
    # 314 Curl-Noise Flow Field: scale -> p1, octaves -> p2, brightness -> p3,
    # anim_mode -> p4 (0=static/1=drift; node choice string decoded to float
    # per pitfall #14 so the preview animates). render_style/colormode are
    # choice strings left unmapped; preview shows the spectral angle->hue field.
    # Closed-form f(uv,t) -> exact parity preview; CPU numpy node authoritative.
    "314": {"shader": "curl_noise_gpu", "type": "procedural",
            "param_map": {"scale": "p1", "octaves": "p2",
                          "brightness": "p3", "anim_mode": "p4"}},
    # 399 CMYK Halftone: spacing -> p1, max_dot -> p2, angle_offset -> p3.
    # ink_set/paper/source/anim_mode are choice strings (pitfall #14) left
    # unmapped; the preview renders the default cmyk ink set on white paper.
    # Closed-form per-pixel screening f(uv, input, params) -> exact parity
    # preview; CPU numpy node stays authoritative for export. Filter twin reads
    # the wired input via u_texture.
    "399": {"shader": "cmyk_halftone_gpu", "type": "filter",
            "param_map": {"spacing": "p1", "max_dot": "p2", "angle_offset": "p3"}},
    # 402 Kaleidoscopic IFS: scale -> p1, fold_angle -> p2, symmetry -> p3,
    # color_shift -> p4. offset_x/offset_y/anim_mode/colormode are choice or
    # multi-value params left unmapped (pitfall #14); the twin renders the
    # default 6-fold orbit coloring. Closed-form f(uv, t) -> exact parity
    # preview; CPU numpy node stays authoritative for export.
    "402": {"shader": "kifs_gpu", "type": "procedural",
            "param_map": {"scale": "p1", "fold_angle": "p2",
                          "symmetry": "p3", "color_shift": "p4"}},
    # 417 Chromatic Aberration → typed-uniform twin. Every uniform name equals a
    # real CPU param of node 417 (contract #5): amount/curve/barrel/vignette/
    # center_drift. `source`/`palette`/`anim_mode` are string choices (pitfall
    # #14) and stay unmapped — the twin filters whatever image is wired in. CPU
    # numpy node stays authoritative for export.
    "417": {"shader": "chromatic_aberration_gpu", "type": "filter", "typed": True,
            "param_map": {"amount": "amount", "curve": "curve",
                          "barrel": "barrel", "vignette": "vignette",
                          "center_drift": "center_drift"}},
    # 419 Thin-Film Interference → typed-uniform twin. Uniform names match node
    # 419's real params: thickness/thickness_range/ior/angle/strength/saturation
    # (contract #5). `source`/`noise_scale`/`palette`/`anim_mode` are string or
    # mode controls (pitfall #14) left unmapped; the twin paints the iridescent
    # radial bands over the wired substrate. CPU numpy node stays authoritative.
    "419": {"shader": "thin_film_gpu", "type": "filter", "typed": True,
            "param_map": {"thickness": "thickness", "thickness_range": "thickness_range",
                          "ior": "ior", "angle": "angle",
                          "strength": "strength", "saturation": "saturation"}},
    # 408 Bloom / Glow → bloom_glow_gpu (typed). Uniform names match node 408's
    # real numeric params: threshold/softness/intensity/radius/streak. source/
    # palette/anim_mode are choice/string (pitfall #14) and stay unmapped.
    "408": {"shader": "bloom_glow_gpu", "type": "filter", "typed": True,
            "param_map": {"threshold": "threshold", "softness": "softness",
                          "intensity": "intensity", "radius": "radius",
                          "streak": "streak"}},
    # 420 Bokeh Lens Blur → bokeh_gpu (typed). radius/blades/anamorphic/rotation/
    # brightness/highlight match node 420's real params. aperture_shape/source/
    # palette/anim_mode are choice/string (pitfall #14) and stay unmapped.
    "420": {"shader": "bokeh_gpu", "type": "filter", "typed": True,
            "param_map": {"radius": "radius", "blades": "blades",
                          "anamorphic": "anamorphic", "rotation": "rotation",
                          "brightness": "brightness", "highlight": "highlight"}},
    # 345 Bilateral Grid → bilateral_grid_gpu (typed). grid_scale/sigma_s/
    # sigma_r/blend match node 345's real params. source/palette/anim_mode are
    # choice/string (pitfall #14) and stay unmapped.
    "345": {"shader": "bilateral_grid_gpu", "type": "filter", "typed": True,
            "param_map": {"grid_scale": "grid_scale", "sigma_s": "sigma_s",
                          "sigma_r": "sigma_r", "blend": "blend"}},
    # ── Categorical coverage expansion (2026-07-11): wire clean name-matching
    # closed-form typed twins onto their existing CPU nodes so the math_art /
    # patterns GPU-source mirror grows by category. Each pairing maps ONLY the
    # CPU node's REAL params that correspond 1:1 to a shader uniform (contract
    # #5); choice/string/multi-value params (pitfall #14) and semantically
    # divergent scalars are intentionally left unmapped. The CPU numpy node
    # stays authoritative for export; these are live-preview twins only.
    # 16 Flow Field → flow_field_typed (procedural). `speed` is the sole 1:1
    # scalar (both = animation speed 0..N). n_particles/wave params diverge.
    "16": {"shader": "flow_field_typed", "type": "procedural", "typed": False,
           "param_map": {"speed": "speed"}},
    # 65 Waveform → waveform_typed (procedural). freq1/2/3 ↔ k1/2/3 (wave
    # number = cyclic frequency, same knob); amp/thick are render-only.
    "65": {"shader": "waveform_typed", "type": "procedural", "typed": False,
           "param_map": {"freq1": "k1", "freq2": "k2", "freq3": "k3"}},
    # 78 Circle Packing → circle_packing_typed (procedural). min_r/max_r are
    # normalized radius bounds (CPU min_radius/max_radius are px but both scale
    # the same normalized range); speed ↔ speed.
    "78": {"shader": "circle_packing_typed", "type": "procedural", "typed": False,
           "param_map": {"min_radius": "min_r", "max_radius": "max_r",
                         "anim_speed": "speed"}},
    # 56 Maze → maze_typed (procedural). wall_thickness ↔ wall (edge width),
    # cell_size ↔ scale (tile size). algorithm/style are choice strings unmapped.
    "56": {"shader": "maze_typed", "type": "procedural", "typed": False,
           "param_map": {"wall_thickness": "wall", "cell_size": "scale"}},
    # 81 Fourier Circles → fourier_circles_typed (procedural). Only `speed`
    # maps cleanly to a CPU param; freq1/2/3 are shader-only harmonic knobs
    # (the CPU node uses n_circles/scale/shape — no freq synth params).
    "81": {"shader": "fourier_circles_typed", "type": "procedural", "typed": False,
           "param_map": {"speed": "speed"}},
    # 406 Harmonograph → harmonograph_typed (procedural). freq1/freq2 ↔ fx/fy,
    # phase ↔ px (both phase offset, default 0), scale ↔ scale (same framing).
    # freq3/4/damping/line_width diverge from the twin's decay/turns/steps.
    "406": {"shader": "harmonograph_typed", "type": "procedural", "typed": False,
            "param_map": {"freq1": "fx", "freq2": "fy", "phase": "px",
                          "scale": "scale"}},
    # 409 Superformula → superformula_typed (procedural). `m` ↔ `m` (exact
    # symmetry count). n1/n2/n3 (CPU supershape exponents) don't match the
    # twin's n/b/c/p uniforms, so they stay unmapped.
    "409": {"shader": "superformula_typed", "type": "procedural", "typed": False,
            "param_map": {"m": "m"}},
    # ── P0.6 field-eval completion (2026-07-12) ──
    # 104 Spherical Harmonics → closed-form twin. max_l/amplitude/glow_strength/
    # anim_speed/twist_amplitude/osc_spread are the node's REAL numeric params
    # (contract #5). The twin declares uniforms= so the client reads them by
    # name (pitfall #14b) — the legacy p1..p4 mapping here is for the
    # param_map-resolves test + documentation only. CPU numpy node stays
    # authoritative for exact spherical-harmonic export.
    "104": {"shader": "spherical_harmonics_gpu", "type": "procedural",
            "param_map": {"max_l": "p1", "amplitude": "p2",
                          "glow_strength": "p3", "anim_speed": "p4"}},
    # 161 Spectral Tapestry → closed-form twin. n_modes/coupling/drift_speed/
    # noise are the node's REAL numeric params (contract #5), wired by name.
    # The CPU spectral-PDE node stays authoritative for export.
    "161": {"shader": "spectral_tapestry_gpu", "type": "procedural",
            "param_map": {"n_modes": "p1", "coupling": "p2",
                          "drift_speed": "p3"}},
    # 473 Gabor Noise -> closed-form twin (live-preview path; the CPU numpy
    # node stays authoritative for exact export). REAL numeric params
    # anisotropy/frequency/falloff (bandwidth) mapped to the twin's p2..p4;
    # the twin's p1 (orientation) is left at its default (0) — the node has no
    # orientation param (anisotropy + wired-image warp drive direction instead).
    "473": {"shader": "gabor_gpu", "type": "procedural",
            "param_map": {"anisotropy": "p2", "frequency": "p3",
                          "falloff": "p4"}},
    # 480 Lens Distortion -> closed-form filter twin (live-preview path; the
    # CPU numpy node stays authoritative for exact export).  REAL numeric
    # params amount/k2/center/aspect/chromatic mapped to the twin's typed
    # uniforms (GPU-First contract #5).
    '480': {'shader': 'lens_distort_gpu', 'type': 'filter', 'typed': True,
            'param_map': {'amount': 'amount', 'k2': 'k2',
                          'center_x': 'center_x', 'center_y': 'center_y',
                          'aspect': 'aspect', 'chromatic': 'chromatic'}},
    # ── Categorical coverage (2026-07-12): recent CPU nodes tagged
    # gpu-twin-candidate (431/432/433/464). Each routes its live preview to a
    # closed-form f(uv,t) GPU twin; the CPU numpy node stays authoritative for
    # export (two-tier precision). Typed-True shims map the node's REAL numeric
    # params 1:1 to the shader's u_<name> uniforms (contract #5). Choice/string
    # params are intentionally left unmapped (pitfall #14).
    '431': {'shader': 'domain_coloring_typed', 'type': 'procedural', 'typed': False,
            'param_map': {'exponent': 'exponent', 'scale': 'scale',
                          'center_x': 'center_x', 'center_y': 'center_y'}},
    '432': {'shader': 'maurer_rose_typed', 'type': 'procedural', 'typed': False,
            'param_map': {'k': 'petals', 'd': 'deg', 'n_lines': 'steps',
                          'line_width': 'thick', 'anim_speed': 'speed'}},
    '433': {'shader': 'low_discrepancy_typed', 'type': 'procedural', 'typed': False,
            'param_map': {'count': 'count', 'radius': 'radius', 'anim_speed': 'speed'}},
    '464': {'shader': 'thin_film_gpu', 'type': 'filter', 'typed': False,
            'param_map': {'thickness': 'thickness', 'thickness_scale': 'thickness_range',
                          'ior': 'ior', 'tilt': 'angle',
                          'intensity': 'strength', 'saturation': 'saturation'}},
    # 503 Conformal Warp -> closed-form Möbius conformal-map twin (live-preview
    # path; the CPU numpy node 503 stays authoritative for export). REAL numeric
    # params scale/warp/anim_speed mapped to the twin's p1/p2/p4. The string
    # choice `function` (moebius/z2/z3/exp/sin/joukowsky) is intentionally
    # left unmapped (pitfall #14) — the twin renders the canonical Möbius map.
    "503": {"shader": "conformal_gpu", "type": "procedural",
            "param_map": {"scale": "p1", "warp": "p2", "anim_speed": "p4"}},
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
    # ── Node 106: Dielectric Breakdown Model (GPU sim twin) ──
    "106": {
        "type": "sim",
        "seed": "dbm_seed",
        "step": "dbm_step",
        "display": "dbm_display",
        "state_channels": 3,
        "substeps": 6,
        "reset_on": ["seed", "param", "loop", "resize"],
        "param_map": {"eta": "p1", "growth_rate": "p2", "cool_rate": "p3", "dielectric": "p4"},
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
    # 499 Sine-Gordon: p1=wave_speed, p2=damping, p3=coupling G, p4=drive_amplitude A.
    "499": {
        "type": "sim",
        "seed": "sine_gordon_seed", "step": "sine_gordon_step", "display": "sine_gordon_display",
        "state_channels": 2, "substeps": 4,
        "reset_on": ["seed", "param", "loop", "resize"],
        "param_map": {"wave_speed": "p1", "damping": "p2",
                      "coupling": "p3", "drive_amplitude": "p4"},
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
        # gravity p1, base_depth p2, amplitude p4 are real numeric params.
        # "nu" was a stale key (node 132 has no such param); the shader's p3
        # slot is unused by the display/step kernels, so p3 is intentionally
        # left unmapped.
        "param_map": {"gravity": "p1", "base_depth": "p2", "amplitude": "p4"},
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
    # ── P1.3b scalar-PDE twins (single/3-channel ping-pong, 5-pt operators) ──
    # 127 Kuramoto-Sivashinsky: nu=p1, dt=p2, noise_amp=p3, aniso_ratio=p4.
    "127": {
        "type": "sim",
        "seed": "ks_seed", "step": "ks_step", "display": "ks_display",
        "state_channels": 1, "substeps": 3,
        "reset_on": ["seed", "param", "loop", "resize"],
        "param_map": {"nu": "p1", "dt": "p2", "noise_amp": "p3", "aniso_ratio": "p4"},
    },
    # 128 Swift-Hohenberg (ε·u − u³ − (1+∇²)²u): epsilon=p1, dt=p2,
    # noise_amp=p3, linear_gain=p4 (preview of (1+∇²) weight).
    "128": {
        "type": "sim",
        "seed": "sh128_seed", "step": "sh128_step", "display": "sh128_display",
        "state_channels": 1, "substeps": 5,
        "reset_on": ["seed", "param", "loop", "resize"],
        "param_map": {"epsilon": "p1", "dt": "p2", "noise_amp": "p3"},
    },
    # 157 Swift-Hohenberg (r·u − (∇²+q0²)²u − u³): r=p1, q0=p2, dt=p3, noise=p4.
    "157": {
        "type": "sim",
        "seed": "sh157_seed", "step": "sh157_step", "display": "sh157_display",
        "state_channels": 1, "substeps": 5,
        "reset_on": ["seed", "param", "loop", "resize"],
        "param_map": {"r": "p1", "q0": "p2", "dt": "p3", "noise": "p4"},
    },
    # 162 Coupled Rössler Oscillator Array (3-var): a=p1, b=p2, c_ross=p3,
    # omega=p4. coupling D fixed in the twin (CPU authoritative for export).
    "162": {
        "type": "sim",
        "seed": "ross_seed", "step": "ross_step", "display": "ross_display",
        "state_channels": 3, "substeps": 2,
        "reset_on": ["seed", "param", "loop", "resize"],
        "param_map": {"a": "p1", "b": "p2", "c_ross": "p3", "omega": "p4"},
    },
    # 170 Phase Field Crystal: epsilon=p1, dt=p2, noise=p3, r2(=r/2)=p4.
    "170": {
        "type": "sim",
        "seed": "pfc_seed", "step": "pfc_step", "display": "pfc_display",
        "state_channels": 1, "substeps": 4,
        "reset_on": ["seed", "param", "loop", "resize"],
        "param_map": {"epsilon": "p1", "dt": "p2", "noise": "p3"},
    },
    # ── P1.5 phase-field solidification ──
    # 122 Dendritic Solidification: Allen-Cahn φ (.r) + passive thermal u (.g).
    # undercooling=p1, anisotropy=p2, symmetry=p3, dt=p4. n_seeds/impurity/
    # anim_mode are count/choice params (pitfall #14) left unmapped — the twin
    # renders the default single-seed "evolve" nucleus. CPU node authoritative.
    "122": {
        "type": "sim",
        "seed": "dendrite_seed", "step": "dendrite_step", "display": "dendrite_display",
        "state_channels": 2, "substeps": 3,
        "reset_on": ["seed", "param", "loop", "resize"],
        "param_map": {"undercooling": "p1", "anisotropy": "p2",
                      "symmetry": "p3", "dt": "p4"},
    },
    # 163 Fractional Laplacian RD: Gray-Scott reaction + α-modulated diffusion
    # breadth (local proxy for the Fourier fractional operator; CPU authoritative
    # for the exact (-∇²)^(α/2)). feed=p1, kill=p2, alpha=p3, diff_v=p4. Reuses
    # grayscott_seed; fire-colormap display. anim_mode/render_style are choice
    # strings (pitfall #14) left unmapped — twin renders the default fire mitosis.
    "163": {
        "type": "sim",
        "seed": "grayscott_seed", "step": "frac_rd_step", "display": "frac_rd_display",
        "state_channels": 2, "substeps": 8,
        "reset_on": ["seed", "param", "loop", "resize"],
        "param_map": {"feed": "p1", "kill": "p2", "alpha": "p3", "diff_v": "p4"},
    },
    # ── P1.6 continuously-evolving field-PDE twins ──
    # 99 Active Nematic: Landau-de Gennes Q-tensor (Qxx=.r, Qxy=.g). activity=p1,
    # elastic_d=p2, A_landau=p3, noise_amp=p4. Thermal noise (defect nucleation)
    # reproduced as state-dependent hash noise. anim_mode is a choice string
    # (pitfall #14) left unmapped — twin renders the default "evolve" turbulence.
    "99": {
        "type": "sim",
        "seed": "nematic_seed", "step": "nematic_step", "display": "nematic_display",
        "state_channels": 2, "substeps": 10,
        "reset_on": ["seed", "param", "loop", "resize"],
        "param_map": {"activity": "p1", "elastic_d": "p2",
                      "A_landau": "p3", "noise_amp": "p4"},
    },
    # 348 Droplet Erosion (was "156 Hydraulic Erosion" before the node was
    # renumbered): 3-field terrain twin (height=.r, water=.g, sediment=.b).
    # The CPU node 348 is the particle/droplet model; this twin is a 3-field
    # grid visual-style parity (CPU stays authoritative for exact routing).
    # Map the node's REAL erosion knobs onto the shader's u_params slots:
    #   erosion_rate → K_e (p2, erosion strength)
    #   deposition   → K_d (p3, deposition rate)
    #   min_slope    → theta (p4, angle of repose)
    # The grid twin's rain (p1) is driven by `evaporation` (closest water-cycle
    # knob on the droplet model); the live preview stays non-black via the
    # shader's clamp on u_params. Choice params (colormap/hillshade/light_angle)
    # are cosmetic and left unmapped (pitfall #14).
    "348": {
        "type": "sim",
        "seed": "erosion_seed", "step": "erosion_step", "display": "erosion_display",
        "state_channels": 3, "substeps": 4,
        "reset_on": ["seed", "param", "loop", "resize"],
        "param_map": {"evaporation": "p1", "erosion_rate": "p2",
                      "deposition": "p3", "min_slope": "p4"},
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
