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


# ── Register all shaders ──────────────────────────────────────────────

for _mid, _sname, _mname in _PROC_SHADERS:
    _make_proc(_mid, _sname, _mname)

for _mid, _sname, _mname in _FILT_SHADERS:
    _make_filt(_mid, _sname, _mname)


# ── Node → shader map for client-side rendering (parity layer / feature #1) ──
# Lets the browser executor render these EXISTING server nodes client-side for
# the live preview, from the same GLSL source (see core/shaders.py). The server
# remains authoritative for one-shot Run and export.
GPU_SHADER_NODE_MAP: dict[str, dict] = {}
for _mid, _sname, _mname in _PROC_SHADERS:
    GPU_SHADER_NODE_MAP[_mid] = {"shader": _sname, "type": "procedural"}
for _mid, _sname, _mname in _FILT_SHADERS:
    GPU_SHADER_NODE_MAP[_mid] = {"shader": _sname, "type": "filter"}


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
