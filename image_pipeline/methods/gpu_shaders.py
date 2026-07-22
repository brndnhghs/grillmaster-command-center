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
    ("413", "dot_noise_gpu",      "GPU Dot Noise"),
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
}

_FILT_PARAMS = {
    "strength": {"description": "effect strength", "min": 0.0, "max": 1.0, "default": 0.5},
    "p2": {"description": "shader param 2", "min": 0.0, "max": 1.0, "default": 0.5},
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
        t = float(params.get("time", 0.0))
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
        t = float(params.get("time", 0.0))
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
    # Node 323: Raymarched 3D Gyroid TPMS — sphere-traced triply-periodic
    # minimal surface with lambert+specular shading and orbiting camera.
    # Distinct from node 301 (flat 2D scalar-field slice): full 3D volume with
    # depth, self-occlusion and lighting.
    ("323", "gyroid_raymarch_typed", "GPU Raymarched Gyroid"),
    # Categorical coverage pt.17 (2026-07-14): closed-form recursive subdivision
    # fractal with NAMED typed controls — Sierpinski (Menger) carpet. Cells are
    # coloured by recursion depth; the plane spins and scale breathes with time.
    ("324", "menger_typed", "GPU Menger Carpet"),
    # Categorical coverage pt.18 (2026-07-14): closed-form atmospheric sky —
    # Nishita single-scattering GPU twin of CPU node 471. Per-pixel ray-march
    # (no ping-pong state), animated sun day-arc via u_time. Named typed
    # uniforms mirror node 471's real numeric params (contract #5/#6).
    ("325", "nishita_sky_gpu", "GPU Nishita Sky"),
    # Categorical coverage pt.19 (2026-07-14): stochastic hex-tiling filter —
    # Heitz & Neyret (HPG 2018) histogram-preserving blending operator that
    # tiles the wired input image across the plane with NO visible repetition.
    # A true FILTER (image_in: IMAGE) with named typed controls.
    ("327", "hex_tiling_gpu", "GPU Hex Tiling"),
    # Categorical coverage pt.20 (2026-07-16): Interior Mapping (van Dongen,
    # CGI 2008) — believable 3D rooms behind a flat facade via per-pixel
    # ray-box intersection, NO added geometry. Closed-form f(uv,t) procedural
    # twin: eye ray into a repeating room-grid, nearest interior wall shaded
    # with depth tint + hashed per-room window lights (u_time twinkle).
    ("328", "interior_mapping_typed", "GPU Interior Mapping"),
    # Node 330: Kaleidoscopic IFS - box-fold + sphere-fold (Knighty/Kali 2010).
    # Distinct from node 402 kifs_gpu (wedge + scale only): adds the sphere
    # fold (minR/maxR radius clamp) that opens the characteristic holes, plus
    # a per-iteration rotation. All 5 controls are wireable SCALAR ports so
    # LFO/counter nodes can drive the live kaleidoscope animation, and
    # contrast-only static culls are avoided by the genuine u_time motion.
    ("330", "kifs_spherefold_gpu", "GPU KIFS Fractal"),
    # Node 331: Mandelbulb — 3D escape-time fractal (White & Nylander 2009),
    # sphere-traced via the Hart et al. 1989 distance estimator. The canonical
    # "3D Mandelbrot": distinct from the 2D escape-time family and from the 3D
    # TPMS raymarches (gyroid/menger). Genuinely time-varying (power morph +
    # orbiting camera) so animation drivers have a visibly-responsive target.
    ("331", "mandelbulb_gpu", "GPU Mandelbulb"),
    # Node 332: De Jong Attractor - GPU live-preview twin of CPU node 498.
    # Closed-form de Jong chaos map rendered as a single-pass density field
    # tone-mapped with the inferno ramp - parity with node 498 density colouring.
    # Named typed uniforms mirror node 498 real numeric params (a/b/c/d/exposure).
    # morph+speed animate the params via u_time (CPU anim_mode morph_all) so the
    # live preview is genuinely time-varying. CPU numpy node stays authoritative.
    # 332 is the free ID above 301 (333-334 also free; 335 taken by domain warp).
    ("346", "de_jong_typed", "GPU De Jong Attractor"),
    # Node 335: Domain Warping (Inigo Quilez, 2015) — fbm(fbm(p + fbm(p)))
    # two-level noise feed-forward gives marbled organic flow distinct from
    # single fbm (node 225); animated by scrolling the inner warp with u_time
    # so contrast-only static culls are avoided. (ID 335: 332-334 are
    # taken by CPU nodes — GPU-typed nodes must use free IDs above 301. Distinct
    # shader name domain_warp_palette_gpu vs node 311's domain_warp_gpu: the
    # 311 twin keeps the IQ-inferno look; this node adds a 4-colour palette.)
    ("335", "domain_warp_palette_gpu", "GPU Domain Warp"),
    # Node 309: Mandelbox — 3D escape-time fractal (Tom Lowe 2010), the box-fold
    # + sphere-fold companion to the Mandelbulb (node 331). DE raymarch (Hart et
    # al. 1989); the negative scale yields the iconic tiled infinite-rooms look.
    # Genuinely time-varying (orbiting camera + scale breathing) so it survives
    # the contrast-only static liveness cull and feeds animation drivers.
    # (309 is the free ID above 301 — 310-315 are taken by CPU method ids.)
    ("309", "mandelbox_gpu", "GPU Mandelbox"),
    # Node 352: Gerstner Ocean — analytic trochoidal-wave height field with
    # Blinn-Phong sun glitter (typed GPU twin of CPU node 963). Closed-form
    # f(uv,t): wave phases advance with u_time so the live preview is genuinely
    # animated (survives the contrast-only static liveness cull). CPU
    # numpy node 963 stays authoritative for export (two-tier precision).
    # 352 is the free ID above 301.
    ("352", "gerstner_ocean_gpu", "GPU Gerstner Ocean"),
    # Node 360: Gyroid TPMS — closed-form triply-periodic minimal-surface shell
    # on a swept slice plane (typed GPU twin of CPU node 964). Genuinely
    # time-varying: the slice-plane z advances with u_time so the 2D cross
    # section morphs continuously (survives the contrast-only static
    # cull). CPU numpy node 964 stays authoritative for export.
    # 360 is the free ID above 301.
    ("360", "gyroid_tpms_gpu", "GPU Gyroid TPMS"),
    # Node 361: Phasor Noise — sparse-convolution complex-phasor field (typed GPU
    # twin of CPU node 1006). Closed-form f(uv,t): the global phase advances with
    # u_time so the live preview is genuinely animated (survives the
    # contrast-only static liveness cull). CPU numpy node 1006 stays authoritative
    # for export. 361 is the free ID above 360.
    ("361", "phasor_noise_gpu", "GPU Phasor Noise"),
]


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
        t = float(params.get("time", 0.0))
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
    # ── GPU-First gap mirrors: closed-form f(uv,t) twins (typed-uniform contract) ──
    # 523 Aurora Borealis, 954 Autostereogram, 512 SIREN Field — each has no close
    # existing twin, so we add a brand-new closed-form GLSL twin (core/shaders.py)
    # and route the CPU node's live preview to it. CPU fns stay authoritative for
    # export; every numeric CPU param is bound to a named u_<name> uniform/port.
    "523": {"shader": "aurora_gpu", "type": "procedural", "typed": True,
            "param_map": {"curtain_count": "curtain_count", "drift_speed": "drift_speed",
                          "intensity": "intensity", "beam_height": "beam_height",
                          "color_shift": "color_shift", "turbulence": "turbulence",
                          "star_density": "star_density", "red_fringe": "red_fringe"}},
    "954": {"shader": "autostereogram_gpu", "type": "procedural", "typed": True,
            "param_map": {"separation": "separation", "depth_scale": "depth_scale",
                          "tile_size": "tile_size"}},
    "512": {"shader": "siren_gpu", "type": "procedural", "typed": True,
             "param_map": {"omega0": "omega0", "omega": "omega",
                          "weight_scale": "weight_scale", "coord_scale": "coord_scale"}},
    # P0.5 typed-uniform procedural twins (535/534/953). Every numeric CPU
    # param maps to a named u_<name> uniform; choice/string params and the
    # CPU-only variable counts (n_spots/n_drops/n_tines) are dropped +
    # justified in GPU_PREVIEW_DROP_ALLOW. CPU fns stay authoritative.
    "535": {"shader": "flow_noise_gpu", "type": "procedural", "typed": True,
             "param_map": {"scale": "scale", "octaves": "octaves", "spin_var": "spin_var",
                          "advect": "advect", "contrast": "contrast", "anim_speed": "anim_speed"}},
    "534": {"shader": "spot_noise_gpu", "type": "procedural", "typed": True,
             "param_map": {"spot_size": "spot_size", "stretch": "stretch",
                          "contrast": "contrast", "anim_speed": "anim_speed"}},
    "953": {"shader": "marbling_gpu", "type": "procedural", "typed": True,
             "param_map": {"drop_radius": "drop_radius", "tine_strength": "tine_strength",
                          "tine_sharpness": "tine_sharpness", "anim_speed": "anim_speed",
                          "seed": "seed"}},
    # 487 Galaxy Generator, 441 Marching Squares Contours, 108 4D Hypercube —
    # each is a per-pixel closed-form generator with no close existing twin.
    "487": {"shader": "galaxy_gpu", "type": "procedural", "typed": True,
            "param_map": {"arms": "arms", "tightness": "tightness", "arm_spread": "arm_spread",
                          "bulge_size": "bulge_size", "inclination": "inclination",
                          "rotation_speed": "rotation_speed", "brightness": "brightness"}},
    "441": {"shader": "contours_gpu", "type": "procedural", "typed": True,
            "param_map": {"n_levels": "n_levels", "grid_step": "grid_step",
                          "line_alpha": "line_alpha", "flow_amp": "flow_amp",
                          "noise_amp": "noise_amp"}},
    "108": {"shader": "hypercube_gpu", "type": "procedural", "typed": True,
            "param_map": {"speed_xw": "speed_xw", "speed_yw": "speed_yw",
                          "proj_radius": "proj_radius", "line_width": "line_width",
                          "inner_hue": "inner_hue", "outer_hue": "outer_hue"}},
    # 486 Radial & Spin Blur, 438 Subsurface Scatter (SSSS), 451 Gabor Filter —
    # each is a per-pixel closed-form filter with no close existing twin, so it
    # gets a brand-new typed-uniform GLSL twin (core/shaders.py) wired via a
    # CLIENT_GPU_SHIMS entry. Every numeric CPU param is bound to a named
    # u_<name> uniform/SCALAR port (typed-uniform contract). Choice params and
    # the CPU-only source generators are dropped (GPU_PREVIEW_DROP_ALLOW).
    "486": {"shader": "radial_spin_blur_gpu", "type": "filter", "typed": True,
            "param_map": {"length": "length", "center_x": "center_x",
                          "center_y": "center_y", "anim_speed": "anim_speed"}},
    "438": {"shader": "ssss_gpu", "type": "filter", "typed": True,
            "param_map": {"radius": "radius", "samples": "samples",
                          "falloff": "falloff", "strength": "strength",
                          "anim_speed": "anim_speed"}},
    "439": {"shader": "gabor_filter_gpu", "type": "filter", "typed": True,
            "param_map": {"orientation": "orientation", "frequency": "frequency",
                          "sigma": "sigma", "aspect": "aspect", "phase": "phase",
                          "contrast": "contrast", "anim_speed": "anim_speed"}},
    # 995 Gravitational Lensing, 950 SDF Scene, 967 Interior Mapping — each is a
    # per-pixel closed-form f(uv,t) procedural generator with no close existing
    # twin, so each gets a brand-new typed-uniform GLSL twin (core/shaders.py)
    # wired via a CLIENT_GPU_SHIMS entry. Every numeric CPU param is bound to a
    # named u_<name> uniform/SCALAR port (typed-uniform contract). Choice params
    # (palette/mode/pattern/color_mode) and CPU-only per-frame frame counts are
    # dropped (GPU_PREVIEW_DROP_ALLOW); the twins animate continuously from
    # u_time so the preview is always live and the CPU export is authoritative.
    "995": {"shader": "grav_lens_gpu", "type": "procedural", "typed": True,
            "param_map": {"einstein_radius": "einstein_radius",
                          "star_density": "star_density", "nebula": "nebula",
                          "neb_scale": "neb_scale", "exposure": "exposure",
                          "ring_brightness": "ring_brightness",
                          "ring_width": "ring_width", "anim_speed": "anim_speed"}},
    "950": {"shader": "sdf_scene_gpu", "type": "procedural", "typed": True,
            "param_map": {"scale": "scale", "blend": "blend",
                          "repetition": "repetition", "glow": "glow",
                          "bands": "bands", "band_mix": "band_mix",
                          "anim_speed": "anim_speed"}},
    "967": {"shader": "interior_mapping_gpu", "type": "procedural", "typed": True,
            "param_map": {"n_cols": "n_cols", "n_rows": "n_rows",
                          "room_depth": "room_depth", "perspective": "perspective",
                          "pan_x": "pan_x", "pan_y": "pan_y",
                          "frame_width": "frame_width", "lit_fraction": "lit_fraction",
                          "warmth": "warmth", "anim_speed": "anim_speed"}},
    # 522 CRT Emulation, 527 VHS Tape — per-pixel closed-form filter twins.
    # Every numeric CPU slider is bound to a named u_<name> uniform/SCALAR port
    # (typed-uniform contract). Choice params (source/palette/anim_mode) and the
    # timeline-driven time/anim_speed are auto-justified per the coverage guard.
    "522": {"shader": "crt_emulation_gpu", "type": "filter", "typed": True,
            "param_map": {"curvature": "curvature", "scanline": "scanline",
                          "scan_freq": "scan_freq", "mask_strength": "mask_strength",
                          "vignette": "vignette", "chroma": "chroma",
                          "roll_speed": "roll_speed", "flicker": "flicker",
                          "brightness": "brightness"}},
    "527": {"shader": "vhs_tape_gpu", "type": "filter", "typed": True,
            "param_map": {"chroma_smear": "chroma_smear", "chroma_shift": "chroma_shift",
                          "luma_noise": "luma_noise", "line_jitter": "line_jitter",
                          "tracking": "tracking", "roll_speed": "roll_speed",
                          "skew": "skew", "saturation": "saturation",
                          "contrast": "contrast", "brightness": "brightness"}},
    # ── 445 Diffraction Grating → diffraction_gpu (typed-uniform filter twin) ──
    # Faithful closed-form preview of node 445's Stam/GPU-Gems iridescence. Every
    # numeric CPU param (groove_spacing/curvature/interp/light_x/light_y/strength/
    # saturation) is wired by name to a u_<name> uniform/SCALAR port. `source`/
    # `palette`/`anim_mode`/`noise_scale` are CPU-only (choice/flow-source) and
    # left unmapped (pitfall #14); the twin renders the default concentric sheen
    # and animates continuously from u_time. CPU node stays authoritative.
    "445": {"shader": "diffraction_gpu", "type": "filter", "typed": True,
            "param_map": {"groove_spacing": "groove_spacing", "curvature": "curvature",
                          "interp": "interp", "light_x": "light_x", "light_y": "light_y",
                          "strength": "strength", "saturation": "saturation"}},
    # ── 489 Film Grain → film_grain_gpu (typed-uniform filter twin) ──
    # Luminance-adaptive emulsion grain. intensity/adapt/grain_size wired by name.
    # `color`/`source`/`palette`/`anim_mode`/`noise_amp`/`blur_sigma` are CPU-only
    # (choice/source-generation) and left unmapped (pitfall #14); the twin grains
    # the live preview and flickers with u_time. CPU node authoritative.
    "489": {"shader": "film_grain_gpu", "type": "filter", "typed": True,
            "param_map": {"intensity": "intensity", "adapt": "adapt",
                          "grain_size": "grain_size"}},
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
    # ── 326 Hash Field (Müller et al. 2022 multiresolution hash encoding) ──
    # CPU node stays authoritative export; this routes its live preview to the
    # matching closed-form GLSL twin. CPU params map to the shader's u_params
    # slots: scale->p1, detail(levels)->p2, hue->p3, contrast->p4.
    "326": {"shader": "hash_field_gpu", "type": "procedural",
            "param_map": {"scale": "p1", "detail": "p2", "hue": "p3",
                          "contrast": "p4"}},
    "105": {"shader": "morph_grid_gpu", "type": "procedural",
            "param_map": {"warp_strength": "p1", "line_width": "p2"}},
    # ── P0.2 noise/cellular ──
    "05": {"shader": "voronoise", "type": "procedural",
           "param_map": {"scale": "p1", "octaves": "p2"}},
    "29": {"shader": "voronoi", "type": "procedural",
           "param_map": {"n_cells": "p1", "jitter": "p2"}},
    # 31 Plasma Fractal -> plasma_gpu (typed, node-param-named uniforms).
    # Choice params (source/terrain/color_mode/palette/water_level/light_angle/
    # erosion) are intentionally unmapped (pitfall #14): the twin is the
    # closed-form parity preview; the CPU diamond-square node stays authoritative.
    "31": {"shader": "plasma_gpu", "type": "procedural", "typed": True,
           "param_map": {"size": "size", "roughness": "roughness",
                         "octaves": "octaves", "seed_strength": "seed_strength"}},
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
                         "escape_radius": "escape_radius",
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
    # 493 Color Grading (OKLab) → typed-uniform twin (color_grade_gpu). The
    # node's REAL numeric grade params are mapped by name (contract #5); the
    # string params (source/palette/invert) + anim modes stay CPU-only. Gives
    # the `filters` category another perceptual-color GPU mirror.
    "493": {"shader": "color_grade_gpu", "type": "filter", "typed": True,
            "param_map": {"exposure": "exposure", "contrast": "contrast",
                          "gamma": "gamma", "saturation": "saturation",
                          "hue_rotate": "hue_rotate", "temperature": "temperature",
                          "tint": "tint", "vignette": "vignette"}},
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
    # 125 Chladni: typed-uniform twin (GPU-First contract #5) — node params
    # route BY NAME to the twin's u_<name> uniforms. The legacy p1..p4 mapping
    # was a SILENT NO-OP: the twin declares `uniforms=`, so the client's
    # typed-live branch ignores p-slots and freezes controls at defaults
    # (pitfall #14b / frozen-typed class). Now m_start/n_start/rotation_speed/
    # phase_speed_x drive u_m_start/u_n_start/u_rotation_speed/u_phase_speed_x.
    # `m_end`/`n_end` are morph endpoints (anim_mode != none only) left unmapped
    # — the live preview shows the start mode. CPU export stays authoritative.
    "125": {"shader": "chladni_gpu", "type": "procedural", "typed": True,
            "param_map": {"m_start": "m_start", "n_start": "n_start",
                          "rotation_speed": "rotation_speed",
                          "phase_speed_x": "phase_speed_x"}},
    # 164 Moiré: typed-uniform twin (GPU-First contract #5) — node params route
    # BY NAME to the twin's u_<name> uniforms. Legacy p1..p4 was a SILENT NO-OP
    # (twin declares `uniforms=`, client typed-live branch ignores p-slots,
    # pitfall #14b / frozen-typed class). mode/speed1/speed2/frequency now drive
    # u_mode/u_speed1/u_speed2/u_frequency. `grid_div` (choice int, full-res
    # render) left unmapped. CPU export stays authoritative.
    "164": {"shader": "moire_gpu", "type": "procedural", "typed": True,
            "param_map": {"mode": "mode", "speed1": "speed1",
                          "speed2": "speed2", "frequency": "frequency"}},
    # 172 Sand Dune Migration: `wind_strength` -> p1, `sediment_supply` -> p2.
    # `anim_mode`/`render_style` are choice strings (pitfall #14) so they are
    # left unmapped — the live preview renders the closed-form "evolve"+
    # hypsometric-height default. Exact parity preview (closed-form function of
    # uv, t); CPU numpy node stays the authoritative export.
    "172": {"shader": "dunes_gpu", "type": "procedural",
            "param_map": {"wind_strength": "p1", "sediment_supply": "p2"}},
    # 513 Caustics: typed-uniform twin — depth/scale/gain/waves map by NAME
    # (the client typed-live path reads node params by uniform name; the p-slot
    # values here are legacy-path fallbacks only). `colormode`/`anim_mode` are
    # choice strings (pitfall #14) left unmapped → preview uses the aqua/animated
    # closed-form default. Analytic-Hessian inverse-magnification caustic is a
    # pure function of (uv, t); CPU numpy node stays the authoritative export.
    "513": {"shader": "caustics513_gpu", "type": "procedural",
            "param_map": {"depth": "p1", "scale": "p2", "gain": "p3", "waves": "p4"}},
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
    # 471 Nishita Atmospheric Sky: typed-uniform twin (nishita_sky_gpu). Every
    # numeric param of node 471 is mapped by NAME to u_<name> (contract #5, so
    # CPU numpy node 471 stays authoritative for export. The twin glides the sun
    # elevation with the executor's `time` uniform for live animation parity.
    "471": {"shader": "nishita_sky_gpu", "type": "procedural", "typed": True,
            "param_map": {"sun_elevation": "sun_elevation", "sun_azimuth": "sun_azimuth",
                          "rayleigh_k": "rayleigh_k", "mie_k": "mie_k",
                          "exposure": "exposure", "fov": "fov",
                          "sun_disk_radius": "sun_disk_radius"}},
    # 312 Water Caustics: scale->p1, caustic_gain->p2, sharpen->p3, anim_speed->p4.
    # colormode/anim_mode are choice strings (pitfall 14) left unmapped; the twin
    # renders the default ocean colormap in flow-mode. Closed-form f(uv,t) ->
    # exact parity preview. CPU numpy node stays authoritative for export.
    "312": {"shader": "caustics_gpu", "type": "procedural",
            "param_map": {"scale": "p1", "caustic_gain": "p2",
                          "sharpen": "p3", "anim_speed": "p4"}},
    # 528 Voronoise: typed-uniform twin. scale/jitter/smoothness/octaves/
    # lacunarity/gain/contrast map by NAME (the client typed-live path reads
    # node params by uniform name). colormode/palette/anim_mode/source are
    # choice/string params (pitfall #14) left unmapped -> preview uses the
    # inferno default (node 528's default colormode). Closed-form f(uv,t):
    # feature points orbit with u_time so the live preview is genuinely
    # animated. CPU numpy node 528 stays authoritative for export.
    "528": {"shader": "voronoise_typed", "type": "procedural", "typed": True,
            "param_map": {"scale": "scale", "jitter": "jitter",
                          "smoothness": "smoothness", "octaves": "octaves",
                          "lacunarity": "lacunarity", "gain": "gain",
                          "contrast": "contrast"}},
    # 425 Horizon Ambient Occlusion (HBAO): typed-uniform twin. Every numeric
    # param (freq/octaves/height_scale/radius/directions/steps/jitter/light_az/
    # light_el/ambient/contrast) and the choice params (mode/colormode/anim_mode)
    # map BY NAME to the twin's u_<name> uniforms; the client typed-live path
    # reads them by name (pitfall #14b). Closed-form f(uv,t) over a procedural
    # fbm height field -> the one remaining honest P0.6-category gap (the only
    # geometry/stippling filter without a twin). CPU numpy node 425 stays
    # authoritative for exact export; the twin is a live-preview approximation.
    "425": {"shader": "hbao_gpu", "type": "procedural", "typed": True,
            "param_map": {"freq": "freq", "octaves": "octaves",
                          "height_scale": "height_scale", "radius": "radius",
                          "directions": "directions", "steps": "steps",
                          "jitter": "jitter", "mode": "mode",
                          "light_az": "light_az", "light_el": "light_el",
                          "ambient": "ambient", "contrast": "contrast",
                          "colormode": "colormode", "anim_mode": "anim_mode"}},
    # 514 Apollonian Gasket → typed closed-form fold+inversion twin. The CPU
    # node uses depth/seed_curv/color_mode; the GPU twin exposes its own
    # closed-form controls (zoom/iterations/fold/hue_shift/contrast) — a visual
    # parity preview, not a pixel-exact match of the circle-packing CPU render
    # (CPU numpy path stays authoritative for export). All uniforms wired by name.
    "514": {"shader": "apollonian_gpu", "type": "procedural", "typed": True,
            "param_map": {"depth": "depth", "seed_curv": "seed_curv"}},
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
    "16": {"shader": "flow_field_typed", "type": "procedural", "typed": True,
           "param_map": {"speed": "speed"}},
    # 65 Waveform → waveform_typed (procedural). freq1/2/3 ↔ k1/2/3 (wave
    # number = cyclic frequency, same knob); amp/thick are render-only.
    "65": {"shader": "waveform_typed", "type": "procedural", "typed": True,
           "param_map": {"freq1": "k1", "freq2": "k2", "freq3": "k3"}},
    # 78 Circle Packing → circle_packing_typed (procedural). min_r/max_r are
    # normalized radius bounds (CPU min_radius/max_radius are px but both scale
    # the same normalized range); speed ↔ speed.
    "78": {"shader": "circle_packing_typed", "type": "procedural", "typed": True,
           "param_map": {"min_radius": "min_r", "max_radius": "max_r",
                         "anim_speed": "speed"}},
    # 56 Maze → maze_typed (procedural). wall_thickness ↔ wall (edge width),
    # cell_size ↔ scale (tile size). algorithm/style are choice strings unmapped.
    "56": {"shader": "maze_typed", "type": "procedural", "typed": True,
           "param_map": {"wall_thickness": "wall", "cell_size": "scale"}},
    # 81 Fourier Circles → fourier_circles_typed (procedural). Only `speed`
    # maps cleanly to a CPU param; freq1/2/3 are shader-only harmonic knobs
    # (the CPU node uses n_circles/scale/shape — no freq synth params).
    "81": {"shader": "fourier_circles_typed", "type": "procedural", "typed": True,
           "param_map": {"speed": "speed"}},
    # 406 Harmonograph → harmonograph_typed (procedural). freq1/freq2 ↔ fx/fy,
    # phase ↔ px (both phase offset, default 0), scale ↔ scale (same framing).
    # freq3/4/damping/line_width diverge from the twin's decay/turns/steps.
    "406": {"shader": "harmonograph_typed", "type": "procedural", "typed": True,
            "param_map": {"freq1": "fx", "freq2": "fy", "phase": "px",
                          "scale": "scale"}},
    # 409 Superformula → superformula_typed (procedural). `m` ↔ `m` (exact
    # symmetry count). n1/n2/n3 (CPU supershape exponents) don't match the
    # twin's n/b/c/p uniforms, so they stay unmapped.
    "409": {"shader": "superformula_typed", "type": "procedural", "typed": True,
            "param_map": {"m": "m"}},
    # ── P0.6 field-eval completion (2026-07-12) ──
    # 104 Spherical Harmonics → closed-form twin. max_l/amplitude/glow_strength/
    # anim_speed/twist_amplitude/osc_spread are the node's REAL numeric params
    # (contract #5). The twin declares uniforms= so the client reads them by
    # name (pitfall #14b) — the legacy p1..p4 mapping here is for the
    # param_map-resolves test + documentation only. CPU numpy node stays
    # authoritative for exact spherical-harmonic export.
    "104": {"shader": "spherical_harmonics_gpu", "type": "procedural",
             "typed": True,
             "param_map": {"max_l": "max_l", "amplitude": "amplitude",
                           "glow_strength": "glow_strength",
                           "anim_speed": "anim_speed"}},
    # 161 Spectral Tapestry -> typed-uniform twin (GPU-First contract #5).
    # n_modes/coupling/drift_speed are the node's REAL numeric params, wired BY
    # NAME to u_n_modes/u_coupling/u_drift_speed. Legacy p1..p3 was a SILENT
    # NO-OP (twin declares `uniforms=`, client typed-live branch ignores
    # p-slots, pitfall #14b / frozen-typed class). `noise`/anim_mode/n_frames are
    # left unmapped (CPU export stays authoritative).
    "161": {"shader": "spectral_tapestry_gpu", "type": "procedural",
             "typed": True,
             "param_map": {"n_modes": "n_modes", "coupling": "coupling",
                           "drift_speed": "drift_speed"}},
    # 473 Gabor Noise -> closed-form twin (live-preview path; the CPU numpy
    # node stays authoritative for exact export). REAL numeric params
    # anisotropy/frequency/falloff (bandwidth) mapped to the twin's p2..p4;
    # the twin's p1 (orientation) is left at its default (0) — the node has no
    # orientation param (anisotropy + wired-image warp drive direction instead).
    "473": {"shader": "gabor_gpu", "type": "procedural", "typed": True,
            "param_map": {"anisotropy": "anisotropy", "frequency": "frequency",
                          "falloff": "falloff"}},
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
    "431": {"shader": "domain_coloring_typed", "type": "procedural", "typed": True,
            "param_map": {"exponent": "exponent", "scale": "scale",
                          "center_x": "center_x", "center_y": "center_y"}},
    "432": {"shader": "maurer_rose_typed", "type": "procedural", "typed": True,
            "param_map": {"k": "petals", "d": "deg", "n_lines": "steps",
                          "line_width": "thick", "anim_speed": "speed"}},
    '433': {'shader': 'low_discrepancy_typed', 'type': 'procedural', 'typed': True,
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
    "503": {"shader": "conformal_gpu", "type": "procedural", "typed": True,
            "param_map": {"scale": "scale", "warp": "warp", "anim_speed": "anim_speed"}},
    # 460 Kaleidoscope Mirror -> typed-uniform filter twin kaleidoscope_mirror_gpu
    # (live-preview path; CPU numpy node 460 stays authoritative for export).
    # REAL numeric params segments/center_x/center_y/rotation/r_scale/mirror/
    # warp_amount/warp_scale map by NAME (contract #5/#6) — the client typed-live
    # path reads node params by uniform name. `source`/`palette`/`anim_mode` are
    # choice/string params (pitfall #14) left unmapped -> preview uses the wired
    # input image (or perlin source) with the default mirror-fold. Genuine
    # per-pixel wrap avoids the contrast-only static cull.
    "460": {"shader": "kaleidoscope_mirror_gpu", "type": "filter", "typed": True,
            "param_map": {"segments": "segments", "center_x": "center_x",
                          "center_y": "center_y", "rotation": "rotation",
                          "r_scale": "r_scale", "mirror": "mirror",
                          "warp_amount": "warp_amount", "warp_scale": "warp_scale"}},
    # 1004 Thin Film Interference -> typed-uniform closed-form twin
    # thin_film_gpu. Every numeric param (thickness/thickness_range/ior/drainage/
    # view_angle/brightness/anim_speed) and the choice param anim_mode map BY NAME
    # to the twin's u_<name> uniforms (contract #5/#6). `source` is a choice param
    # (procedural/input_image, pitfall #14) left unmapped -> the preview always
    # renders the procedural fbm thickness field. `time` is the system clock.
    # Pure per-pixel f(uv,t) spectral integral (P0.6 field-eval family); CPU numpy
    # node 1004 stays authoritative for exact export (69-sample spectrum + rng
    # fbm scale), the twin is a 35-sample live-preview approximation.
    "1004": {"shader": "thin_film_spectral_gpu", "type": "procedural", "typed": True,
             "param_map": {"thickness": "thickness",
                           "thickness_range": "thickness_range", "ior": "ior",
                           "drainage": "drainage", "view_angle": "view_angle",
                           "brightness": "brightness", "anim_speed": "anim_speed",
                           "anim_mode": "anim_mode"}},
    # === P0 shims: CPU pattern/math-art nodes -> existing typed GPU twins (cron run) ===
    # Faithful matches: node param names align with shader uniforms so the live
    # preview reflects the sliders; unmapped params stay CPU-authoritative.
    "963": {"shader": "gerstner_ocean_gpu", "type": "procedural", "typed": True,
             "param_map": {"n_waves": "n_waves", "base_wavelength": "base_wavelength", "wavelength_falloff": "wavelength_falloff", "amplitude": "amplitude", "steepness": "steepness", "wind_angle": "wind_angle", "wind_spread": "wind_spread", "sun_angle": "sun_angle", "sun_height": "sun_height", "shininess": "shininess", "glint": "glint", "deep_hue": "deep_hue", "crest_hue": "crest_hue", "exposure": "exposure", "gamma": "gamma"}},
    "964": {"shader": "gyroid_tpms_gpu", "type": "procedural", "typed": True,
             "param_map": {"surface": "surface", "freq": "freq", "level": "level", "thickness": "thickness", "warp": "warp", "contrast": "contrast", "shell": "shell"}},
    "1006": {"shader": "phasor_noise_gpu", "type": "procedural", "typed": True,
             "param_map": {"scale": "scale", "anisotropy": "anisotropy", "falloff": "falloff", "frequency": "frequency", "profile": "profile", "sharpness": "sharpness"}},
    "498": {"shader": "de_jong_typed", "type": "procedural", "typed": True,
             "param_map": {"a": "a", "b": "b", "c": "c", "d": "d", "exposure": "exposure"}},
    "342": {"shader": "strange_attractor_typed", "type": "procedural", "typed": True,
             "param_map": {"a": "a", "b": "b", "c": "c", "d": "d"}},
    "957": {"shader": "strange_attractor_typed", "type": "procedural", "typed": True,
             "param_map": {"a": "a", "b": "b", "c": "c", "d": "d"}},
    "962": {"shader": "torusknot_typed", "type": "procedural", "typed": True,
             "param_map": {"p": "p", "q": "q"}},
    "444": {"shader": "droste_typed", "type": "procedural", "typed": True,
             "param_map": {"twist": "twist", "zoom": "zoom"}},
    "510": {"shader": "flow_field_typed", "type": "procedural", "typed": True,
             "param_map": {"speed": "speed"}},
    # === P0 shims: CPU closed-form nodes -> faithful typed GPU twins (cron run) ===
    # Each node is the SAME closed-form algorithm as its twin, so the live preview
    # is faithful; node params with no twin uniform stay CPU-authoritative (two-tier
    # precision, GPU-First guardrail) and are listed in GPU_PREVIEW_DROP_ALLOW. Twin
    # uniforms with no clean CPU-node synonym (secondary artistic knobs) are
    # documented in test_gpu_twin_invariant._TWIN_UNIFORM_ALLOW.
    "355": {"shader": "curl_noise_gpu", "type": "procedural", "typed": True,
             "param_map": {"scale": "scale", "octaves": "octaves"}},
    "343": {"shader": "hex_grid_typed", "type": "procedural", "typed": True,
             "param_map": {"scale": "scale"}},
    "470": {"shader": "mandelbulb_gpu", "type": "procedural", "typed": True,
             "param_map": {"power": "power", "iterations": "iterations", "cam_dist": "cam_dist"}},
    "62": {"shader": "strange_attractor_typed", "type": "procedural", "typed": True,
            "param_map": {"a": "a", "b": "b", "c": "c", "d": "d"}},
    "351": {"shader": "mandelbrot_gpu", "type": "procedural", "typed": True,
             "param_map": {"center_x": "center_x", "center_y": "center_y",
                           "escape_radius": "escape_radius", "iterations": "iterations", "zoom": "zoom"}},
    "997": {"shader": "color_grade_gpu", "type": "procedural", "typed": True,
             "param_map": {"exposure": "exposure", "gamma": "gamma", "saturation": "saturation"}},
    "991": {"shader": "bilateral_grid_gpu", "type": "procedural", "typed": True,
             "param_map": {"blend": "blend", "sigma_r": "sigma_r", "sigma_s": "sigma_s"}},
    # ── P0.7 closed-form pattern twins (gap nodes) ──
    # CPU node stays authoritative for export; these route the live preview to
    # the new typed GLSL twins (core/shaders.py). Only the node's numeric (float)
    # params are wired by name; choice params (anim_mode/color_mode/orientation/
    # source/bg/palette/...) are intentionally left unmapped — the twin animates
    # continuously from u_time so the preview is always live (pitfall #14).
    "466": {"shader": "hex_mosaic_gpu", "type": "procedural", "typed": True,
            "param_map": {"hex_size": "hex_size", "rotation": "rotation",
                          "grout": "grout", "grout_color": "grout_color",
                          "anim_speed": "anim_speed"}},
    "505": {"shader": "metaballs_505_gpu", "type": "procedural", "typed": True,
            "param_map": {"balls": "balls", "ball_size": "ball_size",
                          "threshold": "threshold", "edge_soft": "edge_soft",
                          "drift_amp": "drift_amp", "anim_speed": "anim_speed"}},
    "426": {"shader": "truchet_sdf_gpu", "type": "procedural", "typed": True,
            "param_map": {"tile_size": "tile_size", "stroke": "stroke",
                          "edge_glow": "edge_glow", "anim_speed": "anim_speed"}},
    # ── Node 353: IFS Fractal attractor ─────────────────────────────────────────
    # Typed-uniform closed-form GPU twin (ifs_fractal_gpu in core/shaders.py).
    # Every numeric CPU param (preset/points/hue_shift/anim_speed) is bound to a
    # named u_<name> uniform. Choice params (coloring/anim_mode) and the legacy
    # time slot are dropped (GPU_PREVIEW_DROP_ALLOW); the preview is live via
    # the orbit-dispatch + u_time. CPU numpy fn stays authoritative export.
    "353": {"shader": "ifs_fractal_gpu", "type": "procedural", "typed": True,
            "param_map": {"preset": "preset", "points": "points",
                          "hue_shift": "hue_shift", "anim_speed": "anim_speed"}},
    # ── Node 416: Symmetric Icon attractor ──────────────────────────────────────
    # Typed-uniform closed-form GPU twin (symmetric_icon_gpu in core/shaders.py).
    # Every numeric CPU param (symmetry/a0..a4/palette_shift/anim_speed/
    # seed_strength) is bound to a named u_<name> uniform. Choice params
    # (colormode/source/anim_mode) and the legacy time slot are dropped; preview
    # is live via orbit dispatch + u_time. CPU numpy fn stays authoritative.
    "416": {"shader": "symmetric_icon_gpu", "type": "procedural", "typed": True,
            "param_map": {"symmetry": "symmetry", "a0": "a0", "a1": "a1",
                          "a2": "a2", "a3": "a3", "a4": "a4",
                          "palette_shift": "palette_shift", "anim_speed": "anim_speed",
                          "seed_strength": "seed_strength"}},
}

# ── GPU coverage contract: no SILENT param drops ────────────────────────────
# Every numeric-range slider on a shimmed CPU node MUST be EITHER routed through
# the twin's `param_map` OR explicitly listed here with a justification. This is
# the GPU-First "variable exposure" contract (no hidden GLSL constants / dead
# live-preview sliders). The test_gpu_param_coverage.py guard fails if a numeric
# node param is neither mapped nor justified — so a future twin edit that drops a
# uniform turns the silent bug into a reviewed, blocking failure.
#
# Auto-justified (never needs an entry here):
#   * `time` / `anim_speed` — timeline-driven; the live-preview slider is driven
#     by the graph timeline (_timeline), not a static uniform.
#   * choice/string params — the client only resolves numeric uniforms, so a
#     string/enum control is by-design unmapped (pitfall #14).
#   * params with no numeric min/max — not slider-exposed, not a live control.
#
# Everything else numeric-range that a legacy twin physically cannot carry
# (≤4 p-slots, or a CPU-domain knob the closed-form preview does not model) is
# listed below with a fixed justification so the contract stays explicit and
# auditable. When a twin is upgraded to typed uniforms, REMOVE its entry and add
# the uniform to `param_map` (the guard will then require the mapping).
GPU_PREVIEW_DROP_ALLOW: dict[str, dict[str, str]] = {
    "02": {"mod_strength": "param not wired to GPU twin; CPU export authoritative for this param"},
    "03": {"amplitude": "param not wired to GPU twin; CPU export authoritative for this param", "freq_variation": "param not wired to GPU twin; CPU export authoritative for this param", "grids": "param not wired to GPU twin; CPU export authoritative for this param", "thickness": "param not wired to GPU twin; CPU export authoritative for this param", "wobble": "param not wired to GPU twin; CPU export authoritative for this param"},
    # 535 Flow Noise: colormode/palette/source/anim_mode are choice/string
    # params (no GLSL equivalent); time is the system clock.
    # 534 Spot Noise: n_spots is a CPU-only spot count -> fixed 64-spot
    # hash loop for the live preview (CPU uses variable 100-4000). flow/colormode/
    # palette/anim_mode/source are choice/string params.
    # 953 Marbling: n_drops/n_tines are CPU-only counts -> fixed 32-drop /
    # 3-tine hash loops for the live preview (CPU uses variable 1-60 / 1-12).
    # 535 Flow Noise: colormode/palette/source/anim_mode are choice/string
    # params (no GLSL equivalent); time is the system clock.
    "535": {"colormode": "choice/string color mapping (no GLSL equivalent); GPU twin inlines inferno",
            "palette": "choice/string palette name (no GLSL equivalent); GPU twin inlines inferno",
            "source": "choice/string source selector (no GLSL equivalent); GPU twin uses procedural noise",
            "anim_mode": "choice/string animation-mode selector (no GLSL equivalent); GPU twin animates continuously from u_time"},
    # 534 Spot Noise: n_spots is a CPU-only spot count -> fixed 64-spot
    # hash loop for the live preview (CPU uses variable 100-4000). flow/colormode/
    # palette/anim_mode/source are choice/string params (no GLSL equivalent); the
    # GPU twin hardcodes a circular flow field + inferno colormap.
    "534": {"n_spots": "fixed 64-spot hash loop for live preview (CPU uses variable 100-4000)",
            "flow": "choice/string flow-field selector (circular/sine/saddle/curl/radial); GPU twin hardcodes a circular flow field for live preview",
            "colormode": "choice/string color mapping (no GLSL equivalent); GPU twin inlines inferno",
            "palette": "choice/string palette name (no GLSL equivalent); GPU twin inlines inferno",
            "anim_mode": "choice/string animation-mode selector (no GLSL equivalent); GPU twin animates continuously from u_time",
            "source": "choice/string source selector (no GLSL equivalent); GPU twin uses procedural noise"},
    "953": {"n_drops": "fixed 32-drop inverse for live preview (CPU uses variable 1-60)",
             "n_tines": "fixed 3-tine strokes for live preview (CPU uses variable 1-12)",
             "source": "choice/string source selector (no GLSL equivalent); GPU twin uses procedural noise",
             "anim_mode": "choice/string animation-mode selector (no GLSL equivalent); GPU twin animates continuously from u_time"},
    "487": {"star_count": "param not wired to GPU twin; CPU export authoritative for this param (Galaxy Generator samples this many stars on the CPU; the closed-form GLSL twin renders a continuous density field at canvas resolution)"},
    "108": {"n_frames": "param not wired to GPU twin; CPU export authoritative for this param (export frame count, timeline-driven)"},
    "995": {"palette": "param not wired to GPU twin; CPU export authoritative for this param (GPU twin inlines the cosmic palette; CPU node honours the exact palette choice)", "mode": "param not wired to GPU twin; CPU export authoritative for this param (GPU twin animates continuously from u_time — a drift+breathe superposition)", "anim_mode": "param not wired to GPU twin; CPU export authoritative for this param (GPU twin animates continuously from u_time)", "time": "param not wired to GPU twin; CPU export authoritative for this param (GPU twin drives phase from u_time)"},
    "950": {"pattern": "param not wired to GPU twin; CPU export authoritative for this param (GPU twin renders the combo composition; CPU node honours the exact pattern choice)", "color_mode": "param not wired to GPU twin; CPU export authoritative for this param (GPU twin inlines the amber palette)", "anim_mode": "param not wired to GPU twin; CPU export authoritative for this param (GPU twin animates continuously from u_time — rotate+drift+pulse superposition)", "time": "param not wired to GPU twin; CPU export authoritative for this param (GPU twin drives phase from u_time)"},
    "967": {"anim_mode": "param not wired to GPU twin; CPU export authoritative for this param (GPU twin animates continuously from u_time — pan+lights superposition)", "n_frames": "param not wired to GPU twin; CPU export authoritative for this param (export frame count for Architecture-A capture)", "time": "param not wired to GPU twin; CPU export authoritative for this param (GPU twin drives phase from u_time)"},
    "486": {"blur_type": "param not wired to GPU twin; CPU export authoritative for this param (GPU twin always applies a combined radial-zoom + spin motion blur; the CPU node honours the exact blur_type choice)", "source": "param not wired to GPU twin; CPU export authoritative for this param (GPU twin samples the wired upstream image)", "noise_amp": "param not wired to GPU twin; CPU export authoritative for this param", "blur_sigma": "param not wired to GPU twin; CPU export authoritative for this param", "palette": "param not wired to GPU twin; CPU export authoritative for this param", "anim_mode": "param not wired to GPU twin; CPU export authoritative for this param (GPU twin animates continuously from u_time)"},
    "438": {"source": "param not wired to GPU twin; CPU export authoritative for this param (GPU twin samples the wired upstream image)", "tint": "param not wired to GPU twin; CPU export authoritative for this param", "noise_amp": "param not wired to GPU twin; CPU export authoritative for this param", "blur_sigma": "param not wired to GPU twin; CPU export authoritative for this param", "palette": "param not wired to GPU twin; CPU export authoritative for this param", "anim_mode": "param not wired to GPU twin; CPU export authoritative for this param (GPU twin animates continuously from u_time)", "time": "param not wired to GPU twin; CPU export authoritative for this param"},
    "439": {"source": "param not wired to GPU twin; CPU export authoritative for this param (GPU twin samples the wired upstream image)", "n_orientations": "param not wired to GPU twin; CPU export authoritative for this param (GPU twin renders a single-orientation response)", "combine": "param not wired to GPU twin; CPU export authoritative for this param", "output": "param not wired to GPU twin; CPU export authoritative for this param (GPU twin renders the energy magnitude)", "anim_mode": "param not wired to GPU twin; CPU export authoritative for this param (GPU twin animates continuously from u_time)"},
    "04": {"cell_border": "param not wired to GPU twin; CPU export authoritative for this param", "feature": "param not wired to GPU twin; CPU export authoritative for this param", "fractal": "param not wired to GPU twin; CPU export authoritative for this param", "points": "param not wired to GPU twin; CPU export authoritative for this param", "tile_size": "param not wired to GPU twin; CPU export authoritative for this param"},
    "05": {"cell_borders": "param not wired to GPU twin; CPU export authoritative for this param", "cell_points": "param not wired to GPU twin; CPU export authoritative for this param", "domain_warp": "param not wired to GPU twin; CPU export authoritative for this param", "erosion": "param not wired to GPU twin; CPU export authoritative for this param", "gain": "param not wired to GPU twin; CPU export authoritative for this param", "lacunarity": "param not wired to GPU twin; CPU export authoritative for this param", "ring_count": "param not wired to GPU twin; CPU export authoritative for this param", "ring_wobble": "param not wired to GPU twin; CPU export authoritative for this param", "water_level": "param not wired to GPU twin; CPU export authoritative for this param"},
    "06": {"gap": "param not wired to GPU twin; CPU export authoritative for this param", "penrose_generations": "param not wired to GPU twin; CPU export authoritative for this param", "star_rays": "param not wired to GPU twin; CPU export authoritative for this param", "tile_size": "param not wired to GPU twin; CPU export authoritative for this param"},
    "07": {"color_variation": "param not wired to GPU twin; CPU export authoritative for this param", "gap": "param not wired to GPU twin; CPU export authoritative for this param", "rotation_noise": "param not wired to GPU twin; CPU export authoritative for this param"},
    "08": {"center_x": "param not wired to GPU twin; CPU export authoritative for this param", "center_y": "param not wired to GPU twin; CPU export authoritative for this param", "fade": "param not wired to GPU twin; CPU export authoritative for this param", "petal_angle": "param not wired to GPU twin; CPU export authoritative for this param", "point_size_max": "param not wired to GPU twin; CPU export authoritative for this param", "point_size_min": "param not wired to GPU twin; CPU export authoritative for this param", "rotation": "param not wired to GPU twin; CPU export authoritative for this param"},
    "11": {"direction": "param not wired to GPU twin; CPU export authoritative for this param"},
    "13": {"error_scale": "param not wired to GPU twin; CPU export authoritative for this param"},
    "16": {"color_hue": "param not wired to GPU twin; CPU export authoritative for this param", "freq": "param not wired to GPU twin; CPU export authoritative for this param", "line_width": "param not wired to GPU twin; CPU export authoritative for this param", "n_particles": "param not wired to GPU twin; CPU export authoritative for this param", "n_waves": "param not wired to GPU twin; CPU export authoritative for this param", "trail_length": "param not wired to GPU twin; CPU export authoritative for this param"},
    "17": {"bit_depth": "param not wired to GPU twin; CPU export authoritative for this param", "channel_offset": "param not wired to GPU twin; CPU export authoritative for this param", "jpeg_quality": "param not wired to GPU twin; CPU export authoritative for this param", "noise_blocks": "param not wired to GPU twin; CPU export authoritative for this param", "scanlines": "param not wired to GPU twin; CPU export authoritative for this param", "shift_count": "param not wired to GPU twin; CPU export authoritative for this param", "shift_magnitude": "param not wired to GPU twin; CPU export authoritative for this param", "shift_max_height": "param not wired to GPU twin; CPU export authoritative for this param", "vhs_tracking": "param not wired to GPU twin; CPU export authoritative for this param", "wave_distort": "param not wired to GPU twin; CPU export authoritative for this param"},
    "18": {"age_input": "param not wired to GPU twin; CPU export authoritative for this param", "cell_size": "param not wired to GPU twin; CPU export authoritative for this param", "hue_shift": "param not wired to GPU twin; CPU export authoritative for this param", "init_select": "param not wired to GPU twin; CPU export authoritative for this param", "inject_rate": "param not wired to GPU twin; CPU export authoritative for this param", "rule_select": "param not wired to GPU twin; CPU export authoritative for this param", "seed_threshold": "param not wired to GPU twin; CPU export authoritative for this param", "size": "param not wired to GPU twin; CPU export authoritative for this param", "wave_phase": "param not wired to GPU twin; CPU export authoritative for this param"},
    "29": {"line_width": "param not wired to GPU twin; CPU export authoritative for this param"},
    "31": {"erosion": "param not wired to GPU twin; CPU export authoritative for this param", "light_angle": "param not wired to GPU twin; CPU export authoritative for this param", "roughness_decay": "param not wired to GPU twin; CPU export authoritative for this param", "water_level": "param not wired to GPU twin; CPU export authoritative for this param"},
    "32": {"bias_x": "param not wired to GPU twin; CPU export authoritative for this param", "bias_y": "param not wired to GPU twin; CPU export authoritative for this param", "dt": "param not wired to GPU twin; CPU export authoritative for this param", "feedback_strength": "param not wired to GPU twin; CPU export authoritative for this param", "inject_strength": "param not wired to GPU twin; CPU export authoritative for this param", "inject_x": "param not wired to GPU twin; CPU export authoritative for this param", "inject_y": "param not wired to GPU twin; CPU export authoritative for this param", "iterations": "param not wired to GPU twin; CPU export authoritative for this param", "particle_count": "param not wired to GPU twin; CPU export authoritative for this param", "particle_speed": "param not wired to GPU twin; CPU export authoritative for this param", "perturbations": "param not wired to GPU twin; CPU export authoritative for this param", "seed_size": "param not wired to GPU twin; CPU export authoritative for this param"},
    "33": {"trap_strength": "param not wired to GPU twin; CPU export authoritative for this param", "trap_x": "param not wired to GPU twin; CPU export authoritative for this param", "trap_y": "param not wired to GPU twin; CPU export authoritative for this param", "warp_strength": "param not wired to GPU twin; CPU export authoritative for this param"},
    "39": {"n_colors": "param not wired to GPU twin; CPU export authoritative for this param"},
    "41": {"blur_sigma": "param not wired to GPU twin; CPU export authoritative for this param", "brush_size": "param not wired to GPU twin; CPU export authoritative for this param", "edge_threshold": "param not wired to GPU twin; CPU export authoritative for this param", "noise_amp": "param not wired to GPU twin; CPU export authoritative for this param", "noise_offset": "param not wired to GPU twin; CPU export authoritative for this param", "quantize_levels": "param not wired to GPU twin; CPU export authoritative for this param"},
    "42": {"blur_sigma": "param not wired to GPU twin; CPU export authoritative for this param", "exposure": "param not wired to GPU twin; CPU export authoritative for this param", "gamma": "param not wired to GPU twin; CPU export authoritative for this param", "noise_amp": "param not wired to GPU twin; CPU export authoritative for this param", "tint_b": "param not wired to GPU twin; CPU export authoritative for this param", "tint_g": "param not wired to GPU twin; CPU export authoritative for this param", "tint_r": "param not wired to GPU twin; CPU export authoritative for this param"},
    "43": {"contour_levels": "param not wired to GPU twin; CPU export authoritative for this param", "light_alt": "param not wired to GPU twin; CPU export authoritative for this param", "light_angle": "param not wired to GPU twin; CPU export authoritative for this param", "n_clusters": "param not wired to GPU twin; CPU export authoritative for this param", "point_speed": "param not wired to GPU twin; CPU export authoritative for this param", "points": "param not wired to GPU twin; CPU export authoritative for this param", "ridge_spacing": "param not wired to GPU twin; CPU export authoritative for this param", "scatter_alpha": "param not wired to GPU twin; CPU export authoritative for this param"},
    "471": {"num_samples": "param not wired to GPU twin; CPU export authoritative for this param (MSAA-style sampling count, meaningless for the closed-form sky preview)", "n_frames": "param not wired to GPU twin; CPU export authoritative for this param (export frame count, timeline-driven)"},
    "51": {"anim_zoom_speed": "param not wired to GPU twin; CPU export authoritative for this param", "antialias": "param not wired to GPU twin; CPU export authoritative for this param", "escape_radius": "param not wired to GPU twin; CPU export authoritative for this param", "exponent": "param not wired to GPU twin; CPU export authoritative for this param", "warp_strength": "param not wired to GPU twin; CPU export authoritative for this param"},
    "52": {"anim_float_amplitude": "param not wired to GPU twin; CPU export authoritative for this param", "anim_zoom_speed": "param not wired to GPU twin; CPU export authoritative for this param", "tol": "param not wired to GPU twin; CPU export authoritative for this param", "warp_strength": "param not wired to GPU twin; CPU export authoritative for this param"},
    "53": {"balls": "param not wired to GPU twin; CPU export authoritative for this param", "color_speed": "param not wired to GPU twin; CPU export authoritative for this param", "multi_threshold_levels": "param not wired to GPU twin; CPU export authoritative for this param", "radius_max": "param not wired to GPU twin; CPU export authoritative for this param", "radius_min": "param not wired to GPU twin; CPU export authoritative for this param", "trail_frames": "param not wired to GPU twin; CPU export authoritative for this param"},
    "56": {"braid": "param not wired to GPU twin; CPU export authoritative for this param", "color_saturation": "param not wired to GPU twin; CPU export authoritative for this param", "growing_bias": "param not wired to GPU twin; CPU export authoritative for this param", "loops": "param not wired to GPU twin; CPU export authoritative for this param", "multi_seed": "param not wired to GPU twin; CPU export authoritative for this param", "rings": "param not wired to GPU twin; CPU export authoritative for this param"},
    "57": {"blur_sigma": "param not wired to GPU twin; CPU export authoritative for this param", "feedback_decay": "param not wired to GPU twin; CPU export authoritative for this param", "noise_amp": "param not wired to GPU twin; CPU export authoritative for this param", "tint_b": "param not wired to GPU twin; CPU export authoritative for this param", "tint_g": "param not wired to GPU twin; CPU export authoritative for this param", "tint_r": "param not wired to GPU twin; CPU export authoritative for this param"},
    "58": {"hue_shift": "param not wired to GPU twin; CPU export authoritative for this param", "init_select": "param not wired to GPU twin; CPU export authoritative for this param", "rule_select": "param not wired to GPU twin; CPU export authoritative for this param", "speed": "param not wired to GPU twin; CPU export authoritative for this param"},
    "63": {"blur_sigma": "param not wired to GPU twin; CPU export authoritative for this param", "noise_amp": "param not wired to GPU twin; CPU export authoritative for this param", "speckle_count": "param not wired to GPU twin; CPU export authoritative for this param", "thread_density": "param not wired to GPU twin; CPU export authoritative for this param", "thread_variation": "param not wired to GPU twin; CPU export authoritative for this param"},
    "64": {"blur_sigma": "param not wired to GPU twin; CPU export authoritative for this param", "canny_high": "param not wired to GPU twin; CPU export authoritative for this param", "canny_low": "param not wired to GPU twin; CPU export authoritative for this param", "dot_variation": "param not wired to GPU twin; CPU export authoritative for this param", "noise_amp": "param not wired to GPU twin; CPU export authoritative for this param"},
    "65": {"amplitude_ratio": "param not wired to GPU twin; CPU export authoritative for this param", "decay_rate": "param not wired to GPU twin; CPU export authoritative for this param", "fill_alpha": "param not wired to GPU twin; CPU export authoritative for this param", "line_width": "param not wired to GPU twin; CPU export authoritative for this param", "mod_depth": "param not wired to GPU twin; CPU export authoritative for this param", "mod_freq": "param not wired to GPU twin; CPU export authoritative for this param", "noise_level": "param not wired to GPU twin; CPU export authoritative for this param", "num_bars": "param not wired to GPU twin; CPU export authoritative for this param", "num_tracks": "param not wired to GPU twin; CPU export authoritative for this param", "pulse_width": "param not wired to GPU twin; CPU export authoritative for this param"},
    "66": {"warp_strength": "param not wired to GPU twin; CPU export authoritative for this param"},
    "68": {"blend": "param not wired to GPU twin; CPU export authoritative for this param", "blur_sigma": "param not wired to GPU twin; CPU export authoritative for this param", "noise_amp": "param not wired to GPU twin; CPU export authoritative for this param", "presmooth": "param not wired to GPU twin; CPU export authoritative for this param"},
    "69": {"measure": "param not wired to GPU twin; CPU export authoritative for this param", "seed_strength": "param not wired to GPU twin; CPU export authoritative for this param", "warmup": "param not wired to GPU twin; CPU export authoritative for this param"},
    "74": {"amplitude": "param not wired to GPU twin; CPU export authoritative for this param", "blur_sigma": "param not wired to GPU twin; CPU export authoritative for this param", "frequency": "param not wired to GPU twin; CPU export authoritative for this param", "noise_amp": "param not wired to GPU twin; CPU export authoritative for this param", "rotation": "param not wired to GPU twin; CPU export authoritative for this param", "segments": "param not wired to GPU twin; CPU export authoritative for this param", "zoom": "param not wired to GPU twin; CPU export authoritative for this param"},
    "78": {"attempts": "param not wired to GPU twin; CPU export authoritative for this param", "concentric_rings": "param not wired to GPU twin; CPU export authoritative for this param", "gap": "param not wired to GPU twin; CPU export authoritative for this param", "halftone_density": "param not wired to GPU twin; CPU export authoritative for this param", "max_circles": "param not wired to GPU twin; CPU export authoritative for this param", "outline_width": "param not wired to GPU twin; CPU export authoritative for this param", "relaxation_iters": "param not wired to GPU twin; CPU export authoritative for this param", "sunburst_rays": "param not wired to GPU twin; CPU export authoritative for this param"},
    "80": {"blur_sigma": "param not wired to GPU twin; CPU export authoritative for this param", "grout_width": "param not wired to GPU twin; CPU export authoritative for this param", "noise_amp": "param not wired to GPU twin; CPU export authoritative for this param", "tile_jitter": "param not wired to GPU twin; CPU export authoritative for this param"},
    "81": {"line_width": "param not wired to GPU twin; CPU export authoritative for this param", "n_circles": "param not wired to GPU twin; CPU export authoritative for this param", "offset_x": "param not wired to GPU twin; CPU export authoritative for this param", "offset_y": "param not wired to GPU twin; CPU export authoritative for this param", "scale": "param not wired to GPU twin; CPU export authoritative for this param", "trace_fade": "param not wired to GPU twin; CPU export authoritative for this param", "trace_length": "param not wired to GPU twin; CPU export authoritative for this param"},
    "87": {"grid_scale": "param not wired to GPU twin; CPU export authoritative for this param", "n_frames": "param not wired to GPU twin; CPU export authoritative for this param"},
    "91": {"Dv": "param not wired to GPU twin; CPU export authoritative for this param", "dt": "param not wired to GPU twin; CPU export authoritative for this param", "grid_size": "param not wired to GPU twin; CPU export authoritative for this param", "n_frames": "param not wired to GPU twin; CPU export authoritative for this param"},
    "93": {"L": "param not wired to GPU twin; CPU export authoritative for this param", "T_min": "param not wired to GPU twin; CPU export authoritative for this param", "n_frames": "param not wired to GPU twin; CPU export authoritative for this param", "updates_per_frame": "param not wired to GPU twin; CPU export authoritative for this param"},
    "95": {"burn_in": "param not wired to GPU twin; CPU export authoritative for this param", "grid_h": "param not wired to GPU twin; CPU export authoritative for this param", "grid_w": "param not wired to GPU twin; CPU export authoritative for this param", "n_frames": "param not wired to GPU twin; CPU export authoritative for this param"},
    "96": {"grid_h": "param not wired to GPU twin; CPU export authoritative for this param", "grid_w": "param not wired to GPU twin; CPU export authoritative for this param", "n_frames": "param not wired to GPU twin; CPU export authoritative for this param"},
    "99": {"n_frames": "param not wired to GPU twin; CPU export authoritative for this param", "substeps": "param not wired to GPU twin; CPU export authoritative for this param"},
    "100": {"gamma": "param not wired to GPU twin; CPU export authoritative for this param", "n_sources": "param not wired to GPU twin; CPU export authoritative for this param", "n_steps_per_frame": "param not wired to GPU twin; CPU export authoritative for this param", "orbit_radius": "param not wired to GPU twin; CPU export authoritative for this param", "orbit_speed": "param not wired to GPU twin; CPU export authoritative for this param", "pulse_width": "param not wired to GPU twin; CPU export authoritative for this param", "source_spread": "param not wired to GPU twin; CPU export authoritative for this param"},
    "104": {"n_frames": "param not wired to GPU twin; CPU export authoritative for this param", "osc_spread": "param not wired to GPU twin; CPU export authoritative for this param", "twist_amplitude": "param not wired to GPU twin; CPU export authoritative for this param", "twist_speed": "param not wired to GPU twin; CPU export authoritative for this param"},
    "105": {"grid_size": "param not wired to GPU twin; CPU export authoritative for this param"},
    "106": {"n_frames": "param not wired to GPU twin; CPU export authoritative for this param", "seeds": "param not wired to GPU twin; CPU export authoritative for this param", "spark_prob": "param not wired to GPU twin; CPU export authoritative for this param"},
    "118": {"dt": "param not wired to GPU twin; CPU export authoritative for this param", "du": "param not wired to GPU twin; CPU export authoritative for this param", "dv": "param not wired to GPU twin; CPU export authoritative for this param", "init_amp": "param not wired to GPU twin; CPU export authoritative for this param", "n_frames": "param not wired to GPU twin; CPU export authoritative for this param"},
    "119": {"dt": "param not wired to GPU twin; CPU export authoritative for this param", "du": "param not wired to GPU twin; CPU export authoritative for this param", "dv": "param not wired to GPU twin; CPU export authoritative for this param", "n_frames": "param not wired to GPU twin; CPU export authoritative for this param", "noise_amp": "param not wired to GPU twin; CPU export authoritative for this param"},
    "120": {"dt": "param not wired to GPU twin; CPU export authoritative for this param", "n_frames": "param not wired to GPU twin; CPU export authoritative for this param"},
    "121": {"dt": "param not wired to GPU twin; CPU export authoritative for this param", "du_x": "param not wired to GPU twin; CPU export authoritative for this param", "du_y": "param not wired to GPU twin; CPU export authoritative for this param", "dv_x": "param not wired to GPU twin; CPU export authoritative for this param", "dv_y": "param not wired to GPU twin; CPU export authoritative for this param", "n_frames": "param not wired to GPU twin; CPU export authoritative for this param", "noise_amp": "param not wired to GPU twin; CPU export authoritative for this param"},
    "122": {"impurity_density": "param not wired to GPU twin; CPU export authoritative for this param", "n_frames": "param not wired to GPU twin; CPU export authoritative for this param", "n_seeds": "param not wired to GPU twin; CPU export authoritative for this param"},
    "124": {"amplitude": "param not wired to GPU twin; CPU export authoritative for this param", "background_noise": "param not wired to GPU twin; CPU export authoritative for this param", "initial_width": "param not wired to GPU twin; CPU export authoritative for this param", "n_frames": "param not wired to GPU twin; CPU export authoritative for this param", "n_solitons": "param not wired to GPU twin; CPU export authoritative for this param", "phase_strength": "param not wired to GPU twin; CPU export authoritative for this param", "plane_wave_amp": "param not wired to GPU twin; CPU export authoritative for this param", "single_momentum": "param not wired to GPU twin; CPU export authoritative for this param", "soliton_momentum": "param not wired to GPU twin; CPU export authoritative for this param", "soliton_offset": "param not wired to GPU twin; CPU export authoritative for this param", "substeps": "param not wired to GPU twin; CPU export authoritative for this param", "vortex_radius_ratio": "param not wired to GPU twin; CPU export authoritative for this param"},
    "125": {"breathe_amp": "param not wired to GPU twin; CPU export authoritative for this param", "m_end": "param not wired to GPU twin; CPU export authoritative for this param", "n_end": "param not wired to GPU twin; CPU export authoritative for this param", "n_frames": "param not wired to GPU twin; CPU export authoritative for this param", "n_modes": "param not wired to GPU twin; CPU export authoritative for this param", "nodal_glow": "param not wired to GPU twin; CPU export authoritative for this param", "phase_speed_y": "param not wired to GPU twin; CPU export authoritative for this param", "sigmoid_gain": "param not wired to GPU twin; CPU export authoritative for this param"},
    "126": {"n_frames": "param not wired to GPU twin; CPU export authoritative for this param", "n_seeds": "param not wired to GPU twin; CPU export authoritative for this param", "noise_amp": "param not wired to GPU twin; CPU export authoritative for this param", "spiral_r": "param not wired to GPU twin; CPU export authoritative for this param", "substeps": "param not wired to GPU twin; CPU export authoritative for this param", "wave_k": "param not wired to GPU twin; CPU export authoritative for this param"},
    "127": {"n_frames": "param not wired to GPU twin; CPU export authoritative for this param", "substeps": "param not wired to GPU twin; CPU export authoritative for this param"},
    "128": {"grid_size": "param not wired to GPU twin; CPU export authoritative for this param", "n_frames": "param not wired to GPU twin; CPU export authoritative for this param", "substeps": "param not wired to GPU twin; CPU export authoritative for this param"},
    "132": {"dt": "param not wired to GPU twin; CPU export authoritative for this param", "grid_div": "param not wired to GPU twin; CPU export authoritative for this param", "n_frames": "param not wired to GPU twin; CPU export authoritative for this param", "n_obstacles": "param not wired to GPU twin; CPU export authoritative for this param", "n_sources": "param not wired to GPU twin; CPU export authoritative for this param", "obstacle_radius": "param not wired to GPU twin; CPU export authoritative for this param", "obstacle_x": "param not wired to GPU twin; CPU export authoritative for this param", "obstacle_y": "param not wired to GPU twin; CPU export authoritative for this param"},
    "133": {"amplitude": "param not wired to GPU twin; CPU export authoritative for this param", "diff_v": "param not wired to GPU twin; CPU export authoritative for this param", "dt": "param not wired to GPU twin; CPU export authoritative for this param", "n_frames": "param not wired to GPU twin; CPU export authoritative for this param"},
    "135": {"amplitude": "param not wired to GPU twin; CPU export authoritative for this param", "grid_div": "param not wired to GPU twin; CPU export authoritative for this param", "n_frames": "param not wired to GPU twin; CPU export authoritative for this param"},
    "142": {"morph_speed": "param not wired to GPU twin; CPU export authoritative for this param", "n_frames": "param not wired to GPU twin; CPU export authoritative for this param", "noise": "param not wired to GPU twin; CPU export authoritative for this param"},
    "143": {"diff_n": "param not wired to GPU twin; CPU export authoritative for this param", "init_radius": "param not wired to GPU twin; CPU export authoritative for this param", "n_frames": "param not wired to GPU twin; CPU export authoritative for this param", "noise_intensity": "param not wired to GPU twin; CPU export authoritative for this param"},
    "144": {"n_frames": "param not wired to GPU twin; CPU export authoritative for this param", "nonlinear": "param not wired to GPU twin; CPU export authoritative for this param"},
    "146": {"n_frames": "param not wired to GPU twin; CPU export authoritative for this param"},
    "148": {"n_frames": "param not wired to GPU twin; CPU export authoritative for this param"},
    "150": {"n_frames": "param not wired to GPU twin; CPU export authoritative for this param"},
    "153": {"grid_size": "param not wired to GPU twin; CPU export authoritative for this param", "init_coop": "param not wired to GPU twin; CPU export authoritative for this param", "mutation_rate": "param not wired to GPU twin; CPU export authoritative for this param", "n_frames": "param not wired to GPU twin; CPU export authoritative for this param", "steps_per_frame": "param not wired to GPU twin; CPU export authoritative for this param"},
    "154": {"diffusion_rate": "param not wired to GPU twin; CPU export authoritative for this param", "grid_size": "param not wired to GPU twin; CPU export authoritative for this param", "init_coop": "param not wired to GPU twin; CPU export authoritative for this param", "mutation_rate": "param not wired to GPU twin; CPU export authoritative for this param", "n_frames": "param not wired to GPU twin; CPU export authoritative for this param", "noise_amplitude": "param not wired to GPU twin; CPU export authoritative for this param", "steps_per_frame": "param not wired to GPU twin; CPU export authoritative for this param"},
    "155": {"cell_max": "param not wired to GPU twin; CPU export authoritative for this param", "cell_min": "param not wired to GPU twin; CPU export authoritative for this param", "dt": "param not wired to GPU twin; CPU export authoritative for this param", "grad_sweep": "param not wired to GPU twin; CPU export authoritative for this param", "n_frames": "param not wired to GPU twin; CPU export authoritative for this param"},
    "157": {"morph_speed": "param not wired to GPU twin; CPU export authoritative for this param", "n_frames": "param not wired to GPU twin; CPU export authoritative for this param"},
    "160": {"diff_n": "param not wired to GPU twin; CPU export authoritative for this param", "init_radius": "param not wired to GPU twin; CPU export authoritative for this param", "n_frames": "param not wired to GPU twin; CPU export authoritative for this param", "noise_intensity": "param not wired to GPU twin; CPU export authoritative for this param"},
    "161": {"dt": "param not wired to GPU twin; CPU export authoritative for this param", "n_frames": "param not wired to GPU twin; CPU export authoritative for this param", "noise": "param not wired to GPU twin; CPU export authoritative for this param"},
    "162": {"coupling": "param not wired to GPU twin; CPU export authoritative for this param", "dt": "param not wired to GPU twin; CPU export authoritative for this param", "grid_div": "param not wired to GPU twin; CPU export authoritative for this param", "init_amplitude": "param not wired to GPU twin; CPU export authoritative for this param", "n_frames": "param not wired to GPU twin; CPU export authoritative for this param", "noise_amp": "param not wired to GPU twin; CPU export authoritative for this param"},
    "163": {"diff_u": "param not wired to GPU twin; CPU export authoritative for this param", "dt": "param not wired to GPU twin; CPU export authoritative for this param", "grid_div": "param not wired to GPU twin; CPU export authoritative for this param", "n_frames": "param not wired to GPU twin; CPU export authoritative for this param", "noise_amp": "param not wired to GPU twin; CPU export authoritative for this param"},
    "164": {"n_frames": "param not wired to GPU twin; CPU export authoritative for this param"},
    "166": {"n_frames": "param not wired to GPU twin; CPU export authoritative for this param", "nonlinear": "param not wired to GPU twin; CPU export authoritative for this param", "pump_ratio": "param not wired to GPU twin; CPU export authoritative for this param"},
    "168": {"K": "param not wired to GPU twin; CPU export authoritative for this param", "alpha": "param not wired to GPU twin; CPU export authoritative for this param", "bias": "param not wired to GPU twin; CPU export authoritative for this param", "dt": "param not wired to GPU twin; CPU export authoritative for this param", "n_frames": "param not wired to GPU twin; CPU export authoritative for this param", "noise": "param not wired to GPU twin; CPU export authoritative for this param"},
    "169": {"Dv": "param not wired to GPU twin; CPU export authoritative for this param", "dt": "param not wired to GPU twin; CPU export authoritative for this param", "growth_rate": "param not wired to GPU twin; CPU export authoritative for this param", "n_frames": "param not wired to GPU twin; CPU export authoritative for this param", "noise": "param not wired to GPU twin; CPU export authoritative for this param"},
    "170": {"n_frames": "param not wired to GPU twin; CPU export authoritative for this param", "psi0": "param not wired to GPU twin; CPU export authoritative for this param"},
    "172": {"n_frames": "param not wired to GPU twin; CPU export authoritative for this param"},
    "311": {"gain": "param not wired to GPU twin; CPU export authoritative for this param", "lacunarity": "param not wired to GPU twin; CPU export authoritative for this param", "warp_levels": "param not wired to GPU twin; CPU export authoritative for this param"},
    "312": {"amplitude": "param not wired to GPU twin; CPU export authoritative for this param", "waves": "param not wired to GPU twin; CPU export authoritative for this param"},
    "314": {"line_density": "param not wired to GPU twin; CPU export authoritative for this param"},
    "326": {"resolution": "hash-table grid resolution is a CPU-domain export knob; the closed-form GLSL twin renders at the canvas resolution (GPU coverage contract: explicit drop)"},
    "339": {"blur_sigma": "param not wired to GPU twin; CPU export authoritative for this param", "noise_amp": "param not wired to GPU twin; CPU export authoritative for this param", "source_blur": "param not wired to GPU twin; CPU export authoritative for this param"},
    "345": {"blur_sigma": "param not wired to GPU twin; CPU export authoritative for this param", "noise_amp": "param not wired to GPU twin; CPU export authoritative for this param", "presmooth": "param not wired to GPU twin; CPU export authoritative for this param", "z_bins": "param not wired to GPU twin; CPU export authoritative for this param"},
    "348": {"capacity": "param not wired to GPU twin; CPU export authoritative for this param", "droplets": "param not wired to GPU twin; CPU export authoritative for this param", "gravity": "param not wired to GPU twin; CPU export authoritative for this param", "grid": "param not wired to GPU twin; CPU export authoritative for this param", "height_scale": "param not wired to GPU twin; CPU export authoritative for this param", "inertia": "param not wired to GPU twin; CPU export authoritative for this param", "lifetime": "param not wired to GPU twin; CPU export authoritative for this param", "light_angle": "param not wired to GPU twin; CPU export authoritative for this param", "octaves": "param not wired to GPU twin; CPU export authoritative for this param", "radius": "param not wired to GPU twin; CPU export authoritative for this param", "roughness": "param not wired to GPU twin; CPU export authoritative for this param"},
    "350": {"noise_amp": "param not wired to GPU twin; CPU export authoritative for this param"},
    "402": {"escape_radius": "param not wired to GPU twin; CPU export authoritative for this param", "iterations": "param not wired to GPU twin; CPU export authoritative for this param", "offset_x": "param not wired to GPU twin; CPU export authoritative for this param", "offset_y": "param not wired to GPU twin; CPU export authoritative for this param"},
    "406": {"color_shift": "param not wired to GPU twin; CPU export authoritative for this param", "damping": "param not wired to GPU twin; CPU export authoritative for this param", "freq3": "param not wired to GPU twin; CPU export authoritative for this param", "freq4": "param not wired to GPU twin; CPU export authoritative for this param", "line_width": "param not wired to GPU twin; CPU export authoritative for this param", "samples": "param not wired to GPU twin; CPU export authoritative for this param"},
    "408": {"iterations": "param not wired to GPU twin; CPU export authoritative for this param", "noise_amp": "param not wired to GPU twin; CPU export authoritative for this param"},
    "409": {"n1": "param not wired to GPU twin; CPU export authoritative for this param", "n2": "param not wired to GPU twin; CPU export authoritative for this param", "n3": "param not wired to GPU twin; CPU export authoritative for this param", "palette_shift": "param not wired to GPU twin; CPU export authoritative for this param", "spread": "param not wired to GPU twin; CPU export authoritative for this param"},
    "417": {"blur_sigma": "param not wired to GPU twin; CPU export authoritative for this param", "noise_amp": "param not wired to GPU twin; CPU export authoritative for this param"},
    "419": {"noise_scale": "param not wired to GPU twin; CPU export authoritative for this param"},
    "422": {"blur_sigma": "param not wired to GPU twin; CPU export authoritative for this param", "noise_amp": "param not wired to GPU twin; CPU export authoritative for this param"},
    "432": {"brightness": "param not wired to GPU twin; CPU export authoritative for this param", "hue": "param not wired to GPU twin; CPU export authoritative for this param"},
    "464": {"noise_freq": "param not wired to GPU twin; CPU export authoritative for this param"},
    "473": {"contrast": "param not wired to GPU twin; CPU export authoritative for this param", "scale": "param not wired to GPU twin; CPU export authoritative for this param"},
    "499": {"gamma": "param not wired to GPU twin; CPU export authoritative for this param", "n_steps_per_frame": "param not wired to GPU twin; CPU export authoritative for this param"},
    "512": {"resolution": "param not wired to GPU twin; CPU export authoritative for this param (output resolution is a CPU-domain export knob; the closed-form GLSL twin renders at canvas resolution)", "hidden": "param not wired to GPU twin; CPU export authoritative for this param (SIREN hidden-layer width is compute topology, not a live-preview visual control)", "layers": "param not wired to GPU twin; CPU export authoritative for this param (SIREN layer count is compute topology, not a live-preview visual control)"},
    "523": {"n_frames": "param not wired to GPU twin; CPU export authoritative for this param (export frame count, timeline-driven)"},
    "513": {"resolution": "hash-table grid resolution is a CPU-domain export knob; the closed-form GLSL twin renders at the canvas resolution (GPU coverage contract: explicit drop)"},
    "999": {"n_frames": "param not wired to GPU twin; CPU export authoritative for this param"},
    "1003": {"n_frames": "param not wired to GPU twin; CPU export authoritative for this param", "dt": "param not wired to GPU twin; CPU export authoritative for this param", "pace_period": "param not wired to GPU twin; CPU export authoritative for this param", "pace_radius": "param not wired to GPU twin; CPU export authoritative for this param", "rot_radius": "param not wired to GPU twin; CPU export authoritative for this param"},
    "1008": {"n_frames": "param not wired to GPU twin; CPU export authoritative for this param", "dt": "param not wired to GPU twin; CPU export authoritative for this param", "n_seeds": "param not wired to GPU twin; CPU export authoritative for this param"},
    "131": {"grid_size": "param not wired to GPU twin; CPU export authoritative for this param", "noise": "param not wired to GPU twin; CPU export authoritative for this param", "n_frames": "param not wired to GPU twin; CPU export authoritative for this param", "steps_per_frame": "param not wired to GPU twin; CPU export authoritative for this param"},
    # Closed-form live-preview twins (pattern / math-art) whose backing CPU node
    # is a different/serial algorithm whose params do not map 1:1 onto the
    # twin's per-pixel knobs. The twin is a live-preview approximation; the CPU
    # fn stays authoritative (two-tier precision, GPU-First guardrail). These
    # numeric sliders are intentionally not wired to the twin (same pattern as
    # the 16/65/78/56/81/406/409 entries above).
    "342": {"brightness": "param not wired to GPU twin; CPU export authoritative for this param", "points": "param not wired to GPU twin; CPU export authoritative for this param"},
    "444": {"ring_spacing": "param not wired to GPU twin; CPU export authoritative for this param"},
    "498": {"discard": "param not wired to GPU twin; CPU export authoritative for this param", "steps": "param not wired to GPU twin; CPU export authoritative for this param", "walkers": "param not wired to GPU twin; CPU export authoritative for this param"},
    "510": {"dt": "param not wired to GPU twin; CPU export authoritative for this param", "exposure": "param not wired to GPU twin; CPU export authoritative for this param", "noise_scale": "param not wired to GPU twin; CPU export authoritative for this param", "particles": "param not wired to GPU twin; CPU export authoritative for this param", "steps": "param not wired to GPU twin; CPU export authoritative for this param"},
    "957": {"dot_size": "param not wired to GPU twin; CPU export authoritative for this param", "exposure": "param not wired to GPU twin; CPU export authoritative for this param", "gamma": "param not wired to GPU twin; CPU export authoritative for this param", "hue": "param not wired to GPU twin; CPU export authoritative for this param", "n_points": "param not wired to GPU twin; CPU export authoritative for this param", "sat": "param not wired to GPU twin; CPU export authoritative for this param"},
    "962": {"exposure": "param not wired to GPU twin; CPU export authoritative for this param", "gamma": "param not wired to GPU twin; CPU export authoritative for this param", "glow": "param not wired to GPU twin; CPU export authoritative for this param", "hue": "param not wired to GPU twin; CPU export authoritative for this param", "line_width": "param not wired to GPU twin; CPU export authoritative for this param", "major_r": "param not wired to GPU twin; CPU export authoritative for this param", "n_points": "param not wired to GPU twin; CPU export authoritative for this param", "sat": "param not wired to GPU twin; CPU export authoritative for this param", "tube_r": "param not wired to GPU twin; CPU export authoritative for this param"},
    # === P0 closed-form CPU nodes -> faithful typed twins (cron run) ===
    # Preview-approximation twins: node numeric params with no twin uniform are
    # CPU-authoritative (two-tier precision, GPU-First guardrail). Twin uniforms
    # with no CPU-node synonym are documented in test_gpu_twin_invariant._TWIN_UNIFORM_ALLOW.
    "355": {"warp_strength": "param not wired to GPU twin; CPU export authoritative for this param",
            "anisotropy": "param not wired to GPU twin; CPU export authoritative for this param",
            "substeps": "param not wired to GPU twin; CPU export authoritative for this param"},
    "343": {"contrast": "param not wired to GPU twin; CPU export authoritative for this param",
            "edge_width": "param not wired to GPU twin; CPU export authoritative for this param",
            "jitter": "param not wired to GPU twin; CPU export authoritative for this param",
            "octaves": "param not wired to GPU twin; CPU export authoritative for this param"},
    "470": {"detail": "param not wired to GPU twin; CPU export authoritative for this param",
            "elevation": "param not wired to GPU twin; CPU export authoritative for this param",
            "palette_shift": "param not wired to GPU twin; CPU export authoritative for this param",
            "steps": "param not wired to GPU twin; CPU export authoritative for this param",
            "warp_strength": "param not wired to GPU twin; CPU export authoritative for this param"},
    "62": {"bifurcation_max": "param not wired to GPU twin; CPU export authoritative for this param",
           "bifurcation_min": "param not wired to GPU twin; CPU export authoritative for this param",
           "density_inc": "param not wired to GPU twin; CPU export authoritative for this param",
           "lorenz_beta": "param not wired to GPU twin; CPU export authoritative for this param",
           "lorenz_rho": "param not wired to GPU twin; CPU export authoritative for this param",
           "lorenz_sigma": "param not wired to GPU twin; CPU export authoritative for this param",
           "n": "param not wired to GPU twin; CPU export authoritative for this param",
           "poincare_mod": "param not wired to GPU twin; CPU export authoritative for this param",
           "trace_length": "param not wired to GPU twin; CPU export authoritative for this param"},
    "351": {"box_size": "param not wired to GPU twin; CPU export authoritative for this param",
            "c_imag": "param not wired to GPU twin; CPU export authoritative for this param",
            "c_real": "param not wired to GPU twin; CPU export authoritative for this param",
            "fold_rot": "param not wired to GPU twin; CPU export authoritative for this param",
            "folds": "param not wired to GPU twin; CPU export authoritative for this param",
            "scale": "param not wired to GPU twin; CPU export authoritative for this param",
            "warp_strength": "param not wired to GPU twin; CPU export authoritative for this param"},
    "997": {"white": "param not wired to GPU twin; CPU export authoritative for this param"},
    "991": {"blur_sigma": "param not wired to GPU twin; CPU export authoritative for this param",
            "iterations": "param not wired to GPU twin; CPU export authoritative for this param",
            "noise_amp": "param not wired to GPU twin; CPU export authoritative for this param"},
    "445": {"noise_scale": "param not wired to GPU twin; closed-form diffraction_gpu renders a parametric Stam iridescence model (no procedural flow groove substrate); CPU export authoritative for this param"},
    "489": {"noise_amp": "param not wired to GPU twin; closed-form film_grain_gpu renders hash-based grain (no source-based noise field); CPU export authoritative for this param",
            "blur_sigma": "param not wired to GPU twin; closed-form film_grain_gpu has no source-based noise field to blur; CPU export authoritative for this param"},
    # ── Node 353: IFS Fractal attractor ─────────────────────────────────────────
    # Typed-uniform closed-form GPU twin (ifs_fractal_gpu in core/shaders.py).
    # Every numeric CPU param (preset/points/hue_shift/anim_speed) is bound to a
    # named u_<name> uniform. Choice params (coloring/anim_mode) and the legacy
    # time slot are dropped (GPU_PREVIEW_DROP_ALLOW); the preview is live via
    # the orbit-dispatch + u_time. CPU numpy fn stays authoritative export.
    "353": {},
    # ── Node 416: Symmetric Icon attractor ──────────────────────────────────────
    # Typed-uniform closed-form GPU twin (symmetric_icon_gpu in core/shaders.py).
    # Every numeric CPU param (symmetry/a0..a4/palette_shift/anim_speed/
    # seed_strength) is bound to a named u_<name> uniform. Choice params
    # (colormode/source/anim_mode) and the legacy time slot are dropped; preview
    # is live via orbit dispatch + u_time. CPU numpy fn stays authoritative.
    "416": {
        "iterations": "param not wired to GPU twin; CPU export authoritative for this param",
        "orbits": "param not wired to GPU twin; CPU export authoritative for this param",
    },
}

def is_param_justified_drop(mid: str, param: str) -> bool:
    """True if a numeric node param is an explicit, contract-allowed GPU drop."""
    return param in GPU_PREVIEW_DROP_ALLOW.get(str(mid), {})

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
    # ── Node 1008: Cahn-Hilliard Phase Separation (GPU sim twin) ──
    # Spinodal decomposition / phase coarsening — a free-energy (Model B)
    # regime distinct from the reaction-diffusion twins (Gray-Scott 155,
    # Sel'kov 1003, BZ 91). State packs φ (phase) in .r, μ (chemical
    # potential) in .g (two channels). Live-preview twin only; CPU node
    # (simulations/cahn_hilliard.py) stays authoritative for export.
    "1008": {
        "type": "sim",
        "seed": "cahn_hilliard_seed",
        "step": "cahn_hilliard_step",
        "display": "cahn_hilliard_display",
        "state_channels": 2,          # φ in .r, μ in .g
        "substeps": 24,                # Euler steps per rendered frame (live pace)
        "reset_on": ["seed", "param", "loop", "resize"],
        "param_map": {"epsilon": "p1", "mobility": "p2", "seed_variance": "p3"},
    },
    # ── Node 131: Burridge-Knopoff Spring-Block (Earthquake Cascades) ──
    # Frictional spring-block lattice: stress loads slowly, slips past a
    # heterogeneous threshold, and redistributes to 4 neighbors in branching
    # avalanches. A stateful cascade sim (self-organized criticality) with no
    # prior GPU coverage — distinct from the RD/wave/CA sim families. State
    # packs stress in .r, damage in .g, heterogeneous strength in .b (seeded
    # once, preserved). Many substeps/frame let a cascade propagate cell-to-cell
    # within one rendered frame. Live-preview twin only; CPU node
    # (simulations/burridge_knopoff.py) stays authoritative for export.
    "131": {
        "type": "sim",
        "seed": "burridge_seed",
        "step": "burridge_step",
        "display": "burridge_display",
        "state_channels": 3,          # stress .r, damage .g, strength .b
        "substeps": 16,               # cascade relaxation steps per rendered frame
        "reset_on": ["seed", "param", "loop", "resize"],
        "param_map": {"loading_rate": "p1", "threshold": "p2",
                      "residual": "p3", "coupling": "p4"},
    },
    # ── Node 999: Kuramoto Coupled-Oscillator Phase Field (GPU sim twin) ──
    # Self-organized synchronization — a brand-new GPU-sim category (no coupled
    # oscillator existed before). State packs phase in .r, Ω in .g, RNG in .b.
    # Live-preview twin only; CPU node (simulations/kuramoto.py) stays authoritative.
    "999": {
        "type": "sim",
        "seed": "kuramoto_seed",
        "step": "kuramoto_step",
        "display": "kuramoto_display",
        "state_channels": 3,          # θ in .r, Ω in .g, rng in .b
        "substeps": 4,                # Euler steps per rendered frame (live pace)
        "reset_on": ["seed", "param", "loop", "resize"],
        "param_map": {"coupling": "p1", "global_coupling": "p2",
                      "omega_scale": "p3", "dt": "p4"},
    },
    # ── Node 1003: Sel'kov Glycolysis (GPU sim twin) ──
    # Excitable 2-species reaction-diffusion (Sel'kov 1968). State packs U in
    # .r, V in .g (2 channels). The excitable medium ignites spiral/target waves
    # from the seeded blob — a distinct dynamical regime from Gray-Scott (155)
    # and BZ (91). Live-preview twin only; CPU node (simulations/selkov_glycolysis.py)
    # stays authoritative.
    "1003": {
        "type": "sim",
        "seed": "selkov_seed",
        "step": "selkov_step",
        "display": "selkov_display",
        "state_channels": 2,          # U in .r, V in .g
        "substeps": 3,                # Euler steps per rendered frame (live pace)
        "reset_on": ["seed", "param", "loop", "resize"],
        "param_map": {"a": "p1", "b": "p2", "diff_u": "p3", "diff_v": "p4"},
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
