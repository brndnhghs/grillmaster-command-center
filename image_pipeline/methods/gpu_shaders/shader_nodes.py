"""
GPU shader node registration tables — pure data, no logic.

Consumed by _shared.py (the @method factories + registration loops) and
re-exported through the package __init__. Adding a node = adding one tuple
here; the factory derives params, ports, tags, is_time_varying, and the
client-side GPU_SHADER_NODE_MAP entry from it.

- _PROC_SHADERS / _FILT_SHADERS: legacy nodes using the generic p1..p4
  (procedural) or strength/p2 (filter) param shims.
- _TYPED_SHADER_NODES: typed-uniform nodes — every declared GLSL uniform
  becomes a real named node param + a wireable SCALAR port.
- _PROC_PARAMS / _FILT_PARAMS: shared param specs for the generic nodes.
"""



# ── Ordered shader lists (stable IDs) ────────────────────────────────
_PROC_SHADERS = [
    ("162", "mandelbrot",         "GPU Mandelbrot"),
    ("163", "julia",              "GPU Julia"),
    ("164", "plasma",             "GPU Plasma"),
    ("165", "domain_warp",        "GPU Domain Warp"),
    ("166", "voronoi",            "GPU Voronoi"),
    ("167", "voronoise",          "GPU Voronoise"),
    ("168", "ripples",            "GPU Ripples"),
    ("169", "cells",              "GPU Cells"),
    ("170", "bubble_chamber",     "GPU Bubble Chamber"),
    ("171", "stars",              "GPU Stars"),
    ("172", "lightning_fractal",  "GPU Lightning Fractal"),
    ("173", "spiral",             "GPU Spiral"),
    ("174", "dendritic",          "GPU Dendritic"),
    ("175", "barnsley",           "GPU Barnsley Fern"),
    ("176", "spectral",           "GPU Spectral"),
    ("177", "truchet",            "GPU Truchet"),
    ("178", "kaleidoscope_fractal","GPU Kaleidoscope Fractal"),
    ("179", "waves_3d",           "GPU Waves 3D"),
    ("181", "ocean",              "GPU Ocean"),
    ("182", "nebula_gpu",         "GPU Nebula"),
    ("183", "terrain",            "GPU Terrain"),
    ("184", "wood_grain_gpu",     "GPU Wood Grain"),
    ("185", "fire_gpu",           "GPU Fire"),
    ("186", "smoke_gpu",          "GPU Smoke"),
    ("355", "sdf_raymarch_gpu",   "GPU SDF Raymarch"),
    ("356", "dot_noise_gpu",      "GPU Dot Noise"),
]

_FILT_SHADERS = [
    ("187", "shader_bloom",           "GPU Bloom"),
    ("188", "shader_emboss",          "GPU Emboss"),
    ("189", "shader_kaleidoscope",    "GPU Kaleidoscope"),
    ("190", "shader_water_ripple",    "GPU Water Ripple"),
    ("191", "shader_heat_shimmer",    "GPU Heat Shimmer"),
    ("192", "shader_pixelate_gpu",    "GPU Pixelate"),
    ("193", "shader_ink_bleed",       "GPU Ink Bleed"),
    ("194", "shader_halftone_gpu",    "GPU Halftone"),
    ("195", "shader_crt_gpu",         "GPU CRT"),
    ("196", "shader_hologram",        "GPU Hologram"),
    ("197", "shader_mosaic_gpu",      "GPU Mosaic"),
    ("198", "shader_edge_detect_gpu", "GPU Edge Detect"),
    ("199", "shader_warhol",          "GPU Warhol"),
    ("200", "shader_duotone_gpu",     "GPU Duotone"),
    ("201", "shader_rgb_split",       "GPU RGB Split"),
    ("202", "shader_caustics_gpu",    "GPU Caustics"),
    ("203", "shader_glitch_gpu",      "GPU Glitch"),
    ("204", "shader_posterize_gpu",   "GPU Posterize"),
    ("205", "shader_oil_gpu",         "GPU Oil Paint"),
    ("206", "shader_neon_gpu",        "GPU Neon Glow"),
    ("207", "shader_pencil_gpu",      "GPU Pencil"),
    ("208", "shader_motion_blur_gpu", "GPU Motion Blur"),
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
    ("209", "gradient_gpu2",    "GPU Gradient"),
    ("210", "ascii_art_gpu",    "GPU ASCII Art"),
    ("211", "solid_color_gpu",  "GPU Solid Color"),
    ("212", "checker_gpu2",     "GPU Checkerboard"),
    ("213", "wave_pattern_gpu", "GPU Wave Pattern"),
    ("214", "fbm_noise_gpu",    "GPU FBM Noise"),
    # Categorical coverage expansion (2026-07-10): animated plasma, voronoi
    # cells, and the filter family — kaleidoscope / bloom / posterize / edge.
    ("215", "plasma_gpu2",        "GPU Plasma 2"),
    ("216", "voronoi_gpu2",       "GPU Voronoi 2"),
    ("217", "kaleidoscope_gpu",   "GPU Kaleidoscope"),
    ("218", "bloom_gpu",          "GPU Bloom"),
    ("219", "posterize_gpu",      "GPU Posterize"),
    ("220", "edge_gpu",           "GPU Edge Detect"),
    # Categorical coverage expansion (2026-07-10 pt.2): displacement, RGB split,
    # halftone screen, concentric rings, truchet tiling, pixelate/mosaic.
    ("221", "swirl_gpu",          "GPU Swirl"),
    ("222", "chromatic_gpu",      "GPU Chromatic Aberration"),
    ("223", "halftone_gpu",       "GPU Halftone"),
    ("224", "rings_gpu",          "GPU Rings"),
    ("225", "truchet_gpu",       "GPU Truchet"),
    ("226", "pixelate_gpu",       "GPU Pixelate"),
    # Categorical coverage pt.3 (2026-07-10): signature escape-time fractals
    # with NAMED typed controls (zoom/center/iterations/palette/colors) replacing
    # the opaque p1..p4 shims — Mandelbrot, Julia, Burning Ship, Newton,
    # Sierpinski, Lyapunov.
    ("227", "mandelbrot_typed",   "GPU Mandelbrot"),
    ("228", "julia_typed",        "GPU Julia"),
    ("229", "burning_ship_typed", "GPU Burning Ship"),
    ("230", "newton_typed",       "GPU Newton"),
    ("231", "sierpinski_typed",   "GPU Sierpinski"),
    ("232", "lyapunov_typed",     "GPU Lyapunov"),
    # Categorical coverage pt.4 (2026-07-11): per-pixel filter / color-grade
    # family with NAMED typed controls — box blur, unsharp sharpen, vignette,
    # luminance threshold, hue rotate, ordered (Bayer) dither.
    ("233", "box_blur_gpu",       "GPU Box Blur"),
    ("234", "sharpen_gpu",        "GPU Sharpen"),
    ("235", "vignette_gpu",       "GPU Vignette"),
    ("236", "threshold_gpu",      "GPU Threshold"),
    ("237", "hue_shift_gpu",      "GPU Hue Shift"),
    ("238", "dither_gpu",         "GPU Dither"),
    # Categorical coverage expansion (2026-07-11): closed-form field-eval twins
    # moire / chladni / dunes / quasicrystal / metaballs / nebula / wood / ripples.
    ("239", "moire_typed",         "GPU Moiré"),
    ("240", "chladni_typed",       "GPU Chladni"),
    ("241", "dunes_typed",         "GPU Dunes"),
    ("242", "quasicrystal_typed",  "GPU Quasicrystal"),
    ("243", "metaballs_typed",     "GPU Metaballs"),
    ("244", "nebula_typed",        "GPU Nebula"),
    ("245", "wood_grain_typed",   "GPU Wood Grain"),
    ("246", "ripples_typed",      "GPU Ripples"),
    # Categorical coverage pt.6 (2026-07-11): derivative-field filters that
    # derive a FIELD from the upstream image — Sobel magnitude / direction,
    # Laplacian, Scharr, normal map, gradient orientation flow, emboss. Single
    # image_in: IMAGE; every numeric variable is a wireable SCALAR port.
    ("247", "sobel_mag_typed",    "GPU Sobel Magnitude"),
    ("248", "sobel_dir_typed",    "GPU Sobel Direction"),
    ("249", "laplacian_typed",    "GPU Laplacian"),
    ("250", "scharr_typed",       "GPU Scharr"),
    ("251", "normal_map_typed",   "GPU Normal Map"),
    ("252", "gradient_orient_typed", "GPU Gradient Flow"),
    ("253", "emboss_typed",       "GPU Emboss"),
    # Categorical coverage pt.7 (2026-07-11): closed-form pattern generators
    # with NAMED typed controls — spirograph, truchet maze, reaction waves,
    # hex grid, starfield, concentric rings.
    ("254", "spirograph_typed",    "GPU Spirograph"),
    ("255", "truchet_maze_typed",  "GPU Truchet Maze"),
    ("256", "reaction_waves_typed", "GPU Reaction Waves"),
    ("257", "hex_grid_typed",      "GPU Hex Grid"),
    ("258", "starfield_typed",     "GPU Starfield"),
    ("259", "concentric_rings_typed", "GPU Concentric Rings"),
    # Categorical coverage pt.8 (2026-07-11): closed-form math_art patterns
    # with NAMED typed controls — Ulam-spiral homage, hash maze, circle
    # packing, Fourier epicycles, summed waveform, Clifford strange-attractor.
    ("260", "ulam_spiral_typed",     "GPU Ulam Spiral"),
    ("261", "maze_typed",            "GPU Hash Maze"),
    ("262", "circle_packing_typed",  "GPU Circle Packing"),
    ("263", "fourier_circles_typed", "GPU Fourier Circles"),
    ("264", "waveform_typed",        "GPU Waveform"),
    ("265", "strange_attractor_typed", "GPU Strange Attractor"),
    # Categorical coverage pt.8 (2026): closed-form patterns with NAMED typed
    # controls — phyllotaxis dots, guilloché engraving, Lissajous trace, radial
    # wave interference, curl-noise flow field, kaleidoscopic petal bloom.
    ("266", "phyllotaxis_typed",   "GPU Phyllotaxis"),
    ("267", "guilloche_typed",     "GPU Guilloché"),
    ("268", "lissajous_typed",     "GPU Lissajous"),
    ("269", "interference_typed",  "GPU Wave Interference"),
    ("270", "flow_field_typed",    "GPU Flow Field"),
    ("271", "kaleido_bloom_typed", "GPU Kaleido Bloom"),
    # Categorical coverage pt.9 (2026-07-11): closed-form math_art patterns
    # with NAMED typed controls — superformula, harmonograph, Maurer rose,
    # magnetic dipole field, star polygon {n/k}, torus-knot ribbon.
    ("272", "superformula_typed",  "GPU Superformula"),
    ("273", "harmonograph_typed",  "GPU Harmonograph"),
    ("274", "maurer_rose_typed",   "GPU Maurer Rose"),
    ("275", "magnetic_typed",      "GPU Magnetic Field"),
    ("276", "star_polygon_typed",  "GPU Star Polygon"),
    ("277", "torusknot_typed",   "GPU Torus Knot"),
    # Categorical coverage pt.10 (2026-07-11): closed-form pattern nodes with
    # NAMED typed controls — infinite tunnel, vortex/galaxy field, woven fabric,
    # topographic contour map, cross-hatch engraving, domain-warped grid.
    ("278", "tunnel_typed",    "GPU Tunnel"),
    ("279", "vortex_typed",    "GPU Vortex"),
    ("280", "weave_typed",     "GPU Weave"),
    ("281", "contour_typed",   "GPU Contour Map"),
    ("282", "hatch_typed",     "GPU Cross-Hatch"),
    ("283", "gridwarp_typed",  "GPU Warp Grid"),
    # Categorical coverage pt.11 (2026-07-11): extended closed-form procedural
    # family with NAMED typed controls — domain-warped flow, animated caustics,
    # spectral prism, SDF scene, radial energy burst, iridescent bubble foam.
    ("284", "domainwarp_typed", "GPU Domain Warp"),
    ("285", "caustics_typed",   "GPU Caustics"),
    ("286", "prism_typed",      "GPU Spectral Prism"),
    ("287", "sdfscene_typed",   "GPU SDF Scene"),
    ("288", "burst_typed",      "GPU Energy Burst"),
    ("289", "foam_typed",   "GPU Bubble Foam"),
    # Categorical coverage pt.12 (2026-07-11): closed-form pattern node —
    # Gyroid / triply-periodic minimal-surface slice (animation by in-plane
    # spin + slice advance through the 3D field).
    ("290", "gyroid_typed", "GPU Gyroid"),
    # Categorical coverage pt.13 (2026-07-11): closed-form generative-art
    # patterns with NAMED typed controls — Schotter grid, Thue-Morse fractal,
    # crystal diffraction, Apollonian gasket, confocal parabola family,
    # Poincaré-disk hyperbolic tiling.
    ("291", "schotter_typed",    "GPU Schotter"),
    ("292", "thue_morse_typed",  "GPU Thue-Morse"),
    ("293", "crystal_typed",     "GPU Crystal Diffraction"),
    ("294", "apollonian_typed",  "GPU Apollonian Gasket"),
    ("295", "parabola_typed",    "GPU Parabola Family"),
    ("296", "hyperbolic_typed", "GPU Hyperbolic Tiling"),
    # Categorical coverage pt.14 (2026-07-12): real-time volumetric clouds —
    # screen-space fbm density raymarch with single-scatter sun lighting.
    ("297", "clouds_typed", "GPU Volumetric Clouds"),
    # Categorical coverage pt.15 (2026-07-12): closed-form procedural patterns
    # with NAMED typed controls — Droste log-spiral, Voronoi stained glass,
    # Op-Art sinusoidal band distortion. (309 free; 310-315 are CPU method ids.)
    ("304", "droste_typed",        "GPU Droste Spiral"),
    ("305", "stained_glass_typed", "GPU Stained Glass"),
    ("306", "opart_typed",         "GPU Op-Art Waves"),
    ("307", "aurora_typed",        "GPU Aurora Borealis"),
    # Categorical coverage pt.16 (2026-07-13): closed-form procedural — classic
    # Perlin-turbulence marble veining with domain warp (typed, node 320).
    ("308", "marble_typed",         "GPU Marble"),
    # Node 321: Smooth-min Metaballs — Quilez exponential smin union of
    # orbiting SDF spheres. Distinct from node 53 (sum-of-inverse-square
    # field): true SDF + smin so `blend` (k) controls edge softness.
    ("309", "smin_metaballs_gpu",   "GPU Smooth-min Metaballs"),
    # Node 322: Procedural Phasor Noise (Tricard 2019) — sum of complex Gabor
    # kernels; the ARGUMENT (phase) of the accumulated phasor field gives
    # intensity-decoupled oscillating ridges (fingerprint/wood-grain) with
    # locally controllable frequency + orientation. Renders the PHASE, not the
    # magnitude — distinct from any Perlin/Gabor magnitude node.
    ("310", "phasor_noise_gpu",      "GPU Phasor Noise"),
    # Node 323: Raymarched 3D Gyroid TPMS — sphere-traced triply-periodic
    # minimal surface with lambert+specular shading and orbiting camera.
    # Distinct from node 301 (flat 2D scalar-field slice): full 3D volume with
    # depth, self-occlusion and lighting.
    ("311", "gyroid_raymarch_typed", "GPU Raymarched Gyroid"),
    # Categorical coverage pt.17 (2026-07-14): closed-form recursive subdivision
    # fractal with NAMED typed controls — Sierpinski (Menger) carpet. Cells are
    # coloured by recursion depth; the plane spins and scale breathes with time.
    ("312", "menger_typed", "GPU Menger Carpet"),
    # Categorical coverage pt.18 (2026-07-14): closed-form atmospheric sky —
    # Nishita single-scattering GPU twin of CPU node 471. Per-pixel ray-march
    # (no ping-pong state), animated sun day-arc via u_time. Named typed
    # uniforms mirror node 471's real numeric params (contract #5/#6).
    ("313", "nishita_sky_gpu", "GPU Nishita Sky"),
    # Categorical coverage pt.19 (2026-07-14): stochastic hex-tiling filter —
    # Heitz & Neyret (HPG 2018) histogram-preserving blending operator that
    # tiles the wired input image across the plane with NO visible repetition.
    # A true FILTER (image_in: IMAGE) with named typed controls.
    ("315", "hex_tiling_gpu", "GPU Hex Tiling"),
    # Categorical coverage pt.20 (2026-07-16): Interior Mapping (van Dongen,
    # CGI 2008) — believable 3D rooms behind a flat facade via per-pixel
    # ray-box intersection, NO added geometry. Closed-form f(uv,t) procedural
    # twin: eye ray into a repeating room-grid, nearest interior wall shaded
    # with depth tint + hashed per-room window lights (u_time twinkle).
    ("316", "interior_mapping_typed", "GPU Interior Mapping"),
    # Node 330: Kaleidoscopic IFS - box-fold + sphere-fold (Knighty/Kali 2010).
    # Distinct from node 402 kifs_gpu (wedge + scale only): adds the sphere
    # fold (minR/maxR radius clamp) that opens the characteristic holes, plus
    # a per-iteration rotation. All 5 controls are wireable SCALAR ports so
    # LFO/counter nodes can drive the live kaleidoscope animation, and
    # contrast-only static culls are avoided by the genuine u_time motion.
    ("317", "kifs_spherefold_gpu", "GPU KIFS Fractal"),
    # Node 331: Mandelbulb — 3D escape-time fractal (White & Nylander 2009),
    # sphere-traced via the Hart et al. 1989 distance estimator. The canonical
    # "3D Mandelbrot": distinct from the 2D escape-time family and from the 3D
    # TPMS raymarches (gyroid/menger). Genuinely time-varying (power morph +
    # orbiting camera) so animation drivers have a visibly-responsive target.
    ("318", "mandelbulb_gpu", "GPU Mandelbulb"),
    # Node 332: De Jong Attractor - GPU live-preview twin of CPU node 498.
    # Closed-form de Jong chaos map rendered as a single-pass density field
    # tone-mapped with the inferno ramp - parity with node 498 density colouring.
    # Named typed uniforms mirror node 498 real numeric params (a/b/c/d/exposure).
    # morph+speed animate the params via u_time (CPU anim_mode morph_all) so the
    # live preview is genuinely time-varying. CPU numpy node stays authoritative.
    # 332 is the free ID above 301 (333-334 also free; 335 taken by domain warp).
    ("333", "de_jong_typed", "GPU De Jong Attractor"),
    # Node 335: Domain Warping (Inigo Quilez, 2015) — fbm(fbm(p + fbm(p)))
    # two-level noise feed-forward gives marbled organic flow distinct from
    # single fbm (node 225); animated by scrolling the inner warp with u_time
    # so contrast-only static culls are avoided. (ID 335: 332-334 are
    # taken by CPU nodes — GPU-typed nodes must use free IDs above 301. Distinct
    # shader name domain_warp_palette_gpu vs node 311's domain_warp_gpu: the
    # 311 twin keeps the IQ-inferno look; this node adds a 4-colour palette.)
    ("322", "domain_warp_palette_gpu", "GPU Domain Warp"),
    # Node 309: Mandelbox — 3D escape-time fractal (Tom Lowe 2010), the box-fold
    # + sphere-fold companion to the Mandelbulb (node 331). DE raymarch (Hart et
    # al. 1989); the negative scale yields the iconic tiled infinite-rooms look.
    # Genuinely time-varying (orbiting camera + scale breathing) so it survives
    # the contrast-only static liveness cull and feeds animation drivers.
    # (309 is the free ID above 301 — 310-315 are taken by CPU method ids.)
    ("298", "mandelbox_gpu", "GPU Mandelbox"),
    # Node 352: Gerstner Ocean — analytic trochoidal-wave height field with
    # Blinn-Phong sun glitter (typed GPU twin of CPU node 963). Closed-form
    # f(uv,t): wave phases advance with u_time so the live preview is genuinely
    # animated (survives the contrast-only static liveness cull). CPU
    # numpy node 963 stays authoritative for export (two-tier precision).
    # 352 is the free ID above 301.
    ("339", "gerstner_ocean_gpu", "GPU Gerstner Ocean"),
    # Node 360: Gyroid TPMS — closed-form triply-periodic minimal-surface shell
    # on a swept slice plane (typed GPU twin of CPU node 964). Genuinely
    # time-varying: the slice-plane z advances with u_time so the 2D cross
    # section morphs continuously (survives the contrast-only static
    # cull). CPU numpy node 964 stays authoritative for export.
    # 360 is the free ID above 301.
    ("347", "gyroid_tpms_gpu", "GPU Gyroid TPMS"),
    # Node 361: Phasor Noise — sparse-convolution complex-phasor field (typed GPU
    # twin of CPU node 1006). Closed-form f(uv,t): the global phase advances with
    # u_time so the live preview is genuinely animated (survives the
    # contrast-only static liveness cull). CPU numpy node 1006 stays authoritative
    # for export. 361 is the free ID above 360.
    ("348", "phasor_noise_gpu", "GPU Phasor Noise"),
]
