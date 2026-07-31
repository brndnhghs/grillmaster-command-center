"""dither_palette_gpu — GPU shader registration (dynamically loaded by core/shaders.py)."""

from ._registry import _register
from ._helpers import _filter_typed



# ── P0.4 filter twin (node 422 Palette Posterize) ───────────────────────────
# Client GPU live-preview: ordered (Bayer) dithering against a per-channel band
# palette. p1 = levels (0.5->16), p2 = dither strength (0.5->~2.0). The browser
# preview shows the ordered-dither path; the authoritative CPU numpy node
# (median-cut + Floyd-Steinberg + CIELAB) is the export. Per pitfall #15b, no
# local named `step` (the _filter_shader wrapper injects it).
_register("dither_palette_gpu", "GPU palette posterize with ordered dither (client twin of 422)", "filter", _filter_typed('''
    // u_levels = band count per channel; u_dither_scale = ordered-dither strength.
    int levels = int(clamp(2.0 + u_levels * 28.0, 2.0, 32.0));
    float strength = clamp((u_dither_scale - 0.5) * 4.0, 0.0, 4.0);

    vec3 col = orig.rgb;
    // perceptual luma weighting (approximates CIELAB channel weighting)
    float luma = dot(col, vec3(0.2126, 0.7152, 0.0722));
    col = mix(col, vec3(luma), 0.15);

    // 4x4 Bayer matrix (values 0..15)
    const float bayer[16] = float[16](
        0.0,  8.0,  2.0, 10.0,
       12.0,  4.0, 14.0,  6.0,
        3.0, 11.0,  1.0,  9.0,
       15.0,  7.0, 13.0,  5.0);
    ivec2 ip = ivec2(floor(v_uv * u_resolution));
    int bi = (ip.y & 3) * 4 + (ip.x & 3);
    float thr = (bayer[bi] / 16.0 - 0.5) * strength / float(levels);

    vec3 q = floor(col * float(levels) + thr) / float(levels);
    f_color = vec4(clamp(q, 0.0, 1.0), 1.0);
'''), uniforms={
        "levels": {"glsl": "float", "min": 0.0, "max": 1.0, "default": 0.5, "description": "posterize levels"},
        "dither_scale": {"glsl": "float", "min": 0.0, "max": 1.0, "default": 0.5, "description": "ordered dither strength"},
    })