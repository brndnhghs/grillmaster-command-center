"""hdr_gpu — GPU shader registration (dynamically loaded by core/shaders.py)."""

from ._registry import _register
from ._helpers import _filter_typed



# ── P0.4 client-GPU twin shaders for existing CPU filter nodes ──
# Each maps a pre-existing CPU filter node's LIVE preview onto a GLSL twin.
# The CPU numpy path stays the authoritative export (two-tier precision).

# 42 Fake HDR — contrast / saturation / vignette / bloom (GPU live twin)
_register("hdr_gpu", "GPU fake-HDR tonemap (contrast/sat/vignette/bloom)", "filter", _filter_typed('''
    float gray = dot(orig.rgb, vec3(0.299, 0.587, 0.114));
    // contrast around mid-gray
    vec3 c = (orig.rgb - 0.5) * (0.5 + u_contrast * 3.0) + 0.5;
    // saturation toward/away from luma
    c = mix(vec3(gray), c, 0.5 + u_saturation * 2.0);
    // bloom: cheap bright-area lift
    float bright = max(0.0, gray - 0.6) * u_bloom * 2.0;
    c += bright;
    // vignette
    vec2 d = uv - 0.5;
    float vig = 1.0 - dot(d, d) * u_vignette * 2.5;
    c *= clamp(vig, 0.0, 1.0);
    f_color = vec4(clamp(c, 0.0, 1.0), 1.0);
'''), uniforms={
        "contrast": {"glsl": "float", "min": 0.0, "max": 1.0, "default": 0.5, "description": "tone contrast"},
        "saturation": {"glsl": "float", "min": 0.0, "max": 1.0, "default": 0.5, "description": "saturation"},
        "vignette": {"glsl": "float", "min": 0.0, "max": 1.0, "default": 0.5, "description": "vignette strength"},
        "bloom": {"glsl": "float", "min": 0.0, "max": 1.0, "default": 0.5, "description": "bloom lift"},
    })