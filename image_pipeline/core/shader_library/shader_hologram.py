"""shader_hologram — GPU shader registration (dynamically loaded by core/shaders.py)."""

from ._registry import _register
from ._helpers import _filter_typed



_register("shader_hologram", "GPU hologram / scan effect", "filter", _filter_typed('''
    float scan = sin(uv.y * u_resolution.y * 0.5 + u_time * 5.0) * 0.5 + 0.5;
    float scanline = 1.0 - pow(scan, 4.0) * u_scan;
    float edge = abs(uv.x - 0.5) * 2.0;
    float vignette = 1.0 - pow(edge, 3.0) * u_vignette;
    float shift = sin(uv.x * 50.0 + u_time * 3.0) * 0.02;
    vec2 q = uv + vec2(0.0, shift);
    vec3 col = texture(u_texture, q).rgb * scanline * vignette;
    float hue = sin(uv.y * 20.0 + u_time * 2.0) * u_hue + u_hue;
    col += vec3(hue, hue * 0.3, hue * 0.8);
    f_color = vec4(col, 1.0);
'''), uniforms={
    "scan":     {"glsl": "float", "min": 0.0, "max": 0.8, "default": 0.4, "description": "scanline depth"},
    "vignette": {"glsl": "float", "min": 0.0, "max": 1.0, "default": 0.5, "description": "edge vignette"},
    "hue":      {"glsl": "float", "min": 0.0, "max": 0.3, "default": 0.1, "description": "hue tint"},
})