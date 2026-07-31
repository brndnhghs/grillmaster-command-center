"""shader_crt_gpu — GPU shader registration (dynamically loaded by core/shaders.py)."""

from ._registry import _register
from ._helpers import _filter_typed



_register("shader_crt_gpu", "GPU CRT scanlines + bloom", "filter", _filter_typed('''
    float scan = sin(uv.y * u_resolution.y * 3.14159) * 0.5 + 0.5;
    float scanline = 1.0 - (1.0 - scan) * u_scanline;
    // chromatic shift at edges
    vec2 r_uv = uv + vec2(u_chroma * pow(abs(uv.x - 0.5) * 2.0, 2.0), 0.0);
    vec2 b_uv = uv - vec2(u_chroma * pow(abs(uv.x - 0.5) * 2.0, 2.0), 0.0);
    vec3 col;
    col.r = texture(u_texture, r_uv).r;
    col.g = texture(u_texture, uv).g;
    col.b = texture(u_texture, b_uv).b;
    col *= scanline;
    f_color = vec4(col, 1.0);
'''), uniforms={
    "scanline": {"glsl": "float", "min": 0.0, "max": 0.7, "default": 0.3, "description": "scanline darkness"},
    "chroma":   {"glsl": "float", "min": 0.0, "max": 0.004, "default": 0.001, "description": "RGB chroma shift"},
})