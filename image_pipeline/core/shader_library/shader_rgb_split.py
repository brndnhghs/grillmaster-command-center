"""shader_rgb_split — GPU shader registration (dynamically loaded by core/shaders.py)."""

from ._registry import _register
from ._helpers import _filter_typed



_register("shader_rgb_split", "GPU RGB channel separation", "filter", _filter_typed('''
    float shift = u_shift;
    vec2 r_uv = uv + vec2(shift, 0.0);
    vec2 b_uv = uv - vec2(shift, 0.0);
    float r = texture(u_texture, r_uv).r;
    float g = orig.g;
    float b = texture(u_texture, b_uv).b;
    f_color = vec4(r, g, b, 1.0);
'''), uniforms={
    "shift": {"glsl": "float", "min": 0.0, "max": 0.05, "default": 0.02, "description": "channel shift"},
})