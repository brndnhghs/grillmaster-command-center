"""shader_heat_shimmer — GPU shader registration (dynamically loaded by core/shaders.py)."""

from ._registry import _register
from ._helpers import _filter_typed



_register("shader_heat_shimmer", "GPU heat haze / shimmer", "filter", _filter_typed('''
    float haze = sin(uv.x * 30.0 + uv.y * 20.0 + u_time * 3.0) * u_strength * 0.02;
    vec2 off = vec2(0.0, haze * (1.0 - uv.y));
    f_color = texture(u_texture, uv + off);
'''), uniforms={
    "strength": {"glsl": "float", "min": 0.0, "max": 1.0, "default": 0.6, "description": "shimmer amount"},
})