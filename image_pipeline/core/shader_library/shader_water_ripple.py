"""shader_water_ripple — GPU shader registration (dynamically loaded by core/shaders.py)."""

from ._registry import _register
from ._helpers import _filter_typed



_register("shader_water_ripple", "GPU water ripple distortion", "filter", _filter_typed('''
    vec2 off = vec2(
        sin(uv.y * 50.0 + u_time * 2.0) * 0.01 * u_amp,
        cos(uv.x * 50.0 + u_time * 1.5) * 0.01 * u_amp
    );
    f_color = texture(u_texture, uv + off);
'''), uniforms={
    "amp": {"glsl": "float", "min": 0.0, "max": 2.0, "default": 0.8, "description": "ripple amplitude"},
})