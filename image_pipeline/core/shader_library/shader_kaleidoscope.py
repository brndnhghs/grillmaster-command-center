"""shader_kaleidoscope — GPU shader registration (dynamically loaded by core/shaders.py)."""

from ._registry import _register
from ._helpers import _filter_typed



_register("shader_kaleidoscope", "GPU kaleidoscope mirror", "filter", _filter_typed('''
    vec2 p = uv - 0.5;
    float a = atan(p.y, p.x);
    float r = length(p);
    float seg = 3.14159 * 2.0 / max(3.0, u_segments);
    a = mod(a, seg);
    a = abs(a - seg * 0.5);
    vec2 q = vec2(cos(a), sin(a)) * r + 0.5;
    f_color = texture(u_texture, q);
'''), uniforms={
    "segments": {"glsl": "float", "min": 3.0, "max": 16.0, "default": 8.0, "description": "mirror segments"},
})