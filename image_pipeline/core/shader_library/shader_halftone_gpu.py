"""shader_halftone_gpu — GPU shader registration (dynamically loaded by core/shaders.py)."""

from ._registry import _register
from ._helpers import _filter_typed



_register("shader_halftone_gpu", "GPU halftone dot screen", "filter", _filter_typed('''
    float gray = dot(orig.rgb, vec3(0.299, 0.587, 0.114));
    float cell = u_cell;
    vec2 q = fract(uv * u_resolution / cell);
    float d = length(q - 0.5);
    float dot_r = (1.0 - gray) * 0.5;
    float v = d < dot_r ? 0.0 : 1.0;
    f_color = vec4(vec3(v), 1.0);
'''), uniforms={
    "cell": {"glsl": "float", "min": 6.0, "max": 40.0, "default": 16.0, "description": "dot cell size"},
})