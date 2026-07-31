"""shader_posterize_gpu — GPU shader registration (dynamically loaded by core/shaders.py)."""

from ._registry import _register
from ._helpers import _filter_typed



_register("shader_posterize_gpu", "GPU posterization / color reduction", "filter", _filter_typed('''
    float nc = u_n_colors;
    vec3 col = floor(orig.rgb * nc) / nc;
    f_color = vec4(col, 1.0);
'''), uniforms={
    "n_colors": {"glsl": "float", "min": 2.0, "max": 16.0, "default": 9.0, "description": "color levels"},
})