"""shader_duotone_gpu — GPU shader registration (dynamically loaded by core/shaders.py)."""

from ._registry import _register
from ._helpers import _filter_typed



_register("shader_duotone_gpu", "GPU duotone with color controls", "filter", _filter_typed('''
    float gray = dot(orig.rgb, vec3(0.299, 0.587, 0.114));
    f_color = vec4(mix(u_color_shadow, u_color_highlight, gray), 1.0);
'''), uniforms={
    "color_shadow":   {"glsl": "color", "default": "#3366cc", "description": "shadow color"},
    "color_highlight":{"glsl": "color", "default": "#ffcc33", "description": "highlight color"},
})