"""shader_warhol — GPU shader registration (dynamically loaded by core/shaders.py)."""

from ._registry import _register
from ._helpers import _filter_typed



_register("shader_warhol", "GPU Warhol 4-panel duotone", "filter", _filter_typed('''
    float gray = dot(orig.rgb, vec3(0.299, 0.587, 0.114));
    gray = clamp(pow(gray, u_gamma), 0.0, 1.0);
    vec2 p = floor(uv * 2.0);
    vec3 c1, c2;
    if (p.x < 1.0 && p.y < 1.0) { c1 = vec3(0.8, 0.2, 0.2); c2 = vec3(1.0, 1.0, 0.4); }
    else if (p.x >= 1.0 && p.y < 1.0) { c1 = vec3(0.2, 0.4, 0.8); c2 = vec3(0.6, 0.2, 0.6); }
    else if (p.x < 1.0 && p.y >= 1.0) { c1 = vec3(0.2, 0.8, 0.2); c2 = vec3(0.4, 0.2, 0.8); }
    else { c1 = vec3(0.8, 0.6, 0.2); c2 = vec3(0.8, 0.2, 0.2); }
    f_color = vec4(mix(c1, c2, gray), 1.0);
'''), uniforms={
    "gamma": {"glsl": "float", "min": 0.25, "max": 3.0, "default": 1.0, "description": "panel contrast gamma (1.0 = classic)"},
})