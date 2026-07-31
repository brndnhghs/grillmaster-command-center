"""shader_motion_blur_gpu — GPU shader registration (dynamically loaded by core/shaders.py)."""

from ._registry import _register
from ._helpers import _filter_typed



_register("shader_motion_blur_gpu", "GPU directional motion blur", "filter", _filter_typed('''
    float angle = u_angle;
    float dist = u_dist;
    vec2 dir = vec2(cos(angle), sin(angle)) * step * dist;
    vec3 col = vec3(0.0);
    for (int i = -5; i <= 5; i++) {
        float t = float(i) / 5.0;
        col += texture(u_texture, uv + dir * t).rgb * (1.0 - abs(t));
    }
    f_color = vec4(col / 3.5, 1.0);
'''), uniforms={
    "angle": {"glsl": "float", "min": 0.0, "max": 6.2831853, "default": 3.14159265, "description": "blur direction (rad)"},
    "dist":  {"glsl": "float", "min": 0.0, "max": 30.0, "default": 20.0, "description": "blur length (px)"},
})