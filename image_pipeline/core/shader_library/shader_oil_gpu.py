"""shader_oil_gpu — GPU shader registration (dynamically loaded by core/shaders.py)."""

from ._registry import _register
from ._helpers import _filter_typed




_register("shader_oil_gpu", "GPU oil painting simulation", "filter", _filter_typed('''
    float radius = u_radius;
    vec3 sum = vec3(0.0); float total = 0.0;
    float scale = radius / 4.0;
    for (int x = -3; x <= 3; x++) {
        for (int y = -3; y <= 3; y++) {
            vec2 off = vec2(float(x), float(y)) * step * scale;
            float w = exp(-float(x*x + y*y) / (radius * radius));
            sum += texture(u_texture, uv + off).rgb * w;
            total += w;
        }
    }
    f_color = vec4(sum / total, 1.0);
'''), uniforms={
    "radius": {"glsl": "float", "min": 1.0, "max": 8.0, "default": 4.0, "description": "brush radius"},
})