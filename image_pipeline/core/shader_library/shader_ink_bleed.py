"""shader_ink_bleed — GPU shader registration (dynamically loaded by core/shaders.py)."""

from ._registry import _register
from ._helpers import _filter_typed



_register("shader_ink_bleed", "GPU ink bleed / watercolor spread", "filter", _filter_typed('''
    vec3 sum = vec3(0.0);
    float count = 0.0;
    for (int x = -4; x <= 4; x++) {
        for (int y = -4; y <= 4; y++) {
            vec2 off = vec2(float(x), float(y)) * step * u_spread;
            float w = exp(-float(x*x + y*y) / (4.0 * u_spread));
            sum += texture(u_texture, uv + off).rgb * w;
            count += w;
        }
    }
    f_color = vec4(sum / count, 1.0);
'''), uniforms={
    "spread": {"glsl": "float", "min": 0.1, "max": 3.0, "default": 0.6, "description": "bleed radius"},
})