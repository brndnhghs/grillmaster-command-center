"""shader_bloom — GPU shader registration (dynamically loaded by core/shaders.py)."""

from ._registry import _register
from ._helpers import _filter_typed



_register("shader_bloom", "GPU bloom/glow from bright areas", "filter", _filter_typed('''
    float brightness = dot(orig.rgb, vec3(0.299, 0.587, 0.114));
    vec4 glow = vec4(0.0);
    for (int x = -3; x <= 3; x++) {
        for (int y = -3; y <= 3; y++) {
            vec2 off = vec2(float(x), float(y)) * step * 2.0;
            float b = dot(texture(u_texture, uv + off).rgb, vec3(0.299, 0.587, 0.114));
            if (b > 0.7) glow += texture(u_texture, uv + off) * exp(-float(x*x + y*y) / 4.0);
        }
    }
    glow /= 8.0;
    f_color = orig + glow * u_strength;
'''), uniforms={
    "strength": {"glsl": "float", "min": 0.0, "max": 2.0, "default": 0.8, "description": "bloom intensity"},
})