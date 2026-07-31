"""shader_neon_gpu — GPU shader registration (dynamically loaded by core/shaders.py)."""

from ._registry import _register
from ._helpers import _filter_typed



_register("shader_neon_gpu", "GPU neon glow on edges", "filter", _filter_typed('''
    float gx = 0.0, gy = 0.0;
    for (int x = -1; x <= 1; x++) {
        for (int y = -1; y <= 1; y++) {
            vec2 off = vec2(float(x), float(y)) * step;
            float v = dot(texture(u_texture, uv + off).rgb, vec3(0.299, 0.587, 0.114));
            gx += float(x) * v; gy += float(y) * v;
        }
    }
    float edge = sqrt(gx*gx + gy*gy);
    float glow = edge * u_intensity * 3.0;
    vec3 neon = vec3(glow * 0.8, glow * 0.3, glow);
    f_color = vec4(orig.rgb + neon, 1.0);
'''), uniforms={
    "intensity": {"glsl": "float", "min": 0.0, "max": 2.0, "default": 0.5, "description": "neon glow intensity"},
})