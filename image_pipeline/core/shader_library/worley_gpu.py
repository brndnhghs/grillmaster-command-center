"""worley_gpu — GPU shader registration (dynamically loaded by core/shaders.py)."""

from ._registry import _register
from ._helpers import _INFERNO



_register("worley_gpu", "Worley/Cellular F1 noise (client-GPU twin of node 04)", "procedural", _INFERNO + """
vec2 h22(vec2 p){ p = fract(p*vec2(123.34,456.21)); p += dot(p,p+45.32); return fract(vec2(p.x*p.y, p.x+p.y)); }
void main() {
    // u_params.x = jitter (0..1), u_params.y = cell density scale
    float jitter = u_jitter;
    float cells  = 4.0 + u_fractal_gain * 14.0;
    vec2 st = v_uv * cells;
    vec2 g = floor(st), f = fract(st);
    float d = 8.0;
    for (int y = -1; y <= 1; y++) {
        for (int x = -1; x <= 1; x++) {
            vec2 o = vec2(float(x), float(y));
            vec2 fp = 0.5 + (h22(g + o) - 0.5) * jitter;
            d = min(d, length(o + fp - f));
        }
    }
    f_color = vec4(inferno(clamp(d, 0.0, 1.0)), 1.0);
}
""",
    uniforms={
    "jitter": {"glsl": "float", "min": 0.0, "max": 1.0, "default": 0.5, "description": "point jitter"},
    "fractal_gain": {"glsl": "float", "min": 0.0, "max": 1.0, "default": 0.5, "description": "cell density scale"}
}
    )