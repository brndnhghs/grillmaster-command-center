"""morph_grid_gpu — GPU shader registration (dynamically loaded by core/shaders.py)."""

from ._registry import _register
from ._helpers import _INFERNO



_register("morph_grid_gpu",
            "Morphing grid warp (client-GPU twin of node 105)",
            "procedural", _INFERNO + """
void main() {
    // u_params.x = warp strength, .y = line width, .z = palette mix
    float ws = u_warp_strength;
    float lw = clamp(u_line_width, 0.02, 1.0);
    vec2 p = v_uv * 14.0;
    vec2 w = vec2(fbm(p + u_time * 0.1), fbm(p.yx - u_time * 0.1));
    p += (w - 0.5) * ws * 6.0;
    vec2 g = abs(fract(p) - 0.5);
    float line = smoothstep(lw, lw * 0.5, min(g.x, g.y));
    vec3 col = 0.5 + 0.5 * cos(6.2831853 * (0.5 + vec3(0.0, 0.33, 0.67)) + w.x * 4.0);
    f_color = vec4(mix(vec3(line), col, line), 1.0);
}
""",
    uniforms={
    "warp_strength": {"glsl": "float", "min": 0.0, "max": 2.0, "default": 0.5, "description": "warp strength"},
    "line_width": {"glsl": "float", "min": 0.02, "max": 1.0, "default": 0.5, "description": "line width"}
}
    )