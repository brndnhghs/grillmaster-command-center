"""phyllotaxis_gpu — GPU shader registration (dynamically loaded by core/shaders.py)."""

from ._registry import _register
from ._helpers import _INFERNO



_register("phyllotaxis_gpu",
            "Phyllotaxis spiral field (client-GPU twin of node 08)",
            "procedural", _INFERNO + """
void main() {
    // u_params.x = point density, .y = angle goldenness, .z = radius scale
    float dens = mix(0.1, 1.0, clamp(u_points, 0.0, 1.0));
    float phi = 2.39996323 + u_angle * 1.5;        // ~golden angle + jitter
    vec2 c = (v_uv - 0.5) * u_resolution;
    float rmax = 0.5 * min(u_resolution.x, u_resolution.y);
    float acc = 0.0;
    for (int i = 0; i < 220; i++) {
        float fi = float(i) * dens * 12.0;
        float a = fi * phi;
        float rad = sqrt(fi) * (u_radius_scale * 0.5 + 0.05) * rmax * 0.06;
        vec2 pos = rad * vec2(cos(a), sin(a));
        acc += smoothstep(3.0, 0.0, length(c - pos));
    }
    f_color = vec4(inferno(clamp(acc * 0.5, 0.0, 1.0)), 1.0);
}
""",
    uniforms={
    "points": {"glsl": "float", "min": 0.0, "max": 1.0, "default": 0.5, "description": "point density"},
    "angle": {"glsl": "float", "min": 0.0, "max": 1.5, "default": 0.5, "description": "angle goldenness"},
    "radius_scale": {"glsl": "float", "min": 0.0, "max": 2.0, "default": 0.5, "description": "radius scale"}
}
    )