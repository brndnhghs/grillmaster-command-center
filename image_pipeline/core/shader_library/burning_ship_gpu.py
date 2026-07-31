"""burning_ship_gpu — GPU shader registration (dynamically loaded by core/shaders.py)."""

from ._registry import _register
from ._helpers import _FRACTAL_HELPERS



_register("burning_ship_gpu", "Burning Ship fractal (client-GPU twin of node 51)", "procedural",
          _FRACTAL_HELPERS + '''
void main() {
    vec2 uv = (gl_FragCoord.xy - 0.5 * u_resolution) / min(u_resolution.x, u_resolution.y);
    vec2 ctr = vec2(0.5, 0.5);
    vec2 c = ctr + uv * 1.0;  // fixed full view (node 51 has no zoom param)
    vec2 z = vec2(0.0);
    float n = 0.0;
    float last2 = 0.0;
    const float MAXI = 500.0;
    for (int i = 0; i < 500; i++) {
        z = vec2(abs(z.x) - 1.0, abs(z.y)) * abs(z.x) + c; // abs-squared ship map
        z = vec2(z.x*z.x - z.y*z.y, 2.0*z.x*z.y);
        last2 = dot(z, z);
        if (last2 > 16.0 || n >= u_iterations) break;
        n += 1.0;
    }
    float t = (n >= u_iterations - 0.5) ? 0.0 : smooth_iter(n, last2, u_iterations);
    f_color = vec4(fractal_palette(t * (0.6 + 0.4 * u_color_speed) + u_color_offset), 1.0);
}
''',
    uniforms={
    "iterations": {"glsl": "float", "min": 30.0, "max": 500.0, "default": 100, "description": "max iterations"},
    "color_speed": {"glsl": "float", "min": 0.5, "max": 8.0, "default": 2.0, "description": "palette color speed"},
    "color_offset": {"glsl": "float", "min": 0.0, "max": 6.28, "default": 0.0, "description": "palette color offset"}
}
    )