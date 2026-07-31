"""newton_gpu — GPU shader registration (dynamically loaded by core/shaders.py)."""

from ._registry import _register
from ._helpers import _FRACTAL_HELPERS



_register("newton_gpu", "Newton fractal basins (client-GPU twin of node 52)", "procedural",
          _FRACTAL_HELPERS + '''
void main() {
    vec2 uv = (gl_FragCoord.xy - 0.5 * u_resolution) / min(u_resolution.x, u_resolution.y);
    vec2 z = uv * 2.2;  // fixed full view (node 52 has no zoom param)
    const float MAXI = 200.0;
    float n = 0.0;
    for (int i = 0; i < 200; i++) {
        // Newton for z^3 - 1: z - (z^3 - 1) / (3 z^2)
        vec2 z2 = vec2(z.x*z.x - z.y*z.y, 2.0*z.x*z.y);
        vec2 z3 = vec2(z2.x*z.x - z2.y*z.y, 2.0*z2.x*z.y);
        vec2 f = z3 - vec2(1.0, 0.0);
        vec2 dz = 3.0 * z2;
        float denom = dz.x*dz.x + dz.y*dz.y + 1e-8;
        vec2 step = vec2(f.x*dz.x + f.y*dz.y, f.y*dz.x - f.x*dz.y) / denom;
        z -= step;
        n += 1.0;
        if (dot(step, step) < 1e-6 || n >= u_max_iter) break;
    }
    // Color by nearest of the 3 cube roots of unity (angle quantization).
    float ang = atan(z.y, z.x);
    float root = floor((ang + 3.14159) / (2.0 * 3.14159 / 3.0));
    float t = mod(root / 3.0 + u_color_offset + 0.15 * n / MAXI, 1.0);
    f_color = vec4(fractal_palette(t * (0.6 + 0.4 * u_color_speed)), 1.0);
}
''',
    uniforms={
    "max_iter": {"glsl": "float", "min": 10.0, "max": 200.0, "default": 50, "description": "max Newton iterations"},
    "color_speed": {"glsl": "float", "min": 0.5, "max": 8.0, "default": 2.0, "description": "palette color speed"},
    "color_offset": {"glsl": "float", "min": 0.0, "max": 6.28, "default": 0.0, "description": "palette color offset"}
}
    )