"""mandelbrot_gpu — GPU shader registration (dynamically loaded by core/shaders.py)."""

from ._registry import _register
from ._helpers import _FRACTAL_HELPERS



_register("mandelbrot_gpu", "Mandelbrot set (client-GPU twin of node 33)", "procedural",
          _FRACTAL_HELPERS + '''
void main() {
    vec2 uv = (gl_FragCoord.xy - 0.5 * u_resolution) / min(u_resolution.x, u_resolution.y);
    vec2 ctr = vec2(u_center_x, u_center_y);
    vec2 c = ctr + uv * u_zoom;
    vec2 z = vec2(0.0);
    float n = 0.0;
    float last2 = 0.0;
    const float MAXI = 200.0;
    float bail2 = u_escape_radius * u_escape_radius;
    for (int i = 0; i < 200; i++) {
        z = vec2(z.x*z.x - z.y*z.y, 2.0*z.x*z.y) + c;
        last2 = dot(z, z);
        if (last2 > bail2 || n >= u_iterations) break;
        n += 1.0;
    }
    float t = (n >= u_iterations - 0.5) ? 0.0 : smooth_iter(n, last2, u_iterations);
    f_color = vec4(fractal_palette(t + u_color_shift), 1.0);
}
''',
    uniforms={
    "zoom": {"glsl": "float", "min": 0.5, "max": 100000.0, "default": 1.0, "description": "zoom (1 = full view)"},
    "center_x": {"glsl": "float", "min": -2.5, "max": 2.5, "default": -0.5, "description": "center x"},
    "center_y": {"glsl": "float", "min": -2.0, "max": 2.0, "default": 0.0, "description": "center y"},
    "iterations": {"glsl": "float", "min": 50.0, "max": 2000.0, "default": 200, "description": "max iterations"},
    "escape_radius": {"glsl": "float", "min": 1.5, "max": 100.0, "default": 4.0, "description": "escape bailout radius"},
    "color_shift": {"glsl": "float", "min": 0.0, "max": 6.28, "default": 0.0, "description": "palette color offset"}
}
    )