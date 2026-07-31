"""sierpinski_gpu — GPU shader registration (dynamically loaded by core/shaders.py)."""

from ._registry import _register
from ._helpers import _FRACTAL_HELPERS



_register("sierpinski_gpu", "Sierpinski carpet (client-GPU twin of node 67)", "procedural",
          _FRACTAL_HELPERS + '''
void main() {
    vec2 uv = v_uv;
    // p1 = depth (subdivisions), p2 = color_shift, p3/p4 unused (reserved).
    float depth = clamp(u_depth, 1.0, 7.0);
    // Tiling coordinates in [0,1] space.
    vec2 p = uv;
    float hole = 0.0;
    for (float i = 0.0; i < 7.0; i += 1.0) {
        if (i >= depth) break;
        // Carpet rule: remove central third at each scale.
        vec2 cell = floor(p * 3.0);
        if (cell.x == 1.0 && cell.y == 1.0) { hole = 1.0; break; }
        p = fract(p * 3.0);
    }
    float t = fract(0.15 * (depth) + 0.5 + 0.3 * uv.x + 0.2 * uv.y);
    vec3 col = (hole > 0.5) ? vec3(0.04) : fractal_palette(t);
    f_color = vec4(col, 1.0);
}
''',
    uniforms={
    "depth": {"glsl": "float", "min": 1.0, "max": 7.0, "default": 4.0, "description": "subdivision depth"}
}
    )