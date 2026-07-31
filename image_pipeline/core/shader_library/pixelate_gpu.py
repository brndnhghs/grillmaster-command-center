"""pixelate_gpu — GPU shader registration (dynamically loaded by core/shaders.py)."""

from ._registry import _register



_register("pixelate_gpu", "Pixelate / mosaic of the input (typed)",
          "filter", '''
void main() {
    float cells = max(float(u_cells), 2.0);
    vec2 grid = vec2(cells, cells * u_resolution.y / u_resolution.x);
    vec2 uv = (floor(v_uv * grid) + 0.5) / grid;
    vec3 src = texture(u_texture, uv).rgb;
    if (u_levels > 1.5) {
        float lv = float(u_levels);
        src = floor(src * lv + 0.5) / lv;
    }
    f_color = vec4(src, 1.0);
}
''', uniforms={
    "cells":  {"glsl": "float", "min": 4.0, "max": 200.0, "default": 48.0,
               "description": "mosaic cell count (x)"},
    "levels": {"glsl": "int", "min": 1, "max": 32, "default": 1,
               "description": "color quantize levels (1=off)"},
})