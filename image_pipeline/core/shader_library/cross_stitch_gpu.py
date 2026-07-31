"""cross_stitch_gpu — GPU shader registration (dynamically loaded by core/shaders.py)."""

from ._registry import _register
from ._helpers import _filter_typed



# 63 Cross Stitch — grid of stitches on a fabric backdrop (GPU live twin)
_register("cross_stitch_gpu", "GPU cross-stitch embroidery", "filter", _filter_typed('''
    float gstep = max(4.0, 32.0 - u_thread_step * 28.0);   // thread_step
    float lw = 1.0 + u_line_width * 6.0;                  // line_width
    vec2 cell = floor(uv * u_resolution / gstep);
    vec2 cell_uv = (cell + 0.5) * gstep / u_resolution;
    vec3 src = texture(u_texture, cell_uv).rgb;
    // fabric base
    vec3 fabric = vec3(0.95, 0.92, 0.88);
    vec2 q = fract(uv * u_resolution / gstep) - 0.5;
    // cross: two diagonal strokes
    float d1 = abs(q.x + q.y);
    float d2 = abs(q.x - q.y);
    float stroke = min(d1, d2);
    float stitch = 1.0 - smoothstep(lw * 0.35, lw * 0.45, stroke);
    vec3 col = mix(fabric, src, stitch);
    f_color = vec4(col, 1.0);
'''), uniforms={
        "thread_step": {"glsl": "float", "min": 0.0, "max": 1.0, "default": 0.5, "description": "stitch grid step"},
        "line_width": {"glsl": "float", "min": 0.0, "max": 1.0, "default": 0.5, "description": "thread width"},
    })