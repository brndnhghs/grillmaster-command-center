"""test_pattern_gpu — GPU analog of the Test Node's image-pattern generator.

Renders the same diagnostic test patterns the CPU Test Node (`__test__`)
produces, but fully on the GPU as a single-pass procedural fragment shader:
color bars, checkerboard, horizontal/vertical gradients, white/black,
value noise, and an RGB color ramp — all driven by a `pattern` choice uniform
plus named typed controls (tile count, colors, animation).

This mirrors the CPU node's default `anim_mode="none"` (static) while also
supporting gentle time-scrolling gradients/noise via `u_time` so temporal
continuity can be exercised on the GPU path too.
"""
from ._registry import _register


_register("test_pattern_gpu", "GPU Test Pattern (color bars / checker / gradient / noise)",
          "procedural", '''
vec3 _tp_white = vec3(1.0);
vec3 _tp_black = vec3(0.0);

vec3 _tp_color_bars(vec2 uv) {
    // 8 SMPTE-style bars: white, yellow, cyan, green, magenta, red, blue, black
    vec3[8] bars = vec3[8](
        vec3(1.0, 1.0, 1.0), vec3(1.0, 1.0, 0.0), vec3(0.0, 1.0, 1.0), vec3(0.0, 1.0, 0.0),
        vec3(1.0, 0.0, 1.0), vec3(1.0, 0.0, 0.0), vec3(0.0, 0.0, 1.0), vec3(0.0, 0.0, 0.0)
    );
    int i = int(floor(uv.x * 8.0));
    return bars[clamp(i, 0, 7)];
}

vec3 _tp_checker(vec2 uv) {
    float cells = max(u_cells, 1.0);
    float chk = mod(floor(uv.x * cells) + floor(uv.y * cells), 2.0);
    return mix(u_color_a, u_color_b, chk);
}

void main() {
    vec2 uv = v_uv;
    float t = u_time * u_speed;

    vec3 col = _tp_black;

    if (u_pattern == 0) {                       // color_bars
        col = _tp_color_bars(uv);
    } else if (u_pattern == 1) {                // checkerboard
        col = _tp_checker(uv);
    } else if (u_pattern == 2) {                // gradient_h
        float g = fract(uv.x + t * 0.1 * u_animate);
        col = mix(u_color_a, u_color_b, g);
    } else if (u_pattern == 3) {                // gradient_v
        float g = fract(uv.y + t * 0.1 * u_animate);
        col = mix(u_color_a, u_color_b, g);
    } else if (u_pattern == 4) {                // white
        col = _tp_white;
    } else if (u_pattern == 5) {                // black
        col = _tp_black;
    } else if (u_pattern == 6) {                // noise
        float n = hash21(uv * u_cells + vec2(t));
        col = vec3(n);
    } else if (u_pattern == 7) {                // color_ramp
        col = vec3(uv.x, uv.x * 0.667, uv.x * 0.334);
    }

    f_color = vec4(col, 1.0);
}
''', uniforms={
    "pattern": {"glsl": "choice",
                "choices": ["color_bars", "checkerboard", "gradient_h", "gradient_v",
                            "white", "black", "noise", "color_ramp"],
                "default": "color_bars", "description": "test image pattern"},
    "cells":   {"glsl": "float", "min": 1.0, "max": 64.0, "default": 8.0,
                "description": "checker/noise tile count"},
    "color_a": {"glsl": "color", "default": "#101018", "description": "pattern color A"},
    "color_b": {"glsl": "color", "default": "#e8e4d8", "description": "pattern color B"},
    "animate": {"glsl": "float", "min": 0.0, "max": 1.0, "default": 0.0,
                "description": "scroll gradients/noise over time"},
    "speed":   {"glsl": "float", "min": 0.0, "max": 5.0, "default": 0.5,
                "description": "animation speed multiplier"},
})
