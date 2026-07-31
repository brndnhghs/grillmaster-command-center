"""thue_morse_typed — GPU shader registration (dynamically loaded by core/shaders.py)."""

from ._registry import _register



# 303 — Thue-Morse recursive binary fractal: cell parity = popcount(x) XOR
# popcount(y) over a 2^depth grid. Static structure; animation sweeps the
# two-colour palette so the node still responds to time.
_register("thue_morse_typed", "Thue-Morse recursive binary fractal (typed, node 303)",
          "procedural", '''void main() {
    float depth = max(u_depth, 1.0);
    float scale = exp2(depth);
    vec2 cell = floor(v_uv * scale);
    // Popcount parity via floating extraction (no integer bit ops — portable).
    float ix = cell.x + 1.0;
    float iy = cell.y + 1.0;
    float cnt = 0.0;
    for (int b = 0; b < 9; b++) {
        float fb = float(b);
        cnt += mod(floor(ix / exp2(fb)), 2.0);
        cnt += mod(floor(iy / exp2(fb)), 2.0);
    }
    float v = mod(cnt, 2.0);
    // Animation: a global brightness pulse that shifts the two colours over time
    // (independent of the binary value, so it is never a no-op at t=0 vs t=pi).
    float pulse = 0.5 + 0.5 * sin(u_time * u_speed * 0.6);
    vec3 colA = u_color_a * (0.6 + 0.4 * pulse);
    vec3 colB = u_color_b * (1.4 - 0.4 * pulse);
    vec3 col = mix(colA, colB, step(0.5, v));
    f_color = vec4(col, 1.0);
}
''', uniforms={
    "depth":   {"glsl": "float", "min": 1.0, "max": 8.0, "default": 5.0, "description": "recursion depth (2^depth cells)"},
    "speed":   {"glsl": "float", "min": 0.0, "max": 3.0, "default": 0.6, "description": "palette animation speed"},
    "color_a": {"glsl": "color", "default": "#101830", "description": "even parity color"},
    "color_b": {"glsl": "color", "default": "#f0603c", "description": "odd parity color"},
})