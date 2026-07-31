"""menger_typed — GPU shader registration (dynamically loaded by core/shaders.py)."""

from ._registry import _register
from ._helpers import _INFERNO_GPU



_register("menger_typed", "Menger carpet / Sierpinski-carpet recursive subdivision (typed, node 324)",
          "procedural", _INFERNO_GPU + '''void main() {
    // Animated Sierpinski (Menger) carpet: recursive 3x3 subdivision that
    // removes the centre ninth at every level. The plane spins and the scale
    // breathes with time so the static fractal reads as alive in the live
    // preview. Surviving cells are coloured by recursion depth through the
    // inferno map; removed cells show the background colour.
    vec2 p = v_uv - 0.5;
    p = rot(u_time * u_spin) * p;
    float sc = u_scale * (0.85 + 0.15 * sin(u_time * u_pulse));
    vec2 uv = fract(p * sc + 0.5);          // wrap into [0,1)
    bool on = true;
    vec2 q = uv;
    float lvl = 0.0;
    for (int i = 0; i < 6; i++) {
        vec2 cell = floor(q * 3.0);
        if (cell.x == 1.0 && cell.y == 1.0) { on = false; break; }
        q = fract(q * 3.0);
        lvl += 1.0;
    }
    vec3 col;
    if (on) {
        float h = fract(lvl * 0.16 + u_time * 0.04);
        col = inferno(clamp(0.25 + 0.55 * h + 0.25 * q.x, 0.0, 1.0));
    } else {
        col = u_bg;
    }
    f_color = vec4(col, 1.0);
}
''', uniforms={
    "scale": {"glsl": "float", "min": 3.0, "max": 24.0, "default": 8.0,
              "description": "feature density (carpets across view)"},
    "spin":  {"glsl": "float", "min": 0.0, "max": 3.0, "default": 0.4,
              "description": "in-plane rotation speed"},
    "pulse": {"glsl": "float", "min": 0.0, "max": 3.0, "default": 0.8,
              "description": "scale breathing speed"},
    "bg":    {"glsl": "color", "default": "#0a0a18",
              "description": "hole background color"},
})