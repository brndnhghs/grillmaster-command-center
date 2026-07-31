"""spd125_display — GPU shader registration (dynamically loaded by core/shaders.py)."""

from ._registry import _register



_register("spd125_display",
          "SPD #153 display: diverging amber(defect)→blue(coop) by strategy, brightness by payoff",
          "procedural", '''
void main() {
    vec4 s = texture(u_texture, v_uv);
    float strat = s.r;
    vec3 defect_col = vec3(0.86, 0.39, 0.16);  // amber
    vec3 coop_col   = vec3(0.235, 0.55, 0.86); // blue
    vec3 col = mix(defect_col, coop_col, strat);
    float bright = 0.65 + 0.35 * clamp(s.g, 0.0, 1.0);
    f_color = vec4(col * bright, 1.0);
}
''')