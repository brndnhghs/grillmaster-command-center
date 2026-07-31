"""colony_display — GPU shader registration (dynamically loaded by core/shaders.py)."""

from ._registry import _register



_register("colony_display",
          "Bacterial colony display: colony white on dark nutrient field",
          "procedural", '''
void main() {
    vec4 s = texture(u_texture, v_uv);
    float C = clamp(s.g, 0.0, 1.0);
    float N = clamp(s.r, 0.0, 1.0);
    vec3 col = mix(vec3(0.05,0.07,0.10), vec3(0.9,0.95,0.85), C);
    col *= (0.4 + 0.6*N);
    f_color = vec4(col, 1.0);
}
''')