"""rd_display_composite — GPU shader registration (dynamically loaded by core/shaders.py)."""

from ._registry import _register



_register("rd_display_composite",
          "RD display: U in green, V in red (Lotka-Volterra prey/predator look)",
          "procedural", '''
void main() {
    vec4 s = texture(u_texture, v_uv);
    float U = clamp(s.r, 0.0, 1.0); float V = clamp(s.g, 0.0, 1.0);
    vec3 col = vec3(V, U * 0.9 + V * 0.1, U * 0.2);
    f_color = vec4(col, 1.0);
}
''')