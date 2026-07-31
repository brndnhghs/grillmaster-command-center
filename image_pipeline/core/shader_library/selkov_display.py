"""selkov_display — GPU shader registration (dynamically loaded by core/shaders.py)."""

from ._registry import _register



_register("selkov_display",
          "Sel'kov display: U substrate → heat ramp (matches _render_substrate)",
          "procedural", '''
void main() {
    float U = texture(u_texture, v_uv).r;
    float f = clamp(U / 1.5, 0.0, 1.0);
    f = pow(f, 0.6);                       // gamma lift
    vec3 col = vec3(f);                    // grayscale heat
    f_color = vec4(col, 1.0);
}
''')