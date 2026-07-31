"""rd_display_u — GPU shader registration (dynamically loaded by core/shaders.py)."""

from ._registry import _register



_register("rd_display_u",
          "RD display: U activator -> grayscale (sqrt stretch)",
          "procedural", '''
void main() {
    float U = clamp(texture(u_texture, v_uv).r, 0.0, 1.0);
    f_color = vec4(vec3(sqrt(U)), 1.0);
}
''')