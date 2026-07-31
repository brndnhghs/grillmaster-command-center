"""sh128_display — GPU shader registration (dynamically loaded by core/shaders.py)."""

from ._registry import _register



_register("sh128_display",
          "Swift-Hohenberg (128) display: u in [-2,2] -> grayscale (matches _render_field)",
          "procedural", '''
void main() {
    float u = texture(u_texture, v_uv).r;
    float g = clamp((u + 2.0) / 4.0, 0.0, 1.0);
    f_color = vec4(vec3(g), 1.0);
}
''')