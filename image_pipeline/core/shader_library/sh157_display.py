"""sh157_display — GPU shader registration (dynamically loaded by core/shaders.py)."""

from ._registry import _register



_register("sh157_display",
          "Swift-Hohenberg (157) display: mean-centered tanh grayscale (matches _render)",
          "procedural", '''
void main() {
    float u = texture(u_texture, v_uv).r;
    float g = clamp((tanh(u * 1.5) + 1.0) * 0.5, 0.0, 1.0);
    f_color = vec4(vec3(g), 1.0);
}
''')