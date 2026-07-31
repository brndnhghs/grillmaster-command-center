"""acpm_display — GPU shader registration (dynamically loaded by core/shaders.py)."""

from ._registry import _register



_register("acpm_display",
          "AC+PM display: map signed field .r to grayscale",
          "procedural", '''
void main() {
    float c = clamp(texture(u_texture, v_uv).r, -1.0, 1.0);
    float g = c * 0.5 + 0.5;
    f_color = vec4(vec3(g), 1.0);
}
''')