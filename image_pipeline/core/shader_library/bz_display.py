"""bz_display — GPU shader registration (dynamically loaded by core/shaders.py)."""

from ._registry import _register



_register("bz_display",
          "BZ display: V activator -> grayscale",
          "procedural", '''
void main() {
    float V = clamp(texture(u_texture, v_uv).g, 0.0, 1.0);
    f_color = vec4(V, V, V, 1.0);
}
''')