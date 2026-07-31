"""cml142_display — GPU shader registration (dynamically loaded by core/shaders.py)."""

from ._registry import _register



_register("cml142_display",
          "Coupled logistic display (node 142): trail -> grayscale (matches CML render)",
          "procedural", '''
void main() {
    float g = clamp(texture(u_texture, v_uv).g, 0.0, 1.0);
    f_color = vec4(vec3(g), 1.0);
}
''')