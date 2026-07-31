"""pfc_display — GPU shader registration (dynamically loaded by core/shaders.py)."""

from ._registry import _register



_register("pfc_display",
          "Phase Field Crystal display: signed psi -> tanh grayscale (matches PFC render)",
          "procedural", '''
void main() {
    float psi = texture(u_texture, v_uv).r;
    float g = clamp((tanh(psi * 1.5) + 1.0) * 0.5, 0.0, 1.0);
    f_color = vec4(vec3(g), 1.0);
}
''')