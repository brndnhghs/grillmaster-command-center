"""oscillon_display — GPU shader registration (dynamically loaded by core/shaders.py)."""

from ._registry import _register



_register("oscillon_display",
          "Oscillon Resonance display: displacement sigmoid (grayscale, matches _render_displacement)",
          "procedural", '''
void main() {
    float u = texture(u_texture, v_uv).r;
    float sig = tanh(clamp(u, -4.0, 4.0) * 2.5);
    float g = sig * 0.5 + 0.5;
    f_color = vec4(vec3(g), 1.0);
}
''')