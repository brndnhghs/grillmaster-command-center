"""faraday_display — GPU shader registration (dynamically loaded by core/shaders.py)."""

from ._registry import _register



_register("faraday_display",
          "Faraday Waves display: height field sigmoid (grayscale, matches _render_faraday)",
          "procedural", '''
void main() {
    float h = texture(u_texture, v_uv).r;
    float sig = tanh(clamp(h, -4.0, 4.0) * 2.5);
    float g = sig * 0.5 + 0.5;
    f_color = vec4(vec3(g), 1.0);
}
''')