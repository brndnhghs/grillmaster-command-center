"""kuramoto_display — GPU shader registration (dynamically loaded by core/shaders.py)."""

from ._registry import _register



_register("kuramoto_display",
          "Kuramoto display: phase θ → IQ rainbow palette (matches _render_phase)",
          "procedural", '''
void main() {
    float theta = texture(u_texture, v_uv).r;
    float t = theta / 6.2831853;
    vec3 col = 0.5 + 0.5 * cos(6.2831853 * (t + vec3(0.0, 0.3333333, 0.6666667)));
    f_color = vec4(col, 1.0);
}
''')