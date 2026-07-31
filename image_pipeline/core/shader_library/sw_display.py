"""sw_display — GPU shader registration (dynamically loaded by core/shaders.py)."""

from ._registry import _register



_register("sw_display",
          "Shallow Water display: height anomaly -> plasma palette (matches _render_height)",
          "procedural", '''
void main() {
    float h = texture(u_texture, v_uv).r;
    float t = clamp(h * 0.5 + 0.5, 0.0, 1.0);
    vec3 col = mix(vec3(0.05, 0.10, 0.35), vec3(0.10, 0.55, 0.65), t);
    col = mix(col, vec3(0.85, 0.95, 1.0), smoothstep(0.6, 1.0, t));
    f_color = vec4(col, 1.0);
}
''')