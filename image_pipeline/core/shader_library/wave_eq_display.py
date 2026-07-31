"""wave_eq_display — GPU shader registration (dynamically loaded by core/shaders.py)."""

from ._registry import _register



_register("wave_eq_display",
          "Wave Equation display: bipolar displacement -> plasma-like palette",
          "procedural", '''
void main() {
    float u = texture(u_texture, v_uv).r;
    float t = clamp(u * 0.5 + 0.5, 0.0, 1.0);
    vec3 col = mix(vec3(0.10, 0.10, 0.45), vec3(0.95, 0.40, 0.10), t);
    col = mix(col, vec3(1.0, 1.0, 0.65), smoothstep(0.62, 1.0, t));
    f_color = vec4(col, 1.0);
}
''')