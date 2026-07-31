"""cyclic_ca_display — GPU shader registration (dynamically loaded by core/shaders.py)."""

from ._registry import _register



_register("cyclic_ca_display",
          "Cyclic CA display: 8-state cyclic palette (red/green/blue/gold/cyan/magenta/orange/silver)",
          "procedural", '''
void main() {
    vec4 s = texture(u_texture, v_uv);
    float ns = clamp(floor(u_params.x + 0.5), 3.0, 8.0);
    float st = floor(s.r * ns + 0.5);
    // 8 distinct hues around the wheel
    float a = st / ns * 6.2831853;
    vec3 col = 0.5 + 0.5 * cos(a + vec3(0.0, 2.094, 4.188));
    col = mix(vec3(0.05, 0.05, 0.07), col, 0.9);
    f_color = vec4(col, 1.0);
}
''')