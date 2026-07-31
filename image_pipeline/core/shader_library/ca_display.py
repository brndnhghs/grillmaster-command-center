"""ca_display — GPU shader registration (dynamically loaded by core/shaders.py)."""

from ._registry import _register


_register("ca_display",
          "Game of Life display: alive=white, age tints toward warm",
          "procedural", '''
void main() {
    vec4 s = texture(u_texture, v_uv);
    float alive = s.r;
    float age = s.g;
    vec3 col = mix(vec3(0.02, 0.02, 0.05), vec3(0.9, 0.95, 1.0), alive);
    col = mix(col, vec3(1.0, 0.6, 0.2), alive * clamp(age / 12.0, 0.0, 1.0));
    f_color = vec4(col, 1.0);
}
''')