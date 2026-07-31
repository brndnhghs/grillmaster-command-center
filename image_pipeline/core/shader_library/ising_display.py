"""ising_display — GPU shader registration (dynamically loaded by core/shaders.py)."""

from ._registry import _register



_register("ising_display",
          "Ising display: blue-white-red diverging map of spin (+1 white-ish, -1 blue)",
          "procedural", '''
void main() {
    vec4 s = texture(u_texture, v_uv);
    float spin = s.r > 0.5 ? 1.0 : -1.0;
    // diverging: -1 -> blue, +1 -> red, with soft mid-grey
    vec3 col = mix(vec3(0.20, 0.35, 0.85), vec3(0.90, 0.30, 0.25), (spin + 1.0) * 0.5);
    f_color = vec4(col, 1.0);
}
''')