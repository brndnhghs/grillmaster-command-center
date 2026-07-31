"""lv3_display — GPU shader registration (dynamically loaded by core/shaders.py)."""

from ._registry import _register



_register("lv3_display",
          "3-species LV display: U green, V red, W blue (cyclic food web)",
          "procedural", '''
void main() {
    vec4 s = texture(u_texture, v_uv);
    vec3 col = vec3(clamp(s.g,0.0,1.0), clamp(s.r,0.0,1.0), clamp(s.b,0.0,1.0));
    f_color = vec4(col, 1.0);
}
''')