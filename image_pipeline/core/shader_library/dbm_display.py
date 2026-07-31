"""dbm_display — GPU shader registration (dynamically loaded by core/shaders.py)."""

from ._registry import _register



_register("dbm_display",
          "Dielectric Breakdown display: hot tips bright blue-white, cooled trunk dimmer (node 106 twin)",
          "procedural", '''
void main() {
    vec4 s = texture(u_texture, v_uv);
    float occ = s.g;
    float temp = s.b;
    if (occ < -0.5) { f_color = vec4(0.0, 0.0, 0.0, 1.0); return; }   // far-field
    if (occ < 0.5)  { f_color = vec4(0.02, 0.03, 0.06, 1.0); return; } // empty space
    float t = clamp(temp, 0.0, 1.0);
    vec3 hot  = vec3(0.75, 0.85, 1.0);
    vec3 coolc = vec3(0.15, 0.35, 0.9);
    f_color = vec4(mix(coolc, hot, t), 1.0);
}
''')