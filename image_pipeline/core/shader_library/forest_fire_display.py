"""forest_fire_display — GPU shader registration (dynamically loaded by core/shaders.py)."""

from ._registry import _register



_register("forest_fire_display",
          "Forest Fire display: earth/tree/fire_age colormap (node 96 twin)",
          "procedural", '''
void main() {
    vec4 s = texture(u_texture, v_uv);
    float state = s.r;
    float age = s.g;
    vec3 col = vec3(0.16, 0.10, 0.06);          // dark brown earth
    if (state > 0.5 && state < 1.5) {
        col = vec3(0.12, 0.55, 0.20);           // green tree
    } else if (state > 1.5) {
        // fire_age 1-3: dark red -> orange -> bright orange
        vec3 fire = mix(vec3(0.63, 0.12, 0.04), vec3(1.0, 0.63, 0.08),
                        clamp(age / 3.0, 0.0, 1.0));
        col = fire;
    }
    f_color = vec4(col, 1.0);
}
''')