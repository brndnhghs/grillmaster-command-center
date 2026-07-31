"""spd154_display — GPU shader registration (dynamically loaded by core/shaders.py)."""

from ._registry import _register



_register("spd154_display",
          "CSPD #154 display: grayscale cooperation-probability field (matches CPU render)",
          "procedural", '''
void main() {
    float v = clamp(texture(u_texture, v_uv).g, 0.0, 1.0);
    f_color = vec4(vec3(v), 1.0);
}
''')