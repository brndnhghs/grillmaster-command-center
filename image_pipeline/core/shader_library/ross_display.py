"""ross_display — GPU shader registration (dynamically loaded by core/shaders.py)."""

from ._registry import _register



_register("ross_display",
          "Rössler array display: x/y/z -> HSV-ish composite (matches render_style=composite)",
          "procedural", '''
void main() {
    vec3 v = texture(u_texture, v_uv).rgb;
    float x = v.r, y = v.g, z = v.b;
    float xr = clamp((x + 12.0) / 24.0, 0.0, 1.0);
    float yr = clamp((y + 12.0) / 24.0, 0.0, 1.0);
    float zr = clamp(z / 30.0, 0.0, 1.0);
    vec3 col = vec3(xr, yr, 0.3 + 0.7 * zr);
    f_color = vec4(col, 1.0);
}
''')