"""frac_rd_display — GPU shader registration (dynamically loaded by core/shaders.py)."""

from ._registry import _register



_register("frac_rd_display",
          "Fractional-RD display: activator V → fire colormap, gamma (node 163)",
          "procedural", '''
void main() {
    float V = clamp(texture(u_texture, v_uv).g, 0.0, 1.0);
    float v = sqrt(V);                       // gamma stretch (matches CPU V**0.5)
    float r = clamp(4.0 * v,        0.0, 1.0);
    float g = clamp(4.0 * v - 1.0,  0.0, 1.0);
    float b = clamp(4.0 * v - 3.0,  0.0, 1.0);
    f_color = vec4(r, g, b, 1.0);            // dark → red → orange → yellow → white
}
''')