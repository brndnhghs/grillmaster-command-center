"""cml95_display — GPU shader registration (dynamically loaded by core/shaders.py)."""

from ._registry import _register



_register("cml95_display",
          "Coupled logistic display (node 95): trail -> magma-inspired colormap",
          "procedural", '''
void main() {
    float t = clamp(texture(u_texture, v_uv).g, 0.0, 1.0);
    // Piecewise magma-inspired ramp matching _COLORMAP_256 (dark→purple→orange→gold)
    vec3 c0 = vec3(0.016, 0.016, 0.063);
    vec3 c1 = vec3(0.314, 0.0,   0.314);
    vec3 c2 = vec3(0.706, 0.157, 0.471);
    vec3 c3 = vec3(0.941, 0.471, 0.157);
    vec3 c4 = vec3(1.0,   0.863, 0.235);
    vec3 col;
    if (t < 0.25)      col = mix(c0, c1, t / 0.25);
    else if (t < 0.50) col = mix(c1, c2, (t - 0.25) / 0.25);
    else if (t < 0.75) col = mix(c2, c3, (t - 0.50) / 0.25);
    else               col = mix(c3, c4, (t - 0.75) / 0.25);
    f_color = vec4(col, 1.0);
}
''')