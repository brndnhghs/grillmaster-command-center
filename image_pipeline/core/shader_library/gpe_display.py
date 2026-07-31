"""gpe_display — GPU shader registration (dynamically loaded by core/shaders.py)."""

from ._registry import _register



_register("gpe_display",
          "GPE display: phase -> hue, amplitude -> brightness (phase render style)",
          "procedural", '''
void main() {
    vec4 s = texture(u_texture, v_uv);
    float a = s.r, b = s.g;
    float amp = clamp(sqrt(a * a + b * b), 0.0, 1.0);
    float phase = atan(b, a);              // -pi..pi
    float hue = (phase + 3.14159265) / 6.28318530;
    vec3 col = clamp(abs(fract(hue + vec3(0.0, 0.6667, 0.3333)) * 6.0 - 3.0) - 1.0, 0.0, 1.0);
    // density-weighted value: bright at moderate density, dark at vortex cores
    f_color = vec4(col * (0.35 + 0.65 * amp), 1.0);
}
''')