"""sine_gordon_display — GPU shader registration (dynamically loaded by core/shaders.py)."""

from ._registry import _register



_register("sine_gordon_display",
          "Sine-Gordon display: displacement -> plasma-like palette",
          "procedural", '''
void main() {
    float u = texture(u_texture, v_uv).r;
    float t = clamp(u / 6.2831853, 0.0, 1.0);
    vec3 col = mix(vec3(0.05, 0.05, 0.20), vec3(0.90, 0.40, 0.10), t);
    col = mix(col, vec3(1.0, 0.95, 0.60), smoothstep(0.6, 1.0, t));
    f_color = vec4(col, 1.0);
}
''')