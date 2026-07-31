"""fpu_display — GPU shader registration (dynamically loaded by core/shaders.py)."""

from ._registry import _register



_register("fpu_display",
          "FPU display: |displacement| -> fire palette (matches fpu_lattice render)",
          "procedural", '''
void main() {
    float a = abs(texture(u_texture, v_uv).r);
    float t = clamp(a * 1.2, 0.0, 1.0);
    vec3 col = vec3(0.0, 0.0, 0.15);
    col = mix(col, vec3(0.85, 0.25, 0.05), smoothstep(0.0, 0.5, t));
    col = mix(col, vec3(1.0, 0.95, 0.55), smoothstep(0.5, 1.0, t));
    f_color = vec4(col, 1.0);
}
''')