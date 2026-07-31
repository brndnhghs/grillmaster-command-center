"""cahn_hilliard_display — GPU shader registration (dynamically loaded by core/shaders.py)."""

from ._registry import _register
from ._helpers import _INFERNO



_register("cahn_hilliard_display",
          "Cahn-Hilliard display: φ (.r) → inferno colormap (phase look)",
          "procedural", _INFERNO + '''
void main() {
    float phi = texture(u_texture, v_uv).r;
    float t = clamp(phi * 0.5 + 0.5, 0.0, 1.0);
    vec3 col = inferno(t);
    f_color = vec4(col, 1.0);
}
''')