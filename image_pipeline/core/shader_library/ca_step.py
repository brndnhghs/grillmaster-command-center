"""ca_step — GPU shader registration (dynamically loaded by core/shaders.py)."""

from ._registry import _register



_register("ca_step",
          "Game of Life one step: 8-neighbor toroidal count, Conway birth/survival",
          "procedural", '''
void main() {
    vec2 texel = 1.0 / u_resolution;
    float c = texture(u_texture, v_uv).r;
    float n = 0.0;
    for (int y = -1; y <= 1; y++) {
        for (int x = -1; x <= 1; x++) {
            if (x == 0 && y == 0) continue;
            n += texture(u_texture, v_uv + vec2(float(x), float(y)) * texel).r;
        }
    }
    float alive = (n >= 2.5 && n <= 3.5) ? 1.0 : 0.0;  // survive on 2/3
    alive = (c < 0.5 && n > 2.5 && n < 3.5) ? 1.0 : alive;  // birth on 3
    float age = c > 0.5 ? texture(u_texture, v_uv).g + 1.0 : 0.0;
    f_color = vec4(alive, age, 0.0, 1.0);
}
''')