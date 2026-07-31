"""cyclic_ca_step — GPU shader registration (dynamically loaded by core/shaders.py)."""

from ._registry import _register



_register("cyclic_ca_step",
          "Cyclic CA one step: convert to predator state when >= threshold neighbours match (node 87 twin)",
          "procedural", '''
void main() {
    vec2 texel = 1.0 / u_resolution;
    vec4 s = texture(u_texture, v_uv);
    float ns = clamp(floor(u_params.x + 0.5), 3.0, 8.0);
    float thr = clamp(floor(u_params.y + 0.5), 1.0, 5.0);
    float my = floor(s.r * ns + 0.5);
    float pred = mod(my + 1.0, ns);          // predator state index
    float predF = (pred + 0.5) / ns;         // predator state in [0,1)
    float cnt = 0.0;
    for (int y = -1; y <= 1; y++) {
        for (int x = -1; x <= 1; x++) {
            if (x == 0 && y == 0) continue;
            float sn = texture(u_texture, v_uv + vec2(float(x), float(y)) * texel).r;
            float nidx = floor(sn * ns + 0.5);
            if (abs(nidx - pred) < 0.5) cnt += 1.0;
        }
    }
    float alive_pred = cnt >= thr ? 1.0 : 0.0;
    float next = alive_pred > 0.5 ? predF : s.r;
    // advance per-cell RNG carry
    float rng = fract(s.b * 1.4567 + 0.137);
    f_color = vec4(next, 0.0, rng, 1.0);
}
''')