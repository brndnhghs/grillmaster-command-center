"""forest_fire_step — GPU shader registration (dynamically loaded by core/shaders.py)."""

from ._registry import _register



_register("forest_fire_step",
          "Forest Fire one step: growth, neighbour/lightning ignition, fire aging (node 96 twin)",
          "procedural", '''
void main() {
    vec2 texel = 1.0 / u_resolution;
    vec4 s = texture(u_texture, v_uv);
    float p = clamp(u_params.x, 0.001, 0.05);
    float f = clamp(u_params.y, 0.00001, 0.001);
    float state = s.r;
    float age = s.g;
    float rng = fract(s.b * 1.731 + 0.211);

    // count burning neighbours (Moore)
    float burn = 0.0;
    for (int y = -1; y <= 1; y++) {
        for (int x = -1; x <= 1; x++) {
            if (x == 0 && y == 0) continue;
            float sn = texture(u_texture, v_uv + vec2(float(x), float(y)) * texel).r;
            if (sn > 1.5) burn += 1.0;   // state == 2 (burning)
        }
    }

    float next_state = state;
    float next_age = age;

    if (state > 1.5) {
        // currently burning: age it down; age 0 -> empty
        if (age <= 0.5) { next_state = 0.0; next_age = 0.0; }
        else { next_age = age - 1.0; }
    } else if (state > 0.5) {
        // tree: ignite if neighbour burning or lightning
        float lightning = rng < f ? 1.0 : 0.0;
        if (burn > 0.5 || lightning > 0.5) { next_state = 2.0; next_age = 3.0; }
    } else {
        // empty: grow a tree
        if (rng < p) { next_state = 1.0; }
    }
    f_color = vec4(next_state, next_age, rng, 1.0);
}
''')