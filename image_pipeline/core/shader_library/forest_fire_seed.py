"""forest_fire_seed — GPU shader registration (dynamically loaded by core/shaders.py)."""

from ._registry import _register



# ── Node 96: Drossel-Schwabl Forest Fire ──
# 3-state CA: .r encodes state (0 empty, 1 tree, 2 burning); .g = fire_age (0-3);
# .b = per-cell RNG carry. p=growth in u_params.x, f=lightning in u_params.y.
_register("forest_fire_seed",
          "Forest Fire seed: random trees at initial fraction, RNG carry in .b (node 96 twin)",
          "procedural", '''
void main() {
    float init = clamp(u_params.z, 0.1, 0.9);
    float h = hash21(floor(v_uv * u_resolution * 0.5));
    float state = h < init ? 1.0 : 0.0;   // 1 = tree
    float rng = hash21(v_uv * u_resolution + 3.71);
    f_color = vec4(state, 0.0, rng, 1.0);
}
''')