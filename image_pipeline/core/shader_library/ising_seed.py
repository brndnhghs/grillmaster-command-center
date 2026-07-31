"""ising_seed — GPU shader registration (dynamically loaded by core/shaders.py)."""

from ._registry import _register



# ── Node 93: 2D Ising Model (Glauber live approximation of Wolff) ──
# Spins σ=±1 packed as .r in {0,1} (0=-1, 1=+1) to survive fp32; .b = RNG carry.
# Coupling J (period 1) in u_params.x, T/Tc in u_params.y (Glauber p below Tc).
_register("ising_seed",
          "Ising seed: hashed random spin config, RNG carry in .b (node 93 twin)",
          "procedural", '''
void main() {
    float h = hash21(floor(v_uv * u_resolution * 0.5));
    float spin = h < 0.5 ? 0.0 : 1.0;   // 0 = down, 1 = up
    float rng = hash21(v_uv * u_resolution + 11.7);
    f_color = vec4(spin, 0.0, rng, 1.0);
}
''')