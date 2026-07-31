"""ca_seed — GPU shader registration (dynamically loaded by core/shaders.py)."""

from ._registry import _register



# ── Conway's Game of Life (client-GPU sim of nodes 18 / 58) ───────────────────────
# Single-channel CA: state.r = alive mask (0/1), state.g = age (frames alive).
# 8-neighbor toroidal count; birth on 3, survival on 2/3 (classic Conway).
_register("ca_seed",
          "Game of Life seed: hashed random alive cells at given density (nodes 18/58 twin)",
          "procedural", '''
void main() {
    float dens = clamp(u_params.x, 0.02, 0.9);
    float h = hash21(floor(v_uv * u_resolution * 0.5));
    float alive = h < dens ? 1.0 : 0.0;
    f_color = vec4(alive, 0.0, 0.0, 1.0);
}
''')