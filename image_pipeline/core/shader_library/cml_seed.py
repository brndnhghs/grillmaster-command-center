"""cml_seed — GPU shader registration (dynamically loaded by core/shaders.py)."""

from ._registry import _register



# ── Nodes 95 / 142: Coupled Logistic Map Lattice ────────────────────────────
# Both nodes are the SAME dynamical system: each cell x evolves via the logistic
# map f(x)=r·x·(1-x), diffusively coupled to its 4 neighbours with strength ε:
#   x' = (1-ε)·f(x) + (ε/4)·Σ f(x_neighbour)
# Discrete-time recurrence ⇒ raw state strobes, so an EMA trail (decay) is packed
# alongside the raw lattice: state R=raw x, G=accum (trail). display reads accum.
# p1=r (3.5–4.0), p2=ε coupling (0.05–0.5), p3=decay trail (0.5–0.99).
_register("cml_seed",
          "Coupled logistic seed: hashed uniform lattice in [0,1] (nodes 95/142 twin)",
          "procedural", '''
void main() {
    float x = hash21(v_uv * u_resolution + 0.123);
    f_color = vec4(x, x, 0.0, 1.0);  // R=raw x, G=accum(trail)
}
''')