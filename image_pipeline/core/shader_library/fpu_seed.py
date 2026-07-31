"""fpu_seed — GPU shader registration (dynamically loaded by core/shaders.py)."""

from ._registry import _register



# ── Node 150: FPU Chain Lattice ─────────────────────────────────────────────
# Conservative Verlet on a 2D mass-spring grid: displacement u + velocity v.
# Nonlinear springs (k2 linear, k3 cubic, k4 quartic). State packs R=u, G=v.
# p1=k2, p2=k3, p3=k4, p4=dt.
_register("fpu_seed",
          "FPU seed: multi-scale hashed displacement + small velocity (node 150 twin)",
          "procedural", '''
void main() {
    float n = noise(v_uv * 5.0) * 0.5 + noise(v_uv * 13.0) * 0.3
            + noise(v_uv * 29.0) * 0.2;
    f_color = vec4((n - 0.5) * 0.6, 0.0, 0.0, 1.0);  // R=u, G=v
}
''')