"""ks_seed — GPU shader registration (dynamically loaded by core/shaders.py)."""

from ._registry import _register



# ── Node 127: Kuramoto-Sivashinsky Equation ───────────────────────────────
# ∂u/∂t = -ν·∇⁴u - ∇²u - ½|∇u|². Scalar height field u packed in .r; .g is a
# phase accumulator that drives a hashed white-noise source (step shaders get
# u_time=0, pitfall #6b, so noise must be carried in state). Single channel.
# p1=nu (hyperviscosity), p2=dt, p3=noise_amp, p4=aniso_ratio (x/y stretch).
_register("ks_seed",
          "Kuramoto-Sivashinsky seed: small sinusoidal roll field + hashed noise (node 127 twin)",
          "procedural", '''
void main() {
    vec2 uv = v_uv * 6.2831853;
    float u = 0.2 * (sin(uv.x) + sin(uv.y)) + 0.05 * (noise(v_uv * 9.0) - 0.5);
    f_color = vec4(u, 0.0, 0.0, 1.0);  // R=u, G=phase
}
''')