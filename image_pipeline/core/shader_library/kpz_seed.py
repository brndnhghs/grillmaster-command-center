"""kpz_seed — GPU shader registration (dynamically loaded by core/shaders.py)."""

from ._registry import _register



# ── Node 135: KPZ Surface Growth / Erosion ──────────────────────────────────
# ∂h/∂t = ν·∇²h + (λ/2)·|∇h|² + η(x,t). Height h + phase accumulator for noise.
# State packs R=h, G=phase, B=unused. Live preview approximates the KPZ growth.
# p1=nu (diffusion), p2=lambda (nonlinearity), p3=noise_amplitude, p4=dt.
_register("kpz_seed",
          "KPZ seed: flat height field + small hashed perturbation (node 135 twin)",
          "procedural", '''
void main() {
    float n = noise(v_uv * 11.0) * 0.5 + noise(v_uv * 23.0) * 0.5;
    f_color = vec4((n - 0.5) * 0.2, 0.0, 0.0, 1.0);  // R=h, G=phase
}
''')