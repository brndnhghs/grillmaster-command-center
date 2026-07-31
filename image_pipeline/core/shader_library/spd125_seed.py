"""spd125_seed — GPU shader registration (dynamically loaded by core/shaders.py)."""

from ._registry import _register




# ═══════════════════════════════════════════════════════════════════════════
#  P1.4b — Spatial Prisoner's Dilemma (nodes 153 / 154 twins)
#  #153 is a BINARY strategy lattice (cooperate/defect) → same family as the
#  Ising twin: per-cell RNG carry in .b, probabilistic Fermi imitation update.
#  #154 is the CONTINUOUS replicator PDE (s ∈ [0,1]) → same family as the CML
#  / wave twins: packed R=raw field, G=EMA trail, smoothed by PDE + diffusion.
#  CPU numpy nodes stay authoritative (two-tier precision contract).
# ═══════════════════════════════════════════════════════════════════════════

# ── Node 153: Spatial Prisoner's Dilemma — binary lattice ──
_register("spd125_seed",
          "SPD #153 seed: hashed random cooperate/defect lattice, RNG carry in .b",
          "procedural", '''
void main() {
    float h = hash21(floor(v_uv * u_resolution * 0.5));
    float strat = h < 0.5 ? 0.0 : 1.0;   // 0=defect, 1=coop
    float rng = hash21(v_uv * u_resolution + 19.3);
    f_color = vec4(strat, 0.0, rng, 1.0);  // R=strat, G=payoff, B=rng
}
''')