# Method 126 — Continuous Spatial Prisoner's Dilemma (Replicator Dynamics)

**Category:** simulations  
**ID:** 126  
**File:** `simulations/spd_replicator.py`  
**Sibling:** #125 (binary SPD — discrete C/D, Nowak & May style)

## Description

Evolutionary game theory as a **reaction-diffusion PDE**. Each cell holds a continuous strategy `s ∈ [0,1]` (cooperation probability). The replicator equation drives the dynamics:

```
ds/dt = s·(1−s)·(π_coop − π_defect) + D·∇²s + η·N(0,1)
```

Produces **smooth gradient fields** of cooperation probability — flowing traveling waves, vortex spirals where high-cooperation streams curl around low-cooperation cores, and fluid-like mixing patterns. Completely different from #125's binary tile-switching and sharp domain walls.

## Animation Modes

| Mode | Physics | Visual |
|------|---------|--------|
| `evolve` | Perpetual replicator + noise | Never-settling flowing patterns |
| `sweep_temptation` | Ramp T from 1.0 → temptation | Transition from full-cooperation to oscillatory/turbulent |
| `sweep_diffusion` | Ramp D from 0 → diffusion_rate | Sharp domains melt into flowing streams |
| `noise_pulse` | 8× noise burst every 25 frames, then recovery | Dramatic "storms" in the strategy field |
| `parameter_cycle` | Sinusoidal modulation of T | Rhythmic expansion/contraction of cooperative domains |

## Parameters

| Param | Default | Effect |
|-------|---------|--------|
| temptation | 1.5 | Defector temptation payoff T (higher = more defection) |
| reward | 1.0 | Mutual cooperation payoff R |
| sucker_payoff | 0.5 | Sucker's payoff S (cooperator vs defector; S=0.5 centers equilibrium at s≈0.43 with T=1.5) |
| punishment | 0.0 | Mutual defection payoff P |
| diffusion_rate | 0.12 | Spatial diffusion strength (smooths strategy boundaries) |
| noise_amplitude | 0.008 | Gaussian noise per step (prevents convergence) |
| mutation_rate | 0.025 | Drift toward mixed strategies (prevents absorbing boundaries) |
| grid_size | 160 | Internal grid resolution |
| n_frames | 100 | Frames to capture |
| steps_per_frame | 3 | Sim substeps between captures |
| init_mode | clusters | Initial field pattern (clusters/gradient/vortex/random) |
| init_coop | 0.5 | Cooperation density bias |

## Physics Notes

- **Prisoner's Dilemma condition:** T > R > P ≥ S. If this isn't met, the game isn't a true dilemma — cooperation may be trivially optimal.
- **Replicator equation** provides the "reaction" term: `s·(1−s)·(π_coop − π_defect)`. This is zero at s=0 and s=1 (absorbing boundaries) and positive/negative based on which strategy outperforms.
- **Diffusion** creates spatial coupling — without it, each cell evolves independently and no patterns form.
- **Noise** is critical: without it, the system converges to all-cooperate or all-defect. The noise drives perpetual exploration.
- **Moore neighborhood** (8 neighbors) is used for payoff computation.

## Key Differences from #125 (Binary SPD)

| Aspect | #125 Binary SPD | #126 Continuous SPD |
|--------|----------------|-------------------|
| State | `{0, 1}` (cooperate/defect) | `[0, 1]` (continuous strategy) |
| Update rule | Imitate best / Fermi / Moran | Replicator PDE + diffusion + noise |
| Dynamics | Discrete tile-switching | Smooth gradient flow |
| Visual | Sharp domain walls, checkerboard | Smooth waves, fluid-like mixing |
| Patterns | Spiral waves of binary states | Traveling gradients, vortex streams |

## Palette Guidance

The method renders φ grayscale (s = 0.0 → black, s = 1.0 → white). The pipeline applies `--recolor`.

- **Diverging palettes** (coolwarm, seismic) work best — neutral at 0.5, one color for cooperation, the other for defection
- **Narrow two-tone** (cobalt/ember, phantom/ember) emphasizes the contrast between high-cooperation and low-cooperation domains
- Avoid full-HSV rainbow — the field is unipolar (strategy probability) and a diverging map is more appropriate

## Invocation

```bash
# Basic evolve
python -m image_pipeline.pipeline --method 126 --animate --anim-duration 4 --seed 42

# Sweep temptation (watch cooperation collapse)
python -m image_pipeline.pipeline --method 126 --params '{"anim_mode":"sweep_temptation","temptation":2.0,"n_frames":120}' --animate --anim-duration 5 --seed 42

# Noise pulse (dramatic storms)
python -m image_pipeline.pipeline --method 126 --params '{"anim_mode":"noise_pulse","noise_amplitude":0.008}' --animate --anim-duration 4 --seed 7

# Vortex init + diffusion sweep
python -m image_pipeline.pipeline --method 126 --params '{"anim_mode":"sweep_diffusion","init_mode":"vortex","diffusion_rate":0.3,"n_frames":100}' --animate --anim-duration 4 --seed 13
```

Architecture A — internal capture_frame loop.
