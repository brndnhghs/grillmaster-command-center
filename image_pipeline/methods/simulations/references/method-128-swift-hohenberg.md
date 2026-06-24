# Method 128 — Swift-Hohenberg Pattern Formation

**Category:** simulations  
**ID:** 128  
**File:** `simulations/active_brownian.py`

## Description

A canonical pattern-forming PDE (model of Rayleigh-Bénard convection):

```
∂u/∂t = ε·u − u³ − (1 + ∇²)²·u + noise
```

For ε > 0, the uniform state becomes unstable to a finite-wavenumber band of perturbations, producing hexagonal lattices, striped rolls, and localized spots. This is **distinct from reaction-diffusion** — the instability selects a fixed wavelength determined by the biharmonic operator, not by a Turing bifurcation.

## Animation Modes

| Mode | Physics | Visual |
|------|---------|--------|
| `evolve` | Constant ε + noise | Hexagonal/stripe patterns with slow defect drift |
| `sweep_epsilon` | Ramp ε from -0.5 → ε_max | Noise → uniform → pattern onset → coarsening |
| `sweep_noise` | Ramp noise amplitude | Ordered hexagonal → chaotic spatiotemporal |
| `roam` | Gentle ε gradient rotates | Localized structures drift across the field |
| `oscillate` | Sinusoidal ε modulation | Patterns breathe, expand, and contract |

## Parameters

| Param | Default | Effect |
|-------|---------|--------|
| epsilon | 1.2 | Bifurcation parameter (>0 = pattern-forming) |
| noise_amp | 0.05 | Additive noise (prevents locking) |
| init_mode | noise | Initial condition (noise/hex_seed/stripe_seed/spot/quench) |
| grid_size | 192 | Simulation resolution |
| dt | 0.2 | Timestep |
| substeps | 5 | Substep count |

## Physics Notes

- The linear operator `−(1+∇²)²` has a band of unstable wavenumbers near |k| = 1 for ε > 0
- The cubic term `−u³` saturates the growth, giving a finite amplitude
- Pattern wavelength is set by the operator, approximately 2π pixels at the simulation grid resolution
- Noise is essential for the sweep modes — without it patterns get stuck in metastable states

## Why Different from Existing Methods

| Existing | What it does | How SH differs |
|----------|-------------|----------------|
| Cahn-Hilliard (#115) | Spinodal decomposition (conserved) | SH is non-conserved, finite-wavenumber instability |
| Gray-Scott (#32) | Two-species RD (Turing) | SH is single-field, no cross-diffusion |
| BZ Oregonator (#91) | Oscillatory RD chemistry | SH is non-oscillatory, pattern-forming only |
| Wave Equation (#100) | Linear wave propagation | SH is nonlinear pattern formation |

## Invocation

```bash
# Hexagonal patterns (default)
python -m image_pipeline.pipeline --method 128 --params '{"init_mode":"hex_seed"}' --animate 128 --seed 42

# Sweep epsilon (watch pattern onset)
python -m image_pipeline.pipeline --method 128 --params '{"anim_mode":"sweep_epsilon","epsilon":2.0,"init_mode":"noise"}' --animate 128 --seed 42

# Sweep noise (order → chaos)
python -m image_pipeline.pipeline --method 128 --params '{"anim_mode":"sweep_noise","noise_amp":1.0}' --animate 128 --seed 42

# Oscillating patterns
python -m image_pipeline.pipeline --method 128 --params '{"anim_mode":"oscillate","epsilon":2.0}' --animate 128 --seed 42
```

Architecture A — internal capture_frame loop. Fourier-spectral method for linear operator.
