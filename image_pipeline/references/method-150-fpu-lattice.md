# FPU Chain Lattice (ID 150)

Fermi-Pasta-Ulam-Tsingou nonlinear spring lattice on a 2D grid. Conservative Verlet integration — energy conserved, perpetual motion. Energy sloshes between long and short wavelengths in slow recurrent cycles.

**File:** `methods/simulations/chua_lattice.py`
**Function:** `fpu_lattice`

## Parameters

| Param | Range | Default | Description |
|-------|-------|---------|-------------|
| k2 | 0.1–5.0 | 1.0 | Linear spring constant |
| k3 | 0.0–2.0 | 0.5 | Cubic (α-FPU) nonlinearity |
| k4 | 0.0–2.0 | 0.3 | Quartic (β-FPU) nonlinearity |
| mode | waves/impulse/random/checker/vortex | waves | Initial condition |
| n_frames | 100–1500 | 480 | Frame count |
| grid_div | 1–4 | 2 | Coarse grid factor |
| dt | 0.01–0.2 | 0.05 | Time step |

## Modes

- **waves** — standing wave modes (1-3 in x and y), energy conserved
- **impulse** — center impulse, dissipates radially
- **random** — random initial displacements and velocities
- **checker** — checkerboard pattern, symmetric growth
- **vortex** — angular mode with radial decay

## Visual Signature

Thermal colormap (black → red → yellow → white) of displacement magnitude. Energy conserved to ~4 decimal places. Displacement grows slowly as nonlinearities transfer energy between modes.

## Notes

- No damping — perpetual motion
- Velocity Verlet integration for stability
- Energy conservation verified each frame
- Best modes: waves, checker, vortex
