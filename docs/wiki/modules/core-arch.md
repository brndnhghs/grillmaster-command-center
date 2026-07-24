# Module: `core/arch.py`

## Purpose
Detect whether a registered method is Architecture A (simulation with internal loop) or Architecture B (stateless, externally driven).

## Responsibilities
- Classify methods into Architecture A or B
- Provide deterministic heuristic-based detection

## Public Interfaces

### `detect_architecture(meta: MethodMeta) -> str`
Returns `"A"` or `"B"` based on these heuristics:

1. **Has `n_frames` param** → Architecture A (simulation cooks frame list internally)
2. **Has `anim_mode` with non-'none' default** → Architecture A (internal animation loop)
3. **Tags contain 'simulation' or 'sim'** → Architecture A
4. **Has `time` or `_timeline` in params** → Architecture B (externally driven)
5. **Default** → Architecture B

## Architecture A (Simulation)
- Internal simulation loop
- Calls `capture_frame()` to emit intermediate frames
- Cooked once, frames cached in executor
- Examples: Gray-Scott (32), Boids (34), Dynamic Fracture (145), DLA (36)

## Architecture B (Stateless)
- One call = one frame
- Driven by `time`, `_timeline`, or `anim_mode` parameters
- Re-cooked every frame
- Examples: Fractals (07), Noise (05), Dither (13), Glitch (17)

## Dependencies
- `registry.MethodMeta`

## Consumers
- `graph.py` — used by `GraphExecutor` to decide sim cache vs. per-frame cook
- `server.py` — live loop uses architecture to determine dirty propagation
- `animation.py` — `animate_method()` uses architecture to decide animation strategy