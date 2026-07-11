# Render-Health Contracts (Route 0 — Leverage Tier)

Two regression guards lock the pipeline's image-output surface so a method that
registers cleanly in `/api/node-defs` but dies or goes blank at default params
can never ship silently:

| Test | Covers | Command |
|------|--------|---------|
| `image_pipeline/tests/test_sim_render_health.py` | every `simulations`-category method | `env -u PYTHONPATH .venv/bin/python -m pytest image_pipeline/tests/test_sim_render_health.py -q -p no:cacheprovider -m slow` |
| `image_pipeline/tests/test_generator_render_health.py` | `patterns` / `fractals` / `filters` / `math_art` (wire-independent only) | `env -u PYTHONPATH .venv/bin/python -m pytest image_pipeline/tests/test_generator_render_health.py -q -p no:cacheprovider -m slow` |

Both are marked `pytest.mark.slow` and are **deselected by default** (the repo
`addopts` includes `-m "not slow"`), so they never run in the default fast suite.
They are the Leverage-Tier counterpart to the ML e2e probe (`test_ml_nodes_e2e.py`):
the same "valid non-blank frame at defaults" contract, applied to the whole
simulation + generator surface instead of just CLIP/SAM.

## Why they matter

A method can pass `test_gpu_coverage_audit` (structure), `test_shader_parity`
(GPU twins), and still render a blank frame on the CPU export path at default
params. The export path is the authoritative one (GPU = live preview only), so a
blank/blanked export is a user-visible regression. These contracts are the only
thing that catches it automatically.

## How to run them cron-safely

Both spawn one subprocess worker **per method** with a hard `ProcessPoolExecutor`
timeout. The sim contract is the long pole:

- Default sim timeout `PER_METHOD_TIMEOUT = 90s`; cumulative-growth sims
  (`36` DLA, `55` Sandpile) get `CUMULATIVE_TIMEOUT = 150s` and render at 128×128
  instead of 256×256 (their output is an emergent process over many steps — the
  contract validates *validity*, not resolution).
- Observed worst case is ~62s for a normal sim, ~94s for DLA; the caps have
  margin. A full sim run therefore takes several minutes — run it in the
  background (`terminal(background=True, notify_on_complete=True)`), NOT a
  foreground call with a tiny timeout.
- Generators render at 160×160 with `PER_METHOD_TIMEOUT = 30s`; a full generator
  run is a couple of minutes.
- **macOS spawn gotcha:** `ProcessPoolExecutor` on macOS uses `spawn`, which
  re-imports the worker module fresh. Always `import image_pipeline.methods`
  inside the worker (`_run_one`) so `@method` registration fires in the child.
  The tests already do this; if you fork them, keep it.
- **Never pass `--timeout=N` to pytest here** — `pytest-timeout` is not installed
  and the flag is rejected (see skill pitfall #8). The per-method timeout lives
  inside the test via `fut.result(timeout=...)`.

## Blank-check subtlety

A frame is "blank" only if it is a single flat colour: `std < 0.01` **AND**
`np.unique(quantized) <= 2`. A low-contrast-but-VALID render (faint Perlin
relief, shallow-water height field at rest) has many distinct quantized tones and
is deliberately NOT flagged. Do not tighten the threshold — it would false-fail
legitimate low-contrast methods.

## Wire-dependent nodes are auto-excluded

Both contracts exclude methods whose declared input ports include a wirable type
(`image`, `mask`, `field`, `particles`), because those legitimately render
nothing useful with no upstream wire. The exclusions are computed from the live
registry by input-port type, so they stay correct as the graph grows (no
hand-maintained skip list to rot).

## Generator read-back is executor-faithful

The generator contract does NOT require a disk PNG. It treats BOTH a returned
`{"image": arr}` dict AND a `save()`'d PNG as a successful render (mirroring
`core/graph.py` execute()). This avoids false-failing node `05` "Procedural
Noise", which returns its image and lets the executor write the file. It only
fails on a blank / None / missing image or a hard crash.

## What to do when a contract fails

1. **Read the failing id + detail** (`blank output`, `no PNG output`, `TIMEOUT`,
   `worker crashed: ...`).
2. **Reproduce standalone** in a subprocess: `env -u PYTHONPATH .venv/bin/python`
   then `import image_pipeline.methods; from image_pipeline.core.registry import
   get_all; get_all()["<id>"].fn("/tmp/dbg", 42, params={})` — but a forked
   `.py` probe in the repo root is cleanest (see skill pitfall #7).
3. **Fix the method** (seed wiring, dead `params.get`, blank default state) — do
   NOT weaken the contract to make the method pass. A contract failure is a real
   defect signal.
4. **Re-run just that parametrized case** to confirm the fix before committing:
   `pytest ...::test_sim_renders_valid_frame[36:Dielectric...] -m slow`.
