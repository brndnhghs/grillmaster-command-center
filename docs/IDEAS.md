# Ideas

> Free-form capture of speculative, not-yet-scored ideas. Anything that could
> make the project better. Reviewed each loop; promising items graduate to
> `ROADMAP.md` (work streams) or `FEATURE_BACKLOG.md` (features).

---

## Raw Ideas

- **Typed-uniform live sliders for *all* wireable params** — extend the
  typed-uniform contract (#6) beyond the GPU-twin P0 set so every numeric param
  gets a real, un-frozen live-preview slider. (Follows the `7a633c0` /
  `a785fd4` contract work.)

- **Determinism-diff fuzzer** — a CI job that renders the same graph with two
  different but equivalent node orderings / seeds and asserts pixel equality.
  Catches topological-sort / wiring-order regressions proactively.

- **Graph linters** — static checks on a saved graph: dangling ports, cycles
  without feedback flag, terminal-less graphs, type-mismatched wires.

- **Method coverage matrix** — a generated table of which methods have unit
  tests vs. only executor-level coverage, to find dark corners.

- **Sidecar schema registry** — formalize the named sidecar ports
  (`field`, `particles`, `mask`, `palette`, `luminance`) so new methods declare
  them in one place and the executor validates wiring.

- **Headless render CLI parity** — make `runner.py`'s graph path call
  `GraphExecutor` so the CLI and server share one engine (closes TD-11).

- **Per-frame metadata JSON** — emit a sidecar `_frame_meta.json` per render
  (node timings, cache hits, edge transports) so the FX overlay and debugging
  don't depend on in-memory state.

- **Method authorship helper** — `tools/next_id.py` already exists; add a
  template generator that scaffolds a `@method` with correct contract, inputs,
  and Arch-A/Arch-B choice.

---

## Discarded / Parked

- *(none yet)*
