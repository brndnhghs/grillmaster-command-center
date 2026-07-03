# Prompt — Audit & Rewrite a Simulation Node for the Render System

**Give this prompt to the agent (Node Doctor / Hermes) once per simulation or
animated method you want to bring up to standard.** It encodes what we learned
optimizing Cellular Automata (#18) so every sim plays correctly and stays fast
in all three render contexts. Read it in full before touching a method file. It
supplements — does not replace — `AGENT_GUIDE.md` (the method contract) and
`DESIGN.md` → "Live mode" (the authoritative render architecture).

Your target method file: **`{{METHOD_FILE}}`** (method id **`{{METHOD_ID}}`**).

---

## 0. Mission

Make this node behave correctly and performantly under the top-level render
system. "Correctly" means the same node produces sensible output in **all three
render contexts**, and "performantly" means **per-frame cost never grows as the
timeline advances**. Do not change the render pipeline (`core/graph.py`,
`server.py`) — fix the *method*. The one exception already made system-wide is
the sim-cache key; you rely on it but must not re-touch it.

---

## 1. The three render contexts (know how each drives your node)

Every method is invoked by the same executor (`GraphExecutor.execute`), but from
three entry points with different framing:

| Context | Endpoint | `frames` | How your node is driven |
|---|---|---|---|
| **Single still** | `POST /api/graph/execute` (frames=1) | 1 | One cook. Auto mode / param tweak preview. |
| **Clip / sequence** | `POST /api/graph/render-sequence` | N (clip length) | One cook per output frame, `frame`=0..N-1. |
| **Live** | `POST /api/graph/live` | 300 (`LIVE_TOTAL_FRAMES`) | Continuous loop, ~30 fps, `frame`=loop%300, **forces `dirty=True` every frame** and **injects `time=float(frame)`** (unbounded, monotonic). |

On every call the executor injects into `params`: `frame` (int), `frame_seed`
(int), `time` (float — the timeline phase, unless the live loop already set it),
and `_timeline` (a `Timeline` with `.t` in [0,1], `.phase` in [0,2π),
`.total_frames`, `.fps`, `.speed`). Your node must read its animation from these,
never from wall-clock or call-count.

---

## 2. THE decision: Architecture A or B

This is the most important call. Get it right first; everything else follows.

- **Architecture B (stateless generator).** One call = one frame, fully
  determined by the clock. Cost is *independent of how far the timeline has
  run*. Correct for: gradients, noise, pattern generators, pure functions of
  `t`/`phase`. Detected as B by default.

- **Architecture A (stateful simulation).** The sim accumulates state over time
  (a grid, particle set, field that evolves). It cooks its own frame list once,
  the executor caches it, and every subsequent frame is served from cache at
  **O(1)**. Correct for: any CA / reaction-diffusion / boids / fluid / growth
  process. Detected as A when the method declares an **`n_frames` param** (the
  strongest signal), or an `anim_mode` param with a non-"none" default, or a
  `simulation`/`sim` tag. See `core/arch.py::detect_architecture`.

### The failure that forces this decision (the #18 lesson)

**Never write an Architecture-B node whose per-frame work scales with `time`.**
#18 was stateless and ran `int(time·60·speed)` generations *from the seed every
frame*. In live mode `time = float(frame)` climbs without bound, so the work per
frame grew linearly forever — measured **42 ms at frame 1 → 1534 ms at frame
200**, and worse after. A pure generator (Gradient) never slows because its cost
is constant in the clock. **If your node must accumulate state over time, it is
Architecture A. If it is a pure function of the clock, it is Architecture B.
There is no valid third option where a stateless node re-simulates up to `time`.**

**Decision rule for this node:**
> Does correct output at frame *k* require having computed frames *0…k-1*
> (state accumulates)? → **Architecture A.**
> Is frame *k* a self-contained function of the clock? → **Architecture B.**

---

## 3. Architecture-A structure (use this skeleton for stateful sims)

```python
@method(
    id="{{METHOD_ID}}",
    name="...",
    category="simulations",
    tags=[..., "simulation"],          # 'simulation' tag reinforces arch A
    outputs={"image": "IMAGE", "luminance": "SCALAR", ...},
    params={
        ...,
        "speed": {"description": "steps advanced per output frame", "default": 1.0},
        "n_frames": {                  # REQUIRED for arch A — the detector signal
            "description": "frames to cook (sim steps forward once per frame, "
                           "constant cost per frame)",
            "min": 30, "max": 600, "default": 120,
        },
    },
)
def run(out_dir, seed, params=None):
    params = params or {}
    # 1. Resolve params ONCE (see §5 for the -1.0 sentinel rule).
    ...
    # 2. Build the initial state ONCE.
    state = build_initial_state(seed, ...)
    # 3. Cook: persist state across frames, capture BEFORE stepping so frame 0
    #    is the initial state (shows the sim from its beginning — see §4).
    n_frames = max(1, int(params.get("n_frames", 120)))
    steps_per_frame = max(1, int(round(effective_speed)))
    last = None
    for f in range(n_frames):
        last = render(state, ...)      # H×W×3 float32 in [0,1]
        capture_frame("{{METHOD_ID}}", last)   # emit this frame
        for _ in range(steps_per_frame):
            state = step(state, ...)   # advance the PERSISTENT state
    return {"image": last}
```

Key points:
- **State persists across the internal loop** — you step it forward, you do NOT
  rebuild from the seed each captured frame. That is what makes cost flat.
- **`capture_frame(method_id, img)`** (from `core.animation`) is how arch-A
  methods emit their frame list. The executor collects them, caches under
  `(node_id, seed)`, and serves `cached[frame]`. Do not call `save()` per step.
- **Capture before you step**, so frame 0 is the initial state.
- The executor **overrides `n_frames` to the render length** when `frames > 1`
  (clip length, or 300 in live), *only if `n_frames` is present in the node's
  params*. The UI always sends it, so in practice a live cook produces 300
  frames and the loop serves `frame % 300`. Cooking is a **one-time cost** at
  Live-start and on each param change; after that every frame is ~O(1).

---

## 4. "Show the sim from its start" (the generation-floor lesson)

#18 also had a **floor** (`generations = max(60, int(t·60))`) added so a single
still looked evolved. During animation that froze the first frames onto the
floor value — the opening of the sim was missing and "picked up" several frames
in. **Do not floor or offset the frame index to make a preview look nice.** In
Architecture A the fix is free: capture frame 0 = initial state, and the timeline
naturally starts there. If you want a single-still preview to look evolved, gate
that on `_timeline.total_frames <= 1` only — never on the animated path.

---

## 5. The `-1.0` sentinel rule (scalar-override params)

If a param is both a UI control **and** a wireable `SCALAR` override (default
`-1.0` meaning "not wired"), you must **not** gate the override on
`is not None`. The client always sends the param at its default, so
`params.get(name) is not None` is always true and the override permanently
clobbers the UI value. This silently broke #18's `rule`/`seed_pattern`/`size`.

```python
sel = params.get("rule_select")
if sel is not None and float(sel) >= 0:      # a wired channel sends 0..1
    effective_rule = RULE_NAMES[int(float(sel) * len(RULE_NAMES)) % len(RULE_NAMES)]
else:                                         # -1.0 default → honour the UI param
    effective_rule = params.get("rule", "conway")
```

Audit every override port in this node for this pattern.

---

## 6. Cooperating with the render system (invariants you must not fight)

The live loop guarantees four things (see `DESIGN.md` → "Live mode"). Your node
must be compatible with them:

1. **It re-cooks every frame (forces `dirty=True`).** Do not rely on being
   called exactly once or in order; do not stash state in module globals keyed
   on call count. (Arch-A state lives inside one cook, which is fine.)
2. **The clock advances** (`t` sweeps 0→1 over the window; `time` is unbounded
   monotonic). Read motion from `t`/`phase`/`time`/`frame` — a node that reads
   none of them cannot animate.
3. **Injected `time` is preserved** by the executor. Fine to read it; do not
   assume it is bounded (it is `float(frame)` in live).
4. **The sim cache is keyed on *defining* params only** — `_node_params_hash`
   excludes `time`, `frame`, `frame_seed`, `_timeline`, `input_image`. So your
   node's *defining* params (rule, density, size, …) determine cache identity;
   changing any of them re-cooks, changing only the clock does not. **Corollary:
   an arch-A sim must be fully determined by its defining params + `seed` — do
   not let its cooked frames depend on `time`/`frame`, or the cache will serve
   stale-but-wrong frames.**

---

## 7. Design requirements (what "good" looks like)

- **Constant per-frame cost.** After any one-time setup, cost must not grow with
  the timeline. This is the headline requirement.
- **Deterministic.** Same defining params + `seed` ⇒ identical cook. Seed all RNG
  (`random.seed`, `np.random.seed`, or explicit `np.random.default_rng(seed)`).
  Do not seed from `time`/`frame` in a way that breaks cache determinism.
- **Plays from the start.** Frame 0 is the beginning of the sim, not a
  mid-evolution still.
- **Honours its UI params** in every context (see §5).
- **Canvas-relative.** Use `W`, `H` from `core.utils`; never hardcode 768×512.
- **Bounded memory awareness.** Arch-A cooks the whole window into memory (N
  frames × H×W×3 float32). Keep the internal simulation grid at a sensible
  resolution; render up to canvas size, but don't hold N full-res copies of
  giant buffers needlessly. Flag it in your report if this node is heavy.
- **Legible.** One reader understands the whole file. Comment the *why* of any
  non-obvious constant (step count, threshold, cadence).

---

## 8. Code-structure requirements (mechanical)

- `@method` declares `outputs=` (incl. `luminance`) and a one-line `description=`.
- Architecture-A sims declare an **`n_frames` param**; the internal loop calls
  `capture_frame(id, img)` once per frame and returns `{"image": last_frame}`.
- Resolve all params **once**, before the cook loop.
- **Every code path produces an image** (return an error image, save a fallback,
  or re-raise — see `AGENT_GUIDE.md` §2). Broad `except` must not silently
  swallow. Narrow compat handlers that fall through are fine.
- Temp files use a `_` prefix.
- Explicit imports at the top: `from ..core.animation import capture_frame`,
  helpers from `core.utils`.
- Method id is unique and permanent (`tools/next_id.py`); the registry **raises**
  on a duplicate from another module.
- No hidden module-global state keyed on node identity as a substitute for
  Architecture A — the sim cache is the sanctioned state mechanism.

---

## 9. Audit → rewrite → verify procedure for this node

1. **Classify.** Read the method. Does state accumulate over time? Decide A or B
   and state your reasoning in the report.
2. **If it's a stateless sim that scales work with `time`** (the #18 anti-
   pattern) — convert to Architecture A per §3. If it's genuinely a pure clock
   function, leave it Architecture B but confirm §7's constant-cost holds.
3. **Fix §4 (show-from-start)** and **§5 (sentinels)** if present.
4. **Self-check (all must pass):**
   - `uv run python -c "from image_pipeline.server import app"` — imports clean.
   - `uv run python tools/audit_methods.py --fail-on-violations` — exit 0.
   - **No-slowdown timing** (the core proof). Cook the node through an executor
     across the timeline and assert per-frame cost is flat:
     ```python
     import time, tempfile; from pathlib import Path
     import image_pipeline.methods
     from image_pipeline.core.graph import GraphExecutor
     from image_pipeline.core.utils import set_canvas
     set_canvas(768, 512)
     ex = GraphExecutor(Path(tempfile.mkdtemp()), in_memory=True)
     P = {..., "n_frames": 120}          # include every default param the UI sends
     def ms(f):
         n=[{"id":"s","method_id":"{{METHOD_ID}}","params":{**P,"time":float(f)},
             "dirty":True,"render":True}]
         t0=time.time(); ex.execute(n,[],seed=42,frame=f%300,frames=300); return (time.time()-t0)*1000
     ms(1)                                # one-time cook
     assert ms(250) < ms(5)*4 + 20        # late frame ≈ as cheap as an early one
     ```
   - **Plays from start** (clip): render frames 0..N via the executor; assert the
     first frames are not all identical (no clamp onto a later still).
   - **Params honoured**: two renders differing only in a UI param (e.g. rule)
     must differ, with the `-1.0` sentinels present.
5. **Add / extend a regression test** mirroring
   `image_pipeline/tests/test_live_regression.py` (there is a
   `test_stateful_sim_..._does_not_slow_down` pattern to copy).
6. **Report**: architecture chosen and why; what you changed; the timing numbers
   (early vs late frame); any memory/heaviness caveat; the tradeoff if wired
   channels no longer modulate per-frame (arch-A cooks in one pass — channels
   set the sim up but don't drive it per output frame).

---

## 10. Reference: the #18 fix (worked example)

- **Was:** Architecture B, `generations = max(60, int(time·60·speed))`, rebuilt
  grid from seed every frame. Live slowed 42 ms → 1534 ms+; params ignored via
  `is not None` sentinels; first clip frames clamped onto the 60-gen floor.
- **Now:** Architecture A with `n_frames`; grid persists, one generation stepped
  per captured frame, cooked once and cached; sentinels checked `>= 0`; captures
  frame 0 = initial grid. Result: **flat ~8 ms/frame at 768×512 across the whole
  timeline; steady 30 fps over the wire with no decay.**
- **Files:** `image_pipeline/methods/codegen/simulations.py` (the method),
  `image_pipeline/core/graph.py` (`_node_params_hash` volatile-key exclusion —
  system-wide, already done), `image_pipeline/tests/test_live_regression.py`.
- **Tradeoff accepted:** wired SCALAR channels now set the sim up once rather
  than modulating it per output frame. Note this per node if it applies.

---

## Candidate nodes to run this against

Any stateless method whose output evolves with `time`/`frame` and that could
drift slower over a long playback — start with the `simulations/` package and
anything tagged `simulation`/`animation` that is *not* already Architecture A
(no `n_frames`). Prioritise the ones the user actually plays live. For each,
produce the §9 report so the tradeoffs are visible before merging.
