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

## 2. THE decision: three patterns, pick one

Get this right first; everything else follows. There are three valid patterns —
and one anti-pattern that is never valid.

- **Stateless generator (Architecture B, no state).** One call = one frame, a
  pure function of the clock. Cost is independent of how far the timeline has
  run. Correct for gradients, noise, pattern generators. Detected as B by
  default.

- **★ Persistent stateful sim (Architecture B + per-node state) — PREFERRED for
  open-ended sims.** Keep the sim's **last state** in a per-node store and step
  it forward one step per output frame. Runs **forever** at flat per-frame cost,
  constant memory (one state per node), no window, no loop, no reset. This is
  how #18 works now. Correct for CA / reaction-diffusion / most "watch it run"
  sims. Still Architecture B (one call per frame) — do **not** add `n_frames` or
  a `simulation` tag.

- **Cook-a-window sim (Architecture A).** Cook the whole internal frame list in
  one call, `capture_frame()` each; the executor caches and serves it at O(1),
  looping when the live window exceeds the cooked count. Correct when you need a
  *finite, deterministic, scrubbable/exportable* sequence (boids demos, a fixed
  gray-scott clip). Detected as A when the method declares an **`n_frames`** param
  (strongest signal), `anim_mode` non-"none", or a `simulation`/`sim` tag. See
  `core/arch.py::detect_architecture`. Tradeoff: bounded length, higher memory
  (N frames held), a one-time cook hitch on start / param change; it loops rather
  than running truly forever.

- **✗ The anti-pattern (never valid): a stateless node whose per-frame work
  scales with `time`.** #18 originally ran `int(time·60·speed)` generations
  *from the seed every frame*. In live `time = float(frame)` climbs without
  bound, so cost grew linearly forever — **42 ms at frame 1 → 1534 ms at frame
  200**. If you catch a node re-simulating up to `time`, it is broken; convert
  it to one of the two stateful patterns.

**Decision rule for this node:**
> Frame *k* is a self-contained function of the clock (no accumulation)?
> → **Stateless generator (B).**
> State accumulates, and the user wants to watch it run open-endedly?
> → **Persistent stateful sim (★, §3).**
> State accumulates, but you need a fixed deterministic sequence to scrub/export?
> → **Cook-a-window (Architecture A, §3b).**

---

## 3. Persistent stateful structure (★ preferred — run forever)

Architecture B (one call per frame) with a persistent per-node state store. This
is what makes a sim run forever at constant cost. Model this on #18
(`methods/codegen/simulations.py`).

```python
import threading as _threading

# Module-level: last state per node, keyed on out_dir (ends in the node id, so
# nodes never collide; live and clip use different dirs → independent sims).
_STATE: dict[str, dict] = {}
_STATE_LOCK = _threading.Lock()
_STATE_MAX = 64                      # cap so long sessions don't grow unbounded

@method(
    id="{{METHOD_ID}}", name="...", category="simulations",
    tags=[..., "animation"],         # NO 'simulation' tag, NO n_frames → stays Arch B
    outputs={"image": "IMAGE", "luminance": "SCALAR", ...},
    params={..., "speed": {"description": "steps advanced per output frame", "default": 1.0}},
)
def run(out_dir, seed, params=None):
    params = params or {}
    # Resolve params (see §5 for the -1.0 sentinel rule). Split them:
    #   struct params  -> identity of the sim (changing them rebuilds)
    #   render/step params -> applied live every frame (no rebuild)
    struct_sig = (seed, rule, pattern, density, cell_size, seed_image_id, ...)
    steps_per_frame = max(1, int(round(effective_speed)))
    clock = float(params.get("time", 0.0))     # monotonic: unbounded in live, phase in clips
    key = str(out_dir)

    with _STATE_LOCK:
        st = _STATE.get(key)
        rebuild = st is None or st["sig"] != struct_sig or clock < st["clock"] - 1e-9
        if rebuild:
            state = build_initial_state(seed, ...)   # frame 0 = the start (see §4)
            # (optional) if single still — _timeline.total_frames <= 1 — pre-step
            # a few times so an Auto preview looks alive rather than a bare seed.
        else:
            state = st["state"]
            if clock > st["clock"] + 1e-9:            # advance exactly one frame
                for _ in range(steps_per_frame):
                    state = step(state, ...)          # step the PERSISTENT state
        _STATE[key] = {"state": state, "sig": struct_sig, "clock": clock}
        if len(_STATE) > _STATE_MAX:
            for k in list(_STATE)[:-_STATE_MAX]:
                _STATE.pop(k, None)

    return {"image": render(state, ...)}              # H×W×3 float32 [0,1]
```

Why this runs forever: it never reads anything but the **last** state; the clock
is only used to decide *rebuild vs. advance-one-step*, never to size the work.
`clock < last - eps` (scrub back / live restart) and a changed `struct_sig`
(param edit) are the only rebuild triggers. Memory is one state per node.

### 3b. Cook-a-window structure (Architecture A — only for finite sequences)

```python
@method(id="{{METHOD_ID}}", ..., tags=[..., "simulation"],
        params={..., "n_frames": {"min": 30, "max": 600, "default": 120}})
def run(out_dir, seed, params=None):
    state = build_initial_state(seed, ...)
    n = max(1, int(params.get("n_frames", 120)))
    last = None
    for f in range(n):
        last = render(state, ...)
        capture_frame("{{METHOD_ID}}", last)   # frame 0 first → shows from the start
        for _ in range(max(1, int(round(speed)))):
            state = step(state, ...)
    return {"image": last}
```
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

- **Was:** stateless, `generations = max(60, int(time·60·speed))`, rebuilt the
  grid from the seed every frame. Live slowed 42 ms → 1534 ms+; params ignored
  via `is not None` sentinels; first clip frames clamped onto the 60-gen floor.
- **Now:** the **persistent stateful pattern (§3)** — Architecture B with a
  per-node `_CA_STATE` store keyed on `out_dir`; keeps the last grid, steps it
  one generation per frame, rebuilds only on structural-param change or a
  backward clock; sentinels checked `>= 0`. Result: **flat per-frame cost with no
  window and no reset — steady 30 fps at 768×512 held for 70+ seconds of
  continuous live play, still evolving.** Render/step params (color, inject,
  wave) modulate the running sim live; there is no per-frame-channel tradeoff.
- **Files:** `image_pipeline/methods/codegen/simulations.py` (the method),
  `image_pipeline/tests/test_live_regression.py`. The cook-a-window path also got
  a `_node_params_hash` volatile-key exclusion and a modulo-loop cache serve in
  `image_pipeline/core/graph.py` — those benefit any Architecture-A sim.

---

## Candidate nodes to run this against

Any stateless method whose output evolves with `time`/`frame` and that could
drift slower over a long playback — start with the `simulations/` package and
anything tagged `simulation`/`animation` that is *not* already Architecture A
(no `n_frames`). Prioritise the ones the user actually plays live. For each,
produce the §9 report so the tradeoffs are visible before merging.
