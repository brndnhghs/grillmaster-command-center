# Image Pipeline — Full Code Review & Live-Mode Optimization Plan

*Reviewed 2026-07-01 against `image_pipeline/` at HEAD (ea2e561). Every finding below was verified by reading the actual code paths, not inferred. Design docs consulted: DESIGN.md, AGENT_GUIDE.md, PHASE1_PLAN.md, CODEBASE_AUDIT.md, image-pipeline-dev-report.md, timeline-plan.md.*

**Goal:** the pipeline must eventually run **live** (continuous, interactive, TouchDesigner-style cooking) in addition to the existing **render** path (frame-range bake to PNG/MP4) — while staying agent-legible per the design philosophy.

---

## Part 1 — Code Review

### 1.1 Confirmed correctness bugs

**BUG-1 — `field_a`/`field_b` wiring crashes the whole graph run.**
`graph.py:650`: `_farr = slot.get("field") or src_img`. The `field` slot always holds an ndarray (executor sets `"field": extra_outputs.get("field", arr)` at `graph.py:879`), and `or` on a multi-element ndarray raises `ValueError: truth value of an array … is ambiguous` (verified). Worse, edge processing runs **outside** the per-node try/except, so this kills the entire job (or, in the live loop, is swallowed by `except Exception: pass` → silent black frames). Fix: `_farr = slot.get("field"); _farr = _farr if _farr is not None else src_img`.

**BUG-2 — `params_hash` used out of scope for list-returning methods.**
`params_hash` is only assigned inside the `if arch == "A":` block (`graph.py:438`), but the read-back path at `graph.py:817-819` (`isinstance(_fn_result, list)`) writes `self._sim_params_hash[node_id] = params_hash`. If a non-Arch-A method returns a list: `NameError` on the first such node, or — worse — a **stale hash from a previous node in the same loop iteration**, silently corrupting the sim cache. Compute the hash unconditionally (or at read-back time).

**BUG-3 — Non-deterministic seeds across server restarts.**
`node_seed = seed + frame + (hash(node_id) & 0xFFFF)` (`graph.py:430`, `754`). Python string `hash()` is randomized per process (PYTHONHASHSEED). Identical graph + seed produces **different output after every server restart**, directly violating the reproducibility contract in AGENT_GUIDE.md ("Identical seed + params must produce identical output"). Use a stable hash: `zlib.crc32(node_id.encode())`.

**BUG-4 — Live-sim global state races (`server.py:882-933`).**
`/api/graph/live` rebinds the module-global `_live_sim_cancel` on every start. The running loop reads the global each iteration, so: (a) starting a second live sim leaves the first thread alive, both pushing interleaved frames to the shared `_LIVE_FRAME` buffer; (b) there is no way to stop only one. Needs a per-session object (thread + own cancel event) with stop-before-start semantics.

**BUG-5 — `in_memory` capture monkeypatches `utils.save` process-globally.**
`graph.py:770-783` replaces `image_pipeline.core.utils.save` for the duration of a node call. A concurrent job (live loop + render job, or two graph jobs) sees the other job's wrapper — captured frames can land in the wrong executor's `_captured_output`, and an exception in one job can leave the wrapper permanently installed. This must become context-local (ContextVar, like `set_canvas`) or be removed entirely by finishing the dict-return contract.

**BUG-6 — Group nodes lose all state every frame.**
`_execute_group_node` (`graph.py:957`) constructs a **fresh** `GraphExecutor` per frame. Consequences: feedback edges inside groups never see a previous frame; Architecture-A sims inside groups re-run from scratch every frame (also a huge perf hit); dirty-skip never applies. The sub-executor must be cached on the parent executor keyed by group node id.

**BUG-7 — CLI `--all` crashes on dunder method ids.**
`registry.resolve_keys` sorts with `key=lambda x: (int(x), x)` (`registry.py:200`); the registry now contains non-numeric ids (`__counter__`, `__ramp__`, `__timeline__`, …) from `channels.py` and `system/`. `int("__counter__")` raises `ValueError`, so `pipeline.py --all` is broken. Sort with a fallback (`int(x)` if numeric else `(inf, x)`), or exclude the `channels`/`system` categories from CLI resolution.

**BUG-8 — Unbounded memory in two places.**
- `_sim_cache` (`graph.py:306`): full-resolution float32 frame lists per node — a 300-frame sim at 768×512 is ~1.4 GB per node, never evicted.
- Job queues (`server.py:265`): `graph_frame` events carry base64 JPEGs; if a client never opens the SSE stream, the queue grows without bound. Eviction only removes *done* jobs and only when a new job starts.

### 1.2 Design-contract drift (docs vs code)

- **`luminance` is documented as SCALAR everywhere (DESIGN.md, AGENT_GUIDE.md) but the executor now produces a per-pixel `(H,W)` array** (`graph.py:875`). Side effect: implicit scalar inheritance of luminance is dead code — the harvest filter (`isinstance(_v, (int, float))`, `graph.py:583`) silently drops it. Either restore a scalar `luminance` alongside a `luma_field`, or update both docs and the inheritance path. Right now the flagship example of the named-attribute model ("luminance → brightness") doesn't work.
- **DESIGN.md port colors disagree with AGENT_GUIDE.md and `port_types.py`** (white/yellow/blue/orange vs blue/gray/green/orange). Trivial, but these docs are the agent contract — they must agree.
- **`image-pipeline-dev-report.md` says `nd_runner.py` is dead** — it isn't; the Node Doctor endpoints spawn it (`server.py:1394`). Update the report or the audit trail decays.
- **Name-scored implicit injection contradicts "no hidden state, no post-hoc magic"** (AGENT_GUIDE.md). `_score_param` substring matching (`graph.py:151-175`) silently rewrites params from upstream scalars, and payload inheritance cascades them transitively — a scalar three nodes upstream can mutate a param via a substring match (e.g. `speed` → `wind_speed`). For an agent-evolving system this is the most likely source of "why did my param change" confusion. Recommendation: keep exact-name + declared-synonym matches, drop bare substring matching, and log every implicit injection into the payload manifest so the wire inspector shows it.

### 1.3 Performance review — why live mode currently can't hit frame rate

Measured by code inspection of the per-frame path in `GraphExecutor.execute()` + `_run_graph_job` / `_live_loop`:

**P-1 — The disk round-trip tax (the big one).** Per node, per frame, even with `in_memory=True`:
1. Executor writes upstream image → `_input.png` (PNG encode, `graph.py:738-742`);
2. method decodes it (`load_input`), computes, PNG-encodes its output (`save()`);
3. executor PNG-encodes the returned array *again* to `node_dir` (`graph.py:852-856`);
4. writes sidecar `.npy`s + `scalars.json`;
5. merge ports write `_image_a.png`/`_image_b.png` which the Blend node immediately decodes.

PNG encode/decode at 768×512 costs ~10-40 ms each; a 5-node graph does ~10-15 of them per frame → the graph is I/O-bound before any math runs. A 30 fps live target gives a 33 ms budget for *everything*.

**P-2 — `get_all_node_defs()` called per edge, per node, per frame** (`graph.py:629`). It rebuilds NodeDefs for the entire ~200-method registry to answer "is this dst_port an IMAGE port". This is O(edges × registry) each frame. Must be computed once per graph (or per hot-reload) and looked up.

**P-3 — Architecture A is bake-only and pathological in live mode.** The sim cache holds a fixed-length frame list; when the live loop's monotonically increasing `frame` exceeds the cache length (`graph.py:443-456`), the executor **re-runs the entire simulation every frame** and displays its final frame — full sim cost per live frame, frozen output. Batch sims fundamentally cannot serve an infinite live timeline; they need a steppable contract (Part 2, Phase 2).

**P-4 — Live timeline is frozen.** `_live_loop` calls `execute(..., frames=1)`, so `make_timeline(total_frames=1)` yields `t=0, phase=0` forever. Every time-driven (Architecture B) method renders the same frame; only `frame`/`frame_seed` consumers animate. Live mode needs a wall-clock/frame-count timeline, not a normalized render-range one.

**P-5 — Per-frame recompute of static work.**
- Expression strings re-parse AST + compile every frame (`expr.py:86-96`) — cache compiled code objects keyed by the string.
- `_inject_typed` allocates a full `(H,W)` `np.full` array for **every** scalar injection whether or not the method reads `_field_<param>` (`graph.py:217`) — ~1.5 MB alloc per scalar per node per frame.
- Topo sort, terminal detection, GraphNode/GraphEdge dataclass re-construction from dicts — all repeated per frame for an unchanged graph.
- `time.sleep(0.01)` is the only pacing in the live loop — no fixed timestep, no fps target, no frame-drop policy.

**P-6 — Everything is sequential.** Independent branches of the DAG execute one at a time on one thread. NumPy releases the GIL for large ops, so a thread pool over ready nodes is a real (~2-3×) win for wide graphs. The CLI's `run_parallel` exists but is a separate engine the server never uses (long-standing two-engines split).

**P-7 — Method-level hotspots** (spot-checked): `floyd_steinberg_dither` is a pure-Python double pixel loop (`utils.py:484-511`) — seconds per frame; `quantize_to_palette` is chunked (good); several sims use per-pixel Python loops that would need vectorization or numba if used live. Not blocking — live graphs can simply avoid slow nodes — but worth a per-method cost annotation (see Phase 5).

### 1.4 Server & operational review

- **Security: the server is RCE-by-design on 0.0.0.0.** `--tunnel` (ngrok), `/admin/restart` (`os.execv`), and `/api/node-doctor/apply` (writes arbitrary Python that watchdog hot-loads) are each fine for a local tool, but combined with public binding they are remote code execution with zero auth. At minimum: bind 127.0.0.1 by default, require a token when `--tunnel` is used.
- **Hot-reload half-invalidates caches.** After a method file reload, `GraphExecutor._sim_cache`/`_sim_params_hash` and any compiled graph state keep serving results from the old code (params_hash only covers node params). Hot-reload should broadcast an invalidation the executors subscribe to.
- **`_GRAPH_SESSION_DIR` stale-cache risk** (already flagged in the dev report): dirty-skip trusts node ids across sessions; recycled ids serve stale PNGs. A session manifest (graph hash → node ids) would close it.
- **Duplicate SSE stream endpoints** (`stream_job` / `stream_graph_job`) are 90% identical — merge.
- **Repo hygiene:** `methods/cli_tools.nd-bak-728734db.py` still sits in the methods dir (flagged 2026-06-24, still present); ~60+ rendered MP4/PNG artifacts live at repo root and should move under `output/` or be gitignored; `datetime.utcnow()` is deprecated; `_topo_sort`'s `non_feedback_edges` is computed and unused; `ran[node_id] = True` is set twice (`graph.py:885, 897`).
- **The `_DynDim` canvas proxy + PIL/cv2/numpy monkeypatches** (`utils.py:39-228`) are impressively engineered but globally patch `np.mgrid.__getitem__`, `PIL.Image.new/resize`, `cv2.resize/warpAffine/warpPerspective` for the whole process. This is exactly the kind of invisible magic AGENT_GUIDE.md forbids in methods, living in core. It works today; long-term, passing an explicit `ctx.canvas` (or resolution param) through the new node contract (Phase 2) lets these patches retire gradually.

### 1.5 What's genuinely good (keep)

- The `@method` registry + decorator metadata model is clean, and hot-reload with SSE push is a real agent-velocity feature.
- The sidecar protocol as a **contract** (named, typed, declared outputs) is the right abstraction — the problem is only that disk is its *transport* on the hot path.
- `channels.py` already proves the right execution model: stateless dict-returning nodes with zero disk I/O. This is the seed of the live architecture.
- The per-node error placeholder + `node-error` SSE gives excellent failure visibility.
- `expr.py` is a properly sandboxed AST whitelist (no attribute access, no subscripts, whitelisted calls only) — good.
- Thread-dispatch stdout proxy is a clean solution to concurrent job logging.

---

## Part 2 — Optimization Plan: One Engine, Two Clocks

### Guiding principle

Keep the **named-attribute payload as the contract**; change its **transport**. Houdini and TouchDesigner differ exactly here: Houdini bakes to disk for auditability, TD cooks in memory per frame. We want both from one engine:

- **Render mode** = deterministic frame range, disk audit trail on (PNG + sidecars per node, as today).
- **Live mode** = wall-clock timeline, in-memory payload bus, disk off (or async, sampled).

The sidecar files stop being the *mechanism* of dataflow and become a *debug/audit projection* of the in-memory payload — which preserves the "full graph state auditable on disk" philosophy where it matters (render, debugging) without paying for it 30 times a second.

### Phase 0 — Correctness (do first, ~1 session)

Fix BUG-1…BUG-8 from §1.1. None require design changes; all are local. Add a regression test for each in `image_pipeline/tests/` (the field-wire crash and the seed determinism check especially). Also reconcile the `luminance` contract (§1.2) — decide: `luminance` = scalar mean (restores implicit inheritance), plus optional `luma` FIELD output for per-pixel consumers. Update DESIGN.md + AGENT_GUIDE.md in the same commit.

### Phase 1 — Compile the graph; make memory the bus (biggest win/effort ratio)

**1a. `CompiledGraph`.** On graph submit (or change), build once: topo order, terminal, per-node `NodeDef` lookup, per-edge routing decisions (which branch of the edge dispatch applies), compiled expression code objects, stable node seeds. `execute(frame)` then just walks the plan. Eliminates P-2 and most of P-5. The dirty flag becomes "recompile plan" (structure change) vs "recook node" (param change) — same split TD makes.

**1b. In-memory payload bus.** `flat_outputs` already is the bus; finish the job:
- New-contract methods **return** `{"image": arr, "field": arr, "scalar_x": v}` dicts (executor support already exists at `graph.py:823-826`). Kill the `save()`-monkeypatch capture (BUG-5 dies with it).
- Upstream images pass as `_input_image` ndarray only (already injected); `_input.png` written **only in render/debug mode**.
- Merge ports (`image_a/field_a/…`) pass ndarrays directly (`run_params["image_a"]`) instead of temp files; update the 5 compositing nodes (they're the only consumers).
- Node PNG + sidecar writes move behind `if self.audit_to_disk:` — on in render mode, off in live (optionally async-sampled every N frames so the wire inspector still works live).

**1c. Method migration is incremental.** The executor already accepts dict-return, ndarray-return, PIL-return, and disk-only legacy methods. Add a per-method capability flag to `MethodMeta` (auto-detected on first call, or declared: `contract="memory"`) so the executor knows which methods still force a disk round-trip. `tools/audit_methods.py` gains a check; agents migrate methods opportunistically. AGENT_GUIDE.md gets the new preferred contract with the old one documented as legacy.

Expected result: a typical 5-node Architecture-B graph drops from ~150-300 ms/frame of pure I/O overhead to <5 ms overhead — live-viable for stateless graphs immediately.

### Phase 2 — Steppable node contract (the real live unlock)

Architecture A (bake-the-whole-sim) can't serve an infinite timeline (P-3). Introduce **Architecture C: stateful stepping** — the TD cook model:

```python
@method(id="…", …, execution="stepped")
def run(out_dir, seed, params):
    state = params.get("_state")          # None on first cook / after reset
    if state is None:
        state = init_sim(seed, params)    # allocate grids, particles …
    state = step_sim(state, params, dt=params["_timeline"].dt)
    return {"image": render(state), "particles": state["pts"], "_state": state}
```

- Executor keeps `self._node_state[node_id]`; passes it back each cook; clears it when the node is dirtied (param change = live-tweakable without reset if the method reads params each step; structural params can declare `resets_state=True`).
- **Adapters:** Arch B needs nothing (stateless). Arch A methods get a generic wrapper first — run the internal loop **one `capture_frame` at a time** via a generator/greenlet-free trick is fragile, so instead: migrate the ~15 most-used sims (Gray-Scott, Boids, Physarum, FHN, DLA…) by factoring their existing loop body into `init/step` — mechanical refactors, ideal agent tasks with a written recipe in AGENT_GUIDE.md. Unmigrated Arch-A nodes remain render-only (executor marks them `live_capable=False`; UI badges them).
- This also fixes prebake cleanly (`prebake` = call `step` N times at init) and makes feedback edges natural (state is explicit, previous-frame outputs stay on the bus).

### Phase 3 — The live runtime

Replace the `_live_loop` prototype with a proper session object:

- **`LiveSession`**: owns a `CompiledGraph`, node state store, its own thread, target fps, and a **command queue**. Param tweaks/graph edits from the UI post commands; the loop applies them between frames (no shared-dict mutation from the request thread — fixes BUG-4 and makes knob-turning latency one frame).
- **Clock:** fixed-timestep `Timeline` variant for live: `dt = speed / fps`, `global_frame` monotonic, `t`/`phase` derived from wall time or beat clock (the `__timeline__` node's `beat`/`segment` outputs finally get a real driver; MIDI/OSC input becomes a future channel node). Fixes P-4.
- **Pacing:** frame budget = 1/fps; if a cook overruns, skip sleep and report; expose per-node cook times in a `cook_stats` payload (SSE) so the UI can show a TD-style per-node ms overlay — invaluable for agents optimizing methods.
- **Output:** keep MJPEG (it works and is simple); add a WebSocket channel for `cook_stats` + scalar taps (wire inspector live values). WebRTC/WebGL is a later luxury.
- **Selective recooking in live:** only nodes whose inputs changed (upstream cooked, param command, time-dependent) re-cook; static upstream branches cook once and hold — this is the dirty-flag idea applied per-frame, and with the compiled plan it's a cheap bitmask walk.

### Phase 4 — Throughput depth (after live works)

- **Parallel branch cooking:** thread pool over ready nodes in the compiled plan (NumPy/cv2 release the GIL). Wide graphs ~2-3×.
- **Preview-resolution cooking:** live sessions cook at, e.g., 384×256 via the existing `set_canvas` ContextVar and render mode keeps full res — biggest single fps lever, nearly free thanks to the canvas system.
- **Memory discipline:** reuse per-node output buffers where shapes are static; cap `_sim_cache` with an LRU byte budget (BUG-8a); float32 end-to-end (already mostly true).
- **Method vectorization backlog:** annotate per-method cook cost from the Node Tester runs; agents burn down the slow list (floyd_steinberg → error-diffusion via numba or ordered-dither default, etc.).
- **GPU lane (optional):** `moderngl` is already wired for method #82; a `FIELD`-in/`FIELD`-out GLSL wrangle node gives shader-speed post-processing without porting the sim library.

### Phase 5 — Agent-facing consolidation

- **Docs:** update DESIGN.md (transport vs contract, two clocks, Architecture C, audit-to-disk toggle) and AGENT_GUIDE.md (dict-return contract, `init/step` recipe, live-capability flag, determinism rules). Keep the pre-flight checklist authoritative.
- **Audit gate additions:** dict-return compliance, `_state` handling correctness (no module-level state), declared-vs-actual outputs (exists), per-method `live_capable` accuracy.
- **Retire the second engine:** fold the CLI onto the same executor (`CompiledGraph` of a single node), keeping `runner.py`'s cache semantics as an executor option. One engine, one set of bugs.
- **Kill bare-substring implicit injection** (§1.2) and surface all implicit injections in the wire-payload manifest.

### Sequencing & effort

| Phase | Scope | Effort | Unlocks |
|---|---|---|---|
| 0 | 8 bug fixes + luminance contract + tests | 1-2 sessions | trust |
| 1 | CompiledGraph + memory bus + compositing nodes | 2-4 sessions | live-viable stateless graphs |
| 2 | stepped contract + 15 sim migrations | 3-6 sessions (parallelizable across agents) | live sims |
| 3 | LiveSession + live clock + cook stats | 2-3 sessions | real live mode |
| 4 | parallelism, preview res, memory caps | ongoing | fps headroom |
| 5 | docs, audit, engine unification | 1-2 sessions | agent evolvability |

Phases 0-1 are prerequisites for everything; 2 and 3 can proceed in parallel once 1 lands.

---

## Status update — 2026-07-08 (engine + streaming optimization pass)

Profiled the representative live graph (Noise #05 → Glitch #17 → Transform,
768×512, `image_pipeline/tests/profile_live.py`): **390.7 → 149.3 ms/frame
(2.6 → 6.7 fps uncapped)**. The remaining 89% is Procedural Noise's own math
(method backlog, P-7) — engine transport overhead is now ~6 ms/frame.

**Root cause found beyond the plan:** the executor's in-memory capture
monkeypatched the module attribute `utils.save`, but ~150 method files bind
`save` at import time (`from ...core.utils import save`) — so capture never
fired and every live frame paid PNG encode + write + decode read-back per
node (~113 ms each). Fixed by moving capture inside `utils.save()` itself
(per-thread sink, `utils.set_save_capture`), which also kills BUG-5.

Landed (all with regression tests in `tests/test_live_transport.py`):
- Phase 1b transport: `GraphExecutor(audit_to_disk=False)` = zero-disk live
  mode (live loop uses it); render jobs keep the audit trail. Merge ports
  (image_a/b, field_a/b, particles_a/b) now pass ndarrays in-memory; the 4
  compositing consumers prefer them (temp files only in audit mode).
- BUG-1 (field-wire ndarray crash), BUG-5 (save monkeypatch), BUG-6 (group
  sub-executors now cached + inherit live flags), BUG-7 (registry dunder-id
  sort), BUG-8a (sim-cache 1.5 GB byte cap), Arch-A cache-hit now handles
  list-of-dict sim results.
- P-5: expression code objects cached (`expr._COMPILED_CACHE`); scalar
  `_field_<param>` injection is a zero-copy `broadcast_to` view; `_input.png`
  legacy transport writes at PNG compress_level=1 (~10× faster encode).
- Streaming: JPEG via cv2 (`server._encode_jpeg`, PIL fallback) for MJPEG/WS/
  SSE; live loop reads render flag + node names from the per-frame doc
  snapshot (was stale request payload); `/render` endpoint uses one shared
  locked executor (was per-thread, up to N unbounded caches); fixed
  `getattr(_live_last_gid, '', '')` always-`''` bug that recreated the live
  executor (and flushed sim caches) on every live start.
- Fixed server import on eager-resolving pydantic (`GraphRequest` was defined
  200 lines after its first endpoint use); repointed two drifted tests off
  #18 (now Arch B) and pinned test_fidelity's canvas.

Suite: 179 passed / 0 failed (6 errors = playwright not installed).

**Fixed 2026-07-08 (user call: fix in place, accept the visual change):**
noise.py `_grad_at` computed `ix = (x % 1).astype(int64)` — always 0, so
lattice gradient noise sampled a single lattice cell (period-1 tile pattern
instead of Perlin). Rewrote `_lattice_gradient_noise` as proper Perlin
(corner dot products, smoothstep interpolation — a bare floor() index fix
would leave C0 seams at every lattice line) and fixed the same index bug in
`_value_noise`. Two more issues fixed in the same pass:
- **noise_type param was dead**: the morph fallback (`effective_morph_fade
  < 0`) never fired at the default morph=0.0, so every preset rendered as
  `type_cycle[0]` regardless of noise_type — all 13 type choices produced
  bit-identical output. Now morph <= 0 honors noise_type; morph > 0 keeps
  the type-cycle behavior for wired scalars.
- **P-7 perf**: `_grad_table`/`_value_table` hoisted to module level
  (the lru_cache was rebuilt every call, so tables regenerated per frame)
  and converted to float32; simplex math kept float32-clean (math.sqrt, no
  int64 promotion). Noise node 143.9 → 92.6 ms/frame; live graph 149.3 →
  109.4 ms/frame (6.7 → 9.1 fps uncapped).

Seed reproducibility break: previously rendered gradient/value fbm pieces
cannot be regenerated bit-exact from seed+params (renders on disk are
unaffected). Suite: 169 passed / 0 failed after the change (playwright
tests still environment-skipped).

**Fixed 2026-07-08 (same in-place policy): arctan2 branch-cut seam in the
default-style colorizer.** The `style="normal"`/`palette="none"` branch
colorized with `sin(no*6 + theta*0.5)`; theta jumps 2π at the arctan2
branch cut (y=H/2, x<W/2), so the ×0.5 turned it into a π phase jump — a
hard sign flip in the red channel from the left edge to the image center.
Changed to `sin(no*6 + theta)` (integer theta coefficient ⇒ 2π-periodic ⇒
seamless). Verified at 768×512, scale=4, seed 42: the y=256 row-pair diff
for x<384 now matches background row statistics; residual maxima sit at
x=381–383 (the theta pole at the center, inherent to any angular term) and
affect all channels equally. Output-changing for every default-style noise
preset.
