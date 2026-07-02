# Full Code Audit — 2026-07-02

*Verified against source at `ea2e561` (merge of PR #3). Every claim below was checked by reading the code, running the server, executing a real graph end-to-end, and running the repo's own tooling — not inferred from filenames or docs. Supersedes the method-level portions of `CODEBASE_AUDIT.md` (2026-06-20); that document's cleanup section remains accurate history.*

**Audit lens:** the stated goal — a node-based editor taking the best of Houdini (named attribute payloads, procedural cooking) and TouchDesigner (live, always-running, CHOP-style channel operators), infused with LLM agents that evolve the tool with user input, optimized for live/real-time rendering. Each finding is graded against that goal, per `DESIGN.md` and `AGENT_GUIDE.md`.

---

## Verdict

The Houdini half of the vision is genuinely built and working: typed named-attribute payloads, sidecar protocol, implicit scalar inheritance, dirty flags, an open port-type registry, group nodes, per-param keyframes, and a 174-node library behind a coherent `@method` registry. The graph executes end-to-end (verified live: noise → ASCII-art graph rendered and streamed over SSE).

The TouchDesigner half and the LLM half are both **half-wired**. The MJPEG live-preview pipeline exists on the server but is connected to nothing in the UI — the "📺 Live" buttons have no event listeners. The LLM integration (Node Doctor) works only on one specific machine because it shells out to a hardcoded personal agent install (`~/.hermes/...`). And at HEAD, **the server did not boot at all**: the chord_bot cherry-pick renamed `types.py` → `chord_types.py` in its source branch but the renamed file never made it into the commit, so `from image_pipeline.server import app` raised `ModuleNotFoundError`. That is fixed on this branch (see P0).

The single biggest architectural risk is not any one bug — it's that **the three pillars are drifting apart**: real-time infrastructure that the UI doesn't use, an executor whose flagship selective-recook feature the server force-disables, three parallel animation systems, and a registry that silently drops nodes on ID collisions (it has already eaten two).

---

## P0 — Broken at HEAD

### 1. Server could not boot: missing `chord_bot/chord_types.py` — **FIXED on this branch**
Commit `a41e75f` ("cherry-pick: flizz") deleted `chord_bot/types.py` (228 lines) and rewrote every import in `chord_bot/` to `from ..chord_types import …`, but the renamed `chord_types.py` was never added. `server.py:172` mounts chord_bot at import time, so the **entire image pipeline server** failed with `ModuleNotFoundError: No module named 'chord_bot.chord_types'`.

Fix applied: restored the module from `feb0f2f:chord_bot/types.py` as `chord_bot/chord_types.py`. All 8 imported symbols (`HarmonicState`, `SequenceEntry`, `QUALITY_INTERVALS`, `build_chord_name`, `compute_bass`, `compute_voices`, `note_to_pc`, `pc_to_note`) exist in it. Verified: server imports clean, 174 methods register, and all **120 chord_bot tests pass** (`pytest chord_bot/tests`).

Structural lesson: mounting chord_bot inside the image server at module import time couples two unrelated apps' boot paths. A guarded import (`try/except` with a log line, or lazy mount) would have kept the image editor alive.

### 2. Registry silently drops nodes on duplicate IDs — two live collisions
`registry.py:113` is `_registry[id] = meta` — last-write-wins, no warning. The 2026-06-20 audit flagged this exact footgun (the method-#18 incident); it has since fired **twice more**:

| ID | Loser (silently gone) | Winner (import order) |
|----|----|----|
| **83** | `simulations/langtons_ant.py` — Langton's Ant, still listed as #83 in `DESIGN.md`'s outputs table | `p5_sketches.py` — "p5.js Sketch" |
| **146** | `simulations/sand_dune_migration.py` (also not imported, see below) | `simulations/cahn_hilliard.py` — "AC + PM Diffusion" |

Langton's Ant is a real regression: an existing node users may have in saved graphs now resolves to a completely different method (which additionally requires `playwright`, not in `requirements.txt`). Any saved graph containing node #83 now silently produces different output — the exact failure mode `AGENT_GUIDE.md`'s "never reuse an ID" rule exists to prevent.

**Fix:** make `method()` raise (or at minimum log loudly) on duplicate registration; re-home p5.js Sketch to a fresh ID from `tools/next_id.py`; restore #83 to Langton's Ant.

### 3. Five method files are dead weight — written but never imported
Declared `@method` IDs that never reach the registry because no `__init__.py` imports their module:

- `simulations/chua_lattice.py` (#150), `brusselator.py` (#164), `turing_morphogenesis.py` (#169), `pfc.py` (#170), `sand_dune_migration.py` (#146)
- `methods/system/timeline_node.py` (`__timeline__`) — see Timeline finding below

These are complete implementations invisible to the editor. Either wire them into `simulations/__init__.py` (checking #146/#150 for further collisions first) or delete them. The mismatch (195 `@method` decorators in source, 174 in the registry; remainder are the five above plus `.nd-bak` copies) means "the code in the repo" and "the nodes in the editor" have diverged — fatal to the mutual-legibility contract, since an agent reading the file has no signal that the node doesn't exist.

---

## P1 — Core executor (`core/graph.py`)

### 4. Latent `NameError` on the Architecture-B list path
`params_hash` is assigned only inside the `if arch == "A":` block (graph.py:438) but read unconditionally when a method returns a list (graph.py:818). `detect_architecture()` is heuristic (n_frames param / tags); any method that returns a frame list without matching the heuristic crashes the whole graph run with `NameError`, not a node error. Compute `params_hash` next to `sim_cache_key` (graph.py:433), which is already unconditional.

### 5. Reproducibility is broken across restarts: `hash(node_id)`
`node_seed = seed + frame + (hash(node_id) & 0xFFFF)` (graph.py:430, 754). Python string hashing is randomized per process (`PYTHONHASHSEED`), so the same graph + seed produces **different node seeds every server restart**. This directly violates the guide's contract: "Identical seed + params must produce identical output." Use a stable digest, e.g. `int.from_bytes(hashlib.sha1(node_id.encode()).digest()[:2], 'big')`.

### 6. Selective recooking is force-disabled in the main run path
`_run_graph_job` sets `n["dirty"] = True` for every node on every request (server.py:963-964) to work around the disk cache returning stale frames during animation. The result: the headline dirty-flag feature (`DESIGN.md` §Dirty Flag, the frontend's careful `dirty` bookkeeping at index.html:4471/4729) is dead code in `/api/graph/execute`. Every param tweak recooks the entire graph. Additionally each request constructs a fresh `GraphExecutor`, so the in-memory `_sim_cache` for Architecture-A sims is discarded between runs — interactive tweaking re-runs whole simulations.

For the real-time goal this is the highest-leverage fix: keep a **persistent executor per session** (or per saved graph), honor client dirty flags for single-frame runs, and only force-dirty for multi-frame animation renders (the case the workaround was written for).

### 7. O(edges × methods) work per frame: `get_all_node_defs()` inside the edge loop
graph.py:629 rebuilds node defs for **all 174 methods** once per incoming edge per node per frame, just to check one port's type. At 60 nodes × 2 edges this is ~20k `_make_node_def` calls per frame. Hoist one `get_all_node_defs()` per `execute()` call (or cache on the executor; invalidate on hot-reload).

### 8. Disk I/O in the hot loop
Per node per frame the executor writes the output PNG (graph.py:852-856), sidecar `.npy`s, and `_input.png` for image edges — even with `in_memory=True` (the capture hook still calls the original save, and merge-port injection writes temp PNGs at graph.py:641-663). For a 10-node graph at 24 fps that's hundreds of PNG encodes/second aspiration-wise; today it's the main reason live mode can't approach real-time. The payloads are already in memory (`flat_outputs`); disk writes should be an explicit "checkpoint/export" concern, not the default per-frame path — exactly the `DESIGN.md` visibility contract ("the server must not hold long-lived numpy arrays" needs revisiting for the live path; it currently holds them anyway in `_sim_cache`).

### 9. Global monkeypatch of `utils.save` is not thread-safe
In-memory capture swaps `image_pipeline.core.utils.save` module-wide (graph.py:774-783). Two concurrent jobs (e.g. a live sim + a render-sequence) race: node A's frames can be captured under node B's id, and the restore can clobber the other job's patch. Same story for the PIL/cv2/numpy patches in `utils.py` (see #17). Route capture through a `ContextVar` like `set_canvas` already does.

### 10. Payload-type inconsistencies against the documented contract
- `field` falls back to the RGB image: `"field": extra_outputs.get("field", arr)` (graph.py:879, and the cached path at 411). The contract says FIELD is `(H, W)` float32; consumers written to the contract get `(H, W, 3)` silently.
- `luminance` is a per-pixel `(H, W)` array on the normal and cached paths (graph.py:875, 410) but a scalar `float` on the Arch-A cache-hit path (graph.py:526). Docs (`DESIGN.md`, `AGENT_GUIDE.md`) still say SCALAR float. Downstream `isinstance(v, (int, float))` filters (scalar inheritance, graph.py:583) silently skip the ndarray form, so inheritance behaves differently depending on which code path produced the frame.
- Terminal selection disagrees between layers: executor picks the **last** render-flagged node in topo order (graph.py:977-981); the server overrides with the **first** in list order (server.py:994, 920, 1174). With two render flags set, preview and render can show different nodes.

### 11. Name-scoring injection contradicts the "no hidden magic" philosophy
`_score_param`'s substring fallback plus the "no name match → inject into first eligible param" fallback (graph.py:723-725) means a wire labeled `energy` can silently drive `n_frames`, and upstream scalar inheritance can overwrite an int param the user set (round()'d, no UI indication). This is Houdini-flavored convenience but is exactly the "post-hoc magic" `AGENT_GUIDE.md` disavows ("What a method writes is exactly what flows downstream — no hidden state"). Recommendation: keep exact-name and synonym matches; drop the blind first-eligible fallback; surface implicit injections in the wire tooltip so they're inspectable (the visibility contract's own remedy).

### 12. Timeline node: three-quarters wired
`graph.py` and `server.py` both special-case `method_id == "__timeline__"` (graph.py:331, server.py:968), and `methods/system/timeline_node.py` implements it — but the `system` package is never imported, so the node can't be registered; and if a graph JSON contained one anyway, `registry.get_meta` returns `None` → `GraphError: Unknown method` **before** the special-casing helps. Also "first node with `anim_speed != 1.0` wins for the global timeline" (graph.py:337-342) is spooky action at a distance — a per-node param silently mutating global time violates the context-separation principle. Register the node, and make global speed live only on Timeline.

---

## P1 — Real-time pillar (the TouchDesigner half)

### 13. The MJPEG live path is orphaned end-to-end
Server side is complete: `_push_live_frame` (server.py:72), `/api/live/stream` MJPEG multipart (server.py:610), `/api/live/frame.jpg` polling fallback (649), and `/api/graph/live` continuous-execution loop (890). Frontend side: two `📺 Live` buttons exist in the HTML (index.html:1294, 1323) with **zero JavaScript listeners** — no code in the UI references `graph-live-btn`, `/api/live/*`, or `/api/graph/live`. The flagship real-time feature ships as dead buttons. (The cherry-pick commit message claims "MJPEG live preview streaming + polling fallback" — the plumbing landed, the faucet didn't.)

### 14. `/api/graph/live` has no lifecycle management
- No guard against double-start: each POST spawns a new daemon loop; since the loop re-reads the module-global `_live_sim_cancel` each iteration, two loops end up watching the same event — both run, both push frames (flicker + double CPU) until one stop kills both.
- `except Exception: pass` (server.py:927-928) swallows every error silently — a broken graph in live mode just freezes the preview with no signal, the opposite of the node-error surfacing the batch path does well.
- The loop ignores client dirty state and re-cooks everything each iteration (compounds #6/#8).

### 15. The channels family is the right TouchDesigner move — finish its integration
`channels.py` (Counter, Ramp, LFO, Beats, Noise1D, Envelope, Math, Logic, Blend, Strobe, Burst, AgeHeat) is a genuine CHOP analog and the strongest recent addition. Gaps: they're batch-cooked like everything else (a 60 fps LFO cooked at SSE-roundtrip cadence isn't an LFO), and `_find_terminal` had to grow special cases to avoid picking them as sinks. When the live loop gets fixed, channels are the nodes that most deserve a cheap always-cook fast path (they're microseconds of math; skip their disk writes entirely).

### 16. Three parallel animation systems
1. `paramKeyframes` on `GraphNode`, evaluated in the executor with easing/Bézier (graph.py:241-279, 567-575) — the good one.
2. `animParams` linear from/to, interpolated server-side only in `/api/graph/render-sequence` (server.py:1107-1116, 1155-1168).
3. `/api/graph/keyframes` + `_keyframe_store` (server.py:63, 1511-1525) — in-memory, "would be persisted in a full implementation," plus a vestigial `GraphNode.keyframes` field nothing reads.

Same domain concept, three contracts, three failure modes; graphs animate differently depending on which endpoint renders them. Converge on `paramKeyframes` (it's the superset), translate `animParams` into it in one place, delete the keyframe-store endpoints.

### 17. `_DynDim` + PIL/cv2/numpy global monkeypatching (`core/utils.py:39-228`)
The dynamic canvas proxy is impressively engineered and well-commented, but it patches `PIL.Image.new/resize`, `cv2.resize/warpAffine/warpPerspective`, and `np.mgrid.__getitem__` **globally at import**, with silent `except Exception: pass` fallbacks. Risks: breaks unpredictably on library upgrades (behavior depends on private C-API dispatch details), affects every consumer in-process (including chord_bot and user code), and is invisible to an agent reading a method file — the largest single violation of the mutual-legibility contract in the codebase. Medium-term: pass `(w, h)` through `params` (the executor already injects `frame`/`frame_seed`/`_timeline`; canvas is the same kind of context) and deprecate the proxies file-by-file.

---

## P1 — LLM pillar

### 18. Node Doctor only works on one machine
`server.py:1395` hardcodes `_HERMES_PY = Path.home()/".hermes"/"hermes-agent"/"venv"/"bin"/"python"`, and `nd_runner.py` sys-path-injects that personal install and imports `hermes_cli` / `run_agent`. On any other machine (including this audit environment) every Node Doctor chat fails with a subprocess error. For the "constantly evolving with user input" pillar, the LLM backend is the one component that must be portable. Recommendation: make the backend pluggable — env-var-configured command, with a direct Anthropic-API fallback (`anthropic` SDK, streamed) so a fresh clone plus an API key gets a working Doctor. The surrounding design (SSE chat, apply/undo with backups, hot-reload on write, node context in the system prompt, batch-fix via node tester) is solid and worth keeping as-is.

### 19. Node Doctor backup litter is committed and scanned
Five `.nd-bak-*.py` files are tracked in git (`methods/cli_tools.nd-bak-728734db.py` — 977 lines — plus three `ascii_art` and one `gradient` backup in `codegen/`). They carry live `@method` decorators with duplicate IDs; today they're saved from the registry only because nothing imports them, but `tools/audit_methods.py` already scans them (inflating the report) and the watchdog hot-reloader fires on their creation. Add `*.nd-bak-*.py` to `.gitignore` and the audit tool's exclusions, delete the tracked ones, and write backups outside `methods/` (e.g. `output/nd-backups/`) so the watcher never sees them.

### 20. LLM legibility gap: 195/195 methods have empty `description`
`MethodMeta.description` exists, is served by `/api/node-defs`, is shown in the node-header tooltip, and is fed to Node Doctor's context — and every single method leaves it blank (`tools/audit_methods.py` output). For a system whose premise is "an agent reads a node and understands it," one-line descriptions are the cheapest possible win, and they're exactly the kind of task the Node Doctor / a batch agent should backfill.

### 21. The self-audit gate is red
`.pre-commit-config.yaml` runs `tools/audit_methods.py --fail-on-violations`, which currently exits 1 (2 confirmed sidecar gaps: methods writing `write_scalars` keys not declared in `outputs=`, per the regenerated `tools/audit_report.md`). Either fix the two gaps or the gate teaches people to skip it. Also the committed `tools/audit_report.{md,json}` were stale (pre-merge, 175 methods); regenerated on this branch (195).

### 22. Security: the evolve-yourself API is unauthenticated RCE if tunneled
By design, `POST /api/node-doctor/apply` writes arbitrary Python into `methods/` and the watchdog hot-reload **executes it on import**; `POST /admin/restart` re-execs the process; nothing requires auth. That's fine on localhost — but `server.py --tunnel` (pyngrok) and `scripts/tunnel.sh` publish the port. Anyone with the URL owns the machine. Minimum: a shared-secret header checked on `/api/node-doctor/*`, `/admin/*`, and `/api/graph/live` whenever a tunnel is active, and a startup warning. (`core/expr.py`, by contrast, is a properly whitelisted AST evaluator — good.)

---

## P2 — Consistency, hygiene, docs drift

### 23. `DESIGN.md` no longer matches the code
- Port colors: doc says white/yellow/blue/orange/grey; `core/port_types.py` says IMAGE=blue, SCALAR=gray, FIELD=green, PARTICLES=orange, MASK=white, plus COLORMAP (absent from the doc entirely). `AGENT_GUIDE.md` matches the code; `DESIGN.md` doesn't.
- "Port types are declared in `PortType(str, Enum)` in `core/graph.py`" — they now live in the open registry in `core/port_types.py` (a genuine improvement the doc doesn't know about).
- The named-outputs table lists #83 as Langton's Ant (see #2) and omits MASK outputs.
- "Planned Extensions" still lists the wire inspector and dirty flags as future; both are implemented (one of them then disabled, see #6).
Since `DESIGN.md` is "the authoritative architecture document" that agents are told to trust, drift here is worse than drift anywhere else.

### 24. chord_bot is an architecture fork, not a shared kernel
chord_bot re-implements the entire node stack — its own `registry.py`, `executor.py`, `port_types.py`, `keyframes.py`, UI — sharing zero code with `image_pipeline/core`. As proof the node model generalizes to another domain (music), it's encouraging; as a codebase it's a second copy of every kernel concept that will drift (its executor already diverged). Decide explicitly: extract a shared node-graph kernel (registry + topo-sort executor + port registry are all domain-agnostic), or declare chord_bot a deliberately independent experiment and unmount it from the image server's boot path (see #1). Also: `chord_bot/chord_bot.egg-info/` is committed (build artifact — delete, gitignore).

### 25. Repo-root scratch and vault litter
- One-off experiment scripts at root: `gs_portrait.py`, `gs_portrait_hires.py`, `gs_build.py`, `gs_clean.py`, `gen_blobs.py`, `_param_grid.py`, `_regenerate_grids.py`, `_run_all_grids.sh` — all Gray-Scott/grid dev scratch importing the pipeline. Move to `scripts/dev/` or delete.
- `.obsidian/` committed at root **and** inside `image_pipeline/` (editor workspace state).
- `.gitignore` globally ignores `*.html` and `*.txt` — the actual UI (`ui/index.html`) and `requirements.txt` are only tracked because they were force-added; any **new** HTML file or txt asset will silently not be committed. Narrow these patterns to the output dirs they were meant for.
- `ui/__init__.py` — a stray Python package marker in the frontend directory.
- Stale planning docs (`PHASE1_PLAN.md` — largely implemented; `SKILL_UPDATE_PROMPT.md`; `docs/plans/2026-05-19-…refactor-plan.md` — describes the deleted Streamlit app; `README.md` — still describes the Streamlit/vault "Command Center" era and doesn't mention the node editor at all). The README is the first thing a new agent reads and it describes a product that was removed on 2026-06-20.

### 26. Smaller server items
- `/api/graph/wire-payload/{job_id}/…` requires `job_id ∈ _jobs` but reads from the session dir; after the 1-hour job eviction the tooltip 404s even though the payload files are still on disk. Drop the job check or key on the session.
- `_evict_old_jobs` never evicts jobs stuck in `status="running"` — a crashed thread leaks its queue forever.
- Node-tester globals `_test_in_progress`/`_test_cancelled` are unsynchronized module globals (double-POST race).
- `stream_job` and `stream_graph_job` are near-duplicates — one generator with an allowed-event set would do.
- `datetime.utcnow()` (deprecated) in save endpoints.
- `_topo_sort` computes `non_feedback_edges` and never uses it; duplicate `ran[node_id] = True` (graph.py:885, 897).

---

## What is genuinely good (keep, and build on)

- **The payload model.** Typed named-attribute payloads with sidecars + implicit scalar inheritance is exactly the right Houdini translation, and `flat_outputs` as the single source of truth per frame is clean.
- **The open port-type registry** (`core/port_types.py`) — adding COLORMAP without touching core proved the design.
- **The `@method` contract + `AGENT_GUIDE.md`.** The guide is the best document in the repo; the contract (always-write-a-PNG, `_`-prefix temps, declared outputs) is precisely what makes LLM extension safe. `tools/audit_methods.py` + `next_id.py` + pre-commit enforcement is the right self-checking instinct — it just needs to be green (#21) and collision-proof (#2).
- **Hot-reload → SSE `node-defs-updated`** — the editor updating itself when an agent edits a file is the "constantly evolving" pillar working today.
- **Node tester + batch-apply loop** — automated find-broken-nodes → LLM-fix → hot-reload is a real self-healing loop; once #18 lands it works anywhere.
- **`core/expr.py`** — a correctly whitelisted AST expression evaluator with client-side preview parity.
- **Error containment** — per-node tracebacks, dark-red placeholder frames, `node-error` SSE events surfaced on the node UI: failures stay local, graphs keep cooking.
- **chord_bot test discipline** — 120 passing tests; the image side should match it (currently only `image_pipeline/tests/test_fidelity.py` and an empty root `tests/`).

---

## Priority roadmap

| # | Action | Pillar | Effort |
|---|--------|--------|--------|
| 1 | ~~Restore `chord_bot/chord_types.py`~~ **done on this branch**; add a CI smoke test: `python -c "from image_pipeline.server import app"` | all | done / tiny |
| 2 | Registry: raise on duplicate ID; re-home p5 #83, restore Langton's Ant; wire or delete the 5 unregistered method files | Houdini | small |
| 3 | Fix `params_hash` NameError + stable node seeds (`hashlib`, not `hash()`) | Houdini | tiny |
| 4 | Wire the Live buttons → `/api/graph/live` + `<img src=/api/live/stream>`; add double-start guard and error surfacing to the live loop | TouchDesigner | small |
| 5 | Persistent per-session executor; honor dirty flags in single-frame runs; hoist `get_all_node_defs()`; skip disk writes in live mode | TouchDesigner | medium |
| 6 | Pluggable Node Doctor backend with Anthropic-API fallback; move backups out of `methods/`; delete committed `.nd-bak` files | LLM | small |
| 7 | Auth token for mutating endpoints when tunneled | LLM/safety | small |
| 8 | Converge the three animation systems on `paramKeyframes` | TouchDesigner | medium |
| 9 | Update `DESIGN.md` (ports, colors, MASK/COLORMAP, #83) and rewrite `README.md` for the node editor | all | small |
| 10 | Backfill method `description`s (agent batch job); green the pre-commit audit gate | LLM | small |
| 11 | Decide chord_bot: shared kernel vs. independent app; unmount from image-server boot either way | arch | medium |
| 12 | Replace `_DynDim` global patches with explicit canvas in `params` (gradual) | legibility | large |

---

## Changes applied on this branch

1. **`chord_bot/chord_types.py` restored** from `feb0f2f:chord_bot/types.py` (P0 #1). Verified: server imports, 174 methods register, graph executes end-to-end, 120/120 chord_bot tests pass.
2. **`tools/audit_report.md` / `.json` regenerated** — the committed copies predated the PR #3 merge (175 vs 195 scanned methods).

Everything else in this report is findings only — no behavior was changed.
