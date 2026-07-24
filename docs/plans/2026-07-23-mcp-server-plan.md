# MCP Server for the Node Image Pipeline — 2026-07-23

Goal: let an LLM client (Claude Code, Claude Desktop, any MCP host) discover the node
library, author and mutate a graph, render it, and watch the result — through a typed
tool surface instead of hand-rolled `curl` against 50 FastAPI routes.

**The claim this plan rests on: most of the server already exists.** `server.py` carries a
half-built agent layer — a shared graph document, fine-grained patch ops, a broadcast
channel, and a synchronous headless render whose docstring reads *"For agent/programmatic
use"*. Nothing in it is reachable from a tool call today. The MCP server is mostly a typed,
budget-aware skin over that layer, plus six gaps it exposes (G1–G6).

**One finding outranks the plan itself.** Auditing what a headless agent render would hit
turned up a confirmed determinism break on the Architecture-A path: the sim cache's key omits
`frame`, which its cached content depends on, so the same graph at the same seed renders a
different frame 0 depending on which frame was rendered first. It predates MCP; agent
rendering just makes it certain instead of occasional. See "Prerequisite" below — P1 blocks
phase 2.

---

## What already exists (measured 2026-07-23, server running on :7860)

| Substrate | Where | State |
|---|---|---|
| Shared graph doc, `GET/PUT /api/graph/{gid}` | `server.py:1191`, `:1199` | Works. `_touch_graph_meta(doc, by or "agent")` — the "agent" author is already a first-class concept. |
| Fine-grained ops, `PATCH /api/graph/{gid}` | `server.py:1215` | Works. 7 ops: `add_node`, `update_node`, `remove_node`, `add_edge`, `remove_edge`, `set_canvas`, `clear`. |
| Mutation broadcast, `WS /api/graph/ws` | `server.py:1268` | Works server-side. **No subscriber** — see gap G4. |
| Headless sync render, `GET /api/graph/{gid}/render` | `server.py:1343` | Works. Raw PNG/JPEG bytes, warm shared executor, 3D graphs proxied to the Node sidecar. |
| Live loop reads the shared doc every frame | `server.py:1660` | Works. **This is the payoff** — see "Why the live loop matters". |
| Node defs / methods introspection | `/api/node-defs`, `/api/methods` | Works, and is far too large — see the budget table. |
| Diagnostics (per-node timings, errors, cache) | `/api/graph/diagnostics` | Works. |
| Saved graph corpus | `/api/graph/saved` | 48 saved graphs — a ready-made few-shot library. |
| Node source read | `/api/node-doctor/source/{id}` | Works, read-only. |

The editor UI never touches any of it. `ui/js/graph.js` keeps the graph in browser memory
and POSTs `nodes`/`edges` inline to `/api/graph/execute` and `/api/graph/live`; grepping
`ui/js` for `/api/graph/ws`, `PUT /api/graph`, or `PATCH` returns nothing. So the shared-doc
layer is currently written only by the live-start path and read only by `/render`. An MCP
server is its first real consumer.

### Why the live loop matters

`live_graph_sim` persists the request graph into the shared doc and then *re-reads that doc
every frame*:

> "The live loop reads it every frame, so a clear (user or agent) stops the render."
> — `server.py:1690`

So: user hits 📺 Live in the editor → agent `PATCH`es the `active` doc → **the next frame the
user sees is the agent's edit**, with no UI change required. The user's node canvas won't
redraw (G4), but the rendered output stream is live-shared between human and agent from day
one. That is the demo, and it costs nothing to get.

---

## The hard constraint: discovery does not fit in a context window

Measured against the running server:

| Endpoint | Bytes | ≈ tokens |
|---|---:|---:|
| `/api/methods` | 580,920 | ~150k |
| `/api/node-defs` | 746,304 | ~190k |
| `/api/port-types` | 807 | ~200 |
| `/api/palettes` | 167 | ~40 |
| `/api/easing-presets` | 672 | ~170 |
| `/api/graph/saved` | 2,621 | ~650 |

528 registered methods across 15 categories (gpu_shaders 159, simulations 113, filters 87,
patterns 69, math_art 27, fractals 16, codegen 14, channels 12, compositing 10, cli_tools 8,
ml_models 7, io 3, p5_sketches 1, client_3d 1, system 1), carrying ~4,680 params between
them (2026-07-22 spatial audit). `/api/node-defs` returns 538 entries, not 528 — the extra
ten are pseudo-nodes (`__timeline__`, `__geometry__`, the client-3D set) that the agent must
know exist, since `__timeline__` params override the global timeline for the whole graph.

**No tool may return either payload whole.** This single fact drives the whole tool design:
discovery is search-then-detail, never enumerate. Any implementation that starts with
"return `/api/node-defs` and let the model filter" is dead on arrival, and this is the
failure mode to guard against in review.

---

## Architecture

**Python, FastMCP, stdio transport, HTTP client against the running pipeline server.**
Lives at `image_pipeline/mcp/`, ships as `grillmaster_mcp`.

```
image_pipeline/mcp/
  __init__.py
  server.py        FastMCP app, tool registration only — thin
  client.py        httpx client: base URL, X-Api-Token, timeouts, actionable errors
  index.py         cached compact node index (search + detail projection)
  render.py        image encoding, downscale, token-budget policy, disk sink
  models.py        Pydantic input models
  README.md        install + client config
```

### Why wrap HTTP rather than import the pipeline in-process

In-process (`from image_pipeline.core.graph import GraphExecutor`) is tempting and wrong here:

- **Two executors, two sim caches.** `GraphExecutor.SIM_CACHE_MAX_BYTES = 1_500_000_000`
  (`graph.py:486`). The server already runs up to two executors (live + render). An
  in-process MCP adds a third, unmanaged one — 4.5 GB worst case, and the per-clip
  protection invariant (never evict a current-graph sim mid-render) is per-executor, so the
  extra process gets none of it.
- **Duplicate import cost.** 528 methods, a moderngl GL context, and optional torch/diffusers
  imports, paid again per MCP client launch.
- **Split brain.** Method hot-reload, the live doc, `_jobs`, and the node-doctor backups all
  live in the server process. An in-process MCP would see a stale library the moment a node
  is edited.
- **The wrap is genuinely thin.** The endpoints already exist and are already documented as
  agent-facing.

Cost of wrapping: the pipeline server must be running. Handled explicitly — see "Lifecycle".

### Transport and security

- **stdio.** Single user, local, launched by the host as a subprocess. Streamable HTTP buys
  nothing here and adds an auth surface.
- Never log to stdout (stdio protocol channel); logging goes to stderr.
- `GRILLMASTER_API_TOKEN`, when set, is forwarded as `X-Api-Token` on mutating calls —
  same contract the UI uses via `localStorage['api-token']`.
- Base URL defaults to `http://127.0.0.1:7860`, overridable by `GRILLMASTER_URL`. Loopback
  only by default; refuse a non-loopback URL unless `GRILLMASTER_MCP_ALLOW_REMOTE=1`.

### Lifecycle

The MCP server does **not** spawn the pipeline server by default. On any connection failure,
tools return an actionable error, not a stack trace:

> `Pipeline server not reachable at http://127.0.0.1:7860. Start it with
> \`bash scripts/dashboard.sh\` (or \`.venv/bin/python -m image_pipeline.server\`), then retry.`

Optional `GRILLMASTER_MCP_AUTOSTART=1` spawns the launcher and polls `/health` for 30s.
Off by default: an MCP client silently booting a GPU service is a surprise.

---

## Tool surface

14 tools, `grill_` prefix, snake_case, action-first. Annotations on every tool.

### Discovery (read-only)

**`grill_search_nodes`** — `query`, `category`, `tags`, `limit=25`, `offset=0`.
Token match over id, name, category, tags, and description *where one exists* — only 176 of
538 node defs carry a description and `/api/methods` emits none at all (G6). Returns compact
rows: `{id, name, category, tags, summary, in_ports, out_ports}` — **no params**. Paginated
with `total`, `has_more`, `next_offset`. Budget target: ≤ 4 KB for a 25-row page.

**`grill_get_node_def`** — `method_ids: list[str]` (max 8).
Full definition per id: params (type, default, min/max, choices, `spatial`, `content`
flags), input/output ports, `param_ports`, description, `is_time_varying`, deprecation.
Batched, because wiring a graph needs several defs at once and one-at-a-time round-trips are
the main latency sink. Budget target: ≤ 3 KB per node.

**`grill_list_categories`** — no args. 15 categories with counts. ~400 bytes; the natural
entry point before a search.

The index behind these lives in `index.py`: fetch `/api/methods` + `/api/node-defs` once,
project to a compact form, cache in memory with a TTL and an invalidation hook (method
hot-reload changes the library under us — see risk R3). **No server change needed.**

### Graph authoring

**`grill_get_graph`** — `gid="active"`. Current doc, compact (nodes, edges, canvas, meta).

**`grill_patch_graph`** — `gid`, `ops: list[Op]`. Typed mirror of the server's 7 ops, each op
a discriminated Pydantic union so the model can't send `{"op": "add_node"}` with no node.
Runs validation (below) *before* forwarding, so a bad wire fails in milliseconds with a
reason rather than 30 seconds later as a black frame. Annotations: `readOnlyHint: false`,
`destructiveHint: false` (except `clear`/`remove_*`, called out in the description),
`idempotentHint: false`.

**`grill_replace_graph`** — `gid`, `nodes`, `edges`, `canvas`. Wholesale PUT. Destructive;
the description says so and says to prefer `grill_patch_graph`.

**`grill_validate_graph`** — `gid` or inline `nodes`/`edges`. **The highest-value tool in the
set and the one piece of real new capability.** Static checks, no cook:

- unknown `method_id`
- edge referencing a missing node or a port the node doesn't declare
- port type mismatch (IMAGE→SCALAR etc.), against `/api/port-types`
- cycles (the executor's topo sort would reject them, but only after setup)
- no render terminal / no `render: true` node and no inferable terminal
- dead islands — nodes that reach no terminal
- params outside declared min/max; unknown param names

These are exactly the node-graph generation invariants the shootout generator had to
enforce, rediscovered here for a different producer. An agent that can check its own wiring
before rendering is the difference between a 3-call build and a 20-call flail.

### Rendering

**`grill_render_frame`** — `gid`, `frame=0`, `seed`, `width`, `height`, `preview="small"`,
`timeout_s=60`. Returns an MCP `ImageContent` preview **plus** the full-res file path and
render timing. Preview policy in `render.py`:

| `preview` | long edge | JPEG q | ≈ tokens |
|---|---:|---:|---:|
| `none` | — | — | 0 (path only) |
| `small` (default) | 384 | 70 | ~8–12k |
| `full` | canvas | 85 | 40k+ |

Full-res bytes always land on disk under `output/mcp/` and the path is returned, so `full`
is rarely needed. The default is `small` deliberately: an agent iterating on a graph makes
many renders, and `full` on each one exhausts the window in a handful of turns.

**`grill_render_sequence`** — `gid`, `start_frame`, `end_frame`, `fps`, `output_name`.
Wraps the SSE endpoint, consumes the stream, returns a manifest (`frame count`, `dir`,
`video path` if encoded) and optionally one contact-sheet image. Long-running; the tool
description states expected duration so the host doesn't time out silently.

**`grill_get_diagnostics`** — no args. `/api/graph/diagnostics`: per-node cook ms, per-node
errors, cache hit/miss, mem vs disk edges, active node/edge counts. This is how an agent
answers "why is it black" and "why is it slow" without guessing.

### Live

**`grill_live`** — `action: start|stop|status`, `fps`, `fps_limit`, `seed`, `width`,
`height`. Always passes `graph_id="active"` and an empty `nodes` list on start, so the loop
reads the shared doc rather than a snapshot — that's what makes subsequent patches show up
in the user's stream. `status` returns the live stats block.

### Library

**`grill_saved_graphs`** — `action: list|load|save|delete`, `name`. 48 saved graphs are the
best available few-shot corpus for "how is a working graph of this kind wired". `load`
returns the doc; a separate call patches it into `active` (loading and installing are kept
distinct so the agent can inspect first).

**`grill_get_node_source`** — `method_id`. Source text + path from the node-doctor source
endpoint. Read-only, for debugging a node that misbehaves.

### Deliberately excluded from v1

| Endpoint | Why not |
|---|---|
| `/api/node-doctor/chat`, `/apply`, `/undo` | Rewrites method files on disk via a second LLM (Hermes). An MCP agent driving another agent to mutate the library is a loop worth designing separately, not smuggling into v1. |
| `/api/node-tester/batch-apply` | Same — bulk automated edits to method files. |
| `/api/assets/upload` | 512 MB raw-body upload; no good MCP shape, and no demonstrated agent need. |
| `/admin/restart` | Restarting the user's editor from a tool call. No. |
| `/api/generate` (single-method legacy path) | Superseded by graph render; two ways to render is two things to keep correct. |

### Resources (phase 3, optional)

`grillmaster://graph/active`, `grillmaster://node-def/{method_id}`, `grillmaster://saved/{name}`.
Hosts that support resources can pull these without spending a tool call. Strictly additive.

---

## Gaps in the existing server that this work exposes

**G1 — no validation endpoint.** `grill_validate_graph` needs one. Two options: implement
the checks MCP-side against the cached node index (no server change, but the logic drifts
from the executor's real rules), or add `POST /api/graph/validate` in `server.py` reusing
`_make_node_def` and `_topo_sort` (one source of truth, and the UI could use it too).
**Recommend the server endpoint** — the checks belong next to the executor that enforces
them, and the UI's node picker wants the same answers.

**G2 — `/render` hardcodes seed 42.** `server.py:1408` passes `42` to `_ensure_executor` and
`ex.execute(nodes, edges, 42, ...)`. An agent cannot vary the seed on a headless render at
all. Add a `seed: int = 42` query param, threaded to both call sites. Small, and the
seed-change path already flushes the sim cache correctly (`_ensure_executor`'s
`seed_changed` branch).

**G3 — `/render` has no deadline.** A heavy sim graph blocks the request until it finishes,
holding `_render_exec_lock` and hanging the agent's tool call with no feedback. This is a
known failure shape: renders hang *inside* a node, so a per-frame timeout misses it and only
a wall-clock deadline catches it. Add `timeout_s` with a hard deadline returning 504 plus
the last-known node from the executor's timing map, so the error names the culprit rather
than saying "timed out".

**G4 — the editor UI does not subscribe to `/api/graph/ws`.** Verified: no graph-doc fetch,
PUT, PATCH, or ws subscription anywhere in `ui/js`. Consequence: agent edits change what
Live renders but not what the user's canvas shows, so the canvas silently disagrees with the
output. Two honest options:

- **(a) Accept for v1.** Document it: "the agent edits the shared graph; press 📺 Live to see
  its work. The node canvas does not follow agent edits yet."
- **(b) Wire it (phase 4).** Subscribe `graph.js` to `/api/graph/ws`, apply `graph:patch` /
  `graph:replace` to the in-memory graph, and push the result through the existing
  history/undo path. Non-trivial: needs a rule for conflicts with in-flight local edits, and
  agent edits must not silently poison the user's undo stack.

Recommend shipping (a) and scoping (b) separately. (b) is what makes this feel magic, but it
is a UI-state problem, not an MCP problem, and bundling them means neither lands cleanly.

**G5 — no compact mode on `/api/methods`.** Handled MCP-side by `index.py` projection; no
server change. Revisit only if the UI wants the same compact index.

**G6 — two-thirds of the library has no description.** Measured: 176/538 node defs carry a
`description`; `/api/methods` drops the field entirely even for those that have it. Search
by intent ("something that erodes a heightfield") therefore matches on name and tags only,
for most of the library. Three mitigations, cheapest first:

1. Emit `description` from `/api/methods` — one line, recovers 176 nodes for free.
2. Index the *param* descriptions, which are far better populated (every param dumped in
   spot checks had prose) as a secondary, lower-weighted match field. Costs index size, not
   response size.
3. Backfill node descriptions. 362 nodes is a real content task, not a code task — but it is
   the thing that makes intent search actually work, and `AGENT_GUIDE.md` could require a
   description on new nodes so the gap stops growing.

Not a blocker for phase 1; search works on names and tags. It is the ceiling on how good
discovery ever gets, so it belongs on the record now.

---

## Prerequisite — the Architecture-A identity model is broken

Found while auditing what an MCP render would actually hit. Not an MCP gap: these are
pre-existing defects that headless agent rendering makes far easier to trigger, because an
agent renders arbitrary frames in arbitrary order against the persistent `_render_exec_state`
executor. Shipping `grill_render_frame` on top of this bakes non-determinism into the tool
surface, so P1 below blocks phase 2.

### P1 — the Arch-A sim cache key omits `frame`, which the cached content depends on

`graph.py:787` computes `node_seed = seed + frame + _stable_node_offset(node_id)` and
`graph.py:948` cooks the **whole clip** with it. The cache key (`graph.py:790`) is
`(node_id, seed)` — `frame` is in the content but not in the key. Whichever frame first
misses the cache determines the clip that then serves every frame.

Measured 2026-07-23, same graph, seed 42, same params, same output frame 0:

| method | f0 on a fresh executor | f0 after rendering f3 | f0 after rendering f5 |
|---|---|---|---|
| 494 Screen-Space Fluid | `c5f2df5944` | `60753d348f` | `9fe1d96de5` |
| 954 Autostereogram | `05ea06f689` | `249de14d86` | `5432f77c1c` |
| 171 p5.js Sketch | `a1c5c8af50` | `7c8efc490d` | `d8278e2c0a` |

This breaks DESIGN.md's determinism contract, and it is reachable today: `_render_exec_state`
persists across `/render` calls, so frame 5 then frame 0 yields a different frame 0 than a
fresh server. Sequence renders with `start_frame > 0` and live sessions resumed after a
scrub hit the same path.

**Diagnosis:** `+ frame` is Architecture *B*'s seed rule — fresh noise per frame — applied to
A's cook, whose premise is that one cook serves all frames. Lines 787 and 1319 are identical
and it is correct only at 1319.

### P1a — DONE 2026-07-23

`+ frame` dropped on the A path (`graph.py`, Arch-A branch only; the B path at 1319 keeps
it, correctly). The clip is now a pure function of `(seed, params, node)` — exactly what the
existing cache key claims. The key was never wrong; the cook was.

- All **115** Arch-A nodes verified order-independent with the timeline phase held constant.
- Full suite: **8 failed / 2445 passed / 2 errors**, byte-identical to the same run at HEAD.
  Zero regressions. (The 8 are pre-existing: 2 GPU coverage audits, 6 order-dependent flakes
  in `test_lic_flow_coloring` that fail the same way with and without the change.)
- Guard: `image_pipeline/tests/test_arch_a_frame_order.py`.
- **Output-perturbing for Arch-A nodes** — reproducibility policy applies: fixed in place,
  break recorded here.

Probe hygiene, learned the hard way: node 91 ignores both seed and frame, and node 171
(p5.js Sketch) is nondeterministic run-to-run at a fixed seed. Both report "identical"
or "different" for reasons unrelated to the thing under test. 494 and 954 are the reliable
probes.

### P1b — OPEN. The same defect through `time` / `_timeline`

`sim_params["frame"]` was already pinned to 0 for the cook; `time` and `_timeline` were not.
Both are in `_VOLATILE_PARAM_KEYS` — deliberately excluded from the cache key so the clip is
not re-cooked every frame — yet both still reach the cook, so a method reading either cooks
a frame-dependent clip. **Measured: 40 of 115 Arch-A nodes**, including Lattice Boltzmann,
Lenia, Stable Fluids, N-Body Gravity, SPH, Metaballs, Ising, and the whole Lotka-Volterra
family.

The hole only opens when the timeline phase varies — `make_timeline` pins `t = 0.0` at
`total_frames == 1`, so single-frame renders are inert and **every sequence render and every
live playback is exposed**. A first sweep at `frames=1` reported the residual as 1 node; that
number was an artifact of the probe, not a measurement of the code.

**The obvious fix is wrong and was reverted.** Pinning `time`/`_timeline` to frame 0 closes
all 115 — and silently zeroes the spatial-param response of every sim that uses `time` as an
initial-condition input. Measured: `112.u_shear` 0.002724 → 0.000000 and `359.mu` 0.016258 →
0.000000, i.e. two of `test_spatial_params`' certified-SPATIAL params become MEAN_ONLY. The
capability loss is real and silent; the determinism gain is not worth it.

What that reveals is the actual root: **a node whose cooked clip legitimately depends on the
output frame is not a cook-once-and-replay node at all.** It is Architecture B wearing an
`n_frames` param, and the A path cannot fix it from the inside — pinning breaks the honest
`time` readers, and adding `time` to the cache key re-cooks the clip every frame, which is
Architecture B by a slower route. **P1b is therefore a sub-problem of P3 (declared
architecture), not an independent fix.** Sequencing: land P3's `@method(arch=...)`
declaration, reclassify the 40, then P1b closes by construction.

Bounded meanwhile by `test_channel_two_residual_does_not_grow` (marked `slow`), so a refactor
cannot quietly widen it.

### P2 — replay assumes a 1:1 frame mapping the cache does not guarantee

`_store_sim` subsamples an oversized clip, but the read side is `cached[frame % len(cached)]`
with no stride recorded. A 300-frame sim kept as 59 therefore plays original frames
0, 5, 10… at output frames 0, 1, 2… — 5× fast — and loops at output frame 59. The docstring's
claim that playback "spans the full duration at lower temporal resolution" is true of the
stored list and false of the playback. Triggers on any sim past ~59 frames at 1080p, silently.
**Fix:** store the stride alongside the clip and map `output_frame → kept[round(f / stride)]`,
or refuse to subsample and cap `n_frames` with a surfaced warning.

### P3 — the architecture heuristic is inverted on real cases

`detect_architecture` guesses from param names and tags: 115 A / 413 B, classified by
`n_frames` (88), `anim_mode` default (19), and **a tag (8)** — for those eight, editing a
descriptive tag silently changes execution and caching semantics.

Four of the six methods arch.py's own docstring names as canonical Architecture A examples
are classified B: 32 Reaction Diffusion (Gray-Scott), 34 Boids, 35 Flow Field, 36 DLA.
15 of the 113 `simulations` land in B overall — the CA and agent sims the mechanism exists
for. Measured, one executor, 128×128, three consecutive frames:

```
32 Reaction Diffusion  cat=simulations  arch=B   222, 222, 222 ms   1 distinct frame / 3
494 Screen-Space Fluid cat=filters      arch=A    12,   0,   0 ms   1 distinct frame / 3
```

A real sim pays a full re-cook every frame — the `sim */N` blowup the cache exists to
prevent, still live. A *filter* classified A on its `n_frames` param is cached as a one-frame
clip and looped: frozen for the whole session. Both directions fail at once.

**Fix:** declare it. Architecture is a property of how a method is written and only its
author knows it; the registry already carries explicitly declared contracts of this kind
(`new_image_contract`, `is_time_varying`). Add `@method(arch="A"|"B")`, demote the heuristic
to a fallback for unmigrated nodes, and have `tools/audit_methods.py` flag any node whose
declaration disagrees with the heuristic — which converts a silent misbehaviour into a
pre-commit failure.

### P4 — three smaller defects (read, not measured)

- `graph.py:1400` omits the `protect` argument that `graph.py:979` passes
  (`self._active_node_ids`), so the list-returning path's eviction can drop the currently
  rendering graph's sims — the exact BUG-8b condition the protect set was added to prevent.
  The fix was applied to one of two call sites.
- Two out-of-range policies for one data structure: the A path loops (`frame % len`), the
  list path is `if frame < len(...)` with no else, leaving `arr` as None.
- Sidecars are frozen. `write_field` writes a single overwritten `field.npy` and the executor
  stores one dict per `(node, seed)`, so on every cache-hit frame `field`/`mask`/`particles`
  are the sim's *final* state while `image` animates. Better than the `field_defaulted_to_image`
  bug it replaced, still not per-frame correct.
- `_sim_params_hash` is keyed by `node_id` while `_sim_cache` is keyed by `(node_id, seed)`.
  If two seeds coexist for one node, a params change re-cooks one and marks the hash current
  for both, so switching back serves stale frames. Narrow — the server flushes on seed change
  via `_ensure_executor` — but reachable from direct-executor callers: sequence renders,
  tests, and the MCP path once G2 gives it a seed parameter.

---

## Phases and gates

**Phase 1 — discovery, read-only.** Skeleton, `client.py`, `index.py`, `grill_search_nodes`,
`grill_get_node_def`, `grill_list_categories`, `grill_get_graph`. No server changes.
*Gate:* from a cold session, the agent names three plausible nodes for "reaction-diffusion
driven by an audio-reactive scalar" and reports their exact param names — with total tool
output under 15 KB.

**Phase 2 — authoring and render.** P1a is done, which unblocks single-frame agent rendering.
Server: `POST /api/graph/validate` (G1), `seed` + `timeout_s` on `/render` (G2, G3). MCP:
`grill_validate_graph`, `grill_patch_graph`, `grill_replace_graph`, `grill_render_frame`,
`grill_get_diagnostics`.
*Gates:* (a) `test_arch_a_frame_order.py` green, including the `slow` sweeps; (b) the agent
builds a 4-node graph from nothing, validates it, renders a frame, and the frame is not
black — unassisted, in one session.

`grill_render_frame` ships single-frame only, and that is now a deliberate boundary rather
than an oversight: at `frames == 1` the timeline phase is pinned and P1b is inert, so agent
renders are reproducible. **`grill_render_sequence` (phase 3) crosses into the exposed
region** and should not ship until P3 reclassifies the 40 — otherwise the first thing the
tool surface does is hand an agent a reproducibility hole 40 nodes wide.

P2 and P4 are not blockers: they predate MCP and stay wrong at the same rate whether or not
it ships. P3 is now on the critical path — it is what closes P1b and gates phase 3.

**Phase 3 — live, sequences, library.** `grill_live`, `grill_render_sequence`,
`grill_saved_graphs`, `grill_get_node_source`, resources.
*Gate:* with the user watching 📺 Live, the agent patches a param and the user sees the
change in the stream within a second.

**Phase 4 — optional, separate decision.** UI subscribes to `/api/graph/ws` (G4).

---

## Testing

`image_pipeline/tests/test_mcp_server.py`:

- **Schema snapshot** — tool names, descriptions and input schemas are pinned. A silent tool
  rename breaks every host config; this catches it.
- **Budget assertions** — the real teeth. `grill_search_nodes(limit=25)` ≤ 4 KB;
  `grill_get_node_def` ≤ 3 KB/node; `grill_render_frame(preview="small")` image payload
  ≤ 60 KB. These regress the moment someone adds a field "just for completeness".
- **Validation truth table** — each G1 check has a fixture graph that must fail it, and a
  near-miss that must pass.
- **Frame-order determinism** (P1) — belongs in `image_pipeline/tests/`, not the MCP suite,
  since it is an executor contract: for a stochastic Arch-A node, `f0` on a fresh executor,
  `f0` after `f3`, and `f0` after `f5` must hash identically. 494 / 954 / 171 are known-good
  probes; 91 is not (it ignores seed and frame).
- **Live-server integration**, behind a pytest marker (the suite must stay runnable without
  a server): health, search, patch, render, diagnostics against :7860.
- **Manual**: MCP Inspector pass per phase.
- **Evals**: 10 read-only questions in `image_pipeline/mcp/evals.xml` per the MCP-builder
  guidance — each needing several tool calls, each with a stable verifiable answer (e.g.
  "which category has the most registered methods, and how many?" → `gpu_shaders`, 159).

## Risks

**R1 — token budget.** The dominant risk, and the one every other design choice bends
around. Mitigation: the budget assertions above are tests, not guidelines.

**R2 — memory pressure from a third consumer.** Live and render executors each hold up to
1.5 GB of sim cache. MCP renders go through the *existing* `/render` executor, so no third
cache is created — which is precisely why we wrap HTTP. Do not "optimize" this later by
importing `GraphExecutor` into the MCP process.

**R3 — stale node index after hot-reload.** Methods are hot-reloaded from disk by the
watchdog; a cached MCP index would then describe nodes that no longer exist. Mitigation:
short TTL plus a cheap generation check (method count + max mtime, or an etag added to
`/api/methods`) before serving a cached def.

**R4 — determinism. Not a risk; a confirmed defect.** Identical graph + seed + params ⇒
identical output is a load-bearing contract, and P1 shows it already fails on the Arch-A path
before MCP exists. MCP's contribution is amplification: an agent renders arbitrary frames in
arbitrary order against one persistent executor, which is the access pattern that exposes it
every time rather than occasionally. Fix P1 first. Beyond that, the MCP path must never
introduce a second way to set canvas size, seed, or frame that disagrees with the UI's —
everything goes through the same endpoints the UI uses, and G2's seed param defaults to 42 so
existing `/render` callers see no change.

**R5 — agent thrash on the shared doc.** An agent patching `active` while the user edits the
same graph in the editor is a lost-update race (the UI holds its own copy and will overwrite
on the next Live start). Until G4, document `active` as agent-owned and give the agent a
convention of working on a named gid (`mcp-scratch`) unless the user asks for `active`.

---

## Open question for the owner

**Should agent edits drive the user's canvas (G4), or only the rendered output?** Phase 1–3
work either way and I'd ship them first regardless. The answer only decides whether phase 4
gets scheduled — and it's a UI-state design call (conflict resolution, undo semantics) that
shouldn't be made implicitly by whoever writes the MCP tools.
