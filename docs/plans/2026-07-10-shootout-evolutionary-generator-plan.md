# Shootout — Interactive Evolutionary Nodegraph Generator

**Date:** 2026-07-10
**Status:** Plan for implementation (hand to a coding agent)
**Area:** `image_pipeline/`

---

## 1. Concept

An aesthetic-selection ("Picbreeder"-style) evolutionary loop that discovers interesting
animated nodegraphs automatically:

1. The system **generates a batch of random-but-valid nodegraphs**.
2. It **renders each to a short video** and culls the dead ones.
3. It **presents the survivors** in a new web-app page.
4. The user **rates each clip 1–5 stars**.
5. The system **logs every (genome, rating) pair**, **breeds the next generation** from the
   highly-rated parents (mutation + crossover), and — over sessions — **trains a persistent
   "taste" model** that predicts which graphs the user will like.

The novelty vs. a plain param-sweep: selection pressure comes from a human, and the taste
memory persists and compounds across sessions.

---

## 2. Decisions locked with the user (do not re-litigate)

| # | Decision | Choice |
|---|----------|--------|
| 1 | Depth of learning | **Breed winners + persistent taste model.** Session-scoped breeding *and* a cross-session dataset feeding a trainable scorer. |
| 2 | Randomness style | **Wild + repair/reject.** Sample loosely, auto-repair to validity, render, discard the dead. |
| 3 | Review surface | **New page in the existing FastAPI web app.** |
| 4 | Round budget | **6 clips × ~96 frames (~4s @ 24fps).** Over-generate under the hood, show 6 alive. |
| 5 | Selection signal | **1–5 star rating on every clip.** (Positive + graded, per candidate.) |
| 6 | Pre-filter | **Reject dead clips only** (black / static / NaN / flicker). No taste-based pre-ranking in v1 — the model logs & trains first, and is *used* to bias generation in a later phase. |
| 7 | Taste model form | **Tabular regressor over graph features + param stats.** Predicts star rating. Schema designed so visual/embedding features can slot in later. |
| 8 | Gene pool | **Full registry, used by role.** Non-image nodes (Timeline, LFO, Math, FIELD/PARTICLES/COLORMAP sources, compositing) are legal **only in their proper roles** — as drivers/intermediates feeding an IMAGE terminal — never as the terminal render node. This makes generation *port-type-aware*: "wild" within type correctness, not random bytes. |

---

## 3. What already exists (build on it — don't reinvent)

- **Genome == existing graph JSON.** A nodegraph is
  `{version, name, nodes:[{id, method_id, params, x, y, render}], edges:[...]}`.
  See `image_pipeline/output/saved-graphs/spark.json`. One node carries `render: true`
  (the terminal). This *is* the genome — no new format.
- **Render → video is a solved primitive.** `POST /api/graph/render-sequence`
  (`image_pipeline/server.py:1978`) takes `GraphRequest {nodes, edges, seed, frames, frame,
  width, height}`, runs `GraphExecutor.execute(...)` per frame, and assembles
  `frames_to_mp4()` (`core/animation.py`) → `output/sequences/<name>/output.mp4`, served at
  `GET /api/sequences/{name}/video.mp4`. Drive this per candidate.
- **Node catalog + param schemas:** `GET /api/node-defs` → `{method_id: {params, inputs,
  outputs, category, ...}}` from `core/graph.py:get_all_node_defs()`. `params` entries carry
  type/default/range (enriched via `_enrich_params`). This is the sampler's source of truth.
- **Port-type system** (`core/port_types.py`): `IMAGE`, `SCALAR` (accepts_from IMAGE),
  `FIELD`, `PARTICLES`, `MASK`, `COLORMAP`, `ANY`. An edge `src.out_port → dst.in_port` is
  legal iff `dst` type matches `src` type or lists it in `accepts_from` (or dst is `ANY`).
  **These rules define graph validity — the generator and repair pass both key off them.**
- **Two execution modes** share `GraphExecutor`: render (`audit_to_disk=True`) and live
  (`in_memory=True`, zero disk). Use the render-sequence job path (already does the mp4).
- **LLM backend precedent:** Hermes "Node Doctor" (`nd_runner.py`) — precedent for an
  optional LLM assist, *not* required for v1.

---

## 4. Architecture

```
                         ┌──────────────────────────────────────────────┐
                         │  shootout/ (new package under image_pipeline) │
                         └──────────────────────────────────────────────┘
 generator.py   → samples port-type-aware random genomes (wild-within-types)
 repair.py      → makes a genome valid: DAG-ify, fix/drop illegal edges,
                  guarantee exactly one IMAGE terminal (render:true),
                  clamp params to schema ranges
 evaluator.py   → render genome via /api/graph/render-sequence → mp4;
                  compute liveness stats; reject dead clips
 evolve.py      → mutation + crossover operators; builds next generation
                  from rated parents + explorers
 features.py    → genome → feature vector (node histogram, depth, edge stats,
                  param stats) for the taste model
 taste.py       → tabular regressor: train from ratings dataset, predict rating,
                  (phase 2) bias generation
 store.py       → persistence: lineage, per-generation genomes, ratings dataset
                  (JSONL), model artifact
 session.py     → orchestrates a shootout session (generation loop state machine)

 server.py      → NEW endpoints (Section 8) that expose session.py to the UI
 ui/            → NEW page: grid of 6 autoplaying <video>, 1–5 star per clip,
                  "Evolve →" button
```

Everything new lives under `image_pipeline/shootout/`. Server endpoints are thin wrappers.
No changes to the existing executor/registry beyond *reading* their public APIs.

---

## 5. Genome & session data model

**Genome** = existing graph JSON + a metadata envelope:

```jsonc
{
  "genome_id": "g-<uuid8>",
  "generation": 3,
  "parents": ["g-aa11", "g-bb22"],   // [] for gen-0 / explorers
  "origin": "crossover",             // "random" | "mutation" | "crossover" | "explorer"
  "seed": 12345,                     // part of the genome — carried with winners
  "graph": { "version":1, "nodes":[...], "edges":[...] },  // the renderable graph
  "render": { "seq_name":"...", "mp4":"...", "frames":96, "w":768, "h":512 },
  "liveness": { "alive":true, "temporal_var":0.021, "spatial_var":0.13, "nan":false },
  "rating": 4                        // null until the user rates it
}
```

**Ratings dataset** (`shootout/data/ratings.jsonl`, append-only, cross-session): one line per
rated genome = `{features:{...}, rating, genome_id, ts, session_id}`. This is the training
corpus for the taste model and the durable "success library."

**Lineage** (`shootout/data/sessions/<session_id>.json`): ordered generations, each the list
of genome_ids shown + ratings — lets a session resume and lets you trace ancestry.

---

## 6. Generator (`generator.py`) — port-type-aware wild sampling

Goal: emit structurally-random graphs that are *type-plausible* (so repair rarely has to
discard much), honoring the "non-image nodes in their proper roles" rule.

1. **Partition the catalog by output port** (from `/api/node-defs`):
   - `IMAGE`-producers → eligible **terminal** and mid-chain.
   - `FIELD` / `PARTICLES` / `COLORMAP` / `MASK` producers → **source/driver** roles only.
   - `SCALAR`-only / data-only (Timeline `__timeline__`, LFO, Math) → **param drivers** only.
   - Compositing nodes (merge ports `image_a/b`, `field_a/b`, `particles_a/b`) → **combiners**.
   Cache this partition; it's derived purely from node-defs.
2. **Sample a skeleton** with a random depth (2–6 nodes typical):
   pick a terminal IMAGE node, then walk *backwards* filling each required input port with a
   node whose output type is accepted by that port (respecting `accepts_from`). Randomly
   attach optional driver nodes (LFO→a SCALAR param, FIELD source→a filter's field input).
3. **Sample params** per node from its schema: numeric within range (bias toward mid ±
   spread), enums uniform, booleans coin-flip. Occasionally push a param to an extreme for
   surprise.
4. **Random seed** per genome.
5. Emit genome with `origin:"random"`. Positions (`x,y`) auto-laid-out (cheap left→right by
   depth) so the graph is inspectable in the editor.

"Wild" = random topology, node choices, and params. "Valid by construction where cheap,
repaired where not" — the backward-walk already guarantees port types line up; repair is the
safety net for edge cases (cycles from optional attachments, missing terminal, etc.).

---

## 7. Repair & reject

**Repair (`repair.py`, pre-render, cheap):**
- Enforce **DAG** — drop any edge that introduces a cycle.
- Every node input port that is *required* but unfilled → either wire a compatible existing
  node or leave to node default (only if the node tolerates a missing input; else drop node).
- Drop edges whose `src_type`→`dst_type` is illegal per port-type rules.
- Guarantee **exactly one** node with `render:true`, and it must be an IMAGE-producer. If
  zero/multiple, pick the deepest IMAGE node.
- **Clamp params** to schema ranges; coerce enum/bool types.
- If a graph can't be made renderable (no IMAGE terminal reachable) → discard, resample.

**Reject (`evaluator.py`, post-render, "dead clips only"):** compute cheap stats on the
rendered frames (no ML):
- `nan/inf` present → dead.
- **Spatial variance** of mean frame < ε → flat/black/near-constant image → dead.
- **Temporal variance** across frames < ε → static (no motion) → dead. *(4s clips are
  supposed to move.)*
- Optional: extreme temporal variance + no spatial structure → pure flicker/noise → dead.
Thresholds live in config; tune on a first empirical batch. Over-generate (e.g. render up to
~12–15 candidates) to reliably surface **6 alive** survivors.

---

## 8. Evolution engine (`evolve.py`)

**Generation composition** (after gen-0, tunable in config). To fill the survivor pool the
evaluator needs (~12–15 rendered → 6 alive shown):
- **Exploit:** offspring of parents weighted by star rating (4–5★ dominate).
- **Explore:** a fraction of fresh `generator.py` random graphs each round (prevents
  premature convergence / keeps it interesting). Default ~30%.
- (Phase 2) taste-model-biased samples once the model is trained.

**Mutation operators** (pick 1–2 per offspring):
- Param jitter — Gaussian within schema range on a random subset of params.
- Node swap — replace a node with another of the **same output port type**.
- Add/remove a driver node (LFO, filter) on a compatible port.
- Rewire — repoint one edge to another type-compatible port.
- Seed jitter (occasional).

**Crossover:** pick two parents, splice a subgraph from parent B into parent A at a
port-type-compatible boundary; run repair. Parent selection = rating-weighted (rank or
roulette).

Elitism: optionally carry the single highest-rated genome forward unmutated so quality never
regresses.

---

## 9. Taste model (`features.py` + `taste.py`)

**v1 — tabular regressor** predicting star rating (1–5) from a genome feature vector:
- **Features:** node-type histogram (count per method_id / per category), graph depth, node
  count, edge count, mean/branching factor, per-category param summary stats (mean/spread of
  normalized param values), origin one-hot. All derivable without rendering.
- **Model:** gradient-boosted trees or ridge/logistic (small data early on — keep it simple
  and refit-cheap). Train from `ratings.jsonl`, retrain at session end (and/or incrementally).
- **v1 usage:** *log + train only.* Report a held-out correlation metric so you can watch it
  learn. It does **not** gate what's shown yet (per decision #6).
- **Phase 2 usage:** bias generation — score a large candidate pool, sample toward
  high-predicted-rating genomes; optionally pre-rank survivors.
- **Phase 3 (schema-ready, not built):** add visual features — embeddings of sampled frames —
  concatenated to the tabular vector. `features.py` returns a dict so new keys slot in without
  breaking the dataset.

---

## 10. Server endpoints (thin wrappers over `session.py`)

Register **before** the dynamic `/api/graph/{gid}` routes (FastAPI matches in registration
order — see the route-order trap in Section 12).

- `POST /api/shootout/session` → start/resume a session; returns `session_id`, config.
- `POST /api/shootout/generate` `{session_id}` → generate+repair+render+reject a generation;
  returns the 6 survivors `[{genome_id, mp4_url, generation, origin}]`. Long-running → reuse
  the existing job/stream pattern (`/api/jobs/{id}/stream`) for progress.
- `POST /api/shootout/rate` `{session_id, ratings:{genome_id: stars}}` → persist ratings to
  the dataset + lineage.
- `POST /api/shootout/evolve` `{session_id}` → rate (if not already) → breed next generation
  → same shape as `/generate`.
- `GET  /api/shootout/genome/{genome_id}` → full genome JSON (so a favorite can be opened in
  the normal editor / saved via existing `/api/graph/save`).
- `POST /api/shootout/train` (optional/manual) → retrain taste model, return metrics.

Reuse `GraphRequest` shape and `render-sequence` internals — introduce **no new graph
format**.

---

## 11. UI page (`ui/`)

New page (e.g. `/shootout`) in the existing web app:
- **6-up grid** of autoplaying, looping, muted `<video>` (mp4 from
  `/api/sequences/{name}/video.mp4`).
- **1–5 star widget** under each clip.
- **"Evolve →"** button (disabled until all 6 rated, or defaults unrated=skip) → posts
  ratings, triggers next generation, swaps in new clips.
- Header: generation number, session id, "New session", live training metric (once model runs).
- Per-clip **"Open in editor"** (loads the genome graph into the existing node editor) and
  **"Save"** (existing `/api/graph/save`) so a keeper escapes the loop.
- Progress via the existing job-stream SSE while a generation renders.

Keep it consistent with the current UI conventions in `ui/`.

---

## 12. Known traps (from `image_pipeline` memory — heed these)

- **`utils.save()` sink, not monkeypatch.** Methods bind `save` at import; interception must
  go through the per-thread sink (`set_save_capture`). Irrelevant if you only drive
  `render-sequence`, but relevant if you render candidates in-process.
- **Merge-port ndarray contract** — compositing methods pass ndarrays in run_params
  (`image_a/b`, `field_a/b`, `particles_a/b`); `*_path` temp files exist only in audit mode.
  Generated compositing wiring must use the real port names.
- **FastAPI route order** — static `/api/shootout/...` and `/api/graph/<literal>` routes MUST
  be registered before dynamic `/api/graph/{gid}` or they get captured as gids.
- **Determinism epoch** — per `noise-reproducibility-policy`, output-changing bug fixes land
  in place; a genome's `(seed, params)` reproduces only *within a pipeline version*. Store the
  rendered mp4 as the durable artifact; don't assume a genome re-renders bit-exact forever.
- **Excluded from the terminal role:** data-only (`__timeline__`, LFO, Math), 3D stack, ML
  nodes — they may appear as drivers but never as the `render:true` node. Node #18 Cellular
  Automata is Architecture B (no `n_frames`) — handle its animation contract specially or omit
  in v1.
- **149 ms/frame reference** → 96 frames ≈ 14s/clip × ~12–15 rendered ≈ 3–4 min/generation.
  Render candidates **concurrently** (thread pool over `render-sequence`) to keep rounds snappy;
  cap concurrency to avoid thrashing.

---

## 13. Phasing (suggested milestones)

- **M1 — Generate & render.** `generator.py` + `repair.py` + `evaluator.py`; CLI that emits N
  genomes, renders, rejects dead, writes mp4s to a folder. *Proves the wild→valid→video path.*
- **M2 — Review UI + rating.** Shootout page, `/session` `/generate` `/rate` endpoints. Human
  can rate a generation. Ratings persist to `ratings.jsonl`.
- **M3 — Evolution.** `evolve.py` (mutation + crossover + explore mix), `/evolve`. Closed loop:
  rate → breed → next generation.
- **M4 — Taste model (log+train).** `features.py` + `taste.py`; train on the growing dataset,
  surface a metric. Not yet gating.
- **M5 — Taste model (bias).** Use predictions to bias generation / pre-rank. Tune explore
  ratio.
- **Phase 3 (later) — Visual features** into the taste vector.

---

## 14. Config knobs (single `shootout/config.py` or JSON)

`show_n=6`, `render_pool=12–15`, `frames=96`, `fps=24`, `w=768`, `h=512`,
`explore_ratio=0.3`, `elitism=1`, `max_depth=6`, liveness thresholds
(`temporal_var_min`, `spatial_var_min`), mutation rates, render concurrency,
gene-pool include/exclude lists (terminal-eligible = IMAGE-producers minus 3D/ML/data-only).

---

## 15. Test plan

- **Generator/repair unit tests:** every emitted+repaired genome passes the executor's own
  validity checks (DAG, one IMAGE terminal, all edges type-legal, params in range). Fuzz 1000
  genomes → 0 executor `GraphError`s from validity (only legitimate runtime rejects).
- **Evaluator:** synthetic frame stacks (black, static, NaN, moving) classify correctly.
- **Evolve:** offspring of two valid parents are valid; ratings-weighted selection favors
  high-star parents; explore ratio holds.
- **Features/taste:** deterministic feature vector for a fixed genome; model trains and beats a
  mean-baseline on a held-out split once enough ratings exist.
- **Endpoints:** session lifecycle, route-order (static before dynamic), rating persistence
  round-trips.
- **E2E smoke:** one full generation renders 6 alive clips end-to-end under the time budget.

---

## 16. Open questions to resolve during build (defaults chosen; flag if wrong)

- Exact liveness thresholds — empirical, tune on first batch.
- Explore ratio and mutation rates — start at defaults above, expose as knobs.
- Whether to let the user **name/tag** a keeper generation for the success library (nice-to-have).
- Model choice (GBT vs linear) — pick after seeing the first ~100 ratings; both are cheap to
  swap behind `taste.py`.
