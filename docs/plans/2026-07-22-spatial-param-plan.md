# Spatial Params — driving node params with fields, not numbers — 2026-07-22

Goal (owner-stated): a node's inputs should serve that node's unique purpose. Gray-Scott's `feed`/`kill` driven per-pixel by a greyscale bitmap; SVG Vector accepting SVG source or a file; Pixel Mosaic's `tile_size` driven by a field.

Today none of that is expressible. This plan says why, and what to do about it.

**Constraint (revised after measurement):** the migration is **non-perturbing while nothing is wired**. `sparam()`'s scalar path is bit-identical to the `float(params.get(...))` it replaces — verified on #155 by rendering the same frame before and after at `max|Δ| = 0.0`. Only a genuinely wired FIELD changes output, and that is a new graph the user built. So this is NOT the output-perturbing epoch originally assumed; no reproducibility break to record.

## Status — 2026-07-22

| | |
|---|---|
| params marked `spatial: True` | **131** across **64** methods |
| exposing a FIELD port | 131 / 131 |
| probe verdict SPATIAL | **131 / 131** |
| gate | `image_pipeline/tests/test_spatial_params.py` — 265 passing |
| regressions | none (suite identical to baseline: same 6 pre-existing failures / 2 errors) |

**No param in the library now claims spatial support it does not have.** The
three legacy liars are resolved: #11 Gradient migrated (3/3), #01 ASCII Art
partly migrated (`dither_strength`) with its two structural params
de-advertised, #45 Graphviz reclassified structural and its four FIELD ports
removed. The `__test__` node's dead `_field_anim_speed` read is gone too.

### Known caveat — four params pass in standalone mode only

`343.contrast`, `424.brightness`, `425.freq`, `1004.drainage` respond to a field
when the node renders procedurally, and are inert once `image_in` is wired:
wiring an image makes the executor flip `source` to `"input_image"`
(`graph.py`, the `src_spec` branch), and the procedural field these params
modulate is gone on that path. The probe tries both configurations and reports
`standalone_only` for them, so the gate certifies what is actually true rather
than passing them silently. Whether the wired path *should* honour them is a
per-node design question, not a plumbing bug — left open deliberately.

## Findings (measured 2026-07-22, not estimated)

Four mechanisms exist. None delivers a spatially-varying param.

**1. The port generator locks out ~everything.** `core/graph.py::_make_node_def` refuses a port to any param carrying `min`/`max`, on the reasoning that a slider constraint means "internal control, not wireable". Measured across the registry:

| | count |
|---|---|
| methods | 527 |
| total params | 4,680 |
| numeric params wireable today | **42** |
| numeric params blocked purely by `min`/`max` | **3,128** |
| categorical (`choices`/str) | 1,434 |
| list/tuple (the only shape granted a FIELD port) | 1 |

`feed`, `kill`, `diff_u`, `diff_v` (Gray-Scott #155) and `tile_size` (Pixel Mosaic #80) are all in the 3,128. They have no port at all.

**2. A fuzzy back-channel reaches them unpredictably.** `_eligible_params` ignores `min`/`max` entirely, and `_score_param` name-matches upstream outputs onto params (`exact=10, synonym=5, substring=2`). Drivers therefore *do* land on `feed`/`kill` — by name-similarity guessing rather than an explicit wire the user drew.

**3. Fields are averaged into a single number.** `core/graph.py`, driver-injection path:

```python
if isinstance(_dv, np.ndarray):
    _dv = float(np.mean(_dv))
```

Wire a greyscale bitmap into `feed` today and the node receives its mean. This is the reported problem, exactly.

**4. The spatial hook exists, and is 100% decorative.** `_inject_typed` already writes `_field_<param>` alongside every scalar injection, as a zero-copy uniform broadcast. Twelve params across four nodes read it. **Zero respond to the field's structure** — verified by running them, see below.

### The `_field_` convention is theatre (measured, `tools/audit_field_response.py --scan`)

```
summary: MEAN_ONLY=12
0/12 probed params genuinely respond to field structure
```

Per node, and this is the important part — each failure looks like support in source:

- **#11 Gradient** — reads `_field_cx`, `_field_cy`, `_field_direction` into locals, then **never uses them**. Dead reads.
- **#45 Graphviz** (`cli_tools.py`) — reads four `_field_*` arrays and immediately `np.mean()`s every one.
- **#01 ASCII Art** — the subtle one. Reads `_field_font_size`, builds a genuine per-pixel `font_size_arr`, threads it down two call layers as `fs_arr=` … and the receiver does `_fs = int(np.median(fs_arr))`. Median of uniform-0.5 and of a 0→1 ramp are both 0.5, so output is bit-identical.

ASCII Art is the whole argument for a runtime probe: it reads as fully spatial for two layers before collapsing on the third. No amount of code review or grep finds that. Running it finds it in milliseconds.

## The core insight

The three requested examples are three unrelated problems. Treating them as one is what makes this feel intractable.

| Class | Example | What it takes |
|---|---|---|
| **A — already per-cell** | Gray-Scott `feed`/`kill`/`diff_*`, `noise_amp` | Mechanical. `F` is already a scalar inside a per-cell PDE; making it an `(H,W)` array is near-1-line and numpy broadcasts it for free. **The volume lives here.** |
| **B — structural** | Pixel Mosaic `tile_size` | Real algorithm work. Per-pixel tile size means variable-rate tiling, not a broadcast. Selective, by hand. |
| **C — never spatial** | `n_frames`, `dt`, `seed`, `timeout` | Must be *excluded*, or the UI sprouts 3,128 meaningless ports. |
| **D — content inputs** | SVG source, file paths, fonts, shader source | Not a field problem at all. #30 SVG Vector declares `inputs={}` — no ports whatsoever — and the type system conflates "str default" with "categorical choice", so a free-text/code param is currently **inexpressible**. |

## Phases

### Phase 0 — declare intent, don't infer it — STATUS: **done**

`core/spatial.py` (`sparam` / `is_field` / `as_scalar`); `spatial: True` honoured in `_make_node_def`; the driver mean-collapse is now type-aware and hands the array to spatial params.

Add an opt-in `spatial: True` flag to the param spec. `_make_node_def` grants a FIELD port only to flagged params. This decouples wireability from `min`/`max` (which was always the wrong proxy) without exposing 3,128 ports.

One port, dual-mode: the param keeps its slider as the default and accepts a FIELD to override per-pixel — the Houdini/TouchDesigner convention. The existing uniform-broadcast fallback already guarantees the invariant that makes this safe: **a node written against `_field_x` behaves identically whether driven by a slider or a field**, so there is one code path, not two.

Make the mean-collapse type-aware: pass the array through for `spatial` params, keep collapsing everywhere else so untouched nodes are bit-identical.

### Phase 1 — classifier — STATUS: **done** (`tools/classify_params.py`)

Ledger at `data/spatial/param_ledger.csv` — all 4,680 params, all 527 methods:

| class | count | share |
|---|---|---|
| A spatializable | 712 | 15.2% |
| B structural | 1,141 | 24.4% |
| C non-spatial | 2,805 | 59.9% |
| D content | 22 | 0.5% |

Two corrections were forced by running it, both worth keeping: nodes emitting no IMAGE (LFO, Counter — drivers) are class C by construction, since "per-pixel" is meaningless there and they cannot be probed; and a name passed to ANY call outside a broadcast-safe allowlist is structural, because PIL draws, subprocess args and `dot` invocations all need one number. The second correction moved 747 params A → B and is why the keep-rate is trustworthy.

Tool proposing an A/B/C/D label for all 4,680 params: AST-scan whether the param name flows into an expression over `(H,W)` arrays (→ A) versus controlling a count, range, or shape (→ B/C). Emits a reviewable TSV; it does not auto-commit. Expect it to nail the Class-C exclusions and most of A, and to be useless on B — which is fine, B is hand-work by definition.

### Phase 2 — response probe — STATUS: **done** (`tools/audit_field_response.py`)

The verification spine, and deliberately built first. For a param P, render three times at fixed seed and canvas:

- **A** `_field_P` = uniform 0.5
- **B** `_field_P` = horizontal ramp 0→1 (mean also 0.5)
- **C** `_field_P` = vertical ramp 0→1 (mean also 0.5)

All three share a mean, so any node collapsing via `mean`/`median` emits byte-identical output — divergence *proves* the node reads spatial structure. The H-vs-V pair separates genuine spatial response from incidental noise. A determinism pre-check rejects nondeterministic nodes rather than reporting their noise as signal.

Verdicts: `SPATIAL` / `MEAN_ONLY` / `ORIENTED?` / `NONDETERMINISTIC` / `ERROR`.

Validated against synthetic controls — a genuinely per-pixel node scores Δuniform=0.25, a mean-collapsing one Δ=0.00000 — so the 0/12 result above is a real measurement, not a probe that always says no.

### Phase 3 — batched Class-A migration — STATUS: **first pass done** (`tools/migrate_spatial.py`)

Apply → probe → keep-or-revert, in batches so the ~4 s method-tree import is amortised across a whole batch instead of paid per param. Outcome over the 712 candidates:

| outcome | count | meaning |
|---|---|---|
| **KEPT (SPATIAL)** | **117** | field verified to reach the pixels |
| MEAN_ONLY | 155 | rewrite applied, output unchanged — collapse elsewhere in the node |
| ERROR | 213 | array reached a scalar-only use; class-A false positive |
| SKIPPED_SHAPE | 226 | read site not in the `float(params.get(...))` form |
| NOT_PROBED | 105 | node unprobeable in isolation |
| ORIENTED? | 3 | responded, but H and V ramps identical — suspicious, left alone |

Every non-KEPT param is restored byte-for-byte; a failed attempt costs nothing. Outcomes are recorded in `data/spatial/spatial_failures.json` so each batch advances to new params instead of re-attempting the same head of the ledger — without that memory the loop is not monotonic and simply repeats itself (observed, then fixed).

**Gray-Scott #155 is the reference case**, and it needed one hand-fix the codemod cannot do: `anim_mode` defaults to the named regime `"spots"`, whose preset F/k assignment overwrote the wired field. A wired FIELD now outranks the preset. All four of its params (`feed`, `kill`, `diff_u`, `diff_v`) probe SPATIAL.

Still open: the three legacy liars (#01, #11, #45) — none migrated, all still advertising `_field_` support they do not honour. Worth fixing precisely because declaring nothing would be more honest.

### Phase 4 — Class D, content inputs — STATUS: not started

Independent of everything above; can run in parallel. New `text` / `code` / `asset` param types and a `CONTENT` port type, wired to the existing `/api/assets/upload` infrastructure that `io_nodes.py` already uses for image and video import. Unblocks SVG source, shader source, font files and model imports in one pass.

### Phase 5 — lock it — STATUS: **done** (`image_pipeline/tests/test_spatial_params.py`)

237 tests: every declared spatial param must probe SPATIAL and must expose a real FIELD port, plus unit cover on `sparam`'s scalar/field paths. Verified to have teeth by reintroducing the original bug (reverting #155 `feed` to `float(params.get(...))` while leaving `spatial: True` declared) — the gate failed on `[155-feed]` as intended, then went green on restore.

## Risks

- **Output-perturbing.** Phases 0+3 change rendered output; bundle as one epoch (see Constraint above).
- **UI surface.** The `spatial: True` opt-in is the control preventing a 3,128-port explosion. Do not "temporarily" relax the `min`/`max` rule instead — that is the same mistake in the other direction.
- **Class B is a trap.** `tile_size` looks adjacent to `feed`/`kill` but is not; it changes grid topology. Do not let it into a Class-A batch.
- **Phase 1 is optional.** Phases 0/2/3 deliver the value; the classifier only makes the migration cheaper to plan. If it underperforms on real params, drop it and hand-pick from the Phase-2 report.

## Provenance

Numbers from `registry.get_all()` + `_make_node_def` over the live registry, 2026-07-22. Probe results: `data/spatial/field_response_report.json`, regenerate with `python tools/audit_field_response.py --scan`.
