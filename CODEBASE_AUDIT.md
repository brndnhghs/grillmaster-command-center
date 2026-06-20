# Codebase Audit — Image Pipeline / Command Center

*Verified against the source on 2026-06-20. Every claim below was checked by reading the actual import graph, registry wiring, and SQL usage — not inferred from filenames.*

> **Status: cleanup executed (2026-06-20).** The dead code and partials below have been removed/trimmed, the `filters.py` collision renamed, and the method-#18 conflict resolved. See **[Changes applied](#changes-applied)** at the bottom. Sections below describe the codebase *as audited*; struck-through premises are corrected inline.

## Verdict

The prior audit is accurate. Of the issues it raised, all were confirmed; three needed sharpening (one count, one "it's even deader than you thought," and a couple of junk files it missed). The clean core — FastAPI server + node-graph executor + `@method` registry — is coherent and fully wired. Essentially all of the rot is the vault/constellation layer that was bolted on and never removed.

---

## Structure: two independent apps sharing one registry

There are two apps that only overlap at the method registry.

**App 1 — the real tool.** `image_pipeline/server.py` (FastAPI) + `ui/index.html` (87KB single-file frontend). The server calls method functions directly in-process — confirmed at `server.py:260-262`, `meta.fn(out_dir, seed, params=params)`. SSE streaming, live frame preview, node-graph editor. This is the primary product and it's solid.

**App 2 — the utility sidebar.** `app.py` (Streamlit). A local-media browser plus a generate tab that shells out to the CLI through `pipeline_bridge/` (`generate`, `animate`, `promote_still`, `catalog`). Confirmed at `app.py:71-145`. Lighter weight, more of a side panel than a creative surface.

The registry both share: `image_pipeline/core/registry.py` with the `@method` decorator. `methods/__init__.py` auto-imports the method-group modules so each one self-registers on import.

---

## Dead code (safe to delete)

| Item | Status | Evidence |
|---|---|---|
| `vault/` (7 files: scanner, constellations, fragments, artifacts, titles, parser, markdown) | **Dead** | No imports of `vault` anywhere outside `vault/` itself. |
| `core/frontmatter.py` | **Dead** | Only callers are `vault/` and `tests/test_frontmatter.py`. |
| `core/ids.py` | **Dead** | Only imported by `vault/titles.py`, `constellations.py`, `artifacts.py`, `fragments.py`. |
| `core/bms.py` (whole module) | **Dead** | `app.py:10` imports `score_text` but never calls it (import is the only occurrence in the file). All other scoring fns had only vault/test callers. |
| `ui/theme.py` | **Dead** | Zero imports across the repo. |
| `image_pipeline/methods/codegen.py` (113KB) | **Dead** | Shadowed by the `codegen/` package; `from . import codegen` resolves to the package. Holds 12 `@method`s that are duplicated in the package. See correction below. |
| `core/models.py` — `TitleRecord`, `FragmentRecord`, `ConstellationRecord` | **Dead (partial)** | Vault-only. `ArtifactBundle`, `SummonResult`, `BaseRecord` remain active. Note: `core/__init__.py` still re-exports all six, so deleting the records means trimming that `__all__` too. |
| `tests/test_state_machine.py` | **Always errors** | Imports `core.state_machine`, which does not exist anywhere in the repo. ImportError on collection. |
| `tests/test_bms_balance.py`, `tests/test_frontmatter.py` | **Pass but validate dead code** | Green tests covering modules with no live callers. |

### Corrections / additions to the prior audit

1. **`codegen.py` is *more* dead than reported.** The prior note said `_param_grid.py` touches it "by explicit path." It doesn't — `_param_grid.py:17` is `import image_pipeline.methods.codegen`, which (because a package shadows a sibling module of the same name) resolves to the `codegen/` **package**, not `codegen.py`. So `codegen.py` is imported by *nothing*. Delete with zero risk.

2. **Junk files the audit missed:** `image_pipeline/methods/patterns.py.bak` and `image_pipeline/methods/patterns.py.corrupted` (303 bytes each) are leftover artifacts sitting next to the live `patterns.py`. Delete.

---

## Half-baked / partially wired

**`methods/simulations_cellular.py` — NOT method #18's missing half.** *Correction to the prior audit:* method #18 was **never missing** — `image_pipeline/methods/codegen/simulations.py` already registers `id="18"` "Cellular Automata" (category `codegen`, 16 rules, 11 seed patterns, 8 colormaps, 20 animation modes) and is live via the `codegen` package. The orphaned `simulations_cellular.py` *also* claimed `id="18"` — a **competing duplicate**, not a gap. Because the registry is last-write-wins (`_registry[id] = meta`), importing it would have silently *overwritten* the working #18, not added anything. The two are the same concept but different implementations (toroidal vs fixed-edge topology; a 3-state Brian's Brain rule that doesn't fit the live 2-state model; age-based coloring), so they were not mergeable cleanly. **Resolved:** the orphan was re-homed to a free slot as **method #58 "Cellular Automata (Variants)"** (category `simulations`) and wired into `methods/__init__.py`; live #18 is untouched.

**`index/schema.sql` — most tables are write-never-read.** *Correction:* the file defines **10** tables, not 9. Only `artifacts`, `artifact_members`, and `session_state` are used (confirmed via `INSERT`/`FROM`/`UPDATE` in `index/`). The other **7** — `titles`, `title_occurrences`, `fragments`, `constellations`, `relations`, `sandbox_items`, `recent_summons` — are created but have **0** read/write references anywhere. All seven trace back to the vault system.

**`core/runner.py` vs `server.py` — two execution engines.** `runner.py` (caching, threading, progress callbacks) is imported only by the CLI `pipeline.py`. The server bypasses it entirely and calls `meta.fn()` directly, so cache hits and parallel execution never apply to server requests. Two parallel ways to run a graph.

**`core/quality.py` + `core/annotator.py` — CLI-only.** Both imported only by `pipeline.py` (`--quality`, `--demo`/`annotate_batch`). Not wired into `server.py`, so quality checks and annotated demos can't be triggered from the UI.

**`methods/gpu_shaders.py` (method #82) — registers, then crashes.** Pulls `core/shaders.py`, which lazily `import moderngl`. `moderngl` is **not** in `requirements.txt` (confirmed). The method registers fine and fails only on invocation in a clean environment.

---

## Naming collision (not a bug, but a footgun)

- `image_pipeline/core/filters.py` (120KB) — CLI post-processor applied to existing images via `--filter`.
- `image_pipeline/methods/filters.py` (186KB) — registered generative methods (glitch, dither, pixel sort…).

Same name, unrelated jobs. **Resolved:** `core/filters.py` was renamed to `core/postprocess.py` (via `git mv`), and its sole importer in `pipeline.py` updated.

---

## The clean core (leave alone)

FastAPI server + node-graph system is the well-built part: the `@method` registry, `GraphExecutor`, SSE streaming with live-frame intercept, and the HTML/JS frontend are all coherent and fully wired. The mess is almost entirely the vault/constellation layer plus a few CLI-only features that never made it into the server.

---

## Suggested cleanup order

1. **Zero-risk deletes:** `vault/`, `core/frontmatter.py`, `core/ids.py`, `core/bms.py`, `ui/theme.py`, `methods/codegen.py`, `methods/patterns.py.bak`, `methods/patterns.py.corrupted`, `tests/test_state_machine.py`, `tests/test_bms_balance.py`, `tests/test_frontmatter.py`.
2. **Trim partials:** remove the three vault records from `core/models.py` and `core/__init__.__all__`; drop the unused `score_text` import in `app.py:10`; drop the 7 unused tables from `index/schema.sql`.
3. **Decide, don't drift:** for each of `simulations_cellular` (#18), `gpu_shaders` (#82 / `moderngl`), `runner.py`, and `quality.py`/`annotator.py` — either finish wiring it into the server or delete it.
4. **Rename** one of the two `filters.py` files.

---

## Changes applied

*Executed 2026-06-20. All edited modules parse and import clean (the pure-Python layers verified by import; the numpy/scipy/fastapi layers by AST parse + targeted dependency import, since this sandbox lacks `scipy`/`fastapi`/`streamlit`). A full repo grep confirms zero remaining references to any deleted/renamed/moved symbol.*

**Deleted (zero-risk):** `vault/` (7 files), `core/frontmatter.py`, `core/ids.py`, `core/bms.py`, `ui/theme.py`, `image_pipeline/methods/codegen.py` (the 113KB file shadowed by the `codegen/` package), `image_pipeline/methods/patterns.py.bak`, `patterns.py.corrupted`, and the three dead tests (`test_state_machine.py`, `test_bms_balance.py`, `test_frontmatter.py`).

**Trimmed:**
- `core/models.py` reduced to `SummonResult` + `EntityKind` — removed `TitleRecord`/`FragmentRecord`/`ConstellationRecord` (vault-only) and, since `bms.py` is gone, the now-unreferenced `BaseRecord`/`ArtifactBundle`/`ArtifactMediaType`. (`SummonResult` is the only model with live consumers: `app.py`, `index/query.py`.)
- `core/__init__.py` `__all__` updated to match.
- `app.py` — removed the orphan `from core.bms import score_text` import.
- `index/schema.sql` — cut from 10 tables to the 3 actually used (`artifacts`, `artifact_members`, `session_state`) plus their one live index.

**Renamed:** `image_pipeline/core/filters.py` → `image_pipeline/core/postprocess.py` (`git mv`); updated the importer in `pipeline.py`.

**Method #18 conflict resolved:** the orphaned competing `simulations_cellular.py` was re-homed as **method #58 "Cellular Automata (Variants)"** (free slot; was 58/60/61/68/75 open) and wired into `methods/__init__.py`. Live #18 (the codegen version) untouched. Registration of both was verified.

**Left as-is (deliberate):** `gpu_shaders` #82 (still needs `moderngl` in `requirements.txt` to run — kept, not deleted); `runner.py` and `quality.py`/`annotator.py` remain CLI-only.

**Note:** the working tree contained unrelated pre-existing uncommitted changes (e.g. to `server.py`, `fractals.py`, several `codegen/*`, and four already-deleted `tests/*`) that predate this cleanup. Review `git status` before committing so you stage only the intended changes. Recommended pre-commit smoke test in the project `.venv`: boot the FastAPI server, confirm the registry method count, and render methods **#18** and **#58** once.
