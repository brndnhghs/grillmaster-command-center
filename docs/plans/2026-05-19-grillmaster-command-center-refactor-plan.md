# GRILLMASTER Command Center Refactor Plan

> **For Hermes:** Use subagent-driven-development to implement this plan task-by-task.

**Goal:** Replace the current tabbed GRILLMASTER prototype with the exact constellation-engine architecture from the recovered build spec: a single-page Streamlit app with a persistent shell, derived SQLite index, session restore, inspect views for titles/artifacts/fragments/constellations, a collect tray, assembly workbench, and vault-backed constellation promotion.

**Architecture:** This refactor is a full architectural reset, not an incremental polish pass. The existing prototype is treated as a temporary source of visual ideas and vault-path knowledge, while the new app is rebuilt around the recovered canonical spec: three-layer runtime (vault canon + derived index + local session state), four inspectable object kinds, Body/Mind/Spirit scoring, and a write-back workflow to `Grillmaster/Constellations/`. We keep the repo path, but replace the tabbed dashboard model with the single-page shell and new module layout from the spec.

**Tech Stack:** Python 3.11+, Streamlit, Pillow, PyYAML, sqlite3, pathlib, dataclasses, optional rapidfuzz, optional watchdog.

**Primary references:**
- Recovered spec: `/Users/admin/Documents/GitHub/hermes-agent/.hermes/plans/2026-05-18-grillmaster-command-center-build-spec.md`
- Current prototype repo: `/Users/admin/Documents/GitHub/grillmaster-command-center`
- Vault root: `/Users/admin/Documents/Obsidian/Hermes/Grillmaster`
- Constellation output dir: `/Users/admin/Documents/Obsidian/Hermes/Grillmaster/Constellations`

---

## Current-state audit summary

The current repo is an early prototype, not the target architecture.

### What exists now
- `app.py` — single-page Streamlit app, but organized as top-level tabs
- `tabs/man_suite.py` — hardcoded MAN SUITE piece browser
- `tabs/devices.py` — hardcoded device view
- `tabs/cosmology.py` — hardcoded hierarchy/lore browser
- `tabs/social.py` — social planning tab
- `tabs/vault.py` — simple note browser/search
- `utils/vault_reader.py` — ad hoc vault file scanning/markdown reading helpers
- `utils/display.py` — styling helpers
- `.streamlit/config.toml`, `requirements.txt`

### What conflicts with the recovered build spec
- Tabbed dashboard model instead of persistent 3-pane shell
- No SQLite-derived index
- No stable IDs for titles/artifacts/fragments/constellations
- No assembly workbench
- No collect tray
- No session restoration / reconciliation
- No constellation promotion / revision workflow
- No `Grillmaster/Constellations/` write-back path
- No canonical data model or frontmatter round-trip layer
- Hardcoded content in tabs where the new system should be data-driven

### What is salvageable
- Repo path and virtualenv/project shell
- Some CSS/theming ideas from `app.py`
- The knowledge that the app should point at `~/Documents/Obsidian/Hermes/Grillmaster`
- Limited helper ideas from `utils/vault_reader.py` (concept only, not structure)

### What should be removed or superseded
- `tabs/` architecture entirely
- Current `utils/vault_reader.py` as the primary data layer
- The current top-level app layout and footer logic
- Any hardcoded domain pages that bypass the canonical index + inspect model

---

## Refactor strategy

This is a **replace-in-place rebuild**.

We will:
1. preserve the repo path
2. preserve the idea of a local Streamlit app
3. create the recovered spec’s directory structure alongside the old code
4. switch `app.py` to the new shell only after the new architecture is minimally viable
5. delete/sidelined old `tabs/` code once the new shell/index/inspect path is working

This avoids breaking the repo immediately while still treating the prototype as disposable.

---

## File/dir target structure

The refactor target is:

```text
/Users/admin/Documents/GitHub/grillmaster-command-center/
├── app.py
├── requirements.txt
├── README.md
├── .gitignore
├── .streamlit/
│   └── config.toml
├── data/
│   ├── session.sqlite3
│   └── cache/
├── assets/
│   ├── sigils/
│   └── css/
├── docs/
│   └── plans/
├── core/
│   ├── config.py
│   ├── models.py
│   ├── ids.py
│   ├── frontmatter.py
│   ├── state_machine.py
│   └── bms.py
├── vault/
│   ├── scanner.py
│   ├── parser.py
│   ├── artifacts.py
│   ├── titles.py
│   ├── fragments.py
│   ├── constellations.py
│   └── backlinks.py
├── index/
│   ├── build.py
│   ├── query.py
│   ├── reconcile.py
│   └── schema.sql
├── session/
│   ├── store.py
│   └── sandbox.py
├── ui/
│   ├── theme.py
│   ├── shell.py
│   ├── map.py
│   ├── summon.py
│   ├── tray.py
│   ├── workbench.py
│   ├── inspect/
│   │   ├── base.py
│   │   ├── constellation.py
│   │   ├── title.py
│   │   ├── artifact.py
│   │   └── fragment.py
│   ├── panes/
│   │   ├── left_shell.py
│   │   ├── right_inspect.py
│   │   └── right_assemble.py
│   └── widgets/
│       ├── cards.py
│       ├── badges.py
│       ├── relations.py
│       ├── bms_sigil.py
│       └── previews.py
├── services/
│   ├── summon_service.py
│   ├── relation_service.py
│   ├── promotion_service.py
│   └── restore_service.py
└── tests/
    ├── test_frontmatter.py
    ├── test_title_identity.py
    ├── test_artifact_bundles.py
    ├── test_fragment_locators.py
    ├── test_constellation_write_read.py
    ├── test_bms_balance.py
    ├── test_reconciliation.py
    └── test_session_restore.py
```

---

## Migration rules

### Rule 1: Vault is canonical
Never treat the app DB or session store as canonical. Every refactor decision should preserve the invariant that the Obsidian vault wins.

### Rule 2: Replace dashboard tabs with inspectable object kinds
The new primary objects are:
- title
- artifact
- fragment
- constellation

The old tabs are organizational leftovers and should not survive as first-class architecture.

### Rule 3: Build the new engine before deleting the old prototype
Avoid big-bang deletion until the shell, index, and at least one inspect flow are proven.

### Rule 4: Keep v1 local-first
No deployment, auth, or cloud complexity in this refactor.

### Rule 5: Stay faithful to the recovered build spec
If the existing prototype and the recovered spec conflict, the recovered spec wins.

---

## Concrete implementation plan

### Task 1: Snapshot the current prototype and create the new scaffolding

**Objective:** Freeze the current repo shape for reference, then create the recovered-spec directory scaffold without deleting anything yet.

**Files:**
- Create: `README.md`
- Create: `docs/plans/2026-05-19-grillmaster-command-center-refactor-plan.md`
- Create: `core/__init__.py`
- Create: `vault/__init__.py`
- Create: `index/__init__.py`
- Create: `session/__init__.py`
- Create: `ui/__init__.py`
- Create: `ui/inspect/__init__.py`
- Create: `ui/panes/__init__.py`
- Create: `ui/widgets/__init__.py`
- Create: `services/__init__.py`
- Create: `tests/__init__.py`
- Create: `data/cache/.gitkeep`
- Move or leave intact for now: existing `tabs/` and `utils/`

**Step 1: Write the scaffolding directories and package init files**
Create the directory skeleton from the recovered build spec.

**Step 2: Add a README section called `Prototype status`**
State that the repo is under architectural refactor from a tabbed prototype to a constellation engine.

**Step 3: Verify the tree exists**
Run:
```bash
python - <<'PY'
from pathlib import Path
root = Path('/Users/admin/Documents/GitHub/grillmaster-command-center')
required = [
    'core','vault','index','session','ui','ui/inspect','ui/panes','ui/widgets','services','tests','data/cache','docs/plans'
]
for rel in required:
    p = root / rel
    print(rel, p.exists(), p.is_dir())
PY
```
Expected: all required dirs print `True True`.

**Step 4: Commit**
```bash
git add .
git commit -m "chore: scaffold recovered command center architecture"
```

---

### Task 2: Replace the app entrypoint with the new shell bootstrap

**Objective:** Turn `app.py` from a tab router into the top-level bootstrap for the new single-page shell.

**Files:**
- Modify: `app.py`
- Create: `core/config.py`
- Create: `ui/theme.py`
- Create: `ui/shell.py`

**Step 1: Write `core/config.py`**
Include constants for:
- vault root
- constellation dir
- sqlite db path
- cache dir
- helper to ensure local dirs exist

**Step 2: Rewrite `app.py`**
`app.py` should only:
- call `st.set_page_config(...)`
- call `ensure_local_paths()`
- inject theme via `ui.theme`
- initialize DB/index/session bootstrap hooks
- render `ui.shell.render_app_shell()`

It must no longer import or use `tabs.*`.

**Step 3: Write a placeholder shell**
Render three visible panes with placeholder labels:
- left: shell/oracle
- center: inspect/workbench
- right: operational pane

**Step 4: Verify app launch**
Run:
```bash
cd /Users/admin/Documents/GitHub/grillmaster-command-center
streamlit run app.py
```
Expected: app loads with a 3-pane placeholder shell and no tabs.

**Step 5: Commit**
```bash
git add app.py core/config.py ui/theme.py ui/shell.py
git commit -m "refactor: replace tab app with shell bootstrap"
```

---

### Task 3: Build frontmatter and ID primitives first

**Objective:** Establish the low-level primitives the rest of the system depends on.

**Files:**
- Create: `core/frontmatter.py`
- Create: `core/ids.py`
- Create: `tests/test_frontmatter.py`
- Create: `tests/test_title_identity.py`

**Step 1: Write failing tests for frontmatter round-trip**
Test:
- parse frontmatter + body
- write frontmatter + body
- preserve unicode and multiline text

**Step 2: Implement `read_markdown_with_frontmatter()` and `write_markdown_with_frontmatter()`**
Use PyYAML and UTF-8.

**Step 3: Write failing tests for stable IDs**
Test:
- title ID stability
- artifact ID normalization
- fragment ID from source + hash
- constellation ID slug generation

**Step 4: Implement `core/ids.py`**
Add:
- `slugify()`
- `make_title_id()`
- `make_artifact_id()`
- `make_fragment_id()`
- `make_constellation_id()`

**Step 5: Run tests**
Run:
```bash
pytest tests/test_frontmatter.py tests/test_title_identity.py -v
```
Expected: all pass.

**Step 6: Commit**
```bash
git add core/frontmatter.py core/ids.py tests/test_frontmatter.py tests/test_title_identity.py
git commit -m "feat: add frontmatter and stable id primitives"
```

---

### Task 4: Build the vault scanner/parser layer

**Objective:** Replace `utils/vault_reader.py` with a proper scanner/parser layer from the recovered spec.

**Files:**
- Create: `vault/scanner.py`
- Create: `vault/parser.py`
- Create: `core/models.py`
- Leave for now: `utils/vault_reader.py` (deprecated)

**Step 1: Define lightweight models**
In `core/models.py`, define dataclasses for:
- title record
- artifact bundle
- fragment record
- constellation record
- summon result

**Step 2: Write `vault/scanner.py`**
Add recursive discovery for:
- `.md`
- `.png`, `.jpg`, `.jpeg`
- `.wav`, `.mp3`, `.mp4`

**Step 3: Write `vault/parser.py`**
Add helpers for:
- headings
- wikilinks
- note body extraction
- line-aware fragments

**Step 4: Verify counts**
Run a small script that prints note/media counts from the vault.
Expected: non-zero counts aligned with the current Grillmaster vault.

**Step 5: Commit**
```bash
git add core/models.py vault/scanner.py vault/parser.py
git commit -m "feat: add vault scanner and parser layer"
```

---

### Task 5: Build the SQLite schema and startup refresh pipeline

**Objective:** Create the derived index architecture and wire startup/manual refresh.

**Files:**
- Create: `index/schema.sql`
- Create: `index/build.py`
- Modify: `app.py`
- Modify: `ui/panes/left_shell.py`

**Step 1: Write `schema.sql` from the recovered spec**
Include at minimum:
- titles
- title_occurrences
- artifacts
- artifact_members
- fragments
- constellations
- relations
- session_state
- sandbox_items
- recent_summons

**Step 2: Implement DB bootstrap**
In `index/build.py`, add:
- create/open DB
- load schema
- full refresh entrypoint

**Step 3: Wire startup refresh**
On app load:
- ensure DB exists
- optionally refresh index on startup

**Step 4: Add `Refresh Corpus` button in left pane**
This should trigger the rebuild and surface status.

**Step 5: Verify**
Expected:
- `data/session.sqlite3` exists
- refresh button runs successfully

**Step 6: Commit**
```bash
git add index/schema.sql index/build.py app.py ui/panes/left_shell.py
git commit -m "feat: add derived sqlite index and refresh pipeline"
```

---

### Task 6: Index titles, artifacts, fragments, and saved constellations

**Objective:** Populate the canonical four object kinds from the vault.

**Files:**
- Create: `vault/titles.py`
- Create: `vault/artifacts.py`
- Create: `vault/fragments.py`
- Create: `vault/constellations.py`
- Create: `tests/test_artifact_bundles.py`
- Create: `tests/test_fragment_locators.py`
- Create: `tests/test_constellation_write_read.py`
- Modify: `index/build.py`

**Step 1: Implement title indexing**
Use `Title Catalog.md` as primary source.

**Step 2: Implement artifact bundle discovery**
Start with the recovered v1 heuristics:
- image + companion note
- audio + nearby note
- scores as document artifacts
- major concept docs as document artifacts where useful

**Step 3: Implement fragment extraction**
Use paragraph/heading/quote clusters with source locators.

**Step 4: Implement constellation read path**
Read `Grillmaster/Constellations/*.md` and validate frontmatter.

**Step 5: Run focused tests**
Run:
```bash
pytest tests/test_artifact_bundles.py tests/test_fragment_locators.py tests/test_constellation_write_read.py -v
```
Expected: pass.

**Step 6: Verify with a manual sample query script**
Print sample records for one of each kind.

**Step 7: Commit**
```bash
git add vault/titles.py vault/artifacts.py vault/fragments.py vault/constellations.py index/build.py tests/
git commit -m "feat: index titles artifacts fragments and constellations"
```

---

### Task 7: Build summon/oracle and object routing

**Objective:** Replace dashboard browsing with the recovered summon-first interaction model.

**Files:**
- Create: `index/query.py`
- Create: `services/summon_service.py`
- Create: `ui/summon.py`
- Modify: `ui/panes/left_shell.py`
- Modify: `ui/shell.py`

**Step 1: Implement unified search**
Support search across all four kinds.

**Step 2: Define summon result cards**
Each result shows:
- label
- kind badge
- short description/excerpt
- state if constellation
- collect button
- inspect button

**Step 3: Wire left-pane oracle**
The center of the left pane should be oracle input at idle.

**Step 4: Verify**
Search should find:
- at least one title
- at least one artifact
- at least one fragment
- at least one constellation if seeded

**Step 5: Commit**
```bash
git add index/query.py services/summon_service.py ui/summon.py ui/panes/left_shell.py ui/shell.py
git commit -m "feat: add summon oracle and object routing"
```

---

### Task 8: Build the persistent tray and sandbox draft model

**Objective:** Add collection and assembly primitives.

**Files:**
- Create: `session/sandbox.py`
- Create: `ui/tray.py`
- Create: `ui/workbench.py`
- Modify: `ui/shell.py`

**Step 1: Implement tray state**
Allow add/remove/open inspect/add to draft.

**Step 2: Implement draft constellation state**
Fields:
- draft title
- draft state
- summary
- title IDs
- artifact IDs
- fragment IDs
- unresolved bucket

**Step 3: Render workbench placeholder**
Show collected members as cards/chips with counts.

**Step 4: Verify**
- collect from summon
- tray persists in session
- add items into draft

**Step 5: Commit**
```bash
git add session/sandbox.py ui/tray.py ui/workbench.py ui/shell.py
git commit -m "feat: add collect tray and assembly workbench state"
```

---

### Task 9: Build B/M/S scoring and the inspect system

**Objective:** Implement the metaphysical lens and the four inspect views.

**Files:**
- Create: `core/bms.py`
- Create: `tests/test_bms_balance.py`
- Create: `ui/map.py`
- Create: `ui/inspect/base.py`
- Create: `ui/inspect/title.py`
- Create: `ui/inspect/artifact.py`
- Create: `ui/inspect/fragment.py`
- Create: `ui/inspect/constellation.py`
- Create: `ui/widgets/cards.py`
- Create: `ui/widgets/badges.py`
- Create: `ui/widgets/relations.py`
- Create: `ui/widgets/bms_sigil.py`
- Create: `ui/panes/right_inspect.py`

**Step 1: Write failing B/M/S tests**
Use deterministic member mixes.

**Step 2: Implement scoring**
Use the recovered spec’s evidence-based heuristic counts.

**Step 3: Render the left cosmogram**
Do not over-engineer graphics; a clean triangle/sigil is enough for v1.

**Step 4: Implement shared inspect skeleton**
Sections:
1. Invocation
2. Body
3. Mind
4. Spirit
5. Relations
6. Fragments

**Step 5: Implement all four inspect views**
Each kind should plug into the same skeleton with type-specific emphasis.

**Step 6: Wire right inspect pane**
Must show:
- collect/quick add
- related items
- state + B/M/S
- metadata
- revision controls where relevant

**Step 7: Verify**
Manual inspect of one object from each kind.

**Step 8: Commit**
```bash
git add core/bms.py tests/test_bms_balance.py ui/map.py ui/inspect ui/widgets ui/panes/right_inspect.py
git commit -m "feat: add bms scoring map and inspect system"
```

---

### Task 10: Build promotion, revision, backlink proposals, and session restore

**Objective:** Finish the core constellation workflow end-to-end.

**Files:**
- Create: `ui/panes/right_assemble.py`
- Create: `ui/widgets/previews.py`
- Create: `services/promotion_service.py`
- Create: `vault/backlinks.py`
- Create: `session/store.py`
- Create: `index/reconcile.py`
- Create: `services/restore_service.py`
- Create: `tests/test_session_restore.py`
- Create: `tests/test_reconciliation.py`
- Modify: `vault/constellations.py`
- Modify: `ui/workbench.py`
- Modify: `ui/inspect/constellation.py`

**Step 1: Add right assembly pane**
It must show:
- membership
- B/M/S sigil + reading
- name/state/summary fields
- anchor warnings
- preview controls
- reconciliation issues

**Step 2: Implement note generation preview**
Must produce recovered-spec frontmatter/body template.

**Step 3: Implement write-back to `Grillmaster/Constellations/`**
Create note, rescan, and index.

**Step 4: Implement revise flow**
Load saved constellation to editable draft, preview diff, rewrite in place.

**Step 5: Implement backlink proposal engine**
Preview only; no silent edits.

**Step 6: Implement session persistence and reconciliation**
Persist:
- active inspect target
- recent summons
- tray
- draft constellation
- mode
- lens emphasis

Repair/quarantine:
- missing
- renamed_or_moved
- content_drift
- identity_ambiguity

**Step 7: Run focused tests**
Run:
```bash
pytest tests/test_session_restore.py tests/test_reconciliation.py tests/test_constellation_write_read.py -v
```
Expected: all pass.

**Step 8: Manual v1 verification**
Verify against the recovered spec checklist:
- app launches
- refresh works
- summon works
- inspect works for all four kinds
- collect works
- tray persists
- workbench draft assembles
- B/M/S renders
- promotion preview renders
- promotion writes note to vault
- revision writes back
- backlink proposals preview
- session restore works
- broken refs quarantined
- vault precedence preserved

**Step 9: Remove old prototype code**
Delete or archive:
- `tabs/`
- `utils/vault_reader.py`
- any dead imports/usages

Only do this after the new path is proven.

**Step 10: Commit**
```bash
git add .
git commit -m "feat: complete constellation workflow and retire tab prototype"
```

---

## Non-goals for this refactor
- no deployment work
- no auth or multi-user support
- no heavy graph engine
- no automatic ontology generation
- no silent destructive edits to source notes
- no generalized content-management system beyond the recovered spec

---

## Final acceptance criteria

This refactor is complete when:
- the current tabbed prototype is no longer the primary app architecture
- the app matches the recovered build spec’s shell and workflow
- the app is organized around `title / artifact / fragment / constellation`
- the collect → assemble → promote → revise loop is functional
- notes are written to `Grillmaster/Constellations/` with the recovered frontmatter/body structure
- session restore and reconciliation exist
- old tab code is removed or clearly retired

---

## Execution handoff

Plan complete and saved.

Recommended execution order:
1. scaffold + shell bootstrap
2. frontmatter/IDs
3. scanner/parser
4. sqlite index
5. title/artifact/fragment/constellation indexing
6. summon
7. tray/workbench
8. inspect system + B/M/S
9. promotion/revision/backlinks
10. session restore/reconciliation
11. remove old tabs

Ready to execute using subagent-driven-development — one task cluster at a time, with verification after each milestone.