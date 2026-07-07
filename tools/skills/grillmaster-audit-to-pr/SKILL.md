---
name: grillmaster-audit-to-pr
description: Workflow for doing an audit-driven code review of the Grillmaster Command Center (image_pipeline + chord_bot) and landing fixes as verified, per-concern PRs. Use when the user pastes an audit/CODE_AUDIT doc and says "optimize / fix / tackle / take them on / pr then move on". Covers the hard rule that audit claims must be VERIFIED against source before acting, the "fix only the genuine bug" discipline, and the per-concern branch+PR pattern.
---

# Grillmaster Audit → PR Workflow

The Grillmaster Command Center (`documents/github/grillmaster-command-center`, package `image_pipeline`) is a 160+ node generative graph with a FastAPI server (`server.py`), an executor (`core/graph.py`), a method decorator registry (`core/registry.py`), and a single animation clock (`core/timeline.py` → `KeyframeTrack` + `core/graph._evaluate_param_track`). Methods live under `image_pipeline/methods/<category>/`.

## When to use
User drops a design/audit doc (`DESIGN.md`, `CODE_AUDIT_*.md`, `tools/audit_report.md`) and asks to act on it ("take them on", "tackle it", "pr then move on", "yes do that").

## The one hard rule (most important)
**The audit's listed problems are hypotheses, not facts. Verify every claim against source BEFORE editing.** In practice the audit was frequently WRONG:
- It flagged 35 `luminance:"FIELD"` declarations as "mismatches" — but FIELD is correct (`luminance = np.mean(arr, axis=-1)` is a `(H,W)` array = FIELD). The real bug was elsewhere (see below).
- It flagged "phantom output ports" on `noise_node`/`test_node` — but their custom outputs (`amplitude`, `test_scalar`) ARE returned, and `luminance` is a valid executor-synthesized port. No churn needed.
- It counted "183 no description" but ground truth was 181 (and 0 files had ANY `description=`).

So: read the actual code, find the GENUINE defect, fix that, and explicitly NOTE where the audit was wrong rather than manufacturing edits to match its list.

## Sequence
1. **Map + read docs.** `find` the tree (skip `.venv`,`.git`,`__pycache__`). Read the audit doc AND `DESIGN.md` AND `AGENT_GUIDE.md`. The audit gate is `tools/audit_methods.py --fail-on-violations` (hard violations only; ~254 pre-existing warnings are intentionally untouched).
2. **Triage to a priority list.** Group findings; separate genuine bugs from style from "already done". Present the list; the user typically says "take them on" / "tackle it" = do them in order.
3. **For each concern: verify → fix → verify.**
   - Verify the claim with `search_files` / `read_file` / a small `execute_code` scan. If the claim is wrong, say so and move on (no edit).
   - Fix only the genuine defect. Prefer surgical, behavior-preserving changes.
   - Verify with: `py_compile` on changed files, re-run the audit gate (must stay green), and a targeted unit test under a working interpreter (see Environment).
4. **Per-concern branch + PR.** `git checkout -b <type>/<short-name>`, commit, `git push -u origin`, report the PR URL. Types: `fix/`, `perf/`, `refactor/`, `chore/`. One PR per concern so each is reviewable independently. The user reviews/merges; do NOT merge yourself unless told.

## Environment gotchas (learned the hard way)
- The project `.venv` often has a **broken numpy** (`ModuleNotFoundError: numpy._core._multiarray_umath`) — a pre-existing issue, NOT from your edits. The running server on `:7860` is usually a STALE process started from a good state.
- A working interpreter for unit tests is typically `/Users/admin/.local/bin/python3.11` (has numpy+fastapi+PIL) or `.venv/bin/python` AFTER the venv is healed. Confirm with `import numpy` before relying on it.
- The running server emits a `pipeline-server.log` (under `data/logs/`) — grep it for ground-truth proof of live-mode bugs (e.g. `PermissionError … utils.py:248 write_field`).
- CLI/sequence param animation used to be a SEPARATE server-side tween (`_interpolate_params`); it was folded into `paramKeyframes` via `_merge_anim_params_into_nodes`. The single animation model is now `paramKeyframes` (KeyframeTrack) + `Timeline` clock.
- `luminance` is ALWAYS a `(H,W)` FIELD (executor recomputes `np.mean(arr, axis=-1)` at `core/graph.py` in 3 paths — keep them uniform). Declared default type is FIELD (`registry.py`, `server.py` manifest).
- Node ids are hardcoded as literals in `capture_frame("NN", …)` / `mn(NN, …)`. The executor installs a per-thread method-id context (`utils.set_method_id`) so renumbers are safe — do NOT hand-edit 50 literals; rely on the context. Verify renumber-safety with a unit test, not by rewriting each call site.

## Verification checklist (run before each PR)
- [ ] `python -m py_compile <changed files>` clean
- [ ] `python tools/audit_methods.py --fail-on-violations` → "No hard violations"
- [ ] Targeted unit test proving the fix (e.g. folded tween ≈ old tween within tolerance; captured sidecars go to memory not disk; renumber-safe filename generation)
- [ ] If touching live mode: grep `pipeline-server.log` for the prior error to confirm it's gone (or note that a live restart is needed)
- [ ] Note explicitly where the audit was WRONG (so the user trusts the negative result)

## Pitfalls
- Don't trust `search_files` counts blindly — a regex anchored to line-start misses `name=` mid-line in single-line decorators; use `ast` for decorator boundaries.
- Don't blanket-flip 160 nodes' `is_time_varying` — methods that read `time`/`frame`/`anim_mode` genuinely animate; marking them `False` freezes them in live mode (the failure DESIGN.md invariants prevent). Only flip nodes proven static (no time/frame/rng/state evolution).
- The `np.mgrid` global monkeypatch in `noise_node.py` (`yy, xx = np.mgrid[:h,:w]`) is deliberate shared state across many noise/sim methods — leave it unless the user explicitly wants the risky refactor.
- Pyright "reportOptionalMemberAccess" warnings on `group(1) if x else None` in tool scripts are false positives; ignore for one-off `tools/` scripts.
