# Feature Backlog

> Candidate features and enhancements, distinct from debt/refactors. Ranked by
> rough value/effort. Drawn from the architecture review, user-value reasoning,
> and gap-spotting during the engineering loop.

---

## Ranked Candidates

| ID | Feature | Value | Effort | Notes |
|----|---------|:---:|:---:|-------|
| FB-01 | Expose Node Doctor / quality check results in UI render output | High | Med | `core/quality.py` already tested & stable (TD-14) |
| FB-02 | One-click "export graph as reusable preset" from the editor | Med | Med | Builds on existing graph save/load |
| FB-03 | Per-node sim-cache byte budget with eviction (TD-03) | High | Low | Prevents single sim OOM-ing the process |
| FB-04 | Comparison/storyboard view: render N seeds side-by-side | Med | Med | Useful for side-by-side curation |
| FB-05 | Keyboard-driven node graph editing (add/connect/delete) | Med | High | Power-user ergonomics |
| FB-06 | Live preview of a *sub-branch* (render only selected downstream) | Med | Med | Speeds iteration on large graphs |
| FB-07 | Undo/redo history for the graph editor | Med | High | Expected of any node editor |
| FB-08 | Parameter randomize-within-range control | Low | Low | Quick variation generation |
| FB-09 | Batch re-render with param sweep (grid search UI) | Med | Med | Standalone — no existing store to build on |
| FB-10 | Annotated error overlay on failing node in the editor | Med | Low | Surfaces `_write_error_placeholder` reason |

---

## Selection Rule

Pick the top backlog item whose effort fits the current loop budget AND that is
*not* blocked by missing tests/refactors. Re-rank whenever a debt item closes.
