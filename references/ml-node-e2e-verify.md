# ML Node End-to-End Verification

The Grillmaster `grillmaster-image-pipeline` skill (Pitfalls #18-#20) requires
heavy ML/utility nodes to be **executed end-to-end**, not merely registered. A
node can register cleanly in `/api/node-defs` yet fail at runtime — wrong mask
selection, dead outputs (the `write_mask`/`write_scalars`/`write_field` calls
inside the fn never run), or import-only errors that only surface inside the
function body.

This directory provides a reusable recipe + harness for proving the three ML
nodes actually work:

| Node | ID | Contracted outputs | Key risk |
|------|----|--------------------|----------|
| CLIP Score | `__clip_score__` | IMAGE, SCALAR (`score`), FIELD (`weights`) | CLIP import / weights missing |
| SAM Segment | `__sam_segment__` | IMAGE, MASK (`mask`), SCALAR (`score`) | **Pitfall #20**: `automatic` mode picks the ~86% background mask |
| CLIP-guided SAM | `__clip_sam__` | IMAGE, MASK (`mask`), SCALAR (`score`) | Same background-mask bug, + CLIP crop scoring |

## Prerequisites (offline/cron-safe)

CLIP and SAM weights must already be cached so the probe runs fully offline:

- `~/.cache/clip/ViT-B-32.pt`
- `~/.cache/sam_segment/sam_vit_b_01ec64.pth`

The probe forces `device: cpu` and a small 256x256 canvas so SAM's CPU
inference finishes in seconds. If the checkpoints are missing the first run will
download them (~375MB-2.4GB) — not suitable for a tight cron window, so pre-cache
them on the dev machine.

## Recipe (standalone probe)

Run from the repo root with the project venv (never `PYTHONPATH=$PWD` — the
approval guard flags it as interpreter-hijack; use `env -u PYTHONPATH`):

```bash
cd ~/Documents/GitHub/grillmaster-command-center
env -u PYTHONPATH .venv/bin/python scripts/ml_node_probe.py
```

The probe:

1. Builds a synthetic input (gradient for CLIP; a bright disk on a dark
   background for SAM/CLIP-SAM) and writes it to `_input.png` inside a temp
   output dir — exactly the path the graph executor writes when an upstream
   IMAGE wire is connected.
2. Calls `meta.fn(out_dir, seed=42, params)` with `input_image` set (wired-input
   override path, Rule #12 of the method-file rules).
3. Asserts the contracted artifacts exist and have sane values:
   - `scalars.json` exists with `score` in [0, 1]
   - `mask.npy` exists for SAM nodes
   - `field.npy` exists for CLIP Score
   - **Pitfall #20 regression**: the SAM/CLIP-SAM automatic-mode mask must cover
     **< 0.5** of the canvas (foreground disk chosen, NOT the background).

Exit code is non-zero if any assertion fails. Generated artifacts live under
`_probe_out/` and are cleaned up on exit.

## Recipe (pytest regression backstop)

`image_pipeline/tests/test_ml_nodes_e2e.py` wraps the same asserts as a pytest
suite. It is skipped automatically when CLIP/SAM cannot be imported or when the
checkpoints are absent (`@pytest.mark.skipif`), so CI without the models stays
green, and a dev machine with cached weights gets the real end-to-end coverage.

```bash
cd ~/Documents/GitHub/grillmaster-command-center
env -u PYTHONPATH .venv/bin/python -m pytest \
  image_pipeline/tests/test_ml_nodes_e2e.py -q -p no:cacheprovider
```

## Why this matters

Before this harness existed, the skill's Pitfall #18 said "execute, don't just
register," and #20 documented the SAM background-mask fix — but **nothing in the
repo actually ran the nodes**. The fix was correct (verified here: SAM automatic
mode now reports coverage 0.101 on a synthetic disk, down from the ~0.86
background-mask trap), but it had no regression guard. A future edit that
reintroduced `max(pool, key=lambda m: m["segmentation"].sum())` would silently
re-break the node, and the only signal would be a user noticing "SAM segments
everything." This harness turns that into a failing test.

## Result (verified 2026-07-10, cached ViT-B-32 + vit_b)

| Node | result | mask coverage | notes |
|------|--------|---------------|-------|
| `__clip_score__` | PASS | n/a | score=0.202 over 5 labels; FIELD shape (256,256,5) |
| `__sam_segment__` | PASS | 0.101 | background mask correctly dropped (Pitfall #20) |
| `__clip_sam__` | PASS | 0.101 | CLIP picked candidate 4/5 for "a white circle" |
