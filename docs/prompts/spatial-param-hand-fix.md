# Prompt — Hand-Fix One Node's Spatial Params

**Give this prompt to the agent once per node.** One node per run, no batching —
the automated pass (`tools/migrate_spatial.py`) already took everything that
could be taken mechanically. What is left needs a human-shaped decision about
one specific node, and batching those hides which change caused which result.

It supplements — does not replace — `docs/plans/2026-07-22-spatial-param-plan.md`
(why this exists), `AGENT_GUIDE.md` (the method contract) and
`image_pipeline/core/spatial.py` (the `sparam` contract).

Your target: method id **`{{METHOD_ID}}`**, param **`{{PARAM}}`**.

---

## 0. Mission

Make a FIELD wired to `{{PARAM}}` vary this node's output **per pixel**, then
prove it. Done means all three of these hold:

1. `tools/audit_field_response.py --ids {{METHOD_ID}}` reports **SPATIAL** for `{{PARAM}}`.
2. With nothing wired, the node's output is **bit-identical** to before your change.
3. `pytest image_pipeline/tests/test_spatial_params.py -k "{{METHOD_ID}}"` is green.

If you cannot reach all three, the correct outcome is **reclassify, not force**
(§5). A node left half-converted — declaring `spatial: True` while collapsing
the field internally — is worse than one that declares nothing, because the gate
and the UI both now treat that declaration as a promise.

**Do not touch** `core/graph.py`, `core/spatial.py`, or anything in `tools/`.
The plumbing is verified. Fix the *method*.

---

## 1. Read the recorded verdict first

The automated pass already tried this param and wrote down how it failed:

```bash
python -c "import json;d=json.load(open('data/spatial/spatial_failures.json'));print(d.get('{{METHOD_ID}}|{{PARAM}}','(not attempted)'))"
```

That verdict tells you which of §3's failure modes you are in. `(not attempted)`
means the automated pass never tried it — normal for anything not class A, and
normal for all three targets in the last section.

Also pull the static classification and its reason:

```bash
grep '^[A-D],{{METHOD_ID}},' data/spatial/param_ledger.csv | grep ',{{PARAM}},'
```

Both of these are **evidence, not verdicts**. The classifier is a static guess
tuned to be pessimistic: a `B` is usually right and usually means §5, but it also
labels anything crossing a helper boundary `B`, and it labels a field that is
read-then-dropped `C` ("never used arithmetically") — which is a *bug
description*, not a reason to skip. Conversely an `A` is a candidate and never a
promise; #45 Graphviz is class A and still cannot work. Let the probe decide.

---

## 2. Trace the value from read site to pixels

Find where the param is read, then follow the variable it binds **all the way to
the array that gets rendered**. This is the whole job; the edit itself is small.

```bash
grep -n '{{PARAM}}' <method file>
```

The trap that motivated this prompt: **#01 ASCII Art** reads `_field_font_size`,
builds a genuine per-pixel `font_size_arr`, threads it down two call layers as
`fs_arr=` — and the receiver does `_fs = int(np.median(fs_arr))`. Two layers of
correct-looking spatial code, collapsed on the third. Following the value only
one hop would have called that node fixed.

So: do not stop at the read site. Stop at the pixels.

---

## 3. The five failure modes, and the fix for each

### A. Mean/median collapse (`MEAN_ONLY`)
Somewhere between read and render, the array meets `np.mean`, `np.median`,
`float()`, or `.item()`. Fix by keeping the array whole — the surrounding math
is usually already elementwise and will broadcast unchanged.

If the collapse is inside a helper shared with other callers, keep the helper
scalar-compatible: accept either, and branch on `is_field(x)` rather than
forcing every caller to pass an array.

### B. A preset overwrites the wired value (`MEAN_ONLY`)
The node assigns the param from a regime/preset table *after* reading it, so the
wired field is silently discarded. This is what **#155 Gray-Scott** did:
`anim_mode` defaults to the named regime `"spots"`, whose `F`/`k` assignment
clobbered the field.

Fix — a wired FIELD outranks a preset:

```python
if not is_field(F):
    F = regime["F"]
```

Check for this whenever the node has `anim_mode` choices that read like
parameter presets. It is invisible in the diff and common.

### C. The array reaches a scalar-only use (`ERROR`)
`range()`, `int()`, an index, an array shape, a PIL/cv2 call, a subprocess
argument, or an f-string fed to a CLI. **#45 Graphviz** shells out to `dot` with
numeric args — an array cannot survive that.

Two honest options, in order of preference:
1. **Restructure** so the math broadcasts — viable when the scalar use is
   incidental (a bound, a clip limit).
2. **Give up on this param** — go to §5. Do NOT sprinkle `as_scalar()` to make
   the error disappear; that converts a loud failure into a silent `MEAN_ONLY`,
   which is the exact pathology this whole effort exists to remove.

### D. Read site not in the expected shape (`SKIPPED_SHAPE`)
The codemod only rewrites `x = float(params.get("name", default))`. Yours is
computed, nested, conditional, or multi-assigned. Convert it by hand:

```python
x = sparam(params, "{{PARAM}}", <default>)
```

Add `from image_pipeline.core.spatial import sparam, is_field` if absent, and
`"spatial": True,` to the param's spec dict in the `@method` decorator. Both are
required — the flag is what grants the FIELD port.

### E. Unprobeable in isolation (`NOT_PROBED`)
The node needs an external binary or a GPU context, so it cannot render
standalone. Do not mark it `spatial: True`: the gate would fail on it forever.
Go to §5.

**A missing input image is no longer this case.** The probe wires a concentric-
gradient source into any node declaring `image_in`, and probes filters both with
and without it, taking whichever responds. Before that, ASCII Art #01 rendered a
solid error frame with nothing upstream, so every param read as MEAN_ONLY no
matter what the code did — a constant output cannot demonstrate anything. If you
hit a node whose output has one unique value, check that before blaming the param:

```python
img = _render(mid, {}); print(img.std(), len(np.unique(img)))
```

### F. The param is inert at the node's defaults (false `MEAN_ONLY`)
The param is wired correctly but mathematically cancels under default settings,
so every field renders identically. #11 Gradient's `cy` multiplies
`sin(direction)`, which is 0 at the default `direction=0` — a horizontal linear
gradient genuinely has no Y centre. Correct physics, not a broken param.

Confirm by rendering it yourself under a setting that should expose it (for `cy`:
`gradient_type=radial` gave Δ=0.079 where `linear` gave exactly 0). If it
responds there, declare the configuration that makes it observable:

```python
"cy": {"spatial": True, "probe_with": {"gradient_type": "radial"}, ...}
```

The probe merges `probe_with` into every render. Use it only for genuine
inertness you have demonstrated — it is not a way to hunt for a setting where a
broken param happens to wiggle.

### Bonus: `ORIENTED?`
Responded to the ramp, but horizontal and vertical ramps produced identical
output — usually a reduction along one axis (`.mean(axis=0)`, a per-row loop).
Treat as mode A, scoped to the axis being collapsed.

---

## 4. Verify — all three, in this order

**1. The field reaches the pixels:**
```bash
python tools/audit_field_response.py --ids {{METHOD_ID}}
```
Requires `SPATIAL`. `Δuniform` is response vs a same-mean uniform field;
`Δorient` separates real spatial response from noise. Both must exceed 0.

**2. Unwired output is unchanged** — this is the non-negotiable one. `sparam`'s
scalar path must stay bit-identical, or you have perturbed every existing graph
using this node:

```bash
python - <<'PY'
import sys, tempfile, numpy as np
from pathlib import Path
sys.path.insert(0, '.')
import image_pipeline.methods
from image_pipeline.core.graph import GraphExecutor, GraphNode
from image_pipeline.core.utils import set_canvas
set_canvas(96, 72)
n = GraphNode(id="n0", method_id="{{METHOD_ID}}", params={})
with tempfile.TemporaryDirectory() as t:
    ex = GraphExecutor(out_dir=Path(t), fps=24, in_memory=True)
    flat, _, err = ex.execute(nodes=[n.__dict__], edges=[], seed=42, frame=4, frames=5)
    assert not err, err
    np.save(sys.argv[1] if len(sys.argv) > 1 else "/tmp/spatial_check.npy", flat["n0"]["image"])
PY
```
Run it on `git stash`'d code and again on yours; require `max|Δ| == 0.0`.

**3. The gate passes:**
```bash
python -m pytest image_pipeline/tests/test_spatial_params.py -k "{{METHOD_ID}}" -q
```

Then re-run the node's own tests if it has any. Report the actual numbers — a
claim of "works now" without a `Δuniform` and a `max|Δ| == 0.0` is not a result.

---

## 5. When to stop — reclassifying is a valid, correct outcome

Stop and reclassify if the param needs a real scalar to do its job (mode C or E,
or restructuring that would change the node's algorithm). Structural params are
not failures of this effort; they are a different, larger piece of work.

To reclassify, record it so nothing retries it blindly:

```bash
python -c "
import json,pathlib
p=pathlib.Path('data/spatial/spatial_failures.json'); d=json.loads(p.read_text())
d['{{METHOD_ID}}|{{PARAM}}']='STRUCTURAL_BY_HAND'
p.write_text(json.dumps(dict(sorted(d.items())),indent=2))"
```

Then revert your edits (`git checkout -- <method file>`) and say plainly which
scalar-only use blocked it. "`tile_size` drives `range()` in the tiling loop; per-pixel
tile size needs variable-rate tiling, which is an algorithm change" is a complete
and useful answer. Leave the node exactly as you found it.

---

## 6. Hard rules

- **Never** add `"spatial": True` to a param that has not probed `SPATIAL`.
- **Never** use `as_scalar()` to silence an `ERROR`. That trades a loud failure
  for a silent one.
- **Never** change unwired behaviour. Bit-identical or it is not done.
- **One node per run.** Do not opportunistically fix a neighbour you noticed.
- Prefer restructuring the math to branching on `is_field()`; reach for the
  branch only where a preset or a shared helper genuinely needs both paths.
- If the node has no `image` output it is a driver (LFO, Counter) — per-pixel is
  meaningless there. Stop immediately and reclassify.

---

## 7. Report back

- verdict before → after, with `Δuniform` / `Δorient`
- `max|Δ|` for the unwired render (must be `0.0`)
- which failure mode it was, and the one-line reason
- anything you found that the automated pass or this prompt should have caught —
  that belongs back in this file or in `tools/classify_params.py`

---

## Known outstanding targets

Three nodes read `_field_*`, advertise support, and honour none of it — the 12
remaining `MEAN_ONLY` in `data/spatial/field_response_report.json`. They are the
highest-value hand-fixes because they actively mislead.

**None of them appear in `data/spatial/spatial_failures.json`.** The automated pass only
attempts class A, and most of their params are not class A — so §1 will report
`(not attempted)`. That is expected here, not a sign you picked a bad target.
Their real ledger state:

| id | node | params | ledger class | expected mode |
|---|---|---|---|---|
| `01` | ASCII Art | `font_size`, `char_spacing` | **B** — `_render_ascii()` | A — `int(np.median(fs_arr))` inside that helper |
| `11` | Gradient | `cx`, `cy`, `direction` | **C** — "never used arithmetically" | A — `_field_*` read into locals, then dropped |
| `45` | Graphviz | `node_count`, `edge_density` | **A** | C — values become `dot` CLI args |

Read those classes as a warning, not a verdict — each is a *static* guess the
probe already contradicts:

- **`11` — DONE (2026-07-22).** `cx`, `cy`, `direction` all probe SPATIAL. Four
  distinct modes in one node: dead `_field_*` reads (A), `math.cos/sin` refusing
  arrays so the whole helper moved to `np.*` (C-restructure), `center_orbit` /
  `direction_morph` assigning over the wired map (B), and `cy` inert at the
  default linear/direction=0 (F). Also exposed a real gap in the port generator:
  an input declared in `inputs=` that shares a param name was never added to
  `param_ports`, so the UI did not know the port was param-backed. Fixed in
  `_make_node_def`. Expect a node to need more than one mode.
- **`01` — DONE (2026-07-22), but not the way this table predicted.**
  `dither_strength` is SPATIAL (Δuniform=0.00829). `font_size` and `char_spacing`
  went to §5: they set the glyph grid geometry (`step_x`/`step_y`, then the
  cols/rows the render loop walks), so they need one number, and the
  `np.median(fs_arr)` was the node *pretending* otherwise. The dead
  `fs_arr`/`sp_arr` plumbing is deleted rather than left advertising a
  capability — reclassifying means removing the claim, not just the flag.
  The real blocker was mode E: with nothing wired the node returned a solid
  error frame, so nothing could ever probe SPATIAL. Fixed in the probe.
- **`45` — DONE (2026-07-22), via §5.** Class A, and still impossible: every one
  of its four params becomes a Graphviz DOT attribute or a loop bound
  (`range(use_n_nodes)`, `fontsize={use_font_size}`, `len={use_edge_len}`). The
  four FIELD ports were changed to SCALAR and the `np.mean` blocks deleted, so
  the node stops advertising per-pixel support. The clearest case in the repo of
  class A being a candidate, never a promise — and of §5 meaning *remove the
  claim*, not merely skip the work.

All three legacy liars are now resolved; the scan is 131/131 SPATIAL.
