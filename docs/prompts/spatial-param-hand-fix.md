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
python -c "import json;d=json.load(open('tools/spatial_failures.json'));print(d.get('{{METHOD_ID}}|{{PARAM}}','(not attempted)'))"
```

That verdict tells you which of §3's failure modes you are in. `(not attempted)`
means the automated pass never tried it — normal for anything not class A, and
normal for all three targets in the last section.

Also pull the static classification and its reason:

```bash
grep '^[A-D],{{METHOD_ID}},' tools/param_ledger.csv | grep ',{{PARAM}},'
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
The node needs a wired input image, an external binary, or a GPU context, so it
cannot render standalone. Do not mark it `spatial: True`: the gate would fail on
it forever. Go to §5.

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
p=pathlib.Path('tools/spatial_failures.json'); d=json.loads(p.read_text())
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
remaining `MEAN_ONLY` in `tools/field_response_report.json`. They are the
highest-value hand-fixes because they actively mislead.

**None of them appear in `tools/spatial_failures.json`.** The automated pass only
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

- **`11` is the best first target.** Class C "never used arithmetically" is
  literally true of `cx_field` and describes the bug: the field is read and
  discarded. Wiring it into the existing `effective_x` / `effective_y` math is
  small. The classifier cannot distinguish "unused because structural" from
  "unused because someone forgot".
- **`01` is class B only because its params cross a helper boundary.** The
  per-pixel array already reaches `_render_ascii()`; the collapse is the
  `np.median` inside it. Fix the helper, per §3A's note on shared helpers.
- **`45` is class A but will likely end at §5.** Static analysis sees arithmetic
  and stops; it cannot see the values being formatted into `dot` arguments. This
  is the clearest case in the repo of class A being a candidate, never a promise.
