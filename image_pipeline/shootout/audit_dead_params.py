"""Headless per-node liveness audit (Route 8 — dead-param frontier).

The shootout dead-rate was investigated end-to-end (2026-07-18):
  * Route 8 #1 (driver path broken)  -> DISPROVEN: driver-wired genomes die
    at the *baseline* 45% rate, and ``test_driver_wired_reaches_pixels``
    already guards the driver -> SCALAR -> param injection path. Drivers ARE
    sampled every frame.
  * Route 8 #2 (render timeouts)   -> ALREADY FIXED: cost_proxy.py (structural
    ridge regressor) + effective_render_timeout_s + liveness-prior exemption.
    All 58 ``timeout`` deaths in the corpus predate that fix (legacy).
  * born-animated guarantee            -> ALREADY WORKS: ensure_animated (generator.py,
    2026-07-15). All 58 "neither driver nor anim_mode" static deaths
    predate it.

The ONE remaining failure mode is **node-level dead params**: of the 149
``static``/``flat`` deaths, 69 DO contain a driver and 72 DO contain a
non-``none`` anim_mode -- yet they render static. That means the driven
node's own animation logic is a no-op in its render math (the exact
pitfall #4 / #19 / Step-4/5 class the 8-step audit targets: a loop var
or per-frame normalization silently cancels the time/param, so the slider
does nothing and the clip is culled as static).

This script detects exactly that: for every registered time-varying method it
renders the node ALONE in one of its non-``none`` anim_modes and measures
whether the animation actually reaches the pixels (changed_frac + temporal_var,
using the SAME formulas as the liveness gate / test_driver_wired_reaches_pixels).

Usage:
    python3 image_pipeline/shootout/audit_dead_params.py            # all time-varying
    python3 image_pipeline/shootout/audit_dead_params.py --ids 141,137,97
    python3 image_pipeline/shootout/audit_dead_params.py --limit 20

Output: a ranked report written to
    image_pipeline/shootout/data/dead-param-audit.md
and printed to stdout. Suspects (changed_frac <= 0.10 AND temporal_var
<= 1e-3 in a mode that SHOULD move) are the actionable dead-param nodes.

Monotonic-safe: read-only, renders in-memory, writes one report file. Touches
no node, no server path, no GPU.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

import numpy as np

# Self-bootstrap: this script lives under image_pipeline/shootout/, but the
# package import needs the REPO root on sys.path. (parents[2] from here.)
_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

# Register every node (populates core.registry) before reading defs.
import image_pipeline.methods  # noqa: F401
from image_pipeline.core.graph import (  # proven in-process render path
    GraphExecutor,
    GraphEdge,
    GraphNode,
    get_all_node_defs,
)

REPORT_PATH = Path(__file__).resolve().parent / "data" / "dead-param-audit.md"

# Liveness floor -- mirrors test_driver_wired_reaches_pixels / the gate rescue
# thresholds. A genuinely animating node clears these easily; a dead-param
# node does not.
CHANGED_FLOOR = 0.10
TEMPORAL_VAR_FLOOR = 1e-3

N_FRAMES = 24          # total frames for the time ramp
QUARTER = 6            # frame 6 ~ quarter-cycle (opposite sine extreme)
STACK_N = 8             # frames used for temporal_var


def _changed_frac(a: np.ndarray, b: np.ndarray, thr: float = 0.05) -> float:
    diffs = np.abs(a.astype(np.float64) - b.astype(np.float64))
    return float((diffs > thr).mean())


def _temporal_var(stack: list[np.ndarray]) -> float:
    arr = np.stack([s.astype(np.float64) for s in stack])
    return float(arr.var(axis=0).mean())


def _derive_modes_from_source(mid: str) -> list[str]:
    """Recover the anim_mode enum from a method's own source via AST.

    Mirrors the server's ``_derive_anim_mode_choices``: many methods alias
    ``anim_mode`` to a local (``mode = params.get("anim_mode", "none")``) and
    branch on that, so the modes appear NEITHER in an explicit ``choices`` list
    NOR in a paren slash-list description — they live only in the method body.
    Reading the source recovers them and closes the last audit blind spot
    (the alias-only case the description regex cannot see).

    Returns the ordered non-'none' mode list, or ``[]`` if unrecoverable.
    """
    from image_pipeline.core import registry as _reg
    meta = _reg.get_meta(mid)
    fn = getattr(meta, "fn", None) if meta else None
    if fn is None or not hasattr(fn, "__code__"):
        return []
    import ast
    import inspect
    try:
        tree = ast.parse(inspect.getsource(fn))
    except (OSError, TypeError, SyntaxError):
        return []
    # Find the local name bound to anim_mode (direct or via params.get).
    anim_var = "anim_mode"
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and len(node.targets) == 1:
            tgt, val = node.targets[0], node.value
            if isinstance(tgt, ast.Name) and isinstance(val, ast.Call) \
                    and isinstance(val.func, ast.Attribute) and val.func.attr == "get" \
                    and val.args and isinstance(val.args[0], ast.Constant) \
                    and val.args[0].value == "anim_mode":
                anim_var = tgt.id
            elif isinstance(tgt, ast.Name) and isinstance(val, ast.Name) and val.id == "anim_mode":
                anim_var = tgt.id
    found: list[str] = []
    seen: set[str] = set()

    def _add(v):
        if isinstance(v, str) and re.fullmatch(r"[A-Za-z0-9_\-]+", v) and v not in seen:
            seen.add(v)
            found.append(v)

    for node in ast.walk(tree):
        cmp = node.test if isinstance(node, ast.If) and isinstance(node.test, ast.Compare) else \
            (node if isinstance(node, ast.Compare) else None)
        if cmp is None:
            continue
        operands = [cmp.left] + list(cmp.comparators)
        if anim_var in {o.id for o in operands if isinstance(o, ast.Name)}:
            for o in operands:
                if isinstance(o, ast.Constant) and isinstance(o.value, str):
                    _add(o.value)
                elif isinstance(o, (ast.Tuple, ast.List, ast.Set)):
                    for elt in o.elts:
                        if isinstance(elt, ast.Constant) and isinstance(elt.value, str):
                            _add(elt.value)
    return [m for m in found if m.lower() not in ("none", "", "off")]


def _non_none_modes(defn: dict, mid: str | None = None) -> list[str]:
    """Return the non-'none' anim_mode / animation_mode choices for a node def.

    Layered recovery (each fallback catches a strictly harder case):
    1. explicit ``choices`` list in the param spec,
    2. a paren slash-list in the param description (e.g.
       ``"animation mode (none/phase/draw/rotate)"``),
    3. an AST scan of the method's own source (``mid`` given) — catches modes
       declared ONLY via an aliased local (``mode = params.get("anim_mode")``),
       which neither (1) nor (2) can see. This is the last audit blind spot the
       prior runs flagged (false DEAD-PARAM verdicts from a frozen ``none``
       clock — hit 406 Harmonograph / 402 Kaleidoscopic IFS).
    """
    out: list[str] = []
    for key in ("anim_mode", "animation_mode"):
        spec = (defn.get("params") or {}).get(key)
        if not spec:
            continue
        choices = list(spec.get("choices") or [])
        if not choices:
            # Fallback 2: paren slash-list in the description (>= 2 items).
            m = re.search(r"\(([a-z_]+(?:/[a-z_]+)+)\)", str(spec.get("description", "")))
            if m:
                choices = m.group(1).split("/")
        for c in choices:
            cs = str(c).lower()
            if cs not in ("none", "", "off"):
                out.append(str(c))
    # Fallback 3: AST recovery from the method source (alias-only modes).
    if not out and mid is not None:
        params = defn.get("params") or {}
        if any(k in params for k in ("anim_mode", "animation_mode")):
            out = _derive_modes_from_source(mid)
    return out


def _has_time_param(defn: dict) -> bool:
    """True when a node animates via the injected ``time`` clock (Architecture B).

    The GraphExecutor sets ``run_params["time"] = timeline.phase`` every frame
    (graph.py), so any node that DECLARES a ``time`` param (or a ``phase``
    alias) evolves across frames WITHOUT an ``anim_mode`` enum. These nodes were
    previously reported ``no-anim-mode`` and never rendered — a blind spot that
    hid genuine dead-``time`` nodes (the Arch-B analogue of the pitfall #4/#19
    dead-param class). Detect them so the audit can exercise the time path.
    """
    params = defn.get("params") or {}
    return any(k in params for k in ("time", "phase"))


def _time_varying_ids() -> list[str]:
    defs = get_all_node_defs()
    ids: list[str] = []
    for mid, defn in defs.items():
        # Explicit flag
        if defn.get("is_time_varying"):
            ids.append(mid)
            continue
        # An anim_mode enum with a non-'none' choice
        if _non_none_modes(defn, mid):
            ids.append(mid)
            continue
        # Category hints for known time-varying families
        cat = (defn.get("category") or "").lower()
        if cat in ("sim", "simulation", "patterns", "noise", "fractal", "field"):
            # only keep if it actually exposes a time-related param
            params = defn.get("params") or {}
            if any(k in params for k in ("time", "anim_speed", "phase")):
                ids.append(mid)
    return ids


def _render(mid: str, params: dict, frame: int, seed: int = 42) -> np.ndarray:
    node = GraphNode(id="n0", method_id=mid, params=dict(params))
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        ex = GraphExecutor(out_dir=Path(tmp), fps=24, in_memory=True)
        flat, terminal, errors = ex.execute(
            nodes=[node.__dict__],
            edges=[e.__dict__ for e in []],
            seed=seed,
            frame=frame,
            frames=N_FRAMES,
        )
        if errors:
            raise RuntimeError(f"node errors: {errors}")
        if terminal != "n0":
            raise RuntimeError(f"expected terminal n0, got {terminal}")
        img = flat["n0"]["image"]
        if not isinstance(img, np.ndarray) or img.ndim != 3:
            raise RuntimeError("no image out")
        return img.astype(np.float32)


def _anim_param_key(defn: dict) -> str:
    """Return the node's real animation-mode param key.

    Most nodes declare ``anim_mode`` (the pipeline canonical key, see
    graph.py / the executor). A handful of older fractal nodes (49 Buddhabrot,
    51 Burning Ship, 52 Newton) declare ``animation_mode`` instead. The audit
    must inject the SAME key the node actually reads, otherwise it always
    sees ``none`` and a genuinely-animating node is mis-classified as a
    dead-param suspect. Route 8 (2026-07-18): closes this last audit blind
    spot — the harness now honours both spellings.
    """
    params = defn.get("params") or {}
    if "anim_mode" in params:
        return "anim_mode"
    if "animation_mode" in params:
        return "animation_mode"
    return "anim_mode"  # fallback: most nodes use this


def audit_node(mid: str, defn: dict, seed: int = 42, cheap: bool = False) -> dict:
    """Audit one node.

    When ``cheap`` is True, render only 3 frames (f=0, QUARTER, 2·QUARTER)
    instead of the full STACK_N (8) — the high-frequency (cosmetic) modes the
    full stack would expose are irrelevant to the binary dead/alive frontier,
    and the three spread samples still catch a genuinely dead param (which is
    flat at EVERY phase). The cheap path is ~2.7× faster and is the mode the
    cron budget needs to finish the 455-node frontier in one run.
    """
    modes = _non_none_modes(defn, mid)
    anim_key = _anim_param_key(defn)
    # sensible source so motion is visible where the node needs structure
    base: dict[str, object] = {"anim_speed": 1.0}
    params_schema = defn.get("params") or {}
    if "source" in params_schema and "noise" in (params_schema["source"].get("choices") or []):
        base["source"] = "noise"
    result = {
        "id": mid,
        "name": defn.get("name", mid),
        "modes": modes,
        "status": "ok",
        "best_mode": None,
        "best_changed": 0.0,
        "best_tvar": 0.0,
        "detail": "",
    }

    def _probe(rendered: list[np.ndarray]) -> tuple[float, float]:
        """changed_frac(t0 vs quarter) + temporal_var over the rendered stack."""
        if len(rendered) < 2:
            return 0.0, 0.0
        changed = _changed_frac(rendered[0], rendered[len(rendered) // 2])
        tvar = _temporal_var(rendered)
        return changed, tvar

    def _verdict(changed: float, tvar: float, label: str) -> str:
        if changed <= CHANGED_FLOOR and tvar <= TEMPORAL_VAR_FLOOR:
            return "DEAD-PARAM (suspect)"
        if changed <= CHANGED_FLOOR:
            return "weak (changed<=floor)"
        return "alive"

    if not modes:
        # ── Architecture-B fallback: no anim_mode enum, but the node may
        # animate purely via the injected ``time`` clock. Render across the
        # frame stack (the executor varies ``time`` = timeline.phase per frame)
        # and measure whether the time path reaches pixels. This turns the
        # former "no-anim-mode" blind spot into a real verdict.
        if _has_time_param(defn):
            result["modes"] = ["<time>"]
            if cheap:
                frames = [0, QUARTER, min(2 * QUARTER, N_FRAMES - 1)]
            else:
                frames = list(range(STACK_N))
            try:
                stack = [_render(mid, dict(base), frame=f, seed=seed)
                         for f in frames]
                changed, tvar = _probe(stack)
            except Exception as e:
                result["status"] = "render-error"
                result["detail"] = f"time-path err={type(e).__name__}: {str(e)[:80]}"
                return result
            result["best_mode"] = "<time>"
            result["best_changed"] = round(changed, 4)
            result["best_tvar"] = round(tvar, 6)
            result["status"] = _verdict(changed, tvar, "<time>")
            return result
        result["status"] = "no-anim-mode"
        return result
    best_changed = 0.0
    best_tvar = 0.0
    best_mode = None
    for mode in modes:
        try:
            params = {**base, anim_key: mode}
            if cheap:
                frames = [0, QUARTER, min(2 * QUARTER, N_FRAMES - 1)]
            else:
                frames = list(range(STACK_N))
            stack = [_render(mid, params, frame=f, seed=seed) for f in frames]
            changed, tvar = _probe(stack)
        except Exception as e:  # render/param incompatibility -> note, skip mode
            result["detail"] = f"mode={mode} err={type(e).__name__}: {str(e)[:80]}"
            continue
        if changed > best_changed:
            best_changed = changed
            best_tvar = tvar
            best_mode = mode
    result["best_mode"] = best_mode
    result["best_changed"] = round(best_changed, 4)
    result["best_tvar"] = round(best_tvar, 6)
    if best_mode is None:
        result["status"] = "render-error"
    else:
        result["status"] = _verdict(best_changed, best_tvar, best_mode)
    return result


def _progress_path() -> Path:
    """Where the cross-run progress manifest is kept (enables --resume)."""
    return REPORT_PATH.parent / "dead-param-progress.json"


def _load_progress() -> dict:
    p = _progress_path()
    if p.exists():
        try:
            return json.loads(p.read_text())
        except (json.JSONDecodeError, OSError):
            return {}
    return {}


def _save_progress(data: dict) -> None:
    _progress_path().write_text(json.dumps(data, indent=1, sort_keys=True))


def _split_shards(ids: list[str], shard: int, of: int) -> list[str]:
    """Deterministic shard fan-out: 1-based ``shard`` of ``of`` total."""
    return [mid for i, mid in enumerate(ids) if i % of == (shard - 1)]


def _filter_resume(ids: list[str], done: set[str]) -> list[str]:
    """Drop ids already completed in a prior run (enables --resume)."""
    return [mid for mid in ids if mid not in done]


def _merge_shards(merge_dir: Path) -> list[dict]:
    """Collect per-shard result JSON files into one ordered row list.

    Shards are written by a `--shard N/M --merge-dir D` run as ``D/M_<N>.json``.
    Order is stable (sorted by shard index) so the merged report is deterministic.
    """
    rows: list[dict] = []
    for f in sorted(merge_dir.glob("M_*.json"), key=lambda p: int(p.stem.split("_")[1])):
        try:
            rows.extend(json.loads(f.read_text()))
        except (json.JSONDecodeError, OSError) as e:
            print(f"  [merge] skip {f.name}: {e}")
    return rows


def main() -> int:
    ap = argparse.ArgumentParser(description="Headless per-node liveness audit.")
    ap.add_argument("--ids", help="comma-separated method ids to audit (else all time-varying)")
    ap.add_argument("--limit", type=int, help="cap number of nodes audited")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--cheap", action="store_true",
                    help="cheap 3-frame probe (fast; good enough for the dead/alive frontier)")
    ap.add_argument("--resume", action="store_true",
                    help="skip ids already recorded as done in the progress manifest")
    ap.add_argument("--shard", type=int, metavar="N",
                    help="1-based shard index for fan-out (use with --of / --merge-dir)")
    ap.add_argument("--of", type=int, metavar="M", default=1,
                    help="total shard count (default 1 = full run)")
    ap.add_argument("--merge-dir", type=str, default=None,
                    help="dir to write per-shard result JSON (shard mode) or to merge (*.json) when --merge is set")
    ap.add_argument("--merge", action="store_true",
                    help="merge per-shard JSON files from --merge-dir into the combined report")
    args = ap.parse_args()

    # ── Merge mode: assemble the combined report from shard outputs ──
    if args.merge:
        mdir = Path(args.merge_dir or (REPORT_PATH.parent / "shards"))
        rows = _merge_shards(mdir)
        print(f"[merge] {len(rows)} rows from {mdir}")
        return _emit_report(rows)

    defs = get_all_node_defs()
    if args.ids:
        ids = [s.strip() for s in args.ids.split(",") if s.strip()]
    else:
        ids = _time_varying_ids()

    # ── Shard fan-out: deterministically split the id list ──
    if args.of > 1:
        ids = _split_shards(ids, args.shard, args.of)
        print(f"[shard] index={args.shard}/{args.of}  ->  {len(ids)} nodes for this shard")

    # ── Resume: drop ids already completed in a prior run ──
    if args.resume:
        prog = _load_progress()
        done = set(prog.get("done", []))
        skipped = len(_filter_resume(ids, done))
        ids = _filter_resume(ids, done)
        if skipped:
            print(f"[resume] skipping {skipped} already-audited id(s)")

    if args.limit:
        ids = ids[: args.limit]

    print(f"Auditing {len(ids)} time-varying node(s) "
          f"({'cheap' if args.cheap else 'full'})...")
    rows = []
    t0 = time.time()
    for i, mid in enumerate(ids, 1):
        defn = defs.get(mid, {})
        try:
            r = audit_node(mid, defn, seed=args.seed, cheap=args.cheap)
        except Exception as e:
            r = {"id": mid, "name": defs.get(mid, {}).get("name", mid),
                   "status": "exception", "detail": f"{type(e).__name__}: {str(e)[:80]}",
                   "modes": [], "best_mode": None, "best_changed": 0.0, "best_tvar": 0.0}
        rows.append(r)
        flag = "  <-- SUSPECT" if "DEAD-PARAM" in r["status"] else ""
        print(f"  [{i:3d}/{len(ids)}] {mid:8s} {r['status']:22s} "
              f"changed={r['best_changed']:.3f} tvar={r['best_tvar']:.2e} "
              f"mode={r['best_mode']}{flag}")

    # Update the cross-run progress manifest (for --resume).
    if not args.merge:
        prog = _load_progress()
        done = set(prog.get("done", []))
        done.update(r["id"] for r in rows)
        prog["done"] = sorted(done)
        prog["last_run"] = time.strftime("%Y-%m-%dT%H:%M:%S")
        _save_progress(prog)

    # In shard mode we persist just this shard's rows and bail before merging.
    if args.of > 1 and args.merge_dir:
        mdir = Path(args.merge_dir)
        mdir.mkdir(parents=True, exist_ok=True)
        (mdir / f"M_{args.shard}.json").write_text(json.dumps(rows))
        print(f"[shard] wrote {len(rows)} rows -> {mdir / f'M_{args.shard}.json'}")
        return 0

    dt = time.time() - t0
    print(f"\nElapsed: {dt:.0f}s")
    return _emit_report(rows)


def _emit_report(rows: list[dict]) -> int:
    """Write the markdown audit report from a list of per-node rows."""
    suspects = [r for r in rows if "DEAD-PARAM" in r["status"]]
    weak = [r for r in rows if r["status"] == "weak (changed<=floor)"]
    alive = [r for r in rows if r["status"] == "alive"]
    errors = [r for r in rows if r["status"] in ("render-error", "exception", "no-anim-mode")]

    report = [
        "# Dead-Param Liveness Audit",
        "",
        f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}  "
        f"nodes audited: {len(rows)}",
        "",
        "## Summary",
        "",
        f"- **alive** (anim_mode reaches pixels): {len(alive)}",
        f"- **DEAD-PARAM suspects** (should move, does not): {len(suspects)}",
        f"- **weak** (changed<=floor but tvar>0): {len(weak)}",
        f"- **unauditable** (render-error / exception / no anim_mode): {len(errors)}",
        "",
        "A DEAD-PARAM suspect is a node whose non-`none` anim_mode produces a",
        "static frame stack (changed_frac <= 0.10 AND temporal_var <= 1e-3).",
        "That is the root cause of the remaining `static`/`flat` shootout deaths:",
        "the driver / anim_mode is present but the driven node's render math",
        "cancels it (loop-var shadowing, per-frame normalization, or an",
        "unapplied param). Fix per the 8-step animation audit (Step 4/5,",
        "pitfall #4 / #19).",
        "",
        "## Suspects (actionable)",
        "",
    ]
    if suspects:
        for r in sorted(suspects, key=lambda x: x["best_changed"]):
            report.append(
                f"- **{r['id']}** `{r['name']}` -- best mode `{r['best_mode']}` "
                f"changed={r['best_changed']:.3f} tvar={r['best_tvar']:.2e}")
    else:
        report.append("(none)")
    report += ["", "## Weak", ""]
    report += [f"- {r['id']} `{r['name']}` changed={r['best_changed']:.3f} tvar={r['best_tvar']:.2e} mode={r['best_mode']}"
                for r in weak] or ["(none)"]
    report += ["", "## Unauditable", ""]
    report += [f"- {r['id']} `{r['name']}` -> {r['status']} {r.get('detail','')}"
                for r in errors] or ["(none)"]
    report += ["", "## All rows", ""]
    report += [f"- {r['id']:8s} {r['status']:22s} changed={r['best_changed']:.3f} "
                f"tvar={r['best_tvar']:.2e} mode={r['best_mode']}"
                for r in rows]

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text("\n".join(report) + "\n")
    print(f"\nReport -> {REPORT_PATH}")
    print(f"Suspects: {len(suspects)}  weak: {len(weak)}  alive: {len(alive)}  "
          f"unauditable: {len(errors)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
