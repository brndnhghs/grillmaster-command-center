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


def _non_none_modes(defn: dict) -> list[str]:
    """Return the non-'none' anim_mode / animation_mode choices for a node def."""
    out: list[str] = []
    for key in ("anim_mode", "animation_mode"):
        spec = (defn.get("params") or {}).get(key)
        if not spec:
            continue
        choices = spec.get("choices") or []
        for c in choices:
            cs = str(c).lower()
            if cs not in ("none", "", "off"):
                out.append(str(c))
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
        if _non_none_modes(defn):
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


def audit_node(mid: str, defn: dict, seed: int = 42) -> dict:
    modes = _non_none_modes(defn)
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
    if not modes:
        # ── Architecture-B fallback: no anim_mode enum, but the node may
        # animate purely via the injected ``time`` clock. Render across the
        # frame stack (the executor varies ``time`` = timeline.phase per frame)
        # and measure whether the time path reaches pixels. This turns the
        # former "no-anim-mode" blind spot into a real verdict.
        if _has_time_param(defn):
            result["modes"] = ["<time>"]
            try:
                stack = [_render(mid, dict(base), frame=f, seed=seed)
                         for f in range(STACK_N)]
                changed = _changed_frac(stack[0], stack[QUARTER])
                tvar = _temporal_var(stack)
            except Exception as e:
                result["status"] = "render-error"
                result["detail"] = f"time-path err={type(e).__name__}: {str(e)[:80]}"
                return result
            result["best_mode"] = "<time>"
            result["best_changed"] = round(changed, 4)
            result["best_tvar"] = round(tvar, 6)
            if changed <= CHANGED_FLOOR and tvar <= TEMPORAL_VAR_FLOOR:
                result["status"] = "DEAD-PARAM (suspect)"
            elif changed <= CHANGED_FLOOR:
                result["status"] = "weak (changed<=floor)"
            else:
                result["status"] = "alive"
            return result
        result["status"] = "no-anim-mode"
        return result
    best_changed = 0.0
    best_tvar = 0.0
    best_mode = None
    for mode in modes:
        try:
            params = {**base, "anim_mode": mode}
            # build a small stack for temporal_var
            stack = []
            for f in range(STACK_N):
                stack.append(_render(mid, params, frame=f, seed=seed))
            changed = _changed_frac(stack[0], stack[QUARTER])
            tvar = _temporal_var(stack)
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
    elif best_changed <= CHANGED_FLOOR and best_tvar <= TEMPORAL_VAR_FLOOR:
        result["status"] = "DEAD-PARAM (suspect)"
    elif best_changed <= CHANGED_FLOOR:
        result["status"] = "weak (changed<=floor)"
    else:
        result["status"] = "alive"
    return result


def main() -> int:
    ap = argparse.ArgumentParser(description="Headless per-node liveness audit.")
    ap.add_argument("--ids", help="comma-separated method ids to audit (else all time-varying)")
    ap.add_argument("--limit", type=int, help="cap number of nodes audited")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    defs = get_all_node_defs()
    if args.ids:
        ids = [s.strip() for s in args.ids.split(",") if s.strip()]
    else:
        ids = _time_varying_ids()
    if args.limit:
        ids = ids[: args.limit]

    print(f"Auditing {len(ids)} time-varying node(s)...")
    rows: list[dict] = []
    t0 = time.time()
    for i, mid in enumerate(ids, 1):
        defn = defs.get(mid, {})
        try:
            r = audit_node(mid, defn, seed=args.seed)
        except Exception as e:
            r = {"id": mid, "name": defs.get(mid, {}).get("name", mid),
                   "status": "exception", "detail": f"{type(e).__name__}: {str(e)[:80]}",
                   "modes": [], "best_mode": None, "best_changed": 0.0, "best_tvar": 0.0}
        rows.append(r)
        flag = "  <-- SUSPECT" if "DEAD-PARAM" in r["status"] else ""
        print(f"  [{i:3d}/{len(ids)}] {mid:8s} {r['status']:22s} "
              f"changed={r['best_changed']:.3f} tvar={r['best_tvar']:.2e} "
              f"mode={r['best_mode']}{flag}")
    dt = time.time() - t0

    suspects = [r for r in rows if "DEAD-PARAM" in r["status"]]
    weak = [r for r in rows if r["status"] == "weak (changed<=floor)"]
    alive = [r for r in rows if r["status"] == "alive"]
    errors = [r for r in rows if r["status"] in ("render-error", "exception", "no-anim-mode")]

    report = [
        "# Dead-Param Liveness Audit",
        "",
        f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}  "
        f"nodes audited: {len(rows)}  elapsed: {dt:.0f}s",
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
