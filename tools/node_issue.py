#!/usr/bin/env python3
"""node_issue.py — capture, replay and promote esoteric node issues.

The problem this solves: node failures found while *using* the editor are the
hardest to act on. "Node 437 goes black sometimes" is unactionable — the params,
the wiring, the seed and the frame all mattered and none were written down. By
the time anyone looks, the graph has moved on.

The pipeline's determinism contract makes this fixable. DESIGN.md: "Identical
graph + seed + params => identical output, always." So a report does not need
to *describe* a failure — it can *be* the failure, captured as the exact inputs
that produce it. A good report here is a replayable case, not prose.

Lifecycle:

    capture   freeze a failing node + its wiring + seed/frame into
              data/node-issues/<id>.json, recording what it produced at the
              time (types, ranges, error).
    replay    re-run a captured issue against the CURRENT code and diff the
              observed behaviour against what was recorded. This is what tells
              you a fix worked, or that something regressed back.
    list      show captured issues and whether each still reproduces.
    promote   emit a pytest regression test from a captured issue, so a fixed
              bug cannot come back silently.

Because a capture is self-contained, it survives refactors, transfers between
machines, and can be attached to a bug report verbatim.

Usage:
    python tools/node_issue.py capture --method 437 --note "goes black at high blur"
    python tools/node_issue.py capture --graph mygraph.json --node n3 --note "..."
    python tools/node_issue.py replay  --id ni-0001
    python tools/node_issue.py list
    python tools/node_issue.py promote --id ni-0001 > image_pipeline/tests/test_issue_0001.py
"""
from __future__ import annotations

import argparse
import json
import pprint
import sys
import tempfile
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from collections import Counter

REPO = Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

import image_pipeline.methods  # noqa: E402,F401
from image_pipeline.core import registry  # noqa: E402
from image_pipeline.core.graph import GraphExecutor  # noqa: E402
from image_pipeline.core.utils import set_canvas  # noqa: E402

ISSUE_DIR = REPO / "data" / "node-issues"


# ── Observation: what a node actually produced ────────────────────────


def _describe(v) -> dict:
    """A comparable, JSON-safe fingerprint of one payload value."""
    if v is None:
        return {"type": "none"}
    if isinstance(v, np.ndarray):
        finite = bool(np.isfinite(v).all()) if v.dtype.kind == "f" else True
        d = {"type": "ndarray", "shape": list(v.shape), "dtype": str(v.dtype),
             "finite": finite}
        if v.size and v.dtype.kind == "f" and finite:
            d["min"] = round(float(np.nanmin(v)), 6)
            d["max"] = round(float(np.nanmax(v)), 6)
            d["mean"] = round(float(np.nanmean(v)), 6)
            # Constant output is the signature of most "dead node" reports.
            d["constant"] = bool(np.nanmin(v) == np.nanmax(v))
        return d
    if isinstance(v, (int, float, np.integer, np.floating)):
        return {"type": "scalar", "value": round(float(v), 6)}
    return {"type": type(v).__name__}


def _observe(nodes, edges, node_id, seed, frame, frames, canvas) -> dict:
    set_canvas(*canvas)
    ex = GraphExecutor(Path(tempfile.mkdtemp(prefix="ni_")),
                       in_memory=True, audit_to_disk=False)
    t0 = time.perf_counter()
    try:
        flat, _term, errs = ex.execute(nodes=nodes, edges=edges, seed=seed,
                                       frame=frame, frames=frames)
    except Exception:
        return {"raised": traceback.format_exc(limit=6), "elapsed_s": None,
                "payload": {}, "error": None}
    payload = flat.get(node_id) or {}
    return {
        "raised": None,
        "error": str(errs.get(node_id)) if errs.get(node_id) else None,
        "elapsed_s": round(time.perf_counter() - t0, 3),
        "payload": {k: _describe(v) for k, v in sorted(payload.items())},
    }


def _defaults(mid: str) -> dict:
    meta = registry.get_meta(mid)
    return {k: (v.get("default") if isinstance(v, dict) else v)
            for k, v in (meta.params or {}).items()}


# ── Commands ──────────────────────────────────────────────────────────


def cmd_capture(args) -> int:
    if args.graph:
        doc = json.loads(Path(args.graph).read_text(encoding="utf-8"))
        nodes, edges = doc.get("nodes", []), doc.get("edges", [])
        node_id = args.node or (nodes[0]["id"] if nodes else None)
        if node_id is None:
            print("graph has no nodes", file=sys.stderr)
            return 2
        mid = next((n["method_id"] for n in nodes if n["id"] == node_id), None)
    else:
        mid = args.method
        if registry.get_meta(mid) is None:
            print(f"unknown method id {mid!r}", file=sys.stderr)
            return 2
        node_id = "n"
        params = _defaults(mid)
        if args.param:
            for kv in args.param:
                k, _, raw = kv.partition("=")
                try:
                    params[k] = json.loads(raw)
                except json.JSONDecodeError:
                    params[k] = raw
        nodes = [{"id": node_id, "method_id": mid, "params": params, "render": True}]
        edges = []

    canvas = (args.width, args.height)
    observed = _observe(nodes, edges, node_id, args.seed, args.frame,
                        args.frames, canvas)

    ISSUE_DIR.mkdir(parents=True, exist_ok=True)
    n = len(list(ISSUE_DIR.glob("ni-*.json"))) + 1
    issue_id = f"ni-{n:04d}"
    meta = registry.get_meta(mid)
    record = {
        "id": issue_id,
        "captured_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "note": args.note or "",
        "method_id": mid,
        "method_name": getattr(meta, "name", "?"),
        "category": getattr(meta, "category", "?"),
        # Everything needed to reproduce, and nothing else.
        "repro": {"nodes": nodes, "edges": edges, "node_id": node_id,
                  "seed": args.seed, "frame": args.frame,
                  "frames": args.frames, "canvas": list(canvas)},
        "observed_at_capture": observed,
    }
    path = ISSUE_DIR / f"{issue_id}.json"
    path.write_text(json.dumps(record, indent=1), encoding="utf-8")

    print(f"captured {issue_id} -> {path.relative_to(REPO)}")
    print(f"  method {mid} ({record['method_name']})")
    if observed["raised"]:
        print("  RAISED at capture")
    elif observed["error"]:
        print(f"  node error: {observed['error'][:120]}")
    else:
        img = observed["payload"].get("image", {})
        flags = []
        if img.get("constant"):
            flags.append("image is CONSTANT")
        if img.get("finite") is False:
            flags.append("image has NaN/Inf")
        print(f"  image: {img.get('shape')} {img.get('dtype')} "
              f"{'| ' + ', '.join(flags) if flags else 'looks structurally ok'}")
    print(f"\nreplay with:  python tools/node_issue.py replay --id {issue_id}")
    return 0


def _load(issue_id: str) -> dict:
    path = ISSUE_DIR / f"{issue_id}.json"
    if not path.exists():
        raise SystemExit(f"no such issue: {issue_id}")
    return json.loads(path.read_text(encoding="utf-8"))


def _diff(before: dict, after: dict) -> list[str]:
    out = []
    if bool(before.get("raised")) != bool(after.get("raised")):
        out.append(f"raised: {bool(before.get('raised'))} -> {bool(after.get('raised'))}")
    if (before.get("error") or None) != (after.get("error") or None):
        out.append(f"error: {str(before.get('error'))[:60]!r} -> "
                   f"{str(after.get('error'))[:60]!r}")
    keys = set(before.get("payload", {})) | set(after.get("payload", {}))
    for k in sorted(keys):
        b = before.get("payload", {}).get(k)
        a = after.get("payload", {}).get(k)
        if b != a:
            out.append(f"{k}: {b} -> {a}")
    return out


def cmd_replay(args) -> int:
    rec = _load(args.id)
    r = rec["repro"]
    now = _observe(r["nodes"], r["edges"], r["node_id"], r["seed"],
                   r["frame"], r["frames"], tuple(r["canvas"]))
    print(f"=== {rec['id']}  {rec['method_id']} ({rec['method_name']}) ===")
    if rec.get("note"):
        print(f"note: {rec['note']}")
    changes = _diff(rec["observed_at_capture"], now)
    if not changes:
        print("STILL REPRODUCES — behaviour identical to capture.")
        return 1
    print("BEHAVIOUR CHANGED since capture:")
    for c in changes:
        print(f"  {c}")
    return 0


def cmd_list(args) -> int:
    issues = sorted(ISSUE_DIR.glob("ni-*.json")) if ISSUE_DIR.exists() else []
    if not issues:
        print("no captured node issues")
        return 0
    print(f"{'id':<9} {'method':<8} {'status':<18} note")
    for p in issues:
        rec = json.loads(p.read_text(encoding="utf-8"))
        r = rec["repro"]
        try:
            now = _observe(r["nodes"], r["edges"], r["node_id"], r["seed"],
                           r["frame"], r["frames"], tuple(r["canvas"]))
            status = "reproduces" if not _diff(rec["observed_at_capture"], now) else "CHANGED"
        except Exception:
            status = "replay failed"
        print(f"{rec['id']:<9} {rec['method_id']:<8} {status:<18} {rec.get('note','')[:44]}")
    return 0


def cmd_diagnose(args) -> int:
    """Turn "this node feels wrong" into a specific, actionable list.

    Three signals, because "dumbly constructed" has three distinct shapes and
    the fix differs for each:

      contract   the node emits something other than what it declares — a
                 downstream consumer is already receiving the wrong thing.
      dead       a param is exposed but does not reach the pixels. The slider
                 is a lie. (tools/audit_dead_params.py measures this properly
                 by rendering; here we only flag params never read in the body,
                 which is the cheap static half of the same question.)
      buried     the interesting values are hardcoded in the body instead of
                 exposed as params. This is the "not exposing the real
                 interesting parameters" complaint, and it is the most common.

    The buried-constant count is a TRIAGE RANKING, not a defect count: loop
    bounds, colour constants and math literals inflate it. Use it to decide
    which node to open, then read the listed literals and judge.
    """
    import ast
    import inspect

    mid = args.method
    meta = registry.get_meta(mid)
    if meta is None:
        print(f"unknown method id {mid!r}", file=sys.stderr)
        return 2

    print(f"=== {mid}  {meta.name}  [{meta.category}] ===\n")

    # ── contract ──
    try:
        sys.path.insert(0, str(REPO / "tools"))
        from audit_node_contract import audit_method  # type: ignore
        findings = audit_method(mid, meta)
    except Exception as exc:
        findings = []
        print(f"contract check unavailable: {exc}")
    errs = [f for f in findings if f["severity"] == "ERROR"]
    print(f"CONTRACT   {len(errs)} error(s), {len(findings)-len(errs)} other")
    for f in findings[:6]:
        print(f"   [{f['severity']:5}] {f['kind']:24} {f['detail'][:70]}")

    # ── params: declared vs actually read ──
    try:
        src = inspect.getsource(meta.fn)
        tree = ast.parse(src.lstrip())
    except Exception:
        print("\n(could not parse source — skipping param/constant analysis)")
        return 0

    read_names = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)}
    read_strs = {n.value for n in ast.walk(tree)
                 if isinstance(n, ast.Constant) and isinstance(n.value, str)}
    declared = list((meta.params or {}).keys())
    never_read = [p for p in declared
                  if p not in read_names and p not in read_strs]
    print(f"\nPARAMS     {len(declared)} declared, {len(never_read)} never "
          f"referenced in the body")
    for p in never_read:
        print(f"   dead?  {p}")
    if never_read:
        print("   -> confirm by rendering:  python tools/audit_dead_params.py "
              f"--ids {mid}")

    # ── buried constants ──
    lits: list[float] = [
        n.value for n in ast.walk(tree)
        if isinstance(n, ast.Constant) and isinstance(n.value, (int, float))
        and not isinstance(n.value, bool)
        and n.value not in (0, 1, -1, 2, 3, 255, 0.0, 1.0, 0.5)
    ]
    ratio = len(lits) / max(len(declared), 1)
    common = Counter(lits).most_common(8)
    print(f"\nBURIED     {len(lits)} hardcoded constants vs {len(declared)} "
          f"params ({ratio:.1f}x)")
    if common:
        print(f"   most repeated: {[c[0] for c in common]}")
    if ratio >= 5:
        print("   -> strong candidate: the interesting values are in the body, "
              "not on the node")

    # ── Node Doctor brief ──
    if args.brief:
        print("\n" + "=" * 68)
        print("NODE DOCTOR BRIEF (paste into the Node Doctor panel)")
        print("=" * 68)
        print(f"Node {mid} ({meta.name}) needs its parameter surface reworked.")
        print(f"It currently exposes {len(declared)} params: {declared}")
        if errs:
            print("\nContract errors to fix first:")
            for f in errs:
                print(f"  - {f['kind']}: {f['detail']}")
        if never_read:
            print(f"\nThese params are never referenced in the body and may be "
                  f"dead — either wire them into the render math or remove "
                  f"them: {never_read}")
        if ratio >= 5:
            print(f"\nThe body hardcodes {len(lits)} numeric constants, "
                  f"{ratio:.0f}x the number of exposed params. The most "
                  f"repeated are {[c[0] for c in common[:6]]}. Promote the ones "
                  f"that change the LOOK of the output into params with "
                  f"sensible min/max, keep incidental constants inline.")
        print("\nConstraints: keep the method id and signature; declare every "
              "new param in the @method decorator with min/max; anything the "
              "user should be able to animate must be a float/int param. Do "
              "not change what the node fundamentally does.")
    return 0


def cmd_promote(args) -> int:
    rec = _load(args.id)
    r = rec["repro"]
    print(f'''"""Regression: {rec['method_id']} ({rec['method_name']}) — {rec.get('note','captured node issue')}

Promoted from captured node issue {rec['id']} ({rec['captured_at']}).
The pipeline is deterministic, so this is the exact input that failed.
"""
import tempfile
from pathlib import Path

import numpy as np

import image_pipeline.methods  # noqa: F401
from image_pipeline.core.graph import GraphExecutor
from image_pipeline.core.utils import set_canvas

NODES = {pprint.pformat(r["nodes"], width=78)}
EDGES = {pprint.pformat(r["edges"], width=78)}


def test_{rec['id'].replace('-', '_')}_node_{rec['method_id']}_is_healthy():
    set_canvas({r['canvas'][0]}, {r['canvas'][1]})
    ex = GraphExecutor(Path(tempfile.mkdtemp()), in_memory=True, audit_to_disk=False)
    flat, _t, errs = ex.execute(nodes=NODES, edges=EDGES, seed={r['seed']},
                                frame={r['frame']}, frames={r['frames']})
    assert not errs, f"node raised: {{errs}}"
    img = flat.get({r['node_id']!r}, {{}}).get("image")
    assert img is not None, "node produced no image"
    arr = np.asarray(img, dtype=np.float32)
    assert np.isfinite(arr).all(), "output contains NaN/Inf"
    assert arr.min() != arr.max(), "output is a constant image"
''')
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)

    c = sub.add_parser("capture", help="freeze a failing node into a replayable record")
    c.add_argument("--method", help="method id (standalone capture)")
    c.add_argument("--graph", help="graph json to capture from")
    c.add_argument("--node", help="node id within --graph")
    c.add_argument("--param", action="append", help="param override, k=json")
    c.add_argument("--note", help="what looked wrong")
    c.add_argument("--seed", type=int, default=7)
    c.add_argument("--frame", type=int, default=1)
    c.add_argument("--frames", type=int, default=8)
    c.add_argument("--width", type=int, default=256)
    c.add_argument("--height", type=int, default=192)
    c.set_defaults(func=cmd_capture)

    r = sub.add_parser("replay", help="re-run a captured issue against current code")
    r.add_argument("--id", required=True)
    r.set_defaults(func=cmd_replay)

    ls = sub.add_parser("list", help="list captured issues and whether they still reproduce")
    ls.set_defaults(func=cmd_list)

    dg = sub.add_parser("diagnose", help="why does this node feel wrong?")
    dg.add_argument("--method", required=True)
    dg.add_argument("--brief", action="store_true",
                    help="also emit a Node Doctor brief to paste into the panel")
    dg.set_defaults(func=cmd_diagnose)

    p = sub.add_parser("promote", help="emit a pytest regression test")
    p.add_argument("--id", required=True)
    p.set_defaults(func=cmd_promote)

    args = ap.parse_args()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
