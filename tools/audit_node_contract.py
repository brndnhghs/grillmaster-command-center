#!/usr/bin/env python3
"""audit_node_contract.py — does each node actually PRODUCE what it DECLARES?

The existing audits cover neighbouring questions but not this one:

  * ``tools/audit_methods.py``       static AST scan: declared ``outputs=`` vs
                                     signals it can spot in the source.
  * ``tools/validate_image_wiring``  graph-level: is an image input wired to a
                                     real image-emitting upstream port.
  * ``core/node_tester.py``          runs each method in isolation and reports
                                     pass/fail — explicitly "no graph wiring".

None of them execute a node through the real executor and compare the payload
it emits against the port types it declares. That gap matters because the
declared type IS the contract: the UI colours wires from it, ``_inject_typed``
routes values by it, and a downstream node consuming ``FIELD`` will silently
receive garbage if the producer actually emitted something else.

This tool closes it. For every registered method it runs one frame through
``GraphExecutor`` at a small canvas and validates the resulting payload against
the ``core/port_types.py`` registry:

    IMAGE      float32 ndarray (H,W,3), values in [0,1]
    FIELD      float32 ndarray (H,W), arbitrary range
    MASK       float32 ndarray (H,W), values in [0,1]
    PARTICLES  float32 ndarray (N,4) — [x,y,vx,vy]
    COLORMAP   float32 ndarray (N,3) or (N,4)
    SCALAR     a real number

Findings are grouped by kind so a wall of violations from one root cause reads
as one problem. Exit code is 1 if any ERROR-severity finding is present, so it
can gate CI.

Usage:
    python tools/audit_node_contract.py                 # all methods
    python tools/audit_node_contract.py --only 05 17    # specific ids
    python tools/audit_node_contract.py --category filters
    python tools/audit_node_contract.py --json out.json
"""
from __future__ import annotations

import argparse
import json
import sys
import tempfile
import time
import traceback
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

import image_pipeline.methods  # noqa: E402,F401  (registers @method nodes)
from image_pipeline.core import registry  # noqa: E402
from image_pipeline.core.graph import GraphExecutor, _make_node_def  # noqa: E402
from image_pipeline.core.utils import set_canvas  # noqa: E402

# Keys the executor injects into every payload regardless of declaration, so
# their presence is never an "undeclared output" on the method's part.
_EXECUTOR_KEYS = {"image", "luminance", "field", "particles", "mask"}

AUDIT_W, AUDIT_H = 96, 64


# ── Type validators ───────────────────────────────────────────────────


def _is_float_array(v) -> bool:
    return isinstance(v, np.ndarray) and v.dtype.kind == "f"


def _check_image(v):
    if not isinstance(v, np.ndarray):
        return f"expected ndarray, got {type(v).__name__}"
    if v.ndim != 3 or v.shape[2] != 3:
        return f"expected (H,W,3), got {v.shape}"
    if v.dtype.kind != "f":
        return f"expected float dtype, got {v.dtype}"
    if v.size and (np.nanmin(v) < -0.001 or np.nanmax(v) > 1.001):
        return f"values outside [0,1]: [{np.nanmin(v):.3f}, {np.nanmax(v):.3f}]"
    if v.size and not np.isfinite(v).all():
        return "contains NaN/Inf"
    return None


def _check_field(v):
    if not _is_float_array(v):
        return f"expected float ndarray, got {type(v).__name__}"
    if v.ndim != 2:
        return f"expected (H,W), got {v.shape}"
    return None


def _check_mask(v):
    if not _is_float_array(v):
        return f"expected float ndarray, got {type(v).__name__}"
    if v.ndim != 2:
        return f"expected (H,W), got {v.shape}"
    if v.size and (np.nanmin(v) < -0.001 or np.nanmax(v) > 1.001):
        return f"values outside [0,1]: [{np.nanmin(v):.3f}, {np.nanmax(v):.3f}]"
    return None


def _check_particles(v):
    if not _is_float_array(v):
        return f"expected float ndarray, got {type(v).__name__}"
    if v.ndim != 2 or v.shape[1] != 4:
        return f"expected (N,4) [x,y,vx,vy], got {v.shape}"
    return None


def _check_colormap(v):
    if not _is_float_array(v):
        return f"expected float ndarray, got {type(v).__name__}"
    if v.ndim != 2 or v.shape[1] not in (3, 4):
        return f"expected (N,3) or (N,4), got {v.shape}"
    return None


def _check_scalar(v):
    # luminance is contractually allowed to be a per-pixel array (DESIGN.md);
    # any other SCALAR must be a real number.
    if isinstance(v, np.ndarray):
        return None if v.ndim in (0, 2) else f"array scalar with shape {v.shape}"
    if isinstance(v, (int, float, np.integer, np.floating)) and not isinstance(v, bool):
        return None
    return f"expected a number, got {type(v).__name__}"


_CHECKS = {
    "IMAGE": _check_image,
    "FIELD": _check_field,
    "MASK": _check_mask,
    "PARTICLES": _check_particles,
    "COLORMAP": _check_colormap,
    "SCALAR": _check_scalar,
    "ANY": lambda v: None,
}


# ── Audit ─────────────────────────────────────────────────────────────


def _defaults(meta) -> dict:
    out = {}
    for k, spec in (meta.params or {}).items():
        out[k] = spec.get("default") if isinstance(spec, dict) else spec
    # Keep Architecture-A sims short — we are checking types, not dynamics.
    if "n_frames" in out:
        spec = (meta.params or {}).get("n_frames") or {}
        out["n_frames"] = spec.get("min", 2) if isinstance(spec, dict) else 2
    return out


def audit_method(mid: str, meta) -> list[dict]:
    findings: list[dict] = []

    def add(kind, severity, detail, port=None):
        findings.append({"method_id": mid, "name": meta.name,
                         "category": meta.category, "kind": kind,
                         "severity": severity, "port": port, "detail": detail})

    nd = _make_node_def(meta)
    declared = dict(nd.outputs)

    set_canvas(AUDIT_W, AUDIT_H)
    ex = GraphExecutor(Path(tempfile.mkdtemp(prefix="contract_")),
                       in_memory=True, audit_to_disk=False)
    node = {"id": "n", "method_id": mid, "params": _defaults(meta), "render": True}
    t0 = time.perf_counter()
    try:
        flat, _term, errs = ex.execute(nodes=[node], edges=[], seed=7,
                                       frame=1, frames=4)
    except Exception as exc:
        add("raised", "ERROR", f"{type(exc).__name__}: {exc}")
        return findings
    elapsed = time.perf_counter() - t0

    if errs:
        add("node_error", "ERROR", str(errs.get("n", errs))[:300])
        return findings

    payload = flat.get("n") or {}

    # 1. Declared but absent (or None) in the payload.
    for port, ptype in declared.items():
        if port not in payload or payload[port] is None:
            sev = "ERROR" if port == "image" else "WARN"
            add("missing_output", sev,
                f"declares {port}:{ptype} but the payload has no value", port)

    # 2. Present but the value does not satisfy the declared type.
    for port, ptype in declared.items():
        v = payload.get(port)
        if v is None:
            continue
        check = _CHECKS.get(str(ptype).upper())
        if check is None:
            add("unknown_port_type", "WARN",
                f"declared type {ptype!r} is not in the port-type registry", port)
            continue
        problem = check(v)
        if not problem:
            continue
        # Distinguish "this node emitted the wrong thing" from "the executor
        # substituted something". graph.py builds the payload with
        #   "field": extra_outputs.get("field", arr)
        # so a method that declares a FIELD output but writes no field sidecar
        # silently receives the 3-channel IMAGE as its field. That is one
        # executor line, not one bug per node, and lumping them together buries
        # the genuine per-node mismatches under it.
        img = payload.get("image")
        if (port == "field" and isinstance(img, np.ndarray)
                and isinstance(v, np.ndarray) and v is img):
            add("field_defaulted_to_image", "ERROR",
                f"declares field:{ptype} but writes no field sidecar, so the "
                f"executor passes the (H,W,3) image through as the field",
                port)
        else:
            add("type_mismatch", "ERROR", f"{port}:{ptype} — {problem}", port)

    # 3. Emitted a non-trivial value nobody declared. Downstream cannot wire it
    #    (no port exists), so the data is effectively invisible.
    for key, v in payload.items():
        if key in _EXECUTOR_KEYS or key in declared or v is None:
            continue
        if isinstance(v, np.ndarray) or isinstance(v, (int, float)):
            add("undeclared_output", "WARN",
                f"emits {key!r} ({type(v).__name__}) but never declares it in outputs=",
                key)

    # 4. Driveability: the design intent is that nodes are driven by their
    #    parameter inputs. A node with no wireable param port cannot be
    #    animated by a CHOP driver at all.
    if not nd.param_ports:
        add("not_driveable", "INFO",
            "exposes no wireable param port — cannot be driven by an "
            "LFO/Counter/Ramp, so it can only ever be static", None)

    if elapsed > 5.0:
        add("slow", "INFO", f"one frame took {elapsed:.1f}s at {AUDIT_W}x{AUDIT_H}")

    return findings


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*", help="method ids to audit")
    ap.add_argument("--category", help="restrict to one category")
    ap.add_argument("--json", help="write findings to this path")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    methods = registry.get_all()
    items = sorted(methods.items(), key=lambda kv: str(kv[0]))
    if args.only:
        want = set(args.only)
        items = [(m, v) for m, v in items if m in want]
    if args.category:
        items = [(m, v) for m, v in items if v.category == args.category]

    all_findings: list[dict] = []
    for i, (mid, meta) in enumerate(items, 1):
        if not args.quiet and i % 25 == 0:
            print(f"  ... {i}/{len(items)}", file=sys.stderr)
        try:
            all_findings.extend(audit_method(mid, meta))
        except Exception:
            all_findings.append({
                "method_id": mid, "name": getattr(meta, "name", "?"),
                "category": getattr(meta, "category", "?"),
                "kind": "auditor_crash", "severity": "ERROR", "port": None,
                "detail": traceback.format_exc(limit=3)[-300:],
            })

    by_kind: dict[str, list[dict]] = defaultdict(list)
    for f in all_findings:
        by_kind[f["kind"]].append(f)
    sev = Counter(f["severity"] for f in all_findings)

    print(f"\n=== NODE CONTRACT AUDIT — {len(items)} methods, "
          f"{AUDIT_W}x{AUDIT_H} ===")
    print(f"findings: {sev.get('ERROR',0)} ERROR  {sev.get('WARN',0)} WARN  "
          f"{sev.get('INFO',0)} INFO\n")

    order = ["raised", "node_error", "type_mismatch", "missing_output",
             "unknown_port_type", "undeclared_output", "not_driveable",
             "slow", "auditor_crash"]
    for kind in order + [k for k in by_kind if k not in order]:
        group = by_kind.get(kind)
        if not group:
            continue
        s = group[0]["severity"]
        print(f"── {kind}  [{s}]  ×{len(group)}")
        cats = Counter(f["category"] for f in group)
        print(f"     by category: {dict(cats.most_common(6))}")
        for f in group[:6]:
            print(f"     {f['method_id']:>6}  {f['name'][:26]:26}  {f['detail'][:90]}")
        if len(group) > 6:
            print(f"     … and {len(group)-6} more")
        print()

    if args.json:
        Path(args.json).write_text(json.dumps(all_findings, indent=1), encoding="utf-8")
        print(f"wrote {args.json}")

    return 1 if sev.get("ERROR", 0) else 0


if __name__ == "__main__":
    sys.exit(main())
