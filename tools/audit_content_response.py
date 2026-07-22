#!/usr/bin/env python3
"""audit_content_response.py — does a TEXT wire actually change the output?

The TEXT counterpart to ``tools/audit_field_response.py``, and it exists for the
same reason: declaring a port is not the same as honouring it. A param can carry
``content: True``, show a wire in the UI, and still be read via
``params.get(...)`` in a code path the wire never reaches.

The probe
---------
Wire a Text Source upstream of the target param and render twice with two
different strings. If the node reads the wire, the output differs. If it does
not, the renders are byte-identical no matter what the port declares.

Two strings, not one, because a node may ignore the wire and still produce
different output frame to frame; comparing two *wired* renders at the same seed
and frame isolates the wire as the only variable.

Verdicts
--------
    WIRED        output changes with the wired string — the port works
    NOT_WIRED    setting the param DIRECTLY changes output, but wiring it does
                 not: a genuine TEXT-routing defect
    INERT        neither the wire nor the param changes anything — the param
                 itself does nothing, so this is a node bug (or a missing
                 optional dependency sending it down a fallback path), NOT a
                 port problem. QR Code #09 reads this way with `qrcode`
                 uninstalled: its fallback renders the same pattern whatever
                 the payload.
    ERROR        node raised

The direct-param control is what separates the last two. Without it a node whose
param is inert for its own reasons looks identical to one whose wire is broken,
and the fix for those is in completely different places.

Usage:
    python tools/audit_content_response.py --scan
    python tools/audit_content_response.py --ids 09,15
"""
from __future__ import annotations

import argparse
import json
import sys
import tempfile
import traceback
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

import image_pipeline.methods  # noqa: F401,E402 — registers every method
from image_pipeline.core import registry  # noqa: E402
from image_pipeline.core.graph import GraphExecutor, GraphNode, GraphEdge  # noqa: E402
from image_pipeline.core.utils import set_canvas  # noqa: E402

W, H = 96, 72
SEED = 42
EPS = 1e-6
TEXT_SOURCE = "__text_source__"

# Two payloads that differ in length, glyph mix and line count, so any consumer
# — a QR encoder, a typography layout, a shader compiler — renders them apart.
PAYLOAD_A = "ALPHA-0001"
PAYLOAD_B = "zeta 99 // a longer payload\nwith a second line"


def _render(mid: str, param: str, payload: str, wired: bool = True) -> np.ndarray:
    """Render with the payload delivered over a TEXT wire, or set directly."""
    if wired:
        nodes = [
            GraphNode(id="txt", method_id=TEXT_SOURCE, params={"text": payload}).__dict__,
            GraphNode(id="n0", method_id=mid, params={}).__dict__,
        ]
        edges = [GraphEdge(src_node="txt", src_port="text",
                           dst_node="n0", dst_port=param).__dict__]
    else:
        nodes = [GraphNode(id="n0", method_id=mid, params={param: payload}).__dict__]
        edges = []
    with tempfile.TemporaryDirectory() as tmp:
        ex = GraphExecutor(out_dir=Path(tmp), fps=24, in_memory=True)
        flat, _terminal, errors = ex.execute(
            nodes=nodes, edges=edges, seed=SEED, frame=0, frames=1,
        )
        if errors:
            raise RuntimeError(f"node errors: {errors}")
        img = (flat.get("n0") or {}).get("image")
        if not isinstance(img, np.ndarray):
            raise RuntimeError("node emitted no image")
        return img.astype(np.float32)


def _payloads(mid: str, param: str) -> tuple[str, str]:
    """Two payloads valid for this param.

    Prose is wrong for a param that expects GLSL source or a file path — the
    node rejects it and the probe reports ERROR for what is actually correct
    validation. Such params declare their own pair:

        "glsl_code": {"content": True, "content_probe": ["<shader a>", "<shader b>"]}
    """
    meta = registry.get_all().get(mid)
    spec = (meta.params or {}).get(param) if meta else None
    pair = spec.get("content_probe") if isinstance(spec, dict) else None
    if isinstance(pair, (list, tuple)) and len(pair) == 2:
        return str(pair[0]), str(pair[1])
    return PAYLOAD_A, PAYLOAD_B


def probe_param(mid: str, param: str) -> dict:
    pa, pb = _payloads(mid, param)
    try:
        wired_delta = float(np.abs(_render(mid, param, pa) - _render(mid, param, pb)).mean())
    except Exception as e:  # noqa: BLE001 — every failure is a reportable verdict
        return {"method_id": mid, "param": param, "verdict": "ERROR",
                "error": f"{type(e).__name__}: {e}", "delta": 0.0, "direct_delta": 0.0}
    if wired_delta > EPS:
        return {"method_id": mid, "param": param, "verdict": "WIRED",
                "delta": wired_delta, "direct_delta": 0.0}

    # No response over the wire. Is the param inert on its own, or is the wire
    # the broken part? Only the direct-set control can tell those apart.
    try:
        direct = float(np.abs(_render(mid, param, pa, wired=False)
                              - _render(mid, param, pb, wired=False)).mean())
    except Exception as e:  # noqa: BLE001
        return {"method_id": mid, "param": param, "verdict": "ERROR",
                "error": f"direct-set control: {type(e).__name__}: {e}",
                "delta": 0.0, "direct_delta": 0.0}
    return {"method_id": mid, "param": param,
            "verdict": "NOT_WIRED" if direct > EPS else "INERT",
            "delta": wired_delta, "direct_delta": direct}


def declared_content() -> dict[str, set[str]]:
    out: dict[str, set[str]] = {}
    for mid, meta in registry.get_all().items():
        if mid == TEXT_SOURCE:
            continue          # the source itself; nothing upstream to wire
        hits = {p for p, s in (meta.params or {}).items()
                if isinstance(s, dict) and s.get("content")}
        if hits:
            out[mid] = hits
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scan", action="store_true")
    ap.add_argument("--ids")
    ap.add_argument("--json")
    args = ap.parse_args()
    if not args.scan and not args.ids:
        ap.error("pass --scan or --ids")

    set_canvas(W, H)
    targets = declared_content()
    if args.ids:
        want = {s.strip() for s in args.ids.split(",") if s.strip()}
        targets = {k: v for k, v in targets.items() if k in want}
    if not targets:
        print("no params declare content: True for the given selection")
        if args.json:
            Path(args.json).write_text("[]")
        return 0

    metas = registry.get_all()
    results: list[dict] = []
    for mid in sorted(targets):
        name = metas[mid].name if mid in metas else mid
        print(f"\n{mid}  {name}")
        for param in sorted(targets[mid]):
            r = probe_param(mid, param)
            r["name"] = name
            results.append(r)
            mark = {"WIRED": "✓", "NOT_WIRED": "✗", "INERT": "·", "ERROR": "!"}.get(r["verdict"], " ")
            if r["verdict"] == "ERROR":
                extra = "  " + r.get("error", "")[:70]
            else:
                extra = f"  Δwire={r['delta']:.5f} Δdirect={r.get('direct_delta', 0.0):.5f}"
            print(f"  {mark} {param:<20} {r['verdict']:<10}{extra}")

    counts: dict[str, int] = {}
    for r in results:
        counts[r["verdict"]] = counts.get(r["verdict"], 0) + 1
    print("\n" + "─" * 60)
    print("summary: " + "  ".join(f"{k}={v}" for k, v in sorted(counts.items())))
    print(f"{counts.get('WIRED', 0)}/{len(results)} content params read their TEXT wire")
    if args.json:
        Path(args.json).write_text(json.dumps(results, indent=2))
        print(f"wrote {args.json}")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        sys.exit(130)
    except Exception:
        traceback.print_exc()
        sys.exit(2)
