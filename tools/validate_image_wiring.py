#!/usr/bin/env python3
"""
validate_image_wiring.py — verify every node's image input is wired to a real,
image-emitting upstream port *within the same graph document*.

For each node instance in each scanned graph we check:
  1. Does it have an image-typed input port? (image_in, image_a, image_b, seed_image, …)
  2. For each such port, is there at least one inbound edge from a node that
     (a) actually exists in the graph, and
     (b) actually emits an "image" output on the referenced src_port?
  3. Method-contract pass: if a node declares `image_in` but its method body
     never reads the wired image (`_input_image` / `input_image`), the port is
     dead/optional — flag as WARN rather than a hard dangling-input error.

Mandatory image ports (unwired => ERROR):
  - `image_in`  when the method consumes the wired image (filter / most sims)
  - `image_a`, `image_b`  (compositing merge inputs — always required)
Optional named image ports (unwired => WARN): everything else (seed_image, mask_image, …)

Outputs:
  - human-readable report to stdout
  - tools/image_wiring_report.json   (machine-readable, one row per finding + summary)
  - tools/image_wiring_report.md     (markdown summary)
Exit code: 2 if any ERROR-level finding, else 0 (warnings do not fail).
"""
from __future__ import annotations

import argparse
import datetime
import inspect
import json
import sys
import traceback
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

# Mandatory image merge ports — always required, even if the method also has a
# default. Mirrors the executor's special-case handling in graph.py.
REQUIRED_MERGE_PORTS = {"image_a", "image_b"}
PRIMARY_PORT = "image_in"

# Graph doc locations (relative to image_pipeline/output)
GRAPHS_DIR = REPO / "image_pipeline" / "output" / "graphs"
SAVED_DIR = REPO / "image_pipeline" / "output" / "saved-graphs"

# ── Load the real node-definition contract ──────────────────────────────

def _load_contract() -> tuple[dict[str, dict], dict[str, bool]]:
    """Return (node_defs, consumes_input) keyed by method_id."""
    import image_pipeline.methods  # noqa: F401  (populates the registry)
    from image_pipeline.core.graph import get_all_node_defs
    from image_pipeline.core import registry

    defs = get_all_node_defs()
    consumes: dict[str, bool] = {}
    for mid, meta in registry.get_all().items():
        consumes[mid] = _method_consumes_input(meta)
    return defs, consumes


def _method_consumes_input(meta: Any) -> bool:
    """True if the method body reads the wired upstream image."""
    fn = getattr(meta, "fn", None)
    if fn is None:
        return False
    try:
        src = inspect.getsource(fn)
    except (OSError, TypeError):
        return False  # can't introspect — treat as "unknown" (downgrades severity)
    return ("_input_image" in src) or ("input_image" in src)


# ── Validation core ─────────────────────────────────────────────────────

def _src_emits_image(
    src_id: str, src_port: str, nodes_by_id: dict[str, dict], defs: dict[str, dict]
) -> tuple[bool, str]:
    """Return (ok, reason). reason is a short machine token."""
    sdef = nodes_by_id.get(src_id)
    if sdef is None:
        return False, "src-node-missing"
    smid = sdef.get("method_id", "")
    sdefn = defs.get(smid)
    if sdefn is None:
        return False, "src-method-unknown"
    out = sdefn.get("outputs", {})
    t = out.get(src_port)
    if t is None:
        return False, "src-port-absent"
    if str(t).lower() != "image":
        return False, f"src-port-not-image({t})"
    return True, "ok"


def _validate_graph(
    gid: str,
    doc: dict,
    defs: dict[str, dict],
    consumes: dict[str, bool],
) -> list[dict]:
    findings: list[dict] = []
    nodes = doc.get("nodes", []) or []
    edges = doc.get("edges", []) or []
    nodes_by_id = {n.get("id"): n for n in nodes if n.get("id")}

    # Index edges by dst node+port and by src for quick lookup.
    inbound: dict[tuple[str, str], list[dict]] = {}
    for e in edges:
        key = (e.get("dst_node"), e.get("dst_port"))
        inbound.setdefault(key, []).append(e)

    # 0) Orphan/integrity edges: dst or src node missing.
    for e in edges:
        sn, dn = e.get("src_node"), e.get("dst_node")
        if dn not in nodes_by_id:
            findings.append(_f(gid, None, None, "E_DST_MISSING",
                f"edge src={sn}->{dn}:{e.get('dst_port')} references missing dst node"))
        if sn not in nodes_by_id:
            findings.append(_f(gid, None, None, "E_EDGE_SRC_MISSING",
                f"edge {sn}:{e.get('src_port')}->{dn} references missing src node"))

    for n in nodes:
        nid = n.get("id")
        mid = n.get("method_id", "")
        if not mid:
            # Group/placeholder node — no method to validate; skip silently.
            continue
        ndef = defs.get(mid)
        if ndef is None:
            findings.append(_f(gid, nid, mid, "E_ORPHAN_NODE",
                f"node references unknown method_id '{mid}' (no NodeDef)"))
            continue

        inputs = ndef.get("inputs", {})
        image_ports = [p for p, t in inputs.items() if str(t).lower() == "image"]
        if not image_ports:
            continue  # pure generator / no image input — nothing to check

        consumes_in = consumes.get(mid, False)

        for port in image_ports:
            inc = inbound.get((nid, port), [])
            if not inc:
                # No inbound edge to this image port.
                if port == PRIMARY_PORT:
                    if consumes_in:
                        findings.append(_f(gid, nid, mid, "E_DANGLING_REQUIRED",
                            "image_in has no inbound edge and the method consumes the "
                            "wired image — will run without an upstream image (broken)"))
                    else:
                        findings.append(_f(gid, nid, mid, "W_DEAD_PORT",
                            "image_in unwired but method never reads the wired image "
                            "(dead/optional port — safe but worth pruning)"))
                elif port in REQUIRED_MERGE_PORTS:
                    findings.append(_f(gid, nid, mid, "E_DANGLING_MERGE",
                        f"required merge port '{port}' has no inbound edge "
                        "(compositing node needs two image sources)"))
                else:
                    findings.append(_f(gid, nid, mid, "W_OPTIONAL_UNWIRED",
                        f"optional image port '{port}' is unwired "
                        "(method may fall back to an internal/default source)"))
                continue

            # Inbound edge(s) present — verify each is tangible.
            for e in inc:
                sn = e.get("src_node")
                sp = e.get("src_port", "image")
                ok, reason = _src_emits_image(sn, sp, nodes_by_id, defs)
                if not ok:
                    findings.append(_f(gid, nid, mid, "E_EDGE_NOT_TANGIBLE",
                        f"image port '{port}' wired from {sn}:{sp} — {reason} "
                        "(source does not emit a real image on that port)"))

    return findings


def _f(gid, node_id, method_id, code, message) -> dict:
    sev = "ERROR" if code.startswith("E_") else "WARN"
    return {
        "graph": gid,
        "node_id": node_id,
        "method_id": method_id,
        "severity": sev,
        "code": code,
        "message": message,
    }


# ── Graph discovery ─────────────────────────────────────────────────────

def _discover_graphs(include_saved: bool, include_graphs: bool) -> list[tuple[str, dict]]:
    out: list[tuple[str, dict]] = []
    candidates: list[Path] = []
    if include_graphs and GRAPHS_DIR.exists():
        candidates += sorted(GRAPHS_DIR.glob("*.json"))
    if include_saved and SAVED_DIR.exists():
        candidates += sorted(SAVED_DIR.glob("*.json"))
    for p in candidates:
        try:
            doc = json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            continue
        gid = doc.get("id") or doc.get("name") or p.stem
        out.append((gid, doc))
    return out


def _fetch_live(url: str) -> list[tuple[str, dict]]:
    """Optional: pull graphs from a running server instead of disk."""
    import httpx
    out: list[tuple[str, dict]] = []
    base = url.rstrip("/")
    # active graph
    try:
        r = httpx.get(f"{base}/api/graph/active", timeout=10)
        if r.status_code == 200:
            out.append(("active", r.json()))
    except Exception:
        pass
    # saved graphs list
    try:
        r = httpx.get(f"{base}/api/graph/saved", timeout=10)
        if r.status_code == 200:
            for entry in (r.json() or []):
                name = entry.get("name") or entry.get("id")
                if not name:
                    continue
                g = httpx.get(f"{base}/api/graph/saved/{name}", timeout=10)
                if g.status_code == 200:
                    out.append((name, g.json()))
    except Exception:
        pass
    return out


# ── Reporting ───────────────────────────────────────────────────────────

def _render(findings: list[dict], graphs_scanned: int) -> str:
    errors = [f for f in findings if f["severity"] == "ERROR"]
    warns = [f for f in findings if f["severity"] == "WARN"]
    lines: list[str] = []
    lines.append("=" * 72)
    lines.append(" IMAGE-INPUT WIRING VALIDATION")
    lines.append("=" * 72)
    lines.append(f" graphs scanned : {graphs_scanned}")
    lines.append(f" ERROR findings : {len(errors)}")
    lines.append(f" WARN  findings : {len(warns)}")
    lines.append("")

    if not findings:
        lines.append(" ✓ All image inputs are wired to tangible, image-emitting sources.")
        return "\n".join(lines)

    by_graph: dict[str, list[dict]] = {}
    for f in findings:
        by_graph.setdefault(f["graph"], []).append(f)

    for gid in sorted(by_graph):
        items = by_graph[gid]
        e = sum(1 for x in items if x["severity"] == "ERROR")
        w = sum(1 for x in items if x["severity"] == "WARN")
        lines.append(f"── graph: {gid}   ({e} ERROR / {w} WARN) ──")
        for f in items:
            loc = f["node_id"] or "-"
            if f["method_id"]:
                loc += f" ({f['method_id']})"
            lines.append(f"  [{f['severity']}] {f['code']} @ {loc}")
            lines.append(f"        {f['message']}")
        lines.append("")

    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--live", default=None,
                    help="Base URL of a running Grillmaster server (e.g. http://127.0.0.1:7872). "
                         "If set, scans the live active + saved graphs instead of disk.")
    ap.add_argument("--no-saved", action="store_true", help="skip image_pipeline/output/saved-graphs")
    ap.add_argument("--no-graphs", action="store_true", help="skip image_pipeline/output/graphs")
    args = ap.parse_args()

    try:
        defs, consumes = _load_contract()
    except Exception as exc:
        print(f"FATAL: could not load node contract: {exc}", file=sys.stderr)
        traceback.print_exc()
        return 3

    if args.live:
        graphs = _fetch_live(args.live)
    else:
        graphs = _discover_graphs(
            include_saved=not args.no_saved,
            include_graphs=not args.no_graphs,
        )

    all_findings: list[dict] = []
    for gid, doc in graphs:
        all_findings += _validate_graph(gid, doc, defs, consumes)

    report_text = _render(all_findings, len(graphs))
    print(report_text)

    ts = datetime.datetime.now(datetime.timezone.utc).isoformat()
    summary = {
        "generated_at": ts,
        "graphs_scanned": len(graphs),
        "error_count": sum(1 for f in all_findings if f["severity"] == "ERROR"),
        "warn_count": sum(1 for f in all_findings if f["severity"] == "WARN"),
        "findings": all_findings,
    }
    out_json = REPO / "tools" / "image_wiring_report.json"
    out_md = REPO / "tools" / "image_wiring_report.md"
    out_json.write_text(json.dumps(summary, indent=2))
    out_md.write_text(
        f"# Image-Input Wiring Report\n\n"
        f"- generated: {ts}\n- graphs scanned: {len(graphs)}\n"
        f"- errors: {summary['error_count']}\n- warnings: {summary['warn_count']}\n\n"
        f"```\n{report_text}\n```\n"
    )

    return 2 if summary["error_count"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
