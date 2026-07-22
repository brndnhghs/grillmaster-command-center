#!/usr/bin/env python3
"""audit_field_response.py — does a param actually respond to a FIELD, per-pixel?

The sibling audits answer neighbouring questions but not this one:

  * ``tools/audit_node_contract.py``  does a node EMIT what it declares?
  * ``tools/audit_dead_params.py``    does ANIMATION reach the pixels?
  * ``tools/validate_image_wiring``   is an image input wired to a real source?

None of them ask whether a *spatially varying* input actually varies the output
spatially. That gap is invisible to code review: ``codegen/gradient.py`` reads
``_field_cx`` / ``_field_cy`` / ``_field_direction`` into locals and then never
uses them, and ``cli_tools.py`` reads four ``_field_*`` arrays and immediately
``np.mean()``s every one. Both grep as "field consumers". Neither responds to
the field's structure. Only running them tells you.

The probe
---------
For a param P, render the node twice at the same canvas and seed:

    A  ``_field_P`` = uniform 0.5
    B  ``_field_P`` = horizontal ramp 0 -> 1   (mean is also 0.5)

Both carry an identical mean, so any node that collapses the field through
``np.mean`` produces byte-identical output. Divergence therefore proves the
node reads the field's spatial structure, not just its average.

A second ramp (vertical, same mean) separates a genuine spatial response from
incidental noise: a node honouring the field must respond differently to a
horizontal and a vertical ramp, since only their orientation differs.

Verdicts
--------
    SPATIAL    output changes vs uniform AND distinguishes H from V ramp
    MEAN_ONLY  output identical to uniform — field collapsed to its average
    ORIENTED?  changes vs uniform but H and V are identical (suspicious)
    NO_PARAM   node never reads ``_field_<P>`` (no port / not implemented)
    ERROR      node raised

Usage:
    python tools/audit_field_response.py --scan            # every _field_ reader in the tree
    python tools/audit_field_response.py --ids 10,25,30
    python tools/audit_field_response.py --json out.json
"""
from __future__ import annotations

import argparse
import json
import re
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
# Two renders of the same node with identical params must be bit-identical for
# the comparison to mean anything. Anything above this is a real difference.
EPS = 1e-6


# ── Probe fields (all share mean 0.5, so a mean-collapsing node can't tell them apart) ──

def _uniform() -> np.ndarray:
    return np.full((H, W), 0.5, dtype=np.float32)


def _ramp_h() -> np.ndarray:
    return np.tile(np.linspace(0.0, 1.0, W, dtype=np.float32), (H, 1))


def _ramp_v() -> np.ndarray:
    return np.tile(np.linspace(0.0, 1.0, H, dtype=np.float32)[:, None], (1, W))


# Source node wired upstream of filters that need an image. It must be a real
# graph edge, not an injected param: with no upstream edge the executor forces
# run_params["_input_image"] = None, so a filter probed alone renders a constant
# (ASCII Art #01 returns a solid error frame) and no param can appear spatial.
# Concentric gradient = strong ring structure that survives the block-averaging
# and quantisation several filters apply.
SOURCE_ID = "11"
SOURCE_PARAMS = {"gradient_type": "concentric", "style": "solid"}


def _needs_image(mid: str) -> bool:
    meta = registry.get_all().get(mid)
    return bool(meta and "image_in" in (meta.inputs or {}))


def _render(mid: str, params: dict, frame: int = 0, frames: int = 1,
            with_source: bool = True) -> np.ndarray:
    node = GraphNode(id="n0", method_id=mid, params=dict(params))
    nodes = [node.__dict__]
    edges: list[dict] = []
    if with_source and _needs_image(mid) and mid != SOURCE_ID:
        nodes.insert(0, GraphNode(id="src", method_id=SOURCE_ID,
                                  params=dict(SOURCE_PARAMS)).__dict__)
        edges.append(GraphEdge(src_node="src", src_port="image",
                               dst_node="n0", dst_port="image_in").__dict__)
    with tempfile.TemporaryDirectory() as tmp:
        ex = GraphExecutor(out_dir=Path(tmp), fps=24, in_memory=True)
        flat, terminal, errors = ex.execute(
            nodes=nodes, edges=edges, seed=SEED, frame=frame, frames=frames,
        )
        if errors:
            raise RuntimeError(f"node errors: {errors}")
        img = (flat.get("n0") or {}).get("image")
        if not isinstance(img, np.ndarray):
            raise RuntimeError("node emitted no image")
        return img.astype(np.float32)


# Simulations start from a uniform/near-empty state, so a diffusion-style param
# multiplies a laplacian that is identically zero on frame 0 — its field cannot
# reach the pixels yet, and probing only frame 0 reports a false MEAN_ONLY.
# Retry on an evolved frame before concluding.
_STAGES = ((0, 1), (4, 5))


def _probe_with(mid: str, param: str) -> dict:
    """Extra params needed to make this param observable.

    Some params are mathematically inert at their node's defaults — #11
    Gradient's ``cy`` multiplies ``sin(direction)``, which is 0 at the default
    direction=0, so a linear gradient genuinely ignores its Y centre. That is
    correct physics, not a broken param, but it renders identically for every
    field and would report a false MEAN_ONLY.

    A param declares the configuration that exposes it:

        "cy": {"spatial": True, "probe_with": {"gradient_type": "radial"}, ...}
    """
    meta = registry.get_all().get(mid)
    spec = (meta.params or {}).get(param) if meta else None
    hint = spec.get("probe_with") if isinstance(spec, dict) else None
    return dict(hint) if isinstance(hint, dict) else {}


def _probe_at(mid: str, param: str, frame: int, frames: int,
              with_source: bool) -> tuple[float, float]:
    key = f"_field_{param}"
    extra = _probe_with(mid, param)

    def r(field):
        return _render(mid, {**extra, key: field}, frame, frames, with_source)

    base = r(_uniform())
    # Determinism guard: if the node is nondeterministic at fixed seed the
    # comparison below is meaningless, so say so rather than report noise.
    if float(np.abs(base - r(_uniform())).max()) > EPS:
        return (-1.0, -1.0)   # sentinel: nondeterministic
    hor, ver = r(_ramp_h()), r(_ramp_v())
    return (float(np.abs(hor - base).mean()), float(np.abs(hor - ver).mean()))


def _source_modes(mid: str) -> tuple[bool, ...]:
    """Which upstream configurations to try, in order.

    Filters that accept an image often have two code paths — a procedural
    fallback when nothing is wired, and the wired-image path. A param can be
    per-pixel in one and inert in the other, so try both before calling it
    MEAN_ONLY: the question is whether it responds in a real configuration, not
    in one arbitrarily chosen configuration.
    """
    return (True, False) if _needs_image(mid) and mid != SOURCE_ID else (False,)


def probe_param(mid: str, param: str) -> dict:
    """Render uniform / H-ramp / V-ramp and classify the response."""
    d_uniform = d_orient = 0.0
    frame = 0
    modes = _source_modes(mid)
    with_source = modes[0]
    try:
        for with_source in modes:
            for frame, frames in _STAGES:
                d_uniform, d_orient = _probe_at(mid, param, frame, frames, with_source)
                if d_uniform < 0:
                    return {"method_id": mid, "param": param, "verdict": "NONDETERMINISTIC",
                            "d_uniform": 0.0, "d_orient": 0.0, "frame": frame}
                if d_uniform > EPS:
                    break   # responded — no need for the slower evolved stage
            if d_uniform > EPS:
                break
    except Exception as e:  # noqa: BLE001 — every failure is a reportable verdict
        return {"method_id": mid, "param": param, "verdict": "ERROR",
                "error": f"{type(e).__name__}: {e}", "d_uniform": 0.0, "d_orient": 0.0}

    # Which configuration it responded in is a finding, not an implementation
    # detail. A param that works standalone but is inert once an image is wired
    # passes the gate on a technicality: wiring an image flips `source` to
    # "input_image" (the executor does this automatically), and the procedural
    # field the param modulates is gone on that path. Surfacing it keeps the
    # gate from quietly certifying half a capability.
    standalone_only = bool(len(modes) > 1 and not with_source and d_uniform > EPS)

    if d_uniform <= EPS:
        verdict = "MEAN_ONLY"
    elif d_orient <= EPS:
        verdict = "ORIENTED?"
    else:
        verdict = "SPATIAL"
    return {"method_id": mid, "param": param, "verdict": verdict,
            "d_uniform": d_uniform, "d_orient": d_orient, "frame": frame,
            "standalone_only": standalone_only}


# ── Discovery ────────────────────────────────────────────────────────

_FIELD_RE = re.compile(r'_field_([A-Za-z_][A-Za-z0-9_]*)')


def scan_field_readers() -> dict[str, set[str]]:
    """method_id -> {param} for every param participating in the spatial contract.

    Two ways to participate:
      * declaring ``spatial: True`` — the current contract, and what the
        migration marks; the node reads the value via ``sparam()``.
      * a legacy raw ``_field_<param>`` read in the source.

    Source-scanning alone is not enough: a migrated node calls
    ``sparam(params, "feed", ...)`` and never mentions ``_field_feed``, so it
    would be invisible to a grep-only scan and silently skipped by the gate.
    """
    out: dict[str, set[str]] = {}

    # Declared contract.
    for mid, meta in registry.get_all().items():
        declared = {p for p, s in (meta.params or {}).items()
                    if isinstance(s, dict) and s.get("spatial")}
        if declared:
            out[mid] = set(declared)

    # Legacy raw _field_ reads.
    by_module: dict[str, set[str]] = {}
    for path in (REPO / "image_pipeline" / "methods").rglob("*.py"):
        found = set(_FIELD_RE.findall(path.read_text(errors="ignore")))
        if found:
            by_module[path.stem] = found
    for mid, meta in registry.get_all().items():
        stem = (meta.module or "").rsplit(".", 1)[-1]
        if stem in by_module:
            # Only params the node actually declares — a _field_ read for an
            # undeclared name can never be driven by a wire.
            hits = by_module[stem] & set(meta.params or {})
            if hits:
                out.setdefault(mid, set()).update(hits)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scan", action="store_true",
                    help="probe every _field_ reader found in the method tree")
    ap.add_argument("--ids", help="comma-separated method ids")
    ap.add_argument("--json", help="write results to this path")
    args = ap.parse_args()

    set_canvas(W, H)
    targets = scan_field_readers()
    if args.ids:
        want = {s.strip() for s in args.ids.split(",") if s.strip()}
        targets = {k: v for k, v in targets.items() if k in want}
        for mid in want - set(targets):
            print(f"  {mid}: no _field_ reads found in source")
    if not args.scan and not args.ids:
        ap.error("pass --scan or --ids")
    if not targets:
        print("no participating params found for the given selection")
        if args.json:
            Path(args.json).write_text("[]")
        return 0

    metas = registry.get_all()
    results: list[dict] = []
    for mid in sorted(targets, key=lambda m: (metas[m].name if m in metas else m)):
        name = metas[mid].name if mid in metas else mid
        print(f"\n{mid}  {name}")
        for param in sorted(targets[mid]):
            r = probe_param(mid, param)
            r["name"] = name
            results.append(r)
            mark = {"SPATIAL": "✓", "MEAN_ONLY": "·", "ORIENTED?": "?",
                    "ERROR": "!", "NONDETERMINISTIC": "~"}.get(r["verdict"], " ")
            extra = f"  Δuniform={r['d_uniform']:.5f} Δorient={r['d_orient']:.5f}"
            if r.get("standalone_only"):
                extra += "  [standalone only — inert when image_in is wired]"
            if r["verdict"] == "ERROR":
                extra = "  " + r.get("error", "")[:70]
            print(f"  {mark} {param:<24} {r['verdict']:<16}{extra}")

    counts: dict[str, int] = {}
    for r in results:
        counts[r["verdict"]] = counts.get(r["verdict"], 0) + 1
    print("\n" + "─" * 62)
    print("summary: " + "  ".join(f"{k}={v}" for k, v in sorted(counts.items())))
    spatial = counts.get("SPATIAL", 0)
    print(f"{spatial}/{len(results)} probed params genuinely respond to field structure")

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
