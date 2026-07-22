#!/usr/bin/env python3
"""classify_params.py — triage every node param into A/B/C/D for spatial driving.

527 methods and 4,680 params is too many to read. This assigns each param a
class from its source, so the migration works from a ledger instead of a hunch.

    A  SPATIALIZABLE   the param's bound name is only ever used in arithmetic /
                       numpy expressions, so swapping the scalar for an (H,W)
                       array broadcasts with no other change.
    B  STRUCTURAL      the name reaches range(), int(), a shape, an index, a
                       PIL/cv2 call, or a comparison — uses that need ONE
                       number. Converting these raises or silently degrades.
    C  NON_SPATIAL     frame counts, seeds, timesteps, toggles: a per-pixel
                       value is meaningless by definition.
    D  CONTENT         str/path/code params — a different feature (text, file
                       and asset ports), not a field.

Method: find where the param is read (``params.get("<name>", ...)``), take the
name it is bound to, then walk every subsequent use of that name in the
function body. Any single structural use demotes A → B; the analysis is
deliberately pessimistic, since a false A costs a broken node while a false B
only costs a param nobody migrated yet.

Class A is a CANDIDATE list, never a guarantee. ``tools/audit_field_response.py``
is what actually proves a converted param reaches the pixels.

Usage:
    python tools/classify_params.py                     # summary
    python tools/classify_params.py --csv ledger.csv    # full ledger
    python tools/classify_params.py --class A --limit 40
"""
from __future__ import annotations

import argparse
import ast
import csv
import sys
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

import image_pipeline.methods  # noqa: F401,E402 — registers every method
from image_pipeline.core import registry  # noqa: E402

# Names that are never per-pixel regardless of how they are used.
NON_SPATIAL_NAMES = {
    "n_frames", "frames", "seed", "dt", "timeout", "fps", "time",
    "width", "height", "canvas_w", "canvas_h", "size", "resolution",
    "n_seeds", "steps", "n_steps", "iterations", "iters", "substeps",
    "anim_mode", "anim_speed", "render_style", "source", "palette_name",
    "prebake", "start_frame", "end_frame",
}

# Calls that force their argument to a single number.
SCALARISING_CALLS = {
    "range", "int", "round", "len", "reshape", "zeros", "ones", "full",
    "empty", "arange", "linspace", "tile", "repeat", "resize", "new",
    "randint", "sample", "choice", "seed", "truetype", "rotate",
}

# Calls that broadcast elementwise, so an (H,W) array passes through unharmed.
# ANY OTHER call receiving the name is treated as structural: PIL draws,
# subprocess args and f-string formatting all need one number, and being
# permissive here costs a broken node while being strict only defers a param.
BROADCAST_SAFE_CALLS = {
    "abs", "absolute", "clip", "sqrt", "exp", "log", "log2", "log10",
    "sin", "cos", "tan", "arctan", "arctan2", "sinh", "cosh", "tanh",
    "where", "maximum", "minimum", "fmax", "fmin", "power", "hypot",
    "asarray", "array", "float32", "float64", "astype", "nan_to_num",
    "sign", "floor", "ceil", "square", "reciprocal", "negative", "mod",
}


class _UseVisitor(ast.NodeVisitor):
    """Collect how a bound name is used inside a function body."""

    def __init__(self, target: str):
        self.target = target
        self.structural: list[str] = []
        self.arithmetic = 0

    def _is_target(self, node) -> bool:
        return isinstance(node, ast.Name) and node.id == self.target

    def visit_Call(self, node):
        fname = ""
        if isinstance(node.func, ast.Name):
            fname = node.func.id
        elif isinstance(node.func, ast.Attribute):
            fname = node.func.attr
        passed = any(self._is_target(a) for a in node.args) or any(
            self._is_target(kw.value) for kw in node.keywords)
        if passed and fname not in BROADCAST_SAFE_CALLS:
            self.structural.append(f"{fname}()")
        self.generic_visit(node)

    def visit_Subscript(self, node):
        # used as an index -> must be a scalar
        for sub in ast.walk(node.slice):
            if self._is_target(sub):
                self.structural.append("index")
        self.generic_visit(node)

    def visit_Compare(self, node):
        # `if x > 3` on an array raises "truth value is ambiguous"
        if self._is_target(node.left) or any(self._is_target(c) for c in node.comparators):
            self.structural.append("compare")
        self.generic_visit(node)

    def visit_BinOp(self, node):
        if self._is_target(node.left) or self._is_target(node.right):
            self.arithmetic += 1
        self.generic_visit(node)


def _find_func(tree: ast.AST, lineno: int) -> ast.FunctionDef | None:
    """Innermost function containing lineno."""
    best = None
    for n in ast.walk(tree):
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)):
            end = getattr(n, "end_lineno", n.lineno)
            if n.lineno <= lineno <= end:
                if best is None or n.lineno > best.lineno:
                    best = n
    return best


def classify_param(src: str, tree: ast.AST, pname: str, spec: dict) -> tuple[str, str]:
    """Return (class, reason) for one param."""
    if spec.get("spatial"):
        # Already migrated: its read site is sparam(), not params.get(), so the
        # binding search below would miss it and mislabel a working spatial
        # param as C.
        return "A", "already migrated (spatial: True)"
    default = spec.get("default")
    if isinstance(default, str) or spec.get("choices"):
        return "D" if _looks_like_content(pname, default) else "C", "categorical/str"
    if isinstance(default, bool):
        return "C", "bool"
    if not isinstance(default, (int, float)):
        return "C", f"non-numeric default ({type(default).__name__})"
    if pname in NON_SPATIAL_NAMES:
        return "C", "non-spatial by name"

    # Locate `params.get("<pname>" ...)` and the name it binds to.
    binding: tuple[str, int] | None = None
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        f = node.func
        if not (isinstance(f, ast.Attribute) and f.attr == "get"):
            continue
        if not node.args:
            continue
        a0 = node.args[0]
        if not (isinstance(a0, ast.Constant) and a0.value == pname):
            continue
        # walk up: find the Assign whose value contains this call
        for asn in ast.walk(tree):
            if isinstance(asn, ast.Assign) and len(asn.targets) == 1:
                if isinstance(asn.targets[0], ast.Name):
                    for sub in ast.walk(asn.value):
                        if sub is node:
                            binding = (asn.targets[0].id, asn.lineno)
                            break
            if binding:
                break
        if binding:
            break

    if binding is None:
        return "C", "no read site found"

    varname, lineno = binding
    fn = _find_func(tree, lineno)
    if fn is None:
        return "C", "read outside a function"

    v = _UseVisitor(varname)
    v.visit(fn)
    if v.structural:
        uniq = sorted(set(v.structural))
        return "B", f"structural use: {', '.join(uniq[:3])}"
    if v.arithmetic == 0:
        return "C", "never used arithmetically"
    return "A", f"{v.arithmetic} arithmetic use(s) of `{varname}`"


def _looks_like_content(pname: str, default) -> bool:
    hint = ("path", "file", "src", "source_code", "code", "text", "svg",
            "font", "url", "asset", "content")
    return any(h in pname.lower() for h in hint)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--csv", help="write the full ledger here")
    ap.add_argument("--class", dest="klass", choices=list("ABCD"))
    ap.add_argument("--limit", type=int, default=30)
    args = ap.parse_args()

    src_cache: dict[str, tuple[str, ast.AST]] = {}
    rows: list[dict] = []

    for mid, meta in registry.get_all().items():
        mod = sys.modules.get(meta.module or "")
        path = getattr(mod, "__file__", None)
        if not path or not Path(path).exists():
            continue
        if path not in src_cache:
            try:
                text = Path(path).read_text(errors="ignore")
                src_cache[path] = (text, ast.parse(text))
            except SyntaxError:
                continue
        text, tree = src_cache[path]
        # A node that emits no IMAGE cannot have a per-pixel param: drivers
        # (LFO, Counter) and sidecar emitters produce a scalar, and "per-pixel"
        # is meaningless for them. They are also unprobeable, so leaving them
        # in class A costs a guaranteed ERROR per param.
        emits_image = "image" in (meta.outputs or {})
        for pname, spec in (meta.params or {}).items():
            if not isinstance(spec, dict):
                continue
            if not emits_image:
                rows.append({
                    "class": "C", "method_id": mid, "name": meta.name,
                    "param": pname, "reason": "node emits no image",
                    "module": Path(path).relative_to(REPO).as_posix(),
                    "already_spatial": bool(spec.get("spatial")),
                })
                continue
            klass, reason = classify_param(text, tree, pname, spec)
            rows.append({
                "class": klass, "method_id": mid, "name": meta.name,
                "param": pname, "reason": reason,
                "module": Path(path).relative_to(REPO).as_posix(),
                "already_spatial": bool(spec.get("spatial")),
            })

    counts = Counter(r["class"] for r in rows)
    print(f"params classified: {len(rows)}  across {len(set(r['method_id'] for r in rows))} methods\n")
    labels = {"A": "SPATIALIZABLE (candidates)", "B": "STRUCTURAL (needs rework)",
              "C": "NON_SPATIAL (exclude)", "D": "CONTENT (text/file ports)"}
    for k in "ABCD":
        pct = 100.0 * counts[k] / max(1, len(rows))
        print(f"  {k}  {labels[k]:<28} {counts[k]:>5}  ({pct:4.1f}%)")

    if args.klass:
        sel = [r for r in rows if r["class"] == args.klass]
        print(f"\n--- class {args.klass}: {len(sel)} params (showing {min(args.limit,len(sel))}) ---")
        for r in sel[:args.limit]:
            print(f"  {r['method_id']:>8}  {r['param']:<22} {r['name'][:26]:<28} {r['reason']}")

    if args.csv:
        with open(args.csv, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        print(f"\nwrote {args.csv} ({len(rows)} rows)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
