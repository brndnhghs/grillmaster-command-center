#!/usr/bin/env python3
"""migrate_spatial.py — convert Class-A params to per-pixel, and PROVE each one.

Reads the ledger from ``tools/classify_params.py`` and, for every Class-A
candidate, does three things:

  1. rewrite the read site   ``F = float(params.get("feed", 0.035))``
                          -> ``F = sparam(params, "feed", 0.035)``
  2. mark the param          ``"feed": {"spatial": True, ...}``   (grants a FIELD port)
  3. add the import          ``from image_pipeline.core.spatial import sparam``

Then it runs ``tools/audit_field_response.py`` over everything it touched and
**reverts every param that does not come back SPATIAL**. Class A is a static
guess; only the probe knows whether the field reached the pixels. A param that
errors, or that still collapses to its mean, is restored to exactly its previous
source — so a failed migration costs nothing and never leaves a node broken or
advertising support it does not have.

Batching matters: the probe imports the whole method tree (~4 s), so edits are
applied in a batch, probed in ONE subprocess, and only the failures reverted.

Usage:
    python tools/migrate_spatial.py --ids 155 --apply       # one node
    python tools/migrate_spatial.py --limit 40 --apply      # next 40 candidates
    python tools/migrate_spatial.py --limit 10              # dry run (default)
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import subprocess
import sys
import tempfile
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
LEDGER = REPO / "tools" / "param_ledger.csv"
IMPORT_LINE = "from image_pipeline.core.spatial import sparam"

# `X = float(params.get("name", default))`  /  int(...)  /  bare params.get(...)
READ_RE = r'^([ \t]*)([A-Za-z_]\w*)([ \t]*=[ \t]*)(?:float|int)\([ \t]*params\.get\([ \t]*(["\']){name}\4[ \t]*,[ \t]*([^())]+?)[ \t]*\)[ \t]*\)[ \t]*$'


def rewrite_read(src: str, param: str) -> tuple[str, bool]:
    """Point the param's read site at sparam()."""
    pat = re.compile(READ_RE.format(name=re.escape(param)), re.MULTILINE)

    def _sub(m: re.Match) -> str:
        indent, var, eq, q, default = m.groups()
        return f'{indent}{var}{eq}sparam(params, {q}{param}{q}, {default})'

    out, n = pat.subn(_sub, src)
    return out, n > 0


def mark_spatial(src: str, param: str) -> tuple[str, bool]:
    """Add "spatial": True to the param's spec dict in the @method decorator."""
    if re.search(rf'["\']{re.escape(param)}["\']\s*:\s*\{{[^}}]*["\']spatial["\']', src):
        return src, True   # already marked
    pat = re.compile(rf'(["\']){re.escape(param)}\1(\s*:\s*\{{)')
    out, n = pat.subn(lambda m: f'{m.group(1)}{param}{m.group(1)}{m.group(2)}"spatial": True, ', src, count=1)
    return out, n > 0


def ensure_import(src: str) -> str:
    if IMPORT_LINE in src:
        return src
    lines = src.split("\n")
    # after the last top-level import in the header block
    last = 0
    for i, ln in enumerate(lines[:80]):
        if ln.startswith(("import ", "from ")) and "sparam" not in ln:
            last = i
    lines.insert(last + 1, IMPORT_LINE)
    return "\n".join(lines)


FAILURES = REPO / "tools" / "spatial_failures.json"


def load_failures() -> dict[str, str]:
    """(method_id|param) -> verdict for params already tried and reverted.

    Without this the loop is not monotonic: --limit always slices the head of
    the ledger, so every batch re-attempts the same failures and never advances.
    """
    if FAILURES.exists():
        try:
            return json.loads(FAILURES.read_text())
        except json.JSONDecodeError:
            return {}
    return {}


def save_failures(known: dict[str, str]) -> None:
    FAILURES.write_text(json.dumps(dict(sorted(known.items())), indent=2))


def load_candidates(klass: str = "A") -> list[dict]:
    known = load_failures()
    with open(LEDGER, newline="") as fh:
        return [r for r in csv.DictReader(fh)
                if r["class"] == klass
                and r["already_spatial"] != "True"
                and f'{r["method_id"]}|{r["param"]}' not in known]


def probe(pairs: list[tuple[str, str]]) -> dict[tuple[str, str], str]:
    """Run the response probe over the touched method ids; return verdicts."""
    ids = sorted({mid for mid, _ in pairs})
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tf:
        out = Path(tf.name)
    cmd = [sys.executable, str(REPO / "tools" / "audit_field_response.py"),
           "--ids", ",".join(ids), "--json", str(out)]
    subprocess.run(cmd, cwd=REPO, capture_output=True, text=True, timeout=3600)
    verdicts: dict[tuple[str, str], str] = {}
    if out.exists():
        try:
            for r in json.loads(out.read_text()):
                verdicts[(r["method_id"], r["param"])] = r["verdict"]
        except json.JSONDecodeError:
            pass
        out.unlink(missing_ok=True)
    return verdicts


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ids", help="only these method ids (comma-separated)")
    ap.add_argument("--limit", type=int, help="max params to attempt")
    ap.add_argument("--apply", action="store_true", help="write changes (default: dry run)")
    ap.add_argument("--report", help="append outcomes to this JSON")
    args = ap.parse_args()

    cands = load_candidates()
    if args.ids:
        want = {s.strip() for s in args.ids.split(",")}
        cands = [c for c in cands if c["method_id"] in want]
    if args.limit:
        cands = cands[:args.limit]
    if not cands:
        print("no candidates match")
        return 0

    print(f"{len(cands)} Class-A candidates across "
          f"{len({c['method_id'] for c in cands})} methods\n")

    # ── pass 1: apply edits, remembering original file contents ──
    originals: dict[Path, str] = {}
    applied: list[dict] = []
    skipped: list[dict] = []
    for c in cands:
        path = REPO / c["module"]
        if not path.exists():
            continue
        if path not in originals:
            originals[path] = path.read_text()
        src = path.read_text()
        new, ok_read = rewrite_read(src, c["param"])
        if not ok_read:
            skipped.append({**c, "why": "read site did not match the expected form"})
            continue
        new, ok_mark = mark_spatial(new, c["param"])
        if not ok_mark:
            skipped.append({**c, "why": "could not locate param spec in decorator"})
            continue
        new = ensure_import(new)
        # Compile before writing. One malformed rewrite would otherwise take
        # down the probe subprocess for the whole batch, reverting every good
        # edit alongside the bad one.
        try:
            compile(new, str(path), "exec")
        except SyntaxError as e:
            skipped.append({**c, "why": f"rewrite would not compile: {e}"})
            continue
        if args.apply:
            path.write_text(new)
        applied.append(c)

    print(f"  rewritten : {len(applied)}")
    print(f"  skipped   : {len(skipped)} (read site or spec not in the expected shape)")
    if not args.apply:
        print("\ndry run — nothing written. re-run with --apply")
        for c in applied[:15]:
            print(f"    would migrate {c['method_id']:>6} {c['param']}")
        return 0
    if not applied:
        return 0

    # ── pass 2: probe everything touched, in ONE subprocess ──
    print("\nprobing migrated params (uniform vs H-ramp vs V-ramp)…")
    pairs = [(c["method_id"], c["param"]) for c in applied]
    verdicts = probe(pairs)

    kept, reverted = [], []
    for c in applied:
        v = verdicts.get((c["method_id"], c["param"]), "NOT_PROBED")
        (kept if v == "SPATIAL" else reverted).append({**c, "verdict": v})

    # ── pass 3: revert failures ──
    # Whole-file restore then re-apply only the winners: a param-level undo
    # would have to unpick edits that share a file.
    if reverted:
        for path, text in originals.items():
            path.write_text(text)
        for c in kept:
            path = REPO / c["module"]
            src = path.read_text()
            src, _ = rewrite_read(src, c["param"])
            src, _ = mark_spatial(src, c["param"])
            path.write_text(ensure_import(src))

    # Remember failures (including un-migratable read sites) so the next batch
    # advances to new params instead of re-attempting these.
    known = load_failures()
    for c in reverted:
        known[f'{c["method_id"]}|{c["param"]}'] = c["verdict"]
    for c in skipped:
        known[f'{c["method_id"]}|{c["param"]}'] = "SKIPPED_SHAPE"
    save_failures(known)

    print(f"\n  KEPT     {len(kept):>4}  (probe: SPATIAL)")
    print(f"  REVERTED {len(reverted):>4}  (did not respond to field structure)")
    by_v: dict[str, int] = defaultdict(int)
    for c in reverted:
        by_v[c["verdict"]] += 1
    for v, n in sorted(by_v.items()):
        print(f"      {v:<18} {n}")
    if kept:
        print("\n  migrated:")
        for c in kept:
            print(f"    ✓ {c['method_id']:>6}  {c['param']:<20} {c['name'][:34]}")

    if args.report:
        Path(args.report).write_text(json.dumps(
            {"kept": kept, "reverted": reverted, "skipped": skipped}, indent=2))
        print(f"\nwrote {args.report}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
