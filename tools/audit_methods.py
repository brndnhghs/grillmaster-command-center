#!/usr/bin/env python3
"""
audit_methods.py — scan all image_pipeline method files and report what each
method exposes (declared outputs=) vs. what it likely computes (detected signals).

Usage:
    python tools/audit_methods.py          # run from repo root
    python -m tools.audit_methods          # or as module

Outputs:
    tools/audit_report.json
    tools/audit_report.md
"""
from __future__ import annotations

import ast
import json
import re
import sys
from pathlib import Path
from typing import Any

# ── Repo paths ────────────────────────────────────────────────────────────────

REPO = Path(__file__).resolve().parent.parent
METHODS_DIR = REPO / "image_pipeline" / "methods"
TOOLS_DIR = REPO / "tools"

# ── Signal vocabulary ─────────────────────────────────────────────────────────

# Strong particle indicators: variable names used as assignment targets
# Deliberately excludes generic short names (pts, pos, vel, p_arr) that appear
# in HSV color math and graphics control-point lists.
_PARTICLE_VAR_RE = re.compile(
    r"^(_?)(positions?|velocities?|agents?|boids?|ants?|"
    r"particles?_?arr|x_arr|y_arr|pos_arr|vel_arr|"
    r"_pos|_vel|headings|walkers?)$",
    re.IGNORECASE,
)

# Strong field indicators — always counted when present as assignment targets.
_FIELD_STRONG_RE = re.compile(
    r"^(_?)(field|potential|trail|curl|flow_field|flow|psi|"
    r"pressure|temperature|concentration|chemical|ca_grid|"
    r"sandpile|spins|lattice|noise_field|"
    r"divergence|vorticity|height_map|depth_map|"
    r"scalar_field|phase_field)$",
    re.IGNORECASE,
)

# Weak/ambiguous field names — only counted when assigned from a 2-D numpy
# constructor (np.zeros, np.ones, etc.) so generic loop vars like `v` (HSV
# color channel) and `grid` (rendering canvas) are not false-positively flagged.
_FIELD_WEAK_RE = re.compile(
    r"^(_?)(density|grid|angle_arr|phi|u_arr|v_arr|u|v|grad)$",
    re.IGNORECASE,
)

# Scalar state variable indicators
_SCALAR_STATE_VAR_RE = re.compile(
    r"^(energy|order|magnetization|amplitude|spread|entropy|sync_r|"
    r"cohesion|alignment|separation|diversity|pop_spread|fractal_dim|"
    r"avg_speed|mean_vel|kinetic|potential_e|total_e)$",
    re.IGNORECASE,
)

# Category → method class mapping
_CATEGORY_CLASS = {
    "compositing":  "composite",
    "fractals":     "fractal",
    "filters":      "filter",
    "ml_models":    "generator",
    "cli_tools":    "generator",
    "gpu_shaders":  "generator",
    "simulations":  "simulation",
    "patterns":     "generator",
    "math_art":     "generator",
    "codegen":      "generator",
    "simulations_cellular": "simulation",
}

# ── AST helpers ───────────────────────────────────────────────────────────────

def _kw_value(call: ast.Call, key: str) -> ast.expr | None:
    """Return the value node for a keyword arg in a Call node, or None."""
    for kw in call.keywords:
        if kw.arg == key:
            return kw.value
    return None


def _literal(node: ast.expr | None) -> Any:
    """Try ast.literal_eval; return None on failure."""
    if node is None:
        return None
    try:
        return ast.literal_eval(node)
    except Exception:
        return None


def _call_name(node: ast.Call) -> str | None:
    """Return the bare function name of a Call node."""
    f = node.func
    if isinstance(f, ast.Name):
        return f.id
    if isinstance(f, ast.Attribute):
        return f.attr
    return None


def _find_method_decorators(tree: ast.Module) -> list[tuple[ast.FunctionDef, ast.Call]]:
    """Return (func_def, decorator_call) for every @method(...) decorated function."""
    out: list[tuple[ast.FunctionDef, ast.Call]] = []
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        for dec in node.decorator_list:
            if isinstance(dec, ast.Call) and _call_name(dec) == "method":
                out.append((node, dec))  # type: ignore[arg-type]
    return out


# ── Per-function signal analysis ──────────────────────────────────────────────

class _BodyAnalyzer(ast.NodeVisitor):
    """Walk a single method function body, collecting computational signals."""

    def __init__(self) -> None:
        # ─ confirmed (actual write_* calls found) ─
        self.writes_scalars = False
        self.writes_field   = False
        self.writes_particles = False
        self.writes_mask    = False
        self.scalar_keys_written: set[str] = set()

        # ─ inferred from assignment-target names ─
        self.particle_vars: set[str] = set()
        self.field_vars:    set[str] = set()
        self.scalar_state_vars: set[str] = set()

        # ─ 2-D numpy array creations keyed by target name ─
        self.ndarray_2d_vars: set[str] = set()

        # ─ filter signal ─
        self.reads_input_image = False

        # ─ structural diagnostics ─
        self.save_count = 0
        self.has_try_except = False
        self.save_in_except = False
        self.uses_underscore_temps = False

    # ── visits ──────────────────────────────────────────────────────────────

    def visit_Call(self, node: ast.Call) -> None:
        name = _call_name(node)

        if name == "write_scalars":
            self.writes_scalars = True
            for kw in node.keywords:
                if kw.arg:
                    self.scalar_keys_written.add(kw.arg)

        elif name == "write_field":
            self.writes_field = True

        elif name == "write_particles":
            self.writes_particles = True

        elif name == "write_mask":
            self.writes_mask = True

        elif name == "save":
            self.save_count += 1

        # Detect params.get("input_image") / run_params.get("input_image")
        elif name == "get" and node.args:
            arg0 = node.args[0]
            if isinstance(arg0, ast.Constant) and arg0.value == "input_image":
                self.reads_input_image = True

        # Check string literals in *any* argument for underscore temp filenames
        for child in ast.walk(node):
            if (
                isinstance(child, ast.Constant)
                and isinstance(child.value, str)
                and child.value.startswith("_")
                and child.value.endswith((".png", ".npy", ".jpg"))
            ):
                self.uses_underscore_temps = True

        self.generic_visit(node)

    def visit_Assign(self, node: ast.Assign) -> None:
        for t in node.targets:
            self._inspect_target(t, node.value)
        self.generic_visit(node)

    def visit_AugAssign(self, node: ast.AugAssign) -> None:
        self._inspect_target(node.target, None)
        self.generic_visit(node)

    def visit_AnnAssign(self, node: ast.AnnAssign) -> None:
        self._inspect_target(node.target, node.value)
        self.generic_visit(node)

    def visit_Try(self, node: ast.Try) -> None:
        self.has_try_except = True
        # Check each except handler for save() calls
        for handler in node.handlers:
            for child in ast.walk(handler):
                if isinstance(child, ast.Call) and _call_name(child) == "save":
                    self.save_in_except = True
        self.generic_visit(node)

    @staticmethod
    def _is_2d_call(call: ast.Call) -> bool:
        """Return True if call looks like np.zeros((H,W),...) — shape arg is a Tuple."""
        fn = _call_name(call)
        _NP2D = frozenset(("zeros", "ones", "full", "empty",
                           "fft2", "ifft2", "rfft2", "irfft2"))
        if fn not in _NP2D:
            return False
        if fn in ("fft2", "ifft2", "rfft2", "irfft2"):
            return True  # always 2-D transforms
        # For zeros/ones/full/empty the first arg is the shape
        if not call.args:
            return False
        shape_arg = call.args[0]
        return isinstance(shape_arg, (ast.Tuple, ast.List))

    def _inspect_target(self, target: ast.expr, value: ast.expr | None) -> None:
        """Classify the variable name and optionally the RHS shape."""
        if isinstance(target, ast.Name):
            name = target.id
            lo   = name.lower()

            if _PARTICLE_VAR_RE.match(lo):
                self.particle_vars.add(name)

            if _FIELD_STRONG_RE.match(lo):
                self.field_vars.add(name)
                if value is not None and isinstance(value, ast.Call):
                    if self._is_2d_call(value):
                        self.ndarray_2d_vars.add(name)
            elif _FIELD_WEAK_RE.match(lo):
                # Ambiguous name: only count if explicitly a 2-D numpy array
                if value is not None and isinstance(value, ast.Call):
                    if self._is_2d_call(value):
                        self.ndarray_2d_vars.add(name)
                        # Promote to field_vars so audit notes name it
                        self.field_vars.add(name)

            if _SCALAR_STATE_VAR_RE.match(lo):
                self.scalar_state_vars.add(name)

        elif isinstance(target, (ast.Tuple, ast.List)):
            for elt in target.elts:  # type: ignore[attr-defined]
                self._inspect_target(elt, None)


def _analyze_body(func: ast.FunctionDef) -> dict[str, Any]:
    """Run _BodyAnalyzer over a function node and return a plain dict."""
    v = _BodyAnalyzer()
    for child in func.body:
        v.visit(child)
    return {
        "writes_scalars":      v.writes_scalars,
        "writes_field":        v.writes_field,
        "writes_particles":    v.writes_particles,
        "writes_mask":         v.writes_mask,
        "scalar_keys_written": sorted(v.scalar_keys_written),
        "particle_vars":       sorted(v.particle_vars),
        "field_vars":          sorted(v.field_vars),
        "scalar_state_vars":   sorted(v.scalar_state_vars),
        "ndarray_2d_vars":     sorted(v.ndarray_2d_vars),
        "reads_input_image":   v.reads_input_image,
        "save_count":          v.save_count,
        "has_try_except":      v.has_try_except,
        "save_in_except":      v.save_in_except,
        "uses_underscore_temps": v.uses_underscore_temps,
    }


# ── Source-level supplement (regex on raw text) ───────────────────────────────

_INPUT_IMAGE_RE  = re.compile(r'["\']input_image["\']')
_NP_STACK_POS_RE = re.compile(r'np\.stack\s*\(\s*\[.*(?:pos|vel|x_arr|y_arr)')
_WRITE_SCALAR_RE = re.compile(r'write_scalars\s*\(.*?(\w+)\s*=')
_TRAIL_GRID_RE   = re.compile(r'\b(trail|potential|psi|pressure|temperature|concentration|chemical|sandpile|spins|lattice)\b')
_PARTICLE_SRC_RE = re.compile(r'\b(positions|velocities|agents|boids|ants|walkers)\b')


def _source_supplement(src: str) -> dict[str, Any]:
    """Quick regex scan of function source for signals AST might miss."""
    return {
        "src_reads_input_image": bool(_INPUT_IMAGE_RE.search(src)),
        "src_np_stack_pos":      bool(_NP_STACK_POS_RE.search(src)),
        "src_trail_grid":        bool(_TRAIL_GRID_RE.search(src)),
        "src_particle_names":    bool(_PARTICLE_SRC_RE.search(src)),
        "src_scalar_keys":       _WRITE_SCALAR_RE.findall(src),
    }


# ── Classification logic ──────────────────────────────────────────────────────

def _classify(category: str, reads_input: bool, sig: dict[str, Any]) -> str:
    """Return method class: composite | filter | simulation | fractal | generator."""
    if category == "compositing":
        return "composite"
    if reads_input:
        return "filter"
    return _CATEGORY_CLASS.get(category, "generator")


# ── Gap detection ─────────────────────────────────────────────────────────────

def _detect_signals(sig: dict[str, Any], supp: dict[str, Any]) -> list[str]:
    """Build the detected_signals list (what the method appears to compute)."""
    out: list[str] = []

    # Confirmed signals via write_* calls
    if sig["writes_particles"]:
        out.append("PARTICLES")
    if sig["writes_field"]:
        out.append("FIELD")
    if sig["writes_mask"]:
        out.append("MASK")
    if sig["writes_scalars"]:
        out.append("SCALAR")

    # Inferred — only add if not already confirmed
    if "PARTICLES" not in out and (
        sig["particle_vars"] or supp["src_np_stack_pos"] or supp["src_particle_names"]
    ):
        out.append("PARTICLES?")
    if "FIELD" not in out and (
        sig["field_vars"] or sig["ndarray_2d_vars"] or supp["src_trail_grid"]
    ):
        out.append("FIELD?")

    if sig["reads_input_image"] or supp["src_reads_input_image"]:
        out.append("FILTER")

    return out


def _detect_missing(
    sig: dict[str, Any],
    supp: dict[str, Any],
    current_outputs: list[str],
    detected: list[str],
) -> tuple[list[str], list[str], list[str]]:
    """Return (missing_outputs, audit_notes, audit_warnings)."""
    missing: list[str] = []
    notes:   list[str] = []
    warnings: list[str] = []

    out_set = set(current_outputs)

    # ── Confirmed gaps: write_* called but output key absent ─────────────────
    # These become hard violations in _collect_all for field/particles/mask.
    if sig["writes_particles"] and "particles" not in out_set:
        missing.append("PARTICLES")
        notes.append("write_particles() called but 'particles' not in outputs=")

    if sig["writes_field"] and "field" not in out_set:
        missing.append("FIELD")
        notes.append("write_field() called but 'field' not in outputs=")

    if sig["writes_mask"] and "mask" not in out_set:
        missing.append("MASK")
        notes.append("write_mask() called but 'mask' not in outputs=")

    if sig["writes_scalars"]:
        for key in sig["scalar_keys_written"]:
            if key not in out_set:
                missing.append(f"SCALAR:{key}")
                notes.append(f"write_scalars({key}=…) called but '{key}' not in outputs=")

    # ── Inverse warnings: outputs= declares type but write_* never called ─────
    if "field" in out_set and not sig["writes_field"]:
        warnings.append("outputs= declares 'field' but write_field() never called")
    if "particles" in out_set and not sig["writes_particles"]:
        warnings.append("outputs= declares 'particles' but write_particles() never called")
    if "mask" in out_set and not sig["writes_mask"]:
        warnings.append("outputs= declares 'mask' but write_mask() never called")

    # ── Probable gaps: strong inferred signals ────────────────────────────────
    if "PARTICLES?" in detected and "PARTICLES" not in missing and "particles" not in out_set:
        vnames = ", ".join(sig["particle_vars"][:4])
        if vnames:
            missing.append("PARTICLES?")
            notes.append(f"particle-like vars: {vnames} — may warrant particles output")

    if "FIELD?" in detected and "FIELD" not in missing and "field" not in out_set:
        vnames = ", ".join((sig["field_vars"] + sig["ndarray_2d_vars"])[:4])
        if vnames:
            missing.append("FIELD?")
            notes.append(f"field-like vars: {vnames} — may warrant field output")

    # ── Missing luminance (most methods should expose this) ───────────────────
    if "luminance" not in out_set and not out_set:
        notes.append("no outputs= declared; defaults to image+luminance only")

    return missing, notes, warnings


# ── Per-file scan ─────────────────────────────────────────────────────────────

def _scan_file(path: Path) -> list[dict[str, Any]]:
    """Parse one Python file and return a list of method audit entries."""
    try:
        source = path.read_text(encoding="utf-8")
        tree   = ast.parse(source, filename=str(path))
    except SyntaxError as exc:
        return [{"_parse_error": str(exc), "file": str(path)}]

    methods = _find_method_decorators(tree)
    if not methods:
        return []

    source_lines = source.splitlines()
    rel_path = path.relative_to(METHODS_DIR)

    results: list[dict[str, Any]] = []

    for func, dec in methods:
        mid      = str(_literal(_kw_value(dec, "id")) or "?")
        name     = str(_literal(_kw_value(dec, "name")) or func.name)
        category = str(_literal(_kw_value(dec, "category")) or "unknown")

        # ── Description check (warning if missing or empty) ───────────────────
        description_node = _kw_value(dec, "description")
        description_val  = _literal(description_node)
        missing_description = (
            description_node is None
            or description_val is None
            or (isinstance(description_val, str) and not description_val.strip())
        )

        # Declared outputs (empty dict → no explicit outputs= → default)
        declared_outputs_dict = _literal(_kw_value(dec, "outputs")) or {}
        has_explicit_outputs  = _kw_value(dec, "outputs") is not None
        if not has_explicit_outputs:
            # Registry default
            declared_outputs_dict = {"image": "IMAGE", "luminance": "SCALAR"}
        current_outputs = list(declared_outputs_dict.keys())

        # Extract function body source text for regex supplement
        start = func.lineno - 1
        end   = func.end_lineno or len(source_lines)
        func_src = "\n".join(source_lines[start:end])

        sig  = _analyze_body(func)
        supp = _source_supplement(func_src)

        # Override reads_input_image with regex if AST missed it
        reads_input = sig["reads_input_image"] or supp["src_reads_input_image"]
        sig["reads_input_image"] = reads_input

        detected = _detect_signals(sig, supp)
        missing, notes, entry_warnings = _detect_missing(sig, supp, current_outputs, detected)

        # ── Missing description warning ───────────────────────────────────────
        if missing_description:
            entry_warnings.append("no description set")

        klass = _classify(category, reads_input, sig)

        has_fallback_png = (
            sig["save_in_except"]
            or (sig["save_count"] >= 2 and sig["has_try_except"])
        )

        results.append({
            "id":               mid,
            "name":             name,
            "file":             str(rel_path),
            "category":         category,
            "class":            klass,
            "current_outputs":  current_outputs,
            "has_explicit_outputs": has_explicit_outputs,
            "detected_signals": detected,
            "missing_outputs":  missing,
            "has_fallback_png": has_fallback_png,
            "has_try_except":   sig["has_try_except"],
            "uses_underscore_temps": sig["uses_underscore_temps"],
            "audit_notes":      notes,
            "audit_warnings":   entry_warnings,
            # ── internal detail for debugging ─────────────────────────────────
            "_detail": {
                "writes_particles":    sig["writes_particles"],
                "writes_field":        sig["writes_field"],
                "writes_mask":         sig["writes_mask"],
                "writes_scalars":      sig["writes_scalars"],
                "scalar_keys_written": sig["scalar_keys_written"],
                "particle_vars":       sig["particle_vars"],
                "field_vars":          sig["field_vars"],
                "scalar_state_vars":   sig["scalar_state_vars"],
                "ndarray_2d_vars":     sig["ndarray_2d_vars"],
                "save_count":          sig["save_count"],
                "has_try_except":      sig["has_try_except"],
            },
        })

    return results


# ── Walk all method files ─────────────────────────────────────────────────────

def _collect_all() -> tuple[list[dict[str, Any]], list[str], list[str]]:
    """Return (entries, hard_violations, warnings)."""
    py_files = sorted(METHODS_DIR.rglob("*.py"))
    # Skip __init__ files — they only do imports
    py_files = [p for p in py_files if p.name != "__init__.py"]

    all_entries: list[dict[str, Any]] = []
    for path in py_files:
        entries = _scan_file(path)
        all_entries.extend(entries)

    # Sort by integer ID where possible, else string
    def _sort_key(e: dict) -> tuple:
        mid = e.get("id", "?")
        try:
            return (0, int(mid), mid)
        except (ValueError, TypeError):
            return (1, 0, str(mid))

    all_entries.sort(key=_sort_key)

    hard_violations: list[str] = []
    warnings: list[str] = []

    # ── ID collision detection ────────────────────────────────────────────────
    id_to_files: dict[str, list[str]] = {}
    for e in all_entries:
        if "_parse_error" in e:
            continue
        mid = e.get("id", "?")
        id_to_files.setdefault(mid, []).append(e.get("file", "?"))

    for mid, files in id_to_files.items():
        if len(files) > 1:
            hard_violations.append(f"ID COLLISION: id={mid} used in {files}")

    # ── Per-entry checks ──────────────────────────────────────────────────────
    for e in all_entries:
        if "_parse_error" in e:
            continue
        label = f"{e.get('name','?')} (id={e.get('id','?')}) [{e.get('file','?')}]"

        # Hard: write_field/particles/mask called but not declared in outputs=
        for m in e.get("missing_outputs", []):
            if m in ("FIELD", "PARTICLES", "MASK"):
                write_fn = f"write_{m.lower()}"
                hard_violations.append(
                    f"SIDECAR MISMATCH: {label}: "
                    f"{write_fn}() called but '{m.lower()}' not in outputs="
                )

        # Hard: except block exists but no save() call inside it
        if e.get("has_try_except") and not e.get("has_fallback_png"):
            hard_violations.append(
                f"NO PNG FALLBACK: {label}: "
                f"except block has no save() call — node will silently produce no output on error"
            )

        # Warnings from per-entry analysis
        for w in e.get("audit_warnings", []):
            warnings.append(f"{label}: {w}")

    return all_entries, hard_violations, warnings


# ── Report generation ─────────────────────────────────────────────────────────

def _write_json(entries: list[dict[str, Any]], path: Path) -> None:
    # Strip internal _detail from public output
    public = []
    for e in entries:
        if "_parse_error" in e:
            continue
        pub = {k: v for k, v in e.items() if not k.startswith("_")}
        public.append(pub)
    path.write_text(json.dumps(public, indent=2, ensure_ascii=False))
    print(f"  ✓ {path.relative_to(REPO)} ({len(public)} entries)")


_CONFIRMED_TYPES  = {"PARTICLES", "FIELD", "MASK"}   # without ? suffix
_CONFIRMED_PREFIX = "SCALAR:"


def _missing_score(entry: dict) -> int:
    """Score = confirmed missing count × 3 + inferred missing count."""
    score = 0
    for m in entry.get("missing_outputs", []):
        if m in _CONFIRMED_TYPES or m.startswith(_CONFIRMED_PREFIX):
            score += 3
        else:
            score += 1
    return score


def _write_markdown(entries: list[dict[str, Any]], path: Path) -> None:
    valid = [e for e in entries if "_parse_error" not in e]

    # Sort by missing score descending, then by id
    ranked = sorted(valid, key=lambda e: (-_missing_score(e), e.get("id", "?")))

    lines = [
        "# Method Audit Report",
        "",
        f"**{len(valid)} methods scanned** across all category packages.",
        "",
        "Sorted by gap severity (confirmed missing × 3 + inferred missing).",
        "",
        "| ID | Name | File | Class | Current Outputs | Signals | Missing |",
        "|---|---|---|---|---|---|---|",
    ]

    for e in ranked:
        mid      = e["id"]
        name     = e["name"]
        file_    = e["file"].replace("\\", "/")
        cls      = e["class"]
        curr_out = ", ".join(e["current_outputs"])
        signals  = ", ".join(e["detected_signals"]) or "—"
        missing  = ", ".join(e["missing_outputs"])  or "—"
        lines.append(
            f"| `{mid}` | {name} | `{file_}` | {cls} "
            f"| {curr_out} | {signals} | **{missing}** |"
        )

    lines += [
        "",
        "## Legend",
        "",
        "- **PARTICLES** — `write_particles()` called but `particles` not in `outputs=` (confirmed)",
        "- **FIELD** — `write_field()` called but `field` not in `outputs=` (confirmed)",
        "- **MASK** — `write_mask()` called but `mask` not in `outputs=` (confirmed)",
        "- **SCALAR:key** — `write_scalars(key=…)` called but key not in `outputs=` (confirmed)",
        "- **PARTICLES?** — particle-like variable names found in assignments (inferred)",
        "- **FIELD?** — field/grid/trail-like variable names found in assignments (inferred)",
        "- **—** — no gaps detected",
    ]

    path.write_text("\n".join(lines) + "\n")
    print(f"  ✓ {path.relative_to(REPO)}")


# ── Summary to stdout ─────────────────────────────────────────────────────────

def _print_top(entries: list[dict[str, Any]], n: int = 20) -> None:
    valid  = [e for e in entries if "_parse_error" not in e]
    ranked = sorted(valid, key=lambda e: (-_missing_score(e), e.get("id", "?")))

    # Summary stats
    confirmed_gaps = [e for e in valid if any(
        m in _CONFIRMED_TYPES or m.startswith(_CONFIRMED_PREFIX)
        for m in e.get("missing_outputs", [])
    )]
    inferred_gaps  = [e for e in valid if any(
        m.endswith("?") for m in e.get("missing_outputs", [])
    ) and e not in confirmed_gaps]
    clean          = [e for e in valid if not e.get("missing_outputs")]

    no_fallback    = [e for e in valid if e.get("has_try_except") and not e.get("has_fallback_png")]
    no_desc        = [e for e in valid if any(
        "no description set" in w for w in e.get("audit_warnings", [])
    )]

    print(f"\n{'─'*70}")
    print(f"  AUDIT SUMMARY  ({len(valid)} methods)")
    print(f"{'─'*70}")
    print(f"  ✗  {len(confirmed_gaps):3d}  confirmed sidecar gaps  (write_* called but not in outputs=)")
    print(f"  ~  {len(inferred_gaps):3d}  inferred gaps           (signal detected, not declared)")
    print(f"  ✓  {len(clean):3d}  clean                   (no gaps detected)")
    print(f"  ⚠  {len(no_fallback):3d}  no PNG fallback         (except block, no save() inside)")
    print(f"  ⚠  {len(no_desc):3d}  no description          (description missing or empty)")
    print()

    # Breakdown by type
    p_conf = sum(1 for e in valid if "PARTICLES" in e.get("missing_outputs",[]))
    f_conf = sum(1 for e in valid if "FIELD" in e.get("missing_outputs",[]))
    m_conf = sum(1 for e in valid if "MASK" in e.get("missing_outputs",[]))
    s_conf = sum(1 for e in valid if any(m.startswith("SCALAR:") for m in e.get("missing_outputs",[])))
    p_inf  = sum(1 for e in valid if "PARTICLES?" in e.get("missing_outputs",[]))
    f_inf  = sum(1 for e in valid if "FIELD?" in e.get("missing_outputs",[]))
    print(f"  Confirmed: PARTICLES={p_conf}, FIELD={f_conf}, MASK={m_conf}, SCALAR={s_conf}")
    print(f"  Inferred:  PARTICLES?={p_inf}, FIELD?={f_inf}")
    print()

    print(f"  TOP {n} METHODS WITH MOST MISSING OUTPUTS")
    print(f"{'─'*70}")
    hdr = f"  {'ID':>5}  {'Class':12}  {'Missing':30}  Name"
    print(hdr)
    print(f"  {'─'*5}  {'─'*12}  {'─'*30}  {'─'*28}")

    for e in ranked[:n]:
        mid     = e["id"].rjust(5)
        cls     = e["class"].ljust(12)
        missing = (", ".join(e["missing_outputs"]) or "—").ljust(30)
        name    = e["name"]
        print(f"  {mid}  {cls}  {missing}  {name}")

    print()


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> int:
    fail_on_violations = "--fail-on-violations" in sys.argv

    TOOLS_DIR.mkdir(parents=True, exist_ok=True)

    print("Scanning method files …")
    entries, hard_violations, warnings = _collect_all()
    print(f"  Found {len(entries)} method entries across "
          f"{len(set(e.get('file','') for e in entries))} files")

    json_path = TOOLS_DIR / "audit_report.json"
    md_path   = TOOLS_DIR / "audit_report.md"

    _write_json(entries, json_path)
    _write_markdown(entries, md_path)
    _print_top(entries, n=20)

    # ── Warnings ──────────────────────────────────────────────────────────────
    if warnings:
        print(f"{'─'*70}")
        print(f"  WARNINGS  ({len(warnings)} total)")
        print(f"{'─'*70}")
        for w in warnings:
            print(f"  ⚠  {w}")
        print()

    # ── Hard violations ───────────────────────────────────────────────────────
    if hard_violations:
        print(f"{'─'*70}")
        print("  HARD VIOLATIONS")
        print(f"{'─'*70}")
        for v in hard_violations:
            print(f"  ✗  {v}")
        print()
        print(f"{len(hard_violations)} violation(s) found — {len(warnings)} warning(s)")
        if fail_on_violations:
            return 1
        return 0
    else:
        if warnings:
            print(f"No hard violations. {len(warnings)} warning(s) above.")
        else:
            print("All methods clean.")
        return 0


if __name__ == "__main__":
    sys.exit(main())
