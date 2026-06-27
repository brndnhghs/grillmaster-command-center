"""Node Tester — run every registered method in isolation and report results.

Architecture:
  - Each method is tested with default params (no graph wiring).
  - Each method is also tested with edge-case param values (min, max, extremes).
  - Results are structured: pass/fail, error trace, param values used, output stats.
  - Supports batch-apply of Node Doctor fixes to all failing methods.
"""

from __future__ import annotations
import io
import json
import sys
import time
import traceback
from pathlib import Path
from typing import Any

import numpy as np

from . import registry

# ── Test result types ──────────────────────────────────────────────────

class TestResult:
    """Result of testing a single method with a single param set."""

    def __init__(
        self,
        method_id: str,
        method_name: str,
        category: str,
        passed: bool,
        error: str = "",
        error_trace: str = "",
        duration_ms: float = 0.0,
        param_set: str = "default",
        param_values: dict | None = None,
        output_stats: dict | None = None,
    ):
        self.method_id = method_id
        self.method_name = method_name
        self.category = category
        self.passed = passed
        self.error = error
        self.error_trace = error_trace
        self.duration_ms = duration_ms
        self.param_set = param_set
        self.param_values = param_values or {}
        self.output_stats = output_stats or {}

    def to_dict(self) -> dict:
        return {
            "method_id": self.method_id,
            "method_name": self.method_name,
            "category": self.category,
            "passed": self.passed,
            "error": self.error,
            "error_trace": self.error_trace,
            "duration_ms": round(self.duration_ms, 1),
            "param_set": self.param_set,
            "param_values": self.param_values,
            "output_stats": self.output_stats,
        }


class TestReport:
    """Aggregate report for a full test run."""

    def __init__(self):
        self.results: list[TestResult] = []
        self.started_at: float = 0.0
        self.finished_at: float = 0.0
        self.total: int = 0
        self.passed: int = 0
        self.failed: int = 0

    def add(self, r: TestResult):
        self.results.append(r)
        self.total += 1
        if r.passed:
            self.passed += 1
        else:
            self.failed += 1

    def to_dict(self) -> dict:
        return {
            "total": self.total,
            "passed": self.passed,
            "failed": self.failed,
            "duration_s": round(self.finished_at - self.started_at, 1),
            "results": [r.to_dict() for r in self.results],
        }

    def failing_methods(self) -> list[TestResult]:
        return [r for r in self.results if not r.passed]


# ── Test runner ────────────────────────────────────────────────────────

# Default image size for isolated tests
_TEST_H = 256
_TEST_W = 256


def _default_params(meta: registry.MethodMeta) -> dict:
    """Build a default param dict from a method's param spec."""
    params: dict[str, Any] = {}
    for pname, pspec in (meta.params or {}).items():
        if isinstance(pspec, dict):
            default = pspec.get("default", 0.0)
            params[pname] = default
        else:
            params[pname] = 0.0
    params["frame"] = 0
    params["frame_seed"] = 42
    params["time"] = 0.0
    return params


def _edge_case_params(meta: registry.MethodMeta) -> list[tuple[str, dict]]:
    """Generate edge-case param sets for a method.

    Returns a list of (label, params_dict) tuples.
    """
    sets: list[tuple[str, dict]] = []
    base = _default_params(meta)

    # Edge case: all numeric params at min
    min_set = dict(base)
    for pname, pspec in (meta.params or {}).items():
        if isinstance(pspec, dict) and isinstance(pspec.get("min"), (int, float)):
            min_set[pname] = pspec["min"]
    sets.append(("min", min_set))

    # Edge case: all numeric params at max
    max_set = dict(base)
    for pname, pspec in (meta.params or {}).items():
        if isinstance(pspec, dict) and isinstance(pspec.get("max"), (int, float)):
            max_set[pname] = pspec["max"]
    sets.append(("max", max_set))

    # Edge case: extreme seed
    extreme_seed = dict(base)
    extreme_seed["frame_seed"] = 999999
    sets.append(("extreme_seed", extreme_seed))

    return sets


def _compute_output_stats(node_dir: Path) -> dict:
    """Read output PNG and compute basic stats."""
    pngs = sorted(p for p in node_dir.glob("*.png") if not p.name.startswith("_"))
    if not pngs:
        return {"has_output": False}
    try:
        from PIL import Image
        img = Image.open(str(pngs[-1])).convert("RGB")
        arr = np.array(img, dtype=np.float32) / 255.0
        return {
            "has_output": True,
            "width": arr.shape[1],
            "height": arr.shape[0],
            "mean_luminance": round(float(np.mean(arr)), 4),
            "std_luminance": round(float(np.std(arr)), 4),
            "min_luminance": round(float(np.min(arr)), 4),
            "max_luminance": round(float(np.max(arr)), 4),
            "file_size_kb": round(pngs[-1].stat().st_size / 1024, 1),
        }
    except Exception:
        return {"has_output": False, "error": "Could not read output"}


def test_method(
    meta: registry.MethodMeta,
    param_set_label: str,
    param_values: dict,
    out_dir: Path,
    timeout: int = 60,
) -> TestResult:
    """Run a single method with given params and return a TestResult."""
    node_dir = out_dir / f"{meta.id}_{param_set_label}"
    node_dir.mkdir(parents=True, exist_ok=True)

    start = time.time()
    try:
        meta.fn(node_dir, param_values.get("frame_seed", 42), params=param_values)
        duration = (time.time() - start) * 1000
        stats = _compute_output_stats(node_dir)
        return TestResult(
            method_id=meta.id,
            method_name=meta.name,
            category=meta.category,
            passed=True,
            duration_ms=duration,
            param_set=param_set_label,
            param_values=param_values,
            output_stats=stats,
        )
    except Exception as exc:
        duration = (time.time() - start) * 1000
        tb = traceback.format_exc(limit=10)
        return TestResult(
            method_id=meta.id,
            method_name=meta.name,
            category=meta.category,
            passed=False,
            error=str(exc)[:500],
            error_trace=tb,
            duration_ms=duration,
            param_set=param_set_label,
            param_values=param_values,
        )


def run_all_tests(
    out_dir: Path,
    method_ids: list[str] | None = None,
    include_edge_cases: bool = True,
    timeout: int = 60,
    progress_callback=None,
) -> TestReport:
    """Test all registered methods (or a subset) and return a report.

    Args:
        out_dir: Directory for test output artifacts.
        method_ids: If provided, only test these method IDs.
        include_edge_cases: Also test min/max/extreme param sets.
        timeout: Per-method timeout in seconds.
        progress_callback: Optional fn(method_id, method_name, status, detail)
    """
    report = TestReport()
    report.started_at = time.time()

    all_meta = registry.get_all()
    ids_to_test = method_ids or sorted(all_meta.keys())

    for mid in ids_to_test:
        meta = all_meta.get(mid)
        if meta is None:
            continue

        # Default params test
        if progress_callback:
            progress_callback(mid, meta.name, "running", "default")
        default_params = _default_params(meta)
        result = test_method(meta, "default", default_params, out_dir, timeout)
        report.add(result)
        if progress_callback:
            progress_callback(mid, meta.name, "done" if result.passed else "failed", "default")

        # Edge case tests
        if include_edge_cases:
            for label, edge_params in _edge_case_params(meta):
                if progress_callback:
                    progress_callback(mid, meta.name, "running", label)
                result = test_method(meta, label, edge_params, out_dir, timeout)
                report.add(result)
                if progress_callback:
                    progress_callback(mid, meta.name, "done" if result.passed else "failed", label)

    report.finished_at = time.time()
    return report


def batch_apply_fixes(
    fixes: list[dict],
    out_dir: Path,
) -> dict:
    """Apply a batch of Node Doctor fixes to failing methods.

    Each fix: {method_id, source_code, backup_id?}
    Returns {applied: int, failed: list[{method_id, error}]}
    """
    applied = 0
    failures = []

    for fix in fixes:
        method_id = fix.get("method_id", "")
        new_source = fix.get("source", "")
        if not method_id or not new_source:
            failures.append({"method_id": method_id, "error": "Missing method_id or source"})
            continue

        meta = registry.get_meta(method_id)
        if not meta or not meta.module:
            failures.append({"method_id": method_id, "error": "Method not found in registry"})
            continue

        mod = sys.modules.get(meta.module)
        if not mod or not getattr(mod, "__file__", None):
            failures.append({"method_id": method_id, "error": "Module file not found"})
            continue

        path = Path(mod.__file__)
        if not path.exists():
            failures.append({"method_id": method_id, "error": "Source file not found"})
            continue

        # Backup
        import uuid
        backup_id = fix.get("backup_id") or uuid.uuid4().hex[:8]
        backup_path = path.with_suffix(f".nd-bak-{backup_id}.py")
        import shutil
        shutil.copy2(str(path), str(backup_path))

        # Write new source
        path.write_text(new_source)
        applied += 1

    return {"applied": applied, "failed": failures}
