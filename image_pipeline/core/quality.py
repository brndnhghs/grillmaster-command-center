"""
Auto-quality detection: check generated images for issues.

Flags outputs that are:
  - Too small (under size / filesize thresholds)
  - Solid color (too few unique colors)
  - Mostly empty (too few non-zero pixels)
"""
from __future__ import annotations
import math
from pathlib import Path

import numpy as np
from PIL import Image


class QualityReport:
    def __init__(self, path: Path):
        self.path = path
        self.size_kb: int = 0
        self.dims: tuple[int, int] = (0, 0)
        self.unique_colors: int = 0
        self.empty_pct: float = 0.0
        self.mean: float = 0.0
        self.std: float = 0.0
        self.passed: bool = True
        self.issues: list[str] = []

    def __repr__(self) -> str:
        status = "✅" if self.passed else "⚠️"
        issues = f" — {'; '.join(self.issues)}" if self.issues else ""
        return f"  {status} {self.path.name}  ({self.size_kb}KB, {self.unique_colors} colors, {self.empty_pct:.0%} empty){issues}"


def check(path: Path) -> QualityReport:
    """Analyze a generated image and flag issues."""
    report = QualityReport(path)

    if not path.exists():
        report.passed = False
        report.issues.append("missing")
        return report

    report.size_kb = path.stat().st_size // 1024

    try:
        img = Image.open(str(path)).convert("RGB")
    except Exception:
        report.passed = False
        report.issues.append("corrupt")
        return report

    report.dims = img.size

    # Sample pixels for analysis (avoid loading full image into RAM)
    w, h = img.size
    step = max(1, min(w, h) // 64)
    arr = np.array(img)[::step, ::step]
    report.unique_colors = len(np.unique(arr.reshape(-1, arr.shape[-1]), axis=0))

    gray = np.mean(arr.astype(np.float32), axis=2) / 255.0
    report.mean = float(np.mean(gray))
    report.std = float(np.std(gray))
    report.empty_pct = float(np.mean(gray < 0.05))

    # Thresholds
    if report.size_kb < 1:
        report.issues.append(f"tiny file ({report.size_kb}KB)")
    if report.unique_colors < 4:
        report.issues.append(f"only {report.unique_colors} unique colors")
    if report.empty_pct > 0.95:
        report.issues.append(f"{report.empty_pct:.0%} empty pixels")

    report.passed = len(report.issues) == 0
    return report


def verify_batch(paths: list[Path]) -> list[QualityReport]:
    """Check all paths and return reports."""
    return [check(p) for p in paths]


def print_summary(reports: list[QualityReport]):
    """Pretty-print a batch quality report."""
    passed = sum(1 for r in reports if r.passed)
    failed = sum(1 for r in reports if not r.passed)
    print(f"\n{'─' * 50}")
    print(f"Quality check: {passed} passed, {failed} flagged")
    for r in reports:
        print(repr(r))
    if failed:
        print(f"\n⚠️  {failed} images have potential quality issues.")
    print(f"{'─' * 50}\n")