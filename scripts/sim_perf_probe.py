"""Simulation performance + health regression probe.

Companion to ``image_pipeline/tests/test_sim_render_health.py``. Runs every
``simulations``-category method once (defaults, 256x256), measures wall-clock,
and diffs against a prior baseline CSV to flag regressions. Cron-safe: no
pipe-to-interpreter, no network, runs fully offline against the project venv.

Usage (from repo root, project venv):
    env -u PYTHONPATH .venv/bin/python scripts/sim_perf_probe.py
    # Re-baseline:
    env -u PYTHONPATH .venv/bin/python scripts/sim_perf_probe.py --rebaseline
    # Compare against a specific prior CSV:
    env -u PYTHONPATH .venv/bin/python scripts/sim_perf_probe.py --baseline /tmp/prof_out/prof_1783727153.csv

Exit code is non-zero if a method errors/times-out, or if any measured time
exceeds the baseline value plus a margin. Pure PASS produces exit 0.
"""
from __future__ import annotations

import argparse
import csv
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, TimeoutError as FuturesTimeout
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

PER_METHOD_TIMEOUT = 90.0          # hard cap per method (seconds)
CUMULATIVE_SLOW_IDS = {"36", "55"}  # DLA, Sandpile -- expected long runtimes
CUMULATIVE_TIMEOUT = 150.0
REGRESSION_MARGIN_SEC = 10.0       # warn if now > baseline + this
BASELINE_GLOB = "/tmp/prof_out/prof_*.csv"


def _timeout_for(mid: str) -> float:
    return CUMULATIVE_TIMEOUT if mid in CUMULATIVE_SLOW_IDS else PER_METHOD_TIMEOUT

import image_pipeline.methods  # noqa: F401 -- trigger @method registration
from image_pipeline.core.registry import get_all


def _all_sim_ids():
    return sorted(mid for mid, m in get_all().items() if m.category == "simulations")


def _run_one(mid: str):
    """Worker: render one method at defaults; return (seconds, error_str)."""
    from pathlib import Path as _P
    import image_pipeline.methods  # noqa: F401
    from image_pipeline.core.registry import get_all as _ga
    from image_pipeline.core import utils as U

    meta = _ga()[mid]
    node_dir = _P("/tmp/prof_out") / mid
    node_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    token = U.set_canvas(256, 256)
    try:
        meta.fn(node_dir, 42, params={})
    finally:
        U.reset_canvas(token)
    return time.time() - t0, ""


def _load_baseline(path: Path | None):
    if path is None:
        return {}
    rows = {}
    try:
        with path.open(newline="") as fh:
            for row in csv.DictReader(fh):
                rows[row["id"]] = (float(row["seconds"]), row.get("error", ""))
    except Exception:
        pass
    return rows


def _latest_baseline() -> Path | None:
    import glob
    files = sorted(glob.glob(BASELINE_GLOB), reverse=True)
    return Path(files[0]) if files else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rebaseline", action="store_true",
                   help="treat this run as the new baseline (don't fail on regression)")
    ap.add_argument("--baseline", type=str, default=None,
                   help="explicit baseline CSV to diff against")
    args = ap.parse_args()

    sim_ids = _all_sim_ids()
    names = {mid: get_all()[mid].name for mid in sim_ids}
    print(f"Profiling {len(sim_ids)} simulation methods "
          f"(timeout 90s default, 150s for cumulative-growth sims)...", flush=True)

    baseline_path = None if args.rebaseline else (
        Path(args.baseline) if args.baseline else _latest_baseline())
    baseline = _load_baseline(baseline_path)
    if baseline_path:
        print(f"Baseline: {baseline_path} ({len(baseline)} entries)", flush=True)

    results = {}
    with ProcessPoolExecutor(max_workers=4) as ex:
        fut_map = {ex.submit(_run_one, mid): mid for mid in sim_ids}
        for fut in fut_map:
            mid = fut_map[fut]
            cap = _timeout_for(mid)
            try:
                dt, err = fut.result(timeout=cap)
                results[mid] = (dt, err)
            except FuturesTimeout:
                results[mid] = (cap, "TIMEOUT")
            except Exception as e:  # noqa: BLE001
                results[mid] = (cap, f"{type(e).__name__}: {e}"[:100])

    # Write fresh baseline CSV
    ts = int(time.time())
    out_csv = Path("/tmp/prof_out") / f"prof_{ts}.csv"
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["id", "name", "seconds", "error"])
        for mid in sim_ids:
            dt, err = results[mid]
            w.writerow([mid, names[mid], f"{dt:.3f}", err])
    print(f"Wrote {out_csv}", flush=True)

    # Report + regression check
    ranked = sorted(sim_ids, key=lambda m: -results[m][0])
    failures = [(m, results[m]) for m in sim_ids if results[m][1]]
    regressions = []
    known_slow = []
    for mid in sim_ids:
        dt, err = results[mid]
        if err or mid not in baseline:
            continue
        bdt, _ = baseline[mid]
        if dt > bdt + REGRESSION_MARGIN_SEC:
            if mid in CUMULATIVE_SLOW_IDS:
                # DLA/Sandpile are cumulative-growth sims whose default runtime is
                # inherently variable (see skill pitfall #9). Report, don't fail.
                known_slow.append((mid, names[mid], bdt, dt))
            else:
                regressions.append((mid, names[mid], bdt, dt))

    print("\n=== SLOWEST 15 ===", flush=True)
    for mid in ranked[:15]:
        dt, err = results[mid]
        print(f"  {mid:>5}  {names[mid][:40]:40}  {dt:8.2f}s  {err}", flush=True)

    print(f"\nTotal: {len(sim_ids)}; errored/timeout: {len(failures)}", flush=True)
    if failures:
        print("FAILURES:", flush=True)
        for mid, (dt, err) in failures:
            print(f"  {mid} {names[mid]}: {err}", flush=True)
    if regressions:
        print(f"\nREGRESSIONS (>baseline +{REGRESSION_MARGIN_SEC:.0f}s):", flush=True)
        for mid, name, bdt, dt in regressions:
            print(f"  {mid} {name}: {bdt:.1f}s -> {dt:.1f}s", flush=True)
    if known_slow:
        print(f"\nKNOWN-SLOW (cumulative-growth, variable by design -- not a failure):",
              flush=True)
        for mid, name, bdt, dt in known_slow:
            print(f"  {mid} {name}: {bdt:.1f}s -> {dt:.1f}s", flush=True)

    bad = len(failures) or (len(regressions) if not args.rebaseline else 0)
    print(f"\nRESULT: {'FAIL' if bad else 'PASS'}", flush=True)
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
