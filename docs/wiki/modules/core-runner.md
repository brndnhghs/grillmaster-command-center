# Module: `core/runner.py`

## Purpose
Parallel/sequential runner with progress tracking, timing, and caching. CLI-only — not used by `server.py` or `graph.py`.

## Responsibilities
- Run methods sequentially (`run_sequential`)
- Run methods in parallel via thread pool (`run_parallel`)
- Cache integration
- RNG lock for thread-safe seeding

## Functions
- `run_sequential(metas, out_dir, seed, force, params, progress_cb) -> list[tuple]`
- `run_parallel(metas, out_dir, seed, max_workers, force, params, progress_cb) -> list[tuple]`

## Notes
- CLI-only — `server.py` bypasses this entirely
- `_rng_lock` serialises `seed_all()` calls in parallel mode
- Cache at `~/.cache/image-pipeline/`