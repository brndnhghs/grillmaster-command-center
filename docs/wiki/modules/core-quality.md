# Module: `core/quality.py`

## Purpose
Auto-quality detection — checks generated images for issues.

## Flags
- Too small (under size / filesize thresholds)
- Missing file
- Corrupt image
- Too few unique colors (< 4)
- Mostly empty (> 95% empty pixels)

## Public Interfaces
- `QualityReport` — per-image quality report
- `check(path) -> QualityReport` — analyze a single image
- `verify_batch(paths) -> list[QualityReport]` — analyze multiple images
- `print_summary(reports)` — pretty-print summary

## Dependencies
- `numpy`, PIL

## Consumers
- `pipeline.py` — CLI `--quality` flag
- Not wired into `server.py`