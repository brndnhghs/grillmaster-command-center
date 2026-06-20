"""
Image Pipeline v2 — CLI entry point.

Usage:
  python -m image_pipeline.pipeline --all
  python -m image_pipeline.pipeline --group fractals --parallel 4
  python -m image_pipeline.pipeline --group fast --except ml --force
  python -m image_pipeline.pipeline --preset gangstalking
  python -m image_pipeline.pipeline --list
  python -m image_pipeline.pipeline --methods 07,21,49 --composite overlay
"""
from __future__ import annotations
import argparse
import sys
import time
from pathlib import Path

# Ensure the parent is on sys.path for direct python invocations
_parent = Path(__file__).resolve().parent.parent
if str(_parent) not in sys.path:
    sys.path.insert(0, str(_parent))

from image_pipeline.core import registry
from image_pipeline.core import runner as runner_mod
from image_pipeline.core import cache as cache_mod
from image_pipeline.core import quality as quality_mod

# Must import method modules to register them
import image_pipeline.methods  # noqa: F401

DEFAULT_OUT = Path.home() / "Documents" / "GitHub" / "grillmaster-command-center"

# Lazy imports for heavy modules (compositing loads cv2)
BLEND_MODES = [
    "normal","dissolve","multiply","color-burn","linear-burn","darken-only","darker-color",
    "screen","color-dodge","linear-dodge","lighten-only","lighter-color",
    "overlay","hard-light","soft-light-pegtop","soft-light-w3c","vivid-light","linear-light",
    "pin-light","hard-mix","hard-overlay",
    "difference","exclusion","subtract","divide",
    "grain-extract","grain-merge","negation","phoenix","reflect","glow","freeze","heat","stamp",
    "arithmetic","geometric-mean","harmonic-mean","rms","signed-difference","soft-subtract","cross-fade",
    "source-over","destination-over","source-in","destination-in","source-out","destination-out",
    "source-atop","destination-atop","xor","lighter","darker",
    "luminosity","hue","color","heatmap","ascii-quantize","edge-burn","blend","dif","max","diff","min",
    "hstack","vstack","grid","mosaic",
]


def _composite_images(paths, mode, out, cols=3):
    """Lazy import composite_images."""
    from image_pipeline.core.compositing import composite_images as _ci
    _ci(paths, mode, out, cols)

# ── Preset loader ─────────────────────────────────────────────────────


def load_preset(name: str) -> dict:
    """Load a preset YAML file from config/presets/."""
    preset_dir = Path(__file__).resolve().parent / "config" / "presets"
    candidates = list(preset_dir.glob(f"{name}.*")) + list(preset_dir.glob(f"{name}"))
    if candidates:
        import yaml
        return yaml.safe_load(candidates[0].read_text())
    # Fallback: check if preset name is a method group or list
    keys = registry.resolve_keys(name)
    if keys:
        return {"methods": ",".join(keys), "composite": None}
    print(f"  ✗ Preset '{name}' not found.")
    sys.exit(1)


# ── Built-in groups ───────────────────────────────────────────────────

registry.BUILTIN_GROUPS["fast"] = [
    k for k, m in registry.get_all().items()
    if "slow" not in m.tags and "ml" not in m.tags and "gpu" not in m.tags
]
registry.BUILTIN_GROUPS["slow"] = [
    k for k, m in registry.get_all().items() if "slow" in m.tags
]
registry.BUILTIN_GROUPS["ml"] = [
    k for k, m in registry.get_all().items() if "ml" in m.tags or "gpu" in m.tags
]


def print_header(kwargs: dict):
    """Print a nice start banner."""
    print(f"\n{'═' * 60}")
    print(f"  Image Pipeline v2")
    print(f"  Seed: {kwargs.get('seed', 420691337)}")
    print(f"  Output: {kwargs.get('out_dir', DEFAULT_OUT)}")
    print(f"{'─' * 60}")


def print_footer(results: list, elapsed: float, run_all: bool, run_list: bool):
    """Print end summary."""
    if run_list:
        return
    if not results:
        return
    total = len(results)
    cached = sum(1 for _, _, from_cache in results if from_cache)
    generated = total - cached
    avg_time = (
        sum(e for _, e, from_cache in results if not from_cache) / max(1, generated)
        if generated > 0
        else 0
    )
    print(f"\n{'─' * 60}")
    print(f"  Done: {total} methods ({generated} generated, {cached} cached)")
    print(f"  Time: {elapsed:.1f}s total ({avg_time:.1f}s avg per generated)")
    print(f"{'═' * 60}\n")


# ── CLI Argument Parser ───────────────────────────────────────────────


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Image Pipeline v2 — 80 methods + compositing",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    p.add_argument("--all", action="store_true", help="Run all registered methods")
    p.add_argument("--group", type=str, help="Run a named group (e.g., fractals, fast, ml)")
    p.add_argument("--methods", type=str, help="Comma-separated method IDs or names")
    p.add_argument("--except", dest="except_", type=str, default="",
                   help="Comma-separated groups/IDs to exclude (e.g., slow,ml)")
    p.add_argument("--list", action="store_true", help="List all methods by category")
    p.add_argument("--preset", type=str, help="Load a named preset from config/presets/")
    p.add_argument("--output-dir", type=str, default=str(DEFAULT_OUT), help="Output directory")
    p.add_argument("--seed", type=int, default=420691337, help="Random seed")
    p.add_argument("--force", action="store_true", help="Regenerate even if cached")

    p.add_argument("--parallel", type=int, default=0,
                   help="Parallel worker count (0=sequential, 2+ = parallel)")
    p.add_argument("--no-cache", action="store_true", help="Disable caching")
    p.add_argument("--quality", action="store_true", help="Run quality check after generation")

    p.add_argument("--composite", type=str, choices=BLEND_MODES + ["hstack", "vstack", "grid", "mosaic"],
                   help="Composite mode for selected images")
    p.add_argument("--output", type=str, help="Output filename for composite")
    p.add_argument("--cols", type=int, default=3, help="Grid columns for composite")
    p.add_argument("--animate", type=str, default="",
                   help="Animate a method: '07' or '07,34,50'. Uses natural loops or tween.")
    p.add_argument("--anim-duration", type=float, default=4.0, help="Animation duration in seconds (default 4.0)")
    p.add_argument("--anim-fps", type=int, default=24, help="Animation FPS (default 24)")
    p.add_argument("--demo", action="store_true", help="Annotate output images with method params + ranges")
    p.add_argument("--input", type=str, default="",
                    help="Path to input image for filter/effect methods (applied instead of generating source material)")
    p.add_argument("--filter", type=str, default="",
                    help="Post-process filter: effect name or JSON. e.g. 'oil', '{\"effect\":\"colormap\",\"colormap\":\"ocean\"}'")
    p.add_argument("--params", type=str, default="",
                    help="JSON string of param overrides, e.g. '{\"boids\":80,\"frames\":400}'")

    return p


def main():
    p = build_parser()
    args = p.parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Build params dict from --input and --params ──
    user_params: dict = {}
    if args.input:
        user_params["input_image"] = str(Path(args.input).resolve())
    if args.params:
        import json
        try:
            user_params.update(json.loads(args.params))
        except json.JSONDecodeError as e:
            print(f"  ✗ Invalid --params JSON: {e}")
            return

    # ── Animation mode (handled before method resolution) ──
    if args.animate:
        anim_keys = registry.resolve_keys(args.animate)
        anim_metas = [registry.get_meta(k) for k in anim_keys if registry.get_meta(k)]
        if not anim_metas:
            print("  ✗ No methods matched for animation.")
            return
        from image_pipeline.core.animation import animate_method
        for meta in anim_metas:
            print(f"\n  Animating #{meta.id} {meta.name} ({args.anim_duration}s @ {args.anim_fps}fps)")
            animate_method(meta, out_dir, args.seed, fps=args.anim_fps, duration=args.anim_duration, user_params=user_params)
        return

    # ── List mode ──
    if args.list:
        registry.print_help()
        return

    # ── Resolve methods ──
    keys: list[str] = []

    if args.preset:
        preset = load_preset(args.preset)
        methods_spec = preset.get("methods", "")
        if methods_spec:
            keys = registry.resolve_keys(methods_spec)
        if preset.get("composite") and not args.composite:
            args.composite = preset["composite"]

    elif args.all:
        keys = registry.get_ids()

    elif args.group:
        keys = registry.resolve_keys(args.group)

    elif args.methods:
        keys = registry.resolve_keys(args.methods)

    else:
        # Default: show help if nothing specified
        if not args.composite and not args.animate:
            p.print_help()
            return

    # ── Apply --except ──
    if args.except_:
        exclude_keys = registry.resolve_keys(args.except_)
        keys = [k for k in keys if k not in exclude_keys]

    # ── Show summary ──
    metas = [registry.get_meta(k) for k in keys if registry.get_meta(k)]
    if not metas:
        print("  ✗ No methods matched.")
        return

    print_header(vars(args))

    # ── Generation ──
    results = []
    elapsed_total = 0.0

    if metas:
        start = time.time()

        if args.parallel > 1:
            results = runner_mod.run_parallel(
                metas, out_dir, args.seed,
                max_workers=args.parallel,
                force=args.force or args.no_cache,
                params=user_params,
                progress_cb=runner_mod.default_progress,
            )
        else:
            results = runner_mod.run_sequential(
                metas, out_dir, args.seed,
                force=args.force or args.no_cache,
                params=user_params,
                progress_cb=runner_mod.default_progress,
            )

        elapsed_total = time.time() - start

        # ── Quality check ──
        if args.quality and results:
            paths = [out_dir / meta.filename() for meta, _, _ in results if meta]
            reports = quality_mod.verify_batch(paths)
            quality_mod.print_summary(reports)

        print_footer(results, elapsed_total, args.all, args.list)

    # ── Demo annotation ──
    if args.demo and results:
        from image_pipeline.core.annotator import annotate_batch
        demo_ids = [meta.id for meta, _, _ in results if meta]
        print(f"\n{'─'*60}")
        print("  Annotating outputs with method parameters...")
        annotate_batch(out_dir, demo_ids)

    # ── Post-process filter ──
    if args.filter and results:
        import json
        try:
            filter_spec = json.loads(args.filter) if args.filter.startswith("{") else args.filter
        except json.JSONDecodeError:
            filter_spec = args.filter
        from image_pipeline.core.postprocess import apply_filter_batch
        filter_ids = [meta.id for meta, _, _ in results if meta]
        print(f"\n{'─'*60}")
        print(f"  Applying post-process filter: {filter_spec}")
        apply_filter_batch(out_dir, filter_ids, filter_spec, suffix="filtered")

    # ── Compositing ──
    if args.composite:
        if not keys:
            print("  ✗ No methods to composite.")
            return
        paths = [out_dir / registry.get_meta(k).filename() for k in keys if registry.get_meta(k)]
        paths = [p for p in paths if p.exists()]
        if not paths:
            print("  ✗ No generated images found to composite.")
            return
        out_name = args.output or f"composite_{args.composite}.png"
        _composite_images(paths, args.composite, out_dir / out_name, args.cols)


if __name__ == "__main__":
    main()