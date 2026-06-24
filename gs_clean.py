"""Clean Gray-Scott stripes with cell size modifier."""
import sys, os, subprocess
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pathlib import Path
import numpy as np

from image_pipeline.core.animation import enable_frame_capture, get_frames, disable_frame_capture, frames_to_mp4
from image_pipeline.methods.simulations.gray_scott import method_gray_scott
import image_pipeline.methods  # noqa: F401

out_dir = Path(__file__).resolve().parent

enable_frame_capture("134")
method_gray_scott(out_dir, 42, params={
    "anim_mode": "stripes",
    "diff_u": 0.16,
    "diff_v": 0.08,
    "dt": 1.0,
    "n_frames": 720,
    "render_style": "v",
    "cell_mode": "true",
    "cell_min": 1,
    "cell_max": 64,
})
frames = get_frames("134")
disable_frame_capture()

print(f"Captured {len(frames)} frames")

out_path = out_dir / "134-gs-stripes-cells.mp4"
def gen():
    for f in frames:
        yield f
result = frames_to_mp4(gen(), out_path, fps=24, quality=23, max_frames=len(frames))
if result:
    size_mb = result.stat().st_size / 1024 / 1024
    print(f"\n✓ {result.name} ({size_mb:.1f} MB, {len(frames)} frames @ 24fps)")
else:
    print("✗ Failed")
