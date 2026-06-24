"""Gray-Scott phase portrait — full resolution, no pixelation."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pathlib import Path

from image_pipeline.core.animation import enable_frame_capture, get_frames, disable_frame_capture, frames_to_mp4
from image_pipeline.methods.simulations.gray_scott import method_gray_scott
import image_pipeline.methods  # noqa: F401

out_dir = Path(__file__).resolve().parent

enable_frame_capture("134")
method_gray_scott(out_dir, 42, params={
    "anim_mode": "phase_portrait",
    "n_frames": 720,
    "dt": 1.0,
    "cell_mode": "false",  # full resolution, no pixelation
})
frames = get_frames("134")
disable_frame_capture()
print(f"Captured {len(frames)} frames")

out_path = out_dir / "134-gs-portrait-hires.mp4"
def gen():
    for f in frames:
        yield f
result = frames_to_mp4(gen(), out_path, fps=24, quality=23, max_frames=len(frames))
if result:
    size_mb = result.stat().st_size / 1024 / 1024
    print(f"\n✓ {result.name} ({size_mb:.1f} MB, {len(frames)} frames @ 24fps)")
else:
    print("✗ Failed")
