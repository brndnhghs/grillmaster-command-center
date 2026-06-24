"""Test Gray-Scott phase portrait + cell mode."""
import sys, os, subprocess
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pathlib import Path
import numpy as np
from PIL import Image

from image_pipeline.core.animation import enable_frame_capture, get_frames, disable_frame_capture, frames_to_mp4
from image_pipeline.methods.simulations.gray_scott import method_gray_scott
from image_pipeline.core.utils import W, H
import image_pipeline.methods  # noqa: F401

out_dir = Path(__file__).resolve().parent

enable_frame_capture("134")
method_gray_scott(out_dir, 42, params={
    "anim_mode": "phase_portrait",
    "n_frames": 720,
    "dt": 1.0,
    "cell_mode": "true",
    "cell_min": 1,
    "cell_max": 64,
})
frames = get_frames("134")
disable_frame_capture()

print(f"Captured {len(frames)} frames")

out_path = out_dir / "134-gs-portrait.mp4"
def gen():
    for f in frames:
        yield f
result = frames_to_mp4(gen(), out_path, fps=24, quality=23, max_frames=len(frames))
if result:
    size_mb = result.stat().st_size / 1024 / 1024
    print(f"\n✓ Video: {result.name} ({size_mb:.1f} MB, {len(frames)} frames @ 24fps)")
else:
    print("✗ Encoding failed")

# Verify
f0 = frames[0]
row = (f0[256, :, 0] * 255).astype(float)
def avg_block(seg):
    q = np.round(seg / 5).astype(int)
    ch = np.where(np.diff(q) != 0)[0]
    if len(ch) == 0: return len(seg)
    return np.mean(np.diff(np.concatenate([[0], ch, [len(seg)-1]])))
print(f"Pixelation check:")
print(f"  Left:   {avg_block(row[:100]):.1f}px")
print(f"  Mid:    {avg_block(row[334:434]):.1f}px")
print(f"  Right:  {avg_block(row[-100:]):.1f}px")
print(f"  Mean: {f0.mean():.1f}  Min: {f0.min():.4f}  Max: {f0.max():.4f}")
