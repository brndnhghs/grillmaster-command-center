"""Build video using direct frame capture + ffmpeg, bypass animate_method."""
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
    "anim_mode": "stripes",
    "diff_u": 0.16, "diff_v": 0.08,
    "dt": 1.0, "n_frames": 720,  # 30s at 24fps — LONG
    "render_style": "v",
    "cell_mode": "true",
    "cell_min": 1,
    "cell_max": 64,
})
frames = get_frames("134")
disable_frame_capture()

print(f"Captured {len(frames)} frames")

# Encode directly to MP4
out_path = out_dir / "134-gs-cell.mp4"
def gen():
    for f in frames:
        yield f

result = frames_to_mp4(gen(), out_path, fps=24, quality=23, max_frames=len(frames))
if result:
    size_mb = result.stat().st_size / 1024 / 1024
    print(f"\n✓ Video: {result.name} ({size_mb:.1f} MB, {len(frames)} frames @ 24fps)")
else:
    print("✗ Encoding failed")

# Verify pixelation on frame 0
f0 = frames[0]
row = (f0[256, :, 0] * 255).astype(float)

def avg_block(seg):
    q = np.round(seg / 5).astype(int)
    ch = np.where(np.diff(q) != 0)[0]
    if len(ch) == 0:
        return len(seg)
    return np.mean(np.diff(np.concatenate([[0], ch, [len(seg)-1]])))

print(f"\nPixelation check:")
print(f"  Left (0-99):   {avg_block(row[:100]):.1f}px")
print(f"  Mid (334-433): {avg_block(row[334:434]):.1f}px")
print(f"  Right (668-767): {avg_block(row[-100:]):.1f}px")

# Also check direct pixel values
print(f"  Row min={row.min():.0f} max={row.max():.0f}")
print(f"  Row unique values at stride 8: {len(np.unique(row[::8].astype(int)))}")
