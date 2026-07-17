import sys, os, tempfile
sys.path.insert(0, ".")
import numpy as np
from pathlib import Path
import image_pipeline.methods
from image_pipeline.methods.simulations.mls_mpm import method_mls_mpm

# Quick render with minimal frames
with tempfile.TemporaryDirectory() as tmpdir:
    img = method_mls_mpm(Path(tmpdir), seed=42, params={
        "n_particles": 500,
        "n_grid": 32,
        "n_frames": 5,
        "dt": 0.001,
        "material": "elastic",
        "shape": "block",
        "colormap": "velocity",
    })
    arr = np.array(img)
    print(f"Image shape: {arr.shape}")
    print(f"Mean: {arr.mean():.2f}")
    print(f"Std: {arr.std():.4f}")
    print(f"Min: {arr.min()}, Max: {arr.max()}")
    nonblack = np.any(arr > 20, axis=-1).sum()
    total = arr.shape[0] * arr.shape[1]
    print(f"Non-black pixels: {nonblack}/{total} ({100*nonblack/total:.1f}%)")
    if arr.std() < 0.01:
        print("WARNING: near-uniform image!")
    else:
        print("OK: image has variance")
