"""Headless verification for Physarum node 530.
Run: cd ~/Documents/GitHub/grillmaster-command-center && PYTHONPATH=$PWD /Users/admin/Documents/GitHub/hermes-agent/venv/bin/python _verify_physarum.py
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

# Force a small canvas so the CPU sim is fast headless
import image_pipeline.core.utils as U
U.W = 192
U.H = 192

import numpy as np
import image_pipeline.methods  # trigger registration
from image_pipeline.core.registry import get_all

reg = get_all()
meta = reg.get("530")
assert meta is not None, "node 530 not registered"
print(f"[OK] registered: id=530 name={meta.name!r} category={meta.category!r}")
print(f"[OK] outputs={list((meta.outputs or {}).keys())}")
print(f"[OK] total nodes now: {len(reg)}")

def run(**over):
    params = {
        "agents": 6000, "spawn": "random", "sensor_dist": 9.0,
        "sensor_angle": 0.5, "rotation_angle": 0.4, "step_size": 1.0,
        "deposit_amount": 1.0, "decay": 0.93, "diffuse": 0.6,
        "colormode": "palette", "palette": "amber", "bg_style": "dark",
        "n_frames": 120,
    }
    params.update(over)
    from image_pipeline.methods.simulations.physarum import method_physarum
    out = HERE / "_physarum_out"
    out.mkdir(exist_ok=True)
    for f in out.glob("*.png"):
        f.unlink()
    img = method_physarum(out, 42, params)
    arr = np.array(img)
    std = float(arr.std())
    mean = float(arr.mean())
    return arr, std, mean

# 1. non-black
arr, std, mean = run()
print(f"[probe] default -> shape={arr.shape} mean={mean:.3f} std={std:.3f}")
assert std > 0.02, f"image looks flat (std={std})"

def mask_iou(a, b, thr=0.15):
    """IoU of bright-vein masks — robust to sparse vein images where mean
    pixel diff is dominated by the dark background."""
    ma = (a.astype(np.float64).mean(axis=-1) > thr)
    mb = (b.astype(np.float64).mean(axis=-1) > thr)
    inter = np.logical_and(ma, mb).sum()
    union = np.logical_or(ma, mb).sum()
    return float(inter / max(union, 1))

# 2. param liveness via vein-mask IoU (sparse images need structural metric).
#    sensor_dist strongly sets network scale (fine vs coarse veins).
a_near, _, _ = run(sensor_dist=4.0, n_frames=140)
a_far, _, _ = run(sensor_dist=28.0, n_frames=140)
iou_sd = mask_iou(a_near, a_far)
print(f"[probe] sensor_dist near vs far mask IoU = {iou_sd:.3f} (low = structure changed)")
assert iou_sd < 0.80, f"sensor_dist param has no structural effect (IoU={iou_sd})"

# step_size sets how far agents travel per tick -> coarse vs fine mesh
a_slow, _, _ = run(step_size=0.4, n_frames=140)
a_fast, _, _ = run(step_size=4.0, n_frames=140)
iou_ss = mask_iou(a_slow, a_fast)
print(f"[probe] step_size slow vs fast mask IoU = {iou_ss:.3f}")
assert iou_ss < 0.80, f"step_size param has no structural effect (IoU={iou_ss})"

# spawn affects early structure (before convergence erases the init topology)
a_ring_early, _, _ = run(spawn="ring", n_frames=25)
a_rand_early, _, _ = run(spawn="random", n_frames=25)
iou_spawn = mask_iou(a_ring_early, a_rand_early)
print(f"[probe] spawn ring vs random (early) mask IoU = {iou_spawn:.3f}")
assert iou_spawn < 0.80, f"spawn param has no effect (IoU={iou_spawn})"

# 3. decay param liveness (structural)
a_hi, _, _ = run(decay=0.99)
a_lo, _, _ = run(decay=0.82)
iou_decay = mask_iou(a_hi, a_lo)
print(f"[probe] decay hi vs lo mask IoU = {iou_decay:.3f}")
assert iou_decay < 0.85, f"decay param has no structural effect (IoU={iou_decay})"

# 4. end-to-end PNG written
pngs = list((HERE / "_physarum_out").glob("*.png"))
assert pngs, "no PNG written"
print(f"[OK] PNG written: {pngs[0].name}")

print("\nALL PHYSARUM CHECKS PASSED")
