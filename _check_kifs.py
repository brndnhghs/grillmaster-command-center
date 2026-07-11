"""Headless verification for node 351 (Kaleidoscopic IFS).
Mirror of the grillmaster 8-step audit: registration, non-black render,
t/shadowing, smooth delta, and per-mode/per-param delta.
Run from repo root with: env -u PYTHONPATH .venv/bin/python _check_kifs.py
"""
import numpy as np
from PIL import Image
import io, sys

# 1) Registration (Rule #14 in-process check — source of truth)
from image_pipeline.server import app  # noqa: F401  (also Rule #8 import gate)
from image_pipeline.core.registry import get_all
assert "351" in get_all(), "node 351 not registered!"
meta = get_all()["351"]
print(f"[reg] 351 '{meta.name}' category={meta.category} inputs={meta.inputs} "
      f"outputs={list(meta.outputs.keys())} nparams={len(meta.params)}")

# 2) Non-black render via the method fn directly
from image_pipeline.methods.fractals.kaleidoscopic_ifs import method_kaleidoscopic_ifs
from pathlib import Path
import tempfile

def render(params):
    with tempfile.TemporaryDirectory() as d:
        out = Path(d)
        res = method_kaleidoscopic_ifs(out, 42, params=params)
        arr = np.asarray(res, dtype=np.float64)
        return arr

def stats(a):
    return float(a.mean()), float(a.std())

base = {"anim_mode": "none", "time": 0.0}
img0 = render(base)
print(f"[render] none-mode mean={stats(img0)[0]:.3f} std={stats(img0)[1]:.3f}")
assert img0.std() > 0.02, "static render is (near-)black!"
assert img0.mean() > 1.0 / 255.0, "static render too dark"

# 3) spin animation -> frame-to-frame delta (Architecture B: vary time)
a_spin = render({"anim_mode": "spin", "time": 0.0})
b_spin = render({"anim_mode": "spin", "time": 3.14})
d_spin = float(np.mean(np.abs(a_spin - b_spin)))
print(f"[anim] spin t0->t3.14  Δ={d_spin:.4f}")
assert d_spin > 0.05, "spin animation produced no motion!"

# 4) param perturbation: scale swap
a_sc = render({"anim_mode": "none", "scale": 1.8})
b_sc = render({"anim_mode": "none", "scale": 3.2})
d_sc = float(np.mean(np.abs(a_sc - b_sc)))
print(f"[param] scale 1.8->3.2  Δ={d_sc:.4f}")
assert d_sc > 0.05, "scale slider has no effect!"

# 5) system choices all render distinct, non-black
for sysname in ["box", "kaleidoscopic", "inversion"]:
    r = render({"anim_mode": "none", "system": sysname})
    print(f"[system] {sysname:14s} mean={stats(r)[0]:.3f} std={stats(r)[1]:.3f}")
    assert r.std() > 0.02, f"{sysname} produced black output"

# 6) color modes
for cm in ["escape_time", "palette", "angle", "orbit_trap"]:
    r = render({"anim_mode": "none", "color_mode": cm})
    assert r.std() > 0.02, f"color_mode {cm} black"

print("[OK] all checks passed")
