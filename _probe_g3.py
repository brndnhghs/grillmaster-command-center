import numpy as np, tempfile, shutil
from pathlib import Path
import image_pipeline.methods
from image_pipeline.core.graph import GraphExecutor
from image_pipeline.core.utils import set_canvas

W, H = 768, 512
nodes = [
    {"id": "n1", "method_id": "79", "params": {'walkers': 1000, 'steps': 2000, 'step_size': 4.0, 'walker_type': 'classic', 'walk_style': 'random_walk', 'color_mode': 'velocity', 'palette': 'viridis', 'layout': 'grid', 'anim_mode': 'none'}, "dirty": True, "render": True},
    {"id": "n2", "method_id": "68", "params": {'source': 'perlin', 'radius': 5.0, 'anisotropy': 2.0, 'blend': 'screen', 'presmooth': 1.0, 'noise_amp': 0.3, 'anim_mode': 'none'}, "dirty": True},
]
edges = [{"src_node": "n2", "src_port": "image", "dst_node": "n1", "dst_port": "image_in"}]
set_canvas(W, H)
wd = Path(tempfile.mkdtemp(prefix="g3-"))
try:
    ex = GraphExecutor(wd, fps=24, in_memory=True, audit_to_disk=False)
    frames = []
    for fr in range(8):
        flat, term, errs = ex.execute([dict(n) for n in nodes], edges, 42, frame=fr, frames=8)
        arr = (flat.get("n1") or {}).get("image")
        print(f"frame {fr}: arr={'None' if arr is None else arr.shape}, errs={errs}")
        if arr is not None:
            a = np.asarray(arr, dtype=np.float32)
            if a.ndim == 3:
                a = a.mean(-1)
            frames.append(a)
    if len(frames) >= 2:
        st = np.stack(frames)
        print("spatial_var(mean frame var):", float(st.mean(0).var()))
        print("temporal_var:", float(st.var(0).mean()))
        print("frame0 min/max/mean:", float(frames[0].min()), float(frames[0].max()), float(frames[0].mean()))
finally:
    shutil.rmtree(wd, ignore_errors=True)
