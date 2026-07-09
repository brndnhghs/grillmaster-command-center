"""GPU parity harness — compare a node's CPU (numpy) render against its
client-side GPU (WebGL2) render, at a fixed frame + fixed params.

This is the reusable measurement tool for the GPU-first migration (all phases).
It reports SSIM (luminance-structure similarity) and mean-abs-diff (MAD), the
tolerance metrics the migration uses instead of bit-exact equality — GPU fp32
and CPU fp64 differ, and seeded/chaotic nodes never match bit-for-bit.

Two render paths:
  render_cpu(mid, params, seed, w, h)     -> HxWx3 float32 [0,1]   (GraphExecutor)
  render_client(mid, params, seed, w, h)  -> HxWx3 float32 [0,1]   (browser-harness
                                             → client3d.js renderFrame → canvas)

The client path shells out to `browser-harness` (the repo's CDP tool) and needs
the dev server running on :7860. It is skipped/raises cleanly if unavailable, so
this module is safe to import in CI.

CLI:  python -m image_pipeline.tests.gpu_parity <method_id> [--seed N] [--wh WxH] [k=v ...]
"""
from __future__ import annotations

import base64
import io
import json
import subprocess
import sys
from pathlib import Path

import numpy as np


# ── metrics ──────────────────────────────────────────────────────────────────

def _to_gray(a: np.ndarray) -> np.ndarray:
    """HxWx3 [0,1] → HxW luminance float64."""
    a = np.asarray(a, dtype=np.float64)
    if a.ndim == 3:
        a = a[..., :3] @ np.array([0.299, 0.587, 0.114])
    return a


def ssim(a: np.ndarray, b: np.ndarray) -> float:
    """Mean SSIM over the image (Gaussian-windowed, luminance). ~1.0 = identical
    structure; degrades with structural/positional divergence."""
    from scipy.ndimage import gaussian_filter
    ga, gb = _to_gray(a), _to_gray(b)
    if ga.shape != gb.shape:  # resize b to a via nearest — shapes should match
        raise ValueError(f"shape mismatch {ga.shape} vs {gb.shape}")
    sd = 1.5
    mu_a = gaussian_filter(ga, sd); mu_b = gaussian_filter(gb, sd)
    va = gaussian_filter(ga * ga, sd) - mu_a ** 2
    vb = gaussian_filter(gb * gb, sd) - mu_b ** 2
    vab = gaussian_filter(ga * gb, sd) - mu_a * mu_b
    C1, C2 = 0.01 ** 2, 0.03 ** 2
    s = ((2 * mu_a * mu_b + C1) * (2 * vab + C2)) / \
        ((mu_a ** 2 + mu_b ** 2 + C1) * (va + vb + C2))
    return float(np.clip(s, -1, 1).mean())


def mad(a: np.ndarray, b: np.ndarray) -> float:
    """Mean absolute difference over RGB, in [0,1] units (0 = identical)."""
    a = np.asarray(a, dtype=np.float64)[..., :3]
    b = np.asarray(b, dtype=np.float64)[..., :3]
    return float(np.abs(a - b).mean())


# ── CPU render (numpy path, via the real executor) ──────────────────────────

def render_cpu(method_id: str, params: dict | None = None, seed: int = 42,
               w: int = 256, h: int = 256, frame: int = 0) -> np.ndarray:
    """Render a node server-side through GraphExecutor. Returns HxWx3 float32 [0,1]."""
    from image_pipeline.core.graph import GraphExecutor
    from image_pipeline.core.utils import set_canvas
    set_canvas(w, h)
    ex = GraphExecutor(Path("/tmp/gm_gpu_parity_session"), in_memory=True)
    node = {"id": "n", "method_id": method_id, "render": True, "params": dict(params or {})}
    outputs, terminal_id, errs = ex.execute([node], [], seed, frame=frame, frames=1)
    if errs:
        raise RuntimeError(f"CPU render errors for {method_id}: {errs}")
    arr = (outputs.get(terminal_id) or {}).get("image")
    if arr is None:
        raise RuntimeError(f"CPU render produced no image for {method_id}")
    arr = np.asarray(arr, dtype=np.float32)
    if arr.dtype == np.uint8 or arr.max() > 1.5:
        arr = arr.astype(np.float32) / 255.0
    return arr[..., :3]


# ── Client render (WebGL2 path, via browser-harness → client3d.js) ──────────

_CLIENT_JS = r"""
(async function() {
  try {
    const C = await import('/ui/js/client3d.js?v=' + Date.now());
    // Ensure the shader bundle (node_map) is loaded before rendering.
    if (C.prepare) { try { await C.prepare([{method_id: %(MID)s}]); } catch(e){} }
    const nodes = [{id: 'n', method_id: %(MID)s, render: true, params: %(PARAMS)s}];
    const canvas = await C.renderFrame(nodes, [], %(FRAME)d, %(W)d, %(H)d, %(TIME)f);
    // second render — some nodes need a warm frame (bundle/compile) to settle
    await C.renderFrame(nodes, [], %(FRAME)d, %(W)d, %(H)d, %(TIME)f);
    window.__gp = canvas.toDataURL('image/png');
    const errs = C.getNodeErrors ? C.getNodeErrors() : {};
    window.__gpErr = Object.keys(errs).length ? JSON.stringify(errs) : null;
    return 'ok';
  } catch (e) { window.__gpErr = e.message + ' | ' + (e.stack||'').split('\n').slice(0,3).join(' | '); return 'ERR'; }
})()
"""


def render_client(method_id: str, params: dict | None = None, seed: int = 42,
                  w: int = 256, h: int = 256, frame: int = 0, time: float = 0.0,
                  server: str = "http://localhost:7860") -> np.ndarray:
    """Render a node client-side via browser-harness. Returns HxWx3 float32 [0,1].

    Requires the dev server on :7860 and `browser-harness` on PATH. Raises
    RuntimeError with the client-side error if the shader failed to compile/run.
    """
    from PIL import Image
    js = _CLIENT_JS % {
        "MID": json.dumps(str(method_id)),
        "PARAMS": json.dumps(dict(params or {})),
        "FRAME": frame, "W": w, "H": h, "TIME": time,
    }
    # Pass the JS base64-encoded so no quote/newline escaping can corrupt it.
    js_b64 = base64.b64encode(js.encode()).decode()
    script = (
        "from helpers import cdp\n"
        "import time, base64\n"
        "cdp('Network.enable'); cdp('Network.setCacheDisabled', cacheDisabled=True)\n"
        f"goto('{server}/'); wait_for_load(); time.sleep(1.2)\n"
        f"_code = base64.b64decode('{js_b64}').decode()\n"
        "js(_code)\n"
        "time.sleep(2.5)\n"
        "err = js('window.__gpErr'); data = js('window.__gp')\n"
        "print('GPPARITY_ERR:' + (err or 'none'))\n"
        "print('GPPARITY_DATA:' + (data or 'none'))\n"
    )
    out = subprocess.run(["browser-harness"], input=script, capture_output=True,
                         text=True, timeout=90).stdout
    err = next((l[len("GPPARITY_ERR:"):] for l in out.splitlines() if l.startswith("GPPARITY_ERR:")), "none")
    data = next((l[len("GPPARITY_DATA:"):] for l in out.splitlines() if l.startswith("GPPARITY_DATA:")), "none")
    if err and err != "none":
        raise RuntimeError(f"client render error for {method_id}: {err}")
    if not data or data == "none" or "," not in data:
        raise RuntimeError(f"client render produced no image for {method_id}\n{out[-500:]}")
    png = base64.b64decode(data.split(",", 1)[1])
    img = np.asarray(Image.open(io.BytesIO(png)).convert("RGB"), dtype=np.float32) / 255.0
    return img


# ── compare ──────────────────────────────────────────────────────────────────

def compare(method_id: str, params: dict | None = None, seed: int = 42,
            w: int = 256, h: int = 256, frame: int = 0, time: float = 0.0) -> dict:
    """Render both paths and report parity metrics."""
    cpu = render_cpu(method_id, params, seed, w, h, frame)
    cli = render_client(method_id, params, seed, w, h, frame, time)
    if cli.shape[:2] != cpu.shape[:2]:  # client canvas may differ; nearest-resize
        from PIL import Image
        cli_img = Image.fromarray((cli * 255).astype(np.uint8)).resize(
            (cpu.shape[1], cpu.shape[0]), Image.NEAREST)
        cli = np.asarray(cli_img, dtype=np.float32) / 255.0
    return {
        "method_id": method_id, "wh": f"{w}x{h}",
        "ssim": round(ssim(cpu, cli), 4),
        "mad": round(mad(cpu, cli), 4),
    }


def _parse_kv(args: list[str]) -> dict:
    d = {}
    for a in args:
        if "=" in a:
            k, v = a.split("=", 1)
            try:
                v = int(v)
            except ValueError:
                try:
                    v = float(v)
                except ValueError:
                    pass
            d[k] = v
    return d


if __name__ == "__main__":
    import argparse
    import image_pipeline.methods  # noqa: F401 — register nodes
    ap = argparse.ArgumentParser()
    ap.add_argument("method_id")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--wh", default="256x256")
    ap.add_argument("--time", type=float, default=0.0)
    ap.add_argument("kv", nargs="*")
    a = ap.parse_args()
    w, h = (int(x) for x in a.wh.lower().split("x"))
    res = compare(a.method_id, _parse_kv(a.kv), a.seed, w, h, time=a.time)
    print(json.dumps(res, indent=2))
