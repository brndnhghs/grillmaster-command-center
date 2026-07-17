"""Sel'kov Glycolysis — excitable reaction-diffusion spiral waves.

The **Sel'kov model** (E. E. Sel'kov, 1968, *European Journal of Biochemistry*
*4*, 79 — a minimal model of glycolytic oscillations / substrate-depletion
oscillators) is a *two*-variable kinetics that is *excitable*, not simply
bistable. Unlike Gray-Scott (activation–inhibition, id 155) or the BZ
Oregonator (photosensitive, id 91), Sel'kov's nonlinearity is the product
u²v (quadratic in the substrate u), which makes the medium support *traveling
excitable waves* (spirals, target patterns, expanding rings) when two
species diffuse on a lattice:

    du/dt = a − u +  u²·v   +  Du·∇²u
    dv/dt = b·u² − u²·v     +  Dv·∇²v

  u       glycolytic substrate concentration (the FIELD we render)
  v       "active" intermediate (ADP/AMP feedback) concentration
  a       constant supply of substrate
  b       removal rate of the intermediate
  Du, Dv  diffusion coefficients

In the excitable regime (a small, b ≈ 0.5–1) a point perturbation ignites a
wavefront that travels outward and curls into **spirals** where fronts meet —
the same spiral meander seen in real glycolytic waves (Boiteux et al. 1975)
and in cardiac tissue (the medium is a classic *excitable* system). The
waves never settle: they keep re-exciting, so the node is **perpetually
morphing** and survives the shootout contrast-only liveness cull.

Why this node exists (shootout / dead-bucket context): it is *cheap*
(O(H·W) explicit Euler with a 5-point Laplacian, renders in < 1 s, never
hits the 150 s render-timeout cull) and *perpetually morphing* (spiral
wavefronts sweep frame to frame with strong, spatially-varying temporal
variance). It directly dilutes the two dominant shootout death causes
(timeout-culled heavy sims + contrast-static false-culls), exactly the
directive from the 2026-07 candidate manifests that drove nodes #994–#1002.

Architecture A — single-call internal simulation with capture_frame(). The
orchestrator captures the in-memory frame buffer; the GPU twin
(`CLIENT_GPU_SIMS["1003"]`) mirrors the excitable RD dynamics for the live
preview only — the CPU node stays the authoritative export.

Distinct from sibling nodes: Gray-Scott (155) needs F≈0.04/k≈0.06 Turing
spots/mitosis; BZ (91) is photosensitive oscillations; Sel'kov (1003) is the
*excitable* glycolytic substrate-depletion system whose signature is
re-entrant spiral waves, a different dynamical regime of 2-species RD.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
from PIL import Image

from ...core.registry import method
from ...core.utils import (
    save, mn, seed_all, W, H,
    write_scalars, write_field, write_mask,
)
from ...core.animation import capture_frame


# ── IQ cosine palette (smooth, periodic, vivid — matches the GPU twin) ──
def _iq_palette(t: np.ndarray) -> np.ndarray:
    t = np.clip(np.asarray(t, dtype=np.float64), 0.0, 1.0)
    r = 0.5 + 0.5 * np.cos(6.2831853 * (t + 0.00))
    g = 0.5 + 0.5 * np.cos(6.2831853 * (t + 0.3333333))
    b = 0.5 + 0.5 * np.cos(6.2831853 * (t + 0.6666667))
    return np.stack([r, g, b], axis=-1)


def _lap(a: np.ndarray) -> np.ndarray:
    """5-point Laplacian with toroidal wrap (matches GPU RepeatWrapping)."""
    return (np.roll(a, 1, 1) + np.roll(a, -1, 1)
            + np.roll(a, 1, 0) + np.roll(a, -1, 0)
            - 4.0 * a)


# ── Seed presets: where the first excitation ignites ──
SEEDS = {
    "point":    "single central point → expanding target ring",
    "two":      "two points → colliding fronts that curl into spirals",
    "line":     "horizontal line → planar wave that rolls up at the ends",
    "noisy":    "several hashed blobs → a field that self-organises into spirals",
    "none":     "uniform u (no ignition) — static baseline for the liveness test",
}


@method(
    id="1003",
    name="Sel'kov Glycolysis",
    category="simulations",
    tags=["physics", "reaction-diffusion", "excitable", "selkov", "glycolysis",
          "spiral", "waves", "emergent"],
    timeout=120,
    outputs={"image": "IMAGE", "field": "FIELD", "mask": "MASK", "luminance": "SCALAR"},
    params={
        "anim_mode": {
            "description": "ignition seed: point/two/line/noisy/none (static)",
            "choices": ["point", "two", "line", "noisy", "none"],
            "default": "two",
        },
        "n_frames": {
            "description": "number of simulation frames captured",
            "min": 30, "max": 600, "default": 150,
        },
        "dt": {
            "description": "integration timestep (explicit Euler; keep < 0.3 for stability)",
            "min": 0.05, "max": 0.4, "default": 0.2,
        },
        "a": {
            "description": "substrate supply rate a (small → strongly excitable)",
            "min": 0.05, "max": 0.6, "default": 0.18,
        },
        "b": {
            "description": "intermediate removal rate b",
            "min": 0.1, "max": 1.5, "default": 0.6,
        },
        "diff_u": {
            "description": "substrate diffusion Du",
            "min": 0.0, "max": 1.0, "default": 0.25,
        },
        "diff_v": {
            "description": "intermediate diffusion Dv",
            "min": 0.0, "max": 1.0, "default": 0.12,
        },
        "render_style": {
            "description": "what to paint: substrate (u heat), intermediate (v), dual",
            "choices": ["substrate", "intermediate", "dual"],
            "default": "substrate",
        },
    },
)
def method_selkov(out_dir: Path, seed: int, params=None):
    """Sel'kov excitable reaction-diffusion — sustained spiral / target waves.

    Architecture A — internal explicit-Euler simulation; the excitable medium
    keeps re-igniting wavefronts (spirals where fronts meet), so the node is
    alive under the shootout liveness gate. `none` renders a single static
    frame (uniform field, no ignition).
    """
    if params is None:
        params = {}
    seed_kind = str(params.get("anim_mode", "two"))
    is_static = (seed_kind == "none")

    n_frames = int(params.get("n_frames", 150))
    dt = float(params.get("dt", 0.2))
    a = float(params.get("a", 0.18))
    b = float(params.get("b", 0.6))
    Du = float(params.get("diff_u", 0.25))
    Dv = float(params.get("diff_v", 0.12))
    render_style = str(params.get("render_style", "substrate"))

    if is_static:
        n_frames = 1

    # ── Seed wiring ──
    seed_all(seed)
    rng = np.random.default_rng(seed)

    h, w = H, W
    cy, cx = (h - 1) / 2.0, (w - 1) / 2.0

    # ── Initial state: u≈uniform substrate, v≈0; ignite a perturbation ──
    u = np.full((h, w), 0.6, dtype=np.float64)
    v = np.full((h, w), 0.25, dtype=np.float64)
    if not is_static:
        def _ignite(mask_bool):
            u[mask_bool] = 0.05
            v[mask_bool] = 0.85

        if seed_kind == "point":
            rad = max(2, int(min(h, w) * 0.02))
            yy, xx = np.mgrid[0:h, 0:w]
            _ignite((xx - cx) ** 2 + (yy - cy) ** 2 < rad * rad)
        elif seed_kind == "two":
            rad = max(2, int(min(h, w) * 0.018))
            yy, xx = np.mgrid[0:h, 0:w]
            d1 = (xx - cx * 0.6) ** 2 + (yy - cy) ** 2
            d2 = (xx - cx * 1.4) ** 2 + (yy - cy) ** 2
            _ignite((d1 < rad * rad) | (d2 < rad * rad))
        elif seed_kind == "line":
            rad = max(1, int(min(h, w) * 0.012))
            yy, xx = np.mgrid[0:h, 0:w]
            _ignite(np.abs(yy - cy) < rad)
        elif seed_kind == "noisy":
            nblobs = int(rng.integers(5, 10))
            for _ in range(nblobs):
                bx = int(rng.integers(0, w))
                by = int(rng.integers(0, h))
                rad = max(2, int(min(h, w) * rng.uniform(0.015, 0.03)))
                yy, xx = np.mgrid[0:h, 0:w]
                _ignite((xx - bx) ** 2 + (yy - by) ** 2 < rad * rad)

    # ── Render dispatch ──
    def _render_substrate(uu: np.ndarray) -> Image.Image:
        f = np.clip(uu, 0.0, 1.5) / 1.5
        gray = (f ** 0.6 * 255).astype(np.uint8)
        return Image.fromarray(np.stack([gray] * 3, axis=-1), mode="RGB")

    def _render_intermediate(vv: np.ndarray) -> Image.Image:
        f = np.clip(vv, 0.0, 1.0)
        col = _iq_palette(f)
        return Image.fromarray((np.clip(col, 0.0, 1.0) * 255).astype(np.uint8),
                               mode="RGB")

    def _render_dual(uu: np.ndarray, vv: np.ndarray) -> Image.Image:
        fu = np.clip(uu, 0.0, 1.5) / 1.5
        fv = np.clip(vv, 0.0, 1.0)
        col = np.stack([
            0.5 + 0.5 * fv,                       # R: intermediate
            0.5 * fu + 0.3 * fv,                  # G: mix
            1.0 - fu,                              # B: substrate-depleted edges
        ], axis=-1)
        return Image.fromarray((np.clip(col, 0.0, 1.0) * 255).astype(np.uint8),
                               mode="RGB")

    img = None
    last_max_u = 0.0

    # ══════════════════════════════════════════
    #  SIMULATION LOOP (explicit Euler, 5-pt Laplacian)
    # ══════════════════════════════════════════
    for frame in range(n_frames):
        uvv = u * u * v
        du = a - u + uvv + Du * _lap(u)
        dv = b * u * u - uvv + Dv * _lap(v)
        u = np.clip(u + dt * du, 0.0, 2.0)
        v = np.clip(v + dt * dv, 0.0, 2.0)
        last_max_u = float(u.max())

        # ── Render ──
        if render_style == "intermediate":
            canvas = _render_intermediate(v)
        elif render_style == "dual":
            canvas = _render_dual(u, v)
        else:
            canvas = _render_substrate(u)
        img = canvas
        if not is_static:
            capture_frame("1003", np.array(img, dtype=np.float32) / 255.0)

    # ── Final ──
    if img is None:
        img = Image.new("RGB", (int(w), int(h)), (5, 5, 18))

    if not is_static:
        capture_frame("1003", np.array(img, dtype=np.float32) / 255.0)

    # ── Outputs (Rules 4/5/10) ──
    write_field(out_dir, u.astype(np.float32))     # substrate field
    write_field(out_dir, v.astype(np.float32))     # intermediate field
    # Mask = excited (wavefront) regions: high substrate activity above baseline
    wave_mask = np.clip((u - 0.6) / 0.9, 0.0, 1.0).astype(np.float32)
    write_mask(out_dir, wave_mask)
    write_scalars(out_dir,
                  a=a, b=b, dt=dt, diff_u=Du, diff_v=Dv,
                  mean_substrate=float(u.mean()),
                  max_substrate=last_max_u,
                  n_frames=n_frames)
    try:
        save(img, mn(1003, "Sel'kov Glycolysis"), out_dir)
    except Exception as e:  # Rule 1: fallback so a frame still lands on disk
        print(f"  [selkov] save fallback: {e}")
        img.save(str(out_dir / "1003_selkov.png"))
    return img
