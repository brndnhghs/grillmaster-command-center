"""Sel'kov Glycolysis — excitable reaction-diffusion spiral / target waves.

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

Excitable-medium dynamics: a single localized ignition produces *one*
expanding wavefront that sweeps out and then the whole medium relaxes to its
uniform rest state — it does NOT spontaneously sustain motion. To produce the
signature **perpetually-morphing spiral / target waves** (the same spiral
meander seen in real glycolytic waves, Boiteux et al. 1975, and in cardiac
tissue) the node uses a **pacemaker**: it re-excites a (moving) stimulus site
every `pace_period` frames. A central pacemaker → concentric target rings;
a *rotating* pacemaker → genuine Archimedean spiral waves (the standard way
spiral waves are generated experimentally, via a rotating electrode). This is
physically honest and keeps the field alive across the whole sequence instead
of relaxing to a dead uniform state.

Why this node exists (shootout / dead-bucket context): it is *cheap*
(O(H·W) explicit Euler with a 5-point Laplacian, renders in < 1 s, never
hits the 150 s render-timeout cull) and *perpetually morphing* (the pacemaker
keeps re-igniting wavefronts, so the spiral meander sweeps frame to frame with
strong, spatially-varying temporal variance). It directly dilutes the two
dominant shootout death causes (timeout-culled heavy sims + contrast-static
false-culls), exactly the directive from the 2026-07 candidate manifests that
drove nodes #994–#1003.

Architecture A — single-call internal simulation with capture_frame(). The
orchestrator captures the in-memory frame buffer; the GPU twin
(`CLIENT_GPU_SIMS["1003"]`) mirrors the excitable RD dynamics for the live
preview only — the CPU node stays the authoritative export.

Distinct from sibling nodes: Gray-Scott (155) needs F≈0.04/k≈0.06 Turing
spots/mitosis; BZ (91) is photosensitive limit-cycle oscillations; Sel'kov
(1003) is the *excitable* glycolytic substrate-depletion system whose
signature is pacemaker-driven target / spiral waves, a different dynamical
regime of 2-species RD.
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


def _fixed_point(a: float, b: float):
    """Stable rest state (u0, v0) of the Sel'kov kinetics.

    Fixed point: v0 = b and a − u + u²b = 0 → b·u² − u + a = 0.
    The excitable rest is the smaller root. In the oscillatory regime
    (4ab > 1) there is no real rest; fall back to a safe sub-threshold value.
    """
    disc = 1.0 - 4.0 * a * b
    if disc <= 0.0 or b <= 0.0:
        return 0.30, b if b > 0.0 else 0.30
    u0 = (1.0 - math.sqrt(disc)) / (2.0 * b)
    return max(0.02, min(1.5, u0)), b


# ── Ignition presets ──
IGNITION = {
    "rotating":  "rotating pacemaker → sustained Archimedean spiral waves (default)",
    "pacemaker": "central pacemaker → concentric target rings",
    "spiral":    "broken-wavefront seed → a self-sustaining rotor (best effort)",
    "point":     "single central point → one expanding target ring (transient)",
    "line":      "horizontal line → planar wave that rolls up at the ends",
    "noisy":     "several hashed blobs → self-organising waves (transient)",
    "static":    "uniform rest — no ignition (static baseline for liveness tests)",
}


@method(
    id="1003",
    name="Sel'kov Glycolysis",
    category="simulations",
    tags=["physics", "reaction-diffusion", "excitable", "selkov", "glycolysis",
          "spiral", "target-waves", "waves", "emergent", "pacemaker"],
    timeout=120,
    outputs={"image": "IMAGE", "field": "FIELD", "mask": "MASK", "luminance": "SCALAR"},
    params={
        "ignition": {
            "description": "how waves are ignited: rotating pacemaker (spirals) / "
                           "central pacemaker (rings) / broken-wavefront rotor / "
                           "point / line / noisy / static",
            "choices": ["rotating", "pacemaker", "spiral", "point", "line", "noisy", "static"],
            "default": "rotating",
        },
        "n_frames": {
            "description": "number of simulation frames captured",
            "min": 30, "max": 600, "default": 200,
        },
        "dt": {
            "description": "integration timestep (explicit Euler; keep < 0.3 for stability)",
            "min": 0.05, "max": 0.3, "default": 0.2,
        },
        "a": {
            "description": "substrate supply rate a (small → strongly excitable)",
            "min": 0.05, "max": 0.6, "default": 0.10,
        },
        "b": {
            "description": "intermediate removal rate b",
            "min": 0.1, "max": 1.5, "default": 0.6,
        },
        "diff_u": {
            "description": "substrate diffusion Du (controls wave speed; ~1.2 tuned for visible spirals)",
            "min": 0.0, "max": 2.0, "default": 1.2,
        },
        "diff_v": {
            "description": "intermediate diffusion Dv",
            "min": 0.0, "max": 1.0, "default": 0.25,
        },
        "pace_period": {
            "description": "frames between pacemaker re-excitations (smaller → denser waves)",
            "min": 3, "max": 40, "default": 5,
        },
        "pace_radius": {
            "description": "radius (px) of the re-excited stimulus site",
            "min": 2, "max": 30, "default": 4,
        },
        "rot_radius": {
            "description": "orbit radius for the rotating pacemaker (fraction of min dim)",
            "min": 0.0, "max": 0.4, "default": 0.12,
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

    Architecture A — internal explicit-Euler simulation. A pacemaker re-ignites
    the medium every `pace_period` frames so wavefronts keep sweeping the
    canvas (spirals for a rotating pacemaker, concentric rings for a central
    one). The medium is therefore alive across the *whole* sequence, not just a
    transient, so it survives the shootout contrast-only liveness gate.
    `static` renders a single uniform frame (rest state, no ignition).
    """
    if params is None:
        params = {}
    ign = str(params.get("ignition", "rotating"))
    is_static = (ign == "static")

    n_frames = int(params.get("n_frames", 200))
    dt = float(params.get("dt", 0.2))
    a = float(params.get("a", 0.10))
    b = float(params.get("b", 0.6))
    Du = float(params.get("diff_u", 1.2))
    Dv = float(params.get("diff_v", 0.25))
    pace_period = int(params.get("pace_period", 5))
    pace_radius = int(params.get("pace_radius", 4))
    rot_radius = float(params.get("rot_radius", 0.12))
    render_style = str(params.get("render_style", "substrate"))

    if is_static:
        n_frames = 1

    # ── Seed wiring ──
    seed_all(seed)
    rng = np.random.default_rng(seed)

    h, w = H, W
    cy, cx = (h - 1) / 2.0, (w - 1) / 2.0

    # True resting state (so the medium sits still until ignited)
    u0, v0 = _fixed_point(a, b)
    u = np.full((h, w), u0, dtype=np.float64)
    v = np.full((h, w), v0, dtype=np.float64)

    # Ignited (above-threshold) stimulus: deplete substrate, spike the intermediate
    u_ig, v_ig = 0.03, 0.95

    yy, xx = np.mgrid[0:h, 0:w]

    def _ignite_disk(py, px, rad):
        py, px = float(py), float(px)
        m = (xx - px) ** 2 + (yy - py) ** 2 < rad * rad
        u[m] = u_ig
        v[m] = v_ig

    # ── Initial ignition (transient seeds + the broken-wavefront rotor) ──
    if not is_static:
        if ign == "point":
            _ignite_disk(cy, cx, max(2, int(min(h, w) * 0.02)))
        elif ign == "line":
            rad = max(1, int(min(h, w) * 0.012))
            u[np.abs(yy - cy) < rad] = u_ig
            v[np.abs(yy - cy) < rad] = v_ig
        elif ign == "noisy":
            nblobs = int(rng.integers(5, 10))
            for _ in range(nblobs):
                bx = int(rng.integers(0, w))
                by = int(rng.integers(0, h))
                rad = max(2, int(min(h, w) * rng.uniform(0.015, 0.03)))
                _ignite_disk(by, bx, rad)
        elif ign == "spiral":
            # Broken-wavefront rotor: excite the LEFT half of a central disk only,
            # leaving the wavefront tip free so it curls into a spiral.
            R = max(6, int(min(h, w) * 0.18))
            disk = (xx - cx) ** 2 + (yy - cy) ** 2 < R * R
            left = xx < cx
            m = disk & left
            u[m] = u_ig
            v[m] = v_ig
        # rotating / pacemaker: no initial ignition; the loop provides it.

    # ── Render dispatch ──
    def _render_substrate(uu: np.ndarray) -> Image.Image:
        span = max(1e-6, u0 * 2.0)
        f = np.clip(uu, 0.0, u0 * 2.0) / span
        gray = (f ** 0.6 * 255).astype(np.uint8)
        return Image.fromarray(np.stack([gray] * 3, axis=-1), mode="RGB")

    def _render_intermediate(vv: np.ndarray) -> Image.Image:
        f = np.clip(vv, 0.0, 1.0)
        col = _iq_palette(f)
        return Image.fromarray((np.clip(col, 0.0, 1.0) * 255).astype(np.uint8),
                               mode="RGB")

    def _render_dual(uu: np.ndarray, vv: np.ndarray) -> Image.Image:
        span = max(1e-6, u0 * 2.0)
        fu = np.clip(uu, 0.0, u0 * 2.0) / span
        fv = np.clip(vv, 0.0, 1.0)
        col = np.stack([
            0.5 + 0.5 * fv,            # R: intermediate
            0.5 * fu + 0.3 * fv,       # G: mix
            1.0 - fu,                  # B: substrate-depleted edges
        ], axis=-1)
        return Image.fromarray((np.clip(col, 0.0, 1.0) * 255).astype(np.uint8),
                               mode="RGB")

    img = None
    last_max_u = 0.0
    orbit_r = rot_radius * min(h, w)

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

        # ── Pacemaker re-excitation (keeps the medium alive) ──
        if ign in ("rotating", "pacemaker") and frame % pace_period == 0:
            if ign == "rotating":
                ang = 2.0 * math.pi * (frame / max(1, pace_period)) if pace_period > 0 else 0.0
                px = cx + orbit_r * math.cos(ang)
                py = cy + orbit_r * math.sin(ang)
            else:
                px, py = cx, cy
            _ignite_disk(py, px, pace_radius)

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
    # Mask = excited (wavefront) regions: substrate above resting level
    wave_mask = np.clip((u - u0) / max(1e-6, u0), 0.0, 1.0).astype(np.float32)
    write_mask(out_dir, wave_mask)
    write_scalars(out_dir,
                  a=a, b=b, dt=dt, diff_u=Du, diff_v=Dv,
                  pace_period=pace_period, pace_radius=pace_radius,
                  rot_radius=rot_radius,
                  rest_u=float(u0), rest_v=float(v0),
                  mean_substrate=float(u.mean()),
                  max_substrate=last_max_u,
                  n_frames=n_frames)
    try:
        save(img, mn(1003, "Sel'kov Glycolysis"), out_dir)
    except Exception as e:  # Rule 1: fallback so a frame still lands on disk
        print(f"  [selkov] save fallback: {e}")
        img.save(str(out_dir / "1003_selkov.png"))
    return img
