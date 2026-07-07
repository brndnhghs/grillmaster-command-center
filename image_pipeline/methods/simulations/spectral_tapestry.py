"""
Spectral Tapestry (ID 141)

TRAVELING WAVE modal interference — wavefronts ACTUALLY PROPAGATE
across the canvas instead of breathing in place.

u(x,y,t) = Σ a_k(t) · cos(kx_k·x + ky_k·y - ω_k·t + θ_k)

  - kx_k, ky_k: wavenumbers that drift slowly over time
  - ω_k: each mode has its own frequency → waves travel at different speeds
  - a_k(t): amplitude evolves via coupled Landau-Stuart oscillators
  - The cos() creates PROPAGATING wavefronts, not standing waves

Wavenumbers drift slowly → the weave of the pattern itself evolves.
Amplitudes chatter → modes phase-lock and compete.
The result: a living, flowing interference field.
"""

from __future__ import annotations
import math
from pathlib import Path
import numpy as np
from PIL import Image
from ...core.registry import method
from ...core.utils import save, mn, seed_all, W, H
from ...core.animation import capture_frame


@method(
    id="161",
    name="Spectral Tapestry",
    description="Spectral Tapestry — simulations node.",
    category="simulations",
    tags=["waves", "eigenmodes", "interference", "modal"],
    timeout=300,
    params={
        "anim_mode": {
            "description": "Regime",
            "choices": ["evolve", "flow", "chatter", "storm", "obstacle"],
            "default": "evolve",
        },
        "n_modes": {"min": 8, "max": 80, "default": 25},
        "coupling": {"min": 0.0, "max": 2.0, "default": 0.4},
        "n_frames": {"min": 100, "max": 1200, "default": 360},
        "dt": {"min": 0.02, "max": 0.5, "default": 0.1},
        "noise": {"min": 0.0, "max": 0.1, "default": 0.01},
        "drift_speed": {"min": 0.0, "max": 0.05, "default": 0.005},
    }
)
def st(out_dir, seed, params=None):
    if params is None:
        params = {}
    am = str(params.get("anim_mode", "evolve"))
    N = int(params.get("n_modes", 25))
    coupling = float(params.get("coupling", 0.4))
    nf = int(params.get("n_frames", 360))
    dt = float(params.get("dt", 0.1))
    noise_amp = float(params.get("noise", 0.01))
    drift_sp = float(params.get("drift_speed", 0.005))

    seed_all(seed)
    rng = np.random.default_rng(seed)

    if am == "flow":
        coupling = float(params.get("coupling", 0.2))
        drift_sp = float(params.get("drift_speed", 0.01))
        noise_amp = float(params.get("noise", 0.005))
    elif am == "chatter":
        coupling = float(params.get("coupling", 0.8))
        noise_amp = float(params.get("noise", 0.03))
        drift_sp = float(params.get("drift_speed", 0.002))
        N = int(params.get("n_modes", 40))
    elif am == "storm":
        coupling = float(params.get("coupling", 1.2))
        noise_amp = float(params.get("noise", 0.05))
        drift_sp = float(params.get("drift_speed", 0.008))
        N = int(params.get("n_modes", 60))

    h, w = H, W
    xs = np.arange(w, dtype=np.float64)
    ys = np.arange(h, dtype=np.float64)

    # ── Mode basis ──
    # Wavenumbers: range controls pattern density
    kx_m = rng.uniform(1, 18, N) * 2 * math.pi / w
    ky_m = rng.uniform(1, 14, N) * 2 * math.pi / h

    # Each mode's natural frequency (wave speed = ω / |k|)
    freq = rng.uniform(0.5, 4.0, N)

    # Initial phases
    θ = rng.uniform(0, 2 * math.pi, N)

    # Growth rates
    r_k = rng.uniform(0.5, 2.0, N)

    # Coupling matrix in k-space
    K = np.column_stack([kx_m, ky_m])
    K_dist = np.sqrt(((K[:, None] - K[None, :])**2).sum(axis=2))
    K_sim = np.exp(-K_dist / (K_dist.mean() + 1e-10))
    np.fill_diagonal(K_sim, 0.0)
    K_sim /= K_sim.sum(axis=1, keepdims=True) + 1e-10

    # Wavenumber drift targets (random walk anchors)
    kx_target = kx_m.copy()
    ky_target = ky_m.copy()
    kx_vel = np.zeros(N)
    ky_vel = np.zeros(N)

    # Initial amplitudes
    a = rng.uniform(-0.3, 0.3, N).astype(np.float64)

    # Obstacle
    om, od = None, []
    if am == "obstacle":
        nobs = int(params.get("n_obstacles", 3))
        o_rad = float(params.get("obstacle_radius", 0.08))
        yy_o, xx_o = np.ogrid[:h, :w]
        om = np.ones((h, w), dtype=bool)
        nc = int(math.ceil(nobs / 2))
        sx, sy = w / (nc + 1), h / 3.5
        cr = int(min(h, w) * o_rad * 0.7)
        for i in range(nobs):
            ox = int((i % nc + 1) * sx)
            oy = int((i // nc + 0.5 + (0.0 if i % 2 == 0 else 0.5)) * sy)
            d = np.sqrt((xx_o - ox)**2 + (yy_o - oy)**2) <= cr
            om &= ~d
            od.append((ox, oy, cr))

    print(f"[ST141] {N} traveling modes, coupling={coupling}, mode={am}")

    # ── Precompute coordinate grid ──
    XX, YY = np.meshgrid(xs, ys)

    for fr in range(nf):
        t = fr * dt

        # ── Amplitude evolution ──
        coupling_term = coupling * (K_sim @ a - a)
        da = (r_k - a**2) * a + coupling_term
        a += dt * da
        a += rng.normal(0, noise_amp, N)
        a = np.clip(a, -3.0, 3.0)

        # ── Wavenumber drift ──
        # Random walk with spring-back to mean
        kx_vel += drift_sp * rng.normal(0, 1, N) - 0.01 * (kx_m - kx_target)
        ky_vel += drift_sp * rng.normal(0, 1, N) - 0.01 * (ky_m - ky_target)
        kx_m += kx_vel * dt
        ky_m += ky_vel * dt
        # Keep in reasonable range
        kx_m = np.clip(kx_m, 0.5 * 2 * math.pi / w, 25 * 2 * math.pi / w)
        ky_m = np.clip(ky_m, 0.5 * 2 * math.pi / h, 20 * 2 * math.pi / h)
        # New random targets occasionally
        if fr % 50 == 0:
            kx_target = rng.uniform(1, 18, N) * 2 * math.pi / w
            ky_target = rng.uniform(1, 14, N) * 2 * math.pi / h

        # ── Field reconstruction (traveling waves) ──
        u = np.zeros((h, w), dtype=np.float64)
        for k in range(N):
            if abs(a[k]) > 1e-6:
                # Traveling wave: cos(kx·x + ky·y - ω·t + θ)
                phase = kx_m[k] * XX + ky_m[k] * YY - freq[k] * t + θ[k]
                u += a[k] * np.cos(phase)

        # Obstacle
        if om is not None:
            u[~om] = 0.0

        # ── Render ──
        c = u - np.mean(u)
        scale = max(float(np.percentile(np.abs(c), 95)), 1e-10)
        g = ((np.tanh(c / scale * 3.0) + 1.0) * 127.5).astype(np.uint8)
        img = Image.fromarray(np.stack([g] * 3, -1), "RGB")

        if od:
            arr = np.array(img, np.uint8)
            for ox, oy, r in od:
                yy_d, xx_d = np.ogrid[:h, :w]
                d = np.sqrt((xx_d - ox)**2 + (yy_d - oy)**2)
                arr[np.abs(d - r) <= 1.5] = 255
                arr[d <= r] //= 2
            img = Image.fromarray(arr)

        capture_frame("141", np.array(img))

        if fr % max(1, nf // 10) == 0 or fr == nf - 1:
            print(f"  f{fr:4d}/{nf} | |a|∈[{np.abs(a).min():.2f},{np.abs(a).max():.2f}]")

    name = mn(141, f"ST-{am}")
    save(img, name, out_dir)
    print(f"[ST141] Saved {name}")


if __name__ == "__main__":
    st(Path("/tmp/st_test"), 42, {"anim_mode": "flow", "n_frames": 100})
