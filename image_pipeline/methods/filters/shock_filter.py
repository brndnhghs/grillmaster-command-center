from __future__ import annotations
import math
from pathlib import Path

import numpy as np

from ...core.registry import method
from ...core.utils import save, norm, mn, seed_all, W, H, PALETTES, write_scalars
from ...core.animation import capture_frame

try:
    import cv2
    _has_cv2 = True
except ImportError:
    _has_cv2 = False


def _gblur(a: np.ndarray, sigma: float) -> np.ndarray:
    """Gaussian blur with a numpy separable fallback when cv2 is absent."""
    if sigma <= 0.0:
        return a.astype(np.float32)
    if _has_cv2:
        return cv2.GaussianBlur(a, (0, 0), sigmaX=sigma, sigmaY=sigma).astype(np.float32)
    rad = max(1, int(sigma * 3))
    x = np.arange(-rad, rad + 1, dtype=np.float32)
    k = np.exp(-(x ** 2) / (2.0 * sigma * sigma))
    k /= k.sum()
    out = a.astype(np.float32)
    if out.ndim == 2:
        out = out[..., None]
    res = np.empty_like(out)
    for c in range(out.shape[-1]):
        ch = out[..., c]
        pad = np.pad(ch, ((0, 0), (rad, rad)), mode="edge")
        tmp = np.zeros_like(ch)
        for i, kv in enumerate(k):
            tmp += kv * pad[:, i:i + ch.shape[1]]
        pad2 = np.pad(tmp, ((rad, rad), (0, 0)), mode="edge")
        out2 = np.zeros_like(ch)
        for i, kv in enumerate(k):
            out2 += kv * pad2[i:i + ch.shape[0], :]
        res[..., c] = out2
    return res.reshape(a.shape) if a.ndim == 2 else res


@method(
    id="450",
    name="Coherence Shock Filter",
    category="filters",
    new_image_contract=True,
    tags=["npr", "shock-filter", "coherence-enhancing", "structure-tensor",
          "weickert", "osher-rudin", "pde", "woodcut", "sharpen", "animation"],
    inputs={"image_in": "IMAGE"},
    outputs={"image": "IMAGE", "mask": "MASK"},
    params={
        "source": {"description": "source (input_image/blobs/noise/gradient/palette/procedural)", "default": "blobs"},
        "iterations": {"description": "number of shock evolution steps", "min": 1, "max": 60, "default": 16},
        "dt": {"description": "PDE time step per iteration", "min": 0.05, "max": 1.0, "default": 0.4},
        "sigma": {"description": "gradient smoothing sigma (px) for shock term", "min": 0.5, "max": 8.0, "default": 1.5},
        "rho": {"description": "structure-tensor integration sigma (px)", "min": 1.0, "max": 20.0, "default": 5.0},
        "coherence": {"description": "anisotropic diffusion strength along coherence dir", "min": 0.0, "max": 1.0, "default": 0.35},
        "noise_amp": {"description": "noise amplitude for generated sources", "min": 0.1, "max": 1.0, "default": 0.5},
        "blur_sigma": {"description": "gaussian blur sigma for noise/blobs source", "min": 3, "max": 60, "default": 14},
        "palette": {"description": "palette name for palette source", "default": "vapor"},
        "anim_mode": {"description": "animation mode (none/reveal/breathe)", "choices": ["none", "reveal", "breathe"], "default": "none"},
        "anim_speed": {"description": "animation speed multiplier", "min": 0.1, "max": 5.0, "default": 1.0},
        "time": {"description": "animation time in radians", "min": 0.0, "max": 6.2832, "default": 0.0},
    },
)
def method_shock_filter(out_dir: Path, seed: int, params=None):
    """Coherence-Enhancing Shock Filter — Weickert (2003), Osher & Rudin (1990).

    The classic shock filter (Osher & Rudin, SIAM J. Numer. Anal. 1990) evolves
    an image toward a piecewise-constant "cartoon" with infinitely sharp edges
    (shocks) by a hyperbolic PDE:

        u_t = -sign(u_ηη) · |∇u|

    where u_ηη is the second derivative along the gradient direction η. Near an
    edge, the sign of u_ηη flips at the inflection point, so dilation and erosion
    meet exactly at the edge and steepen it into a shock. This is deblurring by
    backward diffusion, stabilised by the |∇u| term.

    Weickert's coherence-enhancing variant (2003) replaces the raw gradient
    direction with the dominant eigenvector of the *structure tensor* smoothed at
    scale ρ, so the shock respects coherent flow structures rather than pixel
    noise. An optional anisotropic diffusion term along the coherence direction v
    (perpendicular to the gradient) suppresses cross-flow noise and yields the
    characteristic flowing, woodcut / hatched-ink look.

    Pipeline per step (orientation from luminance, shock applied per channel):
        1. Structure tensor J_ρ = G_ρ * (∇u_σ ⊗ ∇u_σ)  from the luminance.
        2. Major eigenvector (η) and minor eigenvector (v) of the 2×2 tensor.
        3. u_ηη via the smoothed second-derivative projected onto η.
        4. u ← u − dt · sign(u_ηη) · |∇u|            (shock)
        5. u ← u + dt·coherence · u_vv               (coherence diffusion)

    This CPU path is the authoritative fp64 export, deterministic in the seed.

    Params:
        source:     input_image/blobs/noise/gradient/palette/procedural
        iterations: shock evolution steps (1-60, default 16)
        dt:         PDE time step (0.05-1, default 0.4)
        sigma:      gradient smoothing sigma px (0.5-8, default 1.5)
        rho:        structure-tensor integration sigma px (1-20, default 5)
        coherence:  anisotropic diffusion along coherence dir (0-1, default 0.35)
        noise_amp:  noise amplitude for generated sources
        blur_sigma: gaussian blur sigma for noise/blobs source
        palette:    palette name for palette source
        anim_mode:  none/reveal/breathe
        anim_speed: animation speed multiplier (0.1-5, default 1)
        time:       animation time in radians (0-6.28)
    """
    if params is None:
        params = {}
    seed_all(seed)
    rng = np.random.default_rng(seed)

    source = str(params.get("source", "blobs"))
    iterations = int(params.get("iterations", 16))
    dt = float(params.get("dt", 0.4))
    sigma = float(params.get("sigma", 1.5))
    rho = float(params.get("rho", 5.0))
    coherence = float(params.get("coherence", 0.35))
    noise_amp = float(params.get("noise_amp", 0.5))
    blur_sigma = float(params.get("blur_sigma", 14))
    pal_name = str(params.get("palette", "vapor"))
    anim_mode = str(params.get("anim_mode", "none"))
    anim_speed = float(params.get("anim_speed", 1.0))
    t = float(params.get("time", 0.0)) * anim_speed

    # ── Animation ──
    reveal_frac = 0.0
    if anim_mode == "reveal":
        reveal_frac = 0.5 - 0.5 * math.cos(t)
    elif anim_mode == "breathe":
        # smooth 0.5x..1.5x sweep of iterations (no cusps) — deeper shocks pulse
        iterations = max(1, int(round(iterations * (1.0 + 0.5 * math.sin(t)))))

    # ── Build source (values in [0,1]) ──
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    wired = params.get("_input_image")
    if wired is not None:
        arr = np.asarray(wired, dtype=np.float32)
        if arr.ndim == 3 and arr.shape[-1] >= 3:
            base = arr[..., :3]
        elif arr.ndim == 2:
            base = np.repeat(arr[..., None], 3, axis=-1)
        else:
            base = np.asarray(arr, dtype=np.float32).reshape(H, W, 1)
            base = np.repeat(base, 3, axis=-1)
        base = base.clip(0.0, 1.0)
    elif source == "blobs":
        # Soft colour blobs — the classic shock-filter demo: smooth ramps that
        # steepen into sharp cell boundaries under the PDE.
        n = rng.standard_normal((H, W, 3)).astype(np.float32) * noise_amp + 0.5
        n = _gblur(n, blur_sigma)
        base = norm(n).astype(np.float32)
    elif source == "gradient":
        r = np.sqrt((xx - W / 2) ** 2 + (yy - H / 2) ** 2)
        g = norm(r)
        base = np.stack([g, np.roll(g, 40, 1), 1.0 - g], -1).astype(np.float32)
    elif source == "palette":
        pal = PALETTES.get(pal_name, PALETTES.get("vapor", [(20, 20, 20), (235, 235, 235)]))
        pal_arr = np.array(pal, dtype=np.float32) / 255.0
        n = rng.standard_normal((H, W, 3)).astype(np.float32) * noise_amp + 0.5
        n = _gblur(n, blur_sigma)
        nl = norm(n[..., 0])
        idx = (nl * (len(pal_arr) - 1)).astype(np.int32)
        base = pal_arr[idx][..., :3].astype(np.float32)
    elif source == "procedural":
        g = np.sin(xx * 0.025 + yy * 0.018) * np.cos(xx * 0.017 - yy * 0.028) * 0.5 + 0.5
        base = np.stack([g, np.sqrt(g), 1.0 - g * 0.7], -1).astype(np.float32)
    else:  # noise
        n = rng.standard_normal((H, W, 3)).astype(np.float32) * noise_amp + 0.5
        n = _gblur(n, blur_sigma)
        base = norm(n).astype(np.float32)

    base = base.clip(0.0, 1.0).astype(np.float32)
    if base.ndim == 2:
        base = np.repeat(base[..., None], 3, axis=-1)
    if base.shape[-1] == 1:
        base = np.repeat(base, 3, axis=-1)
    base = base.astype(np.float32)

    # ── Shock filter evolution ──
    u = base.astype(np.float64).copy()
    eps = 1e-9

    for _ in range(max(1, iterations)):
        # Luminance for orientation (shared across channels for coherence)
        lum = (0.299 * u[..., 0] + 0.587 * u[..., 1] + 0.114 * u[..., 2])
        lum_s = _gblur(lum.astype(np.float32), sigma).astype(np.float64)

        gx = np.gradient(lum_s, axis=1)
        gy = np.gradient(lum_s, axis=0)

        # Structure tensor components, integrated at scale rho
        Jxx = _gblur((gx * gx).astype(np.float32), rho).astype(np.float64)
        Jxy = _gblur((gx * gy).astype(np.float32), rho).astype(np.float64)
        Jyy = _gblur((gy * gy).astype(np.float32), rho).astype(np.float64)

        # Major eigenvector (eta) of the 2x2 symmetric tensor
        tr = Jxx + Jyy
        det_term = np.sqrt(np.maximum((Jxx - Jyy) ** 2 + 4.0 * Jxy ** 2, 0.0))
        # eigenvector for the larger eigenvalue lam1 = (tr + det_term)/2
        # (lam1 - Jyy, Jxy) is an eigenvector direction
        ex = (tr + det_term) * 0.5 - Jyy
        ey = Jxy
        enorm = np.sqrt(ex * ex + ey * ey) + eps
        nx = ex / enorm
        ny = ey / enorm
        # coherence (perpendicular) direction
        vx = -ny
        vy = nx

        # Second derivatives (on lightly smoothed luminance for stability)
        uxx = np.gradient(gx, axis=1)
        uyy = np.gradient(gy, axis=0)
        uxy = np.gradient(gx, axis=0)

        # directional second derivative along eta and along v
        u_ee = nx * nx * uxx + 2.0 * nx * ny * uxy + ny * ny * uyy
        u_vv = vx * vx * uxx + 2.0 * vx * vy * uxy + vy * vy * uyy

        s = np.sign(u_ee)  # shock direction from the luminance structure

        for c in range(3):
            cgx = np.gradient(u[..., c], axis=1)
            cgy = np.gradient(u[..., c], axis=0)
            gmag = np.sqrt(cgx * cgx + cgy * cgy)
            # shock: erode/dilate toward the edge
            u[..., c] = u[..., c] - dt * s * gmag
            # coherence-enhancing diffusion along v (uses shared u_vv from lum)
            if coherence > 0.0:
                u[..., c] = u[..., c] + dt * coherence * u_vv
        u = np.clip(u, 0.0, 1.0)

    result = u.astype(np.float32)

    # ── Reveal animation: crossfade original -> shock ──
    if anim_mode == "reveal":
        result = (base * (1.0 - reveal_frac) + result * reveal_frac).astype(np.float32)

    result = np.clip(result, 0.0, 1.0).astype(np.float32)

    # ── Edge mask from the final shock gradient (Rule 10: spatial selection) ──
    fl = 0.299 * result[..., 0] + 0.587 * result[..., 1] + 0.114 * result[..., 2]
    egx = np.gradient(fl, axis=1)
    egy = np.gradient(fl, axis=0)
    mask = norm(np.sqrt(egx * egx + egy * egy)).astype(np.float32)
    try:
        from ...core.utils import write_mask
        write_mask(out_dir, mask)
    except Exception:
        pass

    # ── Scalars ──
    out_std = float(result.std())
    edge_frac = float((mask > 0.25).mean())
    write_scalars(out_dir, out_std=out_std, edge_fraction=edge_frac,
                  iterations=float(iterations))

    capture_frame("450", result)
    save(result, mn(450, f"Coherence Shock it={iterations} c={coherence:.2f}"), out_dir)
    return result
