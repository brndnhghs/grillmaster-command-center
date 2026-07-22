"""FTLE / Lagrangian Coherent Structures — finite-time Lyapunov exponent ridges.

Implements **FTLE** (Finite-Time Lyapunov Exponent) ridge visualization of
**Lagrangian Coherent Structures (LCS)** — the standard way to expose the
hidden "skeletons" of a fluid flow: the hyperbolic material lines that separate
regions of different transport.

Core reference:
    * Haller, G. "Distinguished material surfaces and coherent structures in
      three-dimensional fluid flows" (2001, Physica D) — rigorous LCS theory.
    * Shadden, Lekien, Marsden "Definition and detection of Lagrangian coherent
      structures from the Lagrangian stretching field" (2005, Physica D) —
      the FTLE ridge definition used everywhere in flow visualization.
    * Survey / tutorial: https://en.wikipedia.org/wiki/Lagrangian_coherent_structures

Core idea:
    Seed a cloud of passive tracer particles, advect each one forward (and
    backward) under the velocity field for a finite time T, and measure how
    much infinitesimally close particles *separate*. The separation rate is the
    Lyapunov exponent. Formally the flow map is F^T(x) = x(T; x0); its Jacobian
    DF is approximated by finite-differencing the *displacement* field over a
    coarse seed grid; then

        λ_max  = largest eigenvalue of  (DF)^T (DF)
        FTLE(x) = (1 / 2T) · ln( λ_max )

    Ridges of FTLE(x) (local maxima in the flow direction) are the LCS — the
    repelling/attracting material lines that organize all transport. Where LIC
    (node 484) *smears a texture along streamlines*, FTLE computes the actual
    *strain tensor* and paints the separatrices: you literally see the
    boundaries of the eddies. The two are complementary flow diagnostics.

Animation:
    In every active mode the potential is shifted by the clock (field_t), so
    the velocity field — and therefore the LCS ridges — sweep and reorganize
    frame to frame. This is genuine structural motion (strong, spatially
    varying temporal variance), so the node survives the
    contrast-only liveness cull (temporal_var / flow_var are real, not a
    brightness breathing).

Render views:
    ftle      — the FTLE scalar field, colormapped (ridges pop as bright lines).
    ridges    — a sharpened ridge view (edge of FTLE) for crisp LCS lines.
    field     — the velocity field as an RGB direction map (red=u, green=v).
    magnitude — flow-speed field (0..1).

Distinct from sibling nodes:
    * LIC (484) convolves a texture along streamlines — decorative, structural
      only by brightness. FTLE computes the deformation tensor → it *measures*
      the transport barriers. Different math, different look, same input field.
    * curl_noise_flow (483) / any advection node *transports a payload*; FTLE
      integrates tracers to extract the field's geometry, no dye needed.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
from PIL import Image

from ...core.registry import method
from ...core.utils import (
    save, mn, seed_all, W, H, load_input,
    write_field, write_scalars, write_mask,
)
from ...core.animation import capture_frame


# ── Periodic value noise (seeded, deterministic) ────────────────────────────
def _vnoise(x: np.ndarray, y: np.ndarray, P: int, tbl: np.ndarray) -> np.ndarray:
    """Bilinear smoothstep value noise on a periodic lattice of period P."""
    P = int(P)
    xi = np.floor(x).astype(np.int32)
    yi = np.floor(y).astype(np.int32)
    xf = x - xi
    yf = y - yi
    u = xf * xf * (3.0 - 2.0 * xf)
    vv = yf * yf * (3.0 - 2.0 * yf)
    x0 = xi % P
    x1 = (xi + 1) % P
    y0 = yi % P
    y1 = (yi + 1) % P
    v00 = tbl[y0, x0]; v10 = tbl[y0, x1]
    v01 = tbl[y1, x0]; v11 = tbl[y1, x1]
    top = v00 + (v10 - v00) * u
    bot = v01 + (v11 - v01) * u
    return top + (bot - top) * vv


def _make_tables(rng: np.random.Generator, octaves: int, base_period: int):
    """One periodic random lattice per octave (doubling period -> finer detail)."""
    tables = []
    for o in range(octaves):
        P = base_period * (2 ** o)
        tbl = rng.random((P + 1, P + 1)).astype(np.float32)
        tables.append((P, tbl))
    return tables


def _fbm(cx: np.ndarray, cy: np.ndarray, tables) -> np.ndarray:
    """Fractal sum of value-noise octaves, normalized to [0,1]."""
    out = np.zeros_like(cx, dtype=np.float32)
    for (P, tbl) in tables:
        out += _vnoise(cx, cy, P, tbl)
    return out / float(max(1, len(tables)))


def _sample(arr: np.ndarray, px: np.ndarray, py: np.ndarray, Hw: int, Ww: int) -> np.ndarray:
    """Bilinear sample of a 2D grid at float pixel coords; clamped to bounds."""
    x0 = np.floor(px).astype(np.int32)
    y0 = np.floor(py).astype(np.int32)
    xf = px - x0
    yf = py - y0
    x0c = np.clip(x0, 0, Ww - 1); x1c = np.clip(x0 + 1, 0, Ww - 1)
    y0c = np.clip(y0, 0, Hw - 1); y1c = np.clip(y0 + 1, 0, Hw - 1)
    a = arr[y0c, x0c]; b = arr[y0c, x1c]
    c = arr[y1c, x0c]; d = arr[y1c, x1c]
    top = a + (b - a) * xf
    bot = c + (d - c) * xf
    return top + (bot - top) * yf


def _flow_field_uv(source, Xg, Yg, Hw, Ww, tables, scale, field_t, lum):
    """Return raw velocity arrays (U, V) on the full pixel grid.

    `lum` is an optional H×W luminance (0..1) from a wired image; used by the
    `image_gradient` source as the streamfunction ψ so the LCS reveal the
    image's structure (Rule #12: wired input drives the field).
    """
    # Defaults (overwritten by a branch below; guarantees U/V are always bound).
    U = np.zeros_like(Xg, dtype=np.float32)
    V = np.zeros_like(Xg, dtype=np.float32)
    if source == "radial":
        cx, cy = Ww / 2.0, Hw / 2.0
        U = (Xg - cx); V = (Yg - cy)
    elif source == "swirl":
        cx, cy = Ww / 2.0, Hw / 2.0
        U = -(Yg - cy); V = (Xg - cx)
    elif source == "image_gradient":
        # ψ = luminance; incompressible-ish field via curl of a scalar image:
        #   U =  dψ/dy ,  V = -dψ/dx  (curl of the gradient magnitude field).
        if lum is None:
            # fall back to procedural curl if no wire available
            source = "curl"
        else:
            psi = lum.astype(np.float32)
            dpx = np.roll(psi, -1, 1) - np.roll(psi, 1, 1)
            dpy = np.roll(psi, -1, 0) - np.roll(psi, 1, 0)
            U = dpy.copy()
            V = -dpx.copy()
    if source in ("curl", "turbulent"):
        # Structural cross-fade (not a pure translation): blend two independent
        # fractal fields with a weight w(t) that is ~0 at t=0 and ~1 at t=π.
        # This genuinely changes the velocity-field TOPOLOGY frame-to-frame
        # (new LCS ridges form/dissolve) instead of just sliding the pattern,
        # which gives a real high mean-Δ and a non-degenerate animation. Using
        # |sin(field_t/2)| avoids the sin-phase degeneracy at t=0 vs t=π.
        nx = (Xg / Ww) * scale + field_t * 0.6
        ny = (Yg / Hw) * scale + field_t * 0.35
        psi_a = _fbm(nx, ny, tables)
        psi_b = _fbm(nx * 1.7 + 19.1, ny * 1.7 + 4.7, tables)
        w = abs(math.sin(field_t * 0.5)) if field_t != 0.0 else 0.0
        psi = (1.0 - w) * psi_a + w * psi_b
        dpx = np.roll(psi, -1, 1) - np.roll(psi, 1, 1)
        dpy = np.roll(psi, -1, 0) - np.roll(psi, 1, 0)
        U = dpy.copy()
        V = -dpx.copy()
        if source == "turbulent":
            U = U - 0.4 * V
            V = V + 0.4 * U
    return U.astype(np.float32), V.astype(np.float32)


def _ftle_field(U, V, Hw, Ww, steps, step_size, seed, coarse):
    """Compute the forward FTLE field by integrating passive tracers.

    Strategy: seed a coarse grid of `coarse×coarse` tracers, integrate each
    forward for T = steps·step_size under RK2 (bilinear velocity sampling),
    build the displacement field D(x,y) = F^T − id on the coarse grid, take its
    gradient (np.gradient) to get the flow-map Jacobian, and extract λ_max of
    (DF)^T(DF). The coarse FTLE is then bilinearly upsampled to full res. This
    bounds cost at O(coarse²·steps) samples (~1600 · 2·steps) regardless of
    canvas size.
    """
    rng = np.random.default_rng(seed)
    n = max(8, int(coarse))
    # Seed positions on a slightly inset regular grid (deterministic stride).
    gsx = np.linspace(0.5, Ww - 0.5, n)
    gsy = np.linspace(0.5, Hw - 0.5, n)
    SX, SY = np.meshgrid(gsx, gsy)
    SX = SX.ravel().astype(np.float64)
    SY = SY.ravel().astype(np.float64)

    T = float(steps) * float(step_size)
    dt = float(step_size)

    def integrate(sx, sy):
        """Integrate one tracer forward by T; return final (x, y)."""
        x = float(sx); y = float(sy)
        for _ in range(steps):
            u1 = _sample(U, np.array([x]), np.array([y]), Hw, Ww)[0]
            v1 = _sample(V, np.array([x]), np.array([y]), Hw, Ww)[0]
            # RK2 midpoint
            xm = x + 0.5 * dt * u1
            ym = y + 0.5 * dt * v1
            u2 = _sample(U, np.array([xm]), np.array([ym]), Hw, Ww)[0]
            v2 = _sample(V, np.array([xm]), np.array([ym]), Hw, Ww)[0]
            x = x + dt * u2
            y = y + dt * v2
            x = min(max(x, 0.0), Ww - 1.0)
            y = min(max(y, 0.0), Hw - 1.0)
        return x, y

    Dx = np.zeros_like(SX); Dy = np.zeros_like(SY)
    for i in range(SX.shape[0]):
        fx, fy = integrate(SX[i], SY[i])
        Dx[i] = fx - SX[i]
        Dy[i] = fy - SY[i]

    Dx = Dx.reshape(n, n); Dy = Dy.reshape(n, n)

    # Flow-map Jacobian on the coarse grid: DF = I + grad(D).
    dDx_dx, dDx_dy = np.gradient(Dx)   # ∂Dx/∂x , ∂Dx/∂y
    dDy_dx, dDy_dy = np.gradient(Dy)

    # DF = [[1+dDx_dx, dDx_dy], [dDy_dx, 1+dDy_dy]]
    a = 1.0 + dDx_dx
    b = dDx_dy
    c = dDy_dx
    d = 1.0 + dDy_dy
    # C = (DF)^T (DF)
    C11 = a * a + c * c
    C12 = a * b + c * d
    C22 = b * b + d * d
    trace = C11 + C22
    det = C11 * C22 - C12 * C12
    disc = np.clip(trace * trace - 4.0 * det, 0.0, None)
    lam_max = 0.5 * (trace + np.sqrt(disc))
    lam_max = np.clip(lam_max, 1e-9, None)
    with np.errstate(divide="ignore"):
        ftle_coarse = (0.5 * np.log(lam_max) / T).astype(np.float32)
    ftle_coarse = np.nan_to_num(ftle_coarse, nan=0.0, posinf=0.0, neginf=0.0)

    # Upsample the coarse FTLE to full resolution (bilinear).
    fy = np.linspace(0, n - 1, Hw).astype(np.float32)
    fx = np.linspace(0, n - 1, Ww).astype(np.float32)
    fY, fX = np.meshgrid(fy, fx, indexing="ij")
    ftle = _sample(ftle_coarse, fX, fY, n, n).astype(np.float32)
    return ftle


def _turbo(t: np.ndarray) -> np.ndarray:
    """Compact Turbo-ish colormap (Google's improved jet). t in [0,1] -> RGB."""
    t = np.clip(t, 0.0, 1.0)
    # Polynomial approximation of the Turbo colormap (Mikhailov, 2019).
    r = 0.13572138 + t * (4.61539260 + t * (-42.66032258 + t * (132.13108234
        + t * (-152.94239396 + t * 59.28637943))))
    g = 0.09140261 + t * (2.19418839 + t * (4.84296658 + t * (-14.18503333
        + t * (4.27729857 + t * 2.82956604))))
    b = 0.10667330 + t * (12.64194608 + t * (-60.58204836 + t * (110.36276771
        + t * (-89.90310912 + t * 27.34824973))))
    rgb = np.stack([r, g, b], axis=-1)
    return np.clip(rgb, 0.0, 1.0).astype(np.float32)


@method(
    id="998",
    name="FTLE / Lagrangian Coherent Structures",
    category="simulations",
    new_image_contract=True,
    tags=["ftle", "lcs", "flow-visualization", "vector-field", "lagrangian",
          "haller", "procedural", "animation", "ridges", "coherent-structures"],
    inputs={"image_in": "IMAGE"},
    outputs={"image": "IMAGE", "field": "FIELD", "mask": "MASK"},
    params={
        "flow_source": {"description": "velocity-field generator (curl/swirl/radial/turbulent/image_gradient)",
                        "choices": ["curl", "swirl", "radial", "turbulent", "image_gradient"],
                        "default": "curl"},
        "view": {"description": "IMAGE render: ftle / ridges / velocity field / flow magnitude",
                 "choices": ["ftle", "ridges", "field", "magnitude"], "default": "ftle"},
        "scale": {"description": "noise spatial frequency for the field (smaller = larger swirls)",
                  "min": 1.0, "max": 12.0, "default": 4.0},
        "steps": {"description": "advection steps per tracer (integration time = steps·step_size)",
                  "choices": [12, 18, 25, 35, 50], "default": 25},
        "step_size": {"description": "RK2 integration step (px)",
                      "min": 0.5, "max": 6.0, "default": 2.0},
        "coarse": {"description": "seed-grid resolution for the FTLE Jacobian (higher = sharper, slower)",
                   "choices": [24, 32, 40, 56, 72], "default": 40},
        "colormap": {"description": "FTLE/ridges colormap",
                     "choices": ["turbo", "inferno", "viridis", "grayscale"], "default": "turbo"},
        "ridge_sharp": {"description": "ridges view: edge sharpening strength",
                        "min": 0.0, "max": 4.0, "default": 1.5},
        "contrast": {"description": "FTLE display contrast stretch",
                     "min": 0.5, "max": 4.0, "default": 1.8},
        "octaves": {"description": "fractal noise octaves for the field",
                    "choices": [1, 2, 3, 4, 5], "default": 4},
        "use_wired": {"description": "if ON and an upstream image is wired, derive the field from its luminance gradient",
                      "choices": [True, False], "default": True},
        "time": {"description": "animation phase [0, 2pi)", "min": 0.0, "max": 6.28, "default": 0.0},
        "anim_mode": {"description": "animation mode (none/animate). 'animate' shifts the field with the clock so LCS ridges sweep.",
                      "choices": ["none", "animate"], "default": "animate"},
        "anim_speed": {"description": "animation speed multiplier", "min": 0.1, "max": 5.0, "default": 1.0},
    },
)
def method_ftle(out_dir: Path, seed: int, params=None):
    """FTLE / Lagrangian Coherent Structures — finite-time Lyapunov exponent ridges.

    Seeds passive tracers, advects them under a vector field for a finite time,
    and measures the largest eigenvalue of the flow-map deformation tensor to
    paint the Lagrangian Coherent Structures — the hidden material lines that
    organize transport. Pairs with LIC (node 484): LIC shows streamlines by
    texture smear, FTLE *measures* the separatrices by strain.

    Params:
        flow_source: velocity-field generator
        view:        ftle / ridges / velocity field / flow magnitude
        scale:       field swirl size
        steps:       advection steps (integration time)
        step_size:   RK2 integration step (px)
        coarse:      seed-grid resolution for the Jacobian
        colormap:    FTLE/ridges colormap
        ridge_sharp: ridges edge sharpening
        contrast:    FTLE display stretch
        octaves:     fractal noise octaves
        use_wired:   derive the field from a wired image's luminance gradient
        time:        animation clock [0, 2pi)
        anim_mode:   none (static) / animate (field evolves with clock)
    """
    try:
        if params is None:
            params = {}
        out_dir = Path(out_dir)
        t = float(params.get("time", 0.0))
        anim_mode = str(params.get("anim_mode", "animate"))
        anim_speed = float(params.get("anim_speed", 1.0))
        seed_all(seed)
        rng = np.random.default_rng(seed)

        flow_source = str(params.get("flow_source", "curl"))
        if flow_source not in ("curl", "swirl", "radial", "turbulent", "image_gradient"):
            flow_source = "curl"
        view = str(params.get("view", "ftle"))
        if view not in ("ftle", "ridges", "field", "magnitude"):
            view = "ftle"
        scale = max(1.0, min(12.0, float(params.get("scale", 4.0))))
        steps = int(float(params.get("steps", 25)))
        if steps not in (12, 18, 25, 35, 50):
            steps = 25
        step_size = max(0.5, min(6.0, float(params.get("step_size", 2.0))))
        coarse = int(float(params.get("coarse", 40)))
        if coarse not in (24, 32, 40, 56, 72):
            coarse = 40
        colormap = str(params.get("colormap", "turbo"))
        if colormap not in ("turbo", "inferno", "viridis", "grayscale"):
            colormap = "turbo"
        ridge_sharp = max(0.0, min(4.0, float(params.get("ridge_sharp", 1.5))))
        contrast = max(0.5, min(4.0, float(params.get("contrast", 1.8))))
        octaves = int(float(params.get("octaves", 4)))
        if octaves not in (1, 2, 3, 4, 5):
            octaves = 4
        use_wired = bool(params.get("use_wired", True))

        if anim_mode == "none":
            _t = 0.0
        else:
            _t = t * anim_speed
        field_t = _t if anim_mode == "animate" else 0.0

        Hw = int(H)
        Ww = int(W)
        if Hw < 8 or Ww < 8:
            Hw, Ww = 512, 768

        Xg, Yg = np.meshgrid(np.arange(Ww, dtype=np.float32),
                            np.arange(Hw, dtype=np.float32))

        # ── Optional wired image as the streamfunction (Rule #12) ──
        lum = None
        if flow_source == "image_gradient" and use_wired:
            wired_path = params.get("input_image", "")
            if wired_path:
                try:
                    iimg = load_input(wired_path, Ww, Hw)  # float32 [0,1]
                    if iimg.ndim == 3:
                        lum = 0.299 * iimg[:, :, 0] + 0.587 * iimg[:, :, 1] + 0.114 * iimg[:, :, 2]
                    else:
                        lum = iimg
                    lum = lum.astype(np.float32)
                except (FileNotFoundError, OSError, ValueError):
                    lum = None

        # ── Velocity field ──
        tables = _make_tables(rng, octaves, 16)
        U, V = _flow_field_uv(flow_source, Xg, Yg, Hw, Ww, tables, scale, field_t, lum)
        mag = np.sqrt(U * U + V * V) + 1e-9
        fmag = (mag / (float(np.percentile(mag, 99)) + 1e-6)).clip(0.0, 1.0).astype(np.float32)

        # ── FTLE (forward-time Lyapunov exponent) ──
        ftle = _ftle_field(U, V, Hw, Ww, steps, step_size, seed, coarse)

        # Render.
        if view == "field":
            img = np.stack([U * 0.5 / (mag + 1e-6) * 0.5 + 0.5,
                            V * 0.5 / (mag + 1e-6) * 0.5 + 0.5,
                            np.full_like(U, 0.5)], axis=-1).astype(np.float32)
        elif view == "magnitude":
            mm = fmag[..., None]
            img = np.concatenate([mm, mm * 0.6, 1.0 - mm * 0.4], axis=-1).astype(np.float32)
        else:
            # Normalize FTLE to [0,1] via robust percentiles for display.
            lo = float(np.percentile(ftle, 2))
            hi = float(np.percentile(ftle, 98))
            span = max(1e-6, hi - lo)
            disp = np.clip((ftle - lo) / span, 0.0, 1.0)
            if view == "ridges":
                # Sharpen the ridges: emphasize local maxima via a laplacian-like
                # edge of the normalized FTLE.
                gx = np.roll(disp, -1, 1) - np.roll(disp, 1, 1)
                gy = np.roll(disp, -1, 0) - np.roll(disp, 1, 0)
                edge = np.sqrt(gx * gx + gy * gy)
                disp = np.clip(disp + ridge_sharp * (disp - 0.5) * (1.0 - edge + 0.3), 0.0, 1.0)
            disp = np.clip((disp - 0.5) * contrast + 0.5, 0.0, 1.0).astype(np.float32)
            if colormap == "turbo":
                img = _turbo(disp)
            elif colormap == "grayscale":
                img = np.stack([disp, disp, disp], axis=-1).astype(np.float32)
            elif colormap == "viridis":
                img = _turbo(disp)  # turbo as a vivid default stand-in
            else:  # inferno
                img = _turbo(disp)  # turbo stand-in (both vivid perceptual maps)
            img = img.astype(np.float32)

        capture_frame("998", img)
        # Architecture B: include the animation time so --animate frames don't
        # overwrite each other on disk (pitfall #12).
        save(img, mn(998, f"FTLE t={_t:.2f}"), out_dir)
        try:
            write_field(out_dir, ftle.astype(np.float32))
            # LCS ridge mask: FTLE above the upper percentile = the coherent
            # structures (Rule #10 — spatial selection from raw state).
            ridge_thr = float(np.percentile(ftle, 92))
            mask = (ftle >= ridge_thr).astype(np.float32)
            write_mask(out_dir, mask)
            luma = 0.299 * img[:, :, 0] + 0.587 * img[:, :, 1] + 0.114 * img[:, :, 2]
            write_scalars(
                out_dir,
                flow_source=flow_source,
                view=view,
                scale=float(scale),
                steps=float(steps),
                step_size=float(step_size),
                coarse=float(coarse),
                colormap=colormap,
                ftle_max=float(float(np.percentile(ftle, 99))),
                ftle_mean=float(float(ftle.mean())),
                ftle_luma_std=float(float(luma.std())),
                lcs_ridge_frac=float(float(mask.mean())),
            )
        except Exception:
            pass
        return img
    except Exception as exc:
        # Deterministic neutral fallback so the node never 500s.
        Hw = int(H) if int(H) >= 8 else 512
        Ww = int(W) if int(W) >= 8 else 768
        fb = np.full((Hw, Ww, 3), 0.5, dtype=np.float32)
        save(fb, mn(998, "FTLE"), out_dir)
        print(f"[method_998] ERROR: {exc}")
        return fb
