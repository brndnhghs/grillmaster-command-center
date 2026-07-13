from __future__ import annotations

import math

import numpy as np

from ...core.registry import method
from ...core.utils import (
    save, mn, seed_all, get_canvas, write_scalars, write_field,
)
from ...core.animation import capture_frame


# ── Conformal maps (complex-analysis domain warps) ───────────────────────────
# A conformal map f:C->C preserves angles locally, so straight grids stay
# orthogonal under the deformation — the signature "smooth, non-stretchy" warp
# used in texture/domain parameterization (e.g. the 2024 "Real-Time Conformal
# Maps and Parameterizations" work). Each map is a pure function of z (no
# per-frame state), so the node is a closed-form Architecture-B method.
def _conformal(z: np.ndarray, fn: str, a: complex) -> np.ndarray:
    if fn == "z2":
        return z * z
    if fn == "z3":
        return z * z * z
    if fn == "exp":
        return np.exp(z)
    if fn == "sin":
        return np.sin(z)
    if fn == "joukowsky":
        return z + 1.0 / z
    # default / "moebius": (z - a) / (1 - conj(a) z)  |a| < 1  -> disk to disk
    return (z - a) / (1.0 - np.conj(a) * z)


def _ss(a: float, b: float, x: np.ndarray) -> np.ndarray:
    t = np.clip((x - a) / (b - a), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def _pattern_field(w: np.ndarray, kind: str, freq: float) -> np.ndarray:
    """Scalar pattern in [0,1] sampled at the warped coordinate w."""
    wr = w.real * freq
    wi = w.imag * freq
    if kind == "checker":
        cx = np.floor(wr).astype(np.int64)
        cy = np.floor(wi).astype(np.int64)
        return ((cx + cy) & 1).astype(np.float64)
    if kind == "dots":
        fx = wr - np.floor(wr + 0.5)
        fy = wi - np.floor(wi + 0.5)
        d = np.sqrt(fx * fx + fy * fy)
        return 1.0 - np.clip(d / 0.5, 0.0, 1.0)
    if kind == "stripes":
        return 0.5 + 0.5 * np.sin(wr * math.pi)
    # default "grid": bright lines at integer coordinates
    fx = np.abs(np.mod(wr, 1.0) - 0.5)
    fy = np.abs(np.mod(wi, 1.0) - 0.5)
    return _ss(0.40, 0.49, np.maximum(fx, fy))


@method(id="503", name="Conformal Warp", category="patterns",
        tags=["procedural", "conformal", "complex-analysis", "warp", "domain",
              "animation", "gpu-twin-candidate"],
        inputs={},
        outputs={"image": "IMAGE", "luminance": "FIELD"},
        params={
    "function": {"description": "conformal map f(z) applied as the uv transform",
                 "choices": ["moebius", "z2", "z3", "exp", "sin", "joukowsky"],
                 "default": "moebius"},
    "scale": {"description": "domain radius (zoom of the complex plane)", "min": 0.5, "max": 8.0, "default": 3.0},
    "warp": {"description": "Möbius coefficient magnitude |a| (0=identity, 1=extreme)", "min": 0.0, "max": 0.95, "default": 0.6},
    "pattern": {"description": "base pattern sampled at the warped coordinate",
                "choices": ["grid", "checker", "dots", "stripes"], "default": "grid"},
    "colormode": {"description": "how the warped field is colored",
                  "choices": ["phase", "shade", "mono"], "default": "phase"},
    "anim_mode": {"description": "animation mode (none/rotate/drift/pulse)",
                  "choices": ["none", "rotate", "drift", "pulse"], "default": "none"},
    "anim_speed": {"description": "animation speed multiplier", "min": 0.1, "max": 3.0, "default": 1.0},
    "time": {"description": "animation phase [0, 2pi)", "min": 0.0, "max": 6.28, "default": 0.0},
})
def method_conformal_warp(out_dir, seed: int, params=None):
    """Conformal Warp — complex-domain warping of a base pattern.

    Technique: a *conformal map* f:C->C (complex analysis) is applied as a
    coordinate transform to the uv plane, and a base pattern (grid / checker /
    dots / stripes) is sampled at the warped coordinate w = f(z). Because
    conformal maps preserve angles, the warped pattern stays everywhere
    orthogonal — a smooth, non-stretchy domain deformation used for texture and
    parameterization (see "Real-Time Conformal Maps and Parameterizations",
    2024, and classic Möbius / Joukowsky mappings).

    Available maps:
      * ``moebius``   — (z-a)/(1-conj(a)z), |a|<1 : morphs the disk onto itself
      * ``z2`` / ``z3`` — complex power : folding / multi-fold symmetry
      * ``exp``       — log-spiral unwrapping
      * ``sin``       — periodic banding
      * ``joukowsky`` — z + 1/z : the classic airfoil / circle-to-slit map

    Closed-form per-frame field (Architecture B): the orchestrator re-calls the
    method with an increasing ``time``. Animation modes move the input plane
    (rotate / drift / pulse) and, for ``moebius``, orbit the coefficient ``a``.
    With ``anim_mode="none"`` the field is a pure function of (seed, params), so
    it is a static baseline (Δ ≈ 0) as required.
    """
    try:
        if params is None:
            params = {}
        t = float(params.get("time", 0.0))
        seed_all(seed)

        fn = params.get("function", "moebius")
        scale = float(params.get("scale", 3.0))
        warp = float(np.clip(params.get("warp", 0.6), 0.0, 0.95))
        pattern = params.get("pattern", "grid")
        colormode = params.get("colormode", "phase")
        anim_mode = params.get("anim_mode", "none")
        anim_speed = float(params.get("anim_speed", 1.0))

        # ── Architecture-B time wiring ──
        _t = 0.0 if anim_mode == "none" else t * anim_speed

        # ── Complex plane (centered, aspect-correct) ──
        cw, ch = get_canvas()
        Wpx, Hpx = int(cw), int(ch)
        aspect = Wpx / max(1, Hpx)
        yy, xx = np.mgrid[0:Hpx, 0:Wpx].astype(np.float64)
        zx = (xx / Wpx - 0.5) * 2.0 * scale * aspect
        zy = (yy / Hpx - 0.5) * 2.0 * scale
        z = zx + 1j * zy

        # ── Seed-sourced phase offset (seed wiring; harmless, static) ──
        seed_phase = (seed % 97) / 97.0 * 0.5

        # ── Animation: move the input plane (every map responds) ──
        if anim_mode == "rotate":
            z = z * np.exp(1j * _t)
        elif anim_mode == "drift":
            z = z + (_t * 0.6)
        elif anim_mode == "pulse":
            z = z * (1.0 + 0.3 * math.sin(_t))

        # ── Möbius coefficient a (orbits with time for extra life) ──
        if fn == "moebius" and anim_mode != "none":
            a = warp * np.exp(1j * _t * 0.5)
        else:
            a = warp  # real coefficient (|a| < 1)

        w = _conformal(z, fn, a)
        w = np.nan_to_num(w, nan=0.0, posinf=1e3, neginf=-1e3)

        # ── Sample base pattern at the warped coordinate ──
        pat = _pattern_field(w * np.exp(1j * seed_phase), pattern, 6.0)
        pat = np.clip(pat, 0.0, 1.0)

        # ── Color ──
        if colormode == "mono":
            g = pat
            rgb = np.stack([g, g, g], axis=-1)
        elif colormode == "shade":
            contour = 0.6 + 0.4 * np.sin(np.abs(w) * 2.5)
            g = np.clip(0.06 + 0.94 * pat * contour, 0.0, 1.0)
            rgb = np.stack([g, g, g], axis=-1)
        else:  # phase: hue from arg(w), brightened on pattern lines
            ang = np.angle(w) + seed_phase
            hue = (ang + math.pi) / (2.0 * math.pi)
            base = 0.5 + 0.5 * np.cos(
                2.0 * math.pi * (hue[:, :, None] + np.array([0.0, 0.33, 0.67])[None, None, :])
            )
            dark = (1.0 - pat)[:, :, None] * 0.05
            rgb = base * (0.25 + 0.75 * pat[:, :, None]) + dark

        rgb = rgb.astype(np.float32)

        # ── Provenance / fields (Rule 4 / Rule 5) ──
        write_scalars(out_dir,
                      mean=round(float(rgb.mean()), 4),
                      std=round(float(rgb.std()), 4),
                      warp=round(warp, 3), scale=round(scale, 3))
        write_field(out_dir, pat.astype(np.float32))

        capture_frame("503", rgb)
        save(rgb, mn(503, f"Conformal Warp t={_t:.2f}"), out_dir)
        return rgb
    except Exception as exc:
        cw, ch = get_canvas()
        fallback = np.zeros((int(ch), int(cw), 3), dtype=np.float32)
        save(fallback, mn(503, "Conformal Warp"), out_dir)
        print(f"[method_503] ERROR: {exc}")
        return fallback
