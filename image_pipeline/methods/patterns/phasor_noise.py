"""Phasor Noise — aperiodic band-limited procedural noise (Gilet et al. 2017).

Phasor noise is a successor to Gabor noise that removes the visible grid
artifacts of a regular impulse lattice. Following Gilet, Sauvage, Mesmoudi &
Dischler, "Phasor Noise", SIGGRAPH 2017 (doi:10.1145/3072959.3073642) and its
procedural extension Tricard, Hédelin & Neyret, "Procedural Phasor Noise",
EG 2019, each phasor (band-limited, oriented, windowed sinusoid) is placed on a
*jittered* lattice and — crucially — given its OWN random spatial frequency
within a band. The sum is therefore aperiodic: no global repetition, no
Moiré-like lattice, just natural-looking structured noise. The companion GPU
twin is node 322 (`phasor_noise_gpu`); this CPU node is the authoritative export
path (GPU-First contract: CPU fn stays authoritative).

Closed form per pixel (sum over neighbouring phasors p):

    f(x) = Σ_p  a_p · cos(2π · f_p · u_p(x) + φ_p) · w(‖x − x_p‖)

    u_p(x) = (R(−θ_p) (x − x_p))_x        # projection onto phasor orientation
    w(d)   = exp(−d² / 2σ²)               # local Gaussian window (~ one cell)

Because each phasor carries a random f_p, the field is aperiodic. This is the
key difference from the pipeline's Gabor Noise node (473), which uses a single
shared frequency across a regular lattice.

Animation modes (Architecture B — per-frame re-call with `time`):
    none  — static (phasor lattice fixed by seed): frame Δ ≈ 0.
    boil  — every phasor's phase drifts at a rate ∝ its frequency, so the noise
            "boils" coherently (higher frequencies churn faster). Continuous,
            high Δ, never a false-negative at the audit sample times.
    drift — the whole lattice translates, so features flow across the canvas.
    spin  — the field rotates about its centre (non-symmetry-aligned rate).

Phasor positions / angles / frequencies / base phases are FIXED per seed; only
the time-driven term changes between frames, so animation stays coherent and
there is no per-frame re-randomisation (no t-shadowing trap).
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np

from ...core.registry import method
from ...core.utils import (
    save, mn, seed_all, W, H, write_mask, write_scalars,
)
from ...core.animation import capture_frame


def _hsl_to_rgb(h: float, s: float, l: float):
    h = h % 1.0
    c = (1.0 - abs(2.0 * l - 1.0)) * s
    x = c * (1.0 - abs((h * 6.0) % 2.0 - 1.0))
    m = l - c / 2.0
    if h < 1.0 / 6.0:
        r, g, b = c, x, 0.0
    elif h < 2.0 / 6.0:
        r, g, b = x, c, 0.0
    elif h < 3.0 / 6.0:
        r, g, b = 0.0, c, x
    elif h < 4.0 / 6.0:
        r, g, b = 0.0, x, c
    elif h < 5.0 / 6.0:
        r, g, b = x, 0.0, c
    else:
        r, g, b = c, 0.0, x
    return r + m, g + m, b + m


@method(
    id="959",
    name="Phasor Noise",
    category="patterns",
    new_image_contract=True,
    tags=["noise", "procedural", "phasor", "gabor", "generative", "animation"],
    inputs={},
    outputs={"image": "IMAGE", "mask": "MASK"},
    params={
        "scale": {"description": "base feature size in px (lattice cell)", "min": 8.0, "max": 80.0, "default": 32.0},
        "frequency": {"description": "phasor spatial frequency (× cell size)", "min": 0.5, "max": 4.0, "default": 1.5},
        "octaves": {"description": "number of summed frequency bands", "min": 1.0, "max": 4.0, "default": 2.0},
        "gain": {"description": "contrast / brightness gain", "min": 0.2, "max": 3.0, "default": 1.0},
        "color_mode": {"description": "output colouring (mono/rgb/tint)", "default": "mono"},
        "hue": {"description": "base hue for tint mode", "min": 0.0, "max": 1.0, "default": 0.55},
        "time": {"description": "animation phase [0, 2pi)", "min": 0.0, "max": 6.28, "default": 0.0},
        "anim_mode": {"description": "animation mode (none/boil/drift/spin)", "choices": ["none", "boil", "drift", "spin"], "default": "none"},
        "anim_speed": {"description": "animation speed multiplier", "min": 0.1, "max": 5.0, "default": 1.0},
    },
)
def method_phasor_noise(out_dir: Path, seed: int, params=None):
    """Phasor Noise — aperiodic band-limited procedural noise (Gilet et al. 2017).

    A jittered lattice of oriented, windowed, band-limited sinusoids ("phasors"),
    each with its OWN random frequency, summed into a coherent aperiodic field.
    Distinct from Gabor Noise (473): the per-phasor random frequency kills the
    regular lattice repetition. CPU path is authoritative; GPU twin is node 322.

    Params:
        scale:      lattice cell size (px) — base feature scale
        frequency:  phasor frequency as a multiple of 1/cell
        octaves:    number of summed frequency bands (1..4)
        gain:       contrast / brightness gain
        color_mode: mono (grayscale) / rgb (3 independent fields) / tint (hue ramp)
        hue:        base hue for tint mode
        time:       animation phase [0, 2pi)
        anim_mode:  none / boil / drift / spin
        anim_speed: animation speed (0.1-5.0)
    """
    try:
        if params is None:
            params = {}
        t = float(params.get("time", 0.0))
        anim_mode = str(params.get("anim_mode", "none"))
        anim_speed = float(params.get("anim_speed", 1.0))
        seed_all(seed)

        scale = max(8.0, min(80.0, float(params.get("scale", 32.0))))
        frequency = max(0.5, min(4.0, float(params.get("frequency", 1.5))))
        octaves = int(max(1, min(4, round(float(params.get("octaves", 2.0))))))
        gain = max(0.2, min(3.0, float(params.get("gain", 1.0))))
        color_mode = str(params.get("color_mode", "mono"))
        hue = max(0.0, min(1.0, float(params.get("hue", 0.55))))

        # ── Animation clock (rename t to avoid shadowing the time param) ──
        _t = t * anim_speed

        ncx = max(1, int(round(W / scale)))
        ncy = max(1, int(round(H / scale)))
        cellw = W / ncx
        cellh = H / ncy
        cell_max = max(cellw, cellh)
        sigma = cell_max * 0.55
        inv2s2 = 1.0 / (2.0 * sigma * sigma)

        # ── Build the phasor lattice (fixed per seed) ──
        def _build_phasors(rng_seed: int):
            rng = np.random.default_rng(rng_seed)
            px, py, th, fp, ph, am = [], [], [], [], [], []
            for gy in range(ncy):
                for gx in range(ncx):
                    jx = (gx + 0.5 + (rng.random() * 2 - 1) * 0.5) * cellw
                    jy = (gy + 0.5 + (rng.random() * 2 - 1) * 0.5) * cellh
                    theta = rng.random() * math.pi
                    phase0 = rng.random() * 2.0 * math.pi
                    amp = 0.6 + 0.4 * rng.random()
                    for octv in range(octaves):
                        fmul = frequency * (2.0 ** octv)
                        fpx = fmul / cell_max
                        px.append(jx); py.append(jy)
                        th.append(theta)
                        fp.append(fpx)
                        ph.append(phase0 + octv * 1.7)
                        am.append(amp / (octv + 1))
            return (np.array(px, np.float32), np.array(py, np.float32),
                    np.array(th, np.float32), np.array(fp, np.float32),
                    np.array(ph, np.float32), np.array(am, np.float32))

        # Animation parameters (applied without re-randomising the lattice)
        drift_x = drift_y = 0.0
        spin_a = 0.0
        if anim_mode == "drift":
            drift_x = _t * 12.0
            drift_y = _t * 8.0
        elif anim_mode == "spin":
            spin_a = _t * 0.7
        # boil: per-phasor phase drift handled inside _field via _t and fpx

        # ── Coordinate grid (optionally rotated for spin) ──
        ys, xs = np.mgrid[0:H, 0:W].astype(np.float32)
        cx, cy = W / 2.0, H / 2.0
        if spin_a != 0.0:
            ca_s = math.cos(-spin_a)
            sa_s = math.sin(-spin_a)
            dxc = xs - cx
            dyc = ys - cy
            xs = cx + dxc * ca_s - dyc * sa_s
            ys = cy + dxc * sa_s + dyc * ca_s

        def _field(rng_seed: int):
            PX, PY, TH, FP, PH, AM = _build_phasors(rng_seed)
            N = PX.shape[0]
            ca = np.cos(TH)
            sa = np.sin(TH)
            field = np.zeros((H, W), dtype=np.float32)
            # precompute drift-shifted centres
            cxr = PX + drift_x
            cyr = PY + drift_y
            for i in range(N):
                dx = xs - cxr[i]
                dy = ys - cyr[i]
                u = dx * ca[i] + dy * sa[i]          # projection onto orientation
                d2 = dx * dx + dy * dy
                w = np.exp(-d2 * inv2s2)
                if anim_mode == "boil":
                    # higher-frequency phasors churn faster -> coherent boil
                    pe = PH[i] + _t * (1.0 + 6.0 * FP[i] * cell_max)
                else:
                    pe = PH[i]
                field += AM[i] * np.cos(2.0 * math.pi * FP[i] * u + pe) * w
            # normalise to ~unit std, centre at 0.5
            sd = field.std()
            if sd < 1e-6:
                sd = 1.0
            field = field / sd
            return np.clip(0.5 + 0.5 * field * gain, 0.0, 1.0).astype(np.float32)

        if color_mode == "rgb":
            r = _field(seed)
            g = _field(seed + 1)
            b = _field(seed + 2)
            rgb = np.stack([r, g, b], axis=-1)
            mask = (0.299 * r + 0.587 * g + 0.114 * b).astype(np.float32)
        elif color_mode == "tint":
            v = _field(seed)
            # hue ramp driven by the noise value -> iridescent tint
            rr, gg, bb = _hsl_to_rgb(hue, 0.85, 0.5)
            base = np.array([rr, gg, bb], np.float32)
            rgb = (v[..., None] * base[None, None, :] + (1.0 - v[..., None]) * 0.06).astype(np.float32)
            mask = v.astype(np.float32)
        else:  # mono
            v = _field(seed)
            rgb = np.stack([v, v, v], axis=-1).astype(np.float32)
            mask = v.astype(np.float32)

        capture_frame("959", rgb)
        save(rgb, mn(959, "Phasor Noise"), out_dir)
        try:
            write_scalars(
                out_dir,
                scale=float(scale),
                frequency=float(frequency),
                octaves=float(octaves),
                gain=float(gain),
                mean=float(rgb[..., :3].mean()),
                std=float(rgb[..., :3].std()),
            )
            write_mask(out_dir, mask)
        except Exception:
            pass
        return rgb
    except Exception as exc:
        fallback = np.full((H, W, 3), 0.5, dtype=np.float32)
        save(fallback, mn(959, "Phasor Noise"), out_dir)
        print(f"[method_959] ERROR: {exc}")
        return fallback
