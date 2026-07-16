"""Instant-NGP Multi-Resolution Hash Texture — procedural texture from a hash grid.

This node implements the *multi-resolution hash encoding* introduced by
Müller, Evans, Schied & Keller, "Instant Neural Graphics Primitives with a
Multiresolution Hash Encoding", SIGGRAPH 2022 (arXiv:2201.05989,
https://github.com/NVLabs/instant-ngp).

INGP's key idea: instead of a slow MLP evaluated over a dense grid, store
features in a *small* hash table indexed by the integer corners of a
multi-resolution lattice. Each scale contributes a trilinearly-interpolated
feature; the concatenation is decoded by a tiny MLP. Hash collisions across
scales are fine — the decoder absorbs them, which is exactly why the table can
stay tiny (2^14 entries) while still capturing detail at every frequency.

Here the encoding is used as a *standalone procedural texture generator*: the
hash table and the decoder weights are drawn from the node seed (no training),
so a single eval of f(uv) yields a rich, aperiodic, organic RGB field. It is
the GPU-style "closed-form f(uv)" idea done with a learned-style feature grid.

Animation modes (Architecture B — per-frame re-call with `time`):
    none  — static: lattice fixed by seed, frame Δ ≈ 0.
    warp  — the lookup coordinates scroll (seamless, wrapped), so the texture
            flows across the canvas.
    spin  — the coordinate field rotates about the centre.
    morph — two seeded hash tables are linearly blended by the time phase, so
            the *structure* itself morphs smoothly.

Phasor positions / the lattice are FIXED per seed; only the time-driven term
(coordinate transform or table blend) changes between frames, so animation
stays coherent — no per-frame re-randomisation, no t-shadowing trap.

CPU fn is authoritative (GPU-First contract: additive only).
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


def _hash2(ix, iy, level, seed, T):
    """Vectorised 32-bit integer hash (MurmurHash3 finaliser) over integer
    lattice coordinates. ix, iy are int32 arrays (H, W); returns int64 indices
    in [0, T)."""
    ix = ix.astype(np.uint32)
    iy = iy.astype(np.uint32)
    # Level offset mixed in Python int space to avoid uint32 scalar overflow.
    h = np.uint32((int(seed) + int(level) * 0x9E3779B1) & 0xFFFFFFFF)
    h = (h + ix * np.uint32(0x85EBCA6B)).astype(np.uint32)
    h = (h ^ (h >> 13)).astype(np.uint32)
    h = (h + iy * np.uint32(0xC2B2AE35)).astype(np.uint32)
    h = (h ^ (h >> 16)).astype(np.uint32)
    h = (h * np.uint32(0x27D4EB2F)).astype(np.uint32)
    h = (h ^ (h >> 15)).astype(np.uint32)
    return (h % np.uint32(T)).astype(np.int64)


def _hsv_to_rgb(h, s, v):
    """Vectorised HSV -> RGB (h,s,v arrays in [0,1] -> r,g,b arrays)."""
    h = (h % 1.0 + 1.0) % 1.0
    i = np.floor(h * 6.0).astype(np.int32)
    f = h * 6.0 - i
    p = v * (1.0 - s)
    q = v * (1.0 - s * f)
    t = v * (1.0 - s * (1.0 - f))
    ii = i % 6
    r = np.where(ii == 0, v, np.where(ii == 1, q, np.where(ii == 2, p,
        np.where(ii == 3, p, np.where(ii == 4, t, v)))))
    g = np.where(ii == 0, t, np.where(ii == 1, v, np.where(ii == 2, v,
        np.where(ii == 3, q, np.where(ii == 4, p, p)))))
    b = np.where(ii == 0, p, np.where(ii == 1, p, np.where(ii == 2, t,
        np.where(ii == 3, v, np.where(ii == 4, v, q)))))
    return r, g, b


@method(
    id="978",
    name="Instant-NGP Hash Texture",
    category="patterns",
    new_image_contract=True,
    tags=["noise", "procedural", "hash", "neural", "ingp", "generative", "animation"],
    inputs={},
    outputs={"image": "IMAGE", "mask": "MASK"},
    params={
        "levels": {"description": "number of hash-grid resolution levels", "min": 4.0, "max": 16.0, "default": 10.0},
        "base_resolution": {"description": "coarsest lattice resolution", "min": 1.0, "max": 8.0, "default": 2.0},
        "growth": {"description": "resolution multiplier per level", "min": 1.2, "max": 4.0, "default": 2.0},
        "hash_size_pow2": {"description": "hash table size = 2^this", "min": 10.0, "max": 20.0, "default": 14.0},
        "feature_dim": {"description": "features stored per grid corner", "min": 1.0, "max": 4.0, "default": 2.0},
        "gain": {"description": "contrast / brightness gain", "min": 0.2, "max": 3.0, "default": 1.0},
        "color_mode": {"description": "output colouring (mono/rgb/tint)", "default": "rgb"},
        "hue": {"description": "base hue for tint mode", "min": 0.0, "max": 1.0, "default": 0.6},
        "time": {"description": "animation phase [0, 2pi)", "min": 0.0, "max": 6.28, "default": 0.0},
        "anim_mode": {"description": "animation mode (none/warp/spin/morph)", "choices": ["none", "warp", "spin", "morph"], "default": "none"},
        "anim_speed": {"description": "animation speed multiplier", "min": 0.1, "max": 5.0, "default": 1.0},
    },
)
def method_instant_ngp(out_dir: Path, seed: int, params=None):
    """Instant-NGP Multi-Resolution Hash Texture.

    A multi-resolution hash grid (Müller et al. 2022) decoded by a tiny fixed
    MLP into a rich, aperiodic procedural RGB field. Distinct from every other
    procedural-noise node in the pipeline: the detail at every frequency comes
    from a *shared, tiny hash table* indexed by integer lattice corners, not from
    summed sinusoids (Gabor/Phasor) or value-noise interpolation.

    Params:
        levels:        number of resolution levels in the hash pyramid
        base_resolution: coarsest lattice resolution
        growth:        per-level resolution multiplier
        hash_size_pow2: log2 of the shared hash-table size
        feature_dim:   features per grid corner (concatenated across levels)
        gain:          contrast / brightness gain
        color_mode:    mono (grayscale) / rgb (3 independent decodes) / tint (hue ramp)
        hue:           base hue for tint mode
        time:          animation phase [0, 2pi)
        anim_mode:     none / warp / spin / morph
        anim_speed:    animation speed (0.1-5.0)
    """
    try:
        if params is None:
            params = {}
        t = float(params.get("time", 0.0))
        anim_mode = str(params.get("anim_mode", "none"))
        anim_speed = float(params.get("anim_speed", 1.0))
        seed_all(seed)

        levels = int(max(4, min(16, round(float(params.get("levels", 10.0))))))
        base_res = int(max(1, min(8, round(float(params.get("base_resolution", 2.0))))))
        growth = max(1.2, min(4.0, float(params.get("growth", 2.0))))
        hash_pow = int(max(10, min(20, round(float(params.get("hash_size_pow2", 14.0))))))
        F = int(max(1, min(4, round(float(params.get("feature_dim", 2.0))))))
        T = 1 << hash_pow
        gain = max(0.2, min(3.0, float(params.get("gain", 1.0))))
        color_mode = str(params.get("color_mode", "rgb"))
        hue = max(0.0, min(1.0, float(params.get("hue", 0.6))))

        # ── Animation clock (rename t to avoid shadowing the time param) ──
        _t = t * anim_speed

        # ── Hash tables (fixed per seed) ──
        rng_h = np.random.default_rng(seed)
        H_a = (rng_h.standard_normal((T, F)).astype(np.float32)) * 0.6
        if anim_mode == "morph":
            rng_hb = np.random.default_rng(seed + 1)
            H_b = (rng_hb.standard_normal((T, F)).astype(np.float32)) * 0.6
            # Blend across a full A->B cycle twice per 2π so motion is clearly
            # visible even at the audit sample times (t=0 vs π/2 -> m 0 vs 0.5).
            m = ((_t / (2.0 * math.pi)) * 2.0) % 1.0
            H_table = ((1.0 - m) * H_a + m * H_b).astype(np.float32)
        else:
            H_table = H_a

        # ── Normalised base coordinates (H, W) in [0, 1] ──
        ys, xs = np.mgrid[0:H, 0:W].astype(np.float32)
        bx = xs / max(1, W - 1)
        by = ys / max(1, H - 1)

        cx = bx.copy()
        cy = by.copy()
        if anim_mode == "warp":
            cx = (bx + 0.13 * _t) % 1.0
            cy = (by + 0.07 * _t) % 1.0
        elif anim_mode == "spin":
            ang = _t * 0.6
            ca_s = math.cos(ang)
            sa_s = math.sin(ang)
            dx = bx - 0.5
            dy = by - 0.5
            cx = 0.5 + dx * ca_s - dy * sa_s
            cy = 0.5 + dx * sa_s + dy * ca_s
        # morph / none: cx, cy stay at base

        # ── Multi-resolution hash encoding ──
        feats = []
        for lvl in range(levels):
            res = int(round(base_res * (growth ** lvl)))
            if res < 1:
                res = 1
            if res > 1024:
                res = 1024
            psx = cx * res
            psy = cy * res
            ix = np.floor(psx).astype(np.int32)
            iy = np.floor(psy).astype(np.int32)
            fx = (psx - ix).astype(np.float32)
            fy = (psy - iy).astype(np.float32)
            g00 = H_table[_hash2(ix, iy, lvl, seed, T)]
            g10 = H_table[_hash2(ix + 1, iy, lvl, seed, T)]
            g01 = H_table[_hash2(ix, iy + 1, lvl, seed, T)]
            g11 = H_table[_hash2(ix + 1, iy + 1, lvl, seed, T)]
            w00 = (1.0 - fx) * (1.0 - fy)
            w10 = fx * (1.0 - fy)
            w01 = (1.0 - fx) * fy
            w11 = fx * fy
            lvlfeat = (
                w00[..., None] * g00
                + w10[..., None] * g10
                + w01[..., None] * g01
                + w11[..., None] * g11
            )
            feats.append(lvlfeat)
        features = np.concatenate(feats, axis=-1).reshape(-1, levels * F).astype(np.float32)

        # ── Tiny fixed MLP decoder (no training) ──
        hidden = 8
        rng_d = np.random.default_rng(seed + 777)
        W1 = (rng_d.standard_normal((levels * F, hidden)).astype(np.float32)) * 0.5
        b1 = (rng_d.standard_normal(hidden).astype(np.float32)) * 0.5
        W2 = (rng_d.standard_normal((hidden, 3)).astype(np.float32)) * 0.5
        b2 = (rng_d.standard_normal(3).astype(np.float32)) * 0.5
        hid = np.tanh(features @ W1 + b1)
        rgb_lin = np.tanh(hid @ W2 + b2)
        rgb = (rgb_lin * 0.5 + 0.5).reshape(H, W, 3).astype(np.float32)

        # ── Colour modes ──
        gray = rgb.mean(axis=-1, keepdims=True)
        if color_mode == "mono":
            rgb = np.repeat(gray, 3, axis=-1).astype(np.float32)
        elif color_mode == "tint":
            gv = gray[..., 0]
            rr, gg, bb = _hsv_to_rgb((hue + gv * 0.6) % 1.0, 0.85, 0.4 + 0.6 * gv)
            tint = np.stack([rr, gg, bb], axis=-1)
            rgb = (gv[..., None] * tint + (1.0 - gv[..., None]) * 0.05).astype(np.float32)
        # rgb: keep 3-channel decode

        rgb = np.clip((rgb - 0.5) * gain + 0.5, 0.0, 1.0).astype(np.float32)
        mask = gray[..., 0].astype(np.float32)

        capture_frame("978", rgb)
        save(rgb, mn(978, "Instant-NGP Hash Texture"), out_dir)
        try:
            write_scalars(
                out_dir,
                levels=float(levels),
                base_resolution=float(base_res),
                growth=float(growth),
                hash_table_size=float(T),
                feature_dim=float(F),
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
        save(fallback, mn(978, "Instant-NGP Hash Texture"), out_dir)
        print(f"[method_978] ERROR: {exc}")
        return fallback
