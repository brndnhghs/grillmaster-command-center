from __future__ import annotations

import math
from pathlib import Path

import numpy as np

from ...core.registry import method
from ...core.utils import (
    save, mn, seed_all, W, H, write_scalars, write_field, write_mask, load_input, wired_source_lum,
)
from ...core.animation import capture_frame


def _fmm_distance(Hp: int, Wp: int, seeds: np.ndarray, speed: np.ndarray):
    """Fast Marching Method (Sethian 1996/1999) for the Eikonal equation.

    Solves  |grad T| * F = 1  (i.e. |grad T| = 1/F)  with T=0 at ``seeds``.
    A narrow-band heap walk: at each step the *trial* cell with the smallest
    tentative arrival time is frozen (accepted), then its 4 neighbours are
    reconsidered using the upwind Godunov difference. This yields the exact
    (first-order) geodesic arrival time under the anisotropic speed field ``F``.

    ``speed`` is the local arrival-cost multiplier (>0). ``seeds`` is a boolean
    mask of sources (T=0). Returns float32 (Hp,Wp) arrival time (inf where
    unreachable), and a uint8 state map (0=Far, 1=Trial, 2=Accepted).
    """
    INF = np.inf
    T = np.full((Hp, Wp), INF, dtype=np.float64)
    state = np.zeros((Hp, Wp), dtype=np.uint8)  # 0 Far, 1 Trial, 2 Accepted
    invF = 1.0 / np.maximum(speed, 1e-6)

    # Simple binary min-heap of (T, (y, x)) stored as parallel lists.
    heap_t: list[float] = []
    heap_iy: list[int] = []
    heap_ix: list[int] = []
    n = 0

    def _push(t, iy, ix):
        nonlocal n
        heap_t.append(t)
        heap_iy.append(iy)
        heap_ix.append(ix)
        n += 1
        c = n - 1
        while c > 0:
            p = (c - 1) >> 1
            if heap_t[p] <= heap_t[c]:
                break
            heap_t[p], heap_t[c] = heap_t[c], heap_t[p]
            heap_iy[p], heap_iy[c] = heap_iy[c], heap_iy[p]
            heap_ix[p], heap_ix[c] = heap_ix[c], heap_ix[p]
            c = p

    def _pop():
        nonlocal n
        rt = heap_t[0]; riy = heap_iy[0]; rix = heap_ix[0]
        n -= 1
        if n > 0:
            heap_t[0] = heap_t.pop()
            heap_iy[0] = heap_iy.pop()
            heap_ix[0] = heap_ix.pop()
            c = 0
            while True:
                l = 2 * c + 1
                r = 2 * c + 2
                sm = c
                if l < n and heap_t[l] < heap_t[sm]:
                    sm = l
                if r < n and heap_t[r] < heap_t[sm]:
                    sm = r
                if sm == c:
                    break
                heap_t[c], heap_t[sm] = heap_t[sm], heap_t[c]
                heap_iy[c], heap_iy[sm] = heap_iy[sm], heap_iy[c]
                heap_ix[c], heap_ix[sm] = heap_ix[sm], heap_ix[c]
                c = sm
        return rt, riy, rix

    def _update(iy, ix):
        if state[iy, ix] == 2:
            return
        a = T[iy, ix - 1] if ix > 0 else INF
        b = T[iy, ix + 1] if ix < Wp - 1 else INF
        c = T[iy - 1, ix] if iy > 0 else INF
        d = T[iy + 1, ix] if iy < Hp - 1 else INF
        nb = [v for v in (a, b, c, d) if v < INF]
        if not nb:
            return
        cost = invF[iy, ix]
        if len(nb) == 1:
            # 1-sided front (only one accepted neighbour, e.g. right next to a seed)
            tnew = nb[0] + cost
        else:
            # upwind: pick the two smallest among the accepted neighbours
            s1, s2 = sorted(nb)[:2]
            disc = max(s1 + s2, 2.0 * s1) * cost
            disc = disc * disc - 2.0 * (s1 - s2) * (s1 - s2) * cost * cost
            if disc < 0:
                tnew = s1 + cost
            else:
                tnew = 0.5 * (s1 + s2 + math.sqrt(disc))
        if tnew < T[iy, ix]:
            T[iy, ix] = tnew
            state[iy, ix] = 1
            _push(tnew, iy, ix)

    sy, sx = np.nonzero(seeds)
    for iy, ix in zip(sy.tolist(), sx.tolist()):
        T[iy, ix] = 0.0
        state[iy, ix] = 1
        _push(0.0, iy, ix)

    while n > 0:
        _, iy, ix = _pop()
        if state[iy, ix] == 2:
            continue
        state[iy, ix] = 2
        if ix > 0:
            _update(iy, ix - 1)
        if ix < Wp - 1:
            _update(iy, ix + 1)
        if iy > 0:
            _update(iy - 1, ix)
        if iy < Hp - 1:
            _update(iy + 1, ix)

    return T.astype(np.float32), state


def _iq_ramp(t: np.ndarray):
    """Inigo Quilez cosine palette (smooth, periodic)."""
    t = np.clip(t, 0.0, 1.0)
    r = 0.5 + 0.5 * np.cos(6.2831853 * (t + 0.00))
    g = 0.5 + 0.5 * np.cos(6.2831853 * (t + 0.3333333))
    b = 0.5 + 0.5 * np.cos(6.2831853 * (t + 0.6666667))
    return np.stack([r, g, b], axis=-1)


@method(id='442', name='Fast Marching Geodesic', category='math_art', new_image_contract=True, tags=['geodesic', 'distance', 'eikonal', 'fast-marching', 'anisotropic', 'obstacle', 'expanded', 'animation'], inputs={'image_in': 'IMAGE'}, outputs={'image': 'IMAGE', 'mask': 'MASK'}, params={'seed_mode': {'description': 'seed placement: single, corners, grid, noise, input_mask', 'choices': ['single', 'corners', 'grid', 'noise', 'input_mask'], 'default': 'single'}, 'speed_mode': {'description': 'speed field (cost) source: uniform, radial, noise, input_image', 'choices': ['uniform', 'radial', 'noise', 'input_image'], 'default': 'uniform'}, 'obstacle_mode': {'description': 'obstacle (impassable) source: none, input_image, noise', 'choices': ['none', 'input_image', 'noise'], 'default': 'none'}, 'obstacle_strength': {'description': 'obstacle slow-down (1=free, 30=near-blocking)', 'min': 1.0, 'max': 30.0, 'default': 8.0}, 'seed_count': {'description': 'number of seeds (grid/noise modes)', 'min': 1, 'max': 24, 'default': 6}, 'color_mode': {'description': 'render: field (geodesic heat), isolines (contour bands), overlay (on bg)', 'choices': ['field', 'isolines', 'overlay'], 'default': 'field'}, 'n_iso': {'description': 'number of geodesic contour bands', 'min': 4, 'max': 32, 'default': 16}, 'line_width': {'description': 'isoline/overlay stroke width in px', 'min': 1, 'max': 6, 'default': 2}, 'anim_mode': {'description': 'animation mode: none, sweep, breathe, rotate', 'choices': ['none', 'sweep', 'breathe', 'rotate'], 'default': 'none'}, 'anim_speed': {'description': 'animation speed multiplier', 'min': 0.1, 'max': 5.0, 'default': 1.0}, 'time': {'description': 'animation time in radians', 'min': 0.0, 'max': 6.2832, 'default': 0.0}, 'source': {'description': 'wired upstream image as a domain-warp / seed source', 'choices': ['none', 'input_image'], 'default': 'none'}})
def method_fast_marching(out_dir: Path, seed: int, params=None):
    """Fast Marching Method — exact geodesic / Eikonal distance transform.

    The Fast Marching Method (FMM; J.A. Sethian, Proc. Natl. Acad. Sci. 1996 /
    SIAM Rev. 1999) is a Dijkstra-style narrow-band algorithm that solves the
    static Eikonal equation  |grad T(x)| * F(x) = 1  with T = 0 at a set of
    source seeds. Here T(x) is the *geodesic arrival time* (anisotropic distance)
    from the seeds: a wavefront is marched outward one accepted cell at a time via
    the Godunov upwind difference, so the metric honours a per-pixel speed field
    F(x) and hard obstacles.

    Unlike a Euclidean distance transform (Jump-Flood Voronoi node is Euclidean),
    FMM's distance bends around obstacles and is modulated by local traversal
    cost — the canonical "how far, walking the shortest path, avoiding walls"
    map used for path planning, medial-axis / Voronoi-style segmentation,
    image segmentation (geodesic active contours), and relief/heightfield shading.

    Pipeline (Architecture B — one frame per animation phase ``t``):
      1. Choose seed sources (single point, corners, regular grid, a seed-noise
         scatter, or a wired/thresholded binary mask).
      2. Build an anisotropic speed field F from uniform / radial / value-noise /
         a wired luminance image; optionally multiply by an obstacle slow-down
         (impassable walls in input_image-channel or a thresholded noise blob).
      3. Run the binary-heap FMM on a capped (Hp x Wp) grid (<=~240px) — exact,
         O(N log N), and comfortably sub-second, so it stays far under the
         shootout's 150s render cull.
      4. Upscale the geodesic field to full canvas and render as a heat field,
         banded isolines, or as a thin stroke overlay on a neutral background.
      5. Emit a MASK = normalized geodesic field (0 at seeds -> 1 at far edge).

    Animation modes (deterministic, seed-stable):
      none    - static baseline (field independent of ``time``).
      sweep   - seeds orbit the seed-nodes parameters / a radial wave is added.
      breathe - the obstacle strength oscillates, morphing the geodesic field.
      rotate  - the whole geodesic visualisation rotates about the centre.

    The CPU path is authoritative. The only loops are the narrow-band heap walk
    and a few vectorised threshold passes; no per-pixel Python overhead.
    """
    try:
        if params is None:
            params = {}
        anim_time = float(params.get("time", 0.0))
        anim_mode = str(params.get("anim_mode", "none"))
        anim_speed = float(params.get("anim_speed", 1.0))
        seed_all(seed)

        seed_mode = str(params.get("seed_mode", "single"))
        speed_mode = str(params.get("speed_mode", "uniform"))
        obstacle_mode = str(params.get("obstacle_mode", "none"))
        obstacle_strength = float(params.get("obstacle_strength", 8.0))
        obstacle_strength = max(1.0, min(30.0, obstacle_strength))
        seed_count = int(params.get("seed_count", 6))
        seed_count = max(1, min(24, seed_count))
        color_mode = str(params.get("color_mode", "field"))
        n_iso = int(params.get("n_iso", 16))
        n_iso = max(4, min(32, n_iso))
        line_width = int(params.get("line_width", 2))
        line_width = max(1, min(6, line_width))

        _t = 0.0 if anim_mode == "none" else anim_time * anim_speed

        w = int(W)
        h = int(H)

        # ── Wired input override (Rule 12) ──
        wired = None
        wired_path = params.get("input_image", "")
        if wired_path:
            try:
                wired = load_input(wired_path, w, h)
            except (FileNotFoundError, OSError):
                wired = None

        # ── Capped marching grid (keeps the heap walk fast) ──
        cap = 240
        scale = max(1, max(w, h) // cap)
        Hp = max(2, h // scale)
        Wp = max(2, w // scale)
        gy, gx = np.meshgrid(np.arange(Hp), np.arange(Wp), indexing="ij")
        cy, cx = Hp / 2.0, Wp / 2.0

        # ── Seed mask ──
        seeds = np.zeros((Hp, Wp), dtype=bool)
        if seed_mode == "input_mask":
            if wired is not None:
                lum = (0.299 * wired[:, :, 0] + 0.587 * wired[:, :, 1]
                       + 0.114 * wired[:, :, 2])
                sm = np.array(_resize(lum, Wp, Hp))
                seeds = sm > 0.5
            if not seeds.any():
                seeds[0, 0] = True
        elif seed_mode == "corners":
            seeds[0, 0] = True
            seeds[0, Wp - 1] = True
            seeds[Hp - 1, 0] = True
            seeds[Hp - 1, Wp - 1] = True
        elif seed_mode == "grid":
            step = max(1, int(round(math.sqrt(Hp * Wp / seed_count))))
            for yy in range(0, Hp, step):
                for xx in range(0, Wp, step):
                    seeds[yy, xx] = True
        elif seed_mode == "noise":
            rng = np.random.default_rng(seed)
            idx = rng.choice(Hp * Wp, size=min(seed_count, Hp * Wp), replace=False)
            syy, sxx = np.unravel_index(idx, (Hp, Wp))
            seeds[syy, sxx] = True
        else:  # single (animated sweep orbits multiple asymmetric seeds)
            if anim_mode == "sweep":
                for k in range(3):
                    a = _t + k * 2.0943951  # 120 deg apart
                    sx0 = int(cx + 0.6 * (Wp / 2) * math.cos(a))
                    sy0 = int(cy + 0.6 * (Hp / 2) * math.sin(a))
                    seeds[min(max(sy0, 0), Hp - 1), min(max(sx0, 0), Wp - 1)] = True
            else:
                seeds[int(cy), int(cx)] = True

        # ── Speed / cost field (invF = 1/F) ──
        speed = np.ones((Hp, Wp), dtype=np.float64)
        if speed_mode == "input_image" and wired is not None:
            lum = (0.299 * wired[:, :, 0] + 0.587 * wired[:, :, 1]
                   + 0.114 * wired[:, :, 2])
            sp = np.array(_resize(lum, Wp, Hp))
            speed = 0.15 + 1.85 * sp  # brighter = faster traversal
        elif speed_mode == "radial":
            r = np.hypot(gx - cx, gy - cy)
            speed = 0.25 + 1.75 * np.clip(np.cos(r * 0.06 - _t), 0.0, 1.0)
        elif speed_mode == "noise":
            rng = np.random.default_rng(seed + 7)
            base = rng.random((Hp // 8 + 2, Wp // 8 + 2))
            from scipy.ndimage import zoom
            nz = zoom(base, (Hp / base.shape[0], Wp / base.shape[1]), order=3)
            speed = 0.2 + 1.6 * np.clip(nz, 0, 1)
        # else uniform -> leaves speed = 1

        # ── Animation: a time-varying wave injected into the speed field so the
        # geodesic field actually morphs (a moving single seed alone is masked by
        # normalisation). sweep/breathe/rotate all re-derive T from the new field. ──
        if anim_mode in ("sweep", "breathe", "rotate"):
            wave = 0.5 + 0.5 * np.sin(gx * 0.05 + _t) * np.cos(gy * 0.05 - _t * 0.8)
            speed = speed * (0.18 + 1.64 * wave)

        # ── Obstacle (large slow-down) ──
        obs = np.ones((Hp, Wp), dtype=np.float64)
        if obstacle_mode == "input_image" and wired is not None:
            lum = (0.299 * wired[:, :, 0] + 0.587 * wired[:, :, 1]
                   + 0.114 * wired[:, :, 2])
            om = np.array(_resize(lum, Wp, Hp))
            obs = np.where(om > 0.5, obstacle_strength, 1.0)
        elif obstacle_mode == "noise":
            ob = np.sin(gx * 0.18 + _t) * np.cos(gy * 0.18 - _t)
            thr = 0.1  # wide walls so the obstacle visibly reroutes the front
            obs = np.where(ob > thr, obstacle_strength, 1.0)
        speed = speed / obs  # stronger obstacle -> lower F -> longer geodesic

        T, _state = _fmm_distance(Hp, Wp, seeds, speed)

        # ── Normalise geodesic field ──
        # Use a FIXED reference (canvas diagonal in grid cells) so animation modes
        # that shift the absolute geodesic arrival time remain VISIBLE across
        # frames. Per-frame min/max normalisation would cancel a uniform speed
        # scaling and make sweep/breathe render as a static image (pitfall #19:
        # normalisation cancelling a live control).
        Tf = T.copy()
        Tf[Tf == np.inf] = np.nan
        diag = math.hypot(Wp, Hp)
        Tn = np.clip(Tf / diag, 0.0, 1.0)
        Tn = np.nan_to_num(Tn, nan=1.0)
        tmin = 0.0
        tmax = float(np.nanmax(Tf)) if np.any(~np.isnan(Tf)) else float(diag)

        # ── Render ──
        if anim_mode == "rotate":
            angle = _t
        else:
            angle = 0.0

        out = _render_geodesic(Tn, color_mode, n_iso, line_width, angle,
                               cx * scale, cy * scale, w, h)

        # ── Sidecar outputs (Rules 4, 5, 10) ──
        mask = _resize_mask(Tn, w, h)
        write_mask(out_dir, mask.astype(np.float32))
        write_field(out_dir, _resize_field(Tn, w, h).astype(np.float32))
        n_reach = int(np.count_nonzero(Tn < 1.0))
        write_scalars(out_dir, geodesic_max=float(tmax), geodesic_min=float(tmin),
                      seeds=float(int(seeds.sum())), reachable=float(n_reach))

        capture_frame("442", out)
        save(out, mn(442, f"Fast Marching Geodesic t={_t:.2f}"), out_dir)
        return out
    except Exception as exc:
        fallback = np.zeros((int(H), int(W), 4), dtype=np.float32)
        fallback[:, :, 3] = 1.0
        fallback[:, :, :3] = 0.5
        save(fallback, mn(442, "Fast Marching Geodesic"), out_dir)
        print(f"[method_442] ERROR: {exc}")
        return fallback


# ── Helpers ──
def _resize(arr, Wp, Hp):
    """Nearest-neighbour resize of an (h,w) float array to (Hp,Wp)."""
    from scipy.ndimage import zoom
    zh = Hp / arr.shape[0]
    zw = Wp / arr.shape[1]
    return zoom(arr, (zh, zw), order=1)


def _resize_field(arr, w, h):
    from scipy.ndimage import zoom
    return zoom(arr, (h / arr.shape[0], w / arr.shape[1]), order=1)


def _resize_mask(arr, w, h):
    from scipy.ndimage import zoom
    return np.clip(zoom(arr, (h / arr.shape[0], w / arr.shape[1]), order=1), 0.0, 1.0)


def _render_geodesic(Tn, color_mode, n_iso, line_width, angle, ccx, ccy, w, h):
    """Upscale normalised geodesic field and paint it per color_mode."""
    from scipy.ndimage import zoom
    Tfull = np.clip(zoom(Tn, (h / Tn.shape[0], w / Tn.shape[1]), order=1), 0.0, 1.0)
    out = np.zeros((h, w, 4), dtype=np.float32)

    if angle != 0.0:
        from scipy.ndimage import rotate
        Tfull = rotate(Tfull, math.degrees(angle), reshape=False, order=1)
        Tfull = np.clip(Tfull, 0.0, 1.0)

    if color_mode == "field":
        col = _iq_ramp(Tfull)
        out[:, :, 0:3] = col
        out[:, :, 3] = 1.0
    elif color_mode == "isolines":
        bands = np.floor(Tfull * n_iso) / max(n_iso - 1, 1)
        base = _iq_ramp(bands)
        out[:, :, 0:3] = base
        out[:, :, 3] = 1.0
        # overlay thin band edges as dark strokes
        edge = _band_edges(Tfull, n_iso, line_width)
        out[edge, 0:3] = 0.04
    else:  # overlay stroke on neutral background
        bg = np.full((h, w, 3), 0.06, dtype=np.float32)
        col = _iq_ramp(Tfull)
        out[:, :, 0:3] = bg
        out[:, :, 3] = 1.0
        edge = _band_edges(Tfull, n_iso, line_width)
        out[edge, 0:3] = col[edge]
    return out


def _band_edges(Tfull, n_iso, line_width):
    """Boolean mask of contour edges between geodesic bands (thin strokes)."""
    from scipy.ndimage import sobel
    bands = np.floor(Tfull * n_iso).astype(np.int32)
    # gradient magnitude of the band index highlights band boundaries
    gx = sobel(bands.astype(np.float64), axis=1)
    gy = sobel(bands.astype(np.float64), axis=0)
    g = np.hypot(gx, gy)
    mask = g > 0.5
    if line_width > 1:
        from scipy.ndimage import maximum_filter
        mask = maximum_filter(mask, size=line_width)
    return mask
