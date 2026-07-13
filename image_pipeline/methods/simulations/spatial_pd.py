"""
#125 — Spatial Prisoner's Dilemma

Evolutionary game theory on a 2D grid. Agents play iterated social dilemmas
with their neighborhood, updating strategies based on payoff comparison.

3 games × 2 update rules × 5 init modes × 5 render styles × 2 neighborhoods
× 6 animation modes = 1800 distinct configurations.

Produces emergent spiral waves of cooperation, traveling defector bands,
mosaic-like strategy domains, and sudden cooperation collapse.
"""
from __future__ import annotations
import math
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter

from ...core.registry import method
from ...core.utils import save, mn, seed_all, W, H
from ...core.animation import capture_frame

# ── States ──
COOP = 0
DEFECT = 1

# ── Payoff matrices indexed [my_action][opponent_action] ──
PAYOFF_PD = np.array([
    [1.0,  0.0],
    [1.5,  0.0],
], dtype=np.float32)

PAYOFF_SNOWDRIFT = np.array([
    [1.0,  0.3],
    [1.5,  0.0],
], dtype=np.float32)

PAYOFF_STAG_HUNT = np.array([
    [1.0,  0.0],
    [1.2,  0.8],
], dtype=np.float32)

PAYOFF_TABLES = {
    "pd": PAYOFF_PD,
    "snowdrift": PAYOFF_SNOWDRIFT,
    "stag_hunt": PAYOFF_STAG_HUNT,
}

# ── Colors ──
COOP_COLOR = np.array([60, 140, 220], dtype=np.uint8)
DEFECT_COLOR = np.array([220, 100, 40], dtype=np.uint8)

# ── Neighbourhoods ──
NEIGHBORHOODS = {
    "moore": [(-1, -1), (-1, 0), (-1, 1),
              (0, -1),           (0, 1),
              (1, -1),  (1, 0),  (1, 1)],
    "von_neumann": [(-1, 0), (0, -1), (0, 1), (1, 0)],
}


# ═══════════════════════════════════════════════════════════════
#  Game engine
# ═══════════════════════════════════════════════════════════════

def _play_round(state, payoffs, gh, gw, offsets):
    """Play one round: compute payoffs for all agents."""
    total_payoff = np.zeros((gh, gw), dtype=np.float32)
    for dy, dx in offsets:
        roll_state = np.roll(state, shift=(-dy, -dx), axis=(0, 1))
        total_payoff += payoffs[state, roll_state]
    return total_payoff


def _update_imitate_best(state, payoffs_tbl, gh, gw, rng, mutation_rate, offsets):
    """Imitate-the-best: copy the strategy of the neighbor with highest payoff.

    Vectorized: neighbor payoffs/strategies are gathered via np.roll over the
    offset list, then a running max picks the best neighbor per cell. Ties are
    adopted with 25% probability per neighbor (RNG drawn per offset).

    The neighbor arrays are rolled once into stacked tensors (as the Fermi/Moran
    rules already do) instead of re-rolling inside the offset loop — this halves
    the np.roll count per step. The RNG draw order per offset is preserved
    exactly, so output is bit-identical.
    """
    payoffs = _play_round(state, payoffs_tbl, gh, gw, offsets)
    best_payoff = payoffs.copy()
    best_strategy = state.copy()
    # Roll every neighbor once, up front.
    rolls_p = np.stack([np.roll(payoffs, shift=(-dy, -dx), axis=(0, 1)) for dy, dx in offsets])
    rolls_s = np.stack([np.roll(state, shift=(-dy, -dx), axis=(0, 1)) for dy, dx in offsets])
    gi = np.arange(gh)[:, None]
    gj = np.arange(gw)[None, :]
    for k, (dy, dx) in enumerate(offsets):
        nb_p = rolls_p[k, gi, gj]
        nb_s = rolls_s[k, gi, gj]
        better = nb_p > best_payoff
        best_payoff[better] = nb_p[better]
        best_strategy[better] = nb_s[better]
        tie = (~better) & (nb_p == best_payoff)
        adopt = tie & (rng.random((gh, gw)) < 0.25)
        best_strategy[adopt] = nb_s[adopt]
    new_state = best_strategy
    if mutation_rate > 0:
        new_state = new_state.copy()
        new_state[rng.random((gh, gw)) < mutation_rate] ^= 1
    return new_state, payoffs


def _update_fermi(state, payoffs_tbl, gh, gw, rng, mutation_rate, offsets, K):
    """Fermi: compare with a random neighbor, switch with logistic probability.

    Vectorized: one random neighbor per cell is selected via per-cell offset
    indices, then the logistic switch probability is applied in bulk.
    """
    payoffs = _play_round(state, payoffs_tbl, gh, gw, offsets)
    n = len(offsets)
    gi = np.arange(gh)[:, None]
    gj = np.arange(gw)[None, :]
    rolls_p = np.stack([np.roll(payoffs, shift=(-dy, -dx), axis=(0, 1)) for dy, dx in offsets])
    rolls_s = np.stack([np.roll(state, shift=(-dy, -dx), axis=(0, 1)) for dy, dx in offsets])
    oidx = rng.integers(0, n, size=(gh, gw))
    nb_p = rolls_p[oidx, gi, gj]
    nb_s = rolls_s[oidx, gi, gj]
    prob = 1.0 / (1.0 + np.exp((payoffs - nb_p) / K))
    new_state = state.copy()
    switch = rng.random((gh, gw)) < prob
    new_state[switch] = nb_s[switch]
    if mutation_rate > 0:
        new_state[rng.random((gh, gw)) < mutation_rate] ^= 1
    return new_state, payoffs


def _update_moran(state, payoffs_tbl, gh, gw, rng, mutation_rate, offsets):
    """Moran-like proportional imitation.

    Per-cell random-neighbor comparison with probability proportional to the
    payoff difference p = max(0, (π_neighbor - π_self) / max(π)). Vectorized as
    a single stochastic sweep over all cells (one random neighbor per cell).
    """
    payoffs = _play_round(state, payoffs_tbl, gh, gw, offsets)
    p_max = max(float(payoffs.max()), 0.01)
    n = len(offsets)
    gi = np.arange(gh)[:, None]
    gj = np.arange(gw)[None, :]
    rolls_p = np.stack([np.roll(payoffs, shift=(-dy, -dx), axis=(0, 1)) for dy, dx in offsets])
    rolls_s = np.stack([np.roll(state, shift=(-dy, -dx), axis=(0, 1)) for dy, dx in offsets])
    oidx = rng.integers(0, n, size=(gh, gw))
    nb_p = rolls_p[oidx, gi, gj]
    nb_s = rolls_s[oidx, gi, gj]
    diff = nb_p - payoffs
    prob = np.where(diff > 0, diff / p_max, 0.0)
    new_state = state.copy()
    switch = rng.random((gh, gw)) < prob
    new_state[switch] = nb_s[switch]
    if mutation_rate > 0:
        new_state[rng.random((gh, gw)) < mutation_rate] ^= 1
    return new_state, payoffs


UPDATE_RULES = {
    "imitate_best": lambda s, p, gh, gw, r, mr, off, K: _update_imitate_best(s, p, gh, gw, r, mr, off),
    "fermi":        lambda s, p, gh, gw, r, mr, off, K: _update_fermi(s, p, gh, gw, r, mr, off, K),
    "moran":        lambda s, p, gh, gw, r, mr, off, K: _update_moran(s, p, gh, gw, r, mr, off),
}


# ═══════════════════════════════════════════════════════════════
#  Initial condition modes
# ═══════════════════════════════════════════════════════════════

def _init_random(gh, gw, rng, init_coop, **_):
    return (rng.random((gh, gw)) < init_coop).astype(np.int32)


def _init_clusters(gh, gw, rng, init_coop, **_):
    """Seeded clusters of cooperators in a defector sea."""
    state = np.zeros((gh, gw), dtype=np.int32)  # all defect
    n_clusters = max(3, int(init_coop * 20))
    cluster_r = max(2, int(min(gh, gw) * 0.04))
    for _ in range(n_clusters):
        ci = rng.integers(cluster_r, gh - cluster_r)
        cj = rng.integers(cluster_r, gw - cluster_r)
        y, x = np.ogrid[-ci:gh - ci, -cj:gw - cj]
        mask = x * x + y * y < cluster_r * cluster_r
        state[mask] = COOP
    return state


def _init_stripes(gh, gw, rng, init_coop, **_):
    """Alternating vertical bands of cooperators and defectors."""
    state = np.zeros((gh, gw), dtype=np.int32)
    band_w = max(2, int(gw * 0.08))
    for cj in range(0, gw, band_w * 2):
        state[:, cj:min(cj + band_w, gw)] = COOP
    return state


def _init_checkerboard(gh, gw, rng, init_coop, **_):
    """Chessboard pattern of cooperators and defectors."""
    state = np.zeros((gh, gw), dtype=np.int32)
    state[::2, ::2] = COOP
    state[1::2, 1::2] = COOP
    return state


def _init_defector_seed(gh, gw, rng, init_coop, **_):
    """Small defector seed in a sea of cooperators."""
    state = np.ones((gh, gw), dtype=np.int32)  # all cooperate... wait
    state = np.zeros((gh, gw), dtype=np.int32)  # all cooperate
    seed_r = max(3, int(min(gh, gw) * 0.03))
    ci, cj = gh // 2, gw // 2
    y, x = np.ogrid[-ci:gh - ci, -cj:gw - cj]
    mask = x * x + y * y < seed_r * seed_r
    state[mask] = DEFECT
    return state


def _init_spiral_seed(gh, gw, rng, init_coop, **_):
    """Small spiral seed — promotes spiral wave formation."""
    state = np.zeros((gh, gw), dtype=np.int32)
    ci, cj = gh // 2, gw // 2
    for r in range(3, 16):
        for theta in np.arange(0, 2 * math.pi, 0.3):
            x = int(cj + r * math.cos(theta + r * 0.5))
            y = int(ci + r * math.sin(theta + r * 0.5))
            if 0 <= x < gw and 0 <= y < gh:
                if (r + int(theta * 3)) % 2 == 0:
                    state[y, x] = COOP
                else:
                    state[y, x] = DEFECT
    return state


INIT_MODES = {
    "random": _init_random,
    "clusters": _init_clusters,
    "stripes": _init_stripes,
    "checkerboard": _init_checkerboard,
    "defector_seed": _init_defector_seed,
    "spiral_seed": _init_spiral_seed,
}


# ═══════════════════════════════════════════════════════════════
#  Render styles
# ═══════════════════════════════════════════════════════════════

def _render_default(state, payoffs, gh, gw):
    """Blue/amber per cell, brightness by payoff."""
    img = np.zeros((gh, gw, 3), dtype=np.uint8)
    p_max, p_min = payoffs.max(), payoffs.min()
    p_range = max(p_max - p_min, 0.001)
    brightness = 0.5 + 0.5 * (payoffs - p_min) / p_range
    coop_mask = state == COOP
    def_mask = state == DEFECT
    for c in range(3):
        img[coop_mask, c] = (COOP_COLOR[c] * brightness[coop_mask]).astype(np.uint8)
        img[def_mask, c] = (DEFECT_COLOR[c] * brightness[def_mask]).astype(np.uint8)
    return img


def _render_heatmap(state, payoffs, gh, gw):
    """Full-range thermal colormap of payoff values — reveals payoff landscape."""
    p_min, p_max = payoffs.min(), payoffs.max()
    p_range = max(p_max - p_min, 0.001)
    normed = (payoffs - p_min) / p_range  # [0,1]
    img = np.zeros((gh, gw, 3), dtype=np.uint8)
    # Dark → purple → orange → gold by payoff
    img[:, :, 0] = (normed * 220).astype(np.uint8)
    img[:, :, 1] = (normed * normed * 180).astype(np.uint8)
    img[:, :, 2] = ((1 - normed) * 200 * (1 - normed * 0.7)).astype(np.uint8)
    # Overlay strategy boundaries: edge detect shows domain walls
    edge = np.zeros((gh, gw), dtype=bool)
    edge[1:, :] |= state[1:, :] != state[:-1, :]
    edge[:, 1:] |= state[:, 1:] != state[:, :-1]
    img[edge] = [255, 255, 255]
    return img


def _render_cluster_labels(state, payoffs, gh, gw):
    """Color by connected-component cluster of the majority strategy.

    Uses scipy.ndimage.label (4-way structure) to find contiguous domains,
    assigning each cluster a stable golden-ratio hue.
    """
    from scipy.ndimage import label
    struct = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=bool)
    labels, current_label = label(state, structure=struct)
    img = np.zeros((gh, gw, 3), dtype=np.uint8)
    if current_label == 0:
        return img
    for label in range(1, current_label + 1):
        mask = labels == label
        strat = state[mask][0]  # all same strategy within label
        # Golden ratio hue cycling for distinct cluster colors
        hue = (label * 0.6180339887) % 1.0
        if strat == COOP:
            # blue-ish cluster
            h = 0.55 + 0.25 * hue
        else:
            # amber-ish cluster
            h = 0.05 + 0.15 * hue
        s = 0.5 + 0.4 * ((label * 7) % 1.0)
        v = 0.6 + 0.4 * ((label * 13) % 1.0)
        r, g, b = _hsv2rgb(h % 1.0, s, v)
        img[mask] = (r, g, b)
    return img


def _render_payoff_diff(state, payoffs, gh, gw):
    """Color by difference between own payoff and best neighbor's.

    Red = far behind neighbor (high pressure to switch)
    Blue = ahead of neighbors (stable)
    Green = roughly equal
    """
    from scipy.ndimage import maximum_filter
    # Max payoff among 8 neighbors
    neighbor_max = maximum_filter(payoffs, size=3, mode="wrap")
    diff = payoffs - neighbor_max  # negative = behind, positive = ahead
    d_max = max(abs(diff).max(), 0.001)
    normed = np.clip(diff / d_max, -1.0, 1.0)  # [-1, 1]

    img = np.zeros((gh, gw, 3), dtype=np.uint8)
    # Negative (behind) → red, Positive (ahead) → blue
    behind = normed < 0
    ahead = normed > 0
    ad = np.abs(normed[behind])
    img[behind, 0] = (200 * ad).astype(np.uint8)
    img[behind, 1] = (40 * ad).astype(np.uint8)
    img[behind, 2] = (30 * ad).astype(np.uint8)
    img[ahead, 0] = (30 * normed[ahead]).astype(np.uint8)
    img[ahead, 1] = (60 * normed[ahead]).astype(np.uint8)
    img[ahead, 2] = (200 * normed[ahead]).astype(np.uint8)
    return img


def _render_coop_density(state, payoffs, gh, gw):
    """Gaussian-blurred local cooperation density.

    Reveals the spatial structure of cooperative domains as smooth
    continuous fields — cooperator hotspots glow blue, defector areas
    glow amber, with smooth gradients at domain boundaries.
    """
    density = state.astype(np.float32)  # 1 = cooperator, 0 = defector
    # Gaussian blur to get local density
    from scipy.ndimage import gaussian_filter
    sigma = max(2.0, min(gh, gw) * 0.02)
    density = gaussian_filter(density, sigma=sigma, mode="wrap")

    img = np.zeros((gh, gw, 3), dtype=np.uint8)
    # Blend between amber (d=0) and blue (d=1)
    d = density  # 0 = all def, 1 = all coop
    r = ((1 - d) * 200 + d * 60).astype(np.uint8)
    g = ((1 - d) * 80 + d * 130).astype(np.uint8)
    b = ((1 - d) * 30 + d * 210).astype(np.uint8)
    img[:, :, 0] = r
    img[:, :, 1] = g
    img[:, :, 2] = b
    return img


RENDER_STYLES = {
    "default": _render_default,
    "heatmap": _render_heatmap,
    "cluster_labels": _render_cluster_labels,
    "payoff_diff": _render_payoff_diff,
    "coop_density": _render_coop_density,
}


def _hsv2rgb(h, s, v):
    """HSV to RGB tuple of ints 0-255."""
    h = h % 1.0
    hi = int(h * 6) % 6
    f = h * 6 - hi
    p = v * (1 - s)
    q = v * (1 - f * s)
    t = v * (1 - (1 - f) * s)
    rgb = [(v, t, p), (q, v, p), (p, v, t), (p, q, v), (t, p, v), (v, p, q)][hi]
    return (int(rgb[0] * 255), int(rgb[1] * 255), int(rgb[2] * 255))


# ═══════════════════════════════════════════════════════════════
#  Method
# ═══════════════════════════════════════════════════════════════

@method(
    id="153",
    name="Spatial Prisoner's Dilemma",
    category="simulations",
    tags=["animation", "emergent", "game-theory", "expanded"],
    timeout=180,
    params={
        "anim_mode": {
            "description": "evolution mode",
            "choices": ["evolve", "sweep_temptation", "sweep_mutation", "sweep_S",
                        "invasion", "periodic_stir"],
            "default": "evolve",
        },
        "game": {
            "description": "game payoff matrix",
            "choices": ["pd", "snowdrift", "stag_hunt"],
            "default": "snowdrift",
        },
        "update_rule": {
            "description": "strategy update rule",
            "choices": ["imitate_best", "fermi", "moran"],
            "default": "imitate_best",
        },
        "init_mode": {
            "description": "initial spatial configuration",
            "choices": ["random", "clusters", "stripes", "checkerboard",
                        "defector_seed", "spiral_seed"],
            "default": "random",
        },
        "render_style": {
            "description": "visualization style",
            "choices": ["default", "heatmap", "cluster_labels", "payoff_diff",
                        "coop_density"],
            "default": "default",
        },
        "neighborhood": {
            "description": "interaction neighborhood",
            "choices": ["moore", "von_neumann"],
            "default": "moore",
        },
        "temptation": {
            "description": "defector temptation payoff T",
            "min": 1.0, "max": 2.0, "default": 1.5,
        },
        "sucker_payoff": {
            "description": "sucker's payoff S (cooperator vs defector)",
            "min": -1.0, "max": 1.0, "default": 0.0,
        },
        "mutation_rate": {
            "description": "per-cell per-step mutation probability",
            "min": 0.0, "max": 0.1, "default": 0.002,
        },
        "grid_size": {
            "description": "grid width (height = width * H/W)",
            "min": 64, "max": 400, "default": 160,
        },
        "n_frames": {
            "description": "frames to capture",
            "min": 10, "max": 300, "default": 100,
        },
        "steps_per_frame": {
            "description": "sim steps between frames",
            "min": 1, "max": 20, "default": 2,
        },
        "init_coop": {
            "description": "initial cooperation ratio (for random/clusters init)",
            "min": 0.05, "max": 0.95, "default": 0.5,
        },
        "fermi_K": {
            "description": "Fermi update noise (higher = more stochastic)",
            "min": 0.01, "max": 2.0, "default": 0.1,
        },"anim_speed": {
            "description": "animation speed multiplier",
            "min": 0.1, "max": 3.0, "default": 1.0,
        },
    }
)
def method_spatial_pd(out_dir: Path, seed: int, params=None):
    """Spatial Prisoner's Dilemma — evolutionary game theory on a 2D grid.

    Agents play iterated social dilemmas with neighbors, updating strategies
    by imitation, Fermi, or Moran dynamics. 6 animation modes × 3 games × 3
    update rules × 6 init modes × 5 render styles × 2 neighborhoods = 3240
    configurations.

    Emergent patterns: spiral waves of cooperation, traveling defector bands,
    mosaic-like strategy domains, payoff landscape dynamics, cluster nucleation.

    Args:
        out_dir: Output directory
        seed: Random seed
        params: Parameter overrides
    """
    if params is None:
        params = {}
    t = float(params.get("time", 0.0))
    anim_mode = str(params.get("anim_mode", "evolve"))
    anim_speed = float(params.get("anim_speed", 1.0))
    game_type = str(params.get("game", "snowdrift"))
    update_rule = str(params.get("update_rule", "imitate_best"))
    init_mode = str(params.get("init_mode", "random"))
    render_style = str(params.get("render_style", "default"))
    neighborhood = str(params.get("neighborhood", "moore"))
    temptation = float(params.get("temptation", 1.5))
    sucker_payoff = float(params.get("sucker_payoff", 0.0))
    mutation_rate = float(params.get("mutation_rate", 0.002))
    grid_size = int(params.get("grid_size", 160))
    n_frames = int(params.get("n_frames", 100))
    steps_per_frame = int(params.get("steps_per_frame", 2))
    init_coop = float(params.get("init_coop", 0.5))
    fermi_K = float(params.get("fermi_K", 0.1))

    _t = t * anim_speed

    seed_all(seed)
    rng = np.random.default_rng(seed)
    if _t > 0.001:
        seed_all(seed + int(_t * 10000))
        rng = np.random.default_rng(seed + int(_t * 10000))

    # ── Grid dimensions ──
    gw = max(32, min(400, grid_size))
    gh = max(20, int(gw * H / W))

    # ── Build payoff matrix ──
    base_payoffs = PAYOFF_TABLES.get(game_type, PAYOFF_SNOWDRIFT).copy()
    base_payoffs[DEFECT, COOP] = temptation  # override T
    base_payoffs[COOP, DEFECT] = sucker_payoff  # override S

    # ── Offsets ──
    offsets = NEIGHBORHOODS.get(neighborhood, NEIGHBORHOODS["moore"])

    # ── Initialize ──
    init_fn = INIT_MODES.get(init_mode, _init_random)
    state = init_fn(gh, gw, rng, init_coop)

    # ── Update function ──
    update_fn = UPDATE_RULES.get(update_rule)
    if update_fn is None:
        update_fn = UPDATE_RULES["imitate_best"]

    # ── Render function ──
    render_fn = RENDER_STYLES.get(render_style, _render_default)

    # ── Internal simulation loop ──
    payoffs = _play_round(state, base_payoffs, gh, gw, offsets)
    img = render_fn(state, payoffs, gh, gw)
    pil_img = Image.fromarray(img).resize((W, H), Image.BILINEAR)
    result = np.asarray(pil_img, dtype=np.uint8)
    save(result, mn(125, f"SPD step=0 game={game_type}"), out_dir)
    capture_frame("125", result)

    for frame in range(1, n_frames):
        # The payoff matrix only depends on `frame` (sweep modes) or is
        # constant across the inner sub-steps. Build it ONCE per frame
        # instead of re-copying base_payoffs on every sub-step — identical
        # math, fewer redundant array copies.
        current_payoffs = np.array(base_payoffs, copy=True)

        if anim_mode == "sweep_temptation":
            current_payoffs[DEFECT, COOP] = 1.0 + (temptation - 1.0) * (frame / n_frames)

        mr = mutation_rate
        if anim_mode == "sweep_mutation":
            mr = mutation_rate * (frame / n_frames)
        if anim_mode == "sweep_S":
            current_payoffs[COOP, DEFECT] = sucker_payoff * (frame / n_frames)

        # Periodic stir: inject a random disturbance
        if anim_mode == "periodic_stir" and frame % 15 == 0:
            stir_r = max(3, min(gh, gw) // 10)
            si = rng.integers(stir_r, gh - stir_r)
            sj = rng.integers(stir_r, gw - stir_r)
            y, x = np.ogrid[-si:gh - si, -sj:gw - sj]
            mask = x * x + y * y < stir_r * stir_r
            state[mask] = 1 - state[mask]

        if anim_mode == "invasion":
            # Keep re-seeding the defector patch
            pass  # initial seed was set at init

        for _ in range(steps_per_frame):
            state, payoffs = update_fn(state, current_payoffs, gh, gw, rng, mr, offsets, fermi_K)

        # Render & capture
        payoffs = _play_round(state, base_payoffs, gh, gw, offsets)
        img = render_fn(state, payoffs, gh, gw)
        pil_img = Image.fromarray(img).resize((W, H), Image.BILINEAR)
        result = np.asarray(pil_img, dtype=np.uint8)
        coop_ratio = np.mean(state == COOP)
        save(result, mn(125, f"SPD step={frame} c={coop_ratio:.2f}"), out_dir)
        capture_frame("125", result)

    return result
