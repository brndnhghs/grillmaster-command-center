"""Shootout test plan (docs/plans/2026-07-10-shootout-evolutionary-generator-plan.md §15).

Fast suite: generator/repair fuzz (subset), evaluator synthetic stacks,
evolve validity + selection, features determinism, taste-vs-baseline,
endpoint lifecycle + route order. The 1000-genome fuzz and a real render
smoke are marked slow.
"""
from __future__ import annotations

import json
import random
import shutil

import numpy as np
import pytest

import image_pipeline.methods  # noqa: F401
from image_pipeline.shootout.config import ShootoutConfig
from image_pipeline.shootout.generator import (
    build_gene_pool, random_genome, sample_params,
    _ensure_animated, _graph_has_animation_source,
)
from image_pipeline.shootout.repair import (
    repair_genome, sample_valid_genome, validate_graph,
)
from image_pipeline.shootout.evaluator import evaluate_frames
from image_pipeline.shootout.evolve import (
    crossover, mutate, next_generation, select_parents,
)
from image_pipeline.shootout.features import genome_features
from image_pipeline.shootout import taste, store

CFG = ShootoutConfig()
POOL = build_gene_pool(CFG)


# ── Born-animated floor (Route 8, 2026-07-14) ──────────────────────────
# Guarantees every generated graph carries at least one animation source so
# the liveness gate never wastes a full render on a *genuinely* frozen clip.

def _no_source_graph(cfg, rng):
    """A single architecture-A terminal with no time param, no anim_mode,
    no driver — the exact shape that produces a statically-dead clip."""
    return {
        "version": 1, "name": "",
        "nodes": [{
            "id": "n1", "method_id": "238",
            "params": sample_params(POOL, cfg, rng, "238", False),
            "x": 0, "y": 0, "render": True,
        }],
        "edges": [],
    }


def test_ensure_animated_adds_driver_to_source_less_graph():
    import copy
    cfg = copy.deepcopy(CFG)
    cfg.guarantee_born_animated = True
    g = _no_source_graph(cfg, random.Random(1))
    assert not _graph_has_animation_source(g, POOL)
    out = _ensure_animated(g, POOL, cfg, random.Random(2))
    assert _graph_has_animation_source(out, POOL)
    drv = [n for n in out["nodes"]
           if n["method_id"] in POOL.scalar_drivers]
    assert len(drv) == 1
    # the driver is wired to the terminal's first free driver target
    assert any(e["dst_node"] == "n1" and e["src_node"] == drv[0]["id"]
               for e in out["edges"])


def test_ensure_animated_is_idempotent():
    import copy
    cfg = copy.deepcopy(CFG)
    cfg.guarantee_born_animated = True
    g = _no_source_graph(cfg, random.Random(1))
    out = _ensure_animated(g, POOL, cfg, random.Random(2))
    out2 = _ensure_animated(out, POOL, cfg, random.Random(3))
    drv = [n for n in out2["nodes"]
           if n["method_id"] in POOL.scalar_drivers]
    assert len(drv) == 1  # no second driver added


def test_ensure_animated_disabled_is_noop():
    import copy
    cfg = copy.deepcopy(CFG)
    cfg.guarantee_born_animated = False
    g = _no_source_graph(cfg, random.Random(1))
    out = _ensure_animated(g, POOL, cfg, random.Random(2))
    assert out is g  # unchanged when the floor is disabled


def test_random_genome_is_born_animated():
    import copy
    cfg = copy.deepcopy(CFG)
    cfg.guarantee_born_animated = True
    missing = 0
    for i in range(40):
        g = random_genome(POOL, cfg, random.Random(7000 + i), origin="test")
        if not _graph_has_animation_source(g["graph"], POOL):
            missing += 1
    assert missing == 0, f"{missing}/40 generated genomes lack an animation source"


@pytest.fixture()
def tmp_store(tmp_path, monkeypatch):
    """Redirect the shootout data dir into tmp so tests never touch the
    real ratings dataset or settings overrides."""
    from image_pipeline.shootout import config as cfg_mod
    monkeypatch.setattr(store, "DATA_DIR", tmp_path)
    monkeypatch.setattr(store, "GENOMES_DIR", tmp_path / "genomes")
    monkeypatch.setattr(store, "SESSIONS_DIR", tmp_path / "sessions")
    monkeypatch.setattr(store, "RATINGS_PATH", tmp_path / "ratings.jsonl")
    monkeypatch.setattr(store, "MODEL_PATH", tmp_path / "taste_model.json")
    monkeypatch.setattr(cfg_mod, "_OVERRIDES_PATH", tmp_path / "config.json")
    return tmp_path


# ── Generator + repair ────────────────────────────────────────────────


@pytest.mark.slow
def test_fuzz_generator_repair_fast():
    # NOTE: sample_valid_genome is expensive (~1-2s warm, ~15s cold per call)
    # because the shootout generator exercises real node execution. 150
    # iterations therefore take several minutes, so this belongs in the slow
    # suite alongside its 1000-iteration sibling below. The "fast" name only
    # distinguishes it from the 1000-iteration variant.
    rng = random.Random(42)
    for _ in range(150):
        g = sample_valid_genome(POOL, CFG, rng)
        issues = validate_graph(g["graph"], POOL, CFG)
        assert not issues, issues
        # executor-level structural checks: topo sort + terminal resolution
        from image_pipeline.core.graph import GraphExecutor, GraphNode, GraphEdge
        nodes = [GraphNode(**{k: v for k, v in n.items()
                              if k in GraphNode.__dataclass_fields__})
                 for n in g["graph"]["nodes"]]
        edges = [GraphEdge(**{k: v for k, v in e.items()
                              if k in GraphEdge.__dataclass_fields__})
                 for e in g["graph"]["edges"]]
        ex = GraphExecutor.__new__(GraphExecutor)
        order = ex._topo_sort(nodes, edges)   # raises GraphError on a cycle
        assert len(order) == len(nodes)
        terminal = ex._find_terminal(nodes, edges, order)
        assert terminal is not None


@pytest.mark.slow
def test_fuzz_generator_repair_1000():
    rng = random.Random(1)
    for _ in range(1000):
        g = sample_valid_genome(POOL, CFG, rng)
        assert not validate_graph(g["graph"], POOL, CFG)


def test_repair_fixes_broken_graphs():
    rng = random.Random(7)
    g = random_genome(POOL, CFG, rng)
    graph = g["graph"]
    # break it: cycle, double render flag, illegal edge, out-of-range param
    if len(graph["nodes"]) >= 2:
        a, b = graph["nodes"][0]["id"], graph["nodes"][1]["id"]
        graph["edges"].append({"src_node": a, "src_port": "image",
                               "dst_node": b, "dst_port": "image_in"})
        graph["edges"].append({"src_node": b, "src_port": "image",
                               "dst_node": a, "dst_port": "image_in"})
    for n in graph["nodes"]:
        n["render"] = True
        for k, spec in (POOL.defs[n["method_id"]].get("params") or {}).items():
            if isinstance(spec, dict) and spec.get("max") is not None:
                n["params"][k] = spec["max"] * 100
    graph["edges"].append({"src_node": "ghost", "src_port": "image",
                           "dst_node": graph["nodes"][0]["id"],
                           "dst_port": "image_in"})
    fixed = repair_genome(g, POOL, CFG)
    assert fixed is not None
    assert not validate_graph(fixed["graph"], POOL, CFG)


def test_repair_discards_unrenderable():
    genome = {
        "genome_id": "g-test", "generation": 0, "parents": [],
        "origin": "random", "seed": 1,
        "graph": {"version": 1, "name": "x",
                  "nodes": [{"id": "n1", "method_id": "__lfo__",
                             "params": {}, "render": False}],
                  "edges": []},
    }
    assert repair_genome(genome, POOL, CFG) is None


# ── Evaluator (synthetic frame stacks) ────────────────────────────────


def _stack(fn, n=24, h=64, w=96):
    return [fn(i) for i in range(n)]


def test_evaluator_black_is_flat():
    frames = _stack(lambda i: np.zeros((64, 96, 3), np.float32))
    s = evaluate_frames(frames, CFG)
    assert not s["alive"] and s["reason"] == "flat"


def test_evaluator_spectral_rescue_low_amplitude_coherent():
    """A full-frame tiny-amplitude coherent oscillation (slow breathe) has
    temporal_var and motion_pixel_frac both below the amplitude floors yet is
    genuinely animating. The spectral-liveness rescue must detect the sharp FFT
    spectral peak and keep it alive (Route 8 follow-up, 2026-07-13)."""
    amp = 0.015  # below motion_thresh (0.03)
    grad = (np.arange(96, dtype=np.float32) / 96.0)[None, :]
    def f(i):
        a = np.zeros((64, 96, 3), np.float32)
        a[:] = grad[:, :, None]
        a[:] += amp * float(np.sin(2 * np.pi * i / 24.0))
        return np.clip(a, 0, 1)
    s = evaluate_frames(_stack(f, n=48), CFG)
    assert s["alive"], s
    assert s["spectral_peak"] >= CFG.spectral_corr_min, s


def test_evaluator_spectral_rescue_localized_coherent():
    """A localized low-amplitude coherent breathing blob is also rescued —
    the mean normalized spectral peak is computed only over AC-active pixels so
    a static background does not dilute it."""
    amp = 0.015
    yy, xx = np.mgrid[0:64, 0:96]
    m = ((yy - 32) ** 2 + (xx - 48) ** 2) < 200
    def f(i):
        a = np.full((64, 96, 3), 0.4, np.float32)
        a[m] += amp * float(np.sin(2 * np.pi * i / 16.0))
        return np.clip(a, 0, 1)
    s = evaluate_frames(_stack(f, n=48), CFG)
    assert s["alive"], s
    assert s["spectral_peak"] >= CFG.spectral_corr_min, s


def test_evaluator_spectral_rescue_sparse_small_coverage():
    """Regression (Route 8, 2026-07-13): a SMALL localized coherent breathing
    blob whose coverage is BELOW the 3% motion-rescue floor (≈1.3% of pixels)
    is genuinely alive but was wrongly culled as 'flat'/'static' under the old
    spectral rescue, which reused ``motion_pixel_frac_min`` (0.03) as its
    coverage floor. The dedicated ``spectral_coverage_min`` (0.01) + per-active-
    pixel AC floor must now rescue it, while a single flickering pixel
    (coverage < 0.01) must stay dead. This is exactly the pipeline's
    sparse-content niche (thin strokes / small particles, pitfall #13)."""
    yy, xx = np.mgrid[0:64, 0:96]
    m = ((yy - 32) ** 2 + (xx - 48) ** 2) <= 25   # ~81 px / 6144 ≈ 1.3%
    def f(i):
        a = np.full((64, 96, 3), 0.05, np.float32)
        # clean unclipped coherent breathe: dot oscillates 0.1..0.9 (stays in [0,1])
        a[m] = 0.5 + 0.4 * float(np.sin(2 * np.pi * i / 16.0))
        return a
    s = evaluate_frames(_stack(f, n=48), CFG)
    assert s["alive"], s
    assert s["spectral_peak"] >= CFG.spectral_corr_min, s
    # coverage must be under the old 3% floor or this test does not exercise
    # the bug it claims to (it would pass on the old code too).
    assert s["spectral_active_frac"] < CFG.motion_pixel_frac_min, s

    # Control: a single periodic pixel (~1.6e-5 coverage) must NOT be rescued.
    def g(i):
        a = np.full((64, 96, 3), 0.05, np.float32)
        a[10, 20] = 0.05 + 0.4 * (0.5 + 0.5 * float(np.sin(2 * np.pi * i / 8.0)))
        return a
    s2 = evaluate_frames(_stack(g, n=48), CFG)
    assert not s2["alive"], s2


def test_evaluator_frozen_not_spectral_rescued():
    """A frozen frame has no AC energy, so the spectral rescue must NOT admit
    it (guards against a spurious numerical single-bin FFT peak)."""
    grad = (np.arange(96, dtype=np.float32) / 96.0)[None, :]
    def f(i):
        a = np.zeros((64, 96, 3), np.float32)
        a[:] = grad[:, :, None]
        return a
    s = evaluate_frames(_stack(f, n=48), CFG)
    assert not s["alive"], s
    assert s["spectral_active_frac"] == 0.0, s


def test_evaluator_flicker_not_spectral_rescued():
    """Flat (white) noise has a flat temporal spectrum — normalized peak ~1/K —
    so the spectral rescue must not admit it either."""
    rng = np.random.default_rng(5)
    s = evaluate_frames(
        _stack(lambda i: rng.random((64, 96, 3)).astype(np.float32) * 0.02 + 0.4,
               n=48), CFG)
    assert not s["alive"], s
    assert s["spectral_peak"] < CFG.spectral_corr_min, s
    rng = np.random.default_rng(0)
    img = rng.random((64, 96, 3), dtype=np.float32)
    s = evaluate_frames(_stack(lambda i: img), CFG)
    assert not s["alive"] and s["reason"] == "static"


def test_evaluator_flow_rescue_structured_drift_rejects_flicker_and_static():
    """Sub-problem #3 optical-flow rescue (2026-07-14): a small bright disk
    making a single non-periodic drift across the canvas is genuinely animating
    (pixels move) but is missed by every other rescue — temporal_var is low
    (tiny moving region), motion_pixel_frac is below its 3% floor (sparse
    content), and the motion is non-periodic so it has no sharp spectral peak.
    Dense optical flow must detect the structured displacement (high flow
    magnitude variance + high direction coherence) and keep it alive, while a
    static frame (~0 flow) and incoherent flicker (low coherence) stay dead.
    Frames are sized (256x384 -> 128x192 stride-2 flow buffer) so Farnebäck's
    winsize resolves the motion rather than smoothing it away."""
    H, W = 256, 384
    yy, xx = np.mgrid[0:H, 0:W]
    r = 18
    n = 12
    x0, x1 = 60.0, 60.0 + 3.0 * (n - 1)  # ~3 px/frame, non-periodic (sparse drift)
    def f(i):
        a = np.zeros((H, W, 3), np.float32)
        cx = x0 + (x1 - x0) * (i / (n - 1))
        m = (yy - H // 2) ** 2 + (xx - cx) ** 2 < r * r
        a[m] = 1.0
        return a
    s = evaluate_frames(_stack(f, n=n, h=H, w=W), CFG)
    assert s["alive"], s
    assert s["flow_var"] >= CFG.flow_var_min, s
    assert s["flow_coherence"] >= CFG.flow_coherence_min, s

    # Control 1: static frame -> dead, ~0 flow.
    s_static = evaluate_frames(_stack(lambda i: f(0), h=H, w=W), CFG)
    assert not s_static["alive"], s_static
    assert s_static["flow_var"] < CFG.flow_var_min, s_static

    # Control 2: incoherent low-amplitude flicker -> dead, low coherence.
    rng = np.random.default_rng(7)
    s_flick = evaluate_frames(
        _stack(lambda i: rng.random((H, W, 3)).astype(np.float32) * 0.02,
               n=n, h=H, w=W), CFG)
    assert not s_flick["alive"], s_flick
    assert s_flick["flow_coherence"] < CFG.flow_coherence_min, s_flick


def test_evaluator_flow_telemetry_independent_of_other_rescues():
    """The optical-flow signal (flow_var / flow_coherence) is computed
    independently of the amplitude and spectral rescues, so it remains a valid
    liveness diagnostic even when those rescues are disabled for tuning. A
    structured-drift clip reports high flow_var + coherence regardless; a
    static clip reports ~0 flow under the same config."""
    H, W = 256, 384
    yy, xx = np.mgrid[0:H, 0:W]
    r = 18
    n = 12
    x0, x1 = 60.0, 60.0 + 3.0 * (n - 1)
    def f(i):
        a = np.zeros((H, W, 3), np.float32)
        cx = x0 + (x1 - x0) * (i / (n - 1))
        m = (yy - H // 2) ** 2 + (xx - cx) ** 2 < r * r
        a[m] = 1.0
        return a
    cfg = ShootoutConfig()
    cfg.motion_pixel_frac_min = 1e9  # perceptual rescue off
    cfg.spectral_corr_min = 1e9      # spectral rescue off
    s = evaluate_frames(_stack(f, n=n, h=H, w=W), cfg)
    assert s["flow_var"] >= CFG.flow_var_min, s
    assert s["flow_coherence"] >= CFG.flow_coherence_min, s
    # Static clip still reports ~0 flow under the same config.
    s_static = evaluate_frames(_stack(lambda i: f(0), h=H, w=W), cfg)
    assert s_static["flow_var"] < CFG.flow_var_min, s_static


def test_evaluator_flow_rescue_subthreshold_drift():
    """Sub-problem #3 optical-flow rescue, COMPLETED (2026-07-14): a small
    disk making a SLOW, coherent drift is genuinely animating (pixels move) but
    falls below EVERY other rescue floor — temporal_var < temporal_var_min
    (low global variance), motion_pixel_frac < motion_pixel_frac_min (sparse
    coverage), and no sharp spectral peak. Dense optical flow detects the
    structured displacement (high flow_var + high flow_coherence) and keeps it
    alive. This is the case the prior "flow rescue" telemetry computed but the
    verdict never consulted, so the clip was wrongly culled as "static".

    Controls: a static (non-drifting) disk has ~0 flow -> stays dead; with the
    perceptual + spectral rescues disabled the drift is STILL rescued by flow
    alone (proves flow is the active rescue, not a no-op)."""
    H, W, r, n = 128, 192, 6, 24
    yy, xx = np.mgrid[0:H, 0:W]
    ppf = 1
    x0 = W / 2 - (n - 1) * ppf / 2
    x1 = x0 + (n - 1) * ppf

    def drift(i):
        a = np.zeros((H, W, 3), np.float32)
        cx = x0 + (x1 - x0) * (i / (n - 1))
        m = (yy - H // 2) ** 2 + (xx - cx) ** 2 < r * r
        a[m] = 1.0
        return a

    s = evaluate_frames(_stack(drift, n=n, h=H, w=W), CFG)
    assert s["alive"], s
    assert s["reason"] is None, s
    assert s["flow_var"] >= CFG.flow_var_min, s
    assert s["flow_coherence"] >= CFG.flow_coherence_min, s
    # It really was sub-threshold for the other rescues:
    assert s["temporal_var"] < CFG.temporal_var_min, s
    assert s["motion_pixel_frac"] < CFG.motion_pixel_frac_min, s

    # Control: static disk -> dead, ~0 flow.
    s_static = evaluate_frames(_stack(lambda i: drift(0), h=H, w=W), CFG)
    assert not s_static["alive"], s_static
    assert s_static["flow_var"] < CFG.flow_var_min, s_static

    # Flow alone rescues even with the perceptual + spectral rescues disabled.
    cfg = ShootoutConfig()
    cfg.motion_pixel_frac_min = 1e9
    cfg.spectral_corr_min = 1e9
    s_off = evaluate_frames(_stack(drift, n=n, h=H, w=W), cfg)
    assert s_off["alive"], s_off
    assert s_off["flow_var"] >= CFG.flow_var_min, s_off


def _chroma_only_frames(n=48, amp=0.3, period=24):
    """Synthetic clip whose per-pixel MEAN luminance is a fixed spatial gradient
    (constant over time) while the CHROMA rotates in the plane orthogonal to the
    (1,1,1) mean-luminance axis.

    ``U`` / ``V`` are two orthonormal directions perpendicular to (1,1,1), so
    ``rgb = lum*1 + amp*(U*cos t + V*sin t)`` keeps the channel-MEAN (the
    grayscale the evaluator collapses RGB to) exactly ``lum`` for every ``t``.
    The grayscale-based rescues (temporal variance, perceptual motion, spectral,
    optical-flow) therefore see a frozen clip; only the color-aware rescue can
    detect the chroma motion.
    """
    U = np.array([1.0, -1.0, 0.0], np.float32)
    U = U / np.linalg.norm(U)
    V = np.array([1.0, 1.0, -2.0], np.float32)
    V = V / np.linalg.norm(V)
    grad = (np.arange(96, dtype=np.float32) / 96.0)[None, :]
    lum = (0.3 + 0.4 * grad)[:, :, None]            # [0.3, 0.7], constant over time

    def f(i):
        t = 2.0 * np.pi * i / period                 # smooth coherent rotation
        rgb = lum + amp * (U[None, None, :] * np.cos(t)
                           + V[None, None, :] * np.sin(t))
        return np.clip(rgb, 0.0, 1.0)

    return _stack(f, n=n)


def test_evaluator_color_rescue_chroma_only_oscillation():
    """Color-aware liveness rescue (Route 8 sub-problem #3 closure, 2026-07-16):
    a clip whose HUES/CHANNELS cycle at CONSTANT luminance is genuinely
    animating but reads as 'static' to every grayscale-based rescue (temporal
    variance, perceptual motion, spectral, optical-flow all collapse the color
    vector to the channel mean). A driver modulating an --recolor ``palette``
    (cosmetic color per the color-architecture rule), a LUT/hue-sweep filter, or
    a color_intrinsic method with a luminance-fixed hue sweep is exactly this
    case — the 643-genome scan's 211 static+flat deaths are its fingerprint. The
    color-aware rescue measures per-pixel chroma change on the luminance-
    preserving buffer and must keep it alive, while a frozen color and
    incoherent hue noise stay dead (see the controls below)."""
    s = evaluate_frames(_chroma_only_frames(n=48), CFG)
    assert s["alive"], s
    assert s["reason"] is None, s
    assert s["color_change_frac"] >= CFG.color_change_frac_min, s
    assert s["color_struct_corr"] >= CFG.color_corr_min, s
    # It was sub-threshold for EVERY grayscale-based rescue, so the COLOR rescue
    # is the only thing that could have flipped it alive:
    assert s["temporal_var"] < CFG.temporal_var_min, s
    assert s["motion_pixel_frac"] < CFG.motion_pixel_frac_min, s
    assert s["spectral_peak"] < CFG.spectral_corr_min, s
    assert s["flow_var"] < CFG.flow_var_min, s

    # Color alone rescues even with the other rescues disabled.
    cfg = ShootoutConfig()
    cfg.motion_pixel_frac_min = 1e9
    cfg.spectral_corr_min = 1e9
    cfg.flow_var_min = 1e9
    s_off = evaluate_frames(_chroma_only_frames(n=48), cfg)
    assert s_off["alive"], s_off
    assert s_off["color_change_frac"] >= CFG.color_change_frac_min, s_off


def test_evaluator_color_rescue_frozen_chroma_stays_dead():
    """Control: a clip whose color is frozen (no chroma change) must NOT be
    rescued even though its luminance is spatially structured — there is no
    motion of any kind."""
    U = np.array([1.0, -1.0, 0.0], np.float32)
    U = U / np.linalg.norm(U)
    grad = (np.arange(96, dtype=np.float32) / 96.0)[None, :]
    lum = (0.3 + 0.4 * grad)[:, :, None]

    def g(i):
        return np.clip(lum + 0.3 * U[None, None, :], 0.0, 1.0)  # fixed hue

    s = evaluate_frames(_stack(g, n=48), CFG)
    assert not s["alive"], s
    assert s["color_change_frac"] < CFG.color_change_frac_min, s


def test_evaluator_color_rescue_incoherent_hue_noise_stays_dead():
    """Control: per-PIXEL independent random chroma (luminance-preserving) changes
    color every frame but is temporally INCOHERENT (color_struct_corr ~0), so it
    must not be rescued — only STRUCTURED chroma motion (a palette sweep, a
    coherent hue cycle) is alive. A uniform-luminance canvas avoids the lum
    signal drowning the (low) chroma correlation, isolating the color gate."""
    U = np.array([1.0, -1.0, 0.0], np.float32)
    U = U / np.linalg.norm(U)
    V = np.array([1.0, 1.0, -2.0], np.float32)
    V = V / np.linalg.norm(V)
    rng = np.random.default_rng(7)

    def noise(i):
        a = rng.standard_normal((64, 96)).astype(np.float32)
        b = rng.standard_normal((64, 96)).astype(np.float32)
        chroma = 0.3 * (U[None, None, :] * a[..., None]
                        + V[None, None, :] * b[..., None])
        # uniform luminance (0.5) so spatial_var is degenerate but the clip is
        # clearly NOT alive — the color gate must reject the incoherent chroma.
        return np.clip(0.5 + chroma, 0.0, 1.0)

    s = evaluate_frames(_stack(noise, n=48), CFG)
    assert not s["alive"], s
    # color DOES change...
    assert s["color_change_frac"] >= CFG.color_change_frac_min, s
    # ...but incoherently, so the rescue correctly rejects it:
    assert s["color_struct_corr"] < CFG.color_corr_min, s


def test_evaluator_nan_is_dead():
    def f(i):
        a = np.ones((64, 96, 3), np.float32) * (i / 24)
        a[0, 0, 0] = np.nan
        return a
    s = evaluate_frames(_stack(f), CFG)
    assert not s["alive"] and s["reason"] == "nan"


def test_evaluator_moving_is_alive():
    def f(i):
        a = np.zeros((64, 96, 3), np.float32)
        a[:, (i * 4) % 96: (i * 4) % 96 + 12] = 1.0
        return a
    s = evaluate_frames(_stack(f), CFG)
    assert s["alive"], s


def test_evaluator_flicker_is_dead():
    rng = np.random.default_rng(3)
    s = evaluate_frames(
        _stack(lambda i: rng.random((64, 96, 3)).astype(np.float32)), CFG)
    assert not s["alive"] and s["reason"] == "flicker", s


def test_evaluator_missing_frames_dead():
    s = evaluate_frames([None] * 24, CFG)
    assert not s["alive"] and s["reason"] == "no-output"


def test_timeout_recovers_slow_tailed_dynamic_clip():
    """A clip that hits the wall-clock cap but captured most frames with real
    motion must NOT be culled as 'timeout' (Route 8, item 2). Regression guard
    against the old unconditional ``timed_out -> alive=False`` override."""
    from image_pipeline.shootout.evaluator import LivenessAccumulator

    cfg = ShootoutConfig()
    frames = cfg.frames
    min_frames = int(frames * cfg.min_render_frames_frac)
    # Capture min_frames+1 dynamic frames: a spatial gradient whose vertical
    # offset shifts each frame (real spatial structure AND temporal motion),
    # then stop (cap hit).
    acc = LivenessAccumulator(cfg)
    hs = np.linspace(0.0, 1.0, 96, dtype=np.float32)
    for i in range(min_frames + 1):
        shift = (i * 6) % 96
        col = np.roll(hs, shift)
        acc.add(np.tile(col, (64, 1))[:, :, None].repeat(3, axis=-1))
    # Replicate the render_genome timeout branch exactly.
    stats = acc.stats()
    captured = acc.total - acc.missing
    timed_out = True
    if timed_out:
        if captured >= min_frames and stats.get("alive"):
            stats = {**stats, "truncated": True, "reason": stats.get("reason")}
        else:
            stats = {**stats, "alive": False, "reason": "timeout"}
    assert stats["alive"], stats
    assert stats.get("truncated") is True, stats


def test_timeout_still_culls_too_short_clip():
    """A clip that barely rendered before the cap hit stays culled as 'timeout'."""
    from image_pipeline.shootout.evaluator import LivenessAccumulator

    cfg = ShootoutConfig()
    frames = cfg.frames
    min_frames = int(frames * cfg.min_render_frames_frac)
    acc = LivenessAccumulator(cfg)
    # Capture only 10% of the budget, all static (no motion at all).
    for _ in range(int(frames * 0.1)):
        acc.add(np.zeros((64, 96, 3), np.float32))
    stats = acc.stats()
    captured = acc.total - acc.missing
    timed_out = True
    if timed_out:
        if captured >= min_frames and stats.get("alive"):
            stats = {**stats, "truncated": True, "reason": stats.get("reason")}
        else:
            stats = {**stats, "alive": False, "reason": "timeout"}
    assert not stats["alive"] and stats["reason"] == "timeout", stats


# ── Evolve ────────────────────────────────────────────────────────────


def _rated_generation(rng, ratings):
    out = []
    for r in ratings:
        g = sample_valid_genome(POOL, CFG, rng)
        g["rating"] = r
        out.append(g)
    return out


def test_offspring_are_valid():
    rng = random.Random(5)
    prev = _rated_generation(rng, [5, 4, 3, 2])
    for _ in range(30):
        child = mutate(rng.choice(prev[:2]), POOL, CFG, rng, 1)
        assert child is None or not validate_graph(child["graph"], POOL, CFG)
        cx = crossover(prev[0], prev[1], POOL, CFG, rng, 1)
        assert cx is None or not validate_graph(cx["graph"], POOL, CFG)


def test_graph_distance_bounds_and_identical():
    from image_pipeline.shootout.evolve import graph_distance
    a = sample_valid_genome(POOL, CFG, random.Random(1))
    assert graph_distance(a, a, POOL) == 0.0          # identical → 0
    b = sample_valid_genome(POOL, CFG, random.Random(2))
    d = graph_distance(a, b, POOL)
    assert 0.0 <= d <= 1.0                            # bounded
    # a deliberately different graph (different node types) is not ~0
    assert d > 0.0


def test_mutation_reaches_min_divergence():
    """Bred offspring should clear cfg.min_divergence from the parent at
    least most of the time, and every mutation records a divergence field."""
    from image_pipeline.shootout.evolve import graph_distance
    rng = random.Random(5)
    prev = _rated_generation(rng, [5, 4, 3, 2])
    hit = 0
    n = 60
    for _ in range(n):
        parent = rng.choice(prev[:2])
        child = mutate(parent, POOL, CFG, rng, 1)
        assert child is not None, "divergence loop must always produce a child"
        assert "divergence" in child["deviation"]
        assert graph_distance(parent, child, POOL) == child["deviation"]["divergence"]
        if child["deviation"]["divergence"] >= CFG.min_divergence:
            hit += 1
    # The escalation loop must drive the overwhelming majority past the floor.
    assert hit >= n * 0.8, f"only {hit}/{n} cleared min_divergence={CFG.min_divergence}"


def test_high_min_divergence_forces_more_intensity():
    """Raising min_divergence should push the breeder to use higher
    intensity (more mutation ops) than the default."""
    from image_pipeline.shootout.evolve import mutate
    rng = random.Random(5)
    prev = _rated_generation(rng, [5, 4, 3, 2])
    low = ShootoutConfig(min_divergence=0.1, max_divergence_attempts=8)
    high = ShootoutConfig(min_divergence=0.6, max_divergence_attempts=8)
    rng_low, rng_high = random.Random(99), random.Random(99)
    prev_low = _rated_generation(rng_low, [5, 4, 3, 2])
    prev_high = _rated_generation(rng_high, [5, 4, 3, 2])
    ints_low, ints_high = [], []
    for p in prev_low[:10]:
        c = mutate(p, POOL, low, random.Random(1), 1)
        assert c is not None
        ints_low.append(c["deviation"]["intensity"])
    for p in prev_high[:10]:
        c = mutate(p, POOL, high, random.Random(1), 1)
        assert c is not None
        ints_high.append(c["deviation"]["intensity"])
    assert sum(ints_high) > sum(ints_low), \
        f"higher min_divergence should need more intensity ({ints_high} vs {ints_low})"


def test_gentle_mutation_still_valid_and_recorded():
    rng = random.Random(5)
    prev = _rated_generation(rng, [5, 4, 3, 2])
    child = mutate(prev[0], POOL, CFG, rng, 1, gentle=True)
    assert child is not None
    assert child["deviation"]["kind"] == "protected"
    assert "divergence" in child["deviation"]


def test_crossover_never_produces_duplicate_ids():
    """Regression: the terminal-variance guard seeds a Builder from a crossed
    graph whose ids may be non-contiguous (donor subtree remapped to
    nNx....). Appending driver/filter nodes must not collide with an existing
    id. Previously ~4% of crossovers shipped a duplicate id + self-loop that
    repair silently dropped, thinning the evolved variety."""
    rng = random.Random(5)
    prev = _rated_generation(rng, [5, 4, 3, 2])
    bad = 0
    for _ in range(120):
        pa, pb = rng.choice(prev), rng.choice(prev)
        cx = crossover(pa, pb, POOL, CFG, rng, 1)
        if cx is None:
            continue
        issues = validate_graph(cx["graph"], POOL, CFG)
        if issues:
            bad += 1
            print("bad crossover:", issues)
    assert bad == 0, f"{bad}/120 crossovers produced invalid graphs"


def test_selection_favors_high_stars():
    rng = random.Random(9)
    prev = _rated_generation(rng, [5, 4, 2, 1, None])
    parents, weights = select_parents(prev, CFG)
    assert len(parents) == 3          # 1★ and unrated never breed
    by_rating = {p["rating"]: w for p, w in zip(parents, weights)}
    assert by_rating[5] > by_rating[4] > by_rating[2]


def test_generation_composition():
    rng = random.Random(11)
    prev = _rated_generation(rng, [5, 4, 4, 3, 2, 1])
    gen = next_generation(prev, 1, POOL, CFG, rng)
    assert len(gen) == CFG.render_pool
    origins = [g["origin"] for g in gen]
    n_explore = origins.count("explorer")
    assert n_explore == max(1, round(CFG.explore_ratio * CFG.render_pool))
    assert all(not validate_graph(g["graph"], POOL, CFG) for g in gen)
    for g in gen:
        if g["origin"] in ("mutation", "crossover"):
            assert g["parents"], "bred offspring must record parents"


def test_no_parents_means_all_random():
    rng = random.Random(13)
    prev = _rated_generation(rng, [1, 1, None])
    gen = next_generation(prev, 1, POOL, CFG, rng)
    assert all(g["origin"] in ("random", "explorer") for g in gen)


def test_blended_liveness_breeding_supplements_sparse_ratings():
    """Route 8 follow-up: with a *sparse* rating pool, liveness-fitness
    (unrated but alive) genomes are blended into the breeder pool so the
    abundant liveness signal still drives evolution toward dynamic clips.
    Lightweight dicts: select_parents only reads rating + liveness."""
    prev = [
        {"rating": 5, "liveness": {"alive": True, "temporal_var": 0.05,
                                   "motion_pixel_frac": 0.4}},
        {"rating": 2, "liveness": {"alive": True, "temporal_var": 0.03,
                                   "motion_pixel_frac": 0.25}},
    ]
    for _ in range(5):
        prev.append({"rating": None,
                     "liveness": {"alive": True, "temporal_var": 0.05,
                                  "motion_pixel_frac": 0.4}})
    parents, weights = select_parents(prev, CFG)
    # 2 rated + 5 blended liveness parents
    assert len(parents) == 7
    assert any(g.get("rating") is None for g in parents)
    # rated parents remain present, with rating ordering intact
    rated = [(p["rating"], w) for p, w in zip(parents, weights)
             if p.get("rating") is not None]
    assert rated[0][0] == 5 and rated[1][0] == 2
    # blended (unrated) parents carry real liveness weight
    unrated = [(p, w) for p, w in zip(parents, weights)
               if p.get("rating") is None]
    assert len(unrated) == 5 and all(w > 0 for _, w in unrated)


def test_blended_liveness_off_when_ratings_plentiful():
    """Once humans rate enough clips (>= min_rated) the rated signal is
    trusted and liveness blending disables — pure rating behavior preserved."""
    prev = [{"rating": 5,
             "liveness": {"alive": True, "temporal_var": 0.05,
                          "motion_pixel_frac": 0.4}} for _ in range(25)]
    parents, weights = select_parents(prev, CFG)
    assert len(parents) == 25
    assert all(g.get("rating") is not None for g in parents)


# ── Features + taste ──────────────────────────────────────────────────


def test_features_deterministic():
    rng = random.Random(17)
    g = sample_valid_genome(POOL, CFG, rng)
    assert genome_features(g, POOL, CFG) == genome_features(g, POOL, CFG)
    f = genome_features(g, POOL, CFG)
    assert f["n_nodes"] == len(g["graph"]["nodes"])
    assert f["origin_random"] == 1.0


def test_taste_beats_baseline(tmp_store):
    rng = random.Random(19)
    rng_np = np.random.default_rng(19)
    # One sample pass: features and the synthetic target come from the SAME
    # genomes, so the target actually correlates with the features. The
    # target is depth + a little size, jittered, then linearly rescaled into
    # the 1–5 band so it keeps real spread regardless of how the generator
    # distributes depth (post motif-grammar most graphs are deep).
    feats: list[dict] = []
    raw: list[float] = []
    for _ in range(60):
        g = sample_valid_genome(POOL, CFG, rng)
        f = genome_features(g, POOL, CFG)
        feats.append(f)
        raw.append(f["depth"] + 0.15 * f["n_nodes"] + rng_np.normal(0, 0.15))
    lo, hi = min(raw), max(raw)
    recs = []
    for f, r in zip(feats, raw):
        norm = 1.5 + 3.0 * (r - lo) / max(hi - lo, 1e-9)
        recs.append({"features": f,
                     "rating": float(np.clip(round(norm), 1, 5))})
    art = taste.train(recs)
    assert art["trained"]
    assert art["metrics"]["beats_baseline"]
    assert art["metrics"]["cv_corr"] > 0.3
    pred = taste.predict(recs[0]["features"], art)
    assert 1.0 <= pred <= 5.0


def test_taste_needs_min_samples(tmp_store):
    art = taste.train([{"features": {"n_nodes": 1}, "rating": 3}] * 3)
    assert not art["trained"]


# ── Endpoints ─────────────────────────────────────────────────────────


@pytest.fixture()
def client(tmp_store):
    from fastapi.testclient import TestClient
    from image_pipeline.server import app
    return TestClient(app)


def test_session_lifecycle_and_route_order(client):
    r = client.post("/api/shootout/session", json={})
    assert r.status_code == 200
    sid = r.json()["session_id"]
    assert r.json()["generations"] == []

    # resume returns the same session
    r2 = client.post("/api/shootout/session", json={"session_id": sid})
    assert r2.json()["session_id"] == sid

    # static shootout routes must not be captured by /api/graph/{gid}
    # (route-order trap, plan §12): a genome miss must be OUR 404, not a
    # graph-doc response.
    r3 = client.get("/api/shootout/genome/does-not-exist")
    assert r3.status_code == 404
    assert "Genome" in r3.json()["detail"]

    r4 = client.get("/api/shootout/session/nope")
    assert r4.status_code == 404


def test_rating_persistence_roundtrip(client, tmp_store):
    from image_pipeline.shootout import session as sess
    rng = random.Random(23)
    s = sess.start_session()
    sid = s["session_id"]

    # fabricate a completed generation (no rendering needed for persistence)
    genomes = [sample_valid_genome(POOL, CFG, rng) for _ in range(3)]
    for g in genomes:
        store.save_genome(g)
    s = store.load_session(sid)
    s["generations"].append({
        "gen": 0, "shown": [g["genome_id"] for g in genomes],
        "pool": [g["genome_id"] for g in genomes],
        "ratings": {}, "rated_logged": [],
    })
    store.save_session(s)

    gids = [g["genome_id"] for g in genomes]
    r = client.post("/api/shootout/rate", json={
        "session_id": sid,
        "ratings": {gids[0]: 5, gids[1]: 2, "not-shown": 4},
    })
    assert r.status_code == 200
    assert r.json()["appended"] == 2   # not-shown is ignored

    # dataset round-trip
    lines = [json.loads(l) for l in
             (tmp_store / "ratings.jsonl").read_text().splitlines()]
    assert {l["genome_id"]: l["rating"] for l in lines} == {gids[0]: 5, gids[1]: 2}
    assert all("features" in l and l["session_id"] == sid for l in lines)

    # re-rating updates the session but does not double-append
    r = client.post("/api/shootout/rate", json={
        "session_id": sid, "ratings": {gids[0]: 3}})
    assert r.json()["appended"] == 0
    s = store.load_session(sid)
    assert s["generations"][-1]["ratings"][gids[0]] == 3

    # genome file carries the rating (session_state view)
    state = client.get(f"/api/shootout/session/{sid}").json()
    ratings = {sv["genome_id"]: sv["rating"] for sv in state["survivors"]}
    assert ratings[gids[0]] == 3 and ratings[gids[1]] == 2


# ── Notes + advisor guidance ──────────────────────────────────────────


def test_notes_roundtrip(client, tmp_store):
    from image_pipeline.shootout import session as sess
    rng = random.Random(29)
    s = sess.start_session()
    sid = s["session_id"]
    genomes = [sample_valid_genome(POOL, CFG, rng) for _ in range(2)]
    for g in genomes:
        store.save_genome(g)
    s = store.load_session(sid)
    s["generations"].append({
        "gen": 0, "shown": [g["genome_id"] for g in genomes],
        "pool": [], "ratings": {}, "notes": {}, "rated_logged": [],
    })
    store.save_session(s)
    g0, g1 = (g["genome_id"] for g in genomes)

    r = client.post("/api/shootout/rate", json={
        "session_id": sid,
        "ratings": {g0: 5},
        "notes": {g0: "love the motion, keep this structure",
                  g1: "too static, drop it"},
    })
    assert r.status_code == 200
    assert r.json()["noted"] == 2

    # lineage + genome + dataset all carry the note
    s = store.load_session(sid)
    assert s["generations"][-1]["notes"][g1] == "too static, drop it"
    assert store.load_genome(g0)["notes"].startswith("love the motion")
    line = json.loads((tmp_store / "ratings.jsonl").read_text().splitlines()[0])
    assert line["notes"].startswith("love the motion")

    # notes-only rate (no stars) is accepted
    r = client.post("/api/shootout/rate", json={
        "session_id": sid, "notes": {g1: "updated note"}})
    assert r.status_code == 200
    assert store.load_genome(g1)["notes"] == "updated note"

    # session_state surfaces notes for UI resume
    state = client.get(f"/api/shootout/session/{sid}").json()
    by_id = {sv["genome_id"]: sv for sv in state["survivors"]}
    assert by_id[g1]["notes"] == "updated note"


def _fake_llm_reply(reply):
    return lambda system, user: reply


def test_advisor_guidance_parsing(tmp_store):
    from image_pipeline.shootout import advisor
    rng = random.Random(31)
    rated = [sample_valid_genome(POOL, CFG, rng) for _ in range(2)]
    rated[0]["rating"] = 5
    rated[0]["notes"] = "more nodes please, love the physarum look"
    real_mid = rated[0]["graph"]["nodes"][0]["method_id"]

    reply = json.dumps({
        "prefer_methods": [real_mid, "does-not-exist"],
        "avoid_methods": [],
        "prefer_categories": ["simulations", "bogus-cat"],
        "avoid_categories": [],
        "complexity": "increase",
        "protect_genomes": [rated[0]["genome_id"], "g-unknown"],
        "drop_genomes": [],
        "summary": "grow graphs, favor simulations",
    })
    g = advisor.extract_guidance(rated, POOL, CFG, llm=_fake_llm_reply(
        "Sure! Here is the JSON:\n" + reply))
    assert g["prefer_methods"] == [real_mid]          # unknown id stripped
    assert g["prefer_categories"] == ["simulations"]  # bogus category stripped
    assert g["protect_genomes"] == [rated[0]["genome_id"]]
    assert g["complexity"] == "increase"

    bias = advisor.bias_from_guidance(g)
    assert bias.complexity > 0 and real_mid in bias.prefer_methods

    # no notes → no LLM call, no guidance
    for r in rated:
        r["notes"] = ""
    assert advisor.extract_guidance(rated, POOL, CFG,
                                    llm=_fake_llm_reply(reply)) is None
    # unparseable reply → None
    rated[0]["notes"] = "x"
    assert advisor.extract_guidance(rated, POOL, CFG,
                                    llm=_fake_llm_reply("no json here")) is None
    # LLM unavailable → None
    assert advisor.extract_guidance(rated, POOL, CFG,
                                    llm=lambda s, u: None) is None


def test_guidance_steers_generation():
    rng = random.Random(37)
    prev = _rated_generation(rng, [None] * 4)   # nothing rated → all explorers
    guidance = {"prefer_methods": [], "avoid_methods": [],
                "prefer_categories": [], "avoid_categories": ["gpu_shaders"],
                "complexity": "increase", "protect_genomes": [],
                "drop_genomes": [], "summary": ""}
    gen = next_generation(prev, 1, POOL, CFG, rng, guidance=guidance)
    for g in gen:
        assert all(POOL.defs[n["method_id"]]["category"] != "gpu_shaders"
                   for n in g["graph"]["nodes"]), "avoid_categories ignored"
        assert not validate_graph(g["graph"], POOL, CFG)


def test_guidance_drops_parents():
    rng = random.Random(41)
    prev = _rated_generation(rng, [5, 4])
    dropped = prev[0]["genome_id"]
    guidance = {"prefer_methods": [], "avoid_methods": [],
                "prefer_categories": [], "avoid_categories": [],
                "complexity": "keep", "protect_genomes": [],
                "drop_genomes": [dropped], "summary": ""}
    for _ in range(5):
        gen = next_generation(prev, 1, POOL, CFG, rng, guidance=guidance)
        for g in gen:
            assert dropped not in (g.get("parents") or [])


def test_growth_ops_are_valid_and_uncapped():
    from image_pipeline.shootout.evolve import _op_insert_filter, _op_add_branch
    from image_pipeline.shootout.repair import repair_graph
    rng = random.Random(43)
    g = sample_valid_genome(POOL, CFG, rng)
    graph = g["graph"]
    start = len(graph["nodes"])
    for _ in range(CFG.max_depth + 6):   # grow well past the gen-0 budget
        (_op_insert_filter if rng.random() < 0.5 else _op_add_branch)(
            graph, POOL, CFG, rng)
    fixed = repair_graph(graph, POOL, CFG)
    assert fixed is not None
    assert not validate_graph(fixed, POOL, CFG)
    assert len(fixed["nodes"]) > start, "growth ops never added a node"


def test_complexity_bias_shifts_sizes():
    from image_pipeline.shootout.generator import SamplingBias
    rng = random.Random(47)
    def mean_size(bias):
        return sum(
            len(sample_valid_genome(POOL, CFG, rng, bias=bias)["graph"]["nodes"])
            for _ in range(120)) / 120
    grow = mean_size(SamplingBias(complexity=0.8))
    shrink = mean_size(SamplingBias(complexity=-0.8))
    assert grow > shrink + 0.5, (grow, shrink)


# ── Settings (config overrides) ───────────────────────────────────────


def test_config_endpoint_roundtrip(client, tmp_store):
    from image_pipeline.shootout import config as cfg_mod

    r = client.get("/api/shootout/config")
    assert r.status_code == 200
    fields = {f["name"]: f for f in r.json()["fields"]}
    assert fields["show_n"]["value"] == cfg_mod.DEFAULT_CONFIG.show_n
    assert not fields["show_n"]["overridden"]

    # save: coercion (float→int), clamping, unknown keys dropped
    r = client.post("/api/shootout/config", json={"overrides": {
        "show_n": 4.7,               # → int 5
        "render_pool": 9999,         # clamped to max 64
        "advisor_enabled": False,
        "exclude_methods": ["hack"],  # not tunable — dropped
        "temporal_var_min": 0.001,
    }})
    fields = {f["name"]: f for f in r.json()["fields"]}
    assert fields["show_n"]["value"] == 5 and fields["show_n"]["overridden"]
    assert fields["render_pool"]["value"] == 64
    assert fields["advisor_enabled"]["value"] is False
    assert fields["temporal_var_min"]["value"] == 0.001

    eff = cfg_mod.effective_config()
    assert eff.show_n == 5 and eff.render_pool == 64
    assert eff.advisor_enabled is False
    assert eff.exclude_methods == cfg_mod.DEFAULT_CONFIG.exclude_methods

    # overrides persist across a fresh read
    assert cfg_mod.load_overrides()["show_n"] == 5

    # partial update merges with existing overrides
    client.post("/api/shootout/config", json={"overrides": {"show_n": 3}})
    eff = cfg_mod.effective_config()
    assert eff.show_n == 3 and eff.render_pool == 64

    # reset restores defaults
    r = client.post("/api/shootout/config", json={"reset": True})
    fields = {f["name"]: f for f in r.json()["fields"]}
    assert fields["show_n"]["value"] == cfg_mod.DEFAULT_CONFIG.show_n
    assert not any(f["overridden"] for f in r.json()["fields"])


# ── E2E render smoke (slow — needs ffmpeg) ────────────────────────────


@pytest.mark.slow
@pytest.mark.skipif(not shutil.which("ffmpeg"), reason="ffmpeg required")
def test_e2e_one_tiny_generation(tmp_store, monkeypatch):
    from image_pipeline.shootout import session as sess
    cfg = ShootoutConfig(show_n=2, render_pool=3, frames=12,
                         render_concurrency=2, max_attempts_factor=2)
    s = sess.start_session(cfg=cfg)
    result = sess.run_generation(s["session_id"], cfg,
                                 rng=random.Random(31))
    assert result["generation"] == 0
    assert result["rendered"] >= 3
    for sv in result["survivors"]:
        assert sv["mp4_url"].startswith("/api/sequences/shootout-")
        assert sv["liveness"]["alive"]


# ── Phase 2: utilization audit ──────────────────────────────────────


def test_utilization_counts_methods_and_gaps():
    from image_pipeline.shootout import utilization
    rng = random.Random(3)
    genomes = [sample_valid_genome(POOL, CFG, rng) for _ in range(20)]
    audit = utilization.audit_population(genomes, POOL, CFG)
    # every generated method_id is in the pool and counted
    used = {m for m, d in audit["per_method"].items() if d["count"] > 0}
    assert used
    for g in genomes:
        for n in g["graph"]["nodes"]:
            assert n["method_id"] in audit["per_method"]
    # never_used is disjoint from used
    assert not (set(audit["never_used"]) & used)
    # roles section populated
    assert audit["roles"]["n_terminals"] == len(POOL.terminals)
    assert audit["roles"]["terminals_used_frac"] >= 0.0
    # motif-genomes counted for motif-grammar-generated graphs
    motif_genomes = sum(1 for g in genomes if g["graph"].get("motifs"))
    assert audit["motifs"]["n_motif_genomes"] == motif_genomes
    # summarize returns a non-empty string
    assert utilization.summarize(audit)


def test_utilization_empty_population_is_safe():
    from image_pipeline.shootout import utilization
    audit = utilization.audit_population([], POOL, CFG)
    assert audit["n_genomes"] == 0
    assert audit["n_methods_used"] == 0
    assert audit["n_never_used"] == audit["n_pool_methods"]


def test_utilization_endpoint_fresh(client):
    r = client.get("/api/shootout/utilization")
    assert r.status_code == 200
    body = r.json()
    assert "per_method" in body and "roles" in body
    assert body["n_genomes"] == CFG.render_pool


# ── Phase 3: per-node feedback → advisor ────────────────────────────


def test_node_feedback_to_guidance():
    from image_pipeline.shootout import advisor
    rng = random.Random(53)
    g = sample_valid_genome(POOL, CFG, rng)
    nodes = g["graph"]["nodes"]
    liked = nodes[0]["method_id"]
    disliked = nodes[-1]["method_id"]
    g["node_feedback"] = {
        nodes[0]["id"]: "love this layer",
        nodes[-1]["id"]: "drop this, it's muddy",
    }
    agg = advisor.node_feedback_to_guidance([g], POOL)
    assert liked in agg["prefer"]
    assert disliked in agg["avoid"]
    assert liked not in agg["avoid"]


def test_extract_guidance_uses_node_feedback_without_llm():
    from image_pipeline.shootout import advisor
    rng = random.Random(59)
    g = sample_valid_genome(POOL, CFG, rng)
    nid = g["graph"]["nodes"][0]["id"]
    mid = g["graph"]["nodes"][0]["method_id"]
    g["node_feedback"] = {nid: "hate this node"}
    # no notes, LLM disabled → guidance comes purely from per-node feedback
    g["notes"] = ""
    out = advisor.extract_guidance([g], POOL, CFG, llm=lambda s, u: None)
    assert out is not None
    assert mid in out["avoid_methods"]
    assert out["summary"] == "per-node feedback only"


def test_extract_guidance_merges_node_and_llm():
    from image_pipeline.shootout import advisor
    rng = random.Random(61)
    g = sample_valid_genome(POOL, CFG, rng)
    nid = g["graph"]["nodes"][0]["id"]
    mid = g["graph"]["nodes"][0]["method_id"]
    g["notes"] = "more simulations please"
    g["node_feedback"] = {nid: "love this"}
    llm_reply = json.dumps({
        "prefer_methods": ["999"],  # unknown → sanitized out
        "avoid_methods": [],
        "prefer_categories": ["simulations"],
        "avoid_categories": [],
        "complexity": "increase",
        "protect_genomes": [],
        "drop_genomes": [],
        "summary": "favor simulations",
    })
    out = advisor.extract_guidance([g], POOL, CFG,
                                   llm=lambda s, u: "```json\n" + llm_reply)
    assert mid in out["prefer_methods"]          # node feedback merged in
    assert "simulations" in out["prefer_categories"]
    assert 999 not in out["prefer_methods"]       # LLM unknown id sanitized
    assert "per-node" in out["summary"]


def test_node_feedback_roundtrip(client, tmp_store):
    from image_pipeline.shootout import session as sess
    rng = random.Random(67)
    s = sess.start_session()
    sid = s["session_id"]
    g = sample_valid_genome(POOL, CFG, rng)
    store.save_genome(g)
    s = store.load_session(sid)
    s["generations"].append({
        "gen": 0, "shown": [g["genome_id"]], "pool": [],
        "ratings": {}, "notes": {}, "node_feedback": {}, "rated_logged": [],
    })
    store.save_session(s)
    nid = g["graph"]["nodes"][0]["id"]
    r = client.post("/api/shootout/rate", json={
        "session_id": sid,
        "ratings": {g["genome_id"]: 4},
        "node_feedback": {g["genome_id"]: {nid: "keep this one"}},
    })
    assert r.status_code == 200
    body = r.json()
    assert body["node_feedback"] == 1
    # persisted on the genome + lineage
    saved = store.load_genome(g["genome_id"])
    assert saved["node_feedback"][nid] == "keep this one"
    state = client.get(f"/api/shootout/session/{sid}").json()
    sv = next(x for x in state["survivors"] if x["genome_id"] == g["genome_id"])
    assert sv["node_feedback"][nid] == "keep this one"
    # bad node id is dropped
    r2 = client.post("/api/shootout/rate", json={
        "session_id": sid,
        "node_feedback": {g["genome_id"]: {"ghost_node": "like"}},
    })
    assert r2.status_code == 200
    assert store.load_genome(g["genome_id"]).get("node_feedback", {}) == {} \
        or "ghost_node" not in store.load_genome(g["genome_id"]).get("node_feedback", {})


# ── Explainer enrichment (node names, mini-graph, blurb, deviations) ──


def test_describe_clip_uses_names_and_drivers():
    from image_pipeline.shootout import describe as d
    rng = random.Random(71)
    g = sample_valid_genome(POOL, CFG, rng)
    graph = g["graph"]
    names = d.node_names(graph, POOL)
    assert names  # every node maps to a human name
    for n in graph["nodes"]:
        assert names[n["method_id"]] != n["method_id"] or n["method_id"] in POOL.defs
    cg = d.compact_graph(graph, POOL)
    assert len(cg["nodes"]) == len(graph["nodes"])
    assert all("name" in nd and "is_driver" in nd for nd in cg["nodes"])
    desc = d.describe_clip(graph, POOL)
    assert desc["n_nodes"] == len(graph["nodes"])
    assert isinstance(desc["blurb"], str) and desc["blurb"]
    # driver count matches the pool's scalar_driver classification
    expect_drv = sum(1 for n in graph["nodes"]
                     if n["method_id"] in POOL.scalar_drivers)
    assert desc["n_drivers"] == expect_drv


def test_next_generation_tags_deviations():
    from image_pipeline.shootout import evolve
    rng = random.Random(73)
    # two rated parents
    parents = [sample_valid_genome(POOL, CFG, rng) for _ in range(2)]
    for i, p in enumerate(parents):
        p["rating"] = 5
        p["genome_id"] = f"par{i}"
    kids = evolve.next_generation(parents, 1, POOL, CFG, rng)
    assert kids
    for k in kids:
        assert "deviation" in k
        assert k["deviation"]["kind"] in (
            "mutation", "crossover", "explorer", "random", "protected")
        assert k["deviation"]["text"]
    # with rated parents present, we should see a mix of bred + explore kinds
    kinds = {k["deviation"]["kind"] for k in kids}
    assert "explorer" in kinds          # fresh randoms always present


def test_no_verbatim_survivors():
    """Critique 1: a top-rated genome must NOT roll over unchanged into the
    next generation. Every bred offspring is a new star-weighted variation,
    never the same genome_id carried forward verbatim (elitism removed)."""
    from image_pipeline.shootout import evolve, config as cfg_mod

    # Cross-breed roll off so every child is a single/multi-parent variation.
    cfg = cfg_mod.ShootoutConfig()
    cfg.cross_breed_probability = 0.0
    rng = random.Random(101)
    parents = _rated_generation(rng, [5, 4])[:2]
    for i, p in enumerate(parents):
        p["rating"] = 5
        p["genome_id"] = f"elite-par{i}"
    kids = evolve.next_generation(parents, 1, POOL, cfg, rng)
    assert kids
    parent_ids = {p["genome_id"] for p in parents}
    for k in kids:
        assert k["genome_id"] not in parent_ids, \
            "elite genome rolled over unchanged"


def test_cross_breed_probability_tracks_setting():
    """Critique 2: cross_breed_probability sets the realized fraction of
    offspring that blend TWO parents; raising it yields more crossovers.
    Retrying incompatible pairs keeps the rate from silently decaying."""
    from image_pipeline.shootout import evolve, config as cfg_mod

    def crossover_fraction(prob: float, seed: int, n_gen: int = 24) -> float:
        cfg = cfg_mod.ShootoutConfig()
        cfg.cross_breed_probability = prob
        rng = random.Random(seed)
        parents = _rated_generation(rng, [5, 4])[:2]
        for i, p in enumerate(parents):
            p["rating"] = 5
            p["genome_id"] = f"cb-par{i}-{seed}"
        total = bred = 0
        for _ in range(n_gen):
            for k in evolve.next_generation(parents, 1, POOL, cfg, rng):
                if k["origin"] in ("mutation", "crossover"):
                    bred += 1
                    if k["origin"] == "crossover":
                        total += 1
        return total / bred if bred else 0.0

    lo = crossover_fraction(0.1, 200)
    hi = crossover_fraction(0.9, 200)
    assert hi > lo + 0.3, f"cross-breed rate didn't track setting: {lo:.2f} vs {hi:.2f}"


def test_parent_selection_power_sharpens():
    """Critique 1 (mechanism): higher parent_selection_power makes 5★ parents
    dominate the breeding pool more than 2★ ones (star-weighted, no verbatim
    carry-over)."""
    from image_pipeline.shootout import evolve, config as cfg_mod

    def p5_share(power: float) -> float:
        cfg = cfg_mod.ShootoutConfig()
        cfg.parent_selection_power = power
        rng = random.Random(303)
        parents, weights = evolve.select_parents(_rated_generation(rng, [5, 2]), cfg)
        wi = dict(zip((p["genome_id"] for p in parents), weights))
        p5 = next(p for p in parents if p["rating"] == 5)
        p2 = next(p for p in parents if p["rating"] == 2)
        return wi[p5["genome_id"]] / (wi[p5["genome_id"]] + wi[p2["genome_id"]])

    assert p5_share(3.0) > p5_share(1.0), "power didn't sharpen star weighting"


def test_select_parents_liveness_fallback():
    """Route 8 (2026-07-13): when human ratings are starved (no rating-eligible
    parents), select_parents must fall back to a liveness-fitness parent pool so
    the evolution can still progress instead of collapsing to fresh randoms
    (the gen-0 stagnation seen in the corpus). Static/dead clips must NEVER
    become parents (floor on the liveness fitness)."""
    from image_pipeline.shootout import evolve, config as cfg_mod

    def mk(gid, alive, tvar, mpf, rating=None):
        return {"genome_id": gid, "generation": 1, "rating": rating,
                "liveness": {"alive": alive, "temporal_var": tvar,
                              "motion_pixel_frac": mpf}}

    # No rated parents: a clearly-dynamic alive clip, a dead/static clip, and a
    # barely-alive clip that sits below the fitness floor.
    rated = [
        mk("dyn", True, 0.05, 0.4, rating=None),
        mk("stat", False, 0.0, 0.0, rating=None),
        mk("weak", True, 0.0005, 0.01, rating=None),
    ]
    cfg_on = cfg_mod.ShootoutConfig(); cfg_on.liveness_breed_fallback = True
    cfg_off = cfg_mod.ShootoutConfig(); cfg_off.liveness_breed_fallback = False

    p_off, _ = evolve.select_parents(rated, cfg_off)
    assert p_off == [], "no fallback -> no parents when unrated"

    p_on, w_on = evolve.select_parents(rated, cfg_on)
    ids = {g["genome_id"] for g in p_on}
    assert "dyn" in ids, "dynamic alive clip should breed under fallback"
    assert "stat" not in ids, "dead clip must never breed"
    assert "weak" not in ids, "barely-alive clip below fitness floor must not breed"
    assert any(w > 0 for w in w_on), "dynamic parent must carry real weight"


def test_survivor_view_carries_explainer_fields():
    from image_pipeline.shootout import session as sess, describe as d
    rng = random.Random(79)
    g = sample_valid_genome(POOL, CFG, rng)
    view = sess._survivor_view(g, None, POOL)
    assert view["method_names"]
    assert "graph" in view and view["graph"]["nodes"]
    assert view["blurb"]
    assert view["deviation"] is None  # no evolution yet → no deviation
    # names match the describe module directly
    assert view["method_names"] == d.node_names(g["graph"], POOL)


# ── Phase 4: per-node contribution (ablation) ───────────────────────


def test_contribution_reachability_flags_orphan():
    from image_pipeline.shootout import contribution as contrib
    rng = random.Random(3)
    g = sample_valid_genome(POOL, CFG, rng)
    graph = g["graph"]
    term = contrib._terminal_node_id(graph)
    assert term is not None and term in contrib.reachable_from_terminal(graph)
    # An orphan node with no edges can never reach the output.
    graph["nodes"].append({"id": "orphan1", "params": {},
                           "method_id": POOL.scalar_drivers[0], "render": False})
    reach = contrib.reachable_from_terminal(graph)
    assert "orphan1" not in reach
    assert term in reach  # terminal still reaches itself


def test_contribution_ablate_remove_severs_node():
    from image_pipeline.shootout import contribution as contrib
    rng = random.Random(5)
    g = sample_valid_genome(POOL, CFG, rng)
    graph = g["graph"]
    term = contrib._terminal_node_id(graph)
    victim = next((n["id"] for n in graph["nodes"] if n["id"] != term
                   and any(e["src_node"] == n["id"] or e["dst_node"] == n["id"]
                           for e in graph["edges"])), None)
    if victim is None:
        pytest.skip("sampled graph has no ablatable interior node")
    nodes, edges, mode = contrib.ablate(graph, victim, POOL)
    assert all(n["id"] != victim for n in nodes)   # node dropped
    assert mode in ("bypass", "remove")
    if mode == "remove":
        assert all(e["src_node"] != victim and e["dst_node"] != victim
                   for e in edges)
    else:  # bypass: victim's inputs are gone, nothing still sources from it
        assert all(e["dst_node"] != victim for e in edges)
        assert all(e["src_node"] != victim for e in edges)
    # original graph is untouched
    assert any(n["id"] == victim for n in graph["nodes"])


def test_contribution_delta_and_stack_helpers():
    from image_pipeline.shootout import contribution as contrib
    import numpy as np
    a = np.zeros((4, 6, 6), dtype=np.float32)
    assert contrib._delta(a, a) == 0.0            # identical → no contribution
    b = a.copy()
    b[:] = 1.0
    assert contrib._delta(a, b) == pytest.approx(1.0)  # full-range change
    assert contrib._delta(a, None) is None        # a variant that produced nothing


def test_contribution_structural_only_on_large_graph():
    from image_pipeline.shootout import contribution as contrib
    rng = random.Random(9)
    g = sample_valid_genome(POOL, CFG, rng)
    cfg = ShootoutConfig(contrib_max_nodes=0)     # force the no-render path
    report = contrib.analyze_contribution(g, cfg, POOL)
    assert report["rendered"] is False
    assert report["n_nodes"] == len(g["graph"]["nodes"])
    verdicts = {r["verdict"] for r in report["per_node"]}
    assert verdicts <= {"terminal", "disconnected", "unprobed"}
    assert contrib.summarize(report)


@pytest.mark.slow
@pytest.mark.skipif(not shutil.which("ffmpeg"), reason="ffmpeg required")
def test_contribution_analyze_real_render():
    """End-to-end ablation on a controlled light graph: an animated source
    feeds a filter (render node), plus a driver wired to nothing. Removing
    the source must visibly change the output (contributes); the orphan
    driver must be flagged disconnected. A hand-built graph keeps this fast
    and deterministic — a random genome could sample a heavy sim node whose
    per-probe re-render dominates the run."""
    from image_pipeline.shootout import contribution as contrib
    cfg = ShootoutConfig(width=128, height=96, contrib_frames=6)
    drv = POOL.scalar_drivers[0]
    graph = {"version": 1, "name": "t", "nodes": [
        {"id": "src", "method_id": "312", "params": {}, "render": False},   # Water Caustics
        {"id": "flt", "method_id": "408", "params": {}, "render": True},    # Bloom / Glow
        {"id": "orphan", "method_id": drv, "params": {}, "render": False},  # wired to nothing
    ], "edges": [
        {"src_node": "src", "src_port": "image",
         "dst_node": "flt", "dst_port": "image_in"},
    ]}
    genome = {"genome_id": "gt", "seed": 7, "graph": graph}

    report = contrib.analyze_contribution(genome, cfg, POOL)
    assert report["rendered"] is True
    assert report["terminal"] == "flt"
    verdict = {r["node_id"]: r["verdict"] for r in report["per_node"]}
    assert verdict["flt"] == "terminal"
    assert verdict["orphan"] == "disconnected"   # never reaches the output
    assert verdict["src"] == "contributes"       # removing it changed the frames
    assert [r for r in report["per_node"] if r["node_id"] == "src"][0]["delta"] > 0.05
    # dead_weight is exactly disconnected ∪ silent, and the orphan is in it
    assert set(report["dead_weight"]) == set(report["disconnected"]) | set(report["silent"])
    assert "orphan" in report["dead_weight"]
    assert contrib.summarize(report)


# ── Phase 5: live render telemetry + skip ───────────────────────────


def test_render_monitor_lifecycle_and_skip():
    from image_pipeline.shootout.progress import RenderMonitor
    mon = RenderMonitor()
    mon.begin("gA", total_frames=96, n_nodes=3)
    mon.frame_start("gA", 4)
    mon.node_cooking("gA", "n2", "312", "Water Caustics")
    mon.node_cooking("gA", "n2", "312", "Water Caustics", sim_frame=40)
    snap = mon.snapshot()
    assert "gA" in snap
    s = snap["gA"]
    assert s["frame"] == 4 and s["node_method"] == "312" and s["sim_frame"] == 40

    # skip flips the event + status; a finished genome drops from snapshot
    assert mon.request_skip("gA") is True
    assert mon.is_skipped("gA") and mon.skip_event("gA").is_set()
    assert mon.snapshot()["gA"]["skip_requested"] is True
    mon.finish("gA")
    assert "gA" not in mon.snapshot()
    assert "gA" in mon.snapshot(include_done=True)
    # skip on an unknown genome is inactive, never raises
    assert mon.request_skip("ghost") is False


def test_heartbeat_lines_flag_slow_frames():
    import time
    from image_pipeline.shootout.progress import RenderMonitor, heartbeat_lines
    mon = RenderMonitor()
    mon.begin("gB", total_frames=96, n_nodes=2)
    mon.frame_start("gB", 10)
    mon.node_cooking("gB", "n1", "408", "Bloom / Glow")
    now = time.time()
    # pretend this frame started 30s ago → must be flagged ⚠ SLOW
    mon.snapshot()  # touch
    line = heartbeat_lines(mon.snapshot(), frame_hang_s=15.0, now=now + 30)[0]
    assert "gB" in line and "Bloom / Glow" in line and "408" in line
    assert "frame 11/96" in line and "⚠ SLOW" in line
    # under the threshold → no SLOW flag
    calm = heartbeat_lines(mon.snapshot(), frame_hang_s=15.0, now=now + 1)[0]
    assert "⚠ SLOW" not in calm


def test_skip_and_status_endpoints(client):
    from image_pipeline.shootout import progress
    progress.MONITOR.clear_all()
    # nothing rendering yet
    r = client.get("/api/shootout/render-status")
    assert r.status_code == 200 and r.json()["rendering"] == []
    # a genome that isn't rendering → skip reports inactive
    r = client.post("/api/shootout/skip/gnope")
    assert r.status_code == 200 and r.json()["active"] is False

    # register one on the board and confirm it surfaces + can be skipped
    progress.MONITOR.begin("glive", total_frames=96, n_nodes=2)
    progress.MONITOR.frame_start("glive", 5)
    progress.MONITOR.node_cooking("glive", "n1", "312", "Water Caustics")
    rows = client.get("/api/shootout/render-status").json()["rendering"]
    assert len(rows) == 1 and rows[0]["genome_id"] == "glive"
    assert rows[0]["node_method"] == "312" and rows[0]["frame"] == 5
    assert client.post("/api/shootout/skip/glive").json()["active"] is True
    assert progress.MONITOR.is_skipped("glive")
    progress.MONITOR.clear_all()


@pytest.mark.slow
@pytest.mark.skipif(not shutil.which("ffmpeg"), reason="ffmpeg required")
def test_render_genome_honors_preset_skip():
    """A skip requested before a render starts culls the clip as 'skipped'
    at frame 0 — exercises the executor cancel_event → render-loop path end
    to end without depending on race timing."""
    from image_pipeline.shootout import evaluator, progress
    cfg = ShootoutConfig(width=128, height=96, frames=12)
    graph = {"version": 1, "name": "t", "nodes": [
        {"id": "n1", "method_id": "312", "params": {}, "render": False},
        {"id": "n2", "method_id": "408", "params": {}, "render": True},
    ], "edges": [{"src_node": "n1", "src_port": "image",
                  "dst_node": "n2", "dst_port": "image_in"}]}
    g = {"genome_id": "gpreskip", "seed": 7, "graph": graph}
    progress.MONITOR.request_skip("gpreskip")
    out = evaluator.render_genome(g, cfg)
    progress.MONITOR.clear_all()
    assert out["liveness"]["reason"] == "skipped"
    assert out["liveness"]["alive"] is False


# ── Timeout blame (flag problematic methods for speed/debug work) ──
def _write_timeout_genome(store, gid, reason, timings, nodes, wall=None):
    store.save_genome({
        "genome_id": gid,
        "graph": {"nodes": [{"id": k, "method_id": m} for k, m in nodes]},
        "render": {"node_timings": timings, "wall_s": wall},
        "liveness": {"alive": False, "reason": reason},
    })


def test_timeout_blame_flags_repeat_offenders(tmp_store):
    from image_pipeline.shootout import store, timeout_blame as tb
    # 141 owns ~90% of two timeout clips; 137 owns one. 141 must be
    # flagged, a cheap leaf (8) must not.
    _write_timeout_genome(store, "g-a", "timeout",
                         {"n1": 9000.0, "n2": 1000.0},
                         [("n1", "141"), ("n2", "8")], wall=300)
    _write_timeout_genome(store, "g-b", "timeout",
                         {"n1": 8000.0, "n2": 2000.0},
                         [("n1", "141"), ("n2", "9")], wall=290)
    _write_timeout_genome(store, "g-c", "timeout",
                         {"n1": 7000.0, "n2": 3000.0},
                         [("n1", "137"), ("n2", "10")], wall=280)
    # 137 needs a 2nd timeout appearance to clear the repeat threshold
    _write_timeout_genome(store, "g-h", "timeout",
                         {"n1": 6000.0, "n2": 4000.0},
                         [("n1", "137"), ("n2", "11")], wall=260)
    # a driver leaf present but ~0% compute -> not flagged
    _write_timeout_genome(store, "g-d", "timeout",
                         {"n1": 9500.0, "nL": 10.0},
                         [("n1", "141"), ("nL", "__lfo__")], wall=300)
    # over-budget (pre-render gate, no timings) counts in headline only
    _write_timeout_genome(store, "g-e", "over-budget", {}, [("n1", "141")])
    # alive + non-timeout dead -> excluded
    _write_timeout_genome(store, "g-f", "alive",
                         {"n1": 50.0}, [("n1", "8")], wall=2)
    _write_timeout_genome(store, "g-g", "static", {"n1": 40.0},
                         [("n1", "8")])

    rep = tb.report()
    assert rep["n_timeout"] == 5
    assert rep["n_over_budget"] == 1
    assert rep["n_timed"] == 5
    pids = {m["method_id"] for m in rep["problematic"]}
    assert pids == {"141", "137"}, pids
    assert "__lfo__" not in pids, "driver leaf leaked into problematic"
    # worst clips ordered by wall desc (tie at 300 is order-robust)
    worst_ids = [w["genome_id"] for w in rep["worst_clips"]]
    assert rep["worst_clips"][0]["wall_s"] == 300
    assert "g-a" in worst_ids and "g-d" in worst_ids
    # per-clip attribution: 141 owns 90% of g-a
    b = tb.blame_genome(store.load_genome("g-a"))
    assert b["top_nodes"][0]["method_id"] == "141"
    assert b["top_nodes"][0]["pct"] == 90.0


def test_timeout_blame_endpoint(client, tmp_store):
    from image_pipeline.shootout import store, timeout_blame as tb
    _write_timeout_genome(store, "g-x", "timeout",
                         {"n1": 9000.0, "n2": 1000.0},
                         [("n1", "141"), ("n2", "8")], wall=300)
    _write_timeout_genome(store, "g-y", "timeout",
                         {"n1": 7000.0, "n2": 3000.0},
                         [("n1", "141"), ("n2", "10")], wall=280)
    r = client.get("/api/shootout/timeout-blame")
    assert r.status_code == 200
    d = r.json()
    assert d["n_timeout"] == 2 and d["n_timed"] == 2
    pids = {m["method_id"] for m in d["problematic"]}
    assert pids == {"141"}


# ── node 496 (Local Laplacian) headless feature test ────────────────────────
# Edge-aware tone/detail (Paris et al. 2011). Subtle filters read a small
# mean-Δ, so we assert on CHANGED-PIXEL-FRACTION (the same metric the shootout
# liveness accumulator uses) — mean-Δ is a known false-negative here.

def _render_496(params: dict) -> np.ndarray:
    import image_pipeline.methods  # noqa: F401  (register nodes)
    from image_pipeline.methods.filters.local_laplacian import (
        method_local_laplacian,
    )
    from pathlib import Path
    out = Path("/tmp/_t_llf_496")
    out.mkdir(parents=True, exist_ok=True)
    for p in out.glob("*.png"):
        p.unlink()
    return np.asarray(method_local_laplacian(out_dir=out, seed=42, params=params))


def _changed_frac(a: np.ndarray, b: np.ndarray, thr: float = 0.05) -> float:
    diffs = np.abs(a.astype(np.float64) - b.astype(np.float64))
    return float((diffs > thr).mean())


def test_local_laplacian_registered_and_nonblack():
    import image_pipeline.methods  # noqa: F401
    from image_pipeline.core.graph import get_all_node_defs
    defs = get_all_node_defs()
    assert "496" in defs
    out = _render_496({"anim_mode": "none", "source": "noise",
                       "detail": 1.0, "tone": 0.0})
    assert out.std() > 0.02  # non-black


def test_local_laplacian_none_mode_static():
    a = _render_496({"anim_mode": "none", "source": "noise",
                     "detail": 1.0, "tone": 0.0, "time": 0.0})
    b = _render_496({"anim_mode": "none", "source": "noise",
                     "detail": 1.0, "tone": 0.0, "time": 3.14})
    assert np.mean(np.abs(a.astype(np.float64) - b.astype(np.float64))) < 0.01


def test_local_laplacian_animation_moves_pixels():
    for mode in ("detail_breathe", "tone_sweep"):
        m0 = _render_496({"anim_mode": mode, "source": "noise", "time": 0.0})
        m1 = _render_496({"anim_mode": mode, "source": "noise", "time": 1.57})
        assert _changed_frac(m0, m1) > 0.10, f"{mode} not animating"


def test_local_laplacian_params_live():
    base = _render_496({"anim_mode": "none", "source": "noise",
                        "detail": 1.0, "tone": 0.0})
    detail3 = _render_496({"anim_mode": "none", "source": "noise",
                           "detail": 3.0, "tone": 0.0})
    tone1 = _render_496({"anim_mode": "none", "source": "noise",
                         "detail": 1.0, "tone": 1.0})
    assert _changed_frac(base, detail3) > 0.10


# ── Driver → target pixel-reach integration (Route 8, item 1) ──────────
# The dominant dead-genome failure mode in the shootout is a wired CHOP
# driver (LFO / Counter / Noise1D / Ramp / …) that does NOT actually vary the
# target node's param across frames, so the clip stays flat/static and the
# liveness gate culls it (~63% of 628 genomes historically are dead, and the
# dead-list is dominated by __lfo__ (1041), __counter__ (297), __noise1d__
# (159), __ramp__ (132)). This headless integration test renders a real
# LFO→Gradient(cx) graph frame-by-frame through the GraphExecutor and asserts
# the driver's modulation REACHES THE PIXELS (large temporal_var + changed
# pixel fraction), while the IDENTICAL graph with the driver disconnected
# stays static. It guards the entire driver→SCALAR→param injection path
# (graph.py _inject_typed + the CHOP nodes' global_frame derivation) from
# silently regressing back into the dead-clip trap.

def _render_driver_graph(driver_wired: bool, seed: int = 7,
                         n_frames: int = 24, total_frames: int = 24):
    """Render an LFO→Gradient(cx) graph for n_frames and return the stack."""
    import tempfile
    from pathlib import Path
    from image_pipeline.core.graph import (
        GraphExecutor, GraphNode, GraphEdge,
    )
    import numpy as np

    nodes = [
        GraphNode(id="drv", method_id="__lfo__",
                  params={"waveform": "sine", "min": 0.0, "max": 1.0,
                          "rate": 1.0, "phase": 0.0, "bipolar": False}),
        GraphNode(id="g", method_id="11",
                  params={"gradient_type": "radial", "style": "solid",
                          "cx": 0.5, "cy": 0.5, "direction": 0.0,
                          "anim_mode": "none", "anim_speed": 1.0,
                          "color1": "0.05,0.05,0.2",
                          "color2": "0.95,0.3,0.1"}),
    ]
    edges = []
    if driver_wired:
        # LFO.value (SCALAR) → Gradient.cx (SCALAR port). cx sweeps the
        # gradient center across the full canvas, a large unambiguous Δ.
        edges.append(GraphEdge(src_node="drv", src_port="value",
                               dst_node="g", dst_port="cx"))

    with tempfile.TemporaryDirectory() as tmp:
        ex = GraphExecutor(out_dir=Path(tmp), fps=24, in_memory=True)
        stack = []
        for f in range(n_frames):
            flat, terminal, errors = ex.execute(
                nodes=[n.__dict__ for n in nodes],
                edges=[e.__dict__ for e in edges],
                seed=seed, frame=f, frames=total_frames,
            )
            assert not errors, f"node errors: {errors}"
            assert terminal == "g", f"expected terminal g, got {terminal}"
            img = flat["g"]["image"]
            assert isinstance(img, np.ndarray) and img.ndim == 3, "no image out"
            stack.append(img.astype(np.float32))
    return stack


def test_driver_wired_reaches_pixels():
    """An LFO driving Gradient.cx must produce a visibly animated clip.

    NOTE: do NOT compare frame 0 vs the last frame — with rate=1.0 over 24
    frames the LFO completes an integer number of sine cycles, so frame 0 and
    the last frame land on the SAME phase (the classic sin-phase degeneracy
    false-negative from the 8-step audit, Step 7). Compare frame 0 against a
    quarter-cycle frame instead, which sits at the waveform's opposite extreme.
    """
    stack = _render_driver_graph(driver_wired=True, seed=7)
    arr = np.stack(stack)  # (N, H, W, 3)
    temporal_var = float(arr.var(axis=0).mean())
    # quarter-cycle apart (frame 6) sits at the LFO's opposite extreme from frame 0
    changed = _changed_frac(stack[0], stack[6])
    # The gradient center sweeps 0→1 across the full canvas, so the canvas
    # changes a lot between the two frames; both globals are large.
    assert temporal_var > 1e-3, temporal_var
    assert changed > 0.30, f"driver did not reach pixels (changed={changed:.3f})"


def test_driver_disconnected_is_static():
    """The SAME graph WITHOUT the driver wire must stay static (proves the
    animation in the wired case is caused by the driver, not by a built-in
    anim_mode or seed drift)."""
    stack = _render_driver_graph(driver_wired=False, seed=7)
    arr = np.stack(stack)
    temporal_var = float(arr.var(axis=0).mean())
    changed = _changed_frac(stack[0], stack[-1])
    # cx is pinned at its default 0.5 and anim_mode='none' → a frozen clip.
    assert temporal_var < 1e-4, temporal_var
    assert changed < 0.01, f"disconnected graph moved ({changed:.3f})"


def test_driver_variation_distinct_from_static_baseline():
    """Sanity: the wired driver's temporal_var must be SIGNIFICANTLY larger
    than the disconnected baseline's, confirming the injection path adds real
    motion and is not just numerical noise."""
    wired = np.stack(_render_driver_graph(driver_wired=True, seed=3))
    static = np.stack(_render_driver_graph(driver_wired=False, seed=3))
    tv_wired = wired.var(axis=0).mean()
    tv_static = static.var(axis=0).mean()
    assert tv_wired > 50 * tv_static, (tv_wired, tv_static)


# ── Cost-cull cap-extension fix (Route 8, 2026-07-16) ─────────────────────
# Heavy sims empirically blow the conservative per-method cost estimate by up
# to ~17x (wall/est p99), so the estimate floor silently rejected slow-but-
# dynamic clips (estimated < floor, slipped the gate, culled as 'timeout').
# The fix: a genome CONTAINING a heavy method with a high empirical P(alive)
# is intrinsically heavy and gets the extended cap even when its estimate
# under-predicts. The check runs BEFORE the estimate-floor fallback, so it is
# monotonic-safe (only ever raises the cap for heavy high-prior graphs).

def _fake_model(per_method, per_method_alive, per_method_p90=None,
                n_samples=200, default_ms=50.0):
    return {
        "n_samples": n_samples,
        "per_method": per_method,
        "per_method_alive": per_method_alive,
        "per_method_p90": per_method_p90 or {k: v for k, v in per_method.items()},
        "default_ms": default_ms,
    }


def _genome_with(method_ids):
    return {
        "genome_id": "g-test",
        "graph": {"nodes": [{"id": f"n{i}", "method_id": m}
                             for i, m in enumerate(method_ids)]},
    }


def test_heavy_method_presence_extends_cap_despite_low_estimate():
    """A genome containing a heavy (>= heavy_method_ms_floor) method with a
    high P(alive) must get the extended cap EVEN IF its conservative estimate
    is far below the est-floor (the real under-prediction case)."""
    from image_pipeline.shootout import cost_model as cm
    cfg = ShootoutConfig()
    cfg.heavy_method_ms_floor = 400.0
    cfg.gate_liveness_floor = 0.33
    cfg.heavy_render_timeout_factor = 2.0
    cfg.render_timeout_s = 300.0
    cfg.heavy_extend_est_floor = 0.5
    # Method '999' is genuinely heavy (600 ms/frame) and likely-dynamic (P=0.9)
    # but the P90 estimate is tiny so the est-floor fallback would NOT catch it.
    model = _fake_model(
        per_method={"999": 600.0, "light": 5.0},
        per_method_alive={"999": 0.9, "light": 0.5},
        per_method_p90={"999": 10.0, "light": 5.0},  # tiny → est far under 150s
    )
    g = _genome_with(["light", "999"])
    eff = cm.effective_render_timeout_s(g, cfg, model)
    # 300 × 2 = 600 clamped to max_render_timeout_s default (450).
    assert eff == 450.0, f"heavy-presence must extend cap, got {eff}"
    # Without the heavy method, the light-only genome keeps the base cap.
    g_light = _genome_with(["light", "light"])
    assert cm.effective_render_timeout_s(g_light, cfg, model) == 300.0


def test_heavy_method_without_prior_gets_extension_death_spiral():
    """A heavy method with NO trusted alive-prior (prior is None) must NOW
    trigger the extension (death-spiral closure, Route 8 2026-07-17). The old
    behaviour suppressed it, which guaranteed the heavy sim would be culled as
    'timeout' every generation without ever earning a verdict."""
    from image_pipeline.shootout import cost_model as cm
    cfg = ShootoutConfig()
    cfg.heavy_method_ms_floor = 400.0
    cfg.gate_liveness_floor = 0.33
    cfg.heavy_render_timeout_factor = 2.0
    cfg.render_timeout_s = 300.0
    model = _fake_model(
        per_method={"999": 600.0},
        per_method_alive={"999": None},  # no trusted prior
    )
    g = _genome_with(["999"])
    # 300 × 2 = 600 clamped to max_render_timeout_s default (450).
    assert cm.effective_render_timeout_s(g, cfg, model) == 450.0


def test_est_floor_fallback_still_extends_heavy_sum():
    """The est-floor fallback still extends a graph whose SUM of media methods
    is estimated heavy even with no single heavy method present."""
    from image_pipeline.shootout import cost_model as cm
    cfg = ShootoutConfig()
    cfg.heavy_method_ms_floor = 400.0
    cfg.render_timeout_s = 300.0
    cfg.heavy_extend_est_floor = 0.5
    cfg.heavy_render_timeout_factor = 2.0
    # Three media methods each ~800 ms/frame → sum 2400 ms/frame × 96 frames
    # calibrates (slope 0.557 + intercept 33.7) to ~162s, above the 150s floor.
    model = _fake_model(
        per_method={"a": 800.0, "b": 800.0, "c": 800.0},
        per_method_alive={"a": 0.2, "b": 0.2, "c": 0.2},
        per_method_p90={"a": 800.0, "b": 800.0, "c": 800.0},
    )
    g = _genome_with(["a", "b", "c"])
    # 300 × 2 = 600 clamped to max_render_timeout_s default (450).
    assert cm.effective_render_timeout_s(g, cfg, model) == 450.0


# ── Dead-param audit: Architecture-B (time-param) fallback ──────────────────
# Regression guard for the audit_dead_params.py blind spot fix (2026-07-18):
# nodes that animate via the injected ``time`` clock (no anim_mode enum) were
# reported "no-anim-mode" and never rendered, hiding genuine dead-time nodes.
# The fallback renders them across the frame stack and returns a real verdict.

def test_audit_detects_time_param_node():
    """A node with a ``time`` param but no anim_mode is now auditable."""
    from image_pipeline.shootout.audit_dead_params import _has_time_param
    assert _has_time_param({"params": {"time": {"min": 0.0, "max": 6.28}}})
    assert _has_time_param({"params": {"phase": {"min": 0.0, "max": 6.28}}})
    assert not _has_time_param({"params": {"scale": {"min": 0.0, "max": 1.0}}})
    assert not _has_time_param({"params": {}})
    assert not _has_time_param({})


def test_audit_time_fallback_yields_real_verdict():
    """A pure-filter node (no time param, no anim_mode) stays 'no-anim-mode',
    but a time-param node gets a rendered changed/tvar verdict instead."""
    from image_pipeline.shootout.audit_dead_params import audit_node
    # Pure filter def: no anim_mode, no time param → unauditable as before.
    filt = {"name": "Fake Filter", "params": {"strength": {"min": 0.0, "max": 1.0}}}
    r = audit_node("__nonexistent_filter__", filt)
    assert r["status"] == "no-anim-mode"
    # A time-param def routes into the <time> fallback; the render will error on
    # a nonexistent method id, which the fallback classifies as render-error
    # (NOT the old silent no-anim-mode). Either way it is no longer skipped.
    tdef = {"name": "Fake Time Node", "params": {"time": {"min": 0.0, "max": 6.28}}}
    r2 = audit_node("__nonexistent_time__", tdef)
    assert r2["status"] != "no-anim-mode"
    assert r2["modes"] == ["<time>"]


def test_audit_parses_description_derived_modes():
    """anim_mode choices that live only in the description (paren slash-list)
    must be discovered so the node is audited via its REAL modes, not the
    ``time`` fallback with anim_mode='none' (which freezes the clock → false
    DEAD-PARAM). Regression for nodes 406 Harmonograph / 402 Kaleidoscopic IFS.
    """
    from image_pipeline.shootout.audit_dead_params import _non_none_modes
    # choices only in the description string, no explicit choices list
    desc_only = {"params": {"anim_mode": {
        "description": "animation mode (none/phase/draw/rotate)", "default": "none"}}}
    assert _non_none_modes(desc_only) == ["phase", "draw", "rotate"]
    # explicit choices still take precedence
    explicit = {"params": {"anim_mode": {
        "description": "ignored (a/b)", "choices": ["none", "rotate", "spin"],
        "default": "none"}}}
    assert _non_none_modes(explicit) == ["rotate", "spin"]
    # a description with no paren slash-list yields nothing
    plain = {"params": {"anim_mode": {"description": "the animation", "default": "none"}}}
    assert _non_none_modes(plain) == []


def test_harmonograph_and_kifs_animate_and_declare_modes():
    """Nodes 406/402 must declare explicit anim_mode choices AND actually
    animate in a non-none mode (audit must classify them 'alive')."""
    import image_pipeline.methods  # noqa: F401
    from image_pipeline.core.graph import get_all_node_defs
    from image_pipeline.shootout.audit_dead_params import audit_node
    defs = get_all_node_defs()
    for mid in ("406", "402"):
        defn = defs.get(mid)
        assert defn is not None, f"node {mid} not registered"
        choices = (defn.get("params") or {}).get("anim_mode", {}).get("choices")
        assert choices and "none" in choices and len(choices) >= 3, \
            f"node {mid} missing explicit anim_mode choices: {choices}"
        r = audit_node(mid, defn)
        assert r["status"] == "alive", f"node {mid} audited {r['status']} (should be alive): {r}"


def test_audit_recovers_alias_only_modes_from_source():
    """Fallback 3 (2026-07-18): modes declared ONLY via an aliased local
    (``mode = params.get("anim_mode", "none")`` then branched on) appear in
    NEITHER an explicit ``choices`` list NOR a paren slash-list description.
    An AST scan of the method source must still recover them so the node is
    audited via its real modes rather than the frozen ``none`` time-path.

    We assert the real registry: find a registered node whose anim_mode modes
    are recoverable from source, and prove _derive_modes_from_source returns a
    non-empty list for it. Node 402 (Kaleidoscopic IFS) reads anim_mode via an
    aliased local, so its source-derived modes must be non-empty.
    """
    import image_pipeline.methods  # noqa: F401
    from image_pipeline.shootout.audit_dead_params import (
        _derive_modes_from_source,
        _non_none_modes,
    )
    # 402 aliases anim_mode to a local and branches on it — source recovery
    # must yield its real (non-none) modes.
    modes = _derive_modes_from_source("402")
    assert modes and all(m.lower() != "none" for m in modes), \
        f"source-derived modes for 402 should be non-empty non-none: {modes}"
    # The layered _non_none_modes with a def that has an anim_mode param but no
    # choices/description must fall through to the source scan when mid is given.
    stub = {"params": {"anim_mode": {"description": "the animation", "default": "none"}}}
    assert _non_none_modes(stub) == []            # no mid → no source scan
    assert _non_none_modes(stub, "402") == modes  # mid → source scan recovers


def test_clahe_clip_sweep_reaches_pixels():
    """Node 436 CLAHE `clip_sweep` was a genuine dead-param (Route 8): the
    clip_limit sweep alone is nearly invisible on smooth/low-contrast sources
    (CLAHE has almost nothing to expand), and its 0.4 sweep frequency barely
    moved inside the shootout / audit phase window, so the clip rendered static
    and the liveness gate culled it. The fix co-breathes `strength` and sweeps
    `tile_size` at a faster (1.6) frequency so the animation reaches the pixels
    on the DEFAULT `procedural` source. Guard it via the same audit_node path
    the shootout dead-param frontier uses so a future edit cannot silently
    regress it back to a culled static clip.
    """
    import image_pipeline.methods  # noqa: F401
    from image_pipeline.core.graph import get_all_node_defs
    from image_pipeline.shootout.audit_dead_params import audit_node
    defs = get_all_node_defs()
    defn = defs.get("436")
    assert defn is not None, "node 436 (CLAHE) not registered"
    r = audit_node("436", defn)
    assert r["status"] == "alive", \
        f"CLAHE clip_sweep audited {r['status']} (should be alive): {r}"


def test_fast_bilateral_solver_sweeps_reach_pixels():
    """Node 924 Fast Bilateral Solver was a genuine dead-param (Route 8): on the
    default `noise` source the noise field is static per seed and the sweep
    params (sigma_s / sigma_r / spatial_iterations) barely move the smoothed
    output frame to frame, so every sweep mode rendered a near-static stack
    (changed_frac ~0.08 < 0.10 floor) and the liveness gate culled it. The fix
    (a) drifts the source band with the time clock and (b) co-breathes `amount`
    (source↔full-smoothing blend) at a faster 1.6 frequency in every sweep mode
    so the output visibly restructures inside the audit phase window. Guard it
    via the same audit_node path so a future edit cannot silently regress it.
    """
    import image_pipeline.methods  # noqa: F401
    from image_pipeline.core.graph import get_all_node_defs
    from image_pipeline.shootout.audit_dead_params import audit_node
    defs = get_all_node_defs()
    defn = defs.get("924")
    assert defn is not None, "node 924 (Fast Bilateral Solver) not registered"
    r = audit_node("924", defn)
    assert r["status"] == "alive", \
        f"FBS sweeps audited {r['status']} (should be alive): {r}"


def test_void_and_cluster_renders_non_square_canvas():
    """Node 533 Void-and-Cluster Dither crashed on NON-SQUARE canvases
    (Route 8 dead-param frontier, 2026-07-18).

    Its blue-noise spectrum analysis built a radial mask from a SQUARE n×n
    grid (n = thr.shape[0]) while the FFT magnitude is (H, W) — whenever W != H
    (the executor's default 768×512) indexing ``mag[lp_mask]`` raised
    "boolean index did not match indexed array" and the node fell back to a
    static gray image, which the liveness gate culled as dead. The fix builds
    the mask from the real (H, W) threshold-map shape.

    The audit renders on the default non-square canvas, so a clean `alive`
    verdict proves the crash is gone (a `render-error` status would mean the
    W!=H exception still fires). Added as a permanent guard so a future edit
    cannot silently re-introduce the square-canvas assumption.
    """
    import image_pipeline.methods  # noqa: F401
    from image_pipeline.core.graph import get_all_node_defs
    from image_pipeline.shootout.audit_dead_params import audit_node
    defs = get_all_node_defs()
    defn = defs.get("533")
    assert defn is not None, "node 533 (Void-and-Cluster Dither) not registered"
    r = audit_node("533", defn)
    assert r["status"] != "render-error", \
        f"Void-and-Cluster still crashes on non-square canvas: {r}"
    assert r["status"] == "alive", \
        f"Void-and-Cluster audited {r['status']} (should be alive): {r}"


@pytest.mark.parametrize("mid", ["436", "924", "493", "340", "462", "74", "349", "975", "533"])
def test_dead_param_frontier_filters_alive(mid):
    """Route 8 dead-param frontier regression for the FILTER category.

    A curated, diverse sample of filter nodes that the headless liveness audit
    classified ALIVE must never silently regress to a dead-param (a non-`none`
    anim_mode that renders a static frame stack and gets culled by the liveness
    gate). This is the exact bug class that hit 436 CLAHE `clip_sweep` and 924
    Fast Bilateral Solver `spatial_sweep` — both repaired and now included here
    as permanent guards.

    The sample spans the animation primitives so a regression in ANY of them is
    caught, not just the two previously-fixed nodes:
      * 436 clip_sweep   (clip-limit + strength + tile co-breathe)
      * 924 spatial_sweep(Fast Bilateral Solver — source drift + amount blend)
      * 493 exposure_sweep / 349 lambda_sweep (scalar sweeps)
      * 340 flow          (translation/drift sweep)
      * 462 pulse         (breathing sweep)
      * 74  rotation_cycle(spin sweep)
      * 975 morph         (structural morph sweep)

    We guard on the precise bug class ("DEAD-PARAM (suspect)") rather than the
    stricter "alive": a node that merely becomes 'weak' (changed_frac below the
    0.10 floor but temporal_var still above the 1e-3 liveness floor) still
    passes the shootout gate and is NOT a dead-param regression, so it must not
    spuriously fail this test.
    """
    import image_pipeline.methods  # noqa: F401
    from image_pipeline.core.graph import get_all_node_defs
    from image_pipeline.shootout.audit_dead_params import audit_node
    defs = get_all_node_defs()
    defn = defs.get(mid)
    assert defn is not None, f"node {mid} not registered"
    r = audit_node(mid, defn)
    assert "DEAD-PARAM" not in r["status"], \
        f"node {mid} regressed to dead-param ({r['status']}): {r}"


@pytest.mark.parametrize("mid", ["49", "51", "52"])
def test_dead_param_frontier_animation_mode_key(mid):
    """Route 8 dead-param frontier — `animation_mode` key blind spot.

    Nodes 49 (Buddhabrot), 51 (Burning Ship), 52 (Newton) declare their
    animation-mode enum under the key ``animation_mode`` rather than the
    pipeline-canonical ``anim_mode``. The dead-param audit must inject the
    SAME key the node reads; otherwise it always sees ``none`` and a
    genuinely-animating node is mis-classified (historically as
    ``render-error`` / a false dead-param suspect), corrupting the frontier
    report.

    This test pins the fix: these nodes must be classified ALIVE (their
    ``animation_mode`` reaches pixels) — never a DEAD-PARAM suspect. If the
    harness ever reverts to injecting ``anim_mode`` only, they would render
    static and this guard fails loudly.
    """
    import image_pipeline.methods  # noqa: F401
    from image_pipeline.core.graph import get_all_node_defs
    from image_pipeline.shootout.audit_dead_params import audit_node
    defs = get_all_node_defs()
    defn = defs.get(mid)
    assert defn is not None, f"node {mid} not registered"
    # Sanity: the node really does declare the non-canonical key.
    assert "animation_mode" in (defn.get("params") or {}), \
        f"node {mid} unexpectedly stopped declaring animation_mode"
    r = audit_node(mid, defn)
    assert r["status"] == "alive", \
        f"node {mid} ({defn.get('name')}) animation_mode key not honoured " \
        f"by audit -> {r['status']}: {r}"


# ── Route 8 / sub-problem #6: active-learning rating loop ───────────────
def _fake_genome(gid, alive=True, rated=None, tvar=0.5, n_nodes=3, mp4="/x.mp4"):
    """Minimal genome envelope with exactly the fields the suggester reads."""
    g = {
        "genome_id": gid,
        "graph": {"nodes": [{"id": f"n{i}"} for i in range(n_nodes)]},
        "liveness": {"alive": alive, "temporal_var": tvar},
        "render": {"mp4": mp4},
    }
    if rated is not None:
        g["rating"] = rated
    return g


def test_suggest_for_rating_prefers_unrated_alive_and_carries_mp4():
    """The suggester must surface UNRATED + ALIVE clips, carry mp4_url, and
    skip dead / already-rated genomes. This is the data the UI rating queue
    consumes — if it ever drops mp4_url or leaks rated clips, the queue breaks.
    """
    from image_pipeline.shootout.rating_suggest import suggest_for_rating

    genomes = [
        _fake_genome("g-rated", rated=4),        # already rated -> excluded
        _fake_genome("g-dead", alive=False),     # dead -> excluded
        _fake_genome("g-a", rated=None, tvar=0.9),
        _fake_genome("g-b", rated=None, tvar=0.3),
        _fake_genome("g-c", rated=None, tvar=0.6),
    ]
    out = suggest_for_rating(k=5, genomes=genomes)
    ids = {s["genome_id"] for s in out}
    assert "g-rated" not in ids, "rated clip leaked into suggestions"
    assert "g-dead" not in ids, "dead clip leaked into suggestions"
    assert ids == {"g-a", "g-b", "g-c"}, ids
    for s in out:
        assert s["mp4_url"], f"suggestion {s['genome_id']} missing mp4_url"
        assert isinstance(s["reason"], str) and s["reason"]


def test_rate_external_writes_genome_and_appends_once(tmp_path, monkeypatch):
    """Session-independent rating must persist genome['rating'] AND append to
    the training corpus exactly once (re-rating must not double-count). This
    mirrors server.rate_external without a live server so it runs hermetic."""
    from image_pipeline.shootout import store as shoot_store
    from image_pipeline.shootout import features as shoot_features
    from image_pipeline.shootout import generator as shoot_generator
    from image_pipeline.shootout.config import DEFAULT_CONFIG

    # Point the store at an isolated tmp dir.
    monkeypatch.setattr(shoot_store, "DATA_DIR", tmp_path)
    monkeypatch.setattr(shoot_store, "GENOMES_DIR", tmp_path / "genomes")
    monkeypatch.setattr(shoot_store, "RATINGS_PATH", tmp_path / "ratings.jsonl")
    monkeypatch.setattr(shoot_store, "SESSIONS_DIR", tmp_path / "sessions")
    monkeypatch.setattr(shoot_store, "_ratings_lock", __import__("threading").Lock())
    shoot_store._ensure_dirs()

    gid = "g-a"
    genome = _fake_genome(gid, rated=None)
    shoot_store.save_genome(genome)
    cfg = DEFAULT_CONFIG
    pool = shoot_generator.build_gene_pool(cfg)

    def persist(ratings: dict):
        for g2, stars in ratings.items():
            gm = shoot_store.load_genome(g2)
            assert gm is not None
            gm["rating"] = max(1, min(5, int(stars)))
            shoot_store.save_genome(gm)
            existing = {r["genome_id"] for r in shoot_store.load_ratings() if r}
            if g2 not in existing:
                shoot_store.append_rating(
                    g2, "external", stars,
                    shoot_features.genome_features(gm, pool, cfg))

    before = len(shoot_store.load_ratings())
    persist({gid: 4})
    mid = len(shoot_store.load_ratings())
    persist({gid: 5})  # re-rate
    after = len(shoot_store.load_ratings())

    assert mid == before + 1, (before, mid)
    assert after == mid, "append-once guard broke on re-rate"
    assert shoot_store.load_genome(gid)["rating"] == 5



# ── mine_candidates: schema-correct feedback-loop mining (Route 8) ──
def test_mine_candidates_reads_real_schema():
    """mine_candidates must extract genome_id / graph.motifs / driver count
    from the REAL persisted schema — the old inline probe used non-existent
    keys (id / top-level motifs / n_drivers) and recorded None/[] every run."""
    from image_pipeline.shootout import describe

    pool = build_gene_pool(CFG)
    driver_id = next(iter(pool.scalar_drivers))

    def mk(gid, rating, alive, wall, motifs, ndrivers, gen=1):
        nodes = [{"id": f"n{i}", "method_id": driver_id} for i in range(ndrivers)]
        nodes.append({"id": "src", "method_id": "02"})
        return {
            "genome_id": gid,
            "generation": gen,
            "origin": "explorer",
            "rating": rating,
            "deviation": {"kind": "mutation"},
            "graph": {"nodes": nodes, "edges": [], "motifs": motifs},
            "render": {"wall_s": wall},
            "liveness": {"alive": alive},
        }

    genomes = [
        mk("g-top", 5, True, 12.0, ["sim_backbone", "post_fx"], 2),
        mk("g-mid", 3, True, 8.0, ["pattern_blend"], 1),
        mk("g-slow", 4, True, 120.0, ["post_fx"], 1),
        mk("g-dead", None, False, 200.0, ["sim_backbone"], 0),
    ]

    rep = describe.mine_candidates(genomes=genomes, pool=pool, cheap_wall_s=30.0)

    assert rep["n_genomes"] == 4
    assert rep["n_rated"] == 3
    assert rep["n_alive"] == 3
    # cheap-alive = alive AND wall < 30 → g-top, g-mid (not g-slow at 120s)
    assert rep["cheap_alive"] == 2

    top = rep["top_rated"]
    # highest rating first, real genome_id (not None), real motifs, real drivers
    assert top[0]["genome_id"] == "g-top"
    assert top[0]["rating"] == 5
    assert top[0]["motifs"] == ["sim_backbone", "post_fx"]
    assert top[0]["n_drivers"] == 2
    assert top[0]["deviation_kind"] == "mutation"
    # no None genome_ids leaked (the old-probe failure mode)
    assert all(c["genome_id"] is not None for c in top)

    # surviving-motif coverage only counts ALIVE genomes
    assert rep["motif_coverage"].get("sim_backbone") == 1  # g-top only (g-dead culled)
    assert rep["motif_coverage"].get("post_fx") == 2       # g-top + g-slow


# ── Active-learning rating suggester (Route 8 #6, rating-signal poverty) ──
def test_shootout_suggest_ratings_warm_contract(monkeypatch):
    """suggest_for_rating surfaces diverse, novel, unrated-ALIVE genomes.

    Locks in the cold-start active-learning contract behind the single
    remaining shootout lever (the starved rating corpus, 19/649). Genome
    features are faked to a deterministic cloud so the *selection math*
    (core-set farthest-point greedy + novelty/uncertainty + fitness bias) is
    what is under test, not the feature extractor.

    Contract:
      1. only alive + unrated genomes are ever suggested
      2. the suggestion set is duplicate-free
      3. exactly k are returned (capped at candidate count)
      4. novelty engages once rated genomes exist (active-learning is live)
      5. the selected set is spatially diverse (not collapsed to one point)
    """
    import image_pipeline.shootout.rating_suggest as rs

    def _fake_features(g, pool, cfg):
        h = 0
        for c in g["genome_id"]:
            h = (h * 31 + ord(c)) & 0xFFFFFF
        return {"f0": ((h >> 0) & 0xFF) / 255.0,
                "f1": ((h >> 8) & 0xFF) / 255.0,
                "f2": ((h >> 16) & 0xFF) / 255.0}

    monkeypatch.setattr(rs, "genome_features", _fake_features)

    def _mk(gid, alive, rating=None, tv=0.1):
        return {"genome_id": gid,
                "liveness": {"alive": alive, "temporal_var": tv},
                "rating": rating,
                "graph": {"nodes": [{"method_id": "82"}]}}

    genomes = [
        # 3 rated-alive genomes -> engage the novelty centroid
        _mk("g-1001", True, 4, tv=0.20),
        _mk("g-1002", True, 5, tv=0.25),
        _mk("g-1003", True, 3, tv=0.15),
        # 8 alive + unrated candidates (distinct feature points)
        *[_mk(f"g-000{i}", True, None, tv=0.05 + 0.02 * i) for i in range(1, 9)],
        # 2 dead genomes -> MUST be excluded
        _mk("g-dead1", False), _mk("g-dead2", False),
    ]

    out = rs.suggest_for_rating(k=3, cfg=CFG, pool=POOL, genomes=genomes)
    by_id = {g["genome_id"]: g for g in genomes}
    ids = [s["genome_id"] for s in out]

    # (1) only alive + unrated
    for gid in ids:
        g = by_id[gid]
        assert g["liveness"]["alive"] and g["rating"] is None, \
            f"suggested dead/rated genome {gid}"
    # (2) no duplicates
    assert len(ids) == len(set(ids))
    # (3) exactly k returned (fewer candidates would cap, here 8 >= 3)
    assert len(out) == 3
    # (4) novelty engages (rated corpus present -> some novelty > 0)
    assert any(s["novelty"] > 0.0 for s in out), \
        "novelty collapsed to 0 despite a rated corpus"
    # (5) diversity: selected points are not all identical
    feats = np.array([[_fake_features({"genome_id": i}, POOL, CFG)[k]
                      for k in ("f0", "f1", "f2")] for i in ids])
    if len(feats) > 1:
        d = np.linalg.norm(feats[:, None, :] - feats[None, :, :], axis=2)
        np.fill_diagonal(d, np.inf)
        assert d.min() > 1e-3, "selected set collapsed to identical features (no diversity)"


def test_shootout_suggest_ratings_cold_start(monkeypatch):
    """Cold start (no rated genomes): diversity alone must still drive a
    valid, duplicate-free, alive+unrated suggestion set."""
    import image_pipeline.shootout.rating_suggest as rs

    def _fake_features(g, pool, cfg):
        h = 0
        for c in g["genome_id"]:
            h = (h * 31 + ord(c)) & 0xFFFFFF
        return {"f0": ((h >> 0) & 0xFF) / 255.0,
                "f1": ((h >> 8) & 0xFF) / 255.0,
                "f2": ((h >> 16) & 0xFF) / 255.0}

    monkeypatch.setattr(rs, "genome_features", _fake_features)

    def _mk(gid, alive, rating=None):
        return {"genome_id": gid,
                "liveness": {"alive": alive, "temporal_var": 0.1},
                "rating": rating,
                "graph": {"nodes": [{"method_id": "82"}]}}

    genomes = [
        *[_mk(f"g-000{i}", True, None) for i in range(1, 7)],  # 6 candidates
        _mk("g-dead1", False),
    ]

    out = rs.suggest_for_rating(k=4, cfg=CFG, pool=POOL, genomes=genomes)
    by_id = {g["genome_id"]: g for g in genomes}
    ids = [s["genome_id"] for s in out]

    assert len(out) == 4                       # capped at 6 candidates -> 4
    assert len(ids) == len(set(ids))           # no duplicates
    for gid in ids:
        g = by_id[gid]
        assert g["liveness"]["alive"] and g["rating"] is None
    # cold start -> novelty all zero, but diversity must still hold
    feats = np.array([[_fake_features({"genome_id": i}, POOL, CFG)[k]
                      for k in ("f0", "f1", "f2")] for i in ids])
    d = np.linalg.norm(feats[:, None, :] - feats[None, :, :], axis=2)
    np.fill_diagonal(d, np.inf)
    assert d.min() > 1e-3, "cold-start selection collapsed to identical features"
