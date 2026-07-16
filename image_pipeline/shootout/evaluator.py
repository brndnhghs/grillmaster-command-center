"""Render a genome to a short mp4 and reject dead clips (plan §7).

Rendering happens in-process (same internals as /api/graph/render-sequence:
GraphExecutor per frame), but pipes frames straight into ffmpeg via
frames_to_mp4 instead of writing per-frame PNGs — the mp4 is the durable
artifact (determinism-epoch policy, plan §12). Liveness stats are computed
on a spatially-downsampled grayscale copy of every frame, so memory stays
flat regardless of clip size.
"""
from __future__ import annotations

import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Callable

import numpy as np

try:
    import cv2
    _HAS_CV2 = True
except Exception:  # pragma: no cover - optical-flow rescue degrades gracefully
    cv2 = None
    _HAS_CV2 = False

from image_pipeline.core.animation import frames_to_mp4
from image_pipeline.core.graph import GraphExecutor

from . import progress
from .config import ShootoutConfig, DEFAULT_CONFIG

OUTPUT_ROOT = Path(__file__).resolve().parent.parent / "output"
SEQUENCES_DIR = OUTPUT_ROOT / "sequences"

# Liveness-verdict schema version. Bumped whenever the gate gains a new rescue
# signal (spectral, optical-flow, ...) so the cost model can tell a MODERN
# verdict from a legacy one produced by an evaluator that would have over-culled
# real animation. Legacy genomes (no stamp, or an older stamp) are still trusted
# for timing data but are EXCLUDED from the liveness prior so their stale
# static/flat verdicts do not drag P(alive) down and falsely suppress the
# heavy-cap extension / advisor dead-method feedback (Route 8, 2026-07-16).
EVALUATOR_VERSION = "2026-07-16b"


def seq_name_for(genome_id: str) -> str:
    return f"shootout-{genome_id}"


# ── Liveness stats ────────────────────────────────────────────────────


class LivenessAccumulator:
    """Streaming stats over downsampled grayscale frames."""

    def __init__(self, cfg: ShootoutConfig):
        self.cfg = cfg
        self.small: list[np.ndarray] = []
        self.small_f: list[np.ndarray] = []  # higher-res grayscale for optical flow
        # Color-preserving downsampled copy for the COLOR-AWARE liveness
        # rescue (Route 8 sub-problem #3 closure, 2026-07-16). Plain grayscale
        # (small / small_f) collapses per-pixel RGB -> luminance, so a clip whose
        # HUES/CHANNELS cycle while every pixel's luminance stays constant reads
        # as ``static`` and gets culled. The color stack keeps all 3 channels so
        # we can measure per-pixel chroma change that survives the luminance
        # collapse. Same stride as ``small`` to stay memory-flat.
        self.small_c: list[np.ndarray] = []
        self.nan = False
        self.missing = 0
        self.total = 0

    def add(self, arr: np.ndarray | None) -> None:
        self.total += 1
        if arr is None:
            self.missing += 1
            return
        s = self.cfg.stat_stride
        small = np.asarray(arr, dtype=np.float32)[::s, ::s]
        if small.ndim == 3:
            small = small.mean(axis=-1)
        if not np.isfinite(small).all():
            self.nan = True
            small = np.nan_to_num(small)
        self.small.append(small)
        # Color-preserving copy (stride-downsampled, NOT luminance-collapsed).
        small_c = np.asarray(arr, dtype=np.float32)[::s, ::s]
        if small_c.ndim != 3:
            small_c = small_c[..., None]
        if not np.isfinite(small_c).all():
            self.nan = True
            small_c = np.nan_to_num(small_c)
        self.small_c.append(small_c)
        # Higher-resolution grayscale (half the stat stride) for the optical-flow
        # pass: stride-4 over-smooths sub-pixel drift, so flow needs more detail
        # to resolve sparse, slow, aperiodic motion the other rescues miss.
        s2 = max(1, s // 2)
        small_f = np.asarray(arr, dtype=np.float32)[::s2, ::s2]
        if small_f.ndim == 3:
            small_f = small_f.mean(axis=-1)
        if not np.isfinite(small_f).all():
            small_f = np.nan_to_num(small_f)
        self.small_f.append(small_f)

    def stats(self) -> dict:
        cfg = self.cfg
        if not self.small or self.missing > self.total // 2:
            return {"alive": False, "reason": "no-output", "nan": self.nan,
                    "temporal_var": 0.0, "spatial_var": 0.0,
                    "frame_drop": self.missing}
        # Frames can vary in size if a node changes canvas — crop to smallest.
        h = min(f.shape[0] for f in self.small)
        w = min(f.shape[1] for f in self.small)
        stack = np.stack([f[:h, :w] for f in self.small])  # (T, h, w)

        mean_frame = stack.mean(axis=0)
        spatial_var = float(mean_frame.var())
        temporal_var = float(stack.var(axis=0).mean())

        # Mean consecutive-frame correlation — pure flicker decorrelates.
        corrs = []
        flat = stack.reshape(stack.shape[0], -1)
        for a, b in zip(flat[:-1], flat[1:]):
            sa, sb = a.std(), b.std()
            if sa < 1e-8 or sb < 1e-8:
                corrs.append(1.0)  # constant frames — "correlated", not flicker
            else:
                corrs.append(float(np.dot(a - a.mean(), b - b.mean())
                                   / (len(a) * sa * sb)))
        frame_corr = float(np.mean(corrs)) if corrs else 1.0

        # ── Perceptual motion signal (changed-pixel fraction) ──
        # Global temporal_var averages LOCALIZED motion (a single drifting
        # blob, a rotating thin shape, strokes being drawn) down toward 0,
        # so the variance metric wrongly reports those as "static". Count the
        # fraction of pixels whose per-frame step exceeds motion_thresh; real
        # motion lights up a stable sub-region of the frame, while frozen
        # noise and a global uniform pulse barely register here (the global
        # pulse is already rescued by the temporal_var floor above).
        diffs = np.abs(stack[1:] - stack[:-1])
        if diffs.size == 0:
            # Degenerate: a genome rendered <2 frames, so nothing can move.
            # Guard the empty slice so we never emit a "Mean of empty slice"
            # NaN and leave motion_pixel_frac defined (TD-17).
            motion_pixel_frac = 0.0
        else:
            changed = (diffs > cfg.motion_thresh).mean(axis=0)  # (h, w) per-pixel frac
            motion_pixel_frac = float((changed > 0).mean())

        # ── Spectral-liveness rescue signal (FFT temporal spectrum) ──
        # Amplitude metrics (temporal_var, motion_pixel_frac) miss LOW-AMPLITUDE
        # COHERENT OSCILLATION: a slow breathe / phase-shift / gentle zoom whose
        # per-frame step is below motion_thresh and whose global variance is
        # below temporal_var_min. Coherent periodic motion concentrates its
        # temporal energy into ONE discrete FFT bin (period = T/k frames); frozen
        # noise spreads energy across all bins. For each pixel, take the rFFT of
        # its temporal trace, drop the DC bin, and normalize the magnitude by the
        # total AC energy so the peak bin is ~1.0 for a perfect sine and ~1/K for
        # K bins of flat noise. We take the MEAN of that normalized peak ONLY
        # over pixels that actually carry AC energy (ac_energy >= spectral_ac_min)
        # so a static background does not dilute a localized coherently-oscillating
        # region, and we require that active region to cover a non-trivial
        # fraction of the frame (>= motion_pixel_frac_min) so a single imperceptible
        # flickering pixel can never trigger a rescue. The mean over active pixels
        # of the normalized peak is the spectral-coherence score: ~1.0 for a clean
        # periodic signal, ~1/K for K bins of flat noise. A frozen frame has no
        # active pixels and stays dead.
        T = stack.shape[0]
        centered = stack - stack.mean(axis=0, keepdims=True)        # (T, h, w)
        k = (T // 2) + 1                                            # rFFT bins
        spec = np.abs(np.fft.rfft(centered, axis=0))                # (k, h, w)
        spec = spec[1:] if spec.shape[0] > 1 else spec             # drop DC bin
        ac_energy = spec.sum(axis=0)                                # (h, w) total AC
        # NOTE: spectral_ac is the GLOBAL mean AC over every pixel, so a
        # localized oscillation (small moving object on a dark canvas) has a
        # near-zero global AC and fails any absolute-AC floor — that is the
        # bug this fix addresses. ``spectral_ac_active`` measures the mean AC
        # ONLY over pixels that actually carry AC energy, which is the real
        # "is there a genuine oscillation somewhere" signal.
        spectral_ac = float(ac_energy.mean()) if ac_energy.size else 0.0
        if ac_energy.size and ac_energy.max() > 0:
            # Normalized peak per pixel (0 where there is no AC energy).
            norm = np.zeros_like(ac_energy)
            nz = ac_energy > 0
            norm[nz] = spec.max(axis=0)[nz] / ac_energy[nz]
            active_frac = float(nz.mean())
            spectral_peak = float(norm[nz].mean()) if nz.any() else 0.0
            spectral_ac_active = float(ac_energy[nz].mean()) if nz.any() else 0.0
        else:
            norm = np.zeros_like(ac_energy) if ac_energy.size else np.zeros((1,), np.float32)
            active_frac = 0.0
            spectral_peak = 0.0
            spectral_ac_active = 0.0

        # ── Optical-flow liveness rescue signal (sub-problem #3) ──
        # Dense optical flow measures WHERE pixels move, not just intensity
        # change. It catches the residual after the spectral rescue: LOW-
        # AMPLITUDE NON-PERIODIC STRUCTURED DRIFT (a blob sliding once across, a
        # slow aperiodic pan) that has no sharp spectral peak to trip the
        # periodic-oscillation rescue. ``flow_var`` is the variance of the flow
        # magnitude across the frame; a real drift has a consistent displacement
        # over part of the frame (high variance) while a static frame has ~0
        # flow. ``flow_coherence`` is the alignment of each pixel's per-frame
        # flow with its time-averaged direction (|mean vector| / mean magnitude):
        # ~1.0 for a constant-direction drift, ~0.0 for incoherent flicker — so
        # the coherence gate keeps flicker dead while admitting real drift.
        # Computed on the small stack (already stat_stride-downsampled), further
        # subsampled in time to <= flow_max_frames pairs so the 150s wall is
        # never threatened. Degrades to (0.0, 0.0) if cv2 is unavailable.
        flow_var = 0.0
        flow_coherence = 0.0
        if _HAS_CV2 and T >= 3:
            assert cv2 is not None  # guarded by _HAS_CV2 above
            fseq = np.stack(self.small_f)  # stride-2 grayscale — keeps sub-pixel drift resolvable
            if T > cfg.flow_max_frames:
                fidx = np.linspace(0, T - 1, cfg.flow_max_frames).astype(int)
                fseq = fseq[fidx]
            if cfg.flow_downscale > 1:
                fseq = fseq[:, ::cfg.flow_downscale, ::cfg.flow_downscale]
            # Farnebäck's polynomial expansion needs a uint8 intensity range
            # (0-255); float32 frames in [0,1] collapse to ~0 flow, so scale up.
            fseq = (np.clip(fseq, 0.0, 1.0) * 255.0).astype(np.uint8)
            flows = []
            prev = fseq[0]
            fh, fw = prev.shape
            flow_out = np.zeros((fh, fw, 2), dtype=np.float32)
            for t in range(1, fseq.shape[0]):
                cv2.calcOpticalFlowFarneback(
                    prev, fseq[t], flow_out, 0.5, 3, 15, 3, 5, 1.2, 0)
                flows.append(flow_out.copy())
                prev = fseq[t]
            if flows:
                flows = np.stack(flows, axis=0)            # (K, h, w, 2)
                mags = np.hypot(flows[..., 0], flows[..., 1])
                flow_var = float(mags.var())
                mean_vec = flows.mean(axis=0)              # (h, w, 2)
                mean_mag = np.hypot(mean_vec[..., 0], mean_vec[..., 1])
                per_time_mag = mags.mean(axis=0)          # (h, w)
                moving = per_time_mag > 1e-3
                if moving.any():
                    coh = mean_mag[moving] / per_time_mag[moving]
                    flow_coherence = float(np.mean(np.clip(coh, 0.0, 1.0)))

        # ── Color-aware liveness rescue signal (Route 8 sub-problem #3 closure) ──
        # Every rescue above runs on GRAYSCALE (the luminance-collapsed ``small``
        # / ``small_f`` buffers). That misses CHROMA-ONLY animation: a clip whose
        # per-pixel HUES / CHANNELS cycle while every pixel's luminance stays
        # constant. The Phase-1C research flagged this exact residual — "clips
        # that DO animate but don't change mean-luminance" — and the 643-genome
        # scan's 211 static+flat deaths are its fingerprint. Examples in THIS
        # pipeline: a driver modulating an --recolor ``palette`` (cosmetic color
        # per pitfall color-architecture), a LUT/hue-sweep filter, or a
        # color_intrinsic method whose hue sweeps at fixed luminance. We measure
        # per-pixel color change on the luminance-PRESERVING ``small_c`` buffer.
        # ``color_change_frac`` = fraction of pixels whose mean per-frame RGB step
        # exceeds ``color_thresh``; ``color_struct_corr`` = consecutive-FRAME
        # correlation of the per-pixel color vector (structured color motion, e.g.
        # a palette sweep, has corr ~0.7-0.99; incoherent hue noise ~0.0). A
        # spatial decorrelation (saturation shuffle / channel swap that changes
        # each pixel's color but NOT its local color-vector correlation) is caught
        # by ``color_struct_corr`` being low. Genuinely static color -> ~0 change
        # -> stays dead. Strictly non-destructive: only ever flips
        # static/flat -> alive, never reverse.
        color_change_frac = 0.0
        color_struct_corr = 0.0
        cstack = np.stack([c[:h, :w] for c in self.small_c])  # (T, h, w, 3)
        cd = np.abs(cstack[1:] - cstack[:-1]).mean(axis=-1)   # (T-1, h, w) per-pixel color step
        if cd.size:
            color_change_frac = float((cd > cfg.color_thresh).mean(axis=0).mean())
            # Consecutive-frame correlation of the flattened per-pixel color trace.
            flatc = cstack.reshape(cstack.shape[0], -1)       # (T, h*w*3)
            cc = []
            for a, b in zip(flatc[:-1], flatc[1:]):
                sa, sb = a.std(), b.std()
                if sa < 1e-8 or sb < 1e-8:
                    cc.append(1.0)
                else:
                    cc.append(float(np.dot(a - a.mean(), b - b.mean())
                                    / (len(a) * sa * sb)))
            color_struct_corr = float(np.mean(cc)) if cc else 1.0

        reason = None
        if self.nan:
            reason = "nan"
        elif temporal_var < cfg.temporal_var_min:
            # Not moving by the global variance metric. Try a perceptual
            # rescue: if a meaningful fraction of pixels actually change
            # frame-to-frame AND the motion is temporally STRUCTURED (high
            # consecutive-frame correlation — e.g. a smooth rotation, phase
            # shift, or zoom driven by a control node), keep it.
            #
            # NOTE the correlation sign: ``frame_corr`` is the mean
            # consecutive-frame Pearson correlation over pixels. Smooth
            # structured motion has frame_corr ~0.7–0.99 (each frame is nearly
            # the last, just nudged; even small translating objects overlap
            # most of their area); random flicker/dither has frame_corr ~0.0.
            # The rescue must therefore require frame_corr ABOVE the flicker
            # floor (>= rescue_corr_max, a low threshold ~0.2), NOT below it —
            # a low-correlation clip is flicker, which the flicker gate already
            # handles, and admitting it here would resurrect dead noise. A
            # static clip has frame_corr~1.0 too, but its motion_pixel_frac is
            # ~0 so the first conjunct rejects it. This only ever FLIPS
            # static/flat -> alive, never the reverse.
            if (motion_pixel_frac >= cfg.motion_pixel_frac_min
                    and frame_corr >= cfg.rescue_corr_max):
                reason = None
            elif (spectral_peak >= cfg.spectral_corr_min
                    and spectral_ac_active >= cfg.spectral_ac_min
                    and active_frac >= cfg.spectral_coverage_min):
                # Amplitude metrics (temporal_var, motion_pixel_frac) missed this
                # clip because the motion is LOW-AMPLITUDE but COHERENT (a sharp
                # spectral peak over a meaningful fraction of the frame). A
                # genuine oscillation: rescue it. The spectral_ac floor and the
                # active_frac coverage floor keep a truly frozen frame (no AC
                # energy) or a single flickering pixel (negligible coverage) dead.
                reason = None
            elif (flow_var >= cfg.flow_var_min
                    and flow_coherence >= cfg.flow_coherence_min):
                # Optical-flow rescue (sub-problem #3): a small object making a
                # slow, COHERENT drift produces real pixel displacement but low
                # GLOBAL variance (temporal_var below floor), sparse coverage
                # (motion_pixel_frac below floor), and no sharp spectral peak.
                # Dense optical flow sees the structured displacement: high
                # flow-magnitude variance + high direction coherence. Flicker has
                # low coherence; a static frame has ~0 flow magnitude -> both stay
                # dead. This branch only ever flips dead -> alive, never reverse.
                reason = None
            elif (color_change_frac >= cfg.color_change_frac_min
                    and color_struct_corr >= cfg.color_corr_min):
                # Color-aware rescue (Route 8 sub-problem #3 closure): chroma-only
                # animation. Every other rescue above runs on GRAYSCALE and misses
                # a clip whose hues/channels cycle at constant luminance (palette
                # sweeps, LUT/hue filters, recolor driven by a control node). The
                # motion is genuine but leaves NO luminance footprint, so it is
                # rescued only here: a meaningful fraction of pixels must change
                # color frame-to-frame AND that color change must be temporally
                # STRUCTURED (high consecutive-frame color-vector correlation), so
                # incoherent hue noise / per-frame saturation shuffles stay dead.
                # Strictly non-destructive: flips static/flat -> alive, never reverse.
                reason = None
            else:
                # Not moving. If it is also spatially degenerate
                # (near black/white/uniform) call it "flat"; otherwise it
                # is a structured-but-frozen "static" clip.
                reason = "flat" if spatial_var < cfg.spatial_var_min else "static"
        elif spatial_var < cfg.spatial_var_min:
            # Moving (passed the temporal floor above) but spatially
            # smooth / low-contrast. Motion wins: a smooth gradient or
            # global pulse that animates is a live clip, not a "flat"
            # one. The old gate culled these as "flat" and threw away
            # genuinely dynamic content (e.g. a slow brightness wipe, a
            # smooth field advecting). Only a clip that is BOTH
            # static AND degenerate should be called flat.
            pass
        elif temporal_var > cfg.flicker_var_min and frame_corr < cfg.flicker_corr_max:
            reason = "flicker"

        return {
            "alive": reason is None,
            "reason": reason,
            "nan": self.nan,
            "temporal_var": round(temporal_var, 6),
            "spatial_var": round(spatial_var, 6),
            "frame_corr": round(frame_corr, 4),
            "motion_pixel_frac": round(motion_pixel_frac, 4),
            "spectral_peak": round(spectral_peak, 4),
            "spectral_ac": round(spectral_ac, 6),
            "spectral_ac_active": round(spectral_ac_active, 6),
            "spectral_active_frac": round(active_frac, 4),
            "flow_var": round(flow_var, 4),
            "flow_coherence": round(flow_coherence, 4),
            "color_change_frac": round(color_change_frac, 4),
            "color_struct_corr": round(color_struct_corr, 4),
            "frame_drop": self.missing,
        }


def evaluate_frames(frames: list[np.ndarray | None],
                    cfg: ShootoutConfig = DEFAULT_CONFIG) -> dict:
    """Pure liveness classification of a frame stack (test hook)."""
    acc = LivenessAccumulator(cfg)
    for f in frames:
        acc.add(f)
    return {**acc.stats(), "evaluator_version": EVALUATOR_VERSION}


# ── Executor plumbing shared by render + ablation ─────────────────────


def _pin_n_frames(nodes: list[dict], n_frames: int) -> list[dict]:
    """Pin every node that declares an n_frames param to the clip budget.

    Sims default to their own (often 300+ frame) length; without this a
    short clip cooks the sim's full run. Mutates node dicts in place and
    returns them. Kept as a shared helper so the render path and the
    contribution ablation pin identically (they must, to stay comparable)."""
    from image_pipeline.core.graph import get_all_node_defs
    defs = get_all_node_defs()
    for n in nodes:
        schema = (defs.get(n.get("method_id"), {}).get("params") or {})
        if "n_frames" in schema:
            n["params"] = {**(n.get("params") or {}), "n_frames": n_frames}
    return nodes


def _terminal_image(flat: dict, terminal_id, nodes: list[dict]):
    """The output frame for a cooked graph: the render-flagged node if one
    produced output this frame, else the executor's resolved terminal."""
    render_id = next((n["id"] for n in nodes if n.get("render")), None)
    if render_id and render_id in flat:
        terminal_id = render_id
    return (flat.get(terminal_id) or {}).get("image") if terminal_id else None


def render_stack(nodes: list[dict], edges: list[dict], seed: int,
                 cfg: ShootoutConfig, frames: int,
                 progress_cb: Callable[[str], None] | None = None
                 ) -> LivenessAccumulator:
    """Render (nodes, edges) into a LivenessAccumulator of downsampled
    frames — no mp4, no disk artifact. Node/edge lists are copied, so the
    caller's graph is never mutated. Node failures become dropped frames
    (never raises), same as the full render path.

    Shared by contribution ablation: baseline and every ablated variant go
    through here so their frame stacks are directly comparable."""
    import tempfile
    import shutil
    from image_pipeline.core.utils import set_canvas
    set_canvas(cfg.width, cfg.height)

    nodes = _pin_n_frames([dict(n, dirty=True) for n in nodes], frames)
    edges = [dict(e) for e in edges]

    work_dir = Path(tempfile.mkdtemp(prefix="shootout-contrib-"))
    executor = GraphExecutor(work_dir, fps=cfg.fps, in_memory=True,
                             audit_to_disk=False)
    acc = LivenessAccumulator(cfg)
    t0 = time.time()
    try:
        for frame in range(frames):
            if time.time() - t0 > cfg.render_timeout_s:
                break
            arr = None
            try:
                flat, terminal_id, _errs = executor.execute(
                    nodes, edges, seed, frame=frame, frames=frames)
                arr = _terminal_image(flat, terminal_id, nodes)
            except Exception as exc:  # cycle / unknown method — dropped frame
                if progress_cb:
                    progress_cb(f"ablation frame {frame} error: {exc}")
            acc.add(arr)
    finally:
        shutil.rmtree(work_dir, ignore_errors=True)
    return acc


# ── Rendering ─────────────────────────────────────────────────────────


def render_genome(genome: dict, cfg: ShootoutConfig = DEFAULT_CONFIG,
                  progress_cb: Callable[[str], None] | None = None,
                  render_timeout_s: float | None = None) -> dict:
    """Render genome → output/sequences/shootout-<id>/output.mp4, fill
    genome['render'] + genome['liveness']. Never raises on node failures —
    executor errors become dead clips."""
    from image_pipeline.core.utils import set_canvas
    set_canvas(cfg.width, cfg.height)

    gid = genome["genome_id"]
    name = seq_name_for(gid)
    seq_dir = SEQUENCES_DIR / name
    work_dir = seq_dir / "_work"
    work_dir.mkdir(parents=True, exist_ok=True)

    graph = genome["graph"]
    # Copies: the executor and keyframe folding mutate node dicts.
    nodes = [dict(n, dirty=True) for n in graph.get("nodes", [])]
    edges = [dict(e) for e in graph.get("edges", [])]

    # Pin simulation length to the clip budget. The executor only overrides
    # n_frames when the param is present on the node; the sampler leaves it
    # off (frozen), so without this an Arch-A sim cooks its full default
    # (often 300+ frames) for a 96-frame clip.
    _pin_n_frames(nodes, cfg.frames)
    seed = int(genome.get("seed", 42))

    # Live telemetry + skip: register this render on the shared board, install
    # the skip event the executor's sim loop polls, and a node-progress hook
    # so the heartbeat can report the node cooking right now.
    mon = progress.MONITOR
    skip_ev = mon.skip_event(gid)
    mon.begin(gid, cfg.frames, len(nodes))

    executor = GraphExecutor(work_dir, fps=cfg.fps, in_memory=True,
                             audit_to_disk=False)
    executor.cancel_event = skip_ev
    executor.node_progress = (
        lambda node_id, method_id, name, sim_frame=None:
        mon.node_cooking(gid, node_id, method_id, name, sim_frame=sim_frame))
    acc = LivenessAccumulator(cfg)
    t0 = time.time()
    timed_out = False
    skipped = False
    # Per-genome render cap (Route 8, 2026-07-14): slow-but-likely-dynamic
    # heavy sims get an extended cap so they can finish instead of timing out.
    if render_timeout_s is not None:
        eff_timeout = render_timeout_s
    else:
        # Defensive default for ANY direct caller (not just render_many, which
        # passes the extended cap explicitly): consult the cost model so a
        # standalone render of a heavy-but-likely-dynamic clip still gets the
        # extended cap instead of timing out at the base cap. This can only
        # EVER EXTEND the cap — never shorten it — so it is monotonic-safe and
        # production renders (always via render_many with an explicit cap) are
        # unaffected. Guarded so a cost-model failure falls back to the base
        # cap rather than crashing the render.
        try:
            from . import cost_model as _cm
            eff_timeout = _cm.effective_render_timeout_s(
                genome, cfg, _cm.load_cost_model())
        except Exception:
            eff_timeout = cfg.render_timeout_s
    # Node-error tracking: a method that raises at render (inter-param bugs,
    # OpenCV bad-args, index errors) yields an error-placeholder frame. Record
    # which nodes threw so the clip can be culled instead of shipped.
    node_error_nodes: set[str] = set()
    node_error_sample: str = ""
    # Per-node compute, summed across every rendered frame. The executor
    # reports ms-per-node for the *last* frame only (last_frame_stats), so
    # we fold each frame's timings in here to get total compute per node.
    node_ms: dict[str, float] = {}

    from image_pipeline.core.animation import JobCancelled

    def _frame_gen():
        nonlocal timed_out, skipped, node_error_sample
        for frame in range(cfg.frames):
            if skip_ev.is_set():
                skipped = True
                if progress_cb:
                    progress_cb(f"{gid}: skipped at frame {frame}")
                return
            if time.time() - t0 > eff_timeout:
                timed_out = True
                if progress_cb:
                    progress_cb(f"{gid}: timeout after {frame} frames")
                return
            mon.frame_start(gid, frame)
            arr = None
            try:
                flat, terminal_id, _errs = executor.execute(
                    nodes, edges, seed, frame=frame, frames=cfg.frames)
                # Fold this frame's per-node compute into the running total.
                for nid, ms in (executor.last_frame_stats.get("node_timings") or {}).items():
                    node_ms[nid] = node_ms.get(nid, 0.0) + ms
                # A node that raised is reported here (execute() swaps in an
                # error placeholder and keeps going). Record it so a graph with
                # a broken method gets culled rather than shipped.
                if _errs:
                    node_error_nodes.update(_errs)
                    if not node_error_sample:
                        node_error_sample = str(next(iter(_errs.values()))).splitlines()[0][:120]
                arr = _terminal_image(flat, terminal_id, nodes)
            except JobCancelled:  # skip button / watchdog aborted a wedged node
                skipped = True
                if progress_cb:
                    progress_cb(f"{gid}: skipped mid-frame {frame}")
                return
            except Exception as exc:  # Arch-A raise / cycle / unknown — dead clip
                # Unlike Arch-B node failures (reported via _errs), an exception
                # that escapes execute() isn't tied to a node id — record it so
                # the clip is culled as node_error, not merely "no-output".
                node_error_nodes.add("_exec")
                if not node_error_sample:
                    node_error_sample = str(exc).splitlines()[0][:120]
                if progress_cb:
                    progress_cb(f"{gid}: frame {frame} error: {exc}")
            acc.add(arr)
            if arr is not None:
                yield np.asarray(arr, dtype=np.float32)
                # Live preview thumbnail: capture on the very first rendered
                # frame (so a still shows immediately) and then every
                # `preview_every` frames after, so the user can eyeball +
                # skip the clip before it lands in the survivor pool.
                # Cheap + best-effort.
                if frame == 0 or (cfg.preview_every
                        and frame % cfg.preview_every == 0):
                    mon.capture_preview(gid, frame, arr, cfg.preview_w)
            if progress_cb and frame % 24 == 0:
                progress_cb(f"{gid}: frame {frame + 1}/{cfg.frames}")

    mp4_path = seq_dir / "output.mp4"
    out = None
    try:
        out = frames_to_mp4(_frame_gen(), mp4_path, fps=cfg.fps,
                            max_frames=cfg.frames)
    except Exception as exc:
        if progress_cb:
            progress_cb(f"{gid}: encode failed: {exc}")
    finally:
        mon.finish(gid)

    liveness = acc.stats()
    captured = acc.total - acc.missing
    min_frames = int(cfg.frames * cfg.min_render_frames_frac)
    if skipped:
        # Watchdog hard-wall aborts (total elapsed past the budget) should be
        # treated like a timeout — recover the clip as "truncated" if it
        # captured enough frames and passes liveness, rather than discarding a
        # slow-but-dynamic animation as a plain skip. A manual/frame-stuck skip
        # (elapsed still under the hard wall) stays a plain cull.
        wall_limit = eff_timeout * getattr(cfg, "hard_wall_factor", 1.15)
        hard_walled = (time.time() - t0) >= wall_limit
        if hard_walled and captured >= min_frames and liveness.get("alive"):
            liveness = {**liveness, "truncated": True,
                        "reason": liveness.get("reason")}
        else:
            liveness = {**liveness, "alive": False, "reason": "skipped"}
    elif timed_out:
        # Recover good clips: only cull as "timeout" when we captured too few
        # frames to form a meaningful clip. If most frames rendered and the
        # liveness gate passes, keep the clip (mark it truncated) instead of
        # discarding a slow-tailed but dynamic animation.
        if captured >= min_frames and liveness.get("alive"):
            liveness = {**liveness, "truncated": True,
                        "reason": liveness.get("reason")}
        else:
            liveness = {**liveness, "alive": False, "reason": "timeout"}

    # A node that raised means the clip is showing error placeholders — cull it
    # (unless the user hand-skipped, which already set its own reason). This is
    # the render-time backstop for runtime errors that static validation can't
    # catch; see cfg.reject_node_errors.
    if cfg.reject_node_errors and node_error_nodes and not skipped:
        liveness = {**liveness, "alive": False, "reason": "node_error",
                    "node_error_nodes": sorted(node_error_nodes),
                    "node_error": node_error_sample}

    # The _work dir holds per-node transport PNGs from Arch-A sims — the mp4
    # is the durable artifact, so drop the scratch.
    import shutil
    shutil.rmtree(work_dir, ignore_errors=True)

    return {
        **genome,
        "render": {
            "seq_name": name,
            "mp4": f"/api/sequences/{name}/video.mp4",
            "frames": cfg.frames,
            "fps": cfg.fps,
            "w": cfg.width,
            "h": cfg.height,
            "wall_s": round(time.time() - t0, 1),
            # {node_id: total_ms across all frames} — drives the
            # "slowest node" readout on the shootout card.
            "node_timings": {nid: round(ms, 1) for nid, ms in node_ms.items()},
        },
        "liveness": {**liveness, "evaluator_version": EVALUATOR_VERSION},
    }


def render_many(genomes: list[dict], cfg: ShootoutConfig = DEFAULT_CONFIG,
                progress_cb: Callable[[str], None] | None = None) -> list[dict]:
    """Render candidates concurrently (capped — plan §12 perf note).
    Returns genomes in input order with render/liveness filled.

    A heartbeat thread reads the live render board every cfg.heartbeat_s and
    emits a status line per in-flight clip (current frame, node cooking now,
    seconds on this frame), so a hang is visible second-by-second. It also
    acts as the wedged-render watchdog: any clip whose *current frame* has run
    past render_timeout_s (or the tighter auto_skip_frame_hang_s, if set) is
    force-skipped — the safety net the between-frame timeout can't provide
    when a node is stuck mid-cook."""
    results: dict[int, dict] = {}
    lock = threading.Lock()

    def _safe_progress(msg: str) -> None:
        if progress_cb:
            with lock:
                progress_cb(msg)

    # Fresh board for this batch (drop any finished entries from the last one).
    progress.MONITOR.clear_all()
    stop = threading.Event()

    # Per-genome effective render cap (Route 8, 2026-07-14): slow-but-likely-
    # dynamic heavy sims get an extended cap so they can finish instead of
    # timing out. Computed once here (cost model loaded once) and threaded into
    # both render_genome and the watchdog below so the watchdog never force-
    # skips an extended-cap clip at the base cap.
    try:
        from . import cost_model as _cm
        _cm_model = _cm.load_cost_model()
        eff_timeout: dict[str, float] = {
            g["genome_id"]: _cm.effective_render_timeout_s(g, cfg, _cm_model)
            for g in genomes
        }
    except Exception:
        eff_timeout = {}

    def _heartbeat() -> None:
        hard = cfg.auto_skip_frame_hang_s or 0.0
        while not stop.wait(cfg.heartbeat_s):
            now = time.time()
            snap = progress.MONITOR.snapshot()
            for line in progress.heartbeat_lines(snap, cfg.frame_hang_s, now):
                _safe_progress(line)
            # Watchdog: force-skip a clip wedged on a single frame.
            for gid, s in snap.items():
                if s.get("skip_requested"):
                    continue
                on_frame = now - s.get("t_frame", now)
                limit = hard if hard > 0 else eff_timeout.get(gid, cfg.render_timeout_s)
                if on_frame > limit:
                    progress.MONITOR.request_skip(gid)
                    _safe_progress(
                        f"⏹ {gid}: auto-skip — frame stuck {on_frame:.0f}s "
                        f"(> {limit:.0f}s), likely a wedged node")
                    continue
                # Hard total-wall watchdog: a clip that keeps progressing but is
                # simply slow (each frame < the per-frame limit) never trips the
                # check above and sails past the render budget — empirically up
                # to ~547s against a 300s cap. Force-skip once total elapsed
                # exceeds the (per-genome) render cap × hard_wall_factor so the
                # over-run compute is reclaimed instead of wasted on a clip that
                # will be culled as timeout anyway.
                elapsed = now - s.get("t0", now)
                wall_limit = eff_timeout.get(gid, cfg.render_timeout_s) * getattr(
                    cfg, "hard_wall_factor", 1.15)
                if elapsed > wall_limit:
                    progress.MONITOR.request_skip(gid)
                    _safe_progress(
                        f"⏹ {gid}: auto-skip — total render {elapsed:.0f}s "
                        f"(> {wall_limit:.0f}s hard wall), reclaiming budget")

    hb = threading.Thread(target=_heartbeat, daemon=True)
    hb.start()
    try:
        with ThreadPoolExecutor(max_workers=cfg.render_concurrency) as ex:
            futs = {ex.submit(render_genome, g, cfg, _safe_progress,
                            eff_timeout.get(g["genome_id"], cfg.render_timeout_s)): i
                    for i, g in enumerate(genomes)}
            for fut in as_completed(futs):
                i = futs[fut]
                try:
                    results[i] = fut.result()
                except Exception as exc:
                    _safe_progress(f"{genomes[i]['genome_id']}: render crashed: {exc}")
                    results[i] = {**genomes[i],
                                  "render": None,
                                  "liveness": {"alive": False, "reason": "crash",
                                               "error": str(exc)}}
    finally:
        stop.set()
        hb.join(timeout=2.0)
        # NOTE: do NOT clear_all() here. Each genome's board slot is
        # marked done by render_genome's mon.finish(), and the
        # render-status endpoint + live panel already drop done/finished
        # genomes from the poll, so the board self-prunes. Clearing
        # here would wipe the just-captured preview thumbnails
        # before the 1s live panel can read them (short clips
        # finish faster than the poll), leaving only the black
        # placeholder. Leftover slots from a crashed run are
        # cleared by the clear_all() at the top of the NEXT run.
    return [results[i] for i in range(len(genomes))]
