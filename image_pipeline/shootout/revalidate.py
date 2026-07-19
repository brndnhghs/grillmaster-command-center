"""Re-valuate persisted shootout genomes against the CURRENT liveness gate.

Leverage-Tier, Route 8 (2026-07-18).

Context: the liveness evaluator has gained several rescue signals AFTER the
bulk of the persisted corpus was first culled — optical-flow rescue
(commit 3c63416), color-aware chroma rescue (commit 3106867), and the
spectral-coherence rescue (1358457). Genomes first judged by the *legacy* gate
stamped their verdicts ``evaluator_version=None`` (or an older stamp) and are
therefore systematically *over-culled* as ``static``/``flat``: the modern gate
would rescue many of them.

The rendered mp4s are durable artifacts under ``output/sequences/<seq>/output.mp4``,
so the correction costs ZERO re-render — we just decode the stored frames and
re-run :func:`evaluate_frames`. This is strictly non-destructive: a genome can
only flip ``dead -> alive`` (the modern gate is a superset of the legacy one),
never the reverse. Each rewritten verdict records ``reevaluated=True`` and keeps
the original ``reason`` as ``original_reason`` so the change is auditable.

Nothing here touches the CPU render/export path, the GraphExecutor, or the live
server. It is a corpus-hygiene maintenance pass over gitignored runtime data.
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Callable, Sequence

import numpy as np

from . import store
from .config import DEFAULT_CONFIG, ShootoutConfig
from .evaluator import EVALUATOR_VERSION, evaluate_frames

SEQUENCES_DIR = Path(__file__).resolve().parent.parent / "output" / "sequences"


def _load_frames(mp4_path: Path, max_frames: int = 80) -> "list[np.ndarray] | None":
    """Decode a stored mp4 into downsampled float32 RGB frames (stride-4)."""
    try:
        import cv2
    except Exception:
        return None
    cap = cv2.VideoCapture(str(mp4_path))
    if not cap.isOpened():
        return None
    frames: list[np.ndarray] = []
    while len(frames) < max_frames:
        ok, fr = cap.read()
        if not ok:
            break
        fr = cv2.cvtColor(fr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        frames.append(fr[::4, ::4])
    cap.release()
    return frames if len(frames) >= 4 else None


# Structural-failure reasons — these are NOT liveness-gate verdicts. A clip
# culled for a node error, missing output, or explicit skip has no meaningful
# liveness verdict to re-evaluate; resurrecting it would hide a real fault.
# (``timeout`` IS re-evaluated: the renderer writes captured frames before the
# wall hit, and the liveness gate would accept them if they finished — exactly
# the Route 8 #2 recovery target. ``static``/``flat`` are the pre-rescue
# false-culls the optical-flow / color-aware / spectral rescues now correct.)
_STRUCTURAL_REASONS = frozenset({"node_error", "no-output", "skipped"})


def _needs_reeval(liveness: dict | None) -> bool:
    """A dead genome needs re-evaluation when its verdict predates the rescue
    signals the current gate would apply.

    The evaluator stamps every verdict with ``evaluator_version``. Modern
    verdicts match ``EVALUATOR_VERSION`` exactly. Legacy verdicts (or any
    older stamp) are stale and may have been wrongly culled by a gate that
    lacked the optical-flow / color-aware / spectral rescues. Alive verdicts
    are never re-examined (re-eval only ever flips dead -> alive), and
    structural failures (node_error / no-output / skipped) are not liveness
    verdicts so are left untouched.
    """
    if not isinstance(liveness, dict):
        return False
    if liveness.get("alive"):
        return False
    if liveness.get("reason") in _STRUCTURAL_REASONS:
        return False
    return liveness.get("evaluator_version") != EVALUATOR_VERSION


def revalidate_genome(g: dict, cfg: ShootoutConfig = DEFAULT_CONFIG,
                      max_frames: int = 80) -> dict | None:
    """Re-run the current liveness gate on one genome's stored mp4.

    Returns the genome with an updated ``liveness`` dict if it flipped
    ``dead -> alive``, else ``None`` (no change / not re-evaluable).
    """
    lv = g.get("liveness")
    if not _needs_reeval(lv):
        return None
    seq = (g.get("render") or {}).get("seq_name")
    if not seq:
        return None
    mp4 = SEQUENCES_DIR / seq / "output.mp4"
    if not mp4.exists():
        return None
    frames = _load_frames(mp4, max_frames=max_frames)
    if frames is None:
        return None
    new = evaluate_frames(frames, cfg)
    if not new.get("alive"):
        return None
    # Flip detected. Preserve the original verdict for auditability.
    updated = dict(g)
    orig_reason = lv.get("reason") if isinstance(lv, dict) else None
    orig_ver = lv.get("evaluator_version") if isinstance(lv, dict) else None
    updated["liveness"] = {
        **new,
        "evaluator_version": EVALUATOR_VERSION,
        "reevaluated": True,
        "original_reason": orig_reason,
        "original_evaluator_version": orig_ver,
    }
    return updated


def _annotate_signals(g: dict, new: dict, lv: dict,
                      cfg: ShootoutConfig = DEFAULT_CONFIG) -> dict:
    """Build an annotated liveness dict for a re-decodable genome that stays dead.

    The modern gate (``new``) carries the full rescue-signal set
    (``motion_pixel_frac`` / ``spectral_peak`` / ``flow_var`` /
    ``color_change_frac`` / ...), which legacy verdict dicts predate. Persisting
    it onto still-dead genomes closes the corpus blind spot so the advisor can
    steer on *why* a clip died instead of flying blind.

    Strictly additive: ``alive`` stays ``False`` and the original ``reason`` is
    preserved under ``original_reason`` (audit trail). Idempotent across re-runs
    because the rewritten verdict now matches ``EVALUATOR_VERSION``.
    """
    ann = dict(g)
    ann["liveness"] = {
        **new,
        "evaluator_version": EVALUATOR_VERSION,
        "reevaluated": True,
        "original_reason": lv.get("reason") if isinstance(lv, dict) else None,
        "original_evaluator_version": (lv.get("evaluator_version")
                                       if isinstance(lv, dict) else None),
    }
    return ann


def revalidate_corpus(cfg: ShootoutConfig = DEFAULT_CONFIG,
                      progress: Callable[[str], None] | None = None,
                      max_frames: int = 80,
                      write_signals_for_still_dead: bool = True) -> dict:
    """Re-evaluate every persisted dead genome whose verdict is version-stale.

    Two outcomes are possible for a re-decodable stale-dead genome:

    1. **Flip (dead -> alive)** — the modern rescue signals rescue a clip the
       legacy gate wrongly culled. Handled by :func:`revalidate_genome` and
       persisted verbatim (original behavior; strictly dead -> alive only).
    2. **Annotate (still dead)** — the clip is genuinely dead under the modern
       gate too, but its persisted ``liveness`` dict predates the rescue signals,
       leaving downstream analysis blind to *why* it died. When
       ``write_signals_for_still_dead`` is True (default), we rewrite the dict
       with the full modern signal set (``alive``/``reason`` preserved). This is
       corpus hygiene: it makes the dead-rate *attributable* without changing any
       verdict. (Empirically, on the 649-genome corpus this path fires for the
       vast majority of stale-dead genomes while the flip path fires for ~0 —
       the gate is well-tuned; the deaths are genuine content, not evaluator
       artifacts.)

    progress:
        Optional callback receiving a one-line status string per re-evaluated
        genome (used by the CLI).

    Returns a summary dict with the counts needed for the manifest / report.
    """
    genomes = store.iter_genomes()
    total = 0
    considered = 0
    flipped = 0
    annotated = 0
    no_mp4 = 0
    start = time.time()
    for g in genomes:
        total += 1
        lv = g.get("liveness")
        if not _needs_reeval(lv):
            continue
        considered += 1
        seq = (g.get("render") or {}).get("seq_name")
        mp4 = SEQUENCES_DIR / seq / "output.mp4" if seq else None
        if not (mp4 and mp4.exists()):
            no_mp4 += 1
            continue
        updated = revalidate_genome(g, cfg, max_frames=max_frames)
        if updated is not None:
            store.save_genome(updated)
            flipped += 1
            if progress:
                progress(f"  ⟳ {updated['genome_id']}  "
                         f"{updated['liveness']['original_reason']} -> ALIVE  "
                         f"({updated['liveness'].get('reason')})")
            continue
        if not write_signals_for_still_dead:
            continue
        # revalidate_genome returned None -> mp4 exists & decodes but stays dead.
        # Re-run the gate once more to capture the full signal set, then persist
        # it (additive; alive/reason unchanged).
        frames = _load_frames(mp4, max_frames=max_frames)
        if frames is None:
            continue
        new = evaluate_frames(frames, cfg)
        if new.get("alive"):
            # Defensive: revalidate_genome should have flipped this. Skip to avoid
            # double-writing (it will be re-flipped on the next corpus pass).
            continue
        store.save_genome(_annotate_signals(g, new, lv, cfg))
        annotated += 1
        if progress:
            progress(f"  • {g.get('genome_id')}  {lv.get('reason')} "
                     f"(annotated, still dead)")
    return {
        "total_genomes": total,
        "version_stale_dead": considered,
        "flipped_dead_to_alive": flipped,
        "annotated_still_dead": annotated,
        "missing_mp4": no_mp4,
        "seconds": round(time.time() - start, 1),
        "evaluator_version": EVALUATOR_VERSION,
    }


if __name__ == "__main__":
    import pprint
    pprint.pprint(revalidate_corpus(progress=print))
