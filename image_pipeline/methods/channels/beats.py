"""CHOP-like channel generator nodes.
Auto-imported by channels/__init__.py.
"""
from __future__ import annotations
import math
import random
from pathlib import Path
import numpy as np
from ...core.registry import method
from ...core.utils import seed_all

# ── Per-node state for reset tracking ──────────────────────────────────
# Keyed by _node_id (injected by GraphExecutor into run_params).
# Stores per-instance tracking: {"reset_frame": int, "prev_reset": float}.
_BEATS_STATE: dict[str, dict] = {}
_BEATS_PRUNE_COUNTER = 0


@method(id="__beats__", name="Beats", category="channels",
        tags=["chop", "time", "music", "generator"],
        inputs={"reset": "SCALAR", "swing": "SCALAR"},
        outputs={"value": "SCALAR", "beat": "SCALAR", "bar": "SCALAR",
                 "trigger": "SCALAR", "downbeat": "SCALAR",
                 "beat_index": "SCALAR", "bar_count": "SCALAR"},
        runtime={
            "value": {
                "type": "numeric",
                "label": "Value",
                "observable": True
            },
            "beat": {
                "type": "output",
                "label": "Beat",
                "observable": True
            },
            "bar": {
                "type": "output",
                "label": "Bar",
                "observable": True
            },
            "trigger": {
                "type": "event",
                "label": "Trigger",
                "observable": True
            },
            "downbeat": {
                "type": "event",
                "label": "Downbeat",
                "observable": True
            },
            "beat_index": {
                "type": "output",
                "label": "Beat Index",
                "observable": True
            },
            "bar_count": {
                "type": "output",
                "label": "Bar Count",
                "observable": True
            }
        },
        signal={
            "reset": "control",
            "swing": "numeric",
            "value": "output",
            "beat": "output",
            "bar": "output",
            "trigger": "event",
            "downbeat": "event",
            "beat_index": "output",
            "bar_count": "output",
        },
        params={
            "bpm": {"description": "beats per minute", "min": 20, "max": 300, "default": 120},
            "beats_per_bar": {"description": "beats per bar / time signature numerator", "min": 1, "max": 16, "default": 4},
            "swing": {"description": "swing amount 0–1", "default": 0.0},
            "swing_type": {"description": "swing pattern type",
                          "choices": ["shuffle", "triplet", "off"], "default": "shuffle"},
            "fps": {"description": "frames per second for beat calculation", "min": 1, "max": 120, "default": 24},
        })
def method_beats(out_dir: Path, seed: int, params=None):
    """Musical beat generator — outputs beat phase, bar phase, triggers,
    beat index, bar count, and downbeat marker.

    The beat grid accounts for swing by shifting the onset of swung beats
    in the time domain, so all derived quantities (phase, trigger, bar phase)
    reflect the swung rhythm correctly.

    Outputs:
        beat      (SCALAR): 0→1 phase within current beat
        bar       (SCALAR): 0→1 phase within current bar
        trigger   (SCALAR): 1.0 on first frame of each beat, 0 otherwise
        downbeat  (SCALAR): 1.0 only on first beat of bar, 0 otherwise
        beat_index (SCALAR): 0-indexed beat number within bar
        bar_count  (SCALAR): cumulative completed bars since start
        value     (SCALAR): same as beat (backward-compat alias)
    """
    if params is None:
        params = {}
    seed_all(seed)

    frame = int(params.get("frame", 0))
    bpm = float(params.get("bpm", 120))
    beats_per_bar = int(params.get("beats_per_bar", 4))
    swing = float(params.get("swing", 0.0))
    fps = float(params.get("fps", 24))
    swing_type = params.get("swing_type", "shuffle")
    node_id = params.get("_node_id", "")

    # Derive live frame and fps from injected Timeline
    # The GraphExecutor does NOT inject a raw `frame` integer for CHOP nodes.
    if frame == 0:
        _tl = params.get("_timeline")
        if _tl is not None:
            frame = int(getattr(_tl, "global_frame", 0))
            fps = float(getattr(_tl, "fps", fps))

    # SCALAR input overrides
    reset_val = params.get("reset")
    swing_override = params.get("swing")
    if swing_override is not None:
        swing = float(swing_override)

    # ── Clamp swing ────────────────────────────────────────────────────
    swing = max(0.0, min(1.0, swing))

    # ── Stateful reset ─────────────────────────────────────────────────
    global _BEATS_PRUNE_COUNTER
    _BEATS_PRUNE_COUNTER += 1

    reset_frame_offset = 0
    if node_id and reset_val is not None:
        state = _BEATS_STATE.setdefault(node_id, {
            "reset_frame": 0,
            "prev_reset": 0.0,
        })
        # Rising-edge detection: reset fires when reset_val crosses >0.5
        prev = state.get("prev_reset", 0.0)
        if prev <= 0.5 < float(reset_val):
            state["reset_frame"] = frame
        state["prev_reset"] = float(reset_val)
        reset_frame_offset = state.get("reset_frame", 0)

    # Lazy prune
    if _BEATS_PRUNE_COUNTER % 1000 == 0:
        cutoff = frame - 7200
        for nid in list(_BEATS_STATE):
            if _BEATS_STATE[nid].get("reset_frame", 0) < cutoff:
                del _BEATS_STATE[nid]

    # ── Effective frame (offset by reset) ──────────────────────────────
    eff_frame = max(0, frame - reset_frame_offset)

    # ── Beat grid computation with swing ───────────────────────────────
    fpb = fps * 60.0 / bpm  # frames per beat (non-swung)
    swing_max_shift = 0.45  # max 45% of a beat's duration

    def _is_swung(beat_n: int) -> bool:
        """Return True if beat_n is a swung (delayed) beat."""
        if swing <= 0 or swing_type == "off":
            return False
        if swing_type == "shuffle":
            return (beat_n % 2) == 1
        if swing_type == "triplet":
            return (beat_n % 3) == 2
        return False

    def _beat_start(beat_n: int) -> float:
        """Grid-aligned start frame of beat N.

        Non-swung beats start at their exact grid position (N * fpb).
        Swung beats start later by swing * shift, but the NEXT beat
        snaps back to the grid — the swung beat is simply shorter.
        This matches classic drum-machine swing behavior.
        """
        nominal = beat_n * fpb
        if _is_swung(beat_n) and swing > 0:
            return nominal + swing * swing_max_shift * fpb
        return nominal

    # Find which beat we're on by walking the grid.
    # The grid is non-uniform with swing, so we iterate from the start
    # until we find the beat containing eff_frame.
    beat_n = 0
    beat_start = 0.0
    beat_end = 0.0

    while True:
        start_here = _beat_start(beat_n)
        start_next = _beat_start(beat_n + 1)

        if start_here <= eff_frame < start_next - 1e-12 or start_next <= start_here:
            # eff_frame is in this beat, or this is the last beat we can reach
            # (safety: degenerate case where duration collapsed)
            beat_start = start_here
            beat_end = start_next if start_next > start_here else start_here + fpb
            break
        beat_n += 1

    # ── Beat phase ─────────────────────────────────────────────────────
    beat_dur = beat_end - beat_start
    if beat_dur > 0:
        beat_phase = (eff_frame - beat_start) / beat_dur
    else:
        beat_phase = 0.0
    beat_phase = max(0.0, min(1.0, beat_phase))

    # ── Bar phase ──────────────────────────────────────────────────────
    # Find bar boundaries: bar N starts at beat_start(bar_start_beat)
    bar_start_beat = (beat_n // beats_per_bar) * beats_per_bar
    bar_end_beat = bar_start_beat + beats_per_bar
    bar_start_frame = _beat_start(bar_start_beat)
    bar_end_frame = _beat_start(bar_end_beat)
    bar_dur_actual = bar_end_frame - bar_start_frame
    if bar_dur_actual > 0:
        bar_phase = (eff_frame - bar_start_frame) / bar_dur_actual
    else:
        bar_phase = 0.0
    bar_phase = max(0.0, min(1.0, bar_phase))

    # ── Beat index within bar ──────────────────────────────────────────
    beat_idx = beat_n % beats_per_bar

    # ── Bar count ──────────────────────────────────────────────────────
    bar_count = beat_n // beats_per_bar

    # ── Trigger: 1 on the first frame of each beat ─────────────────────
    # Frame 0 is always the start of the first beat.
    if eff_frame == 0:
        trigger = 1.0
    else:
        prev_eff = eff_frame - 1
        if prev_eff < beat_start:
            prev_beat_n = beat_n - 1
        else:
            prev_beat_n = beat_n
        trigger = 1.0 if prev_beat_n != beat_n else 0.0

    # ── Downbeat: 1 on first beat of each bar ──────────────────────────
    downbeat = 1.0 if trigger > 0 and beat_idx == 0 else 0.0

    return {
        "value": float(beat_phase),
        "beat": float(beat_phase),
        "bar": float(bar_phase),
        "trigger": float(trigger),
        "downbeat": float(downbeat),
        "beat_index": float(beat_idx),
        "bar_count": float(bar_count),
    }
