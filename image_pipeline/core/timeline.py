"""Timeline — structured animation clock for the node graph.

Provides a single source of truth for animation timing across all nodes.
Injected into run_params["_timeline"] by GraphExecutor on every frame.

Usage in a method:
    tl = params.get("_timeline")
    if tl is not None:
        t = tl.t          # normalized [0, 1]
        phase = tl.phase  # [0, 2π) for cyclic methods
        speed = tl.speed  # speed multiplier (default 1.0)
        # Adjust timestep: dt = base_dt * speed
"""

from __future__ import annotations
import math
from dataclasses import dataclass, field
from typing import Any

from .easing import apply_easing, lerp_dict


@dataclass
class Timeline:
    """Structured animation clock for one frame of a node graph render.

    Every node in the graph receives the same Timeline for a given frame,
    then applies its own per-node offset (start_frame / end_frame) to remap
    ``t`` and ``phase``.
    """

    # ── Global frame counters ────────────────────────────────────────
    global_frame: int = 0       # absolute frame number in the render
    total_frames: int = 1       # total frames in this render

    # ── Normalised position ──────────────────────────────────────────
    t: float = 0.0              # [0, 1] — normalised position
    phase: float = 0.0          # [0, 2π) — cyclic phase (for backward compat with "time")

    # ── Output config ────────────────────────────────────────────────
    fps: int = 24               # output frames per second
    speed: float = 1.0          # speed multiplier (methods opt in by reading this)

    # ── Subdivision (per-node substeps) ──────────────────────────────
    substep: int = 0            # which substep within this frame (0 = first)
    total_substeps: int = 1     # total substeps per output frame (1 = no subdivision)

    # ── Per-node timing window (set by GraphExecutor per node) ───────
    start_frame: int = 0        # first frame this node animates
    end_frame: int = 0          # last frame this node animates (exclusive)

    # ── Derived helpers ──────────────────────────────────────────────

    @property
    def progress(self) -> float:
        """Normalised progress within the per-node window [start_frame, end_frame).

        Returns 0.0 before start_frame, 1.0 after end_frame.
        """
        window = self.end_frame - self.start_frame
        if window <= 0:
            return 1.0
        pos = (self.global_frame - self.start_frame) / window
        return max(0.0, min(1.0, pos))

    @property
    def local_phase(self) -> float:
        """Cyclic phase within the per-node window, in [0, 2π)."""
        return self.progress * 2.0 * math.pi

    def to_dict(self) -> dict:
        """Serialise to a plain dict for injection into run_params."""
        return {
            "global_frame": self.global_frame,
            "total_frames": self.total_frames,
            "t": self.t,
            "phase": self.phase,
            "fps": self.fps,
            "speed": self.speed,
            "substep": self.substep,
            "total_substeps": self.total_substeps,
            "start_frame": self.start_frame,
            "end_frame": self.end_frame,
        }


# ── Keyframe data model ────────────────────────────────────────────────


@dataclass
class Keyframe:
    """A single keyframe on a node's animation track.

    Attributes
    ----------
    frame : int
        Absolute frame number this keyframe sits on.
    values : dict[str, Any]
        Param values at this keyframe. Keys are param names (e.g. "r", "gamma", "anim_mode").
        Values are the target value at this frame.
    easing : str
        Easing preset for the segment *after* this keyframe (toward the next one).
        One of: "linear", "ease", "ease-in", "ease-out", "ease-in-out", "step", "bounce", "elastic", "cubic-bezier".
    handle_in : tuple[float, float] | None
        For "cubic-bezier": (x1, y1) control point for the incoming segment.
    handle_out : tuple[float, float] | None
        For "cubic-bezier": (x2, y2) control point for the outgoing segment.
    """

    frame: int
    values: dict[str, Any] = field(default_factory=dict)
    easing: str = "linear"
    handle_in: tuple[float, float] | None = None
    handle_out: tuple[float, float] | None = None


@dataclass
class KeyframeTrack:
    """A collection of keyframes for one node, sorted by frame.

    Attributes
    ----------
    node_id : str
        The node this track belongs to.
    keyframes : list[Keyframe]
        Sorted list of keyframes (ascending frame).
    default_easing : str
        Easing to use when a keyframe doesn't specify one.
    """

    node_id: str
    keyframes: list[Keyframe] = field(default_factory=list)
    default_easing: str = "ease-in-out"

    def __post_init__(self):
        self.keyframes.sort(key=lambda kf: kf.frame)

    def evaluate(self, frame: int) -> dict[str, Any] | None:
        """Evaluate the track at a given frame.

        Returns the interpolated param values dict, or None if the track
        has no keyframes or the frame is before the first keyframe.

        Between keyframes: interpolate using the *next* keyframe's easing.
        Before first keyframe: return first keyframe's values (hold).
        After last keyframe: return last keyframe's values (hold).
        """
        if not self.keyframes:
            return None

        # Before first keyframe — hold
        if frame <= self.keyframes[0].frame:
            return dict(self.keyframes[0].values)

        # After last keyframe — hold
        if frame >= self.keyframes[-1].frame:
            return dict(self.keyframes[-1].values)

        # Find the segment containing this frame
        for i in range(len(self.keyframes) - 1):
            kf_a = self.keyframes[i]
            kf_b = self.keyframes[i + 1]
            if kf_a.frame <= frame < kf_b.frame:
                window = kf_b.frame - kf_a.frame
                if window <= 0:
                    return dict(kf_b.values)
                t = (frame - kf_a.frame) / window
                easing = kf_b.easing or self.default_easing
                t_eased = apply_easing(t, easing,
                                       handle_in=kf_b.handle_in,
                                       handle_out=kf_b.handle_out)
                return lerp_dict(kf_a.values, kf_b.values, t_eased)

        return None

    def to_dict(self) -> dict:
        """Serialise to a plain dict for API transport."""
        return {
            "node_id": self.node_id,
            "default_easing": self.default_easing,
            "keyframes": [
                {
                    "frame": kf.frame,
                    "values": dict(kf.values),
                    "easing": kf.easing,
                    "handle_in": list(kf.handle_in) if kf.handle_in else None,
                    "handle_out": list(kf.handle_out) if kf.handle_out else None,
                }
                for kf in self.keyframes
            ],
        }

    @classmethod
    def from_dict(cls, data: dict) -> KeyframeTrack:
        """Deserialise from a dict (API transport)."""
        return cls(
            node_id=data["node_id"],
            default_easing=data.get("default_easing", "ease-in-out"),
            keyframes=[
                Keyframe(
                    frame=kf["frame"],
                    values=kf.get("values", {}),
                    easing=kf.get("easing", "linear"),
                    handle_in=tuple(kf["handle_in"]) if kf.get("handle_in") else None,
                    handle_out=tuple(kf["handle_out"]) if kf.get("handle_out") else None,
                )
                for kf in data.get("keyframes", [])
            ],
        )


def make_timeline(
    global_frame: int,
    total_frames: int,
    fps: int = 24,
    speed: float = 1.0,
    substep: int = 0,
    total_substeps: int = 1,
    start_frame: int = 0,
    end_frame: int | None = None,
) -> Timeline:
    """Factory: create a Timeline for a given frame.

    Parameters
    ----------
    global_frame : int
        Absolute frame number (0-indexed).
    total_frames : int
        Total frames in the render.
    fps : int
        Output frames per second.
    speed : float
        Speed multiplier (default 1.0). Methods opt in by reading this.
    substep : int
        Which substep within this frame (0 = first).
    total_substeps : int
        Total substeps per output frame (1 = no subdivision).
    start_frame : int
        Per-node start frame (default 0).
    end_frame : int | None
        Per-node end frame (defaults to total_frames).

    Returns
    -------
    Timeline
    """
    if end_frame is None:
        end_frame = total_frames

    # Normalised t across the *global* timeline
    if total_frames > 1:
        t = global_frame / (total_frames - 1)
    else:
        t = 0.0
    t = max(0.0, min(1.0, t))

    phase = t * 2.0 * math.pi

    return Timeline(
        global_frame=global_frame,
        total_frames=total_frames,
        t=t,
        phase=phase,
        fps=fps,
        speed=speed,
        substep=substep,
        total_substeps=total_substeps,
        start_frame=start_frame,
        end_frame=end_frame,
    )
