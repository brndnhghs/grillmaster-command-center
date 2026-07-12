"""Live render telemetry — a process-wide board of what every render thread
is doing *right now*, plus a per-genome skip switch.

Renders run on a thread pool deep inside GraphExecutor.execute(), which is a
single blocking call per frame. When a node wedges mid-cook, nothing returns
to the render loop, so the between-frame timeout never fires and the whole
run appears hung with no output. This module fixes the visibility and the
control:

  * Each render thread reports its phase into the shared MONITOR (current
    frame, the node cooking right now, sim sub-frame, when the frame started).
  * A heartbeat thread (see evaluator.render_many) reads the board every
    ~second and emits a "still on frame N, cooking <node>, Xs elapsed" line —
    so a hang is obvious second-by-second instead of a silent stall.
  * MONITOR.request_skip(gid) sets a threading.Event that the executor's sim
    loop polls (via animation.capture_frame) and that the render loop checks
    between frames — the UI's skip button and the wedged-render watchdog both
    pull this lever.

Everything is a module-level singleton because renders and the skip endpoint
run in the same server process. Pure stdlib, no rendering here.
"""
from __future__ import annotations

import threading
import time


class RenderMonitor:
    """Thread-safe board of in-flight render status + skip events."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._status: dict[str, dict] = {}
        self._skip: dict[str, threading.Event] = {}

    # ── lifecycle (called by the render thread) ───────────────────────
    def begin(self, gid: str, total_frames: int, n_nodes: int) -> None:
        now = time.time()
        with self._lock:
            self._status[gid] = {
                "genome_id": gid, "phase": "render", "frame": 0,
                "total_frames": total_frames, "n_nodes": n_nodes,
                "node_id": None, "node_name": None, "node_method": None,
                "sim_frame": 0, "t0": now, "t_frame": now, "t_node": now,
                "done": False, "skip_requested": False,
            }
            self._skip.setdefault(gid, threading.Event())

    def frame_start(self, gid: str, frame: int) -> None:
        now = time.time()
        with self._lock:
            s = self._status.get(gid)
            if s is not None:
                s.update(frame=frame, t_frame=now, t_node=now,
                         node_id=None, node_name=None, node_method=None,
                         sim_frame=0)

    def node_cooking(self, gid: str, node_id: str, method_id: str,
                     name: str, sim_frame: int | None = None) -> None:
        now = time.time()
        with self._lock:
            s = self._status.get(gid)
            if s is None:
                return
            if sim_frame is not None:
                s["sim_frame"] = sim_frame
            else:
                # A new node started cooking — reset its per-node clock.
                s.update(node_id=node_id, node_method=method_id,
                         node_name=name, t_node=now, sim_frame=0)

    def finish(self, gid: str) -> None:
        with self._lock:
            s = self._status.get(gid)
            if s is not None:
                s["done"] = True

    def clear_all(self) -> None:
        with self._lock:
            self._status.clear()
            self._skip.clear()

    # ── reads (called by heartbeat / server) ──────────────────────────
    def snapshot(self, include_done: bool = False) -> dict[str, dict]:
        with self._lock:
            return {gid: dict(s) for gid, s in self._status.items()
                    if include_done or not s.get("done")}

    # ── skip control ──────────────────────────────────────────────────
    def skip_event(self, gid: str) -> threading.Event:
        with self._lock:
            return self._skip.setdefault(gid, threading.Event())

    def request_skip(self, gid: str) -> bool:
        """Signal the render thread to abort this genome. Returns False if the
        genome isn't (or is no longer) rendering."""
        with self._lock:
            ev = self._skip.setdefault(gid, threading.Event())
            s = self._status.get(gid)
            active = s is not None and not s.get("done")
            if s is not None:
                s["skip_requested"] = True
        ev.set()
        return active

    def is_skipped(self, gid: str) -> bool:
        with self._lock:
            ev = self._skip.get(gid)
        return bool(ev is not None and ev.is_set())


# Process-wide singleton shared by the render pool and the skip endpoint.
MONITOR = RenderMonitor()


def heartbeat_lines(snapshot: dict[str, dict], frame_hang_s: float,
                    now: float | None = None) -> list[str]:
    """One live status line per in-flight genome for the generation log."""
    now = now or time.time()
    out = []
    for gid, s in sorted(snapshot.items()):
        total = s.get("total_frames") or 0
        frame = s.get("frame", 0)
        on_frame = now - s.get("t_frame", now)
        elapsed = now - s.get("t0", now)
        node = s.get("node_name") or "—"
        method = s.get("node_method")
        sim = s.get("sim_frame") or 0
        loc = f"{node} [{method}]" if method else node
        sim_txt = f" · sim {sim}/{total}" if sim else ""
        flag = ""
        if s.get("skip_requested"):
            flag = " ⏹ skip requested"
        elif on_frame >= frame_hang_s:
            flag = f" ⚠ SLOW ({on_frame:.0f}s on this frame)"
        out.append(
            f"⏱ {gid} · frame {frame + 1}/{total} · cooking {loc}{sim_txt}"
            f" · {elapsed:.0f}s total{flag}")
    return out
