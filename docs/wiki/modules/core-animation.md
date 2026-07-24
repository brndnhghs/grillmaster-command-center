# Module: `core/animation.py`

## Purpose
Animation framework for turning a method's incremental generation into an MP4 clip. Supports two capture strategies: explicit internal frame capture (Architecture A) and per-frame time-based animation (Architecture B).

## Responsibilities
- Pipe frames to ffmpeg for MP4 encoding
- Provide `capture_frame()` helper for instrumented methods to submit intermediate frames
- Support per-thread job context for concurrent server requests
- Provide high-level `animate_method()` orchestrator
- Support keyframe-driven parameter ramps (`ParameterRamp`)

## Public Interfaces

### `frames_to_mp4(frame_gen, out_path, fps, quality, max_frames) -> Optional[Path]`
Pipe RGB float32 frames to ffmpeg as MP4 via rawvideo pipe. Returns `out_path` on success, `None` on failure.

### `capture_frame(method_id, arr)`
Called by instrumented methods to submit an intermediate frame. Routes through thread-local context (server path) or module-level globals (CLI path). Uses `get_method_id()` context to handle node renumbering. Raises `JobCancelled` if cancel event is set.

### `get_frames(method_id) -> list[np.ndarray]`
Retrieve captured frames and clear slot. Mirrors the write path: thread-local for server, module-level for CLI.

### `animate_method(meta, out_dir, seed, fps, duration, out_name, user_params) -> Optional[Path]`
High-level orchestrator:
- If method has `_timeline` or `anim_mode` params → time-based animation (per-frame calls with evolving time)
- Otherwise → enable `capture_frame()`, run method once, collect frames
- Falls back to tween animation if no natural frames and no time param

### `ParameterRamp` class
Linear parameter ramp across frames. `at(t)` returns interpolated value.

### Thread-local context functions
| Function | Purpose |
|----------|---------|
| `set_job_context(on_frame, cancel_event)` | Install per-thread context for server path |
| `clear_job_context()` | Remove per-thread context |
| `enable_frame_capture(method_id)` | Signal CLI path to capture frames |
| `disable_frame_capture()` | Disable CLI frame capture |

### `JobCancelled` exception
Raised inside a generation thread to abort cleanly.

## Dependencies
- `subprocess`, `tempfile`, `threading` (stdlib)
- `numpy`, PIL (`Image`)
- `core/utils.py` — `mn`, `W`, `H`
- `core/timeline.py` — `make_timeline`

## Consumers
- `pipeline.py` — CLI animation pipeline
- `server.py` — render sequence endpoint
- Methods with `capture_frame()` calls (Architecture A sims)

## Performance
- ffmpeg rawvideo pipe: avoids intermediate file I/O
- Frame resize to canvas dimensions if needed
- `max_frames=600` cap prevents runaway encoding
- Thread-local context avoids global state in concurrent server requests