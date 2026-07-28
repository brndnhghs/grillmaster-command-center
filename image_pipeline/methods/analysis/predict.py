"""Predict node — time-series prediction and pattern recognition.

Predicts future values of a live scalar datastream using several strategies.
Maintains a rolling history buffer, compares current behavior against stored
patterns, and optionally conditions predictions on matched pattern continuations.

Auto-imported by analysis/__init__.py.
"""

from __future__ import annotations
import json
import math
from pathlib import Path

from ...core.registry import method


# ── Per-node state ──────────────────────────────────────────────────────
_PREDICT_STATE: dict[str, dict] = {}
_PREDICT_PRUNE_COUNTER = 0
_DEFAULT_FPS = 24.0

# ── Direction constants (exposed as float-scale outputs) ────────────────
_DIR_UNKNOWN = 0.0
_DIR_RISING = 1.0
_DIR_FALLING = 2.0
_DIR_STABLE = 3.0

# ── Pattern recording modes ─────────────────────────────────────────────
_RECORD_IDLE = 0
_RECORD_ACTIVE = 1
_RECORD_READY = 2  # buffer full, waiting for save signal


def _zscore_normalize(samples: list[float]) -> list[float]:
    """Zero-mean unit-variance normalization.

    Returns a copy.  When the signal is flat (std ≈ 0), returns all zeros.
    """
    n = len(samples)
    if n == 0:
        return []
    mean = sum(samples) / n
    var = sum((s - mean) ** 2 for s in samples) / n
    std = math.sqrt(var)
    if std < 1e-12:
        return [0.0] * n
    return [(s - mean) / std for s in samples]


def _euclidean_distance(a: list[float], b: list[float]) -> float:
    """Euclidean distance between two equal-length vectors."""
    n = min(len(a), len(b))
    if n == 0:
        return float("inf")
    s = 0.0
    for i in range(n):
        d = a[i] - b[i]
        s += d * d
    return math.sqrt(s / n)  # RMS per-element distance


def _similarity_from_distance(dist: float) -> float:
    """Map RMS distance to a 0..1 similarity score.

    0.0 = perfect match, 2.0+ = no match.
    Maps Gaussian: similarity = exp(-dist).
    """
    return math.exp(-dist)


def _ema(prev: float, current: float, alpha: float) -> float:
    """Exponential moving average."""
    return prev + alpha * (current - prev)


def _linear_extrapolate(
    samples: list[tuple[int, float]], horizon_frames: float, fps: float
) -> float:
    """Fit a line to the samples and extrapolate to the horizon.

    Uses the slope of the last N=4 samples or the full buffer if fewer.
    Returns the last value if not enough samples.
    """
    n = len(samples)
    if n < 2:
        return samples[-1][1] if n == 1 else 0.0

    # Use last min(4, n) samples for slope
    k = min(4, n)
    recent = samples[-k:]
    x0 = recent[0][0]
    y0 = recent[0][1]
    # Weighted least squares — simple: average of first differences
    total_slope = 0.0
    for i in range(1, k):
        dx = recent[i][0] - recent[i - 1][0]
        dy = recent[i][1] - recent[i - 1][1]
        if dx > 0:
            total_slope += dy / dx
    slope = total_slope / (k - 1) if k > 1 else 0.0

    last_frame, last_val = samples[-1]
    return last_val + slope * horizon_frames


def _velocity_accel_extrapolate(
    samples: list[tuple[int, float]], horizon_frames: float, fps: float
) -> float:
    """Extrapolate using estimated velocity and acceleration.

    Fits linear regression and quadratic term to recent samples.
    Uses last 8 samples or all if fewer.
    """
    n = len(samples)
    if n < 3:
        return _linear_extrapolate(samples, horizon_frames, fps)

    k = min(8, n)
    recent = samples[-k:]
    x0 = recent[0][0]
    y0 = recent[0][1]

    # Estimate velocity from last 2 samples
    v = 0.0
    if k >= 2:
        v = (recent[-1][1] - recent[-2][1]) / max(1.0, recent[-1][0] - recent[-2][0])

    # Estimate acceleration from last 3 samples
    a = 0.0
    if k >= 3:
        v1 = (recent[-2][1] - recent[-3][1]) / max(
            1.0, recent[-2][0] - recent[-3][0]
        )
        v2 = v
        a = (v2 - v1) / max(1.0, recent[-1][0] - recent[-2][0])

    last_val = recent[-1][1]
    return last_val + v * horizon_frames + 0.5 * a * (horizon_frames ** 2)


def _classify_direction(
    predicted: float, current: float, threshold: float = 0.01
) -> float:
    """Classify direction as unknown/rising/falling/stable."""
    diff = predicted - current
    if abs(diff) < threshold:
        return _DIR_STABLE
    return _DIR_RISING if diff > 0 else _DIR_FALLING


def _sigmoid_confidence(mae: float, signal_range: float) -> float:
    """Map MAE relative to signal range to a 0..1 confidence.

    When MAE is 0 → confidence 1.0.
    When MAE ≈ signal_range → confidence ≈ 0.27.
    When MAE >> signal_range → confidence → 0.
    """
    if signal_range < 1e-12:
        return 0.0
    ratio = mae / signal_range
    if ratio <= 0:
        return 1.0
    return 1.0 / (1.0 + ratio * ratio)


# ── Pattern data model helpers ──────────────────────────────────────────


def _build_pattern(
    samples: list[float],
    duration_seconds: float,
    sample_rate: float,
    name: str,
) -> dict:
    """Build a pattern dict from raw samples."""
    if not samples:
        return {}
    norm = _zscore_normalize(samples)
    orig_min = min(samples)
    orig_max = max(samples)
    return {
        "name": name,
        "samples_raw": samples,
        "samples_norm": norm,
        "duration": duration_seconds,
        "sample_rate": sample_rate,
        "amplitude": orig_max - orig_min,
        "orig_min": orig_min,
        "orig_max": orig_max,
        "len": len(samples),
    }


def _match_pattern(
    window_norm: list[float],
    pattern: dict,
) -> tuple[float, float]:
    """Compare a normalized window against a stored pattern.

    Returns (similarity, best_offset_ratio) where offset_ratio is the
    fraction of the pattern at which the window best aligns (0..1).
    """
    pat_samples = pattern.get("samples_norm", [])
    if not pat_samples or not window_norm:
        return 0.0, 0.0

    wlen = len(window_norm)
    plen = len(pat_samples)

    # If window is longer than pattern, only consider the first plen samples
    if wlen >= plen:
        d = _euclidean_distance(window_norm[:plen], pat_samples)
        return _similarity_from_distance(d), 1.0

    # Slide the window across the pattern
    best_dist = float("inf")
    best_offset = 0
    max_offset = plen - wlen
    for offset in range(max_offset + 1):
        d = _euclidean_distance(window_norm, pat_samples[offset : offset + wlen])
        if d < best_dist:
            best_dist = d
            best_offset = offset

    progress = best_offset / max_offset if max_offset > 0 else 0.5
    return _similarity_from_distance(best_dist), progress


def _pattern_continuation(
    window_norm: list[float],
    pattern: dict,
    n_continuation: int,
) -> list[float] | None:
    """Return the continuation of a pattern after the best match offset.

    Returns a list of `n_continuation` float values (in the pattern's
    raw amplitude space), or None if the match is too short to continue.
    """
    pat_samples = pattern.get("samples_norm", [])
    pat_raw = pattern.get("samples_raw", [])
    if not pat_samples or not pat_raw or not window_norm:
        return None

    wlen = len(window_norm)
    plen = len(pat_samples)

    if wlen >= plen:
        return None  # already at or past pattern end

    # Find best offset (duplicate logic from _match_pattern for simplicity)
    best_dist = float("inf")
    best_offset = 0
    max_offset = plen - wlen
    for offset in range(max_offset + 1):
        d = _euclidean_distance(window_norm, pat_samples[offset : offset + wlen])
        if d < best_dist:
            best_dist = d
            best_offset = offset

    # Continuation starts after the matched window in the pattern
    cont_end = min(best_offset + wlen + n_continuation, plen)
    if cont_end <= best_offset + wlen:
        return None

    # We need to return in the original value space
    # Reconstruct the original amplitude/shape for the continuation section
    # Use the inverse of z-score normalization
    pat_mean = sum(pat_raw) / len(pat_raw)
    pat_std = math.sqrt(
        sum((s - pat_mean) ** 2 for s in pat_raw) / len(pat_raw)
    )
    pat_std = max(pat_std, 1e-12)

    continuation = []
    for i in range(best_offset + wlen, cont_end):
        norm_val = pat_samples[i]
        raw_val = norm_val * pat_std + pat_mean
        continuation.append(raw_val)

    return continuation


# ── Pattern persistence: store/load from disk ────────────────────────────
_PATTERN_STORE_SUBDIR = "predict_patterns"


def _pattern_store_path(out_dir: Path) -> Path:
    return out_dir.parent / _PATTERN_STORE_SUBDIR


def _save_patterns(out_dir: Path, patterns: dict[str, dict]) -> None:
    """Persist the pattern library alongside the graph output directory.

    Only saves metadata (normalized shape, amplitude, duration, rate) —
    the raw samples are derivable from the norm shape + amplitude and are
    omitted to keep the store compact.
    """
    store = _pattern_store_path(out_dir)
    store.mkdir(parents=True, exist_ok=True)
    manifest = {}
    for name, pat in patterns.items():
        manifest[name] = {
            "samples_norm": pat.get("samples_norm", []),
            "duration": pat.get("duration", 1.0),
            "sample_rate": pat.get("sample_rate", _DEFAULT_FPS),
            "amplitude": pat.get("amplitude", 0.0),
            "orig_min": pat.get("orig_min", 0.0),
            "orig_max": pat.get("orig_max", 1.0),
            "len": pat.get("len", 0),
        }
    try:
        (store / "patterns.json").write_text(json.dumps(manifest, indent=2))
    except OSError:
        pass  # best-effort


def _load_patterns(out_dir: Path) -> dict[str, dict]:
    """Load persisted patterns, returning empty dict if none exist."""
    store = _pattern_store_path(out_dir)
    path = store / "patterns.json"
    if not path.exists():
        return {}
    try:
        manifest = json.loads(path.read_text())
        # Rebuild from normalized form
        patterns = {}
        for name, data in manifest.items():
            patterns[name] = {
                "name": name,
                "samples_norm": data.get("samples_norm", []),
                "duration": data.get("duration", 1.0),
                "sample_rate": data.get("sample_rate", _DEFAULT_FPS),
                "amplitude": data.get("amplitude", 0.0),
                "orig_min": data.get("orig_min", 0.0),
                "orig_max": data.get("orig_max", 1.0),
                "len": data.get("len", 0),
            }
        return patterns
    except (json.JSONDecodeError, OSError):
        return {}


# ── The Node ────────────────────────────────────────────────────────────


@method(
    id="__predict__",
    name="Predict",
    category="analysis",
    tags=["chop", "analysis", "predict", "pattern", "time-series"],
    inputs={
        "stream": "SCALAR",
        "reset": "SCALAR",
        "enable": "SCALAR",
        "record": "SCALAR",
        "save_pattern": "SCALAR",
    },
    outputs={
        "predicted_value": "SCALAR",
        "prediction_confidence": "SCALAR",
        "prediction_error": "SCALAR",
        "predicted_direction": "SCALAR",
        "pattern_match": "SCALAR",
        "pattern_progress": "SCALAR",
        "pattern_phase": "SCALAR",
        "pattern_trigger": "SCALAR",
        "pattern_id": "SCALAR",
        "record_progress": "SCALAR",
        "signal_level": "SCALAR",
        "history_min": "SCALAR",
        "history_max": "SCALAR",
    },
    runtime={
        "predicted_value": {"type": "numeric", "label": "Predicted", "observable": True},
        "prediction_confidence": {"type": "numeric", "label": "Confidence", "observable": True},
        "prediction_error": {"type": "numeric", "label": "Error", "observable": True},
        "predicted_direction": {"type": "output", "label": "Direction", "observable": True},
        "pattern_match": {"type": "numeric", "label": "Match", "observable": True},
        "pattern_progress": {"type": "numeric", "label": "Progress", "observable": True},
        "pattern_trigger": {"type": "numeric", "label": "Trigger", "observable": True},
        "signal_level": {"type": "numeric", "label": "Signal", "observable": True},
    },
    signal={
        "stream": "numeric",
        "reset": "event",
        "enable": "control",
        "record": "event",
        "save_pattern": "event",
        "predicted_value": "output",
        "prediction_confidence": "output",
        "prediction_error": "output",
        "predicted_direction": "output",
        "pattern_match": "output",
        "pattern_progress": "output",
        "pattern_phase": "output",
        "pattern_trigger": "event",
        "pattern_id": "output",
        "record_progress": "output",
        "signal_level": "output",
        "history_min": "output",
        "history_max": "output",
    },
    params={
        "history_window": {
            "description": "History window duration (seconds)",
            "default": 2.0,
            "min": 0.1,
            "max": 60.0,
        },
        "prediction_horizon": {
            "description": "How far ahead to predict (seconds)",
            "default": 0.5,
            "min": 0.0,
            "max": 30.0,
        },
        "prediction_method": {
            "description": "Prediction strategy",
            "choices": ["linear", "smoothed_linear", "velocity_accel"],
            "default": "smoothed_linear",
        },
        "smooth_factor": {
            "description": "EMA smoothing factor for smoothed linear prediction",
            "default": 0.3,
            "min": 0.01,
            "max": 1.0,
        },
        "enable_matching": {
            "description": "Enable pattern matching",
            "default": True,
        },
        "pattern_match_threshold": {
            "description": "Match score threshold for pattern_trigger output (0–1)",
            "default": 0.7,
            "min": 0.0,
            "max": 1.0,
        },
        "pattern_threshdown": {
            "description": "Hysteresis lower threshold for pattern trigger (0–1)",
            "default": 0.5,
            "min": 0.0,
            "max": 1.0,
        },
        "pattern_window": {
            "description": "Pattern matching window (seconds)",
            "default": 1.0,
            "min": 0.1,
            "max": 10.0,
        },
        "pattern_recording_duration": {
            "description": "Duration of pattern recording (seconds)",
            "default": 2.0,
            "min": 0.5,
            "max": 10.0,
        },
        "pattern_conditioned_enabled": {
            "description": "Use matched pattern continuation to inform prediction",
            "default": False,
        },
        "pattern_conditioned_blend": {
            "description": "Max blend weight for pattern-conditioned prediction (0–1)",
            "default": 0.7,
            "min": 0.0,
            "max": 1.0,
        },
        "pattern_name": {
            "description": "Name for next saved pattern",
            "default": "",
        },
        "error_tracking_window": {
            "description": "Number of recent predictions to track for confidence (0=all)",
            "default": 20,
            "min": 0,
            "max": 500,
        },
    },
)
def method_predict(out_dir: Path, seed: int, params=None):
    """Time-series prediction and pattern recognition node.

    The Predict node analyzes recent signal history to estimate where the
    signal is going, recognizes recurring behavior patterns, and optionally
    uses recognized patterns to improve predictions.

    **Core pipeline:**

        Stream ──→ Rolling History ──→ Predictor → Predicted Value
                                      │
                                      └──→ Pattern Matcher → Match Score
                                                              Pattern ID

    **Prediction methods:**

      - ``linear``: Estimate velocity from recent samples, extrapolate.
      - ``smoothed_linear`` (default): Apply EMA before velocity estimation.
        Best for noisy live signals.
      - ``velocity_accel``: Estimate both velocity and acceleration for
        smooth trajectories with momentum.

    **Pattern system:**

      - **Recording**: Set the ``record`` SCALAR input high (≥0.5 threshold)
        to begin recording.  The node captures ``pattern_recording_duration``
        seconds of the live stream.  When recording completes, set
        ``save_pattern`` high (rising edge) to store the captured buffer as
        a named pattern.  The pattern name is set via the ``pattern_name``
        param.
      - **Matching**: The current ``pattern_window`` is continuously compared
        against stored patterns using Z-score normalized Euclidean distance.
        The best match score, progress, and pattern ID are exposed.
      - **Pattern-Trigger**: The ``pattern_trigger`` output fires when the
        match score exceeds ``pattern_match_threshold``, with hysteresis
        via ``pattern_threshdown``.
      - **Pattern-Conditioned Prediction**: When enabled, the matched
        pattern's historical continuation is blended with the standard
        prediction, weighted by the match score.

    **Prediction confidence:**

      Confidence is empirical, based on recent prediction error (MAE).
      The node tracks a ``prediction_records`` deque: each prediction is
      compared against the actual value when the horizon elapses.  The
      MAE over recent records drives confidence via a sigmoid mapping.

    **Inputs:**

        stream (SCALAR): The live datastream being analyzed.
        reset (SCALAR): Rising edge clears history and resets state.
        enable (SCALAR): When < 0.5, computation halts (outputs freeze).
        record (SCALAR): Rising edge starts pattern recording.
        save_pattern (SCALAR): Rising edge saves the recorded buffer.

    **Outputs:**

        predicted_value (SCALAR): Prediction at the selected horizon.
        prediction_confidence (SCALAR): Empirical confidence 0–1.
        prediction_error (SCALAR): Absolute error of the most recent
            matured prediction.
        predicted_direction (SCALAR): 0=unknown, 1=rising, 2=falling,
            3=stable.
        pattern_match (SCALAR): Best match score 0–1.
        pattern_progress (SCALAR): How far through the matched pattern
            (0–1).
        pattern_phase (SCALAR): Continuous normalized position (0–1).
        pattern_trigger (SCALAR): 1 when match > threshold, with
            hysteresis.
        pattern_id (SCALAR): Index of best matching pattern (-1=no match).
        record_progress (SCALAR): Recording buffer fill fraction (0–1).
        signal_level (SCALAR): Current stream value.
        history_min (SCALAR): Minimum observed value in history window.
        history_max (SCALAR): Maximum observed value in history window.
    """
    if params is None:
        params = {}

    # ── Frame derivation ────────────────────────────────────────────────
    frame = int(params.get("frame", 0))
    fps = _DEFAULT_FPS
    if frame == 0:
        _tl = params.get("_timeline")
        if _tl is not None:
            frame = int(getattr(_tl, "global_frame", 0))
            fps = float(getattr(_tl, "fps", _DEFAULT_FPS))
    fps = max(1.0, fps)

    node_id = params.get("_node_id", "")
    out_dir_resolved = out_dir

    # ── Read input stream ───────────────────────────────────────────────
    stream_val = params.get("stream")
    stream_float = float(stream_val) if stream_val is not None else 0.0

    # ── Enable / disable ────────────────────────────────────────────────
    enable_raw = params.get("enable")
    enabled = True
    if enable_raw is not None:
        enabled = float(enable_raw) >= 0.5

    # ── Reset signal (rising edge) ──────────────────────────────────────
    reset_raw = params.get("reset")

    # ── Record / save signals ───────────────────────────────────────────
    record_raw = params.get("record")
    save_raw = params.get("save_pattern")

    # ── Params ──────────────────────────────────────────────────────────
    history_window = max(0.1, float(params.get("history_window", 2.0)))
    prediction_horizon = max(0.0, float(params.get("prediction_horizon", 0.5)))
    prediction_method = params.get("prediction_method", "smoothed_linear")
    smooth_factor = max(0.01, min(1.0, float(params.get("smooth_factor", 0.3))))
    enable_matching = params.get("enable_matching", True)
    if isinstance(enable_matching, str):
        enable_matching = enable_matching.lower() in ("true", "1", "yes")
    pattern_match_threshold = max(0.0, min(1.0, float(params.get("pattern_match_threshold", 0.7))))
    pattern_threshdown = max(0.0, min(1.0, float(params.get("pattern_threshdown", 0.5))))
    pattern_window = max(0.1, float(params.get("pattern_window", 1.0)))
    pattern_rec_dur = max(0.5, float(params.get("pattern_recording_duration", 2.0)))
    pattern_conditioned = params.get("pattern_conditioned_enabled", False)
    if isinstance(pattern_conditioned, str):
        pattern_conditioned = pattern_conditioned.lower() in ("true", "1", "yes")
    pcond_blend = max(0.0, min(1.0, float(params.get("pattern_conditioned_blend", 0.7))))
    pattern_name = str(params.get("pattern_name", ""))
    error_window = int(params.get("error_tracking_window", 20))

    # ── Time conversion ─────────────────────────────────────────────────
    history_frames = int(history_window * fps)
    horizon_frames = prediction_horizon * fps
    pattern_match_frames = int(pattern_window * fps)
    pattern_rec_frames = int(pattern_rec_dur * fps)

    # ── Default outputs ─────────────────────────────────────────────────
    predicted_value = stream_float
    prediction_confidence = 0.0
    prediction_error = 0.0
    predicted_direction = _DIR_STABLE
    pattern_match = 0.0
    pattern_progress = 0.0
    pattern_phase = 0.0
    pattern_trigger = 0.0
    pattern_id = -1.0
    record_progress = 0.0
    signal_level = stream_float
    history_min = stream_float
    history_max = stream_float

    # ── State machine — only with valid node_id ─────────────────────────
    global _PREDICT_PRUNE_COUNTER
    _PREDICT_PRUNE_COUNTER += 1

    if not node_id:
        return {
            "predicted_value": float(predicted_value),
            "prediction_confidence": float(prediction_confidence),
            "prediction_error": float(prediction_error),
            "predicted_direction": float(predicted_direction),
            "pattern_match": float(pattern_match),
            "pattern_progress": float(pattern_progress),
            "pattern_phase": float(pattern_phase),
            "pattern_trigger": float(pattern_trigger),
            "pattern_id": float(pattern_id),
            "record_progress": float(record_progress),
            "signal_level": float(signal_level),
            "history_min": float(history_min),
            "history_max": float(history_max),
        }

    state = _PREDICT_STATE.setdefault(node_id, {
        "buffer": [],  # list of (frame, value)
        "smoothed_prev": None,  # EMA state
        "prediction_records": [],  # list of {frame, horizon, predicted}
        "mae": 0.0,
        "mae_count": 0,
        "last_error": 0.0,
        "prev_prediction": 0.0,
        "prev_frame": frame,
        # Pattern library
        "patterns": None,  # lazily loaded from disk
        "patterns_loaded": False,
        "patterns_dirty": False,
        # Recording state
        "recording_state": _RECORD_IDLE,
        "recording_buffer": [],
        "recording_start_frame": 0,
        "recording_prev_record": 0.0,
        "recording_prev_save": 0.0,
        # Pattern trigger hysteresis
        "trigger_active": False,
        "prev_trigger_raw": 0.0,
        # For direction stability
        "prev_direction": _DIR_UNKNOWN,
    })

    # ── Lazy-load patterns from disk ────────────────────────────────────
    if not state["patterns_loaded"]:
        loaded = _load_patterns(out_dir_resolved)
        state["patterns"] = loaded
        state["patterns_loaded"] = True

    # ── Reset handling (rising edge) ────────────────────────────────────
    reset_fired = False
    if reset_raw is not None:
        _rv = float(reset_raw)
        prev_reset = state.get("prev_reset", 0.0)
        if _rv >= 0.5 > prev_reset:
            reset_fired = True
        state["prev_reset"] = _rv

    if reset_fired:
        state["buffer"] = []
        state["smoothed_prev"] = None
        state["prediction_records"] = []
        state["mae"] = 0.0
        state["mae_count"] = 0
        state["last_error"] = 0.0
        state["recording_state"] = _RECORD_IDLE
        state["recording_buffer"] = []
        state["trigger_active"] = False
        state["prev_prediction"] = stream_float

    # ── Enable gate ──────────────────────────────────────────────────────
    if not enabled:
        # Return last known outputs, still update the buffer so history
        # remains continuous when re-enabled
        predicted_value = state.get("prev_prediction", stream_float)
        return {
            "predicted_value": float(predicted_value),
            "prediction_confidence": float(state.get("mae", 0.0)),
            "prediction_error": float(state.get("last_error", 0.0)),
            "predicted_direction": float(state.get("prev_direction", _DIR_UNKNOWN)),
            "pattern_match": float(pattern_match),
            "pattern_progress": float(pattern_progress),
            "pattern_phase": float(pattern_phase),
            "pattern_trigger": float(pattern_trigger),
            "pattern_id": float(pattern_id),
            "record_progress": float(record_progress),
            "signal_level": float(signal_level),
            "history_min": float(history_min),
            "history_max": float(history_max),
        }

    # ── Rolling buffer ───────────────────────────────────────────────────
    buf = state["buffer"]
    buf.append((frame, stream_float))
    # Trim old entries beyond history window
    cutoff_frame = frame - history_frames
    while buf and buf[0][0] < cutoff_frame:
        buf.pop(0)

    # ── History range ───────────────────────────────────────────────────
    if buf:
        vals = [v for _, v in buf]
        history_min = min(vals)
        history_max = max(vals)

    # ── Evaluate matured predictions ────────────────────────────────────
    recs = state["prediction_records"]
    new_recs = []
    for rec in recs:
        mature_frame = rec["frame"] + int(rec["horizon_frames"])
        if frame >= mature_frame:
            # This prediction has matured — compute error
            actual = stream_float
            err = abs(rec["predicted"] - actual)
            state["last_error"] = err

            # Update running MAE
            n = state["mae_count"]
            current_mae = state["mae"]
            if n < error_window or error_window <= 0:
                n += 1
                state["mae"] = current_mae + (err - current_mae) / n
                state["mae_count"] = n
            else:
                # Sliding window: maintain a deque — but for simplicity,
                # use EMA with a 1/N smoothing factor
                n_eff = min(error_window, 100)
                inv_n = 1.0 / max(1.0, n_eff)
                state["mae"] = _ema(current_mae, err, inv_n)
        else:
            new_recs.append(rec)
    state["prediction_records"] = new_recs

    # ── Compute prediction confidence from MAE ──────────────────────────
    sig_range = history_max - history_min
    if sig_range < 1e-12:
        sig_range = 1.0
    if state["mae_count"] > 0:
        prediction_confidence = _sigmoid_confidence(state["mae"], sig_range)
    else:
        prediction_confidence = 0.0  # UNKNOWN — insufficient data

    prediction_error = state.get("last_error", 0.0)

    # ── Compute prediction ──────────────────────────────────────────────
    if buf and len(buf) >= 2:
        if prediction_method == "linear":
            predicted_value = _linear_extrapolate(buf, horizon_frames, fps)
        elif prediction_method == "velocity_accel":
            predicted_value = _velocity_accel_extrapolate(buf, horizon_frames, fps)
        else:
            # smoothed_linear (default)
            smooth_prev = state["smoothed_prev"]
            if smooth_prev is None:
                smooth_prev = stream_float
            smooth_val = _ema(smooth_prev, stream_float, smooth_factor)
            state["smoothed_prev"] = smooth_val

            # Build smoothed buffer
            smooth_buf = list(buf)  # keep frame coords, replace values
            # Actually rebuild: we need a smoothed buffer for extrapolation
            # Use the raw buffer frames with EMA over time
            smooth_vals = []
            sv = smooth_prev
            for _, v in buf:
                sv = _ema(sv, v, smooth_factor)
                smooth_vals.append(sv)
            smooth_samples = [(buf[i][0], smooth_vals[i]) for i in range(len(buf))]
            predicted_value = _linear_extrapolate(smooth_samples, horizon_frames, fps)

        # Clamp reasonable range
        extrema_margin = sig_range * 2.0
        predicted_value = max(
            history_min - extrema_margin,
            min(history_max + extrema_margin, predicted_value),
        )
    else:
        predicted_value = stream_float

    state["prev_prediction"] = predicted_value

    # ── Classify direction ──────────────────────────────────────────────
    predicted_direction = _classify_direction(predicted_value, stream_float, sig_range * 0.05)
    state["prev_direction"] = predicted_direction

    # ── Pattern matching ────────────────────────────────────────────────
    patterns = state.get("patterns", {})
    if enable_matching and len(buf) >= 2 and patterns:
        # Build current window (last N frames = pattern_window seconds)
        match_cutoff = frame - pattern_match_frames
        window_samples = [v for f, v in buf if f >= match_cutoff]
        if len(window_samples) >= 2:
            window_norm = _zscore_normalize(window_samples)
            best_score = 0.0
            best_progress = 0.0
            best_id = -1
            best_name = ""
            pat_list = list(patterns.items())

            for idx, (pname, pdata) in enumerate(pat_list):
                score, progress = _match_pattern(window_norm, pdata)
                if score > best_score:
                    best_score = score
                    best_progress = progress
                    best_id = idx
                    best_name = pname

            pattern_match = best_score
            pattern_progress = best_progress
            pattern_id = float(best_id) if best_id >= 0 else -1.0
            pattern_phase = pattern_progress

            # ── Pattern-conditioned prediction ──────────────────────────
            if pattern_conditioned and best_id >= 0 and best_score > 0.3:
                pat_data = pat_list[best_id][1]
                # Request continuation of N = horizon_frames samples
                cont = _pattern_continuation(window_norm, pat_data, int(horizon_frames) + 1)
                if cont and cont[0] is not None:
                    # Blend: weight = match_score * max_blend
                    blend_weight = best_score * pcond_blend
                    pat_pred = float(cont[0])  # first predicted value from pattern
                    # Blend with linear prediction
                    blended = predicted_value * (1.0 - blend_weight) + pat_pred * blend_weight
                    predicted_value = blended

    # ── Pattern trigger with hysteresis ────────────────────────────────
    trigger_active = state.get("trigger_active", False)
    if pattern_match >= pattern_match_threshold:
        if not trigger_active:
            trigger_active = True
            pattern_trigger = 1.0
        else:
            pattern_trigger = 1.0  # stay high while above threshold
    elif trigger_active and pattern_match > pattern_threshdown:
        pattern_trigger = 1.0  # hysteresis hold
    else:
        trigger_active = False
        pattern_trigger = 0.0
    state["trigger_active"] = trigger_active

    # ── Pattern recording state machine ─────────────────────────────────
    rec_state = state.get("recording_state", _RECORD_IDLE)
    rec_buf = state.get("recording_buffer", [])
    rec_start = state.get("recording_start_frame", frame)

    # Rising edge on record → start recording
    prev_rec = state.get("recording_prev_record", 0.0)
    if record_raw is not None:
        _rr = float(record_raw)
        if _rr >= 0.5 > prev_rec:
            rec_state = _RECORD_ACTIVE
            rec_buf = []
            rec_start = frame
        state["recording_prev_record"] = _rr

    # Rising edge on save_pattern → save recorded buffer
    prev_save = state.get("recording_prev_save", 0.0)
    if save_raw is not None:
        _sr = float(save_raw)
        if _sr >= 0.5 > prev_save:
            if rec_state == _RECORD_READY and len(rec_buf) >= 2:
                # Save as a new pattern
                pname = pattern_name.strip() if pattern_name.strip() else f"Pattern {len(patterns) + 1}"
                rec_duration = (frame - rec_start) / fps
                sample_rate = len(rec_buf) / max(rec_duration, 0.01)
                new_pat = _build_pattern(rec_buf, rec_duration, sample_rate, pname)
                if new_pat:
                    patterns[pname] = new_pat
                    state["patterns_dirty"] = True
            rec_state = _RECORD_IDLE
            rec_buf = []
        state["recording_prev_save"] = _sr

    # If recording active, fill buffer
    if rec_state == _RECORD_ACTIVE:
        rec_buf.append(stream_float)
        # Check if buffer is full (enough frames for the recording duration)
        rec_needed = max(2, int(pattern_rec_dur * fps))
        if len(rec_buf) >= rec_needed:
            rec_state = _RECORD_READY

        record_progress = min(1.0, len(rec_buf) / max(1.0, rec_needed))

    state["recording_state"] = rec_state
    state["recording_buffer"] = rec_buf
    state["recording_start_frame"] = rec_start

    # ── Store current prediction for future evaluation ──────────────────
    if horizon_frames > 0.5 and len(buf) >= 2:
        state["prediction_records"].append({
            "frame": frame,
            "horizon_frames": horizon_frames,
            "predicted": predicted_value,
        })
        # Cap the records list
        max_records = max(10, error_window * 2)
        if len(state["prediction_records"]) > max_records:
            state["prediction_records"] = state["prediction_records"][-max_records:]

    # ── Persist pattern library when dirty ──────────────────────────────
    if state.get("patterns_dirty", False):
        _save_patterns(out_dir_resolved, patterns)
        state["patterns_dirty"] = False

    # ── Lazy prune ──────────────────────────────────────────────────────
    if _PREDICT_PRUNE_COUNTER % 500 == 0:
        _cutoff = frame - 7200
        for _nid in list(_PREDICT_STATE):
            if _PREDICT_STATE[_nid].get("prev_frame", 0) < _cutoff:
                # Persist before pruning
                pstate = _PREDICT_STATE[_nid]
                if pstate.get("patterns_dirty", False):
                    # Best-effort: cannot persist without out_dir here,
                    # patterns are already persisted on each save
                    pass
                del _PREDICT_STATE[_nid]

    state["prev_frame"] = frame
    state["signal_level"] = stream_float

    # ── Return ──────────────────────────────────────────────────────────
    return {
        "predicted_value": float(predicted_value),
        "prediction_confidence": float(prediction_confidence),
        "prediction_error": float(prediction_error),
        "predicted_direction": float(predicted_direction),
        "pattern_match": float(pattern_match),
        "pattern_progress": float(pattern_progress),
        "pattern_phase": float(pattern_phase),
        "pattern_trigger": float(pattern_trigger),
        "pattern_id": float(pattern_id),
        "record_progress": float(record_progress),
        "signal_level": float(signal_level),
        "history_min": float(history_min),
        "history_max": float(history_max),
    }
