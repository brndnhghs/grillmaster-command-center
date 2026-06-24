from __future__ import annotations
import math
from collections import deque
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from ...core.registry import method
from ...core.utils import save, mn, seed_all, W, H, write_scalars
from ...core.animation import capture_frame

# Canvas constants
RING_RADIUS = min(W, H) // 2 - 50  # ~334 px for 768×512
CENTER_X, CENTER_Y = W // 2, H // 2
DARK_BG = (6, 6, 22)
INDICATOR_Y = H - 32
INDICATOR_H = 8

# ── Waveform type helper (scalar) ──
def _wav(phase: float, wt: str, pw: float) -> float:
    if wt == "sine":
        return math.sin(phase)
    elif wt == "square":
        return 1.0 if math.sin(phase) >= 0 else -1.0
    elif wt == "sawtooth":
        t = phase / (2.0 * math.pi)
        return 2.0 * (t - math.floor(t + 0.5))
    elif wt == "triangle":
        t = phase / (2.0 * math.pi)
        return 2.0 * abs(2.0 * (t - math.floor(t + 0.5))) - 1.0
    elif wt == "pulse":
        t = (phase % (2.0 * math.pi)) / (2.0 * math.pi)
        return 1.0 if t < pw else 0.0
    elif wt == "gaussian":
        sigma = pw * 2.0 * math.pi
        t_m = min(phase % (2.0 * math.pi),
                  (2.0 * math.pi - phase) % (2.0 * math.pi))
        return math.exp(-0.5 * (t_m / sigma) ** 2) * 2.0 - 1.0
    return math.sin(phase)


@method(id="89", name="Kuramoto Sync", category="simulations",
        tags=["oscillators", "emergence", "synchronization"],
        params={
            "n_oscillators": {"description": "number of oscillators", "min": 50, "max": 500, "default": 200},
            "coupling_max": {"description": "max coupling K", "min": 0.5, "max": 8.0, "default": 3.0},
            "K_start": {"description": "starting K (sweep from here to coupling_max)", "min": 0.0, "max": 8.0, "default": 0.0},
            "freq_std": {"description": "natural frequency spread", "min": 0.1, "max": 3.0, "default": 1.0},
            "dot_size": {"description": "oscillator dot radius (px)", "min": 1, "max": 6, "default": 3},
            "trail_length": {"description": "phase trail length (frames)", "min": 0, "max": 30, "default": 12},
            "viz": {"description": "visualization style", "choices": ["ring", "field", "pendulums", "breathing", "bouncing", "oscilloscope"], "default": "pendulums"},
            "wave_type": {"description": "waveform shape for oscilloscope mode", "choices": ["sine", "square", "sawtooth", "triangle", "pulse", "gaussian"], "default": "sine"},
            "n_frames": {"description": "frames", "min": 50, "max": 400, "default": 200},"anim_mode": {"description": "animation mode", "choices": ["none", "sweep"], "default": "none"},
        },
        outputs={"image": "IMAGE", "luminance": "SCALAR", "r": "SCALAR"})
def method_kuramoto(out_dir: Path, seed: int, params=None):
    """Kuramoto synchronization — coupled phase oscillators.

    N oscillators with natural frequencies ω_i interact via
      dθ_i/dt = ω_i + (K/N) Σ sin(θ_j - θ_i)

    Anim_mode "sweep": animate K from 0 → coupling_max — watch the
    synchronization phase transition in real time.

    Visualizations:
      - pendulums (default): vertical bouncing dots. Each dot moves
        up/down as sin(θ). Unsynchronized = chaotic bouncing.
        Synchronized = all bounce together.
      - field: pulsing firefly dots in 2D space
      - ring: phase positions on a circle (original)
    """
    if params is None:
        params = {}

    anim_time = float(params.get("time", 0.0))
    anim_mode = params.get("anim_mode", "none")
    is_sweep = anim_mode == "sweep"

    n = int(params.get("n_oscillators", 200))
    K_max = float(params.get("coupling_max", 3.0))
    K_start = float(params.get("K_start", 0.0))
    freq_std = float(params.get("freq_std", 1.0))
    dot_size = max(1, int(params.get("dot_size", 3)))
    trail_length = int(params.get("trail_length", 12))
    viz = params.get("viz", "pendulums")
    wave_type = params.get("wave_type", "sine")
    pulse_width = float(params.get("pulse_width", 0.15))
    n_frames = int(params.get("n_frames", 200))

    # ── Vectorized waveform function ──
    _wav_arr = np.frompyfunc(lambda p: _wav(p, wave_type, pulse_width), 1, 1)

    seed_all(seed)
    rng = np.random.default_rng(seed)

    # ── Natural frequencies (Gaussian, mean 0) ──
    omega = rng.normal(0.0, freq_std, n).astype(np.float64)

    # ── Initial phases (uniform [0, 2π)) ──
    theta = rng.uniform(0.0, 2.0 * math.pi, n).astype(np.float64)

    # ── Color map: hue by natural frequency ──
    freq_min = -3.0 * freq_std
    freq_max = 3.0 * freq_std
    freq_norm = np.clip((omega - freq_min) / (freq_max - freq_min + 1e-10), 0.0, 1.0)
    hues = (0.6 - freq_norm * 0.6)

    # Vectorized HSV→RGB (s=1.0, v=1.0 for max brightness)
    h6 = hues * 6.0
    i = np.floor(h6).astype(int) % 6
    f = h6 - np.floor(h6)
    s_arr = np.full(n, 1.0)
    v_arr = np.full(n, 1.0)
    p_arr = v_arr * (1.0 - s_arr)
    q_arr = v_arr * (1.0 - f * s_arr)
    t_arr = v_arr * (1.0 - (1.0 - f) * s_arr)

    r_ch = np.zeros(n, dtype=np.float64)
    g_ch = np.zeros(n, dtype=np.float64)
    b_ch = np.zeros(n, dtype=np.float64)

    mask0 = i == 0; r_ch[mask0] = v_arr[mask0]; g_ch[mask0] = t_arr[mask0]; b_ch[mask0] = p_arr[mask0]
    mask1 = i == 1; r_ch[mask1] = q_arr[mask1]; g_ch[mask1] = v_arr[mask1]; b_ch[mask1] = p_arr[mask1]
    mask2 = i == 2; r_ch[mask2] = p_arr[mask2]; g_ch[mask2] = v_arr[mask2]; b_ch[mask2] = t_arr[mask2]
    mask3 = i == 3; r_ch[mask3] = p_arr[mask3]; g_ch[mask3] = q_arr[mask3]; b_ch[mask3] = v_arr[mask3]
    mask4 = i == 4; r_ch[mask4] = t_arr[mask4]; g_ch[mask4] = p_arr[mask4]; b_ch[mask4] = v_arr[mask4]
    mask5 = i == 5; r_ch[mask5] = v_arr[mask5]; g_ch[mask5] = p_arr[mask5]; b_ch[mask5] = q_arr[mask5]

    colors_rgb = np.stack([r_ch, g_ch, b_ch], axis=1)  # (n, 3)
    colors_uint8 = (colors_rgb * 255).astype(np.uint8)

    # ── 2D positions for field mode ──
    margin = 80
    field_positions = np.column_stack([
        rng.uniform(margin, W - margin, n),
        rng.uniform(margin, H - margin - 50, n),  # leave room for indicator bar
    ]).astype(np.int32)

    # ── Pendulum x-positions for pendulums mode ──
    pend_margin = 60
    pend_x = np.linspace(pend_margin, W - pend_margin, n).astype(np.int32)

    # ── Grid positions for breathing mode ──
    grid_cols = int(math.ceil(math.sqrt(n * W / H)))
    grid_rows = int(math.ceil(n / grid_cols))
    gx = np.linspace(60, W - 60, grid_cols)
    gy = np.linspace(40, H - 80, grid_rows)
    grid_positions = np.array([(x, y) for y in gy for x in gx[:grid_cols]][:n], dtype=np.int32)

    # ── Bouncing positions for bouncing mode ──
    bounce_margin = 100
    bounce_rows = 6
    bounce_cols = int(math.ceil(n / bounce_rows))
    bounce_x = np.linspace(bounce_margin, W - bounce_margin, bounce_cols)
    bounce_y_centers = np.linspace(70, H - 90, bounce_rows)
    bounce_pos = []
    for bi in range(bounce_rows):
        for bj in range(bounce_cols):
            if len(bounce_pos) >= n:
                break
            bounce_pos.append((int(bounce_x[bj]), int(bounce_y_centers[bi])))
    bounce_positions = np.array(bounce_pos, dtype=np.int32)

    # ── Phase trail ring buffer ──
    phase_history = deque(maxlen=trail_length)
    for _ in range(trail_length):
        phase_history.append(theta.copy())

    # ── Waveform history for oscilloscope mode ──
    scope_history_len = 50
    # Store the last scope_history_len sin(θ) values per oscillator
    waveform_history = np.zeros((n, scope_history_len), dtype=np.float64)
    waveform_history[:] = _wav_arr(theta).astype(np.float64)[:, np.newaxis]  # initial fill

    # ── Integration ──
    dt = 0.01
    sub_steps = 5

    if is_sweep and anim_time > 0.01:
        n_frames = max(50, int(30 + anim_time * 20))

    for frame in range(n_frames):
        if is_sweep:
            t = frame / max(n_frames - 1, 1)
            K = K_start + (K_max - K_start) * t
        else:
            K = K_max

        # Kuramoto integration (vectorized)
        for _ in range(sub_steps):
            phase_matrix = theta - theta[:, np.newaxis]
            coupling_term = (K / n) * np.sum(np.sin(phase_matrix), axis=1)
            theta += (omega + coupling_term) * (dt / sub_steps)

        # Record phase for trail
        phase_history.append(theta.copy())
        r = np.abs(np.mean(np.exp(1j * theta)))

        # Record waveform for oscilloscope mode
        waveform_history[:, :-1] = waveform_history[:, 1:]  # shift left
        waveform_history[:, -1] = _wav_arr(theta).astype(np.float64)  # newest on right

        # ── Render ──
        img = Image.new("RGB", (W, H), DARK_BG)
        drw = ImageDraw.Draw(img, "RGBA")

        if viz == "field":
            # ── Firefly field visualization ──
            # Each oscillator is a pulsing dot in 2D space.
            # Brightness pulses as (cos(θ) + 1) / 2: θ=0 → fully bright, θ=π → dark.
            # When synchronized (high r), all dots pulse in unison.

            # Draw phase trails as fading glows at old positions
            trail_entries = list(phase_history)
            trail_subsample = max(1, n // 80)
            for t_idx, hist_theta in enumerate(trail_entries):
                alpha = int(8 + (t_idx / max(len(trail_entries) - 1, 1)) * 20)
                brightness = (np.cos(hist_theta) + 1.0) / 2.0  # [0, 1]
                for i in range(0, n, trail_subsample):
                    bx, by = int(field_positions[i, 0]), int(field_positions[i, 1])
                    glow_r = max(1, int(brightness[i] * dot_size * 2.5))
                    gc = (*[int(c) for c in colors_uint8[i]], alpha)
                    drw.ellipse(
                        (bx - glow_r, by - glow_r, bx + glow_r, by + glow_r),
                        fill=gc,
                    )

            # Draw current oscillator dots with pulsing brightness
            current_brightness = (np.cos(theta) + 1.0) / 2.0  # [0, 1]
            for i in range(n):
                bx, by = int(field_positions[i, 0]), int(field_positions[i, 1])
                b = current_brightness[i]
                glow_r = max(1, int(b * dot_size * 3.0))
                # Inner bright core
                core_r = max(1, int(b * dot_size))
                gc = tuple(int(c * (0.3 + 0.7 * b)) for c in colors_uint8[i])
                drw.ellipse(
                    (bx - glow_r, by - glow_r, bx + glow_r, by + glow_r),
                    fill=(*gc, int(40 * b)),
                )
                drw.ellipse(
                    (bx - core_r, by - core_r, bx + core_r, by + core_r),
                    fill=gc,
                )

            # Order parameter r display: pulsing central indicator
            pulse = (np.cos(phase_history[-1]).mean() + 1.0) / 2.0 if len(phase_history) > 0 else 0.5
            glow_radius = int(20 + r * 80)
            glow_alpha = int(30 + r * 80)
            drw.ellipse(
                (W - 90 - glow_radius, 20 - glow_radius,
                 W - 90 + glow_radius, 20 + glow_radius),
                fill=(180, 220, 255, glow_alpha),
            )
            drw.text((W - 95, 28), f"r={r:.2f}", fill=(200, 220, 255))

        elif viz == "pendulums":
            # ── Pendulum array visualization ──
            # Each oscillator is a vertical line with a bouncing dot.
            # y-offset = sin(θ) × amplitude.
            # When unsynchronized: each dot bounces at its own pace.
            # When synchronized: they all bounce in perfect unison.
            pend_amp = 140
            mid_y = H // 2 - 40

            # Draw reference line
            drw.line(
                [(pend_margin, mid_y), (W - pend_margin, mid_y)],
                fill=(40, 40, 60), width=1,
            )

            # Draw phase trails as faint ghost positions
            trail_entries = list(phase_history)
            trail_subsample = max(1, n // 60)
            for t_idx, hist_theta in enumerate(trail_entries):
                alpha = int(6 + (t_idx / max(len(trail_entries) - 1, 1)) * 14)
                hist_y = (np.sin(hist_theta) * pend_amp + mid_y).astype(int)
                for i in range(0, n, trail_subsample):
                    tc = (*[int(c) for c in colors_uint8[i]], alpha)
                    drw.point((int(pend_x[i]), int(hist_y[i])), fill=tc)

            # Draw each oscillator: vertical line + bouncing dot
            dot_y = (np.sin(theta) * pend_amp + mid_y).astype(int)
            plus_r = max(1, dot_size)
            for i in range(n):
                c = tuple(int(ch) for ch in colors_uint8[i])
                # Vertical stem from rest to current position
                stem_y1, stem_y2 = (mid_y, dot_y[i]) if dot_y[i] < mid_y else (dot_y[i], mid_y)
                drw.line([(pend_x[i], stem_y1), (pend_x[i], stem_y2)],
                         fill=(*c, 60), width=1)
                # Dot at current position
                drw.ellipse(
                    (pend_x[i] - plus_r, dot_y[i] - plus_r,
                     pend_x[i] + plus_r, dot_y[i] + plus_r),
                    fill=c,
                )

            # Sync indicator
            sync_color = (100, 255, 100) if r > 0.8 else (255, 200, 60) if r > 0.5 else (200, 100, 100)
            ind_y = mid_y - pend_amp - 20
            ind_r = int(8 + r * 10)
            drw.ellipse(
                (W - 60 - ind_r, ind_y - ind_r, W - 60 + ind_r, ind_y + ind_r),
                fill=(*sync_color, 160),
            )
            drw.text((W - 100, ind_y - 4), f"sync r={r:.2f}", fill=sync_color)

            # Labels
            label_y = mid_y + pend_amp + 16
            drw.text((pend_margin, label_y), f"K={K:.1f}", fill=(100, 140, 200))

        elif viz == "breathing":
            # ── Breathing circles ──
            # Each oscillator is a circle that expands and contracts.
            # Radius = (cos(θ) + 1) / 2 × max_r + min_r
            # θ=0 → fully expanded, θ=π → fully contracted.
            # When synchronized, all circles breathe in unison.
            min_r = 2
            max_r = 20
            radii = ((np.cos(theta) + 1.0) / 2.0 * (max_r - min_r) + min_r).astype(float)
            trail_entries = list(phase_history)
            trail_subsample = max(1, n // 60)
            for t_idx, hist_theta in enumerate(trail_entries):
                alpha = int(6 + (t_idx / max(len(trail_entries) - 1, 1)) * 16)
                hist_r = ((np.cos(hist_theta) + 1.0) / 2.0 * (max_r - min_r) + min_r)
                for i in range(0, n, trail_subsample):
                    bx, by = int(grid_positions[i, 0]), int(grid_positions[i, 1])
                    ri = max(1, int(hist_r[i]))
                    tc = (*[int(c) for c in colors_uint8[i]], alpha)
                    drw.ellipse(
                        (bx - ri, by - ri, bx + ri, by + ri),
                        fill=tc,
                    )

            for i in range(n):
                bx, by = int(grid_positions[i, 0]), int(grid_positions[i, 1])
                ri = max(1, int(radii[i]))
                c = tuple(int(ch) for ch in colors_uint8[i])
                # Outer glow
                drw.ellipse(
                    (bx - ri - 2, by - ri - 2, bx + ri + 2, by + ri + 2),
                    fill=(*c, 60),
                )
                # Solid circle
                drw.ellipse(
                    (bx - ri, by - ri, bx + ri, by + ri),
                    fill=c,
                )

            # Sync indicator
            sync_color = (100, 255, 100) if r > 0.8 else (255, 200, 60) if r > 0.5 else (200, 100, 100)
            drw.text((15, 15), f"r={r:.2f}", fill=sync_color)
            drw.text((15, 30), f"K={K:.1f}", fill=(100, 140, 200))

        elif viz == "bouncing":
            # ── Bouncing balls between walls ──
            # Each oscillator is a ball bouncing horizontally between walls.
            # x-position = sin(θ) mapped to [left_wall, right_wall].
            # When unsynchronized: balls bounce chaotically at different positions.
            # When synchronized: all balls hit the walls at the same time.
            left_x = 40
            right_x = W - 40
            ball_x = ((np.sin(theta) + 1.0) / 2.0 * (right_x - left_x - 20) + left_x + 10).astype(int)

            # Draw wall lines
            drw.line([(left_x, 30), (left_x, H - 50)], fill=(60, 60, 90), width=1)
            drw.line([(right_x, 30), (right_x, H - 50)], fill=(60, 60, 90), width=1)

            # Phase trails
            trail_entries = list(phase_history)
            trail_subsample = max(1, n // 60)
            for t_idx, hist_theta in enumerate(trail_entries):
                alpha = int(6 + (t_idx / max(len(trail_entries) - 1, 1)) * 12)
                hist_x = ((np.sin(hist_theta) + 1.0) / 2.0 * (right_x - left_x - 20) + left_x + 10).astype(int)
                for i in range(0, n, trail_subsample):
                    bi = int(bounce_positions[i, 1])
                    tc = (*[int(c) for c in colors_uint8[i]], alpha)
                    drw.point((int(hist_x[i]), bi), fill=tc)

            # Bouncing balls
            ball_r = max(2, dot_size + 1)
            for i in range(n):
                bx = ball_x[i]
                by = int(bounce_positions[i, 1])
                c = tuple(int(ch) for ch in colors_uint8[i])
                drw.ellipse(
                    (bx - ball_r, by - ball_r, bx + ball_r, by + ball_r),
                    fill=c,
                )

            # Sync indicator & K label
            sync_color = (100, 255, 100) if r > 0.8 else (255, 200, 60) if r > 0.5 else (200, 100, 100)
            drw.text((left_x + 5, H - 38), f"r={r:.2f}", fill=sync_color)
            drw.text((left_x + 5, H - 22), f"K={K:.1f}", fill=(100, 140, 200))

        elif viz == "oscilloscope":
            # ── Oscilloscope grid ──
            # Each oscillator gets its own mini oscilloscope screen showing
            # its waveform sin(θ) scrolling from right to left.
            # When unsynchronized: each trace cycles at a different rate.
            # When synchronized: all traces show identical waveforms.
            max_scopes = 80  # limit for visual clarity
            n_scopes = min(n, max_scopes)
            # Layout: 8 rows × n_cols
            scope_rows = 8
            scope_cols = int(math.ceil(n_scopes / scope_rows))
            # Scope cell dimensions
            margin_left = 12
            margin_top = 14
            cell_w = (W - margin_left * 2 - (scope_cols - 1) * 4) // scope_cols
            cell_h = (H - margin_top * 2 - (scope_rows - 1) * 2) // scope_rows
            cell_w = max(20, min(cell_w, 90))
            cell_h = max(15, min(cell_h, 55))
            total_w = scope_cols * cell_w + (scope_cols - 1) * 4
            total_h = scope_rows * cell_h + (scope_rows - 1) * 2
            offset_x = (W - total_w) // 2
            offset_y = (H - total_h) // 2

            # Subsample oscillators for the grid
            step = max(1, n // n_scopes) if n > n_scopes else 1
            scope_indices = list(range(0, n, step))[:n_scopes]

            # For the background, draw scope screens
            bg_rects = []
            for idx, oi in enumerate(scope_indices):
                row = idx // scope_cols
                col = idx % scope_cols
                sx = offset_x + col * (cell_w + 4)
                sy = offset_y + row * (cell_h + 2)
                bg_rects.append((sx, sy, sx + cell_w, sy + cell_h))
                # Dark scope face
                drw.rectangle((sx, sy, sx + cell_w, sy + cell_h),
                              fill=(8, 8, 28), outline=(40, 40, 60), width=1)

            # Draw traces
            for idx, oi in enumerate(scope_indices):
                sx, sy, ex, ey = bg_rects[idx]
                c = tuple(int(ch) for ch in colors_uint8[oi])
                trace = waveform_history[oi]  # (scope_history_len,)
                # Map trace [-1, 1] → [sy+2, ey-2]
                trace_y = sy + (cell_h - 4) - ((trace + 1.0) / 2.0 * (cell_h - 4))
                trace_y = np.clip(trace_y, sy + 1, ey - 1).astype(int)
                # Draw trace as connected line segments
                n_pts = len(trace)
                for pi in range(n_pts - 1):
                    x1 = sx + 2 + pi * (cell_w - 4) / (n_pts - 1)
                    x2 = sx + 2 + (pi + 1) * (cell_w - 4) / (n_pts - 1)
                    drw.line([(int(x1), int(trace_y[pi])),
                              (int(x2), int(trace_y[pi + 1]))],
                             fill=(*c, 160), width=1)

                # Bright dot at the current (rightmost) position
                last_x = sx + cell_w - 2
                last_y = trace_y[-1]
                drw.ellipse((last_x - 2, last_y - 2, last_x + 2, last_y + 2),
                            fill=c)

            # Sync indicator at top
            sync_color = (100, 255, 100) if r > 0.8 else (255, 200, 60) if r > 0.5 else (200, 100, 100)
            drw.text((10, 2), f"r={r:.2f}  K={K:.1f}  [{wave_type}]", fill=sync_color)

        else:
            # ── Ring visualization (original) ──
            # Phase trails (oldest → newest, fading in)
            trail_entries = list(phase_history)
            trail_subsample = max(1, n // 100)
            for t_idx, hist_theta in enumerate(trail_entries):
                alpha = int(15 + (t_idx / max(len(trail_entries) - 1, 1)) * 50)
                for i in range(0, n, trail_subsample):
                    tx = int(CENTER_X + RING_RADIUS * np.cos(hist_theta[i]))
                    ty = int(CENTER_Y + RING_RADIUS * np.sin(hist_theta[i]))
                    tc = (*[int(c) for c in colors_uint8[i]], alpha)
                    drw.point((tx, ty), fill=tc)

            # Ring track
            drw.ellipse(
                (CENTER_X - RING_RADIUS, CENTER_Y - RING_RADIUS,
                 CENTER_X + RING_RADIUS, CENTER_Y + RING_RADIUS),
                outline=(50, 50, 80),
                width=1,
            )

            # Central glow
            glow_r = int(20 + r * 100)
            glow_alpha = int(25 + r * 60)
            drw.ellipse(
                (CENTER_X - glow_r, CENTER_Y - glow_r,
                 CENTER_X + glow_r, CENTER_Y + glow_r),
                fill=(180, 210, 255, glow_alpha),
            )

            # Radial spikes
            spike_every = max(1, n // 60)
            for i in range(0, n, spike_every):
                sx = CENTER_X + RING_RADIUS * 0.2 * np.cos(theta[i])
                sy = CENTER_Y + RING_RADIUS * 0.2 * np.sin(theta[i])
                ex = CENTER_X + RING_RADIUS * np.cos(theta[i])
                ey = CENTER_Y + RING_RADIUS * np.sin(theta[i])
                spike_color = (*[int(c) for c in colors_uint8[i]], 80)
                drw.line([(sx, sy), (ex, ey)], fill=spike_color, width=1)

            # Oscillator dots
            for i in range(n):
                px = int(CENTER_X + RING_RADIUS * np.cos(theta[i]))
                py = int(CENTER_Y + RING_RADIUS * np.sin(theta[i]))
                c = tuple(int(ch) for ch in colors_uint8[i])
                drw.ellipse(
                    (px - dot_size, py - dot_size, px + dot_size, py + dot_size),
                    fill=c,
                )

        # ── K / r label for ring mode ──
        if isinstance(viz, str) and viz == "ring":
            sync_color = (100, 255, 100) if r > 0.8 else (255, 200, 60) if r > 0.5 else (200, 100, 100)
            drw.text((15, 15), f"r={r:.2f}", fill=sync_color)
            drw.text((15, 30), f"K={K:.1f}", fill=(100, 140, 200))

        if is_sweep:
            capture_frame("89", np.array(img, dtype=np.float32) / 255.0)

    # Final capture
    capture_frame("89", np.array(img, dtype=np.float32) / 255.0)
    write_scalars(out_dir, r=float(r))
    save(img, mn(89, "Kuramoto Sync"), out_dir)
    return img
