from __future__ import annotations

import math
from pathlib import Path

import numpy as np

from ...core.registry import method
from ...core.utils import (
    save, mn, seed_all, W, H, PALETTES,
    write_scalars, write_field, write_particles,
)
from ...core.animation import capture_frame

# ─────────────────────────────────────────────────────────────────────────────
# Aizawa attractor — a 3D chaotic attractor (Tetsuya Aizawa, 1982).
#
# The system of three coupled ODEs
#     ẋ = (z − b)·x − d·y
#     ẏ = d·x + (z − b)·y
#     ż = c + a·z − z³/3 − (x² + y²)·(1 + e·z) + f·z·x³
# with the canonical parameters (a, b, c, d, e, f) = (0.95, 0.7, 0.6, 3.5,
# 0.25, 0.1) produces a distinctive two-lobed "bread-loaf" shell with a fine
# internal ribbon structure — one of the most photogenic of the classical 3D
# strange attractors. See e.g. J. C. Sprott, "Strange Attractators: Creating
# Patterns in Chaos" (2003), §4.3.
#
# Unlike the de Jong / de Jong-style planar maps, this is a genuine 3D flow, so
# the node carries first-class 3D controls (rotation, pan, zoom, perspective)
# and projects the trajectory to 2D with a simple pinhole camera. The attractor
# lives roughly inside a sphere of radius ~1.7 centred near the origin, so the
# default camera frames it directly.
#
# Architecture-B method: no cross-frame state. The integrator is re-run each
# frame; `time` (or `anim_mode`) drives smooth turn-table rotation so it animates
# natively in 📺 Live mode. Determinism: all RNG seeded by `seed`.
# ─────────────────────────────────────────────────────────────────────────────

# Canonical Aizawa parameters. Exposed as editable controls so the shape can be
# morphed — but the defaults reproduce the classic attractor.
_DEFAULT_A = 0.95
_DEFAULT_B = 0.7
_DEFAULT_C = 0.6
_DEFAULT_D = 3.5
_DEFAULT_E = 0.25
_DEFAULT_F = 0.1

_MAX_PARTICLES = 200_000  # cap on written PARTICLES output (disk-friendly)


def _iq_cos_ramp(t: np.ndarray) -> np.ndarray:
    """Inigo-Quilez cosine palette — smooth, periodic, vivid (phase offsets)."""
    t = np.clip(t, 0.0, 1.0)
    r = 0.5 + 0.5 * np.cos(6.2831853 * (t + 0.00))
    g = 0.5 + 0.5 * np.cos(6.2831853 * (t + 0.3333333))
    b = 0.5 + 0.5 * np.cos(6.2831853 * (t + 0.6666667))
    return np.stack([r, g, b], axis=-1)


def _colormap(t: np.ndarray, name: str) -> np.ndarray:
    """Look up `name` in the palette registry, else fall back to an IQ ramp."""
    pal = PALETTES.get(name, [])
    if len(pal) >= 2:
        arr = np.asarray(pal, dtype=np.float32) / 255.0
        idx = np.clip((t * (len(arr) - 1)).astype(np.int64), 0, len(arr) - 1)
        return arr[idx]
    return _iq_cos_ramp(t)


def _integrate(a, b, c, d, e, f, steps, dt, discard, seed):
    """RK4-integrate the Aizawa system once; cache by parameter tuple.

    The trajectory depends only on the ODE parameters (a–f), timestep, length,
    and seed — never on the camera `time`/`yaw`/etc. Caching it means 📺 Live
    mode re-cooks every frame for free: it only re-projects and re-colors the
    (unchanging) curve, so the attractor spins smoothly instead of re-integrating
    200k steps per frame. Bounded to a few entries keyed on rounded params.
    """
    key = (
        round(a, 6), round(b, 6), round(c, 6), round(d, 6),
        round(e, 6), round(f, 6), int(steps), round(dt, 6),
        int(discard), int(seed),
    )
    cached = _TRAJ_CACHE.get(key)
    if cached is not None:
        return cached

    def deriv(p):
        x, y, z = p[:, 0], p[:, 1], p[:, 2]
        dx = (z - b) * x - d * y
        dy = d * x + (z - b) * y
        dz = c + a * z - (z ** 3) / 3.0 - (x ** 2 + y ** 2) * (1.0 + e * z) + f * z * (x ** 3)
        return np.stack([dx, dy, dz], axis=-1)

    p = np.array([0.1, 0.0, 0.0], dtype=np.float64).reshape(1, 3)
    for _ in range(int(discard)):
        k1 = deriv(p)
        p = p + dt * k1

    xyz = np.empty((int(steps), 3), dtype=np.float64)
    for i in range(int(steps)):
        k1 = deriv(p)
        k2 = deriv(p + 0.5 * dt * k1)
        k3 = deriv(p + 0.5 * dt * k2)
        k4 = deriv(p + dt * k3)
        p = p + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        xyz[i] = p[0]

    if len(_TRAJ_CACHE) >= 4:
        _TRAJ_CACHE.pop(next(iter(_TRAJ_CACHE)))
    _TRAJ_CACHE[key] = xyz
    return xyz


_TRAJ_CACHE: dict = {}


def _rotate(points: np.ndarray, yaw: float, pitch: float, roll: float) -> np.ndarray:
    """Rotate N×3 points by yaw (about Y), pitch (about X), roll (about Z)."""
    cy, sy = math.cos(yaw), math.sin(yaw)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cr, sr = math.cos(roll), math.sin(roll)

    # Yaw about Y
    x = cy * points[:, 0] + sy * points[:, 2]
    z = -sy * points[:, 0] + cy * points[:, 2]
    y = points[:, 1]
    # Pitch about X
    y2 = cp * y - sp * z
    z2 = sp * y + cp * z
    # Roll about Z
    x3 = cr * x - sr * y2
    y3 = sr * x + cr * y2
    return np.stack([x3, y3, z2], axis=-1)


@method(
    id="503",
    name="Aizawa Attractor",
    category="math_art",
    new_image_contract=True,
    description="3D chaotic Aizawa attractor: RK4-integrated flow projected through a rotatable pinhole camera and tone-mapped as a glowing shell; colors by depth, density, fire, ice, or mono.",
    tags=["attractor", "aizawa", "chaos", "strange-attractor", "3d", "generative",
          "math-art", "animation", "expanded"],
    inputs={},
    outputs={"image": "IMAGE", "field": "FIELD", "luminance": "SCALAR",
             "particles": "PARTICLES"},
    params={
        "a": {"description": "Aizawa parameter a (flow-curvature / lobe separation)",
              "min": 0.0, "max": 2.0, "default": _DEFAULT_A},
        "b": {"description": "Aizawa parameter b (z-offset term)",
              "min": 0.0, "max": 1.5, "default": _DEFAULT_B},
        "c": {"description": "Aizawa parameter c (constant forcing)",
              "min": 0.0, "max": 1.5, "default": _DEFAULT_C},
        "d": {"description": "Aizawa parameter d (in-plane rotation rate)",
              "min": 0.0, "max": 6.0, "default": _DEFAULT_D},
        "e": {"description": "Aizawa parameter e (anisotropic z-stretch)",
              "min": 0.0, "max": 1.0, "default": _DEFAULT_E},
        "f": {"description": "Aizawa parameter f (cubic z·x³ coupling)",
              "min": 0.0, "max": 0.5, "default": _DEFAULT_F},
        "steps": {"description": "integration steps (trajectory length); the curve is cached so live frames stay fast",
                  "min": 20000, "max": 600000, "default": 80000},
        "dt": {"description": "integration timestep",
               "min": 0.001, "max": 0.02, "default": 0.01},
        "discard": {"description": "initial transient steps discarded (settle onto attractor)",
                    "min": 0, "max": 5000, "default": 1000},
        "color_mode": {"description": "coloring: depth (z → palette), density (accumulation), fire, ice, mono",
                       "choices": ["depth", "density", "fire", "ice", "mono"], "default": "depth"},
        "palette": {"description": "named palette for depth/density coloring",
                    "choices": ["viridis", "inferno", "plasma", "magma", "turbo",
                                "twilight", "rainbow", "ocean", "sunset"], "default": "viridis"},
        "exposure": {"description": "tone-map exposure (higher = brighter / denser glow)",
                     "min": 0.2, "max": 8.0, "default": 1.8},
        "background": {"description": "canvas background",
                       "choices": ["black", "navy", "cream", "white"], "default": "black"},
        # ── 3D camera ──
        "yaw": {"description": "camera yaw (turn-table rotation about vertical axis)",
                "min": -3.1416, "max": 3.1416, "default": 0.6},
        "pitch": {"description": "camera pitch (tilt up/down)",
                  "min": -1.5, "max": 1.5, "default": 0.35},
        "roll": {"description": "camera roll (in-plane spin)",
                 "min": -3.1416, "max": 3.1416, "default": 0.0},
        "pan_x": {"description": "camera horizontal pan (fraction of canvas)",
                  "min": -1.0, "max": 1.0, "default": 0.0},
        "pan_y": {"description": "camera vertical pan (fraction of canvas)",
                  "min": -1.0, "max": 1.0, "default": 0.0},
        "zoom": {"description": "camera zoom (focal scale)",
                 "min": 0.3, "max": 3.0, "default": 1.0},
        "perspective": {"description": "perspective strength (0 = orthographic)",
                        "min": 0.0, "max": 1.5, "default": 0.6},
        # ── Animation ──
        "anim_mode": {"description": "animation mode: none (static), spin_yaw, spin_pitch, orbit",
                      "choices": ["none", "spin_yaw", "spin_pitch", "orbit"], "default": "none"},
        "anim_speed": {"description": "animation speed multiplier",
                       "min": 0.1, "max": 5.0, "default": 1.0},
        "time": {"description": "animation phase [0, 2π)",
                 "min": 0.0, "max": 6.2832, "default": 0.0},
    },
)
def method_aizawa(out_dir: Path, seed: int, params=None):
    """Aizawa strange attractor — a 3D chaotic flow rendered as a glowing shell.

    Integrates the Aizawa ODE system with RK4, rotates the trajectory by the
    camera (yaw/pitch/roll + pan/zoom/perspective), and splats it into a 2D
    density grid. Coloring options: ``depth`` (map the projected z/depth through
    a palette — the default, which reveals the attractor's 3D ribbon structure),
    ``density`` (accumulation → glow), ``fire`` / ``ice`` / ``mono`` ramps.

    Architecture-B method (no cross-frame state): the integrator re-runs each
    frame. ``anim_mode`` drives smooth turn-table/pitch/orbit motion off ``time``
    (via sine/cosine — no ``abs(sin)`` cusps), so the attractor spins natively
    in 📺 Live mode. The 3D camera controls make it a building block for
    fly-through compositions.

    Sidecars: FIELD = normalised accumulation density (usable by downstream
    flow/particle nodes), PARTICLES = a capped sample of projected points with
    per-point depth, SCALAR = parameter + coverage stats.

    Determinism: all RNG seeded by `seed`; in ``none`` mode output is identical
    at every `time`.
    """
    try:
        if params is None:
            params = {}

        t = float(params.get("time", 0.0))
        anim_mode = str(params.get("anim_mode", "none"))
        anim_speed = float(params.get("anim_speed", 1.0))
        _t = 0.0 if anim_mode == "none" else t * anim_speed

        a = float(params.get("a", _DEFAULT_A))
        b = float(params.get("b", _DEFAULT_B))
        c = float(params.get("c", _DEFAULT_C))
        d = float(params.get("d", _DEFAULT_D))
        e = float(params.get("e", _DEFAULT_E))
        f = float(params.get("f", _DEFAULT_F))
        steps = max(20000, min(600000, int(params.get("steps", 200000))))
        dt = max(0.001, min(0.02, float(params.get("dt", 0.01))))
        discard = max(0, min(5000, int(params.get("discard", 1000))))
        color_mode = str(params.get("color_mode", "depth"))
        palette_name = str(params.get("palette", "viridis"))
        exposure = max(0.2, min(8.0, float(params.get("exposure", 1.8))))
        background = str(params.get("background", "black"))

        # ── Animation: smooth camera perturbation (no cusps) ──
        yaw = float(params.get("yaw", 0.6))
        pitch = float(params.get("pitch", 0.35))
        roll = float(params.get("roll", 0.0))
        if anim_mode == "spin_yaw":
            yaw += _t
        elif anim_mode == "spin_pitch":
            pitch += 0.6 * math.sin(_t)
            yaw += 0.3 * math.cos(_t * 0.7)
        elif anim_mode == "orbit":
            yaw += _t
            pitch += 0.5 * math.sin(_t * 0.5)

        pan_x = float(params.get("pan_x", 0.0))
        pan_y = float(params.get("pan_y", 0.0))
        zoom = max(0.3, min(3.0, float(params.get("zoom", 1.0))))
        perspective = max(0.0, min(1.5, float(params.get("perspective", 0.6))))

        w = int(W)
        h = int(H)

        seed_all(seed)
        rng = np.random.default_rng(seed)

        # ── RK4 integrate the Aizawa system (cached by parameter tuple) ──
        # The trajectory depends only on a–f / dt / steps / seed, not on the
        # camera, so it is cached and reused across live frames.
        xyz = _integrate(a, b, c, d, e, f, steps, dt, discard, seed)

        # ── 3D camera transform ──
        rot = _rotate(xyz, yaw, pitch, roll)

        # Pinhole projection: camera looks down -Z after rotation.
        # Pull the cloud in front of the lens by a fixed camera distance.
        cam_dist = 4.0
        zc = rot[:, 2] + cam_dist          # depth in front of camera
        zc = np.clip(zc, 1e-3, None)       # never behind the lens
        focal = zoom * min(w, h) * 0.5 * (1.0 + 0.5 * perspective)
        # Perspective divide (strength scales with `perspective`); ortho when 0.
        persp = 1.0 / (1.0 + perspective * (cam_dist / zc - 1.0))
        sx = (rot[:, 0] * focal * persp) + (w * 0.5) + (pan_x * w)
        sy = (rot[:, 1] * focal * persp * -1.0) + (h * 0.5) + (pan_y * h)  # flip Y for screen

        ix = np.floor(sx).astype(np.int64)
        iy = np.floor(sy).astype(np.int64)
        inside = (ix >= 0) & (ix < w) & (iy >= 0) & (iy < h)
        ix = ix[inside]
        iy = iy[inside]
        # Normalised depth (0 far → 1 near) for coloring; centred on cam_dist.
        depth = zc[inside]
        depth_n = np.clip(1.0 - (depth - depth.min()) / (depth.max() - depth.min() + 1e-9), 0.0, 1.0)

        if ix.size == 0:
            raise RuntimeError("no attractor points projected onto the canvas")

        # ── Accumulate density + a per-pixel depth image ──
        # `depth` is a 1-D array of projected depths for the inside points, so
        # we must splat it into a full (h,w) image before coloring — _colormap
        # expects an (h,w) intensity field and returns (h,w,3). Passing the raw
        # 1-D array returns a (n_points,3) array that cannot broadcast against
        # the (h,w,3) canvas.
        density = np.zeros((h, w), dtype=np.float64)
        np.add.at(density, (iy, ix), 1.0)

        depth_sum = np.zeros((h, w), dtype=np.float64)
        np.add.at(depth_sum, (iy, ix), depth)
        depth_cnt = np.zeros((h, w), dtype=np.float64)
        np.add.at(depth_cnt, (iy, ix), 1.0)
        depth_img = np.divide(depth_sum, depth_cnt,
                              out=np.zeros_like(depth_sum), where=depth_cnt > 0)
        dmin, dmax = float(depth_img.min()), float(depth_img.max())
        depth_norm_img = np.clip((depth_img - dmin) / (dmax - dmin + 1e-9), 0.0, 1.0)

        # ── Tone-map density (1 - exp compression → glowing cloud) ──
        occ = density[density > 0]
        p99 = float(np.percentile(occ, 99)) if occ.size else 1.0
        glow = 1.0 - np.exp(-exposure * density / (p99 + 1e-9))
        glow = np.clip(glow, 0.0, 1.0)

        # ── Coloring ──
        base = {
            "black": np.array([0.0, 0.0, 0.0], dtype=np.float32),
            "navy": np.array([0.04, 0.06, 0.12], dtype=np.float32),
            "cream": np.array([0.96, 0.94, 0.88], dtype=np.float32),
            "white": np.array([1.0, 1.0, 1.0], dtype=np.float32),
        }.get(background, np.array([0.0, 0.0, 0.0], dtype=np.float32)).reshape(1, 1, 3)

        if color_mode == "mono":
            color = np.stack([glow, glow, glow], axis=-1)
        elif color_mode == "fire":
            color = np.stack([
                np.clip(glow * 1.5, 0.0, 1.0),
                np.clip(glow * 0.6, 0.0, 1.0),
                np.clip(glow * 0.2, 0.0, 1.0),
            ], axis=-1)
        elif color_mode == "ice":
            color = np.stack([
                np.clip(glow * 0.3, 0.0, 1.0),
                np.clip(0.35 + glow * 0.5, 0.0, 1.0),
                np.clip(0.55 + glow * 0.45, 0.0, 1.0),
            ], axis=-1)
        elif color_mode == "density":
            color = _colormap(glow, palette_name)
        else:  # depth — map projected depth through the palette to reveal 3D form
            color = _colormap(depth_norm_img, palette_name)

        # Composite color (already brightness-encoded via glow) over background.
        out = base * (1.0 - glow[..., None]) + color
        out = np.clip(out, 0.0, 1.0).astype(np.float32)

        # ── Sidecar outputs (AGENT_GUIDE §"Sidecar protocol") ──
        density_norm = (density / (float(density.max()) + 1e-9)).astype(np.float32)
        write_field(out_dir, density_norm)
        write_scalars(
            out_dir,
            points=float(ix.size),
            a=float(a), b=float(b), c=float(c),
            d=float(d), e=float(e), f=float(f),
            mean_density=float(float(density.mean())),
            max_density=float(float(density.max())),
            bbox_w=float(float(sx.max() - sx.min())),
            bbox_h=float(float(sy.max() - sy.min())),
        )
        # PARTICLES: a capped, deterministic sample of projected points (x, y, vx, vy),
        # where vx/vy carry (normalised depth, 0) for downstream consumers.
        n_p = ix.size
        if n_p > _MAX_PARTICLES:
            pick = rng.choice(n_p, size=_MAX_PARTICLES, replace=False)
            ix_s, iy_s, dp_s = ix[pick], iy[pick], depth_n[pick]
        else:
            ix_s, iy_s, dp_s = ix, iy, depth_n
        particles = np.stack([
            ix_s.astype(np.float32), iy_s.astype(np.float32),
            dp_s.astype(np.float32), np.zeros_like(dp_s, dtype=np.float32),
        ], axis=-1)
        write_particles(out_dir, particles)

        capture_frame("503", out)
        save(out, mn(503, f"Aizawa t={_t:.2f}"), out_dir)
        return out
    except Exception as exc:
        fb = np.zeros((int(H), int(W), 3), dtype=np.float32)
        save(fb, mn(503, "Aizawa"), out_dir)
        print(f"[method_503] ERROR: {exc}")
        return fb
