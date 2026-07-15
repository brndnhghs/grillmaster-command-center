from __future__ import annotations

import math

import numpy as np
from PIL import Image, ImageDraw

from ...core.registry import method
from ...core.utils import save, mn, seed_all, W, H, PALETTES, wired_source_lum
from ...core.animation import capture_frame


# ── Domain-warped flow field ───────────────────────────────────────────────
# A cheap, smooth, fully-deterministic pseudo-flow built from a few warped
# sinusoids. The warp (sampling the field at coordinates displaced by another
# sinusoid) is the classic "domain warping" trick (I. Quilez) — it turns plain
# periodic stripes into organic, rivulet-like streams that drive the Truchet
# tile orientation. `t` evolves the field so the maze flows over time.
def _flow_angle(tx: float, ty: float, t: float, seed: int, warp: float) -> float:
    s1 = seed * 0.137
    s2 = seed * 0.071
    # domain warp: displace the sample point by a slow sinusoid whose strength
    # is the `warp` parameter (0 = plain stripes, 1.5 = tight rivulets)
    wx = tx + warp * math.sin(ty * 1.7 + t * 0.5 + s1)
    wy = ty + warp * math.cos(tx * 1.3 - t * 0.4 + s2)
    a = math.sin(wx * 2.0 + t * 0.6) + math.cos(wy * 2.3 - t * 0.3)
    a += 0.5 * math.sin((wx + wy) * 3.1 + t * 0.8 + s1)
    return a


@method(
    id='531', name='Flowing Truchet', category='patterns',
    tags=['truchet', 'tiling', 'procedural', 'flow-field', 'labyrinth', 'animation', 'domain-warp'],
    params={
        'scale': {'description': 'tiles across the shorter canvas axis', 'min': 4.0, 'max': 80.0, 'default': 28.0},
        'line_width': {'description': 'stroke thickness of the arcs (px)', 'min': 1.0, 'max': 14.0, 'default': 4.0},
        'warp': {'description': 'domain-warp strength of the flow field (0=plain stripes, 1=rivulets)', 'min': 0.0, 'max': 1.5, 'default': 0.6},
        'colormode': {'description': 'color mapping (grayscale/rainbow/inferno/viridis/palette/fire/ice)', 'default': 'inferno'},
        'palette': {'description': 'palette name for palette mode', 'default': 'vapor'},
        'anim_mode': {'description': 'animation mode: none, flow, pulse, drift', 'default': 'none'},
        'anim_speed': {'description': 'animation speed multiplier', 'min': 0.1, 'max': 3.0, 'default': 1.0},
        'time': {'description': 'animation phase [0, 2pi)', 'min': 0.0, 'max': 6.28, 'default': 0.0},
        'source': {'description': "wired upstream image's luminance biases the flow field", 'choices': ['none', 'input_image'], 'default': 'none'},
    },
    inputs={'image_in': 'IMAGE'},
)
def method_flowing_truchet(out_dir, seed: int, params=None):
    """Render a **Flowing Truchet** labyrinth.

    Truchet tiles are square tiles each carrying a quarter-circle arc motif in
    one of two orientations; tiling them randomly yields the famous interlocking
    "tunnel" maze (C. S. Smith / S. Truchet, popularised in generative art).
    The fresh twist here: the tile *orientation* is not random per-tile but
    driven by a **domain-warped flow field**, so neighbouring tiles line up into
    coherent, river-like channels instead of noise.

    Animation modes (Architecture B — the orchestrator re-calls per frame with
    an increasing ``time``):
      * ``flow``  — the flow field evolves, so channels re-route and the maze
                    breathes/morphs.
      * ``pulse`` — arc thickness oscillates (smooth sine, no cusps) so tunnels
                    expand and contract.
      * ``drift`` — the field scrolls diagonally, sending the pattern flowing
                    across the canvas.
    """
    try:
        if params is None:
            params = {}
        t = float(params.get("time", 0.0))
        seed_all(seed)

        scale = float(np.clip(params.get("scale", 28.0), 4.0, 80.0))
        line_width = float(np.clip(params.get("line_width", 4.0), 1.0, 14.0))
        warp = float(np.clip(params.get("warp", 0.6), 0.0, 1.5))
        cmode = params.get("colormode", "inferno")
        pal_name = params.get("palette", "vapor")
        anim_mode = params.get("anim_mode", "none")
        anim_speed = float(params.get("anim_speed", 1.0))

        _t = 0.0 if anim_mode == "none" else t * anim_speed

        # ── Animation modulation (each mode has a distinct visual signature) ──
        # field time: evolves the orientation field for 'flow' & 'drift' only
        _field_t = _t if anim_mode in ("flow", "drift") else 0.0
        # line width: breathes only in 'pulse' (smooth sine, no cusps)
        _lw = line_width * (0.5 + 0.5 * math.sin(_t * 0.8)) if anim_mode == "pulse" else line_width
        # scroll offset: only 'drift' moves the field diagonally
        _drift_x = _t * 0.5 if anim_mode == "drift" else 0.0
        _drift_y = _t * 0.22 if anim_mode == "drift" else 0.0

        # ── Supersampled luminance canvas (white arcs on black) ──
        SS = 3
        lw = max(1.0, _lw * SS)
        cw, ch = W * SS, H * SS
        lum = Image.new("L", (cw, ch), 0)
        d = ImageDraw.Draw(lum)

        # square tiles covering the canvas (pad to keep tiles square)
        tile = cw / scale
        nx = max(1, int(round(cw / tile)))
        ny = max(1, int(round(ch / tile)))
        tile_w = cw / nx
        tile_h = ch / ny
        r_w = tile_w / 2.0
        r_h = tile_h / 2.0

        # Optional wired image biases the orientation field
        _src = wired_source_lum(params, W, H)

        for j in range(ny):
            for i in range(nx):
                # tile center in canvas (supersampled) coords
                cx = (i + 0.5) * tile_w
                cy = (j + 0.5) * tile_h
                # tile-space coordinate fed to the flow field
                fx = (i + 0.5) / max(nx, 1) * 6.2831853 + _drift_x
                fy = (j + 0.5) / max(ny, 1) * 6.2831853 + _drift_y
                ang = _flow_angle(fx, fy, _field_t, seed, warp)
                if _src is not None:
                    # sample wired luminance at this tile's position
                    si = int(np.clip(cx / SS, 0, W - 1))
                    sj = int(np.clip(cy / SS, 0, H - 1))
                    ang += (float(_src[sj, si]) - 0.5) * math.pi
                # binary orientation from the (wrapped) flow angle
                bit = 0 if (ang % (2.0 * math.pi)) < math.pi else 1

                x0 = i * tile_w
                y0 = j * tile_h
                x1 = x0 + tile_w
                y1 = y0 + tile_h
                # PIL arc angles: 0=E, 90=S, 180=W, 270=N
                if bit == 0:
                    # arcs centered at top-left and bottom-right corners
                    d.arc([x0, y0, x0 + 2 * r_w, y0 + 2 * r_h], 0, 90, fill=255, width=int(lw))
                    d.arc([x1 - 2 * r_w, y1 - 2 * r_h, x1, y1], 180, 270, fill=255, width=int(lw))
                else:
                    # arcs centered at top-right and bottom-left corners
                    d.arc([x1 - 2 * r_w, y0, x1, y0 + 2 * r_h], 90, 180, fill=255, width=int(lw))
                    d.arc([x0, y1 - 2 * r_h, x0 + 2 * r_w, y1], 270, 360, fill=255, width=int(lw))

        # downsample to canvas resolution with anti-aliasing
        _RESAMPLE = getattr(getattr(Image, "Resampling", Image), "LANCZOS")
        lum = lum.resize((W, H), _RESAMPLE)
        val = (np.asarray(lum, dtype=np.float64) / 255.0)

        # ── Color mapping ──
        cm = None
        try:
            from matplotlib import cm  # noqa: F811
            _has_mpl = True
        except ImportError:
            _has_mpl = False

        if cmode == "grayscale":
            rgb = np.stack([val, val, val], axis=-1)
        elif cmode == "rainbow":
            hue = val * 2 * math.pi
            rgb = np.stack([
                np.sin(hue) * 0.5 + 0.5,
                np.sin(hue + 2.094) * 0.5 + 0.5,
                np.sin(hue + 4.189) * 0.5 + 0.5,
            ], axis=-1)
        elif cmode == "palette":
            pal = PALETTES.get(pal_name, PALETTES["vapor"])
            idx = (val * (len(pal) - 1)).astype(np.int32)
            rgb = np.array(pal, dtype=np.float32)[idx] / 255.0
        elif cmode == "inferno":
            if _has_mpl:
                rgb = cm.inferno(val)[:, :, :3]
            else:
                rgb = np.stack([val ** 1.4, val ** 0.6 * (1 - val) * 2 + val * 0.2, val ** 0.3 * 0.5], axis=-1)
        elif cmode == "viridis":
            if _has_mpl:
                rgb = cm.viridis(val)[:, :, :3]
            else:
                rgb = np.stack([val * 0.3, val ** 0.5 * 0.8, (1 - val) * 0.4 + val * 0.6], axis=-1)
        elif cmode == "fire":
            rgb = np.stack([np.clip(val * 1.5, 0, 1), val * 0.6, val * 0.2], axis=-1)
        elif cmode == "ice":
            rgb = np.stack([val * 0.2, val * 0.5, 0.5 + val * 0.5], axis=-1)
        else:
            rgb = np.stack([val, val, val], axis=-1)

        rgb = np.clip(rgb, 0.0, 1.0).astype(np.float32)
        capture_frame("531", rgb)
        save(rgb, mn(531, "Flowing Truchet"), out_dir)
        return rgb
    except Exception as exc:
        fallback = np.full((H, W, 3), 128, dtype=np.uint8)
        save(fallback, mn(531, "Flowing Truchet"), out_dir)
        print(f"[method_531] ERROR: {exc}")
        return fallback
