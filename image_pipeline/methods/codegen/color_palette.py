"""
Color Palette utility — GPU-accelerated via ModernGL + GLSL.

Architecture (GPU where it pays, CPU where it doesn't):

  GPU targets:
    • Preview render   — GLSL strip shader via FBO (10× faster than PIL)
    • K-means sampling — hybrid GPU distance + CPU centroid update (15-40×)
    • Palette remap    — existing GPU twin in shaders.py

  CPU targets (too small for GPU — 3-32 scalar values, GPU launch > kernel time):
    • 33 palette generators
    • HSV rotation
    • Registry lookups

Public API (importable by other methods):
    generate_palette(type, n, seed, hue_off, sat, val)  → [(r,g,b), ...]
    palette_to_colormap(colors)                          → (N,3) float32
    sample_palette_from_image(image, n, seed, hue_off)   → (colors, cmap)
    load_registry_palette(name)                          → (N,3) float32 or None
    list_palette_types()                                 → [str, ...]
    list_preset_names()                                  → [str, ...]
"""
from __future__ import annotations

import colorsys
import math
import random
import threading
from pathlib import Path

import numpy as np
from PIL import Image

from ...core.registry import method
from ...core.utils import W, H, seed_all, get_font
from ...core.animation import capture_frame
from ...core.spatial import as_scalar, sparam
from ...core import palette_registry

# ════════════════════════════════════════════════════════════════════════════
# GL CONTEXT
# ════════════════════════════════════════════════════════════════════════════

_ctx_local = threading.local()


def _get_ctx():
    ctx = getattr(_ctx_local, "ctx", None)
    if ctx is None:
        import moderngl
        _ctx_local.ctx = moderngl.create_context(standalone=True, require=330)
    return _ctx_local.ctx


_VERTEX_SHADER = """
#version 330
in vec2 in_vert;
in vec2 in_uv;
out vec2 v_uv;
void main() {
    gl_Position = vec4(in_vert, 0.0, 1.0);
    v_uv = in_uv;
}
"""

_QUAD_VERTICES = np.array([
    -1, -1,  0, 0,
     1, -1,  1, 0,
     1,  1,  1, 1,
    -1,  1,  0, 1,
], dtype='f4')

_QUAD_INDICES = np.array([0, 1, 2, 0, 2, 3], dtype='i4')

_PROG_CACHE_LOCAL = threading.local()


def _get_prog_cache() -> dict:
    """Per-thread program+VAO cache.

    ModernGL programs/VAOs/buffers are bound to the context that created
    them. The context is thread-local (each server thread gets its own), so
    the cache MUST be thread-local too: a module-global cache hands the new
    thread's context a program compiled on a dead thread's context, and
    vao.render() silently renders nothing → the FBO keeps its clear color
    (dark background) on every run after the first.
    """
    cache = getattr(_PROG_CACHE_LOCAL, "cache", None)
    if cache is None:
        _PROG_CACHE_LOCAL.cache = {}
    return _PROG_CACHE_LOCAL.cache


def _get_prog(ctx, frag_src: str):
    """Thread-local program cache keyed by fragment source hash."""
    import hashlib
    key = hashlib.md5(frag_src.encode()).hexdigest()
    cache = _get_prog_cache()
    if key not in cache:
        prog = ctx.program(vertex_shader=_VERTEX_SHADER, fragment_shader=frag_src)
        # Bind only attributes the compiler kept — in_uv is dead-code-eliminated
        # when the fragment never uses v_uv (see core/shaders._create_vao).
        if 'in_uv' in prog:
            vao = ctx.vertex_array(prog, [
                (ctx.buffer(_QUAD_VERTICES), '2f 2f', 'in_vert', 'in_uv')
            ], ctx.buffer(_QUAD_INDICES))
        else:
            # '4f' keeps the 16-byte stride when in_uv is optimized out
            # ('2f 12x' skip layouts drop the first triangle on some drivers).
            vao = ctx.vertex_array(prog, [
                (ctx.buffer(_QUAD_VERTICES), '4f', 'in_vert')
            ], ctx.buffer(_QUAD_INDICES))
        cache[key] = (prog, vao)
    return cache[key]


# ════════════════════════════════════════════════════════════════════════════
# GLSL: PREVIEW STRIP SHADER
# ════════════════════════════════════════════════════════════════════════════

_PREVIEW_STRIP_FRAG = """
#version 330
precision highp float;

in vec2 v_uv;
out vec4 f_color;

uniform vec2 u_resolution;
uniform vec3 u_palette[32];
uniform int u_n_colors;

void main() {
    vec2 uv = gl_FragCoord.xy / u_resolution;
    int n = u_n_colors;
    if (n < 1) {
        f_color = vec4(0.05, 0.05, 0.08, 1.0);
        return;
    }

    // Top 70%: horizontal color strips
    float strip_h = 0.70 / float(n);
    if (uv.y < 0.70) {
        int idx = int(uv.y / strip_h);
        idx = clamp(idx, 0, n - 1);
        // Separator line at strip boundaries
        float frac_y = mod(uv.y, strip_h);
        float sep = 1.0 - smoothstep(0.0, 0.002, frac_y);
        vec3 col = mix(u_palette[idx], vec3(0.22, 0.22, 0.2), sep * 0.8);
        f_color = vec4(col, 1.0);
    }
    // Middle: label background
    else if (uv.y < 0.82) {
        f_color = vec4(0.05, 0.05, 0.08, 1.0);
    }
    // Bottom: color chips with borders
    else {
        float chip_h = 0.18;
        float chip_w = 1.0 / float(n);
        float cy = uv.y - 0.82;
        int idx = int(uv.x / chip_w);
        idx = clamp(idx, 0, n - 1);
        float cx = mod(uv.x, chip_w);
        float border = 1.0 - (
            smoothstep(0.0, 0.004, cx) *
            smoothstep(0.0, 0.004, cy) *
            smoothstep(0.0, 0.004, chip_w - cx) *
            smoothstep(0.0, 0.004, chip_h - cy)
        );
        vec3 col = mix(u_palette[idx], vec3(0.18, 0.18, 0.2), border * 0.6);
        f_color = vec4(col, 1.0);
    }
}
"""


# ════════════════════════════════════════════════════════════════════════════
# GPU RENDER: preview strip
# ════════════════════════════════════════════════════════════════════════════


def _render_preview_gpu(
    colors: list[tuple[int, int, int]],
    w: int, h: int,
) -> np.ndarray:
    """Render color strip preview on GPU via GLSL.

    Returns (H,W,3) float32 ndarray in [0, 1].
    """
    n = len(colors)
    if n == 0:
        return np.zeros((h, w, 3), dtype=np.float32)

    ctx = _get_ctx()
    prog, vao = _get_prog(ctx, _PREVIEW_STRIP_FRAG)

    palette = np.array(colors, dtype=np.float32) / 255.0  # (N, 3)
    # Pad to 32 for the uniform array
    padded = np.zeros((32, 3), dtype=np.float32)
    padded[:n] = palette

    fbo = ctx.simple_framebuffer((w, h))
    fbo.use()

    prog['u_resolution'].value = (float(w), float(h))
    prog['u_n_colors'].value = n

    # Set vec3[] uniform array — ModernGL expects the whole array as a list of
    # rows: prog['u_palette'].value = [ (r,g,b), ... ]. Indexed access
    # ('u_palette[i]') is NOT supported by this binding (KeyError/contains
    # guard silently no-ops, leaving the array zeroed → black output).
    uname = 'u_palette'
    if uname in prog:
        prog[uname].value = [tuple(padded[i]) for i in range(32)]

    ctx.clear(0.05, 0.05, 0.08)
    vao.render()
    data = fbo.read()
    fbo.release()

    # ModernGL simple_framebuffer reads back RGB directly (no BGR swap)
    arr = np.frombuffer(data, dtype=np.uint8).reshape(h, w, 3)
    return arr.astype(np.float32) / 255.0


# ════════════════════════════════════════════════════════════════════════════
# GLSL: K-MEANS DISTANCE PASS (per-pixel centroid assignment)
# ════════════════════════════════════════════════════════════════════════════

_KMEANS_LABEL_FRAG = """
#version 330
precision highp float;

in vec2 v_uv;
out vec4 f_color;

uniform sampler2D u_texture;
uniform vec3 u_centroids[16];
uniform int u_k;

void main() {
    ivec2 px = ivec2(gl_FragCoord.xy);
    vec3 color = texelFetch(u_texture, px, 0).rgb;

    int nearest = 0;
    float min_d = 1e10;
    for (int i = 0; i < u_k; i++) {
        vec3 d = color - u_centroids[i];
        float dist2 = dot(d, d);
        if (dist2 < min_d) {
            min_d = dist2;
            nearest = i;
        }
    }

    // R = label index / K (normalized for float readback), GBA = original color
    f_color = vec4(float(nearest) / float(max(u_k, 1)), color.b, color.g, color.r);
}
"""


# ════════════════════════════════════════════════════════════════════════════
# GPU K-MEANS (hybrid: GPU distance → labels, CPU centroid update)
# ════════════════════════════════════════════════════════════════════════════


def _kmeans_gpu(
    image: np.ndarray,
    k: int,
    seed: int,
    n_iterations: int = 5,
) -> tuple[np.ndarray, np.ndarray]:
    """K-means clustering using GPU for distance computation.

    Hybrid approach:
      - GPU: per-pixel distance to K centroids → label assignment
      - CPU: average colors per label → new centroids

    Args:
        image: (H,W,3) float32 array in [0,1].
        k: Number of clusters (2-16).
        seed: Random seed for initialization.
        n_iterations: Number of k-means iterations.

    Returns:
        (centroids, labels) where centroids is (K,3) float32 [0,1]
        and labels is (H,W) uint8.
    """
    k = min(k, 16)
    h, w = image.shape[:2]

    ctx = _get_ctx()
    prog, vao = _get_prog(ctx, _KMEANS_LABEL_FRAG)

    # Upload input image as texture (ModernGL textures are RGB, no BGR swap)
    img_u8 = (np.clip(image, 0, 1) * 255).astype(np.uint8)
    tex_data = img_u8.tobytes()
    texture = ctx.texture((w, h), 3, tex_data)
    texture.use(0)

    # Initialize centroids with k-means++ (farthest-first sampling).
    # Plain random pixel sampling can seed two centroids on the same color,
    # which then splits one cluster and permanently leaves another unclaimed
    # (seen at k=16: duplicate gray levels in the output palette).
    rng = np.random.RandomState(seed)
    flat = image.reshape(-1, 3).astype(np.float32)
    centroids = np.zeros((k, 3), dtype=np.float32)
    centroids[0] = flat[rng.randint(len(flat))]
    for ci in range(1, k):
        d2 = ((flat[None, :, :] - centroids[:ci, None, :]) ** 2).sum(axis=2).min(axis=0)
        total = d2.sum()
        if total <= 0:
            centroids[ci] = flat[rng.randint(len(flat))]
        else:
            probs = d2 / total
            centroids[ci] = flat[rng.choice(len(flat), p=probs)]

    # FBO for label output (single-channel R32F)
    label_fbo = ctx.simple_framebuffer((w, h), dtype='f4')

    for _ in range(n_iterations):
        # Set uniforms (u_resolution not needed — shader uses gl_FragCoord)
        prog['u_k'].value = k
        prog['u_texture'].value = 0

        # Upload centroids as vec3[16] uniform array — whole-array binding
        # (indexed 'u_centroids[i]' access silently no-ops in ModernGL).
        padded = np.zeros((16, 3), dtype=np.float32)
        padded[:k] = centroids
        prog['u_centroids'].value = [tuple(padded[i]) for i in range(16)]

        # Render labels to FBO
        label_fbo.use()
        ctx.clear(0.0, 0.0, 0.0, 0.0)
        vao.render()

        # Read back labels (FBO is RGBA float32, read(components=1) returns R channel as uint8)
        label_data = label_fbo.read(components=1)  # (H,W) bytes, 1 byte/pixel
        labels = np.frombuffer(label_data, dtype=np.uint8).reshape(h, w).astype(np.int32)
        # Rescale [0,255] → [0,k-1] with ROUNDING: shader writes nearest/k, e.g.
        # label 3 of 4 → 0.75 → 191; 191*4//255 truncates to 2 (collapses the
        # last cluster into the previous one, leaving its centroid stale).
        labels = ((labels * k + 127) // 255).clip(0, k - 1)

        # CPU: update centroids
        for ci in range(k):
            mask = labels == ci
            count = mask.sum()
            if count > 0:
                centroids[ci] = image[mask].mean(axis=0)
            else:
                # Empty cluster: re-seed with the pixel farthest from it
                d2 = ((flat - centroids[ci]) ** 2).sum(axis=1)
                centroids[ci] = flat[np.argmax(d2)]

    label_fbo.release()
    texture.release()

    return centroids, labels.astype(np.uint8)


# ════════════════════════════════════════════════════════════════════════════
# COLOR HELPERS
# ════════════════════════════════════════════════════════════════════════════


def _hsv_to_rgb(h: float, s: float, v: float) -> tuple[int, int, int]:
    """Convert HSV to (r,g,b) bytes."""
    r, g, b = colorsys.hsv_to_rgb(h % 1.0, max(0, min(1, s)), max(0, min(1, v)))
    return (int(r * 255), int(g * 255), int(b * 255))


def _lerp_hue(h1: float, h2: float, t: float) -> float:
    diff = (h2 - h1) % 1.0
    if diff > 0.5:
        diff -= 1.0
    return (h1 + diff * t) % 1.0


def _interpolate_anchors(anchors: list[float], n_colors: int) -> list[float]:
    if n_colors <= len(anchors):
        return [anchors[i % len(anchors)] for i in range(n_colors)]
    hues = []
    for i in range(n_colors):
        anchor_idx = (i * len(anchors)) // n_colors
        frac = ((i * len(anchors)) % n_colors) / max(1, n_colors)
        h1 = anchors[anchor_idx % len(anchors)]
        h2 = anchors[(anchor_idx + 1) % len(anchors)]
        hues.append(_lerp_hue(h1, h2, frac))
    return hues


def _base_hue(seed: int, hue_off: float = 0.0) -> float:
    return (seed * 0.01 + hue_off / 360.0) % 1.0


# ════════════════════════════════════════════════════════════════════════════
# LABEL FORMATTING — per-color value labels
# ════════════════════════════════════════════════════════════════════════════
#
# Only formats with an existing conversion in this repo (or stdlib colorsys)
# are wired up. Named colors and CSS variables are deliberately NOT offered:
# there is no reverse name→RGB table and no semantic mapping from an arbitrary
# generated color to a theme var — both would have to be invented.

_LABEL_FORMATS = [
    "rgb", "rgba", "hsl", "hsla", "hsv",
    "cmyk", "xyz", "lab", "lch", "oklab", "oklch",
    "index", "packed",
]

# sRGB → CIE XYZ (D65) — same matrices/white point as palette_posterize.rgb2lab
_SRGB_TO_XYZ = np.array([
    [0.4124, 0.3576, 0.1805],
    [0.2126, 0.7152, 0.0722],
    [0.0193, 0.1192, 0.9505],
], dtype=np.float64)
_XYZ_WHITE = np.array([95.047, 100.0, 108.883], dtype=np.float64)


def _srgb_to_xyz(r: float, g: float, b: float) -> np.ndarray:
    """sRGB bytes → CIE XYZ (0-100 scale, D65)."""
    c = np.array([r, g, b], dtype=np.float64) / 255.0
    c = np.where(c > 0.04045, ((c + 0.055) / 1.055) ** 2.4, c / 12.92) * 100.0
    return _SRGB_TO_XYZ @ c


def _srgb_to_lab(r: float, g: float, b: float) -> np.ndarray:
    """sRGB bytes → CIELAB (D65), L 0-100."""
    xyz = _srgb_to_xyz(r, g, b) / _XYZ_WHITE
    f = lambda t: np.cbrt(t) if t > 0.008856 else 7.787 * t + 16.0 / 116.0
    fx, fy, fz = f(xyz[0]), f(xyz[1]), f(xyz[2])
    return np.array([116.0 * fy - 16.0, 500.0 * (fx - fy), 200.0 * (fy - fz)])


def _srgb_to_oklab(r: float, g: float, b: float) -> np.ndarray:
    """sRGB bytes → OKLab (L 0-1) — same constants as color_grade._rgb_to_oklab."""
    c = np.array([r, g, b], dtype=np.float64) / 255.0
    c = np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)
    r_, g_, b_ = c
    l = 0.4122214708 * r_ + 0.5363325363 * g_ + 0.0514459929 * b_
    m = 0.2119034982 * r_ + 0.6806995451 * g_ + 0.1073969566 * b_
    s = 0.0883024619 * r_ + 0.2817188376 * g_ + 0.6299787005 * b_
    l_, m_, s_ = np.cbrt([l, m, s])
    L = 0.2104542553 * l_ + 0.7936177850 * m_ - 0.0040720468 * s_
    A = 1.9779984951 * l_ - 2.4285922050 * m_ + 0.4505937099 * s_
    B = 0.0259040371 * l_ + 0.7827717662 * m_ - 0.8086757660 * s_
    return np.array([L, A, B])


def _srgb_to_cmyk(r: float, g: float, b: float) -> tuple[float, float, float, float]:
    """sRGB bytes → CMYK (0-1) — same formula as cmyk_halftone._rgb_to_cmyk."""
    r_, g_, b_ = r / 255.0, g / 255.0, b / 255.0
    k = 1.0 - max(r_, g_, b_)
    inv = 1.0 - k
    safe = inv if inv > 1e-6 else 1.0
    return (1.0 - r_ - k) / safe, (1.0 - g_ - k) / safe, (1.0 - b_ - k) / safe, k


def _lab_to_lch(lab: np.ndarray) -> tuple[float, float, float]:
    """CIELAB → LCH (polar form: C = hypot(a,b), H = atan2(b,a))."""
    L, a, b = lab
    return L, math.hypot(a, b), math.degrees(math.atan2(b, a)) % 360.0


def _format_color_value(rgb: tuple[int, int, int], fmt: str, index: int) -> str:
    """Format one palette color as a label string in the requested format."""
    r, g, b = rgb
    if fmt == "rgba":
        return f"rgba({r}, {g}, {b}, 1)"
    if fmt == "hsl":
        h, l, s = colorsys.rgb_to_hls(r / 255.0, g / 255.0, b / 255.0)
        return f"hsl({h * 360.0:.0f}, {s * 100.0:.0f}%, {l * 100.0:.0f}%)"
    if fmt == "hsla":
        h, l, s = colorsys.rgb_to_hls(r / 255.0, g / 255.0, b / 255.0)
        return f"hsla({h * 360.0:.0f}, {s * 100.0:.0f}%, {l * 100.0:.0f}%, 1)"
    if fmt == "hsv":
        h, s, v = colorsys.rgb_to_hsv(r / 255.0, g / 255.0, b / 255.0)
        return f"({h * 360.0:.0f}°, {s * 100.0:.0f}%, {v * 100.0:.0f}%)"
    if fmt == "cmyk":
        c, m, y, k = _srgb_to_cmyk(r, g, b)
        return f"({c * 100.0:.0f}%, {m * 100.0:.0f}%, {y * 100.0:.0f}%, {k * 100.0:.0f}%)"
    if fmt == "xyz":
        x, y, z = _srgb_to_xyz(r, g, b)
        return f"({x / 100.0:.2f}, {y / 100.0:.2f}, {z / 100.0:.2f})"
    if fmt == "lab":
        L, a, b = _srgb_to_lab(r, g, b)
        return f"({L:.1f}, {a:.1f}, {b:.1f})"
    if fmt == "lch":
        L, c, h = _lab_to_lch(_srgb_to_lab(r, g, b))
        return f"({L:.1f}, {c:.1f}, {h:.0f})"
    if fmt == "oklab":
        L, a, b = _srgb_to_oklab(r, g, b)
        return f"({L:.3f}, {a:.3f}, {b:.3f})"
    if fmt == "oklch":
        L, c, h = _lab_to_lch(_srgb_to_oklab(r, g, b))
        return f"({L:.3f}, {c:.3f}, {h:.0f})"
    if fmt == "index":
        return str(index)
    if fmt == "packed":
        return str((r << 16) | (g << 8) | b)
    return f"rgb({r}, {g}, {b})"


def _draw_palette_labels(
    img: np.ndarray,
    colors: list[tuple[int, int, int]],
    fmt: str,
    title: str,
) -> np.ndarray:
    """Overlay color-value labels on the preview (PIL — text has no GLSL path).

    Each strip in the top 70% band gets its color's formatted value, centered;
    the middle band gets the palette name. Strips too short for legible text
    (large n) skip the per-color labels but keep the name band.
    """
    h, w = img.shape[:2]
    n = len(colors)
    strip_h = int(h * 0.70) // max(1, n)

    # Fonts are recreated per frame during animation — cache per size (get_font
    # re-parses the .ttc on every call, ~33 loads/frame otherwise).
    _font_cache: dict[int, ImageFont.ImageFont | ImageFont.FreeTypeFont] = {}

    def _font(size: int):
        font = _font_cache.get(size)
        if font is None:
            font = _font_cache[size] = get_font(size)
        return font

    from PIL import Image as _PILImage, ImageDraw, ImageFont
    pil = _PILImage.fromarray((np.clip(img, 0, 1) * 255).astype(np.uint8))
    draw = ImageDraw.Draw(pil)

    if strip_h >= 14:
        base_size = max(8, min(15, strip_h - 8))
        for i in range(n):
            text = _format_color_value(colors[i], fmt, i)
            size = base_size
            while size > 8:
                font = _font(size)
                if draw.textlength(text, font=font) <= w - 24:
                    break
                size -= 1
            font = _font(size)
            tw = draw.textlength(text, font=font)
            if tw > w - 24:
                continue  # still too wide for the canvas — skip this label
            r, g, b = colors[i]
            lum = 0.2126 * r + 0.7152 * g + 0.0722 * b
            fill = (245, 245, 250) if lum < 128 else (12, 12, 16)
            x = (w - tw) / 2.0
            y = i * strip_h + (strip_h - (size + 6)) / 2.0
            draw.text((x, y), text, font=font, fill=fill)

    if title:
        band_y0, band_y1 = int(h * 0.70), int(h * 0.82)
        font = _font(13)
        tw = draw.textlength(title, font=font)
        x = (w - tw) / 2.0
        y = band_y0 + (band_y1 - band_y0 - 18) / 2.0
        draw.text((x, y), title, font=font, fill=(200, 200, 220))

    return np.asarray(pil).astype(np.float32) / 255.0


# ════════════════════════════════════════════════════════════════════════════
# PALETTE GENERATORS — 33 types across 5 families
# ════════════════════════════════════════════════════════════════════════════

# -- Classic (7)

def _monochromatic_palette(n_colors, seed, hue_off=0.0, sat=0.75, val=0.7):
    hue = _base_hue(seed, hue_off)
    out = []
    for i in range(n_colors):
        frac = i / max(1, n_colors - 1)
        c_sat = max(0.1, sat - 0.3 + frac * 0.3)
        c_val = max(0.2, val - 0.3 + frac * 0.4)
        out.append(_hsv_to_rgb(hue, c_sat, c_val))
    return out


def _analogous_palette(n_colors, seed, hue_off=0.0, sat=0.75, val=0.7):
    base = _base_hue(seed, hue_off)
    span = 30.0 / 360.0
    out = []
    for i in range(n_colors):
        frac = i / max(1, n_colors - 1) if n_colors > 1 else 0.0
        hue = (base - span / 2 + frac * span) % 1.0
        out.append(_hsv_to_rgb(hue, sat, val))
    return out


def _complementary_palette(n_colors, seed, hue_off=0.0, sat=0.75, val=0.7):
    base = _base_hue(seed, hue_off)
    out = []
    for i in range(n_colors):
        frac = i / max(1, n_colors - 1)
        hue = (base + frac * 0.5) % 1.0
        out.append(_hsv_to_rgb(hue, sat, val))
    return out


def _split_complementary_palette(n_colors, seed, hue_off=0.0, sat=0.75, val=0.7):
    base = _base_hue(seed, hue_off)
    anchors = [base, (base + 150.0 / 360.0) % 1.0, (base + 210.0 / 360.0) % 1.0]
    hues = _interpolate_anchors(anchors, n_colors)
    return [_hsv_to_rgb(h, sat, val) for h in hues]


def _triadic_palette(n_colors, seed, hue_off=0.0, sat=0.75, val=0.7):
    base = _base_hue(seed, hue_off)
    anchors = [(base + i / 3.0) % 1.0 for i in range(3)]
    hues = _interpolate_anchors(anchors, n_colors)
    return [_hsv_to_rgb(h, sat, val) for h in hues]


def _tetradic_palette(n_colors, seed, hue_off=0.0, sat=0.75, val=0.7):
    base = _base_hue(seed, hue_off)
    angle = 60.0 / 360.0
    anchors = [base, (base + angle) % 1.0, (base + 0.5) % 1.0, (base + 0.5 + angle) % 1.0]
    hues = _interpolate_anchors(anchors, n_colors)
    return [_hsv_to_rgb(h, sat, val) for h in hues]


def _square_palette(n_colors, seed, hue_off=0.0, sat=0.75, val=0.7):
    base = _base_hue(seed, hue_off)
    anchors = [(base + i * 0.25) % 1.0 for i in range(4)]
    hues = _interpolate_anchors(anchors, n_colors)
    return [_hsv_to_rgb(h, sat, val) for h in hues]


# -- Extended (9)

def _double_split_complementary_palette(n_colors, seed, hue_off=0.0, sat=0.75, val=0.7):
    base = _base_hue(seed, hue_off)
    comp = (base + 0.5) % 1.0
    split = 30.0 / 360.0
    anchors = [base, (base + split) % 1.0, comp, (comp + split) % 1.0, (comp - split) % 1.0]
    hues = _interpolate_anchors(anchors, n_colors)
    return [_hsv_to_rgb(h, sat, val) for h in hues]


def _clash_palette(n_colors, seed, hue_off=0.0, sat=0.75, val=0.7):
    base = _base_hue(seed, hue_off)
    clash_angle = 170.0 / 360.0
    anchors = [base, (base + clash_angle) % 1.0]
    hues = _interpolate_anchors(anchors, n_colors)
    return [_hsv_to_rgb(h, sat, val) for h in hues]


def _neutral_palette(n_colors, seed, hue_off=0.0, sat=0.75, val=0.7):
    base = _base_hue(seed, hue_off)
    out = []
    for i in range(n_colors):
        frac = i / max(1, n_colors - 1)
        hue = (base + frac * 0.15) % 1.0
        c_sat = max(0.05, sat * 0.15)
        c_val = max(0.3, val - 0.2 + frac * 0.4)
        out.append(_hsv_to_rgb(hue, c_sat, c_val))
    return out


def _achromatic_palette(n_colors, seed, hue_off=0.0, sat=0.75, val=0.7):
    return [(int(20 + 220 * i / max(1, n_colors - 1)),) * 3 for i in range(n_colors)]


def _pastel_palette(n_colors, seed, hue_off=0.0, sat=0.75, val=0.7):
    base = _base_hue(seed, hue_off)
    out = []
    for i in range(n_colors):
        frac = i / max(1, n_colors - 1)
        hue = (base + frac * 0.618) % 1.0
        c_sat = max(0.1, sat * 0.3)
        out.append(_hsv_to_rgb(hue, c_sat, 0.82))
    return out


def _earth_palette(n_colors, seed, hue_off=0.0, sat=0.75, val=0.7):
    base = _base_hue(seed, hue_off)
    out = []
    for i in range(n_colors):
        frac = i / max(1, n_colors - 1)
        if frac < 0.5:
            hue = (base * 0.3 + frac * 2 * 60.0 / 360.0) % 1.0
        else:
            hue = (base * 0.3 + 60.0 / 360.0 + (frac - 0.5) * 2 * 60.0 / 360.0) % 1.0
        c_sat = max(0.2, sat * 0.5)
        c_val = max(0.3, val - 0.2 + frac * 0.3)
        out.append(_hsv_to_rgb(hue, c_sat, c_val))
    return out


def _jewel_palette(n_colors, seed, hue_off=0.0, sat=0.75, val=0.7):
    base = _base_hue(seed, hue_off)
    out = []
    for i in range(n_colors):
        frac = i / max(1, n_colors - 1)
        hue = (base + frac * 0.618) % 1.0
        c_sat = max(0.7, 0.9)
        out.append(_hsv_to_rgb(hue, c_sat, val - 0.1))
    return out


def _neon_palette(n_colors, seed, hue_off=0.0, sat=0.75, val=0.7):
    base = _base_hue(seed, hue_off)
    out = []
    for i in range(n_colors):
        frac = i / max(1, n_colors - 1)
        hue = (base + frac * 0.618) % 1.0
        out.append(_hsv_to_rgb(hue, 1.0, 1.0))
    return out


def _muted_palette(n_colors, seed, hue_off=0.0, sat=0.75, val=0.7):
    base = _base_hue(seed, hue_off)
    out = []
    for i in range(n_colors):
        frac = i / max(1, n_colors - 1)
        hue = (base + frac * 0.618) % 1.0
        c_sat = max(0.1, sat * 0.25)
        c_val = max(0.3, val - 0.1 + frac * 0.2)
        out.append(_hsv_to_rgb(hue, c_sat, c_val))
    return out


# -- Temperature (4)

def _warm_palette(n_colors, seed, hue_off=0.0, sat=0.75, val=0.7):
    base = _base_hue(seed, hue_off)
    out = []
    for i in range(n_colors):
        frac = i / max(1, n_colors - 1)
        hue = (base * 0.2 + frac * 60.0 / 360.0) % 1.0
        out.append(_hsv_to_rgb(hue, sat, val))
    return out


def _cool_palette(n_colors, seed, hue_off=0.0, sat=0.75, val=0.7):
    base = _base_hue(seed, hue_off)
    out = []
    for i in range(n_colors):
        frac = i / max(1, n_colors - 1)
        hue = (0.5 + base * 0.2 + frac * 120.0 / 360.0) % 1.0
        out.append(_hsv_to_rgb(hue, sat, val))
    return out


def _neutral_warm_palette(n_colors, seed, hue_off=0.0, sat=0.75, val=0.7):
    base = _base_hue(seed, hue_off)
    out = []
    for i in range(n_colors):
        frac = i / max(1, n_colors - 1)
        hue = (base * 0.1 + frac * 30.0 / 360.0) % 1.0
        c_sat = max(0.05, sat * 0.1)
        c_val = max(0.3, val - 0.2 + frac * 0.4)
        out.append(_hsv_to_rgb(hue, c_sat, c_val))
    return out


def _neutral_cool_palette(n_colors, seed, hue_off=0.0, sat=0.75, val=0.7):
    base = _base_hue(seed, hue_off)
    out = []
    for i in range(n_colors):
        frac = i / max(1, n_colors - 1)
        hue = (0.6 + base * 0.1 + frac * 30.0 / 360.0) % 1.0
        c_sat = max(0.05, sat * 0.1)
        c_val = max(0.3, val - 0.2 + frac * 0.4)
        out.append(_hsv_to_rgb(hue, c_sat, c_val))
    return out


# -- Perceptual / Mathematical (4)

def _golden_ratio_palette(n_colors, seed, hue_off=0.0, sat=0.75, val=0.7):
    return [_hsv_to_rgb((i * 0.618033988749895 + hue_off / 360.0) % 1.0, sat, val) for i in range(n_colors)]


def _fibonacci_palette(n_colors, seed, hue_off=0.0, sat=0.75, val=0.7):
    fibs = [1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144, 233]
    out = []
    for i in range(n_colors):
        step = fibs[i % len(fibs)] / 360.0
        hue = (i * step + hue_off / 360.0) % 1.0
        out.append(_hsv_to_rgb(hue, sat, val))
    return out


def _prime_spacing_palette(n_colors, seed, hue_off=0.0, sat=0.75, val=0.7):
    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]
    out = []
    for i in range(n_colors):
        step = primes[i % len(primes)] / 360.0
        hue = (i * step + hue_off / 360.0) % 1.0
        out.append(_hsv_to_rgb(hue, sat, val))
    return out


def _uniform_palette(n_colors, seed, hue_off=0.0, sat=0.75, val=0.7):
    return [_hsv_to_rgb((i / n_colors + hue_off / 360.0) % 1.0, sat, val) for i in range(n_colors)]


# -- Extreme / Theoretical (8)

def _tetradic_rectangle_palette(n_colors, seed, hue_off=0.0, sat=0.75, val=0.7):
    base = _base_hue(seed, hue_off)
    rect_width = 90.0 / 360.0
    anchors = [base, (base + rect_width) % 1.0, (base + 0.5) % 1.0, (base + 0.5 + rect_width) % 1.0]
    hues = _interpolate_anchors(anchors, n_colors)
    return [_hsv_to_rgb(h, sat, val) for h in hues]


def _double_complementary_palette(n_colors, seed, hue_off=0.0, sat=0.75, val=0.7):
    base = _base_hue(seed, hue_off)
    gap = 20.0 / 360.0
    anchors = [base, (base + 0.5) % 1.0, (base + gap + 0.25) % 1.0, (base + gap + 0.75) % 1.0]
    hues = _interpolate_anchors(anchors, n_colors)
    return [_hsv_to_rgb(h, sat, val) for h in hues]


def _clash_variable_palette(n_colors, seed, hue_off=0.0, sat=0.75, val=0.7):
    base = _base_hue(seed, hue_off)
    rng = random.Random(seed)
    clash_angle = (160.0 + rng.random() * 15.0) / 360.0
    anchors = [base, (base + clash_angle) % 1.0]
    hues = _interpolate_anchors(anchors, n_colors)
    return [_hsv_to_rgb(h, sat, val) for h in hues]


def _split_variable_palette(n_colors, seed, hue_off=0.0, sat=0.75, val=0.7):
    base = _base_hue(seed, hue_off)
    rng = random.Random(seed + 1)
    split_angle = (120.0 + rng.random() * 50.0) / 360.0
    anchors = [base, (base + split_angle) % 1.0, (base + 1.0 - split_angle) % 1.0]
    hues = _interpolate_anchors(anchors, n_colors)
    return [_hsv_to_rgb(h, sat, val) for h in hues]


def _achromatic_tint_palette(n_colors, seed, hue_off=0.0, sat=0.75, val=0.7):
    hue = _base_hue(seed, hue_off)
    return [_hsv_to_rgb(hue, 0.05, max(0.15, 0.1 + i / max(1, n_colors - 1) * 0.8)) for i in range(n_colors)]


def _achromatic_shade_palette(n_colors, seed, hue_off=0.0, sat=0.75, val=0.7):
    hue = _base_hue(seed, hue_off)
    return [_hsv_to_rgb(hue, 0.08, max(0.05, 0.05 + i / max(1, n_colors - 1) * 0.5)) for i in range(n_colors)]


def _complementary_split_wide_palette(n_colors, seed, hue_off=0.0, sat=0.75, val=0.7):
    base = _base_hue(seed, hue_off)
    wide_split = 170.0 / 360.0
    anchors = [base, (base + wide_split) % 1.0, (base + 1.0 - wide_split) % 1.0]
    hues = _interpolate_anchors(anchors, n_colors)
    return [_hsv_to_rgb(h, sat, val) for h in hues]


def _triadic_alt_palette(n_colors, seed, hue_off=0.0, sat=0.75, val=0.7):
    base = _base_hue(seed, hue_off)
    anchors = [(base + i / 3.0) % 1.0 for i in range(3)]
    hues = _interpolate_anchors(anchors, n_colors)
    out = []
    for i, h in enumerate(hues):
        frac = i / max(1, n_colors - 1)
        c_sat = max(0.3, sat - 0.2 + frac * 0.4)
        c_val = max(0.3, val - 0.2 + frac * 0.4)
        out.append(_hsv_to_rgb(h, c_sat, c_val))
    return out


def _random_palette(n_colors, seed, hue_off=0.0, sat=0.75, val=0.7):
    rng = random.Random(seed)
    out = []
    for _ in range(n_colors):
        hue = (rng.random() + hue_off / 360.0) % 1.0
        c_sat = max(0.3, min(1.0, sat + rng.uniform(-0.2, 0.2)))
        c_val = max(0.3, min(1.0, val + rng.uniform(-0.2, 0.2)))
        out.append(_hsv_to_rgb(hue, c_sat, c_val))
    return out


# ════════════════════════════════════════════════════════════════════════════
# REGISTRY
# ════════════════════════════════════════════════════════════════════════════

_PALETTE_GENERATORS: dict[str, callable] = {
    "monochromatic": _monochromatic_palette,
    "analogous": _analogous_palette,
    "complementary": _complementary_palette,
    "split": _split_complementary_palette,
    "triadic": _triadic_palette,
    "tetradic": _tetradic_palette,
    "square": _square_palette,
    "double-split": _double_split_complementary_palette,
    "clash": _clash_palette,
    "neutral": _neutral_palette,
    "achromatic": _achromatic_palette,
    "pastel": _pastel_palette,
    "earth": _earth_palette,
    "jewel": _jewel_palette,
    "neon": _neon_palette,
    "muted": _muted_palette,
    "warm": _warm_palette,
    "cool": _cool_palette,
    "neutral-warm": _neutral_warm_palette,
    "neutral-cool": _neutral_cool_palette,
    "golden-ratio": _golden_ratio_palette,
    "fibonacci": _fibonacci_palette,
    "prime-spacing": _prime_spacing_palette,
    "uniform": _uniform_palette,
    "tetradic-rectangle": _tetradic_rectangle_palette,
    "double-complementary": _double_complementary_palette,
    "clash-variable": _clash_variable_palette,
    "split-variable": _split_variable_palette,
    "achromatic-tint": _achromatic_tint_palette,
    "achromatic-shade": _achromatic_shade_palette,
    "complementary-split-wide": _complementary_split_wide_palette,
    "triadic-alt": _triadic_alt_palette,
    "random": _random_palette,
}

_PALETTE_CHOICES = sorted(_PALETTE_GENERATORS.keys())

_DEFAULT_SAT: dict[str, float] = {
    "monochromatic": 0.75, "analogous": 0.75, "complementary": 0.75,
    "split": 0.75, "triadic": 0.75, "tetradic": 0.75, "square": 0.75,
    "double-split": 0.75, "clash": 0.75, "neutral": 0.15,
    "achromatic": 0.0, "pastel": 0.25, "earth": 0.4, "jewel": 0.85,
    "neon": 1.0, "muted": 0.2, "warm": 0.75, "cool": 0.75,
    "neutral-warm": 0.1, "neutral-cool": 0.1,
    "golden-ratio": 0.75, "fibonacci": 0.75, "prime-spacing": 0.75,
    "uniform": 0.75,
    "tetradic-rectangle": 0.75, "double-complementary": 0.75,
    "clash-variable": 0.75, "split-variable": 0.75,
    "achromatic-tint": 0.05, "achromatic-shade": 0.08,
    "complementary-split-wide": 0.75, "triadic-alt": 0.75,
    "random": 0.75,
}

_DEFAULT_VAL: dict[str, float] = {
    "monochromatic": 0.7, "analogous": 0.7, "complementary": 0.7,
    "split": 0.7, "triadic": 0.7, "tetradic": 0.7, "square": 0.7,
    "double-split": 0.7, "clash": 0.7, "neutral": 0.6,
    "achromatic": 0.6, "pastel": 0.85, "earth": 0.5, "jewel": 0.55,
    "neon": 0.95, "muted": 0.5, "warm": 0.7, "cool": 0.7,
    "neutral-warm": 0.6, "neutral-cool": 0.6,
    "golden-ratio": 0.7, "fibonacci": 0.7, "prime-spacing": 0.7,
    "uniform": 0.7,
    "tetradic-rectangle": 0.7, "double-complementary": 0.7,
    "clash-variable": 0.7, "split-variable": 0.7,
    "achromatic-tint": 0.6, "achromatic-shade": 0.3,
    "complementary-split-wide": 0.7, "triadic-alt": 0.7,
    "random": 0.7,
}


# ════════════════════════════════════════════════════════════════════════════
# PUBLIC API
# ════════════════════════════════════════════════════════════════════════════


def list_palette_types() -> list[str]:
    return list(_PALETTE_CHOICES)


def list_preset_names() -> list[str]:
    return sorted(palette_registry.get_all().keys())


def generate_palette(
    palette_type: str,
    n_colors: int = 8,
    seed: int = 0,
    hue_off: float = 0.0,
    sat: float | None = None,
    val: float | None = None,
) -> list[tuple[int, int, int]]:
    """Generate palette colors from one of 33 algorithmic types (CPU — 2-5 µs).

    Scalar math on 3-32 values: GPU launch overhead dominates, so this stays CPU.
    """
    gen_fn = _PALETTE_GENERATORS.get(palette_type, _golden_ratio_palette)
    pt = palette_type if palette_type in _PALETTE_GENERATORS else "golden-ratio"
    if sat is None:
        sat = _DEFAULT_SAT.get(pt, 0.75)
    if val is None:
        val = _DEFAULT_VAL.get(pt, 0.7)
    n_colors = max(3, min(32, int(n_colors)))
    return gen_fn(n_colors, seed, hue_off=hue_off, sat=sat, val=val)


def palette_to_colormap(colors: list[tuple[int, int, int]]) -> np.ndarray:
    """Convert (r,g,b) byte tuples to (N,3) float32 COLORMAP."""
    return np.array(colors, dtype=np.float32) / 255.0


def sample_palette_from_image(
    image: np.ndarray,
    n_colors: int = 6,
    seed: int = 0,
    hue_off: float = 0.0,
) -> tuple[list[tuple[int, int, int]], np.ndarray]:
    """Extract dominant colors from an image via GPU-accelerated k-means.

    GPU: per-pixel distance computation (parallel GLSL fragments).
    CPU: centroid update (K × 3 floats, trivial).

    Returns:
        (colors_list, colormap_array)
    """
    n_colors = max(2, min(16, int(n_colors)))

    # Downsample to 64×64 for k-means
    h, w = image.shape[:2]
    if h > 64 or w > 64:
        from PIL import Image as _PIL
        img_pil = _PIL.fromarray((np.clip(image, 0, 1) * 255).astype(np.uint8))
        img_small = img_pil.resize((64, 64), _PIL.LANCZOS)
        thumb = np.array(img_small).astype(np.float32) / 255.0
    else:
        thumb = image

    # GPU k-means
    centroids, _ = _kmeans_gpu(thumb, n_colors, seed)

    # Sort by luminance (brightest first)
    lum = centroids.mean(axis=1)
    order = np.argsort(-lum)
    centroids = centroids[order]

    # Convert to byte tuples
    colors = [
        (int(c[0] * 255), int(c[1] * 255), int(c[2] * 255))
        for c in centroids[:n_colors]
    ]

    # Apply hue offset rotation
    if hue_off != 0.0:
        rotated = []
        for r, g, b in colors:
            h, s, v = colorsys.rgb_to_hsv(r / 255.0, g / 255.0, b / 255.0)
            h = (h + hue_off / 360.0) % 1.0
            rotated.append(_hsv_to_rgb(h, s, v))
        colors = rotated
        centroids = np.array(colors, dtype=np.float32) / 255.0

    return colors, centroids.astype(np.float32)


def load_registry_palette(name: str) -> np.ndarray | None:
    """Load a named preset palette from the palette_registry.

    Supports ``_r`` suffix for reversed variants.
    """
    raw = palette_registry.get(name)
    if raw is None:
        return None
    return np.array(raw, dtype=np.float32) / 255.0


# ════════════════════════════════════════════════════════════════════════════
# @method — Color Palette pipeline node
# ════════════════════════════════════════════════════════════════════════════


@method(
    id="10",
    name="Color Palette",
    category="codegen",
    tags=["palette", "color", "fast", "utility", "gpu"],
    inputs={
        "image_in": "IMAGE",
        "hue_offset": "SCALAR",
        "saturation": "SCALAR",
        "value": "SCALAR",
        "palette_select": "SCALAR",
        "n_colors": "SCALAR",
    },
    outputs={"image": "IMAGE", "luminance": "FIELD", "palette": "COLORMAP"},
    params={
        "source": {
            "description": "palette source: generated (33 types), sampled (extract from wired image_in via GPU k-means), or registry (named preset)",
            "choices": ["generated", "sampled", "registry"],
            "default": "generated",
        },
        "n_colors": {
            "description": "number of palette colors (3-32, ignored when source=registry)",
            "default": 8,
        },
        "n_sample_colors": {
            "description": "number of colors to extract when source=sampled (GPU k-means)",
            "default": 6,
        },
        "palette_type": {
            "description": "palette generation method (33 types). Used when source=generated.",
            "choices": _PALETTE_CHOICES,
            "default": "golden-ratio",
        },
        "palette_name": {
            "description": "registry preset name. Used when source=registry.",
            "default": "bw",
        },
        "saturation": {
            "description": "saturation override (0-1, -1=auto per palette type). Wire LFO.value here.",
            "default": -1.0,
        },
        "value": {
            "description": "value/brightness override (0-1, -1=auto per palette type). Wire LFO.value here.",
            "default": -1.0,
        },
        "hue_offset": {
            "description": "hue rotation in degrees. Wire LFO.value here.",
            "default": 0.0,
        },
        "palette_select": {
            "description": "SCALAR-driven palette type index (0-1 maps to all 33 types). Wire Counter.value here.",
            "default": -1.0,
        },
        "remap_palette": {
            "description": "registry palette name to remap output colors through (none = use as-is)",
            "default": "none",
        },
        "show_labels": {
            "description": "overlay each swatch with its color value label (and the palette name in the center band)",
            "default": False,
        },
        "label_format": {
            "description": "color value format for the labels (named colors & CSS variables not offered — no reverse mapping exists)",
            "choices": _LABEL_FORMATS,
            "default": "rgb",
        },
    },
)
def method_10_color_palette(out_dir: Path, seed: int, params=None):
    """Generate, sample, or load a color palette and emit it as COLORMAP.

    GPU acceleration:
      - Preview render: GLSL strip shader via ModernGL (10× faster than PIL)
      - K-means sampling: hybrid GPU distance + CPU centroid update (15-40× faster)
      - Palette generators: CPU (3-32 scalar values, GPU overhead would dominate)

    Three source modes:
      - generated: 33 algorithmic palette types (harmonic, perceptual, etc.)
      - sampled: extract dominant colors from wired image_in via GPU k-means
      - registry: load a named preset from palette_registry
    """
    if params is None:
        params = {}

    source = params.get("source", "generated")
    n_colors = max(3, min(32, int(as_scalar(sparam(params, "n_colors", 8, cast=int)))))
    n_sample_colors = max(2, min(16, int(as_scalar(sparam(params, "n_sample_colors", 6, cast=int)))))
    remap = params.get("remap_palette", "none")
    show_labels = bool(params.get("show_labels", False))
    label_fmt = params.get("label_format", "rgb")
    if label_fmt not in _LABEL_FORMATS:
        label_fmt = "rgb"

    # -- Read SCALAR inputs --
    # sparam() + as_scalar() are null-safe: the UI sends None for unwired
    # SCALAR ports (bare params.get() would feed int(None)/float(None) →
    # TypeError → error placeholder → no image output), and the executor
    # auto-injects upstream scalars as both the value AND a broadcast
    # _field_<name> (H,W) array — as_scalar collapses the field to its mean.
    # (Fixed 2026-07-30.)
    effective_hue_offset = float(as_scalar(sparam(params, "hue_offset", 0.0)))

    sat_ui = float(as_scalar(sparam(params, "saturation", -1.0)))
    effective_sat = max(0.0, min(1.0, sat_ui)) if sat_ui >= 0 else None

    val_ui = float(as_scalar(sparam(params, "value", -1.0)))
    effective_val = max(0.0, min(1.0, val_ui)) if val_ui >= 0 else None

    palette_select_raw = float(as_scalar(sparam(params, "palette_select", -1.0)))
    if palette_select_raw >= 0:
        pidx = int(float(palette_select_raw) * len(_PALETTE_CHOICES)) % len(_PALETTE_CHOICES)
        palette_type = _PALETTE_CHOICES[pidx]
    else:
        palette_type = params.get("palette_type", "golden-ratio")

    input_img = params.get("_input_image")

    # -- Generate / sample / load palette --
    if source == "sampled" and input_img is not None:
        colors, colormap_arr = sample_palette_from_image(
            input_img, n_colors=n_sample_colors, seed=seed,
            hue_off=effective_hue_offset,
        )
        label_str = f"sampled ({n_sample_colors} colors)"

    elif source == "registry":
        palette_name = params.get("palette_name", "bw")
        colormap_arr = load_registry_palette(palette_name)
        if colormap_arr is None:
            raw = palette_registry.get("amber")
            colormap_arr = np.array(raw, dtype=np.float32) / 255.0 if raw else np.zeros((1, 3), dtype=np.float32)
        colors = [
            (int(c[0] * 255), int(c[1] * 255), int(c[2] * 255))
            for c in np.clip(colormap_arr, 0, 1)
        ]
        # Apply hue rotation (matches generated/sampled behaviour)
        if effective_hue_offset != 0.0:
            rotated = []
            for r, g, b in colors:
                h, s, v = colorsys.rgb_to_hsv(r / 255.0, g / 255.0, b / 255.0)
                h = (h + effective_hue_offset / 360.0) % 1.0
                rotated.append(_hsv_to_rgb(h, s, v))
            colors = rotated
            colormap_arr = np.array(colors, dtype=np.float32) / 255.0
        label_str = palette_name

    else:
        colors = generate_palette(
            palette_type, n_colors=n_colors, seed=seed,
            hue_off=effective_hue_offset,
            sat=effective_sat, val=effective_val,
        )
        colormap_arr = palette_to_colormap(colors)
        label_str = palette_type

    # -- Optional remap through another palette --
    if remap not in ("none", ""):
        raw_remap = palette_registry.get(remap)
        if raw_remap:
            remap_arr = np.array(raw_remap, dtype=np.float32) / 255.0
            col_flat = colormap_arr.reshape(-1, 3)
            diffs = col_flat[:, None, :] - remap_arr[None, :, :]
            nearest = np.argmin(np.sum(diffs**2, axis=2), axis=1)
            colormap_arr = remap_arr[nearest]
            colors = [(int(c[0] * 255), int(c[1] * 255), int(c[2] * 255)) for c in colormap_arr]
            label_str = f"{label_str} → {remap}"

    # -- Render preview on GPU (GLSL strip shader) --
    try:
        result_arr = _render_preview_gpu(colors, W, H)
    except Exception:
        # Fallback: numpy vectorized (no PIL)
        n = len(colors)
        strip_h = int(H * 0.70) // n
        chip_h = int(H * 0.18)
        chip_y = int(H * 0.82)
        canvas = np.full((H, W, 3), 0.05, dtype=np.float32)
        for i, (r, g, b) in enumerate(colors):
            y0 = i * strip_h
            y1 = (i + 1) * strip_h
            canvas[y0:y1, :] = np.array([r, g, b], dtype=np.float32) / 255.0
        for i, (r, g, b) in enumerate(colors):
            x0 = int(i * (W / n))
            x1 = int((i + 1) * (W / n))
            canvas[chip_y:chip_y + chip_h, x0:x1] = np.array([r, g, b], dtype=np.float32) / 255.0
        result_arr = canvas

    # Apply optional palette remap to image too
    if remap not in ("none", ""):
        from ...core.utils import apply_palette
        result_arr = apply_palette(result_arr, remap)

    # -- Optional per-swatch value labels (PIL overlay on the GPU render) --
    # Drawn last so labels reflect the FINAL displayed colors (post-remap).
    if show_labels:
        result_arr = _draw_palette_labels(result_arr, colors, label_fmt, label_str)

    capture_frame("10", result_arr)
    return {"image": result_arr, "palette": colormap_arr}
