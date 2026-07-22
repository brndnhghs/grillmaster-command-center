"""
GPU-accelerated shader pipeline using ModernGL + GLSL fragment shaders.

Two modes:
  - Procedural (method #82): generates imagery from scratch
  - Filter (--filter shader): processes an input image

Runs headlessly on Apple M1 Metal backend (GL 4.1 core profile).

Thread safety: each OS thread gets its own ModernGL context via threading.local().
The live-sim loop thread and the main server thread never share a context, so no
locking is required across threads. Calls from the same thread are always serial.
"""

from __future__ import annotations
from pathlib import Path
import math
import random
import re
import threading

import numpy as np
from PIL import Image


# ═══════════════════════════════════════════════
#  GL CONTEXT (per-thread lazy singleton)
# ═══════════════════════════════════════════════

# One context per OS thread — avoids cross-thread GL state corruption on Metal.
_ctx_local = threading.local()


def _get_ctx():
    ctx = getattr(_ctx_local, "ctx", None)
    if ctx is None:
        import moderngl
        _ctx_local.ctx = moderngl.create_context(standalone=True, require=330)
    return _ctx_local.ctx


# ═══════════════════════════════════════════════
#  QUAD GEOMETRY (full-screen triangle strip)
# ═══════════════════════════════════════════════

_QUAD_VERTICES = np.array([
    -1, -1,  0, 0,
     1, -1,  1, 0,
     1,  1,  1, 1,
    -1,  1,  0, 1,
], dtype='f4')

_QUAD_INDICES = np.array([0, 1, 2, 0, 2, 3], dtype='i4')

# Shared vertex shader
_VERTEX_SHADER = '''
#version 330
in vec2 in_vert;
in vec2 in_uv;
out vec2 v_uv;
void main() {
    gl_Position = vec4(in_vert, 0.0, 1.0);
    v_uv = in_uv;
}
'''


# ═══════════════════════════════════════════════
#  SHADER LIBRARY
# ═══════════════════════════════════════════════

# Each entry: name, description, type (procedural|filter|both), fragment source

SHADERS = {}

# ── Typed uniform specs ───────────────────────────────────────────────
# A shader may declare named, typed uniforms instead of (or alongside) the
# legacy generic u_params vec4. Each entry maps a variable name to a spec:
#
#   {"glsl": "float",  "min": 0, "max": 10, "default": 4.0, "description": …}
#   {"glsl": "int",    "min": 1, "max": 8,  "default": 5,   "description": …}
#   {"glsl": "color",  "default": "#ff2266",                "description": …}
#   {"glsl": "choice", "choices": ["linear", …], "default": "linear", … }
#
# The variable is exposed in GLSL as `uniform <type> u_<name>` (color → vec3,
# choice → int index into `choices`). The node factory in methods/gpu_shaders.py
# turns each spec into a real node param (slider / color picker / dropdown) and
# a wireable SCALAR input port for numeric ones — no more cryptic p1..p4.
# Specs travel to the browser via shader_sources_for_client(), so the client
# parity renderer sets the same uniforms from the same node params.

_UNIFORM_GLSL_TYPES = {"float": "float", "int": "int", "color": "vec3", "choice": "int"}


def uniform_glsl_decls(uniforms: dict) -> str:
    """GLSL declaration block for a shader's typed uniforms."""
    lines = []
    for uname, spec in (uniforms or {}).items():
        gtype = _UNIFORM_GLSL_TYPES.get(spec.get("glsl", "float"), "float")
        desc = spec.get("description", "")
        lines.append(f"uniform {gtype} u_{uname};" + (f"  // {desc}" if desc else ""))
    return ("\n".join(lines) + "\n") if lines else ""


def _parse_color(value) -> tuple[float, float, float]:
    """'#rrggbb' | 'r,g,b' (0-1 or 0-255) | sequence → (r, g, b) floats in [0,1]."""
    if isinstance(value, (tuple, list)) and len(value) >= 3:
        vals = [float(v) for v in value[:3]]
        return tuple(v / 255.0 for v in vals) if max(vals) > 1.0 else tuple(vals)
    s = str(value or "#000000").strip()
    if s.startswith("#") and len(s) >= 7:
        return (int(s[1:3], 16) / 255.0, int(s[3:5], 16) / 255.0, int(s[5:7], 16) / 255.0)
    if "," in s:
        try:
            vals = [float(p) for p in s.split(",")[:3]]
            return tuple(v / 255.0 for v in vals) if max(vals) > 1.0 else tuple(vals)
        except ValueError:
            pass
    return (0.0, 0.0, 0.0)


def coerce_uniform(spec: dict, value) -> float | int | tuple:
    """Coerce a node-param value to the GL-settable value for a typed uniform.

    Mirrors coerceUniform() in ui/js/client3d.js — server and client must agree.
    """
    gtype = spec.get("glsl", "float")
    if value is None:
        value = spec.get("default")
    if gtype == "float":
        try:
            return float(value)
        except (TypeError, ValueError):
            return float(spec.get("default", 0.0))
    if gtype == "int":
        try:
            return int(round(float(value)))
        except (TypeError, ValueError):
            return int(spec.get("default", 0))
    if gtype == "choice":
        choices = spec.get("choices", [])
        if isinstance(value, str):
            return choices.index(value) if value in choices else 0
        try:
            return max(0, min(len(choices) - 1, int(round(float(value)))))
        except (TypeError, ValueError):
            return 0
    if gtype == "color":
        # Pre-swap to BGR: both render paths swap R/B at display time (the
        # server decodes its FBO read as BGR; the client's convention blit
        # swizzles .bgra to match). Feeding B,G,R here means the user's picked
        # color survives the swap and lands on screen as picked — on both
        # targets. The JS mirror (coerceUniform in client3d.js) does the same.
        r, g, b = _parse_color(value)
        return (b, g, r)
    return float(value)


def _register(name: str, description: str, shader_type: str, source: str,
              uniforms: dict | None = None):
    SHADERS[name] = {
        "name": name,
        "description": description,
        "type": shader_type,
        "source": source,
        "uniforms": uniforms or {},
    }


def shader_uses_time(name: str) -> bool:
    """True iff the shader's body actually references the ``u_time`` uniform.

    The common prologue *declares* ``uniform float u_time`` in every shader,
    so presence of the name in the assembled fragment is meaningless — we look
    only at the per-shader ``source`` body. A shader that never reads
    ``u_time`` renders an identical frame for every ``t`` (e.g. static
    fractals like Sierpinski/Mandelbrot, ASCII, gradient, solid color), so its
    node should be marked ``is_time_varying=False``: the executor then cooks it
    once and reuses the result until an upstream input changes. This keeps the
    time-variance contract honest (previously every GPU node defaulted to
    ``is_time_varying=True``, which mislabelled static procedural nodes as
    animated).
    """
    return "u_time" in (SHADERS.get(name, {}).get("source") or "")


# ── COMMON PROLOGUE (injected into every shader) ──

_PROLOGUE = '''
#version 330
precision highp float;

in vec2 v_uv;
out vec4 f_color;

uniform vec2 u_resolution;
uniform float u_time;
uniform vec4 u_params;   // xyzw = 4 generic float params
uniform sampler2D u_texture;  // input image (for filter mode)

// 2D rotation
mat2 rot(float a) { float c=cos(a), s=sin(a); return mat2(c,-s,s,c); }

// 2D noise helpers
float hash21(vec2 p) {
    p = fract(p * vec2(234.34, 435.345));
    p += dot(p, p + 19.19);
    return fract(p.x * p.y);
}

float noise(vec2 p) {
    vec2 i = floor(p); vec2 f = fract(p);
    f = f * f * (3.0 - 2.0 * f);
    float a = hash21(i); float b = hash21(i + vec2(1, 0));
    float c = hash21(i + vec2(0, 1)); float d = hash21(i + vec2(1, 1));
    return mix(mix(a, b, f.x), mix(c, d, f.x), f.y);
}

float fbm(vec2 p) {
    float v = 0.0, a = 0.5;
    for (int i = 0; i < 5; i++) {
        v += a * noise(p); p *= 2.0; a *= 0.5;
    }
    return v;
}
'''


# ═══════════════════════════════════════════════
#  SHADER PARITY LAYER
# ═══════════════════════════════════════════════
#
# One GLSL body per shader runs on BOTH targets:
#   • server  — moderngl desktop GL, "#version 330"      (build_fragment gl330)
#   • browser — WebGL2 / GLSL ES 3.00                    (build_fragment webgl2)
#
# The body/uniforms/helpers are already written in the compatible subset (same
# code the server compiles today). Only the header differs, so the shim is a
# thin version/precision transform. `build_fragment(name,'gl330')` reproduces
# the EXACT string render_shader() compiles today (render_shader is untouched;
# a test locks this equivalence), so the server render path is unchanged.

# Public aliases for the shared shim pieces.
PROLOGUE_GL330 = _PROLOGUE
VERTEX_GL330 = _VERTEX_SHADER

# Vertex shader for client-side fullscreen-quad passes (GLSL ES 3.00). The
# client feeds a [-1,1] quad `position`; v_uv is derived to match the server's
# in_uv (0..1). Kept here so server + client agree on the varying.
VERTEX_WEBGL2 = '''#version 300 es
precision highp float;
in vec3 position;
out vec2 v_uv;
void main() {
    v_uv = position.xy * 0.5 + 0.5;
    gl_Position = vec4(position.xy, 0.0, 1.0);
}'''


def _assemble_gl330(info: dict) -> str:
    """Exactly how render_shader() builds the fragment source.

    Typed-uniform shaders get their `uniform <type> u_<name>;` declarations
    injected between the shared prologue and the body, so the body references
    them like any other uniform. Legacy filter sources embed the prologue
    themselves; typed-uniform filters use the standard prologue (it already
    carries u_texture) so the decl injection applies uniformly.
    """
    decls = uniform_glsl_decls(info.get("uniforms") or {})
    if info["type"] == "filter" and not decls:
        return info["source"]            # legacy filter: source embeds the prologue
    return _PROLOGUE + decls + info["source"]


def _to_webgl2(frag_gl330: str) -> str:
    """Transform an assembled #version 330 fragment into GLSL ES 3.00.

    The body/uniforms/helpers are ES-compatible already; only the header
    changes. `#version` must be the first token in ES, so leading whitespace
    (the prologue starts with a newline) is stripped. The prologue's existing
    `precision highp float;` is preserved, so no duplicate is introduced.
    """
    frag = frag_gl330.lstrip()
    frag = frag.replace("#version 330", "#version 300 es", 1)
    # The prologue always carries `precision highp float;` right after the
    # version line; add one only if some source omitted it (defensive).
    if "precision highp float;" not in frag[:120]:
        frag = frag.replace("#version 300 es", "#version 300 es\nprecision highp float;", 1)
    return frag


# Tokens that would compile on desktop GL but break GLSL ES 3.00 parity.
_WEBGL2_FORBIDDEN = ("texture2D", "textureCube", "gl_FragColor", "varying ", "attribute ")


def build_fragment(name: str, target: str = "gl330") -> str:
    """Assemble a shader's fragment source for a render target.

    target: 'gl330' (server/moderngl) or 'webgl2' (browser/WebGL2).
    """
    if name not in SHADERS:
        raise ValueError(f"Unknown shader: {name}")
    frag = _assemble_gl330(SHADERS[name])
    if target == "gl330":
        return frag
    if target == "webgl2":
        return _to_webgl2(frag)
    raise ValueError(f"Unknown target: {target!r} (expected 'gl330' or 'webgl2')")


def shader_sources_for_client() -> dict:
    """Read-only bundle for the browser executor: every shader's WebGL2 fragment
    plus the shared WebGL2 vertex. Lets the client render any GPU shader node
    from the SAME source the server uses. Additive — no render-path involvement.
    """
    return {
        "vertex": VERTEX_WEBGL2,
        # Server display convention: render_shader() reads the FBO bottom-up and
        # decodes it as BGR (Image.frombytes(..,'raw','BGR')). Verified bit-exact
        # (0.000% diff on plasma/julia/voronoi): a client render matches the
        # server's output after a Y-flip and an R/B swap. Feature #1 applies this
        # so the client live preview matches the server's authoritative export.
        "convention": {"flip_y": True, "swap_rb": True},
        "shaders": {
            name: {
                "type": info["type"],
                "fragment": build_fragment(name, "webgl2"),
                # Typed uniform specs — client sets u_<name> from node params
                # with the same coercion the server uses (coerce_uniform).
                "uniforms": info.get("uniforms") or {},
            }
            for name, info in SHADERS.items()
        },
    }


# ═══════════════════════════════════════════════
#  REGISTER SHADERS
# ═══════════════════════════════════════════════

# ── PROCEDURAL (generate from scratch) ──




#  P0.3 — Escape-time / deterministic fractal CPU-twin shaders (client-GPU live
#  preview of nodes 33/51/52/66/67/69). These are ADDITIVE: the server's CPU
#  numpy path stays the authoritative export; these only drive the browser live
#  preview. They reuse the prologue helpers (rot/hash21/noise/fbm) and the
#  inferno colormap where a fire-style look suits the node's default.

# ── Reusable fractal coloring + escape helper (consumed by the twins below) ──
_FRACTAL_HELPERS = '''
vec3 fractal_palette(float t) {
    // Smooth cosine palette (matches the CPU 'sine' color mode's character).
    return 0.5 + 0.5 * cos(6.28318 * (vec3(1.0, 0.75, 0.5) * t) + vec3(0.0, 2.0, 4.0));
}

// Smooth iteration count → [0,1] using the standard normalized-iteration trick.
float smooth_iter(float n, float last_z2, float max_iter) {
    float nu = n + 1.0 - log(max(log(last_z2) * 0.5, 1.0001)) / log(2.0);
    return clamp(nu / max(max_iter, 1.0), 0.0, 1.0);
}
'''

_register("mandelbrot_gpu", "Mandelbrot set (client-GPU twin of node 33)", "procedural",
          _FRACTAL_HELPERS + '''
void main() {
    vec2 uv = (gl_FragCoord.xy - 0.5 * u_resolution) / min(u_resolution.x, u_resolution.y);
    vec2 ctr = vec2(u_center_x, u_center_y);
    vec2 c = ctr + uv * u_zoom;
    vec2 z = vec2(0.0);
    float n = 0.0;
    float last2 = 0.0;
    const float MAXI = 200.0;
    float bail2 = u_escape_radius * u_escape_radius;
    for (int i = 0; i < 200; i++) {
        z = vec2(z.x*z.x - z.y*z.y, 2.0*z.x*z.y) + c;
        last2 = dot(z, z);
        if (last2 > bail2 || n >= u_iterations) break;
        n += 1.0;
    }
    float t = (n >= u_iterations - 0.5) ? 0.0 : smooth_iter(n, last2, u_iterations);
    f_color = vec4(fractal_palette(t + u_color_shift), 1.0);
}
''',
    uniforms={
    "zoom": {"glsl": "float", "min": 0.5, "max": 100000.0, "default": 1.0, "description": "zoom (1 = full view)"},
    "center_x": {"glsl": "float", "min": -2.5, "max": 2.5, "default": -0.5, "description": "center x"},
    "center_y": {"glsl": "float", "min": -2.0, "max": 2.0, "default": 0.0, "description": "center y"},
    "iterations": {"glsl": "float", "min": 50.0, "max": 2000.0, "default": 200, "description": "max iterations"},
    "escape_radius": {"glsl": "float", "min": 1.5, "max": 100.0, "default": 4.0, "description": "escape bailout radius"},
    "color_shift": {"glsl": "float", "min": 0.0, "max": 6.28, "default": 0.0, "description": "palette color offset"}
}
    )

_register("burning_ship_gpu", "Burning Ship fractal (client-GPU twin of node 51)", "procedural",
          _FRACTAL_HELPERS + '''
void main() {
    vec2 uv = (gl_FragCoord.xy - 0.5 * u_resolution) / min(u_resolution.x, u_resolution.y);
    vec2 ctr = vec2(0.5, 0.5);
    vec2 c = ctr + uv * 1.0;  // fixed full view (node 51 has no zoom param)
    vec2 z = vec2(0.0);
    float n = 0.0;
    float last2 = 0.0;
    const float MAXI = 500.0;
    for (int i = 0; i < 500; i++) {
        z = vec2(abs(z.x) - 1.0, abs(z.y)) * abs(z.x) + c; // abs-squared ship map
        z = vec2(z.x*z.x - z.y*z.y, 2.0*z.x*z.y);
        last2 = dot(z, z);
        if (last2 > 16.0 || n >= u_iterations) break;
        n += 1.0;
    }
    float t = (n >= u_iterations - 0.5) ? 0.0 : smooth_iter(n, last2, u_iterations);
    f_color = vec4(fractal_palette(t * (0.6 + 0.4 * u_color_speed) + u_color_offset), 1.0);
}
''',
    uniforms={
    "iterations": {"glsl": "float", "min": 30.0, "max": 500.0, "default": 100, "description": "max iterations"},
    "color_speed": {"glsl": "float", "min": 0.5, "max": 8.0, "default": 2.0, "description": "palette color speed"},
    "color_offset": {"glsl": "float", "min": 0.0, "max": 6.28, "default": 0.0, "description": "palette color offset"}
}
    )

_register("newton_gpu", "Newton fractal basins (client-GPU twin of node 52)", "procedural",
          _FRACTAL_HELPERS + '''
void main() {
    vec2 uv = (gl_FragCoord.xy - 0.5 * u_resolution) / min(u_resolution.x, u_resolution.y);
    vec2 z = uv * 2.2;  // fixed full view (node 52 has no zoom param)
    const float MAXI = 200.0;
    float n = 0.0;
    for (int i = 0; i < 200; i++) {
        // Newton for z^3 - 1: z - (z^3 - 1) / (3 z^2)
        vec2 z2 = vec2(z.x*z.x - z.y*z.y, 2.0*z.x*z.y);
        vec2 z3 = vec2(z2.x*z.x - z2.y*z.y, 2.0*z2.x*z.y);
        vec2 f = z3 - vec2(1.0, 0.0);
        vec2 dz = 3.0 * z2;
        float denom = dz.x*dz.x + dz.y*dz.y + 1e-8;
        vec2 step = vec2(f.x*dz.x + f.y*dz.y, f.y*dz.x - f.x*dz.y) / denom;
        z -= step;
        n += 1.0;
        if (dot(step, step) < 1e-6 || n >= u_max_iter) break;
    }
    // Color by nearest of the 3 cube roots of unity (angle quantization).
    float ang = atan(z.y, z.x);
    float root = floor((ang + 3.14159) / (2.0 * 3.14159 / 3.0));
    float t = mod(root / 3.0 + u_color_offset + 0.15 * n / MAXI, 1.0);
    f_color = vec4(fractal_palette(t * (0.6 + 0.4 * u_color_speed)), 1.0);
}
''',
    uniforms={
    "max_iter": {"glsl": "float", "min": 10.0, "max": 200.0, "default": 50, "description": "max Newton iterations"},
    "color_speed": {"glsl": "float", "min": 0.5, "max": 8.0, "default": 2.0, "description": "palette color speed"},
    "color_offset": {"glsl": "float", "min": 0.0, "max": 6.28, "default": 0.0, "description": "palette color offset"}
}
    )

# ── Kaleidoscopic IFS (client-GPU twin of node 402) ──
# KIFS: repeated mirror-fold + n-fold wedge fold + rotation + scale + offset.
# The mirror+wedge fold is what yields the snowflake/kaleidoscopic symmetry and
# the negative scale yields the classic self-similar detail. This is a
# closed-form function of (uv, t), so the twin is an exact parity preview; the
# server's CPU numpy node (method_kaleidoscopic_ifs) stays authoritative.
# Neutral: 0.5 on every slot yields the node's canonical default view
# (sym=6, scale=-2, no animation). p1=scale, p2=fold_angle, p3=symmetry(2..8),
# p4=color_shift. anim_mode/offset/colormode are choice strings (pitfall #14)
# so they are NOT mapped — the live preview shows the node's default orbit
# coloring at the default offsets.
_register("kifs_gpu", "Kaleidoscopic IFS fractal (client-GPU twin of node 402)", "procedural",
          _FRACTAL_HELPERS + '''
void main() {
    vec2 uv = (gl_FragCoord.xy - 0.5 * u_resolution) / min(u_resolution.x, u_resolution.y);
    uv *= 3.0;  // match CPU: map pixels to a ~[-3,3] complex-plane region

    // u_time drives a live-preview spin (matches the CPU 'spin' anim mode at
    // t=0; the client advances u_time so the live preview animates). At t=0
    // this is the identity rotation, so parity with the static CPU frame holds.
    float ta = u_time * 0.3;
    uv = rot(ta) * uv;

    float scale = u_scale;                       // 0.5 -> -2.0 canonical
    float ang = (u_fold_angle - 0.5) * 6.2831853; // 0.5 -> 1.0 rad canonical
    float sym = floor(2.0 + u_symmetry * 6.0);    // 0.5 -> 6-fold canonical
    float color_shift = u_color_shift;
    vec2 offs = vec2(1.0, 1.0);                   // default offset (unmapped)

    float r2esc = 144.0;                          // escape_radius = 12 -> r2=144
    float trap = 1e9;
    float esc = 0.0;

    for (int i = 0; i < 24; i++) {
        // 1) Mirror + n-fold wedge fold (kaleidoscopic symmetry)
        vec2 z = abs(uv);
        float a = atan(z.y, z.x);
        float r = length(z);
        float wedge = 3.14159265 / sym;
        a = mod(a, 2.0 * wedge);
        if (a > wedge) a = 2.0 * wedge - a;
        z = vec2(cos(a), sin(a)) * r;
        // 2) Rotation between folds
        z = rot(ang) * z;
        // 3) Scale + offset
        z = z * scale + offs;
        uv = z;
        trap = min(trap, length(z));
        if (dot(z, z) > r2esc) { esc = 1.0; break; }
    }

    float v;
    if (esc > 0.5) {
        v = trap / 12.0;
    } else {
        v = 0.0;  // interior -> dark
    }
    v = clamp(v + color_shift, 0.0, 1.0);
    vec3 col = fractal_palette(v);
    col = mix(vec3(0.02, 0.02, 0.06), col, esc);  // interior fill
    f_color = vec4(col, 1.0);
}
''',
    uniforms={
    "scale": {"glsl": "float", "min": -3.0, "max": 1.0, "default": -2.0, "description": "fold scale (negative = classic detail)"},
    "fold_angle": {"glsl": "float", "min": 0.0, "max": 1.0, "default": 0.5, "description": "fold rotation slot (0.5 -> 1.0 rad)"},
    "symmetry": {"glsl": "float", "min": 0.0, "max": 1.0, "default": 0.5, "description": "kaleidoscopic symmetry order (0.5 -> 6)"},
    "color_shift": {"glsl": "float", "min": 0.0, "max": 1.0, "default": 0.5, "description": "palette color offset"}
}
    )

_register("sierpinski_gpu", "Sierpinski carpet (client-GPU twin of node 67)", "procedural",
          _FRACTAL_HELPERS + '''
void main() {
    vec2 uv = v_uv;
    // p1 = depth (subdivisions), p2 = color_shift, p3/p4 unused (reserved).
    float depth = clamp(u_depth, 1.0, 7.0);
    // Tiling coordinates in [0,1] space.
    vec2 p = uv;
    float hole = 0.0;
    for (float i = 0.0; i < 7.0; i += 1.0) {
        if (i >= depth) break;
        // Carpet rule: remove central third at each scale.
        vec2 cell = floor(p * 3.0);
        if (cell.x == 1.0 && cell.y == 1.0) { hole = 1.0; break; }
        p = fract(p * 3.0);
    }
    float t = fract(0.15 * (depth) + 0.5 + 0.3 * uv.x + 0.2 * uv.y);
    vec3 col = (hole > 0.5) ? vec3(0.04) : fractal_palette(t);
    f_color = vec4(col, 1.0);
}
''',
    uniforms={
    "depth": {"glsl": "float", "min": 1.0, "max": 7.0, "default": 4.0, "description": "subdivision depth"}
}
    )

_register("lyapunov_gpu", "Lyapunov exponent map (client-GPU twin of node 69)", "procedural",
          _FRACTAL_HELPERS + '''
void main() {
    vec2 uv = v_uv;
    // r_min / r_max are the logistic r-range (match node 69's real params);
    // the A/B perturbation sequence and warmup/measure counts are choice/int
    // controls (pitfall #14) left unmapped — the twin renders the default ABAB map.
    vec2 rmin = vec2(u_r_min, u_r_min);
    vec2 rmax = vec2(u_r_max, u_r_max);
    vec2 r = mix(rmin, rmax, uv);
    // Logistic-map A/B perturbation (ABAB...), 8 chars.
    float lambda = 0.0;
    float x = 0.5;
    const float WARM = 30.0;
    const float MEAS = 80.0;
    for (float i = 0.0; i < (WARM + MEAS); i += 1.0) {
        int k = int(mod(i, 8.0));
        float rk = (k == 0 || k == 2 || k == 4 || k == 6) ? r.x : r.y;
        float deriv = rk * (1.0 - 2.0 * x);
        x = rk * x * (1.0 - x);
        if (i >= WARM) {
            lambda += log(abs(deriv) + 1e-8);
        }
    }
    lambda = lambda / MEAS;
    float t = clamp(0.5 + 0.5 * lambda / 2.0, 0.0, 1.0);
    t = t;
    f_color = vec4(fractal_palette(t), 1.0);
}
''',
    uniforms={
    "r_min": {"glsl": "float", "min": 0.0, "max": 1.0, "default": 0.5, "description": "lower logistic r"},
    "r_max": {"glsl": "float", "min": 0.0, "max": 1.0, "default": 0.5, "description": "upper logistic r"}
}
    )

_register("kifs_spherefold_gpu", "Kaleidoscopic IFS - box-fold + sphere-fold (Knighty/Kali 2010)", "procedural",
          _FRACTAL_HELPERS + '''
// KIFS (Kaleidoscopic Iterated Function System) fractal on the 2D plane.
// Core technique (Knighty / Kali, 2010 "Kaleidoscopic IFS" thread):
//   1) box fold  - mirror z into the positive wedge + diagonal fold
//   2) scale + offset  - the "kaleidoscope" growth
//   3) rotation between folds
//   4) sphere fold - pull near-origin points outward, push far points in
//      (the minR/maxR radius clamp that gives the characteristic holes).
// Iterating these affine folds produces the iconic self-similar kaleidoscope.
// Distinct from kifs_gpu (node 402), which only does a wedge + scale fold.
void main() {
    vec2 uv = (gl_FragCoord.xy - 0.5 * u_resolution) / min(u_resolution.x, u_resolution.y);
    uv *= 2.2;

    // Global spin for the live preview (t=0 -> identity, so a static frame
    // still renders the canonical fractal).
    float ta = u_time * 0.2;
    uv = rot(ta) * uv;

    float scale = u_scale;                       // negative scale = classic detail
    float ang = (u_fold_angle - 0.5) * 6.2831853 + u_time * 0.13;  // per-iteration rot
    float minR = max(u_min_r, 0.05);
    float maxR = max(u_max_r, minR + 0.05);
    float minR2 = minR * minR;
    float maxR2 = maxR * maxR;
    // Time-orbiting offset -> the whole kaleidoscope breathes and flows.
    vec2 offs = vec2(1.0 + 0.30 * sin(u_time * 0.7),
                     0.5 + 0.30 * cos(u_time * 0.9));
    float color_shift = u_color_shift;

    vec2 z = uv;
    float trap = 1e9;
    float esc = 0.0;
    float escR2 = 64.0;

    for (int i = 0; i < 16; i++) {
        // 1) box fold - mirror into the first octant, then diagonal fold
        z = abs(z);
        if (z.x < z.y) z = z.yx;
        // 2) scale + offset
        z = z * scale + offs;
        // 3) rotation between folds
        z = rot(ang) * z;
        // 4) sphere fold (radius clamp)
        float r2 = dot(z, z);
        if (r2 < minR2) {
            z *= minR2 / max(r2, 1e-6);
        } else if (r2 > maxR2) {
            z *= maxR2 / r2;
        }
        trap = min(trap, length(z));
        if (dot(z, z) > escR2) { esc = 1.0; break; }
    }

    float v;
    if (esc > 0.5) {
        v = clamp(trap / (maxR * 2.0), 0.0, 1.0);
    } else {
        v = 0.0;  // interior -> dark
    }
    v = clamp(v + color_shift, 0.0, 1.0);
    vec3 col = fractal_palette(v);
    col = mix(vec3(0.02, 0.02, 0.06), col, esc);  // interior fill
    f_color = vec4(col, 1.0);
}
''',
    uniforms={
    "scale": {"glsl": "float", "min": -3.0, "max": 1.0, "default": -1.8, "description": "fold scale (negative = classic kaleidoscopic detail)"},
    "fold_angle": {"glsl": "float", "min": 0.0, "max": 1.0, "default": 0.5, "description": "per-iteration rotation slot (0.5 -> 0 rad)"},
    "min_r": {"glsl": "float", "min": 0.1, "max": 1.5, "default": 0.5, "description": "sphere-fold inner radius"},
    "max_r": {"glsl": "float", "min": 1.0, "max": 4.0, "default": 2.0, "description": "sphere-fold outer radius"},
    "color_shift": {"glsl": "float", "min": 0.0, "max": 1.0, "default": 0.5, "description": "palette color offset"}
}
    )




# Typed-uniform twin of CPU node 528 (Voronoise). Closed-form f(uv,t): the
# Voronoi feature points orbit with u_time so the live preview is genuinely
# animated (survives the shootout contrast-only static liveness cull). Every
# numeric node param is exposed as a named uniform; colormode/palette/anim_mode
# are choice strings (pitfall #14) left unmapped -> preview uses the inferno
# default (node 528's default colormode). CPU numpy node 528 stays authoritative
# for export. 528 is a CPU node id above 301.
_register("voronoise_typed",
          "IQ smooth Voronoi (voronoise) with typed scale/jitter/smoothness/octaves (node 528)",
          "procedural", '''vec2 vhash22(vec2 p) {
    p = vec2(dot(p, vec2(127.1, 311.7)), dot(p, vec2(269.5, 183.3)));
    return fract(sin(p) * 43758.5453);
}

// Self-contained inferno colormap (node 528 default colormode).
vec3 vinferno(float t) {
    t = clamp(t, 0.0, 1.0);
    const vec3 c0 = vec3(0.00021894, 0.00165100, -0.01948090);
    const vec3 c1 = vec3(0.10651342, 0.56395644,  3.93271239);
    const vec3 c2 = vec3(11.60249308, -3.97285397, -15.94239411);
    const vec3 c3 = vec3(-41.70399613, 17.43639888, 44.35414520);
    const vec3 c4 = vec3(77.16293570, -33.40235894, -81.80730926);
    const vec3 c5 = vec3(-71.31942824, 32.62606426, 73.20951986);
    const vec3 c6 = vec3(25.13112622, -12.24266895, -23.07032500);
    return c0 + t * (c1 + t * (c2 + t * (c3 + t * (c4 + t * (c5 + t * c6)))));
}

// IQ-style smooth Voronoi: w blends nearest-distance cells (w=0) into a
// smoothly-averaged noise field (w=1); jitter displaces the cell points.
float voronoise(vec2 x, float w, float jitter, float t) {
    vec2 n = floor(x);
    vec2 f = fract(x);
    float f1 = 8.0, f2 = 8.0;
    for (int j = -1; j <= 1; j++) {
        for (int i = -1; i <= 1; i++) {
            vec2 g = vec2(float(i), float(j));
            vec2 o = vhash22(n + g);
            // Feature points orbit over time so the field is animated.
            vec2 disp = jitter * (0.5 + 0.5 * vec2(sin(t + 6.2831 * o.x),
                                                  cos(t + 6.2831 * o.y)));
            vec2 pnt = g + disp;
            float d = length(pnt - f);
            if (d < f1) { f2 = f1; f1 = d; }
            else if (d < f2) { f2 = d; }
        }
    }
    // w=0 -> crisp Voronoi cell distance; w=1 -> smooth averaged field.
    return mix(f1, 0.5 * (f1 + f2), w);
}

void main() {
    vec2 uv = v_uv * max(u_scale, 0.5);
    float t = u_time * 0.4;
    float v = 0.0, amp = 0.5, norm = 0.0;
    vec2 q = uv;
    for (int o = 0; o < 5; o++) {
        if (o >= int(u_octaves)) break;
        v += amp * voronoise(q, u_smoothness, u_jitter, t + float(o) * 1.3);
        norm += amp;
        q *= u_lacunarity;
        amp *= u_gain;
    }
    v = clamp(v / max(norm, 1e-3), 0.0, 1.0);
    v = clamp((v - 0.5) * u_contrast + 0.5, 0.0, 1.0);
    f_color = vec4(vinferno(v), 1.0);
}
''', uniforms={
    "scale":      {"glsl": "float", "min": 1.0, "max": 24.0, "default": 8.0,
                   "description": "grid frequency / zoom"},
    "jitter":     {"glsl": "float", "min": 0.0, "max": 1.0, "default": 1.0,
                   "description": "cell-point displacement (0=regular, 1=jittered)"},
    "smoothness": {"glsl": "float", "min": 0.0, "max": 1.0, "default": 1.0,
                   "description": "metric: 1=averaged noise, 0=min-distance cells"},
    "octaves":    {"glsl": "float", "min": 1.0, "max": 5.0, "default": 1.0,
                   "description": "fBM octaves stacked for detail"},
    "lacunarity": {"glsl": "float", "min": 1.5, "max": 3.0, "default": 2.0,
                   "description": "frequency multiplier per octave"},
    "gain":       {"glsl": "float", "min": 0.3, "max": 0.8, "default": 0.5,
                   "description": "amplitude falloff per octave"},
    "contrast":   {"glsl": "float", "min": 0.5, "max": 3.0, "default": 1.0,
                   "description": "tone contrast"},
})

_register("apollonian_gpu",
          "IQ Apollonian gasket via iterated fold+inversion; typed depth/seed_curv mirror node 514",
          "procedural", '''// Self-contained inferno colormap (matches other typed twins).
vec3 apinferno(float t) {
    t = clamp(t, 0.0, 1.0);
    const vec3 c0 = vec3(0.00021894, 0.00165100, -0.01948090);
    const vec3 c1 = vec3(0.10651342, 0.56395644,  3.93271239);
    const vec3 c2 = vec3(11.60249308, -3.97285397, -15.94239411);
    const vec3 c3 = vec3(-41.70399613, 17.43639888, 44.35414520);
    const vec3 c4 = vec3(77.16293570, -33.40235894, -81.80730926);
    const vec3 c5 = vec3(-71.31942824, 32.62606426, 73.20951986);
    const vec3 c6 = vec3(25.13112622, -12.24266895, -23.07032500);
    return c0 + t * (c1 + t * (c2 + t * (c3 + t * (c4 + t * (c5 + t * c6)))));
}

// IQ "Apollonian" recurrence: repeatedly wrap into [-1,1] then invert about the
// origin. `u_depth` mirrors node 514's recursion depth (nested-circle density);
// `u_seed_curv` mirrors the seed-circle bend (curvature per inversion). `t`
// slowly rotates the frame so the packing animates when a driver is wired.
void main() {
    vec2 p = (v_uv - 0.5) * 2.4;
    float t = u_time * 0.15;
    float ca = cos(t), sa = sin(t);
    p = mat2(ca, -sa, sa, ca) * p;
    float scale = 1.0;
    float d = 1e9;
    float fold = 0.55 + u_seed_curv * 0.28;   // seed bend -> inversion scale
    for (int i = 0; i < 8; i++) {
        if (i >= int(u_depth)) break;
        p = -1.0 + 2.0 * fract(0.5 * p + 0.5);
        float r2 = dot(p, p);
        float k = fold / max(r2, 1e-4);
        p *= k;
        scale *= k;
        d = min(d, abs(p.y) / scale);
    }
    float v = 1.0 - clamp(pow(d * 6.0, 0.35), 0.0, 1.0);
    v = clamp((v - 0.5) * 1.4 + 0.5, 0.0, 1.0);
    f_color = vec4(apinferno(v), 1.0);
}
''', uniforms={
    "depth":     {"glsl": "float", "min": 1.0, "max": 6.0, "default": 4.0,
                  "description": "recursion depth (denser nested circles)"},
    "seed_curv": {"glsl": "float", "min": 1.0, "max": 4.0, "default": 2.0,
                  "description": "seed-circle bend -> inversion curvature"},
})





_register("hash_field_gpu",
          "Multiresolution Hash Encoding field (client-GPU twin of node 309)",
          "procedural", '''vec3 hash3(vec2 p) {
    vec3 q = vec3(dot(p, vec2(127.1, 311.7)),
                   dot(p, vec2(269.5, 183.3)),
                   dot(p, vec2(419.2, 371.9)));
    return fract(sin(q) * 43758.5453) * 2.0 - 1.0;
}

// Bilinearly-interpolated integer-lattice hash (shared table == NGP look).
float hfeat(vec2 g, float lvl) {
    vec2 i = floor(g);
    vec2 f = g - i;
    vec2 u = f * f * (3.0 - 2.0 * f);          // smoothstep
    // cheap hash into [-1,1] (use a different per-level salt)
    vec2 h00 = hash3(i + vec2(0.0, 0.0) + lvl * 7.0).xy;
    vec2 h10 = hash3(i + vec2(1.0, 0.0) + lvl * 7.0).xy;
    vec2 h01 = hash3(i + vec2(0.0, 1.0) + lvl * 7.0).xy;
    vec2 h11 = hash3(i + vec2(1.0, 1.0) + lvl * 7.0).xy;
    float top = mix(h00.x, h10.x, u.x);
    float bot = mix(h01.x, h11.x, u.x);
    return mix(top, bot, u.y);
}

void main() {
    // Decode real node params (0.5-neutral -> node defaults):
    //   p1 scale [1,16] 0.5 -> 4 ; p2 detail(levels) [1,16] 0.5 -> 10
    //   p3 hue  [0,1]    0.5 -> 0.5 ; p4 contrast [0.2,2.5] 0.5 -> 1.35
    float scale  = 1.0 + u_scale * 15.0;
    float levels = clamp(1.0 + u_detail * 15.0, 1.0, 16.0);
    float hue    = u_hue;
    float contrast = 0.2 + u_contrast * 2.3;

    vec2 uv = v_uv;
    float tt = u_time;
    // smooth coordinate animation (matches CPU anim_mode=pan/zoom/swirl/pulse)
    uv = fract(uv + vec2(tt * 0.05, tt * 0.03));
    float s = exp(-tt * 0.15);
    uv = (uv - 0.5) * s + 0.5;

    float acc = 0.0;
    for (int l = 0; l < 16; l++) {
        if (float(l) >= levels) break;
        float N = max(1.0, scale * pow(2.0, float(l)));
        acc += hfeat(uv * N, float(l));
    }
    acc /= levels;
    float val = clamp(0.5 + 0.5 * acc * contrast, 0.0, 1.0);
    // IQ cosine palette with hue shift
    vec3 col = 0.5 + 0.5 * cos(6.2831853 * (val + vec3(0.0, 0.33, 0.67)) + hue * 6.2831853);
    f_color = vec4(col, 1.0);
}
''',
    uniforms={
    "scale": {"glsl": "float", "min": 1.0, "max": 16.0, "default": 4.0, "description": "coarsest grid resolution"},
    "detail": {"glsl": "float", "min": 1.0, "max": 16.0, "default": 10.0, "description": "hash-grid levels (octaves)"},
    "hue": {"glsl": "float", "min": 0.0, "max": 1.0, "default": 0.5, "description": "palette hue shift"},
    "contrast": {"glsl": "float", "min": 0.2, "max": 2.5, "default": 1.35, "description": "tone contrast"},
})

















# ── Gabor Noise (CPU node 477 twin) — anisotropic sparse Gabor convolution ──
# Lagae et al. 2011. One Gabor kernel per jittered-lattice cell; sum the
# kernels in each pixel's neighbourhood. p1=orientation(deg), p2=anisotropy,
# p3=frequency, p4=bandwidth. Gentle u_time rotation for the live preview.
_register("gabor_gpu", "Gabor noise (Lagae et al. 2011) — anisotropic sparse Gabor convolution", "procedural", '''
void main() {
    vec2 res = u_resolution;
    vec2 px = v_uv * res;
    float orient = radians(90.0);   // fixed: node 473 exposes no orientation param
    float aniso = clamp(u_anisotropy, 0.0, 1.0);
    float freq = clamp(u_frequency * 12.0, 0.5, 12.0);
    float bw = clamp(u_falloff * 6.0, 0.5, 6.0);

    float S = clamp(70.0 / freq, 6.0, 90.0);
    float Fmag = freq * 0.03;
    float bwf = clamp(bw / 2.5, 0.4, 2.4);
    float spar = S * 0.34 * bwf;
    float sper = spar * max(0.12, 1.0 - aniso * 0.85);

    // gentle time rotation (live-preview animation)
    vec2 u = vec2(cos(orient + u_time * 0.5), sin(orient + u_time * 0.5));

    float acc = 0.0;
    float R = 3.0;
    for (float o = 0.0; o < 2.0; o += 1.0) {
        float So = S / pow(2.0, o);
        float Fm = Fmag * pow(2.0, o);
        float spo = spar / pow(2.0, o) * bwf;
        float spe = sper / pow(2.0, o);
        float amp = 1.0 / pow(2.0, o);
        float ci = floor(px.x / So);
        float cj = floor(px.y / So);
        for (float dj = -R; dj <= R; dj += 1.0) {
            for (float di = -R; di <= R; di += 1.0) {
                float nci = ci + di;
                float ncj = cj + dj;
                float jx = (hash21(vec2(nci, ncj)) - 0.5) * So;
                float jy = (hash21(vec2(nci + 131.0, ncj + 517.0)) - 0.5) * So;
                vec2 ip = vec2(nci * So + jx, ncj * So + jy);
                float ph = hash21(vec2(nci + 977.0, ncj + 331.0)) * 6.2831853;
                vec2 d = px - ip;
                float dp = dot(d, u);
                float dq = dot(d, vec2(-u.y, u.x));
                float env = exp(-3.14159265 * (dp * dp / (spo * spo) + dq * dq / (spe * spe)));
                acc += amp * env * cos(6.2831853 * Fm * dp + ph);
            }
        }
    }
    float v = clamp(acc / (abs(acc) + 0.6), -1.0, 1.0);
    vec3 col = 0.5 + 0.5 * cos(6.2831853 * (0.5 + 0.5 * v) + vec3(0.0, 0.33, 0.67));
    f_color = vec4(col, 1.0);
}
''', uniforms={
    "anisotropy": {"glsl": "float", "min": 0.0, "max": 1.0, "default": 0.5,
                   "description": "kernel elongation (0=isotropic, 1=fully stretched)"},
    "frequency":   {"glsl": "float", "min": 0.0, "max": 1.0, "default": 0.5,
                    "description": "spatial frequency (scaled x12 internally)"},
    "falloff":     {"glsl": "float", "min": 0.0, "max": 1.0, "default": 0.5,
                    "description": "envelope bandwidth (scaled x6 internally)"},
})


_register("dot_noise_gpu", "Aperiodic gyroid dot-noise fBm (Xor, GM Shaders 2025)", "procedural", '''
// Dot Noise — a cheap closed-form alternative to 3D simplex noise.
//   Ref: Xor, "Dot Noise", GM Shaders Mini, 2025-09-05
//        https://mini.gmshaders.com/p/dot-noise
// Core idea: gyroid = dot(cos(p), sin(p.yzx)); giving one axis an
// irrational (golden-ratio) frequency makes the sheets never realign,
// yielding aperiodic pseudo-noise with NO hash lookups — ideal for
// many-sample volumetric-style sampling. Here it is fBm-summed and
// animated by sweeping the 3rd (z) coordinate through u_time.
//   u_params.x = base frequency / zoom   (0.5 -> 6.0)
//   u_params.y = fBm octaves            (0.5 -> 4)
//   u_params.z = warp amount (self-domain-warp)  (0.5 -> 0.35)
//   u_params.w = color palette phase    (0.5 -> 0.5)
// PHI = golden ratio -> "most irrational" frequency for aperiodicity.
float dotGyroid(vec3 p) {
    // aperiodic gyroid: one swizzled axis carries a phi-scaled frequency
    const float PHI = 1.61803398875;
    vec3 q = vec3(p.x, p.y * PHI, p.z);
    return dot(cos(q), sin(q.yzx));
}
float dotNoiseFbm(vec3 p, float oct, float warp) {
    // self domain-warp for richer structure (cheap: one extra eval)
    float w = dotGyroid(p * 0.6);
    p += warp * vec3(w, w * 0.7, w * 1.3);
    float sum = 0.0;
    float amp = 0.5;
    float freq = 1.0;
    for (int i = 0; i < 5; i++) {
        if (float(i) >= oct) break;
        sum += amp * dotGyroid(p * freq);
        freq *= 2.0;
        amp *= 0.5;
    }
    return sum;
}
void main() {
    vec2 uv = (v_uv * 2.0 - 1.0);
    uv.x *= u_resolution.x / u_resolution.y;

    float baseFreq = u_freq;
    float oct = floor(u_octaves + 0.5);
    float warp = u_warp;
    float phase = u_palette;

    // sweep the 3rd coordinate through time: animates the noise field
    vec3 p = vec3(uv * baseFreq, u_time * u_flow);

    float n = dotNoiseFbm(p, oct, warp);
    // gyroid dot is in ~[-2,2]; remap to [0,1]
    float v = clamp(n * 0.25 + 0.5, 0.0, 1.0);

    vec3 col = 0.5 + 0.5 * cos(6.2831853 * (phase + v + vec3(0.0, 0.33, 0.67)));
    col *= 0.35 + 0.65 * v;
    f_color = vec4(col, 1.0);
}
''', uniforms={
    "freq":    {"glsl": "float", "min": 2.0, "max": 10.0, "default": 6.0,
                "description": "base frequency / zoom of the noise field"},
    "octaves": {"glsl": "float", "min": 1.0, "max": 6.0, "default": 4.0,
                "description": "number of fBm octaves summed"},
    "warp":    {"glsl": "float", "min": 0.0, "max": 0.7, "default": 0.35,
                "description": "self domain-warp amount"},
    "palette": {"glsl": "float", "min": 0.0, "max": 1.0, "default": 0.5,
                "description": "cosine palette phase"},
    "flow":    {"glsl": "float", "min": 0.0, "max": 2.0, "default": 0.6,
                "description": "time-sweep speed through the noise volume"},
})


# ── FILTER SHADERS (process input image) ──

_register("sdf_raymarch_gpu", "SDF raymarching (signed-distance-field scene)", "procedural", '''
// GPU twin of CPU node 412 — real-time ray marching of a smooth-min
// union of signed-distance primitives (a la Inigo Quilez / Shadertoy).
//   u_params.x = camera distance      (0.5 -> 5.0)
//   u_params.y = scene complexity, # primitives (0.5 -> 4)
//   u_params.z = march steps        (0.5 -> 90)
//   u_params.w = rim glow amount     (0.5 -> 0.5)
// u_time rotates the camera + spins the primitives.
float smin(float a, float b, float k) {
    float h = clamp(0.5 + 0.5 * (b - a) / k, 0.0, 1.0);
    return mix(b, a, h) - k * h * (1.0 - h);
}
float mapScene(vec3 p) {
    float nSph = floor(u_primitives + 0.5);
    float k = 0.4;
    float d = 1e5;
    for (int i = 0; i < 6; i++) {
        if (float(i) >= nSph) break;
        float a = float(i) / max(nSph, 1.0) * 6.2831853 + u_time * 0.3;
        vec3 c = vec3(cos(a) * 1.6, sin(float(i) * 1.3) * 0.4, sin(a) * 1.6);
        float ds = length(p - c) - 0.7;
        d = smin(d, ds, k);
    }
    float ground = p.y + 1.2;
    d = smin(d, ground, 0.5);
    return d;
}
vec3 calcNormal(vec3 p) {
    vec2 e = vec2(0.0015, 0.0);
    return normalize(vec3(
        mapScene(p + e.xyy) - mapScene(p - e.xyy),
        mapScene(p + e.yxy) - mapScene(p - e.yxy),
        mapScene(p + e.yyx) - mapScene(p - e.yyx)));
}
void main() {
    float camDist = u_cam_dist;
    float glow = u_glow;
    float steps = floor(u_steps + 0.5);

    vec2 uv = (v_uv * 2.0 - 1.0);
    uv.x *= u_resolution.x / u_resolution.y;

    float ang = u_time * 0.5;
    vec3 ro = vec3(sin(ang) * camDist, 0.6, cos(ang) * camDist);
    vec3 ta = vec3(0.0);
    vec3 fwd = normalize(ta - ro);
    vec3 rgt = normalize(cross(vec3(0.0, 1.0, 0.0), fwd));
    vec3 upv = cross(fwd, rgt);
    vec3 rd = normalize(fwd * 1.5 + uv.x * rgt + uv.y * upv);

    float t = 0.0;
    float hit = 0.0;
    vec3 p = ro;
    for (int i = 0; i < 160; i++) {
        if (float(i) >= steps) break;
        p = ro + rd * t;
        float d = mapScene(p);
        if (d < 0.001) { hit = 1.0; break; }
        t += d;
        if (t > 30.0) break;
    }

    vec3 bg = mix(vec3(0.05, 0.07, 0.12), vec3(0.18, 0.22, 0.34), uv.y * 0.5 + 0.5);
    vec3 col = bg;
    if (hit > 0.5) {
        vec3 n = calcNormal(p);
        vec3 lig = normalize(vec3(0.6, 0.7, -0.4));
        float dif = clamp(dot(n, lig), 0.0, 1.0);
        float amb = 0.3 + 0.7 * n.y;
        vec3 base = u_surface;
        col = base * (amb * 0.4 + dif * 0.9);
        float rim = pow(1.0 - clamp(dot(n, -rd), 0.0, 1.0), 2.0);
        col += glow * rim * vec3(0.35, 0.7, 1.0);
    }
    f_color = vec4(col, 1.0);
}
''', uniforms={
    "cam_dist":   {"glsl": "float", "min": 3.0, "max": 7.0, "default": 5.0,
                   "description": "camera orbit distance"},
    "primitives": {"glsl": "float", "min": 2.0, "max": 6.0, "default": 4.0,
                   "description": "number of blended SDF spheres"},
    "steps":      {"glsl": "float", "min": 40.0, "max": 140.0, "default": 90.0,
                   "description": "max ray-march steps (quality)"},
    "glow":       {"glsl": "float", "min": 0.0, "max": 1.0, "default": 0.5,
                   "description": "rim-glow amount"},
    "surface":    {"glsl": "color", "default": "#e68c59",
                   "description": "surface base color"},
})

def _filter_shader(source: str) -> str:
    """Wrap a filter shader body with the full image processing prologue."""
    return f'''
{_PROLOGUE}
vec4 sample(vec2 uv) {{ return texture(u_texture, uv); }}

void main() {{
    vec2 uv = v_uv;
    vec2 step = 1.0 / u_resolution;
    vec4 orig = sample(uv);
    {source}
}}
'''

def _filter_typed(body: str) -> str:
    """Wrap a TYPED filter shader body (uses named u_<var> uniforms).

    Unlike _filter_shader, this does NOT embed the prologue — render_shader's
    _assemble_gl330 injects the shared _PROLOGUE + typed-uniform declarations
    for filter shaders that declare `uniforms=`. The body uses v_uv, u_texture,
    u_resolution, u_time (all from the prologue) and the local `step`/`sample`.
    """
    return f'''
vec4 sample(vec2 uv) {{ return texture(u_texture, uv); }}

void main() {{
    vec2 uv = v_uv;
    vec2 step = 1.0 / u_resolution;
    vec4 orig = sample(uv);
    {body}
}}
'''

_register("shader_bloom", "GPU bloom/glow from bright areas", "filter", _filter_typed('''
    float brightness = dot(orig.rgb, vec3(0.299, 0.587, 0.114));
    vec4 glow = vec4(0.0);
    for (int x = -3; x <= 3; x++) {
        for (int y = -3; y <= 3; y++) {
            vec2 off = vec2(float(x), float(y)) * step * 2.0;
            float b = dot(texture(u_texture, uv + off).rgb, vec3(0.299, 0.587, 0.114));
            if (b > 0.7) glow += texture(u_texture, uv + off) * exp(-float(x*x + y*y) / 4.0);
        }
    }
    glow /= 8.0;
    f_color = orig + glow * u_strength;
'''), uniforms={
    "strength": {"glsl": "float", "min": 0.0, "max": 2.0, "default": 0.8, "description": "bloom intensity"},
})

_register("shader_emboss", "GPU emboss / bump mapping", "filter", _filter_typed('''
    float gx = dot(texture(u_texture, uv + vec2(step.x, 0)).rgb, vec3(0.299, 0.587, 0.114));
    float gy = dot(texture(u_texture, uv + vec2(0, step.y)).rgb, vec3(0.299, 0.587, 0.114));
    float gz = dot(orig.rgb, vec3(0.299, 0.587, 0.114));
    float dx = gx - gz; float dy = gy - gz;
    float bump = (dx + dy) * 2.0 + 0.5;
    f_color = vec4(mix(orig.rgb, vec3(bump), u_strength), 1.0);
'''), uniforms={
    "strength": {"glsl": "float", "min": 0.0, "max": 2.0, "default": 0.8, "description": "emboss blend"},
})

_register("shader_kaleidoscope", "GPU kaleidoscope mirror", "filter", _filter_typed('''
    vec2 p = uv - 0.5;
    float a = atan(p.y, p.x);
    float r = length(p);
    float seg = 3.14159 * 2.0 / max(3.0, u_segments);
    a = mod(a, seg);
    a = abs(a - seg * 0.5);
    vec2 q = vec2(cos(a), sin(a)) * r + 0.5;
    f_color = texture(u_texture, q);
'''), uniforms={
    "segments": {"glsl": "float", "min": 3.0, "max": 16.0, "default": 8.0, "description": "mirror segments"},
})

# ── Kaleidoscope Mirror (typed-uniform filter twin of CPU node 460) ──
# Dihedral mirror-fold: map every point into one wedge around a movable
# center, optionally reflect for true dihedral symmetry, apply radial zoom and
# a slow fBm domain-warp. Named typed uniforms mirror node 460's REAL numeric
# params (segments/center_x/center_y/rotation/r_scale/mirror/warp_amount/
# warp_scale) so LFO/counter drivers can modulate the live preview (the
# electrical-engineering trap: a contrast-only static clip is avoided because
# the wrap is genuine per-pixel motion). CPU node 460 stays authoritative;
# this is the live-preview path only. `step` is the prologue-reserved vec2, so
# the warp accumulator uses `q`/`wp` locals (pitfall #15b).
# ══════════════════════════════════════════════════════════════════════════
#  P0.7 closed-form pattern twins (gap nodes 466 / 505 / 426)
#  Live-preview GPU twins for pattern nodes whose CPU algorithm is a faithful
#  closed-form f(uv,t). CPU node stays authoritative for export; these drive the
#  client-side live preview. Choice params (anim_mode / color_mode / orientation
#  / source / bg / palette ...) are intentionally omitted — the twins animate
#  continuously from u_time so the preview is always live; the CPU export honours
#  the exact choice. Helper functions are inlined (only _PROLOGUE helpers +
#  u_time / u_resolution are used) to avoid the late-helper ordering pitfall.
# ══════════════════════════════════════════════════════════════════════════

# ── 466 Hexagonal Mosaic ──
_register("hex_mosaic_gpu", "Hexagonal Mosaic (client-GPU twin of node 466)", "procedural", '''
float hexDist(vec2 p) {
    p = abs(p);
    return max(dot(p, vec2(0.8660254, 0.5)), p.x);
}
void main() {
    float scale = max(4.0, u_hex_size);
    float latRot = u_rotation + u_time * 0.12 * u_anim_speed;
    scale *= 1.0 + 0.15 * sin(u_time * u_anim_speed);   // gentle breathe
    vec2 uv = (v_uv - 0.5) * u_resolution;
    uv = rot(latRot) * uv;                              // rot() is the prologue helper
    vec2 rr = vec2(1.0, 1.7320508) * scale;
    vec2 h = rr * 0.5;
    vec2 a = mod(uv, rr) - h;
    vec2 b = mod(uv + h, rr) - h;
    vec2 gv = dot(a, a) < dot(b, b) ? a : b;
    vec2 cellId = uv - gv;
    float hexR = 0.8660254 * scale;                     // center-to-edge
    float edge = smoothstep(hexR * (1.0 - u_grout), hexR, hexDist(gv));
    float hsh = hash21(cellId * 0.013);
    vec3 col = 0.5 + 0.5 * cos(6.2831853 * (hsh + vec3(0.0, 0.33, 0.67)));
    col = mix(col, vec3(u_grout_color), edge);
    f_color = vec4(col, 1.0);
}
''',
    uniforms={
        "hex_size": {"glsl": "float", "min": 4.0, "max": 60.0, "default": 18.0, "description": "hex cell radius (px)"},
        "rotation": {"glsl": "float", "min": 0.0, "max": 6.2832, "default": 0.0, "description": "lattice rotation (rad)"},
        "grout": {"glsl": "float", "min": 0.0, "max": 1.0, "default": 0.12, "description": "grout line width"},
        "grout_color": {"glsl": "float", "min": 0.0, "max": 1.0, "default": 0.0, "description": "grout grayscale"},
        "anim_speed": {"glsl": "float", "min": 0.1, "max": 3.0, "default": 1.0, "description": "animation speed"},
    })

# ── 505 Metaballs (Procedural) ──
_register("metaballs_505_gpu", "Metaballs (client-GPU twin of node 505)", "procedural", '''
void main() {
    vec2 uv = v_uv;
    float t = u_time * u_anim_speed;
    int N = int(clamp(u_balls, 2.0, 16.0));
    float field = 0.0;
    for (int i = 0; i < 16; i++) {
        if (i >= N) break;
        float fi = float(i);
        float ang = fi * 2.3999632 + t * (0.5 + 0.1 * fi);   // golden-angle spread
        float rad = u_drift_amp * (0.5 + 0.5 * sin(t * 0.7 + fi));
        vec2 c = vec2(0.5) + rad * vec2(cos(ang), sin(ang))
                 + 0.10 * vec2(sin(t + fi), cos(t * 1.1 + fi * 2.0));
        vec2 d = uv - c;
        float bs = u_ball_size * (1.0 + 0.25 * sin(t + fi));  // pulse
        field += (bs * bs) / max(dot(d, d), 1e-4);
    }
    float m = smoothstep(u_threshold - u_edge_soft, u_threshold + u_edge_soft, field);
    float hue = fract(field * 0.15 + t * 0.05);
    vec3 col = 0.5 + 0.5 * cos(6.2831853 * (hue + vec3(0.0, 0.33, 0.67)));
    col *= m;
    f_color = vec4(col, 1.0);
}
''',
    uniforms={
        "balls": {"glsl": "float", "min": 2.0, "max": 16.0, "default": 8.0, "description": "number of metaballs"},
        "ball_size": {"glsl": "float", "min": 0.02, "max": 0.3, "default": 0.1, "description": "ball radius (frac of canvas)"},
        "threshold": {"glsl": "float", "min": 0.3, "max": 3.0, "default": 1.0, "description": "iso-level"},
        "edge_soft": {"glsl": "float", "min": 0.0, "max": 0.6, "default": 0.15, "description": "edge softness"},
        "drift_amp": {"glsl": "float", "min": 0.0, "max": 0.3, "default": 0.12, "description": "orbit amplitude"},
        "anim_speed": {"glsl": "float", "min": 0.1, "max": 3.0, "default": 1.0, "description": "animation speed"},
    })

# ── 426 Smooth Truchet (SDF) ──
_register("truchet_sdf_gpu", "Smooth Truchet (client-GPU twin of node 426)", "procedural", '''
void main() {
    float tile = max(8.0, u_tile_size);
    vec2 uv = v_uv * u_resolution;
    vec2 cell = floor(uv / tile);
    vec2 local = fract(uv / tile);                       // [0,1]
    local = rot(u_time * u_anim_speed) * (local - 0.5) + 0.5;   // flow
    float rnd = hash21(cell);
    if (rnd > 0.5) local.x = 1.0 - local.x;
    float d1 = abs(length(local) - 0.5);                 // arc at corner (0,0)
    float d2 = abs(length(local - 1.0) - 0.5);           // arc at corner (1,1)
    float d = min(d1, d2);
    float aa = 2.0 / tile;
    float lineMask = smoothstep(u_stroke * 0.5 + aa, u_stroke * 0.5 - aa, abs(d));
    float glow = u_edge_glow * smoothstep(0.12, 0.0, abs(d));
    float hue = fract(rnd + u_time * 0.05);
    vec3 base = 0.5 + 0.5 * cos(6.2831853 * (hue + vec3(0.0, 0.33, 0.67)));
    vec3 col = mix(vec3(0.05), base, lineMask);
    col += glow * base;
    f_color = vec4(col, 1.0);
}
''',
    uniforms={
        "tile_size": {"glsl": "float", "min": 24.0, "max": 200.0, "default": 56.0, "description": "tile size (px)"},
        "stroke": {"glsl": "float", "min": 0.04, "max": 0.4, "default": 0.13, "description": "tube width (frac of tile)"},
        "edge_glow": {"glsl": "float", "min": 0.0, "max": 1.0, "default": 0.25, "description": "outer glow strength"},
        "anim_speed": {"glsl": "float", "min": 0.1, "max": 3.0, "default": 0.6, "description": "animation speed"},
    })

_register("kaleidoscope_mirror_gpu", "Kaleidoscope Mirror (client-GPU twin of node 460)", "filter", _filter_typed('''
    vec2 ctr = vec2(u_center_x, u_center_y);
    vec2 p = uv - ctr;
    vec2 wp = p;
    if (u_warp_amount > 0.001) {
        float w = u_warp_amount * 0.3;
        float nx = fbm(p * u_warp_scale + vec2(u_time * 0.1, 0.0)) - 0.5;
        float ny = fbm(p * u_warp_scale + vec2(17.0, u_time * 0.1)) - 0.5;
        wp = p + w * vec2(nx, ny);
    }
    float a = atan(wp.y, wp.x) + radians(u_rotation);
    float r = length(wp) * u_r_scale;
    float seg = 6.2831853 / max(3.0, u_segments);
    a = mod(a, seg);
    if (u_mirror > 0.5) {
        a = abs(a - seg * 0.5);
    }
    vec2 sp = vec2(cos(a), sin(a)) * r + ctr;
    f_color = texture(u_texture, sp);
'''), uniforms={
    "segments": {"glsl": "float", "min": 3.0, "max": 24.0, "default": 8.0, "description": "mirror wedges (dihedral symmetry order)"},
    "center_x": {"glsl": "float", "min": 0.0, "max": 1.0, "default": 0.5, "description": "symmetry center X (0-1)"},
    "center_y": {"glsl": "float", "min": 0.0, "max": 1.0, "default": 0.5, "description": "symmetry center Y (0-1)"},
    "rotation": {"glsl": "float", "min": 0.0, "max": 360.0, "default": 0.0, "description": "pattern rotation in degrees"},
    "r_scale": {"glsl": "float", "min": 0.3, "max": 3.0, "default": 1.0, "description": "radial zoom into the wedge"},
    "mirror": {"glsl": "float", "min": 0.0, "max": 1.0, "default": 1.0, "description": "mirror-fold adjacent wedges (dihedral)"},
    "warp_amount": {"glsl": "float", "min": 0.0, "max": 1.0, "default": 0.0, "description": "fBm domain-warp strength (0 = pure geometry)"},
    "warp_scale": {"glsl": "float", "min": 1.0, "max": 20.0, "default": 6.0, "description": "domain-warp noise frequency"},
})

_register("shader_water_ripple", "GPU water ripple distortion", "filter", _filter_typed('''
    vec2 off = vec2(
        sin(uv.y * 50.0 + u_time * 2.0) * 0.01 * u_amp,
        cos(uv.x * 50.0 + u_time * 1.5) * 0.01 * u_amp
    );
    f_color = texture(u_texture, uv + off);
'''), uniforms={
    "amp": {"glsl": "float", "min": 0.0, "max": 2.0, "default": 0.8, "description": "ripple amplitude"},
})

_register("shader_heat_shimmer", "GPU heat haze / shimmer", "filter", _filter_typed('''
    float haze = sin(uv.x * 30.0 + uv.y * 20.0 + u_time * 3.0) * u_strength * 0.02;
    vec2 off = vec2(0.0, haze * (1.0 - uv.y));
    f_color = texture(u_texture, uv + off);
'''), uniforms={
    "strength": {"glsl": "float", "min": 0.0, "max": 1.0, "default": 0.6, "description": "shimmer amount"},
})

_register("shader_pixelate_gpu", "GPU pixelation with edge preservation", "filter", _filter_typed('''
    float block = u_block;
    vec2 q = floor(uv * u_resolution / block) * block / u_resolution;
    f_color = texture(u_texture, q);
'''), uniforms={
    "block": {"glsl": "float", "min": 4.0, "max": 64.0, "default": 16.0, "description": "pixel block size"},
})

_register("shader_ink_bleed", "GPU ink bleed / watercolor spread", "filter", _filter_typed('''
    vec3 sum = vec3(0.0);
    float count = 0.0;
    for (int x = -4; x <= 4; x++) {
        for (int y = -4; y <= 4; y++) {
            vec2 off = vec2(float(x), float(y)) * step * u_spread;
            float w = exp(-float(x*x + y*y) / (4.0 * u_spread));
            sum += texture(u_texture, uv + off).rgb * w;
            count += w;
        }
    }
    f_color = vec4(sum / count, 1.0);
'''), uniforms={
    "spread": {"glsl": "float", "min": 0.1, "max": 3.0, "default": 0.6, "description": "bleed radius"},
})

_register("shader_halftone_gpu", "GPU halftone dot screen", "filter", _filter_typed('''
    float gray = dot(orig.rgb, vec3(0.299, 0.587, 0.114));
    float cell = u_cell;
    vec2 q = fract(uv * u_resolution / cell);
    float d = length(q - 0.5);
    float dot_r = (1.0 - gray) * 0.5;
    float v = d < dot_r ? 0.0 : 1.0;
    f_color = vec4(vec3(v), 1.0);
'''), uniforms={
    "cell": {"glsl": "float", "min": 6.0, "max": 40.0, "default": 16.0, "description": "dot cell size"},
})

_register("shader_crt_gpu", "GPU CRT scanlines + bloom", "filter", _filter_typed('''
    float scan = sin(uv.y * u_resolution.y * 3.14159) * 0.5 + 0.5;
    float scanline = 1.0 - (1.0 - scan) * u_scanline;
    // chromatic shift at edges
    vec2 r_uv = uv + vec2(u_chroma * pow(abs(uv.x - 0.5) * 2.0, 2.0), 0.0);
    vec2 b_uv = uv - vec2(u_chroma * pow(abs(uv.x - 0.5) * 2.0, 2.0), 0.0);
    vec3 col;
    col.r = texture(u_texture, r_uv).r;
    col.g = texture(u_texture, uv).g;
    col.b = texture(u_texture, b_uv).b;
    col *= scanline;
    f_color = vec4(col, 1.0);
'''), uniforms={
    "scanline": {"glsl": "float", "min": 0.0, "max": 0.7, "default": 0.3, "description": "scanline darkness"},
    "chroma":   {"glsl": "float", "min": 0.0, "max": 0.004, "default": 0.001, "description": "RGB chroma shift"},
})

_register("shader_hologram", "GPU hologram / scan effect", "filter", _filter_typed('''
    float scan = sin(uv.y * u_resolution.y * 0.5 + u_time * 5.0) * 0.5 + 0.5;
    float scanline = 1.0 - pow(scan, 4.0) * u_scan;
    float edge = abs(uv.x - 0.5) * 2.0;
    float vignette = 1.0 - pow(edge, 3.0) * u_vignette;
    float shift = sin(uv.x * 50.0 + u_time * 3.0) * 0.02;
    vec2 q = uv + vec2(0.0, shift);
    vec3 col = texture(u_texture, q).rgb * scanline * vignette;
    float hue = sin(uv.y * 20.0 + u_time * 2.0) * u_hue + u_hue;
    col += vec3(hue, hue * 0.3, hue * 0.8);
    f_color = vec4(col, 1.0);
'''), uniforms={
    "scan":     {"glsl": "float", "min": 0.0, "max": 0.8, "default": 0.4, "description": "scanline depth"},
    "vignette": {"glsl": "float", "min": 0.0, "max": 1.0, "default": 0.5, "description": "edge vignette"},
    "hue":      {"glsl": "float", "min": 0.0, "max": 0.3, "default": 0.1, "description": "hue tint"},
})

_register("shader_mosaic_gpu", "GPU stained glass mosaic", "filter", _filter_typed('''
    float ts = u_tile_size;
    vec2 cell_uv = floor(uv * u_resolution / ts) * ts / u_resolution + ts / u_resolution * 0.5;
    f_color = texture(u_texture, cell_uv);
'''), uniforms={
    "tile_size": {"glsl": "float", "min": 10.0, "max": 60.0, "default": 30.0, "description": "tile size"},
})

_register("shader_edge_detect_gpu", "GPU Sobel edge detection", "filter", _filter_typed('''
    float tl = dot(texture(u_texture, uv + vec2(-step.x, -step.y)).rgb, vec3(0.299, 0.587, 0.114));
    float t  = dot(texture(u_texture, uv + vec2(0, -step.y)).rgb, vec3(0.299, 0.587, 0.114));
    float tr = dot(texture(u_texture, uv + vec2(step.x, -step.y)).rgb, vec3(0.299, 0.587, 0.114));
    float l  = dot(texture(u_texture, uv + vec2(-step.x, 0)).rgb, vec3(0.299, 0.587, 0.114));
    float r  = dot(texture(u_texture, uv + vec2(step.x, 0)).rgb, vec3(0.299, 0.587, 0.114));
    float bl = dot(texture(u_texture, uv + vec2(-step.x, step.y)).rgb, vec3(0.299, 0.587, 0.114));
    float b  = dot(texture(u_texture, uv + vec2(0, step.y)).rgb, vec3(0.299, 0.587, 0.114));
    float br = dot(texture(u_texture, uv + vec2(step.x, step.y)).rgb, vec3(0.299, 0.587, 0.114));
    float gx = -tl - 2.0*l - bl + tr + 2.0*r + br;
    float gy = -tl - 2.0*t - tr + bl + 2.0*b + br;
    float edge = sqrt(gx*gx + gy*gy);
    f_color = vec4(mix(orig.rgb, vec3(edge), u_strength), 1.0);
'''), uniforms={
    "strength": {"glsl": "float", "min": 0.0, "max": 2.0, "default": 1.0, "description": "edge blend"},
})

_register("shader_warhol", "GPU Warhol 4-panel duotone", "filter", _filter_typed('''
    float gray = dot(orig.rgb, vec3(0.299, 0.587, 0.114));
    gray = clamp(pow(gray, u_gamma), 0.0, 1.0);
    vec2 p = floor(uv * 2.0);
    vec3 c1, c2;
    if (p.x < 1.0 && p.y < 1.0) { c1 = vec3(0.8, 0.2, 0.2); c2 = vec3(1.0, 1.0, 0.4); }
    else if (p.x >= 1.0 && p.y < 1.0) { c1 = vec3(0.2, 0.4, 0.8); c2 = vec3(0.6, 0.2, 0.6); }
    else if (p.x < 1.0 && p.y >= 1.0) { c1 = vec3(0.2, 0.8, 0.2); c2 = vec3(0.4, 0.2, 0.8); }
    else { c1 = vec3(0.8, 0.6, 0.2); c2 = vec3(0.8, 0.2, 0.2); }
    f_color = vec4(mix(c1, c2, gray), 1.0);
'''), uniforms={
    "gamma": {"glsl": "float", "min": 0.25, "max": 3.0, "default": 1.0, "description": "panel contrast gamma (1.0 = classic)"},
})

_register("shader_duotone_gpu", "GPU duotone with color controls", "filter", _filter_typed('''
    float gray = dot(orig.rgb, vec3(0.299, 0.587, 0.114));
    f_color = vec4(mix(u_color_shadow, u_color_highlight, gray), 1.0);
'''), uniforms={
    "color_shadow":   {"glsl": "color", "default": "#3366cc", "description": "shadow color"},
    "color_highlight":{"glsl": "color", "default": "#ffcc33", "description": "highlight color"},
})

_register("shader_rgb_split", "GPU RGB channel separation", "filter", _filter_typed('''
    float shift = u_shift;
    vec2 r_uv = uv + vec2(shift, 0.0);
    vec2 b_uv = uv - vec2(shift, 0.0);
    float r = texture(u_texture, r_uv).r;
    float g = orig.g;
    float b = texture(u_texture, b_uv).b;
    f_color = vec4(r, g, b, 1.0);
'''), uniforms={
    "shift": {"glsl": "float", "min": 0.0, "max": 0.05, "default": 0.02, "description": "channel shift"},
})

_register("shader_caustics_gpu", "GPU caustic light overlay", "filter", _filter_typed('''
    float caustic = sin(uv.x * 30.0 + u_time) * cos(uv.y * 25.0 + u_time * 0.7);
    caustic += sin(uv.x * 50.0 - u_time * 1.3) * sin(uv.y * 40.0 + u_time * 0.5) * 0.5;
    caustic = max(0.0, caustic) * u_amount;
    vec3 light = vec3(0.8, 0.9, 1.0) * caustic;
    f_color = vec4(orig.rgb + light, 1.0);
'''), uniforms={
    "amount": {"glsl": "float", "min": 0.0, "max": 1.0, "default": 0.6, "description": "caustic strength"},
})

_register("shader_glitch_gpu", "GPU digital glitch artifacts", "filter", _filter_typed('''
    float band = floor(uv.y * 40.0 * u_intensity);
    float shift = sin(band * 7.0 + u_time * 5.0) * 0.05 * u_intensity;
    float noise = fract(sin(dot(uv * u_resolution, vec2(12.9898, 78.233))) * 43758.5453);
    float glitch = noise > (1.0 - u_intensity * 0.1) ? 1.0 : 0.0;
    vec2 q = uv + vec2(shift + glitch * 0.1, 0.0);
    f_color = texture(u_texture, q);
'''), uniforms={
    "intensity": {"glsl": "float", "min": 0.0, "max": 1.0, "default": 0.5, "description": "glitch intensity"},
})

_register("shader_posterize_gpu", "GPU posterization / color reduction", "filter", _filter_typed('''
    float nc = u_n_colors;
    vec3 col = floor(orig.rgb * nc) / nc;
    f_color = vec4(col, 1.0);
'''), uniforms={
    "n_colors": {"glsl": "float", "min": 2.0, "max": 16.0, "default": 9.0, "description": "color levels"},
})

# ── P0.5 LUT / color client-GPU twins (client-GPU live preview of nodes
# 10/11/39/77) ───────────────────────────────────────────────────────────────
# Additive: the server's CPU numpy nodes stay the authoritative export (two-tier
# precision). These bodies only drive the browser live preview. They reuse the
# prologue helpers (rot/hash21/noise/fbm/_INFERNO) so every new twin is covered
# automatically by test_webgl2_transform_is_valid + the gl330 legacy-equivalence
# parametrized tests.
#
# IMPORTANT (pitfall #15b): filter twins must NOT declare a local named `step` —
# the _filter_shader wrapper injects `vec2 step = 1.0 / u_resolution;` into
# main(). Use `px` / `gstep` / `cell_sz` instead.

_register("gradient_gpu",
          "Gradient generator (client-GPU twin of node 11)",
          "procedural", '''

// sRGB-ish gradient between two endpoint colors expressed as HSV-ish offsets.
vec3 hsv2rgb(vec3 c) {
    vec4 K = vec4(1.0, 2.0/3.0, 1.0/3.0, 3.0);
    vec3 p = abs(fract(c.xxx + K.xyz) * 6.0 - K.www);
    return c.z * mix(K.xxx, clamp(p - K.xxx, 0.0, 1.0), c.y);
}

void main() {
    // u_params.x = direction (radians, 0.5 = 0 rad; maps -PI..PI),
    // u_params.y = center_x (0.5 = middle), u_params.z = center_y,
    // u_params.w = gradient_type (0=linear,1=radial,2=concentric,3=angular,4=diamond).
    float dir = (u_cx - 0.5) * 6.2831853;
    vec2 ctr = vec2(u_cy, 0.5);
    vec2 p = v_uv - ctr;

    float t;
    int gtype = int(floor(0.5 * 4.999));
    if (gtype == 1) {                       // radial
        t = length(p);
    } else if (gtype == 2) {                // concentric (ring index)
        t = fract(length(p) * 8.0 + u_time * 0.05);
    } else if (gtype == 3) {                // angular
        float a = atan(p.y, p.x) - dir;
        t = 0.5 + 0.5 * (a / 3.14159265);
    } else if (gtype == 4) {                // diamond
        t = abs(p.x) + abs(p.y);
    } else {                                // linear
        t = 0.5 + 0.5 * dot(normalize(vec2(cos(dir), sin(dir)) + 1e-5), p);
    }
    t = clamp(t, 0.0, 1.0);

    // Two endpoint hues (cyan -> orange, echoing the node's color1/color2 defaults).
    vec3 c1 = hsv2rgb(vec3(0.62, 0.80, 0.55));
    vec3 c2 = hsv2rgb(vec3(0.05, 0.85, 0.95));
    vec3 col = mix(c1, c2, t);
    f_color = vec4(col, 1.0);
}
''',
    uniforms={
    "cx": {"glsl": "float", "min": 0.0, "max": 1.0, "default": 0.5, "description": "center x"},
    "cy": {"glsl": "float", "min": 0.0, "max": 1.0, "default": 0.5, "description": "center y"}
}
    )

_register("false_color_gpu",
          "False-color IR remap (client-GPU twin of node 77)",
          "filter", _filter_typed('''
    // u_strength: 0 = grayscale, 1 = full false-color blend. The node's
    // color_scheme is a STRING choice (pitfall #14) so the preview locks to
    // the thermal ramp; the CPU fn stays authoritative for all schemes.
    float lum = dot(orig.rgb, vec3(0.299, 0.587, 0.114));
    float s = clamp(u_strength, 0.0, 1.0);
    int scheme = 1;   // preview default: thermal ramp

    vec3 heat;
    if (scheme == 1) {                      // thermal (black-red-yellow-white)
        heat = vec3(smoothstep(0.0, 0.4, lum),
                    smoothstep(0.3, 0.75, lum),
                    smoothstep(0.7, 1.0, lum));
    } else if (scheme == 2) {               // vegetation (brown -> green)
        heat = vec3(0.35 * (1.0 - lum), 0.15 + 0.7 * lum, 0.15 * (1.0 - lum) + 0.1 * lum);
    } else if (scheme == 3) {               // urban (blue-gray -> cyan -> magenta)
        heat = vec3(0.3 + 0.4 * lum, 0.4 + 0.3 * lum, 0.6 + 0.4 * sin(lum * 3.14159));
    } else {                                // standard IR ramp (inferno-like)
        heat = vec3(lum * 1.4, lum * lum * 1.2, (1.0 - lum) * 0.6 + lum * 0.2);
    }
    heat = clamp(heat, 0.0, 1.0);
    vec3 col = mix(vec3(lum), heat, s);
    f_color = vec4(col, 1.0);
'''), uniforms={
        "strength": {"glsl": "float", "min": 0.0, "max": 1.0, "default": 0.5, "description": "false-color blend"},
    })

_register("image_to_mask_gpu",
          "Luminance/channel mask extraction (client-GPU twin of __image_to_mask__)",
          "filter", _filter_shader('''
    // Client-GPU live-preview twin of the Image to Mask compositing node.
    // The node's `mode` is a STRING choice param (luminance/red/green/blue/
    // alpha_from_white/invert_luminance) and per GPU pitfall #14 string params
    // are NOT mapped to numeric uniforms — so this twin renders the DEFAULT
    // `luminance` mode. The CPU fn stays authoritative for all six modes on
    // export. Output is a grayscale mask preview (mask replicated to RGB).
    float lum = dot(orig.rgb, vec3(0.299, 0.587, 0.114));
    float m = clamp(lum, 0.0, 1.0);
    f_color = vec4(vec3(m), 1.0);
'''))

_register("palette_gpu",
          "Color palette swatches (client-GPU twin of node 10)",
          "procedural", '''
void main() {
    // u_params.x = n_colors (2..32), u_params.y = saturation (0.5=auto),
    // u_params.z = hue_offset (0..1), u_params.w = value (0.5=auto).
    float ncols = floor(2.0 + u_hue_offset * 30.0);
    float hueOff = u_value;
    float sat = (u_saturation <= 0.0) ? 0.75 : clamp(u_saturation, 0.0, 1.0);
    float val = 0.95;

    // Arrange the hue ramp as a vertical band of swatches across the canvas.
    int col = int(floor(v_uv.x * ncols));
    float fn = (ncols > 0.5) ? (float(col) / ncols) : 0.0;
    float hue = fract(hueOff + fn);
    // vertical brightness variation inside each swatch so it reads as a palette.
    float band = step(0.08, v_uv.y) * step(v_uv.y, 0.92);
    float v = val * (0.55 + 0.45 * v_uv.y);
    vec3 col3 = clamp(vec3(
        abs(fract(hue + 1.0/3.0) * 2.0 - 1.0),
        abs(fract(hue) * 2.0 - 1.0),
        abs(fract(hue - 1.0/3.0) * 2.0 - 1.0)
    ), 0.0, 1.0);
    col3 = mix(vec3(dot(col3, vec3(0.299,0.587,0.114))), col3, sat) * v;
    f_color = vec4(mix(vec3(0.05), col3, band), 1.0);
}
''',
    uniforms={
    "hue_offset": {"glsl": "float", "min": 0.0, "max": 1.0, "default": 0.5, "description": "hue offset"},
    "saturation": {"glsl": "float", "min": 0.0, "max": 1.0, "default": 0.5, "description": "saturation"},
    "value": {"glsl": "float", "min": 0.0, "max": 1.0, "default": 0.5, "description": "value (kept for parity)"}
}
    )




# ── P0.6 field-eval client-GPU twins (client-GPU live preview of nodes
# 125/164) ──────────────────────────────────────────────────────────────────
# Additive: the server's CPU numpy nodes stay the authoritative export (two-tier
# precision). These bodies only drive the browser live preview. They reuse the
# prologue helpers so they are auto-covered by test_webgl2_transform_is_valid +
# the gl330 legacy-equivalence parametrized tests. Both nodes render a pure
# per-frame field that is a closed-form function of (uv, t), so the twin is an
# exact preview (no seeded-layout divergence, unlike pattern/generative nodes).
#
# IMPORTANT (pitfall #15): encode 0.5 as NEUTRAL so the default u_params
# (0.5,0.5,0.5,0.5) yields the node's canonical full view, not an extreme.

_register("chladni_gpu",
          "Chladni eigenmode field (client-GPU twin of node 125)",
          "procedural", '''
void main() {
    // u_params.x = m-mode (0.5 -> 3.0 canonical start, range 0.5..11.5),
    // u_params.y = n-mode (0.5 -> 3.0),
    // u_params.z = rotation (0.5 -> 0 rad, range -PI..PI),
    // u_params.w = phase shimmer (0.5 -> 0 rad, range -PI..PI).
    float m = 0.5 + u_m_start * 11.0;
    float n = 0.5 + u_n_start * 11.0;
    float rot_ang = (u_rotation_speed - 0.5) * 6.2831853;
    float ph = (u_phase_speed_x - 0.5) * 6.2831853;

    // Centered, normalized coords in [-1, 1] (matches node: xn = X/(W/2)).
    vec2 p = (v_uv - 0.5) * 2.0;
    // Coordinate rotation (plate spin).
    vec2 pr = rot(rot_ang) * p;

    // u_mn(x,y) = sin(m*PI*(x+1)/2 + φx) * sin(n*PI*(y+1)/2 + φy)
    float u = sin(m * 3.14159265 * (pr.x + 1.0) * 0.5 + ph)
            * sin(n * 3.14159265 * (pr.y + 1.0) * 0.5 + ph);

    // Centered, sharp sigmoid emphasis of zero-crossings (nodal lines).
    float sig = tanh(clamp(u, -4.0, 4.0) * 3.5);
    float gray = (sig + 1.0) * 0.5;
    // Nodal-line bright highlight: gaussian bell centered at u=0.
    float nodal = exp(-u * u * 8.0);
    gray = clamp(gray + nodal * 0.35, 0.0, 1.0);
    f_color = vec4(vec3(gray), 1.0);
}
''',
    uniforms={
    "m_start": {"glsl": "float", "min": 0.0, "max": 1.0, "default": 0.5, "description": "m mode"},
    "n_start": {"glsl": "float", "min": 0.0, "max": 1.0, "default": 0.5, "description": "n mode"},
    "rotation_speed": {"glsl": "float", "min": 0.0, "max": 1.0, "default": 0.5, "description": "plate rotation speed"},
    "phase_speed_x": {"glsl": "float", "min": 0.0, "max": 1.0, "default": 0.5, "description": "shimmer phase"}
}
    )

_register("moire_gpu",
          "Moiré interference (client-GPU twin of node 164)",
          "procedural", '''
void main() {
    // u_params.x = mode (0=radial,1=linear,2=spiral,3=hex),
    // u_params.y = speed1 (0.5 -> ~1.0), u_params.z = speed2 (0.5 -> ~1.28),
    // u_params.w = frequency (0.5 -> 20).
    int mode = int(floor(u_mode * 3.999));
    float s1 = 0.1 + u_speed1 * 1.9;      // ~1.0 at default
    float s2 = 0.1 + u_speed2 * 1.9;      // ~1.28 at default
    float freq = 5.0 + u_frequency * 45.0;   // 20 at default
    float t = u_time * 0.05;               // matches node: t = fr*0.05

    vec2 res = u_resolution;
    vec2 p = (v_uv - 0.5) * res;           // pixel-centered coords
    float scale = 1.0 / max(res.x, res.y) * 2.0 * 3.14159265;

    float g1, g2;
    float a1 = s1 * t, a2 = s2 * t;
    if (mode == 1) {                       // linear gratings
        g1 = 0.5 + 0.5 * sin(freq * (p.x * cos(a1) + p.y * sin(a1)) * scale);
        g2 = 0.5 + 0.5 * sin(freq * (p.x * cos(a2) + p.y * sin(a2)) * scale);
    } else if (mode == 2) {                // spiral
        float r = length(p);
        float th = atan(p.y, p.x);
        g1 = 0.5 + 0.5 * sin(freq * (r * scale + th / 6.2831853) * 6.2831853 + a1);
        g2 = 0.5 + 0.5 * sin(freq * (r * scale + th / 6.2831853) * 6.2831853 + a2);
    } else if (mode == 3) {                // hex (3-grating sum)
        float acc = 0.0;
        acc += 0.5 + 0.5 * sin(freq * (p.x) * scale + s1 * t);
        acc += 0.5 + 0.5 * sin(freq * (p.x * cos(1.0471975) + p.y * sin(1.0471975)) * scale + (s1 + 0.3) * t);
        acc += 0.5 + 0.5 * sin(freq * (p.x * cos(2.0943951) + p.y * sin(2.0943951)) * scale + (s1 + 0.6) * t);
        acc = clamp(acc / 3.0, 0.0, 1.0);
        f_color = vec4(vec3(acc), 1.0);
        return;
    } else {                               // radial (default)
        float r = length(p);
        g1 = 0.5 + 0.5 * sin(freq * r * scale);
        g2 = 0.5 + 0.5 * sin(freq * r * scale + a2);
    }
    float g = clamp(g1 * g2 * 2.0, 0.0, 1.0);
    f_color = vec4(vec3(g), 1.0);
}
''',
    uniforms={
    "mode": {"glsl": "float", "min": 0.0, "max": 3.0, "default": 0.0, "description": "grating mode (0-3)"},
    "speed1": {"glsl": "float", "min": 0.0, "max": 1.0, "default": 0.5, "description": "speed 1"},
    "speed2": {"glsl": "float", "min": 0.0, "max": 1.0, "default": 0.5, "description": "speed 2"},
    "frequency": {"glsl": "float", "min": 0.0, "max": 1.0, "default": 0.5, "description": "grating frequency"}
}
    )


# ── Horizon-Based Ambient Occlusion (node 425) ──────────────────────────────
# Closed-form client-GPU twin of the CPU HBAO node. Given a procedural fbm
# height field h(uv), AO at each pixel is the fraction of the sky hemisphere
# NOT blocked by higher neighbours: walk N azimuth rays, record the max horizon
# silhouette angle phi = atan2((h(q)-h(p))*k, dist) along each ray, sum the
# visible fraction 0.5*(1+cos(phi)). No inter-frame state -> pure f(uv,t), the
# P0.6 field-eval family. Helpers are inlined (late _INFERNO_GPU is below this
# region — pitfall #17). Shader-only knobs (EXAG/jitter hash) are documented;
# the node's REAL params are routed by name via param_map in gpu_shaders.py.
_HBAO_HELPERS = '''
float _hbao_hash(vec2 p) {
    p = fract(p * vec2(123.34, 456.21));
    p += dot(p, p + 45.32);
    return fract(p.x * p.y);
}
float _hbao_vnoise(vec2 x) {
    vec2 xi = floor(x); vec2 xf = fract(x);
    vec2 u = xf * xf * (3.0 - 2.0 * xf);
    float a = _hbao_hash(xi);
    float b = _hbao_hash(xi + vec2(1.0, 0.0));
    float c = _hbao_hash(xi + vec2(0.0, 1.0));
    float d = _hbao_hash(xi + vec2(1.0, 1.0));
    return mix(mix(a, b, u.x), mix(c, d, u.x), u.y) * 2.0 - 1.0;
}
float _hbao_fbm(vec2 x, int oct) {
    float outv = 0.0, amp = 1.0, freq = 1.0, norm = 0.0;
    for (int o = 0; o < 6; o++) {
        if (o >= oct) break;
        float a = 2.39996323 * float(o + 1);
        float ca = cos(a), sa = sin(a);
        vec2 r = vec2(x.x * freq * ca - x.y * freq * sa,
                      x.x * freq * sa + x.y * freq * ca);
        outv += amp * _hbao_vnoise(r);
        norm += amp; amp *= 0.5; freq *= 2.0;
    }
    return outv / max(norm, 1e-6);
}
vec3 _hbao_inferno(float t) {
    t = clamp(t, 0.0, 1.0);
    const vec3 c0 = vec3(0.00021894, 0.00165100, -0.01948090);
    const vec3 c1 = vec3(0.10651342, 0.56395644, 3.93271239);
    const vec3 c2 = vec3(11.60249308, -3.97285397, -15.94239411);
    const vec3 c3 = vec3(-41.70399613, 17.43639888, 44.35414520);
    const vec3 c4 = vec3(77.16293570, -33.40235894, -81.80730926);
    const vec3 c5 = vec3(-71.31942824, 32.62606426, 73.20951986);
    const vec3 c6 = vec3(25.13112622, -12.24266895, -23.07032500);
    return c0 + t * (c1 + t * (c2 + t * (c3 + t * (c4 + t * (c5 + t * c6)))));
}
vec3 _hbao_hsv2rgb(vec3 c) {
    vec4 K = vec4(1.0, 2.0 / 3.0, 1.0 / 3.0, 3.0);
    vec3 p = abs(fract(c.xxx + K.xyz) * 6.0 - K.www);
    return c.z * mix(K.xxx, clamp(p - K.xxx, 0.0, 1.0), c.y);
}
'''

_register("hbao_gpu",
          "Horizon-Based Ambient Occlusion of a procedural height field (node 425)",
          "procedural",
          _HBAO_HELPERS + '''
void main() {
    // ── Named uniforms (auto-declared) match node 425's REAL params ──
    // freq, octaves, height_scale, radius, directions, steps, jitter,
    // mode (choice), light_az, light_el, ambient, contrast, colormode (choice).
    // u_time drives the anim_mode (evolve/drift/rotate_light) on the GPU path.
    vec2 res = u_resolution;
    vec2 uv = v_uv;
    vec2 p = (uv - 0.5) * 2.0;

    // ── Animation (GPU live-preview clock) ──
    int amode = int(clamp(floor(u_anim_mode + 0.5), 0.0, 3.0)); // 0 none,1 evolve,2 drift,3 rotate_light
    float t = u_time;
    vec2 off = vec2(0.0);
    float wblend = 0.0;
    if (amode == 2) { off = vec2(t * 0.18, t * 0.08); }       // drift
    else if (amode == 1) { wblend = 0.5 - 0.5 * cos(t); }     // evolve (full 0..1 blend)

    // ── Procedural height field ──
    vec2 fcoord = p * max(u_freq, 0.5);
    float h = _hbao_fbm(fcoord + off, int(clamp(u_octaves, 1.0, 6.0)));
    if (amode == 1) {
        float ang = t * 0.6;
        float ca = cos(ang), sa = sin(ang);
        vec2 rx = fcoord * ca - fcoord.y * sa;
        vec2 ry = fcoord * sa + fcoord.y * ca;
        float h2 = _hbao_fbm(rx + off, int(clamp(u_octaves, 1.0, 6.0)));
        h = mix(h, h2, wblend);
    }
    h = h * 0.5 + 0.5;  // [-1,1] -> [0,1]

    // ── Per-pixel disk rotation (jitter) ──
    float jr = fract(_hbao_hash(uv * res * 0.013 + 3.1)
                     + _hbao_hash(uv * res * 0.027 + 7.7));
    float rot = jr * 6.2831853 * u_jitter;
    float cr = cos(rot), sr = sin(rot);

    int D = int(clamp(u_directions, 3.0, 16.0));
    int K = int(clamp(u_steps, 4.0, 32.0));
    float step_px = max(1.0, u_radius) / float(K);
    // World-height slope factor (EXAG keeps smooth-field relief non-trivial).
    float k = u_height_scale * 12.0;

    float vis_sum = 0.0;
    for (int d = 0; d < 16; d++) {
        if (d >= D) break;
        float base_ang = (6.2831853 / float(D)) * float(d);
        vec2 dd = vec2(cos(base_ang), sin(base_ang));
        vec2 ddir = vec2(dd.x * cr - dd.y * sr, dd.x * sr + dd.y * cr);
        float horizon = -1e9;
        for (int kk = 1; kk <= 32; kk++) {
            if (kk > K) break;
            float d_k = float(kk) * step_px;
            vec2 sp = p * (res * 0.5) + ddir * d_k;   // sample in pixel space
            vec2 suv = sp / res + 0.5;
            suv = clamp(suv, 0.0, 1.0);
            vec2 sc = (suv - 0.5) * 2.0 * max(u_freq, 0.5);
            float hq = _hbao_fbm(sc + off, int(clamp(u_octaves, 1.0, 6.0))) * 0.5 + 0.5;
            float phi = atan((hq - h) * k, d_k);
            horizon = max(horizon, phi);
        }
        horizon = clamp(horizon, 0.0, radians(85.0));
        vis_sum += 0.5 * (1.0 + cos(horizon));
    }
    float ao = vis_sum / float(D);

    // Radius-driven AO intensity gain (the honest, visible radius lever).
    float gain = clamp(0.4 + 0.05 * (u_radius - 4.0), 0.3, 4.0);
    ao = clamp(0.5 + (ao - 0.5) * gain, 0.0, 1.0);

    // ── Hillshade (Lambert from a movable light) ──
    float az = radians(u_light_az);
    float el = radians(u_light_el);
    if (amode == 3) az += t;   // rotate_light sweep
    vec3 L = vec3(cos(el) * cos(az), cos(el) * sin(az), sin(el));
    // surface normal from a cheap height gradient (finite differences)
    float e = 1.0 / max(res.x, res.y);
    vec2 scp = p * max(u_freq, 0.5);
    float hx = _hbao_fbm(scp + vec2(e, 0.0) * 50.0 + off, int(clamp(u_octaves, 1.0, 6.0))) * 0.5 + 0.5;
    float hy = _hbao_fbm(scp + vec2(0.0, e) * 50.0 + off, int(clamp(u_octaves, 1.0, 6.0))) * 0.5 + 0.5;
    vec3 nrm = normalize(vec3(-k * (hx - h) * 30.0, -k * (hy - h) * 30.0, 1.0));
    float shade = max(dot(nrm, L), 0.0);
    shade = u_ambient + (1.0 - u_ambient) * shade;

    // ── Compose ──
    int cmode = int(clamp(floor(u_colormode + 0.5), 0.0, 4.0));
    int mode = int(clamp(floor(u_mode + 0.5), 0.0, 2.0));  // 0 ao,1 shaded,2 height
    float disp;
    if (mode == 2) disp = h;
    else if (mode == 0) disp = clamp((ao - 0.5) * u_contrast + 0.5, 0.0, 1.0);
    else { float comb = clamp(ao * shade, 0.0, 1.0);
           disp = clamp((comb - 0.5) * u_contrast + 0.5, 0.0, 1.0); }

    vec3 rgb;
    if (cmode == 0) rgb = vec3(disp * 0.55, disp * 0.78, disp);          // steel
    else if (cmode == 1) rgb = vec3(disp, disp * 0.72, disp * 0.28);     // amber
    else if (cmode == 2) rgb = _hbao_inferno(disp);                     // inferno
    else if (cmode == 3) {                                               // spectral
        vec3 hsv = vec3(disp, clamp(0.25 + disp * 0.6, 0.0, 1.0), clamp(0.2 + disp * 0.9, 0.0, 1.0));
        rgb = _hbao_hsv2rgb(hsv);
    } else rgb = vec3(disp);                                             // grayscale
    f_color = vec4(clamp(rgb, 0.0, 1.0), 1.0);
}
''',
          uniforms={
              "freq":        {"glsl": "float", "min": 1.0, "max": 12.0, "default": 5.0,
                             "description": "noise frequency of the height field"},
              "octaves":     {"glsl": "int", "min": 1, "max": 6, "default": 4,
                             "description": "fbm octaves"},
              "height_scale": {"glsl": "float", "min": 0.2, "max": 6.0, "default": 2.0,
                               "description": "world height per unit luminance (AO strength)"},
              "radius":      {"glsl": "float", "min": 4.0, "max": 64.0, "default": 24.0,
                              "description": "AO sampling radius in pixels"},
              "directions":  {"glsl": "int", "min": 3, "max": 16, "default": 8,
                              "description": "number of azimuth rays"},
              "steps":       {"glsl": "int", "min": 4, "max": 32, "default": 16,
                             "description": "horizon samples per ray"},
              "jitter":      {"glsl": "float", "min": 0.0, "max": 1.0, "default": 1.0,
                             "description": "per-pixel disk rotation (breaks banding)"},
              "mode":        {"glsl": "choice", "choices": ["ao", "shaded", "height"],
                             "default": "shaded", "description": "output composition"},
              "light_az":    {"glsl": "float", "min": 0.0, "max": 360.0, "default": 135.0,
                             "description": "light azimuth in degrees"},
              "light_el":    {"glsl": "float", "min": 5.0, "max": 85.0, "default": 45.0,
                             "description": "light elevation in degrees"},
              "ambient":     {"glsl": "float", "min": 0.0, "max": 1.0, "default": 0.3,
                             "description": "ambient term of the hillshade combine"},
              "contrast":    {"glsl": "float", "min": 0.5, "max": 3.0, "default": 1.0,
                             "description": "tone contrast of the AO display"},
              "colormode":   {"glsl": "choice", "choices": ["steel", "amber", "inferno", "spectral", "grayscale"],
                             "default": "steel", "description": "output color"},
              "anim_mode":   {"glsl": "choice", "choices": ["none", "evolve", "drift", "rotate_light"],
                             "default": "none", "description": "animation mode"},
          }
          )


# ── Thin-Film Interference (node 1004) ──────────────────────────────────────
# Closed-form client-GPU twin of the CPU spectral thin-film node. A film of
# refractive index n and thickness d reflects a two-beam interference spectrum
# with phase delta = 4*pi*n*d*cos(theta_t)/lambda + pi (single phase reversal at
# the air->film interface, Hecht Optics 4ed §9.5). The reflected spectrum is
# integrated against the CIE 1931 2-deg colour-matching functions (Wyman, Sloan
# & Shirley 2013 analytic gaussian fit) and converted XYZ->linear-sRGB, so the
# full violet->magenta->red band wrap is reproduced rather than a naive RGB pick.
# A procedural fbm thickness field paints the bands; a drainage term thins the
# film toward the top like a real draining bubble. No inter-frame state -> pure
# f(uv,t), the P0.6 field-eval family. CPU numpy node 1004 stays authoritative
# for export; this twin is the live-preview approximation (35-sample spectrum vs
# the CPU's 69, single fbm scale vs the CPU's rng-jittered scale).
_TF_HELPERS = '''
float _tf_hash(vec2 p) {
    p = fract(p * vec2(123.34, 456.21));
    p += dot(p, p + 45.32);
    return fract(p.x * p.y);
}
float _tf_vnoise(vec2 x) {
    vec2 xi = floor(x); vec2 xf = fract(x);
    vec2 u = xf * xf * (3.0 - 2.0 * xf);
    float a = _tf_hash(xi);
    float b = _tf_hash(xi + vec2(1.0, 0.0));
    float c = _tf_hash(xi + vec2(0.0, 1.0));
    float d = _tf_hash(xi + vec2(1.0, 1.0));
    return mix(mix(a, b, u.x), mix(c, d, u.x), u.y) * 2.0 - 1.0;
}
float _tf_fbm(vec2 x) {
    float outv = 0.0, amp = 1.0, freq = 1.0, norm = 0.0;
    for (int o = 0; o < 5; o++) {
        outv += amp * _tf_vnoise(x * freq);
        norm += amp; amp *= 0.5; freq *= 2.0;
    }
    return outv / max(norm, 1e-6);
}
// One-sided gaussian (Wyman et al. 2013 CIE-CMF building block).
float _tf_g(float x, float mu, float s1, float s2) {
    float s = x < mu ? s1 : s2;
    float t = (x - mu) / s;
    return exp(-0.5 * t * t);
}
vec3 _tf_cmf(float lam) {
    float xb = 1.056 * _tf_g(lam, 599.8, 37.9, 31.0)
             + 0.362 * _tf_g(lam, 442.0, 16.0, 26.7)
             - 0.065 * _tf_g(lam, 501.1, 20.4, 26.2);
    float yb = 0.821 * _tf_g(lam, 568.8, 46.9, 40.5)
             + 0.286 * _tf_g(lam, 530.9, 16.3, 31.1);
    float zb = 1.217 * _tf_g(lam, 437.0, 11.8, 36.0)
             + 0.681 * _tf_g(lam, 459.0, 26.0, 13.8);
    return vec3(xb, yb, zb);
}
vec3 _tf_xyz2srgb(vec3 xyz) {
    mat3 M = mat3( 3.2404542, -0.9692660,  0.0556434,
                  -1.5371385,  1.8760108, -0.2040259,
                  -0.4985314,  0.0415560,  1.0572252);
    vec3 lin = max(M * xyz, 0.0);
    vec3 a = vec3(0.055);
    bvec3 hi = greaterThan(lin, vec3(0.0031308));
    vec3 srgb = mix(12.92 * lin, 1.055 * pow(lin, vec3(1.0/2.4)) - a, vec3(hi));
    return clamp(srgb, 0.0, 1.0);
}
'''

_register("thin_film_spectral_gpu",
          "Spectral thin-film interference iridescence (client-GPU twin of node 1004)",
          "procedural",
          _TF_HELPERS + '''
void main() {
    // ── Named uniforms match node 1004's REAL params ──
    // thickness, thickness_range, ior, drainage, view_angle, brightness,
    // anim_speed + the choice param anim_mode (none/flow/swirl/pulse).
    vec2 res = u_resolution;
    vec2 uv = v_uv;
    float mx = max(res.x, res.y);
    // Match the CPU coordinate frame: u = x/max, v = y/max in [0, ~1].
    float u = uv.x * res.x / mx;
    float v = (1.0 - uv.y) * res.y / mx;   // flip: CPU row 0 is the top

    int amode = int(clamp(floor(u_anim_mode + 0.5), 0.0, 3.0)); // 0 none,1 flow,2 swirl,3 pulse
    float t = (amode == 0) ? 0.0 : u_time * max(u_anim_speed, 0.0);

    // ── Thickness field (procedural fbm bands + drainage) ──
    float cx = 0.5, cy = 0.5;
    float dx = u - cx, dy = v - cy;
    if (amode == 2) {           // swirl: rotate the sample frame
        float ca = cos(t), sa = sin(t);
        float rx = ca * dx - sa * dy;
        float ry = sa * dx + ca * dy;
        dx = rx; dy = ry;
    }
    float fx = dx + (amode == 1 ? t : 0.0);   // flow: bands travel in x
    float fy = dy;
    float scale = 4.0;          // CPU uses 3..5 (rng); twin fixes a mid value
    float h = _tf_fbm(vec2(fx * scale * 6.0, fy * scale * 6.0));
    h = 0.5 + 0.5 * h;
    // Drainage: film thins toward the top (v small at top).
    h = clamp(h - u_drainage * (0.5 - v), 0.0, 1.0);
    float thickness01 = clamp(h, 0.0, 1.0);

    // pulse: thickness range breathes (smooth offset sine, no cusp).
    float trange = u_thickness_range;
    if (amode == 3) trange *= (0.1 + 0.9 * (0.5 + 0.5 * sin(t)));

    // ── View-angle cos(theta_t) via Snell (dome normal tilt) ──
    float nx = (uv.x - 0.5) * u_view_angle;
    float ny = (uv.y - 0.5) * u_view_angle;
    float sin_i = clamp(sqrt(nx * nx + ny * ny), 0.0, 0.999);
    float cosT = clamp(sqrt(clamp(1.0 - (sin_i * sin_i) / (u_ior * u_ior), 1e-4, 1.0)), 0.05, 1.0);

    // ── Spectral interference integral against CIE CMF ──
    float d_nm = u_thickness + trange * (thickness01 - 0.5);
    vec3 xyz = vec3(0.0);
    vec3 white = vec3(0.0);
    const float PI = 3.14159265;
    for (int k = 0; k < 35; k++) {
        float lam = 380.0 + float(k) * 10.0;   // 380..720 nm, 10 nm step
        vec3 cmf = _tf_cmf(lam);
        float delta = (4.0 * PI * u_ior * d_nm * cosT) / lam + PI;
        float Rk = 0.5 * (1.0 + cos(delta));
        xyz += cmf * Rk;
        white += cmf;
    }
    xyz /= max(white.y, 1e-6);
    xyz *= u_brightness;
    vec3 rgb = _tf_xyz2srgb(xyz);
    f_color = vec4(rgb, 1.0);
}
''',
          uniforms={
              "thickness":       {"glsl": "float", "min": 100.0, "max": 1200.0, "default": 380.0,
                                  "description": "base film thickness (nm); dominant colour = d*n"},
              "thickness_range": {"glsl": "float", "min": 0.0, "max": 900.0, "default": 420.0,
                                  "description": "thickness variation (nm) — drives the colour bands"},
              "ior":             {"glsl": "float", "min": 1.05, "max": 2.5, "default": 1.33,
                                  "description": "film refractive index (soap 1.33, oil 1.45)"},
              "drainage":        {"glsl": "float", "min": 0.0, "max": 1.0, "default": 0.35,
                                  "description": "vertical thinning gradient (drains toward top)"},
              "view_angle":      {"glsl": "float", "min": 0.0, "max": 1.2, "default": 0.5,
                                  "description": "surface tilt (rad) — oblique edge colour shift"},
              "brightness":      {"glsl": "float", "min": 0.2, "max": 1.5, "default": 0.9,
                                  "description": "overall reflected intensity scale"},
              "anim_speed":      {"glsl": "float", "min": 0.1, "max": 3.0, "default": 1.0,
                                  "description": "animation speed multiplier"},
              "anim_mode":       {"glsl": "choice", "choices": ["none", "flow", "swirl", "pulse"],
                                  "default": "none", "description": "animation mode"},
          }
          )


_register("spherical_harmonics_gpu",
          "Spherical harmonics banding (client-GPU twin of node 104)",
          "procedural", '''
void main() {
    // Named uniforms (auto-declared) match node 104's real params; the client
    // reads them by name (pitfall #14b). Closed-form approximation of
    // spherical-harmonic Y_l^m banding projected to a 2D (theta, phi) map.
    float L = max(1.0, floor(u_max_l + 0.5));
    float th = v_uv.y * 3.14159265;     // polar angle 0..pi
    float ph = v_uv.x * 6.2831853;      // azimuth 0..2pi
    float t = u_time * u_anim_speed * 0.15;

    float f = 0.0;
    float wsum = 0.0;
    for (int li = 1; li <= 8; li++) {
        if (float(li) > L) break;
        float fl = float(li);
        float Pl = cos(fl * th);        // meridional banding (Legendre-like)
        for (int mi = 0; mi <= 8; mi++) {
            if (float(mi) > fl) break;
            float fm = float(mi);
            float az = cos(fm * ph + (fm == 0.0 ? 0.0 : t) + fm * 1.3);
            float w = 1.0 / (fl + 1.0);
            f += Pl * az * w;
            wsum += w;
        }
    }
    f /= max(wsum, 1.0);

    // Twist: azimuthal phase shear (node twist_wave character).
    float twist = sin(ph * (1.0 + u_twist_amplitude) + t * 2.0) * u_osc_spread * 0.15;
    f += twist;

    float val = clamp(f * 0.5 + 0.5, 0.0, 1.0);
    val = pow(val, 1.0 / max(u_glow_strength, 0.2));   // glow emphasis
    float gray = clamp(val * u_amplitude, 0.0, 1.0);
    f_color = vec4(vec3(gray), 1.0);
}
''',
    uniforms={
    "max_l": {"glsl": "float", "min": 1.0, "max": 8.0, "default": 5.0, "description": "max shell l"},
    "amplitude": {"glsl": "float", "min": 0.5, "max": 3.0, "default": 1.5, "description": "brightness amplitude"},
    "glow_strength": {"glsl": "float", "min": 0.5, "max": 3.0, "default": 1.5, "description": "glow emphasis"},
    "anim_speed": {"glsl": "float", "min": 0.1, "max": 5.0, "default": 1.0, "description": "animation speed"},
    "twist_amplitude": {"glsl": "float", "min": 0.5, "max": 5.0, "default": 2.0, "description": "twist amplitude"},
    "osc_spread": {"glsl": "float", "min": 0.0, "max": 5.0, "default": 1.5, "description": "oscillator spread"}
    }
    )

_register("plasma_gpu",
          "Plasma fractal heightfield (client-GPU twin of node 31)",
          "procedural", '''
void main() {
    // Typed uniforms match node 31's real numeric params (pitfall #14):
    //   u_size          plasma grid size (64..1024) -> spatial frequency
    //   u_roughness     initial roughness amplitude (0.05..2) -> warp strength
    //   u_octaves       fBm octaves (1..6) -> detail iterations
    //   u_seed_strength (0..1) -> heightfield seed blend
    // Closed-form diamond-square approximation: domain-warped fBm heightfield
    // with a cosine plasma palette. CPU numpy node stays authoritative for
    // export. Uses prologue helpers noise/fbm (no local redefinition).
    vec2 uv = v_uv;
    float t = u_time * 0.15;                 // height_warp-style evolution

    float freq = u_size * 0.012;             // 512 -> ~6.1 cycles across
    vec2 q = uv * freq;

    // animated domain warp (roughness drives amplitude)
    float warp = u_roughness * 0.6;
    q += warp * vec2(fbm(q + vec2(0.0, t)),
                     fbm(q + vec2(5.2, 1.3 - t)));

    // fBm accumulation (constant loop bound, break on octave count)
    float h = 0.0;
    float amp = 0.5;
    float fr = 1.0;
    int oct = int(clamp(u_octaves, 1.0, 6.0));
    for (int i = 0; i < 6; i++) {
        if (i >= oct) break;
        h += amp * fbm(q * fr + vec2(float(i) * 1.7, t * 0.5));
        fr *= 2.0;
        amp *= 0.5;
    }
    h = mix(h, fract(h + u_seed_strength), 0.5);
    h = clamp(h * 0.5 + 0.5, 0.0, 1.0);

    // cosine plasma palette + height shading
    vec3 col = 0.5 + 0.5 * cos(6.2831853 * (h + vec3(0.0, 0.33, 0.67)));
    col *= 0.6 + 0.4 * h;
    f_color = vec4(col, 1.0);
}
''',
    uniforms={
        "size": {"glsl": "float", "min": 64.0, "max": 1024.0, "default": 512.0,
                 "description": "plasma grid size (frequency)"},
        "roughness": {"glsl": "float", "min": 0.05, "max": 2.0, "default": 0.5,
                      "description": "domain-warp strength"},
        "octaves": {"glsl": "float", "min": 1.0, "max": 6.0, "default": 3.0,
                    "description": "fBm octaves"},
        "seed_strength": {"glsl": "float", "min": 0.0, "max": 1.0, "default": 0.6,
                          "description": "heightfield seed blend"}
    }
    )

_register("spectral_tapestry_gpu",
          "Spectral tapestry interference (client-GPU twin of node 161)",
          "procedural", '''
void main() {
    // Named uniforms match node 161's real params; client reads by name
    // (pitfall #14b). Closed-form approximation of the spectral-PDE field: a
    // golden-angle fan of drifting sinusoidal gratings, contrast-shaped by
    // coupling. The CPU spectral simulation stays authoritative for export.
    float nm = clamp(floor(u_n_modes), 6.0, 40.0);
    float c = u_coupling;
    vec2 res = u_resolution;
    vec2 p = (v_uv - 0.5) * res;
    float scale = 6.2831853 / max(res.x, res.y);
    float t = u_time * (0.05 + u_drift_speed * 20.0);

    float acc = 0.0;
    for (int i = 0; i < 40; i++) {
        if (float(i) >= nm) break;
        float k = float(i);
        float ang = k * 2.39996323;           // golden angle fan
        vec2 d = vec2(cos(ang), sin(ang));
        float w = 3.0 + k * (1.0 + c * 2.0);  // finer modes with coupling
        float ph = t * (0.5 + 0.05 * k) + k * 1.7;
        acc += sin(dot(p, d) * scale * w + ph);
    }
    acc /= max(nm, 1.0);

    float val = acc * 0.5 + 0.5;
    // coupling sharpens the interference (storm-like thresholding)
    val = clamp(mix(val, smoothstep(0.35, 0.65, val), clamp(c * 0.5, 0.0, 1.0)), 0.0, 1.0);
    val += (noise(p * 0.05 + vec2(t)) - 0.5) * u_noise * 4.0;   // stochastic grain
    val = clamp(val, 0.0, 1.0);
    f_color = vec4(vec3(val), 1.0);
}
''',
    uniforms={
    "n_modes": {"glsl": "float", "min": 8.0, "max": 80.0, "default": 25.0, "description": "mode count"},
    "coupling": {"glsl": "float", "min": 0.0, "max": 2.0, "default": 0.4, "description": "mode coupling"},
    "drift_speed": {"glsl": "float", "min": 0.0, "max": 0.05, "default": 0.005, "description": "drift speed"},
    "noise": {"glsl": "float", "min": 0.0, "max": 0.1, "default": 0.01, "description": "stochastic noise"}
    }
    )

_register("dunes_gpu",
          "Sand dune migration height field (client-GPU twin of node 172)",
          "procedural", '''
void main() {
    // u_params.x = wind_strength (0.5 -> ~0.6 default),
    // u_params.y = sediment_supply (0.5 -> ~0.8 default),
    // u_params.z/.w unused (wind rotation + render style are procedural-only
    // live-preview defaults; CPU export is authoritative — two-tier precision).
    // Closed-form wave superposition of (uv, t) -> exact parity preview, no
    // seeded-layout divergence (same family as 125 Chladni / 164 Moiré).
    float wind = 0.1 + u_wind_strength * 1.0;     // ~0.6 at neutral 0.5
    float sed  = 0.1 + u_sediment_supply * 1.4;     // ~0.8 at neutral 0.5
    float t = u_time * 0.05;                 // matches node: t = frame*0.04

    // Slowly rotating wind (node "evolve" mode) -> migrating, merging field.
    float windAngle = t * 0.15;
    vec2 res = u_resolution;
    vec2 p = (v_uv - 0.5) * res;             // pixel-centered coords

    float h = 0.0;

    // ── Small ripples (10 waves spread around wind direction) ──
    for (int i = 0; i < 10; i++) {
        float fi = float(i);
        float amp = (0.22 / (fi + 1.0)) * wind;
        float wl = 8.0 + fi * 5.0;
        float aoff = (fi - 5.0) * 0.055;
        float ang = windAngle + aoff;
        vec2 d = vec2(cos(ang), sin(ang));
        float ph = fi * 1.3 + t * (0.006 + fi * 0.0006);
        float proj = (p.x * d.x + p.y * d.y) / wl + ph;
        h += amp * (sin(proj * 6.2831853) * 0.5 + 0.5);
    }

    // ── Large dune features (4 waves, longer wavelength) ──
    for (int i = 0; i < 4; i++) {
        float fi = float(i);
        float amp = (0.30 + fi * 0.06) * sed;
        float wl = 55.0 + fi * 30.0;
        float wob = sin(t * 0.008 + fi * 2.0) * 0.15;
        float ang = windAngle + wob;
        vec2 d = vec2(cos(ang), sin(ang));
        float ph = fi * 2.5 + t * (0.003 + fi * 0.0004);
        float proj = (p.x * d.x + p.y * d.y) / wl + ph;
        h += amp * (sin(proj * 6.2831853) * 0.5 + 0.5);
    }

    // ── Subtle stochastic texture ──
    h += noise(p * 0.05 + vec2(t * 0.1)) * 0.15;

    // ── Normalize + hypsometric tint (matches node render_style=height) ──
    float hn = clamp(h * 0.55, 0.0, 1.0);
    vec3 col;
    if (hn < 0.20) {
        col = mix(vec3(0.0, 0.0, 1.0), vec3(0.0, 1.0, 1.0), hn / 0.20);
    } else if (hn < 0.40) {
        col = mix(vec3(0.0, 1.0, 1.0), vec3(0.0, 1.0, 0.0), (hn - 0.20) / 0.20);
    } else if (hn < 0.60) {
        col = mix(vec3(0.0, 1.0, 0.0), vec3(1.0, 1.0, 0.0), (hn - 0.40) / 0.20);
    } else if (hn < 0.80) {
        col = mix(vec3(1.0, 1.0, 0.0), vec3(1.0, 0.5, 0.25), (hn - 0.60) / 0.20);
    } else {
        col = mix(vec3(1.0, 0.5, 0.25), vec3(1.0, 0.96, 1.0), (hn - 0.80) / 0.20);
    }
    f_color = vec4(col, 1.0);
}
''',
    uniforms={
    "wind_strength": {"glsl": "float", "min": 0.0, "max": 1.0, "default": 0.5, "description": "wind strength"},
    "sediment_supply": {"glsl": "float", "min": 0.0, "max": 1.0, "default": 0.5, "description": "sediment supply"}
}
    )

# ── Caustics client-GPU twin (node 513) ───────────────────────────────────
# Closed-form f(uv, t) preview of GPU-Gems water caustics (Guardado &
# Sánchez-Crespo 2003). The CPU node 513 computes 1/|det J| of the floor->
# refracted-landing map via a finite-difference Jacobian; the Hessian of the
# sum-of-sines height field is available analytically (second derivative of a
# sine is -k^2 sin), so the same inverse-area-magnification caustic is a pure
# function of (uv, t) here — exact-family parity preview, no seeded-layout
# divergence. Wave directions/freqs are fixed (seed-independent live preview);
# the CPU numpy node stays the authoritative export (two-tier precision).
_register("caustics513_gpu",
          "Water caustics inverse-magnification field (client-GPU twin of node 513)",
          "procedural", '''
void main() {
    // u_depth (0.2..3), u_scale (1..20), u_gain (0.5..8), u_waves (2..7)
    // map directly to the node's real params (typed-uniform live path).
    float depth = u_depth;
    float scl   = u_scale;
    float gain  = u_gain;
    int   nw    = int(u_waves + 0.5);
    float t     = u_time * 0.6;
    vec2  uv    = (v_uv - 0.5) * 2.0;   // [-1, 1] floor coords

    // Analytic Hessian of the sum-of-sines wave height H(u,v,t).
    float Hxx = 0.0, Hyy = 0.0, Hxy = 0.0;
    for (int i = 0; i < 7; i++) {
        if (i >= nw) break;
        float fi   = float(i);
        float ang  = fi * 1.7 + 0.5;                 // fixed directions
        float freq = (0.7 + 0.22 * fi) * scl;
        float amp  = 0.8 / (1.0 + 0.3 * fi);
        float sp   = 0.6 + 0.2 * fi;
        vec2  k    = freq * vec2(cos(ang), sin(ang));
        float ph   = dot(k, uv) + sp * t + fi * 1.3;
        float s    = sin(ph);
        Hxx += -amp * k.x * k.x * s;
        Hyy += -amp * k.y * k.y * s;
        Hxy += -amp * k.x * k.y * s;
    }

    // Jacobian of floor-point -> refracted-landing map ~ I + depth * Hessian.
    float Jxx  = 1.0 + depth * Hxx;
    float Jyy  = 1.0 + depth * Hyy;
    float Jxy  = depth * Hxy;
    float detJ = Jxx * Jyy - Jxy * Jxy;
    float mag  = abs(detJ);

    // Caustic intensity: inverse area magnification, baseline-subtracted then
    // soft-saturated + perceptual falloff (matches the CPU node's shaping).
    float caustic = max(1.0 / max(mag, 1e-3) - 1.0, 0.0);
    caustic = caustic / (caustic + 1.0);
    caustic = pow(caustic, 0.7);
    caustic = clamp(caustic * gain * 0.5, 0.0, 1.0);

    // Deep-water floor + aqua-tinted filaments (default colormode).
    vec3 floorc = vec3(0.02, 0.10, 0.16);
    vec3 tint   = vec3(0.55, 0.95, 1.0);
    vec3 col    = floorc + caustic * tint * 1.4;
    f_color = vec4(clamp(col, 0.0, 1.0), 1.0);
}
''',
    uniforms={
    "depth": {"glsl": "float", "min": 0.2, "max": 3.0, "default": 1.0, "description": "water-column depth (refraction focusing)"},
    "scale": {"glsl": "float", "min": 1.0, "max": 20.0, "default": 6.0, "description": "surface wave spatial frequency"},
    "gain":  {"glsl": "float", "min": 0.5, "max": 8.0, "default": 3.0, "description": "caustic brightness gain"},
    "waves": {"glsl": "float", "min": 2.0, "max": 7.0, "default": 4.0, "description": "number of superimposed directional waves"}
}
    )

# ── P0.6 field-eval client-GPU twins (continued) ──────────────────────────
# Nodes 53 / 43 / 57 are closed-form functions of (uv, t) — exact parity
# previews (no seeded-layout divergence), same family as 125/164/172. The
# CPU numpy nodes stay the authoritative export (two-tier precision). Placed
# here (before the _INFERNO block) so they only use the prologue helpers
# (hash21/fbm/rot) plus a self-contained inline colormap. pitfall #15: encode
# 0.5 as NEUTRAL so the default u_params yields each node's canonical view.

def _inferno_local(t):
    # Compact inferno polynomial (duplicated locally; each twin is a separate
    # shader program so no symbol collision). Kept as a Python string so it can
    # be inlined into the twin bodies below.
    return '''vec3 inferno(float t){
    t = clamp(t, 0.0, 1.0);
    const vec3 c0=vec3(0.00021894,0.00016488,-0.01907227);
    const vec3 c1=vec3(0.10651034,0.56396050,3.93279110);
    const vec3 c2=vec3(11.6028830,-3.9781129,-15.9420510);
    const vec3 c3=vec3(-41.703996,17.4360890,44.3541450);
    const vec3 c4=vec3(77.1629350,-33.402243,-81.8094230);
    const vec3 c5=vec3(-71.319421,32.6260640,73.2095190);
    const vec3 c6=vec3(25.1311300,-12.242810,-23.0709590);
    return c0+t*(c1+t*(c2+t*(c3+t*(c4+t*(c5+t*c6)))));
}'''

_register("metaballs_gpu",
          "Metaballs isosurface field (client-GPU twin of node 53)",
          "procedural", _inferno_local('') + '''
void main() {
    // u_params.x = isovalue (0.5 -> ~0.425), u_params.y = ball_speed (0.5 -> ~2.55).
    float iso   = 0.05 + u_isovalue * 0.75;
    float speed = 0.1  + u_ball_speed * 4.9;
    float t = u_time * 0.05 * speed;

    // Closed-form soft metaball field from N orbiting balls (pure f(uv, t)).
    vec2 p = v_uv;
    float field = 0.0;
    const int N = 14;
    for (int i = 0; i < N; i++) {
        float fi = float(i);
        float ang = fi * 2.399963;                 // golden-angle spread
        float orbit = 0.18 + 0.16 * hash21(vec2(fi, 1.7));
        float wx = 0.5 + orbit * cos(t * (0.6 + 0.05 * fi) + ang);
        float wy = 0.5 + orbit * sin(t * (0.6 + 0.05 * fi) + ang * 1.3);
        vec2 c = vec2(wx, wy);
        float ri = 0.06 + 0.05 * hash21(vec2(fi, 9.1));
        float d2 = dot(p - c, p - c);
        field += (ri * ri) / (ri * ri + d2 + 1e-4);
    }
    float f = clamp(field * 0.5, 0.0, 1.0);
    vec3 col = inferno(f);
    // Bright isosurface edge near the threshold.
    float edge = smoothstep(iso - 0.04, iso, field * (iso + 0.2))
               * (1.0 - smoothstep(iso, iso + 0.04, field * (iso + 0.2)));
    col += edge * 0.35;
    f_color = vec4(clamp(col, 0.0, 1.0), 1.0);
}
''',
    uniforms={
    "isovalue": {"glsl": "float", "min": 0.0, "max": 1.0, "default": 0.5, "description": "isosurface value"},
    "ball_speed": {"glsl": "float", "min": 0.0, "max": 1.0, "default": 0.5, "description": "ball orbit speed"}
    }
    )

# ── Node 321: Smooth-min Metaballs (client-GPU twin, closed-form f(uv,t)) ──
# Research technique: Quilez exponential/quadratic smooth-minimum (smin) merging
# of signed-distance spheres → organically blended "metaball" surfaces. The
# existing node-53 metaballs uses a sum-of-inverse-square field (no true SDF /
# no smin); this twin implements the canonical smin union so the `blend` (k)
# param genuinely controls edge softness the way IQ describes it. Closed-form
# f(uv,t) → exact parity preview; CPU numpy stays authoritative for export.
_register("smin_metaballs_gpu",
          "Smooth-min Metaballs — IQ exponential smin union of orbiting SDF spheres (node 321)",
          "procedural", _inferno_local('') + '''
// Quilez smooth minimum (exponential variant, k = blend).
// Reference: https://iquilezles.org/articles/smin/
float smin_exp(float a, float b, float k) {
    k = max(k, 1e-4);
    float res = exp2(-a / k) + exp2(-b / k);
    return -k * log2(max(res, 1e-4));
}
// Signed distance to a moving ball indexed i.
float ball(vec2 p, float fi, float t, float r) {
    float a = fi * 2.39996323 + t * (0.6 + 0.05 * fi);   // golden-angle spread
    float orbit = 0.18 + 0.16 * hash21(vec2(fi, 1.7));
    vec2 c = vec2(0.5 + orbit * cos(a), 0.5 + orbit * sin(a * 1.3));
    return length(p - c) - r;
}
void main() {
    // 0.5-neutral encoding → node defaults (pitfall #15).
    int   nBalls = int(clamp(3.0 + u_count * 9.0, 3.0, 12.0)); // 3..12
    float k      = mix(0.004, 0.10, u_blend);                 // smin blend
    float thr    = mix(0.55, 0.02, u_threshold);             // edge threshold
    float hue    = u_hue;                                    // base hue 0..1
    float speed  = mix(0.15, 2.6, u_ball_speed);
    float t = u_time * 0.05 * speed;

    vec2 p = v_uv;
    float r = 0.055 + 0.045 * hash21(vec2(7.0, 3.0));
    float d = 1e5;
    for (int i = 0; i < 12; i++) {
        if (float(i) >= float(nBalls)) break;
        float di = ball(p, float(i), t, r);
        d = smin_exp(d, di, k);            // smooth-union the SDFs
    }
    // Normalize the signed field into a 0..1 surface band + interior fill.
    float surf = 1.0 - smoothstep(0.0, thr, abs(d));
    float fill = 1.0 - smoothstep(0.0, thr * 0.6, d);
    float edge = smoothstep(thr, 0.0, abs(d));       // glow at the membrane
    float val = clamp(fill * 0.85 + surf * 0.4 + edge * 0.6, 0.0, 1.0);

    float ang = atan(p.y - 0.5, p.x - 0.5);
    vec3 col = inferno(val);
    // Tint with the hue control so the blob membrane shifts color by angle.
    vec3 tint = 0.5 + 0.5 * cos(6.2831853 * (hue + ang / 6.2831853) + vec3(0.0, 2.0, 4.0));
    col = mix(col, col * (0.6 + 0.6 * tint), 0.45 * edge);
    f_color = vec4(clamp(col, 0.0, 1.0), 1.0);
}
''',
    uniforms={
    "blend": {"glsl": "float", "min": 0.0, "max": 1.0, "default": 0.5, "description": "smin blend (k) — edge softness"},
    "count": {"glsl": "float", "min": 0.0, "max": 1.0, "default": 0.5, "description": "number of metaballs (3..12)"},
    "threshold": {"glsl": "float", "min": 0.0, "max": 1.0, "default": 0.5, "description": "edge threshold (lower = fatter blobs)"},
    "ball_speed": {"glsl": "float", "min": 0.0, "max": 1.0, "default": 0.5, "description": "orbit speed"},
    "hue": {"glsl": "float", "min": 0.0, "max": 1.0, "default": 0.5, "description": "membrane tint hue"}
    }
    )

# ── Node 322: Procedural Phasor Noise (closed-form f(uv,t)) ──
# Research technique: "Procedural Phasor Noise", Tricard, Efremov, Zanni, Neyret,
# Martínez, Lefebvre — ACM TOG / SIGGRAPH 2019.
# Reference: http://thibaulttricard.fr/project_page/phasor_noise/phasor.html
# Phasor noise reformulates Gabor noise as a complex PHASOR field: a sum of
# complex-valued Gabor kernels g_i(x) = A_i·exp(-π b²|x-x_i|²)·exp(i·2π f·(x-x_i)).
# Summing kernels accumulates real+imag parts; the ARGUMENT (phase) of the sum is
# the phasor field. Taking sin(phase) yields oscillating ridge patterns whose
# CONTRAST is decoupled from local intensity (unlike raw Gabor noise), giving the
# characteristic fingerprint/wood-grain ridges with locally controllable
# frequency and orientation. Closed-form per pixel → exact GPU live preview;
# CPU numpy stays authoritative for export.

_register("heatmap_gpu",
          "Density heatmap (client-GPU twin of node 43)",
          "procedural", _inferno_local('') + '''
void main() {
    // u_params.x = sigma proxy (0.5 -> ~0.06), u_params.z = colormap_shift.
    float sigma = 0.01 + u_sigma * 0.10;
    float shift = u_colormap_shift;
    float t = u_time * 0.04;

    // Closed-form kernel-density estimate from K drifting gaussian clusters.
    vec2 p = v_uv;
    float dens = 0.0;
    const int K = 18;
    for (int i = 0; i < K; i++) {
        float fi = float(i);
        vec2 c = vec2(0.15 + 0.7 * hash21(vec2(fi, 3.3)),
                      0.15 + 0.7 * hash21(vec2(fi, 7.7)));
        c += 0.04 * vec2(sin(t + fi), cos(t * 1.1 + fi * 1.7));
        float d2 = dot(p - c, p - c);
        dens += exp(-d2 / (2.0 * sigma * sigma + 1e-4));
    }
    dens = clamp(dens * 0.18, 0.0, 1.0);
    dens = fract(dens + shift);
    f_color = vec4(inferno(dens), 1.0);
}
''',
    uniforms={
    "sigma": {"glsl": "float", "min": 0.0, "max": 1.0, "default": 0.5, "description": "gaussian sigma"},
    "colormap_shift": {"glsl": "float", "min": 0.0, "max": 1.0, "default": 0.5, "description": "colormap shift"}
}
    )

_register("slitscan_gpu",
          "Slit-scan displacement (client-GPU twin of node 57)",
          "procedural", '''
vec3 hsv2rgb(vec3 c) {
    vec4 K = vec4(1.0, 2.0/3.0, 1.0/3.0, 3.0);
    vec3 p = abs(fract(c.xxx + K.xyz) * 6.0 - K.www);
    return c.z * mix(K.xxx, clamp(p - K.xxx, 0.0, 1.0), c.y);
}
void main() {
    // u_params.x = amplitude (0.5 -> ~0.275), u_params.y = frequency
    // (0.5 -> ~0.25), u_params.z = slit_type (0=vertical,1=horizontal,
    // 2=radial,3=spiral,4=angular,5=diagonal).
    int mode = int(floor(u_slit_type * 6.999));
    float amp  = 0.05 + u_amplitude * 0.45;
    float freq = 0.005 + u_frequency * 0.495;
    float t = u_time * 0.05;

    vec2 uv = v_uv;
    vec2 c = uv - 0.5;
    float r = length(c);
    float a = atan(c.y, c.x);

    float disp;
    if (mode == 1)      disp = sin(freq * uv.y * 40.0 + t);
    else if (mode == 2) disp = sin(freq * r * 40.0 - t);
    else if (mode == 3) disp = sin(freq * r * 40.0 + a * 4.0 + t);
    else if (mode == 4) disp = sin(freq * a * 6.0 + t);
    else if (mode == 5) disp = sin(freq * (uv.x + uv.y) * 40.0 + t);
    else                disp = sin(freq * uv.x * 40.0 + t);

    vec2 suv = fract(uv + amp * disp);
    float n = fbm(suv * 5.0);
    float hue = fract(n + t * 0.1);
    vec3 col = mix(vec3(n), hsv2rgb(vec3(hue, 0.7, 0.9)), 0.5);
    f_color = vec4(col, 1.0);
}
''',
    uniforms={
    "amplitude": {"glsl": "float", "min": 0.0, "max": 1.0, "default": 0.5, "description": "displacement amplitude"},
    "frequency": {"glsl": "float", "min": 0.0, "max": 1.0, "default": 0.5, "description": "slit frequency"},
    "slit_type": {"glsl": "float", "min": 0.0, "max": 5.0, "default": 0.0, "description": "slit pattern (0-5)"}
}
    )

# ── CMYK Halftone (node 399) client-GPU twin ──
# Classic print color-separation halftone. The source image is split into
# C/M/Y/K channels; each channel is screened onto its own rotated dot grid and
# the ink layers are recombined subtractively on paper. Per-channel screening is
# a closed-form function of (uv, input, params) -> exact parity preview, no
# seeded-layout divergence (same family as 53/43/57/172). The CPU numpy node
# stays authoritative for export (two-tier precision).
# Real numeric params bound via CLIENT_GPU_SHIMS param_map (spacing/max_dot/
# angle_offset); ink_set/paper/source/anim_mode are choice strings (pitfall
# #14) left unmapped — the preview renders the node's default cmyk ink set on
# white paper. pitfall #15: 0.5 -> node default so a neutral u_params yields the
# canonical view.
_register("cmyk_halftone_gpu",
          "CMYK halftone — classic print color-separation screening (client-GPU twin of node 399)",
          "filter", '''
float halftone_dot(vec2 uv, vec2 res, float value, float angleDeg,
                  float spacing, float maxDot) {
    float ang = radians(angleDeg);
    vec2 p = uv * res;                       // pixel coords
    vec2 g = rot(ang) * p;                   // rotate into screen space
    vec2 cell = mod(g, spacing) - spacing * 0.5;
    float d = length(cell);
    float r = sqrt(clamp(value, 0.0, 1.0)) * maxDot * spacing * 0.5;
    // 1.0 inside the dot, soft 1px edge for AA.
    return 1.0 - smoothstep(r, r + 1.0, d);
}
void main() {
    // Decode real node params via the typed uniforms (u_<name>) — these are
    // driven by the node's spacing/max_dot/angle_offset params on the client
    // AND passed as named_params on the server.
    //   spacing [2,40] ; max_dot [0.3,1.4] ; angle_offset [-45,45] deg
    float spacing  = mix(2.0, 40.0, clamp(u_spacing, 0.0, 1.0));
    float maxDot   = mix(0.3, 1.4, clamp(u_max_dot, 0.0, 1.0));
    float angleOff = (u_angle_offset - 0.5) * 90.0;   // -45..45 deg

    vec3 src = texture(u_texture, v_uv).rgb;
    float luma = dot(src, vec3(0.299, 0.587, 0.114));

    // Standard CMYK screen angles (C 15, M 75, Y 0, K 45) + global offset.
    float c = halftone_dot(v_uv, u_resolution, src.r, 15.0 + angleOff, spacing, maxDot);
    float m = halftone_dot(v_uv, u_resolution, src.g, 75.0 + angleOff, spacing, maxDot);
    float y = halftone_dot(v_uv, u_resolution, src.b,  0.0 + angleOff, spacing, maxDot);
    float k = halftone_dot(v_uv, u_resolution, 1.0 - luma, 45.0 + angleOff, spacing, maxDot);

    // Subtractive recombination on white paper, K overprints the others.
    vec3 col = vec3((1.0 - c) * (1.0 - k),
                    (1.0 - m) * (1.0 - k),
                    (1.0 - y) * (1.0 - k));
    f_color = vec4(clamp(col, 0.0, 1.0), 1.0);
}
''',
    uniforms={
    "spacing": {"glsl": "float", "min": 0.0, "max": 1.0, "default": 0.5, "description": "screen frequency / dot grid spacing"},
    "max_dot": {"glsl": "float", "min": 0.0, "max": 1.0, "default": 0.5, "description": "max dot diameter as fraction of spacing"},
    "angle_offset": {"glsl": "float", "min": 0.0, "max": 1.0, "default": 0.5, "description": "rotate all screens (0.5 = 0 deg)"}
}
    )

# ── Water Caustics (node 312) client-GPU twin ──
# Closed-form Jacobian caustics, same family as 125/164/172/53/43/57: every
# frame is a pure function of (uv, t) -> exact parity preview, no seeded-layout
# divergence. The CPU numpy node stays authoritative for export (two-tier
# precision). Analytic plane-wave derivatives keep it exact and cheap.
# Real params bound via CLIENT_GPU_SHIMS param_map (scale/gain/sharpen/speed);
# default colormode (ocean) is reproduced inline. pitfall #15: 0.5 -> node
# default so the neutral u_params yields the canonical view.
_register("caustics_gpu",
          "Water caustics — Jacobian light-convergence (client-GPU twin of node 312)",
          "procedural", '''\nvec3 caustics_ocean(float c, float floor) {
    // matches CPU colormode='ocean': deep blue floor, bright cyan caustic lines
    vec3 col = vec3(c * 0.35 + 0.02 + floor,
                    c * 0.85 + 0.12 + floor,
                    c * 0.75 + 0.30 + floor);
    return clamp(col, 0.0, 1.0);
}
void main() {
    // Decode real node params (0.5-neutral -> node defaults):
    //   p1 scale [1,16] 0.5->6.0 ; p2 caustic_gain [0.1,3] 0.5->1.2
    //   p3 sharpen [0.5,4] 0.5->1.6 ; p4 anim_speed [0.1,3] 0.5->1.0
    float scale = 0.5 + u_scale * 11.0;
    float gain  = 0.1 + u_caustic_gain * 2.2;
    float sharp = 0.5 + u_sharpen * 2.2;
    float aspd  = 0.1 + u_anim_speed * 1.8;
    float pht   = u_time * aspd;            // flow-mode phase advance

    // Pixel-centered, scale-normalized cartesian coords.
    vec2 res = u_resolution;
    vec2 pc = (gl_FragCoord.xy - 0.5 * res) / max(res.x, res.y) * scale;

    const int NW = 6;
    float H = 0.0, Hx = 0.0, Hy = 0.0;
    float Hxx = 0.0, Hxy = 0.0, Hyy = 0.0;
    for (int i = 0; i < NW; i++) {
        float fi = float(i);
        float ang = hash21(vec2(fi, 1.7)) * 6.2831853;
        float dx = cos(ang), dy = sin(ang);
        float k  = (0.8 + hash21(vec2(fi, 9.1)) * 1.6) * scale * 0.25;
        float ph = hash21(vec2(fi, 4.3)) * 6.2831853 + pht;
        float u  = k * (pc.x * dx + pc.y * dy) + ph;
        float A  = 1.0 / float(NW);          // fixed normalizer (amplitude in play)
        float s  = sin(u), co = cos(u);
        H   += A * s;
        Hx  += A * k * dx * co;
        Hy  += A * k * dy * co;
        Hxx += -A * k * k * dx * dx * s;
        Hxy += -A * k * k * dx * dy * s;
        Hyy += -A * k * k * dy * dy * s;
    }

    // Jacobian of displacement map W = (x + g*Hx, y + g*Hy).
    float j11 = 1.0 + gain * Hxx;
    float j12 = gain * Hxy;
    float j21 = gain * Hxy;
    float j22 = 1.0 + gain * Hyy;
    float det = j11 * j22 - j12 * j21;

    // Light convergence ~ 1/|det|; diverging (det>1) darkens the floor.
    float inv = max(abs(det), 1e-3);
    float caustic = 1.0 / inv;
    caustic = clamp(caustic, 0.0, 4.0);
    caustic = (det > 1.0) ? caustic * 0.25 : caustic;
    float c = clamp(pow(caustic / 4.0, sharp), 0.0, 1.0);

    // Dappled floor noise.
    float floor_ = (0.5 + 0.5 * fbm(pc * 2.0 + vec2(pht * 0.1))) * 0.15;
    vec3 col = caustics_ocean(c, floor_);
    f_color = vec4(col, 1.0);
}
''',
    uniforms={
    "scale": {"glsl": "float", "min": 0.0, "max": 1.0, "default": 0.5, "description": "caustic scale"},
    "caustic_gain": {"glsl": "float", "min": 0.0, "max": 1.0, "default": 0.5, "description": "caustic gain"},
    "sharpen": {"glsl": "float", "min": 0.0, "max": 1.0, "default": 0.5, "description": "sharpen exponent"},
    "anim_speed": {"glsl": "float", "min": 0.0, "max": 1.0, "default": 0.5, "description": "flow speed"}
}
    )


# ── Domain Warping (node 311) client-GPU twin ──
# Inigo Quilez two-level domain warp: fbm noise displaced by a lower-frequency
# copy of itself. Every frame is a pure closed-form function of (uv, t) — same
# family as 125/164/172/53/43/57/312. The CPU numpy node stays authoritative for
# export (two-tier precision). Real params bound via CLIENT_GPU_SHIMS param_map
# (scale/warp_strength/contrast/octaves); colormode defaults to the node's
# 'inferno'. pitfall #15: 0.5 -> node default so neutral u_params yields the
# canonical marbled view. pitfall #19: amplitude is divided by a FIXED normalizer
# (sum of octave amps), so warp_strength stays live and is not silently cancelled.
_register("domain_warp_gpu",
          "Domain Warping — IQ two-level fractal noise warp (client-GPU twin of node 311)",
          "procedural", _inferno_local('') + '''
float dw_fbm(vec2 p, int oct) {
    float v = 0.0, a = 0.5, norm = 0.0;
    for (int i = 0; i < 8; i++) {
        if (i >= oct) break;
        v += a * noise(p);
        norm += a;
        p *= 2.0; a *= 0.5;
    }
    return v / max(norm, 1e-6) * 2.0 - 1.0;   // [-1,1]
}
void main() {
    // Decode real node params (0.5-neutral -> node defaults):
    //   p1 scale [1,12] 0.5->4.0 ; p2 warp_strength [0,8] 0.5->4.0
    //   p3 contrast [0.5,3] 0.5->1.25 ; p4 octaves [1,8] 0.5->4
    float scale = 1.0 + u_scale * 6.0;
    float warp  = u_warp_strength * 8.0;
    float contr = 0.5 + u_contrast * 1.5;
    int oct = int(clamp(1.0 + u_octaves * 7.0, 1.0, 8.0));

    vec2 p = (v_uv - 0.5) * scale;
    // Gentle time drift -> live preview evolves (canonical view at t=0).
    vec2 ph = vec2(u_time * 0.12);

    vec2 q = vec2(dw_fbm(p, oct),
                  dw_fbm(p + vec2(5.2, 1.3), oct));
    vec2 r = vec2(dw_fbm(p + warp * q + vec2(1.7, 9.2) + ph, oct),
                  dw_fbm(p + warp * q + vec2(8.3, 2.8) + ph, oct));
    float val = dw_fbm(p + warp * r, oct);

    val = (val + 1.0) * 0.5;
    val = clamp(0.5 + (val - 0.5) * contr, 0.0, 1.0);

    vec3 col = inferno(val);
    f_color = vec4(clamp(col, 0.0, 1.0), 1.0);
}
''',
    uniforms={
    "scale": {"glsl": "float", "min": 0.0, "max": 1.0, "default": 0.5, "description": "noise scale"},
    "warp_strength": {"glsl": "float", "min": 0.0, "max": 1.0, "default": 0.5, "description": "warp strength"},
    "contrast": {"glsl": "float", "min": 0.0, "max": 1.0, "default": 0.5, "description": "contrast"},
    "octaves": {"glsl": "float", "min": 1.0, "max": 8.0, "default": 4.0, "description": "fbm octaves"}
}
    )


# ── Curl-Noise Flow Field (node 314) client-GPU twin ──
# Divergence-free flow via curl of an fbm potential P: v = (dP/dy, -dP/dx).
# Velocity ANGLE -> hue, MAGNITUDE -> brightness (node default 'spectral' colormap).
# Closed-form function of (uv, t); CPU numpy node stays authoritative for export.
# Real params bound via CLIENT_GPU_SHIMS param_map (scale/octaves/brightness/
# anim_mode). pitfall #15: 0.5 -> node default. A subtle u_time pan keeps the
# live preview in motion (canonical view at t=0).
_register("curl_noise_gpu",
          "Curl-Noise flow field — divergence-free direction field (client-GPU twin of node 314)",
          "procedural", '''vec3 hsv2rgb(vec3 c) {
    vec4 K = vec4(1.0, 2.0/3.0, 1.0/3.0, 3.0);
    vec3 p = abs(fract(c.xxx + K.xyz) * 6.0 - K.www);
    return c.z * mix(K.xxx, clamp(p - K.xxx, 0.0, 1.0), c.y);
}
float cn_fbm(vec2 p, int oct) {
    float v = 0.0, a = 0.5, norm = 0.0;
    for (int i = 0; i < 6; i++) {
        if (i >= oct) break;
        // golden-angle rotate each octave so layers don't align on axes
        float ang = 2.3999632 * float(i + 1);
        vec2 rp = rot(ang) * p;
        v += a * noise(rp);
        norm += a;
        p *= 2.0; a *= 0.5;
    }
    return v / max(norm, 1e-6);
}
void main() {
    // Decode real node params (0.5-neutral -> node defaults):
    //   p1 scale [1,12] 0.5->6.5 ; p2 octaves [1,6] 0.5->3
    //   p3 brightness [0.2,2] 0.5->1.1 ; p4 anim_mode [0,1] 0.5->drift
    float scale = 1.0 + u_scale * 11.0;
    int oct = int(clamp(1.0 + u_octaves * 5.0, 1.0, 6.0));
    float bright = 0.2 + u_brightness * 1.8;
    float drift = step(0.5, u_anim_mode);   // 0=static, 1=drift

    vec2 p = (v_uv - 0.5) * scale;
    vec2 pan = vec2(u_time * 0.6, u_time * 0.25) * drift;

    float e = 0.35;   // finite-difference step in noise space
    float P0 = cn_fbm(p + pan, oct);
    float Px = cn_fbm(p + pan + vec2(e, 0.0), oct);
    float Py = cn_fbm(p + pan + vec2(0.0, e), oct);
    float vx = (Py - P0) / e;        // dP/dy
    float vy = -(Px - P0) / e;       // -dP/dx
    float mag = length(vec2(vx, vy));
    float ang = atan(vy, vx);

    float hue = (ang + 3.14159265) / 6.2831853;
    float sat = clamp(0.5 + mag * 1.5, 0.0, 1.0);
    float val = clamp((0.25 + mag * 2.0) * bright, 0.0, 1.0);
    vec3 col = hsv2rgb(vec3(hue, sat, val));
    f_color = vec4(clamp(col, 0.0, 1.0), 1.0);
}
''',
    uniforms={
    "scale": {"glsl": "float", "min": 0.0, "max": 1.0, "default": 0.5, "description": "noise scale"},
    "octaves": {"glsl": "float", "min": 1.0, "max": 6.0, "default": 3.0, "description": "fbm octaves"},
    "brightness": {"glsl": "float", "min": 0.0, "max": 1.0, "default": 0.5, "description": "brightness"},
    "anim_mode": {"glsl": "float", "min": 0.0, "max": 1.0, "default": 0.5, "description": "drift on/off"}
}
    )


_register("shader_oil_gpu", "GPU oil painting simulation", "filter", _filter_typed('''
    float radius = u_radius;
    vec3 sum = vec3(0.0); float total = 0.0;
    float scale = radius / 4.0;
    for (int x = -3; x <= 3; x++) {
        for (int y = -3; y <= 3; y++) {
            vec2 off = vec2(float(x), float(y)) * step * scale;
            float w = exp(-float(x*x + y*y) / (radius * radius));
            sum += texture(u_texture, uv + off).rgb * w;
            total += w;
        }
    }
    f_color = vec4(sum / total, 1.0);
'''), uniforms={
    "radius": {"glsl": "float", "min": 1.0, "max": 8.0, "default": 4.0, "description": "brush radius"},
})

# Anisotropic Kuwahara — coherence-enhancing painterly abstraction (node 68 twin).
# A rotated, elongated Gaussian kernel is oriented along the local structure-tensor
# direction; the window is split into four angular sectors and the lowest-variance
# sector wins (Kyprianidis et al. 2011). Decoded from normalized u_params:
# p1 = radius∈[0,1] (0.5→8), p2 = anisotropy∈[0,1] (0.5→4). CPU numpy node stays
# authoritative for exact export.
_register("anisotropic_kuwahara_gpu", "GPU anisotropic Kuwahara painterly abstraction", "filter", _filter_typed('''
    // Decode normalized params (client sends radius/anisotropy 0..1 per the
    // 0.5-neutral GPU contract). radius 0.5→8 (node default), aniso 0.5→4.
    int R = int(clamp(2.0 + u_radius * 12.0, 2.0, 15.0));
    float aniso = clamp(1.0 + u_anisotropy * 6.0, 1.0, 12.0);

    // Local structure via Sobel (step is reserved in filter twins — only read it).
    float gx = 0.0, gy = 0.0;
    for (int x = -1; x <= 1; x++) {
        for (int y = -1; y <= 1; y++) {
            float v = dot(texture(u_texture, uv + vec2(float(x), float(y)) * step).rgb,
                          vec3(0.299, 0.587, 0.114));
            gx += float(x) * v;
            gy += float(y) * v;
        }
    }
    float theta = 0.5 * atan(gy, gx + 1e-5);
    float c = cos(-theta), s = sin(-theta);
    float sx = float(R) * 0.6 * sqrt(aniso);
    float sy = float(R) * 0.6 / sqrt(aniso);

    vec3 sc[4]; vec3 sc2[4]; float sw[4];
    for (int q = 0; q < 4; q++) { sc[q] = vec3(0.0); sc2[q] = vec3(0.0); sw[q] = 0.0; }

    for (int i = -8; i <= 8; i++) {
        for (int j = -8; j <= 8; j++) {
            float xe = float(i) * c - float(j) * s;
            float ye = float(i) * s + float(j) * c;
            float w = exp(-(xe * xe / (2.0 * sx * sx) + ye * ye / (2.0 * sy * sy)));
            int quad = (xe > 0.0 ? 2 : 0) + (ye > 0.0 ? 1 : 0);
            vec3 col = texture(u_texture, uv + vec2(float(i), float(j)) * step).rgb;
            sc[quad] += col * w;
            sc2[quad] += col * col * w;
            sw[quad] += w;
        }
    }

    float bestVar = 1e9;
    vec3 bestC = vec3(0.0);
    for (int q = 0; q < 4; q++) {
        if (sw[q] > 1e-4) {
            vec3 m = sc[q] / sw[q];
            vec3 m2 = sc2[q] / sw[q];
            float var = dot(m2 - m * m, vec3(1.0));
            if (var < bestVar) { bestVar = var; bestC = m; }
        }
    }
    f_color = vec4(bestC, 1.0);
'''), uniforms={
        "radius": {"glsl": "float", "min": 0.0, "max": 1.0, "default": 0.5, "description": "Kuwahara radius"},
        "anisotropy": {"glsl": "float", "min": 0.0, "max": 1.0, "default": 0.5, "description": "anisotropy"},
    })

# ── P0.4 filter twin (node 422 Palette Posterize) ───────────────────────────
# Client GPU live-preview: ordered (Bayer) dithering against a per-channel band
# palette. p1 = levels (0.5->16), p2 = dither strength (0.5->~2.0). The browser
# preview shows the ordered-dither path; the authoritative CPU numpy node
# (median-cut + Floyd-Steinberg + CIELAB) is the export. Per pitfall #15b, no
# local named `step` (the _filter_shader wrapper injects it).
_register("dither_palette_gpu", "GPU palette posterize with ordered dither (client twin of 422)", "filter", _filter_typed('''
    // u_levels = band count per channel; u_dither_scale = ordered-dither strength.
    int levels = int(clamp(2.0 + u_levels * 28.0, 2.0, 32.0));
    float strength = clamp((u_dither_scale - 0.5) * 4.0, 0.0, 4.0);

    vec3 col = orig.rgb;
    // perceptual luma weighting (approximates CIELAB channel weighting)
    float luma = dot(col, vec3(0.2126, 0.7152, 0.0722));
    col = mix(col, vec3(luma), 0.15);

    // 4x4 Bayer matrix (values 0..15)
    const float bayer[16] = float[16](
        0.0,  8.0,  2.0, 10.0,
       12.0,  4.0, 14.0,  6.0,
        3.0, 11.0,  1.0,  9.0,
       15.0,  7.0, 13.0,  5.0);
    ivec2 ip = ivec2(floor(v_uv * u_resolution));
    int bi = (ip.y & 3) * 4 + (ip.x & 3);
    float thr = (bayer[bi] / 16.0 - 0.5) * strength / float(levels);

    vec3 q = floor(col * float(levels) + thr) / float(levels);
    f_color = vec4(clamp(q, 0.0, 1.0), 1.0);
'''), uniforms={
        "levels": {"glsl": "float", "min": 0.0, "max": 1.0, "default": 0.5, "description": "posterize levels"},
        "dither_scale": {"glsl": "float", "min": 0.0, "max": 1.0, "default": 0.5, "description": "ordered dither strength"},
    })

_register("shader_neon_gpu", "GPU neon glow on edges", "filter", _filter_typed('''
    float gx = 0.0, gy = 0.0;
    for (int x = -1; x <= 1; x++) {
        for (int y = -1; y <= 1; y++) {
            vec2 off = vec2(float(x), float(y)) * step;
            float v = dot(texture(u_texture, uv + off).rgb, vec3(0.299, 0.587, 0.114));
            gx += float(x) * v; gy += float(y) * v;
        }
    }
    float edge = sqrt(gx*gx + gy*gy);
    float glow = edge * u_intensity * 3.0;
    vec3 neon = vec3(glow * 0.8, glow * 0.3, glow);
    f_color = vec4(orig.rgb + neon, 1.0);
'''), uniforms={
    "intensity": {"glsl": "float", "min": 0.0, "max": 2.0, "default": 0.5, "description": "neon glow intensity"},
})

_register("shader_pencil_gpu", "GPU pencil sketch", "filter", _filter_typed('''
    float gx = 0.0, gy = 0.0;
    for (int x = -1; x <= 1; x++) {
        for (int y = -1; y <= 1; y++) {
            vec2 off = vec2(float(x), float(y)) * step;
            float v = dot(texture(u_texture, uv + off).rgb, vec3(0.299, 0.587, 0.114));
            gx += float(x) * v; gy += float(y) * v;
        }
    }
    float edge = sqrt(gx*gx + gy*gy);
    float sketch = 1.0 - edge * 4.0;
    f_color = vec4(mix(orig.rgb, vec3(sketch), u_strength), 1.0);
'''), uniforms={
    "strength": {"glsl": "float", "min": 0.0, "max": 1.0, "default": 0.5, "description": "pencil strength"},
})

_register("shader_motion_blur_gpu", "GPU directional motion blur", "filter", _filter_typed('''
    float angle = u_angle;
    float dist = u_dist;
    vec2 dir = vec2(cos(angle), sin(angle)) * step * dist;
    vec3 col = vec3(0.0);
    for (int i = -5; i <= 5; i++) {
        float t = float(i) / 5.0;
        col += texture(u_texture, uv + dir * t).rgb * (1.0 - abs(t));
    }
    f_color = vec4(col / 3.5, 1.0);
'''), uniforms={
    "angle": {"glsl": "float", "min": 0.0, "max": 6.2831853, "default": 3.14159265, "description": "blur direction (rad)"},
    "dist":  {"glsl": "float", "min": 0.0, "max": 30.0, "default": 20.0, "description": "blur length (px)"},
})

# ── Stochastic hex-tiling (Heitz & Neyret, HPG 2018) ──
# "High-Performance By-Example Noise using a Histogram-Preserving Blending
# Operator" — tiles a texture across the plane with NO visible repetition.
# Ref: Heitz & Neyret, ACM SIGGRAPH/HPG 2018, doi:10.1145/3233310
# Core idea: overlay a triangle grid (hex lattice); at each pixel the three
# nearest lattice vertices each provide a randomly-offset sample of the input;
# blend them with barycentric weights raised to a contrast exponent so the
# variance-preserving mix hides the tiling seams. The random per-vertex
# translation breaks the lattice periodicity → aperiodic tiling of one image.
_register("hex_tiling_gpu",
          "Stochastic hex-tiling — repetition-free texture tiling (Heitz-Neyret 2018)",
          "filter", _filter_typed('''
    float scale = u_scale;            // tiles across the image
    float contrast = u_contrast;      // blend sharpness (variance preserve)
    float randomness = u_randomness;  // per-tile random offset strength

    vec2 st = uv * scale;
    // skew into triangle-lattice space (hex lattice basis)
    mat2 UNSKEW = mat2(1.0, 0.0, -0.5773503, 1.1547005);
    vec2 sp = UNSKEW * st;
    vec2 baseId = floor(sp);
    vec2 f = fract(sp);
    // pick one of the two triangles of the rhombus by the diagonal
    vec2 id1, id2, id3;
    vec3 w;
    if (f.x + f.y < 1.0) {
        id1 = baseId;
        id2 = baseId + vec2(1.0, 0.0);
        id3 = baseId + vec2(0.0, 1.0);
        w = vec3(1.0 - f.x - f.y, f.x, f.y);
    } else {
        id1 = baseId + vec2(1.0, 1.0);
        id2 = baseId + vec2(1.0, 0.0);
        id3 = baseId + vec2(0.0, 1.0);
        w = vec3(f.x + f.y - 1.0, 1.0 - f.y, 1.0 - f.x);
    }
    // hash each vertex id -> random per-vertex uv translation (breaks periodicity)
    vec2 h1 = fract(sin(vec2(dot(id1, vec2(127.1, 311.7)), dot(id1, vec2(269.5, 183.3)))) * 43758.5453);
    vec2 h2 = fract(sin(vec2(dot(id2, vec2(127.1, 311.7)), dot(id2, vec2(269.5, 183.3)))) * 43758.5453);
    vec2 h3 = fract(sin(vec2(dot(id3, vec2(127.1, 311.7)), dot(id3, vec2(269.5, 183.3)))) * 43758.5453);
    vec4 c1 = texture(u_texture, uv + (h1 - 0.5) * randomness);
    vec4 c2 = texture(u_texture, uv + (h2 - 0.5) * randomness);
    vec4 c3 = texture(u_texture, uv + (h3 - 0.5) * randomness);
    // contrast-preserving (variance) blend: sharpen the bary weights so the
    // dominant tile wins near vertices -> hides seams (Heitz-Neyret operator).
    vec3 wp = pow(max(w, vec3(0.0)), vec3(1.0 + contrast * 6.0));
    wp /= max(dot(wp, vec3(1.0)), 1e-4);
    vec4 blended = c1 * wp.x + c2 * wp.y + c3 * wp.z;
    f_color = vec4(blended.rgb, 1.0);
'''), uniforms={
    "scale":      {"glsl": "float", "min": 1.0, "max": 12.0, "default": 4.0, "description": "tiles across image"},
    "contrast":   {"glsl": "float", "min": 0.0, "max": 1.0, "default": 0.5, "description": "seam-hiding blend sharpness"},
    "randomness": {"glsl": "float", "min": 0.0, "max": 1.0, "default": 0.4, "description": "per-tile random offset"},
})

# ── P0.4 client-GPU twin shaders for existing CPU filter nodes ──
# Each maps a pre-existing CPU filter node's LIVE preview onto a GLSL twin.
# The CPU numpy path stays the authoritative export (two-tier precision).

# 42 Fake HDR — contrast / saturation / vignette / bloom (GPU live twin)
_register("hdr_gpu", "GPU fake-HDR tonemap (contrast/sat/vignette/bloom)", "filter", _filter_typed('''
    float gray = dot(orig.rgb, vec3(0.299, 0.587, 0.114));
    // contrast around mid-gray
    vec3 c = (orig.rgb - 0.5) * (0.5 + u_contrast * 3.0) + 0.5;
    // saturation toward/away from luma
    c = mix(vec3(gray), c, 0.5 + u_saturation * 2.0);
    // bloom: cheap bright-area lift
    float bright = max(0.0, gray - 0.6) * u_bloom * 2.0;
    c += bright;
    // vignette
    vec2 d = uv - 0.5;
    float vig = 1.0 - dot(d, d) * u_vignette * 2.5;
    c *= clamp(vig, 0.0, 1.0);
    f_color = vec4(clamp(c, 0.0, 1.0), 1.0);
'''), uniforms={
        "contrast": {"glsl": "float", "min": 0.0, "max": 1.0, "default": 0.5, "description": "tone contrast"},
        "saturation": {"glsl": "float", "min": 0.0, "max": 1.0, "default": 0.5, "description": "saturation"},
        "vignette": {"glsl": "float", "min": 0.0, "max": 1.0, "default": 0.5, "description": "vignette strength"},
        "bloom": {"glsl": "float", "min": 0.0, "max": 1.0, "default": 0.5, "description": "bloom lift"},
    })

# 63 Cross Stitch — grid of stitches on a fabric backdrop (GPU live twin)
_register("cross_stitch_gpu", "GPU cross-stitch embroidery", "filter", _filter_typed('''
    float gstep = max(4.0, 32.0 - u_thread_step * 28.0);   // thread_step
    float lw = 1.0 + u_line_width * 6.0;                  // line_width
    vec2 cell = floor(uv * u_resolution / gstep);
    vec2 cell_uv = (cell + 0.5) * gstep / u_resolution;
    vec3 src = texture(u_texture, cell_uv).rgb;
    // fabric base
    vec3 fabric = vec3(0.95, 0.92, 0.88);
    vec2 q = fract(uv * u_resolution / gstep) - 0.5;
    // cross: two diagonal strokes
    float d1 = abs(q.x + q.y);
    float d2 = abs(q.x - q.y);
    float stroke = min(d1, d2);
    float stitch = 1.0 - smoothstep(lw * 0.35, lw * 0.45, stroke);
    vec3 col = mix(fabric, src, stitch);
    f_color = vec4(col, 1.0);
'''), uniforms={
        "thread_step": {"glsl": "float", "min": 0.0, "max": 1.0, "default": 0.5, "description": "stitch grid step"},
        "line_width": {"glsl": "float", "min": 0.0, "max": 1.0, "default": 0.5, "description": "thread width"},
    })

# 64 Edge Halftone — Sobel-magnitude-weighted dots (GPU live twin)
_register("edge_halftone_gpu", "GPU edge-weighted halftone dots", "filter", _filter_typed('''
    float tl = dot(texture(u_texture, uv + vec2(-step.x, -step.y)).rgb, vec3(0.299,0.587,0.114));
    float t  = dot(texture(u_texture, uv + vec2(0, -step.y)).rgb, vec3(0.299,0.587,0.114));
    float tr = dot(texture(u_texture, uv + vec2(step.x, -step.y)).rgb, vec3(0.299,0.587,0.114));
    float l  = dot(texture(u_texture, uv + vec2(-step.x, 0)).rgb, vec3(0.299,0.587,0.114));
    float r  = dot(texture(u_texture, uv + vec2(step.x, 0)).rgb, vec3(0.299,0.587,0.114));
    float bl = dot(texture(u_texture, uv + vec2(-step.x, step.y)).rgb, vec3(0.299,0.587,0.114));
    float b  = dot(texture(u_texture, uv + vec2(0, step.y)).rgb, vec3(0.299,0.587,0.114));
    float br = dot(texture(u_texture, uv + vec2(step.x, step.y)).rgb, vec3(0.299,0.587,0.114));
    float gx = -tl - 2.0*l - bl + tr + 2.0*r + br;
    float gy = -tl - 2.0*t - tr + bl + 2.0*b + br;
    float edge = clamp(sqrt(gx*gx + gy*gy), 0.0, 1.0);
    float cell = 4.0 + u_dot_spacing * 16.0;               // dot_spacing
    float base = (1.0 - edge) * 0.5 * (0.5 + u_dot_size * 0.5); // dot_size
    vec2 q = fract(uv * u_resolution / cell) - 0.5;
    float d = length(q);
    float dot_r = clamp(base, 0.02, 0.5);
    float v = d < dot_r ? 0.0 : 1.0;
    vec3 bg = vec3(0.05, 0.05, 0.08);
    f_color = vec4(mix(bg, vec3(1.0), v), 1.0);
'''), uniforms={
        "dot_spacing": {"glsl": "float", "min": 0.0, "max": 1.0, "default": 0.5, "description": "halftone cell spacing"},
        "dot_size": {"glsl": "float", "min": 0.0, "max": 1.0, "default": 0.5, "description": "halftone dot size"},
    })

# 339 Tonal Hatching — pen-and-ink crosshatch screening (GPU live twin of
# node 339, Winkenbach & Salesin 1994). Mirrors the CPU node's geometry:
# ink coverage f = 1 - L^contrast, then screen `layers` rotated stripe
# masks (each layer rotated 180/layers deg) turning on where f exceeds a
# per-layer threshold; darkest tones collapse to solid ink. The CPU fn
# stays authoritative for exact W&D paper/ink palettes + animation modes;
# the twin covers the dominant light-paper / black-ink look live. Numeric
# node params (spacing/line_width/layers/angle/contrast) are mapped by
# name (contract #5); string choices `paper`/`ink_tone` are left
# unmapped (pitfall #14) so the preview uses the canonical light/black.
# NOTE: do NOT redeclare `step` (the wrapper provides a vec2 `step`) — use
# plain comparisons for the stripe test.
_register("tonal_hatching_gpu", "GPU tonal hatching (crosshatch pen-and-ink screening)", "filter", _filter_typed('''
    float lum = dot(orig.rgb, vec3(0.299, 0.587, 0.114));
    lum = clamp(lum, 0.0, 1.0);
    float contrast = max(0.1, u_contrast);
    lum = pow(lum, contrast);
    float f = 1.0 - lum;                          // ink coverage fraction
    float spacing = max(3.0, u_spacing);          // px between strokes
    float lw = clamp(u_line_width, 0.5, spacing - 1.0);
    float nlayers = max(1.0, min(4.0, floor(u_layers + 0.5)));
    float baseA = radians(u_angle);
    vec2 px = uv * u_resolution;
    bool ink = false;
    for (int i = 0; i < 4; i++) {
        if (float(i) >= nlayers) break;
        float thr = (float(i) + 1.0) / (nlayers + 1.0);
        float ang = baseA + float(i) * (3.14159265 / nlayers);
        float ca = cos(ang), sa = sin(ang);
        float lyr = px.x * sa + px.y * ca;
        float phase = mod(lyr, spacing);
        if (f > thr && phase < lw) ink = true;
    }
    if (f > 0.9) ink = true;                      // darkest tones solid ink
    vec3 paper_col = vec3(0.97, 0.96, 0.92);
    vec3 ink_col = vec3(0.09, 0.09, 0.11);
    f_color = vec4(mix(paper_col, ink_col, ink ? 1.0 : 0.0), 1.0);
'''), uniforms={
    "spacing": {"glsl": "float", "min": 4.0, "max": 24.0, "default": 10.0, "description": "hatch line spacing (px)"},
    "line_width": {"glsl": "float", "min": 0.5, "max": 5.0, "default": 1.5, "description": "hatch line width (px)"},
    "layers": {"glsl": "float", "min": 1.0, "max": 4.0, "default": 3.0, "description": "crosshatch layer count"},
    "angle": {"glsl": "float", "min": 0.0, "max": 180.0, "default": 45.0, "description": "base hatch angle (deg)"},
    "contrast": {"glsl": "float", "min": 0.3, "max": 3.0, "default": 1.0, "description": "luminance gamma"},
})

# 493 Color Grading (OKLab) — perceptually-uniform brightness/contrast/gamma/
# saturation/hue/white-balance grade in the OKLab space (Ottosson 2020). Every
# op happens on the wired IMAGE in OKLab so saturation changes do not shift hue
# and contrast changes are perceptually even. Mirrors the CPU node 493 math
# (see methods/filters/color_grade.py) as a typed-uniform GPU filter twin. The
# CPU fn stays authoritative for the synthetic `source` generators + all string
# choice params; string params (source/palette/invert/anim_mode) are unmapped.
_register("color_grade_gpu", "GPU color grading in OKLab perceptual space", "filter", _filter_typed('''
    // sRGB -> linear (avoid GLSL step() — `step` is a reserved vec2 here)
    vec3 c = orig.rgb;
    vec3 lin = vec3(
        c.r <= 0.04045 ? c.r / 12.92 : pow((c.r + 0.055) / 1.055, 2.4),
        c.g <= 0.04045 ? c.g / 12.92 : pow((c.g + 0.055) / 1.055, 2.4),
        c.b <= 0.04045 ? c.b / 12.92 : pow((c.b + 0.055) / 1.055, 2.4));
    // linear RGB -> OKLab
    float l = 0.4122214708*lin.r + 0.5363325363*lin.g + 0.0514459929*lin.b;
    float m = 0.2119034982*lin.r + 0.6806995451*lin.g + 0.1073969566*lin.b;
    float s = 0.0883024619*lin.r + 0.2817188376*lin.g + 0.6299787005*lin.b;
    float l_ = sign(l)*pow(abs(l), 1.0/3.0);
    float m_ = sign(m)*pow(abs(m), 1.0/3.0);
    float s_ = sign(s)*pow(abs(s), 1.0/3.0);
    float L = 0.2104542553*l_ + 0.7936177850*m_ - 0.0040720468*s_;
    float A = 1.9779984951*l_ - 2.4285922050*m_ + 0.4505937099*s_;
    float B = 0.0259040371*l_ + 0.7827717662*m_ - 0.8086757660*s_;
    // Lightness grade: exposure, contrast (pivot 0.5), gamma
    L = L * pow(2.0, u_exposure);
    L = (L - 0.5) * u_contrast + 0.5;
    L = sign(L) * pow(abs(L), 1.0 / max(0.05, u_gamma));
    // Chroma grade: hue rotate + saturation
    float hr = radians(u_hue_rotate);
    float ca = cos(hr), sa = sin(hr);
    vec2 ab = vec2(A * ca - B * sa, A * sa + B * ca) * u_saturation;
    // White balance: temperature (a warm/cool ~ +b/-b via a & b) + tint
    ab.x += u_tint * 0.10;
    ab.y += u_temperature * 0.10;
    A = ab.x; B = ab.y;
    // Radial vignette on lightness
    vec2 vp = uv - 0.5;
    float vr = length(vp) * 1.41421356;
    L *= mix(1.0, 1.0 - vr * vr, clamp(u_vignette, 0.0, 1.0));
    // OKLab -> linear RGB
    float li = L + 0.3963377774*A + 0.2158037573*B;
    float mi = L - 0.1055613458*A - 0.0638541728*B;
    float si = L - 0.0894841775*A - 1.2914855480*B;
    li = li*li*li; mi = mi*mi*mi; si = si*si*si;
    vec3 rgb = vec3(
         4.0767416621*li - 3.3077115913*mi + 0.2309699292*si,
        -1.2684380046*li + 2.6097574011*mi - 0.3413193965*si,
        -0.0041960863*li - 0.7034186147*mi + 1.7076147010*si);
    rgb = clamp(rgb, 0.0, 1.0);
    // linear -> sRGB (avoid step())
    vec3 outc = vec3(
        rgb.r <= 0.0031308 ? rgb.r * 12.92 : 1.055 * pow(rgb.r, 1.0/2.4) - 0.055,
        rgb.g <= 0.0031308 ? rgb.g * 12.92 : 1.055 * pow(rgb.g, 1.0/2.4) - 0.055,
        rgb.b <= 0.0031308 ? rgb.b * 12.92 : 1.055 * pow(rgb.b, 1.0/2.4) - 0.055);
    f_color = vec4(clamp(outc, 0.0, 1.0), orig.a);
'''), uniforms={
    "exposure": {"glsl": "float", "min": -3.0, "max": 3.0, "default": 0.0, "description": "exposure stops (2^ev)"},
    "contrast": {"glsl": "float", "min": 0.2, "max": 3.0, "default": 1.0, "description": "lightness contrast (pivot 0.5)"},
    "gamma": {"glsl": "float", "min": 0.3, "max": 3.0, "default": 1.0, "description": "lightness gamma"},
    "saturation": {"glsl": "float", "min": 0.0, "max": 3.0, "default": 1.0, "description": "OKLab chroma scale"},
    "hue_rotate": {"glsl": "float", "min": -180.0, "max": 180.0, "default": 0.0, "description": "hue rotation (deg)"},
    "temperature": {"glsl": "float", "min": -1.0, "max": 1.0, "default": 0.0, "description": "warm/cool white balance"},
    "tint": {"glsl": "float", "min": -1.0, "max": 1.0, "default": 0.0, "description": "green/magenta tint"},
    "vignette": {"glsl": "float", "min": 0.0, "max": 1.0, "default": 0.0, "description": "radial darkening"},
})

# 350 FXAA Anti-Aliasing — Fast Approximate Anti-Aliasing (Lottes, NVIDIA
# 2009/2011). Real-time screen-space AA: 3x3 luma neighbourhood, edge early-out,
# edge-tangent direction blend (the learnopengl normalize variant, matched by
# the CPU node 350 export). u_params.x = edge_threshold with 0.5 = neutral
# medium strength (pitfall #15: 0.5 must not be a degenerate extreme).
_register("fxaa_gpu", "GPU FXAA anti-aliasing (edge-tangent luma blend)", "filter", _filter_shader('''
    vec3 rgbNW = texture(u_texture, uv + vec2(-step.x,  step.y)).rgb;
    vec3 rgbNE = texture(u_texture, uv + vec2( step.x,  step.y)).rgb;
    vec3 rgbSW = texture(u_texture, uv + vec2(-step.x, -step.y)).rgb;
    vec3 rgbSE = texture(u_texture, uv + vec2( step.x, -step.y)).rgb;
    vec3 rgbM  = orig.rgb;
    float lNW = dot(rgbNW, vec3(0.299, 0.587, 0.114));
    float lNE = dot(rgbNE, vec3(0.299, 0.587, 0.114));
    float lSW = dot(rgbSW, vec3(0.299, 0.587, 0.114));
    float lSE = dot(rgbSE, vec3(0.299, 0.587, 0.114));
    float lM  = dot(rgbM,  vec3(0.299, 0.587, 0.114));

    float lumaMin = min(lM, min(min(lNW, lNE), min(lSW, lSE)));
    float lumaMax = max(lM, max(max(lNW, lNE), max(lSW, lSE)));

    float et = 0.04 + u_params.x * 0.46;            // edge_threshold: 0.04..0.50
    if ((lumaMax - lumaMin) < max(0.001, lumaMax * et)) {
        f_color = orig;                            // flat region — pass through
        return;
    }

    float dirX = -lNW - lNE + lSW + lSE;
    float dirY = -lNW - lSW + lNE + lSE;
    float dirL = max(1e-6, sqrt(dirX * dirX + dirY * dirY));
    vec2 dir = vec2(dirX, dirY) / dirL;

    vec3 rgbA = (texture(u_texture, uv + dir * (1.0/6.0) * step).rgb +
                 texture(u_texture, uv - dir * (1.0/6.0) * step).rgb) * 0.5;
    vec3 rgbB = (rgbA * 0.5) + (texture(u_texture, uv + dir * (1.0/2.0) * step).rgb +
                                texture(u_texture, uv - dir * (1.0/2.0) * step).rgb) * 0.25;
    float lB = dot(rgbB, vec3(0.299, 0.587, 0.114));
    vec3 rgb = ((lB < lumaMin) || (lB > lumaMax)) ? rgbA : rgbB;   // anti-ringing clamp
    f_color = vec4(rgb, 1.0);
'''))

# 74 Swirl Displacement — polar swirl remap (GPU live twin)

# 13 Dithering — Bayer 8x8 ordered dither with N-level quantization (GPU live
# twin). The CPU node's default `fs` (Floyd-Steinberg error diffusion) is an
# inherently serial scan that cannot be reproduced per-pixel on the GPU, so this
# twin renders the ORDERED (Bayer) approximation and the CPU fn stays
# authoritative for all error-diffusion algorithms. `levels` -> p1 (2..8),
# `contrast` -> p2. `algorithm`/`palette`/`noise_type` are string choices
# (pitfall #14) and are left unmapped.
_register("dither13_gpu", "GPU Bayer-4 ordered dithering (node 13 twin)", "filter", _filter_typed('''
    float gray = dot(orig.rgb, vec3(0.299, 0.587, 0.114));
    float contrast = 0.5 + u_contrast * 2.5;            // contrast
    gray = clamp((gray - 0.5) * contrast + 0.5, 0.0, 1.0);
    // Bayer 4x4 ordered threshold matrix (values 0..15) via a small lookup.
    float bx = mod(gl_FragCoord.x, 4.0);
    float by = mod(gl_FragCoord.y, 4.0);
    float m[16];
    m[0]=0.0;  m[1]=8.0;  m[2]=2.0;  m[3]=10.0;
    m[4]=12.0; m[5]=4.0;  m[6]=14.0; m[7]=6.0;
    m[8]=3.0;  m[9]=11.0; m[10]=1.0; m[11]=9.0;
    m[12]=15.0;m[13]=7.0; m[14]=13.0;m[15]=5.0;
    int idx = int(by) * 4 + int(bx);
    float threshold = (m[idx] + 0.5) / 16.0;            // in (0,1)
    float levels = floor(2.0 + u_levels * 6.0 + 0.5); // levels
    float steps = max(1.0, levels - 1.0);
    float scaled = gray * steps;
    float lower = floor(scaled);
    float frac = scaled - lower;
    float q = (lower + (frac > threshold ? 1.0 : 0.0)) / steps;
    f_color = vec4(vec3(clamp(q, 0.0, 1.0)), 1.0);
'''), uniforms={
        "levels": {"glsl": "float", "min": 0.0, "max": 1.0, "default": 0.5, "description": "quantization levels"},
        "contrast": {"glsl": "float", "min": 0.0, "max": 1.0, "default": 0.5, "description": "contrast"},
    })


# ── GPU-First gap mirrors: closed-form f(uv,t) twins for CPU nodes 523/954/512 ──
# These three CPU nodes are per-pixel closed-form fields with NO close existing
# twin (Aurora Borealis, Autostereogram, SIREN Field). Each maps to a client-GPU
# shim in image_pipeline/methods/gpu_shaders.py (typed-uniform contract: every
# numeric CPU param becomes a named u_<name> uniform, bound to a real SCALAR
# port). CPU fns stay authoritative for export; parity is approximate by design.
# No late helpers (inferno/hsv2rgb) are used — a local cosine palette is inlined
# to avoid the late-helper ordering pitfall.

_register("aurora_gpu", "Aurora Borealis (client-GPU twin of node 523)", "procedural",
'''void main() {
    vec2 uv = (gl_FragCoord.xy - 0.5 * u_resolution) / min(u_resolution.x, u_resolution.y);
    float t = u_time;
    vec3 col = vec3(0.01, 0.02, 0.06) * (1.0 - 0.4 * uv.y);
    // starfield
    vec2 sg = floor(uv * 220.0);
    float h = hash21(sg);
    if (h > 1.0 - clamp(u_star_density, 0.0, 1.0) * 0.5) {
        float d = length(fract(uv * 220.0) - 0.5);
        col += vec3(smoothstep(0.12, 0.0, d)) * (0.4 + 0.6 * hash21(sg + 2.1));
    }
    int N = int(clamp(u_curtain_count, 1.0, 8.0));
    for (int i = 0; i < 8; i++) {
        if (i >= N) break;
        float fi = float(i);
        float baseY = -0.45 + fi * 0.22;
        float wob = fbm(vec2(uv.x * u_turbulence + t * u_drift_speed * 0.25 + fi * 3.1, t * 0.12));
        float yc = baseY + (wob - 0.5) * 0.7 + 0.12 * sin(t * 0.3 + fi);
        float dist = abs(uv.y - yc);
        float width = 0.16 * (0.6 + 0.6 * wob);
        float beam = exp(-dist * dist / (width * width));
        float streak = 0.6 + 0.4 * sin(uv.x * 38.0 + t * u_drift_speed * 2.0 + fi * 5.0);
        beam *= streak;
        vec3 ac = mix(vec3(0.1, 1.0, 0.45), vec3(1.0, 0.25, 0.35),
                      clamp(u_red_fringe, 0.0, 1.0) * abs(uv.x));
        ac = mix(ac, vec3(0.35, 0.75, 1.0), 0.25 * sin(fi * 1.7));
        col += ac * beam * u_intensity * (0.35 + 0.65 * fbm(vec2(uv.x * 3.0 - t * 0.2, fi)));
    }
    // vertical extent mask driven by beam_height
    col *= smoothstep(u_beam_height + 0.4, u_beam_height - 0.4, max(uv.y, -1.0));
    col += vec3(0.05, 0.0, 0.08) * u_color_shift;
    f_color = vec4(clamp(col, 0.0, 1.0), 1.0);
}
''',
uniforms={
    "curtain_count": {"glsl": "float", "min": 1.0, "max": 8.0, "default": 4.0, "description": "number of aurora curtains"},
    "drift_speed": {"glsl": "float", "min": 0.0, "max": 3.0, "default": 1.0, "description": "horizontal drift speed"},
    "intensity": {"glsl": "float", "min": 0.2, "max": 2.5, "default": 1.0, "description": "overall brightness"},
    "beam_height": {"glsl": "float", "min": 0.2, "max": 0.9, "default": 0.6, "description": "vertical extent of curtains"},
    "color_shift": {"glsl": "float", "min": -1.0, "max": 1.0, "default": 0.0, "description": "palette color offset"},
    "turbulence": {"glsl": "float", "min": 0.5, "max": 6.0, "default": 2.5, "description": "fbm turbulence frequency"},
    "star_density": {"glsl": "float", "min": 0.0, "max": 1.0, "default": 0.35, "description": "starfield density"},
    "red_fringe": {"glsl": "float", "min": 0.0, "max": 1.0, "default": 0.5, "description": "red fringe amount"},
})

_register("siren_gpu", "SIREN Field (client-GPU twin of node 512)", "procedural",
'''void main() {
    vec2 uv = (gl_FragCoord.xy - 0.5 * u_resolution) / min(u_resolution.x, u_resolution.y);
    float t = u_time;
    vec2 p = uv * u_coord_scale;
    float v = 0.0;
    for (int k = 0; k < 6; k++) {
        float fk = float(k);
        vec2 w = vec2(cos(fk * 1.7), sin(fk * 2.3));
        float b = fk * 0.9;
        float om = mix(u_omega0, u_omega, fract(fk * 0.37));
        float s = sin(om * dot(w, p + 0.08 * t) + b + t * 0.3);
        v += s * (0.5 + 0.5 * sin(fk * 1.1));
    }
    v *= u_weight_scale / 6.0;
    v = v * 0.5 + 0.5;
    // cosine palette (Inigo Quilez) — no dependency on late helpers
    vec3 cmap = 0.5 + 0.5 * cos(6.2831853 * (v + vec3(0.0, 0.33, 0.67)));
    f_color = vec4(clamp(cmap, 0.0, 1.0), 1.0);
}
''',
uniforms={
    "omega0": {"glsl": "float", "min": 1.0, "max": 60.0, "default": 30.0, "description": "base frequency"},
    "omega": {"glsl": "float", "min": 1.0, "max": 60.0, "default": 30.0, "description": "top frequency"},
    "weight_scale": {"glsl": "float", "min": 0.1, "max": 3.0, "default": 1.0, "description": "output gain"},
    "coord_scale": {"glsl": "float", "min": 0.5, "max": 12.0, "default": 3.0, "description": "coordinate scale"},
})

_register("autostereogram_gpu", "Autostereogram (client-GPU twin of node 954)", "procedural",
'''void main() {
    vec2 res = u_resolution;
    float px = gl_FragCoord.x;
    float py = gl_FragCoord.y;
    vec2 uv = (vec2(px, py) - 0.5 * res) / min(res.x, res.y);
    float t = u_time;
    float r2 = dot(uv, uv);
    float depth = clamp(1.0 - r2 * 2.5 + 0.06 * sin(t + py * 0.01), 0.0, 1.0);
    depth *= u_depth_scale;
    float shift = depth * u_separation;
    float sx = px - shift;
    vec2 gp = vec2(sx, py) / u_tile_size;
    vec2 cell = fract(gp) - 0.5;
    float dotm = smoothstep(0.35, 0.3, length(cell));
    vec3 base = vec3(0.82);
    vec3 col = mix(base, vec3(0.08), dotm);
    f_color = vec4(col, 1.0);
}
''',
uniforms={
    "separation": {"glsl": "float", "min": 4.0, "max": 80.0, "default": 40.0, "description": "stereo separation (px)"},
    "depth_scale": {"glsl": "float", "min": 0.1, "max": 1.5, "default": 1.0, "description": "depth relief"},
    "tile_size": {"glsl": "float", "min": 4.0, "max": 48.0, "default": 16.0, "description": "dot tile size (px)"},
})


# ── GPU-First gap mirrors: closed-form f(uv,t) twins for CPU nodes 487/441/108 ──
# Each CPU node is a per-pixel closed-form generator with NO close existing
# twin, so it gets a brand-new GLSL twin wired via a typed-uniform
# CLIENT_GPU_SHIMS entry (gpu_shaders.py). Every numeric CPU param becomes a
# named u_<name> uniform/SCALAR port (typed-uniform contract). CPU fns stay
# authoritative for export; GPU live preview is approximate by design. Palettes
# are inlined (cosine / hsv) to avoid the late-helper (inferno) ordering pitfall.

_register("galaxy_gpu", "Galaxy Generator (client-GPU twin of node 487)", "procedural",
'''void main() {
    vec2 uv = (gl_FragCoord.xy - 0.5 * u_resolution) / min(u_resolution.x, u_resolution.y);
    float incl = mix(0.45, 1.0, clamp(u_inclination, 0.0, 1.0));
    uv.y /= incl;
    float r = length(uv);
    float ang = atan(uv.y, uv.x);
    float t = u_time * 0.2 * (0.3 + u_rotation_speed);

    // central bulge
    float bulge = exp(-r * r / (u_bulge_size * u_bulge_size)) * 1.2;

    // logarithmic spiral arms
    float spiral = 0.0;
    int N = int(clamp(u_arms, 1.0, 6.0));
    for (int i = 0; i < 6; i++) {
        if (i >= N) break;
        float fi = float(i);
        float a = ang + log(r + 0.06) / max(u_tightness, 0.05)
                  - fi * 6.2831853 / float(N) + t;
        float c = cos(a);
        spiral += exp((c - 1.0) * (4.0 / max(u_arm_spread, 0.02)));
    }
    spiral *= exp(-r * 1.3);

    float density = bulge + spiral;

    // procedural star sparkle
    float tw = hash21(floor(uv * 90.0));
    density += smoothstep(0.93, 1.0, tw) * (0.3 + 0.7 * fbm(uv * 18.0 + t)) * (0.4 + density);
    density = clamp(density * u_brightness, 0.0, 1.0);

    // natural palette: warm core -> blue rim, depth by radius
    vec3 core = vec3(1.0, 0.85, 0.6);
    vec3 midc = vec3(1.0, 0.95, 0.85);
    vec3 rim  = vec3(0.55, 0.7, 1.0);
    vec3 col = mix(core, midc, smoothstep(0.0, 0.3, r));
    col = mix(col, rim, smoothstep(0.3, 0.9, r));
    col *= density;
    col += vec3(0.02, 0.03, 0.05) * (1.0 - r);
    f_color = vec4(clamp(col, 0.0, 1.0), 1.0);
}
''',
uniforms={
    "arms": {"glsl": "float", "min": 1.0, "max": 6.0, "default": 2.0, "description": "number of spiral arms"},
    "tightness": {"glsl": "float", "min": 0.1, "max": 1.5, "default": 0.5, "description": "spiral winding tightness"},
    "arm_spread": {"glsl": "float", "min": 0.02, "max": 0.4, "default": 0.15, "description": "arm width"},
    "bulge_size": {"glsl": "float", "min": 0.05, "max": 0.5, "default": 0.2, "description": "central bulge radius"},
    "inclination": {"glsl": "float", "min": 0.0, "max": 1.0, "default": 0.3, "description": "view inclination (vertical squash)"},
    "rotation_speed": {"glsl": "float", "min": 0.0, "max": 3.0, "default": 1.0, "description": "arm rotation speed"},
    "brightness": {"glsl": "float", "min": 0.2, "max": 3.0, "default": 1.0, "description": "overall brightness"},
})

_register("contours_gpu", "Marching Squares Contours (client-GPU twin of node 441)", "procedural",
'''void main() {
    vec2 uv = (gl_FragCoord.xy - 0.5 * u_resolution) / min(u_resolution.x, u_resolution.y);
    float t = u_time;

    // scalar field: fbm noise blended with a radial wave
    vec2 flow = vec2(t * u_flow_amp, -t * u_flow_amp * 0.5);
    float n = fbm(uv * 3.0 + flow);
    float radial = 0.5 + 0.5 * sin(length(uv) * 6.0 - t * 0.3);
    float field = mix(n, radial, 0.4) * u_noise_amp + (1.0 - u_noise_amp) * n;
    field = clamp(field, 0.0, 1.0);

    int N = int(clamp(u_n_levels, 3.0, 24.0));
    float lv = field * float(N);
    float f = fract(lv);
    float d = min(f, 1.0 - f);                 // distance to nearest iso-level
    float w = fwidth(lv) * 1.5 + 0.015;
    float line = 1.0 - smoothstep(0.0, w, d);

    // faint reference grid (grid_step ~ pixels per cell)
    vec2 g = abs(fract(uv * (10.0 / max(u_grid_step, 1.0))) - 0.5);
    float grid = 1.0 - smoothstep(0.0, 0.04, min(g.x, g.y));
    line = max(line * u_line_alpha, grid * 0.12);

    // color by level (level mode)
    float lev = floor(lv) / float(N);
    vec3 cmap = 0.5 + 0.5 * cos(6.2831853 * (lev + vec3(0.0, 0.33, 0.67)));
    vec3 col = mix(vec3(0.05, 0.06, 0.08), cmap, line);
    f_color = vec4(clamp(col, 0.0, 1.0), 1.0);
}
''',
uniforms={
    "n_levels": {"glsl": "float", "min": 3.0, "max": 24.0, "default": 10.0, "description": "number of contour levels"},
    "grid_step": {"glsl": "float", "min": 2.0, "max": 16.0, "default": 5.0, "description": "reference grid cell size"},
    "line_alpha": {"glsl": "float", "min": 0.1, "max": 1.0, "default": 0.9, "description": "contour line opacity"},
    "flow_amp": {"glsl": "float", "min": 0.0, "max": 0.5, "default": 0.2, "description": "animated flow amplitude"},
    "noise_amp": {"glsl": "float", "min": 0.1, "max": 1.0, "default": 0.6, "description": "noise contribution to field"},
})

_register("hypercube_gpu", "4D Hypercube (client-GPU twin of node 108)", "procedural",
'''float _hc_pc(int x) {
    int c = 0;
    for (int k = 0; k < 4; k++) { c += (x >> k) & 1; }
    return float(c);
}
float _hc_distSeg(vec2 p, vec2 a, vec2 b) {
    vec2 pa = p - a, ba = b - a;
    float h = clamp(dot(pa, ba) / max(dot(ba, ba), 1e-6), 0.0, 1.0);
    return length(pa - ba * h);
}
void main() {
    vec2 uv = (gl_FragCoord.xy - 0.5 * u_resolution) / min(u_resolution.x, u_resolution.y);
    float t = u_time;

    float a1 = t * (0.2 + u_speed_xw * 0.5);
    float a2 = t * (0.2 + u_speed_yw * 0.5);
    float cx = cos(a1), sx = sin(a1);
    float cy = cos(a2), sy = sin(a2);

    vec2 pos[16];
    vec3 colw[16];
    for (int i = 0; i < 16; i++) {
        float x = ((i >> 0) & 1) == 1 ? 1.0 : -1.0;
        float y = ((i >> 1) & 1) == 1 ? 1.0 : -1.0;
        float z = ((i >> 2) & 1) == 1 ? 1.0 : -1.0;
        float w = ((i >> 3) & 1) == 1 ? 1.0 : -1.0;
        float rx = x * cx - w * sx;
        float rw = x * sx + w * cx;
        float ry = y * cy - z * sy;
        float rz = y * sy + z * cy;
        float k = 1.0 / (u_proj_radius - rw);
        vec3 p3 = vec3(rx, ry, rz) * k * u_proj_radius * 0.32;
        pos[i] = p3.xy;
        float hue = 0.5 * (rw + 1.0);
        colw[i] = vec3(0.5 + 0.5 * cos(6.2831853 * (u_inner_hue + hue)),
                       0.5 + 0.5 * cos(6.2831853 * (u_outer_hue + hue + 0.33)),
                       0.5 + 0.5 * cos(6.2831853 * (u_inner_hue + hue + 0.67)));
    }

    float dmin = 1e9;
    vec3 edgeCol = vec3(0.0);
    for (int i = 0; i < 16; i++) {
        for (int j = i + 1; j < 16; j++) {
            if (_hc_pc(i ^ j) == 1.0) {
                float d = _hc_distSeg(uv, pos[i], pos[j]);
                if (d < dmin) { dmin = d; edgeCol = 0.5 * (colw[i] + colw[j]); }
            }
        }
    }
    float lw = u_line_width * 0.0025;
    float line = 1.0 - smoothstep(0.0, lw, dmin);
    vec3 col = line * edgeCol + vec3(0.02, 0.02, 0.03) * (1.0 - line);
    f_color = vec4(clamp(col, 0.0, 1.0), 1.0);
}
''',
uniforms={
    "speed_xw": {"glsl": "float", "min": 0.1, "max": 3.0, "default": 0.5, "description": "XW-plane rotation speed"},
    "speed_yw": {"glsl": "float", "min": 0.1, "max": 3.0, "default": 0.3, "description": "YW-plane rotation speed"},
    "proj_radius": {"glsl": "float", "min": 2.0, "max": 6.0, "default": 3.5, "description": "4D perspective radius"},
    "line_width": {"glsl": "float", "min": 1.0, "max": 4.0, "default": 3.0, "description": "edge thickness"},
    "inner_hue": {"glsl": "float", "min": 0.0, "max": 1.0, "default": 0.55, "description": "inner vertex hue"},
    "outer_hue": {"glsl": "float", "min": 0.0, "max": 1.0, "default": 0.08, "description": "outer vertex hue"},
})


# ═══════════════════════════════════════════════
#  RENDER ENGINE
# ═══════════════════════════════════════════════

# Per-thread program + VAO cache keyed by shader_name.
# Avoids recompiling GLSL and rebuilding VAOs every frame — the dominant cost
# for small shaders on the live loop.  Each thread owns its own cache because
# GL programs are bound to the context that created them.
_prog_cache_local = threading.local()


def _get_prog_cache() -> dict:
    cache = getattr(_prog_cache_local, "cache", None)
    if cache is None:
        _prog_cache_local.cache = {}
    return _prog_cache_local.cache


def _create_vao(ctx, prog):
    """Create full-screen quad VAO."""
    vbo = ctx.buffer(_QUAD_VERTICES.tobytes())
    ibo = ctx.buffer(_QUAD_INDICES.tobytes())
    vao = ctx.vertex_array(prog, [
        (vbo, '2f 2f', 'in_vert', 'in_uv'),
    ], ibo)
    return vao


def render_shader(shader_name: str, resolution: tuple[int, int] = (512, 512),
                   params: tuple[float, float, float, float] = (0.5, 0.5, 0.5, 0.5),
                   time: float = 0.0,
                   input_image: np.ndarray | None = None,
                   named_params: dict | None = None) -> Image.Image:
    """Render a shader to an image.

    Args:
        shader_name: Name in SHADERS dict
        resolution: (width, height) output size
        params: 4 float uniforms mapped to u_params (legacy shaders)
        time: Time value for u_time animation
        input_image: Optional numpy array (H,W,3) float32 [0,1] or uint8,
                     for filter shaders
        named_params: values for the shader's typed uniforms, keyed by the
                      declared name (set as u_<name>; coerced per spec)

    Returns: PIL Image
    """
    if shader_name not in SHADERS:
        raise ValueError(f"Unknown shader: {shader_name}. Available: {list(SHADERS.keys())}")

    info = SHADERS[shader_name]
    ctx = _get_ctx()
    cache = _get_prog_cache()

    w, h = resolution

    # Build fragment shader source (single assembly path — shared with the
    # parity layer so build_fragment('gl330') is exactly what compiles here).
    frag_src = _assemble_gl330(info)

    # Cache program + VAO per shader name (recompile on first use per thread)
    if shader_name not in cache:
        try:
            prog = ctx.program(vertex_shader=_VERTEX_SHADER, fragment_shader=frag_src)
        except Exception as e:
            raise RuntimeError(f"Shader compilation failed for '{shader_name}': {e}")
        vao = _create_vao(ctx, prog)
        cache[shader_name] = (prog, vao)
    else:
        prog, vao = cache[shader_name]

    # Framebuffer is resolution-specific — create fresh each call (cheap)
    fbo = ctx.simple_framebuffer((w, h))
    fbo.use()

    # Set uniforms (some may be optimised out by the GLSL compiler)
    for uniform_name, uniform_value in [('u_resolution', (float(w), float(h))),
                                         ('u_time', time),
                                         ('u_params', params)]:
        if uniform_name in prog:
            prog[uniform_name].value = uniform_value

    # Typed uniforms: u_<name> per the shader's declared spec.
    # Missing values fall back to the spec default (NOT 0.5) so a variable left
    # unwired still renders at its authored neutral instead of going black.
    uspec = info.get("uniforms") or {}
    if uspec:
        vals = named_params or {}
        for uname, spec in uspec.items():
            gl_name = f"u_{uname}"
            if gl_name in prog:
                prog[gl_name].value = coerce_uniform(spec, vals.get(uname, spec.get("default")))

    # Handle input texture — accept float32 [0,1] or uint8 [0,255]
    texture = None
    if input_image is not None and 'u_texture' in prog:
        if input_image.dtype != np.uint8:
            img_u8 = (np.clip(input_image, 0.0, 1.0) * 255).astype(np.uint8)
        else:
            img_u8 = input_image
        tex_data = img_u8[:, :, ::-1].tobytes()  # RGB -> BGR for GL
        texture = ctx.texture((img_u8.shape[1], img_u8.shape[0]), 3, tex_data)
        texture.use(0)
        prog['u_texture'].value = 0

    ctx.clear(0.0, 0.0, 0.0)
    vao.render()
    data = fbo.read()

    # Convert to PIL
    img = Image.frombytes('RGB', (w, h), data, 'raw', 'BGR')

    # Release per-frame resources (program + VAO stay in cache)
    fbo.release()
    if texture is not None:
        texture.release()

    return img


def _have_cv2():
    """Check if OpenCV is available."""
    try:
        import cv2
        return True
    except ImportError:
        return False


# ═══════════════════════════════════════════════
#  HANDY HELPERS
# ═══════════════════════════════════════════════

def list_shaders(shader_type: str | None = None) -> list[dict]:
    """List all available shaders, optionally filtered by type."""
    if shader_type:
        return [v for v in SHADERS.values() if v["type"] == shader_type]
    return list(SHADERS.values())


def render_procedural(shader_name: str, resolution=(512, 512), params=(0.5, 0.5, 0.5, 0.5),
                       time=0.0) -> Image.Image:
    """Render a procedural shader (no input image needed)."""
    info = SHADERS.get(shader_name)
    if info and info["type"] == "filter":
        raise ValueError(f"'{shader_name}' is a filter shader, use render_filter() instead")
    return render_shader(shader_name, resolution, params, time)


def render_filter(shader_name: str, input_image: np.ndarray,
                   params=(0.5, 0.5, 0.5, 0.5), time=0.0) -> Image.Image:
    """Apply a filter shader to an input image."""
    info = SHADERS.get(shader_name)
    if info and info["type"] == "procedural":
        raise ValueError(f"'{shader_name}' is a procedural shader, use render_procedural() instead")
    h, w = input_image.shape[:2]
    return render_shader(shader_name, (w, h), params, time, input_image)


# ── Default starter template exposed to the UI ─────────────────────
CUSTOM_SHADER_TEMPLATE = '''void main() {
    vec2 uv = v_uv;
    float t = u_time * 0.5;

    // u_params.x = p1, u_params.y = p2, u_params.z = p3, u_params.w = p4
    float v = sin(uv.x * 10.0 + t) * cos(uv.y * 8.0 + t * 0.7);
    v = v * 0.5 + 0.5;

    f_color = vec4(v, v * 0.5, 1.0 - v, 1.0);
}'''


def render_custom_shader(
    glsl_body: str,
    resolution: tuple[int, int] = (512, 512),
    params: tuple[float, float, float, float] = (0.5, 0.5, 0.5, 0.5),
    time: float = 0.0,
    input_image: np.ndarray | None = None,
) -> Image.Image:
    """Compile and render a user-supplied GLSL fragment shader.

    `glsl_body` is the full void main(){...} block. The _PROLOGUE (uniforms +
    helpers) is prepended automatically — the user does not write #version or
    uniform declarations.

    Raises RuntimeError with a human-readable message on compile failure.
    """
    import hashlib
    frag_src = _PROLOGUE + "\n" + glsl_body

    ctx = _get_ctx()
    cache = _get_prog_cache()

    # Cache key = SHA-1 of the full fragment source (thread-local per ctx)
    cache_key = "__custom__:" + hashlib.sha1(frag_src.encode()).hexdigest()

    if cache_key not in cache:
        try:
            prog = ctx.program(vertex_shader=_VERTEX_SHADER, fragment_shader=frag_src)
        except Exception as e:
            raise RuntimeError(str(e))
        vao = _create_vao(ctx, prog)
        cache[cache_key] = (prog, vao)
    else:
        prog, vao = cache[cache_key]

    w, h = resolution
    fbo = ctx.simple_framebuffer((w, h))
    fbo.use()

    for uniform_name, uniform_value in [
        ('u_resolution', (float(w), float(h))),
        ('u_time', time),
        ('u_params', params),
    ]:
        if uniform_name in prog:
            prog[uniform_name].value = uniform_value

    texture = None
    if input_image is not None and 'u_texture' in prog:
        if input_image.dtype != np.uint8:
            img_u8 = (np.clip(input_image, 0.0, 1.0) * 255).astype(np.uint8)
        else:
            img_u8 = input_image
        tex_data = img_u8[:, :, ::-1].tobytes()
        texture = ctx.texture((img_u8.shape[1], img_u8.shape[0]), 3, tex_data)
        texture.use(0)
        prog['u_texture'].value = 0

    ctx.clear(0.0, 0.0, 0.0)
    vao.render()
    data = fbo.read()

    img = Image.frombytes('RGB', (w, h), data, 'raw', 'BGR')

    fbo.release()
    if texture is not None:
        texture.release()

    return img


# ═══════════════════════════════════════════════
#  P0 client-GPU parity shaders for existing CPU nodes
# ═══════════════════════════════════════════════
# Render EXISTING CPU pattern nodes (04 Worley, 02 Quasicrystal) on the browser
# GPU for the live preview (see methods/gpu_shaders.py CLIENT_GPU_SHIMS). The CPU
# numpy node stays the authoritative export (two-tier precision).
#
# NOTE (determinism): both CPU nodes seed feature-point positions / per-wave
# phases with numpy PCG64 (np.random.default_rng), which GLSL cannot reproduce.
# These shaders replicate the STRUCTURE via a GLSL hash, so the live look matches
# the node's character but not the exact seeded layout. High/exact parity needs a
# derived-uniforms path (compute the RNG values server-side -> uniforms) - deferred.

_INFERNO = """
vec3 inferno(float t){ t = clamp(t, 0.0, 1.0);
  const vec3 c0=vec3(0.00021894,0.00016488,-0.01907227);
  const vec3 c1=vec3(0.10651034,0.56396050, 3.93279110);
  const vec3 c2=vec3(11.6028830,-3.9781129,-15.9420510);
  const vec3 c3=vec3(-41.703996,17.4360890, 44.3541450);
  const vec3 c4=vec3(77.1629350,-33.402243,-81.8094230);
  const vec3 c5=vec3(-71.319421,32.6260640, 73.2095190);
  const vec3 c6=vec3(25.1311300,-12.242810,-23.0709590);
  return c0+t*(c1+t*(c2+t*(c3+t*(c4+t*(c5+t*c6)))));
}
"""

# ── P0.5 typed-uniform procedural twins: Flow Noise / Spot Noise / Mathematical Marbling ──
# Client-GPU live-preview mirrors of CPU nodes 535 / 534 / 953.
# Every numeric CPU param is bound to a named u_<name> uniform (typed-uniform
# contract). Choice/string params (colormode/palette/source/anim_mode/flow) and the
# CPU-only variable counts (n_spots/n_drops/n_tines) are dropped + justified in
# GPU_PREVIEW_DROP_ALLOW. CPU numpy fns stay authoritative; the GPU twins are
# approximate-by-design live previews (fixed 64-spot / 32-drop / 3-tine loops).
_register("flow_noise_gpu", "Flow Noise — rotating-gradient Perlin (client-GPU twin of node 535)", "procedural", _INFERNO + """
vec3 flow_color(float v){ return inferno(clamp(v,0.0,1.0)); }
void main() {
    vec2 uv = v_uv;
    float t = u_time * u_anim_speed;
    float sc = u_scale * (1.0 + 0.35 * sin(t));
    vec2 p = uv * sc;
    if (u_advect > 0.0) {
        vec2 w = vec2(fbm(p*0.4+3.1), fbm(p*0.4+9.7)) - 0.5;
        p += w * u_advect * 6.0 * (0.5 + 0.5*sin(t*0.7));
    }
    float ang = t * (0.6 + u_spin_var);
    vec2 warp = rot(ang) * vec2(fbm(p*0.5), fbm(p*0.5+21.0));
    float v = 0.0; float amp = 0.5; float norm = 0.0; float ss = 1.0;
    for (int o = 0; o < 6; o++) {
        if (float(o) >= u_octaves) break;
        vec2 q = (p + warp*(1.0+u_spin_var*2.0)) * ss;
        v += amp * fbm(q);
        norm += amp; amp *= 0.5; ss *= 2.0;
    }
    v = (norm > 0.0) ? v/norm : 0.0;
    v = 0.5 + 0.5 * v * u_contrast;
    v = clamp(v, 0.0, 1.0);
    f_color = vec4(flow_color(v), 1.0);
}
""",
    uniforms={
        "scale": {"glsl": "float", "min": 12.0, "max": 260.0, "default": 90.0, "description": "feature size in pixels (lattice spacing)"},
        "octaves": {"glsl": "float", "min": 1.0, "max": 6.0, "default": 4.0, "description": "fractal octaves (turbulent detail)"},
        "spin_var": {"glsl": "float", "min": 0.0, "max": 1.0, "default": 0.6, "description": "0 = uniform global spin, 1 = per-cell chaotic spin"},
        "advect": {"glsl": "float", "min": 0.0, "max": 3.0, "default": 1.2, "description": "pseudo-advection strength (domain transport)"},
        "contrast": {"glsl": "float", "min": 0.4, "max": 2.5, "default": 1.15, "description": "final tone contrast"},
        "anim_speed": {"glsl": "float", "min": 0.1, "max": 3.0, "default": 1.0, "description": "animation speed multiplier"},
    }
    )

_register("spot_noise_gpu", "Spot Noise — flow-oriented anisotropic spots (client-GPU twin of node 534)", "procedural", _INFERNO + """
void main() {
    vec2 uv = v_uv;
    vec2 res = u_resolution;
    float t = u_time * u_anim_speed;
    float contrast = u_contrast;
    float spot = u_spot_size;
    float stretch = u_stretch;
    float field = 0.0;
    for (int i = 0; i < 64; i++) {
        float fi = float(i);
        vec2 hc = vec2(hash21(vec2(fi, 1.0)), hash21(vec2(fi, 7.0)));
        vec2 c = hc * res;
        vec2 rel = (c / res) - 0.5;
        float theta = atan(rel.y, rel.x) + 1.5707963;
        c += vec2(cos(theta), sin(theta)) * (t * 22.0);
        c = mod(c, res);
        theta += t * 0.4;
        float ct = cos(theta), st = sin(theta);
        float sa = spot * stretch;
        float sb = spot / sqrt(max(stretch, 1e-3));
        vec2 d = (uv * res) - c;
        float uu = ct*d.x + st*d.y;
        float vv = -st*d.x + ct*d.y;
        float g = exp(-(uu*uu/(2.0*sa*sa) + vv*vv/(2.0*sb*sb)));
        float amp = (hash21(vec2(fi, 13.0)) - 0.5) * 2.0;
        amp *= (0.5 + 0.5*sin(t*0.7));
        field += amp * g;
    }
    float val = 0.5 + field / (0.6 * 4.0);
    val = clamp(0.5 + (val - 0.5) * contrast, 0.0, 1.0);
    f_color = vec4(inferno(val), 1.0);
}
""",
    uniforms={
        "spot_size": {"glsl": "float", "min": 3.0, "max": 40.0, "default": 14.0, "description": "base spot radius in px (before stretch)"},
        "stretch": {"glsl": "float", "min": 1.0, "max": 12.0, "default": 5.0, "description": "anisotropy: elongation along the flow direction"},
        "contrast": {"glsl": "float", "min": 0.5, "max": 3.0, "default": 1.4, "description": "final tone contrast"},
        "anim_speed": {"glsl": "float", "min": 0.1, "max": 3.0, "default": 1.0, "description": "animation speed multiplier"},
    }
    )

_register("marbling_gpu", "Mathematical Marbling — closed-form fluid advection (client-GPU twin of node 953)", "procedural", _INFERNO + """
void main() {
    vec2 uv = v_uv;
    vec2 res = u_resolution;
    float t = u_time * u_anim_speed;
    float minwh = min(res.x, res.y);
    float base_r = u_drop_radius * minwh;
    float tstr = u_tine_strength * minwh;
    float tc = max(1e-3, u_tine_sharpness * minwh);
    vec3 bg = vec3(0.96);
    vec2 q = uv * res;
    for (int k = 0; k < 3; k++) {
        float fk = float(k);
        float ang = 6.2831853 * fk / 3.0 + 0.3;
        vec2 that = vec2(cos(ang), sin(ang));
        vec2 nhat = vec2(-that.y, that.x);
        float sweep = sin(t*0.6 + fk*1.3) * 0.5 + 0.5;
        vec2 p0 = vec2(sweep * res.x, (0.5 + 0.3*cos(fk)) * res.y);
        vec2 rel = q - p0;
        float along = dot(rel, that);
        float dd = abs(along);
        float decay = exp(-dd / tc);
        float disp = tstr * decay;
        q -= disp * nhat;
    }
    vec3 outc = bg;
    for (int i = 31; i >= 0; i--) {
        float fi = float(i);
        vec2 hc = vec2(hash21(vec2(fi + u_seed, 2.0)), hash21(vec2(fi + u_seed, 8.0)));
        vec2 ctr = (0.08 + 0.84*hc) * res;
        float rr = base_r * (0.6 + 0.8*hash21(vec2(fi + u_seed, 17.0)));
        vec2 d = q - ctr;
        if (dot(d,d) <= rr*rr) {
            float hh = hash21(vec2(fi + u_seed, 23.0));
            outc = 0.5 + 0.5*vec3(sin(hh*6.2831853), sin(hh*6.2831853+2.0943951), sin(hh*6.2831853+4.1887902));
        }
    }
    f_color = vec4(clamp(outc, 0.0, 1.0), 1.0);
}
""",
    uniforms={
        "drop_radius": {"glsl": "float", "min": 0.01, "max": 0.3, "default": 0.09, "description": "base drop radius (fraction of min(W,H))"},
        "tine_strength": {"glsl": "float", "min": 0.0, "max": 0.6, "default": 0.22, "description": "tine displacement magnitude"},
        "tine_sharpness": {"glsl": "float", "min": 0.02, "max": 0.5, "default": 0.14, "description": "tine sharpness c (smaller = sharper)"},
        "anim_speed": {"glsl": "float", "min": 0.1, "max": 5.0, "default": 1.0, "description": "animation speed multiplier"},
        "seed": {"glsl": "float", "min": 0.0, "max": 99999.0, "default": 42.0, "description": "random seed for drop placement"},
    }
    )

_register("worley_gpu", "Worley/Cellular F1 noise (client-GPU twin of node 04)", "procedural", _INFERNO + """
vec2 h22(vec2 p){ p = fract(p*vec2(123.34,456.21)); p += dot(p,p+45.32); return fract(vec2(p.x*p.y, p.x+p.y)); }
void main() {
    // u_params.x = jitter (0..1), u_params.y = cell density scale
    float jitter = u_jitter;
    float cells  = 4.0 + u_fractal_gain * 14.0;
    vec2 st = v_uv * cells;
    vec2 g = floor(st), f = fract(st);
    float d = 8.0;
    for (int y = -1; y <= 1; y++) {
        for (int x = -1; x <= 1; x++) {
            vec2 o = vec2(float(x), float(y));
            vec2 fp = 0.5 + (h22(g + o) - 0.5) * jitter;
            d = min(d, length(o + fp - f));
        }
    }
    f_color = vec4(inferno(clamp(d, 0.0, 1.0)), 1.0);
}
""",
    uniforms={
    "jitter": {"glsl": "float", "min": 0.0, "max": 1.0, "default": 0.5, "description": "point jitter"},
    "fractal_gain": {"glsl": "float", "min": 0.0, "max": 1.0, "default": 0.5, "description": "cell density scale"}
}
    )

_register("quasicrystal_gpu", "Quasicrystal wave superposition (client-GPU twin of node 02)", "procedural", _INFERNO + """
float h11(float n){ return fract(sin(n*127.1)*43758.5453); }
void main() {
    // u_params.x = frequency, .y = amplitude, .z = rotation, .w = wave count
    float freq = max(u_frequency, 0.005);
    float amp  = (u_amplitude <= 0.0) ? 1.0 : u_amplitude;
    float rot  = u_rotation;
    int nwaves = int(clamp(u_waves, 2.0, 24.0));
    vec2 p = v_uv * u_resolution - 0.5 * u_resolution;      // centered pixel coords
    float phi = 3.14159265 * (1.0 + 2.2360679) / 2.0;        // pi*(1+sqrt5)/2 (penrose)
    float field = 0.0;
    for (int i = 0; i < 24; i++) {
        if (i >= nwaves) break;
        float fi = float(i);
        float theta = mod(fi * 6.2831853 / phi + rot, 6.2831853);
        float ph = h11(fi + 1.0) * 6.2831853;                // hash phase (!= numpy RNG)
        float f  = freq * (0.5 + h11(fi + 100.0));           // hash freq jitter
        float proj = p.x * cos(theta) + p.y * sin(theta);
        field += sin(proj * f + ph) * amp;
    }
    float result = field / float(nwaves) * 0.5 + 0.5;        // approx of CPU norm()
    f_color = vec4(inferno(clamp(result, 0.0, 1.0)), 1.0);
}
""",
    uniforms={
    "frequency": {"glsl": "float", "min": 0.005, "max": 2.0, "default": 0.5, "description": "wave frequency"},
    "amplitude": {"glsl": "float", "min": 0.0, "max": 2.0, "default": 1.0, "description": "wave amplitude"},
    "rotation": {"glsl": "float", "min": 0.0, "max": 6.2831853, "default": 0.0, "description": "wave rotation"},
    "waves": {"glsl": "float", "min": 2.0, "max": 24.0, "default": 12.0, "description": "wave count"}
}
    )


# ═══════════════════════════════════════════════════════════════════════════
#  P1 — GPU reaction-diffusion sim shaders (client ping-pong; server untouched)
# ═══════════════════════════════════════════════════════════════════════════
# Three-shader set for a WebGL2 ping-pong float-state simulation. The client
# (client3d.js) owns the {a,b} RGBA-float state pair and the substep loop; these
# GLSL bodies define seed → step → display. State packs U in .r, V in .g.
# The CPU node (methods/simulations/gray_scott.py, id 155) stays the
# authoritative export — nothing here is rendered server-side.

_register("wallpaper_gpu",
            "Wallpaper-group tiling (client-GPU twin of node 06)",
            "procedural", _INFERNO + """
void main() {
    // u_params.x = tile size (log-scaled), .y = color variation, .z = rotation noise
    float ts = mix(8.0, 64.0, clamp(u_scale_variation, 0.0, 1.0));
    float cv = u_color_variation;
    vec2 st = v_uv * u_resolution / ts;
    vec2 g = floor(st), f = fract(st);
    // per-tile slight rotation + hue offset (echoes rotation_noise/color_variation)
    float r = (hash21(g) - 0.5) * 6.2831853 * cv;
    float hue = hash21(g + 3.17) * cv;
    vec2 p = (f - 0.5) * rot(r);
    float d = abs(p.x) + abs(p.y);            // diamond motif
    vec3 col = 0.5 + 0.5 * cos(6.2831853 * (hue + vec3(0.0, 0.33, 0.67)));
    col *= smoothstep(0.5, 0.45, d);
    f_color = vec4(inferno(clamp(length(col) * 0.6 + d * 0.4, 0.0, 1.0)), 1.0);
}
""",
    uniforms={
    "scale_variation": {"glsl": "float", "min": 0.0, "max": 1.0, "default": 0.5, "description": "tile size variation"},
    "color_variation": {"glsl": "float", "min": 0.0, "max": 1.0, "default": 0.5, "description": "color variation"}
}
    )

_register("morph_grid_gpu",
            "Morphing grid warp (client-GPU twin of node 105)",
            "procedural", _INFERNO + """
void main() {
    // u_params.x = warp strength, .y = line width, .z = palette mix
    float ws = u_warp_strength;
    float lw = clamp(u_line_width, 0.02, 1.0);
    vec2 p = v_uv * 14.0;
    vec2 w = vec2(fbm(p + u_time * 0.1), fbm(p.yx - u_time * 0.1));
    p += (w - 0.5) * ws * 6.0;
    vec2 g = abs(fract(p) - 0.5);
    float line = smoothstep(lw, lw * 0.5, min(g.x, g.y));
    vec3 col = 0.5 + 0.5 * cos(6.2831853 * (0.5 + vec3(0.0, 0.33, 0.67)) + w.x * 4.0);
    f_color = vec4(mix(vec3(line), col, line), 1.0);
}
""",
    uniforms={
    "warp_strength": {"glsl": "float", "min": 0.0, "max": 2.0, "default": 0.5, "description": "warp strength"},
    "line_width": {"glsl": "float", "min": 0.02, "max": 1.0, "default": 0.5, "description": "line width"}
}
    )

_register("phyllotaxis_gpu",
            "Phyllotaxis spiral field (client-GPU twin of node 08)",
            "procedural", _INFERNO + """
void main() {
    // u_params.x = point density, .y = angle goldenness, .z = radius scale
    float dens = mix(0.1, 1.0, clamp(u_points, 0.0, 1.0));
    float phi = 2.39996323 + u_angle * 1.5;        // ~golden angle + jitter
    vec2 c = (v_uv - 0.5) * u_resolution;
    float rmax = 0.5 * min(u_resolution.x, u_resolution.y);
    float acc = 0.0;
    for (int i = 0; i < 220; i++) {
        float fi = float(i) * dens * 12.0;
        float a = fi * phi;
        float rad = sqrt(fi) * (u_radius_scale * 0.5 + 0.05) * rmax * 0.06;
        vec2 pos = rad * vec2(cos(a), sin(a));
        acc += smoothstep(3.0, 0.0, length(c - pos));
    }
    f_color = vec4(inferno(clamp(acc * 0.5, 0.0, 1.0)), 1.0);
}
""",
    uniforms={
    "points": {"glsl": "float", "min": 0.0, "max": 1.0, "default": 0.5, "description": "point density"},
    "angle": {"glsl": "float", "min": 0.0, "max": 1.5, "default": 0.5, "description": "angle goldenness"},
    "radius_scale": {"glsl": "float", "min": 0.0, "max": 2.0, "default": 0.5, "description": "radius scale"}
}
    )


_register("grayscott_seed",
          "Gray-Scott initial state: U=1, V=hashed seed blobs (client-GPU sim of node 155)",
          "procedural", '''
void main() {
    // U (substrate) starts ~1 everywhere; V (activator) as a few blobs, echoing
    // the CPU node's n_seeds gaussian patches (positions from a stable hash — the
    // GPU twin cannot reproduce numpy's PCG64 layout; it just needs live seeds).
    float U = 1.0;
    float V = 0.0;
    for (int i = 0; i < 20; i++) {
        float fi = float(i);
        vec2 c = vec2(hash21(vec2(fi + 0.5, 1.37)),
                      hash21(vec2(fi + 0.5, 7.91)));
        c = 0.05 + 0.90 * c;                 // keep blobs off the very edge
        float d = distance(v_uv, c);
        V += 0.5 * exp(-(d * d) / 0.0016);
    }
    V = clamp(V, 0.0, 1.0);
    f_color = vec4(U, V, 0.0, 1.0);
}
''')

_register("grayscott_step",
          "Gray-Scott one Euler step (5-point Laplacian, toroidal) — reads/writes RG state",
          "procedural", '''
void main() {
    vec2 texel = 1.0 / u_resolution;
    vec4 s = texture(u_texture, v_uv);
    float U = s.r, V = s.g;
    // 5-point Laplacian; RepeatWrapping on the state texture makes it toroidal,
    // matching the CPU np.roll boundaries.
    vec4 sl = texture(u_texture, v_uv + vec2(-texel.x, 0.0));
    vec4 sr = texture(u_texture, v_uv + vec2( texel.x, 0.0));
    vec4 su = texture(u_texture, v_uv + vec2(0.0,  texel.y));
    vec4 sd = texture(u_texture, v_uv + vec2(0.0, -texel.y));
    float lapU = sl.r + sr.r + su.r + sd.r - 4.0 * U;
    float lapV = sl.g + sr.g + su.g + sd.g - 4.0 * V;
    float F  = u_params.x;   // feed
    float k  = u_params.y;   // kill
    float Du = u_params.z;   // diffusion U
    float Dv = u_params.w;   // diffusion V
    float uvv = U * V * V;
    const float dt = 1.0;    // CPU default timestep; substeps control pace
    float nU = U + dt * (Du * lapU - uvv + F * (1.0 - U));
    float nV = V + dt * (Dv * lapV + uvv - (F + k) * V);
    f_color = vec4(clamp(nU, 0.0, 1.0), clamp(nV, 0.0, 1.0), 0.0, 1.0);
}
''')

_register("grayscott_display",
          "Gray-Scott display: V activator → grayscale (gamma 0.5, matches _render_v)",
          "procedural", '''
void main() {
''')

# ── Sel'kov Glycolysis (client-GPU sim of node 1003) ─────────────────────────
# Excitable two-variable reaction-diffusion (Sel'kov 1968): substrate-depletion
# kinetics u²v with TWO diffusing species. State packs U in .r, V in .g. The
# medium is *excitable* (not Turing): a perturbation ignites a wavefront that
# travels and curls into spirals — the signature of glycolytic waves and a
# different dynamical regime from Gray-Scott (id 155) and BZ (id 91). CPU node
# stays authoritative for export (two-tier precision); this is the live twin.
_register("selkov_seed",
          "Sel'kov initial state: U~0.6, V~0.25 with a hashed ignition blob (node 1003 twin)",
          "procedural", '''
void main() {
    float U = 0.6;
    float V = 0.25;
    // Ignite one seeded blob near center so the excitable wave actually starts.
    // (The CPU node supports several seed shapes; the twin just needs ONE live
    // ignition to show the same spiral dynamics.)
    vec2 c = vec2(0.5);
    float d = distance(v_uv, c);
    float ign = smoothstep(0.04, 0.0, d);
    U = mix(U, 0.05, ign);
    V = mix(V, 0.85, ign);
    f_color = vec4(U, V, 0.0, 1.0);
}
''')

_register("selkov_step",
          "Sel'kov one Euler step (5-pt Laplacian, toroidal) — excitable u²v kinetics",
          "procedural", '''
void main() {
    vec2 texel = 1.0 / u_resolution;
    vec4 s = texture(u_texture, v_uv);
    float U = s.r, V = s.g;
    vec4 sl = texture(u_texture, v_uv + vec2(-texel.x, 0.0));
    vec4 sr = texture(u_texture, v_uv + vec2( texel.x, 0.0));
    vec4 su = texture(u_texture, v_uv + vec2(0.0,  texel.y));
    vec4 sd = texture(u_texture, v_uv + vec2(0.0, -texel.y));
    float lapU = sl.r + sr.r + su.r + sd.r - 4.0 * U;
    float lapV = sl.g + sr.g + su.g + sd.g - 4.0 * V;
    float a  = u_params.x;   // substrate supply
    float b  = u_params.y;   // intermediate removal
    float Du = u_params.z;   // diffusion U
    float Dv = u_params.w;   // diffusion V
    float uvv = U * U * V;
    float dt = 0.2;          // matches CPU default; substeps control pace
    float nU = U + dt * (a - U + uvv + Du * lapU);
    float nV = V + dt * (b * U * U - uvv + Dv * lapV);
    f_color = vec4(clamp(nU, 0.0, 2.0), clamp(nV, 0.0, 2.0), 0.0, 1.0);
}
''')

_register("selkov_display",
          "Sel'kov display: U substrate → heat ramp (matches _render_substrate)",
          "procedural", '''
void main() {
    float U = texture(u_texture, v_uv).r;
    float f = clamp(U / 1.5, 0.0, 1.0);
    f = pow(f, 0.6);                       // gamma lift
    vec3 col = vec3(f);                    // grayscale heat
    f_color = vec4(col, 1.0);
}
''')

# ── BZ Oregonator (client-GPU sim of node 91) ───────────────────────────────
# Two-variable reaction-diffusion with Oregonator kinetics. State packs U in
# .r, V in .g (same channel layout as Gray-Scott). CPU node is Arch-A sim; this
# is the live-preview twin only — server export stays authoritative.
_register("bz_seed",
          "BZ Oregonator initial state: U~1, V~0 with hashed seed blobs (node 91 twin)",
          "procedural", '''
void main() {
    float U = 1.0;
    float V = 0.0;
    for (int i = 0; i < 16; i++) {
        float fi = float(i);
        vec2 c = vec2(hash21(vec2(fi + 0.5, 1.37)),
                      hash21(vec2(fi + 0.5, 7.91)));
        c = 0.05 + 0.90 * c;
        float d = distance(v_uv, c);
        V += 0.6 * exp(-(d * d) / 0.004);
    }
    V = clamp(V, 0.0, 0.9);
    f_color = vec4(U, V, 0.0, 1.0);
}
''')

_register("bz_step",
          "BZ Oregonator one step (5-pt Laplacian, toroidal) — Oregonator kinetics",
          "procedural", '''
void main() {
    vec2 texel = 1.0 / u_resolution;
    vec4 s = texture(u_texture, v_uv);
    float U = s.r, V = s.g;
    vec4 sl = texture(u_texture, v_uv + vec2(-texel.x, 0.0));
    vec4 sr = texture(u_texture, v_uv + vec2( texel.x, 0.0));
    vec4 su = texture(u_texture, v_uv + vec2(0.0,  texel.y));
    vec4 sd = texture(u_texture, v_uv + vec2(0.0, -texel.y));
    float lapU = sl.r + sr.r + su.r + sd.r - 4.0 * U;
    float lapV = sl.g + sr.g + su.g + sd.g - 4.0 * V;
    float eps = u_params.x;   // epsilon (timescale separation)
    float q   = u_params.y;   // q
    float f   = u_params.z;   // f
    float Du  = u_params.w;   // diffusion U (Dv ~ 0 for classic BZ)
    float uvq = (U + q) > 0.0 ? (U * V * (U - q) / (U + q)) : 0.0;
    float dU = (U - U * U - f * uvq + Du * lapU) / max(eps, 1e-3);
    float dV = U - V + 0.0 * lapV;   // Dv ~ 0 -> V is reaction-dominated
    float nU = U + dU * 0.02;
    float nV = V + dV * 0.02;
    f_color = vec4(clamp(nU, 0.0, 1.0), clamp(nV, 0.0, 1.0), 0.0, 1.0);
}
''')

_register("bz_display",
          "BZ display: V activator -> grayscale",
          "procedural", '''
void main() {
    float V = clamp(texture(u_texture, v_uv).g, 0.0, 1.0);
    f_color = vec4(V, V, V, 1.0);
}
''')


# ── Cahn-Hilliard Phase Separation (client-GPU sim of node 1008) ───────────────
# Spinodal decomposition / phase coarsening — the free-energy model behind
# emulsions and alloy decomposition. A distinct regime from the reaction-
# diffusion twins (Gray-Scott 155, Sel'kov 1003, BZ 91): there is
# NO reaction term, only a double-well potential + interfacial energy.
# State packs φ (phase) in .r and μ (chemical potential) in .g (two
# channels). The CPU node (methods/simulations/cahn_hilliard.py) stays
# authoritative for export; this is the live-preview twin.
_register("cahn_hilliard_seed",
          "Cahn-Hilliard initial state: small-noise φ in .r, μ=0 in .g (node 1008 twin)",
          "procedural", '''
void main() {
    float amp = max(u_params.z, 0.05);   // seed_variance (p3)
    float hh = hash21(v_uv * 137.13 + 0.123);
    float phi = (hh - 0.5) * 2.0 * amp;
    f_color = vec4(phi, 0.0, 0.0, 1.0);  // .r = phi, .g = mu(0)
}
''')

_register("cahn_hilliard_step",
          "Cahn-Hilliard one step (5-pt Laplacian, toroidal) — two-channel state (.r=φ, .g=μ)",
          "procedural", '''
void main() {
    vec2 texel = 1.0 / u_resolution;
    vec4 s  = texture(u_texture, v_uv);
    float phi = s.r;
    float mu  = s.g;
    vec4 sl = texture(u_texture, v_uv + vec2(-texel.x, 0.0));
    vec4 sr = texture(u_texture, v_uv + vec2( texel.x, 0.0));
    vec4 su = texture(u_texture, v_uv + vec2(0.0,  texel.y));
    vec4 sd = texture(u_texture, v_uv + vec2(0.0, -texel.y));
    float eps = max(u_params.x, 0.01);   // interface width (p1)
    float mob = max(u_params.y, 0.01);   // mobility (p2)
    // Stable explicit dt for Model B: dt < 2/(mob*eps^2*kmax^2); kmax^2~9.87
    float dt = min(0.05, 1.5 / (mob * eps * eps * 9.87 + 1e-3));
    float lap_phi = sl.r + sr.r + su.r + sd.r - 4.0 * phi;
    float mu_new  = phi * phi * phi - phi - eps * eps * lap_phi;
    float lap_mu  = sl.g + sr.g + su.g + sd.g - 4.0 * mu;
    float phi_new = phi + dt * lap_mu;
    f_color = vec4(clamp(phi_new, -1.5, 1.5), mu_new, 0.0, 1.0);
}
''')

_register("cahn_hilliard_display",
          "Cahn-Hilliard display: φ (.r) → inferno colormap (phase look)",
          "procedural", _INFERNO + '''
void main() {
    float phi = texture(u_texture, v_uv).r;
    float t = clamp(phi * 0.5 + 0.5, 0.0, 1.0);
    vec3 col = inferno(t);
    f_color = vec4(col, 1.0);
}
''')

# ── Conway's Game of Life (client-GPU sim of nodes 18 / 58) ───────────────────────
# Single-channel CA: state.r = alive mask (0/1), state.g = age (frames alive).
# 8-neighbor toroidal count; birth on 3, survival on 2/3 (classic Conway).
_register("ca_seed",
          "Game of Life seed: hashed random alive cells at given density (nodes 18/58 twin)",
          "procedural", '''
void main() {
    float dens = clamp(u_params.x, 0.02, 0.9);
    float h = hash21(floor(v_uv * u_resolution * 0.5));
    float alive = h < dens ? 1.0 : 0.0;
    f_color = vec4(alive, 0.0, 0.0, 1.0);
}
''')

_register("ca_step",
          "Game of Life one step: 8-neighbor toroidal count, Conway birth/survival",
          "procedural", '''
void main() {
    vec2 texel = 1.0 / u_resolution;
    float c = texture(u_texture, v_uv).r;
    float n = 0.0;
    for (int y = -1; y <= 1; y++) {
        for (int x = -1; x <= 1; x++) {
            if (x == 0 && y == 0) continue;
            n += texture(u_texture, v_uv + vec2(float(x), float(y)) * texel).r;
        }
    }
    float alive = (n >= 2.5 && n <= 3.5) ? 1.0 : 0.0;  // survive on 2/3
    alive = (c < 0.5 && n > 2.5 && n < 3.5) ? 1.0 : alive;  // birth on 3
    float age = c > 0.5 ? texture(u_texture, v_uv).g + 1.0 : 0.0;
    f_color = vec4(alive, age, 0.0, 1.0);
}
''')


# ═══════════════════════════════════════════════════════════════════════════
#  P1.2 — RD family (Lotka-Volterra, FitzHugh-Nagumo, Turing, Colony)
#  Client-GPU sim twins of nodes 118-121, 133, 143/160, 168, 169.
#  All share the 5-pt toroidal Laplacian + RGBA-float ping-pong; only the
#  reaction term differs. CPU numpy nodes stay authoritative export.
# ═══════════════════════════════════════════════════════════════════════════

# Generic seeded RD state: U~1, V~0 with hashed seed blobs in V (node 118/119/133/169).
_register("rd_seed",
          "Generic RD seed: U~1, V~0 with hashed seed blobs (P1.2 RD twins)",
          "procedural", '''
void main() {
    float U = 1.0; float V = 0.0;
    for (int i = 0; i < 18; i++) {
        float fi = float(i);
        vec2 c = vec2(hash21(vec2(fi + 0.5, 1.37)), hash21(vec2(fi + 0.5, 7.91)));
        c = 0.05 + 0.90 * c;
        float d = distance(v_uv, c);
        V += 0.5 * exp(-(d * d) / 0.002);
    }
    V = clamp(V, 0.0, 1.0);
    f_color = vec4(U, V, 0.0, 1.0);
}
''')

_register("rd_display_u",
          "RD display: U activator -> grayscale (sqrt stretch)",
          "procedural", '''
void main() {
    float U = clamp(texture(u_texture, v_uv).r, 0.0, 1.0);
    f_color = vec4(vec3(sqrt(U)), 1.0);
}
''')

_register("rd_display_composite",
          "RD display: U in green, V in red (Lotka-Volterra prey/predator look)",
          "procedural", '''
void main() {
    vec4 s = texture(u_texture, v_uv);
    float U = clamp(s.r, 0.0, 1.0); float V = clamp(s.g, 0.0, 1.0);
    vec3 col = vec3(V, U * 0.9 + V * 0.1, U * 0.2);
    f_color = vec4(col, 1.0);
}
''')

# Lotka-Volterra 2-var (nodes 118, 119): du = a*u - b*u*v + Du*Lap(u);
# dv = d*u*v - g*v + Dv*Lap(v). p1=a, p2=b, p3=g, p4=d. Du/Dv fixed scale.
_register("lv_step",
          "Lotka-Volterra RD step (5-pt toroidal Laplacian)",
          "procedural", '''
void main() {
    vec2 texel = 1.0 / u_resolution;
    vec4 s = texture(u_texture, v_uv);
    float U = s.r, V = s.g;
    float lu = texture(u_texture, v_uv + vec2(-texel.x,0.0)).r
             + texture(u_texture, v_uv + vec2(texel.x,0.0)).r
             + texture(u_texture, v_uv + vec2(0.0,texel.y)).r
             + texture(u_texture, v_uv + vec2(0.0,-texel.y)).r - 4.0*U;
    float lv = texture(u_texture, v_uv + vec2(-texel.x,0.0)).g
             + texture(u_texture, v_uv + vec2(texel.x,0.0)).g
             + texture(u_texture, v_uv + vec2(0.0,texel.y)).g
             + texture(u_texture, v_uv + vec2(0.0,-texel.y)).g - 4.0*V;
    float a = u_params.x, b = u_params.y, g = u_params.z, d = u_params.w;
    float Du = 0.12, Dv = 0.30;
    float nU = U + 0.2 * (a*U - b*U*V + Du*lu);
    float nV = V + 0.2 * (d*U*V - g*V + Dv*lv);
    f_color = vec4(clamp(nU,0.0,1.0), clamp(nV,0.0,1.0), 0.0, 1.0);
}
''')

# FitzHugh-Nagumo (node 133): du = (u - u^3/3 - v)/e + Du*Lap(u);
# dv = e*(u + a - b*v) + Dv*Lap(v). p1=e, p2=a, p3=b, p4=Du.
_register("fhn_step",
          "FitzHugh-Nagumo step (5-pt toroidal Laplacian, excitable media)",
          "procedural", '''
void main() {
    vec2 texel = 1.0 / u_resolution;
    vec4 s = texture(u_texture, v_uv);
    float U = s.r, V = s.g;
    float lu = texture(u_texture, v_uv + vec2(-texel.x,0.0)).r
             + texture(u_texture, v_uv + vec2(texel.x,0.0)).r
             + texture(u_texture, v_uv + vec2(0.0,texel.y)).r
             + texture(u_texture, v_uv + vec2(0.0,-texel.y)).r - 4.0*U;
    float lv = texture(u_texture, v_uv + vec2(-texel.x,0.0)).g
             + texture(u_texture, v_uv + vec2(texel.x,0.0)).g
             + texture(u_texture, v_uv + vec2(0.0,texel.y)).g
             + texture(u_texture, v_uv + vec2(0.0,-texel.y)).g - 4.0*V;
    float e = max(u_params.x, 1e-3), a = u_params.y, b = u_params.z, Du = u_params.w;
    float Dv = 0.0;
    float nU = U + 0.08 * ((U - U*U*U/3.0 - V)/e + Du*lu);
    float nV = V + 0.08 * (e*(U + a - b*V) + Dv*lv);
    f_color = vec4(clamp(nU,-1.0,1.0), clamp(nV,-1.0,1.0), 0.0, 1.0);
}
''')

# Turing / Schnakenberg (node 169): ru = g*(a - u + u^2 v); rv = g*(b - u^2 v).
# Diffusion Du, Dv; p1=a, p2=b, p3=g, p4=Du.
_register("turing_step",
          "Schnakenberg/Turing RD step (5-pt toroidal Laplacian)",
          "procedural", '''
void main() {
    vec2 texel = 1.0 / u_resolution;
    vec4 s = texture(u_texture, v_uv);
    float U = s.r, V = s.g;
    float lu = texture(u_texture, v_uv + vec2(-texel.x,0.0)).r
             + texture(u_texture, v_uv + vec2(texel.x,0.0)).r
             + texture(u_texture, v_uv + vec2(0.0,texel.y)).r
             + texture(u_texture, v_uv + vec2(0.0,-texel.y)).r - 4.0*U;
    float lv = texture(u_texture, v_uv + vec2(-texel.x,0.0)).g
             + texture(u_texture, v_uv + vec2(texel.x,0.0)).g
             + texture(u_texture, v_uv + vec2(0.0,texel.y)).g
             + texture(u_texture, v_uv + vec2(0.0,-texel.y)).g - 4.0*V;
    float a = u_params.x, b = u_params.y, g = u_params.z, Du = u_params.w;
    float Dv = 0.5;
    float u2v = U*U*V;
    float nU = U + 0.02 * (g*(a - U + u2v) + Du*lu);
    float nV = V + 0.02 * (g*(b - u2v) + Dv*lv);
    f_color = vec4(clamp(nU,0.0,1.0), clamp(nV,0.0,1.0), 0.0, 1.0);
}
''')

# Allen-Cahn + Perona-Malik anisotropic diffusion (node 146 "AC + PM Diffusion" twin).
# Single scalar field c in [-1,1], packed in .r. p1=alpha (diffusion strength),
# p2=K (PM edge sensitivity), p3=bias (constant double-well shift), p4=dt.
# Live-preview approximation of the CPU sim: omits per-frame noise + the time
# ramp on bias (CPU authoritative for export). Resuses the 5-pt ping-pong template.
_register("acpm_seed",
          "AC+PM seed: signed +/-1 blobs in .r (node 146 twin)",
          "procedural", '''
void main() {
    float c = 0.0;
    for (int i = 0; i < 24; i++) {
        float fi = float(i);
        vec2 ctr = vec2(hash21(vec2(fi + 0.5, 1.37)), hash21(vec2(fi + 0.5, 7.91)));
        ctr = 0.05 + 0.90 * ctr;
        float r = 0.03 + 0.05 * hash21(vec2(fi + 2.3, 4.1));
        float d = distance(v_uv, ctr);
        float signv = (mod(fi, 2.0) < 0.5) ? 1.0 : -1.0;
        c += signv * exp(-(d * d) / (r * r));
    }
    c = clamp(c, -1.0, 1.0);
    f_color = vec4(c, 0.0, 0.0, 1.0);
}
''')

_register("acpm_step",
          "AC+PM step: Allen-Cahn reaction + Perona-Malik anisotropic diffusion (5-pt)",
          "procedural", '''
void main() {
    vec2 texel = 1.0 / u_resolution;
    vec4 s = texture(u_texture, v_uv);
    float c = s.r;
    float cl = texture(u_texture, v_uv + vec2(-texel.x,0.0)).r;
    float cr = texture(u_texture, v_uv + vec2(texel.x,0.0)).r;
    float cd = texture(u_texture, v_uv + vec2(0.0,texel.y)).r;
    float cu = texture(u_texture, v_uv + vec2(0.0,-texel.y)).r;
    // Perona-Malik anisotropic diffusion (4-neighbour, edge-preserving)
    float K2 = max(u_params.y * u_params.y, 1e-4);
    float gx = (cr - c) / (1.0 + (cr - c) * (cr - c) / K2);
    float gy = (cu - c) / (1.0 + (cu - c) * (cu - c) / K2);
    float gxl = (c - cl) / (1.0 + (c - cl) * (c - cl) / K2);
    float gyl = (c - cd) / (1.0 + (c - cd) * (c - cd) / K2);
    float diff = (gx - gxl) + (gy - gyl);
    // Allen-Cahn double-well reaction + constant bias
    float ac = c - c * c * c + u_params.z;
    float dt = u_params.w;
    float alpha = u_params.x;
    float nc = c + dt * (ac + alpha * diff);
    f_color = vec4(clamp(nc, -1.5, 1.5), 0.0, 0.0, 1.0);
}
''')

_register("acpm_display",
          "AC+PM display: map signed field .r to grayscale",
          "procedural", '''
void main() {
    float c = clamp(texture(u_texture, v_uv).r, -1.0, 1.0);
    float g = c * 0.5 + 0.5;
    f_color = vec4(vec3(g), 1.0);
}
''')

# 3-species Lotka-Volterra (node 120): U,V,W in r,g,b — cyclic predation.
# p1,p2,p3,p4 = interaction strengths (live preview approx; CPU authoritative).
_register("lv3_seed",
          "3-species LV seed: U~1, V~0.5, W~0.5 with hashed blobs (node 120 twin)",
          "procedural", '''
void main() {
    float U = 1.0, V = 0.5, W = 0.5;
    for (int i = 0; i < 14; i++) {
        float fi = float(i);
        vec2 c = vec2(hash21(vec2(fi+0.5,1.37)), hash21(vec2(fi+0.5,7.91)));
        c = 0.05 + 0.90*c; float d = distance(v_uv, c);
        float blob = exp(-(d*d)/0.002);
        U -= 0.4*blob; V += 0.5*blob; W += 0.3*blob;
    }
    f_color = vec4(clamp(U,0.0,1.0), clamp(V,0.0,1.0), clamp(W,0.0,1.0), 1.0);
}
''')

_register("lv3_step",
          "3-species Lotka-Volterra step (5-pt toroidal Laplacian on 3 channels)",
          "procedural", '''
void main() {
    vec2 texel = 1.0 / u_resolution;
    vec4 s = texture(u_texture, v_uv);
    float U = s.r, V = s.g, W = s.b;
    float lu = texture(u_texture, v_uv + vec2(-texel.x,0.0)).r
             + texture(u_texture, v_uv + vec2(texel.x,0.0)).r
             + texture(u_texture, v_uv + vec2(0.0,texel.y)).r
             + texture(u_texture, v_uv + vec2(0.0,-texel.y)).r - 4.0*U;
    float lv = texture(u_texture, v_uv + vec2(-texel.x,0.0)).g
             + texture(u_texture, v_uv + vec2(texel.x,0.0)).g
             + texture(u_texture, v_uv + vec2(0.0,texel.y)).g
             + texture(u_texture, v_uv + vec2(0.0,-texel.y)).g - 4.0*V;
    float lw = texture(u_texture, v_uv + vec2(-texel.x,0.0)).b
             + texture(u_texture, v_uv + vec2(texel.x,0.0)).b
             + texture(u_texture, v_uv + vec2(0.0,texel.y)).b
             + texture(u_texture, v_uv + vec2(0.0,-texel.y)).b - 4.0*W;
    float k1 = u_params.x, k2 = u_params.y, k3 = u_params.z, k4 = u_params.w;
    float nU = U + 0.15 * (U - k1*U*V + 0.08*lu);
    float nV = V + 0.15 * (k2*U*V - k3*V*W + 0.08*lv);
    float nW = W + 0.15 * (k4*V*W - W + 0.08*lw);
    f_color = vec4(clamp(nU,0.0,1.0), clamp(nV,0.0,1.0), clamp(nW,0.0,1.0), 1.0);
}
''')

_register("lv3_display",
          "3-species LV display: U green, V red, W blue (cyclic food web)",
          "procedural", '''
void main() {
    vec4 s = texture(u_texture, v_uv);
    vec3 col = vec3(clamp(s.g,0.0,1.0), clamp(s.r,0.0,1.0), clamp(s.b,0.0,1.0));
    f_color = vec4(col, 1.0);
}
''')

# Bacterial colony (nodes 143, 160): nutrient N (.r), colony C (.g).
# growth of C where N present; consumption of N by C; diffusion of N.
# p1=growth, p2=diff_c, p3=consumption, p4=death.
_register("colony_seed",
          "Bacterial colony seed: nutrient full, colony disc at center (nodes 143/160 twin)",
          "procedural", '''
void main() {
    float N = 1.0;
    float d = distance(v_uv, vec2(0.5));
    float C = d < 0.06 ? 1.0 : 0.0;
    f_color = vec4(N, C, 0.0, 1.0);
}
''')

_register("colony_step",
          "Bacterial colony step: N nutrient, C colony (5-pt toroidal Laplacian)",
          "procedural", '''
void main() {
    vec2 texel = 1.0 / u_resolution;
    vec4 s = texture(u_texture, v_uv);
    float N = s.r, C = s.g;
    float ln = texture(u_texture, v_uv + vec2(-texel.x,0.0)).r
             + texture(u_texture, v_uv + vec2(texel.x,0.0)).r
             + texture(u_texture, v_uv + vec2(0.0,texel.y)).r
             + texture(u_texture, v_uv + vec2(0.0,-texel.y)).r - 4.0*N;
    float lc = texture(u_texture, v_uv + vec2(-texel.x,0.0)).g
             + texture(u_texture, v_uv + vec2(texel.x,0.0)).g
             + texture(u_texture, v_uv + vec2(0.0,texel.y)).g
             + texture(u_texture, v_uv + vec2(0.0,-texel.y)).g - 4.0*C;
    float growth = u_params.x, diff_c = u_params.y, cons = u_params.z, death = u_params.w;
    float dC = growth * C * N - cons * C + diff_c * lc;
    float dN = -cons * C * N + 0.05 * ln;   // nutrient consumed + diffuses in
    float nC = clamp(C + 0.1 * dC, 0.0, 1.0);
    float nN = clamp(N + 0.1 * dN, 0.0, 1.0);
    f_color = vec4(nN, nC, 0.0, 1.0);
}
''')

_register("colony_display",
          "Bacterial colony display: colony white on dark nutrient field",
          "procedural", '''
void main() {
    vec4 s = texture(u_texture, v_uv);
    float C = clamp(s.g, 0.0, 1.0);
    float N = clamp(s.r, 0.0, 1.0);
    vec3 col = mix(vec3(0.05,0.07,0.10), vec3(0.9,0.95,0.85), C);
    col *= (0.4 + 0.6*N);
    f_color = vec4(col, 1.0);
}
''')
_register("ca_display",
          "Game of Life display: alive=white, age tints toward warm",
          "procedural", '''
void main() {
    vec4 s = texture(u_texture, v_uv);
    float alive = s.r;
    float age = s.g;
    vec3 col = mix(vec3(0.02, 0.02, 0.05), vec3(0.9, 0.95, 1.0), alive);
    col = mix(col, vec3(1.0, 0.6, 0.2), alive * clamp(age / 12.0, 0.0, 1.0));
    f_color = vec4(col, 1.0);
}
''')


# ── Kuramoto coupled-oscillator phase field (client-GPU sim of node 999) ─────
# Client-GPU sim twin of the Arch-A Kuramoto node. The phase field is NOT a
# velocity advection (every flow node does that) — it is self-organized
# synchronization: each pixel is an oscillator whose phase θ is nudged toward
# its neighbours AND toward the global mean phase. State packs:
#   .r = phase θ (wrapped to [0, 2π])
#   .g = natural frequency Ω (frozen at seed — per-oscillator intrinsic rate)
#   .b = RNG carry (see pitfall #6b: renderGpuSim gives step NO u_time, so we
#        carry a per-cell random in state instead of hashing the clock).
# CPU node stays authoritative for export (two-tier precision).
_register("kuramoto_seed",
          "Kuramoto seed: hashed phase + spatially-structured natural frequency Ω (node 999 twin)",
          "procedural", '''
void main() {
    float h1 = hash21(floor(v_uv * u_resolution * 0.37));
    float h2 = hash21(floor(v_uv * u_resolution * 0.91) + 5.3);
    // phase: scattered so the field starts incoherent
    float theta = h1 * 6.2831853;
    // Ω: smooth spatial gradient + mild noise → travelling spiral waves
    vec2 c = v_uv - 0.5;
    float omega = (c.x + c.y) * u_params.z * 1.4 + (h2 - 0.5) * u_params.z * 0.4;
    f_color = vec4(theta, omega, h2, 1.0);
}
''')

_register("kuramoto_step",
          "Kuramoto one Euler step: θ += dt·(Ω + K·Σsin(θⱼ−θ) + gK·R·sin(Ψ−θ))",
          "procedural", '''
void main() {
    vec2 texel = 1.0 / u_resolution;
    vec4 s = texture(u_texture, v_uv);
    float theta = s.r;
    float omega = s.g;
    float rng = s.b;
    // Nearest-neighbour coupling term Σⱼ sin(θⱼ − θᵢ) (toroidal wrap).
    float cl = texture(u_texture, v_uv + vec2(-texel.x, 0.0)).r;
    float cr = texture(u_texture, v_uv + vec2( texel.x, 0.0)).r;
    float cu = texture(u_texture, v_uv + vec2(0.0,  texel.y)).r;
    float cd = texture(u_texture, v_uv + vec2(0.0, -texel.y)).r;
    float coupling = sin(cl - theta) + sin(cr - theta)
                   + sin(cu - theta) + sin(cd - theta);
    // Global mean-field term: approximate R·sin(Ψ−θᵢ) with a frame-stable
    // proxy using the local-averaged phase (keeps the twin lively without a
    // full-canvas reduction in GLSL). Ψ ≈ neighbourhood mean phase.
    float neigh = (cl + cr + cu + cd) * 0.25;
    float mean_sin = sin(neigh - theta);
    float K  = u_params.x;            // local coupling
    float gK = u_params.y;            // global coupling
    float dt = max(u_params.w, 0.02);
    // local coherence ~ |coupling|/4 in [0,1] stands in for R so the global
    // term stays bounded and the pattern still forms chimeras.
    float Rloc = clamp(abs(coupling) / 4.0, 0.0, 1.0);
    float dtheta = omega + K * coupling + gK * Rloc * mean_sin;
    float ntheta = mod(theta + dt * dtheta, 6.2831853);
    // advance RNG carry
    float nrng = fract(rng * 1.4567 + 0.137);
    f_color = vec4(ntheta, omega, nrng, 1.0);
}
''')

_register("kuramoto_display",
          "Kuramoto display: phase θ → IQ rainbow palette (matches _render_phase)",
          "procedural", '''
void main() {
    float theta = texture(u_texture, v_uv).r;
    float t = theta / 6.2831853;
    vec3 col = 0.5 + 0.5 * cos(6.2831853 * (t + vec3(0.0, 0.3333333, 0.6666667)));
    f_color = vec4(col, 1.0);
}
''')


# ═══════════════════════════════════════════════════════════════════════════
# ── Node 106: Dielectric Breakdown Model (GPU sim twin) ─────────────────────
# Client-GPU sim twin of the Arch-A dielectric-breakdown node (DBM, Niemeyer
# 1984): a Jacobi-relaxed Laplace potential field with stochastic growth
# probability proportional to |grad(phi)|^eta at the tree frontier. State packs
# the potential phi in .r, occupancy in .g (-1 far-field boundary / 0 empty /
# 1 tree), and temperature (brightness) in .b. The CPU numpy node stays
# authoritative for export (two-tier precision).
_register("dbm_seed",
          "Dielectric Breakdown seed: center electrode + fixed far-field boundary (node 106 twin)",
          "procedural", '''
void main() {
    vec2 res = u_resolution;
    vec2 uv = v_uv;
    float occ = 0.0;
    float phi = 0.0;
    float temp = 0.0;
    // Single seed electrode at center (the twin uses n_seeds = 1).
    if (distance(uv, vec2(0.5)) < 1.5 / res.x) {
        occ = 1.0; temp = 1.0; phi = 1.0;
    }
    // Fixed Dirichlet far-field: potential 0, never grows.
    float m = 2.0 / res.x;
    if (uv.x < m || uv.x > 1.0 - m || uv.y < m || uv.y > 1.0 - m) {
        occ = -1.0; phi = 0.0; temp = 0.0;
    }
    f_color = vec4(phi, occ, temp, 1.0);
}
''')

_register("dbm_step",
          "Dielectric Breakdown step: relax Laplace potential, grow frontier proportional to |grad(phi)|^eta (node 106 twin)",
          "procedural", '''
float dbmNbPhi(vec2 off) {
    vec4 n = texture(u_texture, v_uv + off);
    if (n.g > 0.5) return 1.0;    // tree electrode
    if (n.g < -0.5) return 0.0;   // far-field boundary
    return n.r;
}

void main() {
    vec2 texel = 1.0 / u_resolution;
    vec4 s = texture(u_texture, v_uv);
    float occ = s.g;
    float temp = s.b;

    float eta  = u_params.x;
    float grow = u_params.y;   // growth_rate
    float cool = u_params.z;   // cool_rate
    float diel = u_params.w;   // dielectric strength

    // Far-field boundary: frozen.
    if (occ < -0.5) { f_color = vec4(0.0, -1.0, 0.0, 1.0); return; }
    // Tree electrode: hold phi=1, cool the temperature each step.
    if (occ > 0.5)  { f_color = vec4(1.0, 1.0, temp * cool, 1.0); return; }

    // One Jacobi relaxation sweep of the harmonic potential.
    float pl = dbmNbPhi(vec2(-texel.x, 0.0));
    float pr = dbmNbPhi(vec2( texel.x, 0.0));
    float pd = dbmNbPhi(vec2(0.0, -texel.y));
    float pu = dbmNbPhi(vec2(0.0,  texel.y));
    float phiNew = 0.25 * (pl + pr + pu + pd);

    // Gradient magnitude of the potential at this cell (tip vs flat front).
    float gx = pr - pl;
    float gy = pu - pd;
    float grad = length(vec2(gx, gy)) * 0.5;

    bool frontier = (pl > 0.5 || pr > 0.5 || pu > 0.5 || pd > 0.5);

    float newOcc = 0.0;
    float newTemp = 0.0;
    if (frontier) {
        // Per-cell dielectric weakness (stable hash) — weak spots grow easier.
        float weak = hash21(floor(v_uv * u_resolution) + 3.17);
        float dieMul = mix(1.0, weak, diel);
        float w = pow(max(grad, 0.0), eta);
        float prob = clamp(w * grow * 0.06 * dieMul, 0.0, 1.0);
        float rng = hash21(floor(v_uv * u_resolution) + 0.5 + phiNew * 53.0);
        if (rng < prob) { newOcc = 1.0; newTemp = 1.0; }
    }

    f_color = vec4(phiNew, newOcc, newTemp, 1.0);
}
''')

_register("dbm_display",
          "Dielectric Breakdown display: hot tips bright blue-white, cooled trunk dimmer (node 106 twin)",
          "procedural", '''
void main() {
    vec4 s = texture(u_texture, v_uv);
    float occ = s.g;
    float temp = s.b;
    if (occ < -0.5) { f_color = vec4(0.0, 0.0, 0.0, 1.0); return; }   // far-field
    if (occ < 0.5)  { f_color = vec4(0.02, 0.03, 0.06, 1.0); return; } // empty space
    float t = clamp(temp, 0.0, 1.0);
    vec3 hot  = vec3(0.75, 0.85, 1.0);
    vec3 coolc = vec3(0.15, 0.35, 0.9);
    f_color = vec4(mix(coolc, hot, t), 1.0);
}
''')

#  P1.4 — Discrete cellular automata / statistical-mechanics twins
#  Client-GPU sim twins of nodes 87 (Cyclic CA), 96 (Forest Fire), 93 (Ising).
#  All use RGBA-float ping-pong: a single-channel integer CA state in .r,
#  an auxiliary channel for age/aux data in .g, and a per-cell RNG carry in .b
#  (advanced each step via hash21 so the live sim does not require u_time).
#  CPU numpy nodes stay authoritative for export (two-tier precision).
# ═══════════════════════════════════════════════════════════════════════════

# ── Node 87: Cyclic (Rock-Paper-Scissors) CA ──
# State in .r ∈ [0,1) encodes state index = floor(.r * n_states); n_states from
# u_params.x (3-8). .b carries the per-cell RNG seed, advanced each step.
_register("cyclic_ca_seed",
          "Cyclic CA seed: hashed random state in [0,n_states), RNG carry in .b (node 87 twin)",
          "procedural", '''
void main() {
    float ns = clamp(floor(u_params.x + 0.5), 3.0, 8.0);
    float h = hash21(floor(v_uv * u_resolution * 0.5));
    float s = floor(h * ns) / ns;   // quantize into n_states buckets
    float rng = hash21(v_uv * u_resolution + 7.13);
    f_color = vec4(s, 0.0, rng, 1.0);
}
''')

_register("cyclic_ca_step",
          "Cyclic CA one step: convert to predator state when >= threshold neighbours match (node 87 twin)",
          "procedural", '''
void main() {
    vec2 texel = 1.0 / u_resolution;
    vec4 s = texture(u_texture, v_uv);
    float ns = clamp(floor(u_params.x + 0.5), 3.0, 8.0);
    float thr = clamp(floor(u_params.y + 0.5), 1.0, 5.0);
    float my = floor(s.r * ns + 0.5);
    float pred = mod(my + 1.0, ns);          // predator state index
    float predF = (pred + 0.5) / ns;         // predator state in [0,1)
    float cnt = 0.0;
    for (int y = -1; y <= 1; y++) {
        for (int x = -1; x <= 1; x++) {
            if (x == 0 && y == 0) continue;
            float sn = texture(u_texture, v_uv + vec2(float(x), float(y)) * texel).r;
            float nidx = floor(sn * ns + 0.5);
            if (abs(nidx - pred) < 0.5) cnt += 1.0;
        }
    }
    float alive_pred = cnt >= thr ? 1.0 : 0.0;
    float next = alive_pred > 0.5 ? predF : s.r;
    // advance per-cell RNG carry
    float rng = fract(s.b * 1.4567 + 0.137);
    f_color = vec4(next, 0.0, rng, 1.0);
}
''')

_register("cyclic_ca_display",
          "Cyclic CA display: 8-state cyclic palette (red/green/blue/gold/cyan/magenta/orange/silver)",
          "procedural", '''
void main() {
    vec4 s = texture(u_texture, v_uv);
    float ns = clamp(floor(u_params.x + 0.5), 3.0, 8.0);
    float st = floor(s.r * ns + 0.5);
    // 8 distinct hues around the wheel
    float a = st / ns * 6.2831853;
    vec3 col = 0.5 + 0.5 * cos(a + vec3(0.0, 2.094, 4.188));
    col = mix(vec3(0.05, 0.05, 0.07), col, 0.9);
    f_color = vec4(col, 1.0);
}
''')

# ── Node 96: Drossel-Schwabl Forest Fire ──
# 3-state CA: .r encodes state (0 empty, 1 tree, 2 burning); .g = fire_age (0-3);
# .b = per-cell RNG carry. p=growth in u_params.x, f=lightning in u_params.y.
_register("forest_fire_seed",
          "Forest Fire seed: random trees at initial fraction, RNG carry in .b (node 96 twin)",
          "procedural", '''
void main() {
    float init = clamp(u_params.z, 0.1, 0.9);
    float h = hash21(floor(v_uv * u_resolution * 0.5));
    float state = h < init ? 1.0 : 0.0;   // 1 = tree
    float rng = hash21(v_uv * u_resolution + 3.71);
    f_color = vec4(state, 0.0, rng, 1.0);
}
''')

_register("forest_fire_step",
          "Forest Fire one step: growth, neighbour/lightning ignition, fire aging (node 96 twin)",
          "procedural", '''
void main() {
    vec2 texel = 1.0 / u_resolution;
    vec4 s = texture(u_texture, v_uv);
    float p = clamp(u_params.x, 0.001, 0.05);
    float f = clamp(u_params.y, 0.00001, 0.001);
    float state = s.r;
    float age = s.g;
    float rng = fract(s.b * 1.731 + 0.211);

    // count burning neighbours (Moore)
    float burn = 0.0;
    for (int y = -1; y <= 1; y++) {
        for (int x = -1; x <= 1; x++) {
            if (x == 0 && y == 0) continue;
            float sn = texture(u_texture, v_uv + vec2(float(x), float(y)) * texel).r;
            if (sn > 1.5) burn += 1.0;   // state == 2 (burning)
        }
    }

    float next_state = state;
    float next_age = age;

    if (state > 1.5) {
        // currently burning: age it down; age 0 -> empty
        if (age <= 0.5) { next_state = 0.0; next_age = 0.0; }
        else { next_age = age - 1.0; }
    } else if (state > 0.5) {
        // tree: ignite if neighbour burning or lightning
        float lightning = rng < f ? 1.0 : 0.0;
        if (burn > 0.5 || lightning > 0.5) { next_state = 2.0; next_age = 3.0; }
    } else {
        // empty: grow a tree
        if (rng < p) { next_state = 1.0; }
    }
    f_color = vec4(next_state, next_age, rng, 1.0);
}
''')

_register("forest_fire_display",
          "Forest Fire display: earth/tree/fire_age colormap (node 96 twin)",
          "procedural", '''
void main() {
    vec4 s = texture(u_texture, v_uv);
    float state = s.r;
    float age = s.g;
    vec3 col = vec3(0.16, 0.10, 0.06);          // dark brown earth
    if (state > 0.5 && state < 1.5) {
        col = vec3(0.12, 0.55, 0.20);           // green tree
    } else if (state > 1.5) {
        // fire_age 1-3: dark red -> orange -> bright orange
        vec3 fire = mix(vec3(0.63, 0.12, 0.04), vec3(1.0, 0.63, 0.08),
                        clamp(age / 3.0, 0.0, 1.0));
        col = fire;
    }
    f_color = vec4(col, 1.0);
}
''')

# ── Node 93: 2D Ising Model (Glauber live approximation of Wolff) ──
# Spins σ=±1 packed as .r in {0,1} (0=-1, 1=+1) to survive fp32; .b = RNG carry.
# Coupling J (period 1) in u_params.x, T/Tc in u_params.y (Glauber p below Tc).
_register("ising_seed",
          "Ising seed: hashed random spin config, RNG carry in .b (node 93 twin)",
          "procedural", '''
void main() {
    float h = hash21(floor(v_uv * u_resolution * 0.5));
    float spin = h < 0.5 ? 0.0 : 1.0;   // 0 = down, 1 = up
    float rng = hash21(v_uv * u_resolution + 11.7);
    f_color = vec4(spin, 0.0, rng, 1.0);
}
''')

_register("ising_step",
          "Ising one Glauber step: flip spin by Metropolis-like prob from 4-neighbour sum (node 93 twin)",
          "procedural", '''
void main() {
    vec2 texel = 1.0 / u_resolution;
    vec4 s = texture(u_texture, v_uv);
    float J = clamp(u_params.x, 0.5, 2.0);
    float T = clamp(u_params.y, 0.5, 3.0);
    float beta = 1.0 / (T * 2.2691853);     // Tc(Tc-scaled) = 2.269*J, J folded out
    float spin = s.r > 0.5 ? 1.0 : -1.0;
    float rng = fract(s.b * 1.824 + 0.317);

    // 4-neighbour von Neumann sum (+1/-1 each)
    float nsum = 0.0;
    nsum += (texture(u_texture, v_uv + vec2(-texel.x, 0.0)).r > 0.5 ? 1.0 : -1.0);
    nsum += (texture(u_texture, v_uv + vec2( texel.x, 0.0)).r > 0.5 ? 1.0 : -1.0);
    nsum += (texture(u_texture, v_uv + vec2(0.0, texel.y)).r > 0.5 ? 1.0 : -1.0);
    nsum += (texture(u_texture, v_uv + vec2(0.0,-texel.y)).r > 0.5 ? 1.0 : -1.0);

    float dE = 2.0 * J * spin * nsum;        // energy cost of flipping
    float p_flip = dE > 0.0 ? exp(-beta * dE) : 1.0;
    float nspin = (rng < p_flip) ? -spin : spin;
    f_color = vec4(nspin > 0.5 ? 1.0 : 0.0, 0.0, rng, 1.0);
}
''')

_register("ising_display",
          "Ising display: blue-white-red diverging map of spin (+1 white-ish, -1 blue)",
          "procedural", '''
void main() {
    vec4 s = texture(u_texture, v_uv);
    float spin = s.r > 0.5 ? 1.0 : -1.0;
    // diverging: -1 -> blue, +1 -> red, with soft mid-grey
    vec3 col = mix(vec3(0.20, 0.35, 0.85), vec3(0.90, 0.30, 0.25), (spin + 1.0) * 0.5);
    f_color = vec4(col, 1.0);
}
''')


# ═══════════════════════════════════════════════════════════════════════════
#  P1.4b — Spatial Prisoner's Dilemma (nodes 153 / 154 twins)
#  #153 is a BINARY strategy lattice (cooperate/defect) → same family as the
#  Ising twin: per-cell RNG carry in .b, probabilistic Fermi imitation update.
#  #154 is the CONTINUOUS replicator PDE (s ∈ [0,1]) → same family as the CML
#  / wave twins: packed R=raw field, G=EMA trail, smoothed by PDE + diffusion.
#  CPU numpy nodes stay authoritative (two-tier precision contract).
# ═══════════════════════════════════════════════════════════════════════════

# ── Node 153: Spatial Prisoner's Dilemma — binary lattice ──
_register("spd125_seed",
          "SPD #153 seed: hashed random cooperate/defect lattice, RNG carry in .b",
          "procedural", '''
void main() {
    float h = hash21(floor(v_uv * u_resolution * 0.5));
    float strat = h < 0.5 ? 0.0 : 1.0;   // 0=defect, 1=coop
    float rng = hash21(v_uv * u_resolution + 19.3);
    f_color = vec4(strat, 0.0, rng, 1.0);  // R=strat, G=payoff, B=rng
}
''')

_register("spd125_step",
          "SPD #153 one Fermi-imitation step: probabilistic strategy switch from neighbors (snowdrift matrix)",
          "procedural", '''
void main() {
    vec2 texel = 1.0 / u_resolution;
    vec4 s = texture(u_texture, v_uv);
    float my = s.r;
    float rng = fract(s.b * 1.4567 + 0.137);

    // Moore neighborhood (8 cells)
    float n00 = texture(u_texture, v_uv + vec2(-texel.x,-texel.y)).r;
    float n01 = texture(u_texture, v_uv + vec2(0.0,-texel.y)).r;
    float n02 = texture(u_texture, v_uv + vec2(texel.x,-texel.y)).r;
    float n10 = texture(u_texture, v_uv + vec2(-texel.x,0.0)).r;
    float n12 = texture(u_texture, v_uv + vec2(texel.x,0.0)).r;
    float n20 = texture(u_texture, v_uv + vec2(-texel.x,texel.y)).r;
    float n21 = texture(u_texture, v_uv + vec2(0.0,texel.y)).r;
    float n22 = texture(u_texture, v_uv + vec2(texel.x,texel.y)).r;
    float nc = n00+n01+n02+n10+n12+n20+n21+n22;  // #coop neighbors
    float nn = 8.0 - nc;                         // #defect neighbors

    float T = clamp(u_params.x, 1.0, 2.0);     // temptation payoff
    float S = clamp(u_params.y, -1.0, 1.0);    // sucker payoff
    float K = clamp(u_params.z, 0.01, 2.0);    // Fermi stochasticity

    // Snowdrift payoffs: coop reward R=1.0, defect gets T vs coop / S vs defect
    float my_pay  = (my < 0.5) ? (nc * 1.0 + nn * S) : (nc * T);
    float r2 = fract(rng * 2.137 + 0.71);
    float nbr = step(0.5, r2);                  // a random neighbor strategy
    float nbr_pay = (nbr < 0.5) ? (nc * 1.0 + nn * S) : (nc * T);

    float prob = 1.0 / (1.0 + exp((my_pay - nbr_pay) / K));
    float r3 = fract(rng * 3.11 + 0.43);
    float nstrat = (r3 < prob) ? nbr : my;

    // rough normalized payoff for display brightness
    float pay = clamp(my_pay / (8.0 * max(T, 1.0)), 0.0, 1.0);
    f_color = vec4(nstrat, pay, rng, 1.0);
}
''')

_register("spd125_display",
          "SPD #153 display: diverging amber(defect)→blue(coop) by strategy, brightness by payoff",
          "procedural", '''
void main() {
    vec4 s = texture(u_texture, v_uv);
    float strat = s.r;
    vec3 defect_col = vec3(0.86, 0.39, 0.16);  // amber
    vec3 coop_col   = vec3(0.235, 0.55, 0.86); // blue
    vec3 col = mix(defect_col, coop_col, strat);
    float bright = 0.65 + 0.35 * clamp(s.g, 0.0, 1.0);
    f_color = vec4(col * bright, 1.0);
}
''')

# ── Node 154: Continuous Spatial PD (replicator dynamics) — PDE field ──
_register("spd154_seed",
          "CSPD #154 seed: hashed continuous strategy field s∈[0,1] (R=raw, G=trail)",
          "procedural", '''
void main() {
    float x = hash21(v_uv * u_resolution + 0.321);
    f_color = vec4(x * 0.6, x * 0.6, 0.0, 1.0);  // R=s, G=accum trail
}
''')

_register("spd154_step",
          "CSPD #154 one Euler step: replicator reaction + diffusion + mutation drift + noise",
          "procedural", '''
void main() {
    vec2 texel = 1.0 / u_resolution;
    vec4 s = texture(u_texture, v_uv);
    float s0 = s.r;
    float accum = s.g;
    float rng = fract(s.b * 1.245 + 0.371);

    float T = u_params.x;   // temptation
    float R = u_params.y;   // reward (mutual coop)
    float S = u_params.z;   // sucker
    float P = u_params.w;   // punishment

    // 8-neighbour sum for replicator payoff fields
    float sum_s = 0.0;
    sum_s += texture(u_texture, v_uv + vec2(-texel.x,-texel.y)).r;
    sum_s += texture(u_texture, v_uv + vec2(0.0,-texel.y)).r;
    sum_s += texture(u_texture, v_uv + vec2(texel.x,-texel.y)).r;
    sum_s += texture(u_texture, v_uv + vec2(-texel.x,0.0)).r;
    sum_s += texture(u_texture, v_uv + vec2(texel.x,0.0)).r;
    sum_s += texture(u_texture, v_uv + vec2(-texel.x,texel.y)).r;
    sum_s += texture(u_texture, v_uv + vec2(0.0,texel.y)).r;
    sum_s += texture(u_texture, v_uv + vec2(texel.x,texel.y)).r;

    float coop_sum = R * sum_s + S * (8.0 - sum_s);
    float def_sum  = T * sum_s + P * (8.0 - sum_s);
    float replicator = s0 * (1.0 - s0) * (coop_sum - def_sum);

    float mutation = 0.025;
    float mutation_drift = mutation * (0.5 - s0);

    // 5-point Laplacian for diffusion
    float lap = texture(u_texture, v_uv + vec2(-texel.x,0.0)).r
              + texture(u_texture, v_uv + vec2(texel.x,0.0)).r
              + texture(u_texture, v_uv + vec2(0.0,texel.y)).r
              + texture(u_texture, v_uv + vec2(0.0,-texel.y)).r
              - 4.0 * s0;
    float D = 0.12;
    float diffusion = D * lap;

    // cheap pseudo-Gaussian noise (sum of 3 uniforms, variance-normalized)
    float gnoise = (fract(rng * 1.7 + 0.13) + fract(rng * 2.3 + 0.57)
                    + fract(rng * 3.1 + 0.91) - 1.5) * 0.5773502;
    float noise_amp = 0.008;
    float DT = 0.2;
    float ds = DT * (replicator + mutation_drift + diffusion)
               + noise_amp * gnoise * sqrt(DT);
    float sn = clamp(s0 + ds, 0.0, 1.0);

    float decay = 0.9;
    float an = decay * accum + (1.0 - decay) * sn;
    f_color = vec4(sn, clamp(an, 0.0, 1.0), rng, 1.0);
}
''')

_register("spd154_display",
          "CSPD #154 display: grayscale cooperation-probability field (matches CPU render)",
          "procedural", '''
void main() {
    float v = clamp(texture(u_texture, v_uv).g, 0.0, 1.0);
    f_color = vec4(vec3(v), 1.0);
}
''')


# ═══════════════════════════════════════════════════════════════════════════
#  P1.3 — Wave-equation family (client-GPU sim twins of nodes 100, 144, 166)
#  All three are scalar displacement u + velocity v leapfrog field systems
#  (plus a pump/drive phase accumulator). State packs R=u, G=v, B=pump_phase
#  in RGBA-float ping-pong, stepped `substeps` times per rendered frame. The
#  CPU numpy nodes stay the authoritative export (two-tier precision).
# ═══════════════════════════════════════════════════════════════════════════

# ── Node 100: Wave Equation ── -------------------------------------------------
# 2D wave equation u_tt = c^2 laplacian(u) via velocity-Verlet on (u, v).
# p1=wave_speed, p2=damping, p3=source_frequency, p4=source_amplitude.
_register("wave_eq_seed",
          "Wave Equation seed: small hashed noise displacement, zero velocity (node 100 twin)",
          "procedural", '''
void main() {
    float n = noise(v_uv * 9.0) * 0.5 + noise(v_uv * 23.0) * 0.25;
    f_color = vec4((n - 0.375) * 0.6, 0.0, 0.0, 1.0);  // R=u (small), G=v=0, B=phase=0
}
''')

_register("wave_eq_step",
          "Wave Equation one step (velocity-Verlet, 5-pt toroidal Laplacian)",
          "procedural", '''
void main() {
    vec2 texel = 1.0 / u_resolution;
    vec4 s = texture(u_texture, v_uv);
    float u = s.r, v = s.g, phase = s.b;
    float lu = texture(u_texture, v_uv + vec2(-texel.x, 0.0)).r
             + texture(u_texture, v_uv + vec2( texel.x, 0.0)).r
             + texture(u_texture, v_uv + vec2(0.0, texel.y)).r
             + texture(u_texture, v_uv + vec2(0.0,-texel.y)).r - 4.0 * u;
    float c = clamp(u_params.x, 0.3, 2.5);
    float c2 = 0.20 * c * c;             // stable ( < 0.5 )
    float damp = clamp(u_params.y, 0.90, 1.0);
    float freq = clamp(u_params.z, 0.2, 8.0);
    float amp = clamp(u_params.w, 0.2, 5.0);

    // Source injection: two detuned point sources (mirrors the CPU node).
    float dphi = 6.2831853 * freq;
    phase = mod(phase + dphi, 6.2831853);
    float src = amp * sin(phase);
    vec2 p0 = vec2(0.33, 0.5), p1v = vec2(0.66, 0.5);
    float ds0 = distance(v_uv, p0), ds1 = distance(v_uv, p1v);
    float src_inj = src * (exp(-(ds0*ds0)/0.0008) + exp(-(ds1*ds1)/0.0008) * 0.85);

    float vn = (v + c2 * lu) * damp;       // dv = c2*lap ; velocity damping
    float un = u + vn + src_inj;           // du = v
    f_color = vec4(clamp(un, -8.0, 8.0), clamp(vn, -8.0, 8.0), phase, 1.0);
}
''')

_register("wave_eq_display",
          "Wave Equation display: bipolar displacement -> plasma-like palette",
          "procedural", '''
void main() {
    float u = texture(u_texture, v_uv).r;
    float t = clamp(u * 0.5 + 0.5, 0.0, 1.0);
    vec3 col = mix(vec3(0.10, 0.10, 0.45), vec3(0.95, 0.40, 0.10), t);
    col = mix(col, vec3(1.0, 1.0, 0.65), smoothstep(0.62, 1.0, t));
    f_color = vec4(col, 1.0);
}
''')

# ── Node 499: Sine-Gordon Equation ── ------------------------------------------
# 2D Sine-Gordon u_tt = c^2 lap(u) - G*sin(u) + A*drive. Same leapfrog (u, v)
# fields as the Wave Equation (node 100) with the addition of the -G*sin(u)
# restoring term that produces kink/antikink solitons and breathers.
# p1=wave_speed (c), p2=damping, p3=coupling (G), p4=drive_amplitude (A).
# c2 = min(0.20*c*c, 0.45) (CFL-safe); S = 0.20*G; drive = A*0.05*(sin6.28*3x+sin6.28*3y).
_register("sine_gordon_seed",
          "Sine-Gordon seed: kink-antikink initial displacement, zero velocity (node 499 twin)",
          "procedural", '''
void main() {
    float k = 8.0;
    float x = v_uv.x;
    float u0 = 4.0 * (atan(exp(k * (x - 0.35))) - atan(exp(k * (x - 0.65))));
    f_color = vec4(u0, 0.0, 0.0, 1.0);  // R=u (kink pair), G=v=0
}
''')

_register("sine_gordon_step",
          "Sine-Gordon one step (leapfrog): v += c2*lap - G*sin(u); u += v + drive",
          "procedural", '''
void main() {
    vec2 texel = 1.0 / u_resolution;
    vec4 s = texture(u_texture, v_uv);
    float u = s.r, v = s.g;
    float lu = texture(u_texture, v_uv + vec2(-texel.x, 0.0)).r
             + texture(u_texture, v_uv + vec2( texel.x, 0.0)).r
             + texture(u_texture, v_uv + vec2(0.0, texel.y)).r
             + texture(u_texture, v_uv + vec2(0.0,-texel.y)).r - 4.0 * u;
    float c = clamp(u_params.x, 0.5, 1.5);
    float c2 = min(0.20 * c * c, 0.45);
    float damp = clamp(u_params.y, 0.95, 1.0);
    float G = clamp(u_params.z, 0.1, 4.0);
    float S = 0.20 * G;
    float A = clamp(u_params.w, 0.0, 2.0);
    float drive = A * 0.05 * (sin(6.2831853 * 3.0 * v_uv.x) + sin(6.2831853 * 3.0 * v_uv.y));
    float vn = (v + c2 * lu - S * sin(u)) * damp;
    float un = u + vn + drive;
    f_color = vec4(clamp(un, -8.0, 8.0), clamp(vn, -8.0, 8.0), 0.0, 1.0);
}
''')

_register("sine_gordon_display",
          "Sine-Gordon display: displacement -> plasma-like palette",
          "procedural", '''
void main() {
    float u = texture(u_texture, v_uv).r;
    float t = clamp(u / 6.2831853, 0.0, 1.0);
    vec3 col = mix(vec3(0.05, 0.05, 0.20), vec3(0.90, 0.40, 0.10), t);
    col = mix(col, vec3(1.0, 0.95, 0.60), smoothstep(0.6, 1.0, t));
    f_color = vec4(col, 1.0);
}
''')

# ── Node 144: Faraday Waves ── -------------------------------------------------
# Parametrically-driven damped wave: force = nu*lap - gamma*v - (w0^2 + A*cos p)*u + a*u^3
# p1=amplitude A, p2=omega0, p3=damping gamma, p4=capillary nu. Alpha fixed 0.5.
_register("faraday_seed",
          "Faraday Waves seed: multi-scale hashed noise height field (node 144 twin)",
          "procedural", '''
void main() {
    float n = noise(v_uv * 6.0) * 0.5 + noise(v_uv * 14.0) * 0.3
            + noise(v_uv * 30.0) * 0.2;
    f_color = vec4((n - 0.5) * 0.5, 0.0, 0.0, 1.0);  // R=h (small), G=v, B=phase
}
''')

_register("faraday_step",
          "Faraday Waves one step (parametric pump, 5-pt toroidal Laplacian)",
          "procedural", '''
void main() {
    vec2 texel = 1.0 / u_resolution;
    vec4 s = texture(u_texture, v_uv);
    float u = s.r, v = s.g, phase = s.b;
    float lu = texture(u_texture, v_uv + vec2(-texel.x, 0.0)).r
             + texture(u_texture, v_uv + vec2( texel.x, 0.0)).r
             + texture(u_texture, v_uv + vec2(0.0, texel.y)).r
             + texture(u_texture, v_uv + vec2(0.0,-texel.y)).r - 4.0 * u;
    float A = u_params.x, w0 = clamp(u_params.y, 0.5, 6.0);
    float gamma = clamp(u_params.z, 0.02, 1.5);
    float nu = clamp(u_params.w, 0.05, 4.0);
    float dt = 0.08;
    float Omega = 2.0 * w0;                  // drive at 2*omega0 (subharmonic)
    phase = mod(phase + dt * Omega, 6.2831853);
    float drive = A * cos(phase);
    float alpha = 0.5;
    float force = nu * lu - gamma * v - (w0 * w0 + drive) * u + alpha * u * u * u;
    float vn = v + dt * force;
    float un = u + dt * vn;
    // soft clamp to avoid blowup
    float peak = max(abs(un), 1.0);
    if (peak > 8.0) { un *= 8.0 / peak; vn *= 8.0 / peak; }
    f_color = vec4(un, vn, phase, 1.0);
}
''')

_register("faraday_display",
          "Faraday Waves display: height field sigmoid (grayscale, matches _render_faraday)",
          "procedural", '''
void main() {
    float h = texture(u_texture, v_uv).r;
    float sig = tanh(clamp(h, -4.0, 4.0) * 2.5);
    float g = sig * 0.5 + 0.5;
    f_color = vec4(vec3(g), 1.0);
}
''')

# ── Node 166: Parametric Oscillator Lattice (Oscillon Resonance) ── ------------
# d2u/dt2 = D*lap - gamma*v - w0^2(1 + eps*sin p)*u - beta*u^3
# p1=epsilon, p2=omega0, p3=damping gamma, p4=diffusion D. Beta fixed 0.3.
_register("oscillon_seed",
          "Oscillon Resonance seed: multi-scale hashed noise displacement (node 166 twin)",
          "procedural", '''
void main() {
    float n = noise(v_uv * 5.0) * 0.5 + noise(v_uv * 13.0) * 0.3
            + noise(v_uv * 28.0) * 0.2;
    f_color = vec4((n - 0.5) * 0.4, 0.0, 0.0, 1.0);  // R=u (small), G=v, B=phase
}
''')

_register("oscillon_step",
          "Oscillon Resonance one step (parametric Mathieu pump, 5-pt toroidal Laplacian)",
          "procedural", '''
void main() {
    vec2 texel = 1.0 / u_resolution;
    vec4 s = texture(u_texture, v_uv);
    float u = s.r, v = s.g, phase = s.b;
    float lu = texture(u_texture, v_uv + vec2(-texel.x, 0.0)).r
             + texture(u_texture, v_uv + vec2( texel.x, 0.0)).r
             + texture(u_texture, v_uv + vec2(0.0, texel.y)).r
             + texture(u_texture, v_uv + vec2(0.0,-texel.y)).r - 4.0 * u;
    float eps = clamp(u_params.x, 0.05, 1.5);
    float w0 = clamp(u_params.y, 0.5, 6.0);
    float gamma = clamp(u_params.z, 0.01, 1.0);
    float D = clamp(u_params.w, 0.05, 4.0);
    float dt = 0.1;
    float pump = 2.0 * w0;                   // omega_p = 2*omega0
    phase = mod(phase + dt * pump, 6.2831853);
    float stiff = w0 * w0 * (1.0 + eps * sin(phase));
    float beta = 0.3;
    float force = D * lu - gamma * v - stiff * u - beta * u * u * u;
    float vn = v + dt * force;
    float un = u + dt * vn;
    float peak = max(abs(un), 1.0);
    if (peak > 8.0) { un *= 8.0 / peak; vn *= 8.0 / peak; }
    f_color = vec4(un, vn, phase, 1.0);
}
''')

_register("oscillon_display",
          "Oscillon Resonance display: displacement sigmoid (grayscale, matches _render_displacement)",
          "procedural", '''
void main() {
    float u = texture(u_texture, v_uv).r;
    float sig = tanh(clamp(u, -4.0, 4.0) * 2.5);
    float g = sig * 0.5 + 0.5;
    f_color = vec4(vec3(g), 1.0);
}
''')

# ── Node 131: Burridge-Knopoff Spring-Block (Earthquake Cascades) ── -----------
# 2D grid of frictional blocks slowly driven by a plate. Stress builds until a
# block exceeds its heterogeneous friction threshold and slips, resetting to a
# residual level and redistributing a coupling fraction of released stress to
# its 4 neighbors — which may trigger a branching cascade. The CPU node runs an
# inner while-loop to fully relax a cascade per frame; the GPU twin performs one
# threshold+redistribute relaxation per substep, so a cascade propagates over
# consecutive substeps (many substeps/frame → visually equivalent avalanches).
# State packs: .r = stress, .g = damage (accumulated slip count), .b = strength
# (heterogeneous friction, seeded once and preserved). CPU numpy node stays the
# authoritative export (two-tier precision).
# p1=loading_rate, p2=threshold, p3=residual, p4=coupling(α).
_register("burridge_seed",
          "Burridge-Knopoff seed: heterogeneous strength (.b), near-threshold stress (.r), zero damage (node 131 twin)",
          "procedural", '''
void main() {
    float thr = clamp(u_params.y, 0.5, 5.0);
    // Heterogeneous per-cell strength in [0.7, 1.3] (matches CPU 0.7+0.6*rand).
    float hs = hash21(v_uv * 71.31 + 3.7);
    float strength = 0.7 + 0.6 * hs;
    // Initial stress near each block's own threshold (0.5..1.0 of thr).
    float hr = hash21(v_uv * 137.13 + 0.123);
    float stress = thr * (0.5 + 0.5 * hr);
    f_color = vec4(stress, 0.0, strength, 1.0);  // .r=stress .g=damage .b=strength
}
''')

_register("burridge_step",
          "Burridge-Knopoff one step: load + threshold slip + 4-neighbor stress redistribution (toroidal)",
          "procedural", '''
void main() {
    vec2 texel = 1.0 / u_resolution;
    vec4 s = texture(u_texture, v_uv);
    float stress   = s.r;
    float damage   = s.g;
    float strength = s.b;

    float rate  = clamp(u_params.x, 0.001, 0.1);
    float thr   = clamp(u_params.y, 0.5, 5.0);
    float resid = clamp(u_params.z, 0.0, 0.5);
    float alpha = clamp(u_params.w, 0.0, 0.25);

    // Slow tectonic loading + tiny per-cell noise (breaks symmetry / nucleates).
    float nz = (hash21(v_uv * 311.7 + fract(stress * 53.13)) - 0.5) * 0.008;
    stress += rate + nz;

    // Neighbor slip stress: a neighbor over its own effective threshold released
    // its stress; we receive alpha * that released stress from each of 4 sides.
    vec4 sl = texture(u_texture, v_uv + vec2(-texel.x, 0.0));
    vec4 sr = texture(u_texture, v_uv + vec2( texel.x, 0.0));
    vec4 su = texture(u_texture, v_uv + vec2(0.0,  texel.y));
    vec4 sd = texture(u_texture, v_uv + vec2(0.0, -texel.y));
    float rel_l = (sl.r > thr * sl.b) ? sl.r : 0.0;
    float rel_r = (sr.r > thr * sr.b) ? sr.r : 0.0;
    float rel_u = (su.r > thr * su.b) ? su.r : 0.0;
    float rel_d = (sd.r > thr * sd.b) ? sd.r : 0.0;
    stress += alpha * (rel_l + rel_r + rel_u + rel_d);

    // This block's own slip: if over effective threshold, reset to residual and
    // record a damage event (permanent scar for the fracture render).
    float eff = thr * strength;
    bool over = stress > eff;
    float new_stress = over ? resid : stress;
    float new_damage = damage + (over ? 1.0 : 0.0);

    f_color = vec4(clamp(new_stress, 0.0, 8.0), new_damage, strength, 1.0);
}
''')

_register("burridge_display",
          "Burridge-Knopoff display (tectonic): stress field + edge-detected crack lines (grayscale)",
          "procedural", '''
void main() {
    vec2 texel = 1.0 / u_resolution;
    float st = texture(u_texture, v_uv).r;
    // Contrast-stretched stress background (matches CPU tectonic: (s-0.2)/0.6 ^0.8).
    float s = clamp(st, 0.0, 1.0);
    float ss = clamp((s - 0.2) / 0.6, 0.0, 1.0);
    float bg = pow(ss, 0.8) * 0.59 + 0.08;
    // 4-directional stress gradient → bright crack edges.
    float c  = st;
    float gl = abs(texture(u_texture, v_uv + vec2(-texel.x, 0.0)).r - c);
    float gr = abs(texture(u_texture, v_uv + vec2( texel.x, 0.0)).r - c);
    float gu = abs(texture(u_texture, v_uv + vec2(0.0,  texel.y)).r - c);
    float gd = abs(texture(u_texture, v_uv + vec2(0.0, -texel.y)).r - c);
    float grad = max(max(gl, gr), max(gu, gd));
    float edges = clamp(grad * 3.0, 0.0, 1.0) * 0.86;
    // Faint permanent damage scars.
    float dmg = texture(u_texture, v_uv).g;
    float scar = clamp(log(1.0 + dmg) / 3.0, 0.0, 1.0) * 0.08;
    float g = max(max(bg, edges), scar);
    f_color = vec4(vec3(g), 1.0);
}
''')

# ═══════════════════════════════════════════════════════════════════════════
#  P1.3b — Fluid / surface-growth / lattice sim twins (nodes 132, 135, 150)
#  Same RGBA-float ping-pong contract as P1.3: seed writes initial state,
#  step reads u_texture + u_params (p1..p4) and writes new state, display maps
#  state -> RGB. CPU numpy nodes stay authoritative (two-tier precision).
# ═══════════════════════════════════════════════════════════════════════════

# ── Node 132: Shallow Water Waves ───────────────────────────────────────────
# 2D shallow-water surrogate: height h + velocity (u,v). A wave-like advection/
# diffusion of h coupled to velocity (gravity g, base depth, viscosity nu,
# source amplitude) gives a faithful live preview of the CPU solver.
# p1=gravity, p2=base_depth, p3=viscosity(nu), p4=source_amplitude.
_register("sw_seed",
          "Shallow Water seed: hashed noise height field, zero velocity (node 132 twin)",
          "procedural", '''
void main() {
    float n = noise(v_uv * 7.0) * 0.5 + noise(v_uv * 17.0) * 0.3
            + noise(v_uv * 31.0) * 0.2;
    f_color = vec4((n - 0.5) * 0.4, 0.0, 0.0, 1.0);  // R=h, G=u, B=v
}
''')

_register("sw_step",
          "Shallow Water one step (wave-coupled height/velocity, 5-pt toroidal Laplacian)",
          "procedural", '''
void main() {
    vec2 texel = 1.0 / u_resolution;
    vec4 s = texture(u_texture, v_uv);
    float h = s.r, u = s.g, v = s.b;
    float lh = texture(u_texture, v_uv + vec2(-texel.x, 0.0)).r
             + texture(u_texture, v_uv + vec2( texel.x, 0.0)).r
             + texture(u_texture, v_uv + vec2(0.0, texel.y)).r
             + texture(u_texture, v_uv + vec2(0.0,-texel.y)).r - 4.0 * h;
    float g = clamp(u_params.x, 1.0, 20.0);
    float h0 = clamp(u_params.y, 0.3, 3.0);
    float nu = clamp(u_params.z, 0.0001, 0.005);
    float amp = clamp(u_params.w, 0.02, 0.5);
    float dt = 0.08;
    // Gravity-driven height gradient -> velocity (wave coupling)
    float du = -g * (texture(u_texture, v_uv + vec2(texel.x,0.0)).r
                     - texture(u_texture, v_uv + vec2(-texel.x,0.0)).r) * 0.5;
    float dv = -g * (texture(u_texture, v_uv + vec2(0.0,texel.y)).r
                     - texture(u_texture, v_uv + vec2(0.0,-texel.y)).r) * 0.5;
    // Viscosity smooths velocity
    u += (nu * lh * 30.0 + du) * dt;
    v += (nu * lh * 30.0 + dv) * dt;
    // Height advected by velocity + diffusion
    float hn = h + dt * (0.20 * lh - (u + v) * 0.5);
    // Central ripple source (mirrors CPU two-point source)
    vec2 p0 = vec2(0.33, 0.5);
    float ds = distance(v_uv, p0);
    hn += amp * sin(u_time * 3.0) * exp(-(ds*ds)/0.002);
    float peak = max(abs(hn), 1.0);
    if (peak > 6.0) { hn *= 6.0 / peak; u *= 6.0 / peak; v *= 6.0 / peak; }
    f_color = vec4(clamp(hn, -6.0, 6.0), clamp(u, -6.0, 6.0), clamp(v, -6.0, 6.0), 1.0);
}
''')

_register("sw_display",
          "Shallow Water display: height anomaly -> plasma palette (matches _render_height)",
          "procedural", '''
void main() {
    float h = texture(u_texture, v_uv).r;
    float t = clamp(h * 0.5 + 0.5, 0.0, 1.0);
    vec3 col = mix(vec3(0.05, 0.10, 0.35), vec3(0.10, 0.55, 0.65), t);
    col = mix(col, vec3(0.85, 0.95, 1.0), smoothstep(0.6, 1.0, t));
    f_color = vec4(col, 1.0);
}
''')

# ── Node 135: KPZ Surface Growth / Erosion ──────────────────────────────────
# ∂h/∂t = ν·∇²h + (λ/2)·|∇h|² + η(x,t). Height h + phase accumulator for noise.
# State packs R=h, G=phase, B=unused. Live preview approximates the KPZ growth.
# p1=nu (diffusion), p2=lambda (nonlinearity), p3=noise_amplitude, p4=dt.
_register("kpz_seed",
          "KPZ seed: flat height field + small hashed perturbation (node 135 twin)",
          "procedural", '''
void main() {
    float n = noise(v_uv * 11.0) * 0.5 + noise(v_uv * 23.0) * 0.5;
    f_color = vec4((n - 0.5) * 0.2, 0.0, 0.0, 1.0);  // R=h, G=phase
}
''')

_register("kpz_step",
          "KPZ one step (diffusion + nonlinear growth + white-noise source)",
          "procedural", '''
void main() {
    vec2 texel = 1.0 / u_resolution;
    vec4 s = texture(u_texture, v_uv);
    float h = s.r, phase = s.b;
    float lh = texture(u_texture, v_uv + vec2(-texel.x, 0.0)).r
             + texture(u_texture, v_uv + vec2( texel.x, 0.0)).r
             + texture(u_texture, v_uv + vec2(0.0, texel.y)).r
             + texture(u_texture, v_uv + vec2(0.0,-texel.y)).r - 4.0 * h;
    float nu = clamp(u_params.x, 0.05, 3.0);
    float lam = clamp(u_params.y, -3.0, 3.0);
    float sigma = clamp(u_params.z, 0.01, 1.0);
    float dt = clamp(u_params.w, 0.01, 1.0);
    // Gradient magnitude squared |∇h|²
    float dx = (texture(u_texture, v_uv + vec2(texel.x,0.0)).r
                - texture(u_texture, v_uv + vec2(-texel.x,0.0)).r) * 0.5;
    float dy = (texture(u_texture, v_uv + vec2(0.0,texel.y)).r
                - texture(u_texture, v_uv + vec2(0.0,-texel.y)).r) * 0.5;
    float grad2 = dx*dx + dy*dy;
    // White-noise source (hashed, time-evolving)
    phase = fract(phase + dt);
    float eta = (hash21(floor(v_uv * u_resolution) + phase * 137.0) - 0.5);
    float dh = nu * lh + 0.5 * lam * grad2 + sigma * eta * 2.0;
    float hn = h + dt * dh;
    float peak = max(abs(hn), 1.0);
    if (peak > 10.0) { hn *= 10.0 / peak; }
    f_color = vec4(clamp(hn, -10.0, 10.0), phase, 0.0, 1.0);
}
''')

_register("kpz_display",
          "KPZ display: hillshaded height -> terrain grayscale (matches _render_terrain)",
          "procedural", '''
void main() {
    vec2 texel = 1.0 / u_resolution;
    float h = texture(u_texture, v_uv).r;
    float dx = (texture(u_texture, v_uv + vec2(texel.x,0.0)).r
                - texture(u_texture, v_uv + vec2(-texel.x,0.0)).r) * 0.5;
    float dy = (texture(u_texture, v_uv + vec2(0.0,texel.y)).r
                - texture(u_texture, v_uv + vec2(0.0,-texel.y)).r) * 0.5;
    vec3 nrm = normalize(vec3(-dx, -dy, 0.08));
    vec3 sun = normalize(vec3(0.6, 0.6, 0.7));
    float shade = clamp(dot(nrm, sun), 0.0, 1.0);
    float t = clamp(h * 0.25 + 0.5, 0.0, 1.0);
    float g = clamp(shade * 0.7 + t * 0.4, 0.0, 1.0);
    f_color = vec4(vec3(g), 1.0);
}
''')

# ── Node 150: FPU Chain Lattice ─────────────────────────────────────────────
# Conservative Verlet on a 2D mass-spring grid: displacement u + velocity v.
# Nonlinear springs (k2 linear, k3 cubic, k4 quartic). State packs R=u, G=v.
# p1=k2, p2=k3, p3=k4, p4=dt.
_register("fpu_seed",
          "FPU seed: multi-scale hashed displacement + small velocity (node 150 twin)",
          "procedural", '''
void main() {
    float n = noise(v_uv * 5.0) * 0.5 + noise(v_uv * 13.0) * 0.3
            + noise(v_uv * 29.0) * 0.2;
    f_color = vec4((n - 0.5) * 0.6, 0.0, 0.0, 1.0);  // R=u, G=v
}
''')

_register("fpu_step",
          "FPU one step (Verlet, 5-pt nonlinear spring coupling)",
          "procedural", '''
void main() {
    vec2 texel = 1.0 / u_resolution;
    vec4 s = texture(u_texture, v_uv);
    float u = s.r, v = s.g;
    float up = texture(u_texture, v_uv + vec2( texel.x, 0.0)).r;
    float um = texture(u_texture, v_uv + vec2(-texel.x, 0.0)).r;
    float vp = texture(u_texture, v_uv + vec2(0.0, texel.y)).r;
    float vm = texture(u_texture, v_uv + vec2(0.0,-texel.y)).r;
    float k2 = clamp(u_params.x, 0.1, 5.0);
    float k3 = clamp(u_params.y, 0.0, 2.0);
    float k4 = clamp(u_params.z, 0.0, 2.0);
    float dt = clamp(u_params.w, 0.01, 0.2);
    // Nonlinear spring force (discrete laplacian of force) — mirrors CPU acceleration()
    float dxp = up - u, dxm = u - um;
    float dyp = vp - u, dym = u - vm;
    float fx = k2 * (dxp - dxm) + k3 * (dxp*dxp - dxm*dxm) + k4 * (dxp*dxp*dxp - dxm*dxm*dxm);
    float fy = k2 * (dyp - dym) + k3 * (dyp*dyp - dym*dym) + k4 * (dyp*dyp*dyp - dym*dym*dym);
    float f = fx + fy;
    // Velocity-Verlet (per-frame v carried; u advanced by v + accel)
    float vn = v + f * dt;
    float un = u + vn * dt;
    float peak = max(abs(un), 1.0);
    if (peak > 30.0) { un *= 30.0 / peak; vn *= 30.0 / peak; }
    f_color = vec4(clamp(un, -30.0, 30.0), clamp(vn, -30.0, 30.0), 0.0, 1.0);
}
''')

_register("fpu_display",
          "FPU display: |displacement| -> fire palette (matches fpu_lattice render)",
          "procedural", '''
void main() {
    float a = abs(texture(u_texture, v_uv).r);
    float t = clamp(a * 1.2, 0.0, 1.0);
    vec3 col = vec3(0.0, 0.0, 0.15);
    col = mix(col, vec3(0.85, 0.25, 0.05), smoothstep(0.0, 0.5, t));
    col = mix(col, vec3(1.0, 0.95, 0.55), smoothstep(0.5, 1.0, t));
    f_color = vec4(col, 1.0);
}
''')

# ── Node 127: Kuramoto-Sivashinsky Equation ───────────────────────────────
# ∂u/∂t = -ν·∇⁴u - ∇²u - ½|∇u|². Scalar height field u packed in .r; .g is a
# phase accumulator that drives a hashed white-noise source (step shaders get
# u_time=0, pitfall #6b, so noise must be carried in state). Single channel.
# p1=nu (hyperviscosity), p2=dt, p3=noise_amp, p4=aniso_ratio (x/y stretch).
_register("ks_seed",
          "Kuramoto-Sivashinsky seed: small sinusoidal roll field + hashed noise (node 127 twin)",
          "procedural", '''
void main() {
    vec2 uv = v_uv * 6.2831853;
    float u = 0.2 * (sin(uv.x) + sin(uv.y)) + 0.05 * (noise(v_uv * 9.0) - 0.5);
    f_color = vec4(u, 0.0, 0.0, 1.0);  // R=u, G=phase
}
''')

_register("ks_step",
          "Kuramoto-Sivashinsky one step: -nu*∇⁴u - ∇²u - ½|∇u|² (5-pt operators + hashed noise)",
          "procedural", '''
void main() {
    vec2 texel = 1.0 / u_resolution;
    vec4 s = texture(u_texture, v_uv);
    float u = s.r, phase = s.g;
    // 5-pt Laplacian
    float c  = u;
    float l  = texture(u_texture, v_uv + vec2(-texel.x, 0.0)).r;
    float r  = texture(u_texture, v_uv + vec2( texel.x, 0.0)).r;
    float d  = texture(u_texture, v_uv + vec2(0.0,-texel.y)).r;
    float uu = texture(u_texture, v_uv + vec2(0.0, texel.y)).r;
    float lap = (l + r + d + uu - 4.0 * c);
    // 5-pt biharmonic (∇⁴) from the Laplacian
    float l2 = texture(u_texture, v_uv + vec2(-texel.x*2.0, 0.0)).r
             + texture(u_texture, v_uv + vec2( texel.x*2.0, 0.0)).r
             + texture(u_texture, v_uv + vec2(0.0,-texel.y*2.0)).r
             + texture(u_texture, v_uv + vec2(0.0, texel.y*2.0)).r
             - 4.0 * ((l+r+d+uu)*0.25);
    float lap2 = lap + (l2 - lap) * 0.5;  // cheap ∇⁴ stand-in
    // gradient magnitude squared |∇u|²
    float dx = (r - l) * 0.5, dy = (uu - d) * 0.5;
    float grad2 = dx*dx + dy*dy;
    float nu = clamp(u_params.x, 0.01, 0.5);
    float dt = clamp(u_params.y, 0.001, 0.05);
    float sigma = clamp(u_params.z, 0.0, 0.3);
    phase = fract(phase + dt);
    float eta = (hash21(floor(v_uv * u_resolution) + phase * 211.0) - 0.5) * sigma;
    float un = c + dt * (-nu * lap2 - lap - 0.5 * grad2 + eta);
    float peak = max(abs(un), 1.0);
    if (peak > 6.0) { un *= 6.0 / peak; }
    f_color = vec4(clamp(un, -6.0, 6.0), phase, 0.0, 1.0);
}
''')

_register("ks_display",
          "Kuramoto-Sivashinsky display: signed u -> tanh-sigmoid grayscale (matches _render_ks)",
          "procedural", '''
void main() {
    float u = texture(u_texture, v_uv).r;
    float g = clamp((tanh(u * 1.5) + 1.0) * 0.5, 0.0, 1.0);
    f_color = vec4(vec3(g), 1.0);
}
''')

# ── Node 128: Swift-Hohenberg (ε·u − u³ − (1+∇²)²·u) ───────────────────────
# Spectral-style pattern formation. Local approximation of the biharmonic
# operator via a 5-pt stencil. Scalar u packed in .r; .g phase drives noise.
# p1=epsilon, p2=dt, p3=noise_amp, p4=linear_gain (~0.5 ctrl of (1+∇²) weight).
_register("sh128_seed",
          "Swift-Hohenberg (128) seed: small hashed pattern field (node 128 twin)",
          "procedural", '''
void main() {
    float u = 0.2 * (sin(v_uv.x * 12.0) * 0.5 + sin(v_uv.y * 9.0) * 0.5)
            + 0.05 * (noise(v_uv * 7.0) - 0.5);
    f_color = vec4(u, 0.0, 0.0, 1.0);  // R=u, G=phase
}
''')

_register("sh128_step",
          "Swift-Hohenberg (128) one step: epsilon*u - u^3 - (1+lap)^2 u + noise",
          "procedural", '''
void main() {
    vec2 texel = 1.0 / u_resolution;
    vec4 s = texture(u_texture, v_uv);
    float u = s.r, phase = s.g;
    // 5-pt Laplacian
    float c  = u;
    float l  = texture(u_texture, v_uv + vec2(-texel.x, 0.0)).r;
    float r  = texture(u_texture, v_uv + vec2( texel.x, 0.0)).r;
    float d  = texture(u_texture, v_uv + vec2(0.0,-texel.y)).r;
    float uu = texture(u_texture, v_uv + vec2(0.0, texel.y)).r;
    float lap = (l + r + d + uu - 4.0 * c);
    // (1 + lap)^2 u  = u + 2*lap + lap*lagain  (local ∇⁴ stand-in)
    float lap2 = (l + r + d + uu - 4.0 * lap) - 4.0 * c;  // ∇⁴ approx
    float lap4 = lap + lap2;
    float eps = clamp(u_params.x, -0.5, 3.0);
    float dt = clamp(u_params.y, 0.01, 1.0);
    float sigma = clamp(u_params.z, 0.0, 1.0);
    float gain = clamp(u_params.w, 0.0, 2.0);
    phase = fract(phase + dt);
    float eta = (hash21(floor(v_uv * u_resolution) + phase * 97.0) - 0.5) * sigma;
    float reaction = eps * c - c * c * c;
    // -(1 + gain*lap)^2 u  ~  -u - 2*gain*lap - gain*gain*lap2
    float lin = -c - 2.0 * gain * lap - gain * gain * lap2;
    float un = c + dt * (reaction + lin + eta);
    float peak = max(abs(un), 1.0);
    if (peak > 4.0) { un *= 4.0 / peak; }
    f_color = vec4(clamp(un, -4.0, 4.0), phase, 0.0, 1.0);
}
''')

_register("sh128_display",
          "Swift-Hohenberg (128) display: u in [-2,2] -> grayscale (matches _render_field)",
          "procedural", '''
void main() {
    float u = texture(u_texture, v_uv).r;
    float g = clamp((u + 2.0) / 4.0, 0.0, 1.0);
    f_color = vec4(vec3(g), 1.0);
}
''')

# ── Node 157: Swift-Hohenberg (r·u − (∇²+q₀²)²·u − u³) ─────────────────────
# Same ε·u − u³ structure with a tuned wavenumber band. q0 packs into p2.
# p1=r, p2=q0, p3=dt, p4=noise_amp.
_register("sh157_seed",
          "Swift-Hohenberg (157) seed: small hashed field + q0-scale hex hint (node 157 twin)",
          "procedural", '''
void main() {
    float q0 = clamp(u_params.y, 0.02, 0.3);
    float u = 0.15 * (cos(q0 * v_uv.x * 6.2831853)
                      + cos(q0 * 0.5 * v_uv.x * 6.2831853 + 0.8660254 * q0 * v_uv.y * 6.2831853)
                      + cos(q0 * 0.5 * v_uv.x * 6.2831853 - 0.8660254 * q0 * v_uv.y * 6.2831853));
    u += 0.05 * (noise(v_uv * 7.0) - 0.5);
    f_color = vec4(u, 0.0, 0.0, 1.0);  // R=u, G=phase
}
''')

_register("sh157_step",
          "Swift-Hohenberg (157) one step: r*u - (lap+q0^2)^2 u - u^3 + noise",
          "procedural", '''
void main() {
    vec2 texel = 1.0 / u_resolution;
    vec4 s = texture(u_texture, v_uv);
    float u = s.r, phase = s.g;
    float c  = u;
    float l  = texture(u_texture, v_uv + vec2(-texel.x, 0.0)).r;
    float r  = texture(u_texture, v_uv + vec2( texel.x, 0.0)).r;
    float d  = texture(u_texture, v_uv + vec2(0.0,-texel.y)).r;
    float uu = texture(u_texture, v_uv + vec2(0.0, texel.y)).r;
    float lap = (l + r + d + uu - 4.0 * c);
    float lap2 = (l + r + d + uu - 4.0 * lap) - 4.0 * c;  // ∇⁴ approx
    float q0 = clamp(u_params.y, 0.02, 0.3);
    float rq02 = (lap + q0 * q0) * (lap + q0 * q0);  // (∇²+q0²) for lin term
    float rq04 = rq02 * rq02;                          // (∇²+q0²)²
    float rr = clamp(u_params.x, -1.0, 5.0);
    float dt = clamp(u_params.z, 0.01, 0.5);
    float sigma = clamp(u_params.w, 0.0, 0.1);
    phase = fract(phase + dt);
    float eta = (hash21(floor(v_uv * u_resolution) + phase * 53.0) - 0.5) * sigma * 10.0;
    float reaction = rr * c - c * c * c;
    float lin = -rq04;
    float un = c + dt * (reaction + lin + eta);
    float peak = max(abs(un), 1.0);
    if (peak > 4.0) { un *= 4.0 / peak; }
    f_color = vec4(clamp(un, -4.0, 4.0), phase, 0.0, 1.0);
}
''')

_register("sh157_display",
          "Swift-Hohenberg (157) display: mean-centered tanh grayscale (matches _render)",
          "procedural", '''
void main() {
    float u = texture(u_texture, v_uv).r;
    float g = clamp((tanh(u * 1.5) + 1.0) * 0.5, 0.0, 1.0);
    f_color = vec4(vec3(g), 1.0);
}
''')

# ── Node 162: Coupled Rössler Oscillator Array ─────────────────────────────
# 3-variable chaotic oscillator per cell, diffusively coupled on a 2D grid.
# State packs R=x (slow), G=y (fast), B=z (fold). p1=a, p2=b, p3=c_ross,
# p4=omega. coupling D is fixed (CPU authoritative); live preview approximates.
_register("ross_seed",
          "Rössler array seed: near-fixed-point oscillation, hashed per-cell (node 162 twin)",
          "procedural", '''
void main() {
    float x = -5.7 + 0.5 * (noise(v_uv * 13.0) - 0.5);
    float y = -5.7 + 0.5 * (noise(v_uv * 13.0 + 3.1) - 0.5);
    float z = 5.7  + 0.5 * (noise(v_uv * 13.0 + 7.7) - 0.5);
    f_color = vec4(x, y, z, 1.0);  // R=x, G=y, B=z
}
''')

_register("ross_step",
          "Rössler array one step: per-cell Rössler ODE + 5-pt diffusive coupling",
          "procedural", '''
void main() {
    vec2 texel = 1.0 / u_resolution;
    vec4 s = texture(u_texture, v_uv);
    float x = s.r, y = s.g, z = s.b;
    // 5-pt Laplacian of each channel
    vec4 ll = texture(u_texture, v_uv + vec2(-texel.x, 0.0));
    vec4 rr = texture(u_texture, v_uv + vec2( texel.x, 0.0));
    vec4 dd = texture(u_texture, v_uv + vec2(0.0,-texel.y));
    vec4 uu = texture(u_texture, v_uv + vec2(0.0, texel.y));
    vec3 lap = (ll.rgb + rr.rgb + dd.rgb + uu.rgb - 4.0 * vec3(x, y, z));
    float a = clamp(u_params.x, 0.1, 0.6);
    float b = clamp(u_params.y, 0.05, 0.5);
    float c = clamp(u_params.z, 3.0, 15.0);
    float omega = clamp(u_params.w, 0.5, 2.0);
    float D = 0.5;  // fixed coupling strength (CPU authoritative)
    float dt = 0.08;
    float dx = -omega * y - z + D * lap.x;
    float dy =  omega * x + a * y + D * lap.y;
    float dz =  b + z * (x - c) + D * lap.z;
    vec3 nn = vec3(x, y, z) + dt * vec3(dx, dy, dz);
    nn.z = clamp(nn.z, 0.0, 30.0);  // z always positive in Rössler
    f_color = vec4(nn, 1.0);
}
''')

_register("ross_display",
          "Rössler array display: x/y/z -> HSV-ish composite (matches render_style=composite)",
          "procedural", '''
void main() {
    vec3 v = texture(u_texture, v_uv).rgb;
    float x = v.r, y = v.g, z = v.b;
    float xr = clamp((x + 12.0) / 24.0, 0.0, 1.0);
    float yr = clamp((y + 12.0) / 24.0, 0.0, 1.0);
    float zr = clamp(z / 30.0, 0.0, 1.0);
    vec3 col = vec3(xr, yr, 0.3 + 0.7 * zr);
    f_color = vec4(col, 1.0);
}
''')

# ── Node 170: Phase Field Crystal ───────────────────────────────────────────
# PFC amplitude equation live-preview approximation. Single scalar ψ packed in
# .r; .g phase drives noise. p1=epsilon, p2=dt, p3=noise_amp, p4=r2 (=r/2).
_register("pfc_seed",
          "Phase Field Crystal seed: small hashed amplitude field (node 170 twin)",
          "procedural", '''
void main() {
    float psi = 0.2 * (noise(v_uv * 11.0) - 0.5) + 0.05 * (noise(v_uv * 23.0) - 0.5);
    f_color = vec4(psi, 0.0, 0.0, 1.0);  // R=psi, G=phase
}
''')

_register("pfc_step",
          "Phase Field Crystal one step: lap(psi + lap psi) - r/2 psi^2 + psi^3 + noise",
          "procedural", '''
void main() {
    vec2 texel = 1.0 / u_resolution;
    vec4 s = texture(u_texture, v_uv);
    float psi = s.r, phase = s.g;
    float c  = psi;
    float l  = texture(u_texture, v_uv + vec2(-texel.x, 0.0)).r;
    float r  = texture(u_texture, v_uv + vec2( texel.x, 0.0)).r;
    float d  = texture(u_texture, v_uv + vec2(0.0,-texel.y)).r;
    float uu = texture(u_texture, v_uv + vec2(0.0, texel.y)).r;
    float lap = (l + r + d + uu - 4.0 * c);
    float lap2 = (l + r + d + uu - 4.0 * lap) - 4.0 * c;  // ∇⁴ approx
    float eps = clamp(u_params.x, 0.01, 0.5);
    float dt = clamp(u_params.y, 0.01, 0.5);
    float sigma = clamp(u_params.z, 0.0, 0.1);
    float r2 = clamp(u_params.w, 0.0, 2.0);  // = r/2 quadratic coefficient
    float lin = lap + lap2;
    float reaction = eps * c - r2 * c * c + (1.0 / 3.0) * c * c * c;
    phase = fract(phase + dt);
    float eta = (hash21(floor(v_uv * u_resolution) + phase * 31.0) - 0.5) * sigma * 12.0;
    float pn = c + dt * (lin + reaction + eta);
    float peak = max(abs(pn), 1.0);
    if (peak > 4.0) { pn *= 4.0 / peak; }
    f_color = vec4(clamp(pn, -4.0, 4.0), phase, 0.0, 1.0);
}
''')

_register("pfc_display",
          "Phase Field Crystal display: signed psi -> tanh grayscale (matches PFC render)",
          "procedural", '''
void main() {
    float psi = texture(u_texture, v_uv).r;
    float g = clamp((tanh(psi * 1.5) + 1.0) * 0.5, 0.0, 1.0);
    f_color = vec4(vec3(g), 1.0);
}
''')

# ── Nodes 95 / 142: Coupled Logistic Map Lattice ────────────────────────────
# Both nodes are the SAME dynamical system: each cell x evolves via the logistic
# map f(x)=r·x·(1-x), diffusively coupled to its 4 neighbours with strength ε:
#   x' = (1-ε)·f(x) + (ε/4)·Σ f(x_neighbour)
# Discrete-time recurrence ⇒ raw state strobes, so an EMA trail (decay) is packed
# alongside the raw lattice: state R=raw x, G=accum (trail). display reads accum.
# p1=r (3.5–4.0), p2=ε coupling (0.05–0.5), p3=decay trail (0.5–0.99).
_register("cml_seed",
          "Coupled logistic seed: hashed uniform lattice in [0,1] (nodes 95/142 twin)",
          "procedural", '''
void main() {
    float x = hash21(v_uv * u_resolution + 0.123);
    f_color = vec4(x, x, 0.0, 1.0);  // R=raw x, G=accum(trail)
}
''')

_register("cml_step",
          "Coupled logistic one step: logistic map + diffusive coupling + EMA trail",
          "procedural", '''
void main() {
    vec2 texel = 1.0 / u_resolution;
    vec4 s = texture(u_texture, v_uv);
    float x = s.r, accum = s.g;
    float r = clamp(u_params.x, 3.5, 4.0);
    float eps = clamp(u_params.y, 0.05, 0.5);
    float decay = clamp(u_params.z, 0.5, 0.99);
    // f(x) at this cell and its 4 toroidal neighbours
    float fx  = r * x * (1.0 - x);
    float xu  = texture(u_texture, v_uv + vec2(0.0, -texel.y)).r;
    float xd  = texture(u_texture, v_uv + vec2(0.0,  texel.y)).r;
    float xl  = texture(u_texture, v_uv + vec2(-texel.x, 0.0)).r;
    float xr  = texture(u_texture, v_uv + vec2( texel.x, 0.0)).r;
    float fsum = r*xu*(1.0-xu) + r*xd*(1.0-xd) + r*xl*(1.0-xl) + r*xr*(1.0-xr);
    float xn = (1.0 - eps) * fx + (eps * 0.25) * fsum;
    xn = clamp(xn, 0.0, 1.0);
    // Exponential moving-average trail to suppress discrete-time strobing
    float an = decay * accum + (1.0 - decay) * xn;
    f_color = vec4(xn, clamp(an, 0.0, 1.0), 0.0, 1.0);
}
''')

_register("cml95_display",
          "Coupled logistic display (node 95): trail -> magma-inspired colormap",
          "procedural", '''
void main() {
    float t = clamp(texture(u_texture, v_uv).g, 0.0, 1.0);
    // Piecewise magma-inspired ramp matching _COLORMAP_256 (dark→purple→orange→gold)
    vec3 c0 = vec3(0.016, 0.016, 0.063);
    vec3 c1 = vec3(0.314, 0.0,   0.314);
    vec3 c2 = vec3(0.706, 0.157, 0.471);
    vec3 c3 = vec3(0.941, 0.471, 0.157);
    vec3 c4 = vec3(1.0,   0.863, 0.235);
    vec3 col;
    if (t < 0.25)      col = mix(c0, c1, t / 0.25);
    else if (t < 0.50) col = mix(c1, c2, (t - 0.25) / 0.25);
    else if (t < 0.75) col = mix(c2, c3, (t - 0.50) / 0.25);
    else               col = mix(c3, c4, (t - 0.75) / 0.25);
    f_color = vec4(col, 1.0);
}
''')

_register("cml142_display",
          "Coupled logistic display (node 142): trail -> grayscale (matches CML render)",
          "procedural", '''
void main() {
    float g = clamp(texture(u_texture, v_uv).g, 0.0, 1.0);
    f_color = vec4(vec3(g), 1.0);
}
''')


# ── Complex Ginzburg-Landau (client-GPU sim of node 126) ────────────────────
# Complex field A packed as .r = Re(A), .g = Im(A). Explicit Euler with a
# 5-point Laplacian (toroidal). CGL: dA/dt = A + (1+i*alpha)*lap(A)
#   - (1+i*beta)*|A|^2*A. CPU node is Arch-A sim; this is the live-preview twin
# only — server export stays authoritative (seeded layout differs, as expected).
_register("cgl_seed",
          "CGL initial state: small random complex noise in RG (node 126 twin)",
          "procedural", '''
void main() {
    vec2 p = v_uv * u_resolution;
    float a = (hash21(p + 0.19) - 0.5) * 0.2;
    float b = (hash21(p + 7.31) - 0.5) * 0.2;
    f_color = vec4(a, b, 0.0, 1.0);
}
''')

_register("cgl_step",
          "CGL one Euler step (5-pt Laplacian, toroidal) — complex field in RG",
          "procedural", '''
void main() {
    vec2 texel = 1.0 / u_resolution;
    vec4 s = texture(u_texture, v_uv);
    float a = s.r, b = s.g;
    vec4 sl = texture(u_texture, v_uv + vec2(-texel.x, 0.0));
    vec4 sr = texture(u_texture, v_uv + vec2( texel.x, 0.0));
    vec4 su = texture(u_texture, v_uv + vec2(0.0,  texel.y));
    vec4 sd = texture(u_texture, v_uv + vec2(0.0, -texel.y));
    float lapR = sl.r + sr.r + su.r + sd.r - 4.0 * a;
    float lapI = sl.g + sr.g + su.g + sd.g - 4.0 * b;
    float alpha = clamp(u_params.x, -3.0, 3.0);   // p1: node alpha (-3..3)
    float beta  = clamp(u_params.y, -3.0, 3.0);   // p2: node beta (-3..3)
    float dt    = clamp(u_params.z, 0.005, 0.2);  // p3: node dt
    float m = a * a + b * b;
    // (1+i*alpha)*lap
    float dispR = lapR - alpha * lapI;
    float dispI = lapI + alpha * lapR;
    // (1+i*beta)*|A|^2*A
    float nlR = m * (a - beta * b);
    float nlI = m * (b + beta * a);
    float na = a + dt * (a + dispR - nlR);
    float nb = b + dt * (b + dispI - nlI);
    // clamp amplitude to avoid blowup in the live preview
    float mag = sqrt(na * na + nb * nb);
    if (mag > 3.0) { na *= 3.0 / mag; nb *= 3.0 / mag; }
    f_color = vec4(na, nb, 0.0, 1.0);
}
''')

_register("cgl_display",
          "CGL display: phase -> hue, amplitude -> brightness (phase_amp mode)",
          "procedural", '''
void main() {
    vec4 s = texture(u_texture, v_uv);
    float a = s.r, b = s.g;
    float amp = clamp(sqrt(a * a + b * b), 0.0, 1.0);
    float phase = atan(b, a);              // -pi..pi
    float hue = (phase + 3.14159265) / 6.28318530;
    vec3 col = clamp(abs(fract(hue + vec3(0.0, 0.6667, 0.3333)) * 6.0 - 3.0) - 1.0, 0.0, 1.0);
    f_color = vec4(col * (0.35 + 0.65 * amp), 1.0);
}
''')

# ── P1.3 complex-field PDE — Nonlinear Schrödinger (node 124). Same R/G complex
# field packing as CGL. NLSE in real space: ψ=a+ib, ∂ψ/∂t = i(β∇²ψ − g|ψ|²ψ + Vψ)
#   → ∂a/∂t = −β·∇²b + g·|ψ|²·b − V·b ;  ∂b/∂t = β·∇²a − g|ψ|²·a + V·a
# Explicit Euler on the 5-pt (toroidal) Laplacian. CPU node is a split-step
# Fourier Arch-A sim; this is the live-preview twin only — server export stays
# authoritative (seeded layout differs, as expected for this PDE family).
_register("nls_seed",
          "NLSE initial state: small random complex noise in RG (node 124 twin)",
          "procedural", '''
void main() {
    vec2 p = v_uv * u_resolution;
    float a = (hash21(p + 0.19) - 0.5) * 0.3;
    float b = (hash21(p + 7.31) - 0.5) * 0.3;
    f_color = vec4(a, b, 0.0, 1.0);
}
''')

_register("nls_step",
          "NLSE one Euler step (5-pt Laplacian, toroidal) — complex field in RG",
          "procedural", '''
void main() {
    vec2 texel = 1.0 / u_resolution;
    vec4 s = texture(u_texture, v_uv);
    float a = s.r, b = s.g;
    vec4 sl = texture(u_texture, v_uv + vec2(-texel.x, 0.0));
    vec4 sr = texture(u_texture, v_uv + vec2( texel.x, 0.0));
    vec4 su = texture(u_texture, v_uv + vec2(0.0,  texel.y));
    vec4 sd = texture(u_texture, v_uv + vec2(0.0, -texel.y));
    float lapR = sl.r + sr.r + su.r + sd.r - 4.0 * a;
    float lapI = sl.g + sr.g + su.g + sd.g - 4.0 * b;
    float beta = clamp(u_params.x, -2.0, 2.0);   // p1: node dispersion β
    float gnl  = clamp(u_params.y, -3.0, 3.0);   // p2: node nonlinearity g (+focus)
    float dt   = clamp(u_params.z, 0.002, 0.1);  // p3: node dt
    float trap = u_params.w;                      // p4: node trap_strength
    float r2 = (v_uv.x - 0.5) * (v_uv.x - 0.5)
             + (v_uv.y - 0.5) * (v_uv.y - 0.5);
    float V = trap * 400.0 * r2;                  // harmonic trap (live scale)
    float m = a * a + b * b;
    // ∂a/∂t = -β·lapI + g·m·b - V·b ;  ∂b/∂t = β·lapR - g·m·a + V·a
    float da = -beta * lapI + gnl * m * b - V * b;
    float db =  beta * lapR - gnl * m * a + V * a;
    float na = a + dt * da;
    float nb = b + dt * db;
    // clamp amplitude to avoid blowup in the live preview
    float mag = sqrt(na * na + nb * nb);
    if (mag > 4.0) { na *= 4.0 / mag; nb *= 4.0 / mag; }
    f_color = vec4(na, nb, 0.0, 1.0);
}
''')

_register("nls_display",
          "NLSE display: phase -> hue, amplitude -> brightness (combined style)",
          "procedural", '''
void main() {
    vec4 s = texture(u_texture, v_uv);
    float a = s.r, b = s.g;
    float amp = clamp(sqrt(a * a + b * b), 0.0, 1.0);
    float phase = atan(b, a);              // -pi..pi
    float hue = (phase + 3.14159265) / 6.28318530;
    vec3 col = clamp(abs(fract(hue + vec3(0.0, 0.6667, 0.3333)) * 6.0 - 3.0) - 1.0, 0.0, 1.0);
    f_color = vec4(col * (0.35 + 0.65 * amp), 1.0);
}
''')

# ── P1.3 complex-field PDE — Gross-Pitaevskii (node 148). Same R/G complex
# field packing as CGL/NLSE. ψ=a+ib, split-step symplectic Euler: half-step
# nonlinear (kinetic in k-space via a precomputed k² texture) + half-step
# potential. The live twin approximates the spectral kinetic step with a 5-pt
# Laplacian proxy (lapR, lapI) which carries the same smoothing dynamics; the
# CPU node stays authoritative for frame-accurate export (seeded layout +
# full split-step Fourier differ, as expected for this PDE family).
#   Re(k²ψ) = lapR, Im(k²ψ) = lapI ; D = (g·m + V) is real potential.
#   half-nonlin: a' = a·cos(D·dt/2) - b·sin(D·dt/2)
#                b' = b·cos(D·dt/2) + a·sin(D·dt/2)
#   kinetic:     a'' = a' + α·lapI·dt ;  b'' = b' - α·lapR·dt
#                (∂a/∂t = +α·∇²b, ∂b/∂t = -α·∇²a → curl-free rotation)
_register("gpe_seed",
          "GPE initial state: small random complex Gaussian bump in RG (node 148 twin)",
          "procedural", '''
void main() {
    vec2 p = v_uv * u_resolution;
    vec2 d = v_uv - 0.5;
    float bump = exp(-dot(d, d) * 8.0);                 // central condensate
    float a = bump * (1.0 + (hash21(p + 0.19) - 0.5) * 0.06);
    float b = bump * (hash21(p + 7.31) - 0.5) * 0.06;
    f_color = vec4(a, b, 0.0, 1.0);                     // .b = accumulated sim-time (0)
}
''')

_register("gpe_step",
          "GPE one symplectic Euler step (5-pt Laplacian proxy for kinetic) — complex field in RG",
          "procedural", '''
void main() {
    // NOTE: u_time is 0 here (renderGpuSim passes no time to step shaders,
    // pitfall #6b). We carry an accumulating sim-time in the .b state channel
    // so the stirrer orbits and the live preview actually moves frame to frame.
    vec2 texel = 1.0 / u_resolution;
    vec4 s = texture(u_texture, v_uv);
    float a = s.r, b = s.g;
    float tt = s.b;                            // accumulated sim-time (advances each step)
    vec4 sl = texture(u_texture, v_uv + vec2(-texel.x, 0.0));
    vec4 sr = texture(u_texture, v_uv + vec2( texel.x, 0.0));
    vec4 su = texture(u_texture, v_uv + vec2(0.0,  texel.y));
    vec4 sd = texture(u_texture, v_uv + vec2(0.0, -texel.y));
    float lapR = sl.r + sr.r + su.r + sd.r - 4.0 * a;
    float lapI = sl.g + sr.g + su.g + sd.g - 4.0 * b;
    float gnl  = clamp(u_params.x, 0.0, 4.0);    // p1: node nonlinearity g (0.2..4.0)
    float ss   = clamp(u_params.y, 0.02, 1.5);   // p2: node stir_speed (0.05..1.5)
    float alpha = clamp(u_params.z, 0.02, 2.0);  // p3: node alpha / kinetic coeff (0.05..2.0)
    float stp  = clamp(u_params.w, 0.0, 20.0);   // p4: node stir_amp (1..20)
    float dt = 0.05;                             // fixed live-preview timestep
    // Moving repulsive stirrer (single gaussian, orbits via accumulated time)
    vec2 ctr = vec2(0.5 + 0.18 * sin(tt * 0.6 * ss), 0.5 + 0.14 * cos(tt * 0.5 * ss));
    vec2 dd = v_uv - ctr;
    float V = stp * 6.0 * exp(-dot(dd, dd) * 30.0);
    float m = a * a + b * b;
    float D = 0.5 * (gnl * m + V) * dt;          // half-step potential phase
    float c = cos(D), sn = sin(D);
    float a1 = a * c - b * sn;
    float b1 = b * c + a * sn;
    // kinetic step (5-pt Laplacian proxy for spectral k²)
    float a2 = a1 + alpha * lapI * dt;
    float b2 = b1 - alpha * lapR * dt;
    float mag = sqrt(a2 * a2 + b2 * b2);
    if (mag > 4.0) { a2 *= 4.0 / mag; b2 *= 4.0 / mag; }
    f_color = vec4(a2, b2, tt + dt, 1.0);        // .b carries the advancing sim-time
}
''')

_register("gpe_display",
          "GPE display: phase -> hue, amplitude -> brightness (phase render style)",
          "procedural", '''
void main() {
    vec4 s = texture(u_texture, v_uv);
    float a = s.r, b = s.g;
    float amp = clamp(sqrt(a * a + b * b), 0.0, 1.0);
    float phase = atan(b, a);              // -pi..pi
    float hue = (phase + 3.14159265) / 6.28318530;
    vec3 col = clamp(abs(fract(hue + vec3(0.0, 0.6667, 0.3333)) * 6.0 - 3.0) - 1.0, 0.0, 1.0);
    // density-weighted value: bright at moderate density, dark at vortex cores
    f_color = vec4(col * (0.35 + 0.65 * amp), 1.0);
}
''')


# ═══════════════════════════════════════════════
#  TYPED-UNIFORM SHADERS (named vars, no p1..p4)
# ═══════════════════════════════════════════════
#
# Each declares its variables via `uniforms=` — the node factory exposes them
# as real params (sliders / color pickers / dropdowns) AND wireable SCALAR
# ports. Bodies stay in the GL330/ES300-compatible parity subset.

_register("gradient_gpu2", "Gradient with typed controls (linear/radial/conic/diamond)",
          "procedural", '''
void main() {
    vec2 uv = v_uv;
    vec2 ctr = vec2(u_center_x, u_center_y);
    float a = radians(u_angle);
    vec2 dir = vec2(cos(a), sin(a));
    float t;
    if (u_mode == 1) {                       // radial
        t = length(uv - ctr) * 1.41421356;
    } else if (u_mode == 2) {                // conic
        vec2 d = uv - ctr;
        t = fract((atan(d.y, d.x) - a) / 6.28318530 + 1.0);
    } else if (u_mode == 3) {                // diamond
        vec2 d = abs(uv - ctr);
        t = (d.x + d.y) * 1.2;
    } else {                                 // linear
        t = dot(uv - ctr, dir) + 0.5;
    }
    t = clamp(t, 0.0, 1.0);
    if (u_bands > 1.5) t = floor(t * u_bands) / max(u_bands - 1.0, 1.0);  // posterized bands
    // Ordered-dither the ramp to hide 8-bit banding on smooth gradients.
    float dth = (hash21(gl_FragCoord.xy) - 0.5) * u_dither * 0.02;
    t = clamp(t + dth, 0.0, 1.0);
    f_color = vec4(mix(u_color_a, u_color_b, t), 1.0);
}
''', uniforms={
    "mode":     {"glsl": "choice", "choices": ["linear", "radial", "conic", "diamond"],
                 "default": "linear", "description": "gradient geometry"},
    "angle":    {"glsl": "float", "min": 0.0, "max": 360.0, "default": 0.0,
                 "description": "gradient angle (deg)"},
    "center_x": {"glsl": "float", "min": 0.0, "max": 1.0, "default": 0.5,
                 "description": "center X"},
    "center_y": {"glsl": "float", "min": 0.0, "max": 1.0, "default": 0.5,
                 "description": "center Y"},
    "color_a":  {"glsl": "color", "default": "#0b1026", "description": "start color"},
    "color_b":  {"glsl": "color", "default": "#4a9eff", "description": "end color"},
    "bands":    {"glsl": "float", "min": 0.0, "max": 32.0, "default": 0.0,
                 "description": "posterize bands (0 = smooth)"},
    "dither":   {"glsl": "float", "min": 0.0, "max": 1.0, "default": 0.25,
                 "description": "dither strength (hides banding)"},
})

_register("ascii_art_gpu", "ASCII-art the input image with a procedural bitmap font",
          "filter", '''
// 4x5 glyphs bit-packed in floats (movAX13h encoding): brightness ramp
// . : * o & 8 @ # — classic, WebGL-safe (float exp2/mod, no int precision).
float glyph_px(float n, vec2 p) {
    p = floor(p * vec2(-4.0, 4.0) + 2.5);
    if (clamp(p.x, 0.0, 4.0) == p.x && clamp(p.y, 0.0, 4.0) == p.y) {
        float k = p.x + 5.0 * p.y;
        if (int(mod(n / exp2(k), 2.0)) == 1) return 1.0;
    }
    return 0.0;
}

float glyph_for(float g) {
    float n = 0.0;                        // ' '
    if (g > 0.1) n = 4096.0;              // .
    if (g > 0.2) n = 65600.0;             // :
    if (g > 0.3) n = 332772.0;            // *
    if (g > 0.4) n = 15255086.0;          // o
    if (g > 0.5) n = 23385164.0;          // &
    if (g > 0.6) n = 15252014.0;          // 8
    if (g > 0.7) n = 13199452.0;          // @
    if (g > 0.8) n = 11512810.0;          // #
    return n;
}

void main() {
    float cell = max(u_cell_size, 4.0);
    vec2 cellOrigin = floor(gl_FragCoord.xy / cell) * cell;
    vec2 cellCenterUV = (cellOrigin + 0.5 * cell) / u_resolution;
    vec3 src = texture(u_texture, cellCenterUV).rgb;
    float g = dot(src, vec3(0.299, 0.587, 0.114));
    g = pow(clamp(g, 0.0, 1.0), max(u_gamma, 0.05));
    if (u_invert == 1) g = 1.0 - g;
    vec2 p = (gl_FragCoord.xy - cellOrigin) / cell * 2.0 - 1.0;   // [-1,1] in cell
    float px = glyph_px(glyph_for(g), p);
    vec3 col;
    if (u_mode == 1)      col = mix(u_bg_color, src, px);                    // colored
    else if (u_mode == 2) col = mix(vec3(0.0, 0.05, 0.0), vec3(0.2, 1.0, 0.3) * (0.4 + 0.6 * g), px); // terminal
    else                  col = mix(u_bg_color, u_fg_color, px);             // mono
    f_color = vec4(col, 1.0);
}
''', uniforms={
    "cell_size": {"glsl": "float", "min": 4.0, "max": 32.0, "default": 8.0,
                  "description": "character cell size (px)"},
    "mode":      {"glsl": "choice", "choices": ["mono", "colored", "terminal"],
                  "default": "colored", "description": "coloring mode"},
    "fg_color":  {"glsl": "color", "default": "#e8e8e8", "description": "glyph color (mono mode)"},
    "bg_color":  {"glsl": "color", "default": "#0a0a10", "description": "background color"},
    "invert":    {"glsl": "int", "min": 0, "max": 1, "default": 0,
                  "description": "invert brightness ramp"},
    "gamma":     {"glsl": "float", "min": 0.2, "max": 3.0, "default": 1.0,
                  "description": "brightness gamma before ramp"},
})

_register("solid_color_gpu", "Solid color fill (typed color picker)",
          "procedural", '''
void main() {
    f_color = vec4(u_color, 1.0);
}
''', uniforms={
    "color": {"glsl": "color", "default": "#4a9eff", "description": "fill color"},
})

_register("checker_gpu2", "Checkerboard with typed tile counts, colors, rotation",
          "procedural", '''
void main() {
    vec2 uv = v_uv - 0.5;
    float a = radians(u_angle);
    uv = rot(a) * uv + 0.5;
    vec2 tiles = vec2(max(u_tiles_x, 1.0), max(u_tiles_y, 1.0));
    vec2 cellPos = fract(uv * tiles);
    float chk = mod(floor(uv.x * tiles.x) + floor(uv.y * tiles.y), 2.0);
    vec3 col = mix(u_color_a, u_color_b, chk);
    // Optional grid lines between tiles.
    if (u_line_width > 0.001) {
        vec2 edge = min(cellPos, 1.0 - cellPos);
        float line = step(min(edge.x, edge.y), u_line_width * 0.5);
        col = mix(col, u_line_color, line);
    }
    f_color = vec4(col, 1.0);
}
''', uniforms={
    "tiles_x":    {"glsl": "float", "min": 1.0, "max": 64.0, "default": 8.0,
                   "description": "tiles across"},
    "tiles_y":    {"glsl": "float", "min": 1.0, "max": 64.0, "default": 8.0,
                   "description": "tiles down"},
    "angle":      {"glsl": "float", "min": 0.0, "max": 360.0, "default": 0.0,
                   "description": "rotation (deg)"},
    "color_a":    {"glsl": "color", "default": "#101018", "description": "tile color A"},
    "color_b":    {"glsl": "color", "default": "#e8e4d8", "description": "tile color B"},
    "line_width": {"glsl": "float", "min": 0.0, "max": 0.3, "default": 0.0,
                   "description": "grid line width (0 = none)"},
    "line_color": {"glsl": "color", "default": "#4a9eff", "description": "grid line color"},
})

_register("wave_pattern_gpu", "Periodic wave stripes: sine/triangle/square/saw, typed controls",
          "procedural", '''
void main() {
    vec2 uv = v_uv - 0.5;
    float a = radians(u_angle);
    float x = dot(uv, vec2(cos(a), sin(a))) * u_frequency + u_time * u_phase_speed;
    float ph = fract(x);
    float w;
    if (u_waveform == 1)      w = 1.0 - abs(ph * 2.0 - 1.0);          // triangle
    else if (u_waveform == 2) w = step(ph, clamp(u_duty, 0.01, 0.99)); // square
    else if (u_waveform == 3) w = ph;                                  // saw
    else                      w = 0.5 + 0.5 * sin(ph * 6.28318530);    // sine
    f_color = vec4(mix(u_color_a, u_color_b, w), 1.0);
}
''', uniforms={
    "waveform":    {"glsl": "choice", "choices": ["sine", "triangle", "square", "saw"],
                    "default": "sine", "description": "wave shape"},
    "frequency":   {"glsl": "float", "min": 0.5, "max": 64.0, "default": 8.0,
                    "description": "stripe frequency"},
    "angle":       {"glsl": "float", "min": 0.0, "max": 360.0, "default": 45.0,
                    "description": "stripe angle (deg)"},
    "phase_speed": {"glsl": "float", "min": -4.0, "max": 4.0, "default": 0.5,
                    "description": "phase drift speed (per second)"},
    "duty":        {"glsl": "float", "min": 0.0, "max": 1.0, "default": 0.5,
                    "description": "duty cycle (square wave)"},
    "color_a":     {"glsl": "color", "default": "#0b1026", "description": "trough color"},
    "color_b":     {"glsl": "color", "default": "#ff9d2e", "description": "crest color"},
})

_register("fbm_noise_gpu", "Fractal Brownian motion noise with typed octave controls",
          "procedural", '''
float fbm_typed(vec2 p) {
    float v = 0.0, amp = 0.5, freq = 1.0, norm = 0.0;
    for (int i = 0; i < 10; i++) {
        if (i >= u_octaves) break;
        v += amp * noise(p * freq);
        norm += amp;
        freq *= u_lacunarity;
        amp *= u_gain;
    }
    return norm > 0.0 ? v / norm : 0.0;
}

void main() {
    vec2 p = (v_uv - 0.5) * u_scale;
    p += u_time * u_drift * vec2(0.31, 0.17);
    if (u_warp > 0.001) {
        vec2 q = vec2(fbm_typed(p + vec2(5.2, 1.3)), fbm_typed(p + vec2(8.3, 2.8)));
        p += u_warp * 4.0 * (q - 0.5);
    }
    float t = clamp(fbm_typed(p), 0.0, 1.0);
    t = pow(t, max(u_contrast, 0.05));
    f_color = vec4(mix(u_color_a, u_color_b, t), 1.0);
}
''', uniforms={
    "scale":      {"glsl": "float", "min": 0.5, "max": 32.0, "default": 6.0,
                   "description": "noise scale"},
    "octaves":    {"glsl": "int", "min": 1, "max": 10, "default": 5,
                   "description": "fbm octaves"},
    "gain":       {"glsl": "float", "min": 0.1, "max": 0.9, "default": 0.5,
                   "description": "per-octave gain"},
    "lacunarity": {"glsl": "float", "min": 1.2, "max": 4.0, "default": 2.0,
                   "description": "per-octave frequency multiplier"},
    "warp":       {"glsl": "float", "min": 0.0, "max": 1.0, "default": 0.0,
                   "description": "domain warp amount"},
    "drift":      {"glsl": "float", "min": 0.0, "max": 2.0, "default": 0.2,
                   "description": "animation drift speed"},
    "contrast":   {"glsl": "float", "min": 0.2, "max": 3.0, "default": 1.0,
                   "description": "output contrast (gamma)"},
    "color_a":    {"glsl": "color", "default": "#06080f", "description": "low color"},
    "color_b":    {"glsl": "color", "default": "#d8e8ff", "description": "high color"},
})

# ── Typed-uniform nodes 226-231 (categorical coverage expansion) ──────
# Each declares its variables via `uniforms=` so the node factory exposes them
# as real params (sliders / color pickers / dropdowns) AND wireable SCALAR
# input ports, with data-typed outputs (image: IMAGE, luminance: FIELD). Bodies
# stay in the GL330/ES300-compatible parity subset (prologue helpers + no
# forbidden tokens).

_register("plasma_gpu2", "Animated plasma with typed scale/colors/warp",
          "procedural", '''
void main() {
    vec2 uv = v_uv;
    vec2 p = (uv - 0.5) * u_scale;
    // Slow, smooth multi-octave drift (no discrete cusps).
    float t = u_time * u_speed * 0.25;
    float v = sin(p.x * 6.0 + t) * cos(p.y * 4.0 + t * 0.7);
    v += sin(p.x * 11.0 - t * 1.2) * cos(p.y * 9.0 + t * 0.5) * 0.6;
    v += sin((p.x + p.y) * 16.0 + t * 0.3) * 0.3;
    if (u_warp > 0.001) {
        v += 0.4 * sin(length(p) * u_warp * 8.0 - t * 1.5);
    }
    v = v * 0.5 + 0.5;
    v = clamp(pow(v, max(u_contrast, 0.05)), 0.0, 1.0);
    f_color = vec4(mix(u_color_a, u_color_b, v), 1.0);
}
''', uniforms={
    "scale":    {"glsl": "float", "min": 0.5, "max": 16.0, "default": 4.0,
                 "description": "plasma spatial scale"},
    "speed":    {"glsl": "float", "min": 0.0, "max": 4.0, "default": 1.0,
                 "description": "animation speed"},
    "warp":     {"glsl": "float", "min": 0.0, "max": 1.0, "default": 0.0,
                 "description": "radial warp amount"},
    "contrast": {"glsl": "float", "min": 0.2, "max": 3.0, "default": 1.0,
                 "description": "output contrast (gamma)"},
    "color_a":  {"glsl": "color", "default": "#10071f", "description": "low color"},
    "color_b":  {"glsl": "color", "default": "#ffcf4d", "description": "high color"},
})

_register("voronoi_gpu2", "Voronoi/worley cellular cells with typed controls",
          "procedural", '''
vec2 _cell(vec2 g, float seed) {
    return g + 0.5 + 0.5 * vec2(
        sin(seed + 3.1 * g.x + 1.7 * g.y),
        cos(seed + 2.3 * g.x - 4.1 * g.y));
}
void main() {
    vec2 uv = v_uv * max(u_scale, 0.5);
    uv += u_time * u_drift * vec2(0.13, 0.07);
    float seed = u_seed * 6.2831;
    vec2 g = floor(uv), f = fract(uv);
    float d1 = 1e9, d2 = 1e9;
    for (int j = -1; j <= 1; j++)
    for (int i = -1; i <= 1; i++) {
        vec2 off = vec2(float(i), float(j));
        vec2 c = _cell(g + off, seed);
        float d = length(c - f);
        if (d < d1) { d2 = d1; d1 = d; } else if (d < d2) { d2 = d; }
    }
    float t = (u_metric == 1) ? (d2 - d1) : d1;   // F2-F1 edges vs nearest
    t = clamp(t * 1.6, 0.0, 1.0);
    if (u_cells > 0.5) t = step(0.5, t);           // hard cell regions
    vec3 col = mix(u_color_a, u_color_b, t);
    if (u_edge > 0.001) {
        float e = smoothstep(0.0, u_edge, abs(d1 - 0.5 * u_scale * 0.04));
        col = mix(u_edge_color, col, e);
    }
    f_color = vec4(col, 1.0);
}
''', uniforms={
    "scale":     {"glsl": "float", "min": 1.0, "max": 32.0, "default": 8.0,
                  "description": "cell density"},
    "seed":      {"glsl": "float", "min": 0.0, "max": 1.0, "default": 0.5,
                  "description": "cell layout seed"},
    "drift":     {"glsl": "float", "min": 0.0, "max": 2.0, "default": 0.1,
                  "description": "animation drift speed"},
    "metric":    {"glsl": "choice", "choices": ["nearest", "edges"],
                  "default": "nearest", "description": "distance metric"},
    "cells":     {"glsl": "int", "min": 0, "max": 1, "default": 0,
                  "description": "hard cell regions (0=smooth)"},
    "edge":      {"glsl": "float", "min": 0.0, "max": 0.5, "default": 0.0,
                  "description": "cell boundary line width"},
    "color_a":   {"glsl": "color", "default": "#0a0a12", "description": "cell color A"},
    "color_b":   {"glsl": "color", "default": "#37e0c8", "description": "cell color B"},
    "edge_color":{"glsl": "color", "default": "#ffffff", "description": "boundary color"},
})

_register("kaleidoscope_gpu", "Kaleidoscope mirror of the input image (typed)",
          "filter", '''
void main() {
    vec2 uv = v_uv - 0.5;
    float a = radians(u_angle) + u_time * u_spin;
    uv = rot(a) * uv;
    float seg = max(float(u_segments), 2.0);
    float ang = atan(uv.y, uv.x);
    float rad = length(uv);
    // Fold angle into one wedge, then mirror within the wedge.
    float wedge = 6.28318530 / seg;
    ang = mod(ang, wedge);
    ang = abs(ang - wedge * 0.5);
    vec2 p = vec2(cos(ang), sin(ang)) * rad + 0.5;
    vec3 src = texture(u_texture, fract(p)).rgb;
    f_color = vec4(mix(src, src * (0.6 + 0.8 * u_zoom), u_zoom), 1.0);
}
''', uniforms={
    "segments": {"glsl": "int", "min": 2, "max": 24, "default": 6,
                 "description": "mirror segments"},
    "angle":    {"glsl": "float", "min": 0.0, "max": 360.0, "default": 0.0,
                 "description": "base rotation (deg)"},
    "spin":     {"glsl": "float", "min": -2.0, "max": 2.0, "default": 0.0,
                 "description": "auto-rotation speed"},
    "zoom":     {"glsl": "float", "min": 0.0, "max": 1.0, "default": 0.0,
                 "description": "center zoom"},
})

_register("bloom_gpu", "Soft additive bloom / glow on the input (typed)",
          "filter", '''
vec3 _bloom_sample(vec2 uv, float r) {
    vec3 s = vec3(0.0);
    for (int k = 0; k < 8; k++) {
        float a = float(k) / 7.0 * 6.28318530;
        s += texture(u_texture, uv + vec2(cos(a), sin(a)) * r).rgb;
    }
    return s / 8.0;
}
void main() {
    vec2 uv = v_uv;
    vec3 src = texture(u_texture, uv).rgb;
    float r = u_radius * 0.03;
    // Two-pass cheap bloom (wide + tight) for a soft halo.
    vec3 glow = _bloom_sample(uv, r) * 0.6 + _bloom_sample(uv, r * 0.4) * 0.4;
    glow = pow(glow, vec3(max(u_threshold, 0.01)));   // emphasize bright areas
    vec3 col = mix(src, src + glow * u_strength * 1.6, clamp(u_strength, 0.0, 1.0));
    f_color = vec4(col, 1.0);
}
''', uniforms={
    "strength": {"glsl": "float", "min": 0.0, "max": 1.0, "default": 0.6,
                 "description": "glow strength"},
    "radius":   {"glsl": "float", "min": 0.0, "max": 1.0, "default": 0.4,
                 "description": "glow radius"},
    "threshold":{"glsl": "float", "min": 0.1, "max": 2.0, "default": 1.0,
                 "description": "brightness threshold (gamma)"},
})

_register("posterize_gpu", "Posterize / reduce color levels of input (typed)",
          "filter", '''
void main() {
    vec3 src = texture(u_texture, v_uv).rgb;
    float levels = max(float(u_levels), 2.0);
    vec3 q = floor(src * levels + 0.5) / levels;
    if (u_gamma != 1.0) q = pow(clamp(q, 0.0, 1.0), vec3(u_gamma));
    q = mix(src, q, clamp(u_amount, 0.0, 1.0));
    f_color = vec4(q, 1.0);
}
''', uniforms={
    "levels": {"glsl": "int", "min": 2, "max": 32, "default": 5,
               "description": "color levels per channel"},
    "amount": {"glsl": "float", "min": 0.0, "max": 1.0, "default": 1.0,
               "description": "effect amount"},
    "gamma":  {"glsl": "float", "min": 0.3, "max": 3.0, "default": 1.0,
               "description": "post-posterize gamma"},
})

_register("edge_gpu", "Sobel edge detection on the input (typed)",
          "filter", '''
float _lum(vec2 uv) {
    return dot(texture(u_texture, uv).rgb, vec3(0.299, 0.587, 0.114));
}
void main() {
    vec2 px = u_thickness / u_resolution;
    float tl = _lum(v_uv + px * vec2(-1.0,  1.0));
    float  l = _lum(v_uv + px * vec2(-1.0,  0.0));
    float bl = _lum(v_uv + px * vec2(-1.0, -1.0));
    float  t = _lum(v_uv + px * vec2( 0.0,  1.0));
    float  b = _lum(v_uv + px * vec2( 0.0, -1.0));
    float tr = _lum(v_uv + px * vec2( 1.0,  1.0));
    float  r = _lum(v_uv + px * vec2( 1.0,  0.0));
    float br = _lum(v_uv + px * vec2( 1.0, -1.0));
    float gx = -tl - 2.0 * l - bl + tr + 2.0 * r + br;
    float gy =  tl + 2.0 * t + tr - bl - 2.0 * b - br;
    float e = clamp(length(vec2(gx, gy)) * u_strength, 0.0, 1.0);
    vec3 col = mix(u_bg, u_edge, e);   // edges in u_edge over u_bg
    f_color = vec4(col, 1.0);
}
''', uniforms={
    "strength":   {"glsl": "float", "min": 0.2, "max": 4.0, "default": 1.5,
                   "description": "edge gain"},
    "thickness":  {"glsl": "float", "min": 0.5, "max": 6.0, "default": 1.5,
                   "description": "edge kernel thickness (px)"},
    "bg":         {"glsl": "color", "default": "#000000", "description": "background color"},
    "edge":       {"glsl": "color", "default": "#39ff88", "description": "edge color"},
})

# ── Typed-uniform nodes 232-237 (categorical coverage expansion, 2026-07-10) ──
# swirl displacement, chromatic aberration, halftone, concentric rings,
# truchet tiles, pixelate/mosaic. Same typed-uniform contract as 226-231:
# every variable is a real node param + wireable SCALAR port; filters take
# image_in: IMAGE. Bodies stay in the GL330/ES300 parity subset.

_register("swirl_gpu", "Swirl / vortex displacement of the input (typed)",
          "filter", '''
void main() {
    vec2 uv = v_uv - 0.5;
    float r = length(uv);
    float amt = (u_strength) * smoothstep(u_radius, 0.0, r);
    float a = amt + u_time * u_spin;
    uv = rot(a) * uv;
    vec3 src = texture(u_texture, fract(uv + 0.5)).rgb;
    f_color = vec4(src, 1.0);
}
''', uniforms={
    "strength": {"glsl": "float", "min": -6.0, "max": 6.0, "default": 3.0,
                 "description": "swirl strength (signed)"},
    "radius":   {"glsl": "float", "min": 0.1, "max": 1.2, "default": 0.6,
                 "description": "swirl falloff radius"},
    "spin":     {"glsl": "float", "min": 0.0, "max": 3.0, "default": 0.0,
                 "description": "animated spin speed"},
})

_register("chromatic_gpu", "Chromatic aberration RGB split of the input (typed)",
          "filter", '''
void main() {
    vec2 uv = v_uv;
    vec2 dir = (uv - 0.5);
    float amt = u_amount * 0.05;
    float ph = u_time * u_pulse;
    float k = amt * (1.0 + 0.3 * sin(ph));
    float rC = texture(u_texture, uv + dir * k).r;
    float gC = texture(u_texture, uv).g;
    float bC = texture(u_texture, uv - dir * k).b;
    f_color = vec4(rC, gC, bC, 1.0);
}
''', uniforms={
    "amount": {"glsl": "float", "min": 0.0, "max": 1.0, "default": 0.4,
               "description": "aberration amount"},
    "pulse":  {"glsl": "float", "min": 0.0, "max": 4.0, "default": 0.0,
               "description": "animated pulse speed"},
})

# ── Typed-uniform shims for CPU filter nodes 417 / 419 ──
# Mirrors node 417 (Chromatic Aberration) and node 419 (Thin-Film Interference)
# with NAMED typed uniforms that equal the CPU node's real params (contract #5),
# so the live preview tracks the sliders. The CPU numpy node stays authoritative
# for export (two-tier precision). Each uniform is verified live by
# test_typed_uniforms_drive_output (MAD >= 1.0 when perturbed to an extreme).
_register("chromatic_aberration_gpu", "Chromatic aberration RGB split (client-GPU twin of node 417)",
          "filter", '''
void main() {
    // Optical center can be nudged by center_drift (kept static here — the CPU
    // node only orbits it in spin mode); at the default 0.4 it sits at (0.5,0.5).
    vec2 ctr = vec2(0.5) + (u_center_drift - 0.4) * vec2(0.25, -0.15);
    vec2 d = v_uv - ctr;
    float rn = length(d);
    vec2 dir = d / max(rn, 1e-4);
    // Lateral split grows as r^curve (k=2 reproduces physical lateral CA).
    float amt = u_amount * 0.012;
    float k = amt * pow(rn, u_curve);
    // Optional barrel/pincushion radial distortion.
    float rbar = rn * (1.0 + u_barrel * rn * rn);
    float rR = rbar + k;          // R sampled outward
    float rB = rbar - k;          // B sampled inward
    float rC = texture(u_texture, ctr + dir * rR).r;
    float gC = texture(u_texture, v_uv).g;
    float bC = texture(u_texture, ctr + dir * rB).b;
    vec3 col = vec3(rC, gC, bC);
    col *= (1.0 - u_vignette * rn * rn);
    f_color = vec4(clamp(col, 0.0, 1.0), 1.0);
}
''', uniforms={
    "amount":       {"glsl": "float", "min": 0.0, "max": 60.0, "default": 20.0,
                    "description": "max lateral RGB split (px)"},
    "curve":        {"glsl": "float", "min": 1.0, "max": 4.0, "default": 2.0,
                    "description": "radial falloff exponent"},
    "barrel":       {"glsl": "float", "min": -0.4, "max": 0.4, "default": 0.0,
                    "description": "barrel/pincushion distortion"},
    "vignette":     {"glsl": "float", "min": 0.0, "max": 1.0, "default": 0.0,
                    "description": "edge darkening"},
    "center_drift": {"glsl": "float", "min": 0.0, "max": 1.0, "default": 0.4,
                    "description": "aberration-center offset"},
})

_register("lens_distort_gpu", "Lens Distortion — Brown–Conrady radial (barrel/pincushion) + chromatic split (client-GPU twin of node 480)",
          "filter", '''
void main() {
    vec2 ctr = vec2(u_center_x, u_center_y);
    vec2 d = v_uv - ctr;
    d.x *= u_aspect;
    float r = length(d);
    float r2 = r * r;
    float rd = r * (1.0 + u_amount * r2 + u_k2 * r2 * r2);
    vec2 dir = d / max(r, 1e-4);
    vec2 base = ctr + (dir * rd) / max(u_aspect, 1e-4);
    vec3 col = texture(u_texture, clamp(base, 0.0, 1.0)).rgb;
    if (u_chromatic > 0.0) {
        float k = u_chromatic * 0.02 * rd;
        float rR = texture(u_texture, clamp(ctr + (dir * (rd + k)) / max(u_aspect, 1e-4), 0.0, 1.0)).r;
        float bB = texture(u_texture, clamp(ctr + (dir * (rd - k)) / max(u_aspect, 1e-4), 0.0, 1.0)).b;
        col = vec3(rR, col.g, bB);
    }
    f_color = vec4(clamp(col, 0.0, 1.0), 1.0);
}
''', uniforms={
    "amount":     {"glsl": "float", "min": -0.6, "max": 0.6, "default": 0.25,
                   "description": "Brown–Conrady k1 radial distortion (barrel<0 / pincushion>0)"},
    "k2":         {"glsl": "float", "min": -0.3, "max": 0.3, "default": 0.0,
                   "description": "higher-order k2 radial term"},
    "center_x":   {"glsl": "float", "min": 0.0, "max": 1.0, "default": 0.5,
                   "description": "distortion centre X (uv)"},
    "center_y":   {"glsl": "float", "min": 0.0, "max": 1.0, "default": 0.5,
                   "description": "distortion centre Y (uv)"},
    "aspect":     {"glsl": "float", "min": 0.3, "max": 3.0, "default": 1.0,
                   "description": "aspect correction (elliptical distortion)"},
    "chromatic":  {"glsl": "float", "min": 0.0, "max": 1.0, "default": 0.0,
                   "description": "radial chromatic-aberration split"},
})

_register("thin_film_gpu", "Thin-film interference iridescence (client-GPU twin of node 419)",
          "filter", '''
void main() {
    // Radial thickness field (matches the CPU node's default 'radial' source):
    // d grows from the frame center outward, so the iridescent bands form a
    // soap-bubble / oil-slick ring pattern over the wired substrate.
    vec2 p = v_uv - 0.5;
    float r = length(p) * 1.4;
    // Live-preview animation: advance the radial thickness with the preview
    // clock u_time so the iridescent bands drift (mirrors the CPU node's
    // anim_mode/time). The client feeds u_time every frame.
    float d = u_thickness + u_thickness_range * (r + 0.06 * sin(u_time * 0.6));
    float ang = radians(u_angle);
    float sin_a = sin(ang);
    float c = sin_a / max(u_ior, 1.001);
    float cos_t = sqrt(max(0.0, 1.0 - c * c));
    // Optical path difference (nm); per-wavelength reflectance via R(λ)=cos².
    float opd = 2.0 * u_ior * d * cos_t;
    vec3 lam = vec3(650.0, 550.0, 450.0);
    vec3 phase = (6.2831853 * opd / lam) + 3.14159265;
    vec3 iri = (1.0 - cos(phase)) * 0.5;
    // Saturation control around the band luminance.
    float lum = dot(iri, vec3(0.3333333));
    iri = clamp(lum + u_saturation * (iri - lum), 0.0, 1.0);
    vec3 src = texture(u_texture, v_uv).rgb;
    vec3 col = mix(src, iri, u_strength);
    f_color = vec4(clamp(col, 0.0, 1.0), 1.0);
}
''', uniforms={
    "thickness":        {"glsl": "float", "min": 100.0, "max": 1200.0, "default": 380.0,
                        "description": "base film thickness (nm)"},
    "thickness_range":  {"glsl": "float", "min": 0.0, "max": 1200.0, "default": 320.0,
                        "description": "thickness variation (nm)"},
    "ior":              {"glsl": "float", "min": 1.0, "max": 2.5, "default": 1.33,
                        "description": "film refractive index"},
    "angle":            {"glsl": "float", "min": 0.0, "max": 80.0, "default": 0.0,
                        "description": "incidence angle (deg)"},
    "strength":         {"glsl": "float", "min": 0.0, "max": 1.0, "default": 1.0,
                        "description": "overlay blend over source"},
    "saturation":       {"glsl": "float", "min": 0.0, "max": 1.5, "default": 1.0,
                        "description": "band color saturation"},
})

_register("halftone_gpu", "Halftone dot-screen of the input (typed)",
          "filter", '''
void main() {
    vec2 uv = v_uv;
    float ang = radians(u_angle);
    vec2 rp = rot(ang) * (uv - 0.5) + 0.5;
    float scale = max(u_scale, 4.0);
    vec2 cell = fract(rp * scale) - 0.5;
    float d = length(cell);
    float lum = dot(texture(u_texture, uv).rgb, vec3(0.299, 0.587, 0.114));
    float radius = (1.0 - lum) * 0.7 * u_dot;
    float dot_ = smoothstep(radius, radius - 0.08, d);
    vec3 col = mix(u_bg, u_ink, dot_);
    f_color = vec4(col, 1.0);
}
''', uniforms={
    "scale": {"glsl": "float", "min": 8.0, "max": 120.0, "default": 48.0,
              "description": "dot grid density"},
    "angle": {"glsl": "float", "min": 0.0, "max": 90.0, "default": 15.0,
              "description": "screen angle (deg)"},
    "dot":   {"glsl": "float", "min": 0.2, "max": 2.0, "default": 1.0,
              "description": "dot size multiplier"},
    "bg":    {"glsl": "color", "default": "#ffffff", "description": "paper color"},
    "ink":   {"glsl": "color", "default": "#101010", "description": "ink color"},
})

# ── Typed filter twins for CPU filter nodes (categorical GPU coverage) ──
# Each is a closed-form per-pixel approximation of the CPU node it shadows;
# the CPU numpy node stays authoritative for export (two-tier precision).
# Uniform names equal the CPU node's real numeric params (contract #5) so the
# browser live preview tracks the sliders. Choice/string params (source,
# palette, anim_mode, mode, paper, ink, aperture_shape) are intentionally
# unmapped (pitfall #14) — the twin filters whatever image is wired in.
_register("bloom_glow_gpu", "Bloom / glow with optional anamorphic streak (typed twin of node 408)",
          "filter", '''
void main() {
    vec3 src = texture(u_texture, v_uv).rgb;
    float thr = u_threshold;
    float knee = max(thr * u_softness, 0.001);
    vec2 px = 1.0 / u_resolution;
    float r = max(u_radius, 1.0);
    vec3 glow = vec3(0.0);
    float wsum = 0.0;
    const int N = 16;
    float ga = 2.39996323;
    // Golden-angle disc sampling of the bright-pass = a cheap single-pass glow.
    for (int i = 0; i < N; i++) {
        float fi = float(i);
        float rad = sqrt((fi + 0.5) / float(N)) * r;
        float ang = fi * ga;
        vec2 off = vec2(cos(ang) * u_streak, sin(ang)) * rad;
        vec3 s = texture(u_texture, v_uv + off * px).rgb;
        float l = dot(s, vec3(0.2126, 0.7152, 0.0722));
        float f = clamp((l - (thr - knee)) / (2.0 * knee), 0.0, 1.0);
        f = f * f;
        glow += s * f;
        wsum += max(f, 0.0001);
    }
    glow /= max(wsum, 0.001);
    vec3 col = src + u_intensity * glow;
    f_color = vec4(clamp(col, 0.0, 1.0), 1.0);
}
''', uniforms={
    "threshold": {"glsl": "float", "min": 0.0, "max": 1.0, "default": 0.5,
                  "description": "brightness cutoff for the bloom prefilter"},
    "softness":  {"glsl": "float", "min": 0.0, "max": 1.0, "default": 0.5,
                  "description": "soft-knee width as fraction of threshold"},
    "intensity": {"glsl": "float", "min": 0.0, "max": 3.0, "default": 1.2,
                  "description": "glow additive strength"},
    "radius":    {"glsl": "float", "min": 1.0, "max": 40.0, "default": 12.0,
                  "description": "blur radius in px (glow spread)"},
    "streak":    {"glsl": "float", "min": 1.0, "max": 8.0, "default": 1.0,
                  "description": "anamorphic streak anisotropy (1=round)"},
})

_register("bokeh_gpu", "Bokeh lens blur with shaped aperture (typed twin of node 420)",
          "filter", '''
void main() {
    vec3 src = texture(u_texture, v_uv).rgb;
    vec2 px = 1.0 / u_resolution;
    float R = max(u_radius, 1.0);
    const int N = 24;
    float ga = 2.39996323;
    float rot_ang = radians(u_rotation);
    mat2 ROT = rot(rot_ang);
    float seg = 3.14159265 / max(u_blades, 3.0);
    vec3 acc = vec3(0.0);
    float wsum = 0.0;
    // Disc sampling weighted by a regular-N-gon aperture SDF (the iris shape);
    // horizontal anamorphic stretch bakes the cinematic streak into highlights.
    for (int i = 0; i < N; i++) {
        float fi = float(i);
        float rad = sqrt((fi + 0.5) / float(N)) * R;
        float ang = fi * ga;
        vec2 off = vec2(cos(ang) * u_anamorphic, sin(ang)) * rad;
        vec2 pr = ROT * off;
        float a = atan(pr.y, pr.x);
        float rr = length(pr);
        float m = mod(a + 3.14159265, 2.0 * seg) - seg;
        float edge = R * cos(seg - abs(m));
        float w = (rr <= edge) ? 1.0 : 0.0;
        vec3 s = texture(u_texture, v_uv + off * px).rgb;
        acc += s * w;
        wsum += w;
    }
    vec3 blurred = acc / max(wsum, 0.001);
    float lum = dot(src, vec3(0.2126, 0.7152, 0.0722));
    vec3 hot = src * smoothstep(0.6, 1.0, lum) * u_highlight;
    vec3 col = (blurred + hot) * u_brightness;
    f_color = vec4(clamp(col, 0.0, 1.0), 1.0);
}
''', uniforms={
    "radius":     {"glsl": "float", "min": 2.0, "max": 48.0, "default": 16.0,
                   "description": "bokeh radius in px (defocus amount)"},
    "blades":     {"glsl": "float", "min": 3.0, "max": 12.0, "default": 6.0,
                   "description": "iris blade count (polygon sides)"},
    "anamorphic": {"glsl": "float", "min": 1.0, "max": 8.0, "default": 4.0,
                   "description": "horizontal streak stretch"},
    "rotation":   {"glsl": "float", "min": 0.0, "max": 360.0, "default": 0.0,
                   "description": "aperture rotation (deg)"},
    "brightness": {"glsl": "float", "min": 0.2, "max": 2.5, "default": 1.0,
                   "description": "output brightness gain"},
    "highlight":  {"glsl": "float", "min": 0.0, "max": 1.0, "default": 0.35,
                   "description": "re-add hot cores to out-of-focus lights"},
})

_register("bilateral_grid_gpu", "Edge-preserving bilateral smoothing (typed twin of node 345)",
          "filter", '''
void main() {
    vec3 center = texture(u_texture, v_uv).rgb;
    vec2 px = 1.0 / u_resolution;
    float R = max(u_sigma_s, 0.5) * max(u_grid_scale, 1.0);
    const int N = 24;
    float ga = 2.39996323;
    float invR2 = 1.0 / (2.0 * max(u_sigma_r, 0.5) * max(u_sigma_r, 0.5));
    float sp2 = max(R * R * 0.25, 1.0);
    vec3 acc = vec3(0.0);
    float wsum = 0.0;
    // Joint bilateral: weight neighbours by BOTH spatial Gaussian and range
    // (color) similarity to the center, so smooth regions melt while silhouettes
    // survive. A genuine single-pass approximation of the bilateral grid.
    for (int i = 0; i < N; i++) {
        float fi = float(i);
        float rad = sqrt((fi + 0.5) / float(N)) * R;
        float ang = fi * ga;
        vec2 off = vec2(cos(ang), sin(ang)) * rad;
        vec3 s = texture(u_texture, v_uv + off * px).rgb;
        float ws = exp(-(rad * rad) / sp2);
        float dc = distance(s, center);
        float wr = exp(-(dc * dc) * invR2);
        float w = ws * wr;
        acc += s * w;
        wsum += w;
    }
    vec3 bilat = acc / max(wsum, 0.001);
    vec3 col = mix(bilat, center, u_blend);
    f_color = vec4(clamp(col, 0.0, 1.0), 1.0);
}
''', uniforms={
    "grid_scale": {"glsl": "float", "min": 1.0, "max": 8.0, "default": 4.0,
                   "description": "spatial cell size in px (smoother when larger)"},
    "sigma_s":    {"glsl": "float", "min": 0.5, "max": 8.0, "default": 2.0,
                   "description": "spatial blur radius in grid cells"},
    "sigma_r":    {"glsl": "float", "min": 0.5, "max": 8.0, "default": 2.0,
                   "description": "range (intensity) blur radius — smaller = sharper edges"},
    "blend":      {"glsl": "float", "min": 0.0, "max": 1.0, "default": 0.0,
                   "description": "blend original source back in (1=original)"},
})

_register("rings_gpu", "Concentric animated rings (typed procedural)",
          "procedural", '''
void main() {
    vec2 uv = (v_uv - 0.5);
    uv.x *= u_resolution.x / u_resolution.y;
    float r = length(uv) * u_freq;
    float t = u_time * u_speed;
    float w = 0.5 + 0.5 * sin(r * 6.28318530 - t * 2.0);
    w = pow(w, max(u_sharp, 0.05));
    vec3 col = mix(u_color_a, u_color_b, w);
    f_color = vec4(col, 1.0);
}
''', uniforms={
    "freq":  {"glsl": "float", "min": 1.0, "max": 40.0, "default": 10.0,
              "description": "ring frequency"},
    "speed": {"glsl": "float", "min": 0.0, "max": 4.0, "default": 1.0,
              "description": "animation speed"},
    "sharp": {"glsl": "float", "min": 0.2, "max": 6.0, "default": 1.0,
              "description": "ring edge sharpness"},
    "color_a": {"glsl": "color", "default": "#05070f", "description": "trough color"},
    "color_b": {"glsl": "color", "default": "#4de0ff", "description": "crest color"},
})

_register("truchet_gpu", "Truchet arc tiling (typed procedural)",
          "procedural", '''
void main() {
    vec2 uv = v_uv * max(u_scale, 1.0);
    uv += u_time * u_drift * vec2(0.1, 0.06);
    vec2 g = floor(uv), f = fract(uv);
    float flip = step(0.5, hash21(g));
    if (flip > 0.5) f.x = 1.0 - f.x;
    float d1 = length(f - vec2(0.0, 0.0));
    float d2 = length(f - vec2(1.0, 1.0));
    float lw = u_width * 0.5;
    float arc = min(abs(d1 - 0.5), abs(d2 - 0.5));
    float line = smoothstep(lw, lw - 0.06, arc);
    vec3 col = mix(u_bg, u_fg, line);
    f_color = vec4(col, 1.0);
}
''', uniforms={
    "scale": {"glsl": "float", "min": 2.0, "max": 40.0, "default": 10.0,
              "description": "tile density"},
    "width": {"glsl": "float", "min": 0.05, "max": 0.6, "default": 0.25,
              "description": "arc line width"},
    "drift": {"glsl": "float", "min": 0.0, "max": 2.0, "default": 0.0,
              "description": "animated drift"},
    "bg":    {"glsl": "color", "default": "#0a0a14", "description": "background"},
    "fg":    {"glsl": "color", "default": "#ffd166", "description": "arc color"},
})

_register("pixelate_gpu", "Pixelate / mosaic of the input (typed)",
          "filter", '''
void main() {
    float cells = max(float(u_cells), 2.0);
    vec2 grid = vec2(cells, cells * u_resolution.y / u_resolution.x);
    vec2 uv = (floor(v_uv * grid) + 0.5) / grid;
    vec3 src = texture(u_texture, uv).rgb;
    if (u_levels > 1.5) {
        float lv = float(u_levels);
        src = floor(src * lv + 0.5) / lv;
    }
    f_color = vec4(src, 1.0);
}
''', uniforms={
    "cells":  {"glsl": "float", "min": 4.0, "max": 200.0, "default": 48.0,
               "description": "mosaic cell count (x)"},
    "levels": {"glsl": "int", "min": 1, "max": 32, "default": 1,
               "description": "color quantize levels (1=off)"},
})

# ── Typed escape-time fractal nodes (ids 238-243) ───────────────────────
# Categorical coverage for the signature fractal family (Mandelbrot / Julia /
# Burning Ship / Newton / Sierpinski / Lyapunov). These expose NAMED, typed
# controls (zoom, center, iteration count, palette, colors) + wireable SCALAR
# ports, replacing the opaque p1..p4 shims for these nodes. CPU fns stay
# authoritative; these are an additive typed-uniform convenience layer.
_TYPED_FRACTAL_HELPERS = _FRACTAL_HELPERS + '''
vec3 inferno_l(float t){
    t = clamp(t, 0.0, 1.0);
    const vec3 c0 = vec3(0.00021894, 0.00016488, -0.01907227);
    const vec3 c1 = vec3(0.10651034, 0.56396050, 3.93279110);
    const vec3 c2 = vec3(11.6028830, -3.9781129, -15.9420510);
    const vec3 c3 = vec3(-41.703996, 17.4360890, 44.3541450);
    const vec3 c4 = vec3(77.1629350, -33.402243, -81.8094230);
    const vec3 c5 = vec3(-71.319421, 32.6260640, 73.2095190);
    const vec3 c6 = vec3(25.1311300, -12.242810, -23.0709590);
    return c0 + t*(c1 + t*(c2 + t*(c3 + t*(c4 + t*(c5 + t*c6)))));
}
vec3 _fractalColor(float t, int mode, vec3 ca, vec3 cb, float shift){
    if (mode == 1) return inferno_l(t);
    if (mode == 2) return mix(ca, cb, clamp(t, 0.0, 1.0));
    return fractal_palette(t + shift);
}
'''

_register("mandelbrot_typed", "Mandelbrot set with typed zoom/center/iter/palette (node 238)",
          "procedural", _TYPED_FRACTAL_HELPERS + '''
void main() {
    vec2 uv = (gl_FragCoord.xy - 0.5 * u_resolution) / min(u_resolution.x, u_resolution.y);
    vec2 c = vec2(u_center_x, u_center_y) + uv * (3.0 / max(u_zoom, 0.001));
    vec2 z = vec2(0.0);
    float n = 0.0; float last2 = 0.0;
    const int CAP = 500;
    for (int i = 0; i < CAP; i++) {
        z = vec2(z.x*z.x - z.y*z.y, 2.0*z.x*z.y) + c;
        last2 = dot(z, z); n += 1.0;
        if (last2 > 16.0 || n >= float(u_max_iter)) break;
    }
    float t = (n >= float(u_max_iter) - 0.5) ? 1.0 : smooth_iter(n, last2, float(u_max_iter));
    f_color = vec4(_fractalColor(t, u_palette, u_color_a, u_color_b, u_color_shift), 1.0);
}
''', uniforms={
    "zoom":       {"glsl": "float", "min": 0.01, "max": 8.0, "default": 1.0,
                   "description": "zoom (1 = full view)"},
    "center_x":   {"glsl": "float", "min": -2.0, "max": 0.5, "default": -0.5,
                   "description": "center X"},
    "center_y":   {"glsl": "float", "min": -1.5, "max": 1.5, "default": 0.0,
                   "description": "center Y"},
    "max_iter":   {"glsl": "int", "min": 20, "max": 500, "default": 200,
                   "description": "max iterations"},
    "palette":    {"glsl": "choice", "choices": ["sine", "inferno", "grayscale"],
                   "default": "sine", "description": "color palette"},
    "color_shift": {"glsl": "float", "min": 0.0, "max": 1.0, "default": 0.0,
                    "description": "palette shift"},
    "color_a":    {"glsl": "color", "default": "#05010a",
                   "description": "color A (grayscale / holes)"},
    "color_b":    {"glsl": "color", "default": "#ffd166",
                   "description": "color B (grayscale)"},
})

_register("julia_typed", "Julia set with typed c/zoom/iter/palette (node 239)",
          "procedural", _TYPED_FRACTAL_HELPERS + '''
void main() {
    vec2 uv = (gl_FragCoord.xy - 0.5 * u_resolution) / min(u_resolution.x, u_resolution.y);
    vec2 c = vec2(u_cx, u_cy);
    vec2 z = uv * (3.0 / max(u_zoom, 0.001));
    float n = 0.0; float last2 = 0.0;
    const int CAP = 500;
    for (int i = 0; i < CAP; i++) {
        z = vec2(z.x*z.x - z.y*z.y, 2.0*z.x*z.y) + c;
        last2 = dot(z, z); n += 1.0;
        if (last2 > 16.0 || n >= float(u_max_iter)) break;
    }
    float t = (n >= float(u_max_iter) - 0.5) ? 1.0 : smooth_iter(n, last2, float(u_max_iter));
    f_color = vec4(_fractalColor(t, u_palette, u_color_a, u_color_b, u_color_shift), 1.0);
}
''', uniforms={
    "cx":         {"glsl": "float", "min": -1.0, "max": 1.0, "default": -0.7269,
                   "description": "Julia c (real)"},
    "cy":         {"glsl": "float", "min": -1.0, "max": 1.0, "default": 0.1889,
                   "description": "Julia c (imag)"},
    "zoom":       {"glsl": "float", "min": 0.01, "max": 8.0, "default": 1.0,
                   "description": "zoom (1 = full view)"},
    "max_iter":   {"glsl": "int", "min": 20, "max": 500, "default": 200,
                   "description": "max iterations"},
    "palette":    {"glsl": "choice", "choices": ["sine", "inferno", "grayscale"],
                   "default": "sine", "description": "color palette"},
    "color_shift": {"glsl": "float", "min": 0.0, "max": 1.0, "default": 0.0,
                    "description": "palette shift"},
    "color_a":    {"glsl": "color", "default": "#05010a",
                   "description": "color A (grayscale / holes)"},
    "color_b":    {"glsl": "color", "default": "#ffd166",
                   "description": "color B (grayscale)"},
})

_register("burning_ship_typed", "Burning Ship set with typed zoom/center/iter/palette (node 240)",
          "procedural", _TYPED_FRACTAL_HELPERS + '''
void main() {
    vec2 uv = (gl_FragCoord.xy - 0.5 * u_resolution) / min(u_resolution.x, u_resolution.y);
    uv.y = -uv.y;
    vec2 c = vec2(u_center_x, u_center_y) + uv * (3.0 / max(u_zoom, 0.001));
    vec2 z = vec2(0.0);
    float n = 0.0; float last2 = 0.0;
    const int CAP = 500;
    for (int i = 0; i < CAP; i++) {
        z = vec2(abs(z.x), abs(z.y));
        z = vec2(z.x*z.x - z.y*z.y, 2.0*z.x*z.y) + c;
        last2 = dot(z, z); n += 1.0;
        if (last2 > 16.0 || n >= float(u_max_iter)) break;
    }
    float t = (n >= float(u_max_iter) - 0.5) ? 1.0 : smooth_iter(n, last2, float(u_max_iter));
    f_color = vec4(_fractalColor(t, u_palette, u_color_a, u_color_b, u_color_shift), 1.0);
}
''', uniforms={
    "zoom":       {"glsl": "float", "min": 0.01, "max": 8.0, "default": 1.0,
                   "description": "zoom (1 = full view)"},
    "center_x":   {"glsl": "float", "min": -2.0, "max": 0.5, "default": -0.5,
                   "description": "center X"},
    "center_y":   {"glsl": "float", "min": -2.0, "max": 0.5, "default": -0.5,
                   "description": "center Y"},
    "max_iter":   {"glsl": "int", "min": 20, "max": 500, "default": 200,
                   "description": "max iterations"},
    "palette":    {"glsl": "choice", "choices": ["sine", "inferno", "grayscale"],
                   "default": "sine", "description": "color palette"},
    "color_shift": {"glsl": "float", "min": 0.0, "max": 1.0, "default": 0.0,
                    "description": "palette shift"},
    "color_a":    {"glsl": "color", "default": "#05010a",
                   "description": "color A (grayscale / holes)"},
    "color_b":    {"glsl": "color", "default": "#ffd166",
                   "description": "color B (grayscale)"},
})

_register("newton_typed", "Newton fractal (z^3-1) basins with typed zoom/palette (node 241)",
          "procedural", _TYPED_FRACTAL_HELPERS + '''
void main() {
    vec2 uv = (gl_FragCoord.xy - 0.5 * u_resolution) / min(u_resolution.x, u_resolution.y);
    vec2 z = uv * (2.2 / max(u_zoom, 0.001));
    float n = 0.0;
    const int CAP = 80;
    for (int i = 0; i < CAP; i++) {
        vec2 z2 = vec2(z.x*z.x - z.y*z.y, 2.0*z.x*z.y);
        vec2 z3 = vec2(z2.x*z.x - z2.y*z.y, 2.0*z2.x*z.y);
        vec2 f = z3 - vec2(1.0, 0.0);
        vec2 dz = 3.0 * z2;
        float denom = dz.x*dz.x + dz.y*dz.y + 1e-8;
        vec2 stp = vec2(f.x*dz.x + f.y*dz.y, f.y*dz.x - f.x*dz.y) / denom;
        z -= stp; n += 1.0;
        if (dot(stp, stp) < 1e-6) break;
    }
    float ang = atan(z.y, z.x);
    float root = floor((ang + 3.14159) / (2.0 * 3.14159 / 3.0));
    float t = mod(root / 3.0 + u_color_offset + 0.15 * n / 80.0, 1.0);
    vec3 col = (u_palette == 1) ? inferno_l(t * (0.6 + 0.4 * u_color_speed))
              : (u_palette == 2) ? mix(u_color_a, u_color_b, clamp(t, 0.0, 1.0))
              : fractal_palette(t * (0.6 + 0.4 * u_color_speed));
    f_color = vec4(col, 1.0);
}
''', uniforms={
    "zoom":         {"glsl": "float", "min": 0.01, "max": 8.0, "default": 1.0,
                     "description": "zoom (1 = full view)"},
    "color_speed":  {"glsl": "float", "min": 0.0, "max": 2.0, "default": 1.0,
                     "description": "color cycling speed"},
    "color_offset": {"glsl": "float", "min": 0.0, "max": 1.0, "default": 0.0,
                     "description": "color offset"},
    "palette":      {"glsl": "choice", "choices": ["sine", "inferno", "grayscale"],
                     "default": "sine", "description": "color palette"},
    "color_a":      {"glsl": "color", "default": "#05010a",
                     "description": "color A (grayscale)"},
    "color_b":      {"glsl": "color", "default": "#ffd166",
                     "description": "color B (grayscale)"},
})

_register("sierpinski_typed", "Sierpinski carpet with typed depth/palette (node 242)",
          "procedural", _TYPED_FRACTAL_HELPERS + '''
void main() {
    vec2 p = v_uv;
    float depth = clamp(floor(u_depth), 1.0, 7.0);
    float hole = 0.0;
    for (float i = 0.0; i < 7.0; i += 1.0) {
        if (i >= depth) break;
        vec2 cell = floor(p * 3.0);
        if (cell.x == 1.0 && cell.y == 1.0) { hole = 1.0; break; }
        p = fract(p * 3.0);
    }
    float t = fract(0.15 * depth + u_color_shift + 0.3 * v_uv.x + 0.2 * v_uv.y);
    vec3 col = (hole > 0.5) ? u_color_a
             : _fractalColor(t, u_palette, u_color_a, u_color_b, u_color_shift);
    f_color = vec4(col, 1.0);
}
''', uniforms={
    "depth":       {"glsl": "int", "min": 1, "max": 7, "default": 4,
                    "description": "subdivision depth"},
    "palette":     {"glsl": "choice", "choices": ["sine", "inferno", "grayscale"],
                    "default": "sine", "description": "color palette"},
    "color_shift": {"glsl": "float", "min": 0.0, "max": 1.0, "default": 0.0,
                    "description": "palette shift"},
    "color_a":     {"glsl": "color", "default": "#0a0a12",
                    "description": "hole color"},
    "color_b":     {"glsl": "color", "default": "#ffd166",
                    "description": "color B (grayscale)"},
})

_register("lyapunov_typed", "Lyapunov exponent map with typed r-range/palette (node 243)",
          "procedural", _TYPED_FRACTAL_HELPERS + '''
void main() {
    vec2 uv = v_uv;
    float rx = mix(u_r_min, u_r_max, uv.x);
    float ry = mix(u_r_min, u_r_max, uv.y);
    float lambda = 0.0; float x = 0.5;
    const float WARM = 30.0; const float MEAS = 120.0;
    for (float i = 0.0; i < (WARM + MEAS); i += 1.0) {
        int k = int(mod(i, 8.0));
        float rk = (k == 0 || k == 2 || k == 4 || k == 6) ? rx : ry;
        float deriv = rk * (1.0 - 2.0 * x);
        x = rk * x * (1.0 - x);
        if (i >= WARM) lambda += log(abs(deriv) + 1e-8);
    }
    lambda /= MEAS;
    float t = clamp(0.5 + 0.5 * lambda / 2.0, 0.0, 1.0);
    t = fract(t + u_color_shift);
    vec3 col = (u_palette == 2) ? mix(u_color_a, u_color_b, t)
              : (u_palette == 1) ? inferno_l(t)
              : fractal_palette(t);
    f_color = vec4(col, 1.0);
}
''', uniforms={
    "r_min":       {"glsl": "float", "min": 0.0, "max": 4.0, "default": 2.5,
                    "description": "r min (AB row)"},
    "r_max":       {"glsl": "float", "min": 0.0, "max": 4.0, "default": 4.0,
                    "description": "r max (AB row)"},
    "palette":     {"glsl": "choice", "choices": ["sine", "inferno", "grayscale"],
                    "default": "sine", "description": "color palette"},
    "color_shift": {"glsl": "float", "min": 0.0, "max": 1.0, "default": 0.0,
                    "description": "palette shift"},
    "color_a":     {"glsl": "color", "default": "#05010a",
                    "description": "color A (grayscale)"},
    "color_b":     {"glsl": "color", "default": "#ffd166",
                    "description": "color B (grayscale)"},
})

# ── Typed filter / color-grade nodes (ids 244-249) ──────────────────────
# Categorical coverage pt.4 (2026-07-11): the per-pixel filter / color-grade
# family with NAMED typed controls + wireable SCALAR ports — box blur, unsharp
# sharpen, vignette, luminance threshold, hue rotate, and ordered dither.
# Filters take image_in: IMAGE. CPU fns stay authoritative; additive layer.

_register("box_blur_gpu", "Box blur of the input (typed radius/samples)",
          "filter", '''
void main() {
    vec2 px = u_radius / u_resolution;
    int n = int(clamp(float(u_samples), 1.0, 6.0));
    vec3 acc = vec3(0.0); float wsum = 0.0;
    for (int j = -6; j <= 6; j++) {
        if (j < -n || j > n) continue;
        for (int i = -6; i <= 6; i++) {
            if (i < -n || i > n) continue;
            acc += texture(u_texture, v_uv + px * vec2(float(i), float(j))).rgb;
            wsum += 1.0;
        }
    }
    vec3 blurred = acc / max(wsum, 1.0);
    vec3 src = texture(u_texture, v_uv).rgb;
    f_color = vec4(mix(src, blurred, clamp(u_amount, 0.0, 1.0)), 1.0);
}
''', uniforms={
    "radius":  {"glsl": "float", "min": 0.5, "max": 12.0, "default": 2.0,
                "description": "sample spacing (px)"},
    "samples": {"glsl": "int", "min": 1, "max": 6, "default": 3,
                "description": "kernel half-width"},
    "amount":  {"glsl": "float", "min": 0.0, "max": 1.0, "default": 1.0,
                "description": "blend amount"},
})

_register("sharpen_gpu", "Unsharp-mask sharpen of the input (typed)",
          "filter", '''
void main() {
    vec2 px = u_radius / u_resolution;
    vec3 c  = texture(u_texture, v_uv).rgb;
    vec3 nb = texture(u_texture, v_uv + px * vec2( 0.0,  1.0)).rgb
            + texture(u_texture, v_uv + px * vec2( 0.0, -1.0)).rgb
            + texture(u_texture, v_uv + px * vec2( 1.0,  0.0)).rgb
            + texture(u_texture, v_uv + px * vec2(-1.0,  0.0)).rgb;
    vec3 blur = nb * 0.25;
    vec3 sharp = c + (c - blur) * u_strength;
    f_color = vec4(clamp(sharp, 0.0, 1.0), 1.0);
}
''', uniforms={
    "strength": {"glsl": "float", "min": 0.0, "max": 5.0, "default": 1.5,
                 "description": "sharpen strength"},
    "radius":   {"glsl": "float", "min": 0.5, "max": 6.0, "default": 1.0,
                 "description": "sample radius (px)"},
})

_register("vignette_gpu", "Vignette darkening of the input (typed)",
          "filter", '''
void main() {
    vec3 src = texture(u_texture, v_uv).rgb;
    vec2 d = v_uv - 0.5;
    d.x *= u_resolution.x / u_resolution.y;
    float r = length(d) * 1.41421356;
    float v = smoothstep(u_outer, u_inner, r);
    v = mix(1.0, v, clamp(u_amount, 0.0, 1.0));
    f_color = vec4(mix(u_color, src, v), 1.0);
}
''', uniforms={
    "inner":  {"glsl": "float", "min": 0.0, "max": 1.0, "default": 0.3,
               "description": "inner (bright) radius"},
    "outer":  {"glsl": "float", "min": 0.2, "max": 1.6, "default": 1.0,
               "description": "outer (dark) radius"},
    "amount": {"glsl": "float", "min": 0.0, "max": 1.0, "default": 0.8,
               "description": "vignette amount"},
    "color":  {"glsl": "color", "default": "#000000", "description": "vignette color"},
})

_register("threshold_gpu", "Luminance threshold / two-tone of the input (typed)",
          "filter", '''
void main() {
    vec3 src = texture(u_texture, v_uv).rgb;
    float l = dot(src, vec3(0.299, 0.587, 0.114));
    float e = smoothstep(u_threshold - u_softness, u_threshold + u_softness, l);
    vec3 col = mix(u_low, u_high, e);
    f_color = vec4(col, 1.0);
}
''', uniforms={
    "threshold": {"glsl": "float", "min": 0.0, "max": 1.0, "default": 0.5,
                  "description": "luminance cutoff"},
    "softness":  {"glsl": "float", "min": 0.0, "max": 0.5, "default": 0.05,
                  "description": "edge softness"},
    "low":       {"glsl": "color", "default": "#0a0a12", "description": "below-threshold color"},
    "high":      {"glsl": "color", "default": "#ffffff", "description": "above-threshold color"},
})

_register("hue_shift_gpu", "Hue rotate + saturation of the input (typed)",
          "filter", '''
vec3 _rgb2hsv(vec3 c) {
    vec4 K = vec4(0.0, -1.0/3.0, 2.0/3.0, -1.0);
    vec4 p = mix(vec4(c.bg, K.wz), vec4(c.gb, K.xy), step(c.b, c.g));
    vec4 q = mix(vec4(p.xyw, c.r), vec4(c.r, p.yzx), step(p.x, c.r));
    float d = q.x - min(q.w, q.y);
    float e = 1.0e-10;
    return vec3(abs(q.z + (q.w - q.y) / (6.0 * d + e)), d / (q.x + e), q.x);
}
vec3 _hsv2rgb(vec3 c) {
    vec4 K = vec4(1.0, 2.0/3.0, 1.0/3.0, 3.0);
    vec3 p = abs(fract(c.xxx + K.xyz) * 6.0 - K.www);
    return c.z * mix(K.xxx, clamp(p - K.xxx, 0.0, 1.0), c.y);
}
void main() {
    vec3 src = texture(u_texture, v_uv).rgb;
    vec3 hsv = _rgb2hsv(src);
    hsv.x = fract(hsv.x + u_hue);
    hsv.y = clamp(hsv.y * u_saturation, 0.0, 1.0);
    hsv.z = clamp(hsv.z * u_value, 0.0, 1.0);
    f_color = vec4(_hsv2rgb(hsv), 1.0);
}
''', uniforms={
    "hue":        {"glsl": "float", "min": 0.0, "max": 1.0, "default": 0.0,
                   "description": "hue rotation (0..1)"},
    "saturation": {"glsl": "float", "min": 0.0, "max": 3.0, "default": 1.0,
                   "description": "saturation gain"},
    "value":      {"glsl": "float", "min": 0.0, "max": 3.0, "default": 1.0,
                   "description": "brightness gain"},
})

_register("dither_gpu", "Ordered (Bayer 4x4) dither of the input (typed)",
          "filter", '''
void main() {
    vec3 src = texture(u_texture, v_uv).rgb;
    vec2 gp = mod(floor(v_uv * u_resolution / max(u_scale, 1.0)), 4.0);
    int ix = int(gp.x); int iy = int(gp.y);
    float bayer[16];
    bayer[0]=0.0;  bayer[1]=8.0;  bayer[2]=2.0;  bayer[3]=10.0;
    bayer[4]=12.0; bayer[5]=4.0;  bayer[6]=14.0; bayer[7]=6.0;
    bayer[8]=3.0;  bayer[9]=11.0; bayer[10]=1.0; bayer[11]=9.0;
    bayer[12]=15.0;bayer[13]=7.0; bayer[14]=13.0;bayer[15]=5.0;
    int bi = iy * 4 + ix;
    float thr = (bayer[bi] + 0.5) / 16.0;
    float lv = max(float(u_levels), 2.0);
    vec3 dithered = floor(src * lv + (thr - 0.5)) / (lv - 1.0);
    dithered = clamp(dithered, 0.0, 1.0);
    f_color = vec4(mix(src, dithered, clamp(u_amount, 0.0, 1.0)), 1.0);
}
''', uniforms={
    "levels": {"glsl": "int", "min": 2, "max": 16, "default": 3,
               "description": "output levels per channel"},
    "scale":  {"glsl": "float", "min": 1.0, "max": 8.0, "default": 1.0,
               "description": "dither pattern scale (px)"},
    "amount": {"glsl": "float", "min": 0.0, "max": 1.0, "default": 1.0,
               "description": "effect amount"},
})

# ── Typed-uniform closed-form field-eval twins (250-257) ──────────────────────
# Each exposes its key visual parameters as named u_* uniforms so the GPU node
# is fully editable AND wireable (data-typed SCALAR inputs) per the typed-uniform
# contract. All are pure functions of (uv, t) -> exact parity preview, no seeded
# layout divergence (same family as 125 Chladni / 164 Moiré / 172 Dunes).

# Shared inferno colormap (each _register is a separate program).
_INFERNO_GPU = '''
vec3 inferno(float t){
    t = clamp(t, 0.0, 1.0);
    const vec3 c0=vec3(0.00021894,0.00016488,-0.01907227);
    const vec3 c1=vec3(0.10651034,0.56396050,3.93279110);
    const vec3 c2=vec3(11.6028830,-3.9781129,-15.9420510);
    const vec3 c3=vec3(-41.703996,17.4360890,44.3541450);
    const vec3 c4=vec3(77.1629350,-33.402243,-81.8094230);
    const vec3 c5=vec3(-71.319421,32.6260640,73.2095190);
    const vec3 c6=vec3(25.1311300,-12.242810,-23.0709590);
    return c0+t*(c1+t*(c2+t*(c3+t*(c4+t*(c5+t*c6)))));
}
'''

_register("moire_typed", "Moiré interference gratings with typed mode/speed/freq (node 250)",
          "procedural", '''void main() {
    int mode = int(clamp(floor(u_mode + 0.5), 0.0, 3.0));
    float s1 = max(u_speed1, 0.05);
    float s2 = max(u_speed2, 0.05);
    float freq = max(u_freq, 1.0);
    float t = u_time * 0.05;
    vec2 res = u_resolution;
    vec2 p = (v_uv - 0.5) * res;
    float scale = 1.0 / max(res.x, res.y) * 2.0 * 3.14159265;
    float a1 = s1 * t, a2 = s2 * t;
    float g1, g2;
    if (mode == 1) {            // linear
        g1 = 0.5 + 0.5 * sin(freq * (p.x * cos(a1) + p.y * sin(a1)) * scale);
        g2 = 0.5 + 0.5 * sin(freq * (p.x * cos(a2) + p.y * sin(a2)) * scale);
    } else if (mode == 2) {     // spiral
        float r = length(p);
        g1 = 0.5 + 0.5 * sin(freq * r * scale + a1 * 4.0);
        g2 = 0.5 + 0.5 * sin(freq * r * scale + a2 * 4.0);
    } else if (mode == 3) {     // hex
        vec2 h = vec2(p.x, abs(fract(p.y * 0.5) - 0.25)) * scale;
        g1 = 0.5 + 0.5 * sin(freq * (h.x + a1));
        g2 = 0.5 + 0.5 * sin(freq * (h.y + a2));
    } else {                    // radial
        float r = length(p);
        g1 = 0.5 + 0.5 * sin(freq * r * scale + a1 * 4.0);
        g2 = 0.5 + 0.5 * sin(freq * r * scale + a2 * 4.0 + 1.57);
    }
    float v = clamp(0.5 + 0.5 * sin((g1 - g2) * 3.14159), 0.0, 1.0);
    f_color = vec4(mix(u_color_a, u_color_b, v), 1.0);
}
''', uniforms={
    "mode":   {"glsl": "choice", "choices": ["radial", "linear", "spiral", "hex"],
               "default": "radial", "description": "interference geometry"},
    "speed1": {"glsl": "float", "min": 0.0, "max": 4.0, "default": 1.0,
               "description": "grating 1 speed"},
    "speed2": {"glsl": "float", "min": 0.0, "max": 4.0, "default": 1.28,
               "description": "grating 2 speed"},
    "freq":   {"glsl": "float", "min": 1.0, "max": 60.0, "default": 20.0,
               "description": "grating frequency"},
    "color_a": {"glsl": "color", "default": "#0b1026", "description": "low color"},
    "color_b": {"glsl": "color", "default": "#ffcf4d", "description": "high color"},
})

_register("chladni_typed", "Chladni nodal plate with typed mode/rotation/phase (node 251)",
          "procedural", _INFERNO_GPU + '''void main() {
    float m = max(u_m_mode, 0.5);
    float n = max(u_n_mode, 0.5);
    float rot_ang = u_rotation;
    float ph = u_phase;
    vec2 p = (v_uv - 0.5) * 2.0;
    vec2 pr = rot(rot_ang) * p;
    float u = sin(m * 3.14159265 * (pr.x + 1.0) * 0.5 + ph)
            * sin(n * 3.14159265 * (pr.y + 1.0) * 0.5 + ph);
    float sig = tanh(u * u_contrast * 4.0);
    float v = 0.5 + 0.5 * sig;
    f_color = vec4(inferno(v), 1.0);
}
''', uniforms={
    "m_mode":    {"glsl": "float", "min": 1.0, "max": 12.0, "default": 3.0,
                  "description": "x mode number"},
    "n_mode":    {"glsl": "float", "min": 1.0, "max": 12.0, "default": 3.0,
                  "description": "y mode number"},
    "rotation":  {"glsl": "float", "min": -3.14159, "max": 3.14159, "default": 0.0,
                  "description": "plate rotation (rad)"},
    "phase":     {"glsl": "float", "min": -3.14159, "max": 3.14159, "default": 0.0,
                  "description": "phase shimmer (rad)"},
    "contrast":  {"glsl": "float", "min": 0.2, "max": 4.0, "default": 1.0,
                  "description": "nodal-line sharpness"},
})

_register("dunes_typed", "Sand dune migration with typed wind/sediment (node 252)",
          "procedural", '''void main() {
    float wind = max(u_wind_strength, 0.0);
    float sed = max(u_sediment, 0.0);
    float t = u_time * 0.05;
    float windAngle = t * 0.15;
    vec2 res = u_resolution;
    vec2 p = (v_uv - 0.5) * res;
    float h = 0.0;
    // Layered wave superposition -> migrating dune field.
    // wind_strength scales the wave amplitude (stronger wind -> taller, higher-contrast
    // dunes); sediment controls wavelength (more sediment -> finer ripples).
    float amp_scale = 0.2 + 1.5 * clamp(wind, 0.0, 1.5);   // height grows with wind
    float wlen_base = mix(60.0, 8.0, clamp(sed, 0.0, 1.0)); // finesse with sediment
    for (int i = 0; i < 5; i++) {
        float fi = float(i);
        float ang = windAngle + fi * 0.7;
        vec2 dir = vec2(cos(ang), sin(ang));
        float wlen = wlen_base * (1.0 - fi / 8.0);
        float amp = amp_scale * (1.0 - fi / 6.0);
        h += amp * sin(dot(p, dir) / wlen + t * (1.0 + fi * 0.2));
    }
    // Fixed normalization (independent of wind) so wind_strength changes contrast.
    float v = clamp(0.5 + 0.5 * (h / 5.0), 0.0, 1.0);
    vec3 col = mix(u_sand_low, u_sand_high, v);
    f_color = vec4(col, 1.0);
}
''', uniforms={
    "wind_strength": {"glsl": "float", "min": 0.0, "max": 1.5, "default": 0.6,
                      "description": "wind strength (dune height)"},
    "sediment":      {"glsl": "float", "min": 0.0, "max": 1.5, "default": 0.8,
                      "description": "sediment supply (ripple fineness)"},
    "sand_low":  {"glsl": "color", "default": "#5a3a1a", "description": "shadow sand"},
    "sand_high": {"glsl": "color", "default": "#e8c89a", "description": "lit sand"},
})

_register("quasicrystal_typed", "Quasicrystal interference with typed freq/rot/waves (node 253)",
          "procedural", _INFERNO_GPU + '''void main() {
    float freq = max(u_frequency, 0.005);
    float amp  = max(u_amplitude, 0.01);
    float rot  = u_rotation;
    int nwaves = int(clamp(u_waves, 2.0, 24.0));
    vec2 p = (v_uv - 0.5) * u_resolution;
    float t = u_time * 0.05;
    float sum = 0.0;
    for (int i = 0; i < 24; i++) {
        if (i >= nwaves) break;
        float fi = float(i);
        float a = rot + fi * 2.3999632 + t * 0.1;
        vec2 dir = vec2(cos(a), sin(a));
        sum += amp * sin(dot(p, dir) * freq * 0.01 + fi * 1.7);
    }
    float v = clamp(0.5 + 0.5 * (sum / float(max(nwaves, 1))), 0.0, 1.0);
    f_color = vec4(inferno(v), 1.0);
}
''', uniforms={
    "frequency": {"glsl": "float", "min": 0.5, "max": 10.0, "default": 3.0,
                  "description": "wave frequency"},
    "amplitude": {"glsl": "float", "min": 0.1, "max": 3.0, "default": 1.0,
                  "description": "wave amplitude"},
    "rotation":  {"glsl": "float", "min": 0.0, "max": 6.2831853, "default": 0.0,
                  "description": "global rotation (rad)"},
    "waves":     {"glsl": "int", "min": 2, "max": 24, "default": 12,
                  "description": "number of interfering waves"},
})

_register("metaballs_typed", "Metaballs isosurface field with typed isovalue/speed/colors (node 254)",
          "procedural", _INFERNO_GPU + '''void main() {
    float iso   = 0.05 + u_isovalue * 0.75;
    float speed = 0.1  + u_ball_speed * 4.9;
    float t = u_time * 0.05 * speed;
    vec2 p = v_uv;
    float field = 0.0;
    const int N = 14;
    for (int i = 0; i < N; i++) {
        float fi = float(i);
        float ang = fi * 2.399963;
        float orbit = 0.18 + 0.16 * hash21(vec2(fi, 1.7));
        float wx = 0.5 + orbit * cos(t * (0.6 + 0.05 * fi) + ang);
        float wy = 0.5 + orbit * sin(t * (0.6 + 0.05 * fi) + ang * 1.3);
        vec2 c = vec2(wx, wy);
        float ri = 0.06 + 0.05 * hash21(vec2(fi, 9.1));
        float d2 = dot(p - c, p - c);
        field += (ri * ri) / (ri * ri + d2 + 1e-4);
    }
    float f = clamp(field * 0.5, 0.0, 1.0);
    vec3 col = inferno(f);
    float edge = smoothstep(iso - 0.04, iso, field * (iso + 0.2))
               * (1.0 - smoothstep(iso, iso + 0.04, field * (iso + 0.2)));
    col += edge * 0.35;
    col = mix(col, u_tint, u_tint_strength);
    f_color = vec4(clamp(col, 0.0, 1.0), 1.0);
}
''', uniforms={
    "isovalue":   {"glsl": "float", "min": 0.0, "max": 1.0, "default": 0.5,
                   "description": "iso threshold"},
    "ball_speed": {"glsl": "float", "min": 0.0, "max": 1.0, "default": 0.5,
                   "description": "orbit speed"},
    "tint":       {"glsl": "color", "default": "#ffffff", "description": "edge tint"},
    "tint_strength": {"glsl": "float", "min": 0.0, "max": 1.0, "default": 0.0,
                      "description": "edge tint strength"},
})

_register("nebula_typed", "Procedural nebula with typed scale/warp/colors (node 255)",
          "procedural", '''void main() {
    vec2 uv = v_uv * max(u_scale, 0.5);
    float t = u_time * 0.03 * max(u_warp, 0.1);
    vec2 q = vec2(fbm(uv + t), fbm(uv + vec2(5.2, 1.3) + t * 0.7));
    vec2 r = vec2(fbm(uv + 3.0 * q + vec2(1.7, 9.2) + t * 0.3),
                  fbm(uv + 3.0 * q + vec2(8.3, 2.8) + t * 0.4));
    float v = fbm(uv + 3.0 * r);
    float mask = 1.0 - abs(v_uv.y - 0.5) * 2.0 * u_vignette;
    vec3 col = u_shadow + (u_highlight - u_shadow) * (0.5 + 0.5 * cos(v * 4.0 + vec3(0, 1, 2)));
    col *= mask;
    f_color = vec4(clamp(col, 0.0, 1.0), 1.0);
}
''', uniforms={
    "scale":     {"glsl": "float", "min": 0.5, "max": 6.0, "default": 2.0,
                  "description": "turbulence scale"},
    "warp":      {"glsl": "float", "min": 0.1, "max": 3.0, "default": 1.0,
                  "description": "domain-warp / drift speed"},
    "vignette":  {"glsl": "float", "min": 0.0, "max": 1.0, "default": 1.0,
                  "description": "vertical vignette"},
    "shadow":    {"glsl": "color", "default": "#0a0a1f", "description": "dark cloud"},
    "highlight": {"glsl": "color", "default": "#5aa0ff", "description": "nebula glow"},
})

_register("wood_grain_typed", "Wood grain rings with typed rings/scale/colors (node 256)",
          "procedural", '''void main() {
    vec2 uv = v_uv - 0.5;
    float d = length(uv) * max(u_scale, 1.0);
    float grain = sin(d * max(u_rings, 1.0) + fbm(uv * 10.0) * u_turb) * 0.5 + 0.5;
    vec3 col = mix(u_dark, u_light, grain);
    f_color = vec4(col, 1.0);
}
''', uniforms={
    "rings": {"glsl": "float", "min": 1.0, "max": 40.0, "default": 8.0,
              "description": "ring frequency"},
    "scale": {"glsl": "float", "min": 1.0, "max": 30.0, "default": 10.0,
              "description": "ring spread"},
    "turb":  {"glsl": "float", "min": 0.0, "max": 2.0, "default": 0.5,
              "description": "grain turbulence"},
    "dark":  {"glsl": "color", "default": "#3a1d0a", "description": "dark wood"},
    "light": {"glsl": "color", "default": "#9a5a2a", "description": "light wood"},
})

_register("ripples_typed", "Concentric color ripples with typed freq/speed/colors (node 257)",
          "procedural", '''void main() {
    vec2 uv = v_uv - 0.5;
    float d = length(uv);
    float ph = u_time * max(u_speed, 0.1) - d * max(u_freq, 1.0);
    float r = 0.5 + 0.5 * sin(ph);
    float g = 0.5 + 0.5 * sin(ph + 2.0);
    float b = 0.5 + 0.5 * sin(ph + 4.0);
    vec3 col = mix(u_color_a, u_color_b, d);
    col *= vec3(r, g, b);
    col *= (1.0 - d);
    f_color = vec4(col, 1.0);
}
''', uniforms={
    "freq":  {"glsl": "float", "min": 1.0, "max": 40.0, "default": 30.0,
              "description": "ripple frequency"},
    "speed": {"glsl": "float", "min": 0.1, "max": 4.0, "default": 2.0,
              "description": "ripple speed"},
    "color_a": {"glsl": "color", "default": "#10071f", "description": "inner color"},
    "color_b": {"glsl": "color", "default": "#4affd0", "description": "outer color"},
})

# ── Typed-uniform derivative-field nodes (258-264, 2026-07-11) ───────────────
# Single-input IMAGE filters that derive a FIELD from the upstream frame:
# Sobel magnitude / direction, Laplacian, Scharr, normal map, gradient
# orientation flow, emboss. Every variable is a real node param + wireable
# SCALAR port; filters take image_in: IMAGE (same contract as 244-249).
# Bodies stay in the GL330/ES300 parity subset. NOTE: the pixel-step uniform is
# named `u_texel` (NOT `step`, which is reserved in every filter twin — pitfall
# #15b). The 3x3 stencil is shared via _DERIV_GPU.

_DERIV_GPU = '''
float _dlum(vec2 uv) {
    return dot(texture(u_texture, uv).rgb, vec3(0.299, 0.587, 0.114));
}
vec3 _dfetch(vec2 uv) { return texture(u_texture, uv).rgb; }
'''

_register("sobel_mag_typed", "Sobel gradient magnitude of the input (typed, node 258)",
          "filter", _DERIV_GPU + '''void main() {
    vec2 px = u_texel / u_resolution;
    float tl = _dlum(v_uv + px * vec2(-1.0,  1.0));
    float  l = _dlum(v_uv + px * vec2(-1.0,  0.0));
    float bl = _dlum(v_uv + px * vec2(-1.0, -1.0));
    float  t = _dlum(v_uv + px * vec2( 0.0,  1.0));
    float  b = _dlum(v_uv + px * vec2( 0.0, -1.0));
    float tr = _dlum(v_uv + px * vec2( 1.0,  1.0));
    float  r = _dlum(v_uv + px * vec2( 1.0,  0.0));
    float br = _dlum(v_uv + px * vec2( 1.0, -1.0));
    float gx = -tl - 2.0*l - bl + tr + 2.0*r + br;
    float gy =  tl + 2.0*t + tr - bl - 2.0*b - br;
    float m = clamp(length(vec2(gx, gy)) * u_gain * 0.25, 0.0, 1.0);
    vec3 col = mix(u_low, u_high, m);
    f_color = vec4(col, 1.0);
}
''', uniforms={
    "gain":    {"glsl": "float", "min": 0.2, "max": 4.0, "default": 1.5,
                "description": "magnitude gain"},
    "texel":   {"glsl": "float", "min": 0.5, "max": 6.0, "default": 1.5,
                "description": "kernel thickness (px)"},
    "low":     {"glsl": "color", "default": "#000814", "description": "low (flat) color"},
    "high":    {"glsl": "color", "default": "#39ff88", "description": "edge (high) color"},
})

_register("sobel_dir_typed", "Sobel gradient direction (HSL flow) of the input (typed, node 259)",
          "filter", _DERIV_GPU + '''vec3 _hue2rgb(float h) {
    float k = mod(h * 6.0, 6.0);
    float x = clamp(abs(mod(k, 2.0) - 1.0), 0.0, 1.0);
    if (k < 1.0) return vec3(1.0, x, 0.0);
    if (k < 2.0) return vec3(x, 1.0, 0.0);
    if (k < 3.0) return vec3(0.0, 1.0, x);
    if (k < 4.0) return vec3(0.0, x, 1.0);
    if (k < 5.0) return vec3(x, 0.0, 1.0);
    return vec3(1.0, 0.0, x);
}
void main() {
    vec2 px = u_texel / u_resolution;
    float tl = _dlum(v_uv + px * vec2(-1.0,  1.0));
    float  l = _dlum(v_uv + px * vec2(-1.0,  0.0));
    float bl = _dlum(v_uv + px * vec2(-1.0, -1.0));
    float  t = _dlum(v_uv + px * vec2( 0.0,  1.0));
    float  b = _dlum(v_uv + px * vec2( 0.0, -1.0));
    float tr = _dlum(v_uv + px * vec2( 1.0,  1.0));
    float  r = _dlum(v_uv + px * vec2( 1.0,  0.0));
    float br = _dlum(v_uv + px * vec2( 1.0, -1.0));
    float gx = -tl - 2.0*l - bl + tr + 2.0*r + br;
    float gy =  tl + 2.0*t + tr - bl - 2.0*b - br;
    float ang = atan(gy, gx);                 // [-pi, pi]
    float hue = (ang + 3.14159265) / 6.2831853;
    float mag = clamp(length(vec2(gx, gy)) * u_gain * 0.25, 0.0, 1.0);
    vec3 col = mix(vec3(u_flat), _hue2rgb(hue), mag);
    f_color = vec4(col, 1.0);
}
''', uniforms={
    "gain":  {"glsl": "float", "min": 0.2, "max": 4.0, "default": 1.5,
              "description": "direction gain"},
    "texel": {"glsl": "float", "min": 0.5, "max": 6.0, "default": 1.5,
              "description": "kernel thickness (px)"},
    "flat":  {"glsl": "color", "default": "#101018", "description": "flat-region color"},
})

_register("laplacian_typed", "Laplacian zero-crossing / edge field of the input (typed, node 260)",
          "filter", _DERIV_GPU + '''void main() {
    vec2 px = u_texel / u_resolution;
    float c  = _dlum(v_uv);
    float l  = _dlum(v_uv + px * vec2(-1.0,  0.0));
    float r  = _dlum(v_uv + px * vec2( 1.0,  0.0));
    float t  = _dlum(v_uv + px * vec2( 0.0,  1.0));
    float b  = _dlum(v_uv + px * vec2( 0.0, -1.0));
    float lap = (l + r + t + b - 4.0 * c);
    float e = clamp(abs(lap) * u_gain * 0.5, 0.0, 1.0);
    vec3 col = mix(u_flat, u_edge, e);
    f_color = vec4(col, 1.0);
}
''', uniforms={
    "gain":  {"glsl": "float", "min": 0.2, "max": 6.0, "default": 2.0,
              "description": "laplacian gain"},
    "texel": {"glsl": "float", "min": 0.5, "max": 6.0, "default": 1.5,
              "description": "kernel spacing (px)"},
    "flat":  {"glsl": "color", "default": "#080810", "description": "flat color"},
    "edge":  {"glsl": "color", "default": "#ff5cf0", "description": "edge color"},
})

_register("scharr_typed", "Scharr operator (sharper Sobel) magnitude (typed, node 261)",
          "filter", _DERIV_GPU + '''void main() {
    vec2 px = u_texel / u_resolution;
    float tl = _dlum(v_uv + px * vec2(-1.0,  1.0));
    float  l = _dlum(v_uv + px * vec2(-1.0,  0.0));
    float bl = _dlum(v_uv + px * vec2(-1.0, -1.0));
    float  t = _dlum(v_uv + px * vec2( 0.0,  1.0));
    float  b = _dlum(v_uv + px * vec2( 0.0, -1.0));
    float tr = _dlum(v_uv + px * vec2( 1.0,  1.0));
    float  r = _dlum(v_uv + px * vec2( 1.0,  0.0));
    float br = _dlum(v_uv + px * vec2( 1.0, -1.0));
    float gx = -3.0*tl - 10.0*l - 3.0*bl + 3.0*tr + 10.0*r + 3.0*br;
    float gy =  3.0*tl + 10.0*t + 3.0*tr - 3.0*bl - 10.0*b - 3.0*br;
    float m = clamp(length(vec2(gx, gy)) * u_gain * 0.0625, 0.0, 1.0);
    vec3 col = mix(u_low, u_high, m);
    f_color = vec4(col, 1.0);
}
''', uniforms={
    "gain":  {"glsl": "float", "min": 0.2, "max": 4.0, "default": 1.6,
              "description": "magnitude gain"},
    "texel": {"glsl": "float", "min": 0.5, "max": 6.0, "default": 1.5,
              "description": "kernel thickness (px)"},
    "low":   {"glsl": "color", "default": "#001018", "description": "low color"},
    "high":  {"glsl": "color", "default": "#4dffd0", "description": "edge color"},
})

_register("normal_map_typed", "Normal map (bump) from luminance gradient (typed, node 262)",
          "filter", _DERIV_GPU + '''void main() {
    vec2 px = u_texel / u_resolution;
    float l = _dlum(v_uv + px * vec2(-1.0,  0.0));
    float r = _dlum(v_uv + px * vec2( 1.0,  0.0));
    float t = _dlum(v_uv + px * vec2( 0.0,  1.0));
    float b = _dlum(v_uv + px * vec2( 0.0, -1.0));
    float dx = (r - l) * u_strength;
    float dy = (t - b) * u_strength;
    vec3 n = normalize(vec3(-dx, -dy, 1.0));
    vec3 col = n * 0.5 + 0.5;
    f_color = vec4(col, 1.0);
}
''', uniforms={
    "strength": {"glsl": "float", "min": 0.1, "max": 8.0, "default": 2.0,
                 "description": "surface bumpiness"},
    "texel":    {"glsl": "float", "min": 0.5, "max": 6.0, "default": 1.5,
                 "description": "sample spacing (px)"},
})

_register("gradient_orient_typed", "Gradient orientation flow field (typed, node 263)",
          "filter", _DERIV_GPU + '''void main() {
    vec2 px = u_texel / u_resolution;
    float tl = _dlum(v_uv + px * vec2(-1.0,  1.0));
    float  l = _dlum(v_uv + px * vec2(-1.0,  0.0));
    float bl = _dlum(v_uv + px * vec2(-1.0, -1.0));
    float  t = _dlum(v_uv + px * vec2( 0.0,  1.0));
    float  b = _dlum(v_uv + px * vec2( 0.0, -1.0));
    float tr = _dlum(v_uv + px * vec2( 1.0,  1.0));
    float  r = _dlum(v_uv + px * vec2( 1.0,  0.0));
    float br = _dlum(v_uv + px * vec2( 1.0, -1.0));
    float gx = -tl - 2.0*l - bl + tr + 2.0*r + br;
    float gy =  tl + 2.0*t + tr - bl - 2.0*b - br;
    vec2 dir = (abs(gx) + abs(gy) < 1e-4) ? vec2(1.0, 0.0) : normalize(vec2(gx, gy));
    // rotate the orientation vector by the wind angle and tint by strength
    float ang = u_wind + u_time * u_spin;
    vec2 d = rot(ang) * dir;
    float mag = clamp(length(vec2(gx, gy)) * u_gain * 0.25, 0.0, 1.0);
    vec3 col = mix(vec3(u_flat), vec3(d * 0.5 + 0.5, 0.5), mag);
    f_color = vec4(col, 1.0);
}
''', uniforms={
    "gain":  {"glsl": "float", "min": 0.2, "max": 4.0, "default": 1.5,
              "description": "flow strength"},
    "wind":  {"glsl": "float", "min": -3.14159, "max": 3.14159, "default": 0.0,
              "description": "flow rotation (rad)"},
    "spin":  {"glsl": "float", "min": 0.0, "max": 3.0, "default": 0.0,
              "description": "animated spin speed"},
    "texel": {"glsl": "float", "min": 0.5, "max": 6.0, "default": 1.5,
              "description": "kernel thickness (px)"},
    "flat":  {"glsl": "color", "default": "#0a0a12", "description": "flat color"},
})

_register("emboss_typed", "Directional emboss (relief) of the input (typed, node 264)",
          "filter", _DERIV_GPU + '''void main() {
    vec2 px = u_texel / u_resolution;
    // 3x3 emboss kernel rotated by u_angle
    float tl = _dlum(v_uv + px * vec2(-1.0,  1.0));
    float  l = _dlum(v_uv + px * vec2(-1.0,  0.0));
    float bl = _dlum(v_uv + px * vec2(-1.0, -1.0));
    float  t = _dlum(v_uv + px * vec2( 0.0,  1.0));
    float  b = _dlum(v_uv + px * vec2( 0.0, -1.0));
    float tr = _dlum(v_uv + px * vec2( 1.0,  1.0));
    float  r = _dlum(v_uv + px * vec2( 1.0,  0.0));
    float br = _dlum(v_uv + px * vec2( 1.0, -1.0));
    float gx = -tl - 2.0*l - bl + tr + 2.0*r + br;
    float gy =  tl + 2.0*t + tr - bl - 2.0*b - br;
    float relief = gx * cos(u_angle) + gy * sin(u_angle);
    float e = clamp(0.5 + relief * u_gain * 0.25, 0.0, 1.0);
    vec3 col = mix(u_dark, u_light, e);
    f_color = vec4(col, 1.0);
}
''', uniforms={
    "gain":  {"glsl": "float", "min": 0.2, "max": 4.0, "default": 1.5,
              "description": "relief strength"},
    "angle": {"glsl": "float", "min": 0.0, "max": 6.2831853, "default": 2.3561945,
              "description": "light direction (rad)"},
    "texel": {"glsl": "float", "min": 0.5, "max": 6.0, "default": 1.5,
              "description": "kernel thickness (px)"},
    "dark":  {"glsl": "color", "default": "#161a2a", "description": "shadow color"},
    "light": {"glsl": "color", "default": "#f3f0e6", "description": "highlight color"},
})

# ── Typed-uniform pattern expansions (ids 265–270, 2026-07-11) ───────────────
# Closed-form f(uv,t) pattern generators — every frame a pure function of the
# fragment coordinate and the animation clock, so the CPU-vs-GPU live preview is
# EXACTLY reproducible (no seeded-layout divergence). Additive GPU live path;
# no CPU fn is touched. Each declares NAMED typed uniforms that the factory
# turns into node params + wireable SCALAR ports.

_register("spirograph_typed", "Hypotrochoid/epitrochoid spirograph ribbons (typed, node 265)",
          "procedural", '''void main() {
    float R = max(u_ring_radius, 0.01);
    float r = max(u_wheel_radius, 0.001);
    float d = u_pen_offset;
    int np = int(clamp(u_petals, 1.0, 60.0));
    float speed = u_time * 0.02 * max(u_spin, 0.0);
    vec2 ctr = (vec2(u_center_x, u_center_y) - 0.5) * 2.0;
    vec2 p = (v_uv - 0.5) * 2.0 - ctr;
    float best = 1e9;
    for (int i = 0; i < 60; i++) {
        if (i >= np) break;
        float a = (float(i) / float(np)) * 6.2831853 + speed;
        float ca = cos(a), sa = sin(a);
        // hypotrochoid point for this phase
        vec2 q = (R - r) * vec2(ca, sa) + d * vec2(cos(((R - r) / r) * a), sin(((R - r) / r) * a));
        best = min(best, distance(p, q));
    }
    float rib = smoothstep(u_line_width, u_line_width * 0.25, best);
    f_color = vec4(mix(u_bg, u_ink, rib), 1.0);
}
''', uniforms={
    "ring_radius":  {"glsl": "float", "min": 0.1, "max": 1.0, "default": 0.7,
                     "description": "fixed ring radius R"},
    "wheel_radius": {"glsl": "float", "min": 0.02, "max": 0.9, "default": 0.27,
                     "description": "rolling wheel radius r"},
    "pen_offset":   {"glsl": "float", "min": 0.0, "max": 0.9, "default": 0.45,
                     "description": "pen distance from wheel center"},
    "petals":       {"glsl": "float", "min": 1.0, "max": 60.0, "default": 24.0,
                     "description": "number of lobes"},
    "spin":         {"glsl": "float", "min": 0.0, "max": 8.0, "default": 1.0,
                     "description": "rotation animation speed"},
    "center_x":     {"glsl": "float", "min": 0.0, "max": 1.0, "default": 0.5,
                     "description": "rosette center x"},
    "center_y":     {"glsl": "float", "min": 0.0, "max": 1.0, "default": 0.5,
                     "description": "rosette center y"},
    "line_width":   {"glsl": "float", "min": 0.005, "max": 0.08, "default": 0.02,
                     "description": "ribbon thickness"},
    "bg":   {"glsl": "color", "default": "#0c0e1a", "description": "background"},
    "ink":  {"glsl": "color", "default": "#5ef2c0", "description": "ribbon color"},
})

_register("truchet_maze_typed", "Random-rotated Truchet arc/maze tiling (typed, node 266)",
          "procedural", '''float _truchet_hash(vec2 p) {
    p = fract(p * vec2(123.34, 345.45));
    p += dot(p, p + 34.345);
    return fract(p.x * p.y);
}
void main() {
    int cells = int(clamp(u_cells, 1.0, 40.0));
    float cellSize = 1.0 / float(cells);
    // Continuous rotation of the whole tiling with time -> every frame differs
    // and the maze appears to spin/re-tile as it animates.
    float ang = u_time * 0.15 * max(u_anim_speed, 0.0);
    mat2 R = mat2(cos(ang), -sin(ang), sin(ang), cos(ang));
    vec2 uv = R * (v_uv - 0.5) / cellSize + 0.5;
    vec2 id = floor(uv);
    vec2 f = fract(uv) - 0.5;
    float h = _truchet_hash(id);
    bool flip = h > 0.5;
    if (flip) f = vec2(f.y, f.x);
    float r = u_arc_radius;
    float d1 = abs(distance(f, vec2(-0.5 + r, -0.5 + r)) - r);
    float d2 = abs(distance(f, vec2( 0.5 - r,  0.5 - r)) - r);
    float d = min(d1, d2);
    float line = smoothstep(u_line_width, u_line_width * 0.4, d);
    vec3 col = mix(u_bg, u_ink, line);
    if (u_show_nodes > 0.5) {
        float dn = min(distance(f, vec2(-0.5, -0.5)), distance(f, vec2(0.5, 0.5)));
        col = mix(col, u_node_color, smoothstep(0.06, 0.02, dn));
    }
    f_color = vec4(col, 1.0);
}
''', uniforms={
    "cells":       {"glsl": "float", "min": 1.0, "max": 40.0, "default": 8.0,
                    "description": "tiles per axis"},
    "line_width":  {"glsl": "float", "min": 0.01, "max": 0.25, "default": 0.09,
                    "description": "stroke width"},
    "arc_radius":  {"glsl": "float", "min": 0.1, "max": 0.5, "default": 0.5,
                    "description": "arc radius (1=semicircle)"},
    "anim_speed":  {"glsl": "float", "min": 0.0, "max": 6.0, "default": 1.0,
                    "description": "tile rotation animation speed"},
    "show_nodes":  {"glsl": "float", "min": 0.0, "max": 1.0, "default": 0.0,
                    "description": "draw connection nodes (0/1)"},
    "bg":          {"glsl": "color", "default": "#10131f", "description": "background"},
    "ink":         {"glsl": "color", "default": "#e8d9a0", "description": "stroke color"},
    "node_color":  {"glsl": "color", "default": "#ff6b6b", "description": "node color"},
})

_register("reaction_waves_typed", "Autonomous reaction-diffusion wave pattern (typed, node 267)",
          "procedural", _INFERNO_GPU + '''void main() {
    vec2 res = u_resolution;
    vec2 p = (v_uv - 0.5) * res;
    float t = u_time * 0.05;
    float v = 0.0;
    // Layered concentric reaction fronts from jittered seed centers.
    for (int i = 0; i < 8; i++) {
        float fi = float(i);
        vec2 seed = (vec2(hash21(vec2(fi, 3.1)), hash21(vec2(fi, 7.7))) - 0.5) * res;
        float d = distance(p, seed);
        float k = (u_wavelength * (1.0 + 0.25 * sin(fi * 1.7)));
        float ph = (d / k) - t * (u_speed * (1.0 + 0.15 * cos(fi * 2.3)));
        v += (0.5 + 0.5 * sin(ph * 6.2831853));
    }
    v = (v / 8.0 - 0.5) * u_contrast + 0.5;
    v = clamp(v, 0.0, 1.0);
    f_color = vec4(inferno(v), 1.0);
}
''', uniforms={
    "speed":       {"glsl": "float", "min": 0.0, "max": 6.0, "default": 1.0,
                    "description": "wave propagation speed"},
    "wavelength":  {"glsl": "float", "min": 4.0, "max": 120.0, "default": 38.0,
                    "description": "front spacing (px)"},
    "contrast":    {"glsl": "float", "min": 0.2, "max": 4.0, "default": 1.0,
                    "description": "band sharpness"},
})

_register("hex_grid_typed", "Hexagonal lattice with tri-planar cell tinting (typed, node 268)",
          "procedural", '''vec4 _hexDist(vec2 p) {
    // p in hex-tile space (unit cell). Returns vec4(r, g, b, minDist) where
    // the first three components carry the three edge distances of the hexagon.
    vec2 q = abs(p);
    float c = dot(q, normalize(vec2(1.0, 1.7320508)));
    float a = max(c, q.x);
    float b = max(c, q.y);
    // distance to the two relevant edge orientations + vertical edge
    vec2 r = vec2(max(a, b), max(q.x * 0.8660254 + q.y * 0.5, q.y));
    return vec4(a, b, q.y, min(a, r.y));
}
void main() {
    float sc = max(u_scale, 0.5);
    vec2 p = (v_uv - 0.5) * u_resolution / sc * 2.0;
    p += vec2(u_offset_x, u_offset_y) * u_resolution / sc * 2.0;
    p.y += u_time * u_flow * 0.5;
    const vec2 s = vec2(1.0, 1.7320508);
    vec2 a = mod(p, s) - s * 0.5;
    vec2 b = mod(p + s * 0.5, s) - s * 0.5;
    vec4 ha = _hexDist(a);
    vec4 hb = _hexDist(b);
    float d = (length(a) < length(b)) ? ha.w : hb.w;
    // thickness is in CELL units (0..0.5); convert so 0.1 reads as a thin wall.
    float th = max(u_thickness * 0.1, 0.001);
    float edge = smoothstep(th, th * 0.4, d);
    vec2 cell = (length(a) < length(b)) ? floor(p / s) : floor((p + s * 0.5) / s);
    float idh = fract(sin(dot(cell, vec2(127.1, 311.7))) * 43758.5453);
    vec3 fill = mix(u_fill_a, u_fill_b, idh);
    f_color = vec4(mix(fill, u_line, edge), 1.0);
}
''', uniforms={
    "scale":       {"glsl": "float", "min": 4.0, "max": 120.0, "default": 24.0,
                    "description": "hex cell size (px)"},
    "thickness":   {"glsl": "float", "min": 0.5, "max": 8.0, "default": 2.0,
                    "description": "wall thickness (px)"},
    "flow":        {"glsl": "float", "min": 0.0, "max": 6.0, "default": 1.0,
                    "description": "downward drift speed"},
    "offset_x":    {"glsl": "float", "min": -0.5, "max": 0.5, "default": 0.0,
                    "description": "horizontal offset"},
    "offset_y":    {"glsl": "float", "min": -0.5, "max": 0.5, "default": 0.0,
                    "description": "vertical offset"},
    "fill_a":      {"glsl": "color", "default": "#14233f", "description": "cell tint A"},
    "fill_b":      {"glsl": "color", "default": "#2a4d6e", "description": "cell tint B"},
    "line":        {"glsl": "color", "default": "#9fe3ff", "description": "wall color"},
})

_register("starfield_typed", "Parallax starfield with twinkling (typed, node 269)",
          "procedural", '''float _sfield(vec2 uv, float seed) {
    vec2 g = floor(uv);
    vec2 f = fract(uv);
    float h = hash21(g + seed);
    float star = smoothstep(0.5 - u_star_size, 0.5 - u_star_size * 0.4,
                            distance(f, vec2(h, fract(h * 13.3))));
    return star;
}
void main() {
    vec2 uv = v_uv * u_density;
    float t = u_time * 0.1 * u_twinkle;
    float total = 0.0;
    vec3 col = u_bg_color;
    for (int i = 1; i <= 4; i++) {
        float fi = float(i);
        float depth = fi / 4.0;
        vec2 suv = (uv * depth) + vec2(t * depth, t * depth * 0.3) + fi * 17.0;
        float s = _sfield(suv, fi * 3.7);
        float tw = 0.6 + 0.4 * sin(t * (2.0 + fi) + hash21(suv) * 6.2831);
        total += s * tw * (1.0 - depth * 0.4);
        col += u_star_color * s * tw * (1.0 - depth * 0.5);
    }
    col = max(col, u_bg_color);
    f_color = vec4(col, 1.0);
}
''', uniforms={
    "density":     {"glsl": "float", "min": 4.0, "max": 80.0, "default": 30.0,
                    "description": "stars per screen"},
    "star_size":   {"glsl": "float", "min": 0.01, "max": 0.2, "default": 0.06,
                    "description": "star radius"},
    "twinkle":     {"glsl": "float", "min": 0.0, "max": 6.0, "default": 1.0,
                    "description": "twinkle animation speed"},
    "bg_color":    {"glsl": "color", "default": "#03040a", "description": "sky color"},
    "star_color":  {"glsl": "color", "default": "#ffffff", "description": "star color"},
})

_register("concentric_rings_typed", "Smooth concentric rings / ripples (typed, node 270)",
          "procedural", _INFERNO_GPU + '''void main() {
    vec2 res = u_resolution;
    vec2 p = (v_uv - 0.5) * res;
    p += (vec2(u_center_x, u_center_y) - 0.5) * res;
    p = rot(u_skew) * p;
    float r = length(p);
    float t = u_time * 0.05 * u_speed;
    float rings = 0.5 + 0.5 * sin(r / max(u_spacing, 1.0) * 6.2831853 - t * 6.2831853);
    rings = pow(rings, u_sharpness);
    f_color = vec4(inferno(clamp(rings, 0.0, 1.0)), 1.0);
}
''', uniforms={
    "spacing":     {"glsl": "float", "min": 4.0, "max": 120.0, "default": 28.0,
                    "description": "ring spacing (px)"},
    "speed":       {"glsl": "float", "min": 0.0, "max": 6.0, "default": 1.0,
                    "description": "ripple expansion speed"},
    "sharpness":   {"glsl": "float", "min": 0.3, "max": 6.0, "default": 1.0,
                    "description": "band sharpness"},
    "skew":        {"glsl": "float", "min": -1.5707963, "max": 1.5707963, "default": 0.0,
                    "description": "ellipse skew (rad)"},
    "center_x":    {"glsl": "float", "min": 0.0, "max": 1.0, "default": 0.5,
                    "description": "center x"},
    "center_y":    {"glsl": "float", "min": 0.0, "max": 1.0, "default": 0.5,
                    "description": "center y"},
})

# ── Typed math_art pattern nodes (ids 271-276) ───────────────────────────
# Categorical coverage for the math_art family: closed-form visual patterns
# (Ulam-spiral homage, hash maze, circle packing, Fourier epicycles, summed
# waveform, Clifford strange-attractor bands). Each exposes NAMED typed
# controls + wireable SCALAR ports (the _make_typed factory derives them from
# `uniforms`). CPU fns stay authoritative; these are an additive typed-uniform
# live-preview layer. No per-frame seeds — every frame is a pure function of
# (uv, t), so GPU/CPU parity is exact (no seeded-layout divergence).
_register("ulam_spiral_typed", "Ulam-spiral homage: sparse glowing dots along a number spiral (typed, node 271)",
          "procedural", _INFERNO_GPU + '''void main() {
    vec2 uv = (v_uv - 0.5);
    uv.x *= u_resolution.x / u_resolution.y;
    float rad = length(uv);
    float ang = atan(uv.y, uv.x);
    float turns = max(u_turns, 0.5);
    float idx = rad * turns * 6.28318530;
    vec2 cell = vec2(floor(idx / max(u_cells, 1.0)),
                     floor((ang + 3.14159265) / (6.28318530 / max(u_arms, 1.0))));
    float h = hash21(cell + 0.5);
    float isPrime = step(1.0 - u_density, h);
    float t = u_time * 0.03 * u_speed;
    float glow = 0.5 + 0.5 * sin(idx * 0.25 - t * 6.28318530);
    float v = max(isPrime, glow * 0.22);
    vec3 col = mix(u_bg, u_fg, v);
    f_color = vec4(col, 1.0);
}
''', uniforms={
    "turns":   {"glsl": "float", "min": 1.0, "max": 40.0, "default": 12.0,
                "description": "spiral turns"},
    "cells":   {"glsl": "float", "min": 4.0, "max": 80.0, "default": 24.0,
                "description": "cells per turn"},
    "arms":    {"glsl": "float", "min": 1.0, "max": 16.0, "default": 6.0,
                "description": "radial arms"},
    "density": {"glsl": "float", "min": 0.02, "max": 0.6, "default": 0.18,
                "description": "prime-dot density"},
    "speed":   {"glsl": "float", "min": 0.0, "max": 6.0, "default": 1.0,
                "description": "animation speed"},
    "bg":      {"glsl": "color", "default": "#05060f", "description": "background"},
    "fg":      {"glsl": "color", "default": "#ffcf5c", "description": "dot color"},
})

_register("maze_typed", "Hash maze: procedural wall grid (typed, node 272)",
          "procedural", '''void main() {
    vec2 uv = v_uv * max(u_scale, 1.0);
    uv += vec2(u_time * u_drift * 0.05, 0.0);
    vec2 g = floor(uv);
    vec2 f = fract(uv);
    float hw = max(u_wall, 0.02) * 0.5;
    float h1 = hash21(g);
    float h2 = hash21(g + 17.3);
    float vwall = step(1.0 - u_density, h1) * step(f.x, hw);
    float hwall = step(1.0 - u_density, h2) * step(f.y, hw);
    float wall = clamp(vwall + hwall, 0.0, 1.0);
    vec3 col = mix(u_bg, u_fg, wall);
    f_color = vec4(col, 1.0);
}
''', uniforms={
    "scale":   {"glsl": "float", "min": 4.0, "max": 60.0, "default": 18.0,
                "description": "grid density"},
    "wall":    {"glsl": "float", "min": 0.04, "max": 0.5, "default": 0.18,
                "description": "wall thickness"},
    "density": {"glsl": "float", "min": 0.02, "max": 0.6, "default": 0.25,
                "description": "wall probability"},
    "drift":   {"glsl": "float", "min": 0.0, "max": 4.0, "default": 0.0,
                "description": "scroll drift"},
    "bg":      {"glsl": "color", "default": "#0a0a14", "description": "background"},
    "fg":      {"glsl": "color", "default": "#9be7ff", "description": "wall color"},
})

_register("circle_packing_typed", "Circle packing: grid of disks with hashed radii (typed, node 273)",
          "procedural", _INFERNO_GPU + '''void main() {
    vec2 uv = v_uv * max(u_scale, 1.0);
    vec2 g = floor(uv);
    vec2 f = fract(uv) - 0.5;
    float h = hash21(g + 3.1);
    float rad = u_min_r + h * (u_max_r - u_min_r);
    float d = length(f);
    float disk = smoothstep(rad, rad - 0.05, d) * step(d, rad);
    float t = u_time * 0.05 * u_speed;
    vec3 tint = inferno(h);
    vec3 col = mix(u_bg, tint, disk);
    f_color = vec4(col, 1.0);
}
''', uniforms={
    "scale":  {"glsl": "float", "min": 2.0, "max": 40.0, "default": 10.0,
               "description": "pack density"},
    "min_r":  {"glsl": "float", "min": 0.05, "max": 0.6, "default": 0.15,
               "description": "min disk radius"},
    "max_r":  {"glsl": "float", "min": 0.1, "max": 0.95, "default": 0.5,
               "description": "max disk radius"},
    "speed":  {"glsl": "float", "min": 0.0, "max": 6.0, "default": 1.0,
               "description": "animation speed"},
    "bg":     {"glsl": "color", "default": "#04060d", "description": "background"},
})

_register("fourier_circles_typed", "Fourier epicycles: traced harmonic curve (typed, node 274)",
          "procedural", '''void main() {
    vec2 p = (v_uv - 0.5);
    p.x *= u_resolution.x / u_resolution.y;
    float t = u_time * 0.1 * u_speed;
    float best = 1e9;
    for (int i = 0; i < 128; i++) {
        float s = float(i) / 127.0;
        float ph = s * 6.28318530 + t;
        vec2 q = vec2(0.0);
        q.x += sin(ph * u_freq1 + u_phase1) * (0.32 / max(u_freq1, 1.0));
        q.y += cos(ph * u_freq2 + u_phase2) * (0.32 / max(u_freq2, 1.0));
        q += 0.22 * vec2(sin(ph * u_freq3), cos(ph * u_freq3));
        best = min(best, length(p - q));
    }
    float line = smoothstep(u_thick, 0.0, best);
    vec3 col = mix(u_bg, u_fg, line);
    f_color = vec4(col, 1.0);
}
''', uniforms={
    "freq1":  {"glsl": "float", "min": 1.0, "max": 12.0, "default": 3.0,
               "description": "harmonic 1 freq"},
    "freq2":  {"glsl": "float", "min": 1.0, "max": 12.0, "default": 5.0,
               "description": "harmonic 2 freq"},
    "freq3":  {"glsl": "float", "min": 1.0, "max": 12.0, "default": 2.0,
               "description": "harmonic 3 freq"},
    "phase1": {"glsl": "float", "min": 0.0, "max": 6.2831853, "default": 0.0,
               "description": "harmonic 1 phase"},
    "phase2": {"glsl": "float", "min": 0.0, "max": 6.2831853, "default": 1.2,
               "description": "harmonic 2 phase"},
    "thick":  {"glsl": "float", "min": 0.005, "max": 0.08, "default": 0.02,
               "description": "line thickness"},
    "speed":  {"glsl": "float", "min": 0.0, "max": 6.0, "default": 1.0,
               "description": "animation speed"},
    "bg":     {"glsl": "color", "default": "#05070f", "description": "background"},
    "fg":     {"glsl": "color", "default": "#62f0c8", "description": "curve color"},
})

_register("waveform_typed", "Waveform: summed sine oscillators (typed, node 275)",
          "procedural", '''void main() {
    vec2 p = (v_uv - 0.5);
    p.x *= u_resolution.x / u_resolution.y;
    float t = u_time * 0.08 * u_speed;
    float y = 0.0;
    y += sin(p.x * u_k1 * 6.28318530 + t);
    y += 0.5 * sin(p.x * u_k2 * 6.28318530 + t * 1.3);
    y += 0.3 * sin(p.x * u_k3 * 6.28318530 + t * 0.7);
    y *= u_amp * 0.25;
    float line = smoothstep(u_thick, 0.0, abs(p.y - y));
    vec3 col = mix(u_bg, u_fg, line);
    f_color = vec4(col, 1.0);
}
''', uniforms={
    "k1":    {"glsl": "float", "min": 0.5, "max": 16.0, "default": 3.0,
              "description": "osc 1 wavenumber"},
    "k2":    {"glsl": "float", "min": 0.5, "max": 16.0, "default": 6.0,
              "description": "osc 2 wavenumber"},
    "k3":    {"glsl": "float", "min": 0.5, "max": 16.0, "default": 9.0,
              "description": "osc 3 wavenumber"},
    "amp":   {"glsl": "float", "min": 0.2, "max": 2.0, "default": 1.0,
              "description": "amplitude"},
    "thick": {"glsl": "float", "min": 0.005, "max": 0.08, "default": 0.02,
              "description": "trace thickness"},
    "speed": {"glsl": "float", "min": 0.0, "max": 6.0, "default": 1.0,
              "description": "scroll speed"},
    "bg":    {"glsl": "color", "default": "#04060d", "description": "background"},
    "fg":    {"glsl": "color", "default": "#ff6bd6", "description": "trace color"},
})

_register("strange_attractor_typed", "Strange-attractor bands: Clifford map density (typed, node 276)",
          "procedural", _INFERNO_GPU + '''void main() {
    vec2 uv = (v_uv - 0.5);
    uv.x *= u_resolution.x / u_resolution.y;
    vec2 p = uv * 2.6;
    float t = u_time * 0.05 * u_speed;
    float a = u_a + 0.12 * sin(t);
    float b = u_b + 0.12 * cos(t);
    float c = u_c;
    float d = u_d;
    vec2 q = p;
    float acc = 0.0;
    for (int i = 0; i < 16; i++) {
        vec2 nx = vec2(sin(a * q.y) + c * cos(a * q.x),
                       sin(b * q.x) + d * cos(b * q.y));
        acc += exp(-dot(nx - p, nx - p) * u_band);
        q = nx;
    }
    float v = clamp(acc * u_gain, 0.0, 1.0);
    vec3 col = inferno(v);
    f_color = vec4(col, 1.0);
}
''', uniforms={
    "a":    {"glsl": "float", "min": -2.5, "max": 2.5, "default": -1.4,
             "description": "Clifford a"},
    "b":    {"glsl": "float", "min": -2.5, "max": 2.5, "default": 1.6,
             "description": "Clifford b"},
    "c":    {"glsl": "float", "min": -2.0, "max": 2.0, "default": 1.0,
             "description": "Clifford c"},
    "d":    {"glsl": "float", "min": -2.0, "max": 2.0, "default": 0.7,
             "description": "Clifford d"},
    "band": {"glsl": "float", "min": 1.0, "max": 60.0, "default": 18.0,
             "description": "band tightness"},
    "gain": {"glsl": "float", "min": 0.1, "max": 3.0, "default": 1.0,
             "description": "density gain"},
    "speed":{"glsl": "float", "min": 0.0, "max": 6.0, "default": 1.0,
             "description": "animation speed"},
})

# ── Node 332: De Jong Attractor — GPU live-preview twin of CPU node 498 ──
# De Jong recurrence x'=sin(a·y)−cos(b·x), y'=sin(c·x)−cos(d·y). The CPU node
# iterates many parallel walkers, splats a density grid, and tone-maps with an
# inferno ramp; this twin does the SAME in a single fragment pass (a coarse
# accumulation grid + 3×3 gaussian splat) so the live preview is a faithful
# parity of the density (inferno) colouring. Named typed uniforms mirror node
# 498's real numeric params (a/b/c/d/exposure) — contract #5/#6. `morph`+`speed`
# animate the parameters via u_time (matches CPU anim_mode="morph_all"), so the
# live preview is genuinely time-varying and survives the contrast-only static
# liveness cull. The CPU numpy node stays authoritative for export; this is the
# live GPU source for the De Jong category (categorical coverage gap: 498 had
# no GPU twin until now).
_register("de_jong_typed", "De Jong attractor density field (typed, node 498)",
          "procedural", _INFERNO_GPU + '''void main() {
    vec2 uv = (v_uv - 0.5);
    uv.x *= u_resolution.x / u_resolution.y;
    float t = u_time * 0.05 * u_speed;
    float a = u_a, b = u_b, c = u_c, d = u_d;
    if (u_morph > 0.5) {
        // morph_all: smoothly perturb all four params (no abs(sin) cusp)
        a += 0.9 * sin(t);
        b += 0.9 * cos(t * 0.9);
        c += 0.9 * sin(t * 1.1 + 1.0);
        d += 0.9 * cos(t * 0.8 + 2.0);
    }
    const int ACC = 40;      // accumulation starts (parallel trajectories)
    const int STP = 64;      // points splatted per start
    const int GRID = 40;     // density accumulation grid (coarse, cheap)
    float dens[GRID * GRID];
    for (int k = 0; k < GRID * GRID; k++) dens[k] = 0.0;
    float span = 2.0 + max(max(abs(a), abs(b)), max(abs(c), abs(d)));
    for (int i = 0; i < ACC; i++) {
        float seed = float(i) / float(ACC);
        vec2 p = vec2(sin(seed * 12.9) * 1.7, cos(seed * 7.3) * 1.7);
        for (int k = 0; k < 8; k++) {            // transient discard
            p = vec2(sin(a * p.y) - cos(b * p.x),
                     sin(c * p.x) - cos(d * p.y));
        }
        for (int s = 0; s < STP; s++) {
            p = vec2(sin(a * p.y) - cos(b * p.x),
                     sin(c * p.x) - cos(d * p.y));
            vec2 g = clamp(p / span * 0.5 + 0.5, 0.0, 0.999);
            int gx = int(g.x * float(GRID));
            int gy = int(g.y * float(GRID));
            dens[gy * GRID + gx] += 1.0;
        }
    }
    // gaussian-splat the accumulation grid at the pixel position
    vec2 g = clamp(uv / span * 0.5 + 0.5, 0.0, 0.999);
    float fx = g.x * float(GRID) - 0.5;
    float fy = g.y * float(GRID) - 0.5;
    float acc = 0.0;
    for (int dy = -1; dy <= 1; dy++) {
        for (int dx = -1; dx <= 1; dx++) {
            float cx = floor(fx) + float(dx);
            float cy = floor(fy) + float(dy);
            if (cx < 0.0 || cy < 0.0 || cx >= float(GRID) || cy >= float(GRID)) continue;
            int idx = int(cy) * GRID + int(cx);
            float dist = length(vec2(fx, fy) - vec2(cx + 0.5, cy + 0.5));
            acc += dens[idx] * exp(-dist * dist * u_sharp);
        }
    }
    float glow = 1.0 - exp(-u_exposure * acc * u_density_scale);
    vec3 col = inferno(clamp(glow, 0.0, 1.0));
    f_color = vec4(col, 1.0);
}
''', uniforms={
    "a":    {"glsl": "float", "min": -3.0, "max": 3.0, "default": -2.0,
             "description": "de Jong parameter a (shape control)"},
    "b":    {"glsl": "float", "min": -3.0, "max": 3.0, "default": -2.0,
             "description": "de Jong parameter b (shape control)"},
    "c":    {"glsl": "float", "min": -3.0, "max": 3.0, "default": -1.2,
             "description": "de Jong parameter c (shape control)"},
    "d":    {"glsl": "float", "min": -3.0, "max": 3.0, "default": 2.0,
             "description": "de Jong parameter d (shape control)"},
    "morph": {"glsl": "float", "min": 0.0, "max": 1.0, "default": 0.0,
              "description": "morph a/b/c/d over time (0=static, 1=morph_all)"},
    "exposure": {"glsl": "float", "min": 0.2, "max": 6.0, "default": 1.6,
                 "description": "tone-map exposure (glow brightness)"},
    "sharp": {"glsl": "float", "min": 1.0, "max": 60.0, "default": 18.0,
              "description": "splat tightness (band tightness)"},
    "density_scale": {"glsl": "float", "min": 0.1, "max": 3.0, "default": 1.0,
                      "description": "density accumulation normaliser"},
    "speed": {"glsl": "float", "min": 0.0, "max": 6.0, "default": 1.0,
              "description": "animation speed"},
})

# ── Categorical coverage pt.8 (typed closed-form patterns, nodes 277-282) ──
# phyllotaxis dots, guilloché engraving, Lissajous trace, radial wave
# interference, curl-noise flow field, kaleidoscopic petal bloom. Each is a
# pure f(uv, t) → exact CPU/GPU parity (P0.6). Continuous-time motion only.

_register("phyllotaxis_typed", "Phyllotaxis: golden-angle sunflower dot spiral (typed, node 277)",
          "procedural", _INFERNO_GPU + '''void main() {
    vec2 uv = (v_uv - 0.5);
    uv.x *= u_resolution.x / u_resolution.y;
    float t = u_time * 0.05 * u_speed;
    float ga = 2.39996323;
    float best = 1e9;
    float bestk = 0.0;
    int N = int(u_count);
    for (int i = 0; i < 512; i++) {
        if (i >= N) break;
        float fi = float(i);
        float r = u_spread * sqrt(fi) / sqrt(float(N));
        float ang = fi * ga + t;
        vec2 pc = vec2(cos(ang), sin(ang)) * r;
        float d = length(uv - pc);
        if (d < best) { best = d; bestk = fi / float(N); }
    }
    float dot = smoothstep(u_dotsize, u_dotsize * 0.4, best);
    vec3 col = mix(u_bg, inferno(bestk), dot);
    f_color = vec4(col, 1.0);
}
''', uniforms={
    "count":   {"glsl": "int", "min": 16, "max": 512, "default": 240,
                "description": "seed count"},
    "spread":  {"glsl": "float", "min": 0.2, "max": 1.2, "default": 0.85,
                "description": "spiral radius"},
    "dotsize": {"glsl": "float", "min": 0.004, "max": 0.06, "default": 0.02,
                "description": "dot radius"},
    "speed":   {"glsl": "float", "min": 0.0, "max": 6.0, "default": 1.0,
                "description": "rotation speed"},
    "bg":      {"glsl": "color", "default": "#05070e", "description": "background"},
})

_register("guilloche_typed", "Guilloché: rose-curve engraving lattice (typed, node 278)",
          "procedural", '''void main() {
    vec2 p = (v_uv - 0.5);
    p.x *= u_resolution.x / u_resolution.y;
    float t = u_time * 0.06 * u_speed;
    float r = length(p);
    float a = atan(p.y, p.x);
    float rose = cos(a * u_petals + t) * u_amp;
    float bands = sin((r - rose) * u_freq * 6.28318530);
    float line = smoothstep(1.0 - u_sharp, 1.0, abs(bands));
    vec3 col = mix(u_bg, u_ink, line);
    f_color = vec4(col, 1.0);
}
''', uniforms={
    "petals": {"glsl": "float", "min": 2.0, "max": 24.0, "default": 7.0,
               "description": "rose petal count"},
    "amp":    {"glsl": "float", "min": 0.0, "max": 0.3, "default": 0.08,
               "description": "rose amplitude"},
    "freq":   {"glsl": "float", "min": 4.0, "max": 60.0, "default": 24.0,
               "description": "ring frequency"},
    "sharp":  {"glsl": "float", "min": 0.05, "max": 0.9, "default": 0.4,
               "description": "line sharpness"},
    "speed":  {"glsl": "float", "min": 0.0, "max": 6.0, "default": 1.0,
               "description": "animation speed"},
    "bg":     {"glsl": "color", "default": "#060a10", "description": "background"},
    "ink":    {"glsl": "color", "default": "#7cf0ff", "description": "engraving"},
})

_register("lissajous_typed", "Lissajous: traced harmonic figure (typed, node 279)",
          "procedural", '''void main() {
    vec2 p = (v_uv - 0.5) * 2.0;
    p.x *= u_resolution.x / u_resolution.y;
    float t = u_time * 0.1 * u_speed;
    float best = 1e9;
    for (int i = 0; i < 240; i++) {
        float s = float(i) / 240.0 * 6.28318530;
        vec2 q = vec2(sin(u_fx * s + u_phase + t), sin(u_fy * s)) * u_scale;
        best = min(best, length(p - q));
    }
    float line = smoothstep(u_thick, u_thick * 0.3, best);
    vec3 col = mix(u_bg, u_fg, line);
    f_color = vec4(col, 1.0);
}
''', uniforms={
    "fx":    {"glsl": "float", "min": 1.0, "max": 12.0, "default": 3.0,
              "description": "x frequency"},
    "fy":    {"glsl": "float", "min": 1.0, "max": 12.0, "default": 2.0,
              "description": "y frequency"},
    "phase": {"glsl": "float", "min": 0.0, "max": 6.28, "default": 1.57,
              "description": "phase offset"},
    "scale": {"glsl": "float", "min": 0.3, "max": 0.95, "default": 0.8,
              "description": "figure size"},
    "thick": {"glsl": "float", "min": 0.01, "max": 0.12, "default": 0.04,
              "description": "trace thickness"},
    "speed": {"glsl": "float", "min": 0.0, "max": 6.0, "default": 1.0,
              "description": "drift speed"},
    "bg":    {"glsl": "color", "default": "#04060c", "description": "background"},
    "fg":    {"glsl": "color", "default": "#ffe66b", "description": "trace color"},
})

_register("interference_typed", "Radial wave interference from N sources (typed, node 280)",
          "procedural", _INFERNO_GPU + '''void main() {
    vec2 p = (v_uv - 0.5);
    p.x *= u_resolution.x / u_resolution.y;
    float t = u_time * 0.5 * u_speed;
    float acc = 0.0;
    int N = int(u_sources);
    for (int i = 0; i < 8; i++) {
        if (i >= N) break;
        float ang = float(i) / float(N) * 6.28318530;
        vec2 src = vec2(cos(ang), sin(ang)) * u_radius;
        float d = length(p - src);
        acc += sin(d * u_freq * 6.28318530 - t);
    }
    float v = 0.5 + 0.5 * acc / float(N);
    v = clamp(pow(v, u_contrast), 0.0, 1.0);
    vec3 col = inferno(v);
    f_color = vec4(col, 1.0);
}
''', uniforms={
    "sources":  {"glsl": "int", "min": 2, "max": 8, "default": 4,
                 "description": "wave source count"},
    "radius":   {"glsl": "float", "min": 0.1, "max": 0.6, "default": 0.35,
                 "description": "source ring radius"},
    "freq":     {"glsl": "float", "min": 2.0, "max": 40.0, "default": 14.0,
                 "description": "wave frequency"},
    "contrast": {"glsl": "float", "min": 0.3, "max": 4.0, "default": 1.4,
                 "description": "contrast gamma"},
    "speed":    {"glsl": "float", "min": 0.0, "max": 6.0, "default": 1.0,
                 "description": "wave speed"},
})

_register("flow_field_typed", "Curl-noise flow field streamlines (typed, node 281)",
          "procedural", '''void main() {
    vec2 p = (v_uv - 0.5);
    p.x *= u_resolution.x / u_resolution.y;
    float t = u_time * 0.1 * u_speed;
    vec2 q = p * u_zoom;
    float ang = fbm(q + vec2(t, -t)) * 6.28318530 * u_swirl;
    vec2 dir = vec2(cos(ang), sin(ang));
    float stripe = sin(dot(p, dir) * u_freq * 6.28318530 + t * 4.0);
    float line = smoothstep(1.0 - u_density, 1.0, abs(stripe));
    vec3 col = mix(u_bg, u_fg, line);
    f_color = vec4(col, 1.0);
}
''', uniforms={
    "zoom":    {"glsl": "float", "min": 0.5, "max": 6.0, "default": 2.5,
                "description": "noise zoom"},
    "swirl":   {"glsl": "float", "min": 0.2, "max": 3.0, "default": 1.0,
                "description": "flow curl amount"},
    "freq":    {"glsl": "float", "min": 4.0, "max": 40.0, "default": 16.0,
                "description": "streamline density"},
    "density": {"glsl": "float", "min": 0.05, "max": 0.8, "default": 0.35,
                "description": "line coverage"},
    "speed":   {"glsl": "float", "min": 0.0, "max": 6.0, "default": 1.0,
                "description": "animation speed"},
    "bg":      {"glsl": "color", "default": "#070510", "description": "background"},
    "fg":      {"glsl": "color", "default": "#8affc1", "description": "streamlines"},
})

_register("kaleido_bloom_typed", "Kaleidoscopic petal bloom (typed, node 282)",
          "procedural", _INFERNO_GPU + '''void main() {
    vec2 p = (v_uv - 0.5);
    p.x *= u_resolution.x / u_resolution.y;
    float t = u_time * 0.15 * u_speed;
    float r = length(p);
    float a = atan(p.y, p.x);
    float seg = 6.28318530 / u_slices;
    a = mod(a + t, seg);
    a = abs(a - seg * 0.5);
    float petal = cos(a * u_slices * 0.5) * sin(r * u_rings * 6.28318530 - t * 2.0);
    float v = clamp(0.5 + 0.5 * petal * u_gain, 0.0, 1.0);
    vec3 col = inferno(v);
    f_color = vec4(col, 1.0);
}
''', uniforms={
    "slices": {"glsl": "float", "min": 3.0, "max": 24.0, "default": 8.0,
               "description": "mirror slices"},
    "rings":  {"glsl": "float", "min": 1.0, "max": 20.0, "default": 6.0,
               "description": "radial rings"},
    "gain":   {"glsl": "float", "min": 0.3, "max": 3.0, "default": 1.3,
               "description": "intensity gain"},
    "speed":  {"glsl": "float", "min": 0.0, "max": 6.0, "default": 1.0,
               "description": "rotation speed"},
})

# ── Categorical coverage pt.9 (typed closed-form patterns, nodes 283-288) ──
# superformula, harmonograph, Maurer rose, magnetic dipole field, star polygon,
# torus-knot ribbon. Each is a pure f(uv, t) → exact CPU/GPU parity (P0.6),
# continuous-time motion only. Six more distinct math_art generators in the
# same family as 265-282.

_register("superformula_typed", "Superformula: Gielis radial curve sweep (typed, node 283)",
          "procedural", _INFERNO_GPU + '''void main() {
    vec2 p = (v_uv - 0.5);
    p.x *= u_resolution.x / u_resolution.y;
    float t = u_time * 0.1 * u_speed;
    float r = length(p);
    float a = atan(p.y, p.x);
    // Superformula radius for normalized angle a (continuous t rotation).
    float aa = a + t;
    float ca = cos(u_m * aa / 4.0);
    float sa = sin(u_n * aa / 4.0);
    float ra = pow(abs(ca), u_b) + pow(abs(sa), u_c);
    ra = pow(max(ra, 1e-4), -1.0 / u_p);
    float rr = ra * u_scale;
    float d = abs(r - rr);
    float line = smoothstep(u_thick, u_thick * 0.3, d);
    vec3 col = mix(u_bg, inferno(clamp(r / max(u_scale, 1e-3), 0.0, 1.0)), line);
    f_color = vec4(col, 1.0);
}
''', uniforms={
    "m":     {"glsl": "float", "min": 1.0, "max": 20.0, "default": 6.0,
              "description": "superformula m (symmetry)"},
    "n":     {"glsl": "float", "min": 1.0, "max": 20.0, "default": 8.0,
              "description": "superformula n"},
    "b":     {"glsl": "float", "min": 0.2, "max": 6.0, "default": 1.0,
              "description": "exponent b"},
    "c":     {"glsl": "float", "min": 0.2, "max": 6.0, "default": 1.0,
              "description": "exponent c"},
    "p":     {"glsl": "float", "min": 0.2, "max": 6.0, "default": 1.0,
              "description": "exponent p"},
    "scale": {"glsl": "float", "min": 0.3, "max": 1.2, "default": 0.85,
              "description": "curve radius"},
    "thick": {"glsl": "float", "min": 0.006, "max": 0.08, "default": 0.02,
              "description": "line thickness"},
    "speed": {"glsl": "float", "min": 0.0, "max": 6.0, "default": 1.0,
              "description": "rotation speed"},
    "bg":    {"glsl": "color", "default": "#04060c", "description": "background"},
})

_register("harmonograph_typed", "Harmonograph: decaying Lissajous trace (typed, node 284)",
          "procedural", '''void main() {
    vec2 p = (v_uv - 0.5) * 2.0;
    p.x *= u_resolution.x / u_resolution.y;
    float t = u_time * 0.3 * u_speed;
    float best = 1e9;
    int N = int(u_steps);
    for (int i = 0; i < 400; i++) {
        if (i >= N) break;
        float s = float(i) / float(N) * 6.28318530 * u_turns;
        float env = exp(-u_decay * float(i) / float(N));
        vec2 q = vec2(
            sin(u_fx * s + u_px + t) * env,
            sin(u_fy * s + u_py) * env
        ) * u_scale;
        best = min(best, length(p - q));
    }
    float line = smoothstep(u_thick, u_thick * 0.3, best);
    vec3 col = mix(u_bg, u_fg, line);
    f_color = vec4(col, 1.0);
}
''', uniforms={
    "fx":    {"glsl": "float", "min": 1.0, "max": 12.0, "default": 2.0,
              "description": "x frequency"},
    "fy":    {"glsl": "float", "min": 1.0, "max": 12.0, "default": 3.0,
              "description": "y frequency"},
    "px":    {"glsl": "float", "min": 0.0, "max": 6.28, "default": 0.0,
              "description": "x phase"},
    "py":    {"glsl": "float", "min": 0.0, "max": 6.28, "default": 1.57,
              "description": "y phase"},
    "decay": {"glsl": "float", "min": 0.0, "max": 4.0, "default": 1.2,
              "description": "amplitude decay"},
    "turns": {"glsl": "float", "min": 1.0, "max": 12.0, "default": 6.0,
              "description": "number of turns"},
    "steps": {"glsl": "int", "min": 60, "max": 400, "default": 300,
              "description": "trace resolution"},
    "scale": {"glsl": "float", "min": 0.3, "max": 0.95, "default": 0.8,
              "description": "figure size"},
    "thick": {"glsl": "float", "min": 0.01, "max": 0.12, "default": 0.04,
              "description": "trace thickness"},
    "speed": {"glsl": "float", "min": 0.0, "max": 6.0, "default": 1.0,
              "description": "drift speed"},
    "bg":    {"glsl": "color", "default": "#05070e", "description": "background"},
    "fg":    {"glsl": "color", "default": "#7ad7ff", "description": "trace color"},
})

_register("maurer_rose_typed", "Maurer rose: polygonal line sculpture (typed, node 285)",
          "procedural", _INFERNO_GPU + '''void main() {
    vec2 p = (v_uv - 0.5);
    p.x *= u_resolution.x / u_resolution.y;
    float t = u_time * 0.08 * u_speed;
    float best = 1e9;
    int N = int(u_steps);
    float d = 3.14159265 / 180.0 * u_deg;
    for (int i = 0; i < 720; i++) {
        if (i >= N) break;
        float k = float(i);
        float ang = k * d + t;
        float rr = u_scale * sin(u_petals * ang);
        vec2 q = vec2(cos(ang), sin(ang)) * rr;
        best = min(best, length(p - q));
    }
    float line = smoothstep(u_thick, u_thick * 0.3, best);
    vec3 col = mix(u_bg, inferno(clamp(length(p) / max(u_scale,1e-3), 0.0, 1.0)), line);
    f_color = vec4(col, 1.0);
}
''', uniforms={
    "petals": {"glsl": "float", "min": 2.0, "max": 20.0, "default": 6.0,
               "description": "rose petal count"},
    "deg":    {"glsl": "float", "min": 1.0, "max": 180.0, "default": 29.0,
               "description": "connector angle (deg)"},
    "steps":  {"glsl": "int", "min": 60, "max": 720, "default": 360,
               "description": "vertex count"},
    "scale":  {"glsl": "float", "min": 0.3, "max": 1.2, "default": 0.9,
               "description": "flower radius"},
    "thick":  {"glsl": "float", "min": 0.004, "max": 0.06, "default": 0.015,
               "description": "line thickness"},
    "speed":  {"glsl": "float", "min": 0.0, "max": 6.0, "default": 1.0,
               "description": "rotation speed"},
    "bg":     {"glsl": "color", "default": "#05060c", "description": "background"},
})

_register("magnetic_typed", "Magnetic dipole field: field-line ribbons (typed, node 286)",
          "procedural", _INFERNO_GPU + '''void main() {
    vec2 p = (v_uv - 0.5);
    p.x *= u_resolution.x / u_resolution.y;
    float t = u_time * 0.1 * u_speed;
    // Distance to a dipole at origin; field strength falls as 1/r^3 inside.
    float r = max(length(p), 0.04);
    float pa = atan(p.y, p.x) + t;
    // Dipole potential ~ cos^2(theta) - 0.5 ; draw iso-lines of it.
    float pot = cos(pa) * cos(pa) - 0.5;
    float bands = sin(pot * u_lines * 6.28318530 / max(r, 0.04) * u_tight);
    float line = smoothstep(1.0 - u_sharp, 1.0, abs(bands));
    vec3 col = mix(u_bg, inferno(clamp(1.0 - r / (u_scale + 1e-3), 0.0, 1.0)), line);
    f_color = vec4(col, 1.0);
}
''', uniforms={
    "lines":  {"glsl": "float", "min": 2.0, "max": 40.0, "default": 14.0,
               "description": "field-line count"},
    "tight":  {"glsl": "float", "min": 0.2, "max": 3.0, "default": 1.0,
               "description": "line tightness"},
    "sharp":  {"glsl": "float", "min": 0.05, "max": 0.9, "default": 0.4,
               "description": "line sharpness"},
    "scale":  {"glsl": "float", "min": 0.3, "max": 1.2, "default": 0.9,
               "description": "field radius"},
    "speed":  {"glsl": "float", "min": 0.0, "max": 6.0, "default": 1.0,
               "description": "rotation speed"},
    "bg":     {"glsl": "color", "default": "#04060c", "description": "background"},
})

_register("star_polygon_typed", "Star polygon {n/k}: connected vertex rosette (typed, node 287)",
          "procedural", _INFERNO_GPU + '''void main() {
    vec2 p = (v_uv - 0.5);
    p.x *= u_resolution.x / u_resolution.y;
    float t = u_time * 0.12 * u_speed;
    int N = int(u_points);
    int K = int(clamp(u_skip, 1.0, float(N) - 1.0));
    float best = 1e9;
    for (int i = 0; i < 240; i++) {
        if (i >= N) break;
        float a0 = (float(i) / float(N)) * 6.28318530 + t;
        float a1 = (float((i + K) % N) / float(N)) * 6.28318530 + t;
        vec2 v0 = vec2(cos(a0), sin(a0)) * u_scale;
        vec2 v1 = vec2(cos(a1), sin(a1)) * u_scale;
        vec2 d = v1 - v0;
        float l2 = max(dot(d, d), 1e-6);
        float h = clamp(dot(p - v0, d) / l2, 0.0, 1.0);
        best = min(best, length(p - (v0 + d * h)));
    }
    float line = smoothstep(u_thick, u_thick * 0.3, best);
    vec3 col = mix(u_bg, inferno(clamp(length(p) / max(u_scale,1e-3), 0.0, 1.0)), line);
    f_color = vec4(col, 1.0);
}
''', uniforms={
    "points": {"glsl": "int", "min": 5, "max": 40, "default": 12,
               "description": "vertex count n"},
    "skip":   {"glsl": "int", "min": 2, "max": 20, "default": 5,
               "description": "step k ({n/k})"},
    "scale":  {"glsl": "float", "min": 0.3, "max": 1.2, "default": 0.9,
               "description": "polygon radius"},
    "thick":  {"glsl": "float", "min": 0.004, "max": 0.06, "default": 0.012,
               "description": "line thickness"},
    "speed":  {"glsl": "float", "min": 0.0, "max": 6.0, "default": 1.0,
               "description": "rotation speed"},
    "bg":     {"glsl": "color", "default": "#05060c", "description": "background"},
})

_register("torusknot_typed", "Torus knot ribbon: parametric (p,q) knot (typed, node 288)",
          "procedural", _INFERNO_GPU + '''void main() {
    vec2 p = (v_uv - 0.5) * 2.0;
    p.x *= u_resolution.x / u_resolution.y;
    float t = u_time * 0.2 * u_speed;
    float best = 1e9;
    int N = int(u_steps);
    for (int i = 0; i < 600; i++) {
        if (i >= N) break;
        float s = float(i) / float(N) * 6.28318530;
        float r = cos(u_q * s) + u_rad;
        vec2 q = vec2(sin(u_p * s + t) * r, cos(u_p * s + t) * r) * u_scale;
        best = min(best, length(p - q));
    }
    float line = smoothstep(u_thick, u_thick * 0.3, best);
    float hue = clamp(atan(p.y, p.x) / 6.28318530 + 0.5, 0.0, 1.0);
    vec3 col = mix(u_bg, inferno(hue), line);
    f_color = vec4(col, 1.0);
}
''', uniforms={
    "p":     {"glsl": "float", "min": 1.0, "max": 12.0, "default": 2.0,
              "description": "knot winding p"},
    "q":     {"glsl": "float", "min": 1.0, "max": 12.0, "default": 3.0,
              "description": "knot winding q"},
    "rad":   {"glsl": "float", "min": 0.0, "max": 2.0, "default": 0.8,
              "description": "tube offset"},
    "steps": {"glsl": "int", "min": 120, "max": 600, "default": 400,
              "description": "curve resolution"},
    "scale": {"glsl": "float", "min": 0.2, "max": 0.8, "default": 0.45,
              "description": "knot size"},
    "thick": {"glsl": "float", "min": 0.01, "max": 0.12, "default": 0.035,
              "description": "ribbon thickness"},
    "speed": {"glsl": "float", "min": 0.0, "max": 6.0, "default": 1.0,
              "description": "rotation speed"},
    "bg":    {"glsl": "color", "default": "#04060c", "description": "background"},
})

# ── Typed closed-form patterns pt.10 (ids 289-294) ─────────────────────────
# Categorical coverage continuation (2026-07-11): classic generative-art
# patterns with NAMED typed controls — infinite zoom tunnel, vortex/galaxy
# field, woven fabric, topographic contour map, cross-hatch engraving, and a
# domain-warped grid lattice. All closed-form f(uv,t); additive live-preview
# twins. CPU fns stay authoritative; these are a convenience layer.

_register("tunnel_typed", "Infinite zoom tunnel: polar depth-warp with typed arms/freq/falloff (node 289)",
          "procedural", _INFERNO_GPU + '''void main() {
    vec2 p = (v_uv - 0.5) * 2.0;
    p.x *= u_resolution.x / u_resolution.y;
    float r = max(length(p), 1e-3);
    float a = atan(p.y, p.x);
    float t = u_time * u_speed;
    float depth = u_scale / r + t * 0.5;
    float rings = 0.5 + 0.5 * sin(depth * u_freq);
    float spokes = 0.5 + 0.5 * sin(a * u_arms + depth * 0.5);
    float v = rings * 0.6 + spokes * 0.4;
    vec3 col = inferno(fract(depth) * 0.9 + 0.05);
    col = mix(u_bg, col, smoothstep(0.0, u_falloff, r));
    f_color = vec4(col, 1.0);
}
''', uniforms={
    "speed":   {"glsl": "float", "min": 0.0, "max": 4.0, "default": 1.0,
                "description": "zoom speed"},
    "scale":   {"glsl": "float", "min": 0.05, "max": 1.5, "default": 0.35,
                "description": "tunnel depth scale"},
    "freq":    {"glsl": "float", "min": 1.0, "max": 40.0, "default": 12.0,
                "description": "ring frequency"},
    "arms":    {"glsl": "float", "min": 1.0, "max": 16.0, "default": 6.0,
                "description": "spoke count"},
    "falloff": {"glsl": "float", "min": 0.1, "max": 1.5, "default": 0.55,
                "description": "edge fade"},
    "bg":      {"glsl": "color", "default": "#040610", "description": "vanishing point"},
})

_register("vortex_typed", "Spiral vortex / galaxy field with typed arms/twist/falloff (node 290)",
          "procedural", '''void main() {
    vec2 p = (v_uv - 0.5) * 2.0;
    p.x *= u_resolution.x / u_resolution.y;
    float r = length(p);
    float a = atan(p.y, p.x);
    float t = u_time * u_speed;
    float swirl = a + r * u_twist - t;
    float bands = 0.5 + 0.5 * sin(swirl * u_arms);
    float density = exp(-r * u_falloff);
    vec3 col = mix(u_bg, mix(u_color_a, u_color_b, bands), density);
    f_color = vec4(col, 1.0);
}
''', uniforms={
    "speed":   {"glsl": "float", "min": 0.0, "max": 4.0, "default": 0.8,
                "description": "spin speed"},
    "arms":    {"glsl": "float", "min": 1.0, "max": 24.0, "default": 4.0,
                "description": "spiral arm count"},
    "twist":   {"glsl": "float", "min": -8.0, "max": 8.0, "default": 3.0,
                "description": "winding tightness"},
    "falloff": {"glsl": "float", "min": 0.5, "max": 8.0, "default": 2.6,
                "description": "core brightness falloff"},
    "color_a": {"glsl": "color", "default": "#1b2a6b", "description": "arm color A"},
    "color_b": {"glsl": "color", "default": "#ffd27a", "description": "arm color B"},
    "bg":      {"glsl": "color", "default": "#050308", "description": "background"},
})

_register("weave_typed", "Woven fabric: over/under threads on a typed checker grid (node 291)",
          "procedural", '''void main() {
    vec2 uv = v_uv * u_scale;
    uv += u_time * u_speed * 0.05;
    vec2 g = floor(uv);
    vec2 fv = fract(uv);
    float parity = mod(g.x + g.y, 2.0);
    float bulge;
    float along;
    if (parity < 0.5) { bulge = sin(fv.y * 3.14159265); along = fv.x; }
    else              { bulge = sin(fv.x * 3.14159265); along = fv.y; }
    float thread = smoothstep(0.0, 0.5, bulge) * smoothstep(1.0, 0.5, bulge);
    float shade = 0.5 + 0.5 * sin(along * 3.14159265);
    vec3 base = (parity < 0.5) ? u_color_a : u_color_b;
    vec3 col = base * (0.45 + 0.6 * shade);
    col *= (0.35 + 0.65 * thread);
    f_color = vec4(col, 1.0);
}
''', uniforms={
    "scale":   {"glsl": "float", "min": 2.0, "max": 40.0, "default": 12.0,
                "description": "thread count"},
    "speed":   {"glsl": "float", "min": 0.0, "max": 4.0, "default": 1.0,
                "description": "drift speed"},
    "color_a": {"glsl": "color", "default": "#b3421f", "description": "weft color"},
    "color_b": {"glsl": "color", "default": "#1f5ab3", "description": "warp color"},
})

_register("contour_typed", "Topographic contour map of FBM terrain with typed levels/thickness (node 292)",
          "procedural", '''void main() {
    vec2 p = (v_uv - 0.5) * 2.0;
    p.x *= u_resolution.x / u_resolution.y;
    float n = fbm(p * u_freq + u_time * 0.03 * u_speed);
    vec3 col = mix(u_color_a, u_color_b, clamp(n, 0.0, 1.0));
    float c = n * u_levels;
    float d = abs(fract(c) - 0.5) * 2.0;
    float line = smoothstep(u_thick, u_thick * 0.3, d);
    col = mix(col, u_line, line);
    f_color = vec4(col, 1.0);
}
''', uniforms={
    "speed":  {"glsl": "float", "min": 0.0, "max": 4.0, "default": 1.0,
               "description": "terrain drift"},
    "freq":   {"glsl": "float", "min": 1.0, "max": 12.0, "default": 3.5,
               "description": "terrain feature size"},
    "levels": {"glsl": "float", "min": 2.0, "max": 40.0, "default": 14.0,
               "description": "contour line count"},
    "thick":  {"glsl": "float", "min": 0.02, "max": 0.4, "default": 0.12,
               "description": "line thickness"},
    "color_a": {"glsl": "color", "default": "#0d3b2e", "description": "low elevation"},
    "color_b": {"glsl": "color", "default": "#e8d8a0", "description": "high elevation"},
    "line":   {"glsl": "color", "default": "#1a1208", "description": "contour ink"},
})

_register("hatch_typed", "Cross-hatch engraving shading over procedural luminance (node 293)",
          "procedural", '''void main() {
    vec2 p = (v_uv - 0.5) * 2.0;
    p.x *= u_resolution.x / u_resolution.y;
    float lum = fbm(p * u_freq + u_time * 0.04 * u_speed);
    float ang = radians(u_angle);
    vec2 dir = vec2(cos(ang), sin(ang));
    float h1 = step(0.5, fract(dot(p, dir) * u_density));
    vec2 dir2 = vec2(cos(ang + 1.5707963), sin(ang + 1.5707963));
    float h2 = step(0.5, fract(dot(p, dir2) * u_density));
    float ink = (1.0 - lum) * h1;
    ink = max(ink, (1.0 - lum * 0.5) * h2 * step(0.5, lum));
    vec3 col = mix(u_paper, u_ink, clamp(ink, 0.0, 1.0));
    f_color = vec4(col, 1.0);
}
''', uniforms={
    "speed":   {"glsl": "float", "min": 0.0, "max": 4.0, "default": 1.0,
                "description": "luminance drift"},
    "freq":    {"glsl": "float", "min": 1.0, "max": 12.0, "default": 3.0,
                "description": "shading feature size"},
    "angle":   {"glsl": "float", "min": 0.0, "max": 90.0, "default": 35.0,
                "description": "hatch angle (deg)"},
    "density": {"glsl": "float", "min": 4.0, "max": 80.0, "default": 28.0,
                "description": "line density"},
    "paper":   {"glsl": "color", "default": "#f2efe2", "description": "paper"},
    "ink":     {"glsl": "color", "default": "#15110c", "description": "ink"},
})

_register("gridwarp_typed", "Domain-warped grid lattice with typed warp/cells/width (node 294)",
          "procedural", '''void main() {
    vec2 g = v_uv * u_cells;
    vec2 w = vec2(
        fbm(g * 0.5 + u_time * 0.05 * u_speed),
        fbm(g * 0.5 + 7.3 - u_time * 0.05 * u_speed)
    ) - 0.5;
    g += w * u_warp;
    vec2 f = fract(g);
    float lx = smoothstep(u_width, 0.0, min(f.x, 1.0 - f.x));
    float ly = smoothstep(u_width, 0.0, min(f.y, 1.0 - f.y));
    float line = max(lx, ly);
    vec3 col = mix(u_bg, u_line, line);
    f_color = vec4(col, 1.0);
}
''', uniforms={
    "speed":  {"glsl": "float", "min": 0.0, "max": 4.0, "default": 1.0,
               "description": "warp flow speed"},
    "cells":  {"glsl": "float", "min": 2.0, "max": 60.0, "default": 14.0,
               "description": "grid cell count"},
    "width":  {"glsl": "float", "min": 0.02, "max": 0.4, "default": 0.12,
               "description": "line width"},
    "warp":   {"glsl": "float", "min": 0.0, "max": 2.0, "default": 0.7,
               "description": "domain warp strength"},
    "bg":     {"glsl": "color", "default": "#0a0a12", "description": "background"},
    "line":   {"glsl": "color", "default": "#43e8d8", "description": "grid line"},
})

# ── Typed-uniform closed-form pattern batch (2026-07-11, nodes 295-300) ──
# Extended family of single-output procedural nodes with NAMED typed controls.
# Each is a pure function of (uv, t) — exact parity live preview, no seeded
# layout divergence. Reuses the prologue helpers (fbm/noise/hash21/rot) and an
# inlined hsv2rgb (no dependency on the late _INFERNO_GPU helper).

_register("domainwarp_typed", "Domain-warped fractal flow field (typed, node 295)",
          "procedural", '''vec3 _hsv(float h, float s, float v) {
    vec3 k = vec3(1.0, 2.0/3.0, 1.0/3.0);
    vec3 p = abs(fract(vec3(h) + k) * 6.0 - 3.0);
    return v * mix(vec3(1.0), clamp(p - 1.0, 0.0, 1.0), s);
}
void main() {
    vec2 p = (v_uv - 0.5) * 2.0;
    p.x *= u_resolution.x / u_resolution.y;
    float t = u_time * 0.05 * u_speed;
    vec2 q = vec2(fbm(p * u_scale + t), fbm(p * u_scale + 5.2 - t));
    vec2 r = vec2(fbm(p * u_scale + u_warp * q + 1.7),
                  fbm(p * u_scale + u_warp * q + 9.2));
    float v = fbm(p * u_scale + u_warp * r);
    float hue = fract(v * u_hue_spread + u_hue_shift);
    vec3 col = _hsv(hue, u_sat, 0.35 + 0.65 * v);
    f_color = vec4(col, 1.0);
}
''', uniforms={
    "speed":        {"glsl": "float", "min": 0.0, "max": 4.0, "default": 1.0,
                    "description": "flow speed"},
    "scale":        {"glsl": "float", "min": 1.0, "max": 12.0, "default": 3.5,
                    "description": "noise frequency"},
    "warp":         {"glsl": "float", "min": 0.0, "max": 6.0, "default": 3.0,
                    "description": "domain-warp iterations"},
    "hue_shift":    {"glsl": "float", "min": 0.0, "max": 1.0, "default": 0.6,
                    "description": "base hue"},
    "hue_spread":   {"glsl": "float", "min": 0.0, "max": 1.5, "default": 0.7,
                    "description": "hue range across field"},
    "sat":          {"glsl": "float", "min": 0.0, "max": 1.0, "default": 0.8,
                    "description": "saturation"},
})

_register("caustics_typed", "Animated water caustics (typed, node 296)",
          "procedural", '''vec3 _hsv(float h, float s, float v) {
    vec3 k = vec3(1.0, 2.0/3.0, 1.0/3.0);
    vec3 p = abs(fract(vec3(h) + k) * 6.0 - 3.0);
    return v * mix(vec3(1.0), clamp(p - 1.0, 0.0, 1.0), s);
}
void main() {
    vec2 p = v_uv * u_scale;
    p.x *= u_resolution.x / u_resolution.y;
    float t = u_time * 0.1 * u_speed;
    vec2 w = vec2(0.0);
    for (int i = 0; i < 5; i++) {
        float fi = float(i);
        w += vec2(sin(p.y * (1.0 + fi * 0.3) + t + fi),
                  cos(p.x * (1.0 + fi * 0.3) - t * 1.1 + fi * 1.7));
        p *= 1.4;
    }
    float c = 1.0 - abs(sin(w.x + w.y) * 0.5 + 0.5);
    c = pow(c, u_sharp);
    vec3 col = mix(u_deep, u_shallow, c);
    f_color = vec4(col, 1.0);
}
''', uniforms={
    "speed":    {"glsl": "float", "min": 0.0, "max": 4.0, "default": 1.0,
                "description": "ripple speed"},
    "scale":    {"glsl": "float", "min": 1.0, "max": 16.0, "default": 6.0,
                "description": "ripple density"},
    "sharp":    {"glsl": "float", "min": 1.0, "max": 12.0, "default": 4.0,
                "description": "caustic sharpness"},
    "deep":     {"glsl": "color", "default": "#02121f", "description": "deep water"},
    "shallow":  {"glsl": "color", "default": "#4fd6ff", "description": "lit water"},
})

_register("prism_typed", "Spectral prism / diffraction grating (typed, node 297)",
          "procedural", '''vec3 _hsv(float h, float s, float v) {
    vec3 k = vec3(1.0, 2.0/3.0, 1.0/3.0);
    vec3 p = abs(fract(vec3(h) + k) * 6.0 - 3.0);
    return v * mix(vec3(1.0), clamp(p - 1.0, 0.0, 1.0), s);
}
void main() {
    vec2 p = (v_uv - 0.5) * 2.0;
    p.x *= u_resolution.x / u_resolution.y;
    float t = u_time * 0.03 * u_speed;
    float d = dot(p, vec2(cos(u_angle), sin(u_angle)));
    float spec = sin(d * u_freq + t) * 0.5 + 0.5;
    float bands = spec * u_rainbow;
    vec3 col = _hsv(fract(bands + u_hue_shift), u_sat, 1.0);
    float vig = smoothstep(u_falloff, 0.0, length(p));
    col *= mix(1.0, vig, u_darken);
    f_color = vec4(col, 1.0);
}
''', uniforms={
    "speed":     {"glsl": "float", "min": 0.0, "max": 4.0, "default": 1.0,
                "description": "phase drift"},
    "freq":      {"glsl": "float", "min": 2.0, "max": 60.0, "default": 22.0,
                "description": "grating frequency"},
    "angle":     {"glsl": "float", "min": 0.0, "max": 360.0, "default": 30.0,
                "description": "grating angle (deg)"},
    "rainbow":   {"glsl": "float", "min": 0.2, "max": 3.0, "default": 1.2,
                "description": "hue spread"},
    "hue_shift": {"glsl": "float", "min": 0.0, "max": 1.0, "default": 0.0,
                "description": "hue offset"},
    "sat":       {"glsl": "float", "min": 0.0, "max": 1.0, "default": 0.9,
                "description": "saturation"},
    "falloff":   {"glsl": "float", "min": 0.2, "max": 2.0, "default": 1.0,
                "description": "edge falloff"},
    "darken":    {"glsl": "float", "min": 0.0, "max": 1.0, "default": 0.5,
                "description": "edge darkening"},
})

_register("sdfscene_typed", "Minimal signed-distance scene (typed, node 298)",
          "procedural", '''vec3 _hsv(float h, float s, float v) {
    vec3 k = vec3(1.0, 2.0/3.0, 1.0/3.0);
    vec3 p = abs(fract(vec3(h) + k) * 6.0 - 3.0);
    return v * mix(vec3(1.0), clamp(p - 1.0, 0.0, 1.0), s);
}
float _sdCircle(vec2 p, float r) { return length(p) - r; }
float _sdBox(vec2 p, vec2 b) {
    vec2 d = abs(p) - b;
    return length(max(d, 0.0)) + min(max(d.x, d.y), 0.0);
}
void main() {
    vec2 p = (v_uv - 0.5) * 2.0;
    p.x *= u_resolution.x / u_resolution.y;
    float t = u_time * 0.2 * u_speed;
    p *= u_zoom;
    vec2 c = vec2(cos(t), sin(t * 0.7)) * u_orbit;
    float d = min(_sdCircle(p - c, u_rad),
                  _sdBox(rot(t * u_spin) * p, vec2(u_box)));
    float aa = fwidth(d) + 0.002;
    float mask = 1.0 - smoothstep(0.0, aa, d);
    float rim = smoothstep(0.0, aa, abs(d) - u_rim);
    vec3 col = mix(u_bg, u_fill, mask);
    col = mix(col, u_rimc, rim * (1.0 - mask));
    f_color = vec4(col, 1.0);
}
''', uniforms={
    "speed":   {"glsl": "float", "min": 0.0, "max": 4.0, "default": 1.0,
                "description": "scene spin speed"},
    "zoom":    {"glsl": "float", "min": 0.3, "max": 3.0, "default": 1.0,
                "description": "camera zoom"},
    "orbit":   {"glsl": "float", "min": 0.0, "max": 1.5, "default": 0.45,
                "description": "circle orbit radius"},
    "rad":     {"glsl": "float", "min": 0.05, "max": 1.0, "default": 0.3,
                "description": "circle radius"},
    "box":     {"glsl": "float", "min": 0.05, "max": 1.0, "default": 0.28,
                "description": "box half-size"},
    "spin":    {"glsl": "float", "min": -3.0, "max": 3.0, "default": 0.6,
                "description": "box spin rate"},
    "rim":     {"glsl": "float", "min": 0.0, "max": 0.1, "default": 0.04,
                "description": "rim width"},
    "bg":      {"glsl": "color", "default": "#0b0b14", "description": "background"},
    "fill":    {"glsl": "color", "default": "#ff5d73", "description": "shape fill"},
    "rimc":    {"glsl": "color", "default": "#ffe66d", "description": "rim color"},
})

_register("burst_typed", "Radial energy burst / shockwave (typed, node 299)",
          "procedural", '''vec3 _hsv(float h, float s, float v) {
    vec3 k = vec3(1.0, 2.0/3.0, 1.0/3.0);
    vec3 p = abs(fract(vec3(h) + k) * 6.0 - 3.0);
    return v * mix(vec3(1.0), clamp(p - 1.0, 0.0, 1.0), s);
}
void main() {
    vec2 p = (v_uv - 0.5) * 2.0;
    p.x *= u_resolution.x / u_resolution.y;
    float t = u_time * 0.1 * u_speed;
    float r = length(p);
    float a = atan(p.y, p.x);
    float wave = sin(r * u_freq - t * u_velocity) * 0.5 + 0.5;
    float spokes = pow(abs(cos(a * u_spokes * 0.5 + t * 0.2)), u_sharpness);
    float ring = smoothstep(u_thick, 0.0, abs(r - fract(t * u_velocity * 0.05) * u_reach)) * u_intensity;
    float energy = (wave * 0.4 + spokes * 0.4 + ring * 0.8);
    energy *= smoothstep(u_reach, 0.0, r);
    vec3 col = mix(u_bg, u_hot, clamp(energy, 0.0, 1.0));
    col = mix(col, u_core, smoothstep(0.6, 1.0, energy));
    f_color = vec4(col, 1.0);
}
''', uniforms={
    "speed":      {"glsl": "float", "min": 0.0, "max": 4.0, "default": 1.0,
                "description": "animation speed"},
    "freq":       {"glsl": "float", "min": 2.0, "max": 40.0, "default": 12.0,
                "description": "radial wave frequency"},
    "velocity":   {"glsl": "float", "min": 1.0, "max": 20.0, "default": 6.0,
                "description": "shockwave speed"},
    "spokes":     {"glsl": "float", "min": 1.0, "max": 24.0, "default": 8.0,
                "description": "spoke count"},
    "sharpness":  {"glsl": "float", "min": 1.0, "max": 12.0, "default": 4.0,
                "description": "spoke sharpness"},
    "thick":      {"glsl": "float", "min": 0.01, "max": 0.3, "default": 0.06,
                "description": "ring thickness"},
    "reach":      {"glsl": "float", "min": 0.4, "max": 2.0, "default": 1.4,
                "description": "burst reach"},
    "intensity":  {"glsl": "float", "min": 0.0, "max": 1.0, "default": 0.8,
                "description": "ring intensity"},
    "bg":         {"glsl": "color", "default": "#05060f", "description": "background"},
    "hot":        {"glsl": "color", "default": "#ff7a18", "description": "hot ring"},
    "core":       {"glsl": "color", "default": "#fff2c4", "description": "core flash"},
})

_register("foam_typed", "Procedural bubble foam / Voronoi cell membrane (typed, node 300)",
          "procedural", '''vec3 _hsv(float h, float s, float v) {
    vec3 k = vec3(1.0, 2.0/3.0, 1.0/3.0);
    vec3 p = abs(fract(vec3(h) + k) * 6.0 - 3.0);
    return v * mix(vec3(1.0), clamp(p - 1.0, 0.0, 1.0), s);
}
void main() {
    vec2 uv = v_uv * u_cells;
    uv.x *= u_resolution.x / u_resolution.y;
    float t = u_time * 0.05 * u_speed;
    vec2 g = floor(uv); vec2 f = fract(uv);
    float md = 1e9; vec2 mp = vec2(0.0);
    for (int j = -1; j <= 1; j++) {
        for (int i = -1; i <= 1; i++) {
            vec2 o = vec2(float(i), float(j));
            vec2 cell = g + o;
            vec2 off = vec2(hash21(cell), hash21(cell + 3.3));
            off = 0.5 + 0.45 * sin(t + 6.2831 * off);
            vec2 r = o + off - f;
            float dd = dot(r, r);
            if (dd < md) { md = dd; mp = r; }
        }
    }
    float dist = sqrt(md);
    float edge = smoothstep(u_thick, 0.0, dist);
    float irid = fract(dist * u_irid + u_hue_shift);
    vec3 col = mix(u_bg, _hsv(irid, u_sat, 1.0), edge);
    f_color = vec4(col, 1.0);
}
''', uniforms={
    "speed":    {"glsl": "float", "min": 0.0, "max": 4.0, "default": 1.0,
                "description": "cell jitter speed"},
    "cells":    {"glsl": "float", "min": 2.0, "max": 40.0, "default": 12.0,
                "description": "cell count"},
    "thick":    {"glsl": "float", "min": 0.01, "max": 0.4, "default": 0.12,
                "description": "membrane thickness"},
    "irid":     {"glsl": "float", "min": 0.0, "max": 4.0, "default": 1.5,
                "description": "iridescence bands"},
    "hue_shift":{"glsl": "float", "min": 0.0, "max": 1.0, "default": 0.55,
                "description": "hue offset"},
    "sat":      {"glsl": "float", "min": 0.0, "max": 1.0, "default": 0.7,
                "description": "saturation"},
    "bg":       {"glsl": "color", "default": "#06121a", "description": "background"},
})


# ═══════════════════════════════════════════════════════════════════════════
#  P1.5 — Dendritic Solidification (node 122) phase-field ping-pong twin
# ═══════════════════════════════════════════════════════════════════════════
# Allen-Cahn phase field φ (.r) coupled to a passive thermal field u (.g).
# Anisotropic interface width W(θ)=W0(1+ε cos(kθ)) gives the 4-fold dendrite
# branching. Faithful to the CPU node's ACTUAL (simplified) update — it uses the
# W²∇²φ diffusion form, constant driving force, double-well f'(φ). The CPU numpy
# node stays the authoritative export; every param is clamped to its documented
# range so the twin is robust to the client's neutral u_params fallback.
_register("dendrite_seed",
          "Dendritic seed: single tanh nucleus at center + thermal bump (node 122 twin)",
          "procedural", '''
void main() {
    vec2 res = u_resolution;
    vec2 p = v_uv * res;
    float dist = length(p - 0.5 * res);
    float W0 = 0.5, seedR = 12.0;
    float phi = tanh((seedR - dist) / (W0 * 1.41421356));   // φ=+1 solid core → −1 liquid
    float u0 = clamp(u_params.x, -1.0, -0.1);               // undercooling
    float u  = clamp(u0 + 0.3 * exp(-(dist * dist) / 100.0), -1.0, 1.0);
    f_color = vec4(phi, u, 0.0, 1.0);
}
''')

_register("dendrite_step",
          "Dendritic Allen-Cahn step: anisotropic W(θ)²∇²φ + double-well + thermal (node 122)",
          "procedural", '''
void main() {
    vec2 texel = 1.0 / u_resolution;
    vec4 s  = texture(u_texture, v_uv);
    float phi = s.r, uu = s.g;
    vec4 sl = texture(u_texture, v_uv + vec2(-texel.x, 0.0));
    vec4 sr = texture(u_texture, v_uv + vec2( texel.x, 0.0));
    vec4 st = texture(u_texture, v_uv + vec2(0.0,  texel.y));
    vec4 sb = texture(u_texture, v_uv + vec2(0.0, -texel.y));
    float lapPhi = sl.r + sr.r + st.r + sb.r - 4.0 * phi;
    float lapU   = sl.g + sr.g + st.g + sb.g - 4.0 * uu;
    float phix = (sr.r - sl.r) * 0.5;
    float phiy = (st.r - sb.r) * 0.5;
    float theta = atan(phiy, phix);
    float eps = clamp(u_params.y, 0.0, 0.1);                 // anisotropy
    float k   = floor(clamp(u_params.z, 3.0, 8.0) + 0.5);    // symmetry (int fold)
    float W0 = 0.5;
    float w  = W0 * (1.0 + eps * cos(k * theta));
    float aniso = (w * w) * lapPhi;
    float fp    = 4.0 * phi * (phi * phi - 1.0);             // f'(φ)=(φ²−1)²'
    float drive = 4.0 * (1.0 - phi * phi);                   // D_DRIVE=4
    float M  = 50.0;
    // Cap the step for explicit-scheme stability (M·dt·w² must stay <~0.25, else
    // the interior checkerboards). The CPU node masks this with a periodic
    // Gaussian blur; the twin caps dt + lightly smooths in display instead.
    float dt = min(clamp(u_params.w, 0.005, 0.2), 0.02);
    float dphi = M * (aniso - fp + drive);
    float phiN = clamp(phi + dt * dphi, -1.0, 1.0);
    float du   = 6.0 * lapU + 0.3 * max(dphi, 0.0);          // D_THERMAL=6 + latent heat
    float uN   = clamp(uu + dt * du, -1.0, 1.0);
    f_color = vec4(phiN, uN, 0.0, 1.0);
}
''')

_register("dendrite_display",
          "Dendritic display: φ → grayscale + thin interface outline (node 122)",
          "procedural", '''
void main() {
    vec2 texel = 1.0 / u_resolution;
    float phi = texture(u_texture, v_uv).r;
    float pl = texture(u_texture, v_uv + vec2(-texel.x, 0.0)).r;
    float pr = texture(u_texture, v_uv + vec2( texel.x, 0.0)).r;
    float pt = texture(u_texture, v_uv + vec2(0.0,  texel.y)).r;
    float pb = texture(u_texture, v_uv + vec2(0.0, -texel.y)).r;
    float pa = texture(u_texture, v_uv + vec2(-texel.x,  texel.y)).r;
    float pc = texture(u_texture, v_uv + vec2( texel.x,  texel.y)).r;
    float pe = texture(u_texture, v_uv + vec2(-texel.x, -texel.y)).r;
    float pf = texture(u_texture, v_uv + vec2( texel.x, -texel.y)).r;
    // 3×3 Gaussian-ish smooth of φ (suppresses the explicit-scheme checkerboard,
    // mirroring the CPU node's light periodic blur).
    float phiS = (phi * 4.0 + (pl + pr + pt + pb) * 2.0 + (pa + pc + pe + pf)) / 16.0;
    float gray = (phiS + 1.0) * 0.5;
    float gmag = length(vec2((pr - pl) * 0.5, (pt - pb) * 0.5));
    if (abs(phi) < 0.2 && gmag > 0.15) gray = 1.0;   // thin interface outline
    f_color = vec4(gray, gray, gray, 1.0);
}
''')


# ═══════════════════════════════════════════════════════════════════════════
#  P1.5 — Fractional Laplacian Reaction-Diffusion (node 163) RD twin
# ═══════════════════════════════════════════════════════════════════════════
# Gray-Scott reaction (identical kinetics: −UV²+F(1−U) / +UV²−(F+k)V) with the
# fractional exponent α approximated LOCALLY as an effective substrate-diffusion
# breadth (lower α → broader Du → coarser/more-replicating spots). The true
# Fourier (-∇²)^(α/2) operator is not reproduced — the CPU numpy node stays the
# authoritative export. Seed reuses grayscott_seed; display uses the node's fire
# colormap. State: U in .r, V in .g.
_register("frac_rd_step",
          "Fractional-RD step: Gray-Scott reaction + α-modulated diffusion breadth (node 163)",
          "procedural", '''
void main() {
    vec2 texel = 1.0 / u_resolution;
    vec4 s  = texture(u_texture, v_uv);
    float U = s.r, V = s.g;
    vec4 sl = texture(u_texture, v_uv + vec2(-texel.x, 0.0));
    vec4 sr = texture(u_texture, v_uv + vec2( texel.x, 0.0));
    vec4 st = texture(u_texture, v_uv + vec2(0.0,  texel.y));
    vec4 sb = texture(u_texture, v_uv + vec2(0.0, -texel.y));
    float lapU = sl.r + sr.r + st.r + sb.r - 4.0 * U;
    float lapV = sl.g + sr.g + st.g + sb.g - 4.0 * V;
    float F     = clamp(u_params.x, 0.01, 0.08);
    float k     = clamp(u_params.y, 0.03, 0.08);
    float alpha = clamp(u_params.z, 0.5, 2.0);
    float Dv    = clamp(u_params.w, 0.005, 0.2);
    // Lévy-flight proxy: lower α → broader substrate diffusion (sharper fronts,
    // more self-replication). α=2 → Du=1.5·Dv; α=0.5 → Du≈3.75·Dv.
    float Du = Dv * (1.5 + (2.0 - alpha) * 1.5);
    float uvv = U * V * V;
    float nU = clamp(U + (Du * lapU - uvv + F * (1.0 - U)), 0.0, 1.0);
    float nV = clamp(V + (Dv * lapV + uvv - (F + k) * V), 0.0, 1.0);
    f_color = vec4(nU, nV, 0.0, 1.0);
}
''')

_register("frac_rd_display",
          "Fractional-RD display: activator V → fire colormap, gamma (node 163)",
          "procedural", '''
void main() {
    float V = clamp(texture(u_texture, v_uv).g, 0.0, 1.0);
    float v = sqrt(V);                       // gamma stretch (matches CPU V**0.5)
    float r = clamp(4.0 * v,        0.0, 1.0);
    float g = clamp(4.0 * v - 1.0,  0.0, 1.0);
    float b = clamp(4.0 * v - 3.0,  0.0, 1.0);
    f_color = vec4(r, g, b, 1.0);            // dark → red → orange → yellow → white
}
''')


# ═══════════════════════════════════════════════════════════════════════════
#  P1.6 — Active Nematic Liquid Crystals (node 99) Q-tensor field twin
# ═══════════════════════════════════════════════════════════════════════════
# Simplified Landau-de Gennes Q-tensor model. State packs the two independent
# components of the traceless symmetric 2×2 tensor: Qxx in .r, Qxy in .g. Explicit
# Euler: ∂Q/∂t = Γ·H + α·Q + D·∇²Q + noise, with H = -(A·Q + C·Tr(Q²)·Q). Thermal
# noise (which nucleates the ±½ defects) is reproduced as state-dependent hash
# noise that varies each substep. Display = schlieren texture (director hue,
# order-parameter brightness) + defect glow. CPU numpy node authoritative.
_register("nematic_seed",
          "Active-nematic seed: small random Q-tensor (Qxx=.r, Qxy=.g) (node 99)",
          "procedural", '''
void main() {
    float a = hash21(v_uv * 311.0 + 1.7) - 0.5;
    float b = hash21(v_uv * 517.0 + 9.3) - 0.5;
    f_color = vec4(a * 0.12, b * 0.12, 0.0, 1.0);   // ~N(0,0.05) initial order
}
''')

_register("nematic_step",
          "Active-nematic step: Landau-de Gennes + activity + elastic ∇²Q + hash noise (node 99)",
          "procedural", '''
void main() {
    vec2 texel = 1.0 / u_resolution;
    vec4 s  = texture(u_texture, v_uv);
    float Qxx = s.r, Qxy = s.g;
    vec4 sl = texture(u_texture, v_uv + vec2(-texel.x, 0.0));
    vec4 sr = texture(u_texture, v_uv + vec2( texel.x, 0.0));
    vec4 st = texture(u_texture, v_uv + vec2(0.0,  texel.y));
    vec4 sb = texture(u_texture, v_uv + vec2(0.0, -texel.y));
    float Lxx = sl.r + sr.r + st.r + sb.r - 4.0 * Qxx;
    float Lxy = sl.g + sr.g + st.g + sb.g - 4.0 * Qxy;
    float alpha = clamp(u_params.x, -0.2, 0.2);      // activity α
    float D     = clamp(u_params.y, 0.01, 2.0);      // elastic constant
    float A     = clamp(u_params.z, -0.5, 0.1);      // Landau A
    float noise = clamp(u_params.w, 0.0, 0.15);      // thermal noise amplitude
    const float C = 1.0, G = 1.0, dt = 0.05;
    float S2 = 2.0 * (Qxx * Qxx + Qxy * Qxy);        // Tr(Q²)
    float Hxx = -(A * Qxx + C * S2 * Qxx);
    float Hxy = -(A * Qxy + C * S2 * Qxy);
    float dQxx = dt * (G * Hxx + alpha * Qxx + D * Lxx);
    float dQxy = dt * (G * Hxy + alpha * Qxy + D * Lxy);
    // Thermal noise (nucleates defects) — state-dependent so it varies per substep.
    float n1 = hash21(v_uv * 512.0 + vec2(Qxx, Qxy) * 813.0 + 2.3) - 0.5;
    float n2 = hash21(v_uv * 727.0 + vec2(Qxy, Qxx) * 611.0 + 7.1) - 0.5;
    dQxx += noise * n1 * 2.0 * 0.2236;               // 2·(hash−0.5)·√dt
    dQxy += noise * n2 * 2.0 * 0.2236;
    f_color = vec4(clamp(Qxx + dQxx, -2.0, 2.0),
                   clamp(Qxy + dQxy, -2.0, 2.0), 0.0, 1.0);
}
''')

_register("nematic_display",
          "Active-nematic display: director-hue schlieren + order brightness + defect glow (node 99)",
          "procedural", '''
float _dir(vec4 q) { return 0.5 * atan(2.0 * q.g, q.r + 1e-6); }
void main() {
    vec2 texel = 1.0 / u_resolution;
    vec4 s = texture(u_texture, v_uv);
    float Qxx = s.r, Qxy = s.g;
    float S = 2.0 * sqrt(Qxx * Qxx + Qxy * Qxy);
    float theta = _dir(s);
    float hue = fract(theta / 3.14159265 * 2.0);     // 2 cycles per π (nematic)
    float val = clamp(abs(S) * 1.5 + 0.2, 0.2, 1.0);
    float sat = clamp(abs(S) * 2.0 + 0.2, 0.2, 1.0);
    float phi = hue * 6.2831853;
    vec3 col = 0.5 + 0.5 * vec3(cos(phi), cos(phi - 2.094), cos(phi + 2.094));
    col = (1.0 - sat) * 0.3 + sat * col;
    col *= val;
    // Defect glow: wrapped director-gradient magnitude (bend) → warm cores.
    float tl = _dir(texture(u_texture, v_uv + vec2(-texel.x, 0.0)));
    float tr = _dir(texture(u_texture, v_uv + vec2( texel.x, 0.0)));
    float tt = _dir(texture(u_texture, v_uv + vec2(0.0,  texel.y)));
    float tb = _dir(texture(u_texture, v_uv + vec2(0.0, -texel.y)));
    float bend = sqrt(sin(tr - tl) * sin(tr - tl) + sin(tt - tb) * sin(tt - tb));
    float glow = clamp((bend - 0.3) * 2.0, 0.0, 1.0);
    col = clamp(col + glow * vec3(1.0, 0.85, 0.3) * 0.5, 0.0, 1.0);
    f_color = vec4(col, 1.0);
}
''')


# ═══════════════════════════════════════════════════════════════════════════
#  P1.6 — Hydraulic Erosion / River Network (node 156) 3-field terrain twin
# ═══════════════════════════════════════════════════════════════════════════
# 3-field grid sim: terrain height h (.r), water w (.g), sediment s (.b). Local
# model (visual-style parity; CPU authoritative for the exact steepest-descent
# routing): rain → water pools down the (h+w) surface gradient → stream-power
# erosion (K_e·w·slope) lifts sediment → deposition (K_d) settles it on flats →
# thermal creep smooths toward the angle of repose → evaporation. Display is the
# CPU's grayscale hillshade + water-channel brightening.
_register("erosion_seed",
          "Hydraulic-erosion seed: fbm fractal terrain, dry (h=.r, w=.g=0, s=.b=0) (node 156)",
          "procedural", '''
void main() {
    float t = fbm(v_uv * 4.0) + 0.5 * fbm(v_uv * 8.0 + 3.1) + 0.25 * fbm(v_uv * 16.0 + 7.7);
    float h = (t - 0.6) * 0.6;               // broad fractal landscape, centered
    f_color = vec4(h, 0.0, 0.0, 1.0);
}
''')

_register("erosion_step",
          "Hydraulic-erosion step: rain + surface-gradient flow + stream-power erosion/deposition (node 156)",
          "procedural", '''
void main() {
    vec2 texel = 1.0 / u_resolution;
    vec4 c = texture(u_texture, v_uv);
    float h = c.r, w = c.g, s = c.b;
    vec4 cl = texture(u_texture, v_uv + vec2(-texel.x, 0.0));
    vec4 cr = texture(u_texture, v_uv + vec2( texel.x, 0.0));
    vec4 cu = texture(u_texture, v_uv + vec2(0.0,  texel.y));
    vec4 cd = texture(u_texture, v_uv + vec2(0.0, -texel.y));
    float rain  = clamp(u_params.x, 0.0, 0.05);
    float K_e   = clamp(u_params.y, 0.001, 0.5);
    float K_d   = clamp(u_params.z, 0.01, 1.0);
    float theta = clamp(u_params.w, 0.02, 0.5);
    // Rainfall.
    w += rain;
    // Water flows to lower total surface (h+w): Laplacian of the surface pools
    // water into valleys and drains it off ridges.
    float surfC = h + w;
    float lapSurf = (cl.r + cl.g) + (cr.r + cr.g) + (cu.r + cu.g) + (cd.r + cd.g) - 4.0 * surfC;
    w = max(w + 0.25 * lapSurf, 0.0);
    // Terrain slope → stream-power erosion; sediment picked up.
    float dx = (cr.r - cl.r) * 0.5, dy = (cu.r - cd.r) * 0.5;
    float slope = sqrt(dx * dx + dy * dy);
    float erode = K_e * w * slope;
    h -= erode; s += erode;
    // Deposition where flow slackens (low slope).
    float dep = K_d * s * (1.0 - clamp(slope * 6.0, 0.0, 1.0));
    h += dep; s = max(s - dep, 0.0);
    // Thermal creep toward the angle of repose.
    float lapH = cl.r + cr.r + cu.r + cd.r - 4.0 * h;
    h += theta * 0.15 * lapH;
    // Evaporation + tiny continuous noise (keeps drainage networks reorganizing).
    w *= 0.97;
    h += (hash21(v_uv * 331.0 + vec2(h, w) * 57.0) - 0.5) * 0.003;
    f_color = vec4(h, w, s, 1.0);
}
''')

_register("erosion_display",
          "Hydraulic-erosion display: grayscale hillshade + water-channel brightening (node 156)",
          "procedural", '''
void main() {
    vec2 texel = 1.0 / u_resolution;
    vec4 c = texture(u_texture, v_uv);
    float h = c.r, w = c.g;
    float hL = texture(u_texture, v_uv + vec2(-texel.x, 0.0)).r;
    float hR = texture(u_texture, v_uv + vec2( texel.x, 0.0)).r;
    float hU = texture(u_texture, v_uv + vec2(0.0,  texel.y)).r;
    float hD = texture(u_texture, v_uv + vec2(0.0, -texel.y)).r;
    float dx = (hR - hL) * 0.5, dy = (hU - hD) * 0.5;
    float slope  = atan(sqrt(dx * dx + dy * dy) * 8.0);
    float aspect = atan(dy, -dx);
    float az = radians(315.0), alt = radians(45.0);
    float shade = clamp(sin(alt) * cos(slope) + cos(alt) * sin(slope) * cos(az - aspect), 0.0, 1.0);
    float hn = clamp(h * 0.7 + 0.5, 0.0, 1.0);
    float g = 0.55 * shade + 0.35 * hn;
    float wn = clamp(w * 6.0, 0.0, 1.0);          // water channels brighten
    g = clamp(max(g, wn * 0.3) + wn * 0.15, 0.0, 1.0);
    f_color = vec4(g, g, g, 1.0);
}
''')

# ── Typed closed-form pattern node (id 301) ──────────────────────────────
# Gyroid / triply-periodic minimal surface — a 2D slice of the 3D field
#   g(x,y,z) = sin(x)cos(y) + sin(y)cos(z) + sin(z)cos(x)
# popularized in shader art (Inigo Quilez, ~2010s). The viewing plane is spun
# in-plane and the slice depth advances with time, so the woven surface appears
# to rotate and flow through 3D space. Closed-form f(uv,t) → exact GPU/CPU
# parity preview (no seeded-layout divergence). Additive typed-uniform layer.
_register("gyroid_typed", "Gyroid / triply-periodic minimal-surface slice (typed, node 301)",
          "procedural", _INFERNO_GPU + '''void main() {
    // 2D slab of the 3D gyroid scalar field. Spinning the plane in-plane and
    // advancing the slice depth with time makes the woven surface appear to
    // rotate and flow through 3D space.
    vec2 p = (v_uv - 0.5) * u_scale;
    float t = u_time;
    p = rot(t * u_spin) * p;                // in-plane spin
    float z = u_slice + t * u_anim_speed;   // advance the slice through volume
    float g = sin(p.x) * cos(p.y) + sin(p.y) * cos(z) + sin(z) * cos(p.x);
    float v = 0.5 + 0.5 * sin(g * 1.3);     // colormap coordinate
    vec3 base = inferno(clamp(v, 0.0, 1.0));
    float wall = smoothstep(u_thickness, u_thickness * 0.25, abs(g));
    vec3 col = mix(base, u_wall, wall);
    f_color = vec4(col, 1.0);
}
''', uniforms={
    "scale":      {"glsl": "float", "min": 4.0, "max": 40.0, "default": 14.0,
                   "description": "feature density (cells across view)"},
    "thickness":  {"glsl": "float", "min": 0.02, "max": 0.6, "default": 0.12,
                   "description": "gyroid wall thickness"},
    "slice":      {"glsl": "float", "min": 0.0, "max": 6.2831853, "default": 0.0,
                   "description": "static slice depth offset (rad)"},
    "anim_speed": {"glsl": "float", "min": 0.0, "max": 6.0, "default": 1.0,
                   "description": "slice advance speed through volume"},
    "spin":       {"glsl": "float", "min": 0.0, "max": 6.0, "default": 0.6,
                   "description": "in-plane rotation speed"},
    "wall":       {"glsl": "color", "default": "#ff5cf0",
                   "description": "wall highlight color"},
})

_register("menger_typed", "Menger carpet / Sierpinski-carpet recursive subdivision (typed, node 324)",
          "procedural", _INFERNO_GPU + '''void main() {
    // Animated Sierpinski (Menger) carpet: recursive 3x3 subdivision that
    // removes the centre ninth at every level. The plane spins and the scale
    // breathes with time so the static fractal reads as alive in the live
    // preview. Surviving cells are coloured by recursion depth through the
    // inferno map; removed cells show the background colour.
    vec2 p = v_uv - 0.5;
    p = rot(u_time * u_spin) * p;
    float sc = u_scale * (0.85 + 0.15 * sin(u_time * u_pulse));
    vec2 uv = fract(p * sc + 0.5);          // wrap into [0,1)
    bool on = true;
    vec2 q = uv;
    float lvl = 0.0;
    for (int i = 0; i < 6; i++) {
        vec2 cell = floor(q * 3.0);
        if (cell.x == 1.0 && cell.y == 1.0) { on = false; break; }
        q = fract(q * 3.0);
        lvl += 1.0;
    }
    vec3 col;
    if (on) {
        float h = fract(lvl * 0.16 + u_time * 0.04);
        col = inferno(clamp(0.25 + 0.55 * h + 0.25 * q.x, 0.0, 1.0));
    } else {
        col = u_bg;
    }
    f_color = vec4(col, 1.0);
}
''', uniforms={
    "scale": {"glsl": "float", "min": 3.0, "max": 24.0, "default": 8.0,
              "description": "feature density (carpets across view)"},
    "spin":  {"glsl": "float", "min": 0.0, "max": 3.0, "default": 0.4,
              "description": "in-plane rotation speed"},
    "pulse": {"glsl": "float", "min": 0.0, "max": 3.0, "default": 0.8,
              "description": "scale breathing speed"},
    "bg":    {"glsl": "color", "default": "#0a0a18",
              "description": "hole background color"},
})

# ═══════════════════════════════════════════════
#  TYPED-UNIFORM UPGRADE — dedicated GPU procedural nodes 173-197
#  These re-register the same shader names with named, typed `uniforms=`
#  and bodies that read `u_<name>` instead of the legacy `u_params` p-slots.
#  The node factory (methods/gpu_shaders.py) routes these ids through
#  `_make_typed`, so each variable becomes a real param + wireable SCALAR
#  port + typed IMAGE/FIELD outputs. Additive — CPU/fp64 export untouched.
# ═══════════════════════════════════════════════

_register("mandelbrot", 'Mandelbrot set zoom region', "procedural", '\nvoid main() {\n    vec2 uv = (gl_FragCoord.xy - 0.5 * u_resolution) / min(u_resolution.x, u_resolution.y);\n    float zoom = exp(u_zoom * 3.0);\n    vec2 c = vec2(-0.5, 0.0) + uv * zoom;\n    vec2 z = vec2(0.0);\n    int n = 0;\n    for (int i = 0; i < 100; i++) {\n        z = vec2(z.x*z.x - z.y*z.y, 2.0*z.x*z.y) + c;\n        if (dot(z, z) > 4.0) break;\n        n++;\n    }\n    float t = float(n) / 100.0;\n    f_color = vec4(0.5 + 0.5 * cos(t * 6.28 + vec3(0.0, 2.0, 4.0) + u_color_shift * 6.28), 1.0);\n}\n',
          uniforms={
  "zoom": {
    "glsl": "float",
    "min": 0.0,
    "max": 1.0,
    "default": 0.5,
    "description": "zoom (0.5 = full view)"
  },
  "color_shift": {
    "glsl": "float",
    "min": 0.0,
    "max": 1.0,
    "default": 0.5,
    "description": "hue rotation"
  }
})

_register("julia", 'Julia set fractal (client-GPU twin of node 66)', "procedural",
          '''
void main() {
    vec2 uv = (gl_FragCoord.xy - 0.5 * u_resolution) / min(u_resolution.x, u_resolution.y);
    vec2 c = vec2(-0.7269, 0.1889);  // node 66's famous default constant (string param unmapped)
    vec2 z = uv * 3.0;              // fixed full view (node 66 has no zoom param)
    int n = 0;
    float last2 = 0.0;
    const float MAXI = 500.0;
    for (int i = 0; i < 500; i++) {
        z = vec2(z.x*z.x - z.y*z.y, 2.0*z.x*z.y) + c;
        last2 = dot(z, z);
        if (last2 > u_escape_radius * u_escape_radius || n >= u_iterations) break;
        n++;
    }
    float t = (n >= u_iterations - 0.5) ? 0.0
            : clamp((n + 1.0 - log(max(log(last2)*0.5, 1.0001))/log(2.0)) / u_iterations, 0.0, 1.0);
    f_color = vec4(0.5 + 0.5 * cos(t * 6.28318 + vec3(0.0, 2.0, 4.0)), 1.0);
}
''',
          uniforms={
  "iterations": {
    "glsl": "float",
    "min": 30.0,
    "max": 500.0,
    "default": 100,
    "description": "max iterations"
  },
  "escape_radius": {
    "glsl": "float",
    "min": 1.5,
    "max": 10.0,
    "default": 2.0,
    "description": "escape radius"
  }
})

_register("plasma", 'Multi-octave colored plasma', "procedural", '\nvoid main() {\n    vec2 uv = v_uv;\n    float t = u_time * 0.1;\n    float v = sin(uv.x * u_scale + t) * cos(uv.y * u_scale * 0.75 + t * 0.7);\n    v += sin(uv.x * u_scale * 2.0 - t * 1.2) * cos(uv.y * u_scale * 1.5 + t * 0.5) * 0.5;\n    v += sin((uv.x + uv.y) * u_scale * 3.0 + t * 0.3) * 0.25;\n    v = v * 0.5 + 0.5;\n    f_color = vec4(0.5 + 0.5 * cos(v * 6.28 + vec3(0.0, 2.0, 4.0) + u_hue_shift * 6.28), 1.0);\n}\n',
          uniforms={
  "scale": {
    "glsl": "float",
    "min": 1.0,
    "max": 32.0,
    "default": 8.0,
    "description": "spatial frequency"
  },
  "hue_shift": {
    "glsl": "float",
    "min": 0.0,
    "max": 1.0,
    "default": 0.0,
    "description": "hue rotation"
  }
})

_register("domain_warp", 'Domain-warped fractal noise', "procedural", '\nvoid main() {\n    vec2 uv = v_uv * 3.0;\n    float t = u_time * 0.05;\n    float w = 2.0 + u_warp * 3.0;\n    vec2 q = vec2(fbm(uv + t), fbm(uv + vec2(5.2, 1.3) + t * 0.7));\n    vec2 r = vec2(fbm(uv + w * q + vec2(1.7, 9.2) + t * 0.3),\n                  fbm(uv + w * q + vec2(8.3, 2.8) + t * 0.4));\n    float v = fbm(uv + w * r);\n    f_color = vec4(0.5 + 0.5 * cos(v * 6.28 + vec3(0.0, 2.0, 4.0) + u_hue_shift * 6.28), 1.0);\n}\n',
          uniforms={
  "warp": {
    "glsl": "float",
    "min": 0.0,
    "max": 1.0,
    "default": 0.5,
    "description": "warp strength"
  },
  "hue_shift": {
    "glsl": "float",
    "min": 0.0,
    "max": 1.0,
    "default": 0.0,
    "description": "hue rotation"
  }
})

_register("voronoi", 'Voronoi/Worley noise cells', "procedural", '\nvoid main() {\n    vec2 uv = v_uv * u_scale;\n    vec2 i = floor(uv); vec2 f = fract(uv);\n    float md = 1.0;\n    for (int y = -1; y <= 1; y++) {\n        for (int x = -1; x <= 1; x++) {\n            vec2 n = vec2(float(x), float(y));\n            vec2 p = hash21(i + n) * vec2(1.0);\n            float d = length(n + p - f);\n            md = min(md, d);\n        }\n    }\n    f_color = vec4(md, md * 0.5, 1.0 - md, 1.0);\n}\n',
          uniforms={
  "scale": {
    "glsl": "float",
    "min": 1.0,
    "max": 15.0,
    "default": 7.5,
    "description": "cell density"
  }
})

_register("voronoise", 'Smooth voronoi layers', "procedural", '\nvoid main() {\n    vec2 uv = v_uv * u_scale;\n    float t = u_time * 0.02;\n    vec2 q = vec2(fbm(uv + t), fbm(uv + vec2(3.7, 1.2) + t));\n    vec2 r = vec2(fbm(uv + 4.0 * q + vec2(1.7, 9.2)),\n                  fbm(uv + 4.0 * q + vec2(8.3, 2.8)));\n    float v = fbm(uv + 4.0 * r);\n    f_color = vec4(0.5 + 0.5 * cos(v * 4.0 + vec3(0.0, 2.0, 4.0) + u_hue_shift * 6.28), 1.0);\n}\n',
          uniforms={
  "scale": {
    "glsl": "float",
    "min": 1.0,
    "max": 8.0,
    "default": 4.0,
    "description": "layer frequency"
  },
  "hue_shift": {
    "glsl": "float",
    "min": 0.0,
    "max": 1.0,
    "default": 0.0,
    "description": "hue rotation"
  }
})

_register("ripples", 'Concentric ripple pattern', "procedural", '\nvoid main() {\n    vec2 uv = v_uv - 0.5;\n    float d = length(uv);\n    float ph = d * u_frequency - u_time * u_speed;\n    float r = sin(ph) * 0.5 + 0.5;\n    float g = sin(ph + 2.0) * 0.5 + 0.5;\n    float b = sin(ph + 4.0) * 0.5 + 0.5;\n    f_color = vec4(r, g, b, 1.0) * (1.0 - d);\n}\n',
          uniforms={
  "frequency": {
    "glsl": "float",
    "min": 5.0,
    "max": 60.0,
    "default": 30.0,
    "description": "ripple frequency"
  },
  "speed": {
    "glsl": "float",
    "min": 0.0,
    "max": 5.0,
    "default": 2.0,
    "description": "ripple speed"
  }
})

_register("cells", 'Cellular growth simulation', "procedural", '\nvoid main() {\n    vec2 uv = v_uv * u_scale;\n    vec2 i = floor(uv); vec2 f = fract(uv);\n    float md = 8.0;\n    vec2 mp = vec2(0.0);\n    for (int y = -1; y <= 1; y++) {\n        for (int x = -1; x <= 1; x++) {\n            vec2 n = vec2(float(x), float(y));\n            vec2 p = hash21(i + n) * vec2(1.0);\n            float d = length(n + p - f);\n            if (d < md) { md = d; mp = n + p; }\n        }\n    }\n    float c = hash21(i + mp);\n    vec3 col = 0.5 + 0.5 * cos(c * 6.28 + vec3(0, 2, 4));\n    col *= 1.0 - md * 1.2;\n    col += vec3(0.05) / (md * md + 0.01);\n    f_color = vec4(col, 1.0);\n}\n',
          uniforms={
  "scale": {
    "glsl": "float",
    "min": 1.0,
    "max": 16.0,
    "default": 8.0,
    "description": "cell scale"
  }
})

_register("bubble_chamber", 'Simulated bubble chamber trails', "procedural", '\nvoid main() {\n    vec2 uv = v_uv * 2.0 - 1.0;\n    float t = u_time * u_speed;\n    float v = 0.0;\n    for (int i = 0; i < 40; i++) {\n        if (i >= u_count) break;\n        float fi = float(i);\n        vec2 p = vec2(sin(fi * 1.7 + t * 0.5), cos(fi * 2.3 + t * 0.7)) * 0.8;\n        float d = length(uv - p) - 0.03;\n        v += 0.005 / (d * d + 0.001);\n    }\n    f_color = vec4(v * 0.5, v * 0.8, v, 1.0);\n}\n',
          uniforms={
  "count": {
    "glsl": "int",
    "min": 1,
    "max": 40,
    "default": 20,
    "description": "number of trails"
  },
  "speed": {
    "glsl": "float",
    "min": 0.0,
    "max": 3.0,
    "default": 0.3,
    "description": "drift speed"
  }
})

_register("stars", 'Starfield with parallax', "procedural", '\nvoid main() {\n    vec2 uv = v_uv;\n    float t = u_time * u_speed;\n    vec3 col = vec3(0.0);\n    for (int i = 0; i < 120; i++) {\n        if (i >= u_count) break;\n        float fi = float(i);\n        vec2 p = fract(vec2(sin(fi * 127.1 + t), cos(fi * 311.7 + t * 0.7)));\n        float d = length(uv - p);\n        float brightness = 0.003 / (d * d);\n        vec3 star_col = 0.5 + 0.5 * cos(fi + vec3(0, 2, 4));\n        col += brightness * star_col;\n    }\n    f_color = vec4(col, 1.0);\n}\n',
          uniforms={
  "count": {
    "glsl": "int",
    "min": 10,
    "max": 120,
    "default": 50,
    "description": "star count"
  },
  "speed": {
    "glsl": "float",
    "min": 0.0,
    "max": 3.0,
    "default": 0.05,
    "description": "parallax speed"
  }
})

_register("lightning_fractal", 'Fractal lightning branching', "procedural", '\nvoid main() {\n    vec2 uv = v_uv * 2.0 - 1.0;\n    float t = u_time * u_speed;\n    vec2 p = vec2(0.0);\n    float v = 0.0;\n    for (int i = 0; i < 128; i++) {\n        if (i >= u_segments) break;\n        float fi = float(i);\n        p += vec2(sin(fi * 0.3 + t), cos(fi * 0.7 + t * 0.5)) * 0.02;\n        float d = length(uv - p);\n        v += 0.02 / (d + 0.01);\n    }\n    f_color = vec4(v * 0.3, v * 0.5, v, 1.0);\n}\n',
          uniforms={
  "segments": {
    "glsl": "int",
    "min": 8,
    "max": 128,
    "default": 64,
    "description": "branch segments"
  },
  "speed": {
    "glsl": "float",
    "min": 0.0,
    "max": 3.0,
    "default": 0.2,
    "description": "flicker speed"
  }
})

_register("spiral", 'Logarithmic spiral galaxy', "procedural", '\nvoid main() {\n    vec2 uv = v_uv - 0.5;\n    float a = atan(uv.y, uv.x);\n    float r = length(uv);\n    float spiral = sin(a * u_arms - r * u_tightness + u_time * u_speed) * 0.5 + 0.5;\n    float fade = exp(-r * 3.0);\n    float col = spiral * fade;\n    f_color = vec4(col * 1.2, col * 0.8, col * fade + 0.1, 1.0);\n}\n',
          uniforms={
  "arms": {
    "glsl": "float",
    "min": 1.0,
    "max": 12.0,
    "default": 4.0,
    "description": "spiral arms"
  },
  "tightness": {
    "glsl": "float",
    "min": 5.0,
    "max": 30.0,
    "default": 15.0,
    "description": "winding tightness"
  },
  "speed": {
    "glsl": "float",
    "min": 0.0,
    "max": 3.0,
    "default": 0.5,
    "description": "rotation speed"
  }
})

_register("dendritic", 'Dendritic / tree-like branching', "procedural", '\nvoid main() {\n    vec2 uv = v_uv * 2.0 - 1.0;\n    float t = u_time * u_speed;\n    float d = length(uv);\n    float a = atan(uv.y, uv.x) * u_branches;\n    float branch = sin(a * 8.0 + log(d + 0.001) * 10.0 + t) * 0.5 + 0.5;\n    float v = branch * exp(-d * 2.0);\n    f_color = vec4(v * 0.3, v * 0.6, v * 0.2, 1.0);\n}\n',
          uniforms={
  "branches": {
    "glsl": "float",
    "min": 2.0,
    "max": 16.0,
    "default": 8.0,
    "description": "branch count"
  },
  "speed": {
    "glsl": "float",
    "min": 0.0,
    "max": 3.0,
    "default": 0.1,
    "description": "growth speed"
  }
})

_register("barnsley", 'Barnsley fern approximation', "procedural", '\nvoid main() {\n    vec2 uv = v_uv * 3.0 - 1.5;\n    float t = u_time * u_speed;\n    float v = 0.0;\n    for (int i = 0; i < 200; i++) {\n        if (i >= u_iterations) break;\n        float fi = float(i);\n        vec2 p = vec2(sin(fi * 0.5 + t), cos(fi * 0.3 + t * 0.7));\n        float dx = uv.x - p.x * 0.5;\n        float dy = uv.y - p.y * 0.8 - 0.5;\n        v += 0.001 / (dx*dx + dy*dy + 0.001);\n    }\n    f_color = vec4(v * 0.2, v * 0.8, v * 0.2, 1.0);\n}\n',
          uniforms={
  "iterations": {
    "glsl": "int",
    "min": 20,
    "max": 200,
    "default": 100,
    "description": "sample iterations"
  },
  "speed": {
    "glsl": "float",
    "min": 0.0,
    "max": 3.0,
    "default": 0.1,
    "description": "sway speed"
  }
})

_register("spectral", 'Spectral / rainbow interference', "procedural", '\nvoid main() {\n    vec2 uv = v_uv * 2.0 - 1.0;\n    float t = u_time * u_speed;\n    float a = atan(uv.y, uv.x);\n    float r = length(uv);\n    float v = sin(r * u_rings - t) + cos(a * u_arms + t * 0.5);\n    v = v * 0.25 + 0.5;\n    f_color = vec4(0.5 + 0.5 * cos(v * 6.28 + vec3(0, 2, 4)), 1.0);\n}\n',
          uniforms={
  "rings": {
    "glsl": "float",
    "min": 5.0,
    "max": 40.0,
    "default": 20.0,
    "description": "radial rings"
  },
  "arms": {
    "glsl": "float",
    "min": 2.0,
    "max": 10.0,
    "default": 5.0,
    "description": "angular arms"
  },
  "speed": {
    "glsl": "float",
    "min": 0.0,
    "max": 3.0,
    "default": 0.1,
    "description": "animation speed"
  }
})

_register("truchet", 'Truchet tile pattern', "procedural", '\nvoid main() {\n    vec2 uv = v_uv * u_scale;\n    vec2 i = floor(uv); vec2 f = fract(uv) - 0.5;\n    float flip = hash21(i) > 0.5 ? 1.0 : -1.0;\n    float d = length(f * flip);\n    float v = smoothstep(0.4, 0.5, d);\n    float c = hash21(i + vec2(1.0));\n    vec3 col = mix(vec3(0.9, 0.9, 0.95), 0.5 + 0.5 * cos(c * 6.28 + vec3(0, 2, 4)), v);\n    f_color = vec4(col, 1.0);\n}\n',
          uniforms={
  "scale": {
    "glsl": "float",
    "min": 1.0,
    "max": 12.0,
    "default": 6.0,
    "description": "tile scale"
  }
})

_register("kaleidoscope_fractal", 'Kaleidoscope IFS fractal', "procedural", '\nvoid main() {\n    vec2 uv = v_uv * 2.0 - 1.0;\n    float t = u_time * u_speed;\n    for (int i = 0; i < 20; i++) {\n        if (i >= u_iterations) break;\n        uv = abs(uv);\n        float a = sin(t + float(i) * 0.5);\n        uv = rot(a) * uv;\n        uv = uv * 1.5 - vec2(0.5);\n    }\n    float v = length(uv);\n    f_color = vec4(0.5 + 0.5 * cos(v * 10.0 + vec3(0, 2, 4)), 1.0);\n}\n',
          uniforms={
  "iterations": {
    "glsl": "int",
    "min": 3,
    "max": 20,
    "default": 10,
    "description": "IFS iterations"
  },
  "speed": {
    "glsl": "float",
    "min": 0.0,
    "max": 3.0,
    "default": 0.1,
    "description": "rotation speed"
  }
})

_register("waves_3d", '3D wave interference', "procedural", '\nvoid main() {\n    vec2 uv = v_uv * u_scale - 2.0;\n    float t = u_time * u_speed;\n    float v = 0.0;\n    for (int i = 0; i < 20; i++) {\n        if (float(i) >= u_waves) break;\n        float fi = float(i);\n        vec2 p = vec2(sin(fi * 1.3 + t), cos(fi * 1.7 + t * 0.8));\n        v += sin(dot(uv, p) * 3.0 + t) * 0.1;\n    }\n    vec3 col = 0.5 + 0.5 * cos(v * 4.0 + vec3(0, 2, 4));\n    f_color = vec4(col, 1.0);\n}\n',
          uniforms={
  "waves": {
    "glsl": "float",
    "min": 2.0,
    "max": 20.0,
    "default": 10.0,
    "description": "wave sources"
  },
  "scale": {
    "glsl": "float",
    "min": 1.0,
    "max": 8.0,
    "default": 4.0,
    "description": "field scale"
  },
  "speed": {
    "glsl": "float",
    "min": 0.0,
    "max": 3.0,
    "default": 0.5,
    "description": "animation speed"
  }
})

_register("pixel_sort_gpu", 'Edge-directed pixel sorting on GPU', "procedural", '\nvoid main() {\n    vec2 uv = v_uv;\n    float v = fbm(uv * 4.0 + u_time * u_speed);\n    vec2 stepv = vec2(1.0 / u_resolution.x, 1.0 / u_resolution.y);\n    float dx = fbm((uv + vec2(stepv.x, 0)) * 4.0) - v;\n    float dy = fbm((uv + vec2(0, stepv.y)) * 4.0) - v;\n    float edge = abs(dx) + abs(dy);\n    float bands = floor(uv.x * u_bands + v * 10.0) / u_bands + v * 0.1;\n    vec3 col = 0.5 + 0.5 * cos(bands * 6.28 + vec3(0, 2, 4));\n    col = mix(col, vec3(0.1), edge * 5.0);\n    f_color = vec4(col, 1.0);\n}\n',
          uniforms={
  "bands": {
    "glsl": "float",
    "min": 4.0,
    "max": 40.0,
    "default": 20.0,
    "description": "sort bands"
  },
  "speed": {
    "glsl": "float",
    "min": 0.0,
    "max": 3.0,
    "default": 0.02,
    "description": "flow speed"
  }
})

_register("ocean", 'Procedural ocean waves', "procedural", '\nvoid main() {\n    vec2 uv = v_uv * 3.0;\n    float t = u_time * u_speed;\n    float v = sin(uv.x * 5.0 + t) * cos(uv.y * 3.0 + t * 0.7);\n    v += sin(uv.x * 8.0 - t * 1.3) * sin(uv.y * 6.0 + t) * 0.5 * u_choppiness;\n    v += sin((uv.x + uv.y) * 12.0 + t * 0.5) * 0.25;\n    v = v * 0.5 + 0.5;\n    vec3 col = mix(vec3(0.0, 0.2, 0.5), vec3(0.1, 0.6, 0.8), v);\n    col += vec3(0.3, 0.4, 0.5) * pow(v, 4.0);\n    f_color = vec4(col, 1.0);\n}\n',
          uniforms={
  "choppiness": {
    "glsl": "float",
    "min": 0.0,
    "max": 2.0,
    "default": 1.0,
    "description": "wave amplitude"
  },
  "speed": {
    "glsl": "float",
    "min": 0.0,
    "max": 3.0,
    "default": 0.3,
    "description": "wave speed"
  }
})

_register("nebula_gpu", 'Space nebula gas clouds', "procedural", '\nvoid main() {\n    vec2 uv = v_uv * u_scale;\n    float t = u_time * u_speed;\n    vec2 q = vec2(fbm(uv + t), fbm(uv + vec2(5.2, 1.3) + t * 0.7));\n    vec2 r = vec2(fbm(uv + 3.0 * q + vec2(1.7, 9.2) + t * 0.3),\n                  fbm(uv + 3.0 * q + vec2(8.3, 2.8) + t * 0.4));\n    float v = fbm(uv + 3.0 * r);\n    float mask = 1.0 - abs(v_uv.y - 0.5) * 2.0;\n    vec3 col = 0.3 + 0.7 * (0.5 + 0.5 * cos(v * 4.0 + vec3(0, 1, 2)));\n    col *= mask;\n    f_color = vec4(col, 1.0);\n}\n',
          uniforms={
  "scale": {
    "glsl": "float",
    "min": 1.0,
    "max": 6.0,
    "default": 2.0,
    "description": "cloud scale"
  },
  "speed": {
    "glsl": "float",
    "min": 0.0,
    "max": 3.0,
    "default": 0.03,
    "description": "drift speed"
  }
})

_register("terrain", 'Procedural terrain heightmap', "procedural", '\nvoid main() {\n    vec2 uv = v_uv * u_scale;\n    float t = u_time * u_speed;\n    float h = fbm(uv + t);\n    float h2 = fbm(uv * 2.0 + t * 1.5) * 0.5;\n    float h3 = fbm(uv * 4.0 + t * 2.0) * 0.25;\n    h = h * 0.6 + h2 * 0.3 + h3 * 0.1;\n    vec3 col;\n    if (h < u_sea_level) col = vec3(0.1, 0.3, 0.6);\n    else if (h < u_sea_level + 0.15) col = vec3(0.2, 0.5, 0.2);\n    else if (h < 0.6) col = vec3(0.3, 0.3, 0.1);\n    else if (h < 0.75) col = vec3(0.4, 0.25, 0.1);\n    else col = vec3(0.8, 0.8, 0.9);\n    float shade = 0.5 + 0.5 * cos(h * 20.0);\n    f_color = vec4(col * shade, 1.0);\n}\n',
          uniforms={
  "scale": {
    "glsl": "float",
    "min": 1.0,
    "max": 8.0,
    "default": 3.0,
    "description": "terrain scale"
  },
  "sea_level": {
    "glsl": "float",
    "min": 0.0,
    "max": 1.0,
    "default": 0.3,
    "description": "water line"
  },
  "speed": {
    "glsl": "float",
    "min": 0.0,
    "max": 2.0,
    "default": 0.02,
    "description": "erosion speed"
  }
})

_register("wood_grain_gpu", 'Concentric wood grain rings', "procedural", '\nvoid main() {\n    vec2 uv = v_uv - 0.5;\n    float d = length(uv) * u_rings;\n    float grain = sin(d * u_turbulence + fbm(uv * u_turbulence) * 0.5) * 0.5 + 0.5;\n    vec3 col = mix(vec3(0.3, 0.15, 0.05), vec3(0.6, 0.3, 0.1), grain);\n    f_color = vec4(col, 1.0);\n}\n',
          uniforms={
  "rings": {
    "glsl": "float",
    "min": 2.0,
    "max": 20.0,
    "default": 10.0,
    "description": "ring density"
  },
  "turbulence": {
    "glsl": "float",
    "min": 1.0,
    "max": 10.0,
    "default": 8.0,
    "description": "grain wobble"
  }
})

_register("fire_gpu", 'Animated fire/flame', "procedural", '\nvoid main() {\n    vec2 uv = v_uv;\n    float t = u_time * u_speed;\n    float v = fbm(vec2(uv.x * 3.0, (1.0 - uv.y) * 5.0 + t));\n    v = v * (1.0 - uv.y) * u_intensity;\n    vec3 col = mix(vec3(1.0, 0.9, 0.4), vec3(0.8, 0.2, 0.0), v);\n    col = mix(col, vec3(0.1, 0.0, 0.0), 1.0 - v);\n    f_color = vec4(col, 1.0);\n}\n',
          uniforms={
  "intensity": {
    "glsl": "float",
    "min": 0.0,
    "max": 2.0,
    "default": 1.0,
    "description": "flame intensity"
  },
  "speed": {
    "glsl": "float",
    "min": 0.0,
    "max": 3.0,
    "default": 0.5,
    "description": "rise speed"
  }
})

_register("smoke_gpu", 'Rising smoke / steam', "procedural", '\nvoid main() {\n    vec2 uv = v_uv;\n    float t = u_time * u_speed;\n    float v = fbm(uv * 3.0 + vec2(0.0, t));\n    v = v * (1.0 - uv.y) * 0.8 * u_density;\n    vec3 col = mix(vec3(0.8, 0.8, 0.85), vec3(0.2, 0.2, 0.25), v);\n    f_color = vec4(col, 1.0);\n}\n',
          uniforms={
  "density": {
    "glsl": "float",
    "min": 0.0,
    "max": 2.0,
    "default": 1.0,
    "description": "smoke density"
  },
  "speed": {
    "glsl": "float",
    "min": 0.0,
    "max": 3.0,
    "default": 0.1,
    "description": "rise speed"
  }
})


# ═══════════════════════════════════════════════
#  CLOSED-FORM TYPED-UNIFORM NODES — pt.13 (nodes 302-307)
#  Pure f(uv,t) field-eval twins: each variable is a named, typed uniform
#  wired through _make_typed (real param + wireable SCALAR port). No ping-pong
#  state, exact server/browser parity. Additive — CPU/fp64 export untouched.
# ═══════════════════════════════════════════════

# 302 — Schotter (Georg Nees, 1968): a rigid grid of squares whose jitter and
# rotation grow with distance from the centre. The canonical generative-art
# "ordered disorder" piece.
_register("schotter_typed", "Schotter — Georg Nees generative grid of jittered squares (typed, node 302)",
          "procedural", '''void main() {
    float N = max(u_cells, 2.0);
    vec2 g = v_uv * N;
    vec2 id = floor(g);
    vec2 f = fract(g) - 0.5;
    vec2 ctr = (id + 0.5) / N - 0.5;
    float d = length(ctr);
    float amt = u_jitter * smoothstep(0.0, 0.7, d);
    float r1 = hash21(id + 1.3);
    float r2 = hash21(id + 7.7);
    float r3 = hash21(id + 3.1);
    float ang = (r1 - 0.5) * amt * 1.5 + u_time * u_speed * (r2 - 0.5) * 0.3;
    vec2 disp = (vec2(r2 - 0.5, r3 - 0.5)) * amt * 0.6;
    vec2 q = f - disp;
    q = rot(ang) * q;
    float s = 0.5 * u_square;
    vec2 a = abs(q);
    float inside = step(max(a.x, a.y), s);
    vec3 col = mix(u_bg, u_fg, inside);
    f_color = vec4(col, 1.0);
}
''', uniforms={
    "cells":  {"glsl": "float", "min": 2.0, "max": 24.0, "default": 11.0, "description": "grid cells per axis"},
    "jitter": {"glsl": "float", "min": 0.0, "max": 1.5, "default": 0.85, "description": "displacement (grows outward)"},
    "square": {"glsl": "float", "min": 0.3, "max": 0.95, "default": 0.72, "description": "square fill fraction"},
    "speed":  {"glsl": "float", "min": 0.0, "max": 3.0, "default": 0.5, "description": "animation speed"},
    "fg":     {"glsl": "color", "default": "#f4c020", "description": "square color"},
    "bg":     {"glsl": "color", "default": "#0e0e16", "description": "background color"},
})

# 303 — Thue-Morse recursive binary fractal: cell parity = popcount(x) XOR
# popcount(y) over a 2^depth grid. Static structure; animation sweeps the
# two-colour palette so the node still responds to time.
_register("thue_morse_typed", "Thue-Morse recursive binary fractal (typed, node 303)",
          "procedural", '''void main() {
    float depth = max(u_depth, 1.0);
    float scale = exp2(depth);
    vec2 cell = floor(v_uv * scale);
    // Popcount parity via floating extraction (no integer bit ops — portable).
    float ix = cell.x + 1.0;
    float iy = cell.y + 1.0;
    float cnt = 0.0;
    for (int b = 0; b < 9; b++) {
        float fb = float(b);
        cnt += mod(floor(ix / exp2(fb)), 2.0);
        cnt += mod(floor(iy / exp2(fb)), 2.0);
    }
    float v = mod(cnt, 2.0);
    // Animation: a global brightness pulse that shifts the two colours over time
    // (independent of the binary value, so it is never a no-op at t=0 vs t=pi).
    float pulse = 0.5 + 0.5 * sin(u_time * u_speed * 0.6);
    vec3 colA = u_color_a * (0.6 + 0.4 * pulse);
    vec3 colB = u_color_b * (1.4 - 0.4 * pulse);
    vec3 col = mix(colA, colB, step(0.5, v));
    f_color = vec4(col, 1.0);
}
''', uniforms={
    "depth":   {"glsl": "float", "min": 1.0, "max": 8.0, "default": 5.0, "description": "recursion depth (2^depth cells)"},
    "speed":   {"glsl": "float", "min": 0.0, "max": 3.0, "default": 0.6, "description": "palette animation speed"},
    "color_a": {"glsl": "color", "default": "#101830", "description": "even parity color"},
    "color_b": {"glsl": "color", "default": "#f0603c", "description": "odd parity color"},
})

# 304 — Crystal diffraction: sum of N cosinusoidal gratings evenly fanned around
# the circle, coloured with the inferno map. Rotating the fan gives the
# classic X-ray-diffraction look.
_register("crystal_typed", "Crystal diffraction — sum of N sinusoidal gratings (typed, node 304)",
          "procedural", _INFERNO_GPU + '''void main() {
    vec2 p = (v_uv - 0.5) * u_scale;
    float t = u_time * u_speed;
    int N = int(max(u_arms, 1.0));
    float acc = 0.0;
    for (int k = 0; k < 64; k++) {
        if (k >= N) break;
        float a = (float(k) / float(N)) * 6.2831853 + u_rotation + t * 0.15;
        vec2 dir = vec2(cos(a), sin(a));
        acc += cos(dot(p, dir) * u_freq);
    }
    acc /= float(N);
    float v = 0.5 + 0.5 * acc;
    vec3 col = inferno(clamp(v, 0.0, 1.0));
    f_color = vec4(col, 1.0);
}
''', uniforms={
    "arms":     {"glsl": "float", "min": 2.0, "max": 24.0, "default": 6.0, "description": "grating directions"},
    "freq":     {"glsl": "float", "min": 1.0, "max": 40.0, "default": 12.0, "description": "grating frequency"},
    "scale":    {"glsl": "float", "min": 2.0, "max": 12.0, "default": 6.0, "description": "spatial scale"},
    "rotation": {"glsl": "float", "min": 0.0, "max": 6.2831853, "default": 0.0, "description": "base rotation"},
    "speed":    {"glsl": "float", "min": 0.0, "max": 3.0, "default": 0.5, "description": "animation speed"},
})

# 305 — Apollonian gasket: iterate circle inversions in three mutually tangent
# circles inscribed in the unit disk. The limit set is the gasket.
_register("apollonian_typed", "Apollonian gasket via circle inversions (typed, node 305)",
          "procedural", _INFERNO_GPU + '''void main() {
    vec2 p = (v_uv - 0.5) * u_scale;
    float t = u_time * u_speed;
    p = rot(t * 0.4) * p;
    // three mutually-tangent unit-ish circles in the disk
    vec2 c0 = vec2(-0.5, 0.0);
    vec2 c1 = vec2( 0.5, 0.0);
    vec2 c2 = vec2( 0.0, 0.8660254);
    float r = 0.5;
    for (int i = 0; i < 8; i++) {
        vec2 c = (i % 3 == 0) ? c0 : (i % 3 == 1) ? c1 : c2;
        vec2 d = p - c;
        float d2 = max(dot(d, d), 1e-6);
        p = c + (r * r / d2) * d;
    }
    // Band the final inversion radius so the limit-set structure fills the full
    // inferno range (the previous clamp(length*0.5) saturated most of the disk
    // to white, hiding the u_scale/u_speed response — a silent no-op live
    // preview caught by test_typed_uniforms_drive_output). The +t*2.0 phase
    // keeps the rotation/animation visible (radius alone is rotation-invariant).
    float v = fract(length(p) * 0.5);
    v = 0.5 + 0.5 * sin(v * 6.2831853 * u_scale + t * 2.0);
    vec3 col = inferno(v);
    f_color = vec4(col, 1.0);
}
''', uniforms={
    "scale": {"glsl": "float", "min": 2.0, "max": 12.0, "default": 5.0, "description": "view scale"},
    "speed": {"glsl": "float", "min": 0.0, "max": 3.0, "default": 0.5, "description": "rotation speed"},
})

# 306 — Confocal parabola family (op-art): sum of parabolas f(s^2/c) over N
# directions, coloured with inferno. Distinct from the domain-warp grid.
_register("parabola_typed", "Confocal parabola family — op-art interference (typed, node 306)",
          "procedural", _INFERNO_GPU + '''void main() {
    vec2 p = (v_uv - 0.5) * u_scale;
    float t = u_time * u_speed;
    int N = int(max(u_arms, 1.0));
    float acc = 0.0;
    for (int k = 0; k < 48; k++) {
        if (k >= N) break;
        float a = (float(k) / float(N)) * 3.14159265;
        vec2 dir = vec2(cos(a), sin(a));
        float s = dot(p, dir);
        float c = dot(p, vec2(-dir.y, dir.x));
        acc += cos((s * s / max(abs(c), 0.05)) * u_freq + t);
    }
    acc /= float(N);
    vec3 col = inferno(0.5 + 0.5 * acc);
    f_color = vec4(col, 1.0);
}
''', uniforms={
    "arms":  {"glsl": "float", "min": 2.0, "max": 24.0, "default": 8.0, "description": "parabola directions"},
    "freq":  {"glsl": "float", "min": 1.0, "max": 30.0, "default": 8.0, "description": "curvature frequency"},
    "scale": {"glsl": "float", "min": 2.0, "max": 12.0, "default": 6.0, "description": "spatial scale"},
    "speed": {"glsl": "float", "min": 0.0, "max": 3.0, "default": 0.5, "description": "animation speed"},
})

# 307 — Poincaré-disk hyperbolic {p,q} tiling via repeated inversion in the
# edge-circles of the central regular p-gon. Edges glow; interior fills with
# inferno by inversion depth.
_register("hyperbolic_typed", "Poincaré-disk hyperbolic {p,q} tiling (typed, node 307)",
          "procedural", _INFERNO_GPU + '''void main() {
    int p = int(clamp(u_sides, 3.0, 12.0));
    int q = int(clamp(u_verts, 3.0, 12.0));
    float cp = cos(3.14159265 / float(p));
    float cq = cos(3.14159265 / float(q));
    float R0 = cq / cp;
    float dm = (R0 * R0 + 1.0) / (2.0 * R0 * cp);
    float rho = sqrt(max(dm * dm - 1.0, 1e-4));
    vec2 uv = (v_uv - 0.5) * 2.0;
    float t = u_time * u_speed;
    uv = rot(t * 0.25) * uv;
    vec3 col = u_bg;
    if (length(uv) < 1.0) {
        vec2 pnt = uv;
        float it = 0.0;
        for (int i = 0; i < 6; i++) {
            float best = 1e9;
            vec2 bc = vec2(0.0);
            for (int j = 0; j < 12; j++) {
                if (j >= p) break;
                float th = (float(j) + 0.5) * 6.2831853 / float(p);
                vec2 c = vec2(cos(th), sin(th)) * dm;
                float d2 = dot(pnt - c, pnt - c);
                if (d2 < best) { best = d2; bc = c; }
            }
            vec2 d = pnt - bc;
            float d2 = max(dot(d, d), 1e-6);
            pnt = bc + (rho * rho / d2) * d;
            it += 1.0;
        }
        col = mix(u_bg, inferno(clamp(0.2 + 0.6 * fract(it * 0.25), 0.0, 1.0)), 0.85);
        col = mix(col, u_edge, smoothstep(0.05, 0.0, abs(length(pnt) - rho)));
    }
    f_color = vec4(col, 1.0);
}
''', uniforms={
    "sides": {"glsl": "float", "min": 3.0, "max": 12.0, "default": 5.0, "description": "p — polygon sides"},
    "verts": {"glsl": "float", "min": 3.0, "max": 12.0, "default": 4.0, "description": "q — polygons at a vertex"},
    "speed": {"glsl": "float", "min": 0.0, "max": 3.0, "default": 0.4, "description": "rotation speed"},
    "bg":    {"glsl": "color", "default": "#05060f", "description": "background"},
    "edge":  {"glsl": "color", "default": "#39e0ff", "description": "edge glow color"},
})


# ═══════════════════════════════════════════════════════════════
#  Volumetric Clouds (node 308) — ray-marched fbm density field
# ═══════════════════════════════════════════════════════════════
# Screen-space volumetric cloud render: march a fixed ray range through a
# world-Y slab, sample an fbm density field (advected by a wind vector driven
# by u_time), and accumulate single-scatter sunlight (a short light march
# toward the sun for self-shadowing) with Beer-Lambert absorption. Background
# is a sky gradient + sun glow. Closed-form f(uv,t) — no depth buffer needed,
# so it is a pure procedural twin (no input image).
_register("clouds_typed",
          "Raymarched volumetric clouds — screen-space fbm density march with "
          "single-scatter sun lighting over a sky gradient (typed, node 308)",
          "procedural", '''void main() {
    vec2 uv = v_uv * 2.0 - 1.0;
    uv.x *= u_resolution.x / u_resolution.y;

    vec3 ro = vec3(0.0, 0.5, 3.0);
    vec3 rd = normalize(vec3(uv * 1.2, -1.6));

    float t = u_time * u_speed;
    vec3 wind = vec3(t * u_wind, 0.0, t * u_wind * 0.4);

    float az = u_sun_azim * 6.2831853;
    float el = mix(0.05, 1.2, u_sun_elev);
    vec3 sun = normalize(vec3(cos(az) * cos(el), sin(el), sin(az) * cos(el)));

    float h = clamp(rd.y * 0.5 + 0.5, 0.0, 1.0);
    vec3 sky = mix(u_sky_bottom, u_sky_top, h);
    sky += vec3(1.0, 0.92, 0.72) * pow(max(dot(rd, sun), 0.0), 12.0) * 0.5;

    float y0 = -1.0, y1 = 2.0;
    float transmittance = 1.0;
    vec3 scattered = vec3(0.0);
    const int STEPS = 40;
    float tB = 6.0;
    float stepSize = tB / float(STEPS);

    for (int i = 0; i < STEPS; i++) {
        float tt = (float(i) + 0.5) * stepSize;
        vec3 pos = ro + rd * tt + wind;
        float vert = smoothstep(y0, y0 + 0.6, pos.y) * (1.0 - smoothstep(y1 - 0.6, y1, pos.y));
        float d = fbm(pos.xy * 0.6 + pos.z * 0.4 + 10.0);
        d = smoothstep(1.0 - u_coverage, 1.0, d) * vert;
        d *= u_density;
        if (d > 0.001) {
            float ls = 0.0;
            vec3 lp = pos;
            for (int j = 0; j < 4; j++) {
                lp += sun * 0.25;
                float ld = fbm((lp + wind).xy * 0.6 + (lp.z) * 0.4 + 10.0);
                ls += smoothstep(1.0 - u_coverage, 1.0, ld);
            }
            float lightT = exp(-ls * 0.5 * u_density);
            float absorb = exp(-d * stepSize * 3.0);
            vec3 sunCol = vec3(1.0, 0.95, 0.85);
            scattered += transmittance * (1.0 - absorb) * lightT * sunCol;
            transmittance *= absorb;
        }
        if (transmittance < 0.02) break;
    }

    vec3 col = sky * transmittance + scattered;
    col = pow(clamp(col, 0.0, 1.0), vec3(0.9));
    f_color = vec4(col, 1.0);
}
''', uniforms={
    "coverage":  {"glsl": "float", "min": 0.05, "max": 0.98, "default": 0.5,
                  "description": "cloud coverage threshold"},
    "density":   {"glsl": "float", "min": 0.0, "max": 2.5, "default": 1.0,
                  "description": "cloud density multiplier"},
    "wind":      {"glsl": "float", "min": 0.0, "max": 2.0, "default": 0.6,
                  "description": "wind speed (advection)"},
    "sun_elev":  {"glsl": "float", "min": 0.0, "max": 1.0, "default": 0.6,
                  "description": "sun elevation"},
    "sun_azim":  {"glsl": "float", "min": 0.0, "max": 1.0, "default": 0.25,
                  "description": "sun azimuth"},
    "speed":     {"glsl": "float", "min": 0.0, "max": 3.0, "default": 0.6,
                  "description": "animation speed"},
    "sky_top":   {"glsl": "color", "default": "#1a4a8a", "description": "zenith sky color"},
    "sky_bottom":{"glsl": "color", "default": "#cfe3f2", "description": "horizon sky color"},
})

# ── GPU-First categorical coverage: recent CPU nodes tagged gpu-twin-candidate ──
# 431 Domain Coloring / 433 Low-Discrepancy Field. Closed-form f(uv,t) twins so
# the recent CPU nodes get a client-GPU live-preview mirror. CPU numpy node stays
# authoritative for export (two-tier precision). NAMED typed uniforms equal the
# CPU node's real numeric params (contract #5).

_register("domain_coloring_typed",
          "Domain coloring of complex functions: phase portrait + contour grid (typed, node 431)",
          "procedural", '''void main() {
    // Complex plane: uv in [-scale, scale] around (center_x, center_y).
    vec2 uv = (v_uv - 0.5) * 2.0;
    uv.x *= u_resolution.x / u_resolution.y;
    vec2 z = uv * u_scale + vec2(u_center_x, u_center_y);
    // Animate via the live-preview clock u_time (the client advances it so the
    // live preview moves). Rotate the plane while gently drifting the center —
    // mirrors the CPU node's rotate/drift anim modes without a dead phase param.
    float a = u_time * 0.4;
    float ca = cos(a), sa = sin(a);
    z = mat2(ca, -sa, sa, ca) * z;
    z += vec2(sin(u_time * 0.3), cos(u_time * 0.22)) * u_scale * 0.12;
    // f(z) = z^n (the node default 'poly' with exponent n == z_n family).
    float n = max(u_exponent, 2.0);
    float r = length(z), th = atan(z.y, z.x);
    vec2 f = pow(r, n) * vec2(cos(n * th), sin(n * th));
    // Phase portrait: hue = arg f / 2pi; lightness = (2/pi) atan|f|.
    float arg = atan(f.y, f.x) / 6.2831853 + 0.5;
    float mag = atan(length(f)) * 2.0 / 3.14159265;
    vec3 col = 0.5 + 0.5 * cos(6.2831853 * (arg + vec3(0.0, 0.333, 0.667)));
    // 'enhanced'/'grid' contour: darken on log|f| & phase lattice lines.
    float gl = abs(fract(log(length(f) + 1e-3) * 3.0) - 0.5);
    float lp = abs(fract(arg * 12.0) - 0.5);
    float grid = smoothstep(0.02, 0.12, min(gl, lp));
    col *= mix(1.0, grid, u_grid);
    col *= mag;
    f_color = vec4(clamp(col, 0.0, 1.0), 1.0);
}
''', uniforms={
    "exponent":  {"glsl": "float", "min": 2.0, "max": 12.0, "default": 3.0,
                  "description": "power n for z^n"},
    "scale":     {"glsl": "float", "min": 0.5, "max": 8.0, "default": 3.0,
                  "description": "view half-extent in the complex plane"},
    "center_x":  {"glsl": "float", "min": -4.0, "max": 4.0, "default": 0.0,
                  "description": "real part of view center"},
    "center_y":  {"glsl": "float", "min": -4.0, "max": 4.0, "default": 0.0,
                  "description": "imaginary part of view center"},
    "grid":      {"glsl": "float", "min": 0.0, "max": 1.0, "default": 1.0,
                  "description": "contour/grid overlay strength (artistic knob; CPU 'coloring' is a string mode, not a float synonym)"},
})

_register("low_discrepancy_typed",
          "Low-discrepancy (R2) point field: stipple / dot pattern (typed, node 433)",
          "procedural", _INFERNO_GPU + '''void main() {
    vec2 p = (v_uv - 0.5);
    p.x *= u_resolution.x / u_resolution.y;
    float t = u_time * 0.15 * u_speed;
    vec3 col = u_bg;
    // R2 low-discrepancy sequence (Roberts 2018): alpha = (1/phi^2, 1/phi^3).
    vec2 alpha = vec2(0.7548776662, 0.5698402909);
    int N = int(u_count);
    float best = 1e9;
    // Rasterise N dots; highlight the single nearest dot per pixel.
    for (int i = 0; i < 20000; i++) {
        if (i >= N) break;
        float fi = float(i);
        vec2 q = fract(alpha * fi + vec2(u_ox, u_oy) + t * 0.05);
        q -= 0.5; q.x *= u_resolution.x / u_resolution.y;
        // gentle rotation so animation is visible on the point cloud
        float ca = cos(t * 0.3), sa = sin(t * 0.3);
        q = mat2(ca, -sa, sa, ca) * q;
        best = min(best, length(p - q));
    }
    float dot = smoothstep(u_radius, u_radius * 0.3, best);
    col = mix(u_bg, inferno(clamp(1.0 - best * 1.5, 0.0, 1.0)), dot);
    f_color = vec4(col, 1.0);
}
''', uniforms={
    "count":   {"glsl": "int", "min": 50, "max": 20000, "default": 2000,
                "description": "number of sampled points N"},
    "radius":  {"glsl": "float", "min": 0.5, "max": 8.0, "default": 1.5,
                "description": "dot radius in px"},
    "ox":      {"glsl": "float", "min": 0.0, "max": 1.0, "default": 0.0,
                "description": "sequence x offset (seed)"},
    "oy":      {"glsl": "float", "min": 0.0, "max": 1.0, "default": 0.0,
                "description": "sequence y offset (seed)"},
    "speed":   {"glsl": "float", "min": 0.0, "max": 6.0, "default": 1.0,
                "description": "rotation speed"},
    "bg":      {"glsl": "color", "default": "#05060c", "description": "background"},
})

# ── Categorical coverage pt.15 (2026-07-12): closed-form procedural patterns
# with NAMED typed controls — Droste log-spiral, Voronoi stained glass, Op-Art
# sinusoidal band distortion. Each is a pure f(uv,t) field (no ping-pong state)
# so it verifies headlessly via render_shader. ──

# 316 — Droste log-spiral: conformal log-polar mapping (Escher "Print Gallery"
# homage). Rings tile self-similarly in log-radius while the angle winds them
# into a spiral; animation rotates the spiral phase.
_register("droste_typed", "Droste log-polar self-similar spiral (typed, node 316)",
          "procedural", '''void main() {
    vec2 p = v_uv - 0.5;
    p.x *= u_resolution.x / u_resolution.y;
    float r = length(p);
    float a = atan(p.y, p.x);
    float lr = log(max(r, 1e-4));
    // Spiral coordinate: log-radius zoom + angular winding + time phase.
    float coord = lr * u_zoom + a * u_twist + u_time * u_speed * 0.5;
    float band = fract(coord * u_bands);
    float ring = smoothstep(0.46, 0.5, abs(band - 0.5));
    // Radial vignette so the singular centre fades cleanly.
    float vig = smoothstep(0.02, 0.15, r);
    vec3 col = mix(u_bg, u_fg, ring * vig);
    f_color = vec4(col, 1.0);
}
''', uniforms={
    "zoom":   {"glsl": "float", "min": 0.5, "max": 6.0, "default": 2.5, "description": "log-radius zoom (ring density)"},
    "twist":  {"glsl": "float", "min": 0.0, "max": 12.0, "default": 3.0, "description": "angular winding (spiral arms)"},
    "bands":  {"glsl": "float", "min": 1.0, "max": 16.0, "default": 4.0, "description": "band repetition"},
    "speed":  {"glsl": "float", "min": 0.0, "max": 4.0, "default": 0.8, "description": "spiral animation speed"},
    "fg":     {"glsl": "color", "default": "#f0d060", "description": "ring color"},
    "bg":     {"glsl": "color", "default": "#101020", "description": "background color"},
})

# 317 — Voronoi stained glass: F1/F2 cellular decomposition with a flat random
# facet color per cell and dark leaded seams along cell boundaries. Site jitter
# animates the cells so seams drift.
_register("stained_glass_typed", "Voronoi stained-glass facets with leaded seams (typed, node 317)",
          "procedural", '''void main() {
    vec2 p = v_uv;
    p.x *= u_resolution.x / u_resolution.y;
    float N = max(u_cells, 2.0);
    vec2 g = p * N;
    vec2 id = floor(g);
    vec2 f = fract(g);
    float d1 = 8.0, d2 = 8.0;
    vec2 bestId = id;
    for (int y = -1; y <= 1; y++) {
        for (int x = -1; x <= 1; x++) {
            vec2 o = vec2(float(x), float(y));
            vec2 rnd = vec2(hash21(id + o + 0.5), hash21(id + o + 31.4));
            rnd = 0.5 + u_jitter * 0.5 * sin(u_time * u_speed + 6.2831 * rnd);
            vec2 pt = o + rnd - f;
            float d = length(pt);
            if (d < d1) { d2 = d1; d1 = d; bestId = id + o; }
            else if (d < d2) { d2 = d; }
        }
    }
    vec3 facet = 0.35 + 0.65 * vec3(hash21(bestId + 2.1),
                                    hash21(bestId + 5.3),
                                    hash21(bestId + 9.7));
    float lum = dot(facet, vec3(0.333));
    facet = mix(vec3(lum), facet, u_saturation);
    float seam = smoothstep(0.0, max(u_seam, 1e-3), d2 - d1);
    vec3 col = mix(u_seam_color, facet, seam);
    f_color = vec4(col, 1.0);
}
''', uniforms={
    "cells":      {"glsl": "float", "min": 2.0, "max": 30.0, "default": 9.0, "description": "cells per axis"},
    "jitter":     {"glsl": "float", "min": 0.0, "max": 1.0, "default": 0.6, "description": "site drift amount"},
    "seam":       {"glsl": "float", "min": 0.01, "max": 0.25, "default": 0.07, "description": "leaded seam width"},
    "saturation": {"glsl": "float", "min": 0.0, "max": 1.0, "default": 0.85, "description": "facet color saturation"},
    "speed":      {"glsl": "float", "min": 0.0, "max": 3.0, "default": 0.5, "description": "cell animation speed"},
    "seam_color": {"glsl": "color", "default": "#0a0a0f", "description": "seam (lead) color"},
})

# 318 — Op-Art band distortion: parallel bands whose position is sinusoidally
# displaced (Bridget Riley homage). The whole field rotates and the wave phase
# animates for a shimmering moiré effect.
_register("opart_typed", "Op-Art sinusoidal band distortion (typed, node 318)",
          "procedural", '''void main() {
    vec2 p = v_uv - 0.5;
    p.x *= u_resolution.x / u_resolution.y;
    p = rot(u_rotation) * p;
    float disp = u_amplitude * sin(p.x * u_freq_x * 6.2831 + u_time * u_speed);
    float y = p.y + disp;
    float band = fract(y * u_bands);
    float stripe = smoothstep(0.47, 0.5, abs(band - 0.5));
    vec3 col = mix(u_bg, u_fg, stripe);
    f_color = vec4(col, 1.0);
}
''', uniforms={
    "bands":     {"glsl": "float", "min": 2.0, "max": 40.0, "default": 14.0, "description": "band frequency"},
    "amplitude": {"glsl": "float", "min": 0.0, "max": 0.5, "default": 0.12, "description": "wave displacement"},
    "freq_x":    {"glsl": "float", "min": 0.5, "max": 8.0, "default": 2.0, "description": "horizontal wave frequency"},
    "rotation":  {"glsl": "float", "min": 0.0, "max": 3.14159, "default": 0.0, "description": "field rotation (radians)"},
    "speed":     {"glsl": "float", "min": 0.0, "max": 5.0, "default": 1.2, "description": "wave animation speed"},
    "fg":        {"glsl": "color", "default": "#f5f5f5", "description": "stripe color"},
    "bg":        {"glsl": "color", "default": "#101014", "description": "background color"},
})

# 319 — Aurora Borealis: real-time procedural northern-lights curtain. A
# domain-warped sinusoidal energy field produces vertical "rays" that drift with
# time; a Gaussian sky-band window localises the curtain, and an x/y-driven hue
# ramp sweeps green→violet. Closed-form f(uv,t) — no texture, no raymarch loop —
# so it is a cheap procedural twin (good live-preview + fast export). References:
# Roy Theunissen's "Aurora Borealis: A Breakdown" (2022) for the layered-curtain
# model and the GodotShaders volumetric-aurora approach for the energy-field look.
_register("aurora_typed", "Aurora Borealis — real-time drifting light-curtain (typed, node 319)",
          "procedural", '''vec3 _hsv(float h, float s, float v) {
    vec3 k = vec3(1.0, 2.0/3.0, 1.0/3.0);
    vec3 p = abs(fract(vec3(h) + k) * 6.0 - 3.0);
    return v * mix(vec3(1.0), clamp(p - 1.0, 0.0, 1.0), s);
}
void main() {
    vec2 uv = v_uv;
    vec2 p = (uv - 0.5) * 2.0;
    p.x *= u_resolution.x / u_resolution.y;
    float t = u_time * u_speed * 0.2;

    // Sky gradient background (dark zenith, faint blue at horizon).
    vec3 sky = mix(u_sky_bottom, u_sky_top, pow(uv.y, 0.6));

    // Aurora occupies an upper band of the sky.
    float band = exp(-pow((uv.y - u_center) / u_thickness, 2.0));

    // Warped horizontal field drives the curtain ribbons (vertical streaks).
    float x = p.x * u_scale;
    float warp = fbm(vec2(x * 0.4, t)) * 2.5;
    float ph = x + warp + t * 1.3;

    // Several drifting ribbon layers; rays fade upward so they rise like light.
    float ribbon = 0.0;
    float wsum = 0.0;
    for (int i = 0; i < 3; i++) {
        float fi = float(i);
        float amp = 1.0 - fi * 0.25;
        float s = sin(ph * (1.0 + fi * 0.5) + fi * 2.0);
        float streak = smoothstep(0.75, 1.0, abs(s)) * amp;
        streak *= smoothstep(u_center + u_thickness, u_center - u_thickness, uv.y);
        ribbon += streak;
        wsum += amp;
    }
    ribbon /= max(wsum, 0.001);

    float rays = ribbon * band;

    // Colour: green base with violet tips, swept by x and height.
    float hue = mix(u_hue_green, u_hue_violet,
                    clamp(0.5 + 0.5 * sin(x * 0.15 + t), 0.0, 1.0));
    hue = fract(hue - uv.y * 0.15);
    vec3 aurora = _hsv(hue, 0.9, 1.0) * rays * 1.6;

    // Faint static star speckle outside the band.
    float stars = step(0.996, hash21(floor(uv * u_resolution / 2.0))) * (1.0 - band);
    sky += stars * 0.5;

    vec3 col = clamp(sky + aurora, 0.0, 1.0);
    f_color = vec4(col, 1.0);
}
''', uniforms={
    "speed":      {"glsl": "float", "min": 0.0, "max": 4.0, "default": 1.0,
                   "description": "curtain drift speed"},
    "scale":      {"glsl": "float", "min": 0.5, "max": 12.0, "default": 3.0,
                   "description": "ribbon frequency"},
    "center":     {"glsl": "float", "min": 0.2, "max": 0.9, "default": 0.62,
                   "description": "curtain height (sky band centre)"},
    "thickness":  {"glsl": "float", "min": 0.02, "max": 0.4, "default": 0.14,
                   "description": "curtain thickness"},
    "hue_green":  {"glsl": "float", "min": 0.0, "max": 1.0, "default": 0.33,
                   "description": "base green hue"},
    "hue_violet": {"glsl": "float", "min": 0.0, "max": 1.0, "default": 0.78,
                   "description": "tip violet hue"},
    "sky_bottom": {"glsl": "color", "default": "#02030a", "description": "horizon sky"},
    "sky_top":    {"glsl": "color", "default": "#0a1430", "description": "zenith sky"},
})


_register("marble_typed", "Marble — Perlin-turbulence veining with domain warp (typed, node 320)",
          "procedural", '''float _turb(vec2 p, int oct) {
    // Absolute-value fbm (Perlin turbulence): sharp filaments instead of soft fbm.
    float sum = 0.0, amp = 1.0, freq = 1.0, norm = 0.0;
    for (int i = 0; i < 6; i++) {
        if (i >= oct) break;
        sum  += amp * abs(noise(p * freq) * 2.0 - 1.0);
        norm += amp;
        amp  *= 0.5;
        freq *= 2.0;
    }
    return sum / max(norm, 0.001);
}
void main() {
    vec2 uv = v_uv;
    vec2 p = uv;
    p.x *= u_resolution.x / u_resolution.y;
    float t = u_time * u_speed * 0.1;

    // Slowly drifting turbulence field warps the vein coordinate.
    vec2 warp = vec2(t, -t * 0.6);
    float turb = _turb(p * u_scale + warp, int(u_octaves + 0.5));

    // Directional vein axis (angle in turns) + turbulence distortion.
    float ang = u_angle * 6.2831853;
    float axis = p.x * cos(ang) + p.y * sin(ang);
    float veins = axis * u_freq + turb * u_distortion;

    // Sharp sinusoidal bands -> marble veins.
    float m = 0.5 + 0.5 * sin(veins * 6.2831853);
    m = pow(m, max(u_sharpness, 0.01));

    vec3 col = mix(u_base_color, u_vein_color, clamp(m, 0.0, 1.0));

    // Subtle self-shadow from the raw turbulence for depth.
    col *= 0.75 + 0.25 * (1.0 - turb);

    f_color = vec4(clamp(col, 0.0, 1.0), 1.0);
}
''', uniforms={
    "speed":      {"glsl": "float", "min": 0.0, "max": 4.0, "default": 1.0,
                   "description": "turbulence drift speed"},
    "scale":      {"glsl": "float", "min": 0.5, "max": 12.0, "default": 3.0,
                   "description": "turbulence field scale"},
    "octaves":    {"glsl": "int", "min": 1.0, "max": 6.0, "default": 5.0,
                   "description": "turbulence octaves (detail)"},
    "freq":       {"glsl": "float", "min": 0.5, "max": 16.0, "default": 4.0,
                   "description": "vein frequency"},
    "angle":      {"glsl": "float", "min": 0.0, "max": 1.0, "default": 0.12,
                   "description": "vein direction (turns)"},
    "distortion": {"glsl": "float", "min": 0.0, "max": 6.0, "default": 2.0,
                   "description": "turbulence distortion of veins"},
    "sharpness":  {"glsl": "float", "min": 0.2, "max": 8.0, "default": 2.5,
                   "description": "vein contrast/sharpness"},
    "base_color": {"glsl": "color", "default": "#1a1c22", "description": "stone base colour"},
    "vein_color": {"glsl": "color", "default": "#e8e4d8", "description": "vein colour"},
})


# ── Node 323: Raymarched 3D Gyroid TPMS — sphere-traced triply-periodic
#    minimal surface with lambert+specular lighting and camera orbit. Distinct
#    from node 301 (flat 2D slice of the scalar field): this sphere-traces the
#    thick-walled gyroid volume in 3D, so it reads as a solid woven lattice with
#    real depth, self-occlusion and shading. TPMS ref: Inigo Quilez raymarching
#    + gyroid f(p)=dot(sin(p),cos(p.yzx)).
_register("gyroid_raymarch_typed",
          "Raymarched 3D Gyroid TPMS — sphere-traced minimal surface (typed, node 323)",
          "procedural", '''float gyroid(vec3 p) {
    // Signed thick-shell distance to the gyroid iso-surface. The raw field is
    // dot(sin(p), cos(p.yzx)); subtracting u_bias thickens the walls (a
    // level-set band around 0), and the /scale factor is a Lipschitz correction
    // so sphere-tracing stays conservative.
    float f = dot(sin(p), cos(p.yzx));
    return (abs(f) - u_bias) / (u_freq * 2.5);
}
vec3 gyro_normal(vec3 p) {
    vec2 e = vec2(0.001, 0.0);
    return normalize(vec3(
        gyroid(p + e.xyy) - gyroid(p - e.xyy),
        gyroid(p + e.yxy) - gyroid(p - e.yxy),
        gyroid(p + e.yyx) - gyroid(p - e.yyx)));
}
void main() {
    vec2 uv = (v_uv - 0.5);
    uv.x *= u_resolution.x / u_resolution.y;
    float t = u_time * u_anim_speed;

    // Orbiting camera around the lattice.
    float ca = t * 0.3 + u_cam_angle * 6.2831853;
    vec3 ro = vec3(sin(ca), 0.35, cos(ca)) * u_cam_dist;
    vec3 fw = normalize(-ro);
    vec3 rt = normalize(cross(vec3(0.0, 1.0, 0.0), fw));
    vec3 up = cross(fw, rt);
    vec3 rd = normalize(fw + uv.x * rt * 1.2 + uv.y * up * 1.2);

    // Frequency scales the lattice; drift advances the field through space.
    vec3 drift = vec3(0.0, t * 0.4, 0.0);

    float d = 0.0;
    float hit = 0.0;
    vec3 pos = ro;
    for (int i = 0; i < 90; i++) {
        pos = ro + rd * d;
        float ds = gyroid(pos * u_freq + drift);
        if (ds < 0.0008) { hit = 1.0; break; }
        d += ds;
        if (d > 14.0) break;
    }

    vec3 col = u_bg;
    if (hit > 0.5) {
        vec3 n = gyro_normal(pos * u_freq + drift);
        vec3 ld = normalize(vec3(0.6, 0.8, 0.4));
        float dif = clamp(dot(n, ld), 0.0, 1.0);
        float spe = pow(clamp(dot(reflect(-ld, n), -rd), 0.0, 1.0), 32.0);
        // Depth-tinted base colour cycling with camera distance travelled.
        float depth = clamp(d / 10.0, 0.0, 1.0);
        vec3 base = mix(u_near_color, u_far_color, depth);
        col = base * (0.18 + 0.82 * dif) + vec3(1.0) * spe * u_spec;
        // Fake AO from march-step count folded into depth darkening.
        col *= 1.0 - depth * 0.35;
    }
    col = pow(clamp(col, 0.0, 1.0), vec3(0.4545));  // gamma
    f_color = vec4(col, 1.0);
}
''', uniforms={
    "freq":       {"glsl": "float", "min": 0.4, "max": 4.0, "default": 1.4,
                   "description": "lattice frequency (cells per unit)"},
    "bias":       {"glsl": "float", "min": 0.0, "max": 1.2, "default": 0.35,
                   "description": "wall thickness (level-set band)"},
    "cam_dist":   {"glsl": "float", "min": 2.0, "max": 8.0, "default": 4.5,
                   "description": "camera distance from lattice"},
    "cam_angle":  {"glsl": "float", "min": 0.0, "max": 1.0, "default": 0.0,
                   "description": "static camera azimuth offset (turns)"},
    "anim_speed": {"glsl": "float", "min": 0.0, "max": 4.0, "default": 1.0,
                   "description": "orbit + drift speed"},
    "spec":       {"glsl": "float", "min": 0.0, "max": 1.5, "default": 0.6,
                   "description": "specular highlight strength"},
    "near_color": {"glsl": "color", "default": "#ff7a3c",
                   "description": "near-surface colour"},
    "far_color":  {"glsl": "color", "default": "#2a6cff",
                   "description": "far-surface colour"},
    "bg":         {"glsl": "color", "default": "#05060a",
                   "description": "background colour"},
})


# ── Node 503: Conformal Warp — Möbius map deforms a base grid (client-GPU twin) ──
_register("conformal_gpu", "Conformal warp — Möbius map deforms a base grid (client-GPU twin of node 503)", "procedural", '''
vec2 cz_mul(vec2 a, vec2 b){ return vec2(a.x*b.x - a.y*b.y, a.x*b.y + a.y*b.x); }
vec2 cz_conj(vec2 a){ return vec2(a.x, -a.y); }
vec2 cz_div(vec2 a, vec2 b){ float d = dot(b,b) + 1e-6; return vec2(a.x*b.x + a.y*b.y, a.y*b.x - a.x*b.y) / d; }
float cz_ss(float a, float b, float x){ float t = clamp((x-a)/(b-a), 0.0, 1.0); return t*t*(3.0-2.0*t); }

void main() {
    vec2 res = u_resolution;
    float aspect = res.x / res.y;
    // centered, aspect-correct complex coordinate
    vec2 z = (v_uv - 0.5) * 2.0;
    z.x *= aspect;
    float scale = clamp(u_scale * 6.0, 0.5, 8.0);   // domain radius
    z *= scale;
    float warp = clamp(u_warp, 0.0, 0.92);          // |a|
    float anim = u_time * clamp(u_anim_speed * 2.0, 0.05, 4.0); // anim speed

    // Möbius coefficient a orbits inside the unit disk => disk->disk conformal map
    vec2 a = warp * vec2(cos(anim), sin(anim));
    // also rotate the input plane for extra life
    float grot = anim * 0.5;
    z = vec2(z.x*cos(grot) - z.y*sin(grot), z.x*sin(grot) + z.y*cos(grot));
    vec2 num = z - a;
    vec2 den = vec2(1.0, 0.0) - cz_mul(cz_conj(a), z);
    vec2 w = cz_div(num, den);

    // base grid pattern sampled at the warped coordinate
    float k = 6.0;
    vec2 g = w * k;
    float fx = abs(fract(g.x) - 0.5);
    float fy = abs(fract(g.y) - 0.5);
    float line = cz_ss(0.40, 0.49, max(fx, fy));
    float r = length(w);
    float shade = clamp(line * (0.55 + 0.45 * sin(r * 2.5)), 0.0, 1.0);
    vec3 base = mix(vec3(0.04, 0.06, 0.11), vec3(0.92, 0.96, 1.0), shade);
    // subtle hue from argument(w)
    float ang = atan(w.y, w.x);
    vec3 tint = 0.14 * vec3(cos(ang), cos(ang + 2.094), cos(ang + 4.188)) * (1.0 - shade);
    f_color = vec4(base + tint, 1.0);
}
''', uniforms={
    "scale":      {"glsl": "float", "min": 0.0, "max": 1.0, "default": 0.5,
                   "description": "domain radius (scaled x6 internally, clamped 0.5-8.0)"},
    "warp":       {"glsl": "float", "min": 0.0, "max": 1.0, "default": 0.5,
                   "description": "Mobius coefficient magnitude |a| (0-0.92)"},
    "anim_speed": {"glsl": "float", "min": 0.0, "max": 1.0, "default": 0.5,
                   "description": "sun/orbit animation speed (scaled x2 internally)"},
})

# ── GPU-First categorical coverage (2026-07-14): Nishita atmospheric
# single-scattering sky — the GPU live-preview twin of CPU node 471. Closed-form
# per-pixel ray-march (no ping-pong state) so it verifies headlessly. Works in
# km units so the scale-height exponentials stay in a fp32-safe range (the CPU
# node's metre units would underflow exp() in fp32). The sun rides a day-arc
# driven by u_time (which advances one step per frame; pace is set by the
# graph's wired drivers/substeps, not a time_scale multiplier), so the live
# preview animates; sun_elevation/sun_azimuth are the base pose the graph can
# also wire
# CHOP generators into. CPU numpy node stays authoritative for export. ──
_register("nishita_sky_gpu",
          "Nishita atmospheric single-scattering sky (GPU twin of CPU node 471) "
          "— Rayleigh + Mie scattering integrated along view and light rays through "
          "a spherical atmosphere; animated sun day-arc via u_time.",
          "procedural", '''void main() {
    vec2 uv = v_uv * 2.0 - 1.0;
    uv.x *= u_resolution.x / u_resolution.y;
    float fov = radians(u_fov);
    float f = tan(fov * 0.5);
    vec3 rd = normalize(vec3(uv.x * f, uv.y * f, -1.0));
    // Camera 1 km above the ground, inside the spherical atmosphere (units: km).
    vec3 ro = vec3(0.0, 6361.0, 0.0);

    // Animated sun: a day-arc around the node's base elevation/azimuth.
    // u_time advances one step per frame (pace via wired drivers/substeps),
    // so the live preview moves smoothly.
    float baseEl = radians(u_sun_elevation);
    float baseAz = radians(u_sun_azimuth);
    float el = clamp(baseEl + sin(u_time) * radians(40.0), -1.4, 1.55);
    float az = baseAz + u_time * 0.25;
    vec3 sun = normalize(vec3(cos(el) * sin(az), sin(el), -cos(el) * cos(az)));

    float Rg = 6360.0, Rt = 6420.0, Hr = 8.0, Hm = 1.2;

    // Ray vs top-of-atmosphere and ground spheres (a = 1).
    float b = dot(ro, rd);
    float cT = dot(ro, ro) - Rt * Rt;
    float tTop = -b + sqrt(max(b * b - cT, 0.0));
    float cG = dot(ro, ro) - Rg * Rg;
    float discG = b * b - cG;
    float tGround = -b - sqrt(max(discG, 0.0));
    bool ground = (discG >= 0.0) && (tGround > 0.0);
    float tMax = ground ? min(tTop, tGround) : tTop;
    tMax = max(tMax, 0.0);

    // Phase functions (view/sun angle).
    float cosT = dot(rd, sun);
    float phaseR = 3.0 / (16.0 * 3.14159265) * (1.0 + cosT * cosT);
    float g = 0.76;
    float denom = (2.0 + g * g) * pow(1.0 + g * cosT, 2.0);
    float phaseM = 3.0 / (8.0 * 3.14159265) * (1.0 - g * g) * (1.0 + cosT * cosT) / max(denom, 1e-8);

    vec3 betaR = vec3(5.8e-3, 13.5e-3, 33.1e-3) * u_rayleigh_k;
    vec3 betaM = vec3(21e-3) * u_mie_k;
    float SUN_I = 20.0;

    const int NS = 16, NL = 8;
    float seg = tMax / float(NS);
    vec3 sumR = vec3(0.0), sumM = vec3(0.0);
    vec3 lastAtten = vec3(1.0);
    float odr = 0.0, odm = 0.0;

    for (int i = 0; i < NS; i++) {
        float t = (float(i) + 0.5) * seg;
        vec3 p = ro + t * rd;
        float alt = length(p) - Rg;
        float hr = exp(-alt / Hr) * seg;
        float hm = exp(-alt / Hm) * seg;
        odr += hr; odm += hm;
        // Light ray optical depth toward the sun.
        float bL = dot(p, sun);
        float cL = dot(p, p) - Rt * Rt;
        float tTopL = -bL + sqrt(max(bL * bL - cL, 0.0));
        float segL = max(tTopL, 0.0) / float(NL);
        float odlr = 0.0, odlm = 0.0;
        for (int j = 0; j < NL; j++) {
            float tl = (float(j) + 0.5) * segL;
            vec3 pl = p + tl * sun;
            float altL = length(pl) - Rg;
            odlr += exp(-altL / Hr) * segL;
            odlm += exp(-altL / Hm) * segL;
        }
        vec3 tau = betaR * (odr + odlr) + betaM * (1.1 * (odm + odlm));
        vec3 atten = exp(-tau);
        lastAtten = atten;
        sumR += atten * hr * betaR * phaseR;
        sumM += atten * hm * betaM * phaseM;
    }

    vec3 col = (sumR + sumM) * SUN_I;
    // Sun disk + soft halo, modulated by transmittance toward the sun.
    float ang = acos(clamp(cosT, -1.0, 1.0));
    float rDisk = radians(u_sun_disk_radius);
    float disk = clamp(1.0 - ang / rDisk, 0.0, 1.0);
    float halo = clamp(1.0 - ang / (rDisk * 4.0), 0.0, 1.0) * 0.3;
    float glow = disk + halo;
    float trans = (lastAtten.r + lastAtten.g + lastAtten.b) / 3.0;
    col += glow * trans * vec3(1.0, 0.95, 0.85) * 20.0;
    // Tonemap + clamp.
    col = vec3(1.0) - exp(-col * u_exposure);
    f_color = vec4(clamp(col, 0.0, 1.0), 1.0);
}
''', uniforms={
    "sun_elevation":   {"glsl": "float", "min": -10.0, "max": 90.0, "default": 6.0,
                        "description": "sun elevation in degrees (base for the animated day-arc)"},
    "sun_azimuth":     {"glsl": "float", "min": 0.0, "max": 360.0, "default": 90.0,
                        "description": "sun azimuth in degrees"},
    "rayleigh_k":      {"glsl": "float", "min": 0.2, "max": 3.0, "default": 1.0,
                        "description": "Rayleigh (molecular) scattering strength"},
    "mie_k":           {"glsl": "float", "min": 0.0, "max": 3.0, "default": 1.0,
                        "description": "Mie (aerosol) scattering strength"},
    "exposure":        {"glsl": "float", "min": 0.1, "max": 3.0, "default": 1.0,
                        "description": "tonemap exposure"},
    "fov":             {"glsl": "float", "min": 20.0, "max": 120.0, "default": 60.0,
                        "description": "vertical field of view in degrees"},
    "sun_disk_radius": {"glsl": "float", "min": 0.3, "max": 5.0, "default": 1.2,
                        "description": "apparent sun disk radius in degrees"},
})


_register("interior_mapping_typed",
          "Interior Mapping — fake 3D rooms behind a facade via ray-box intersection (van Dongen 2008, typed, node 328)",
          "procedural", '''
// Interior Mapping (Joost van Dongen, CGI 2008): render believable interior
// rooms behind a flat facade WITHOUT any extra geometry. For each screen pixel
// we build an eye ray into a virtual room-box (one cell of a repeating grid of
// rooms) and intersect it against the 6 walls; the nearest wall behind the
// facade plane is shaded with a simple depth-tinted colour + a glowing window
// light. Ceiling/floor/back/side walls each get their own tint so the illusion
// of depth reads clearly. Windows are lit per-room by a hashed on/off so the
// building looks inhabited, and u_time drives a slow twinkle.
float _im_hash(vec2 c){ return fract(sin(dot(c, vec2(41.3, 289.7))) * 43758.5453); }
void main() {
    vec2 uv = v_uv;
    float aspect = u_resolution.x / u_resolution.y;
    // Facade plane coords, scaled into a grid of rooms.
    vec2 fp = vec2(uv.x * aspect, uv.y) * u_rooms;
    vec2 cell = floor(fp);            // which room
    vec2 f = fract(fp);               // position on the facade within the room [0,1]

    // Per-room hash → window-lit state + slight room-depth variation.
    float h = _im_hash(cell);
    float lit = step(1.0 - u_lit_frac, _im_hash(cell + 3.1));
    // Twinkle: lit rooms flicker slowly.
    float tw = 0.75 + 0.25 * sin(u_time * u_speed * 2.0 + h * 30.0);
    lit *= tw;

    // Eye ray. Camera sits in front of the facade (z = -dist); we look toward
    // +z into the room. Screen offset from room centre gives ray direction.
    float depth = u_depth * (0.7 + 0.6 * h);   // room depth varies per room
    vec3 ro = vec3(0.0, 0.0, -u_cam_dist);
    // Ray direction: parallax from facade-local coords centred at 0.
    vec3 rd = normalize(vec3((f - 0.5) * u_parallax, 1.0));

    // Room box spans x,y in [-0.5,0.5], z in [0, depth]. Intersect ray with the
    // far planes; pick the nearest positive-z surface = the wall we see.
    // Slab method on each axis for the exit face.
    vec3 inv = 1.0 / rd;
    // x walls at +/-0.5, y walls at +/-0.5, back wall at z=depth.
    float tx = ((rd.x > 0.0 ? 0.5 : -0.5) - ro.x) * inv.x;
    float ty = ((rd.y > 0.0 ? 0.5 : -0.5) - ro.y) * inv.y;
    float tz = (depth - ro.z) * inv.z;
    float tHit = min(min(tx, ty), tz);
    vec3 hit = ro + rd * tHit;

    // Classify which wall was hit → base tint.
    vec3 wallCol;
    float shade;
    if (tHit == tz) {
        wallCol = u_back;                       // back wall
        shade = 1.0;
    } else if (tHit == ty) {
        wallCol = (rd.y > 0.0) ? u_ceiling : u_floor;
        shade = 0.85;
    } else {
        wallCol = u_side;                        // side walls
        shade = 0.7;
    }
    // Depth fog: deeper points darker.
    float zf = clamp(hit.z / max(depth, 0.001), 0.0, 1.0);
    shade *= mix(1.0, u_ambient, zf);

    // Window glow: a warm rectangle on the back wall for lit rooms.
    vec3 col = wallCol * shade;
    if (lit > 0.001) {
        // back-wall panel glow, brighter toward the back.
        float panel = smoothstep(0.55, 0.35, abs(hit.x)) *
                      smoothstep(0.55, 0.35, abs(hit.y));
        col += u_light * lit * panel * (0.4 + 0.6 * zf);
    }

    // Thin mullion frame on the facade between rooms (window bars).
    float bar = min(smoothstep(0.0, u_frame, f.x) * smoothstep(1.0, 1.0 - u_frame, f.x),
                    smoothstep(0.0, u_frame, f.y) * smoothstep(1.0, 1.0 - u_frame, f.y));
    col = mix(u_frame_col, col, bar);

    f_color = vec4(clamp(col, 0.0, 1.0), 1.0);
}
''', uniforms={
    "rooms":     {"glsl": "float", "min": 1.0, "max": 20.0, "default": 7.0,
                  "description": "rooms across the facade (grid density)"},
    "depth":     {"glsl": "float", "min": 0.3, "max": 4.0, "default": 1.5,
                  "description": "room depth (parallax strength)"},
    "cam_dist":  {"glsl": "float", "min": 0.2, "max": 3.0, "default": 1.0,
                  "description": "camera distance from facade"},
    "parallax":  {"glsl": "float", "min": 0.2, "max": 2.0, "default": 1.0,
                  "description": "eye-ray spread (viewing angle)"},
    "lit_frac":  {"glsl": "float", "min": 0.0, "max": 1.0, "default": 0.5,
                  "description": "fraction of rooms with lights on"},
    "speed":     {"glsl": "float", "min": 0.0, "max": 4.0, "default": 1.0,
                  "description": "window twinkle speed"},
    "ambient":   {"glsl": "float", "min": 0.05, "max": 1.0, "default": 0.3,
                  "description": "deep-room ambient floor"},
    "frame":     {"glsl": "float", "min": 0.0, "max": 0.3, "default": 0.06,
                  "description": "window mullion frame width"},
    "back":      {"glsl": "color", "default": "#3a4152", "description": "back wall colour"},
    "ceiling":   {"glsl": "color", "default": "#5a6273", "description": "ceiling colour"},
    "floor":     {"glsl": "color", "default": "#2a2e38", "description": "floor colour"},
    "side":      {"glsl": "color", "default": "#454b5a", "description": "side wall colour"},
    "light":     {"glsl": "color", "default": "#ffd28a", "description": "window light colour"},
    "frame_col": {"glsl": "color", "default": "#12141a", "description": "mullion frame colour"},
})


# ── Node 331: Mandelbulb — 3D escape-time fractal (Daniel White & Paul
# Nylander 2009), sphere-traced with the Hart, Sandin & Kauffman (1989)
# analytical distance estimator. Distinct from the 2D escape-time family
# (Mandelbrot/Julia/Burning Ship) and from the 3D TPMS raymarches (gyroid/
# menger): this is the canonical "3D Mandelbrot". The power exponent slowly
# morphs with u_time (sin breathing) so the bulb is genuinely time-varying
# (avoids the contrast-only static cull), and the camera orbits it. ──
_register("mandelbulb_gpu",
          "Mandelbulb — 3D escape-time fractal (White & Nylander 2009), sphere-traced "
          "via the Hart et al. 1989 distance estimator; animated power morph + orbiting camera",
          "procedural", '''float mandelDE(vec3 pos, float power, out float orbit) {
    vec3 z = pos;
    float dr = 1.0;
    float r = 0.0;
    float trap = 0.0;
    for (int i = 0; i < 24; i++) {
        if (float(i) >= u_iterations) break;
        r = length(z);
        if (r > u_bailout) break;
        float theta = acos(clamp(z.z / max(r, 1e-6), -1.0, 1.0));
        float phi = atan(z.y, z.x);
        dr = pow(r, power - 1.0) * power * dr + 1.0;
        float zr = pow(r, power);
        theta *= power;
        phi *= power;
        z = zr * vec3(sin(theta) * cos(phi), sin(theta) * sin(phi), cos(theta)) + pos;
        trap = float(i);
    }
    orbit = clamp(trap / max(u_iterations, 1.0), 0.0, 1.0);
    return 0.5 * log(max(r, 1.0001)) * r / max(dr, 1e-6);
}
vec3 mandelNormal(vec3 p, float power) {
    vec2 e = vec2(0.0008, 0.0);
    float oa, ob, oc, od;
    return normalize(vec3(
        mandelDE(p + e.xyy, power, oa) - mandelDE(p - e.xyy, power, ob),
        mandelDE(p + e.yxy, power, oc) - mandelDE(p - e.yxy, power, od),
        mandelDE(p + e.yyx, power, oa) - mandelDE(p - e.yyx, power, ob)));
}
void main() {
    vec2 uv = (v_uv - 0.5);
    uv.x *= u_resolution.x / u_resolution.y;
    float t = u_time * u_anim_speed;
    // Power morph: the signature bulb shape breathes between powers so the
    // fractal genuinely animates even with a static camera.
    float power = clamp(u_power + 1.0 * sin(t * 0.3), 2.0, 12.0);

    float ca = t * 0.3 + u_cam_angle * 6.2831853;
    vec3 ro = vec3(sin(ca), 0.35, cos(ca)) * u_cam_dist;
    vec3 fw = normalize(-ro);
    vec3 rt = normalize(cross(vec3(0.0, 1.0, 0.0), fw));
    vec3 up = cross(fw, rt);
    vec3 rd = normalize(fw + uv.x * rt * 1.3 + uv.y * up * 1.3);

    float d = 0.0;
    float hit = 0.0;
    vec3 pos = ro;
    float orbit = 0.0;
    int steps = 0;
    for (int i = 0; i < 160; i++) {
        pos = ro + rd * d;
        float o;
        float ds = mandelDE(pos, power, o);
        orbit = o;
        if (ds < 0.0008) { hit = 1.0; break; }
        d += ds;
        steps = i;
        if (d > 12.0) break;
    }

    vec3 col = u_bg;
    if (hit > 0.5) {
        vec3 n = mandelNormal(pos, power);
        vec3 ld = normalize(vec3(0.6, 0.8, 0.4));
        float dif = clamp(dot(n, ld), 0.0, 1.0);
        float spe = pow(clamp(dot(reflect(-ld, n), -rd), 0.0, 1.0), 24.0);
        vec3 base = mix(u_base_color, u_glow_color, orbit);
        col = base * (0.2 + 0.8 * dif) + vec3(1.0) * spe * u_spec;
        col += u_glow_color * (1.0 - orbit) * 0.4;  // inner-trap glow
        float ao = 1.0 - float(steps) / 160.0 * 0.4;  // step-count fake AO
        col *= ao;
    }
    col = pow(clamp(col, 0.0, 1.0), vec3(0.4545));  // gamma
    f_color = vec4(col, 1.0);
}
''', uniforms={
    "power":      {"glsl": "float", "min": 2.0, "max": 12.0, "default": 8.0,
                   "description": "fractal exponent (signature bulb power)"},
    "iterations": {"glsl": "float", "min": 4.0, "max": 20.0, "default": 12.0,
                   "description": "escape-time iteration cap (detail)"},
    "cam_dist":   {"glsl": "float", "min": 1.5, "max": 5.0, "default": 2.6,
                   "description": "camera distance from the bulb"},
    "cam_angle":  {"glsl": "float", "min": 0.0, "max": 1.0, "default": 0.0,
                   "description": "static camera azimuth offset (turns)"},
    "anim_speed": {"glsl": "float", "min": 0.0, "max": 4.0, "default": 1.0,
                   "description": "camera orbit + power-morph speed"},
    "bailout":    {"glsl": "float", "min": 2.0, "max": 8.0, "default": 4.0,
                   "description": "escape radius (shape / level of detail)"},
    "spec":       {"glsl": "float", "min": 0.0, "max": 1.5, "default": 0.6,
                   "description": "specular highlight strength"},
    "base_color": {"glsl": "color", "default": "#ffb24a",
                   "description": "surface base colour"},
    "glow_color": {"glsl": "color", "default": "#3aa0ff",
                   "description": "orbit-trap glow colour"},
    "bg":         {"glsl": "color", "default": "#05060a",
                   "description": "background colour"},
})


# ── Node 336: Mandelbox — 3D escape-time fractal (Tom Lowe, 2010), the
# box-fold + sphere-fold companion to the Mandelbulb. Each iteration applies a
# per-axis mirror "box fold" (z -> 2*clamp(z,-s,s) - z) then a radius-clamped
# "sphere fold" (pull near-origin points outward, push far points in) and the
# scale*+c affine map; the Hart et al. 1989 scalar distance estimator divides
# the length of z by |dr| (the accumulated derivative) to give a raymarchable
# DE. The negative scale (default -1.5) yields the iconic tiled infinite-rooms
# look, distinct from the bulb (power morph) and from the KIFS wedge-fold
# (node 402/330). An orbiting camera + a subtle scale breathing keep it
# genuinely time-varying so animation drivers have a visibly-responsive target
# and the contrast-only static liveness cull is avoided. ──
_register("mandelbox_gpu",
          "Mandelbox — 3D escape-time fractal (Tom Lowe 2010), box-fold + sphere-fold "
          "DE raymarch (Hart et al. 1989); orbiting camera + scale breathing",
          "procedural", '''float mandelboxDE(vec3 p, float scale, out float orbit) {
    vec3 z = p;
    vec3 c = p;            // Mandelbox: offset c = starting point (connected tiling)
    float dr = 1.0;
    float trap = 0.0;
    float fr2 = u_fixed_radius * u_fixed_radius;
    float mr2 = u_min_radius * u_min_radius;
    for (int i = 0; i < 24; i++) {
        if (float(i) >= u_iterations) break;
        // Box fold: mirror z back into the [-fold, fold] box.
        z = clamp(z, -u_fold, u_fold) * 2.0 - z;
        // Sphere fold: conditional radius rescale.
        float r2 = dot(z, z);
        if (r2 < mr2) {
            float t = fr2 / mr2;
            z *= t; dr *= t;
        } else if (r2 < fr2) {
            float t = fr2 / max(r2, 1e-6);
            z *= t; dr *= t;
        }
        z = scale * z + c;
        dr = dr * abs(scale) + 1.0;
        trap = float(i);
        if (dot(z, z) > 16.0) break;   // escape
    }
    orbit = clamp(trap / max(u_iterations, 1.0), 0.0, 1.0);
    return length(z) / abs(dr);
}
vec3 mandelboxNormal(vec3 p, float scale) {
    vec2 e = vec2(0.0009, 0.0);
    float oa, ob, oc, od;
    return normalize(vec3(
        mandelboxDE(p + e.xyy, scale, oa) - mandelboxDE(p - e.xyy, scale, ob),
        mandelboxDE(p + e.yxy, scale, oc) - mandelboxDE(p - e.yxy, scale, od),
        mandelboxDE(p + e.yyx, scale, oa) - mandelboxDE(p - e.yyx, scale, ob)));
}
void main() {
    vec2 uv = (v_uv - 0.5);
    uv.x *= u_resolution.x / u_resolution.y;
    float t = u_time * u_anim_speed;
    // Scale breathing: keep the iconic negative-scale tiling while morphing.
    float scale = clamp(u_scale + 0.25 * sin(t * 0.25), -2.5, 3.0);

    float ca = t * 0.3 + u_cam_angle * 6.2831853;
    vec3 ro = vec3(sin(ca), 0.35, cos(ca)) * u_cam_dist;
    vec3 fw = normalize(-ro);
    vec3 rt = normalize(cross(vec3(0.0, 1.0, 0.0), fw));
    vec3 up = cross(fw, rt);
    vec3 rd = normalize(fw + uv.x * rt * 1.3 + uv.y * up * 1.3);

    float d = 0.0;
    float hit = 0.0;
    vec3 pos = ro;
    float orbit = 0.0;
    int steps = 0;
    for (int i = 0; i < 160; i++) {
        pos = ro + rd * d;
        float o;
        float ds = mandelboxDE(pos, scale, o);
        orbit = o;
        if (ds < 0.0008) { hit = 1.0; break; }
        d += ds;
        steps = i;
        if (d > 12.0) break;
    }

    vec3 col = u_bg;
    if (hit > 0.5) {
        vec3 n = mandelboxNormal(pos, scale);
        vec3 ld = normalize(vec3(0.6, 0.8, 0.4));
        float dif = clamp(dot(n, ld), 0.0, 1.0);
        float spe = pow(clamp(dot(reflect(-ld, n), -rd), 0.0, 1.0), 24.0);
        vec3 base = mix(u_base_color, u_glow_color, orbit);
        col = base * (0.2 + 0.8 * dif) + vec3(1.0) * spe * u_spec;
        col += u_glow_color * (1.0 - orbit) * 0.4;  // inner-trap glow
        float ao = 1.0 - float(steps) / 160.0 * 0.4;  // step-count fake AO
        col *= ao;
    }
    col = pow(clamp(col, 0.0, 1.0), vec3(0.4545));  // gamma
    f_color = vec4(col, 1.0);
}
''', uniforms={
    "scale":      {"glsl": "float", "min": -2.5, "max": 3.0, "default": -1.5,
                   "description": "fractal scale (negative = iconic tiled-rooms look)"},
    "fold":       {"glsl": "float", "min": 0.1, "max": 2.0, "default": 1.0,
                   "description": "box-fold clamp half-width"},
    "min_radius": {"glsl": "float", "min": 0.1, "max": 1.5, "default": 0.5,
                   "description": "sphere-fold inner radius"},
    "fixed_radius": {"glsl": "float", "min": 0.3, "max": 2.0, "default": 1.0,
                     "description": "sphere-fold outer radius"},
    "iterations": {"glsl": "float", "min": 4.0, "max": 24.0, "default": 14.0,
                   "description": "escape-time iteration cap (detail)"},
    "cam_dist":   {"glsl": "float", "min": 2.0, "max": 12.0, "default": 6.0,
                   "description": "camera distance from the box"},
    "cam_angle":  {"glsl": "float", "min": 0.0, "max": 1.0, "default": 0.0,
                   "description": "static camera azimuth offset (turns)"},
    "anim_speed": {"glsl": "float", "min": 0.0, "max": 4.0, "default": 1.0,
                   "description": "camera orbit + scale-breathing speed"},
    "spec":       {"glsl": "float", "min": 0.0, "max": 1.5, "default": 0.6,
                   "description": "specular highlight strength"},
    "base_color": {"glsl": "color", "default": "#e8c07a",
                   "description": "surface base colour"},
    "glow_color": {"glsl": "color", "default": "#3aa0ff",
                   "description": "orbit-trap glow colour"},
    "bg":         {"glsl": "color", "default": "#05060a",
                   "description": "background colour"},
})


_register("domain_warp_palette_gpu",
          "Domain warping — fbm(fbm(p + fbm(p))) feedback (Inigo Quilez, 2015) with "
          "a 4-colour palette; two-level noise feed-forward produces marbled, organic "
          "flow; animated by scrolling the inner warp offset with u_time",
          "procedural", '''float _warpfbm(vec2 p) {
    float v = 0.0, a = 0.5;
    for (int i = 0; i < 6; i++) {
        v += a * noise(p); p *= 2.02; a *= 0.5;
    }
    return v;
}
void main() {
    vec2 uv = (v_uv - 0.5);
    uv.x *= u_resolution.x / u_resolution.y;
    float t = u_time * u_speed;
    // Two-level domain warp: q = fbm(p + t), r = fbm(p + q + t), final = fbm(p + r).
    vec2 p = uv * u_scale + u_offset;
    vec2 q = vec2(_warpfbm(p + vec2(0.0, 0.0) + 0.15 * t),
                  _warpfbm(p + vec2(5.2, 1.3) - 0.10 * t));
    vec2 r = vec2(_warpfbm(p + u_warp * q + vec2(1.7, 9.2) + 0.12 * t),
                  _warpfbm(p + u_warp * q + vec2(8.3, 2.8) - 0.08 * t));
    float f = _warpfbm(p + u_warp * r);
    // Mix palette by the three warp layers for the classic iq marble look.
    vec3 col = mix(u_color_a, u_color_b, clamp(f * f * 2.0, 0.0, 1.0));
    col = mix(col, u_color_c, clamp(length(q), 0.0, 1.0));
    col = mix(col, u_color_d, clamp(r.x, 0.0, 1.0));
    col = u_color_bg + col * (0.25 + 0.6 * f);
    col = pow(clamp(col, 0.0, 1.0), vec3(0.4545));  // gamma
    f_color = vec4(col, 1.0);
}
''', uniforms={
    "scale":   {"glsl": "float", "min": 0.5, "max": 8.0, "default": 3.0,
                  "description": "pattern frequency (zoom into the warp field)"},
    "warp":    {"glsl": "float", "min": 0.0, "max": 4.0, "default": 1.6,
                  "description": "warp strength (feed-forward coupling)"},
    "speed":   {"glsl": "float", "min": 0.0, "max": 4.0, "default": 1.0,
                  "description": "animation scroll speed"},
    "offset":  {"glsl": "float", "min": 0.0, "max": 20.0, "default": 0.0,
                  "description": "static field offset into the warp field"},
    "color_a": {"glsl": "color", "default": "#1a2a6c",
                  "description": "base palette colour A"},
    "color_b": {"glsl": "color", "default": "#b21f1f",
                  "description": "palette colour B (f-weighted)"},
    "color_c": {"glsl": "color", "default": "#fdbb2d",
                  "description": "palette colour C (q-weighted)"},
    "color_d": {"glsl": "color", "default": "#16a085",
                  "description": "palette colour D (r.x-weighted)"},
    "color_bg": {"glsl": "color", "default": "#04060f",
                  "description": "background tint"},
})



# -- Categorical coverage: Gerstner Ocean (typed GPU node 352) --
# Closed-form trochoidal-wave ocean (Fournier & Reeves 1986; GPU Gems ch.1
# normal, Finch 2004) as a shaded height field with Blinn-Phong sun glitter.
# Mirrors the CPU node 963 (patterns/gerstner_ocean.py) which explicitly flags
# "a clean GPU twin is a natural follow-up". Genuinely time-varying (wave phases
# advance with u_time) so it survives the shootout contrast-only static cull.
# CPU numpy node stays authoritative for export (two-tier precision).
_register("gerstner_ocean_gpu",
          "Gerstner Ocean -- analytic trochoidal wave height field with sun "
          "glitter (typed GPU twin of node 963); genuinely animated (wave "
          "phases advance with u_time)",
          "procedural", """
void main() {
    vec2 uv = (v_uv - 0.5);
    uv.x *= u_resolution.x / u_resolution.y;
    float t = u_time * 0.6;
    float N = clamp(u_n_waves, 1.0, 24.0);
    float baseL = max(u_base_wavelength, 1e-3);
    float falloff = clamp(u_wavelength_falloff, 0.5, 0.95);
    float amp = u_amplitude;
    float Q = clamp(u_steepness, 0.0, 1.0);
    float windA = u_wind_angle * 6.2831853;
    float spread = u_wind_spread * 6.2831853;

    float z = 0.0;
    vec2 slope = vec2(0.0);
    float nz = 1.0;
    float L = baseL;
    float A = amp;
    float sumA = 0.0;
    for (float i = 0.0; i < 24.0; i += 1.0) {
        if (i >= N) break;
        float frac = (N > 1.0) ? (i / (N - 1.0) - 0.5) : 0.0;
        float ang = windA + frac * spread + hash21(vec2(i, 3.7)) * 0.4;
        vec2 D = vec2(cos(ang), sin(ang));
        float k = 6.2831853 / L;
        float w = sqrt(9.8 * k * 0.02);
        float phi = hash21(vec2(i, 11.3)) * 6.2831853;
        float f = k * dot(D, uv) - w * t + phi;
        z += A * sin(f);
        slope += D * (k * A * cos(f));
        nz -= Q * k * A * sin(f);
        sumA += A;
        L *= falloff;
        A *= falloff;
    }
    if (sumA > 0.0) z /= sumA;

    vec3 Nn = normalize(vec3(-slope, max(nz, 0.05)));
    float sunAz = u_sun_angle * 6.2831853;
    float sunEl = clamp(u_sun_height, 0.05, 1.0) * 1.5707963;
    vec3 sun = normalize(vec3(cos(sunAz) * cos(sunEl),
                              sin(sunAz) * cos(sunEl), sin(sunEl)));
    vec3 viewv = vec3(0.0, 0.0, 1.0);
    vec3 hv = normalize(sun + viewv);
    float diff = max(dot(Nn, sun), 0.0);
    float spec = pow(max(dot(Nn, hv), 0.0), max(u_shininess, 1.0)) * u_glint;

    float crest = clamp(z * 3.0 + 0.5, 0.0, 1.0);
    vec3 deep = 0.5 + 0.5 * cos(6.2831853 * (u_deep_hue + vec3(0.0, 0.15, 0.3)));
    deep *= vec3(0.25, 0.5, 0.7);
    vec3 crestc = 0.5 + 0.5 * cos(6.2831853 * (u_crest_hue + vec3(0.0, 0.1, 0.2)));
    crestc = mix(crestc, vec3(0.9, 0.95, 1.0), 0.5);
    vec3 col = mix(deep, crestc, crest);
    col = col * (0.25 + 0.9 * diff) + vec3(spec);
    col *= u_exposure;
    col = pow(clamp(col, 0.0, 1.0), vec3(1.0 / max(u_gamma, 1e-3)));
    f_color = vec4(col, 1.0);
}
""", uniforms={
    "n_waves":            {"glsl": "float", "min": 1.0, "max": 24.0, "default": 9.0, "description": "number of Gerstner wave components"},
    "base_wavelength":    {"glsl": "float", "min": 0.15, "max": 2.5, "default": 1.1, "description": "longest wavelength (screen units)"},
    "wavelength_falloff": {"glsl": "float", "min": 0.5, "max": 0.95, "default": 0.82, "description": "per-wave wavelength scale (<1 shortens)"},
    "amplitude":          {"glsl": "float", "min": 0.02, "max": 0.6, "default": 0.16, "description": "master wave amplitude"},
    "steepness":          {"glsl": "float", "min": 0.0, "max": 1.0, "default": 0.7, "description": "trochoid steepness Q (0=round, 1=sharp crests)"},
    "wind_angle":         {"glsl": "float", "min": 0.0, "max": 1.0, "default": 0.12, "description": "dominant wind direction (turns)"},
    "wind_spread":        {"glsl": "float", "min": 0.0, "max": 0.4, "default": 0.14, "description": "angular spread of wave directions (turns)"},
    "sun_angle":          {"glsl": "float", "min": 0.0, "max": 1.0, "default": 0.62, "description": "sun azimuth (turns)"},
    "sun_height":         {"glsl": "float", "min": 0.05, "max": 1.0, "default": 0.35, "description": "sun elevation (0=horizon, 1=zenith)"},
    "shininess":          {"glsl": "float", "min": 4.0, "max": 400.0, "default": 90.0, "description": "specular sharpness of the sun glitter"},
    "glint":              {"glsl": "float", "min": 0.0, "max": 4.0, "default": 1.6, "description": "sun-glitter intensity"},
    "deep_hue":           {"glsl": "float", "min": 0.0, "max": 1.0, "default": 0.53, "description": "deep-water base hue"},
    "crest_hue":          {"glsl": "float", "min": 0.0, "max": 1.0, "default": 0.5, "description": "crest/foam hue"},
    "exposure":           {"glsl": "float", "min": 0.2, "max": 3.0, "default": 1.1, "description": "overall brightness multiplier"},
    "gamma":              {"glsl": "float", "min": 0.3, "max": 2.5, "default": 1.0, "description": "tonal gamma"},
})


# -- Categorical coverage: Gyroid TPMS (typed GPU node 360) --
# Closed-form triply-periodic minimal-surface field (Schoen 1970 gyroid,
# Schwarz 1890 P/D, Neovius, Schoen I-WP) evaluated on a swept slice plane.
# Mirrors CPU node 964 (patterns/gyroid_tpms.py); the third coordinate z is
# animated with u_time so the 2D cross-section morphs continuously as the plane
# sweeps through the 3D volume -> genuinely time-varying, survives the shootout
# contrast-only static liveness cull. CPU numpy node 964 stays authoritative
# for export (two-tier precision). Inferno colormap for the default view.
_register("gyroid_tpms_gpu",
          "Gyroid TPMS -- closed-form triply-periodic minimal-surface shell "
          "on a swept slice plane (typed GPU twin of node 964); genuinely "
          "animated (slice z advances with u_time)",
          "procedural", _INFERNO + '''
void main() {
    vec2 p = (v_uv - 0.5) * 2.0 * 3.14159265 * max(u_freq, 1e-3);
    float t = u_time * 0.6;
    // Optional lattice rotation.
    float rot_ang = u_rotate * t * 0.5;
    float ca = cos(rot_ang), sa = sin(rot_ang);
    p = vec2(p.x * ca - p.y * sa, p.x * sa + p.y * ca);
    // Domain warp (animated so warp>0 stays alive).
    float wstr = u_warp;
    if (wstr > 0.0) {
        p.x += wstr * sin(p.y * 0.5 + t);
        p.y += wstr * cos(p.x * 0.5 + t * 0.8);
    }
    float z = t;   // swept slice plane
    float sx = sin(p.x), cx = cos(p.x);
    float sy = sin(p.y), cy = cos(p.y);
    float sz = sin(z),   cz = cos(z);

    int surf = int(clamp(u_surface + 0.5, 0.0, 4.0));
    float g, gmax;
    if (surf == 1) {            // schwarz_p
        g = cx + cy + cz; gmax = 3.0;
    } else if (surf == 2) {     // diamond (Schwarz D)
        g = sx*sy*sz + sx*cy*cz + cx*sy*cz + cx*cy*sz; gmax = 1.5;
    } else if (surf == 3) {     // neovius
        g = 3.0*(cx+cy+cz) + 4.0*cx*cy*cz; gmax = 3.0;
    } else if (surf == 4) {     // I-WP (Schoen)
        g = 2.0*(cx*cy + cy*cz + cz*cx) - (cos(2.0*p.x)+cos(2.0*p.y)+cos(2.0*z));
        gmax = 3.0;
    } else {                    // gyroid (Schoen)
        g = sx*cy + sy*cz + sz*cx; gmax = 1.5;
    }
    float gn = g / gmax;        // ~[-1,1]
    float lvl = u_level / gmax;

    float val;
    if (u_shell > 0.5) {
        float d = abs(gn - lvl);
        val = clamp(1.0 - d / max(u_thickness, 1e-6), 0.0, 1.0);
    } else {
        val = clamp((gn - lvl) * 0.5 + 0.5, 0.0, 1.0);
    }
    val = clamp(0.5 + (val - 0.5) * u_contrast, 0.0, 1.0);
    vec3 col = inferno(val);
    f_color = vec4(col, 1.0);
}
''', uniforms={
    "surface":   {"glsl": "float", "min": 0.0, "max": 4.0, "default": 0.0, "description": "minimal surface: 0 gyroid, 1 schwarz_p, 2 diamond, 3 neovius, 4 iwp"},
    "freq":      {"glsl": "float", "min": 1.0, "max": 16.0, "default": 5.0, "description": "spatial frequency (cells across canvas)"},
    "level":     {"glsl": "float", "min": -1.5, "max": 1.5, "default": 0.0, "description": "iso-level of the surface (shifts shell in/out)"},
    "thickness": {"glsl": "float", "min": 0.02, "max": 0.8, "default": 0.22, "description": "shell half-thickness of the surface band"},
    "warp":      {"glsl": "float", "min": 0.0, "max": 1.5, "default": 0.0, "description": "domain-warp strength (organic distortion)"},
    "contrast":  {"glsl": "float", "min": 0.5, "max": 3.0, "default": 1.2, "description": "final tone contrast"},
    "shell":     {"glsl": "float", "min": 0.0, "max": 1.0, "default": 1.0, "description": "render mode: 0 smooth field, 1 shell band"},
    "rotate":    {"glsl": "float", "min": 0.0, "max": 1.0, "default": 0.0, "description": "lattice rotation amount (0 off, 1 spin with time)"},
})

# ── Node 361: Phasor Noise (Tricard et al. 2019) typed GPU twin of CPU node
# 1006. Sums complex Gaussian-windowed phasors over a 3x3 jittered-grid
# neighbourhood, extracts the ARGUMENT (constant-amplitude phase field), then
# feeds it through a periodic profile — reproducing the crisp fingerprint/
# wood-grain stripes of phasor noise. Closed-form f(uv,t): the global phase
# advances with u_time so the live preview genuinely animates (survives the
# shootout contrast-only static-liveness cull). CPU numpy node 1006 stays
# authoritative for export (two-tier precision).
_register("phasor_noise_gpu",
          "Phasor Noise -- sparse-convolution complex-phasor field with a "
          "periodic profile (typed GPU twin of node 1006); anisotropic "
          "fingerprint/wood-grain stripes; genuinely animated (global phase "
          "advances with u_time)",
          "procedural", _INFERNO + '''
// per-cell random impulse: orientation angle, phase offset, frequency jitter
vec3 cell_impulse(vec2 cell) {
    float a = hash21(cell + 0.13) * 6.28318530718;          // orientation
    float psi = hash21(cell + 3.71) * 6.28318530718;        // phase offset
    float fj = 0.75 + 0.5 * hash21(cell + 7.42);            // freq jitter
    return vec3(a, psi, fj);
}

void main() {
    // Map uv into grid space; u_scale = cells across the canvas.
    float cells = max(2.0, u_scale);
    vec2 gp = v_uv * cells;
    vec2 base = floor(gp);

    float t = u_time * 0.6;

    // Accumulate the complex phasor sum over a 3x3 neighbourhood.
    vec2 acc = vec2(0.0);
    for (int dy = -1; dy <= 1; dy++) {
        for (int dx = -1; dx <= 1; dx++) {
            vec2 cell = base + vec2(float(dx), float(dy));
            // jittered impulse centre inside the cell
            vec2 jit = vec2(hash21(cell + 1.7), hash21(cell + 9.3));
            vec2 ctr = cell + jit;
            vec3 imp = cell_impulse(cell);
            float ang = imp.x;
            // optional domain warp bends orientation toward a smooth flow
            ang += u_anisotropy * (noise(cell * 0.5 + t * 0.1) - 0.5) * 3.14159265;
            float psi = imp.y;
            float fj = imp.z;

            vec2 d = gp - ctr;
            // oriented coordinate
            float ca = cos(ang), sa = sin(ang);
            float u = d.x * ca + d.y * sa;
            float vv = -d.x * sa + d.y * ca;
            // anisotropic Gaussian envelope (bandwidth <- u_falloff)
            float bw = max(0.2, u_falloff);
            float aM = bw;
            float am = bw / (1.0 + u_anisotropy * 4.0);
            float env = exp(-3.14159265 * (aM*aM*u*u + am*am*vv*vv));
            // windowed unit oscillation (frequency <- u_frequency)
            float theta = 6.28318530718 * u_frequency * fj * u + psi;
            acc += env * vec2(cos(theta), sin(theta));
        }
    }

    // phasor field = ARGUMENT of the complex sum (constant amplitude)
    float phase = atan(acc.y, acc.x);
    float ph = phase + (u_animate > 0.5 ? t : 0.0);

    // periodic profile
    int prof = int(clamp(u_profile + 0.5, 0.0, 3.0));
    float val;
    if (prof == 1) {              // sawtooth
        val = fract(ph / 6.28318530718 + 0.5);
    } else if (prof == 2) {       // square (smooth)
        float k = max(1.0, u_sharpness * 12.0);
        val = 0.5 + 0.5 * tanh(k * sin(ph));
    } else if (prof == 3) {       // triangle
        float saw = fract(ph / 6.28318530718 + 0.5);
        val = 1.0 - abs(2.0 * saw - 1.0);
    } else {                      // sine
        val = 0.5 + 0.5 * sin(ph);
    }

    // coherence fade where the phasor magnitude is tiny (phase undefined)
    float mag = length(acc);
    float coh = clamp(mag / 0.6, 0.0, 1.0);
    val = 0.5 + (val - 0.5) * coh;
    val = clamp(0.5 + (val - 0.5) * u_contrast, 0.0, 1.0);

    vec3 col = inferno(val);
    f_color = vec4(col, 1.0);
}
''', uniforms={
    "scale":      {"glsl": "float", "min": 2.0, "max": 40.0, "default": 12.0, "description": "grain feature count (cells across the canvas)"},
    "anisotropy": {"glsl": "float", "min": 0.0, "max": 1.0, "default": 0.8, "description": "0 isotropic, 1 strongly directional streaks"},
    "falloff":    {"glsl": "float", "min": 0.3, "max": 1.6, "default": 0.7, "description": "Gaussian envelope bandwidth (higher = more compact kernels)"},
    "frequency":  {"glsl": "float", "min": 0.2, "max": 3.0, "default": 1.2, "description": "base stripe frequency (cycles per feature)"},
    "profile":    {"glsl": "float", "min": 0.0, "max": 3.0, "default": 0.0, "description": "periodic profile: 0 sine, 1 sawtooth, 2 square, 3 triangle"},
    "sharpness":  {"glsl": "float", "min": 0.1, "max": 2.0, "default": 0.6, "description": "edge sharpness for the square profile"},
    "contrast":   {"glsl": "float", "min": 0.5, "max": 3.0, "default": 1.2, "description": "final tone contrast"},
    "animate":    {"glsl": "float", "min": 0.0, "max": 1.0, "default": 1.0, "description": "advance global phase with time (0 static, 1 glide)"},
})


# ── 471 Nishita Atmospheric Sky (client-GPU twin of node 471) ─────────────────
# Faithful GLSL port of the CPU node's single-scattering Nishita model
# (Nishita et al. 1993): for each pixel we march the view ray through a
# spherical shell atmosphere accumulating in-scattered sunlight from Rayleigh
# (molecular, blue) + Mie (aerosol, sun halo) scattering, with a second march
# toward the sun for optical depth (horizon reddening). This is O(W·H) with a
# small fixed sample budget = the same cheap closed-form f(uv,t) profile as the
# CPU node, so it is an honest live-preview twin (CPU numpy node 471 stays the
# authoritative export source). All numeric params are REAL node-471 params,
# wired by name via the typed param_map (contract #5). No string/choice params
# (anim_mode) are mapped — pitfall #14.
_register("nishita_sky_gpu", "Nishita atmospheric scattering sky (client-GPU twin of node 471)",
          "procedural", '''
const float R_GROUND = 6360e3;
const float R_TOP    = 6420e3;
const float H_R      = 7994.0;
const float H_M      = 1200.0;
const float PI       = 3.14159265;
const vec3  BETA_R   = vec3(5.8e-6, 13.5e-6, 33.1e-6);
const vec3  BETA_M   = vec3(21.0e-6, 21.0e-6, 21.0e-6);
const float MIE_G    = 0.76;
const float SUN_I    = 20.0;

// ray vs sphere centered at origin; returns (t_near, t_far)
vec2 _rsph(vec3 ro, vec3 rd, float R) {
    float b = 2.0 * dot(ro, rd);
    float c = dot(ro, ro) - R * R;
    float disc = max(b * b - 4.0 * c, 0.0);
    float sq = sqrt(disc);
    return vec2((-b - sq) * 0.5, (-b + sq) * 0.5);
}

// in-scatter along one view ray for a normalized sun direction sd.
vec3 _sky(vec3 rd, vec3 sd, float rk, float mk, float exposure,
          float fov, float disk_deg) {
    vec3 ro = vec3(0.0, R_GROUND + 1000.0, 0.0);
    vec2 a = _rsph(ro, rd, R_TOP);
    vec2 g = _rsph(ro, rd, R_GROUND);
    float ground = step(0.0, g.x);
    float t_max = (ground > 0.5) ? min(a.y, g.x) : a.y;
    t_max = max(t_max, 0.0);

    float ct = dot(rd, sd);
    float phase_r = 3.0 / (16.0 * PI) * (1.0 + ct * ct);
    float denom = (2.0 + MIE_G * MIE_G) * (1.0 + MIE_G * ct) * (1.0 + MIE_G * ct);
    float phase_m = 3.0 / (8.0 * PI) * (1.0 - MIE_G * MIE_G) * (1.0 + ct * ct)
                    / max(denom, 1e-8);

    float ns = 16.0, nsl = 8.0;
    float seg = t_max / ns;
    vec3 sum_r = vec3(0.0), sum_m = vec3(0.0);
    vec3 last_atten = vec3(1.0);
    for (int i = 0; i < 16; i++) {
        float t = (float(i) + 0.5) * seg;
        vec3 p = ro + t * rd;
        float hgt = length(p) - R_GROUND;
        float hr = exp(-hgt / H_R) * seg;
        float hm = exp(-hgt / H_M) * seg;
        // light march toward sun
        float tl = max(_rsph(p, sd, R_TOP).y, 0.0) / nsl;
        float tl_i = 0.5 * tl;
        float odlr = 0.0, odlm = 0.0;
        for (int j = 0; j < 8; j++) {
            vec3 pl = p + tl_i * sd;
            float hl = length(pl) - R_GROUND;
            odlr += exp(-hl / H_R) * tl;
            odlm += exp(-hl / H_M) * tl;
            tl_i += tl;
        }
        vec3 tau = BETA_R * (hr + odlr * rk) + BETA_M * (hm + odlm * 1.1 * mk);
        vec3 atten = exp(-tau);
        last_atten = atten;
        sum_r += atten * hr * BETA_R * rk * phase_r;
        sum_m += atten * hm * BETA_M * mk * phase_m;
    }
    vec3 col = (sum_r + sum_m) * SUN_I;
    float ang = acos(clamp(ct, -1.0, 1.0));
    float r_disk = radians(disk_deg);
    float disk = clamp(1.0 - ang / r_disk, 0.0, 1.0);
    float halo = clamp(1.0 - ang / (r_disk * 4.0), 0.0, 1.0) * 0.3;
    float glow = disk + halo;
    float trans = (last_atten.r + last_atten.g + last_atten.b) / 3.0;
    vec3 sun_tint = vec3(1.0, 0.95, 0.85);
    col += glow * trans * sun_tint * 20.0;
    col = 1.0 - exp(-col * exposure);
    return clamp(col, 0.0, 1.0);
}

void main() {
    // pixel -> view dir (camera looks down -Z, y up), fov in degrees
    vec2 uv = v_uv * 2.0 - 1.0;
    float aspect = u_resolution.x / u_resolution.y;
    float f = tan(radians(u_fov) * 0.5);
    vec3 rd = normalize(vec3(uv.x * aspect * f, uv.y * f, -1.0));
    // sun dir from elevation/azimuth (degrees); time glides elevation
    float el = radians(u_sun_elevation + u_time * 30.0);
    float az = radians(u_sun_azimuth);
    vec3 sd = vec3(cos(el) * sin(az), sin(el), -cos(el) * cos(az));
    rd = normalize(rd);
    sd = normalize(sd);
    vec3 col = _sky(rd, sd, u_rayleigh_k, u_mie_k, u_exposure, u_fov, u_sun_disk_radius);
    f_color = vec4(col, 1.0);
}
''', uniforms={
    "sun_elevation":   {"glsl": "float", "min": -10.0, "max": 90.0, "default": 6.0,
                        "description": "sun elevation (deg); time glides it for preview"},
    "sun_azimuth":     {"glsl": "float", "min": 0.0, "max": 360.0, "default": 90.0,
                        "description": "sun azimuth (deg)"},
    "rayleigh_k":      {"glsl": "float", "min": 0.2, "max": 3.0, "default": 1.0,
                        "description": "Rayleigh scattering strength"},
    "mie_k":           {"glsl": "float", "min": 0.0, "max": 3.0, "default": 1.0,
                        "description": "Mie (aerosol/haze) strength"},
    "exposure":        {"glsl": "float", "min": 0.1, "max": 3.0, "default": 1.0,
                        "description": "tonemap exposure"},
    "fov":             {"glsl": "float", "min": 20.0, "max": 120.0, "default": 60.0,
                        "description": "vertical field-of-view (deg)"},
    "sun_disk_radius": {"glsl": "float", "min": 0.3, "max": 5.0, "default": 1.2,
                        "description": "apparent sun disk radius (deg)"},
})

# ══════════════════════════════════════════════════════════════════════════
#  CPU-node closed-form filter twins (Route 0 / GPU-First gap mirror)
#  Live-preview GPU twins for filter nodes whose CPU algorithm is a faithful
#  per-pixel f(uv, input) operator. CPU node stays authoritative for export;
#  these drive the client-side live preview. Choice params (blur_type /
#  source / tint / palette / anim_mode / combine / output / n_orientations)
#  are intentionally omitted from param_map — the twins animate continuously
#  from u_time so the preview is always live, and the CPU export honours the
#  exact choices. Helper functions are the prologue's (rot / hash21 / fbm);
#  `step` is the prologue-reserved vec2 so the bodies use `mix`/manual compares
#  (never the `step()` builtin). Animation uses cos(u_time)/linear terms so the
#  t=0 vs t=π audit is never a sin-phase false negative.
# ══════════════════════════════════════════════════════════════════════════

# ── 486 Radial & Spin Blur (client-GPU twin) ──
_register("radial_spin_blur_gpu",
          "Radial & Spin Blur (client-GPU twin of node 486)",
          "filter", _filter_typed('''
    // Motion-blur kernel: average samples laid along a radial (zoom) and
    // rotational (spin) path about a pivot. Continuous spin + a cos breathe
    // keep the live preview animated (no sin-phase 0/pi degeneracy).
    int n = int(u_length) + 1;
    n = clamp(n, 2, 32);
    vec2 ctr = vec2(u_center_x, u_center_y);
    vec2 p = uv - ctr;
    float maxr = max(length(p * u_resolution), 1.0);
    float breathe = 1.0 + 0.3 * cos(u_time * u_anim_speed);
    float ang = u_time * u_anim_speed * 0.5;
    vec3 acc = vec3(0.0);
    for (int i = 0; i < 32; i++) {
        if (i >= n) break;
        float f = (float(i) / float(n - 1)) - 0.5;          // -0.5 .. 0.5
        float disp = (u_length / maxr) * f * breathe;
        vec2 q = ctr + p * (1.0 - disp);                    // radial zoom
        vec2 qr = ctr + rot(ang) * p;                       // rotational spin
        vec2 samp = mix(q, qr, 0.35);
        acc += sample(clamp(samp, 0.0, 1.0)).rgb;
    }
    acc /= float(n);
    f_color = vec4(acc, 1.0);
'''), uniforms={
    "length":     {"glsl": "float", "min": 0.0, "max": 64.0, "default": 14.0,
                   "description": "blur strength in px (edge displacement)"},
    "center_x":   {"glsl": "float", "min": 0.0, "max": 1.0, "default": 0.5,
                   "description": "blur pivot x (0-1)"},
    "center_y":   {"glsl": "float", "min": 0.0, "max": 1.0, "default": 0.5,
                   "description": "blur pivot y (0-1)"},
    "anim_speed": {"glsl": "float", "min": 0.1, "max": 5.0, "default": 1.0,
                   "description": "animation speed"},
})

# ── 438 Subsurface Scatter / SSSS (client-GPU twin) ──
_register("ssss_gpu",
          "Subsurface Scatter SSSS (client-GPU twin of node 438)",
          "filter", _filter_typed('''
    // Separable exponential-profile blur (Jimenez & Gutierrez 2010): a sharp
    // core term + a broad halo term, two 1-D passes (X then Y). Strength
    // breathes with cos(u_time) so the preview is live.
    int N = int(u_samples);
    N = clamp(N, 4, 25);
    float ext = max(1e-3, u_radius * 3.0);
    float stp = ext / float(max(N, 1));
    float cs = max(1e-3, u_radius * max(0.1, u_falloff));
    float invCs = 1.0 / cs;
    float invCs2 = 1.0 / (cs * 2.5);
    float strength = clamp(u_strength * (0.7 + 0.3 * cos(u_time * u_anim_speed)), 0.0, 1.0);
    vec3 accx = orig.rgb; float wsumx = 1.0;
    for (int i = 0; i < 25; i++) {
        if (i >= N) break;
        float off = (float(i) + 0.5) * stp;
        float w = exp(-off * invCs) + 0.5 * exp(-off * invCs2);
        vec2 d = vec2(off, 0.0) / u_resolution;
        accx += w * (sample(clamp(uv + d, 0.0, 1.0)).rgb + sample(clamp(uv - d, 0.0, 1.0)).rgb);
        wsumx += 2.0 * w;
    }
    accx /= wsumx;
    vec3 accy = accx; float wsumy = 1.0;
    for (int i = 0; i < 25; i++) {
        if (i >= N) break;
        float off = (float(i) + 0.5) * stp;
        float w = exp(-off * invCs) + 0.5 * exp(-off * invCs2);
        vec2 d = vec2(0.0, off) / u_resolution;
        accy += w * (sample(clamp(uv + d, 0.0, 1.0)).rgb + sample(clamp(uv - d, 0.0, 1.0)).rgb);
        wsumy += 2.0 * w;
    }
    accy /= wsumy;
    vec3 outc = mix(orig.rgb, accy, strength);
    f_color = vec4(outc, 1.0);
'''), uniforms={
    "radius":     {"glsl": "float", "min": 2.0, "max": 60.0, "default": 18.0,
                   "description": "scatter radius in px"},
    "samples":    {"glsl": "float", "min": 4.0, "max": 25.0, "default": 11.0,
                   "description": "profile samples per axis"},
    "falloff":    {"glsl": "float", "min": 0.2, "max": 6.0, "default": 1.4,
                   "description": "profile sharpness"},
    "strength":   {"glsl": "float", "min": 0.0, "max": 1.0, "default": 0.85,
                   "description": "subsurface blend amount"},
    "anim_speed": {"glsl": "float", "min": 0.1, "max": 5.0, "default": 1.0,
                   "description": "animation speed"},
})

# ── 439 Gabor Filter (client-GPU twin) ──
_register("gabor_filter_gpu",
          "Gabor Filter (client-GPU twin of node 439)",
          "filter", _filter_typed('''
    // Single-orientation Gabor kernel response (magnitude). The filter bank +
    // hue-vs-energy output modes of the CPU node are dropped; this twin shows
    // the energy magnitude of one Gabor at the chosen orientation. Orientation
    // rotates with u_time so the live preview is animated.
    int hk = int(clamp(u_sigma * 3.0 / max(u_aspect, 0.2), 3.0, 15.0));
    float theta = u_orientation + u_time * 0.3 * u_anim_speed;
    float ct = cos(theta), st = sin(theta);
    float sigma2 = 2.0 * u_sigma * u_sigma;
    vec3 acc = vec3(0.0);
    vec3 wsum = vec3(0.0);
    for (int y = -15; y <= 15; y++) {
        if (abs(y) > hk) break;
        for (int x = -15; x <= 15; x++) {
            if (abs(x) > hk) break;
            vec2 d = vec2(float(x), float(y)) / u_resolution;
            vec3 s = sample(clamp(uv + d, 0.0, 1.0)).rgb;
            float xr = float(x) * ct + float(y) * st;
            float yr = -float(x) * st + float(y) * ct;
            float env = exp(-(xr * xr + (u_aspect * yr) * (u_aspect * yr)) / sigma2);
            float k = env * cos(6.2831853 * u_frequency * xr + u_phase);
            acc += s * k;
            wsum += vec3(k);
        }
    }
    vec3 resp = acc / max(abs(wsum), vec3(1e-3));
    float mag = clamp(length(resp) * u_contrast, 0.0, 1.0);
    f_color = vec4(vec3(mag), 1.0);
'''), uniforms={
    "orientation": {"glsl": "float", "min": 0.0, "max": 3.14159, "default": 0.0,
                    "description": "filter orientation (rad)"},
    "frequency":   {"glsl": "float", "min": 0.02, "max": 0.5, "default": 0.12,
                    "description": "Gabor spatial frequency (cycles/px)"},
    "sigma":       {"glsl": "float", "min": 2.0, "max": 24.0, "default": 8.0,
                    "description": "Gaussian envelope std (px)"},
    "aspect":      {"glsl": "float", "min": 0.2, "max": 1.0, "default": 0.5,
                    "description": "envelope elongation gamma"},
    "phase":       {"glsl": "float", "min": 0.0, "max": 6.28318, "default": 0.0,
                    "description": "sinusoid phase (rad)"},
    "contrast":    {"glsl": "float", "min": 0.5, "max": 3.0, "default": 1.0,
                    "description": "response contrast boost"},
    "anim_speed":  {"glsl": "float", "min": 0.1, "max": 3.0, "default": 1.0,
                    "description": "animation speed"},
})


# ── 522 CRT Emulation (client-GPU twin) ──
_register("crt_emulation_gpu",
          "CRT Emulation (client-GPU twin of node 522)",
          "filter", _filter_typed('''
    // Barrel distortion (curvature), aperture-grille mask, scanlines, edge
    // vignette, RGB chroma shift, and a u_time-driven vertical roll + brightness
    // flicker so the live preview is animated (cos term, not sin, to avoid the
    // 0/pi phase degeneracy).
    vec2 p = uv * 2.0 - 1.0;
    float r2 = dot(p, p);
    vec2 quv = (p * (1.0 + u_curvature * r2)) * 0.5 + 0.5;
    quv.y = fract(quv.y + u_time * 0.05 * u_roll_speed);
    vec3 col;
    col.r = sample(clamp(quv + vec2(u_chroma * 0.01 * (quv.x - 0.5), 0.0), 0.0, 1.0)).r;
    col.g = sample(clamp(quv, 0.0, 1.0)).g;
    col.b = sample(clamp(quv - vec2(u_chroma * 0.01 * (quv.x - 0.5), 0.0), 0.0, 1.0)).b;
    float scan = 0.5 + 0.5 * sin(quv.y * u_resolution.y * 0.5 * u_scan_freq);
    col *= 1.0 - u_scanline * (1.0 - scan);
    float m = 0.5 + 0.5 * cos(quv.x * u_resolution.x * 1.04719755);
    col *= 1.0 - u_mask_strength * (1.0 - m);
    col *= 1.0 - u_vignette * r2;
    col *= u_brightness * (1.0 - u_flicker * (0.5 + 0.5 * cos(u_time * 7.0)));
    f_color = vec4(clamp(col, 0.0, 1.0), 1.0);
'''), uniforms={
    "curvature":  {"glsl": "float", "min": 0.0, "max": 0.45, "default": 0.18, "description": "barrel distortion amount"},
    "scanline":   {"glsl": "float", "min": 0.0, "max": 1.0,  "default": 0.35, "description": "scanline darkness"},
    "scan_freq":  {"glsl": "float", "min": 1.0, "max": 8.0,  "default": 2.5,  "description": "scanline frequency"},
    "mask_strength": {"glsl": "float", "min": 0.0, "max": 1.0,  "default": 0.35, "description": "aperture-grille mask strength"},
    "vignette":   {"glsl": "float", "min": 0.0, "max": 1.0,  "default": 0.3,  "description": "edge vignette"},
    "chroma":     {"glsl": "float", "min": 0.0, "max": 1.0,  "default": 0.25, "description": "RGB chroma shift"},
    "roll_speed": {"glsl": "float", "min": 0.0, "max": 3.0,  "default": 1.0,  "description": "vertical roll speed"},
    "flicker":    {"glsl": "float", "min": 0.0, "max": 0.3,  "default": 0.06, "description": "brightness flicker"},
    "brightness": {"glsl": "float", "min": 0.4, "max": 2.0,  "default": 1.1,  "description": "overall brightness"},
})

# ── 527 VHS Tape (client-GPU twin) ──
_register("vhs_tape_gpu",
          "VHS Tape (client-GPU twin of node 527)",
          "filter", _filter_typed('''
    // Chroma smear + per-line chroma shift, horizontal line jitter and a
    // tracking-band distortion driven by u_time, luma noise, and
    // saturation/contrast/brightness grading. cos/linear temporal terms (not
    // sin) keep the live preview honest.
    float n = hash21(vec2(uv.x * 100.0, floor(uv.y * u_resolution.y)));
    float jit = (hash21(vec2(floor(uv.y * 40.0), u_time)) - 0.5) * u_line_jitter * 0.05;
    vec2 quv = vec2(uv.x + jit + (uv.y - 0.5) * u_skew * 0.15,
                    fract(uv.y + u_time * 0.02 * u_roll_speed));
    float off = u_chroma_smear * 0.02 + u_chroma_shift * 0.0005 * sin(uv.y * 80.0 + u_time);
    vec3 col;
    col.r = sample(clamp(quv + vec2(off, 0.0), 0.0, 1.0)).r;
    col.g = sample(clamp(quv, 0.0, 1.0)).g;
    col.b = sample(clamp(quv - vec2(off, 0.0), 0.0, 1.0)).b;
    col += (n - 0.5) * u_luma_noise;
    float track = smoothstep(0.0, 0.08,
        abs(fract(uv.y - u_time * 0.1 * u_roll_speed) - 0.5) - (0.45 - 0.1 * u_tracking));
    col *= 1.0 - track * 0.5;
    float l = dot(col, vec3(0.299, 0.587, 0.114));
    col = (col - l) * u_saturation + l;
    col = (col - 0.5) * u_contrast + 0.5;
    col *= u_brightness;
    f_color = vec4(clamp(col, 0.0, 1.0), 1.0);
'''), uniforms={
    "chroma_smear": {"glsl": "float", "min": 0.0, "max": 1.0,  "default": 0.55, "description": "horizontal chroma smear"},
    "chroma_shift": {"glsl": "float", "min": 0.0, "max": 24.0, "default": 8.0,  "description": "per-line chroma offset"},
    "luma_noise":   {"glsl": "float", "min": 0.0, "max": 0.6,  "default": 0.12, "description": "luma noise amount"},
    "line_jitter":  {"glsl": "float", "min": 0.0, "max": 1.0,  "default": 0.45, "description": "horizontal line jitter"},
    "tracking":     {"glsl": "float", "min": 0.0, "max": 1.0,  "default": 0.5,  "description": "tracking band strength"},
    "roll_speed":     {"glsl": "float", "min": 0.0, "max": 3.0,  "default": 1.0,  "description": "vertical roll speed"},
    "skew":         {"glsl": "float", "min": 0.0, "max": 1.0,  "default": 0.35, "description": "tape skew"},
    "saturation":   {"glsl": "float", "min": 0.0, "max": 2.0,  "default": 1.25, "description": "saturation"},
    "contrast":     {"glsl": "float", "min": 0.3, "max": 2.0,  "default": 1.1,  "description": "contrast"},
    "brightness":   {"glsl": "float", "min": 0.4, "max": 2.0,  "default": 1.05, "description": "brightness"},
})

# ── 445 Diffraction Grating (client-GPU twin) ──
# Faithful closed-form preview of node 445's per-pixel Stam / GPU-Gems
# diffraction formula: groove tangent g (default concentric CD/record layout),
# half-vector H = normalize(L+V), grating vector G = g−(g·H)H, then
# w = λ0·(g·H) + a·|G| and intensity = Σ cos(2π·frac(w/D)) summed over
# λ=650/550/450 nm → iridescent R/G/B. CPU node stays authoritative for export;
# this drives the client-side live preview. The groove is the default concentric
# layout — `source`/`palette`/`anim_mode` are CPU-only choices (pitfall #14) so
# the preview shows the canonical concentric sheen. Numeric params
# (groove_spacing/curvature/interp/light_x/light_y/strength/saturation) are wired
# by name. The preview animates continuously from u_time (light rotates + curvature
# breathes); cos terms so the t=0 vs t=pi audit is never a sin-phase false negative.
_register("diffraction_gpu",
          "Diffraction Grating Iridescence (client-GPU twin of node 445)",
          "filter", _filter_typed('''
    vec3 srgb = orig.rgb;
    float mx = max(u_resolution.x, u_resolution.y);
    vec2 p = (uv - 0.5) * u_resolution / mx;          // centred, aspect-correct
    // concentric groove tangent (tangent to the radial direction)
    float a0 = atan(p.y, p.x);
    vec2 g = vec2(cos(a0 + 1.5707963), sin(a0 + 1.5707963));
    // view direction (curved disc; breathes with time)
    float curv = u_curvature * (1.0 + 0.35 * cos(u_time * 0.5));
    vec3 V = normalize(vec3(p.x * curv, p.y * curv, 1.0));
    // light direction (rotates with time for a live, always-on preview)
    float llen = length(vec2(u_light_x, u_light_y)) + 1e-4;
    float la = atan(u_light_y, u_light_x) + u_time;
    vec2 Lxy = vec2(cos(la), sin(la)) * llen;
    vec3 L = normalize(vec3(Lxy, 1.0));
    vec3 Hh = normalize(L + V);                       // half vector
    float gdotH = dot(g, Hh.xy);
    vec3 G = vec3(g - gdotH * Hh.xy, -gdotH * Hh.z);   // grating vector
    float Gmag = length(G);
    // per-wavelength interference (pulse mode modulates the groove period)
    float D = u_groove_spacing * (1.0 + 0.4 * cos(u_time * 0.5));
    float lam[3]; lam[0] = 650.0; lam[1] = 550.0; lam[2] = 450.0;
    vec3 iri = vec3(0.0);
    for (int i = 0; i < 3; i++) {
        float wi = lam[i] * gdotH + u_interp * Gmag;
        float fr = fract(wi / D);
        float t = 6.2831853 * fr;
        float a = (1.0 - cos(t)) * 0.5;
        float b = (1.0 - cos(2.0 * t)) * 0.5;
        float c = (1.0 - cos(3.0 * t)) * 0.5;
        iri[i] = (a + b + c);
    }
    iri = clamp(iri, 0.0, 1.0);
    // saturation control (luminance-preserving)
    float lum_i = dot(iri, vec3(0.3333333));
    iri = clamp(lum_i + u_saturation * (iri - lum_i), 0.0, 1.0);
    // composite over the source
    vec3 outc = mix(srgb, iri, u_strength);
    f_color = vec4(clamp(outc, 0.0, 1.0), 1.0);
'''), uniforms={
    "groove_spacing": {"glsl": "float", "min": 400.0, "max": 3000.0, "default": 1300.0, "description": "grating period D (nm)"},
    "curvature":      {"glsl": "float", "min": 0.0, "max": 2.0, "default": 0.8, "description": "view tilt across frame"},
    "interp":         {"glsl": "float", "min": 0.0, "max": 1.0, "default": 0.5, "description": "spectral-order interpolation"},
    "light_x":        {"glsl": "float", "min": -1.0, "max": 1.0, "default": 0.0, "description": "incident light x"},
    "light_y":        {"glsl": "float", "min": -1.0, "max": 1.0, "default": 0.3, "description": "incident light y"},
    "strength":       {"glsl": "float", "min": 0.0, "max": 1.0, "default": 1.0, "description": "iridescent overlay blend"},
    "saturation":     {"glsl": "float", "min": 0.0, "max": 1.5, "default": 1.0, "description": "band colour saturation"},
})

# ── 489 Film Grain (client-GPU twin) ──
# Luminance-adaptive emulsion grain — closed-form preview of node 489. Per-pixel
# hash grain, shadow-weighted by luminance (real film grain reads stronger in
# shadows). CPU node is authoritative for export; this is the live-preview path.
# `color`/`source`/`palette`/`anim_mode` are CPU-only choices (pitfall #14) — the
# preview animates the grain field with u_time (flicker) so it stays live and
# is_time_varying is honest (no sin-phase degeneracy). intensity/adapt/grain_size
# are wired by name to the shader's u_<name> uniforms (typed-uniform contract).
_register("film_grain_gpu",
          "Film Grain (client-GPU twin of node 489)",
          "filter", _filter_typed('''
    vec3 srgb = orig.rgb;
    // blocky grain: quantize by grain_size, hash per block (+u_time = flicker)
    vec2 cell = floor(uv * u_resolution / max(1.0, u_grain_size));
    float r1 = hash21(cell + vec2(0.123, 0.0) + u_time * 13.0);
    float r2 = hash21(cell + vec2(7.77, 3.33) + u_time * 17.0);
    float r3 = hash21(cell + vec2(2.22, 9.99) + u_time * 11.0);
    vec3 grain = vec3(r1, r2, r3) * 2.0 - 1.0;          // [-1,1]
    float lum = dot(srgb, vec3(0.299, 0.587, 0.114));
    float k = u_intensity * (1.0 + 2.0 * u_adapt * (1.0 - lum));
    vec3 outc = clamp(srgb + grain * k, 0.0, 1.0);
    f_color = vec4(clamp(outc, 0.0, 1.0), 1.0);
'''), uniforms={
    "intensity":  {"glsl": "float", "min": 0.0, "max": 0.6, "default": 0.12, "description": "grain strength / ISO-like amount"},
    "adapt":      {"glsl": "float", "min": 0.0, "max": 1.0, "default": 0.7, "description": "shadow-weighting of grain"},
    "grain_size": {"glsl": "float", "min": 1.0, "max": 8.0, "default": 1.0, "description": "grain pixel scale (chunkier if >1)"},
})


# ══════════════════════════════════════════════════════════════════════════
#  CPU-node closed-form procedural twins (Route 0 / GPU-First gap mirror)
#  Live-preview GPU twins for three pattern/math_art nodes whose CPU algorithm
#  is a faithful per-pixel closed-form f(uv, t) generator with NO close
#  existing twin. CPU node stays authoritative for export; these drive the
#  client-side live preview. Every numeric CPU param becomes a named u_<name>
#  uniform/SCALAR port (typed-uniform contract). Choice params (palette / mode
#  / pattern / color_mode) are dropped to GPU_PREVIEW_DROP_ALLOW — the twins
#  animate continuously from u_time so the preview is always live, and the CPU
#  export honours the exact choices. Palettes are inlined (no late-helper
#  _INFERNO ordering pitfall). Animation uses cos()/linear terms so the
#  t=0 vs t=pi audit is never a sin-phase false negative.
# ══════════════════════════════════════════════════════════════════════════

# ── 995 Gravitational Lensing / Einstein Ring (client-GPU twin) ──
_register("grav_lens_gpu",
          "Gravitational Lensing Einstein Ring (client-GPU twin of node 995)",
          "procedural",
'''void main() {
    // Thin-lens deflection: beta = theta * (1 - thetaE^2/|theta|^2). Sample a
    // procedural sky (fbm nebula + hashed stars) at the mapped source position
    // and brighten by the magnification mu. Continuous drift keeps it live.
    vec2 uv = (gl_FragCoord.xy - 0.5 * u_resolution) / min(u_resolution.x, u_resolution.y);
    uv *= 2.2;                                    // world scale ~ CPU node
    float t = u_time * u_anim_speed;

    float tE = u_einstein_radius;
    // drift the source in a slow circle (matches CPU 'drift' mode)
    vec2 p = uv + 0.25 * vec2(sin(t), cos(t));
    // breathe the lens mass a touch (cos, no 0/pi degeneracy)
    tE *= 0.85 + 0.15 * cos(t * 0.7);

    float r2 = dot(p, p) + 1e-4;
    float inv = (tE * tE) / r2;
    vec2 beta = p * (1.0 - inv);                  // source position
    float mu = 1.0 / abs(1.0 - inv * inv);
    mu = clamp(mu, 1.0, 8.0);

    // procedural sky at beta
    vec2 sky_uv = beta * (2.0 + u_neb_scale * 0.6);
    float neb = pow(clamp(fbm(sky_uv + vec2(0.0, t * 0.05)) , 0.0, 1.0), 1.6) * u_nebula;
    float star = smoothstep(1.0 - u_star_density * 40.0, 1.0, hash21(floor(beta * 240.0)));
    star += 0.5 * smoothstep(0.985, 1.0, hash21(floor(beta * 90.0 + 7.0)));

    // palette (cosmic default, inlined)
    vec3 neb_rgb = vec3(0.32, 0.22, 0.55);
    vec3 star_rgb = vec3(0.80, 0.86, 1.0);
    vec3 sky = neb_rgb * neb + star_rgb * star;

    float rr = sqrt(r2);
    float ring = exp(-((rr - tE) * (rr - tE)) / (2.0 * u_ring_width * u_ring_width));
    vec3 glow = star_rgb * ring * u_ring_brightness;

    vec3 col = clamp(sky * mu + glow, 0.0, 1.0);
    col = clamp(col * u_exposure, 0.0, 1.0);
    f_color = vec4(col, 1.0);
}
''',
uniforms={
    "einstein_radius": {"glsl": "float", "min": 0.05, "max": 0.9, "default": 0.35, "description": "Einstein radius (lens mass, ring size)"},
    "star_density":    {"glsl": "float", "min": 0.0005, "max": 0.02, "default": 0.004, "description": "fraction of pixels that are stars"},
    "nebula":          {"glsl": "float", "min": 0.0, "max": 1.0, "default": 0.5, "description": "nebula cloud intensity"},
    "neb_scale":       {"glsl": "float", "min": 1.0, "max": 8.0, "default": 3.0, "description": "nebula fbm frequency"},
    "exposure":        {"glsl": "float", "min": 0.2, "max": 4.0, "default": 1.4, "description": "output brightness multiplier"},
    "ring_brightness": {"glsl": "float", "min": 0.0, "max": 3.0, "default": 1.2, "description": "Einstein-ring glow strength"},
    "ring_width":      {"glsl": "float", "min": 0.01, "max": 0.3, "default": 0.06, "description": "Einstein-ring glow width"},
    "anim_speed":      {"glsl": "float", "min": 0.1, "max": 5.0, "default": 1.0, "description": "animation speed"},
})

# ── 950 SDF Scene (client-GPU twin) ──
_register("sdf_scene_gpu",
          "SDF Scene (client-GPU twin of node 950)",
          "procedural",
'''float sd_circle(vec2 p, float r) { return length(p) - r; }
float sd_box(vec2 p, vec2 b) {
    vec2 d = abs(p) - b;
    return length(max(d, 0.0)) + min(max(d.x, d.y), 0.0);
}
float sd_ring(vec2 p, float r, float th) { return abs(length(p) - r) - th; }
float smin_p(float a, float b, float k) {
    float h = clamp(0.5 + 0.5 * (b - a) / k, 0.0, 1.0);
    return mix(b, a, h) - k * h * (1.0 - h);
}
void main() {
    vec2 uv = (gl_FragCoord.xy - 0.5 * u_resolution) / min(u_resolution.x, u_resolution.y);
    uv *= 2.0 * u_scale;
    float t = u_time * u_anim_speed;

    // rotate + drift (closed-form, live). cos/sin drift, angle=t.
    vec2 p = uv + 0.30 * vec2(sin(t), cos(t * 0.7));
    p = rot(t * 0.5) * p;

    // domain repetition (tiling)
    if (u_repetition > 1e-4) {
        float rep = max(1e-3, u_repetition);
        p = mod(p + 0.5 * rep, rep) - 0.5 * rep;
    }
    float k = max(1e-3, u_blend);
    float dc = sd_circle(p, 0.16);
    float db = sd_box(p, vec2(0.20));
    float dr = sd_ring(p, 0.34, 0.022);
    float d = smin_p(smin_p(dc, db, k), dr, k);

    // shading from the field
    vec3 bg = vec3(0.03, 0.02, 0.05);
    vec3 ink = vec3(0.98, 0.78, 0.36);
    float edge = 0.014;
    float inside = clamp(0.5 - d / (2.0 * edge), 0.0, 1.0);
    float glow_eff = u_glow * (1.0 + 0.5 * cos(t));
    float glow_f = exp(-3.0 * max(d, 0.0)) * glow_eff;
    float band_f = 0.5 + 0.5 * sin(d * max(0.0, u_bands) - t);
    float band_factor = (1.0 - u_band_mix) + u_band_mix * band_f;

    vec3 col = mix(bg, ink, inside) + ink * glow_f;
    col *= band_factor;
    f_color = vec4(clamp(col, 0.0, 1.0), 1.0);
}
''',
uniforms={
    "scale":      {"glsl": "float", "min": 0.5, "max": 4.0, "default": 1.6, "description": "scene zoom / world scale"},
    "blend":      {"glsl": "float", "min": 0.01, "max": 0.6, "default": 0.18, "description": "smooth-min blend softness"},
    "repetition": {"glsl": "float", "min": 0.0, "max": 1.5, "default": 0.0, "description": "domain-repetition cell size (0=off)"},
    "glow":       {"glsl": "float", "min": 0.0, "max": 2.0, "default": 0.8, "description": "outside halo strength"},
    "bands":      {"glsl": "float", "min": 0.0, "max": 40.0, "default": 12.0, "description": "distance isoline count"},
    "band_mix":   {"glsl": "float", "min": 0.0, "max": 1.0, "default": 0.5, "description": "contour-band modulation amount"},
    "anim_speed": {"glsl": "float", "min": 0.1, "max": 5.0, "default": 1.0, "description": "animation speed"},
})

# ── 967 Interior Mapping (client-GPU twin) ──
_register("interior_mapping_gpu",
          "Interior Mapping (client-GPU twin of node 967)",
          "procedural",
'''void main() {
    // Fake 3D rooms behind a flat facade (van Dongen 2008): per-pixel ray-box
    // intersection into a tiled window grid. Continuous camera pan keeps it live.
    vec2 gu = gl_FragCoord.xy / u_resolution;
    float t = u_time * u_anim_speed;

    float pan_x = u_pan_x + 0.6 * sin(t);
    float pan_y = u_pan_y + 0.25 * cos(t * 0.7);
    float persp = u_perspective;

    vec2 f = gu * vec2(u_n_cols, u_n_rows);
    vec2 ci = floor(f);
    vec2 lxy = (f - ci) - 0.5;                    // local window coord [-0.5,0.5]
    float lx = lxy.x, ly = lxy.y;

    bool in_frame = (abs(lx) > (0.5 - u_frame_width)) || (abs(ly) > (0.5 - u_frame_width));

    // per-window hashed room params
    float rd = hash21(ci + 1.0);
    float depth = u_room_depth * (0.75 + 0.5 * rd);
    float wall_h = hash21(ci + 2.0);
    float lit_h = hash21(ci + 3.0);
    // flicker lit set over time
    float flick = 0.5 + 0.5 * sin(t * (0.6 + lit_h * 2.0) + wall_h * 6.2831853);
    float lit_val = clamp(lit_h * 0.5 + flick * 0.5, 0.0, 1.0);
    bool lit = lit_val < u_lit_fraction;

    // cast interior ray, intersect room box
    float dz = 1.0;
    float dx = lx * persp + pan_x;
    float dy = ly * persp + pan_y;
    float eps = 1e-6;
    float dxs = abs(dx) < eps ? eps : dx;
    float dys = abs(dy) < eps ? eps : dy;
    float tz = depth / dz;
    float tx = dx > 0.0 ? (0.5 - lx) / dxs : (-0.5 - lx) / dxs;
    float ty = dy > 0.0 ? (0.5 - ly) / dys : (-0.5 - ly) / dys;
    tx = tx > 0.0 ? tx : 1e9;
    ty = ty > 0.0 ? ty : 1e9;
    float t_hit = min(min(tz, tx), ty);
    bool hit_back = (tz <= tx) && (tz <= ty);
    bool hit_side = (tx < tz) && (tx <= ty);
    bool hit_ud = (ty < tz) && (ty < tx);

    float hx = lx + dx * t_hit;
    float hy = ly + dy * t_hit;
    float hz = dz * t_hit;

    // base wall colour warm vs cool
    vec3 warm = vec3(0.62, 0.50, 0.38);
    vec3 cool = vec3(0.40, 0.45, 0.52);
    vec3 base = warm * u_warmth + cool * (1.0 - u_warmth);
    vec3 rgb = base * (0.75 + 0.5 * wall_h);

    float shade = 1.0;
    if (hit_side) shade = 0.82;
    if (hit_ud && hy > 0.0) shade = 1.05;
    if (hit_ud && hy <= 0.0) shade = 0.62;
    rgb *= shade;

    // back-wall picture detail
    float bx = hx + 0.5, by = hy + 0.5;
    if (hit_back && abs(bx - 0.5) < 0.22 && abs(by - 0.55) < 0.16)
        rgb = vec3(0.20, 0.28, 0.42);

    // depth attenuation
    float dnorm = clamp(hz / (u_room_depth * 1.3), 0.0, 1.0);
    rgb *= (1.0 - 0.55 * dnorm);

    // ceiling light glow for lit rooms
    float gl = exp(-((hx * hx) / 0.12 + ((hy - 0.35) * (hy - 0.35)) / 0.10));
    if (lit) rgb += vec3(1.0, 0.92, 0.72) * (gl * 0.9);
    rgb *= lit ? 1.0 : 0.30;
    if (!lit) rgb *= vec3(0.7, 0.8, 1.0);

    // faint sky reflection
    rgb += vec3(0.10, 0.14, 0.22) * (1.0 - gu.y) * 0.12;

    // facade mullions
    if (in_frame) rgb = vec3(0.14, 0.14, 0.16);

    f_color = vec4(clamp(rgb, 0.0, 1.0), 1.0);
}
''',
uniforms={
    "n_cols":      {"glsl": "float", "min": 1.0, "max": 24.0, "default": 8.0, "description": "window columns across facade"},
    "n_rows":      {"glsl": "float", "min": 1.0, "max": 24.0, "default": 6.0, "description": "window rows down facade"},
    "room_depth":  {"glsl": "float", "min": 0.4, "max": 3.0, "default": 1.4, "description": "virtual room depth"},
    "perspective": {"glsl": "float", "min": 0.2, "max": 2.5, "default": 1.1, "description": "parallax strength"},
    "pan_x":       {"glsl": "float", "min": -1.0, "max": 1.0, "default": 0.0, "description": "horizontal camera offset"},
    "pan_y":       {"glsl": "float", "min": -1.0, "max": 1.0, "default": 0.15, "description": "vertical camera offset"},
    "frame_width": {"glsl": "float", "min": 0.0, "max": 0.25, "default": 0.06, "description": "facade mullion thickness"},
    "lit_fraction":{"glsl": "float", "min": 0.0, "max": 1.0, "default": 0.6, "description": "fraction of lit windows"},
    "warmth":      {"glsl": "float", "min": 0.0, "max": 1.0, "default": 0.5, "description": "room colour warmth"},
    "anim_speed":  {"glsl": "float", "min": 0.1, "max": 5.0, "default": 1.0, "description": "animation speed"},
})

# ── IFS Fractal attractor (node 353 twin) ──────────────────────────────────────
# Per-pixel chaos game: orbit start seeded by pixel+time, 30-50 iterations
# through a preset's affine maps, final position deposited as soft density.
# Named typed uniforms mirror the CPU node's numeric params; preset/coloring/
# anim_mode are choice strings (pitfall #14) dropped for GPU preview.
_register("ifs_fractal_gpu",
          "IFS Fractal attractor — chaos game (client-GPU twin of node 353)",
          "procedural", '''vec3 ifspal(float t, float shift) {
    t = clamp(t + shift, 0.0, 1.0);
    return 0.5 + 0.5 * cos(6.2831853 * (t + vec3(0.0, 0.33, 0.67)));
}

vec2 hash22in(vec2 p) {
    float a = hash21(p), b = hash21(p + vec2(1.0, 0.0));
    return vec2(a, b);
}

vec2 ifs_map(vec2 z, int preset, int ch) {
    float h3 = 0.3333333, h6 = 0.1666667;
    if (preset == 0) { // sierpinski
        if (ch == 0) return vec2(h3*z.x, h3*z.y);
        else if (ch == 1) return vec2(h3*z.x + h3, h3*z.y);
        else return vec2(h3*z.x + h6, h3*z.y + 0.4330127);
    } else if (preset == 1) { // dragon
        float s = 0.353553;
        if (ch == 0) return vec2(s*(z.x - z.y), s*(z.x + z.y));
        else return vec2(-s*(z.x + z.y) + 1.0, s*(z.x - z.y));
    } else { // snowflake
        float a = float(ch) * 1.0471976;
        float c = cos(a), sn = sin(a);
        return vec2(h3*(c*z.x - sn*z.y) + 0.5, h3*(sn*z.x + c*z.y) + 0.5);
    }
}

void main() {
    vec2 uv = v_uv;
    int preset = int(u_preset + 0.5);
    float density = 0.0;
    int N = int(u_points * 0.0004 + 8.0);
    if (N > 60) N = 60;
    float t_rot = u_time * 0.25;
    float ca = cos(t_rot), sa = sin(t_rot);

    for (int orbit = 0; orbit < 8; orbit++) {
        vec2 z = hash22in(vec2(float(orbit)*1.7+0.5, float(orbit)*3.1+0.3));
        z = z * 2.0 - 1.0;
        int nmaps = (preset == 1) ? 2 : 6;
        for (int i = 0; i < 40; i++) {
            float h = fract(sin(dot(z + float(i)*0.17, vec2(127.1, 311.7))) * 43758.5453);
            z = ifs_map(z, preset, int(h * float(nmaps)));
        }
        float ox = z.x - 0.5, oy = z.y - 0.5;
        vec2 sp = vec2(ca*ox - sa*oy, sa*ox + ca*oy) + 0.5;
        density += exp(-length(uv - sp) * 22.0);
    }
    density = clamp(density / 8.0, 0.0, 1.0);
    f_color = vec4(ifspal(density * 1.3, u_hue_shift), 1.0);
}
''', uniforms={
    "preset": {"glsl": "choice", "choices": ["sierpinski", "dragon", "snowflake"], "default": "sierpinski", "description": "IFS preset family"},
    "points": {"glsl": "float", "min": 10000.0, "max": 500000.0, "default": 120000.0, "description": "chaos-game iterations (density proxy)"},
    "hue_shift": {"glsl": "float", "min": 0.0, "max": 1.0, "default": 0.0, "description": "hue rotation"},
    "anim_speed": {"glsl": "float", "min": 0.1, "max": 5.0, "default": 1.0, "description": "animation speed multiplier"},
})

# ── Symmetric Icon attractor (node 416 twin) ───────────────────────────────────
# Orbits the Field & Golubitsky symmetric-chaos map per pixel, deposits density.
# Named typed uniforms mirror the CPU node's numeric params; colormode/palette/
# source/anim_mode are choice strings (pitfall #14) dropped for preview.
_register("symmetric_icon_gpu",
          "Symmetric Icon attractor — Field & Golubitsky (client-GPU twin of node 416)",
          "procedural", '''vec3 iconpal(float t, float shift) {
    t = clamp(t + shift, 0.0, 1.0);
    return 0.5 + 0.5 * cos(6.2831853 * (t + vec3(0.0, 0.33, 0.67)));
}

vec2 cpow(vec2 z, int n) {
    vec2 r = vec2(1.0, 0.0);
    for (int i = 0; i < 10; i++) {
        if (i >= n) break;
        r = vec2(r.x*z.x - r.y*z.y, r.x*z.y + r.y*z.x);
    }
    return r;
}

vec2 hash22in(vec2 p) {
    float a = hash21(p), b = hash21(p + vec2(1.0, 0.0));
    return vec2(a, b);
}

void main() {
    int sym = int(u_symmetry + 0.5);
    vec2 uv = v_uv;
    float density = 0.0;
    float t_rot = u_time * 0.3;
    float time_warp = 1.0 + 0.15 * sin(u_time * 0.7);

    for (int orbit = 0; orbit < 8; orbit++) {
        vec2 z = hash22in(vec2(float(orbit)*1.3+0.5, float(orbit)*2.9+0.3));
        z = z * 1.4 - 0.7;
        for (int i = 0; i < 32; i++) {
            float z2 = dot(z, z);
            vec2 zn = cpow(z, sym);
            vec2 coeff = vec2(u_a0 + u_a1*z2 + u_a2*zn.x, u_a3);
            vec2 conj_pow = cpow(vec2(z.x, -z.y), sym - 1);
            z = coeff * z + u_a4 * conj_pow;
        }
        // subtle time-warp of position (anim_mode=evolve analogue)
        z = vec2(z.x + 0.003 * sin(u_time * 1.1 + float(orbit)), z.y);
        float ox = z.x * 0.7 + 0.5;
        float oy = z.y * 0.7 + 0.5;
        float c = cos(t_rot), s = sin(t_rot);
        vec2 sp = vec2(c*(ox-0.5) - s*(oy-0.5) + 0.5, s*(ox-0.5) + c*(oy-0.5) + 0.5);
        density += exp(-length(uv - sp) * 20.0);
    }
    density = clamp(density / 8.0, 0.0, 1.0);
    float v = density * time_warp;
    f_color = vec4(iconpal(v * 1.1, u_palette_shift), 1.0);
}
''', uniforms={
    "symmetry": {"glsl": "int", "min": 2, "max": 9, "default": 6, "description": "rotational symmetry order n"},
    "a0": {"glsl": "float", "min": -3.0, "max": 3.0, "default": -2.0, "description": "real constant term"},
    "a1": {"glsl": "float", "min": -3.0, "max": 3.0, "default": 1.5, "description": "modulus |z|^2 coupling"},
    "a2": {"glsl": "float", "min": -3.0, "max": 3.0, "default": -0.1, "description": "Re(z^n) coupling"},
    "a3": {"glsl": "float", "min": -1.5, "max": 1.5, "default": 0.0, "description": "imaginary term (chiral twist)"},
    "a4": {"glsl": "float", "min": -1.5, "max": 1.5, "default": 0.6, "description": "conjugate z^(n-1) coupling"},
    "palette_shift": {"glsl": "float", "min": 0.0, "max": 1.0, "default": 0.5, "description": "cosine palette hue offset"},
    "anim_speed": {"glsl": "float", "min": 0.1, "max": 3.0, "default": 1.0, "description": "animation speed multiplier"},
    "seed_strength": {"glsl": "float", "min": 0.0, "max": 1.0, "default": 0.6, "description": "blend with wired luminance"},
})
