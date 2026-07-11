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

_register("mandelbrot", "Mandelbrot set zoom region", "procedural", '''
void main() {
    vec2 uv = (gl_FragCoord.xy - 0.5 * u_resolution) / min(u_resolution.x, u_resolution.y);
    float zoom = exp(u_params.x * 3.0);
    vec2 c = vec2(-0.5, 0.0) + uv * zoom;
    vec2 z = vec2(0.0);
    int n = 0;
    for (int i = 0; i < 100; i++) {
        z = vec2(z.x*z.x - z.y*z.y, 2.0*z.x*z.y) + c;
        if (dot(z, z) > 4.0) break;
        n++;
    }
    float t = float(n) / 100.0;
    f_color = vec4(0.5 + 0.5 * cos(t * 6.28 + vec3(0, 2, 4)), 1.0);
}
''')

_register("julia", "Julia set fractal", "procedural", '''
void main() {
    vec2 uv = (gl_FragCoord.xy - 0.5 * u_resolution) / min(u_resolution.x, u_resolution.y);
    vec2 c = vec2(-0.7269 + (u_params.x - 0.5) * 0.4, 0.1889 + (u_params.y - 0.5) * 0.4);
    vec2 z = uv * exp((u_params.z - 0.5) * 3.0) * 3.0;  // 0.5 -> full view (±1.5)
    int n = 0;
    float last2 = 0.0;
    const float MAXI = 200.0;
    for (int i = 0; i < 200; i++) {
        z = vec2(z.x*z.x - z.y*z.y, 2.0*z.x*z.y) + c;
        last2 = dot(z, z);
        if (last2 > 16.0) break;
        n++;
    }
    float t = (n >= MAXI - 0.5) ? 0.0 : clamp((n + 1.0 - log(max(log(last2)*0.5, 1.0001))/log(2.0)) / MAXI, 0.0, 1.0);
    f_color = vec4(0.5 + 0.5 * cos(t * 6.28318 + vec3(0.0, 2.0, 4.0)), 1.0);
}
''')

_register("plasma", "Multi-octave colored plasma", "procedural", '''
void main() {
    vec2 uv = v_uv;
    float t = u_time * 0.1;
    float v = sin(uv.x * 8.0 + t) * cos(uv.y * 6.0 + t * 0.7);
    v += sin(uv.x * 16.0 - t * 1.2) * cos(uv.y * 12.0 + t * 0.5) * 0.5;
    v += sin((uv.x + uv.y) * 24.0 + t * 0.3) * 0.25;
    v = v * 0.5 + 0.5;
    f_color = vec4(0.5 + 0.5 * cos(v * 6.28 + vec3(0, 2, 4)), 1.0);
}
''')

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
    // p1 = zoom (0.5 = full view), p2 = color_shift, p3 = center_x, p4 = center_y.
    float zoom = u_zoom;
    vec2 ctr = vec2(u_center_x, u_center_y);
    vec2 c = ctr + uv * zoom;
    vec2 z = vec2(0.0);
    float n = 0.0;
    float last2 = 0.0;
    const float MAXI = 200.0;
    for (int i = 0; i < 200; i++) {
        z = vec2(z.x*z.x - z.y*z.y, 2.0*z.x*z.y) + c;
        last2 = dot(z, z);
        if (last2 > 16.0) break;
        n += 1.0;
    }
    float t = (n >= MAXI - 0.5) ? 0.0 : smooth_iter(n, last2, MAXI);
    f_color = vec4(fractal_palette(t + u_color_shift), 1.0);
}
''',
    uniforms={
    "zoom": {"glsl": "float", "min": 0.05, "max": 20.0, "default": 1.0, "description": "zoom (0.5=1x full view)"},
    "color_shift": {"glsl": "float", "min": 0.0, "max": 1.0, "default": 0.5, "description": "palette color offset"},
    "center_x": {"glsl": "float", "min": -1.5, "max": 1.5, "default": 0.5, "description": "center x"},
    "center_y": {"glsl": "float", "min": -1.5, "max": 1.5, "default": 0.5, "description": "center y"}
}
    )

_register("burning_ship_gpu", "Burning Ship fractal (client-GPU twin of node 51)", "procedural",
          _FRACTAL_HELPERS + '''
void main() {
    vec2 uv = (gl_FragCoord.xy - 0.5 * u_resolution) / min(u_resolution.x, u_resolution.y);
    // p1 = zoom (0.5 = full view), p2 = color_shift, p3 = center_x, p4 = center_y.
    float zoom = exp((u_color_speed - 0.5) * 6.0);
    vec2 ctr = vec2(0.5, 0.5);
    vec2 c = ctr + uv * zoom;
    vec2 z = vec2(0.0);
    float n = 0.0;
    float last2 = 0.0;
    const float MAXI = 200.0;
    for (int i = 0; i < 200; i++) {
        z = vec2(abs(z.x) - 1.0, abs(z.y)) * abs(z.x) + c; // abs-squared ship map
        z = vec2(z.x*z.x - z.y*z.y, 2.0*z.x*z.y);
        last2 = dot(z, z);
        if (last2 > 16.0) break;
        n += 1.0;
    }
    float t = (n >= MAXI - 0.5) ? 0.0 : smooth_iter(n, last2, MAXI);
    f_color = vec4(fractal_palette(t + u_color_offset), 1.0);
}
''',
    uniforms={
    "color_speed": {"glsl": "float", "min": 0.0, "max": 2.0, "default": 0.5, "description": "palette color speed"},
    "color_offset": {"glsl": "float", "min": 0.0, "max": 1.0, "default": 0.5, "description": "palette color offset"}
}
    )

_register("newton_gpu", "Newton fractal basins (client-GPU twin of node 52)", "procedural",
          _FRACTAL_HELPERS + '''
void main() {
    vec2 uv = (gl_FragCoord.xy - 0.5 * u_resolution) / min(u_resolution.x, u_resolution.y);
    // p1 = color_speed, p2 = color_offset, p3 = zoom (0.5 = full view), p4 = unused.
    vec2 z = uv * 2.2;
    const float MAXI = 60.0;
    float n = 0.0;
    for (int i = 0; i < 60; i++) {
        // Newton for z^3 - 1: z - (z^3 - 1) / (3 z^2)
        vec2 z2 = vec2(z.x*z.x - z.y*z.y, 2.0*z.x*z.y);
        vec2 z3 = vec2(z2.x*z.x - z2.y*z.y, 2.0*z2.x*z.y);
        vec2 f = z3 - vec2(1.0, 0.0);
        vec2 dz = 3.0 * z2;
        float denom = dz.x*dz.x + dz.y*dz.y + 1e-8;
        vec2 step = vec2(f.x*dz.x + f.y*dz.y, f.y*dz.x - f.x*dz.y) / denom;
        z -= step;
        n += 1.0;
        if (dot(step, step) < 1e-6) break;
    }
    // Color by nearest of the 3 cube roots of unity (angle quantization).
    float ang = atan(z.y, z.x);
    float root = floor((ang + 3.14159) / (2.0 * 3.14159 / 3.0));
    float t = mod(root / 3.0 + u_color_offset + 0.15 * n / MAXI, 1.0);
    f_color = vec4(fractal_palette(t * (0.6 + 0.4 * u_color_speed)), 1.0);
}
''',
    uniforms={
    "color_speed": {"glsl": "float", "min": 0.0, "max": 2.0, "default": 0.5, "description": "palette color speed"},
    "color_offset": {"glsl": "float", "min": 0.0, "max": 1.0, "default": 0.5, "description": "palette color offset"}
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
    // p1 = r_min, p2 = r_max, p3 = color_mode(0=lyapunov), p4 = color_shift.
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

_register("domain_warp", "Domain-warped fractal noise", "procedural", '''
void main() {
    vec2 uv = v_uv * 3.0;
    float t = u_time * 0.05;
    float warp = 2.0 + u_params.x * 3.0;
    vec2 q = vec2(fbm(uv + t), fbm(uv + vec2(5.2, 1.3) + t * 0.7));
    vec2 r = vec2(fbm(uv + warp * q + vec2(1.7, 9.2) + t * 0.3),
                  fbm(uv + warp * q + vec2(8.3, 2.8) + t * 0.4));
    float v = fbm(uv + warp * r);
    f_color = vec4(0.5 + 0.5 * cos(v * 6.28 + vec3(0, 2, 4)), 1.0);
}
''')

_register("voronoi", "Voronoi/Worley noise cells", "procedural", '''
void main() {
    vec2 uv = v_uv * (5.0 + u_params.x * 5.0);
    vec2 i = floor(uv); vec2 f = fract(uv);
    float md = 1.0;
    for (int y = -1; y <= 1; y++) {
        for (int x = -1; x <= 1; x++) {
            vec2 n = vec2(float(x), float(y));
            vec2 p = hash21(i + n) * vec2(1.0);
            float d = length(n + p - f);
            md = min(md, d);
        }
    }
    f_color = vec4(md, md * 0.5, 1.0 - md, 1.0);
}
''')

_register("voronoise", "Smooth voronoi layers", "procedural", '''
void main() {
    vec2 uv = v_uv * 4.0;
    float t = u_time * 0.02;
    vec2 q = vec2(fbm(uv + t), fbm(uv + vec2(3.7, 1.2) + t));
    vec2 r = vec2(fbm(uv + 4.0 * q + vec2(1.7, 9.2)),
                  fbm(uv + 4.0 * q + vec2(8.3, 2.8)));
    float v = fbm(uv + 4.0 * r);
    f_color = vec4(0.5 + 0.5 * cos(v * 4.0 + vec3(0, 2, 4)), 1.0);
}
''')

_register("ripples", "Concentric ripple pattern", "procedural", '''
void main() {
    vec2 uv = v_uv - 0.5;
    float d = length(uv);
    float r = sin(d * 30.0 - u_time * 2.0) * 0.5 + 0.5;
    float g = sin(d * 30.0 - u_time * 2.0 + 2.0) * 0.5 + 0.5;
    float b = sin(d * 30.0 - u_time * 2.0 + 4.0) * 0.5 + 0.5;
    f_color = vec4(r, g, b, 1.0) * (1.0 - d);
}
''')

_register("cells", "Cellular growth simulation", "procedural", '''
void main() {
    vec2 uv = v_uv * 8.0;
    vec2 i = floor(uv); vec2 f = fract(uv);
    float md = 8.0;
    vec2 mp = vec2(0.0);
    for (int y = -1; y <= 1; y++) {
        for (int x = -1; x <= 1; x++) {
            vec2 n = vec2(float(x), float(y));
            vec2 p = hash21(i + n) * vec2(1.0);
            float d = length(n + p - f);
            if (d < md) { md = d; mp = n + p; }
        }
    }
    float c = hash21(i + mp);
    vec3 col = 0.5 + 0.5 * cos(c * 6.28 + vec3(0, 2, 4));
    col *= 1.0 - md * 1.2;
    col += vec3(0.05) / (md * md + 0.01);
    f_color = vec4(col, 1.0);
}
''')

_register("bubble_chamber", "Simulated bubble chamber trails", "procedural", '''
void main() {
    vec2 uv = v_uv * 2.0 - 1.0;
    float t = u_time * 0.3;
    float v = 0.0;
    for (int i = 0; i < 20; i++) {
        float fi = float(i);
        vec2 p = vec2(sin(fi * 1.7 + t * 0.5), cos(fi * 2.3 + t * 0.7)) * 0.8;
        float d = length(uv - p) - 0.03;
        v += 0.005 / (d * d + 0.001);
    }
    f_color = vec4(v * 0.5, v * 0.8, v, 1.0);
}
''')

_register("stars", "Starfield with parallax", "procedural", '''
void main() {
    vec2 uv = v_uv;
    float t = u_time * 0.05;
    vec3 col = vec3(0.0);
    for (int i = 0; i < 50; i++) {
        float fi = float(i);
        vec2 p = fract(vec2(sin(fi * 127.1 + t), cos(fi * 311.7 + t * 0.7)));
        float d = length(uv - p);
        float brightness = 0.003 / (d * d);
        vec3 star_col = 0.5 + 0.5 * cos(fi + vec3(0, 2, 4));
        col += brightness * star_col;
    }
    f_color = vec4(col, 1.0);
}
''')

_register("lightning_fractal", "Fractal lightning branching", "procedural", '''
void main() {
    vec2 uv = v_uv * 2.0 - 1.0;
    float t = u_time * 0.2;
    vec2 p = vec2(0.0);
    float v = 0.0;
    for (int i = 0; i < 64; i++) {
        float fi = float(i);
        p += vec2(sin(fi * 0.3 + t), cos(fi * 0.7 + t * 0.5)) * 0.02;
        float d = length(uv - p);
        v += 0.02 / (d + 0.01);
    }
    f_color = vec4(v * 0.3, v * 0.5, v, 1.0);
}
''')

_register("spiral", "Logarithmic spiral galaxy", "procedural", '''
void main() {
    vec2 uv = v_uv - 0.5;
    float a = atan(uv.y, uv.x);
    float r = length(uv);
    float spiral = sin(a * 4.0 - r * 15.0 + u_time * 0.5) * 0.5 + 0.5;
    float fade = exp(-r * 3.0);
    float col = spiral * fade;
    f_color = vec4(col * 1.2, col * 0.8, col * fade + 0.1, 1.0);
}
''')

_register("dendritic", "Dendritic / tree-like branching", "procedural", '''
void main() {
    vec2 uv = v_uv * 2.0 - 1.0;
    float t = u_time * 0.1;
    float d = length(uv);
    float a = atan(uv.y, uv.x) * 3.0;
    float branch = sin(a * 8.0 + log(d + 0.001) * 10.0 + t) * 0.5 + 0.5;
    float v = branch * exp(-d * 2.0);
    f_color = vec4(v * 0.3, v * 0.6, v * 0.2, 1.0);
}
''')

_register("barnsley", "Barnsley fern approximation", "procedural", '''
void main() {
    vec2 uv = v_uv * 3.0 - 1.5;
    float t = u_time * 0.1;
    float v = 0.0;
    for (int i = 0; i < 100; i++) {
        float fi = float(i);
        vec2 p = vec2(sin(fi * 0.5 + t), cos(fi * 0.3 + t * 0.7));
        float dx = uv.x - p.x * 0.5;
        float dy = uv.y - p.y * 0.8 - 0.5;
        v += 0.001 / (dx*dx + dy*dy + 0.001);
    }
    f_color = vec4(v * 0.2, v * 0.8, v * 0.2, 1.0);
}
''')

_register("spectral", "Spectral / rainbow interference", "procedural", '''
void main() {
    vec2 uv = v_uv * 2.0 - 1.0;
    float t = u_time * 0.1;
    float a = atan(uv.y, uv.x);
    float r = length(uv);
    float v = sin(r * 20.0 - t) + cos(a * 5.0 + t * 0.5);
    v = v * 0.25 + 0.5;
    f_color = vec4(0.5 + 0.5 * cos(v * 6.28 + vec3(0, 2, 4)), 1.0);
}
''')

_register("truchet", "Truchet tile pattern", "procedural", '''
void main() {
    vec2 uv = v_uv * 6.0;
    vec2 i = floor(uv); vec2 f = fract(uv) - 0.5;
    float flip = hash21(i) > 0.5 ? 1.0 : -1.0;
    float d = length(f * flip);
    float v = smoothstep(0.4, 0.5, d);
    float c = hash21(i + vec2(1.0));
    vec3 col = mix(vec3(0.9, 0.9, 0.95), 0.5 + 0.5 * cos(c * 6.28 + vec3(0, 2, 4)), v);
    f_color = vec4(col, 1.0);
}
''')

_register("kaleidoscope_fractal", "Kaleidoscope IFS fractal", "procedural", '''
void main() {
    vec2 uv = v_uv * 2.0 - 1.0;
    float t = u_time * 0.1;
    for (int i = 0; i < 10; i++) {
        uv = abs(uv);
        float a = sin(t + float(i) * 0.5);
        uv = rot(a) * uv;
        uv = uv * 1.5 - vec2(0.5);
    }
    float v = length(uv);
    f_color = vec4(0.5 + 0.5 * cos(v * 10.0 + vec3(0, 2, 4)), 1.0);
}
''')

_register("waves_3d", "3D wave interference", "procedural", '''
void main() {
    vec2 uv = v_uv * 4.0 - 2.0;
    float t = u_time * 0.5;
    float v = 0.0;
    for (int i = 0; i < 10; i++) {
        float fi = float(i);
        vec2 p = vec2(sin(fi * 1.3 + t), cos(fi * 1.7 + t * 0.8));
        v += sin(dot(uv, p) * 3.0 + t) * 0.1;
    }
    vec3 col = 0.5 + 0.5 * cos(v * 4.0 + vec3(0, 2, 4));
    f_color = vec4(col, 1.0);
}
''')

_register("pixel_sort_gpu", "Edge-directed pixel sorting on GPU", "procedural", '''
void main() {
    vec2 uv = v_uv;
    float v = fbm(uv * 4.0 + u_time * 0.02);
    vec2 step = vec2(1.0 / u_resolution.x, 1.0 / u_resolution.y);
    float dx = fbm((uv + vec2(step.x, 0)) * 4.0) - v;
    float dy = fbm((uv + vec2(0, step.y)) * 4.0) - v;
    float edge = abs(dx) + abs(dy);
    float bands = floor(uv.x * 20.0 + v * 10.0) / 20.0 + v * 0.1;
    vec3 col = 0.5 + 0.5 * cos(bands * 6.28 + vec3(0, 2, 4));
    col = mix(col, vec3(0.1), edge * 5.0);
    f_color = vec4(col, 1.0);
}
''')

_register("ocean", "Procedural ocean waves", "procedural", '''
void main() {
    vec2 uv = v_uv * 3.0;
    float t = u_time * 0.3;
    float v = sin(uv.x * 5.0 + t) * cos(uv.y * 3.0 + t * 0.7);
    v += sin(uv.x * 8.0 - t * 1.3) * sin(uv.y * 6.0 + t) * 0.5;
    v += sin((uv.x + uv.y) * 12.0 + t * 0.5) * 0.25;
    v = v * 0.5 + 0.5;
    vec3 col = mix(vec3(0.0, 0.2, 0.5), vec3(0.1, 0.6, 0.8), v);
    col += vec3(0.3, 0.4, 0.5) * pow(v, 4.0);
    f_color = vec4(col, 1.0);
}
''')

_register("nebula_gpu", "Space nebula gas clouds", "procedural", '''
void main() {
    vec2 uv = v_uv * 2.0;
    float t = u_time * 0.03;
    vec2 q = vec2(fbm(uv + t), fbm(uv + vec2(5.2, 1.3) + t * 0.7));
    vec2 r = vec2(fbm(uv + 3.0 * q + vec2(1.7, 9.2) + t * 0.3),
                  fbm(uv + 3.0 * q + vec2(8.3, 2.8) + t * 0.4));
    float v = fbm(uv + 3.0 * r);
    float mask = 1.0 - abs(v_uv.y - 0.5) * 2.0;
    vec3 col = 0.3 + 0.7 * (0.5 + 0.5 * cos(v * 4.0 + vec3(0, 1, 2)));
    col *= mask;
    f_color = vec4(col, 1.0);
}
''')

_register("terrain", "Procedural terrain heightmap", "procedural", '''
void main() {
    vec2 uv = v_uv * 3.0;
    float t = u_time * 0.02;
    float h = fbm(uv + t);
    float h2 = fbm(uv * 2.0 + t * 1.5) * 0.5;
    float h3 = fbm(uv * 4.0 + t * 2.0) * 0.25;
    h = h * 0.6 + h2 * 0.3 + h3 * 0.1;
    vec3 col;
    if (h < 0.3) col = vec3(0.1, 0.3, 0.6);
    else if (h < 0.45) col = vec3(0.2, 0.5, 0.2);
    else if (h < 0.6) col = vec3(0.3, 0.3, 0.1);
    else if (h < 0.75) col = vec3(0.4, 0.25, 0.1);
    else col = vec3(0.8, 0.8, 0.9);
    float shade = 0.5 + 0.5 * cos(h * 20.0);
    f_color = vec4(col * shade, 1.0);
}
''')

_register("wood_grain_gpu", "Concentric wood grain rings", "procedural", '''
void main() {
    vec2 uv = v_uv - 0.5;
    float d = length(uv) * 10.0;
    float grain = sin(d * 8.0 + fbm(uv * 10.0) * 0.5) * 0.5 + 0.5;
    vec3 col = mix(vec3(0.3, 0.15, 0.05), vec3(0.6, 0.3, 0.1), grain);
    f_color = vec4(col, 1.0);
}
''')

_register("fire_gpu", "Animated fire/flame", "procedural", '''
void main() {
    vec2 uv = v_uv;
    float t = u_time * 0.5;
    float v = fbm(vec2(uv.x * 3.0, (1.0 - uv.y) * 5.0 + t));
    v = v * (1.0 - uv.y);
    vec3 col = mix(vec3(1.0, 0.9, 0.4), vec3(0.8, 0.2, 0.0), v);
    col = mix(col, vec3(0.1, 0.0, 0.0), 1.0 - v);
    f_color = vec4(col, 1.0);
}
''')

_register("smoke_gpu", "Rising smoke / steam", "procedural", '''
void main() {
    vec2 uv = v_uv;
    float t = u_time * 0.1;
    float v = fbm(uv * 3.0 + vec2(0.0, t));
    v = v * (1.0 - uv.y) * 0.8;
    vec3 col = mix(vec3(0.8, 0.8, 0.85), vec3(0.2, 0.2, 0.25), v);
    f_color = vec4(col, 1.0);
}
''')


# ── FILTER SHADERS (process input image) ──

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
    float cell = u_cell;
    vec2 cell_uv = floor(uv * u_resolution / cell) * cell / u_resolution + cell / u_resolution * 0.5;
    f_color = texture(u_texture, cell_uv);
'''), uniforms={
    "cell": {"glsl": "float", "min": 10.0, "max": 60.0, "default": 30.0, "description": "tile size"},
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

_register("shader_warhol", "GPU Warhol 4-panel duotone", "filter", _filter_shader('''
    float gray = dot(orig.rgb, vec3(0.299, 0.587, 0.114));
    vec2 p = floor(uv * 2.0);
    vec3 c1, c2;
    if (p.x < 1.0 && p.y < 1.0) { c1 = vec3(0.8, 0.2, 0.2); c2 = vec3(1.0, 1.0, 0.4); }
    else if (p.x >= 1.0 && p.y < 1.0) { c1 = vec3(0.2, 0.4, 0.8); c2 = vec3(0.6, 0.2, 0.6); }
    else if (p.x < 1.0 && p.y >= 1.0) { c1 = vec3(0.2, 0.8, 0.2); c2 = vec3(0.4, 0.2, 0.8); }
    else { c1 = vec3(0.8, 0.6, 0.2); c2 = vec3(0.8, 0.2, 0.2); }
    f_color = vec4(mix(c1, c2, gray), 1.0);
'''))

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
    float band = floor(uv.y * 40.0 * u_amount);
    float shift = sin(band * 7.0 + u_time * 5.0) * 0.05 * u_amount;
    float noise = fract(sin(dot(uv * u_resolution, vec2(12.9898, 78.233))) * 43758.5453);
    float glitch = noise > (1.0 - u_amount * 0.1) ? 1.0 : 0.0;
    vec2 q = uv + vec2(shift + glitch * 0.1, 0.0);
    f_color = texture(u_texture, q);
'''), uniforms={
    "amount": {"glsl": "float", "min": 0.0, "max": 1.0, "default": 0.5, "description": "glitch intensity"},
})

_register("shader_posterize_gpu", "GPU posterization / color reduction", "filter", _filter_typed('''
    float levels = u_levels;
    vec3 col = floor(orig.rgb * levels) / levels;
    f_color = vec4(col, 1.0);
'''), uniforms={
    "levels": {"glsl": "float", "min": 2.0, "max": 16.0, "default": 9.0, "description": "color levels"},
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
          "filter", _filter_shader('''
    // u_params.x = strength (0 = grayscale, 1 = full false-color),
    // u_params.y = scheme (0=standard,1=thermal,2=vegetation,3=urban).
    float lum = dot(orig.rgb, vec3(0.299, 0.587, 0.114));
    float s = clamp(u_params.x, 0.0, 1.0);
    int scheme = int(floor(u_params.y * 3.999));

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
'''))

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


_register("shader_oil_gpu", "GPU oil painting simulation", "filter", _filter_shader('''
    float radius = 2.0 + u_params.x * 4.0;
    vec3 sum = vec3(0.0); float total = 0.0;
    for (int x = -3; x <= 3; x++) {
        for (int y = -3; y <= 3; y++) {
            vec2 off = vec2(float(x), float(y)) * step;
            float w = exp(-float(x*x + y*y) / (radius * radius));
            sum += texture(u_texture, uv + off).rgb * w;
            total += w;
        }
    }
    f_color = vec4(sum / total, 1.0);
'''))

# Anisotropic Kuwahara — coherence-enhancing painterly abstraction (node 68 twin).
# A rotated, elongated Gaussian kernel is oriented along the local structure-tensor
# direction; the window is split into four angular sectors and the lowest-variance
# sector wins (Kyprianidis et al. 2011). Decoded from normalized u_params:
# p1 = radius∈[0,1] (0.5→8), p2 = anisotropy∈[0,1] (0.5→4). CPU numpy node stays
# authoritative for exact export.
_register("anisotropic_kuwahara_gpu", "GPU anisotropic Kuwahara painterly abstraction", "filter", _filter_shader('''
    // Decode normalized params (client sends p1 = radius∈[0,1], p2 = anisotropy∈[0,1],
    // per the 0.5-neutral GPU contract). radius 0.5→8 (node default), aniso 0.5→4.
    int R = int(clamp(2.0 + u_params.x * 12.0, 2.0, 15.0));
    float aniso = clamp(1.0 + u_params.y * 6.0, 1.0, 12.0);

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
'''))

_register("shader_neon_gpu", "GPU neon glow on edges", "filter", _filter_shader('''
    float gx = 0.0, gy = 0.0;
    for (int x = -1; x <= 1; x++) {
        for (int y = -1; y <= 1; y++) {
            vec2 off = vec2(float(x), float(y)) * step;
            float v = dot(texture(u_texture, uv + off).rgb, vec3(0.299, 0.587, 0.114));
            gx += float(x) * v; gy += float(y) * v;
        }
    }
    float edge = sqrt(gx*gx + gy*gy);
    float glow = edge * u_params.x * 3.0;
    vec3 neon = vec3(glow * 0.8, glow * 0.3, glow);
    f_color = vec4(orig.rgb + neon, 1.0);
'''))

_register("shader_pencil_gpu", "GPU pencil sketch", "filter", _filter_shader('''
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
    f_color = vec4(mix(orig.rgb, vec3(sketch), u_params.x), 1.0);
'''))

_register("shader_motion_blur_gpu", "GPU directional motion blur", "filter", _filter_shader('''
    float angle = u_params.x * 6.2832;
    float dist = 10.0 + u_params.y * 20.0;
    vec2 dir = vec2(cos(angle), sin(angle)) * step * dist;
    vec3 col = vec3(0.0);
    for (int i = -5; i <= 5; i++) {
        float t = float(i) / 5.0;
        col += texture(u_texture, uv + dir * t).rgb * (1.0 - abs(t));
    }
    f_color = vec4(col / 3.5, 1.0);
'''))

# ── P0.4 client-GPU twin shaders for existing CPU filter nodes ──
# Each maps a pre-existing CPU filter node's LIVE preview onto a GLSL twin.
# The CPU numpy path stays the authoritative export (two-tier precision).

# 42 Fake HDR — contrast / saturation / vignette / bloom (GPU live twin)
_register("hdr_gpu", "GPU fake-HDR tonemap (contrast/sat/vignette/bloom)", "filter", _filter_shader('''
    float gray = dot(orig.rgb, vec3(0.299, 0.587, 0.114));
    // contrast around mid-gray
    vec3 c = (orig.rgb - 0.5) * (0.5 + u_params.x * 3.0) + 0.5;
    // saturation toward/away from luma
    c = mix(vec3(gray), c, 0.5 + u_params.y * 2.0);
    // bloom: cheap bright-area lift
    float bright = max(0.0, gray - 0.6) * u_params.w * 2.0;
    c += bright;
    // vignette
    vec2 d = uv - 0.5;
    float vig = 1.0 - dot(d, d) * u_params.z * 2.5;
    c *= clamp(vig, 0.0, 1.0);
    f_color = vec4(clamp(c, 0.0, 1.0), 1.0);
'''))

# 63 Cross Stitch — grid of stitches on a fabric backdrop (GPU live twin)
_register("cross_stitch_gpu", "GPU cross-stitch embroidery", "filter", _filter_shader('''
    float gstep = max(4.0, 32.0 - u_params.x * 28.0);   // thread_step -> p1
    float lw = 1.0 + u_params.y * 6.0;                  // line_width -> p2
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
'''))

# 64 Edge Halftone — Sobel-magnitude-weighted dots (GPU live twin)
_register("edge_halftone_gpu", "GPU edge-weighted halftone dots", "filter", _filter_shader('''
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
    float cell = 4.0 + u_params.x * 16.0;               // dot_spacing -> p1
    float base = (1.0 - edge) * 0.5 * (0.5 + u_params.y * 0.5); // dot_size -> p2
    vec2 q = fract(uv * u_resolution / cell) - 0.5;
    float d = length(q);
    float dot_r = clamp(base, 0.02, 0.5);
    float v = d < dot_r ? 0.0 : 1.0;
    vec3 bg = vec3(0.05, 0.05, 0.08);
    f_color = vec4(mix(bg, vec3(1.0), v), 1.0);
'''))

# 74 Swirl Displacement — polar swirl remap (GPU live twin)
_register("swirl_gpu", "GPU swirl/pinch displacement", "filter", _filter_shader('''
    vec2 p = uv - 0.5;
    float r = length(p);
    float a = atan(p.y, p.x);
    float strength = (u_params.x - 0.5) * 6.0;          // 0.5 -> none
    float swirl = strength * (1.0 - r);
    float ca = cos(a + swirl), sa = sin(a + swirl);
    vec2 q = vec2(ca, sa) * r + 0.5;
    f_color = texture(u_texture, q);
'''))

# 13 Dithering — Bayer 8x8 ordered dither with N-level quantization (GPU live
# twin). The CPU node's default `fs` (Floyd-Steinberg error diffusion) is an
# inherently serial scan that cannot be reproduced per-pixel on the GPU, so this
# twin renders the ORDERED (Bayer) approximation and the CPU fn stays
# authoritative for all error-diffusion algorithms. `levels` -> p1 (2..8),
# `contrast` -> p2. `algorithm`/`palette`/`noise_type` are string choices
# (pitfall #14) and are left unmapped.
_register("dither13_gpu", "GPU Bayer-4 ordered dithering (node 13 twin)", "filter", _filter_shader('''
    float gray = dot(orig.rgb, vec3(0.299, 0.587, 0.114));
    float contrast = 0.5 + u_params.y * 2.5;            // contrast -> p2 (0.5..3.0)
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
    float levels = floor(2.0 + u_params.x * 6.0 + 0.5); // levels -> p1 (2..8)
    float steps = max(1.0, levels - 1.0);
    float scaled = gray * steps;
    float lower = floor(scaled);
    float frac = scaled - lower;
    float q = (lower + (frac > threshold ? 1.0 : 0.0)) / steps;
    f_color = vec4(vec3(clamp(q, 0.0, 1.0)), 1.0);
'''))


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

_register("worley_gpu", "Worley/cellular F1 noise (client-GPU twin of node 04)", "procedural", _INFERNO + """
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


# ── Conway's Game of Life (client-GPU sim of nodes 18 / 58) ─────────────────
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


# ═══════════════════════════════════════════════════════════════════════════
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

_register("julia", 'Julia set fractal', "procedural", '\nvoid main() {\n    vec2 uv = (gl_FragCoord.xy - 0.5 * u_resolution) / min(u_resolution.x, u_resolution.y);\n    vec2 c = vec2(u_c_re, u_c_im);\n    vec2 z = uv * exp((u_zoom - 0.5) * 3.0) * 3.0;\n    int n = 0;\n    float last2 = 0.0;\n    const float MAXI = 200.0;\n    for (int i = 0; i < 200; i++) {\n        z = vec2(z.x*z.x - z.y*z.y, 2.0*z.x*z.y) + c;\n        last2 = dot(z, z);\n        if (last2 > 16.0) break;\n        n++;\n    }\n    float t = (n >= MAXI - 0.5) ? 0.0 : clamp((n + 1.0 - log(max(log(last2)*0.5, 1.0001))/log(2.0)) / MAXI, 0.0, 1.0);\n    f_color = vec4(0.5 + 0.5 * cos(t * 6.28318 + vec3(0.0, 2.0, 4.0) + u_color_shift * 6.28), 1.0);\n}\n',
          uniforms={
  "c_re": {
    "glsl": "float",
    "min": -1.0,
    "max": 1.0,
    "default": -0.7269,
    "description": "Julia constant real"
  },
  "c_im": {
    "glsl": "float",
    "min": -1.0,
    "max": 1.0,
    "default": 0.1889,
    "description": "Julia constant imaginary"
  },
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

