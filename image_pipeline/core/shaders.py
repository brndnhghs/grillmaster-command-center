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
    float zoom = exp((u_params.x - 0.5) * 6.0);
    vec2 ctr = vec2(u_params.z, u_params.w);
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
    f_color = vec4(fractal_palette(t + u_params.y), 1.0);
}
''')

_register("burning_ship_gpu", "Burning Ship fractal (client-GPU twin of node 51)", "procedural",
          _FRACTAL_HELPERS + '''
void main() {
    vec2 uv = (gl_FragCoord.xy - 0.5 * u_resolution) / min(u_resolution.x, u_resolution.y);
    // p1 = zoom (0.5 = full view), p2 = color_shift, p3 = center_x, p4 = center_y.
    float zoom = exp((u_params.x - 0.5) * 6.0);
    vec2 ctr = vec2(u_params.z, u_params.w);
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
    f_color = vec4(fractal_palette(t + u_params.y), 1.0);
}
''')

_register("newton_gpu", "Newton fractal basins (client-GPU twin of node 52)", "procedural",
          _FRACTAL_HELPERS + '''
void main() {
    vec2 uv = (gl_FragCoord.xy - 0.5 * u_resolution) / min(u_resolution.x, u_resolution.y);
    // p1 = color_speed, p2 = color_offset, p3 = zoom (0.5 = full view), p4 = unused.
    vec2 z = uv * exp((u_params.z - 0.5) * 5.0) * 2.2;
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
    float t = mod(root / 3.0 + u_params.y + 0.15 * n / MAXI, 1.0);
    f_color = vec4(fractal_palette(t * (0.6 + 0.4 * u_params.x)), 1.0);
}
''')

_register("sierpinski_gpu", "Sierpinski carpet (client-GPU twin of node 67)", "procedural",
          _FRACTAL_HELPERS + '''
void main() {
    vec2 uv = v_uv;
    // p1 = depth (subdivisions), p2 = color_shift, p3/p4 unused (reserved).
    float depth = clamp(floor(u_params.x * 7.0) + 1.0, 1.0, 7.0);
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
    float t = fract(0.15 * (depth) + u_params.y + 0.3 * uv.x + 0.2 * uv.y);
    vec3 col = (hole > 0.5) ? vec3(0.04) : fractal_palette(t);
    f_color = vec4(col, 1.0);
}
''')

_register("lyapunov_gpu", "Lyapunov exponent map (client-GPU twin of node 69)", "procedural",
          _FRACTAL_HELPERS + '''
void main() {
    vec2 uv = v_uv;
    // p1 = r_min, p2 = r_max, p3 = color_mode(0=lyapunov), p4 = color_shift.
    vec2 rmin = vec2(u_params.x, u_params.x);
    vec2 rmax = vec2(u_params.y, u_params.y);
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
    t = (u_params.z > 0.5) ? fract(t + u_params.w) : t;
    f_color = vec4(fractal_palette(t), 1.0);
}
''')

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

_register("shader_bloom", "GPU bloom/glow from bright areas", "filter", _filter_shader('''
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
    f_color = orig + glow * u_params.x;
'''))

_register("shader_emboss", "GPU emboss / bump mapping", "filter", _filter_shader('''
    float gx = dot(texture(u_texture, uv + vec2(step.x, 0)).rgb, vec3(0.299, 0.587, 0.114));
    float gy = dot(texture(u_texture, uv + vec2(0, step.y)).rgb, vec3(0.299, 0.587, 0.114));
    float gz = dot(orig.rgb, vec3(0.299, 0.587, 0.114));
    float dx = gx - gz; float dy = gy - gz;
    float bump = (dx + dy) * 2.0 + 0.5;
    f_color = vec4(mix(orig.rgb, vec3(bump), u_params.x), 1.0);
'''))

_register("shader_kaleidoscope", "GPU kaleidoscope mirror", "filter", _filter_shader('''
    vec2 p = uv - 0.5;
    float a = atan(p.y, p.x);
    float r = length(p);
    float seg = 3.14159 * 2.0 / max(4.0, 8.0 - u_params.x * 4.0);
    a = mod(a, seg);
    a = abs(a - seg * 0.5);
    vec2 q = vec2(cos(a), sin(a)) * r + 0.5;
    f_color = texture(u_texture, q);
'''))

_register("shader_water_ripple", "GPU water ripple distortion", "filter", _filter_shader('''
    vec2 off = vec2(
        sin(uv.y * 50.0 + u_time * 2.0) * 0.01 * u_params.x,
        cos(uv.x * 50.0 + u_time * 1.5) * 0.01 * u_params.x
    );
    f_color = texture(u_texture, uv + off);
'''))

_register("shader_heat_shimmer", "GPU heat haze / shimmer", "filter", _filter_shader('''
    float haze = sin(uv.x * 30.0 + uv.y * 20.0 + u_time * 3.0) * u_params.x * 0.02;
    vec2 off = vec2(0.0, haze * (1.0 - uv.y));
    f_color = texture(u_texture, uv + off);
'''))

_register("shader_pixelate_gpu", "GPU pixelation with edge preservation", "filter", _filter_shader('''
    float block = max(4.0, 64.0 - u_params.x * 60.0);
    vec2 q = floor(uv * u_resolution / block) * block / u_resolution;
    f_color = texture(u_texture, q);
'''))

_register("shader_ink_bleed", "GPU ink bleed / watercolor spread", "filter", _filter_shader('''
    vec3 sum = vec3(0.0);
    float count = 0.0;
    for (int x = -4; x <= 4; x++) {
        for (int y = -4; y <= 4; y++) {
            vec2 off = vec2(float(x), float(y)) * step * u_params.x;
            float w = exp(-float(x*x + y*y) / (4.0 * u_params.x));
            sum += texture(u_texture, uv + off).rgb * w;
            count += w;
        }
    }
    f_color = vec4(sum / count, 1.0);
'''))

_register("shader_halftone_gpu", "GPU halftone dot screen", "filter", _filter_shader('''
    float gray = dot(orig.rgb, vec3(0.299, 0.587, 0.114));
    float cell = 8.0 + u_params.x * 20.0;
    vec2 q = fract(uv * u_resolution / cell);
    float d = length(q - 0.5);
    float dot_r = (1.0 - gray) * 0.5;
    float v = d < dot_r ? 0.0 : 1.0;
    f_color = vec4(vec3(v), 1.0);
'''))

_register("shader_crt_gpu", "GPU CRT scanlines + bloom", "filter", _filter_shader('''
    float scan = sin(uv.y * u_resolution.y * 3.14159) * 0.5 + 0.5;
    float scanline = 1.0 - (1.0 - scan) * 0.3;
    // chromatic shift at edges
    vec2 r_uv = uv + vec2(0.001 * pow(abs(uv.x - 0.5) * 2.0, 2.0), 0.0);
    vec2 b_uv = uv - vec2(0.001 * pow(abs(uv.x - 0.5) * 2.0, 2.0), 0.0);
    vec3 col;
    col.r = texture(u_texture, r_uv).r;
    col.g = texture(u_texture, uv).g;
    col.b = texture(u_texture, b_uv).b;
    col *= scanline;
    f_color = vec4(col, 1.0);
'''))

_register("shader_hologram", "GPU hologram / scan effect", "filter", _filter_shader('''
    float scan = sin(uv.y * u_resolution.y * 0.5 + u_time * 5.0) * 0.5 + 0.5;
    float scanline = 1.0 - pow(scan, 4.0) * 0.4;
    float edge = abs(uv.x - 0.5) * 2.0;
    float vignette = 1.0 - pow(edge, 3.0) * 0.5;
    float shift = sin(uv.x * 50.0 + u_time * 3.0) * 0.02;
    vec2 q = uv + vec2(0.0, shift);
    vec3 col = texture(u_texture, q).rgb * scanline * vignette;
    float hue = sin(uv.y * 20.0 + u_time * 2.0) * 0.1 + 0.1;
    col += vec3(hue, hue * 0.3, hue * 0.8);
    f_color = vec4(col, 1.0);
'''))

_register("shader_mosaic_gpu", "GPU stained glass mosaic", "filter", _filter_shader('''
    float cell = 20.0 + u_params.x * 40.0;
    vec2 cell_uv = floor(uv * u_resolution / cell) * cell / u_resolution + cell / u_resolution * 0.5;
    f_color = texture(u_texture, cell_uv);
'''))

_register("shader_edge_detect_gpu", "GPU Sobel edge detection", "filter", _filter_shader('''
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
    f_color = vec4(mix(orig.rgb, vec3(edge), u_params.x), 1.0);
'''))

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

_register("shader_duotone_gpu", "GPU duotone with color controls", "filter", _filter_shader('''
    float gray = dot(orig.rgb, vec3(0.299, 0.587, 0.114));
    vec3 c1 = vec3(u_params.x, u_params.y, u_params.z);
    vec3 c2 = vec3(u_params.w, 0.2, 0.8);
    f_color = vec4(mix(c1, c2, gray), 1.0);
'''))

_register("shader_rgb_split", "GPU RGB channel separation", "filter", _filter_shader('''
    float shift = 0.02 * u_params.x;
    vec2 r_uv = uv + vec2(shift, 0.0);
    vec2 b_uv = uv - vec2(shift, 0.0);
    float r = texture(u_texture, r_uv).r;
    float g = orig.g;
    float b = texture(u_texture, b_uv).b;
    f_color = vec4(r, g, b, 1.0);
'''))

_register("shader_caustics_gpu", "GPU caustic light overlay", "filter", _filter_shader('''
    float caustic = sin(uv.x * 30.0 + u_time) * cos(uv.y * 25.0 + u_time * 0.7);
    caustic += sin(uv.x * 50.0 - u_time * 1.3) * sin(uv.y * 40.0 + u_time * 0.5) * 0.5;
    caustic = max(0.0, caustic) * u_params.x * 0.8;
    vec3 light = vec3(0.8, 0.9, 1.0) * caustic;
    f_color = vec4(orig.rgb + light, 1.0);
'''))

_register("shader_glitch_gpu", "GPU digital glitch artifacts", "filter", _filter_shader('''
    float band = floor(uv.y * 40.0 * u_params.x);
    float shift = sin(band * 7.0 + u_time * 5.0) * 0.05 * u_params.x;
    float noise = fract(sin(dot(uv * u_resolution, vec2(12.9898, 78.233))) * 43758.5453);
    float glitch = noise > (1.0 - u_params.x * 0.1) ? 1.0 : 0.0;
    vec2 q = uv + vec2(shift + glitch * 0.1, 0.0);
    f_color = texture(u_texture, q);
'''))

_register("shader_posterize_gpu", "GPU posterization / color reduction", "filter", _filter_shader('''
    float levels = max(2.0, 16.0 - u_params.x * 14.0);
    vec3 col = floor(orig.rgb * levels) / levels;
    f_color = vec4(col, 1.0);
'''))

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
    float dir = (u_params.x - 0.5) * 6.2831853;
    vec2 ctr = vec2(u_params.y, u_params.z);
    vec2 p = v_uv - ctr;

    float t;
    int gtype = int(floor(u_params.w * 4.999));
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
''')

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

_register("palette_gpu",
          "Color palette swatches (client-GPU twin of node 10)",
          "procedural", '''
void main() {
    // u_params.x = n_colors (2..32), u_params.y = saturation (0.5=auto),
    // u_params.z = hue_offset (0..1), u_params.w = value (0.5=auto).
    float ncols = floor(2.0 + u_params.x * 30.0);
    float hueOff = u_params.z;
    float sat = (u_params.y <= 0.0) ? 0.75 : clamp(u_params.y, 0.0, 1.0);
    float val = (u_params.w <= 0.0) ? 0.95 : clamp(u_params.w, 0.0, 1.0);

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
''')




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
    float m = 0.5 + u_params.x * 11.0;
    float n = 0.5 + u_params.y * 11.0;
    float rot_ang = (u_params.z - 0.5) * 6.2831853;
    float ph = (u_params.w - 0.5) * 6.2831853;

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
''')

_register("moire_gpu",
          "Moiré interference (client-GPU twin of node 164)",
          "procedural", '''
void main() {
    // u_params.x = mode (0=radial,1=linear,2=spiral,3=hex),
    // u_params.y = speed1 (0.5 -> ~1.0), u_params.z = speed2 (0.5 -> ~1.28),
    // u_params.w = frequency (0.5 -> 20).
    int mode = int(floor(u_params.x * 3.999));
    float s1 = 0.1 + u_params.y * 1.9;      // ~1.0 at default
    float s2 = 0.1 + u_params.z * 1.9;      // ~1.28 at default
    float freq = 5.0 + u_params.w * 45.0;   // 20 at default
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
''')


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
    float wind = 0.1 + u_params.x * 1.0;     // ~0.6 at neutral 0.5
    float sed  = 0.1 + u_params.y * 1.4;     // ~0.8 at neutral 0.5
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
''')

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
    float iso   = 0.05 + u_params.x * 0.75;
    float speed = 0.1  + u_params.y * 4.9;
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
''')

_register("heatmap_gpu",
          "Density heatmap (client-GPU twin of node 43)",
          "procedural", _inferno_local('') + '''
void main() {
    // u_params.x = sigma proxy (0.5 -> ~0.06), u_params.z = colormap_shift.
    float sigma = 0.01 + u_params.x * 0.10;
    float shift = u_params.z;
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
''')

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
    int mode = int(floor(u_params.z * 6.999));
    float amp  = 0.05 + u_params.x * 0.45;
    float freq = 0.005 + u_params.y * 0.495;
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
''')

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
    float jitter = u_params.x;
    float cells  = 4.0 + u_params.y * 14.0;
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
""")

_register("quasicrystal_gpu", "Quasicrystal wave superposition (client-GPU twin of node 02)", "procedural", _INFERNO + """
float h11(float n){ return fract(sin(n*127.1)*43758.5453); }
void main() {
    // u_params.x = frequency, .y = amplitude, .z = rotation, .w = wave count
    float freq = max(u_params.x, 0.005);
    float amp  = (u_params.y <= 0.0) ? 1.0 : u_params.y;
    float rot  = u_params.z;
    int nwaves = int(clamp(u_params.w, 2.0, 24.0));
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
""")


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
    float ts = mix(8.0, 64.0, clamp(u_params.x, 0.0, 1.0));
    float cv = u_params.y;
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
""")

_register("morph_grid_gpu",
            "Morphing grid warp (client-GPU twin of node 105)",
            "procedural", _INFERNO + """
void main() {
    // u_params.x = warp strength, .y = line width, .z = palette mix
    float ws = u_params.x;
    float lw = clamp(u_params.y, 0.02, 1.0);
    vec2 p = v_uv * 14.0;
    vec2 w = vec2(fbm(p + u_time * 0.1), fbm(p.yx - u_time * 0.1));
    p += (w - 0.5) * ws * 6.0;
    vec2 g = abs(fract(p) - 0.5);
    float line = smoothstep(lw, lw * 0.5, min(g.x, g.y));
    vec3 col = 0.5 + 0.5 * cos(6.2831853 * (u_params.z + vec3(0.0, 0.33, 0.67)) + w.x * 4.0);
    f_color = vec4(mix(vec3(line), col, line), 1.0);
}
""")

_register("phyllotaxis_gpu",
            "Phyllotaxis spiral field (client-GPU twin of node 08)",
            "procedural", _INFERNO + """
void main() {
    // u_params.x = point density, .y = angle goldenness, .z = radius scale
    float dens = mix(0.1, 1.0, clamp(u_params.x, 0.0, 1.0));
    float phi = 2.39996323 + u_params.y * 1.5;        // ~golden angle + jitter
    vec2 c = (v_uv - 0.5) * u_resolution;
    float rmax = 0.5 * min(u_resolution.x, u_resolution.y);
    float acc = 0.0;
    for (int i = 0; i < 220; i++) {
        float fi = float(i) * dens * 12.0;
        float a = fi * phi;
        float rad = sqrt(fi) * (u_params.z * 0.5 + 0.05) * rmax * 0.06;
        vec2 pos = rad * vec2(cos(a), sin(a));
        acc += smoothstep(3.0, 0.0, length(c - pos));
    }
    f_color = vec4(inferno(clamp(acc * 0.5, 0.0, 1.0)), 1.0);
}
""")


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
