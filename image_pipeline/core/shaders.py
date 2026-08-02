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
import re
import threading

import numpy as np
from PIL import Image

# Shader registrations live in one file per shader under core/shader_library/
# (dynamically loaded at the bottom of this module). SHADERS/_register live in
# the library's _registry.py; _PROLOGUE + shared GLSL helpers in _helpers.py.
from .shader_library._registry import SHADERS, _register  # noqa: E402,F401
from .shader_library._helpers import _PROLOGUE  # noqa: E402


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

_UNIFORM_GLSL_TYPES = {"float": "float", "int": "int", "color": "vec3", "choice": "int", "sampler2D": "sampler2D"}


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


_prog_cache_local = threading.local()


def _get_prog_cache() -> dict:
    cache = getattr(_prog_cache_local, "cache", None)
    if cache is None:
        _prog_cache_local.cache = {}
    return _prog_cache_local.cache


def _create_vao(ctx, prog):
    """Create full-screen quad VAO.

    Bind only the attributes the GLSL compiler actually kept: when a fragment
    shader never uses v_uv (fractals, math art), drivers dead-code-eliminate
    in_uv and its whole chain, so a hardcoded 'in_uv' binding raises KeyError
    (seen on NVIDIA and modern Intel GLSL compilers; macOS kept it).
    """
    vbo = ctx.buffer(_QUAD_VERTICES.tobytes())
    ibo = ctx.buffer(_QUAD_INDICES.tobytes())
    if 'in_uv' in prog:
        vao = ctx.vertex_array(prog, [
            (vbo, '2f 2f', 'in_vert', 'in_uv'),
        ], ibo)
    else:
        # 16-byte vertex stride: bind the full quad as '4f' — the fragment
        # only reads xy; uv floats are dead data (a '2f 12x' skip layout
        # drops the first triangle on some GL drivers).
        vao = ctx.vertex_array(prog, [
            (vbo, '4f', 'in_vert'),
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
        tex_data = img_u8[:, :, ::-1].tobytes()  # RGB -> BGR for GL (pipeline-wide BGR convention)
        texture = ctx.texture((img_u8.shape[1], img_u8.shape[0]), 3, tex_data)
        texture.use(0)
        prog['u_texture'].value = 0

    # Bind glyph atlas for ascii_art_gpu shader
    atlas_tex = None
    if shader_name == 'ascii_art_gpu' and 'u_glyph_atlas' in prog:
        from .ascii_gpu_fonts_json import get_atlas_texture
        font_idx = 0
        if named_params and 'font' in named_params:
            font_choices = info.get('uniforms', {}).get('font', {}).get('choices', [])
            font_name = named_params['font']
            if font_name in font_choices:
                font_idx = font_choices.index(font_name)
        atlas_tex = get_atlas_texture(ctx, prog, font_idx)
        if atlas_tex is not None:
            atlas_tex.use(1)
            prog['u_glyph_atlas'].value = 1

    ctx.clear(0.0, 0.0, 0.0)
    vao.render()
    data = fbo.read()

    # Convert to PIL. The pipeline carries colors BGR end-to-end (coerce_uniform
    # pre-swaps color uniforms, input textures are uploaded BGR), so the readback
    # bytes are BGR and the raw decoder swaps them back to RGB.
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
#  SHADER LIBRARY (dynamically loaded)
# ═══════════════════════════════════════════════
# Each shader registration lives in its own file under core/shader_library/
# (one module per shader name). Importing this package runs every module's
# _register(...) call, populating SHADERS above. The per-shader files are the
# link targets for the Node Doctor source window.
from . import shader_library as _shader_library  # noqa: E402
_shader_library.load_all()
