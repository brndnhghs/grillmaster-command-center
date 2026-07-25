"""Transform-feedback GPU particle systems.

macOS OpenGL caps at 4.1, so compute shaders (glDispatchCompute / SSBO, GL 4.3)
are unavailable. **Transform feedback** (GL 3.3+) is the portable GPGPU path:
a vertex shader reads a particle's state (x,y,vx,vy), computes the next state,
and the new state is captured back into a buffer via a varying — ping-ponged
between two VBOs each frame. A second pass rasterises the particles as additive
soft points into an FBO → IMAGE.

Per-node state (the double-buffered VBOs + compiled programs) is persistent,
keyed by the node's dir path, and lives on the shared per-thread GL context from
core.shaders (so it shares the live loop's context and needs no locking beyond
the single-cook-at-a-time guarantee).
"""
from __future__ import annotations

import hashlib

import numpy as np
from PIL import Image

from .shaders import _get_ctx, uniform_glsl_decls, coerce_uniform

# node-dir path → dict(ctx, prog_step, prog_draw, bufs, step_vaos, draw_vaos,
#                       cur, count, body_hash)
_STATE: dict[str, dict] = {}

# Shared GLSL helpers available inside an agent's update body.
_PARTICLE_HELPERS = '''
float hash11(float p){ p = fract(p * 0.1031); p *= p + 33.33; p *= p + p; return fract(p); }
vec2  hash21(float p){ vec3 p3 = fract(vec3(p) * vec3(.1031,.1030,.0973));
                       p3 += dot(p3, p3.yzx + 33.33); return fract((p3.xx + p3.yz) * p3.zy); }
'''

_STEP_VS = '''#version 330
in vec4 in_p;              // x, y (in [0,1]) , vx, vy
out vec4 out_p;
uniform float u_time;
uniform float u_dt;
uniform int   u_count;
uniform vec2  u_resolution;
{uniform_decls}
{helpers}
void main() {{
    vec4 p = in_p;
    int  id = gl_VertexID;
    vec4 out_p_default = p;
    {body}
}}
'''

_DRAW_VS = '''#version 330
in vec4 in_p;
uniform float u_point_size;
void main() {
    gl_Position  = vec4(in_p.xy * 2.0 - 1.0, 0.0, 1.0);   // [0,1] -> clip
    gl_PointSize = u_point_size;
}
'''

_DRAW_FS = '''#version 330
out vec4 f_color;
uniform vec3 u_color;
void main() {
    vec2 d = gl_PointCoord - 0.5;
    float a = smoothstep(0.5, 0.0, length(d));   // soft round sprite
    f_color = vec4(u_color * a, a);
}
'''


def _seed_particles(count: int, seed: int) -> np.ndarray:
    """Initial (N,4) state: uniform positions in [0,1], small random velocities."""
    rng = np.random.default_rng(seed)
    pos = rng.random((count, 2), dtype=np.float32)
    vel = (rng.random((count, 2), dtype=np.float32) - 0.5) * 0.02
    return np.concatenate([pos, vel], axis=1).astype('f4')


def compile_particle_programs(glsl_body: str, uniforms: dict):
    """Build the step (transform-feedback) + draw programs. Raises RuntimeError
    with a readable message on GLSL compile failure (the agent's feedback)."""
    ctx = _get_ctx()
    step_src = _STEP_VS.format(
        uniform_decls=uniform_glsl_decls(uniforms), helpers=_PARTICLE_HELPERS,
        body=glsl_body,
    )
    try:
        prog_step = ctx.program(vertex_shader=step_src, varyings=['out_p'])
        prog_draw = ctx.program(vertex_shader=_DRAW_VS, fragment_shader=_DRAW_FS)
    except Exception as e:
        raise RuntimeError(str(e))
    return ctx, prog_step, prog_draw


def _ensure_state(key: str, glsl_body: str, uniforms: dict, count: int,
                  seed: int, reseed: bool):
    body_hash = hashlib.sha1((glsl_body + repr(sorted(uniforms))).encode()).hexdigest()
    st = _STATE.get(key)
    ctx = _get_ctx()
    stale = (st is None or st["ctx"] is not ctx
             or st["body_hash"] != body_hash or st["count"] != count)
    if stale:
        _ctx, prog_step, prog_draw = compile_particle_programs(glsl_body, uniforms)
        data = _seed_particles(count, seed)
        bufs = [ctx.buffer(data.tobytes()), ctx.buffer(reserve=data.nbytes)]
        step_vaos = [ctx.vertex_array(prog_step, [(bufs[i], '4f', 'in_p')]) for i in (0, 1)]
        draw_vaos = [ctx.vertex_array(prog_draw, [(bufs[i], '4f', 'in_p')]) for i in (0, 1)]
        st = {"ctx": ctx, "prog_step": prog_step, "prog_draw": prog_draw,
              "bufs": bufs, "step_vaos": step_vaos, "draw_vaos": draw_vaos,
              "cur": 0, "count": count, "body_hash": body_hash}
        _STATE[key] = st
    elif reseed:
        st["bufs"][st["cur"]].write(_seed_particles(count, seed).tobytes())
    return st


def render_particles(key: str, glsl_body: str, uniforms_spec: dict, params: dict,
                     count: int, cw: int, ch: int, *, seed: int = 0,
                     point_size: float = 3.0, color=(0.6, 0.8, 1.0),
                     dt: float = 1.0, emit: bool = False):
    """Step the particle system one frame on the GPU and rasterise it.

    Returns (image float32 (ch,cw,3), particles (N,4) ndarray or None).
    Reseeds when the injected frame is 0 (timeline restart).
    """
    import moderngl
    frame = int(params.get("frame", params.get("time", 0)) or 0)
    st = _ensure_state(key, glsl_body, uniforms_spec, count, seed, reseed=(frame == 0))
    ctx = st["ctx"]
    prog_step, prog_draw = st["prog_step"], st["prog_draw"]

    # ── uniforms on the step program ──
    for uname, val in (("u_time", float(params.get("time", 0.0))),
                       ("u_dt", float(dt)), ("u_count", int(count)),
                       ("u_resolution", (float(cw), float(ch)))):
        if uname in prog_step:
            prog_step[uname].value = val
    for name, spec in (uniforms_spec or {}).items():
        u = f"u_{name}"
        if u in prog_step:
            prog_step[u].value = coerce_uniform(spec, params.get(name, spec.get("default")))

    # ── step: front → back via transform feedback, then swap ──
    cur = st["cur"]
    st["step_vaos"][cur].transform(st["bufs"][1 - cur], mode=moderngl.POINTS)
    cur = 1 - cur
    st["cur"] = cur

    # ── draw: additive soft points into an FBO ──
    if "u_point_size" in prog_draw:
        prog_draw["u_point_size"].value = float(point_size)
    if "u_color" in prog_draw:
        prog_draw["u_color"].value = tuple(float(c) for c in color)
    fbo = ctx.simple_framebuffer((cw, ch))
    fbo.use()
    ctx.clear(0.0, 0.0, 0.0)
    ctx.enable(moderngl.PROGRAM_POINT_SIZE | moderngl.BLEND)
    ctx.blend_func = (moderngl.SRC_ALPHA, moderngl.ONE)     # additive glow
    st["draw_vaos"][cur].render(mode=moderngl.POINTS)
    data = fbo.read()
    ctx.disable(moderngl.BLEND)

    # Match the server's shader readback convention (BGR decode, no Y-flip) so a
    # particle node composites like every other GPU node.
    arr = np.array(Image.frombytes('RGB', (cw, ch), data, 'raw', 'BGR'),
                   dtype=np.float32) / 255.0
    fbo.release()

    parts = None
    if emit:
        parts = np.frombuffer(st["bufs"][cur].read(), dtype='f4').reshape(count, 4).copy()
    return arr, parts


def drop_state(key: str) -> None:
    """Release a node's particle GL resources (buffers + programs)."""
    st = _STATE.pop(key, None)
    if not st:
        return
    for b in st["bufs"]:
        try:
            b.release()
        except Exception:
            pass
