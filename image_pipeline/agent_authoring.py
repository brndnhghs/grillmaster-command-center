"""Runtime, agent-driven node authoring.

Lets an LLM (over MCP) invent a brand-new GPU node *while the app is running* —
no file edit, no restart, no recompile. The node it authors is a GLSL fragment
body plus a small typed-uniform manifest; this module compiles it, registers it
as a first-class @method node (params + wireable ports auto-derived), and makes
the live DAG see it.

Everything here is orchestration over primitives that already exist:

    core.shaders._register        — add source+uniforms to the live shader lib
    core.shaders.render_shader    — runtime GLSL compile + cache + render
    methods.gpu_shaders._make_typed — shader+uniforms -> @method node w/ ports
    core.registry.unregister      — drop a node
    core.graph.clear_node_defs_cache — make the DAG re-read the registry
    core.port_types.register_port_type — runtime-registrable port types

The only genuinely new logic is: allocate a collision-free id, use a unique
internal shader key (so the per-thread program cache can never serve a stale
compile for a re-authored node), compile-test *now* so the agent gets the GLSL
error as feedback instead of a broken node, and clean up on failure.

Agent-authored ids live in a high namespace (>= AGENT_ID_BASE) so they never
collide with the file-based method ids that tools/next_id.py allocates.
"""
from __future__ import annotations

import uuid
from pathlib import Path
from typing import Any

AGENT_ID_BASE = 9000


# ── id allocation ─────────────────────────────────────────────────────

def next_agent_id() -> str:
    """Lowest unused numeric id in the agent namespace (>= AGENT_ID_BASE)."""
    from .core import registry
    used = {int(i) for i in registry.get_ids() if i.isdigit()}
    n = max([i for i in used if i >= AGENT_ID_BASE] + [AGENT_ID_BASE - 1]) + 1
    return str(n)


# ── the authoring entry point ─────────────────────────────────────────

def register_node_type(manifest: dict) -> dict:
    """Compile + register an agent-authored GPU node at runtime.

    manifest = {
        "name":        "Curl Warp",                 # display name (required)
        "type": "procedural"|"filter"|"feedback"|"expression"|"particles",  # default procedural
        "glsl":        "void main(){ ... }",         # fragment body (GPU types)
        "description": "…",                          # optional
        "uniforms": {                                # optional typed params
            "scale": {"glsl": "float", "min": 0, "max": 10,
                      "default": 3.0, "description": "warp amount"},
        },
    }

    The GLSL body is the raw ``void main(){...}`` — the shared prologue
    (``#version``, ``u_resolution``/``u_time``/``u_params``/``u_texture``,
    noise helpers, and every ``uniform <t> u_<name>`` from the manifest) is
    injected for you, exactly as the built-in shader nodes get it.

    Uniform keys are BARE names: manifest key ``scale`` is declared as
    ``uniform float u_scale`` and referenced in the body as ``u_scale``. A
    filter node also gets ``sampler2D u_texture`` (the upstream image). Valid
    ``glsl`` types: float, int, color, choice.

    type="feedback" authors a SIMULATION node (reaction-diffusion, life,
    growth, infinite-feedback canvas). It is a filter — ``u_texture`` is the
    node's OWN PREVIOUS FRAME (ping-pong is handled by a self-feedback edge;
    use plan_wire/wire's ``feedback_self`` to add it automatically). Feedback
    nodes are forced ``is_time_varying=True`` so the live loop re-cooks them
    every frame — a filter that reads only ``u_texture`` (never ``u_time``)
    would otherwise be judged static, cooked once, and frozen. On frame 0 the
    previous frame is black; seed your initial state when ``u_time < 1.0``.

    Returns on success:
        {"ok": True, "id": "9001",
         "ports": {"inputs": {...}, "outputs": {...}},
         "params": {...}}

    Returns on a GLSL compile error (the agent's feedback loop):
        {"ok": False, "compile_error": "0:14: 'vec4' : ..."}
    """
    from .core import registry
    from .core.shaders import _register, render_shader, SHADERS, _PROLOGUE
    from .core.graph import clear_node_defs_cache
    from .methods.gpu_shaders import _make_typed

    name = (manifest.get("name") or "").strip()
    if not name:
        return {"ok": False, "error": "manifest.name is required"}

    # Expression nodes are CPU math, not shaders — separate path entirely.
    if manifest.get("type") == "expression":
        return _register_expression_node(name, manifest)
    # Particle nodes are transform-feedback GPU systems — their own path.
    if manifest.get("type") == "particles":
        return _register_particle_node(name, manifest)

    glsl = manifest.get("glsl") or ""
    if "void main" not in glsl:
        return {"ok": False, "error": "manifest.glsl must contain a void main(){...} body"}

    stype = manifest.get("type", "procedural")
    if stype not in ("procedural", "filter", "feedback"):
        return {"ok": False, "error":
                f"type must be 'procedural', 'filter' or 'feedback', got {stype!r}"}
    is_feedback = stype == "feedback"
    # A feedback node is a filter under the hood: u_texture = its own prev frame.
    shader_type = "filter" if is_feedback else stype

    # Common agent mistake: keying a uniform "u_freq" declares "u_u_freq".
    bad_keys = [k for k in (manifest.get("uniforms") or {}) if k.startswith("u_")]
    if bad_keys:
        return {"ok": False, "error": (
            f"uniform keys must be bare names, not {bad_keys} — drop the 'u_' "
            f"prefix (key 'freq' is declared as 'uniform ... u_freq' and read "
            f"as u_freq in the body)")}

    node_id = next_agent_id()
    # Unique internal key => the thread-local program cache (keyed by shader
    # name) can never hand back a stale compile for a re-authored node.
    shader_key = f"agent_{node_id}"

    # 1. Put source + typed uniforms into the live shader library.
    # Keep the agent contract uniform (always a bare `void main(){...}` with the
    # prologue auto-injected). _assemble_gl330 injects the prologue for
    # procedural and for filters WITH uniforms, but returns a uniform-less
    # filter's source verbatim (legacy filters embed their own #version). So for
    # the filter/feedback-without-uniforms case, prepend the prologue here.
    uniforms = manifest.get("uniforms") or {}
    source = glsl
    if shader_type == "filter" and not uniforms:
        source = _PROLOGUE + "\n" + glsl
    _register(shader_key, manifest.get("description", ""), shader_type,
              source, uniforms)

    # 2. Compile NOW at tiny resolution so the agent gets the GLSL error back,
    #    not a node that explodes on first cook.
    try:
        render_shader(shader_key, (16, 16), (0.5, 0.5, 0.5, 0.5), 0.0)
    except Exception as e:  # ValueError (bad uniform) or RuntimeError (GLSL)
        SHADERS.pop(shader_key, None)
        return {"ok": False, "compile_error": str(e)}

    # 3. shader + uniforms  ->  a real @method node (params + wireable ports).
    try:
        _make_typed(node_id, shader_key, name)
    except Exception as e:
        SHADERS.pop(shader_key, None)
        return {"ok": False, "error": f"node registration failed: {e}"}

    meta = registry.get_meta(node_id)

    # Feedback nodes MUST re-cook every frame. _make_typed derives
    # is_time_varying from whether the body references u_time, so a sim that
    # reads only u_texture would be judged static and frozen by the live loop's
    # dirty-skip. Force it on (and tag it) before the node-defs are rebuilt.
    if is_feedback:
        meta.is_time_varying = True
        if "feedback" not in meta.tags:
            meta.tags = list(meta.tags) + ["feedback"]

    # 4. Make the live DAG re-read the registry (picks up is_time_varying).
    clear_node_defs_cache()

    result = {
        "ok": True,
        "id": node_id,
        "name": name,
        "type": stype,
        "ports": {"inputs": dict(meta.inputs or {}), "outputs": dict(meta.outputs)},
        "params": dict(meta.params),
    }
    if is_feedback:
        result["feedback"] = True
        result["hint"] = ("wire a self-feedback edge (image -> image_in, "
                          "feedback=True) — or pass feedback_self:true in wire's "
                          "add_nodes. Seed initial state when u_time < 1.0.")
    return result


# ── expression body kind (CPU scalar math) ───────────────────────────
# A SCALAR-output node whose body is a safe math expression, not GLSL — LFOs,
# envelopes, math over time and other scalars, to modulate any wireable param.
# Reuses core.expr's function whitelist + AST safety approach, extended so the
# node's own free variables are allowed names (core.expr only permits t/frame/
# seed). `t` is the frame-based time (same value injected as u_time).

def _parse_expression(expr: str):
    """Return (code, sorted free_vars, uses_t) or raise ValueError with a message
    the agent can act on. Free vars = names that aren't math builtins or `t`."""
    import ast
    from .core.expr import _SAFE_NAMES, _ALLOWED, _SAFE_CALL_NAMES

    expr = (expr or "").strip()
    if not expr:
        raise ValueError("expression is empty")
    try:
        tree = ast.parse(expr, mode="eval")
    except SyntaxError as e:
        raise ValueError(f"syntax error: {e}")

    names = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)}
    free_vars = sorted(names - set(_SAFE_NAMES) - {"t"})
    allowed = set(_SAFE_NAMES) | {"t"} | set(free_vars)

    def _safe(node) -> bool:
        if not isinstance(node, _ALLOWED):
            return False
        if isinstance(node, ast.Call) and not (
                isinstance(node.func, ast.Name) and node.func.id in _SAFE_CALL_NAMES):
            return False
        if isinstance(node, ast.Name) and node.id not in allowed:
            return False
        return all(_safe(c) for c in ast.iter_child_nodes(node))

    if not _safe(tree):
        raise ValueError("expression uses a disallowed construct (only +-*/%**, "
                         "comparisons, if/else, and sin/cos/sqrt/abs/min/max/… "
                         "over t and your variables are permitted)")
    uses_t = "t" in names
    return compile(tree, "<expr>", "eval"), free_vars, uses_t


def _register_expression_node(name: str, manifest: dict) -> dict:
    """Register a SCALAR-output expression node into the live registry."""
    from .core import registry
    from .core.registry import method as _method_dec
    from .core.expr import _SAFE_NAMES
    from .core.graph import clear_node_defs_cache

    expr = manifest.get("expr") or ""
    try:
        code, free_vars, uses_t = _parse_expression(expr)
    except ValueError as e:
        return {"ok": False, "expr_error": str(e)}   # the agent's feedback loop

    var_specs = manifest.get("vars") or {}
    defaults = {v: float((var_specs.get(v) or {}).get("default", 0.0)) for v in free_vars}

    def _param_spec(v: str) -> dict:
        s = var_specs.get(v) or {}
        p = {"default": defaults[v], "description": s.get("description", "")}
        if "min" in s:
            p["min"] = s["min"]
        if "max" in s:
            p["max"] = s["max"]
        return p

    node_id = next_agent_id()

    @_method_dec(
        id=node_id, name=name, category="expression",
        inputs={v: "SCALAR" for v in free_vars},   # each free var is a wireable SCALAR port
        outputs={"value": "SCALAR"},
        params={v: _param_spec(v) for v in free_vars},
        tags=["expression", "cpu", "fast"],
        is_time_varying=uses_t,                    # static unless it reads t
        description=manifest.get("description", ""),
    )
    def _fn(out_dir, seed, params=None, _code=code, _vars=tuple(free_vars),
            _defaults=defaults):
        params = params or {}
        ctx = {**_SAFE_NAMES, "t": float(params.get("time", 0.0))}
        for v in _vars:
            try:
                ctx[v] = float(params.get(v, _defaults[v]))
            except (TypeError, ValueError):
                ctx[v] = _defaults[v]
        try:
            return {"value": float(eval(_code, {"__builtins__": {}}, ctx))}  # noqa: S307
        except Exception:
            return {"value": 0.0}

    _fn.__name__ = f"expr_{node_id}"
    clear_node_defs_cache()
    meta = registry.get_meta(node_id)
    return {
        "ok": True, "id": node_id, "name": name, "type": "expression",
        "expr": expr, "vars": free_vars, "is_time_varying": uses_t,
        "ports": {"inputs": dict(meta.inputs or {}), "outputs": dict(meta.outputs)},
        "params": dict(meta.params),
        "hint": "outputs a SCALAR on port 'value'; wire it into any node's "
                "wireable param (freq, decay, gain, …). 't' is frame time.",
    }


# ── particle body kind (transform-feedback GPU) ───────────────────────
# GL 4.1 on macOS has no compute shaders, so particles step via transform
# feedback (core.particles). The agent authors a vertex-shader update body that
# sets out_p = vec4(x,y,vx,vy) from `p`; the system handles double-buffering,
# seeding, point rasterisation, and per-node persistent state.

def _register_particle_node(name: str, manifest: dict) -> dict:
    from .core import registry
    from .core.registry import method as _method_dec
    from .core.graph import clear_node_defs_cache
    from .core import particles as _particles
    from .core.shaders import _parse_color
    from .core.utils import get_canvas
    from .methods.gpu_shaders import _param_from_uniform

    glsl = manifest.get("glsl") or ""
    if "out_p" not in glsl:
        return {"ok": False, "error":
                "particle body must assign out_p = vec4(x, y, vx, vy) — the next "
                "state; `p` is the current state, x/y in [0,1]"}
    uniforms = manifest.get("uniforms") or {}
    bad = [k for k in uniforms if k.startswith("u_")]
    if bad:
        return {"ok": False, "error": f"uniform keys must be bare names, not {bad}"}

    # compile-test the transform-feedback + draw programs now (agent feedback).
    try:
        _particles.compile_particle_programs(glsl, uniforms)
    except Exception as e:
        return {"ok": False, "compile_error": str(e)}

    node_id = next_agent_id()
    params: dict = {
        "count":      {"default": 20000, "min": 100, "max": 200000,
                       "description": "particle count (realloc on change)"},
        "point_size": {"default": 3.0, "min": 1, "max": 64,
                       "description": "point sprite size (px)"},
        "dt":         {"default": 1.0, "min": 0.0, "max": 4.0,
                       "description": "simulation step scale"},
        "color":      {"glsl": "color", "default": "#99ccff",
                       "description": "particle colour"},
        "emit_particles": {"choices": ["false", "true"], "default": "false",
                       "description": "also output the (N,4) PARTICLES buffer "
                                      "(adds a GPU→CPU readback)"},
    }
    for uname, spec in uniforms.items():
        params[uname] = _param_from_uniform(spec)

    inputs = {"point_size": "SCALAR", "dt": "SCALAR"}
    for uname, spec in uniforms.items():
        if spec.get("glsl", "float") in ("float", "int"):
            inputs[uname] = "SCALAR"

    @_method_dec(
        id=node_id, name=name, category="particles",
        new_image_contract=True, is_time_varying=True,
        inputs=inputs, outputs={"image": "IMAGE", "particles": "PARTICLES"},
        params=params, tags=["gpu", "particles", "transform-feedback"],
        description=manifest.get("description", ""),
    )
    def _fn(out_dir, seed, params=None, _body=glsl, _uspec=uniforms):
        params = params or {}
        count = int(params.get("count", 20000) or 20000)
        emit = str(params.get("emit_particles", "false")).lower() in ("true", "1", "yes")
        cw, ch = get_canvas()
        arr, parts = _particles.render_particles(
            str(out_dir), _body, _uspec, params, count, cw, ch,
            seed=int(seed), point_size=float(params.get("point_size", 3.0)),
            color=_parse_color(params.get("color", "#99ccff")),
            dt=float(params.get("dt", 1.0)), emit=emit)
        out = {"image": arr}
        if parts is not None:
            out["particles"] = parts
        return out

    _fn.__name__ = f"particles_{node_id}"
    clear_node_defs_cache()
    meta = registry.get_meta(node_id)
    return {
        "ok": True, "id": node_id, "name": name, "type": "particles",
        "ports": {"inputs": dict(meta.inputs or {}), "outputs": dict(meta.outputs)},
        "params": dict(meta.params),
        "hint": "GPU transform-feedback particles. Body sets out_p=vec4(x,y,vx,vy), "
                "x/y in [0,1]; `p`=current state, id=gl_VertexID. Uniforms: "
                "u_time, u_dt, u_count, u_resolution + your typed ones; helpers "
                "hash11(f)/hash21(f). Reseeds when frame==0.",
    }
# ── graph wiring (validated) ──────────────────────────────────────────
# The raw /api/graph/{gid}/patch appends edges with zero checking, so an agent
# wiring blind produces type-mismatched edges that only fail at cook time. These
# helpers validate a connection against the live node-defs + port-type registry
# BEFORE it lands, and return an error the agent can read and fix.

def _port_compatible(src_type: str, dst_type: str) -> bool:
    """True if an output of src_type may feed an input of dst_type.

    Same rule the type system encodes: identical types match; ANY is a
    wildcard; otherwise the dst port type's accepts_from list may whitelist the
    src type (e.g. SCALAR accepts_from=['IMAGE'] — an image reduces to a scalar).
    """
    s, d = src_type.lower(), dst_type.lower()
    if s == d or d == "any" or s == "any":
        return True
    from .core.port_types import get_port_type
    spec = get_port_type(d.upper())
    return bool(spec and s.upper() in {a.upper() for a in spec.accepts_from})


def validate_connection(src_method_id: str, src_port: str,
                        dst_method_id: str, dst_port: str) -> dict:
    """Check an edge against the live node-defs. {ok:True, src_type, dst_type}
    or {ok:False, error:...} naming the available ports on a miss."""
    from .core.graph import get_all_node_defs
    defs = get_all_node_defs()
    if src_method_id not in defs:
        return {"ok": False, "error": f"unknown source node type {src_method_id!r}"}
    if dst_method_id not in defs:
        return {"ok": False, "error": f"unknown target node type {dst_method_id!r}"}
    souts = defs[src_method_id].get("outputs", {}) or {}
    dins = defs[dst_method_id].get("inputs", {}) or {}
    if src_port not in souts:
        return {"ok": False, "error":
                f"{src_method_id} has no output port {src_port!r}; outputs: {list(souts)}"}
    if dst_port not in dins:
        return {"ok": False, "error":
                f"{dst_method_id} has no input port {dst_port!r}; inputs: {list(dins)}"}
    st, dt = souts[src_port], dins[dst_port]
    if not _port_compatible(st, dt):
        return {"ok": False, "error":
                f"type mismatch: {src_method_id}.{src_port}:{st} -> "
                f"{dst_method_id}.{dst_port}:{dt}"}
    return {"ok": True, "src_type": st, "dst_type": dt}


def plan_wire(doc: dict, add_nodes: list[dict] | None,
              connect: list[dict] | None) -> dict:
    """Validate an atomic wiring intent against a graph doc WITHOUT mutating it.

    add_nodes: [{"ref": "a", "method_id": "9000", "params": {...},
                 "x": .., "y": .., "render": bool,
                 "feedback_self": bool}]  — ref is a local alias.
                 feedback_self adds a self-feedback edge (image -> image_in,
                 feedback=True) so a "feedback"-type sim node ping-pongs on its
                 own previous frame.
    connect:   [{"src": "a"|<existing node id>, "src_port": "image",
                 "dst": "b"|<existing node id>, "dst_port": "image_in",
                 "feedback": bool}]

    Returns {ok:True, nodes:[node dicts], edges:[edge dicts], ref_ids:{ref:id}}
    ready to append to doc, or {ok:False, errors:[...]} — all-or-nothing, so the
    agent fixes and retries instead of leaving a half-wired graph.
    """
    from .core import registry
    errors: list[str] = []
    ref_ids: dict[str, str] = {}
    new_nodes: list[dict] = []
    self_feedback_refs: list[str] = []  # node ids that get an auto self-feedback edge

    for spec in add_nodes or []:
        mid = str(spec.get("method_id", ""))
        if registry.get_meta(mid) is None and mid not in _threejs_ids():
            errors.append(f"add_nodes: unknown method_id {mid!r}")
            continue
        nid = spec.get("id") or f"n{uuid.uuid4().hex[:8]}"
        if "ref" in spec:
            ref_ids[spec["ref"]] = nid
        new_nodes.append({
            "id": nid, "method_id": mid,
            "params": dict(spec.get("params", {})),
            "x": float(spec.get("x", 0.0)), "y": float(spec.get("y", 0.0)),
            "render": bool(spec.get("render", False)),
        })
        # Sugar: a feedback sim node ping-ponging on its own previous frame.
        if spec.get("feedback_self"):
            self_feedback_refs.append(nid)

    # method_id lookup across new nodes (by ref/id) and existing doc nodes
    method_of: dict[str, str] = {n["id"]: n["method_id"] for n in doc.get("nodes", [])}
    for n in new_nodes:
        method_of[n["id"]] = n["method_id"]
    resolve = {**ref_ids, **{nid: nid for nid in method_of}}

    new_edges: list[dict] = []
    for c in connect or []:
        src_id = resolve.get(c.get("src"))
        dst_id = resolve.get(c.get("dst"))
        if not src_id or src_id not in method_of:
            errors.append(f"connect: unknown src {c.get('src')!r}"); continue
        if not dst_id or dst_id not in method_of:
            errors.append(f"connect: unknown dst {c.get('dst')!r}"); continue
        v = validate_connection(method_of[src_id], c.get("src_port", ""),
                                method_of[dst_id], c.get("dst_port", ""))
        if not v["ok"]:
            errors.append(f"connect: {v['error']}"); continue
        new_edges.append({
            "id": f"e{uuid.uuid4().hex[:8]}",
            "src_node": src_id, "src_port": c["src_port"],
            "dst_node": dst_id, "dst_port": c["dst_port"],
            "feedback": bool(c.get("feedback", False)),
        })

    # Auto self-feedback edges for feedback_self nodes: image -> own image_in.
    for nid in self_feedback_refs:
        v = validate_connection(method_of[nid], "image", method_of[nid], "image_in")
        if not v["ok"]:
            errors.append(f"feedback_self: {v['error']} (node needs an image_in "
                          f"input — author it as type 'feedback' or 'filter')")
            continue
        new_edges.append({
            "id": f"e{uuid.uuid4().hex[:8]}",
            "src_node": nid, "src_port": "image",
            "dst_node": nid, "dst_port": "image_in",
            "feedback": True,
        })

    if errors:
        return {"ok": False, "errors": errors}
    return {"ok": True, "nodes": new_nodes, "edges": new_edges, "ref_ids": ref_ids}


def _threejs_ids() -> set[str]:
    try:
        from .core.threejs_nodes import THREEJS_3D_NODE_DEFS
        return set(THREEJS_3D_NODE_DEFS)
    except Exception:
        return set()


def unregister_node_type(node_id: str) -> dict:
    """Drop an agent-authored node and free its shader-library slot."""
    from .core import registry
    from .core.shaders import SHADERS, _get_prog_cache
    from .core.graph import clear_node_defs_cache

    if registry.get_meta(node_id) is None:
        return {"ok": False, "error": f"no node with id {node_id!r}"}

    registry.unregister(node_id)
    shader_key = f"agent_{node_id}"
    SHADERS.pop(shader_key, None)
    # Best-effort: drop this thread's cached program for the key. Other threads'
    # caches self-heal on their next cook (the SHADERS entry is already gone).
    try:
        _get_prog_cache().pop(shader_key, None)
    except Exception:
        pass
    clear_node_defs_cache()
    return {"ok": True, "id": node_id}


def render_node(node_id: str, params: dict | None = None,
                out_dir: str | Path | None = None) -> dict:
    """Cook a single authored node once and report where the frame landed.

    Proves an authored node actually executes. Full graph wiring uses the
    existing /api/graph/{gid}/execute path — this is the single-node probe.
    """
    import tempfile
    from .core import registry

    meta = registry.get_meta(node_id)
    if meta is None:
        return {"ok": False, "error": f"no node with id {node_id!r}"}
    out = Path(out_dir or tempfile.mkdtemp(prefix="agent_render_"))
    out.mkdir(parents=True, exist_ok=True)

    result: Any = meta.fn(out, 0, params=params or {})
    if isinstance(result, dict) and "image" in result:  # new_image_contract
        img = result["image"]
        return {"ok": True, "id": node_id, "shape": list(img.shape),
                "dtype": str(img.dtype), "in_memory": True}
    # legacy contract writes a PNG to out_dir
    png = out / meta.filename()
    return {"ok": True, "id": node_id, "path": str(png), "exists": png.exists()}
