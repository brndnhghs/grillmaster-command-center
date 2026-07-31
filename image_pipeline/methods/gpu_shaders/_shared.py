"""
GPU shader nodes — @method factories + registration for every GLSL shader node.

Node REGISTRATION DATA lives in sibling modules (one concern per file):
  * shader_nodes.py — _PROC_SHADERS / _FILT_SHADERS / _TYPED_SHADER_NODES /
    _PROC_PARAMS / _FILT_PARAMS (the node tables; add a node = add a tuple)
  * client_shims.py — CLIENT_GPU_SHIMS / CLIENT_GPU_SIMS (CPU-node live-preview
    routes) and GPU_PREVIEW_DROP_ALLOW + is_param_justified_drop (coverage
    contract)

This module holds the only logic: the @method factories, the registration
loops, and the derived GPU_SHADER_NODE_MAP. Procedural shaders generate
imagery from scratch; filter shaders consume _input_image and return the
modified image. The legacy combined method #82 is kept for backward
compatibility with existing graphs that reference it by ID.

All methods are tagged "gpu" so the ⚡ badge renders in the palette.
Filter methods set new_image_contract=True — the executor skips _input.png
writes and they receive the upstream ndarray directly as params["_input_image"].
"""
from __future__ import annotations
from pathlib import Path

import numpy as np

from ...core.registry import method
from ...core.utils import get_canvas
from ...core.shaders import render_shader, SHADERS, shader_uses_time

from .shader_nodes import (
    _PROC_SHADERS,
    _FILT_SHADERS,
    _TYPED_SHADER_NODES,
    _PROC_PARAMS,
    _FILT_PARAMS,
)
from .client_shims import (
    CLIENT_GPU_SHIMS,
    CLIENT_GPU_SIMS,
    GPU_PREVIEW_DROP_ALLOW,
    is_param_justified_drop,
)


# ── Factory: procedural ───────────────────────────────────────────────

def _make_proc(method_id: str, shader_name: str, method_name: str):
    @method(id=method_id, name=method_name, category="gpu_shaders",
            new_image_contract=True,
            tags=["gpu", "fast"],
            is_time_varying=shader_uses_time(shader_name),
            params=_PROC_PARAMS)
    def _fn(out_dir: Path, seed: int, params=None):
        if params is None:
            params = {}
        t = float(params.get("time", 0.0))
        p = tuple(float(params.get(f"p{i}", 0.5)) for i in range(1, 5))
        cw, ch = get_canvas()
        img = render_shader(shader_name, (cw, ch), p, t)
        arr = np.array(img, dtype=np.uint8)
        # Return dict: executor captures image directly; no disk write needed in live mode.
        # Disk mode: executor writes the output PNG at graph.py:891 when in_memory=False.
        return {"image": arr.astype(np.float32) / 255.0}

    _fn.__name__ = f"gpu_proc_{shader_name}"
    return _fn


# ── Factory: filter ───────────────────────────────────────────────────

def _make_filt(method_id: str, shader_name: str, method_name: str):
    @method(id=method_id, name=method_name, category="gpu_shaders",
            new_image_contract=True,
            inputs={"image_in": "IMAGE"},
            tags=["gpu", "fast"],
            is_time_varying=shader_uses_time(shader_name),
            params=_FILT_PARAMS)
    def _fn(out_dir: Path, seed: int, params=None):
        if params is None:
            params = {}
        inp = params.get("_input_image")  # float32 [0,1] or None
        t = float(params.get("time", 0.0))
        strength = float(params.get("strength", 0.5))
        p2 = float(params.get("p2", 0.5))
        p = (strength, p2, 0.5, 0.5)
        cw, ch = get_canvas()
        img = render_shader(shader_name, (cw, ch), p, t, inp)
        arr = np.array(img, dtype=np.uint8)
        return {"image": arr.astype(np.float32) / 255.0}

    _fn.__name__ = f"gpu_filt_{shader_name}"
    return _fn


# ── Typed-uniform shader nodes (ids 220+) ─────────────────────────────
# These shaders declare named, typed variables (core/shaders.py `uniforms=`)
# instead of the generic p1..p4 vec4. The factory turns every declared
# variable into:
#   • a real node param — slider (float/int), color picker (color, '#rrggbb'
#     default renders a swatch in the UI), or dropdown (choice), AND
#   • a wireable, data-typed SCALAR input port (float/int uniforms), so any
#     scalar output (LFO, luminance mean, counter, …) can drive the variable.
# Inputs/outputs are explicitly data-typed: filters take image_in: IMAGE;
# every node emits image: IMAGE + luminance: FIELD.

def _param_from_uniform(spec: dict) -> dict:
    """Node param spec from a typed uniform spec (same shape the UI expects)."""
    gtype = spec.get("glsl", "float")
    p: dict = {"description": spec.get("description", "")}
    if gtype == "choice":
        p["choices"] = list(spec.get("choices", []))
        p["default"] = spec.get("default", p["choices"][0] if p["choices"] else "")
    elif gtype == "color":
        p["default"] = spec.get("default", "#ffffff")
    else:  # float / int — slider with the uniform's declared range
        if "min" in spec:
            p["min"] = spec["min"]
        if "max" in spec:
            p["max"] = spec["max"]
        p["default"] = spec.get("default", 0)
    return p


def _make_typed(method_id: str, shader_name: str, method_name: str):
    info = SHADERS[shader_name]
    uspec: dict = info.get("uniforms") or {}
    is_filter = info["type"] == "filter"

    params = {uname: _param_from_uniform(spec) for uname, spec in uspec.items()}

    inputs: dict[str, str] = {}
    if is_filter:
        inputs["image_in"] = "IMAGE"
    for uname, spec in uspec.items():
        if spec.get("glsl", "float") in ("float", "int"):
            inputs[uname] = "SCALAR"

    @method(id=method_id, name=method_name, category="gpu_shaders",
            new_image_contract=True,
            inputs=inputs,
            outputs={"image": "IMAGE", "luminance": "FIELD"},
            tags=["gpu", "fast", "typed-uniforms"],
            is_time_varying=shader_uses_time(shader_name),
            description=info.get("description", ""),
            params=params)
    def _fn(out_dir: Path, seed: int, params=None,
            _shader=shader_name, _uspec=uspec, _is_filter=is_filter):
        if params is None:
            params = {}
        t = float(params.get("time", 0.0))
        named = {u: params.get(u, spec.get("default"))
                 for u, spec in _uspec.items()}
        inp = params.get("_input_image") if _is_filter else None
        cw, ch = get_canvas()
        img = render_shader(_shader, (cw, ch), (0.5, 0.5, 0.5, 0.5), t, inp,
                            named_params=named)
        arr = np.array(img, dtype=np.uint8)
        return {"image": arr.astype(np.float32) / 255.0}

    _fn.__name__ = f"gpu_typed_{shader_name}"
    return _fn


# ── Register all shaders ──────────────────────────────────────────────

for _mid, _sname, _mname in _PROC_SHADERS:
    # Typed-uniform shaders (those with a `uniforms=` spec) get named params +
    # wireable SCALAR ports; the rest keep the legacy generic-p1..p4 path.
    if SHADERS.get(_sname, {}).get("uniforms"):
        _make_typed(_mid, _sname, _mname)
    else:
        _make_proc(_mid, _sname, _mname)

for _mid, _sname, _mname in _FILT_SHADERS:
    if SHADERS.get(_sname, {}).get("uniforms"):
        _make_typed(_mid, _sname, _mname)
    else:
        _make_filt(_mid, _sname, _mname)

for _mid, _sname, _mname in _TYPED_SHADER_NODES:
    _make_typed(_mid, _sname, _mname)


# ── Node → shader map for client-side rendering (parity layer / feature #1) ──
# Lets the browser executor render these EXISTING server nodes client-side for
# the live preview, from the same GLSL source (see core/shaders.py). The server
# remains authoritative for one-shot Run and export.
GPU_SHADER_NODE_MAP: dict[str, dict] = {}
# 173-197 are registered as typed-uniform nodes (each shader declares named
# variables → real params + wireable SCALAR ports + IMAGE/FIELD outputs).
for _mid, _sname, _mname in _PROC_SHADERS:
    GPU_SHADER_NODE_MAP[_mid] = {"shader": _sname,
                                 "type": SHADERS[_sname]["type"], "typed": True}
for _mid, _sname, _mname in _FILT_SHADERS:
    GPU_SHADER_NODE_MAP[_mid] = {"shader": _sname, "type": "filter",
                                 "typed": bool(SHADERS.get(_sname, {}).get("uniforms"))}
for _mid, _sname, _mname in _TYPED_SHADER_NODES:
    # typed: client sets u_<name> uniforms from node params (no p1..p4).
    GPU_SHADER_NODE_MAP[_mid] = {"shader": _sname,
                                 "type": SHADERS[_sname]["type"], "typed": True}

# P0 client-GPU shims + P1 sim shims (defined in client_shims.py) route
# EXISTING CPU nodes' live previews to GLSL twins. Merged into the map so the
# /api/shader-sources endpoint serves them; CPU numpy fns stay authoritative
# for export (two-tier precision).
GPU_SHADER_NODE_MAP.update(CLIENT_GPU_SHIMS)
GPU_SHADER_NODE_MAP.update(CLIENT_GPU_SIMS)


# ── Shader names for the legacy combined method #82 ─────────────────
SHADER_NAMES = sorted([k for k, v in SHADERS.items() if v["type"] == "procedural"])
