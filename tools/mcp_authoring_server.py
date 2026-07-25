"""MCP server — lets an LLM author & wire GPU nodes in the LIVE image_pipeline.

This is the agent-native surface. An assistant connected here can invent a new
node type at runtime (GLSL body + typed-uniform manifest), connect authored
nodes into the running graph (with port/type validation), render, and tear
down — no file edit, no restart. New nodes and edges appear in the open editor
in real time via the server's node-defs-updated / graph:patch events.

It is a thin HTTP client over the running FastAPI server (image_pipeline.server),
so every call targets the SAME live registry and graph the user sees. Start the
pipeline server first (default http://127.0.0.1:7860); override with
GRILLMASTER_URL. If GRILLMASTER_API_TOKEN is set on the server, set it here too.

Run:
    .venv/bin/python tools/mcp_authoring_server.py

Register with Claude Code:
    claude mcp add grillmaster-authoring -- \
        /ABS/PATH/.venv/bin/python /ABS/PATH/tools/mcp_authoring_server.py
"""
from __future__ import annotations

import os

import httpx
from mcp.server.fastmcp import FastMCP

BASE = os.environ.get("GRILLMASTER_URL", "http://127.0.0.1:7860").rstrip("/")
_TOKEN = os.environ.get("GRILLMASTER_API_TOKEN", "")
_HEADERS = {"x-api-token": _TOKEN} if _TOKEN else {}

mcp = FastMCP("grillmaster-authoring")


def _req(method: str, path: str, **kw) -> dict:
    try:
        r = httpx.request(method, f"{BASE}{path}", headers=_HEADERS, timeout=120, **kw)
    except httpx.ConnectError:
        return {"ok": False, "error": f"cannot reach pipeline server at {BASE} — "
                                      f"start it (python -m image_pipeline.server) "
                                      f"or set GRILLMASTER_URL"}
    if r.status_code >= 400:
        return {"ok": False, "error": f"HTTP {r.status_code}: {r.text[:300]}"}
    try:
        return r.json()
    except Exception:
        return {"ok": True, "raw": r.text[:300]}


@mcp.tool()
def get_catalog(agent_only: bool = True) -> dict:
    """List node types the agent can wire, plus live port types. Read this first.

    agent_only=True (default) hides the 354 built-in method nodes but KEEPS
    agent-authored nodes (id >= 9000) AND source/input nodes (category io/input:
    Camera, Video Import, Image Import, Text Source) so live inputs are wireable.
    Pass agent_only=False to see everything."""
    defs = _req("GET", "/api/node-defs")
    ports = _req("GET", "/api/port-types")
    nodes = {}
    for mid, nd in (defs.items() if isinstance(defs, dict) else []):
        is_agent = mid.isdigit() and int(mid) >= 9000
        is_source = nd.get("category") in ("io", "input")
        if agent_only and not (is_agent or is_source):
            continue
        nodes[mid] = {"name": nd.get("name", ""), "category": nd.get("category", ""),
                      "inputs": nd.get("inputs", {}), "outputs": nd.get("outputs", {}),
                      "params": nd.get("params", {})}
    return {"port_types": ports, "nodes": nodes}


@mcp.tool()
def get_graph(gid: str = "active") -> dict:
    """Read the live graph document (nodes, edges, canvas) so you can decide
    what to add and where to connect."""
    return _req("GET", f"/api/graph/{gid}")


@mcp.tool()
def register_node_type(name: str, type: str = "procedural", glsl: str = "",
                       description: str = "", uniforms: dict | None = None,
                       expr: str = "", vars: dict | None = None) -> dict:
    """Author a new node at runtime. Four body kinds via `type`:

    GPU shader kinds (use `glsl` + `uniforms`) — `glsl` is a raw
    `void main(){ ... }` body; the prologue (#version, u_resolution, u_time,
    u_params, u_texture, noise helpers) and one `uniform <t> u_<key>` per
    uniform are injected, do NOT write them:
      • "procedural" — generates imagery from scratch.
      • "filter"     — processes an upstream image (image_in / u_texture).
      • "feedback"   — a simulation: u_texture is the node's OWN previous frame
                       (reaction-diffusion, life, growth, feedback canvas). Seed
                       when u_time < 1.0; wire it with feedback_self:true.
    uniforms: {"<bare_key>":{"glsl":"float|int|color|choice","min":..,"max":..,
              "default":..,"description":..}}; key "gain" -> u_gain + a node
              param + (float/int) a wireable SCALAR port.
    On a GLSL error returns {ok:false, compile_error:...} — read it, fix, retry.

    Particle kind ("particles", use `glsl` + `uniforms`) — a GPU
    transform-feedback particle system (macOS GL 4.1 has no compute shaders).
    `glsl` is a vertex-shader UPDATE body that sets `out_p = vec4(x,y,vx,vy)`
    (next state) from `p` (current state; x/y in [0,1]); available: u_time,
    u_dt, u_count, u_resolution + your typed uniforms, id=gl_VertexID, helpers
    hash11(f)/hash21(f). Outputs IMAGE (additive soft points) + PARTICLES (N,4).
    Node params: count/point_size/dt/color/emit_particles. Reseeds on frame 0.
    Expression kind (use `expr` + optional `vars`) — a SCALAR-output CPU math
    node (LFOs, envelopes) to modulate any wireable param:
      • expr: safe math over `t` (frame time) and your free variables, e.g.
        "0.5 + 0.5*sin(t*2.0 + phase)". Functions: sin/cos/tan/sqrt/abs/floor/
        ceil/round/log/pow/min/max/noise; if/else; comparisons. Each free var
        becomes a param + a wireable SCALAR input; output is on port "value".
      • vars: {"phase":{"default":0.0,"min":..,"max":..,"description":..}} (opt).
    On a bad expression returns {ok:false, expr_error:...}.
    """
    return _req("POST", "/api/nodes/author", json={
        "name": name, "type": type, "glsl": glsl, "description": description,
        "uniforms": uniforms or {}, "expr": expr, "vars": vars or {}})


@mcp.tool()
def validate_edge(src_method_id: str, src_port: str,
                  dst_method_id: str, dst_port: str) -> dict:
    """Type-check one connection against the live node-defs without touching the
    graph. Use it to confirm ports before wiring."""
    return _req("POST", "/api/graph/validate-edge", json={
        "src_method_id": src_method_id, "src_port": src_port,
        "dst_method_id": dst_method_id, "dst_port": dst_port})


@mcp.tool()
def wire_graph(add_nodes: list[dict] | None = None,
               connect: list[dict] | None = None, gid: str = "active") -> dict:
    """Atomically add authored nodes AND connect them in the live graph, with
    port/type validation up front (all-or-nothing).

    add_nodes: [{"ref":"a","method_id":"9000","params":{...},"x":0,"y":0,
                 "render":false}] — ref is a local alias.
    connect:   [{"src":"a"|<node id>,"src_port":"image",
                 "dst":"b"|<node id>,"dst_port":"image_in"}]

    On success the nodes+edges land in the running graph and the editor repaints.
    On any error nothing is applied and {ok:false, errors:[...]} comes back."""
    return _req("POST", f"/api/graph/{gid}/wire",
                json={"add_nodes": add_nodes or [], "connect": connect or []})


@mcp.tool()
def render_node(node_id: str, params: dict | None = None) -> dict:
    """Cook a single authored node once (proves it executes)."""
    return _req("POST", "/api/nodes/render", json={"node_id": node_id, "params": params})


@mcp.tool()
def unregister_node_type(node_id: str) -> dict:
    """Remove an agent-authored node and free its shader slot."""
    return _req("DELETE", f"/api/nodes/{node_id}")


@mcp.tool()
def register_port_type(name: str, color: str, description: str,
                       accepts_from: list[str] | None = None) -> dict:
    """Declare a new semantic port type at runtime (a new routable data type).
    color is a UI hex like "#4a9eff"; accepts_from whitelists source types that
    may feed this one (e.g. ["IMAGE"])."""
    return _req("POST", "/api/port-types", json={
        "name": name, "color": color, "description": description,
        "accepts_from": accepts_from or []})


if __name__ == "__main__":
    mcp.run()  # stdio transport
