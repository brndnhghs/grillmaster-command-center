"""Prompt assembly for the tuning builder (Graph Smith) and the lesson distiller.

The builder system prompt is the novel core: role + graph-JSON contract + port
rules + the catalog digest + the accumulated playbook + few-shots. The playbook
is what makes the agent improve across sessions — it is injected verbatim.
"""
from __future__ import annotations

import json

# Two compact, real-shaped examples (kept inline so prompting is hermetic and
# self-documenting — no disk dependency). One single-node terminal, one with a
# driver chain feeding a param via a SCALAR wire.
_EXAMPLE_SINGLE = {
    "version": 1,
    "name": "spark",
    "nodes": [
        {"id": "n1", "method_id": "18",
         "params": {"rule": "conway", "seed_pattern": "spark_center",
                    "density": 0.3, "color": "mono", "size": 4},
         "render": True},
    ],
    "edges": [],
}

_EXAMPLE_DRIVEN = {
    "version": 1,
    "name": "rule_cycle",
    "nodes": [
        {"id": "n1", "method_id": "__counter__",
         "params": {"mode": "loop", "end": 15}, "render": False},
        {"id": "n2", "method_id": "__math__",
         "params": {"operation": "map_range", "map_src_min": 0, "map_src_max": 15,
                    "map_dst_min": 0, "map_dst_max": 1}, "render": False},
        {"id": "n3", "method_id": "18",
         "params": {"rule": "conway", "seed_pattern": "random",
                    "density": 0.3, "color": "mono", "size": 4}, "render": True},
    ],
    "edges": [
        {"src_node": "n1", "src_port": "value", "dst_node": "n2", "dst_port": "a"},
        {"src_node": "n2", "src_port": "value", "dst_node": "n3", "dst_port": "rule_select"},
    ],
}

_CONTRACT = """\
You are GRAPH SMITH. You translate an image brief into a valid node-graph for
this generative pipeline. The user describes an image (or a change to one); you
choose nodes from the catalog, set their params, and wire them into a graph that
produces that image.

HARD RULES for the graph you emit:
- It is JSON: {"version":1, "name":"<short-slug>", "nodes":[...], "edges":[...]}.
- Each node: {"id":"n<N>", "method_id":"<from catalog>", "params":{...},
  "render":<bool>}. You MAY omit x/y — they are auto-laid-out.
- Exactly ONE node has "render":true, and it MUST be an IMAGE node (a generator
  or filter from the IMAGE NODES section). That is the terminal — the image the
  user sees is whatever it outputs.
- Edges: {"src_node","src_port","dst_node","dst_port"}. An edge is legal only if
  the source's output port type is accepted by the destination's input port. Use
  the in:/out: types in the catalog. DRIVERS (SCALAR) wire into a node's param
  ports to animate them; SOURCES (FIELD/MASK/PARTICLES/COLORMAP) feed filters and
  compositors; IMAGE flows into filter image inputs / compositor image_a/image_b.
- params is a dict keyed by the exact param name from the catalog. Only set params
  you mean to change; unset params take their defaults.
- Prefer a SMALL graph that clearly serves the brief (2–5 nodes typical). Do not
  add nodes that don't contribute to the described image.
- For motion briefs (sparkle, rotate, drip, warp), pick nodes/params that evolve
  over time and/or add a DRIVER (LFO/Counter/Math) into an animatable param.

RESPONSE FORMAT:
- First, 1–3 sentences of rationale: which nodes you chose and why they serve the
  brief. Reference the playbook if it applies.
- Then the graph as a SINGLE fenced ```json block. Nothing after it.
"""


def build_system(catalog_digest: str, playbook: str) -> str:
    """Assemble the full builder system prompt."""
    return f"""{_CONTRACT}

=== EXAMPLES (shape only — not templates to copy) ===
A single-node terminal:
```json
{json.dumps(_EXAMPLE_SINGLE)}
```
A driver chain feeding a param over a SCALAR wire:
```json
{json.dumps(_EXAMPLE_DRIVEN)}
```

=== NODE-CRAFT PLAYBOOK (what you have learned so far — USE THIS) ===
{playbook}

=== NODE CATALOG (method_id  Name — desc / in:out / params) ===
{catalog_digest}
"""


def build_user(brief: str) -> str:
    return (f"BRIEF: {brief}\n\n"
            f"Build a graph that produces this image. Reply with rationale then "
            f"the ```json graph.")


def revise_user(brief: str, current_graph: dict,
                critique_history: list[str], critique: str) -> str:
    hist = ""
    if critique_history:
        hist = "PRIOR CRITIQUES:\n" + "\n".join(f"- {c}" for c in critique_history) + "\n\n"
    return (
        f"ORIGINAL BRIEF: {brief}\n\n"
        f"{hist}"
        f"CURRENT GRAPH:\n```json\n{json.dumps(current_graph)}\n```\n\n"
        f"NEW CRITIQUE (what to fix): {critique}\n\n"
        f"Revise the graph to address the critique while still serving the brief. "
        f"Keep what works; change what the critique calls out. Reply with rationale "
        f"then the full revised ```json graph."
    )


# ── Lesson distiller ──────────────────────────────────────────────────
_LEARN_SYSTEM = """\
You are the LESSON DISTILLER for a node-graph image pipeline. Given a brief, the
graph that was built, and the user's rating + critique, write ONE durable,
GENERAL, reusable lesson about the node pipeline — the kind of craft knowledge
that would help build a *different* image better next time.

RULES:
- One or two sentences. Concrete. Name the node(s) and param(s) involved.
- Generalize beyond this exact brief ("For warping backgrounds, Domain Warp
  driven by an LFO on `amount` reads as liquid; keep amount < 0.4 or overlaid
  text becomes unreadable") — not "this graph was rated 2 stars".
- If the attempt FAILED, capture the failure mode and the fix direction.
- Also output a short SECTION name to file it under (an effect/theme like
  "Warping backgrounds", "Particles", "Text overlay", "Color / palette").

RESPONSE FORMAT (exactly):
SECTION: <section name>
LESSON: <the one lesson>
"""


def learn_system() -> str:
    return _LEARN_SYSTEM


def learn_user(brief: str, graph: dict, rating: int, critique: str) -> str:
    return (
        f"BRIEF: {brief}\n\n"
        f"GRAPH:\n```json\n{json.dumps(graph)}\n```\n\n"
        f"RATING: {rating}/5\n"
        f"CRITIQUE: {critique}\n\n"
        f"Distill one durable lesson. Reply in the SECTION/LESSON format."
    )


def parse_lesson(text: str) -> tuple[str, str]:
    """Parse the distiller's `SECTION:/LESSON:` reply. Falls back gracefully."""
    section, lesson = "General craft", ""
    for line in text.splitlines():
        s = line.strip()
        if s.upper().startswith("SECTION:"):
            section = s.split(":", 1)[1].strip() or section
        elif s.upper().startswith("LESSON:"):
            lesson = s.split(":", 1)[1].strip()
    if not lesson:
        # No structured lesson — use the whole reply as the lesson body.
        lesson = text.strip()
    return section, lesson
