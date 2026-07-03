# GRILLMASTER Command Center

A **node-based generative image & video editor** that takes the best of Houdini and TouchDesigner and infuses it with LLM agents:

- **Houdini's data model** — every node produces a typed, named-attribute payload (IMAGE, SCALAR, FIELD, PARTICLES, MASK, COLORMAP). Wires carry named attributes, not blobs; downstream nodes pick up what they need by name. Non-destructive, procedural, fully on-disk and auditable.
- **TouchDesigner's live instinct** — a 📺 Live mode cooks the graph continuously (~30 fps for light graphs) and streams frames to the editor over MJPEG; CHOP-style channel nodes (LFO, Counter, Beats, Envelope, Math, Logic…) drive parameters over time; dirty-flag selective recooking keeps interactive tweaks fast. Real-time rendering is the optimization target. The live-loop architecture is a load-bearing milestone with documented non-regression invariants — see `DESIGN.md` → "Live mode" and `image_pipeline/tests/test_live_regression.py`.
- **LLM-infused evolution** — the pipeline is designed to be read, extended, and repaired by agents. The built-in **Node Doctor** (backed by the **Hermes agent, the sole LLM backend**) chats about any node with its source in context, rewrites it, and hot-reloads it into the running editor; the node tester finds broken methods and batch-applies fixes. The tool evolves continuously with user input.

180+ generative methods ship in the library: physics and biology simulations (reaction-diffusion, boids, physarum, fluid instabilities, N-body…), fractals, patterns, math art, filters, compositing nodes, and CLI-tool wrappers.

## Running

```bash
uv venv .venv && uv pip install -r requirements.txt --python .venv/bin/python
.venv/bin/python -m image_pipeline.server            # http://localhost:7860
```

The editor is a single-page app served at `/`. Build a graph (Tab or right-click opens the node picker), wire typed ports, hit **Run** for a frame or a sequence, or **📺 Live** for the continuous loop.

### Configuration

| Env var | Purpose |
|---|---|
| `HERMES_AGENT_DIR` | Hermes agent install for Node Doctor (default `~/.hermes/hermes-agent`) |
| `HERMES_PYTHON` | Override the exact interpreter for the Hermes runner |
| `GRILLMASTER_API_TOKEN` | When set, mutating endpoints require the `X-Api-Token` header — set this whenever you tunnel the server (`--tunnel`). Put the token in the UI's `localStorage['api-token']`. |

## Repository layout

```
image_pipeline/
  core/        executor (graph.py), registry, port types, timeline, utils
  methods/     the node library — one file per method, grouped by category
  server.py    FastAPI app: node defs, graph execution, SSE + MJPEG streaming,
               Node Doctor, node tester, sequences
ui/index.html  the entire editor frontend (single file)
chord_bot/     an independent chord-progression node system (music domain),
               mounted at /chordbot
tools/         audit_methods.py (contract enforcement, pre-commit), next_id.py
DESIGN.md      authoritative architecture document — read this first
AGENT_GUIDE.md the method-file contract for anyone (human or agent) adding nodes
```

## Extending it

Read `AGENT_GUIDE.md` before touching any method file — it is the contract. The short version: get an ID from `tools/next_id.py`, declare every output you write, produce an image on every code path, seed everything stochastic, and keep it legible. `tools/audit_methods.py --fail-on-violations` (pre-commit) enforces the contract.

Current state of the codebase, known gaps, and the remediation roadmap: `CODE_AUDIT_2026-07-02.md` and `docs/plans/2026-07-02-audit-remediation-plan.md`.
