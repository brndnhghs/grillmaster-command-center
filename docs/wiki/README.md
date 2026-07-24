# Grillmaster Command Center

Grillmaster Command Center is a local-first generative art and music studio. The **Image Pipeline** turns parameterised node-graphs into images, animations, and videos; **Chord Bot** composes chord progressions as left-to-right node graphs and exports them as MIDI, text, or JSON. A small **Dashboard** process supervises both services and serves a unified control-panel UI.

The system is designed to run entirely locally on a laptop: no cloud dependencies, no accounts, no build step for the front-end.

## Key Concepts

- **Method** — a single generative algorithm registered with the `@method` decorator (373 methods across 8 categories + top-level files). Each method declares its parameters, input/output ports, and tags.
- **Node Graph** — a directed acyclic graph of method nodes wired together. The executor topologically sorts the graph and runs each node in order, passing image/field/mask data along edges.
- **Architecture A vs B** — Architecture-A methods (simulations) cook an entire frame list internally and are cached by the executor. Architecture-B methods are stateless single-frame generators driven by a timeline or `time` parameter.
- **Live Simulation** — the graph can run continuously, re-reading the shared graph document every frame so edits are absorbed by the running loop without restarting.
- **Chord Bot Graph** — a separate node-graph engine where horizontal nodes advance a beat clock and vertical nodes augment harmonic state; execution is a single pass producing a `SequenceEntry` list.

## Entry Points

- [`image_pipeline/pipeline.py`](https://github.com/brndnhghs/grillmaster-command-center/blob/3e085d44fccca63896b5f6543aaa54ab4216e4b3/image_pipeline/pipeline.py) — CLI batch runner (`python -m image_pipeline.pipeline --all`)
- [`image_pipeline/server.py`](https://github.com/brndnhghs/grillmaster-command-center/blob/3e085d44fccca63896b5f6543aaa54ab4216e4b3/image_pipeline/server.py) — FastAPI server (default `:7860`) serving the node-graph UI and REST/SSE/WebSocket API
- [`chord_bot/server.py`](https://github.com/brndnhghs/grillmaster-command-center/blob/3e085d44fccca63896b5f6543aaa54ab4216e4b3/chord_bot/server.py) — Chord Bot FastAPI server (default `:7861`)
- [`dashboard/__init__.py`](https://github.com/brndnhghs/grillmaster-command-center/blob/3e085d44fccca63896b5f6543aaa54ab4216e4b3/dashboard/__init__.py) — Dashboard supervisor (default `:7870`) that spawns and monitors the two services
- [`scripts/grillmaster-launcher.sh`](https://github.com/brndnhghs/grillmaster-command-center/blob/3e085d44fccca63896b5f6543aaa54ab4216e4b3/scripts/grillmaster-launcher.sh) — one-shot launcher for both services

## High-Level Architecture

The Image Pipeline and Chord Bot are independent FastAPI apps. The Dashboard is a third FastAPI app that spawns each as a child process and proxies their status. The browser UI (`ui/index.html`) talks directly to the Image Pipeline server over REST, Server-Sent Events, and WebSocket; Chord Bot has its own UI under `/chordbot`. A three.js sidecar on `:7862` serves the 3D viewport.

See [architecture.md](architecture.md).

## Module Map

| Module | Purpose |
|--------|---------|
| [`core-graph`](modules/core-graph.md) | Node/edge schema, topological executor, live-dirty tracking |
| [`core-registry`](modules/core-registry.md) | `@method` decorator, `MethodMeta`, auto-discovery |
| [`core-port_types`](modules/core-port_types.md) | Port-type system (IMAGE, FIELD, MASK, SCALAR, TEXT, PARTICLES) |
| [`core-timeline`](modules/core-timeline.md) | Timeline, keyframe tracks, `make_timeline` helper |
| [`core-arch`](modules/core-arch.md) | Architecture-A/B split, sim-cache, parameter hashing |
| [`core-cache`](modules/core-cache.md) | LRU frame cache, selective invalidation |
| [`core-compositing`](modules/core-compositing.md) | Blend-mode compositing (normal, screen, overlay, grid, …) |
| [`core-easing`](modules/core-easing.md) | Easing functions + presets for animated params |
| [`core-expr`](modules/core-expr.md) | Expression evaluator for param strings (`$input.mean` etc.) |
| [`core-node_tester`](modules/core-node_tester.md) | Automated per-method test runner + report |
| [`core-quality`](modules/core-quality.md) | Quality presets (fast / balanced / HQ) |
| [`core-runner`](modules/core-runner.md) | Method runner helper (single-method execution) |
| [`core-utils`](modules/core-utils.md) | Canvas sizing, palette quantisation, save helpers, sidecar protocol |
| [`core-animation`](modules/core-animation.md) | `animate_method`, frame capture, per-job context |
| [`core-postprocess`](modules/core-postprocess.md) | OpenCV filter library (~56 effects: oil, edge, bloom, warp, …) |
| [`core-annotator`](modules/core-annotator.md) | Demo overlay renderer (stamps params onto output images) |
| [`server`](modules/server.md) | FastAPI server: REST, SSE, WebSocket, job queue, live sim, Node Doctor |
| [`methods-library`](modules/methods-library.md) | 373 generative methods across 8 categories + top-level files |
| [`ui-editor`](modules/ui-editor.md) | Browser SPA: method browser, node-graph canvas, 3D viewport, diagnostics |
| [`chord-bot`](modules/chord-bot.md) | Standalone chord-progression node graph + MIDI/text/JSON export |
| [`dashboard`](modules/dashboard.md) | Process supervisor + unified control-panel UI |

## Getting Started

See [getting-started.md](getting-started.md).

## Diagrams

- [Architecture flowchart](diagrams/architecture.md)
- [Class diagram](diagrams/class-diagram.md)
- [Sequence diagrams](diagrams/sequences.md)

## Repository

- **Source:** https://github.com/brndnhghs/grillmaster-command-center
- **SHA:** `3e085d44fccca63896b5f6543aaa54ab4216e4b3`
- **Generated:** 2026-07-24 by `hermes-agent code-wiki skill v0.1.0`
