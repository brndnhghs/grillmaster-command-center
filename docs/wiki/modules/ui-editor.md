# Module: `ui/` (Node-Graph Editor SPA)

## Purpose
The browser front-end for the Image Pipeline — a single-page application (`ui/index.html`, ~654 lines) backed by a modular ES-module JS bundle. It is the primary human interface: browse/run individual methods, build and run node graphs on a canvas, watch a live MJPEG/WebSocket preview, edit a 3D viewport, and inspect diagnostics.

## Architecture (architectural overview — not an exhaustive JS inventory)
The shell is a **tabbed layout** defined in `index.html`:
- **Methods** tab — the classic single-method UI: sidebar search, parameter editor, output viewer (image/video), generate/stop/auto buttons, elapsed timer, progress log.
- **Node Graph** tab — the graph editor: a draggable node canvas, a method palette sidebar, a toolbar (run, live, auto, clear, save/load, layout, canvas-size preset, 3D-edit, FX overlay), and a shared center column where the **graph canvas** and the **3D viewport** dock side-by-side (split by a draggable divider — the viewport does not take over the preview).
- **Diagnostics** tab — node timings, cache hit/miss, edge transport (mem vs disk), GPU/CPU split, cook-rate limiter status.

### JS module system
`ui/js/` holds the bundle. Observed entry/orchestration points:
- `app.js` (~32 KB) — Methods-tab controller: fetches `/api/methods`, wires the generate/stop/auto buttons, drives the SSE job stream, and applies a token-aware `fetch` wrapper (when `localStorage['api-token']` is set, every request carries `X-Api-Token`).
- `graph.js` (~275 KB) — the node-graph canvas editor (nodes, edges, drag, wiring, save/load).
- `client3d.js` / `editor3d.js` — the three.js viewport and its editor controls.
- `diagnostics.js` — the Diagnostics tab.
- `theme.js` — theme bootstrap (runs before first paint to avoid flash; mirrors the `gm-theme` / `gm-theme-custom` localStorage keys).
- `graph-history.js`, `graph-clipboard.js`, `timeline-menu.js`, `node-tester.js`, `server-restart.js` — supporting modules.

### three.js integration
`index.html` declares an **import map** pointing `three` → `/ui/vendor/three.module.js` and `three/addons/` → `/ui/vendor/addons/`. This lets the vendored three.js r185 build (and its addons that `import 'three'`) load verbatim from `ui/vendor/` without hand-rewriting on every upgrade. There is **no** `three/webgpu` or `three/tsl` entry in this build.

### Mobile responsiveness
`app.js` detects `(max-width: 768px)` and moves the viewer + button row into sticky top/bottom bars; the graph toolbar collapses to a mobile top bar with the same button groups.

## Key API surfaces consumed (from `server.py`)
- `GET /api/methods`, `GET /api/node-defs`, `GET /api/port-types`, `GET /api/palettes`, `GET /api/easing-presets`
- `POST /api/generate` (+ `GET /api/jobs/{id}/stream`, `DELETE /api/jobs/{id}`)
- `POST /api/graph/execute`, `POST /api/graph/live`, `GET /api/graph/live/stream` (MJPEG), `WS /api/live/ws`
- `GET/POST /api/graph/{gid}`, `WS /api/graphs/{gid}/ws`, `POST /api/graph/save`
- `POST /api/node-doctor/chat`, `POST /api/node-tester/run`

## Dependencies
- Pure browser ES modules + vendored `three.js` r185 (no build step)
- Talks to `server.py` over REST + SSE + WebSocket

## Consumers
- Served by `server.py` at `/` (mounted static at `/ui`, `/assets`)
- Also reachable through the `dashboard` SPA switcher

## Known Assumptions
- The token wrapper is a no-op when `localStorage['api-token']` is unset (local/dev use)
- The 3D viewport and graph canvas are co-equal panes; both stay live during a render

## Source
[`ui/index.html`](https://github.com/brndnhghs/grillmaster-command-center/blob/3e085d44fccca63896b5f6543aaa54ab4216e4b3/ui/index.html) · [`ui/js/app.js`](https://github.com/brndnhghs/grillmaster-command-center/blob/3e085d44fccca63896b5f6543aaa54ab4216e4b3/ui/js/app.js)
