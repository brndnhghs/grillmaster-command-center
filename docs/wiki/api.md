# API Reference (Index)

This page is a **functional index** — every endpoint is documented with its path and behaviour in [`modules/server.md`](modules/server.md). Below are the groups and the canonical entry points.

## Image Pipeline (`:7860`)

| Group | Representative endpoints | Purpose |
|-------|--------------------------|---------|
| **Methods** | `GET /api/methods`, `GET /api/node-defs`, `GET /api/port-types`, `GET /api/palettes`, `GET /api/easing-presets` | Browse the 373-method library and its port/param schema |
| **Generation** | `POST /api/generate`, `GET /api/jobs/{id}/stream` (SSE), `DELETE /api/jobs/{id}` | Run a single method, stream progress, cancel |
| **Graph Execute** | `POST /api/graph/execute`, `GET /api/graph/{gid}/render`, `POST /api/graph/save`, `GET /api/graph/saved` | Headless graph render → bytes, save/load shared graph docs |
| **Live Sim** | `POST /api/graph/live`, `GET /api/graph/live/stream` (MJPEG), `WS /api/live/ws`, `GET /api/graph/live/status` | Continuous simulation, frame streaming, live stats |
| **Graph WS** | `WS /api/graph/ws`, `POST /api/graph/{gid}/execute` | Edit a shared graph over WebSocket; execute by id |
| **Assets** | `POST /api/assets/upload`, `GET /api/assets` | Upload USD/GLTF models & textures (512 MB cap) |
| **Groups** | `POST /api/groups/save`, `GET /api/groups`, `DELETE /api/groups/{name}` | Save/load node groups |
| **Node Doctor** | `POST /api/node-doctor/chat`, `POST /api/node-doctor/apply`*, `POST /api/node-doctor/undo/{id}`* | LLM-assisted method editing (apply/undo require token) |
| **Node Tester** | `POST /api/node-tester/run`, `GET /api/node-tester/status`, `GET /api/node-tester/report` | Automated per-method test runner |
| **Diagnostics** | `GET /api/graph/diagnostics`, `GET /api/sequences`, `GET /api/sequences/{name}/video.{ext}` | Graph health, sequence encode/stream |
| **Admin** | `POST /admin/restart`* | Restart the server to pick up code changes (requires token) |

`*` = protected by `GRILLMASTER_API_TOKEN` (see [getting-started.md](getting-started.md#configuration)).

## Data Models (Image Pipeline)

- **`GenerateRequest`** — `method_id`, `seed`, `params`, `animate`, `fps`, `duration`, `filter`, `demo`, `width`, `height`
- **`GraphRequest`** — `nodes[]`, `edges[]`, `seed`, `frames`, `frame`, `width`, `height`, `fps_limit`, `graph_id`
- **`NodeModel`** — `id`, `type`, `x`, `y`, `params`, `paramKeyframes`, `dirty`
- **`EdgeModel`** — `src_node`, `dst_node`, `src_port`, `dst_port`

## Auth

When `GRILLMASTER_API_TOKEN` is set, protected endpoints require header `X-Api-Token: <token>`. The UI attaches it automatically from `localStorage['api-token']` (see `ui/js/app.js`).
