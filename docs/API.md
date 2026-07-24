# API Reference — Grillmaster Command Center

> Generated: 2026-07-13 · Phase 5

---

## Base URL

- Image Pipeline: `http://localhost:7860`
- Dashboard: `http://localhost:7870`

## Authentication

When `GRILLMASTER_API_TOKEN` env var is set:
- Mutating endpoints require `X-Api-Token` header
- The UI attaches it from `localStorage['api-token']`

## Endpoints

### Static & UI

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/` | Serve editor UI (`ui/index.html`) |
| `GET` | `/health` | Health check → `{"ok": true}` |

### Port Types & Methods

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/api/port-types` | Port type registry (colors, descriptions) |
| `GET` | `/api/palettes` | List available color palettes |
| `GET` | `/api/methods` | List all registered methods (id, name, category, params) |
| `GET` | `/api/node-defs` | Full node definitions with port info |

### Generation

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/api/generate` | Generate single image or animation |
| `DELETE` | `/api/jobs/{job_id}` | Cancel a running job |
| `GET` | `/api/jobs/{job_id}/stream` | SSE job progress stream |
| `GET` | `/api/jobs/{job_id}/result` | Download job result file |

**POST /api/generate** body:
```json
{
  "method_id": "32",
  "seed": 42,
  "params": {},
  "animate": false,
  "fps": 24,
  "duration": 3.0,
  "filter": null,
  "demo": false,
  "width": 768,
  "height": 512
}
```

### Node Graph

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/api/graph/run` | Execute a graph (single frame or multi-frame) |
| `POST` | `/api/graph/live` | Start/stop/update live mode loop |
| `GET` | `/api/graph/live/status` | Live mode status |
| `POST` | `/api/graph/render-sequence` | Render a multi-frame sequence |
| `GET` | `/api/graph/wire-payload/{job_id}/{src_node_id}` | Inspect wire payload |

**POST /api/graph/run** body:
```json
{
  "nodes": [{"id": "n1", "method_id": "32", "params": {}, "x": 100, "y": 200}],
  "edges": [{"src_node": "n1", "src_port": "image", "dst_node": "n2", "dst_port": "image_in"}],
  "seed": 42,
  "frame": 0,
  "frames": 1
}
```

### Graph Document Store

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/api/graphs/{gid}` | Load graph document |
| `POST` | `/api/graphs/{gid}` | Save graph document |
| `WS` | `/api/graphs/{gid}/ws` | WebSocket graph document broadcast |

### Live Preview

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/api/live/stream` | MJPEG multipart/x-mixed-replace stream |
| `WS` | `/api/live/ws` | WebSocket JSON frames with base64 JPEG |
| `GET` | `/api/live/frame.jpg` | Polling fallback — latest JPEG frame |

### SSE Events

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/api/events` | SSE stream for hot-reload events |

### Node Doctor

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/api/node-doctor/{method_id}` | Node Doctor chat (GET) |
| `POST` | `/api/node-doctor/{method_id}` | Apply Node Doctor fix |

### Node Tester

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/api/node-tester/run` | Run all method tests |
| `POST` | `/api/node-tester/batch-fix` | Batch-apply fixes to failing methods |

### Broken-Node Ledger

User-filed "this node misbehaves" reports, persisted to `docs/reports/broken-nodes.json`.
Read the whole ledger before diagnosing a single node — the point of the file is that
patterns across reports (a shared helper, a whole category) are visible where one report
is not. Each report carries `source_path` and the `params` in play when it was filed.

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/api/broken-nodes` | List reports; `?status=open` filters to unresolved |
| `POST` | `/api/broken-nodes` | File a report — `{method_id, note, node_id?, params?, graph_name?, reported_by?}` |
| `PATCH` | `/api/broken-nodes/{report_id}` | Edit `note`, or set `status` to `open` / `resolved` |
| `DELETE` | `/api/broken-nodes/{report_id}` | Drop a report entirely |

### Saved Graphs

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/api/graph/saved` | List saved graphs |
| `POST` | `/api/graph/save` | Save a graph |
| `GET` | `/api/graph/saved/{name}` | Load a saved graph |
| `DELETE` | `/api/graph/saved/{name}` | Delete a saved graph |

### Sequences

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/api/sequences` | List rendered sequences |
| `GET` | `/api/sequences/{name}` | List frames in a sequence |
| `GET` | `/api/sequences/{name}/{frame}` | Download a specific frame |

### 3D

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/api/3d/render` | Render a 3D scene via headless sidecar |
| `POST` | `/api/3d/upload-model` | Upload a 3D model file |

### Admin

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/api/admin/restart` | Restart the server process (requires token) |

## SSE Event Types

| Event | Source | Data |
|-------|--------|------|
| `progress` | Generation job | `{"message": "...", "elapsed": 1.2}` |
| `frame` | Generation job | Base64 JPEG string |
| `done` | Generation job | `{"output_path": "...", "type": "image"}` |
| `error` | Generation job | `{"message": "..."}` |
| `node-defs-updated` | Hot-reload | `{}` |
| `broken-nodes-updated` | Broken-node ledger changed | `{}` |
| `graph` | Graph document mutation | JSON payload |

## WebSocket Message Format (Live)

JSON message from server:
```json
{
  "frame": 42,
  "cook_ms": 15.3,
  "fps": 29.5,
  "node_timings": {"n1": 12.1, "n2": 3.2},
  "node_names": {"n1": "Gray-Scott"},
  "node_errors": {},
  "gpu_nodes": 0,
  "cpu_nodes": 2,
  "mem_edges": 1,
  "disk_edges": 0,
  "edge_transport": {"n1->n2": "mem"},
  "canvas_w": 768,
  "canvas_h": 512,
  "img": "<base64-jpeg>"
}
```