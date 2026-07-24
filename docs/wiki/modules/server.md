# Module: `server.py` (image_pipeline)

## Purpose
FastAPI web server (3,015 lines) serving the node-graph editor frontend, the generation API, live preview streaming, graph execution, SSE events, hot-reload, Node Doctor, and Node Tester.

## Responsibilities
- Serve the single-page editor UI (`ui/index.html`)
- Generate single images and animations via `/api/generate`
- Execute node graphs via `/api/graph/run` and `/api/graph/render-sequence`
- Continuous live preview loop (`POST /api/graph/live`, MJPEG stream, WebSocket)
- SSE streaming for job progress, hot-reload events, and graph events
- WebSocket graph document broadcast (shared doc between user and agent)
- Hot-reload method files via watchdog
- Graph document persistence (shared doc store)
- Node Doctor endpoint (LLM-powered method repair via Hermes agent)
- Node Tester endpoint (batch method testing)
- 3D scene rendering via headless sidecar
- Method file upload/download
- Admin restart

## Startup Sequence
1. Import `image_pipeline.core.registry` → loads all methods
2. Import `image_pipeline.methods` → auto-registers all methods via `@method` decorator
3. Mount static files at `/output`, `/ui`, `/assets`
4. Install `_ThreadDispatchWriter` for per-thread stdout/stderr proxy
5. Start watchdog observer for method hot-reload
7. Enter lifespan → accept requests

## Ports
- Default: 7860 (image pipeline)
- Dashboard: 7870 (separate process)

## Key Modules Referenced
- `core/registry.py` — method lookup
- `core/graph.py` — `GraphExecutor`, `NodeDef`, `GraphError`
- `core/port_types.py` — `all_port_types`
- `core/animation.py` — `animate_method`, `JobCancelled`, `set_job_context`
- `core/quality.py` — quality check
- `core/postprocess.py` — `apply_filter`
- `core/annotator.py` — `annotate_image`
- `core/cache.py` — content-addressed cache
- `core/registry.py` — `unregister`, `get_ids_by_module`
- `nd_runner.py` — Hermes Node Doctor subprocess runner

## Graph Store
- `_graph_docs` in-memory dict + persisted to `output/graphs/{id}.json`
- `_graph_store_lock` for thread safety
- `_load_graph_doc(gid)` / `_persist_graph_doc(doc)` / `_touch_graph_meta(doc, by)`
- `_broadcast_graph_event` — pushes mutations to all WebSocket graph clients

## Live Preview System
- `_LIVE_FRAME` global bytes buffer (MJPEG)
- `_LIVE_FRAME_COND` — condition variable for MJPEG waiters
- `_LIVE_FRAME_ID` — monotonic counter
- `_push_live_frame(arr, ws_meta)` — encode JPEG, update buffer, broadcast to WS
- `_broadcast_ws_frame(jpeg_bytes, ws_meta)` — JSON with base64 image + diagnostics
- `_encode_jpeg(arr, quality, max_width, halve)` — cv2-backed JPEG encode with PIL fallback
- **MJPEG stream**: `GET /api/live/stream` (multipart/x-mixed-replace)
- **WebSocket**: `WS /api/live/ws` (JSON frames with base64 JPEG)
- **Polling**: `GET /api/live/frame.jpg` (latest JPEG frame)

## Hot-Reload
- `_MethodWatcher` (watchdog `FileSystemEventHandler`) watches `image_pipeline/methods/`
- `_hot_reload_path(filepath)` — unregisters old methods, re-imports module, broadcasts SSE
- `_sse_clients` — list of asyncio queues for `/api/events` SSE endpoint

## Auth
- `GRILLMASTER_API_TOKEN` env var — when set, mutating endpoints require `X-Api-Token` header
- `require_token()` dependency — used by `/admin/restart`, Node Doctor, Node Tester

## Thread Dispatch Writer
- `_ThreadDispatchWriter` — per-thread stdout/stderr proxy
- Installed at module load time replacing `sys.stdout`/`sys.stderr`
- Each job thread installs its own writer, falls back to real stream
- Enables concurrent job output isolation

## API Endpoints

### Static & UI
| Method | Path | Description |
|--------|------|-------------|
| GET | `/` | Serve editor UI |
| GET | `/health` | Health check |

### Generation
| Method | Path | Description |
|--------|------|-------------|
| POST | `/api/generate` | Generate single image or animation |
| DELETE | `/api/jobs/{job_id}` | Cancel a running job |
| GET | `/api/jobs/{job_id}/stream` | SSE job progress stream |
| GET | `/api/jobs/{job_id}/result` | Download job result file |

### Node Graph
| Method | Path | Description |
|--------|------|-------------|
| GET | `/api/port-types` | Port type registry |
| GET | `/api/palettes` | List available palettes |
| GET | `/api/methods` | List all registered methods |
| GET | `/api/node-defs` | Full node definitions with port info |
| POST | `/api/graph/run` | Execute a graph (single frame or multi-frame) |
| POST | `/api/graph/live` | Start/stop/update live mode loop |
| GET | `/api/graph/live/status` | Live mode status |
| POST | `/api/graph/render-sequence` | Render a multi-frame sequence |
| GET | `/api/sequences` | List rendered sequences |
| GET | `/api/sequences/{name}` | List frames in a sequence |
| GET | `/api/sequences/{name}/{frame}` | Download a specific frame |
| GET | `/api/graph/saved` | List saved graphs |
| POST | `/api/graph/save` | Save a graph |
| GET | `/api/graph/saved/{name}` | Load a saved graph |
| DELETE | `/api/graph/saved/{name}` | Delete a saved graph |
| GET | `/api/graph/wire-payload/{job_id}/{src_node_id}` | Inspect wire payload |

### Live Preview
| Method | Path | Description |
|--------|------|-------------|
| GET | `/api/live/stream` | MJPEG live stream |
| WS | `/api/live/ws` | WebSocket live preview |
| GET | `/api/live/frame.jpg` | Polling frame fallback |

### Graph Document Store
| Method | Path | Description |
|--------|------|-------------|
| GET | `/api/graphs/{gid}` | Load graph document |
| POST | `/api/graphs/{gid}` | Save graph document |
| WS | `/api/graphs/{gid}/ws` | WebSocket graph document broadcast |

### Node Doctor & Tester
| Method | Path | Description |
|--------|------|-------------|
| GET | `/api/node-doctor/{method_id}` | Node Doctor chat |
| POST | `/api/node-doctor/{method_id}` | Apply Node Doctor fix |
| POST | `/api/node-tester/run` | Run all method tests |
| POST | `/api/node-tester/batch-fix` | Batch-apply fixes to failing methods |

### Admin
| Method | Path | Description |
|--------|------|-------------|
| POST | `/api/admin/restart` | Restart the server process |
| GET | `/api/events` | SSE event stream (hot-reload) |

### 3D
| Method | Path | Description |
|--------|------|-------------|
| POST | `/api/3d/render` | Render a 3D scene via headless sidecar |
| POST | `/api/3d/upload-model` | Upload a 3D model file |

## Dependencies
- `fastapi`, `uvicorn`, `pydantic`
- `watchdog` (hot-reload)
- `numpy`, `PIL`, `cv2` (image encoding)
- `image_pipeline.core.*` (all core modules)

## Performance
- MJPEG frame rate: ~30 fps cap (`_frame_interval = 1/30`)
- JPEG encode: cv2 (libjpeg-turbo) with PIL fallback
- Live mode: in-memory payload bus, zero disk writes, `audit_to_disk=False`
- Graph store: in-memory dict + disk persistence
- SSE: keepalive every 30s to prevent timeout