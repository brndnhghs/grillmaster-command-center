# Performance Analysis — Grillmaster Command Center

> Generated: 2026-07-13 · Phase 8

---

## Current Performance Profile

### Bottleneck Analysis

| Component | Bottleneck | Estimated Impact |
|-----------|-----------|------------------|
| JPEG encode (live mode) | cv2 `imencode` per frame | ~5-15ms per 768×512 frame |
| Arch-A sim cache | 1.5 GB budget, ~1.4 GB per 300-frame sim | Memory pressure with 2+ large sims |
| Dirty-flag skip | O(1) dict lookup | Negligible |
| Image wiring | PIL write + read for legacy methods | ~10-100ms per edge per frame |
| Expression eval | AST parse + compile per frame | ~50-200µs per expression |
| ffmpeg pipe | Rawvideo pipe for MP4 encoding | Limited by disk I/O |
| Server startup | Import all method files | ~2-5 seconds |
| WebSocket broadcast | Base64 encode + JSON serialization | ~1-5ms per frame |

### Hot Path Analysis

**Single-frame interactive run (typical):**
- Topological sort: O(V + E) — negligible
- Dirty check: O(1) per node (dict lookup)
- Arch-A cache hit: O(1) (dict lookup + modulo)
- Arch-B per node: time spent in `meta.fn()` — varies by method
- Total: dominated by method execution time

**Live mode (continuous 30fps):**
- JPEG encode: 5-15ms (cv2)
- JSON serialization: 1-3ms
- WebSocket broadcast: 1-5ms (N clients)
- MJPEG: no encoding per client (shared buffer)
- Total overhead: ~10-25ms per frame → 40-100fps capacity

**Multi-frame sequence render:**
- Per-frame: same as single-frame
- No live overhead (no JPEG encode, no WS broadcast)
- Each frame saved to disk: ~10-50ms per PNG write

## Memory Analysis

| Component | Memory Usage | Notes |
|-----------|-------------|-------|
| Sim cache (max) | 1.5 GB | Single 300-frame 768×512 sim ≈ 1.4 GB |
| Canvas (768×512) | 4.5 MB per image | float32 RGB |
| Canvas (1920×1080) | 24.8 MB per image | float32 RGB |
| Method registry | ~500 KB | 180+ MethodMeta objects |
| Node graph state | ~100 KB | 20-node graph JSON |
| Python process baseline | ~50-100 MB | FastAPI + numpy + opencv |

## Optimization Opportunities

### 1. Sim Cache Compression

**Issue:** Arch-A sim cache stores full float32 RGB frames. A 300-frame sim at 768×512 is ~1.4 GB.

**Opportunity:** Store frames as uint8 JPEG in memory (decompress on read). JPEG at quality 85 reduces memory by 10-20× at negligible quality cost.

**Estimated impact:** 1.5 GB → 150 MB for equivalent frame count

**Effort:** 2-3 days

### 2. Zero-Copy Image Passthrough

**Issue:** When a filter node with `is_time_varying=False` receives an upstream image, it currently makes a copy (`arr.copy()`) even when it doesn't modify the image.

**Opportunity:** Use read-only views or reference counting for passthrough nodes.

**Estimated impact:** Eliminates ~4.5 MB copy per passthrough per frame

**Effort:** 1 day

### 3. Parallel Node Execution

**Issue:** Nodes execute sequentially in topological order, even when they have no data dependencies.

**Opportunity:** Execute independent subgraphs in parallel using a thread pool. This is most impactful for graphs with branches (e.g., two generators feeding a composite node).

**Estimated impact:** 2× throughput for branched graphs; negligible for linear chains

**Effort:** 3-5 days (significant complexity)

### 4. PNG Compression Level

**Issue:** `PIL.save()` uses default compression (level 6) for transport files (`_input.png`). These are transient and don't need high compression.

**Fix already applied:** `compress_level=1` for transport files in `graph.py` (10× faster encode).

### 5. Lazy Method Import

**Issue:** `import image_pipeline.methods` at server startup imports all method files, even those with optional dependencies (torch, moderngl, etc.).

**Opportunity:** Lazy-import individual method files on first use, or use a dependency registry to skip methods whose dependencies are missing.

**Estimated impact:** Reduces startup time from 2-5s to <1s

**Effort:** 2-3 days

## Algorithmic Complexity

| Operation | Complexity | Notes |
|-----------|------------|-------|
| Topological sort | O(V + E) | Kahn's algorithm |
| Name-based param scoring | O(P × S) | Param count × synonym table size |
| Dirty propagation | O(V + E) | BFS through DAG |
| Sim cache lookup | O(1) | Dict keyed by (node_id, seed) |
| Sim cache eviction | O(N) | Oldest-first, protects active nodes |
| Image thumbnail | O(1) | Fixed-size JPEG encode |
| Palette quantization | O(H × W × N) | N = palette size, chunked for memory |
| Floyd-Steinberg dither | O(H × W) | Sequential per-pixel (cannot be parallelized) |

## Startup Cost

| Phase | Time | Dominant Factor |
|-------|------|-----------------|
| Python import | ~0.5s | 180+ method file imports |
| numpy import | ~0.3s | Large C extension |
| opencv import | ~0.5s | Large C extension |
| FastAPI app creation | ~0.1s | Route registration |
| Watchdog observer | ~0.1s | File system watcher |
| **Total** | **~2-5s** | Depends on disk speed |

## Rendering Performance

| Method Type | Typical Time | Notes |
|-------------|-------------|-------|
| Simple filter (glitch, dither) | 10-50ms | Fast pixel operations |
| Fractal (Mandelbrot, Julia) | 50-500ms | Escape-time per pixel |
| Pattern (noise, Voronoi) | 10-200ms | Parallelizable |
| Simulation (Gray-Scott, CA) | 100-5000ms | Per-frame step, first frame is full cook |
| ML model (Stable Diffusion) | 10-60s | GPU-dependent, lazy import |
| Blender render | 5-60s | External process |