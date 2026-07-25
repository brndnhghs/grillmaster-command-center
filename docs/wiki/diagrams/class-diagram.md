# Class Diagram

The Image Pipeline uses a decorator-registry + executor pattern. The classes below are the load-bearing types — every field shown was observed in the source reads (registry.py, graph.py, timeline.py).

```mermaid
classDiagram
    class MethodMeta {
        +str id
        +str name
        +str category
        +list tags
        +dict params
        +dict inputs
        +dict outputs
        +Callable fn
        +int version
        +bool new_image_contract
    }
    class NodeDef {
        +str method_id
        +dict inputs
        +dict outputs
        +set param_ports
    }
    class GraphNode {
        +str id
        +str method_id
        +dict params
        +float x
        +float y
        +bool render
        +bool dirty
        +list keyframes
        +dict paramKeyframes
    }
    class GraphEdge {
        +str src_node
        +str src_port
        +str dst_node
        +str dst_port
        +bool feedback
    }
    class GraphExecutor {
        +execute(nodes, edges, seed, frame, frames)
        +selective_invalidate(old_n, new_n, old_e, new_e, seed) int
    }
    class Timeline {
        +make_timeline(n_frames, fps)
        +KeyframeTrack tracks
    }
    class KeyframeTrack {
        +list keyframes
        +eval_at(frame) float
    }

    MethodMeta <.. NodeDef : _make_node_def()
    GraphExecutor o-- GraphNode
    GraphExecutor o-- GraphEdge
    NodeDef <.. GraphNode : derived from
    Timeline *-- KeyframeTrack
```

## Notes

- **`NodeDef` is auto-generated** from `MethodMeta` by `graph._make_node_def()` — it is not hand-authored. Input ports are derived from `meta.inputs` plus one auto `image_in` port (unless `inputs=None` or `inputs={}`), and one wireable port per `SCALAR`/`FIELD` param default.
- **`GraphNode` carries the live graph state** — `params`, `keyframes`, `paramKeyframes`, and the `dirty` flag that drives live invalidation. The executor reads these, not the `NodeDef`.
- **`GraphExecutor` is not thread-safe.** The server serializes every use behind a lock (`_live_exec_lock` / `_render_exec_lock`) and keeps one persistent executor across hot-swaps so Architecture-A sim caches survive.
- **Architecture-A caching** lives inside `GraphExecutor` (`_sim_cache`, keyed by node-id + param hash + frame). `selective_invalidate()` returns the count of cache entries cleared.
- **Port types** (`IMAGE`, `FIELD`, `MASK`, `SCALAR`, `TEXT`, `PARTICLES`) are a string-keyed registry in `port_types.py`, not a class — shown as edge labels in the architecture diagram rather than here.
