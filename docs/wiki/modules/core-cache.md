# Module: `core/cache.py`

## Purpose
Content-addressed output cache. Each generated image is stored by method key + seed hash.

## Functions
| Function | Description |
|----------|-------------|
| `exists(method_id, seed, out_dir, params)` | Check cached output, return path or None |
| `store(method_id, seed, out_path, params)` | Record cached output path |
| `clear(method_id)` | Clear cache for one method or all |
| `_cache_key(method_id, seed, params_hash)` | SHA256-based key (16 hex chars) |
| `_hash_params(params)` | MD5 param hash |

## Notes
- Cache dir: `~/.cache/image-pipeline/`
- Used by `runner.py` (CLI) — not by `server.py` or `graph.py`
- Effectively superseded by the executor's sim cache and dirty-flag system