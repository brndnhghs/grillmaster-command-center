# Module: `core/port_types.py`

## Purpose
Open port-type registry allowing new data types to be added without modifying core engine code. Port types define what data flows through wire connections in the node graph.

## Responsibilities
- Maintain a global registry of `PortTypeSpec` definitions
- Provide `register_port_type()` for adding new types
- Provide `get_port_type()` and `all_port_types()` for lookup
- Define built-in types: IMAGE, SCALAR, FIELD, PARTICLES, MASK, COLORMAP, ANY

## Public Interfaces

### `PortTypeSpec` dataclass
| Field | Type | Description |
|-------|------|-------------|
| `name` | str | Type name |
| `color` | str | Hex color for UI rendering |
| `description` | str | Human-readable description |
| `accepts_from` | list[str] | Coercible input types |

### Functions
| Function | Returns | Description |
|----------|---------|-------------|
| `register_port_type(name, color, description, accepts_from)` | None | Register a new port type |
| `get_port_type(name)` | `PortTypeSpec` or `None` | Lookup by name |
| `all_port_types()` | `dict[str, PortTypeSpec]` | Full registry copy |

## Built-in Types

| Type | Color | Description | Accepts From |
|------|-------|-------------|-------------|
| `IMAGE` | `#4a9eff` (blue) | float32 ndarray (H,W,3) [0,1] | — |
| `SCALAR` | `#888888` (gray) | Python float | IMAGE (luminance) |
| `FIELD` | `#4caf50` (green) | float32 ndarray (H,W) arbitrary range | — |
| `PARTICLES` | `#ff9800` (orange) | float32 ndarray (N,4) [x,y,vx,vy] | — |
| `MASK` | `#e8e8e8` (white) | float32 ndarray (H,W) [0,1] | — |
| `COLORMAP` | `#e040fb` (magenta) | float32 ndarray (N,3/4) color palette | — |
| `ANY` | `#444444` (dark gray) | wildcard input type | — |

## Dependencies
- stdlib: `dataclasses`

## Consumers
- `graph.py`: imports `all_port_types` to ensure registry loads
- `server.py`: serves `GET /api/port-types` from `all_port_types()`
- `ui/index.html`: fetches port types for dynamic wire color rendering

## Key Design
- Added after Phase 1 refactor — previously port types were a hardcoded enum in `graph.py`
- COLORMAP was the first type added without touching core (proof of extensibility)
- Type coercion: SCALAR→int uses `round()`, SCALAR→float passes through
- Mismatched types skip silently with log warning (no crash)