"""Open port-type registry — add new types here without touching core."""
from __future__ import annotations

from dataclasses import dataclass, field as dc_field


@dataclass
class PortTypeSpec:
    name: str
    color: str               # hex color for UI
    description: str
    accepts_from: list[str] = dc_field(default_factory=list)


_PORT_REGISTRY: dict[str, PortTypeSpec] = {}


def register_port_type(
    name: str,
    color: str,
    description: str,
    accepts_from: list[str] | None = None,
) -> None:
    _PORT_REGISTRY[name] = PortTypeSpec(
        name=name, color=color, description=description,
        accepts_from=accepts_from or [],
    )


def get_port_type(name: str) -> PortTypeSpec | None:
    return _PORT_REGISTRY.get(name)


def all_port_types() -> dict[str, PortTypeSpec]:
    return dict(_PORT_REGISTRY)


# Built-in registrations
register_port_type("IMAGE",     "#4a9eff", "float32 ndarray (H,W,3) values [0,1]")
register_port_type("SCALAR",    "#888888", "Python float", accepts_from=["IMAGE"])
register_port_type("FIELD",     "#4caf50", "float32 ndarray (H,W) arbitrary range")
register_port_type("PARTICLES", "#ff9800", "float32 ndarray (N,4) — [x,y,vx,vy]")
register_port_type("MASK",      "#e8e8e8", "float32 ndarray (H,W) values [0,1]")
register_port_type("ANY",       "#444444", "wildcard input type")
