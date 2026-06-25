"""Port-type registry for Chord Bot — open registry mirroring image_pipeline/core/port_types.py."""
from __future__ import annotations

from dataclasses import dataclass, field as dc_field


@dataclass
class PortTypeSpec:
    name: str
    color: str
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
        name=name,
        color=color,
        description=description,
        accepts_from=accepts_from or [],
    )


def get_port_type(name: str) -> PortTypeSpec | None:
    return _PORT_REGISTRY.get(name)


def all_port_types() -> dict[str, PortTypeSpec]:
    return dict(_PORT_REGISTRY)


# Built-in registrations
register_port_type(
    "HARMONIC",
    "#9b59b6",
    "HarmonicState — key, mode, chord, voices, tension, duration",
)
register_port_type(
    "BEAT",
    "#e67e22",
    "Python float — beat position",
    accepts_from=["HARMONIC"],
)
