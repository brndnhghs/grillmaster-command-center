"""Compact node-catalog digest for the Hermes builder prompt.

Source of truth is `get_all_node_defs()` (the live registry), partitioned by role
with `shootout.generator.build_gene_pool()` so we reuse the exact IMAGE-terminal /
driver / source classification the repair pass enforces. The digest is terse by
design — Hermes needs real method_ids and the most useful params, not full
schemas — so it fits in context alongside the playbook and few-shots.
"""
from __future__ import annotations

import image_pipeline.methods  # noqa: F401 — registers the full node catalog
from image_pipeline.core.graph import get_all_node_defs
from image_pipeline.shootout.generator import GenePool, build_gene_pool


def gene_pool() -> GenePool:
    return build_gene_pool()


def _fmt_param(name: str, spec: dict) -> str:
    """One param as `name(default[; range|choices])`, abbreviated."""
    default = spec.get("default")
    bits = f"{name}={default!r}"
    choices = spec.get("choices")
    if choices:
        shown = choices[:6]
        more = "…" if len(choices) > 6 else ""
        bits += " {" + ",".join(map(str, shown)) + more + "}"
    elif "min" in spec and "max" in spec:
        bits += f" [{spec['min']}..{spec['max']}]"
    return bits


def _fmt_ports(ports: dict) -> str:
    return ",".join(sorted(set(ports.values()))) if ports else "-"


def _node_line(mid: str, d: dict, params_limit: int) -> str:
    params = d.get("params") or {}
    # Prefer params with ranges/choices (the expressive knobs) first.
    ordered = sorted(
        params.items(),
        key=lambda kv: (0 if ("choices" in kv[1] or "min" in kv[1]) else 1),
    )
    shown = [_fmt_param(n, s) for n, s in ordered[:params_limit]]
    more = f" +{len(params) - params_limit} more" if len(params) > params_limit else ""
    ins = _fmt_ports(d.get("inputs") or {})
    outs = _fmt_ports(d.get("outputs") or {})
    desc = (d.get("description") or "").strip().split("\n")[0][:80]
    return (f"  {mid}  {d.get('name','?')} — {desc}\n"
            f"      in:{ins} out:{outs} | " + "; ".join(shown) + more)


def _by_category(defs: dict, ids: list[str]) -> dict[str, list[str]]:
    groups: dict[str, list[str]] = {}
    for mid in ids:
        cat = defs[mid].get("category", "misc")
        groups.setdefault(cat, []).append(mid)
    return groups


def digest(params_limit: int = 6, max_per_category: int | None = None) -> str:
    """Render the catalog digest string for the builder prompt.

    params_limit    — params shown per node.
    max_per_category — cap nodes listed per category (None = all). Use to shrink
                       the prompt if context pressure demands it.
    """
    pool = gene_pool()
    defs = pool.defs
    lines: list[str] = []

    def emit_group(title: str, ids: list[str]) -> None:
        if not ids:
            return
        lines.append(title)
        for cat, mids in sorted(_by_category(defs, ids).items()):
            lines.append(f"[{cat}]")
            for mid in sorted(mids, key=lambda m: defs[m].get("name", "")):
                lines.append(_node_line(mid, defs[mid], params_limit))
            if max_per_category:
                pass  # cap applied below
        lines.append("")

    # Terminals / filters — the IMAGE producers (one is the render node).
    image_ids = list(pool.image_producers)
    if max_per_category:
        capped: list[str] = []
        for cat, mids in _by_category(defs, image_ids).items():
            capped.extend(sorted(mids, key=lambda m: defs[m].get("name", ""))[:max_per_category])
        image_ids = capped
    emit_group(
        "=== IMAGE NODES (generators & filters — exactly one is render:true) ===",
        image_ids,
    )

    # Drivers — scalar-only, animate params, never terminal.
    emit_group(
        "=== DRIVERS (SCALAR output — wire into params to animate; never terminal) ===",
        list(pool.scalar_drivers),
    )

    # Other sources — FIELD / MASK / PARTICLES / COLORMAP producers that are not
    # themselves IMAGE producers (those already appear above).
    other: list[str] = []
    seen = set(pool.image_producers) | set(pool.scalar_drivers)
    for t in ("field", "mask", "particles", "colormap"):
        for mid in pool.producers_by_type.get(t, []):
            if mid not in seen:
                other.append(mid)
                seen.add(mid)
    emit_group(
        "=== SOURCES (FIELD/MASK/PARTICLES/COLORMAP — feed filters & compositors) ===",
        other,
    )

    return "\n".join(lines).rstrip() + "\n"


def describe_graph(graph: dict) -> str:
    """Human one-liner-per-node summary of a graph, for the UI/logs."""
    defs = gene_pool().defs
    out: list[str] = []
    for n in graph.get("nodes", []):
        mid = n.get("method_id")
        name = defs.get(mid, {}).get("name", mid)
        flag = " (render)" if n.get("render") else ""
        out.append(f"{n.get('id')}: {name}{flag}")
    for e in graph.get("edges", []):
        out.append(f"  {e.get('src_node')}.{e.get('src_port')} → "
                   f"{e.get('dst_node')}.{e.get('dst_port')}")
    return "\n".join(out)
