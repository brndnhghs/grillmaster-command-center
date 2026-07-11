"""Motif grammar — educated randomness for graph construction.

Instead of growing graphs one type-legal edge at a time (which is knob
turning), composition samples from a library of *workflow motifs*: the
recurring shapes humans actually build — a sim into a post-FX chain, two
branches blended, a masked composite, a field modulating a filter, a
feedback loop, a control node fanned across related params.

Each motif is a small builder function with a weight. Weights come from
_DEFAULT_WEIGHTS, can be edited in shootout/motifs.json, and are further
multiplied by advisor guidance (prefer_motifs / avoid_motifs), so both the
user and the learner tune which workflows dominate.

The driver policy is separate and unconditional: after the structure
stands, every node's most animation-relevant params get control-node
drivers (LFO / ramp / counter / noise / strobe / envelope), with the
driver's output range mapped onto the target param's schema range — a
driver that oscillates 0..1 into a 0..500 param is *not* driving it.
"""
from __future__ import annotations

import json
import random
from pathlib import Path

from .config import ShootoutConfig, DEFAULT_CONFIG
from .generator import (
    GenePool, SamplingBias, _declared_ports, _fillable_ports, _is_needy,
    _pick_producer, sample_budget, sample_params,
)

MOTIFS_JSON = Path(__file__).resolve().parent / "motifs.json"


# ── Graph builder ─────────────────────────────────────────────────────


class Builder:
    def __init__(self, pool: GenePool, cfg: ShootoutConfig,
                 rng: random.Random, bias: SamplingBias | None):
        self.pool = pool
        self.cfg = cfg
        self.rng = rng
        self.bias = bias
        self.nodes: list[dict] = []
        self.edges: list[dict] = []
        self._n = 0

    def add(self, method_id: str, has_image_input: bool = False,
            render: bool = False) -> str:
        self._n += 1
        nid = f"n{self._n}"
        self.nodes.append({
            "id": nid, "method_id": method_id,
            "params": sample_params(self.pool, self.cfg, self.rng,
                                    method_id, has_image_input),
            "x": 0, "y": 0, "render": render,
        })
        return nid

    def wire(self, src_id: str, src_port: str, dst_id: str, dst_port: str,
             feedback: bool = False) -> None:
        e = {"src_node": src_id, "src_port": src_port,
             "dst_node": dst_id, "dst_port": dst_port}
        if feedback:
            e["feedback"] = True
        self.edges.append(e)

    def node(self, nid: str) -> dict:
        return next(n for n in self.nodes if n["id"] == nid)

    def mid(self, nid: str) -> str:
        return self.node(nid)["method_id"]

    def terminal_id(self) -> str | None:
        return next((n["id"] for n in self.nodes if n.get("render")), None)

    def fed_ports(self) -> set[tuple[str, str]]:
        return {(e["dst_node"], e["dst_port"]) for e in self.edges}

    def pick(self, dst_type: str, leaf: bool) -> str | None:
        return _pick_producer(self.pool, self.cfg, self.rng, dst_type,
                              leaf_only=leaf, bias=self.bias,
                              prefer_continuation=not leaf)

    # ── shared building blocks ────────────────────────────────────
    def source(self) -> str | None:
        """A pure image source (no image feed expected)."""
        mid = self.pick("image", leaf=True)
        return self.add(mid) if mid else None

    def chain(self, upstream: str, length: int) -> str:
        """Append `length` image filters after node `upstream`; returns the
        last node id."""
        cur = upstream
        for _ in range(length):
            mid = None
            for _try in range(6):
                cand = self.pick("image", leaf=False)
                if cand and any(t == "image" for _, t
                                in _fillable_ports(self.pool.defs[cand])):
                    mid = cand
                    break
            if mid is None:
                break
            port = next(p for p, t in _fillable_ports(self.pool.defs[mid])
                        if t == "image")
            nid = self.add(mid, has_image_input=True)
            self.wire(cur, self.pool.output_port_for(self.mid(cur), "image"),
                      nid, port)
            cur = nid
        return cur

    def branch(self, max_len: int = 2) -> str | None:
        """source [+ short chain] — returns the branch's output node id."""
        src = self.source()
        if src is None:
            return None
        if max_len > 0 and self.rng.random() < 0.6:
            return self.chain(src, self.rng.randint(1, max_len))
        return src


# ── Motifs ────────────────────────────────────────────────────────────
# Each returns the id of its image-output head node (or None on failure).


def m_sim_backbone(b: Builder, budget: int) -> str | None:
    """source → 1..n image filters. The bread-and-butter chain."""
    src = b.source()
    if src is None:
        return None
    return b.chain(src, min(max(budget - 1, 0), b.rng.randint(1, 3)))


def m_pattern_blend(b: Builder, budget: int) -> str | None:
    """Two branches → Image Blend (137)."""
    if "137" not in b.pool.defs:
        return None
    a = b.branch(max_len=min(budget // 2, 2))
    c = b.branch(max_len=min(budget // 2, 2))
    if a is None or c is None:
        return a or c
    blend = b.add("137")
    b.wire(a, b.pool.output_port_for(b.mid(a), "image"), blend, "image_a")
    b.wire(c, b.pool.output_port_for(b.mid(c), "image"), blend, "image_b")
    return blend


def m_masked_composite(b: Builder, budget: int) -> str | None:
    """Branch A through Apply Mask (141), mask derived from branch B via
    Image-to-Mask — the classic 'show this texture where that one is
    bright' composite."""
    if "141" not in b.pool.defs or "__image_to_mask__" not in b.pool.defs:
        return None
    a = b.branch(max_len=min(budget // 2, 2))
    m_src = b.branch(max_len=1)
    if a is None or m_src is None:
        return a or m_src
    to_mask = b.add("__image_to_mask__", has_image_input=True)
    b.wire(m_src, b.pool.output_port_for(b.mid(m_src), "image"),
           to_mask, "image_in")
    apply_m = b.add("141", has_image_input=True)
    b.wire(a, b.pool.output_port_for(b.mid(a), "image"), apply_m, "image_in")
    b.wire(to_mask, "mask", apply_m, "mask")
    return apply_m


def m_field_modulate(b: Builder, budget: int) -> str | None:
    """Feed a FIELD source into a field-accepting port somewhere in the
    existing graph (spatial modulation instead of a flat param)."""
    fed = b.fed_ports()
    targets = [(n, p) for n in b.nodes
               for p, t in _declared_ports(b.pool.defs[n["method_id"]])
               if t == "field" and (n["id"], p) not in fed]
    if not targets:
        return None
    tgt, port = b.rng.choice(targets)
    src_mid = _pick_producer(b.pool, b.cfg, b.rng, "field",
                             leaf_only=True, bias=b.bias)
    if src_mid is None:
        return None
    src = b.add(src_mid)
    b.wire(src, b.pool.output_port_for(src_mid, "field"), tgt["id"], port)
    return None  # modulator — doesn't change the image head


def m_post_fx(b: Builder, budget: int) -> str | None:
    """Extend after the current terminal: 1-2 more filters, render moves."""
    term = b.terminal_id()
    if term is None:
        return None
    b.node(term)["render"] = False
    head = b.chain(term, b.rng.randint(1, min(max(budget, 1), 2)))
    if head == term:
        b.node(term)["render"] = True
        return None
    b.node(head)["render"] = True
    return head


def m_feedback_loop(b: Builder, budget: int) -> str | None:
    """Terminal image fed back (previous frame) into an early image port —
    trails, decay, self-similarity. Risky but occasionally gorgeous."""
    term = b.terminal_id()
    if term is None:
        return None
    fed = b.fed_ports()
    early = [(n, p) for n in b.nodes if n["id"] != term
             for p, t in _fillable_ports(b.pool.defs[n["method_id"]])
             if t == "image" and (n["id"], p) not in fed]
    if not early:
        return None
    tgt, port = b.rng.choice(early)
    b.wire(term, b.pool.output_port_for(b.mid(term), "image"),
           tgt["id"], port, feedback=True)
    return None


_BACKBONES = ["sim_backbone", "pattern_blend", "masked_composite"]

_MOTIFS: dict[str, tuple] = {
    #  name               (builder,            default_weight, backbone?)
    "sim_backbone":      (m_sim_backbone,      3.0),
    "pattern_blend":     (m_pattern_blend,     1.5),
    "masked_composite":  (m_masked_composite,  1.2),
    "field_modulate":    (m_field_modulate,    1.2),
    "post_fx":           (m_post_fx,           2.0),
    "feedback_loop":     (m_feedback_loop,     0.5),
}


def motif_names() -> list[str]:
    return list(_MOTIFS)


def load_motif_weights() -> dict[str, float]:
    """Defaults, overridden by shootout/motifs.json (user-editable)."""
    weights = {k: w for k, (_, w) in _MOTIFS.items()}
    if MOTIFS_JSON.exists():
        try:
            for k, v in json.loads(MOTIFS_JSON.read_text()).items():
                if k in weights and isinstance(v, (int, float)) and v >= 0:
                    weights[k] = float(v)
        except Exception:
            pass
    return weights


# ── Driver policy — control-node-driven animation, always ─────────────

# param-name fragments → preferred driver methods (ordered by fit)
_DRIVER_AFFINITY = [
    (("speed", "rate", "freq", "phase", "flow", "vel", "time"),
     ["__lfo__", "__ramp__", "__noise1d__"]),
    (("angle", "rot", "dir", "orient"), ["__lfo__", "__ramp__"]),
    (("amp", "strength", "intensity", "gain", "mix", "blend", "opacity",
      "bright", "weight"), ["__lfo__", "__envelope__", "__noise1d__"]),
    (("threshold", "cutoff", "level", "density"), ["__lfo__", "__strobe__"]),
    (("steps", "count", "iter", "num", "n_", "rule", "mode_i", "index"),
     ["__counter__", "__strobe__"]),
]
_DRIVER_FALLBACK = ["__lfo__", "__lfo__", "__lfo__", "__noise1d__", "__ramp__"]

# driver method → (low-end param, high-end param) for output-range mapping
_DRIVER_RANGE_PARAMS = {
    "__lfo__": ("min", "max"),
    "__noise1d__": ("min", "max"),
    "__ramp__": ("start", "end"),
    "__counter__": ("start", "end"),
    "__strobe__": ("off_value", "on_value"),
}


def _driver_for(pname: str, rng: random.Random,
                available: set[str]) -> str | None:
    lname = pname.lower()
    for frags, drivers in _DRIVER_AFFINITY:
        if any(f in lname for f in frags):
            opts = [d for d in drivers if d in available]
            if opts:
                return opts[0] if rng.random() < 0.7 else rng.choice(opts)
    opts = [d for d in _DRIVER_FALLBACK if d in available]
    return rng.choice(opts) if opts else None


def _drivable_params(pool: GenePool, cfg: ShootoutConfig,
                     method_id: str) -> list[tuple[str, dict | None]]:
    """(param, schema) pairs a driver can usefully feed, best-first.

    Includes BOTH port-declared scalar params (driver_targets — the
    executor's _field_<param> wire path) AND every ranged numeric param in
    the node schema. Most nodes expose ordinary params (strength, angle,
    threshold, speed, mix, …) that are *not* declared as wiring ports but
    are exactly what _DRIVER_AFFINITY targets by name — without them the
    driver policy attaches almost nothing. Name affinity ranks
    motion-related params first; range-mappable params get a small bonus so
    the output-range mapping in _configure_driver can do real work.
    """
    d = pool.defs[method_id]
    schema = d.get("params") or {}
    port_targets = set(pool.driver_targets(method_id))
    seen: set[str] = set()
    out: list[tuple[float, str, dict | None]] = []

    def _score(p: str, spec: dict | None, port: bool) -> tuple[float, str, dict | None]:
        default = (spec or {}).get("default")
        ranged = spec is not None and spec.get("min") is not None \
            and spec.get("max") is not None \
            and isinstance(default, (int, float)) \
            and not isinstance(default, bool)
        score = 2.5 if port else 2.0
        if ranged:
            score += 0.5
        lname = p.lower()
        for frags, _ in _DRIVER_AFFINITY:
            if any(f in lname for f in frags):
                score += 1.5
                break
        return score, p, spec if ranged else None

    for p in port_targets:
        if p in cfg.frozen_params or p in seen:
            continue
        seen.add(p)
        spec = schema.get(p) if isinstance(schema.get(p), dict) else None
        out.append(_score(p, spec, port=True))
    for p, spec in schema.items():
        if p in seen or p in cfg.frozen_params:
            continue
        if not isinstance(spec, dict):
            continue
        default = spec.get("default")
        if not (isinstance(default, (int, float)) and not isinstance(default, bool)):
            continue
        lo, hi = spec.get("min"), spec.get("max")
        if lo is None or hi is None or hi <= lo:
            continue
        seen.add(p)
        out.append(_score(p, spec, port=False))

    out.sort(key=lambda t: -t[0])
    return [(p, spec) for _, p, spec in out]


def _configure_driver(b: Builder, drv_id: str, drv_mid: str,
                      target_spec: dict | None) -> None:
    """Map the driver's output range onto the target param's schema range
    and pick a musically-sane rate. Without this, an 0..1 LFO into a 0..500
    param 'drives' nothing."""
    rng = b.rng
    params = b.node(drv_id)["params"]
    if target_spec is not None and drv_mid in _DRIVER_RANGE_PARAMS:
        lo_k, hi_k = _DRIVER_RANGE_PARAMS[drv_mid]
        lo, hi = float(target_spec["min"]), float(target_spec["max"])
        span = hi - lo
        # random sub-window covering 30–100% of the param's range
        width = span * rng.uniform(0.3, 1.0)
        start = lo + rng.uniform(0, span - width)
        params[lo_k] = round(start, 4)
        params[hi_k] = round(start + width, 4)
    if drv_mid in ("__lfo__", "__noise1d__", "__strobe__"):
        params["rate"] = round(rng.uniform(0.1, 1.5), 3)   # gentle cycles
    if drv_mid == "__ramp__":
        params["duration_frames"] = int(b.cfg.frames * rng.uniform(0.5, 1.0))
        params["mode"] = rng.choice(["loop", "pingpong", "once"]) \
            if "mode" in params else params.get("mode", "loop")


def apply_driver_policy(b: Builder) -> None:
    """All animation control-node driven when possible: give every
    non-driver node a driver on its most animation-relevant param
    (p_drive_primary), sometimes a second (p_drive_secondary), and
    occasionally fan one driver across two related params."""
    pool, cfg, rng = b.pool, b.cfg, b.rng
    available = set(pool.scalar_drivers)
    if not available:
        return
    fed = b.fed_ports()
    for n in list(b.nodes):
        if n["method_id"] in pool.scalar_drivers:
            continue
        cands = [(p, s) for p, s in _drivable_params(pool, cfg, n["method_id"])
                 if (n["id"], p) not in fed]
        if not cands:
            continue
        n_drive = (rng.random() < cfg.p_drive_primary) \
            + (len(cands) > 1 and rng.random() < cfg.p_drive_secondary)
        prev_driver = None
        for i in range(int(n_drive)):
            pname, spec = cands[i]
            # fan-out: reuse the previous driver for the sibling param when
            # the ranges are compatible (one clock, several params)
            if prev_driver and spec is None and rng.random() < 0.5:
                drv_id, drv_mid = prev_driver
            else:
                drv_mid = _driver_for(pname, rng, available)
                if drv_mid is None:
                    continue
                drv_id = b.add(drv_mid)
                _configure_driver(b, drv_id, drv_mid, spec)
            b.wire(drv_id, pool.output_port_for(drv_mid, "scalar") or "value",
                   n["id"], pname)
            fed.add((n["id"], pname))
            prev_driver = (drv_id, drv_mid)


# ── Composer ──────────────────────────────────────────────────────────


def compose_graph(pool: GenePool, cfg: ShootoutConfig, rng: random.Random,
                  bias: SamplingBias | None = None,
                  motif_weights: dict[str, float] | None = None) -> dict:
    """Sample a graph by stacking motifs up to the size budget, then run
    the driver policy. Returns graph dict with a 'motifs' provenance list."""
    weights = dict(load_motif_weights())
    for k, v in (motif_weights or {}).items():
        if k in weights:
            weights[k] *= v

    b = Builder(pool, cfg, rng, bias)
    used: list[str] = []
    target = sample_budget(cfg, rng, bias.complexity if bias else 0.0) + 1

    # 1. Backbone — produces the render head.
    bb_names = [m for m in _BACKBONES if weights.get(m, 0) > 0] or ["sim_backbone"]
    bb = rng.choices(bb_names, weights=[weights.get(m, 1.0) for m in bb_names])[0]
    head = _MOTIFS[bb][0](b, max(target - 1, 0))
    if head is None:   # catalog too filtered — fall back to a bare source
        head = b.source() or b.add(rng.choice(pool.terminals))
    b.node(head)["render"] = True
    used.append(bb)

    # 2. Extensions until the budget is spent.
    ext_names = [m for m in _MOTIFS if m not in _BACKBONES
                 and weights.get(m, 0) > 0]
    tries = 0
    while len(b.nodes) < target and ext_names and tries < target * 2:
        tries += 1
        m = rng.choices(ext_names,
                        weights=[weights.get(x, 1.0) for x in ext_names])[0]
        before = len(b.nodes) + len(b.edges)
        _MOTIFS[m][0](b, target - len(b.nodes))
        if len(b.nodes) + len(b.edges) > before:
            used.append(m)

    # 3. Animation: control-node drivers everywhere they fit.
    apply_driver_policy(b)

    from .generator import _auto_layout
    _auto_layout(b.nodes, b.edges)
    return {"version": 1, "name": "", "nodes": b.nodes, "edges": b.edges,
            "motifs": used}
