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

import numpy as np

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
        existing = {n["id"] for n in self.nodes}
        while nid in existing:
            self._n += 1
            nid = f"n{self._n}"
        self.nodes.append({
            "id": nid, "method_id": method_id,
            "params": sample_params(self.pool, self.cfg, self.rng,
                                    method_id, has_image_input),
            "x": 0, "y": 0, "render": render,
        })
        return nid

    def wire(self, src_id: str, src_port: str, dst_id: str, dst_port: str) -> None:
        # No `feedback` option by design: the auto-generator must never emit a
        # feedback edge (image output looped back to an upstream input). The
        # pipeline has no layering/accumulation node to make that render
        # correctly, so feedback stays a manual/UI-only concept.
        self.edges.append({"src_node": src_id, "src_port": src_port,
                           "dst_node": dst_id, "dst_port": dst_port})

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


    # ── Terminal variance guard (Route 8: kill flat/static reject waste) ──

    def _primary_image_in(self, mid: str) -> str | None:
        d = self.pool.defs[mid]
        for p, t in _fillable_ports(d):
            if t == "image":
                return p
        for p, t in (d.get("outputs") or {}).items():
            if t == "image":
                return p
        return None

    def _probe_terminal_variance(self, cfg) -> "tuple[float, float] | None":
        """Render 2 tiny frames of the render head + its ancestor subgraph
        (drivers are ancestors, so modulation is captured; parallel heavy
        branches are excluded) and return (spatial_var, temporal_var), or None
        on failure/timeout. Sim heads are skipped (n_frames param)."""
        import shutil
        import tempfile
        import threading

        head = self.terminal_id()
        if head is None:
            return None
        if "n_frames" in (self.pool.defs[self.mid(head)].get("params") or {}):
            return None
        # Subgraph: head + ancestors only (BFS up the edge list).
        keep = {head}
        changed = True
        while changed:
            changed = False
            for e in self.edges:
                if e["src_node"] in keep and e["dst_node"] not in keep:
                    keep.add(e["dst_node"])
                    changed = True
        nodes = [dict(n, dirty=True) for n in self.nodes if n["id"] in keep]
        edges = [dict(e) for e in self.edges
                 if e["src_node"] in keep and e["dst_node"] in keep]
        W, H = 112, 72
        try:
            from image_pipeline.core.graph import GraphExecutor
            from image_pipeline.core.utils import set_canvas
        except Exception:
            return None
        wd = Path(tempfile.mkdtemp(prefix="svar-"))
        res: dict = {}

        def _run() -> None:
            try:
                set_canvas(W, H)
                ex = GraphExecutor(wd, fps=cfg.fps, in_memory=True,
                                   audit_to_disk=False)
                frames = []
                for frame in range(2):
                    flat, _t, _e = ex.execute(nodes, edges, 1,
                                              frame=frame, frames=2)
                    img = (flat.get(head) or {}).get("image")
                    if img is None:
                        return
                    s = np.asarray(img, dtype=np.float32)[::4, ::4]
                    if s.ndim == 3:
                        s = s.mean(axis=-1)
                    frames.append(s)
                if len(frames) < 2:
                    return
                stack = np.stack(frames)
                res["spatial"] = float(stack.mean(axis=0).var())
                res["temporal"] = float(stack.var(axis=0).mean())
            except Exception:
                pass

        th = threading.Thread(target=_run)
        th.start()
        th.join(timeout=2.5)
        shutil.rmtree(wd, ignore_errors=True)
        if th.is_alive() or "spatial" not in res:
            return None
        return res["spatial"], res["temporal"]

    def _boost_head_motion(self, rng: random.Random) -> None:
        """Ensure the render head moves: widen an existing driver, attach one
        to a drivable param, or force a strong live anim_mode on a TV head."""
        head = self.terminal_id()
        if head is None:
            return
        head_mid = self.mid(head)
        fed = self.fed_ports()
        # Widen an existing driver feeding the head.
        drv_edge = next((e for e in self.edges
                         if e["dst_node"] == head
                         and self.mid(e["src_node"]) in self.pool.scalar_drivers),
                        None)
        if drv_edge is not None:
            drv = self.node(drv_edge["src_node"])
            drv_mid = drv["method_id"]
            target = (self.pool.defs[head_mid].get("params") or {}).get(
                drv_edge["dst_port"])
            if target is not None and drv_mid in _DRIVER_RANGE_PARAMS:
                lo_k, hi_k = _DRIVER_RANGE_PARAMS[drv_mid]
                lo, hi = float(target["min"]), float(target["max"])
                drv["params"][lo_k] = round(lo, 4)
                drv["params"][hi_k] = round(hi, 4)
                if drv_mid in ("__lfo__", "__noise1d__", "__strobe__"):
                    drv["params"]["rate"] = round(rng.uniform(0.8, 2.0), 3)
            return
        # Attach a fresh driver to the best drivable param.
        cands = [(p, s) for p, s in _drivable_params(self.pool, self.cfg, head_mid)
                 if (head, p) not in fed]
        if cands:
            pname, spec = cands[0]
            drv_mid = _driver_for(pname, rng, set(self.pool.scalar_drivers))
            if drv_mid is not None:
                drv_id = self.add(drv_mid)
                _configure_driver(self, drv_id, drv_mid, spec, pname)
                self.wire(drv_id,
                          self.pool.output_port_for(drv_mid, "scalar") or "value",
                          head, pname)
            return
        # TV but nothing drivable: force a strong live anim_mode.
        if self.pool.defs[head_mid].get("is_time_varying"):
            for pname, spec in (self.pool.defs[head_mid].get("params") or {}).items():
                if isinstance(spec, dict) and "none" in (spec.get("choices") or []):
                    live = [c for c in spec["choices"] if c != "none"]
                    if live:
                        self.node(head)["params"][pname] = rng.choice(live)
                        return

    def _reroll_head_params(self, rng: random.Random) -> None:
        head = self.terminal_id()
        if head is None:
            return
        mid = self.mid(head)
        has_img = any(e["dst_node"] == head
                      and self.mid(e["src_node"]) not in self.pool.scalar_drivers
                      for e in self.edges)
        self.node(head)["params"] = sample_params(self.pool, self.cfg, rng,
                                                   mid, has_img)

    def _reroll_upstream_sources(self, rng: random.Random) -> None:
        """Flatness often lives in the head's upstream image source, not the
        head itself — re-rolling only the head can't fix a flat source feeding
        it. Re-sample the direct image/field/mask/particles sources of the head
        toward higher-variance params so the head receives varied input."""
        head = self.terminal_id()
        if head is None:
            return
        head_def = self.pool.defs[self.mid(head)]
        src_ids: set[str] = set()
        for e in self.edges:
            if e["dst_node"] != head:
                continue
            ptype = (head_def.get("inputs") or {}).get(e["dst_port"])
            if ptype in ("image", "field", "mask", "particles"):
                src_ids.add(e["src_node"])
        for sid in src_ids:
            mid = self.mid(sid)
            has_img = any(e2["dst_node"] == sid
                          and self.mid(e2["src_node"]) not in self.pool.scalar_drivers
                          for e2 in self.edges)
            self.node(sid)["params"] = sample_params(self.pool, self.cfg, rng,
                                                      mid, has_img)

    def _swap_terminal_to_filter(self, rng: random.Random) -> None:
        """Replace the render head (in place, keeping its id) with a TV filter
        that accepts an image input — keeps upstream edges valid and is
        inherently high-variance. Last resort after param re-rolls fail."""
        head = self.terminal_id()
        if head is None:
            return
        opts = [m for m in self.pool.image_producers
                if self.pool.defs[m].get("is_time_varying")
                and self._primary_image_in(m) is not None
                and m != self.mid(head)]
        if not opts:
            return
        new_mid = rng.choice(opts)
        node = self.node(head)
        node["method_id"] = new_mid
        node["params"] = sample_params(self.pool, self.cfg, rng, new_mid, True)
        # Drop incoming edges that no longer match the new node's ports.
        valid_in = set((self.pool.defs[new_mid].get("inputs") or {}).keys())
        valid_params = set((self.pool.defs[new_mid].get("params") or {}).keys())
        self.edges = [e for e in self.edges
                      if not (e["dst_node"] == head
                              and e["dst_port"] not in valid_in
                              and e["dst_port"] not in valid_params)]
        _terminal_animated_floor(self)
        self._boost_head_motion(rng)

    def ensure_terminal_variance(self, cfg, rng: random.Random) -> None:
        """Route 8 terminal guard: guarantee the render head is animated AND
        spatially/temporally varied. Cheap 2-frame probe (head + ancestors)
        re-rolls the head params and its upstream sources, or swaps the head to
        a variance-friendly filter, when output is flat/static. Sim heads are
        skipped (structural bias only — they vary)."""
        _terminal_animated_floor(self)
        head = self.terminal_id()
        if head is None:
            return
        head_mid = self.mid(head)
        if not self.pool.defs[head_mid].get("is_time_varying"):
            self._boost_head_motion(rng)
            # A non-TV head that still can't be driven must be swapped.
            if not self.pool.defs[self.mid(head)].get("is_time_varying"):
                self._swap_terminal_to_filter(rng)
                self._reroll_upstream_sources(rng)
                return
        if "n_frames" in (self.pool.defs[head_mid].get("params") or {}):
            # Sim head: structural bias only. DON'T bail — still apply best-effort
            # structural repair (boost head motion / reroll upstream sources) so a
            # sim fed by a flat source gets a varied input. The probe can't measure
            # sims cheaply, so we lean on the structural heuristics and return.
            self._boost_head_motion(rng)
            self._reroll_upstream_sources(rng)
            return
        # Up to `retries` fix cycles: re-roll head + upstream sources, re-probe.
        for _ in range(max(1, cfg.terminal_variance_retries)):
            probe = self._probe_terminal_variance(cfg)
            if probe is None:
                # Probe failed/timed out (heavy subgraph, sim ancestor, or render
                # error). NEVER bail — apply best-effort structural repair so the
                # genome still ships variance-friendly. A flat clip is far more
                # likely if we do nothing.
                self._boost_head_motion(rng)
                self._reroll_head_params(rng)
                self._reroll_upstream_sources(rng)
                continue
            spatial, temporal = probe
            if (spatial >= cfg.spatial_var_min * 1.5
                    and temporal >= cfg.temporal_var_min * 1.5):
                return
            if temporal < cfg.temporal_var_min * 1.5:
                self._boost_head_motion(rng)
            if spatial < cfg.spatial_var_min * 1.5:
                self._reroll_head_params(rng)
                self._reroll_upstream_sources(rng)
        # Final fallback: swap head to a variance-friendly filter and re-roll
        # the upstream sources + head params, retrying with different filters
        # until one passes the probe (high-variance filters like edge/threshold
        # vary far more than a random pick, so try several).
        for _ in range(5):
            self._swap_terminal_to_filter(rng)
            self._reroll_upstream_sources(rng)
            self._reroll_head_params(rng)
            probe = self._probe_terminal_variance(cfg)
            if probe is None:
                # Probe unavailable — keep the swapped filter (it's inherently
                # high-variance) and stop. Do NOT discard the improvement.
                return
            spatial, temporal = probe
            if (spatial >= cfg.spatial_var_min * 1.5
                    and temporal >= cfg.temporal_var_min * 1.5):
                return


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


# NOTE: there is deliberately no image-feedback motif. Feeding a terminal's
# image output back into an upstream image port (a "feedback loop") requires
# layering/accumulation nodes the pipeline does not yet have, so an
# auto-generated feedback edge renders incorrectly. repair.py strips any
# feedback edge and validate_graph rejects one — see BUG "auto feedback loops".
# Re-introduce a feedback motif here only once real layering nodes exist.

_BACKBONES = ["sim_backbone", "pattern_blend", "masked_composite"]

_MOTIFS: dict[str, tuple] = {
    #  name               (builder,            default_weight, backbone?)
    "sim_backbone":      (m_sim_backbone,      3.0),
    "pattern_blend":     (m_pattern_blend,     1.5),
    "masked_composite":  (m_masked_composite,  1.2),
    "field_modulate":    (m_field_modulate,    1.2),
    "post_fx":           (m_post_fx,           2.0),
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


# Per-kind default oscillation half-width for wireable params that have a
# numeric default but NO schema min/max. The generator's _drivable_params
# (and the executor's port wiring) expose many such params — phase, morph,
# offset_x, rotation, zoom, wobble, color_shift, … — that accept ANY python
# float. An LFO stuck at its node default 0..1 barely nudges these (a 0..1
# sweep into a phase that's happy at ~5.0 is invisible → the clip reads
# static and gets culled). When the target schema has no range to map onto,
# centre the driver on the param's own default and widen it by an amount
# appropriate to the kind of quantity, so the oscillation actually moves the
# rendered output. Empirics (Route 8, 2026-07-12): this single change lifts
# LFO→phase/morph/rotation graphs from temporal_var≈1e-4 (dead) to
# temporal_var≈1e-2 (alive) across the whole 96-frame clip.
_DRIVER_DEFAULT_SPAN: dict[tuple, float] = {
    # (kind frag substrings, half-width). Calibrated so an LFO sweeping the
    # param at rate≈0.6 over a 96-frame clip clears the shootout liveness
    # floor (temporal_var >= 3e-3, changed-pixel frac >= 0.03) on node 05:
    #   phase/morph half-span 3.0 → tvar≈2e-3 (DEAD); 6.0 → tvar≈6e-3 (alive);
    #   10.0 → tvar≈1.4e-2 (comfortably alive). Pad well above the floor.
    ("phase", "offset", "warp", "drift", "morph", "wobble"): 8.0,
    ("rotation", "rot", "angle", "orient", "dir"): 2.0,
    ("zoom", "scale", "size"): 3.0,
    ("color_shift", "hue", "shift", "tint"): 1.5,
    ("speed", "rate", "vel", "flow", "freq"): 1.5,
    ("amp", "strength", "intensity", "gain", "amount"): 1.0,
    ("threshold", "cutoff", "level", "density", "mix", "blend"): 0.8,
}
_DRIVER_DEFAULT_FALLBACK = 1.0  # absolute span if no kind matches


def _default_span_for(param_name: str) -> float:
    ln = param_name.lower()
    for frags, span in _DRIVER_DEFAULT_SPAN.items():
        if any(f in ln for f in frags):
            return span
    return _DRIVER_DEFAULT_FALLBACK


def _configure_driver(b: Builder, drv_id: str, drv_mid: str,
                      target_spec: dict | None,
                      target_param_name: str | None = None) -> None:
    """Map the driver's output range onto the target param and pick a
    musically-sane rate. Without this, an 0..1 LFO into a 0..500 param (or a
    wireable param with no schema range) 'drives' nothing.

    ``target_param_name`` MUST be passed (the param being driven); the edge may
    not exist yet when this runs, so it cannot be looked up from the graph. It
    selects the per-kind oscillation span for Case 2 below.

    Range mapping priority:
      1. Target schema has min/max → map onto that range (sub-window).
      2. Target has a numeric default but no min/max (the common case for
         wireable port params like phase/rotation/zoom) → centre the driver on
         the default and widen by a per-kind span so the oscillation is visible.
      3. No usable target info → leave the driver at its node default (0..1);
         this is the weak case the headless test documents as a regression risk.
    """
    rng = b.rng
    params = b.node(drv_id)["params"]
    if drv_mid in _DRIVER_RANGE_PARAMS:
        lo_k, hi_k = _DRIVER_RANGE_PARAMS[drv_mid]
        # Case 1: target schema exposes a range.
        if target_spec is not None and target_spec.get("min") is not None \
                and target_spec.get("max") is not None \
                and float(target_spec["max"]) > float(target_spec["min"]):
            lo, hi = float(target_spec["min"]), float(target_spec["max"])
            span = hi - lo
            width = span * rng.uniform(0.3, 1.0)
            start = lo + rng.uniform(0, span - width)
            params[lo_k] = round(start, 4)
            params[hi_k] = round(start + width, 4)
        # Case 2: no schema range, but the target has a numeric default we can
        # centre on (phase=0.0 → oscillate around 0 over a kind-appropriate
        # span; rotation=0.0 → ±1.5; zoom=1.0 → ~0..3; …).
        elif target_spec is not None:
            default = target_spec.get("default")
            if isinstance(default, (int, float)) and not isinstance(default, bool):
                centre = float(default)
                half = _default_span_for(target_param_name or "")
                lo = round(centre - half, 4)
                hi = round(centre + half, 4)
                params[lo_k] = lo
                params[hi_k] = hi
    if drv_mid in ("__lfo__", "__noise1d__", "__strobe__"):
        params["rate"] = round(rng.uniform(0.1, 1.5), 3)   # gentle cycles
    if drv_mid == "__ramp__":
        params["duration_frames"] = int(b.cfg.frames * rng.uniform(0.5, 1.0))
        params["mode"] = rng.choice(["loop", "pingpong", "once"]) \
            if "mode" in params else params.get("mode", "loop")


def _terminal_animated_floor(b: Builder) -> None:
    """Safety net (Route 8, 2026-07-12): a time-varying node that has NEITHER
    a control-node driver NOR a live ``anim_mode`` is born frozen and gets
    culled as ``static`` by the liveness gate even though it *could* animate.

    For every TV node not driven by a control node, ensure its animation-mode
    enum (one whose choices include ``"none"``) is set to a live (non-``none``)
    choice. ``apply_driver_policy`` already attaches drivers with high
    probability, but this catches the residual case (p_drive_primary < 1) and
    the fallback ``random_graph`` path, guaranteeing TV nodes are born
    animated. Does not touch nodes that already have a driver or a non-none
    mode — so it only ever ADDS motion, never removes it.
    """
    pool, rng = b.pool, b.rng
    fed = b.fed_ports()
    driver_ids = {n["id"] for n in b.nodes
                  if n["method_id"] in pool.scalar_drivers}
    for n in b.nodes:
        if n["method_id"] in driver_ids:
            continue
        if not pool.defs[n["method_id"]].get("is_time_varying"):
            continue
        driven = any((n["id"], p) in fed for p, _ in
                     _drivable_params(pool, b.cfg, n["method_id"]))
        if driven:
            continue
        schema = pool.defs[n["method_id"]].get("params") or {}
        for pname, spec in schema.items():
            if not isinstance(spec, dict):
                continue
            choices = spec.get("choices")
            if not choices or "none" not in choices:
                continue
            if n["params"].get(pname, "none") == "none":
                live = [c for c in choices if c != "none"]
                if live:
                    n["params"][pname] = rng.choice(live)


def apply_driver_policy(b: Builder) -> None:
    """All animation control-node driven when possible: give every
    non-driver node a driver on its most animation-relevant param
    (p_drive_primary), sometimes a second (p_drive_secondary), and
    occasionally fan one driver across two related params. Then guarantee
    every time-varying node is born animated even if no driver attached.
    """
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
                # Pass the FULL param schema (not the ranged-filtered spec that
                # _drivable_params returns) so _configure_driver can read the
                # param's default for drivers targeting wireable params that
                # have no min/max (e.g. phase, morph, rotation).
                full_spec = (pool.defs[n["method_id"]].get("params") or {}).get(pname)
                _configure_driver(b, drv_id, drv_mid, full_spec, pname)
            b.wire(drv_id, pool.output_port_for(drv_mid, "scalar") or "value",
                   n["id"], pname)
            fed.add((n["id"], pname))
            prev_driver = (drv_id, drv_mid)

    # Safety net: guarantee every TV node is born animated (Route 8).
    _terminal_animated_floor(b)


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
