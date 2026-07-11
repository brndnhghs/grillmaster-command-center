"""Advisor — turn the user's free-text pros/cons notes into structured
breeding guidance.

The user rates clips 1–5★ AND writes what works / what doesn't. Stars drive
the numeric selection pressure; this module handles the text: it asks the
Hermes LLM (same backend as Node Doctor, via nd_runner.py) to compress the
notes into a strict-JSON guidance object that the evolve step can apply
mechanically:

    {
      "prefer_methods":   ["86", "137"],     // method ids to sample more
      "avoid_methods":    ["58"],            // method ids to stop sampling
      "prefer_categories": ["simulations"],
      "avoid_categories":  [],
      "complexity": "increase" | "decrease" | "keep",
      "protect_genomes": ["g-aaaa1111"],     // mutate gently (params only)
      "drop_genomes":    ["g-bbbb2222"],     // never breed, regardless of stars
      "summary": "one-line restatement of the feedback"
    }

No Hermes → no guidance (stars still work); the notes are always persisted
either way, so nothing the user writes is ever lost.

Phase 3 adds *per-node* feedback: the UI lets the user tag individual nodes
in a genome ("like"/"dislike"/free text). That signal is aggregated into the
same guidance dict (prefer/avoid the node's method) WITHOUT an LLM call, so
node-level preferences steer breeding even when the LLM is unavailable.
"""
from __future__ import annotations

import json
import os
import re
import subprocess
from pathlib import Path
from typing import Callable

from .config import ShootoutConfig, DEFAULT_CONFIG
from .generator import GenePool, SamplingBias, build_gene_pool

# Same backend resolution as server.py's Node Doctor.
HERMES_AGENT_DIR = Path(
    os.environ.get("HERMES_AGENT_DIR", str(Path.home() / ".hermes" / "hermes-agent"))
)
HERMES_PYTHON = Path(
    os.environ.get("HERMES_PYTHON", str(HERMES_AGENT_DIR / "venv" / "bin" / "python"))
)
ND_RUNNER = Path(__file__).resolve().parent.parent / "nd_runner.py"

_COMPLEXITY_TO_BIAS = {"increase": 0.7, "decrease": -0.7, "keep": 0.0}

_SYSTEM_PROMPT = """You are the breeding advisor for an evolutionary generative-art system.
Users rate short clips rendered from node graphs and write free-text notes on what
works and what doesn't. Convert their notes into ONE strict JSON object with exactly
these keys (all required):

  "prefer_methods":    list of method-id strings to sample MORE often
  "avoid_methods":     list of method-id strings to sample LESS/never
  "prefer_categories": list of category names to favor
  "avoid_categories":  list of category names to suppress
  "complexity":        "increase" | "decrease" | "keep"  (graph size/wiring richness)
  "protect_genomes":   genome ids whose structure should be kept (only gentle
                       parameter mutation) — use when a note says "keep this,
                       just tweak X"
  "drop_genomes":      genome ids that should not breed at all
  "summary":           one short sentence restating the actionable feedback

Rules:
- Use ONLY method ids and category names from the provided catalog/context.
- Empty lists are fine; do not invent feedback that isn't in the notes.
- "more complex", "more nodes", "richer" → complexity "increase";
  "simpler", "too busy", "too chaotic" → "decrease"; otherwise "keep".
- Output ONLY the JSON object. No markdown, no commentary."""


# ── Per-node feedback (phase 3) ──────────────────────────────────────
# Sentiment keywords that let the advisor act on per-node feedback WITHOUT an
# LLM round-trip (cheap, deterministic). A node flagged with any negative
# word is pushed to avoid_methods; any positive word (or bare free text) is
# pushed to prefer_methods. Explicit like/dislike words win over neutral
# free text, which defaults to a mild prefer.
_POSITIVE = ("good", "like", "love", "nice", "great", "keep", "cool", "pretty",
             "beautiful", "favorite", "fav")
_NEGATIVE = ("bad", "hate", "ugly", "drop", "remove", "kill", "weird", "mess",
             "muddy", "noisy", "too much", "awful", "worst", "dislike")


def _node_sentiment(text: str) -> int:
    """-1 dislike, +1 like, 0 unknown/neutral (for a per-node note)."""
    t = text.lower()
    if any(w in t for w in _NEGATIVE):
        return -1
    if any(w in t for w in _POSITIVE):
        return 1
    return 0


def node_feedback_to_guidance(rated: list[dict],
                              pool: GenePool) -> dict[str, set]:
    """Aggregate per-node feedback into raw prefer/avoid method sets.

    `rated` elements carry a `node_feedback` map: {node_id: text}. For each
    genome we look up the node's method_id in its graph and bin it into
    prefer/avoid based on sentiment. Returns {"prefer": set, "avoid": set}
    (raw, unsanitized) — the caller merges these into the sanitized guidance
    so unknown/deprecated method ids are still dropped downstream.
    """
    prefer: set[str] = set()
    avoid: set[str] = set()
    for g in rated:
        feedback = g.get("node_feedback") or {}
        if not feedback:
            continue
        by_id = {n["id"]: n for n in g.get("graph", {}).get("nodes", [])}
        for nid, note in feedback.items():
            note = (note or "").strip()
            if not note:
                continue
            node = by_id.get(nid)
            if node is None:
                continue
            mid = node.get("method_id")
            if mid not in pool.defs:
                continue
            s = _node_sentiment(note)
            if s < 0:
                avoid.add(mid)
            else:
                prefer.add(mid)  # 0 (neutral free text) → mild prefer
    return {"prefer": prefer, "avoid": avoid}


def hermes_llm(system_prompt: str, user_msg: str,
               timeout: float = 90.0) -> str | None:
    """One-shot chat via the Hermes subprocess runner. None on any failure."""
    if not HERMES_PYTHON.exists():
        return None
    payload = json.dumps({
        "system_prompt": system_prompt,
        "messages": [{"role": "user", "content": user_msg}],
    }).encode()
    try:
        proc = subprocess.run(
            [str(HERMES_PYTHON), str(ND_RUNNER)],
            input=payload, capture_output=True, timeout=timeout,
        )
    except Exception:
        return None
    chunks: list[str] = []
    for raw in proc.stdout.decode(errors="replace").splitlines():
        try:
            d = json.loads(raw.strip())
        except Exception:
            continue
        if "text" in d:
            chunks.append(d["text"])
        elif "error" in d:
            return None
    return "".join(chunks) or None


def _method_catalog(pool: GenePool) -> str:
    """Compact id:name catalog grouped by category (LLM context)."""
    by_cat: dict[str, list[str]] = {}
    seen = set()
    for mid in pool.image_producers + pool.scalar_drivers:
        if mid in seen:
            continue
        seen.add(mid)
        d = pool.defs[mid]
        by_cat.setdefault(d.get("category", "?"), []).append(
            f"{mid}:{d.get('name', '')}")
    lines = []
    for cat in sorted(by_cat):
        lines.append(f"[{cat}] " + ", ".join(sorted(by_cat[cat])))
    return "\n".join(lines)


def _describe_generation(rated: list[dict], pool: GenePool) -> str:
    lines = []
    for g in rated:
        rating = g.get("rating")
        note = (g.get("notes") or "").strip()
        methods = ", ".join(
            f"{n['method_id']}:{pool.defs.get(n['method_id'], {}).get('name', '?')}"
            for n in g["graph"].get("nodes", []))
        lines.append(
            f"- {g['genome_id']}  rating={rating if rating is not None else 'unrated'}"
            f"  nodes={len(g['graph'].get('nodes', []))}  [{methods}]"
            + (f"\n  NOTE: {note}" if note else ""))
    return "\n".join(lines)


def _extract_json(text: str) -> dict | None:
    """First balanced {...} block in the reply, parsed."""
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if not m:
        return None
    try:
        obj = json.loads(m.group(0))
        return obj if isinstance(obj, dict) else None
    except Exception:
        return None


def _sanitize(raw: dict, rated: list[dict], pool: GenePool) -> dict:
    """Clamp the LLM output to known ids/categories — never trust it raw."""
    known_methods = set(pool.defs)
    known_cats = {d.get("category") for d in pool.defs.values()}
    known_genomes = {g["genome_id"] for g in rated}

    def _ids(key: str, universe: set) -> list[str]:
        vals = raw.get(key) or []
        if not isinstance(vals, list):
            return []
        return [str(v) for v in vals if str(v) in universe]

    complexity = raw.get("complexity")
    if complexity not in _COMPLEXITY_TO_BIAS:
        complexity = "keep"
    return {
        "prefer_methods": _ids("prefer_methods", known_methods),
        "avoid_methods": _ids("avoid_methods", known_methods),
        "prefer_categories": _ids("prefer_categories", known_cats),
        "avoid_categories": _ids("avoid_categories", known_cats),
        "complexity": complexity,
        "protect_genomes": _ids("protect_genomes", known_genomes),
        "drop_genomes": _ids("drop_genomes", known_genomes),
        "summary": str(raw.get("summary") or "")[:300],
    }


def extract_guidance(rated: list[dict],
                     pool: GenePool | None = None,
                     cfg: ShootoutConfig = DEFAULT_CONFIG,
                     llm: Callable[[str, str], str | None] | None = None,
                     ) -> dict | None:
    """Notes of a rated generation → sanitized guidance dict, or None when
    there is no signal at all (no notes AND no per-node feedback).

    Phase 3: per-node feedback is aggregated deterministically (no LLM) and
    merged with the LLM's whole-genome guidance. Either source alone is
    enough to produce guidance, so node-level preferences steer breeding
    even when the LLM backend is unavailable.
    """
    pool = pool or build_gene_pool(cfg)

    has_notes = any((g.get("notes") or "").strip() for g in rated)
    has_node_fb = any((g.get("node_feedback") or {}) for g in rated)
    if not has_notes and not has_node_fb:
        return None

    # Whole-genome notes → LLM guidance (may be None if no LLM / no notes).
    llm_guidance = None
    if has_notes:
        if llm is None:
            llm = lambda s, u: hermes_llm(s, u, timeout=cfg.advisor_timeout_s)  # noqa: E731
        user_msg = (
            "GENERATION UNDER REVIEW:\n"
            + _describe_generation(rated, pool)
            + "\n\nAVAILABLE NODE CATALOG (id:name by category):\n"
            + _method_catalog(pool)
            + "\n\nProduce the guidance JSON now."
        )
        reply = llm(_SYSTEM_PROMPT, user_msg)
        if reply:
            raw = _extract_json(reply)
            if raw is not None:
                llm_guidance = _sanitize(raw, rated, pool)

    # Per-node feedback → deterministic prefer/avoid (phase 3). Always
    # applied; it narrows the search toward/away from specific methods the
    # user tagged, independent of the LLM.
    node = node_feedback_to_guidance(rated, pool)
    if llm_guidance is None:
        if not node["prefer"] and not node["avoid"]:
            return None
        return {
            "prefer_methods": sorted(node["prefer"]),
            "avoid_methods": sorted(node["avoid"]),
            "prefer_categories": [],
            "avoid_categories": [],
            "complexity": "keep",
            "protect_genomes": [],
            "drop_genomes": [],
            "summary": "per-node feedback only",
        }

    # Merge: node feedback is unioned onto the LLM guidance.
    merged = dict(llm_guidance)
    merged["prefer_methods"] = sorted(
        set(llm_guidance.get("prefer_methods") or []) | node["prefer"])
    merged["avoid_methods"] = sorted(
        set(llm_guidance.get("avoid_methods") or []) | node["avoid"])
    note = (llm_guidance.get("summary") or "").strip()
    tags = []
    if node["prefer"]:
        tags.append(f"+{len(node['prefer'])} node(s)")
    if node["avoid"]:
        tags.append(f"-{len(node['avoid'])} node(s)")
    if tags:
        merged["summary"] = (note + f" [per-node: {' '.join(tags)}]").strip()[:300]
    return merged


def bias_from_guidance(guidance: dict | None) -> SamplingBias:
    """Guidance dict → SamplingBias for generator/evolve sampling."""
    if not guidance:
        return SamplingBias()
    return SamplingBias(
        prefer_methods=set(guidance.get("prefer_methods") or []),
        avoid_methods=set(guidance.get("avoid_methods") or []),
        prefer_categories=set(guidance.get("prefer_categories") or []),
        avoid_categories=set(guidance.get("avoid_categories") or []),
        complexity=_COMPLEXITY_TO_BIAS.get(guidance.get("complexity"), 0.0),
    )
