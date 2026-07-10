"""Typed-port contract for GPU-sourced nodes (GPU-First guardrail #3).

This is the missing test called out in the GPU-First Build Plan's "Verification
additions (headless, pre-commit)" list. It enforces the *data-typed port
contract* the graph wiring layer relies on to reject type-incompatible
connections.

The authoritative port-type vocabulary is the registry's INTERNAL types
(``MethodMeta.inputs`` / ``.outputs``), which ``core/graph.py`` lowers and
serializes to the client node-defs JSON. The internal vocabulary (see
``core/port_types.py`` + ``graph._PORT_TYPE_MAP``) is:

    IMAGE      -- RGBA float full-color frame
    MASK       -- single-channel alpha / binary
    FIELD      -- scalar field (luminance = mean over channels)
    SCALAR     -- single numeric output (score / 0..1)
    VEC2/VE3/COLOR -- structured params
    STATE      -- ping-pong RGBA-float sim state (sims only)
    PARTICLES  -- agent/particle positions (x/y/vx/vy)
    COLORMAP   -- color palette / lookup table (N,3 or N,4)
    ANY        -- non-image data structure with no vocab slot (escape hatch)

A node that emits a port typed with a typo (e.g. ``IMGAGE``) is a real bug: the
serializer falls through to ``any`` and silently flattens the type, so the
client resolver and the wiring layer both lose the real type.

Two documented follow-ups (do NOT widen the allowed set without a reason):

* ``STATE`` exposure on sims -- the GPU-First plan states "sims additionally
  expose STATE", but that step is NOT yet implemented in node-defs: the
  ping-pong state is owned client-side and sims currently expose ``IMAGE`` (and
  sometimes ``FIELD``/``luminance``). The test therefore documents STATE as a
  known follow-up and asserts the weaker, currently-true property that every GPU
  sim exposes at least one ``IMAGE`` output.

* ``COLORMAP`` (node 10 "Color Palette") serializes to ``any`` in the client
  JSON because ``graph._PORT_TYPE_MAP`` has no COLORMAP→client entry yet. The
  internal type is correct; this is a serialization gap, not a registry
  violation. The test allows COLORMAP (it is a registered port type) and notes
  the mapping is future work.

Run (headless, no browser needed):
  cd ~/Documents/GitHub/grillmaster-command-center
  env -u PYTHONPATH .venv/bin/python -m pytest image_pipeline/tests/test_gpu_node_typed_ports.py -q
"""
from image_pipeline.core.registry import get_all
from image_pipeline.methods.gpu_shaders import GPU_SHADER_NODE_MAP

# Authoritative internal port-type vocabulary (core/port_types.py + graph.py).
VOCAB = {
    "image", "mask", "field", "scalar",
    "vec2", "vec3", "color", "state",
    "particles", "colormap", "any",
}


def _meta_for(mid: str, allm: dict):
    return allm.get(str(mid)) or allm.get(str(mid).zfill(2))


def test_gpu_sourced_ports_use_vocabulary():
    """Every GPU-sourced node's input/output port types are in the vocabulary.

    The registry's internal types are the single source of truth; a typo (e.g.
    ``IMGAGE``) would fall through to ``any`` in the client serializer and
    silently flatten the type. This fails the build on any such regression.
    """
    import image_pipeline.methods  # noqa: F401 — ensure registration

    allm = get_all()
    breaches = []

    for mid in GPU_SHADER_NODE_MAP:
        meta = _meta_for(str(mid), allm)
        if meta is None:
            breaches.append((str(mid), "<meta missing>", None, None))
            continue
        for port, typ in (meta.inputs or {}).items():
            if typ.lower() not in VOCAB:
                breaches.append((str(mid), "inputs", port, typ))
        for port, typ in (meta.outputs or {}).items():
            if typ.lower() not in VOCAB:
                breaches.append((str(mid), "outputs", port, typ))

    assert not breaches, (
        "GPU-sourced node(s) advertise port type(s) outside the typed-port "
        "vocabulary (graph wiring / client resolver would misbehave -- a typo "
        "here silently falls through to `any`):\n"
        + "\n".join(
            f"  node {m}: {direction}['{port}'] = {typ!r} "
            f"(not in {sorted(VOCAB)})"
            for m, direction, port, typ in breaches
        )
    )


def test_gpu_sim_exposes_image_output():
    """GPU sims must at least expose an ``IMAGE`` output (STATE is future work).

    The GPU-First plan's full contract is "sims additionally expose STATE", but
    STATE ports are not yet surfaced in node-defs (the ping-pong state is
    client-internal). Until that lands, the minimum guaranteed contract is that
    a sim that renders produces an ``IMAGE`` output.
    """
    import image_pipeline.methods  # noqa: F401 — ensure registration

    allm = get_all()
    sims_missing_image = []
    for mid, entry in GPU_SHADER_NODE_MAP.items():
        if entry.get("type") != "sim":
            continue
        meta = _meta_for(str(mid), allm)
        if meta is None:
            continue
        outs = {t.lower() for t in (meta.outputs or {}).values()}
        if "image" not in outs:
            sims_missing_image.append((str(mid), meta.name, meta.outputs))

    assert not sims_missing_image, (
        "GPU sim node(s) expose no `IMAGE` output (the sim must render to an "
        "image at minimum):\n"
        + "\n".join(
            f"  node {m} ({name}): outputs={outs}"
            for m, name, outs in sims_missing_image
        )
    )
