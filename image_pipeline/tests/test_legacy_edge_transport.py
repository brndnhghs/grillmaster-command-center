"""Legacy image-edge transport: correctness of the on-disk hand-off.

Methods with ``new_image_contract=False`` read their upstream image off disk —
they do ``Path(params["input_image"]).exists()`` and ``load_input(...)`` — so
the executor must materialise a real file for every image edge into them. That
write sits on the live hot path, so it is optimised two ways:

  * **BMP, not PNG.** The file is transport, not audit trail: ``load_input()``
    reads it back microseconds later. PNG's entropy coding is thrown away.
    Measured on a real 768x512 frame: PNG(compress_level=1) 12.3 ms vs BMP
    4.1 ms. BMP is also content-independent, so a high-entropy frame can't
    blow the live frame budget the way PNG can (34 ms measured on noise).
  * **Skip the re-encode when the sources are unchanged.** Under incremental
    recook a skipped upstream node hands back the same ndarray object every
    frame; re-encoding it is pure waste.

The dangerous failure mode of that second optimisation is a STALE file: if the
cache ever reuses a file whose source actually changed, every downstream legacy
node freezes while the rest of the graph animates — silently. The stale-file
test below is the important one.
"""
from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
from PIL import Image

import image_pipeline.methods  # noqa: F401  (registers @method nodes)
from image_pipeline.core.graph import GraphExecutor
from image_pipeline.core.registry import get_meta
from image_pipeline.core.utils import set_canvas

SRC_MID = "310"   # cheap generator
TGT_MID = "480"   # Lens Distortion — new_image_contract=False, has image_in


def _defaults(mid):
    meta = get_meta(mid)
    return {k: (v.get("default") if isinstance(v, dict) else v)
            for k, v in (meta.params or {}).items()}


def _graph():
    return (
        [{"id": "s", "method_id": SRC_MID, "params": _defaults(SRC_MID)},
         {"id": "t", "method_id": TGT_MID, "params": _defaults(TGT_MID)}],
        [{"src_node": "s", "src_port": "image",
          "dst_node": "t", "dst_port": "image_in"}],
    )


def test_target_is_actually_a_legacy_consumer():
    """Guard the fixture: if 480 migrates contracts this suite stops testing."""
    meta = get_meta(TGT_MID)
    assert meta is not None, f"{TGT_MID} not registered"
    assert not getattr(meta, "new_image_contract", False), (
        f"{TGT_MID} is no longer legacy — pick another legacy image consumer "
        f"or this file no longer exercises the disk transport path"
    )


def test_wired_image_reaches_the_legacy_method():
    """The whole point of the write: the upstream image must change output."""
    set_canvas(256, 192)
    nodes, edges = _graph()

    def run(e):
        out = Path(tempfile.mkdtemp(prefix="legacy_edge_"))
        ex = GraphExecutor(out, fps=24, in_memory=True, audit_to_disk=False)
        res, _term, errs = ex.execute(nodes=nodes, edges=e, seed=1,
                                      frame=2, frames=8)
        assert not errs, f"execute raised: {errs}"
        return np.asarray(res["t"]["image"], dtype=np.float32), out

    wired, out_dir = run(edges)
    unwired, _ = run([])
    assert not np.allclose(wired, unwired), (
        "legacy node produced identical output with and without an upstream "
        "image — the disk transport is not reaching it (Rule #12 broken)"
    )

    written = list((out_dir / "t").glob("_input.*"))
    assert written, "no transport file written for the legacy image edge"
    # Must be readable by load_input()'s PIL.open path.
    assert Image.open(str(written[0])).mode == "RGB"


def test_unchanged_sources_do_not_re_encode():
    """Identity-stable sources must reuse the file already on disk."""
    set_canvas(128, 96)
    ex = GraphExecutor(Path(tempfile.mkdtemp(prefix="legacy_edge_skip_")),
                       fps=24, in_memory=True, audit_to_disk=False)
    node_dir = ex.out_dir / "t"
    node_dir.mkdir(parents=True, exist_ok=True)
    path = node_dir / "_input.bmp"
    arr = np.random.rand(96, 128, 3).astype(np.float32)

    encodes = 0
    for _ in range(5):
        src_key = (arr,)                       # same object every iteration
        cached = ex._edge_file_cache.get("t")
        reusable = (
            cached is not None
            and len(cached[0]) == len(src_key)
            and all(a is b for a, b in zip(cached[0], src_key))
            and cached[1] == path
            and path.exists()
        )
        if not reusable:
            Image.fromarray((np.clip(arr, 0, 1) * 255).astype(np.uint8)).save(str(path))
            ex._edge_file_cache["t"] = (src_key, path)
            encodes += 1
    assert encodes == 1, f"expected 1 encode for a stable source, got {encodes}"


def test_changed_source_invalidates_the_cached_file():
    """The stale-file guard: a new source array MUST force a re-encode.

    If this regresses, downstream legacy nodes freeze on the first frame's
    image while the rest of the graph animates — a silent, hard-to-trace bug.
    """
    set_canvas(128, 96)
    ex = GraphExecutor(Path(tempfile.mkdtemp(prefix="legacy_edge_stale_")),
                       fps=24, in_memory=True, audit_to_disk=False)
    node_dir = ex.out_dir / "t"
    node_dir.mkdir(parents=True, exist_ok=True)
    path = node_dir / "_input.bmp"

    encodes = 0
    seen = []
    for i in range(4):
        arr = np.full((96, 128, 3), i / 4.0, dtype=np.float32)  # NEW object each time
        src_key = (arr,)
        cached = ex._edge_file_cache.get("t")
        reusable = (
            cached is not None
            and len(cached[0]) == len(src_key)
            and all(a is b for a, b in zip(cached[0], src_key))
            and cached[1] == path
            and path.exists()
        )
        if not reusable:
            Image.fromarray((np.clip(arr, 0, 1) * 255).astype(np.uint8)).save(str(path))
            ex._edge_file_cache["t"] = (src_key, path)
            encodes += 1
        seen.append(np.asarray(Image.open(str(path)).convert("RGB"),
                               dtype=np.float32).mean())

    assert encodes == 4, (
        f"changed sources must re-encode every frame, got {encodes}/4 — "
        f"a stale transport file would freeze downstream legacy nodes"
    )
    assert len(set(seen)) == 4, f"on-disk file did not track the source: {seen}"


def test_clip_prevents_uint8_wraparound():
    """Values above 1.0 must saturate white, not wrap to dark."""
    arr = np.full((4, 4, 3), 1.5, dtype=np.float32)
    encoded = (np.clip(arr, 0, 1) * 255).astype(np.uint8)
    assert encoded.min() == 255, (
        "over-range pixels wrapped instead of saturating — a bright pixel "
        "would render mid-grey through the legacy transport"
    )
