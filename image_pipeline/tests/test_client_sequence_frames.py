"""Client-rendered graphs (3D / p5) must produce a timeline clip, like server ones.

The 3D scene-render node cooks on the browser GPU, so the server never sees its
pixels. Run used to mean two different things depending on which engine owned
the graph: the server path rendered start..end, wrote a PNG per frame and
returned a seq_name (→ clip on the timeline), while the client path rendered a
single frame into a canvas and stopped — no frames on disk, no clip.

These tests guard the piece that closes that gap: an upload route the browser
pushes each rendered frame to, landing them in the same sequence store the
timeline already reads from.
"""
import base64
import io
import re
import shutil
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from image_pipeline import server as _server

SEQ = "pytest-client-seq"


def _png_b64(color, size=(8, 6), data_url=False):
    buf = io.BytesIO()
    Image.new("RGB", size, color).save(buf, format="PNG")
    raw = base64.b64encode(buf.getvalue()).decode()
    return f"data:image/png;base64,{raw}" if data_url else raw


@pytest.fixture(scope="module")
def client():
    # No `with` — the lifespan may only be entered once per process.
    yield TestClient(_server.app)
    shutil.rmtree(str(_server.SEQUENCES_DIR / re.sub(r'[^a-zA-Z0-9_-]', '_', SEQ)),
                  ignore_errors=True)


@pytest.fixture
def seq_dir():
    d = _server.SEQUENCES_DIR / re.sub(r'[^a-zA-Z0-9_-]', '_', SEQ)
    shutil.rmtree(str(d), ignore_errors=True)
    yield d
    shutil.rmtree(str(d), ignore_errors=True)


def _upload(client, frame, color=(10, 20, 30), reset=False, data_url=False):
    return client.post(
        f"/api/sequences/{SEQ}/frames",
        json={"frame": frame, "data": _png_b64(color, data_url=data_url), "reset": reset},
    )


def test_uploaded_frames_land_where_the_timeline_reads_them(client, seq_dir):
    """A browser-rendered range becomes a scrubbable sequence, not one canvas."""
    for f in range(0, 5):
        assert _upload(client, f, reset=(f == 0)).status_code == 200

    assert sorted(p.name for p in seq_dir.glob("frame_*.png")) == [
        f"frame_{f:04d}.png" for f in range(5)
    ]
    # The timeline fetches through this route; every frame it asks for is there.
    for f in range(0, 5):
        r = client.get(f"/api/sequences/{SEQ}/{f}")
        assert r.status_code == 200, f"frame {f} missing from the sequence"

    # And the sequence is a first-class one — listing/encoding see it too.
    listed = {s["name"]: s for s in client.get("/api/sequences").json()}
    assert listed[SEQ]["frame_count"] == 5


def test_data_url_prefix_is_accepted(client, seq_dir):
    """canvas.toDataURL output posts verbatim — no client-side stripping needed."""
    assert _upload(client, 0, reset=True, data_url=True).status_code == 200
    assert (seq_dir / "frame_0000.png").exists()


def test_rerun_does_not_serve_the_previous_run_frames(client, seq_dir):
    """The JPEG the read route caches next to each PNG must not outlive it.

    Without invalidation, a second Run into the same sequence name replays the
    first run's pixels: the PNG is new, but the stale sibling JPEG is what the
    route actually serves.
    """
    _upload(client, 0, color=(255, 0, 0), reset=True)
    first = client.get(f"/api/sequences/{SEQ}/0")
    assert first.status_code == 200
    assert (seq_dir / "frame_0000.jpg").exists(), "read route should cache a JPEG"

    _upload(client, 0, color=(0, 0, 255))
    second = Image.open(io.BytesIO(client.get(f"/api/sequences/{SEQ}/0").content)).convert("RGB")
    r, g, b = second.getpixel((0, 0))
    assert b > r, f"served the stale frame — got rgb({r},{g},{b}), expected blue"


def test_reset_clears_a_longer_previous_run(client, seq_dir):
    """A re-render over a shorter range must not leave the old tail behind."""
    for f in range(0, 6):
        _upload(client, f, reset=(f == 0))
    for f in range(0, 3):
        _upload(client, f, reset=(f == 0))
    assert len(list(seq_dir.glob("frame_*.png"))) == 3


def test_bad_base64_is_rejected(client, seq_dir):
    r = client.post(f"/api/sequences/{SEQ}/frames",
                    json={"frame": 0, "data": "not base64 !!!", "reset": True})
    assert r.status_code == 400


def test_name_is_sanitised_like_the_other_sequence_routes(client):
    """Traversal characters in the name can't escape the sequence root.

    Slashes never reach the handler (they don't match the route), so the
    exposure is the rest: dots and backslashes, same as the read/delete routes.
    """
    r = client.post(r"/api/sequences/..\..\evil/frames",
                    json={"frame": 0, "data": _png_b64((1, 2, 3)), "reset": True})
    assert r.status_code == 200
    assert not (_server.SEQUENCES_DIR.parent / "evil").exists()
    assert (_server.SEQUENCES_DIR / "______evil" / "frame_0000.png").exists()
    shutil.rmtree(str(_server.SEQUENCES_DIR / "______evil"), ignore_errors=True)


# ── Browser wiring: Run on a client graph renders the range, not one frame ───

UI = Path(__file__).resolve().parents[2] / "ui" / "js"


def test_client_run_renders_the_whole_range_and_creates_a_clip():
    src = (UI / "graph.js").read_text()
    run = src[src.index("async function gDoRun()"):]
    run = run[:run.index("\n}\n")]
    assert "gClientRunSequence()" in run, \
        "Run on a client graph must render the timeline range, not a single frame"

    seq = src[src.index("async function gClientRunSequence()"):]
    seq = seq[:seq.index("\n// ── Run ")]
    assert "renderSequence" in seq
    assert "gUploadSequenceFrame" in seq, "rendered frames must reach the sequence store"
    assert "gGraphDoneSwap" in seq, "a client run must create a clip like a server run"


def test_client_executor_exposes_a_sequence_renderer():
    src = (UI / "client3d.js").read_text()
    assert "export async function renderSequence(" in src
    body = src[src.index("export async function renderSequence("):]
    body = body[:body.index("\n}\n")]
    assert "readbackPNG" in body, "frames must come back as PNGs for upload"
    assert "onFrame" in body


def test_timeline_frame_fetches_are_cache_busted_per_run():
    """Frames are served `immutable`; a re-run needs a fresh URL to be visible."""
    src = (UI / "graph.js").read_text()
    fetch = src[src.index("function _tlFetchFrame("):]
    fetch = fetch[:fetch.index("\n}\n")]
    assert "?v=" in fetch, "re-running a sequence name would replay cached frames"
