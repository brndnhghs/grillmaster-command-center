"""Chord Bot FastAPI server — serves the node-graph UI and REST API.

Launch:
    python -m chord_bot.server          # default: http://127.0.0.1:7861
    python -m chord_bot.server --port 8000
"""
from __future__ import annotations

import tempfile
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, PlainTextResponse

import chord_bot  # noqa: F401 — triggers all node registrations
from .executor import ChordExecutor, ChordGraphError
from .export.midi import write_midi
from .export.text import progression_to_text
from .port_types import all_port_types
from .registry import get_node_defs

_UI_FILE = Path(__file__).parent / "ui" / "index.html"

app = FastAPI(title="Chord Bot", version="1.0.0", docs_url=None, redoc_url=None)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── UI ─────────────────────────────────────────────────────────────────────────


@app.get("/")
async def index() -> HTMLResponse:
    if _UI_FILE.exists():
        return HTMLResponse(_UI_FILE.read_text())
    return HTMLResponse(
        "<html><body style='font-family:monospace;background:#0d0d14;color:#e0e0f0'>"
        "<h2>Chord Bot</h2><p>UI not found — expected at chord_bot/ui/index.html</p>"
        "</body></html>"
    )


# ── Registry endpoints ─────────────────────────────────────────────────────────


@app.get("/api/node-defs")
async def api_node_defs() -> dict:
    return get_node_defs()


@app.get("/api/port-types")
async def api_port_types() -> dict:
    return {
        name: {
            "name":         pt.name,
            "color":        pt.color,
            "description":  pt.description,
            "accepts_from": pt.accepts_from,
        }
        for name, pt in all_port_types().items()
    }


# ── Graph execution ─────────────────────────────────────────────────────────────


@app.post("/api/graph/execute")
async def api_execute(request: Request) -> list:
    body  = await request.json()
    nodes = body.get("nodes", [])
    edges = body.get("edges", [])
    try:
        seq = ChordExecutor().execute(nodes, edges)
        return [e.to_dict() for e in seq]
    except ChordGraphError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Execution error: {exc}")


# ── MIDI export ────────────────────────────────────────────────────────────────


@app.post("/api/graph/export-midi")
async def api_export_midi(request: Request) -> FileResponse:
    body  = await request.json()
    nodes = body.get("nodes", [])
    edges = body.get("edges", [])
    tempo = int(body.get("tempo", 120))
    try:
        seq = ChordExecutor().execute(nodes, edges)
        tmp = tempfile.mktemp(suffix=".mid")
        write_midi(seq, tmp, tempo_bpm=tempo, include_bass=True, include_arp=True)
        return FileResponse(
            tmp,
            media_type="audio/midi",
            filename="chord_bot.mid",
            headers={"Content-Disposition": "attachment; filename=chord_bot.mid"},
        )
    except ChordGraphError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Export error: {exc}")


# ── Spec-compliant aliases + text export ─────────────────────────────────────
# The /api/graph/* routes above are the original paths; these expose the
# flat /api/* names called by the UI and required by the server spec.


@app.get("/health")
async def health() -> dict:
    return {"ok": True}


@app.get("/api/nodes")
async def api_nodes() -> dict:
    return get_node_defs()


@app.post("/api/execute")
async def api_execute_flat(request: Request) -> list:
    body = await request.json()
    nodes = body.get("nodes", [])
    edges = body.get("edges", [])
    try:
        seq = ChordExecutor().execute(nodes, edges)
        return [e.to_dict() for e in seq]
    except ChordGraphError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Execution error: {exc}")


@app.post("/api/export/midi")
async def api_export_midi_flat(request: Request) -> FileResponse:
    body  = await request.json()
    nodes = body.get("nodes", [])
    edges = body.get("edges", [])
    tempo = int(body.get("tempo", 120))
    try:
        seq = ChordExecutor().execute(nodes, edges)
        tmp = tempfile.mktemp(suffix=".mid")
        write_midi(seq, tmp, tempo_bpm=tempo, include_bass=True, include_arp=True)
        return FileResponse(
            tmp,
            media_type="audio/midi",
            filename="chord_bot.mid",
            headers={"Content-Disposition": "attachment; filename=chord_bot.mid"},
        )
    except ChordGraphError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Export error: {exc}")


@app.post("/api/export/text")
async def api_export_text(request: Request) -> PlainTextResponse:
    body = await request.json()
    nodes = body.get("nodes", [])
    edges = body.get("edges", [])
    try:
        seq = ChordExecutor().execute(nodes, edges)
        return PlainTextResponse(progression_to_text(seq))
    except ChordGraphError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Export error: {exc}")


# ── Entry point ────────────────────────────────────────────────────────────────


def run(host: str = "127.0.0.1", port: int = 7861) -> None:
    """Launch the Chord Bot server."""
    import uvicorn
    print(f"  Chord Bot  →  http://{host}:{port}")
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Chord Bot server")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7861)
    args = parser.parse_args()
    run(host=args.host, port=args.port)
