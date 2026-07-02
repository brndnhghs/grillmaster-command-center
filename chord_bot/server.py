"""Chord Bot FastAPI server — serves the node-graph UI and REST API.

Launch:
    python -m chord_bot.server          # default: http://127.0.0.1:7861
    python -m chord_bot.server --port 8000
"""
from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, PlainTextResponse
from pydantic import BaseModel, Field, field_validator

import chord_bot  # noqa: F401 — triggers all node registrations
from .executor import ChordExecutor, ChordGraphError
from .export.midi import write_midi
from .export.text import progression_to_text
from .port_types import all_port_types
from .registry import get_node_defs

_UI_DIR  = Path(__file__).parent / "ui"
_UI_FILE = _UI_DIR / "index.html"

app = FastAPI(title="Chord Bot", version="1.0.0", docs_url=None, redoc_url=None)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Request models ─────────────────────────────────────────────────────────────


class NodeModel(BaseModel):
    id:             str
    type:           str
    x:              float = 0.0
    y:              float = 0.0
    params:         dict[str, Any] = Field(default_factory=dict)
    paramKeyframes: dict[str, list[dict[str, Any]]] = Field(default_factory=dict)
    dirty:          bool = True

    @field_validator("id", "type")
    @classmethod
    def not_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("must be a non-empty string")
        return v


class EdgeModel(BaseModel):
    src_node: str
    dst_node: str
    src_port: str = "harmonic_out"
    dst_port: str = "harmonic_in"


class GraphRequest(BaseModel):
    nodes: list[NodeModel] = Field(default_factory=list)
    edges: list[EdgeModel] = Field(default_factory=list)
    tempo: int = Field(default=120, ge=20, le=400)

    def node_dicts(self) -> list[dict]:
        return [n.model_dump() for n in self.nodes]

    def edge_dicts(self) -> list[dict]:
        return [e.model_dump() for e in self.edges]


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
async def api_execute(req: GraphRequest) -> list:
    try:
        seq = ChordExecutor().execute(req.node_dicts(), req.edge_dicts())
        return [e.to_dict() for e in seq]
    except ChordGraphError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Execution error: {exc}")


# ── MIDI export ────────────────────────────────────────────────────────────────


@app.post("/api/graph/export-midi")
async def api_export_midi(req: GraphRequest) -> FileResponse:
    try:
        seq = ChordExecutor().execute(req.node_dicts(), req.edge_dicts())
        tmp = tempfile.mktemp(suffix=".mid")
        write_midi(seq, tmp, tempo_bpm=req.tempo, include_bass=True, include_arp=True)
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


@app.get("/health")
async def health() -> dict:
    return {"ok": True}


TUNNEL_INFO_PATH = Path(__file__).parent.parent / "data" / "tunnel-info.json"


@app.get("/api/tunnel-url")
async def tunnel_url() -> dict:
    try:
        data = json.loads(TUNNEL_INFO_PATH.read_text())
        return data
    except (FileNotFoundError, json.JSONDecodeError):
        return {
            "chord": {"url": None, "local": "http://127.0.0.1:7861"},
            "pipeline": {"url": None, "local": "http://127.0.0.1:7860"},
        }


@app.get("/api/nodes")
async def api_nodes() -> dict:
    return get_node_defs()


@app.post("/api/execute")
async def api_execute_flat(req: GraphRequest) -> list:
    try:
        seq = ChordExecutor().execute(req.node_dicts(), req.edge_dicts())
        return [e.to_dict() for e in seq]
    except ChordGraphError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Execution error: {exc}")


@app.post("/api/export/midi")
async def api_export_midi_flat(req: GraphRequest) -> FileResponse:
    try:
        seq = ChordExecutor().execute(req.node_dicts(), req.edge_dicts())
        tmp = tempfile.mktemp(suffix=".mid")
        write_midi(seq, tmp, tempo_bpm=req.tempo, include_bass=True, include_arp=True)
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
async def api_export_text(req: GraphRequest) -> PlainTextResponse:
    try:
        seq = ChordExecutor().execute(req.node_dicts(), req.edge_dicts())
        return PlainTextResponse(progression_to_text(seq))
    except ChordGraphError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Export error: {exc}")


# ── UI static assets (JS modules) ─────────────────────────────────────────────
# Serves chord_bot/ui/*.js so the browser can fetch ES modules referenced by
# index.html's <script type="module" src="app.js">.
# This route must come AFTER all API routes so it never shadows them.

@app.get("/wiki")
async def wiki() -> HTMLResponse:
    f = _UI_DIR / "wiki.html"
    if f.exists():
        return HTMLResponse(f.read_text())
    raise HTTPException(status_code=404, detail="Wiki not found")


@app.get("/{js_file:path}")
async def ui_static(js_file: str) -> FileResponse:
    if not js_file.endswith(".js"):
        raise HTTPException(status_code=404)
    candidate = (_UI_DIR / js_file).resolve()
    # Block path traversal
    if not str(candidate).startswith(str(_UI_DIR.resolve())):
        raise HTTPException(status_code=403)
    if not candidate.is_file():
        raise HTTPException(status_code=404)
    return FileResponse(candidate, media_type="application/javascript")


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
