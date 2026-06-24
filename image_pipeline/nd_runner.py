#!/usr/bin/env python3
"""NODE DOCTOR subprocess runner — executed under hermes's own Python venv.

stdin  : JSON  {system_prompt: str, messages: [{role, content}]}
stdout : newline-delimited JSON lines
         {"text": "..."}   — streaming delta
         {"done": true}    — finished successfully
         {"error": "..."}  — fatal error
"""
from __future__ import annotations
import json, logging, os, sys
from pathlib import Path

HERMES_AGENT = Path.home() / ".hermes" / "hermes-agent"
sys.path.insert(0, str(HERMES_AGENT))

os.environ.setdefault("HERMES_YOLO_MODE", "1")
os.environ.setdefault("HERMES_ACCEPT_HOOKS", "1")
logging.disable(logging.CRITICAL)

# Suppress all stderr noise from hermes internals
sys.stderr = open(os.devnull, "w")

real_stdout = sys.stdout


def emit(obj: dict):
    real_stdout.write(json.dumps(obj) + "\n")
    real_stdout.flush()


try:
    data = json.load(sys.stdin)
    system_prompt = data.get("system_prompt", "")
    messages: list = data.get("messages", [])

    # Split into history + current user message
    if messages and messages[-1]["role"] == "user":
        user_msg = messages[-1]["content"]
        history = messages[:-1]
    else:
        user_msg = ""
        history = messages

    from hermes_cli.config import load_config
    from run_agent import AIAgent

    cfg = load_config()
    model_cfg = cfg.get("model") or {}
    cfg_model = (
        (model_cfg.get("default") or model_cfg.get("model") or "")
        if isinstance(model_cfg, dict)
        else str(model_cfg)
    )

    agent = AIAgent(
        model=cfg_model,
        enabled_toolsets=[],   # pure chat — no tool execution
        quiet_mode=True,
        ephemeral_system_prompt=system_prompt,
    )

    def on_stream(text: str):
        emit({"text": text})

    agent.run_conversation(
        user_msg,
        system_message=system_prompt,
        conversation_history=history,
        stream_callback=on_stream,
    )
    emit({"done": True})

except Exception as exc:
    emit({"error": str(exc)})
