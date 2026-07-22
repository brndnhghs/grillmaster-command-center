"""Hermes backend resolution must work on both platforms and find real installs.

`docs/CONFIGURATION.md` advertises `HERMES_PYTHON` as "Auto-detected". It was
not: the default was

    HERMES_AGENT_DIR / "venv" / "bin" / "python"

— a hardcoded POSIX path under a single hardcoded directory. Two consequences,
both of which made Node Doctor silently dead:

  * On Windows the venv interpreter lives at ``venv/Scripts/python.exe``, so
    the probe missed it even when HERMES_AGENT_DIR pointed at a valid install.
  * An install anywhere other than ``~/.hermes/hermes-agent`` was never found,
    and the only signal was a startup warning that scrolled past.

Hermes is the sole LLM backend for the whole pipeline (DESIGN.md), so this took
out Node Doctor and Node Tester fixes entirely.

These tests pin the resolution rules. They build fake install trees rather than
touching the real one, so they pass on any machine.
"""
from __future__ import annotations

import os

import pytest

from image_pipeline.server import _hermes_venv_python, _resolve_hermes


def _make_install(root, layout: str):
    """Create a fake hermes-agent checkout with a venv in the given layout."""
    agent = root / "hermes-agent"
    if layout == "windows":
        py = agent / "venv" / "Scripts" / "python.exe"
    else:
        py = agent / "venv" / "bin" / "python"
    py.parent.mkdir(parents=True, exist_ok=True)
    py.write_text("", encoding="utf-8")
    return agent, py


def test_finds_windows_venv_layout(tmp_path):
    agent, py = _make_install(tmp_path, "windows")
    assert _hermes_venv_python(agent) == py


def test_finds_posix_venv_layout(tmp_path):
    agent, py = _make_install(tmp_path, "posix")
    assert _hermes_venv_python(agent) == py


def test_returns_none_when_no_interpreter(tmp_path):
    agent = tmp_path / "hermes-agent"
    agent.mkdir(parents=True)
    assert _hermes_venv_python(agent) is None


def test_explicit_agent_dir_wins(tmp_path, monkeypatch):
    """An operator setting HERMES_AGENT_DIR must always be obeyed."""
    agent, py = _make_install(tmp_path, "windows")
    monkeypatch.setenv("HERMES_AGENT_DIR", str(agent))
    monkeypatch.delenv("HERMES_PYTHON", raising=False)
    got_dir, got_py = _resolve_hermes()
    assert got_dir == agent
    assert got_py == py


def test_explicit_python_wins_over_probe(tmp_path, monkeypatch):
    agent, _py = _make_install(tmp_path, "windows")
    override = tmp_path / "some-other-python"
    override.write_text("", encoding="utf-8")
    monkeypatch.setenv("HERMES_AGENT_DIR", str(agent))
    monkeypatch.setenv("HERMES_PYTHON", str(override))
    _got_dir, got_py = _resolve_hermes()
    assert got_py == override


def test_autodetects_localappdata_install(tmp_path, monkeypatch):
    """The real-world case: install under %LOCALAPPDATA%, no env var set.

    Before this, resolution looked only at ~/.hermes/hermes-agent and reported
    "backend not found" for a perfectly good install.
    """
    agent, py = _make_install(tmp_path / "hermes", "windows")
    monkeypatch.delenv("HERMES_AGENT_DIR", raising=False)
    monkeypatch.delenv("HERMES_PYTHON", raising=False)
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path))
    # Point HOME somewhere empty so the ~/.hermes candidate cannot match.
    monkeypatch.setattr("pathlib.Path.home", lambda: tmp_path / "empty-home")
    got_dir, got_py = _resolve_hermes()
    assert got_dir == agent, f"did not auto-detect LOCALAPPDATA install: {got_dir}"
    assert got_py == py


def test_falls_back_to_documented_default_when_nothing_found(tmp_path, monkeypatch):
    """With no install anywhere, resolution must still name the default path.

    The startup warning tells the user where to point HERMES_AGENT_DIR, so the
    fallback has to stay the documented location rather than something random.
    """
    monkeypatch.delenv("HERMES_AGENT_DIR", raising=False)
    monkeypatch.delenv("HERMES_PYTHON", raising=False)
    monkeypatch.delenv("LOCALAPPDATA", raising=False)
    home = tmp_path / "empty-home"
    monkeypatch.setattr("pathlib.Path.home", lambda: home)
    got_dir, got_py = _resolve_hermes()
    assert got_dir == home / ".hermes" / "hermes-agent"
    assert got_py is None
