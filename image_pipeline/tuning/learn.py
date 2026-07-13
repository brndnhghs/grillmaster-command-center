"""Lesson distillation — turn a rating + critique into durable playbook craft.

This is the mechanism by which the agent "begins to understand": each rated
attempt is distilled (by Hermes) into one general, reusable lesson and filed
under an effect/theme section of playbook.md, which is then fed back into every
future build. The Hermes runner is injectable for tests.
"""
from __future__ import annotations

from typing import Callable

from . import prompt, store
from .hermes import run_hermes

Runner = Callable[[str, list[dict]], str]


def distill_lesson(brief: str, graph: dict, rating: int, critique: str,
                   *, runner: Runner | None = None) -> tuple[str, str]:
    """Ask Hermes for one lesson. Returns (section, lesson)."""
    runner = runner or run_hermes
    text = runner(
        prompt.learn_system(),
        [{"role": "user", "content": prompt.learn_user(brief, graph, rating, critique)}],
    )
    return prompt.parse_lesson(text)


def learn(brief: str, graph: dict, rating: int, critique: str,
          *, runner: Runner | None = None) -> dict:
    """Distill and file a lesson. Returns {section, lesson, written}."""
    section, lesson = distill_lesson(brief, graph, rating, critique, runner=runner)
    written = store.append_lesson(section, lesson)
    return {"section": section, "lesson": lesson, "written": written}
