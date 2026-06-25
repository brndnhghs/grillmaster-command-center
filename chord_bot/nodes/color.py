"""Color node — adds or removes a specific color tone / alteration."""
from __future__ import annotations

from ..registry import chord
from ..types import HarmonicState, note_to_pc


# Tension annotation name → semitone offset from root
_COLOR_SEMITONES: dict[str, int] = {
    "b9":  1,
    "9":   2,
    "#9":  3,
    "11":  5,
    "#11": 6,
    "b13": 8,
    "13":  9,
}


@chord(
    id="color",
    name="Color",
    category="vertical",
    axis="vertical",
    description=(
        "Adds or removes a specific color tone (b9, 9, #9, 11, #11, b13, 13) "
        "from the current chord."
    ),
    params={
        "tone": {
            "description": "color tone to affect (b9/9/#9/11/#11/b13/13)",
            "default": "9",
        },
        "action": {
            "description": "add or remove the tone",
            "default": "add",
        },
    },
)
def node_color(state: HarmonicState, params: dict) -> HarmonicState:
    tone   = str(params.get("tone",   "9"))
    action = str(params.get("action", "add"))

    semitone = _COLOR_SEMITONES.get(tone)
    if semitone is None:
        return state.copy()

    out = state.copy()
    if action == "add":
        if semitone not in out.tensions:
            out.tensions.append(semitone)
            # Adding a tension slightly raises the tension level
            out.tension = round(min(1.0, out.tension + 0.05), 3)
    elif action == "remove":
        if semitone in out.tensions:
            out.tensions.remove(semitone)
            out.tension = round(max(0.0, out.tension - 0.05), 3)

    return out
