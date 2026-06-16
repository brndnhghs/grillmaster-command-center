from __future__ import annotations

from core.state_machine import (
    DEFAULT_CONSTELLATION_STATE,
    available_constellation_states,
    describe_state,
    normalize_constellation_state,
    validate_transition,
)



def test_state_machine_normalizes_values_to_spec_states():
    assert normalize_constellation_state(None) == DEFAULT_CONSTELLATION_STATE
    assert normalize_constellation_state("MANIFESTED") == "manifested"
    assert normalize_constellation_state("weird") == "latent"



def test_state_machine_exposes_expected_states():
    assert available_constellation_states() == ["latent", "manifested", "stalled"]



def test_state_machine_describes_and_validates_transitions():
    assert "Latent" in describe_state("latent")
    allowed = validate_transition("latent", "manifested")
    assert allowed.allowed is True
    blocked = validate_transition("latent", "latent")
    assert blocked.allowed is True
