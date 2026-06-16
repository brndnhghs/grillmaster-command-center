from __future__ import annotations

import pytest

from core.bms import BMSBalance, apply_record_scores, dominant_axis, score_draft_summary, score_membership_mix, score_record
from core.models import ArtifactBundle, ConstellationRecord, FragmentRecord, TitleRecord


@pytest.fixture
def title_record() -> TitleRecord:
    return TitleRecord(
        id="title_weather_girl",
        canonical_text="Weather Girl Score",
        description="Title catalog entry for a score design concept and notation map.",
        aliases=["WG score concept"],
        occurrence_headings=["Design Notes", "Score Sketches"],
        extracted_fragments=["fragment_weather_girl_01"],
        notes_count=3,
    )


@pytest.fixture
def artifact_record() -> ArtifactBundle:
    return ArtifactBundle(
        id="artifact_knock_audio",
        title="Knock Percussion Device",
        description="Audio performance document for a percussion device and rhythm study.",
        media_type="audio",
        primary_path="audio/knock-piece.wav",
        companion_note_paths=["audio/knock-design.md"],
        member_paths=["audio/knock-piece.wav", "audio/knock-alt.wav"],
        signature="audio/knock-piece.wav",
        state="indexed",
    )


@pytest.fixture
def fragment_record() -> FragmentRecord:
    return FragmentRecord(
        id="fragment_invocation",
        source_path="notes/invocation.md",
        source_title="Invocation Note",
        heading="Theory",
        excerpt="A theory note mapping language, pattern, and conceptual structure.",
        context_before="Opening question",
        context_after="Next design pass",
        normalized_hash="abc123",
    )


@pytest.fixture
def constellation_record() -> ConstellationRecord:
    return ConstellationRecord(
        id="constellation_weather_girl",
        title="Weather Girl Constellation",
        summary="A ritual invocation that binds memory, desire, and portal logic.",
        invocation="Summon the latent vision and carry it into form.",
        state="latent",
        body_reading="A gesture of breath and voice grounds the image.",
        mind_reading="The schema links title, score, and fragment into one map.",
        spirit_reading="The sigil becomes an omen and a sacred spell.",
        source_note_path="Constellations/weather-girl.md",
    )


def assert_balance_shape(balance: BMSBalance) -> None:
    assert balance.body == pytest.approx(round(balance.body, 12))
    assert balance.mind == pytest.approx(round(balance.mind, 12))
    assert balance.spirit == pytest.approx(round(balance.spirit, 12))
    assert balance.body + balance.mind + balance.spirit == pytest.approx(1.0)


def test_score_record_title_heavy_mix_is_mind_dominant(title_record: TitleRecord) -> None:
    balance = score_record(title_record)

    assert_balance_shape(balance)
    assert dominant_axis(balance) == "mind"
    assert balance.mind > balance.spirit > balance.body


def test_score_record_artifact_heavy_mix_is_body_dominant(artifact_record: ArtifactBundle) -> None:
    balance = score_record(artifact_record)

    assert_balance_shape(balance)
    assert dominant_axis(balance) == "body"
    assert balance.body > balance.mind > balance.spirit


def test_score_record_fragment_heavy_mix_is_mind_dominant(fragment_record: FragmentRecord) -> None:
    balance = score_record(fragment_record)

    assert_balance_shape(balance)
    assert dominant_axis(balance) == "mind"
    assert balance.mind > balance.body
    assert balance.mind > balance.spirit


def test_score_record_constellation_is_spirit_dominant(constellation_record: ConstellationRecord) -> None:
    balance = score_record(constellation_record)

    assert_balance_shape(balance)
    assert dominant_axis(balance) == "spirit"
    assert balance.spirit > balance.mind > balance.body


def test_apply_record_scores_sets_base_record_fields(title_record: TitleRecord) -> None:
    balance = apply_record_scores(title_record)

    assert title_record.body_score == balance.body
    assert title_record.mind_score == balance.mind
    assert title_record.spirit_score == balance.spirit


def test_score_membership_mix_balances_a_deliberate_member_set(
    title_record: TitleRecord,
    artifact_record: ArtifactBundle,
    fragment_record: FragmentRecord,
    constellation_record: ConstellationRecord,
) -> None:
    balance = score_membership_mix([title_record, artifact_record, fragment_record, constellation_record])

    assert_balance_shape(balance)
    assert balance.mind > balance.spirit > balance.body


def test_score_draft_summary_can_shift_mix_toward_spirit(
    title_record: TitleRecord,
    artifact_record: ArtifactBundle,
    fragment_record: FragmentRecord,
) -> None:
    balance = score_draft_summary(
        title="Oracle of the Weather Portal",
        summary="A ritual invocation, sacred omen, and memory spell for summoning form.",
        members=[title_record, artifact_record, fragment_record],
    )

    assert_balance_shape(balance)
    assert dominant_axis(balance) == "spirit"
    assert balance.spirit > balance.mind > balance.body
