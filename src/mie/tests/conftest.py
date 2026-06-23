import pretty_midi
import pytest

from mie.basic_pitch_analyzer import BasicPitchAnalyzer


def make_midi(
    notes: list[tuple[int, float, float]], program: int = 0
) -> pretty_midi.PrettyMIDI:
    """Build an in-memory MIDI from (pitch, start, end) tuples."""
    pm = pretty_midi.PrettyMIDI()
    inst = pretty_midi.Instrument(program=program)
    for pitch, start, end in notes:
        inst.notes.append(
            pretty_midi.Note(velocity=100, pitch=pitch, start=start, end=end)
        )
    pm.instruments.append(inst)
    return pm


@pytest.fixture(scope="session")
def analyzer() -> BasicPitchAnalyzer:
    """One analyzer for the whole test session."""
    return BasicPitchAnalyzer()
