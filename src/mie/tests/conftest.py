import pretty_midi
import pytest

from mie.basic_pitch_analyzer import BasicPitchAnalyzer


def make_midi(notes: list[tuple[int, float, float]], program: int = 0) -> pretty_midi.PrettyMIDI:
    """Build an in-memory MIDI from (pitch, start, end) tuples."""
    pm = pretty_midi.PrettyMIDI()
    inst = pretty_midi.Instrument(program=program)
    for pitch, start, end in notes:
        inst.notes.append(pretty_midi.Note(velocity=100, pitch=pitch, start=start, end=end))
    pm.instruments.append(inst)
    return pm


def score(analyzer: BasicPitchAnalyzer, est: pretty_midi.PrettyMIDI, ref: pretty_midi.PrettyMIDI):
    """Extract notes from both MIDIs and run both scorers. Returns (note_accuracy, timing_accuracy)."""
    est_intervals, est_pitches = analyzer._extract_notes_from_midi(est)
    ref_intervals, ref_pitches = analyzer._extract_notes_from_midi(ref)
    note_acc = analyzer._get_note_accuracy(est_intervals, est_pitches, ref_intervals, ref_pitches)
    time_acc = analyzer._get_timing_accuracy(est_intervals, est_pitches, ref_intervals, ref_pitches)
    return note_acc, time_acc


@pytest.fixture(scope="session")
def analyzer() -> BasicPitchAnalyzer:
    """One analyzer for the whole test session."""
    return BasicPitchAnalyzer()
