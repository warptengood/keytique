import pretty_midi

from mie.basic_pitch_analyzer import BasicPitchAnalyzer

from .conftest import make_midi

# A simple ascending melody: (MIDI pitch, start_sec, end_sec). Reused as the
# "reference" in most tests; we perturb copies of it to simulate mistakes.
MELODY = [
    (60, 0.0, 0.4),  # C4
    (64, 0.5, 0.9),  # E4
    (67, 1.0, 1.4),  # G4
    (72, 1.5, 1.9),  # C5
]


# --- Scoring logic: the core of the module -------------------------------------


def test_perfect_match_scores_one(analyzer: BasicPitchAnalyzer):
    """Identical performance and reference => P = R = F1 = 1, nothing missed/extra."""
    ref = make_midi(MELODY)
    est = make_midi(MELODY)

    acc = analyzer._get_pitch_accuracy(est, ref)

    assert acc.f1 == 1.0
    assert acc.precision == 1.0
    assert acc.recall == 1.0
    assert acc.missed_notes == []
    assert acc.extra_notes == []


def test_dropped_note_is_missed(analyzer: BasicPitchAnalyzer):
    """Performer skips G4 => recall drops, precision stays 1, G4 shows up as missed."""
    ref = make_midi(MELODY)
    est = make_midi([n for n in MELODY if n[0] != 67])  # drop G4

    acc = analyzer._get_pitch_accuracy(est, ref)

    assert acc.precision == 1.0  # everything played was correct
    assert acc.recall == 3 / 4  # but one of four ref notes is missing
    assert len(acc.missed_notes) == 1
    assert acc.missed_notes[0].pitch == "G4"
    assert acc.extra_notes == []


def test_added_note_is_extra(analyzer: BasicPitchAnalyzer):
    """Performer plays an extra A4 => precision drops, it shows up as extra."""
    ref = make_midi(MELODY)
    est = make_midi(MELODY + [(69, 2.0, 2.4)])  # extra A4

    acc = analyzer._get_pitch_accuracy(est, ref)

    assert acc.recall == 1.0  # all ref notes were played
    assert acc.precision == 4 / 5  # but one played note was spurious
    assert len(acc.extra_notes) == 1
    assert acc.extra_notes[0].pitch == "A4"
    assert acc.missed_notes == []


def test_wrong_note_is_both_missed_and_extra(analyzer: BasicPitchAnalyzer):
    """Playing F4 where E4 was written => the right note is missed AND the wrong one is extra."""
    wrong = [(65, 0.5, 0.9) if n[0] == 64 else n for n in MELODY]  # E4 -> F4
    ref = make_midi(MELODY)
    est = make_midi(wrong)

    acc = analyzer._get_pitch_accuracy(est, ref)

    assert {n.pitch for n in acc.missed_notes} == {"E4"}
    assert {n.pitch for n in acc.extra_notes} == {"F4"}


def test_constant_time_offset_is_corrected(analyzer: BasicPitchAnalyzer):
    """A performance shifted by a constant (e.g. leading silence) should still score perfectly."""
    ref = make_midi(MELODY)
    est = make_midi(
        [(p, s + 5.0, e + 5.0) for p, s, e in MELODY]
    )  # whole thing 5s late

    acc = analyzer._get_pitch_accuracy(est, ref)

    assert acc.f1 == 1.0  # offset alignment removes the constant shift


def test_empty_estimate_returns_zeros(analyzer: BasicPitchAnalyzer):
    """No detected notes => zeroed accuracy, not a crash."""
    ref = make_midi(MELODY)
    est = make_midi([])

    acc = analyzer._get_pitch_accuracy(est, ref)

    assert acc.f1 == 0.0
    assert acc.precision == 0.0
    assert acc.recall == 0.0


# --- Helper methods ------------------------------------------------------------


def test_extract_notes_skips_drums(analyzer: BasicPitchAnalyzer):
    """Drum tracks carry no pitched content and must be ignored."""
    pm = make_midi(MELODY)
    drums = pretty_midi.Instrument(program=0, is_drum=True)
    drums.notes.append(pretty_midi.Note(velocity=100, pitch=38, start=0.0, end=0.1))
    pm.instruments.append(drums)

    _, pitches = analyzer._extract_notes_from_midi(pm)

    assert len(pitches) == len(MELODY)  # the drum note is not counted


def test_instrument_names_are_deduped(analyzer: BasicPitchAnalyzer):
    """Multiple tracks of the same program collapse to one name (fugue-voice case)."""
    pm = make_midi(MELODY, program=0)
    pm.instruments.append(
        make_midi(MELODY, program=0).instruments[0]
    )  # second piano track

    names = analyzer._get_instrument_names(pm)

    assert names == ["Acoustic Grand Piano"]


# --- End-to-end test on a real recording ---------------------------------------


def test_analyze_on_real_sample(analyzer: BasicPitchAnalyzer):
    """End-to-end on a real recording, running the actual basic-pitch model.

    We don't assert an exact F1 (the model isn't deterministic to the digit) — we
    assert it lands in the plausible range we measured (~0.8) and the metadata is filled.
    """
    result = analyzer.analyze("samples/1/1.mp3", "samples/1/1.mid")

    assert result.duration_sec > 0
    assert result.instruments == ["Acoustic Grand Piano"]
    assert "BWV 850" in result.piece
    assert 0.6 < result.pitch_accuracy.f1 < 0.95
