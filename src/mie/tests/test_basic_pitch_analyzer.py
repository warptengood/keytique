import math

import pretty_midi

from mie.basic_pitch_analyzer import BasicPitchAnalyzer, TIME_ACCURACY_ONSET_TOLERANCE

from .conftest import make_midi, score

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

    note_acc, time_acc = score(analyzer, est, ref)

    assert note_acc.f1 == 1.0
    assert note_acc.precision == 1.0
    assert note_acc.recall == 1.0
    assert note_acc.missed_notes == []
    assert note_acc.extra_notes == []
    assert time_acc.mean_onset_error_sec == 0.0
    assert time_acc.mean_abs_onset_error_sec == 0.0
    assert time_acc.matched_note_count == len(MELODY)

    for time_deviation in time_acc.deviations:
        assert time_deviation.onset_error_sec == 0.0


def test_dropped_note_is_missed(analyzer: BasicPitchAnalyzer):
    """Performer skips G4 => recall drops, precision stays 1, G4 shows up as missed."""
    ref = make_midi(MELODY)
    est = make_midi([n for n in MELODY if n[0] != 67])  # drop G4

    note_acc, time_acc = score(analyzer, est, ref)

    assert note_acc.precision == 1.0  # everything played was correct
    assert note_acc.recall == (len(MELODY) - 1) / len(MELODY)  # but one of ref notes is missing
    assert len(note_acc.missed_notes) == 1
    assert note_acc.missed_notes[0].pitch == "G4"
    assert note_acc.extra_notes == []
    assert time_acc.mean_onset_error_sec == 0.0
    assert time_acc.mean_abs_onset_error_sec == 0.0
    assert time_acc.matched_note_count == len(MELODY) - 1

    for time_deviation in time_acc.deviations:
        assert time_deviation.onset_error_sec == 0.0


def test_added_note_is_extra(analyzer: BasicPitchAnalyzer):
    """Performer plays an extra A4 => precision drops, it shows up as extra."""
    ref = make_midi(MELODY)
    est = make_midi(MELODY + [(69, 2.0, 2.4)])  # extra A4

    note_acc, time_acc = score(analyzer, est, ref)

    assert note_acc.recall == 1.0  # all ref notes were played
    assert note_acc.precision == len(MELODY) / (len(MELODY) + 1)  # but one played note was spurious
    assert len(note_acc.extra_notes) == 1
    assert note_acc.extra_notes[0].pitch == "A4"
    assert note_acc.missed_notes == []
    assert time_acc.mean_onset_error_sec == 0.0
    assert time_acc.mean_abs_onset_error_sec == 0.0
    assert time_acc.matched_note_count == len(MELODY)

    for time_deviation in time_acc.deviations:
        assert time_deviation.onset_error_sec == 0.0


def test_wrong_note_is_both_missed_and_extra(analyzer: BasicPitchAnalyzer):
    """Playing F4 where E4 was written => the right note is missed AND the wrong one is extra."""
    wrong = [(65, 0.5, 0.9) if n[0] == 64 else n for n in MELODY]  # E4 -> F4
    ref = make_midi(MELODY)
    est = make_midi(wrong)

    note_acc, _ = score(analyzer, est, ref)

    assert {n.pitch for n in note_acc.missed_notes} == {"E4"}
    assert {n.pitch for n in note_acc.extra_notes} == {"F4"}


def test_global_onset_error_is_corrected(analyzer: BasicPitchAnalyzer):
    """A performance shifted by a constant (e.g. leading silence) should still score perfectly."""
    ref = make_midi(MELODY)
    est = make_midi([(p, s + 5.0, e + 5.0) for p, s, e in MELODY])  # whole thing 5s late

    est_intervals, est_pitches = analyzer._extract_notes_from_midi(est)
    ref_intervals, ref_pitches = analyzer._extract_notes_from_midi(ref)

    analyzer._fix_global_onset_error(est_intervals, ref_intervals)

    note_acc = analyzer._get_note_accuracy(est_intervals, est_pitches, ref_intervals, ref_pitches)

    assert note_acc.f1 == 1.0  # offset alignment removes the constant shift


def test_empty_estimate_returns_zeros(analyzer: BasicPitchAnalyzer):
    """No detected notes => zeroed accuracy, not a crash."""
    ref = make_midi(MELODY)
    est = make_midi([])

    note_acc, time_acc = score(analyzer, est, ref)

    assert note_acc.f1 == 0.0
    assert note_acc.precision == 0.0
    assert note_acc.recall == 0.0
    assert time_acc is None


def test_timing_rushing(analyzer: BasicPitchAnalyzer):
    """Timing tempo is rushing"""
    ref = make_midi(MELODY)
    shift = TIME_ACCURACY_ONSET_TOLERANCE / len(MELODY)
    est = make_midi([(p, s + shift * (i + 1), e + shift * (i + 1)) for i, (p, s, e) in enumerate(MELODY)])

    _, time_acc = score(analyzer, est, ref)

    expected = [shift * (i + 1) for i in range(len(MELODY))]  # in reference-onset order

    assert math.isclose(time_acc.mean_onset_error_sec, sum(expected) / len(expected), rel_tol=1e-4)
    assert time_acc.matched_note_count == len(MELODY)
    assert len(time_acc.deviations) == len(MELODY)  # all deviations kept, not just the worst

    for dev, exp in zip(time_acc.deviations, expected):
        assert math.isclose(dev.onset_error_sec, exp, rel_tol=1e-4)


def test_timing_chaos(analyzer: BasicPitchAnalyzer):
    """Timing tempo is chaotic"""
    ref = make_midi(MELODY)
    shift = TIME_ACCURACY_ONSET_TOLERANCE / len(MELODY)
    est = make_midi([(p, s + shift * (i + 1) * ((-1) ** i), e + shift * (i + 1) * ((-1) ** i)) for i, (p, s, e) in enumerate(MELODY)])

    _, time_acc = score(analyzer, est, ref)

    expected = [shift * (i + 1) * ((-1) ** i) for i in range(len(MELODY))]  # in reference-onset order

    assert math.isclose(time_acc.mean_abs_onset_error_sec, sum(abs(d) for d in expected) / len(expected), rel_tol=1e-4)
    assert time_acc.matched_note_count == len(MELODY)
    assert len(time_acc.deviations) == len(MELODY)

    for dev, exp in zip(time_acc.deviations, expected):
        assert math.isclose(dev.onset_error_sec, exp, rel_tol=1e-4)


# --- Helper methods ------------------------------------------------------------


def test_extract_notes_from_midi(analyzer: BasicPitchAnalyzer):
    """Extracted notes correspond to the data in MIDI"""
    pm = make_midi(MELODY)
    intervals, pitches = analyzer._extract_notes_from_midi(pm)

    ref_pitches = [note[0] for note in MELODY]
    ref_intervals = [[note[1], note[2]] for note in MELODY]

    assert (pitches == ref_pitches).all()
    assert (intervals == ref_intervals).all()


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
    pm.instruments.append(make_midi(MELODY, program=0).instruments[0])  # second piano track

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
    assert 0.6 < result.note_accuracy.f1 < 0.95
