from mie.processor import CLUSTER_BIN_SEC, TIMING_TENDENCY_EPS_SEC, TOP_WORST_DEVIATIONS, Processor
from mie.schemas import (
    AnalysisResult,
    Note,
    NoteAccuracy,
    TimingAccuracy,
    TimingDeviation,
)


def make_result(
    *,
    missed: list[tuple[str, float]] = (),
    extra: list[tuple[str, float]] = (),
    f1: float = 0.0,
    precision: float = 0.0,
    recall: float = 0.0,
    duration_sec: float = 30.0,
    timing: TimingAccuracy | None = None,
) -> AnalysisResult:
    """Build a raw AnalysisResult from (pitch, onset_sec) tuples — no audio needed."""
    return AnalysisResult(
        piece="Test Piece",
        instruments=["Acoustic Grand Piano"],
        duration_sec=duration_sec,
        note_accuracy=NoteAccuracy(
            f1=f1,
            precision=precision,
            recall=recall,
            missed_notes=[Note(pitch=p, onset_sec=t) for p, t in missed],
            extra_notes=[Note(pitch=p, onset_sec=t) for p, t in extra],
        ),
        timing_accuracy=timing,
    )


# --- Pitch aggregation ---------------------------------------------------------


def test_most_missed_pitches_counted_and_sorted():
    """Pitches are counted and returned most-frequent-first."""
    result = make_result(missed=[("C#5", 1.0), ("C#5", 2.0), ("F#4", 3.0), ("C#5", 4.0), ("F#4", 5.0)])

    processed = Processor().process(result)

    assert processed.most_missed_pitches == [("C#5", 3), ("F#4", 2)]
    assert processed.missed_count == 5


def test_pitch_classes_strip_octave():
    """C#5 and C#4 collapse to one pitch class C# — the key-signature signal."""
    result = make_result(missed=[("C#5", 1.0), ("C#4", 2.0), ("F#3", 3.0), ("F#5", 4.0)])

    processed = Processor().process(result)

    assert dict(processed.most_missed_pitch_classes) == {"C#": 2, "F#": 2}


def test_scalars_and_counts_pass_through():
    result = make_result(
        missed=[("A4", 1.0)],
        extra=[("B4", 2.0), ("C5", 3.0)],
        f1=0.8,
        precision=0.75,
        recall=0.85,
    )

    processed = Processor().process(result)

    assert (processed.f1, processed.precision, processed.recall) == (0.8, 0.75, 0.85)
    assert processed.missed_count == 1
    assert processed.extra_count == 2
    assert processed.piece == "Test Piece"


# --- Time clustering -----------------------------------------------------------


def test_clusters_count_missed_and_extra_together():
    result = make_result(missed=[("A4", 1.0), ("B4", 2.0)], extra=[("C5", 3.0)], duration_sec=10.0)

    processed = Processor().process(result)

    assert len(processed.error_time_clusters) == 1
    cluster = processed.error_time_clusters[0]
    assert (cluster.start_sec, cluster.end_sec, cluster.error_count) == (0.0, CLUSTER_BIN_SEC, 3)


def test_boundary_note_counted_once():
    """A note exactly on a bin edge must not be double-counted across adjacent bins."""
    result = make_result(
        missed=[("A4", 9.9), ("B4", 10.0), ("C5", 10.1)],
        duration_sec=20.0,
    )

    processed = Processor().process(result)

    counts = [c.error_count for c in processed.error_time_clusters]
    assert sum(counts) == 3  # no double counting
    assert counts == [1, 2]  # 9.9 in [0,10); 10.0 and 10.1 in [10,20)


def test_clusters_cover_errors_past_duration():
    """Spurious detections after the nominal duration are still binned, not dropped."""
    result = make_result(missed=[("A4", 12.0)], duration_sec=5.0)

    processed = Processor().process(result)

    # Range must extend past duration_sec to reach the 12.0s error.
    total = sum(c.error_count for c in processed.error_time_clusters)
    assert total == 1
    assert processed.error_time_clusters[-1].end_sec >= 12.0


def test_no_errors_yields_no_clusters():
    processed = Processor().process(make_result())

    assert processed.error_time_clusters == []


# --- Timing --------------------------------------------------------------------


def make_timing(mean: float, deviations: list[TimingDeviation] | None = None) -> TimingAccuracy:
    if deviations is None:
        deviations = [TimingDeviation(reference_note=Note(pitch="A4", onset_sec=1.0), onset_error_sec=mean)]
    return TimingAccuracy(
        mean_onset_error_sec=mean,
        mean_abs_onset_error_sec=abs(mean),
        matched_note_count=10,
        deviations=deviations,
    )


def test_timing_tendency_rushing():
    """Negative mean onset error (plays early) => rushing."""
    processed = Processor().process(make_result(timing=make_timing(-(TIMING_TENDENCY_EPS_SEC + 0.05))))

    assert processed.timing.tendency == "rushing"


def test_timing_tendency_dragging():
    processed = Processor().process(make_result(timing=make_timing(TIMING_TENDENCY_EPS_SEC + 0.05)))

    assert processed.timing.tendency == "dragging"


def test_timing_tendency_steady_within_epsilon():
    """A few ms of jitter is steady, not a tempo problem."""
    processed = Processor().process(make_result(timing=make_timing(TIMING_TENDENCY_EPS_SEC / 2)))

    assert processed.timing.tendency == "steady"


def test_timing_absent_passes_through_as_none():
    processed = Processor().process(make_result(timing=None))

    assert processed.timing is None


def test_worst_deviations_are_largest_by_magnitude():
    """Processed view keeps only the top-N deviations by absolute error, largest first."""
    devs = [
        TimingDeviation(reference_note=Note(pitch="A4", onset_sec=float(i)), onset_error_sec=err)
        for i, err in enumerate([0.01, -0.30, 0.05, -0.02, 0.20, -0.10])
    ]
    processed = Processor().process(make_result(timing=make_timing(0.0, deviations=devs)))

    errors = [d.onset_error_sec for d in processed.timing.worst_deviations]
    assert len(errors) == TOP_WORST_DEVIATIONS  # truncated from 6 to 5
    assert errors == [-0.30, 0.20, -0.10, 0.05, -0.02]  # sorted by |error| desc
