from bisect import bisect_left
from collections import Counter

from .schemas import (
    AnalysisResult,
    Note,
    ProcessedAnalysisResult,
    ProcessedTiming,
    TimeCluster,
    TimingAccuracy,
)

# Width of each time bin used to locate where errors bunch up, in seconds.
CLUSTER_BIN_SEC = 10
# Mean onset error below this magnitude is treated as "steady" rather than
# rushing/dragging — basic-pitch onsets jitter by a few ms even on a clean take.
TIMING_TENDENCY_EPS_SEC = 0.02
# How many of the largest timing deviations to surface in the processed view.
TOP_WORST_DEVIATIONS = 5


def _pitch_class(pitch: str) -> str:
    """Strip the octave from a note name: "C#5" -> "C#", "A2" -> "A"."""
    return pitch.rstrip("-0123456789")


class Processor:
    def process(self, result: AnalysisResult) -> ProcessedAnalysisResult:
        missed = result.note_accuracy.missed_notes
        extra = result.note_accuracy.extra_notes

        return ProcessedAnalysisResult(
            piece=result.piece,
            instruments=result.instruments,
            duration_sec=result.duration_sec,
            f1=result.note_accuracy.f1,
            precision=result.note_accuracy.precision,
            recall=result.note_accuracy.recall,
            missed_count=len(missed),
            extra_count=len(extra),
            most_missed_pitches=Counter(n.pitch for n in missed).most_common(),
            most_missed_pitch_classes=Counter(_pitch_class(n.pitch) for n in missed).most_common(),
            error_time_clusters=self._error_time_clusters(missed + extra, result.duration_sec),
            timing=self._process_timing(result.timing_accuracy),
        )

    def _error_time_clusters(self, notes: list[Note], duration_sec: float) -> list[TimeCluster]:
        """Bin error onsets into fixed windows to show where mistakes bunch up.

        Bins are half-open [start, stop) so a note on a boundary is counted once.
        The range extends to cover any error past duration_sec (e.g. spurious
        late detections), not just up to the nominal duration.
        """
        if not notes:
            return []

        onsets = sorted(n.onset_sec for n in notes)
        end = max(duration_sec, onsets[-1])

        clusters = []
        start = 0.0
        while start < end:
            stop = start + CLUSTER_BIN_SEC
            count = bisect_left(onsets, stop) - bisect_left(onsets, start)
            clusters.append(TimeCluster(start_sec=start, end_sec=stop, error_count=count))
            start = stop
        return clusters

    def _process_timing(self, timing: TimingAccuracy | None) -> ProcessedTiming | None:
        if timing is None:
            return None

        mean = timing.mean_onset_error_sec
        if mean < -TIMING_TENDENCY_EPS_SEC:
            tendency = "rushing"  # plays ahead of the beat (detected onset earlier than reference)
        elif mean > TIMING_TENDENCY_EPS_SEC:
            tendency = "dragging"  # lags behind the beat
        else:
            tendency = "steady"

        worst_deviations = sorted(timing.deviations, key=lambda d: abs(d.onset_error_sec), reverse=True)[:TOP_WORST_DEVIATIONS]

        return ProcessedTiming(
            mean_onset_error_sec=timing.mean_onset_error_sec,
            mean_abs_onset_error_sec=timing.mean_abs_onset_error_sec,
            matched_note_count=timing.matched_note_count,
            tendency=tendency,
            worst_deviations=worst_deviations,
        )
