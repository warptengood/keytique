from pathlib import Path

import pretty_midi
import numpy as np
from mido import MidiFile, MidiTrack
from basic_pitch import ICASSP_2022_MODEL_PATH
from basic_pitch.inference import predict, Model
from mir_eval.transcription import match_notes

from ._analyzer import Analyzer
from .schemas import AnalysisResult, NoteAccuracy, Note, TimingAccuracy, TimingDeviation


TIME_ACCURACY_ONSET_TOLERANCE = 0.15
TOP_WORST_DEVIATIONS = 5


class BasicPitchAnalyzer(Analyzer):
    def __init__(self) -> None:
        self.model: Model = Model(ICASSP_2022_MODEL_PATH)

    def analyze(self, audio_path: str, midi_path: str) -> AnalysisResult:
        _, est_midi, _ = predict(audio_path, self.model)
        ref_midi = pretty_midi.PrettyMIDI(midi_path)

        est_intervals, est_pitches = self._extract_notes_from_midi(est_midi)
        ref_intervals, ref_pitches = self._extract_notes_from_midi(ref_midi)

        self._fix_global_onset_error(est_intervals, ref_intervals)

        return AnalysisResult(
            piece=self._get_piece_name(midi_path),
            instruments=self._get_instrument_names(ref_midi),
            duration_sec=ref_midi.get_end_time(),
            note_accuracy=self._get_note_accuracy(est_intervals, est_pitches, ref_intervals, ref_pitches),
            timing_accuracy=self._get_timing_accuracy(est_intervals, est_pitches, ref_intervals, ref_pitches),
        )

    def _fix_global_onset_error(self, est_intervals: np.ndarray, ref_intervals: np.ndarray) -> None:
        # Remove the leading silence
        global_error = est_intervals[:, 0].min() - ref_intervals[:, 0].min()
        est_intervals -= global_error

    def _get_note_accuracy(
        self,
        est_intervals: np.ndarray,
        est_pitches: np.ndarray,
        ref_intervals: np.ndarray,
        ref_pitches: np.ndarray,
    ) -> NoteAccuracy:
        if len(ref_pitches) == 0 or len(est_pitches) == 0:
            return NoteAccuracy()

        matching = match_notes(
            ref_intervals,
            pretty_midi.note_number_to_hz(ref_pitches),
            est_intervals,
            pretty_midi.note_number_to_hz(est_pitches),
            offset_ratio=None,
        )

        hits = len(matching)
        precision = hits / len(est_pitches)
        recall = hits / len(ref_pitches)
        f1 = 2 * precision * recall / (precision + recall) if hits else 0.0

        matched_ref = {i for i, _ in matching}
        matched_est = {j for _, j in matching}

        missed_notes = [
            Note(
                pitch=pretty_midi.note_number_to_name(ref_pitches[i]),
                onset_sec=float(ref_intervals[i][0]),
            )
            for i in range(len(ref_pitches))
            if i not in matched_ref
        ]
        extra_notes = [
            Note(
                pitch=pretty_midi.note_number_to_name(est_pitches[i]),
                onset_sec=float(est_intervals[i][0]),
            )
            for i in range(len(est_pitches))
            if i not in matched_est
        ]

        return NoteAccuracy(
            f1=f1,
            precision=precision,
            recall=recall,
            missed_notes=missed_notes,
            extra_notes=extra_notes,
        )

    def _get_timing_accuracy(
        self,
        est_intervals: np.ndarray,
        est_pitches: np.ndarray,
        ref_intervals: np.ndarray,
        ref_pitches: np.ndarray,
    ) -> TimingAccuracy | None:
        if len(ref_pitches) == 0 or len(est_pitches) == 0:
            return None

        matching = match_notes(
            ref_intervals,
            pretty_midi.note_number_to_hz(ref_pitches),
            est_intervals,
            pretty_midi.note_number_to_hz(est_pitches),
            onset_tolerance=TIME_ACCURACY_ONSET_TOLERANCE,
            offset_ratio=None,
        )

        if not matching:
            return None

        pairs = np.array(matching)  # shape (N, 2): columns are (ref_i, est_j)
        onset_error_sec = est_intervals[pairs[:, 1], 0] - ref_intervals[pairs[:, 0], 0]
        abs_error = np.abs(onset_error_sec)

        worst = np.argsort(-abs_error, kind="stable")[:TOP_WORST_DEVIATIONS]

        return TimingAccuracy(
            mean_onset_error_sec=float(onset_error_sec.mean()),
            mean_abs_onset_error_sec=float(abs_error.mean()),
            matched_note_count=len(matching),
            worst_deviations=[
                TimingDeviation(
                    reference_note=Note(
                        pitch=pretty_midi.note_number_to_name(ref_pitches[pairs[k, 0]]),
                        onset_sec=float(ref_intervals[pairs[k, 0]][0]),
                    ),
                    onset_error_sec=float(onset_error_sec[k]),
                )
                for k in worst
            ],
        )

    def _extract_notes_from_midi(self, midi_data: pretty_midi.PrettyMIDI) -> tuple[np.ndarray, np.ndarray]:
        intervals, pitches = [], []

        for instrument in midi_data.instruments:
            instrument: pretty_midi.Instrument
            if instrument.is_drum:
                continue

            for note in instrument.notes:
                note: pretty_midi.Note
                intervals.append([note.start, note.end])
                pitches.append(note.pitch)

        return np.array(intervals), np.array(pitches)

    def _get_piece_name(self, midi_path: str) -> str:
        mid = MidiFile(midi_path)
        for track in mid.tracks:
            track: MidiTrack
            if track.name != "":
                return track.name
        return Path(midi_path).stem

    def _get_instrument_names(self, midi_data: pretty_midi.PrettyMIDI) -> list[str]:
        names = []
        for instrument in midi_data.instruments:
            instrument: pretty_midi.Instrument
            name = "Drum" if instrument.is_drum else pretty_midi.program_to_instrument_name(instrument.program)
            names.append(name)
        return list(dict.fromkeys(names))
