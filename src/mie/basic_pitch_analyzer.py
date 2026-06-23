import os

import numpy as np
from mido import MidiFile, MidiTrack
from basic_pitch import ICASSP_2022_MODEL_PATH
from basic_pitch.inference import predict, Model
from mir_eval.transcription import match_notes
from pretty_midi import (
    Note,
    Instrument,
    PrettyMIDI,
    note_number_to_hz,
    note_number_to_name,
    program_to_instrument_name,
)

from ._analyzer import Analyzer
from .schemas import AnalysisResult, PitchAccuracy, NoteOnset


class BasicPitchAnalyzer(Analyzer):
    def __init__(self) -> None:
        self.model: Model = Model(ICASSP_2022_MODEL_PATH)

    def analyze(self, audio_path: str, midi_path: str) -> AnalysisResult:
        _, est_midi, _ = predict(audio_path, self.model)
        ref_midi = PrettyMIDI(midi_path)

        return AnalysisResult(
            piece=self._get_piece_name(midi_path),
            instruments=self._get_instrument_names(ref_midi),
            duration_sec=ref_midi.get_end_time(),
            pitch_accuracy=self._get_pitch_accuracy(est_midi, ref_midi),
        )

    def _get_pitch_accuracy(
        self, est_midi: PrettyMIDI, ref_midi: PrettyMIDI
    ) -> PitchAccuracy:
        est_intervals, est_pitches = self._extract_notes_from_midi(est_midi)
        ref_intervals, ref_pitches = self._extract_notes_from_midi(ref_midi)

        if len(ref_pitches) == 0 or len(est_pitches) == 0:
            return PitchAccuracy(
                f1=0, precision=0, recall=0, missed_notes=[], extra_notes=[]
            )

        global_offset = est_intervals[:, 0].min() - ref_intervals[:, 0].min()
        est_intervals -= global_offset

        matching = match_notes(
            ref_intervals,
            note_number_to_hz(ref_pitches),
            est_intervals,
            note_number_to_hz(est_pitches),
            offset_ratio=None,
        )

        hits = len(matching)
        precision = hits / len(est_pitches)
        recall = hits / len(ref_pitches)
        f1 = 2 * precision * recall / (precision + recall) if hits else 0.0

        matched_ref = {i for i, _ in matching}
        matched_est = {j for _, j in matching}

        missed_notes = [
            NoteOnset(
                pitch=note_number_to_name(ref_pitches[i]),
                time_sec=float(ref_intervals[i][0]),
            )
            for i in range(len(ref_pitches))
            if i not in matched_ref
        ]
        extra_notes = [
            NoteOnset(
                pitch=note_number_to_name(est_pitches[i]),
                time_sec=float(est_intervals[i][0]),
            )
            for i in range(len(est_pitches))
            if i not in matched_est
        ]

        return PitchAccuracy(
            f1=f1,
            precision=precision,
            recall=recall,
            missed_notes=missed_notes,
            extra_notes=extra_notes,
        )

    def _extract_notes_from_midi(
        self, midi_data: PrettyMIDI
    ) -> tuple[np.ndarray, np.ndarray]:
        intervals, pitches = [], []

        for instrument in midi_data.instruments:
            instrument: Instrument
            if instrument.is_drum:
                continue

            for note in instrument.notes:
                note: Note
                intervals.append([note.start, note.end])
                pitches.append(note.pitch)

        return np.array(intervals), np.array(pitches)

    def _get_piece_name(self, midi_path: str) -> str:
        mid = MidiFile(midi_path)
        for track in mid.tracks:
            track: MidiTrack
            if track.name != "":
                return track.name
        return os.path.splitext(os.path.basename(midi_path))[0]

    def _get_instrument_names(self, midi_data: PrettyMIDI) -> list[str]:
        names = []
        for instrument in midi_data.instruments:
            instrument: Instrument
            name = (
                "Drum"
                if instrument.is_drum
                else program_to_instrument_name(instrument.program)
            )
            names.append(name)
        return list(dict.fromkeys(names))
