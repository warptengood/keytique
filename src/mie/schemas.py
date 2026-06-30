from typing import Literal

from pydantic import BaseModel, Field


class Note(BaseModel):
    pitch: str
    onset_sec: float


class NoteAccuracy(BaseModel):
    f1: float = 0.0
    precision: float = 0.0
    recall: float = 0.0
    missed_notes: list[Note] = Field(default_factory=list)
    extra_notes: list[Note] = Field(default_factory=list)


class TimingDeviation(BaseModel):
    reference_note: Note
    onset_error_sec: float


class TimingAccuracy(BaseModel):
    mean_onset_error_sec: float
    mean_abs_onset_error_sec: float
    matched_note_count: int
    deviations: list[TimingDeviation]


class AnalysisResult(BaseModel):
    piece: str
    instruments: list[str]
    duration_sec: float
    note_accuracy: NoteAccuracy
    timing_accuracy: TimingAccuracy | None = None


class TimeCluster(BaseModel):
    start_sec: float
    end_sec: float
    error_count: int


class ProcessedTiming(BaseModel):
    mean_onset_error_sec: float
    mean_abs_onset_error_sec: float
    matched_note_count: int
    tendency: Literal["rushing", "dragging", "steady"]
    worst_deviations: list[TimingDeviation]


class ProcessedAnalysisResult(BaseModel):
    piece: str
    instruments: list[str]
    duration_sec: float
    f1: float
    precision: float
    recall: float
    missed_count: int
    extra_count: int
    most_missed_pitches: list[tuple[str, int]]
    most_missed_pitch_classes: list[tuple[str, int]]
    error_time_clusters: list[TimeCluster]
    timing: ProcessedTiming | None = None
