from pydantic import BaseModel


class NoteOnset(BaseModel):
    pitch: str
    time_sec: float


class PitchAccuracy(BaseModel):
    f1: float
    precision: float
    recall: float
    missed_notes: list[NoteOnset]
    extra_notes: list[NoteOnset]


class AnalysisResult(BaseModel):
    piece: str
    instruments: list[str]
    duration_sec: float
    pitch_accuracy: PitchAccuracy
