from typing import Protocol, runtime_checkable

from .schemas import AnalysisResult


@runtime_checkable
class Analyzer(Protocol):
    def analyze(self, audio_path: str, midi_path: str) -> AnalysisResult: ...
