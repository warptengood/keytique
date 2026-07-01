import json
import os
from pathlib import Path

from dotenv import load_dotenv
from litellm import completion

from src.rag.schemas import Payload
from src.mie.schemas import ProcessedAnalysisResult


load_dotenv()

PROMPTS_DIR = Path(__file__).parent / "prompts"


class LLMClient:
    def __init__(self, model: str) -> None:
        self.model = model
        self.api_base = None
        if model.startswith("ollama"):
            self.api_base = os.getenv("OLLAMA_API_BASE")

    def build_rag_query(self, analysis: ProcessedAnalysisResult) -> str:
        return self._complete("query-builder.md", self._query_input(analysis), temperature=0)

    def get_feedback(
        self,
        analysis: ProcessedAnalysisResult,
        context: list[Payload],
        student_input: str | None = None,
    ) -> str:
        parts = [
            f"Analysis:\n{analysis.model_dump_json()}",
            f"Context:\n{self._render_context(context)}",
        ]
        if student_input and student_input.strip():
            parts.append(f"Student note:\n{student_input.strip()}")
        return self._complete("feedback.md", "\n\n".join(parts), temperature=0)

    def _render_context(self, context: list[Payload]) -> str:
        if not context:
            return "(no relevant material retrieved)"
        return "\n\n".join(f"[{i}] {p.section_title}\n{p.text}" for i, p in enumerate(context, 1))

    def _complete(self, prompt_name: str, user_message: str, *, temperature: float = 0) -> str:
        response = completion(
            model=self.model,
            messages=[
                {"role": "system", "content": self.get_prompt(prompt_name)},
                {"role": "user", "content": user_message},
            ],
            api_base=self.api_base,
            # temperature=temperature,
        )
        content = response.choices[0].message.content or ""
        return content.strip().strip('"')

    def _query_input(self, analysis: ProcessedAnalysisResult) -> str:
        payload = {
            "f1": round(analysis.f1, 2),
            "precision": round(analysis.precision, 2),
            "recall": round(analysis.recall, 2),
            "missed_count": analysis.missed_count,
            "extra_count": analysis.extra_count,
            "most_missed_pitch_classes": analysis.most_missed_pitch_classes[:5],
            "timing": None
            if analysis.timing is None
            else {
                "tendency": analysis.timing.tendency,
                "mean_abs_onset_error_sec": round(analysis.timing.mean_abs_onset_error_sec, 3),
            },
        }
        return json.dumps(payload)

    def get_prompt(self, prompt_name: str) -> str:
        text = (PROMPTS_DIR / prompt_name).read_text()
        start = text.index("## System") + len("## System")
        end = text.find("\n## ", start)
        section = text[start:] if end == -1 else text[start:end]
        return section.strip()
