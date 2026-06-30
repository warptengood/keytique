import os
from pathlib import Path

from dotenv import load_dotenv
from litellm import completion

from src.mie.schemas import ProcessedAnalysisResult


load_dotenv()


class LLMClient:

    def __init__(self, model: str) -> None:
        self.model = model
        self.api_base = None
        if model.startswith("ollama"):
            self.api_base = os.getenv("OLLAMA_API_BASE")
    
    def build_rag_query(self, analysis: ProcessedAnalysisResult) -> str:
        response = completion(
            model=self.model,
            messages=[
                {"role": "system", "content": self.get_prompt("query-builder.md")},
                {"role": "user", "content": analysis.model_dump_json()},
            ],
            api_base=self.api_base,
        )
        return response.choices[0].message.content
    
    def get_prompt(self, prompt_name: str) -> str:
        prompt_path = Path(f"src/llm/prompts/{prompt_name}").read_text()
        return prompt_path
