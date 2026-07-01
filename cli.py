import argparse

from qdrant_client import QdrantClient

from src.rag.indexer import Indexer
from src.llm.client import LLMClient
from src.mie._analyzer import Analyzer
from src.mie.processor import Processor
from src.rag.retriever import Retriever
from src.rag.embedding_model import EMBEDDING_MODELS
from src.mie.basic_pitch_analyzer import BasicPitchAnalyzer


ANALYZERS = {"basic_pitch": BasicPitchAnalyzer}


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("CLI Interface for Music Information Extraction.")
    parser.add_argument("input", nargs=2, help="Path to audio file and .mid file. Enter two paths (audio then midi) separated by a space.")
    parser.add_argument("--method", choices=[ANALYZERS.keys()], default="basic_pitch", help="Method to analyze input data.")
    parser.add_argument("--embedding-model", default="bge-small-en-v1.5", choices=EMBEDDING_MODELS.keys(), help="Embedding model for RAG.")
    parser.add_argument("--llm-provider", default="openai/gpt-4o", help="LLM provider like `openai/gpt-4o` or `ollama/deepseek-r1:14b`.")
    parser.add_argument("--user-input", default=None, help="Additional input for LLM.")
    args = parser.parse_args()
    return args


def main():
    args = get_args()
    analyzer: Analyzer = ANALYZERS[args.method]()
    analysis_result = analyzer.analyze(*args.input)

    processor = Processor()
    processed_analysis_result = processor.process(analysis_result)

    llm_client = LLMClient(args.llm_provider)
    query= llm_client.build_rag_query(processed_analysis_result)

    qdrant_client = QdrantClient(url="http://localhost:6333")
    embedding_model = EMBEDDING_MODELS[args.embedding_model]()

    indexer = Indexer(qdrant_client, embedding_model)
    indexer.delete_collection()
    indexer.build_index()

    retriever = Retriever(qdrant_client, embedding_model)
    context = retriever.retrieve(query)

    response = llm_client.get_feedback(processed_analysis_result, context, args.user_input)
    print(response)

if __name__ == "__main__":
    main()
