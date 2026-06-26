import json
import math
import argparse
from enum import StrEnum, auto

from pydantic import BaseModel
from qdrant_client import QdrantClient

from ..indexer import Indexer
from ..schemas import SkillArea
from ..retriever import Retriever
from ..config import LOCAL_CORPUS_PATH
from ..embedding_model import EMBEDDING_MODELS


class Category(StrEnum):
    PITCH = auto()
    TIMING = auto()
    BOTH = auto()
    NOT_RELEVANT = auto()
    DISTANT = auto()
    THEORY = auto()


class Chunk(BaseModel):
    doc_id: str
    section_title: str
    grade: int


class Query(BaseModel):
    id: str
    query: str
    category: str
    skill_area: SkillArea | None
    relevant: list[Chunk]
    notes: str


class EvaluationSuit(BaseModel):
    version: str
    queries: list[Query]


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluation for Retriever")
    parser.add_argument("--eval-file-path", default=LOCAL_CORPUS_PATH / "eval_set.json", help="Path to JSON file with evaluation suit.")
    parser.add_argument("--embedder", choices=EMBEDDING_MODELS.keys(), default="all-MiniLM-L6-v2", help=f"Type of embedding model to use. Choose among: {EMBEDDING_MODELS.keys()}.")
    args = parser.parse_args()
    return args    


def recall_at_k(relevant_dict: dict[tuple, int], retrieved_list: list[tuple], k: int) -> float:
    relevant_in_k = sum(1 for item in retrieved_list[:k] if item in relevant_dict)
    return relevant_in_k / len(relevant_dict)


def reciprocal_rank_at_k(relevant_dict: dict[tuple, int], retrieved_list: list[tuple], k: int) -> float:
    for index, item in enumerate(retrieved_list[:k], start=1):
        if item in relevant_dict:
            return 1 / index
    return 0.0


def normalized_discounted_cumulative_gaint_at_k(relevant_dict: dict[tuple, int], retrieved_list: list[tuple], k: int) -> float:
    dcg_at_k = 0.0
    for index, item in enumerate(retrieved_list[:k], start=1):
        if item in relevant_dict:
            dcg_at_k += relevant_dict[item] / math.log2(index + 1)
    
    idcg_at_k = sum(grade / math.log2(index + 1) for index, grade in enumerate(sorted(relevant_dict.values(), reverse=True)[:k], start=1))
    return dcg_at_k / idcg_at_k


def main():
    args = get_args()
    with open(args.eval_file_path, "r") as f:
        eval_suit = EvaluationSuit(**json.load(f))
    
    client = QdrantClient(url="http://localhost:6333")
    embedder = EMBEDDING_MODELS[args.embedder]()

    indexer = Indexer(client, embedder)
    indexer.delete_collection()
    indexer.build_index()

    retriever = Retriever(client, embedder)

    recall_at_3, recall_at_5 = [], []
    rr_at_3, rr_at_5 = [], []
    ndcg_at_3, ndcg_at_5 = [], [] 

    for query_object in eval_suit.queries:
        payloads = retriever.retrieve(query_object.query, top_k=5)

        relevant_dict = {(chunk.doc_id, chunk.section_title): chunk.grade for chunk in query_object.relevant}
        retrieved_list = [(payload.doc_id, payload.section_title) for payload in payloads]

        if len(relevant_dict) == 0:
            # skip the queries with no relevant chunks for now, not sure what to do with them
            continue
        
        recall_at_3.append(recall_at_k(relevant_dict, retrieved_list, 3))
        recall_at_5.append(recall_at_k(relevant_dict, retrieved_list, 5))
        rr_at_3.append(reciprocal_rank_at_k(relevant_dict, retrieved_list, 3))
        rr_at_5.append(reciprocal_rank_at_k(relevant_dict, retrieved_list, 5))
        ndcg_at_3.append(normalized_discounted_cumulative_gaint_at_k(relevant_dict, retrieved_list, 3))
        ndcg_at_5.append(normalized_discounted_cumulative_gaint_at_k(relevant_dict, retrieved_list, 5))

    print("=" * 100)
    print(f"Evaluation suit: {args.eval_file_path}")
    print(f"Embedding model: {args.embedder}")
    print("=" * 100)
    print(f"Mean Recall@3: {sum(recall_at_3)/len(recall_at_3):.2f}")
    print(f"Mean Recall@5: {sum(recall_at_5)/len(recall_at_5):.2f}")
    print(f"MRR@3:         {sum(rr_at_3)/len(rr_at_3):.2f}")
    print(f"MRR@5:         {sum(rr_at_5)/len(rr_at_5):.2f}")
    print(f"Mean nDCG@3:   {sum(ndcg_at_3)/len(ndcg_at_3):.2f}")
    print(f"Mean nDCG@5:   {sum(ndcg_at_5)/len(ndcg_at_5):.2f}")
    print("=" * 100)


if __name__ == "__main__":
    main()