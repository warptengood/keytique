from qdrant_client import QdrantClient, models

from .config import COLLECTION_NAME
from .schemas import SkillArea, Payload
from .embedding_model import BasicEmbeddingModel


class Retriever:
    def __init__(self, client: QdrantClient, embedder: BasicEmbeddingModel) -> None:
        self._client = client
        self._embedder = embedder

    def retrieve(self, query: str, skill_area: SkillArea | None = None, top_k: int = 5) -> list[Payload]:
        search_result = self._client.query_points(
            collection_name=COLLECTION_NAME,
            query=self._embedder.encode_query(query),
            query_filter=None
            if skill_area is None
            else models.Filter(must=[models.FieldCondition(key="frontmatter.skill_area", match=models.MatchValue(value=skill_area))]),
            with_payload=True,
            limit=top_k,
        ).points

        return [Payload(**r.payload) for r in search_result]
