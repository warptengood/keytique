import re
from pathlib import Path

import yaml
from qdrant_client import QdrantClient, models

from .schemas import Payload, FrontMatter
from .embedding_model import BasicEmbeddingModel
from .config import COLLECTION_NAME, LOCAL_CORPUS_PATH


class Indexer:
    def __init__(self, client: QdrantClient, embedder: BasicEmbeddingModel) -> None:
        self._client = client
        self._embedder = embedder

    def delete_collection(self) -> None:
        if not self.collection_exists():
            return
        self._client.delete_collection(COLLECTION_NAME)

    def collection_exists(self) -> bool:
        return self._client.collection_exists(COLLECTION_NAME)

    def build_index(self) -> None:
        if self.collection_exists():
            return

        self._client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=models.VectorParams(size=self._embedder.dim, distance=models.Distance.COSINE),
        )

        self.upload_document(list(Path(LOCAL_CORPUS_PATH).rglob("*.md")))

    def upload_document(self, document: Path | list[Path]) -> None:
        if isinstance(document, Path):
            document = [document]

        payloads: list[Payload] = []
        for doc in document:
            payloads.extend(self._extract_chunks(doc))

        vectors = self._embedder.encode_documents([payload.text for payload in payloads])

        self._client.upload_collection(
            collection_name=COLLECTION_NAME,
            vectors=vectors,
            payload=[p.model_dump() for p in payloads],
            parallel=4,
            max_retries=3,
        )

    def _extract_chunks(self, document: Path) -> list[Payload]:
        # TODO: split chunks that exceeds model's max length
        content = document.read_text(encoding="utf-8").strip()

        if not content.startswith("---"):
            raise ValueError(f"Missing frontmatter in {document}")

        _, frontmatter_raw, body = content.split("---", maxsplit=2)

        frontmatter = FrontMatter(**yaml.safe_load(frontmatter_raw))

        sections = re.split(r"^##\s+", body.strip(), flags=re.MULTILINE)

        payloads = []
        for index, section in enumerate(sections[1:]):
            lines = section.strip().splitlines()

            if not lines:
                continue

            title = lines[0].strip()
            text = "\n".join(lines[1:]).strip()

            if text:
                payloads.append(
                    Payload(
                        frontmatter=frontmatter,
                        text=text,
                        section_title=title,
                        doc_id=document.stem,
                        chunk_index=index,
                    )
                )
        return payloads
