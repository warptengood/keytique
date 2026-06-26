import torch
from dotenv import load_dotenv
from transformers import AutoTokenizer, AutoModel, AutoConfig, BatchEncoding
from transformers.modeling_outputs import BaseModelOutput


load_dotenv()


def mean_pooling(last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    last_hidden = last_hidden_state.masked_fill(~attention_mask[..., None].bool(), 0.0)
    denom = attention_mask.sum(dim=1, keepdim=True).clamp(min=1e-9)
    return last_hidden.sum(dim=1) / denom


def cls_token(last_hidden_state: torch.Tensor) -> torch.Tensor:
    return last_hidden_state[:, 0]


class BasicEmbeddingModel:
    def __init__(self, model_id: str) -> None:
        config = AutoConfig.from_pretrained(model_id)

        self.dim = config.hidden_size
        self.model_id = model_id

        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModel.from_pretrained(model_id)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.model.to(self.device)
        self.model.eval()

    def encode_query(self, query: str) -> list[float]:
        query_embeddings = self._encode(query)
        return query_embeddings.squeeze(0).cpu().tolist()

    def encode_documents(self, documents: list[str], batch_size: int = 64) -> list[list[float]]:
        embeddings: list[list[float]] = []
        for start in range(0, len(documents), batch_size):
            batch = documents[start : start + batch_size]
            embeddings.extend(self._encode(batch).cpu().tolist())
        return embeddings

    def _encode(self, text: str | list[str]) -> torch.Tensor:
        encoded_input = self.tokenizer(text, padding=True, truncation=True, return_tensors="pt")
        encoded_input.to(self.device)

        with torch.no_grad():
            model_output = self.model(**encoded_input)

        sentence_embeddings = self._pooling(model_output, encoded_input)
        sentence_embeddings = self._norm(sentence_embeddings)
        return sentence_embeddings

    def _pooling(self, model_output: BaseModelOutput, encoded_input: BatchEncoding) -> torch.Tensor:
        return mean_pooling(model_output.last_hidden_state, encoded_input["attention_mask"])

    def _norm(self, sentence_embeddings: torch.Tensor) -> torch.Tensor:
        # L2-normalization on 1st dimension
        return torch.nn.functional.normalize(sentence_embeddings, p=2, dim=1)


class MiniLM(BasicEmbeddingModel):
    def __init__(self, model_id: str = "sentence-transformers/all-MiniLM-L6-v2") -> None:
        super().__init__(model_id)


class BGE(BasicEmbeddingModel):
    def __init__(self, model_id: str = "BAAI/bge-small-en-v1.5") -> None:
        super().__init__(model_id)

    def encode_query(self, query: str) -> list[float]:
        query = "Represent this sentence for searching relevant passages: " + query
        return super().encode_query(query)

    def _pooling(self, model_output: BaseModelOutput, _: BatchEncoding) -> torch.Tensor:
        return cls_token(model_output.last_hidden_state)


class E5(BasicEmbeddingModel):
    def __init__(self, model_id: str = "intfloat/e5-small-v2") -> None:
        super().__init__(model_id)

    def encode_query(self, query: str) -> list[float]:
        query = "query: " + query
        return super().encode_query(query)

    def encode_documents(self, documents: list[str], batch_size: int = 64) -> list[list[float]]:
        documents = ["passage: " + doc for doc in documents]
        return super().encode_documents(documents, batch_size)


EMBEDDING_MODELS = {
    "all-MiniLM-L6-v2": MiniLM,
    "bge-small-en-v1.5": BGE,
    "e5-small-v2": E5,
}