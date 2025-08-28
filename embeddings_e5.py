from typing import List, Optional
from langchain_core.embeddings import Embeddings
from sentence_transformers import SentenceTransformer

class E5Embeddings(Embeddings):
    def __init__(self, model_name: str = "intfloat/e5-base-v2", normalize: bool = True, device: Optional[str] = None):
        # Default to CPU if no device is passed
        if device is None:
            device = "cpu"
        self.model = SentenceTransformer(model_name, device="cpu")
        self.normalize = normalize

    @property
    def dimension(self) -> int:
        return self.model.get_sentence_embedding_dimension()

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        # Prefix with "passage:" as required by E5 models
        texts = [f"passage: {t}" for t in texts]
        embs = self.model.encode(
            texts, 
            normalize_embeddings=self.normalize, 
            show_progress_bar=False
        )
        return [e.tolist() for e in embs]

    def embed_query(self, text: str) -> List[float]:
        # Prefix with "query:" for queries
        emb = self.model.encode(
            [f"query: {text}"], 
            normalize_embeddings=self.normalize, 
            show_progress_bar=False
        )
        return emb[0].tolist()
