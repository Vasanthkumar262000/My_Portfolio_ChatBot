
from sentence_transformers import SentenceTransformer
import numpy as np
from typing import List


class EmbeddingService:
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):

        print(f"Loading embedding model: {model_name}...")
        self.model = SentenceTransformer(model_name)

        self.model_name = model_name

        self.embedding_dimension = self.model.get_sentence_embedding_dimension()
        print(f"Embedding model loaded. Dimension: {self.embedding_dimension}")
    
    def embed_text(self, text: str) -> np.ndarray:

        embedding = self.model.encode(text, convert_to_numpy=True)
        return embedding
    
    def embed_batch(self, texts: List[str]) -> np.ndarray:

        embeddings = self.model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
        return embeddings

