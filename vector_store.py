import re
import faiss
import numpy as np
import pickle
import os
from typing import List, Dict, Any, Set
from embedding_service import EmbeddingService


def _tokenize_for_search(text: str) -> Set[str]:
    if not text:
        return set()
    words = re.findall(r"[a-z0-9]{2,}", text.lower())
    return set(words)


class VectorStore:

    def __init__(self, embedding_service: EmbeddingService, index_path: str = "data/faiss_index"):

        self.embedding_service = embedding_service
        self.index_path = index_path
        self.dimension = embedding_service.embedding_dimension

        os.makedirs(index_path, exist_ok=True)

        self.index = faiss.IndexFlatL2(self.dimension)

        self.texts = []
        self.metadata = []

        self.load()

    def add_documents(self, texts: List[str], metadata: List[Dict[str, Any]] = None):

        if not texts:
            return

        print(f"Generating embeddings for {len(texts)} chunks...")
        embeddings = self.embedding_service.embed_batch(texts)

        faiss.normalize_L2(embeddings)

        self.index.add(embeddings)

        self.texts.extend(texts)

        if metadata:
            self.metadata.extend(metadata)
        else:
            self.metadata.extend([{}] * len(texts))

        print(f"Added {len(texts)} documents to vector store. Total: {self.index.ntotal}")

    def search(self, query: str, top_k: int = 5, min_similarity: float = 0.3) -> List[Dict[str, Any]]:
        if self.index.ntotal == 0:
            return []

        query_embedding = self.embedding_service.embed_text(query)
        query_embedding = query_embedding.reshape(1, -1)
        faiss.normalize_L2(query_embedding)

        search_k = min(top_k * 3, self.index.ntotal)
        distances, indices = self.index.search(query_embedding, search_k)

        results = []
        for i, (distance, idx) in enumerate(zip(distances[0], indices[0])):
            if idx >= len(self.texts):
                continue

            similarity = max(0.0, 1.0 - (distance ** 2) / 2.0)

            if similarity < min_similarity:
                continue

            results.append({
                "text": self.texts[idx],
                "metadata": self.metadata[idx],
                "score": float(similarity),
                "distance": float(distance),
                "rank": len(results) + 1
            })

            if len(results) >= top_k:
                break

        results.sort(key=lambda x: x["score"], reverse=True)

        return results

    def search_keyword(self, query: str, top_k: int = 10) -> List[Dict[str, Any]]:
        if self.index.ntotal == 0:
            return []
        q_tokens = _tokenize_for_search(query)
        if not q_tokens:
            return []
        scored = []
        for idx, text in enumerate(self.texts):
            chunk_tokens = _tokenize_for_search(text)
            overlap = len(q_tokens & chunk_tokens) / len(q_tokens)
            if overlap > 0:
                scored.append({
                    "text": text,
                    "metadata": self.metadata[idx],
                    "score": float(overlap),
                    "rank": 0,
                })
        scored.sort(key=lambda x: x["score"], reverse=True)
        for i, r in enumerate(scored[:top_k], 1):
            r["rank"] = i
        return scored[:top_k]

    def search_hybrid(
        self,
        query: str,
        top_k: int = 5,
        min_similarity: float = 0.12,
        vector_weight: float = 0.55,
        keyword_weight: float = 0.45,
    ) -> List[Dict[str, Any]]:
        if self.index.ntotal == 0:
            return []
        vector_results = self.search(query, top_k=min(top_k * 4, self.index.ntotal), min_similarity=0.0)
        keyword_results = self.search_keyword(query, top_k=top_k * 3)
        by_key = {}
        for r in vector_results:
            key = r["text"]
            by_key[key] = {"vector_score": r["score"], "keyword_score": 0.0, "r": r}
        for r in keyword_results:
            key = r["text"]
            if key not in by_key:
                by_key[key] = {"vector_score": 0.0, "keyword_score": r["score"], "r": r}
            else:
                by_key[key]["keyword_score"] = r["score"]
        combined = []
        for v in by_key.values():
            vec = v["vector_score"]
            kw = v["keyword_score"]
            combined_score = vector_weight * vec + keyword_weight * kw
            r = v["r"].copy()
            r["score"] = round(combined_score, 4)
            r["vector_score"] = vec
            r["keyword_score"] = kw
            combined.append(r)
        combined.sort(key=lambda x: x["score"], reverse=True)
        for i, r in enumerate(combined[:top_k], 1):
            r["rank"] = i
        return combined[:top_k]

    def save(self):
        try:
            if self.index.ntotal == 0:
                print(f"[WARNING] Attempting to save empty index (0 vectors). Skipping save.")
                return

            if len(self.texts) == 0:
                print(f"[WARNING] Attempting to save empty texts list. Skipping save.")
                return

            if len(self.texts) != self.index.ntotal:
                print(f"[WARNING] Mismatch: {len(self.texts)} texts but {self.index.ntotal} vectors. Saving anyway...")

            index_file = os.path.join(self.index_path, "index.faiss")
            data_file = os.path.join(self.index_path, "data.pkl")

            faiss.write_index(self.index, index_file)

            with open(data_file, "wb") as f:
                pickle.dump({
                    "texts": self.texts,
                    "metadata": self.metadata,
                    "dimension": self.dimension
                }, f)

            print(f"Vector store saved to {self.index_path} ({self.index.ntotal} vectors, {len(self.texts)} texts)")
        except Exception as e:
            print(f"Error saving vector store: {str(e)}")
            import traceback
            traceback.print_exc()

    def load(self):
        try:
            index_file = os.path.join(self.index_path, "index.faiss")
            data_file = os.path.join(self.index_path, "data.pkl")

            if os.path.exists(index_file) and os.path.exists(data_file):
                loaded_index = faiss.read_index(index_file)

                with open(data_file, "rb") as f:
                    data = pickle.load(f)
                    loaded_texts = data.get("texts", [])
                    loaded_metadata = data.get("metadata", [])

                if loaded_index.ntotal == 0 or len(loaded_texts) == 0:
                    print(f"[WARNING] Vector store files exist but are empty. Starting fresh.")
                    return False

                if loaded_index.ntotal == len(loaded_texts):
                    self.index = loaded_index
                    self.texts = loaded_texts
                    self.metadata = loaded_metadata
                    print(f"Vector store loaded from {self.index_path}. Documents: {self.index.ntotal}")
                    return True
                else:
                    print(f"[WARNING] Mismatch: {loaded_index.ntotal} vectors but {len(loaded_texts)} texts. Starting fresh.")
                    return False
            else:
                print("No existing vector store found. Starting fresh.")
                return False
        except Exception as e:
            print(f"Error loading vector store: {str(e)}")
            import traceback
            traceback.print_exc()
            return False

    def get_stats(self) -> Dict[str, Any]:
        return {
            "total_documents": self.index.ntotal,
            "embedding_dimension": self.dimension,
            "index_path": self.index_path
        }

    def clear(self):
        self.index = faiss.IndexFlatL2(self.dimension)
        self.texts = []
        self.metadata = []

        try:
            index_file = os.path.join(self.index_path, "index.faiss")
            data_file = os.path.join(self.index_path, "data.pkl")
            if os.path.exists(index_file):
                os.remove(index_file)
            if os.path.exists(data_file):
                os.remove(data_file)
        except Exception as e:
            print(f"Error clearing vector store files: {str(e)}")

        print("Vector store cleared")
