"""
Vector Store - Manages FAISS vector database for storing and searching embeddings
"""
import faiss
import numpy as np
import pickle
import os
from typing import List, Dict, Any
from embedding_service import EmbeddingService


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
    
    def search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:

        if self.index.ntotal == 0:
            return []

        query_embedding = self.embedding_service.embed_text(query)

        query_embedding = query_embedding.reshape(1, -1)

        faiss.normalize_L2(query_embedding)
        
        distances, indices = self.index.search(query_embedding, min(top_k, self.index.ntotal))

        results = []
        for i, (distance, idx) in enumerate(zip(distances[0], indices[0])):
            if idx < len(self.texts):

                similarity = 1 - (distance / 2)
                
                results.append({
                    "text": self.texts[idx],
                    "metadata": self.metadata[idx],
                    "score": float(similarity),
                    "rank": i + 1
                })
        
        return results
    
    def save(self):
        """Save the FAISS index and associated data to disk."""
        try:
            index_file = os.path.join(self.index_path, "index.faiss")
            faiss.write_index(self.index, index_file)
            
            data_file = os.path.join(self.index_path, "data.pkl")
            with open(data_file, "wb") as f:
                pickle.dump({
                    "texts": self.texts,
                    "metadata": self.metadata,
                    "dimension": self.dimension
                }, f)
            
            print(f"Vector store saved to {self.index_path}")
        except Exception as e:
            print(f"Error saving vector store: {str(e)}")
    
    def load(self):
        """Load the FAISS index and associated data from disk."""
        try:
            index_file = os.path.join(self.index_path, "index.faiss")
            data_file = os.path.join(self.index_path, "data.pkl")
            
            if os.path.exists(index_file) and os.path.exists(data_file):
                self.index = faiss.read_index(index_file)

                with open(data_file, "rb") as f:
                    data = pickle.load(f)
                    self.texts = data.get("texts", [])
                    self.metadata = data.get("metadata", [])
                
                print(f"Vector store loaded from {self.index_path}. Documents: {self.index.ntotal}")
                return True
            else:
                print("No existing vector store found. Starting fresh.")
                return False
        except Exception as e:
            print(f"Error loading vector store: {str(e)}")
            return False
    
    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about the vector store."""
        return {
            "total_documents": self.index.ntotal,
            "embedding_dimension": self.dimension,
            "index_path": self.index_path
        }
    
    def clear(self):
        """Clear all documents from the vector store."""

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

