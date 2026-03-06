import os
import sys
import json
from pathlib import Path

from embedding_service import EmbeddingService
from vector_store import VectorStore
from dotenv import load_dotenv

load_dotenv()

DEFAULT_KB_PATH = "data/vasanth_profile_kb.json"


def load_kb(kb_path: str) -> dict:
    path = Path(kb_path)
    if not path.exists():
        raise FileNotFoundError(f"Knowledge base not found: {kb_path}")
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if "documents" not in data or not isinstance(data["documents"], list):
        raise ValueError("Knowledge base must contain a 'documents' array")
    return data


def main():
    kb_path = DEFAULT_KB_PATH
    if len(sys.argv) > 1 and not sys.argv[1].startswith("--"):
        kb_path = sys.argv[1]
    for arg in sys.argv[1:]:
        if arg.startswith("--kb-path="):
            kb_path = arg.split("=", 1)[1]
            break

    print("=" * 60)
    print("Training RAG from Knowledge Base (JSON)")
    print("=" * 60)
    print(f"KB file: {kb_path}\n")

    try:
        data = load_kb(kb_path)
        documents = data["documents"]
        kb_meta = data.get("knowledge_base", {})
        print(f"Document: {kb_meta.get('document', 'N/A')}")
        print(f"Version: {kb_meta.get('version', 'N/A')}")
        print(f"Entries: {len(documents)}\n")

        if not documents:
            print("[ERROR] No documents in knowledge base.")
            sys.exit(1)

        texts = []
        metadata = []
        for doc in documents:
            q = doc.get("question", "")
            a = doc.get("answer", "")
            if not a and not q:
                continue
            text = f"Question: {q}\nAnswer: {a}".strip()
            texts.append(text)
            meta = {
                "doc_id": doc.get("doc_id", ""),
                "category": doc.get("category", ""),
                "keywords": ", ".join(doc.get("keywords", []))[:200],
            }
            metadata.append(meta)

        print("Initializing embedding service and vector store...")
        embedding_service = EmbeddingService()
        vector_store = VectorStore(embedding_service)
        initial_count = vector_store.index.ntotal
        print(f"Current documents in vector store: {initial_count}\n")

        print("[Reindex] Clearing existing vector store...")
        vector_store.clear()
        print("[Reindex] Cleared.\n")

        print(f"Generating embeddings for {len(texts)} chunks...")
        vector_store.add_documents(texts, metadata)

        print("Saving vector store...")
        vector_store.save()

        stats = vector_store.get_stats()
        print("\n" + "=" * 60)
        print("Training complete")
        print("=" * 60)
        print(f"Chunks indexed: {len(texts)}")
        print(f"Total vectors: {stats['total_documents']}")
        print("\nRestart the backend (or redeploy) to use the updated index.")
        print("=" * 60)

    except FileNotFoundError as e:
        print(f"[ERROR] {e}")
        sys.exit(1)
    except ValueError as e:
        print(f"[ERROR] {e}")
        sys.exit(1)
    except Exception as e:
        print(f"[ERROR] {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
