import os
import sys
from pdf_processor import process_pdf
from embedding_service import EmbeddingService
from vector_store import VectorStore
from dotenv import load_dotenv

load_dotenv()


def main():
    if len(sys.argv) < 2:
        print("Usage: python process_pdf.py <path_to_pdf>")
        print("\nExample:")
        print("  python process_pdf.py data/your_document.pdf")
        sys.exit(1)

    pdf_path = sys.argv[1]

    if not os.path.exists(pdf_path):
        print(f"Error: PDF file not found: {pdf_path}")
        sys.exit(1)

    print("=" * 60)
    print("PDF Processing Script")
    print("=" * 60)
    print(f"Processing: {pdf_path}\n")

    try:
        print("Step 1: Initializing embedding service...")
        embedding_service = EmbeddingService()

        print("Step 2: Initializing vector store...")
        vector_store = VectorStore(embedding_service)

        print(f"\nStep 3: Processing PDF...")
        chunks = process_pdf(pdf_path, chunk_size=1000, chunk_overlap=200)

        if not chunks:
            print("Error: No text could be extracted from PDF.")
            print("Make sure the PDF contains text (not just images).")
            sys.exit(1)

        print(f"[OK] Extracted {len(chunks)} chunks from PDF\n")

        filename = os.path.basename(pdf_path)
        metadata = [{"filename": filename, "chunk_id": i} for i in range(len(chunks))]

        print("Step 4: Generating embeddings and adding to vector store...")
        vector_store.add_documents(chunks, metadata)

        print("\nStep 5: Saving vector store to disk...")
        vector_store.save()

        stats = vector_store.get_stats()
        print("\n" + "=" * 60)
        print("[OK] Processing Complete!")
        print("=" * 60)
        print(f"Total documents in vector store: {stats['total_documents']}")
        print(f"Embedding dimension: {stats['embedding_dimension']}")
        print(f"Vector store location: {stats['index_path']}")
        print("\nYou can now start the API server with:")
        print("  uvicorn main:app --reload --host 0.0.0.0 --port 8000")
        print("=" * 60)

    except Exception as e:
        print(f"\n[ERROR] Error: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
