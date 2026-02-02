"""
Script to add a new document (PDF or DOCX) to the vector database.
Usage: python add_document.py <path_to_file> [--reindex]
  --reindex: Clear existing index and re-add only this document (use after fixing RAG pipeline).
"""
import os
import sys
from pdf_processor import process_document
from embedding_service import EmbeddingService
from vector_store import VectorStore
from dotenv import load_dotenv

load_dotenv()

# Chunk sizes tuned for better RAG retrieval (smaller chunks, more overlap)
CHUNK_SIZE = 700
CHUNK_OVERLAP = 180


def main():
    """Add a document to the vector store."""
    args = [a for a in sys.argv[1:] if a != "--reindex"]
    reindex = "--reindex" in sys.argv
    
    if len(args) < 1:
        print("Usage: python add_document.py <path_to_file> [--reindex]")
        print("\nExample:")
        print("  python add_document.py data/new_resume.pdf")
        print("  python add_document.py data/Vasanth information doc.pdf --reindex")
        sys.exit(1)
    
    file_path = args[0]
    
    if not os.path.exists(file_path):
        print(f"Error: File not found: {file_path}")
        sys.exit(1)
    
    file_ext = os.path.splitext(file_path)[1].lower()
    if file_ext not in ['.pdf', '.docx', '.doc']:
        print(f"Error: Unsupported file type: {file_ext}")
        print("Supported: .pdf, .docx, .doc")
        sys.exit(1)
    
    # Suggest DOCX when it exists (often cleaner than PDF extraction)
    base = os.path.splitext(file_path)[0]
    if file_ext == ".pdf" and os.path.isfile(base + ".docx"):
        print(f"[Tip] {base}.docx exists. DOCX often gives cleaner text for RAG; consider using it for --reindex.")
    
    print("=" * 60)
    print("Adding Document to Vector Database")
    print("=" * 60)
    print(f"File: {file_path}")
    if reindex:
        print("Mode: REINDEX (clearing existing index first)")
    print()
    
    try:
        # Initialize services
        print("Initializing services...")
        embedding_service = EmbeddingService()
        vector_store = VectorStore(embedding_service)
        initial_count = vector_store.index.ntotal
        print(f"Current documents in vector store: {initial_count}\n")
        
        if reindex:
            print("[Reindex] Clearing existing vector store...")
            vector_store.clear()
            print("[Reindex] Cleared.\n")
        
        # Process document with improved chunking (smaller chunks for better retrieval)
        print(f"Processing {file_ext.upper()} file (chunk_size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP})...")
        chunks = process_document(file_path, chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
        
        if not chunks:
            print("[ERROR] No text could be extracted from the file.")
            print("Possible reasons:")
            print("  - File is image-based (scanned PDF)")
            print("  - File is corrupted")
            print("  - File is password protected")
            sys.exit(1)
        
        print(f"[OK] Extracted {len(chunks)} chunks\n")
        
        # Create rich metadata for better retrieval
        filename = os.path.basename(file_path)
        file_size = os.path.getsize(file_path)
        
        metadata = []
        for i, chunk in enumerate(chunks):
            # Extract key terms from chunk for better searchability
            words = chunk.lower().split()[:10]  # First 10 words as keywords
            keywords = ' '.join(words)
            
            metadata.append({
                "filename": filename,
                "chunk_id": i,
                "chunk_index": i,
                "total_chunks": len(chunks),
                "file_size": file_size,
                "keywords": keywords[:100]  # Limit keyword length
            })
        
        # Add to vector store
        print("[Processing] Generating embeddings (this may take a moment)...")
        vector_store.add_documents(chunks, metadata)
        
        # Save vector store
        print("[Saving] Writing vector store to disk...")
        vector_store.save()
        
        # Show statistics
        stats = vector_store.get_stats()
        print("\n" + "=" * 60)
        print("[OK] Document Added Successfully!")
        print("=" * 60)
        print(f"Chunks added: {len(chunks)}")
        print(f"Total documents in vector store: {stats['total_documents']}")
        print(f"New documents: {stats['total_documents'] - initial_count}")
        if reindex:
            print("\nReindex complete. Restart your backend to use the updated index.")
        else:
            print("\nRestart your backend to load the new documents!")
        print("=" * 60)
    
    except Exception as e:
        print(f"\n[ERROR] Error: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

