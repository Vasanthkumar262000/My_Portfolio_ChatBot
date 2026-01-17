"""
Script to add a new document (PDF or DOCX) to the vector database.
Usage: python add_document.py <path_to_file>
"""
import os
import sys
from pdf_processor import process_document
from embedding_service import EmbeddingService
from vector_store import VectorStore
from dotenv import load_dotenv

load_dotenv()


def main():
    """Add a document to the vector store."""
    
    if len(sys.argv) < 2:
        print("Usage: python add_document.py <path_to_file>")
        print("\nExample:")
        print("  python add_document.py data/new_resume.pdf")
        print("  python add_document.py data/new_resume.docx")
        sys.exit(1)
    
    file_path = sys.argv[1]
    
    if not os.path.exists(file_path):
        print(f"Error: File not found: {file_path}")
        sys.exit(1)
    
    file_ext = os.path.splitext(file_path)[1].lower()
    if file_ext not in ['.pdf', '.docx', '.doc']:
        print(f"Error: Unsupported file type: {file_ext}")
        print("Supported: .pdf, .docx, .doc")
        sys.exit(1)
    
    print("=" * 60)
    print("Adding Document to Vector Database")
    print("=" * 60)
    print(f"File: {file_path}\n")
    
    try:
        # Initialize services
        print("Initializing services...")
        embedding_service = EmbeddingService()
        vector_store = VectorStore(embedding_service)
        initial_count = vector_store.index.ntotal
        print(f"Current documents in vector store: {initial_count}\n")
        
        # Process document
        print(f"Processing {file_ext.upper()} file...")
        chunks = process_document(file_path, chunk_size=1000, chunk_overlap=200)
        
        if not chunks:
            print("Error: No text could be extracted from the file.")
            sys.exit(1)
        
        print(f"[OK] Extracted {len(chunks)} chunks\n")
        
        # Create metadata
        filename = os.path.basename(file_path)
        metadata = [{"filename": filename, "chunk_id": i} for i in range(len(chunks))]
        
        # Add to vector store
        print("Generating embeddings and adding to vector store...")
        vector_store.add_documents(chunks, metadata)
        
        # Save vector store
        print("Saving vector store to disk...")
        vector_store.save()
        
        # Show statistics
        stats = vector_store.get_stats()
        print("\n" + "=" * 60)
        print("[OK] Document Added Successfully!")
        print("=" * 60)
        print(f"Chunks added: {len(chunks)}")
        print(f"Total documents in vector store: {stats['total_documents']}")
        print(f"New documents: {stats['total_documents'] - initial_count}")
        print("\nRestart your backend to load the new documents!")
        print("=" * 60)
    
    except Exception as e:
        print(f"\n[ERROR] Error: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

