"""Check if PDF data is loaded in vector store"""
import pickle
import os
import faiss

index_path = "data/faiss_index"
index_file = os.path.join(index_path, "index.faiss")
data_file = os.path.join(index_path, "data.pkl")

print("=" * 60)
print("Checking PDF Data in Vector Store")
print("=" * 60)

# Check if files exist
if not os.path.exists(index_file):
    print(f"[ERROR] Index file not found: {index_file}")
    print("The PDF has not been processed yet.")
    print("\nTo process the PDF, run:")
    print(f"  python process_pdf.py data/Vasanthkumar_Ramamoorthi_Resume.pdf")
    exit(1)

if not os.path.exists(data_file):
    print(f"[ERROR] Data file not found: {data_file}")
    print("The PDF has not been processed yet.")
    exit(1)

# Load the index
try:
    index = faiss.read_index(index_file)
    print(f"[OK] Index file loaded")
    print(f"Total vectors in index: {index.ntotal}")
except Exception as e:
    print(f"[ERROR] Failed to load index: {e}")
    exit(1)

# Load the data
try:
    with open(data_file, "rb") as f:
        data = pickle.load(f)
    
    texts = data.get("texts", [])
    metadata = data.get("metadata", [])
    
    print(f"[OK] Data file loaded")
    print(f"Total text chunks: {len(texts)}")
    print(f"Total metadata entries: {len(metadata)}")
    
    # Check if data matches
    if index.ntotal != len(texts):
        print(f"[WARNING] Mismatch: Index has {index.ntotal} vectors but {len(texts)} text chunks")
    else:
        print(f"[OK] Data consistency check passed")
    
    # Show sample data
    if texts:
        print("\n" + "=" * 60)
        print("Sample Data:")
        print("=" * 60)
        print(f"First chunk (first 200 chars):")
        print(f"  {texts[0][:200]}...")
        print(f"\nMetadata for first chunk:")
        print(f"  {metadata[0]}")
        
        # Check for PDF filename
        pdf_found = False
        for meta in metadata:
            if 'Vasanthkumar' in str(meta.get('filename', '')):
                pdf_found = True
                break
        
        if pdf_found:
            print(f"\n[OK] PDF filename found in metadata")
        else:
            print(f"\n[WARNING] PDF filename not found in metadata")
    
    if index.ntotal == 0:
        print("\n[ERROR] Vector store is empty!")
        print("You need to process the PDF first:")
        print(f"  python process_pdf.py data/Vasanthkumar_Ramamoorthi_Resume.pdf")
    else:
        print(f"\n[OK] Vector store contains {index.ntotal} documents")
        print("PDF data appears to be loaded correctly!")
        
except Exception as e:
    print(f"[ERROR] Failed to load data: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

print("=" * 60)







