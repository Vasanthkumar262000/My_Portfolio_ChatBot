"""
Document Processor - Extracts text from PDF and DOCX files and creates chunks for RAG
"""
import os
from typing import List
from PyPDF2 import PdfReader

try:
    from docx import Document
    DOCX_AVAILABLE = True
except ImportError:
    DOCX_AVAILABLE = False


def extract_text_from_pdf(pdf_path: str) -> str:
    """
    Extract all text from a PDF file.
    
    Args:
        pdf_path: Path to the PDF file
        
    Returns:
        Extracted text as a string
    """
    try:
        reader = PdfReader(pdf_path)
        text = ""
        for page in reader.pages:
            text += page.extract_text() + "\n"
        return text
    except Exception as e:
        raise Exception(f"Error reading PDF: {str(e)}")


def chunk_text(text: str, chunk_size: int = 1000, chunk_overlap: int = 200) -> List[str]:
    """
    Split text into chunks with overlap for better context retention.
    
    Args:
        text: The text to chunk
        chunk_size: Maximum size of each chunk (in characters)
        chunk_overlap: Number of characters to overlap between chunks
        
    Returns:
        List of text chunks
    """
    if not text:
        return []
    
    chunks = []
    start = 0
    text_length = len(text)
    
    while start < text_length:
        end = start + chunk_size
        chunk = text[start:end]
        
        # Try to break at sentence boundary if not at end
        if end < text_length:
            # Look for sentence endings
            last_period = chunk.rfind('.')
            last_newline = chunk.rfind('\n')
            break_point = max(last_period, last_newline)
            
            if break_point > chunk_size * 0.5:  # Only break if we're past halfway
                chunk = chunk[:break_point + 1]
                end = start + break_point + 1
        
        chunks.append(chunk.strip())
        start = end - chunk_overlap  # Overlap for context
    
    return chunks


def extract_text_from_docx(docx_path: str) -> str:
    """
    Extract all text from a DOCX file.
    
    Args:
        docx_path: Path to the DOCX file
        
    Returns:
        Extracted text as a string
    """
    if not DOCX_AVAILABLE:
        raise ImportError("python-docx is not installed. Install it with: pip install python-docx")
    
    try:
        doc = Document(docx_path)
        text = ""
        for paragraph in doc.paragraphs:
            text += paragraph.text + "\n"
        return text
    except Exception as e:
        raise Exception(f"Error reading DOCX: {str(e)}")


def process_document(file_path: str, chunk_size: int = 1000, chunk_overlap: int = 200) -> List[str]:
    """
    Process a document (PDF or DOCX) and return chunks.
    
    Args:
        file_path: Path to the document file
        chunk_size: Maximum size of each chunk
        chunk_overlap: Overlap between chunks
        
    Returns:
        List of text chunks ready for embedding
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    
    # Determine file type
    file_ext = os.path.splitext(file_path)[1].lower()
    
    if file_ext == '.pdf':
        print(f"Extracting text from PDF: {file_path}")
        text = extract_text_from_pdf(file_path)
    elif file_ext in ['.docx', '.doc']:
        print(f"Extracting text from DOCX: {file_path}")
        text = extract_text_from_docx(file_path)
    else:
        raise ValueError(f"Unsupported file type: {file_ext}. Supported: .pdf, .docx")
    
    if not text.strip():
        raise ValueError(f"No text could be extracted from {file_path}")
    
    print(f"Text extracted: {len(text)} characters")
    
    print(f"Chunking text into pieces...")
    chunks = chunk_text(text, chunk_size, chunk_overlap)
    print(f"Created {len(chunks)} chunks")
    
    return chunks


def process_pdf(pdf_path: str, chunk_size: int = 1000, chunk_overlap: int = 200) -> List[str]:
    """
    Complete PDF processing pipeline: extract text and chunk it.
    (Kept for backward compatibility)
    
    Args:
        pdf_path: Path to the PDF file
        chunk_size: Maximum size of each chunk
        chunk_overlap: Overlap between chunks
        
    Returns:
        List of text chunks ready for embedding
    """
    return process_document(pdf_path, chunk_size, chunk_overlap)


