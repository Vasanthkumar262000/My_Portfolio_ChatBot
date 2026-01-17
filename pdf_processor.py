"""
Document Processor - Extracts text from PDF and DOCX files and creates chunks for RAG
Professional-grade extraction with multiple fallback strategies
"""
import os
import re
from typing import List
from PyPDF2 import PdfReader

try:
    from docx import Document
    DOCX_AVAILABLE = True
except ImportError:
    DOCX_AVAILABLE = False


def extract_text_from_pdf(pdf_path: str) -> str:
    """
    Extract all text from a PDF file with professional-grade extraction.
    Uses multiple strategies and intelligent text cleaning.
    
    Args:
        pdf_path: Path to the PDF file
        
    Returns:
        Extracted text as a string
    """
    try:
        reader = PdfReader(pdf_path)
        total_pages = len(reader.pages)
        text_parts = []
        
        for page_num, page in enumerate(reader.pages, 1):
            page_text = ""
            
            # Strategy 1: Standard text extraction
            try:
                page_text = page.extract_text()
            except:
                pass
            
            # Strategy 2: If standard extraction yields little, try alternative
            if len(page_text.strip()) < 50:
                try:
                    # Try with different extraction modes
                    if hasattr(page, 'extract_text'):
                        page_text = page.extract_text(extraction_mode="layout")
                except:
                    pass
            
            # Strategy 3: Extract from form fields if available
            if len(page_text.strip()) < 50 and hasattr(page, 'get_form_text_fields'):
                try:
                    form_fields = page.get_form_text_fields()
                    if form_fields:
                        page_text = ' '.join([str(v) for v in form_fields.values() if v])
                except:
                    pass
            
            if page_text:
                # Advanced text cleaning
                page_text = _clean_extracted_text(page_text)
                if page_text.strip():
                    text_parts.append(page_text)
        
        # Combine all pages with proper spacing
        full_text = "\n\n".join(text_parts)
        
        # Final normalization
        full_text = _normalize_text(full_text)
        
        if not full_text.strip():
            raise ValueError(f"No text could be extracted from PDF: {pdf_path}")
        
        return full_text.strip()
    
    except Exception as e:
        raise Exception(f"Error reading PDF '{pdf_path}': {str(e)}")


def _clean_extracted_text(text: str) -> str:
    """Clean extracted text while preserving structure"""
    if not text:
        return ""
    
    # Remove control characters except newlines and tabs
    text = re.sub(r'[\x00-\x08\x0b-\x0c\x0e-\x1f\x7f-\x9f]', '', text)
    
    # Normalize whitespace but preserve line breaks
    lines = text.split('\n')
    cleaned_lines = []
    
    for line in lines:
        # Remove excessive spaces but keep single spaces
        line = re.sub(r' +', ' ', line.strip())
        
        # Remove lines that are just punctuation or numbers
        if line and not re.match(r'^[\s\W\d]+$', line):
            cleaned_lines.append(line)
    
    return '\n'.join(cleaned_lines)


def _normalize_text(text: str) -> str:
    """Final text normalization"""
    if not text:
        return ""
    
    # Remove excessive blank lines (more than 2 consecutive)
    text = re.sub(r'\n{3,}', '\n\n', text)
    
    # Fix common PDF extraction issues
    # Fix broken words (space in middle of word)
    text = re.sub(r'(\w) (\w)', r'\1\2', text)  # But be careful with this
    
    # Normalize unicode quotes and dashes
    text = text.replace('\u201c', '"').replace('\u201d', '"')
    text = text.replace('\u2018', "'").replace('\u2019', "'")
    text = text.replace('\u2013', '-').replace('\u2014', '--')
    
    return text.strip()


def chunk_text(text: str, chunk_size: int = 1500, chunk_overlap: int = 400) -> List[str]:
    """
    Professional-grade semantic chunking with intelligent boundary detection.
    Preserves context and semantic meaning across chunks.
    
    Args:
        text: The text to chunk
        chunk_size: Maximum size of each chunk (in characters)
        chunk_overlap: Number of characters to overlap between chunks
        
    Returns:
        List of text chunks optimized for RAG
    """
    if not text or not text.strip():
        return []
    
    # Optimized chunking parameters for better retrieval
    chunk_size = max(chunk_size, 1500)  # Larger chunks for better context
    chunk_overlap = max(chunk_overlap, 400)  # More overlap for continuity
    
    # Split into paragraphs first (semantic units)
    paragraphs = _split_into_paragraphs(text)
    
    chunks = []
    current_chunk = []
    current_size = 0
    
    for para in paragraphs:
        para_size = len(para)
        
        # If adding this paragraph would exceed chunk size
        if current_size + para_size > chunk_size and current_chunk:
            # Save current chunk
            chunk_text = '\n\n'.join(current_chunk)
            if chunk_text.strip():
                chunks.append(chunk_text.strip())
            
            # Start new chunk with overlap from previous
            overlap_text = _get_overlap_text(current_chunk, chunk_overlap)
            current_chunk = [overlap_text] if overlap_text else []
            current_size = len(overlap_text)
        
        # If paragraph itself is larger than chunk_size, split it
        if para_size > chunk_size:
            # Split large paragraph at sentence boundaries
            sentences = _split_into_sentences(para)
            for sentence in sentences:
                sent_size = len(sentence)
                if current_size + sent_size > chunk_size and current_chunk:
                    chunk_text = '\n\n'.join(current_chunk)
                    if chunk_text.strip():
                        chunks.append(chunk_text.strip())
                    current_chunk = []
                    current_size = 0
                
                current_chunk.append(sentence)
                current_size += sent_size + 2  # +2 for '\n\n'
        else:
            current_chunk.append(para)
            current_size += para_size + 2
    
    # Add final chunk
    if current_chunk:
        chunk_text = '\n\n'.join(current_chunk)
        if chunk_text.strip():
            chunks.append(chunk_text.strip())
    
    # Filter out very small chunks (likely artifacts)
    chunks = [chunk for chunk in chunks if len(chunk.strip()) >= 100]
    
    return chunks


def _split_into_paragraphs(text: str) -> List[str]:
    """Split text into paragraphs"""
    # Split on double newlines (paragraph breaks)
    paragraphs = re.split(r'\n\s*\n', text)
    # Clean and filter
    paragraphs = [p.strip() for p in paragraphs if p.strip()]
    return paragraphs


def _split_into_sentences(text: str) -> List[str]:
    """Split text into sentences"""
    # Simple sentence splitting (can be improved with NLTK/spaCy)
    sentences = re.split(r'(?<=[.!?])\s+', text)
    sentences = [s.strip() for s in sentences if s.strip()]
    return sentences


def _get_overlap_text(paragraphs: List[str], overlap_size: int) -> str:
    """Get overlap text from end of paragraphs"""
    if not paragraphs:
        return ""
    
    overlap_text = ""
    for para in reversed(paragraphs):
        if len(overlap_text) + len(para) <= overlap_size:
            overlap_text = para + '\n\n' + overlap_text
        else:
            # Take last part of paragraph
            remaining = overlap_size - len(overlap_text)
            if remaining > 0:
                overlap_text = para[-remaining:] + '\n\n' + overlap_text
            break
    
    return overlap_text.strip()


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


def process_document(file_path: str, chunk_size: int = 1500, chunk_overlap: int = 400) -> List[str]:
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
        print(f"[PDF] Extracting text from: {file_path}")
        text = extract_text_from_pdf(file_path)
    elif file_ext in ['.docx', '.doc']:
        print(f"[DOCX] Extracting text from: {file_path}")
        text = extract_text_from_docx(file_path)
    else:
        raise ValueError(f"Unsupported file type: {file_ext}. Supported: .pdf, .docx")
    
    if not text or not text.strip():
        raise ValueError(f"No text could be extracted from {file_path}. File may be image-based or corrupted.")
    
    char_count = len(text)
    word_count = len(text.split())
    print(f"[OK] Text extracted: {char_count:,} characters, {word_count:,} words")
    
    print(f"[Processing] Chunking text with size={chunk_size}, overlap={chunk_overlap}...")
    chunks = chunk_text(text, chunk_size, chunk_overlap)
    
    if not chunks:
        raise ValueError(f"Failed to create chunks from {file_path}. Text may be too short or malformed.")
    
    avg_chunk_size = sum(len(c) for c in chunks) / len(chunks)
    print(f"[OK] Created {len(chunks)} chunks (avg size: {int(avg_chunk_size):,} chars)")
    
    return chunks


def process_pdf(pdf_path: str, chunk_size: int = 1500, chunk_overlap: int = 400) -> List[str]:
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


