"""
PDF text extraction module using PyMuPDF for processing MEB documents
"""
import fitz  # PyMuPDF
from pathlib import Path
from typing import List, Dict, Union
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PDFProcessor:
    """
    PDF processor for extracting text from MEB PDF documents
    Uses PyMuPDF which is excellent for Turkish text extraction
    """
    
    def __init__(self):
        self.processed_docs = []
    
    def extract_text_from_pdf(self, pdf_path: Path) -> Dict[str, Union[str, int]]:
        """
        Extract text from a single PDF file
        
        Args:
            pdf_path: Path to the PDF file
            
        Returns:
            Dictionary with filename and extracted text
        """
        try:
            doc = fitz.open(pdf_path)
            text_blocks = []
            
            for page_num in range(len(doc)):
                page = doc.load_page(page_num)
                # Extract text with better formatting preservation
                text = page.get_text("text")  # type: ignore
                if text.strip():  # Only add non-empty pages
                    text_blocks.append(f"--- Sayfa {page_num + 1} ---\n{text}")
            
            doc.close()
            
            full_text = "\n\n".join(text_blocks)
            
            result = {
                "filename": pdf_path.name,
                "text": full_text,
                "page_count": len(text_blocks),
                "char_count": len(full_text)
            }
            
            logger.info(f"Extracted {result['char_count']} characters from {result['filename']} ({result['page_count']} pages)")
            return result
            
        except Exception as e:
            logger.error(f"Error processing {pdf_path}: {str(e)}")
            return {
                "filename": pdf_path.name,
                "text": "",
                "page_count": 0,
                "char_count": 0,
                "error": str(e)
            }
    
    def process_pdf_directory(self, pdf_directory: Path) -> List[Dict[str, Union[str, int]]]:
        """
        Process all PDF files in a directory
        
        Args:
            pdf_directory: Path to directory containing PDF files
            
        Returns:
            List of dictionaries with extracted text and metadata
        """
        pdf_files = list(pdf_directory.glob("*.pdf"))
        if not pdf_files:
            logger.warning(f"No PDF files found in {pdf_directory}")
            return []
        
        logger.info(f"Found {len(pdf_files)} PDF files to process")
        
        processed_docs = []
        for pdf_path in pdf_files:
            logger.info(f"Processing {pdf_path.name}...")
            result = self.extract_text_from_pdf(pdf_path)
            if result["text"]:  # Only add successfully processed files
                processed_docs.append(result)
        
        logger.info(f"Successfully processed {len(processed_docs)} PDF files")
        self.processed_docs = processed_docs
        return processed_docs
    
    def get_total_content_stats(self) -> Dict[str, int]:
        """Get statistics about processed content"""
        if not self.processed_docs:
            return {"total_files": 0, "total_pages": 0, "total_chars": 0}
        
        return {
            "total_files": len(self.processed_docs),
            "total_pages": sum(doc["page_count"] for doc in self.processed_docs),
            "total_chars": sum(doc["char_count"] for doc in self.processed_docs)
        } 