"""
Document chunking module for splitting extracted PDF text into manageable chunks
"""
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from typing import List, Dict, Any
import logging
from config import CHUNK_SIZE, CHUNK_OVERLAP

logger = logging.getLogger(__name__)

class DocumentChunker:
    """
    Document chunker that splits long texts into smaller, manageable chunks
    optimized for Turkish text and educational content
    """
    
    def __init__(self, chunk_size: int = CHUNK_SIZE, chunk_overlap: int = CHUNK_OVERLAP):
        """
        Initialize the chunker
        
        Args:
            chunk_size: Target size for each chunk
            chunk_overlap: Number of characters to overlap between chunks
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
        # Custom separators optimized for Turkish educational content
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=[
                "\n\n\n",  # Multiple line breaks
                "\n\n",    # Double line breaks  
                "\n",      # Single line breaks
                ". ",      # Sentence endings
                "! ",      # Exclamation sentences
                "? ",      # Question sentences
                "; ",      # Semicolons
                ", ",      # Commas
                " ",       # Spaces
                ""         # Character level
            ],
            keep_separator=True,
            add_start_index=True
        )
    
    def chunk_document(self, document_data: Dict[str, str]) -> List[Document]:
        """
        Chunk a single document
        
        Args:
            document_data: Dictionary containing filename and text
            
        Returns:
            List of LangChain Document objects with metadata
        """
        filename = document_data["filename"]
        text = document_data["text"]
        
        if not text.strip():
            logger.warning(f"Empty text for document {filename}")
            return []
        
        # Split the text into chunks
        chunks = self.text_splitter.split_text(text)
        
        # Create Document objects with metadata
        documents = []
        for i, chunk in enumerate(chunks):
            doc = Document(
                page_content=chunk,
                metadata={
                    "source": filename,
                    "chunk_id": i,
                    "total_chunks": len(chunks),
                    "chunk_size": len(chunk),
                    "document_type": "meb_pdf"
                }
            )
            documents.append(doc)
        
        logger.info(f"Created {len(documents)} chunks from {filename}")
        return documents
    
    def chunk_documents(self, documents_data: List[Dict[str, str]]) -> List[Document]:
        """
        Chunk multiple documents
        
        Args:
            documents_data: List of document dictionaries
            
        Returns:
            List of all Document chunks from all documents
        """
        all_chunks = []
        
        for doc_data in documents_data:
            chunks = self.chunk_document(doc_data)
            all_chunks.extend(chunks)
        
        logger.info(f"Created total of {len(all_chunks)} chunks from {len(documents_data)} documents")
        return all_chunks
    
    def get_chunking_stats(self, documents: List[Document]) -> Dict[str, Any]:
        """
        Get statistics about the chunking process
        
        Args:
            documents: List of chunked documents
            
        Returns:
            Dictionary with chunking statistics
        """
        if not documents:
            return {}
        
        chunk_sizes = [len(doc.page_content) for doc in documents]
        sources = list(set(doc.metadata["source"] for doc in documents))
        
        return {
            "total_chunks": len(documents),
            "unique_sources": len(sources),
            "avg_chunk_size": sum(chunk_sizes) / len(chunk_sizes),
            "min_chunk_size": min(chunk_sizes),
            "max_chunk_size": max(chunk_sizes),
            "sources": sources
        } 