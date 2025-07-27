"""
Configuration settings for the Self-RAG system
"""
from pathlib import Path

# Project paths
PROJECT_ROOT = Path(__file__).parent
PDF_DIR = PROJECT_ROOT.parent / "pdf_files"  # PDFs are in pdf_files directory

# Chunking configuration
CHUNK_SIZE = 500  # Smaller chunks for better precision
CHUNK_OVERLAP = 50

# Retrieval configuration 
MAX_CHUNKS_PER_QUERY = 2  # As specified: at most 2 chunks per question

# LLM configuration
OLLAMA_MODEL = "llama3.2:1b"
OLLAMA_BASE_URL = "http://localhost:11434"

# Embedding configuration
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"  # Lightweight Turkish-compatible model

# Vespa configuration
VESPA_APP_NAME = "selfrag"
VESPA_SCHEMA_NAME = "selfrag_doc"  # Changed from "document" to avoid conflicts
VESPA_PORT = 8080

# Grading thresholds
RELEVANCE_THRESHOLD = 0.7
SUPPORT_THRESHOLD = 0.8 