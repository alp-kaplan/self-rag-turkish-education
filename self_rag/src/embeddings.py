"""
Local embedding system using sentence-transformers
Optimized for Turkish text and educational content
"""
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Any
import numpy as np
import logging
from config import EMBEDDING_MODEL

logger = logging.getLogger(__name__)

class LocalEmbeddings:
    """
    Local embedding system using sentence-transformers
    Uses a lightweight model that works well with Turkish text
    """
    
    def __init__(self, model_name: str = EMBEDDING_MODEL):
        """
        Initialize the embedding model
        
        Args:
            model_name: Name of the sentence-transformers model to use
        """
        self.model_name = model_name
        self.model = None
        self.embedding_dim = None
        self._load_model()
    
    def _load_model(self):
        """Load the sentence transformer model"""
        try:
            logger.info(f"Loading embedding model: {self.model_name}")
            self.model = SentenceTransformer(self.model_name)
            
            # Get embedding dimension by encoding a test sentence
            test_embedding = self.model.encode("test")
            self.embedding_dim = len(test_embedding)
            
            logger.info(f"Model loaded successfully. Embedding dimension: {self.embedding_dim}")
            
        except Exception as e:
            logger.error(f"Failed to load embedding model {self.model_name}: {str(e)}")
            raise
    
    def embed_text(self, text: str) -> List[float]:
        """
        Encode a single text into embeddings
        
        Args:
            text: Text to encode
            
        Returns:
            List of float values representing the embedding
        """
        if not self.model:
            raise ValueError("Model not loaded")
        
        try:
            embedding = self.model.encode(text)
            return embedding.tolist()
        except Exception as e:
            logger.error(f"Error encoding text: {str(e)}")
            return []
    
    def embed_texts(self, texts: List[str], batch_size: int = 32) -> List[List[float]]:
        """
        Encode multiple texts into embeddings
        
        Args:
            texts: List of texts to encode
            batch_size: Batch size for processing
            
        Returns:
            List of embeddings
        """
        if not self.model:
            raise ValueError("Model not loaded")
        
        if not texts:
            return []
        
        try:
            logger.info(f"Encoding {len(texts)} texts in batches of {batch_size}")
            
            embeddings = []
            for i in range(0, len(texts), batch_size):
                batch = texts[i:i + batch_size]
                batch_embeddings = self.model.encode(batch)
                embeddings.extend([emb.tolist() for emb in batch_embeddings])
                
                if (i // batch_size + 1) % 10 == 0:
                    logger.info(f"Processed {i + len(batch)} / {len(texts)} texts")
            
            logger.info(f"Successfully encoded {len(embeddings)} texts")
            return embeddings
            
        except Exception as e:
            logger.error(f"Error encoding texts: {str(e)}")
            return []
    
    def embed_documents(self, documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Embed a list of document chunks and add embeddings to metadata
        
        Args:
            documents: List of document dictionaries with page_content and metadata
            
        Returns:
            List of documents with embeddings added
        """
        if not documents:
            return []
        
        # Extract texts for batch embedding
        texts = [doc.get("page_content", "") for doc in documents]
        embeddings = self.embed_texts(texts)
        
        # Add embeddings to documents
        embedded_docs = []
        for doc, embedding in zip(documents, embeddings):
            embedded_doc = doc.copy()
            embedded_doc["embedding"] = embedding
            embedded_docs.append(embedded_doc)
        
        logger.info(f"Added embeddings to {len(embedded_docs)} documents")
        return embedded_docs
    
    def get_embedding_dimension(self) -> int:
        """Get the dimension of the embedding vectors"""
        return self.embedding_dim if self.embedding_dim else 0
    
    def similarity(self, embedding1: List[float], embedding2: List[float]) -> float:
        """
        Calculate cosine similarity between two embeddings
        
        Args:
            embedding1: First embedding vector
            embedding2: Second embedding vector
            
        Returns:
            Cosine similarity score
        """
        if len(embedding1) != len(embedding2):
            raise ValueError("Embeddings must have the same dimension")
        
        # Convert to numpy arrays for efficient computation
        vec1 = np.array(embedding1)
        vec2 = np.array(embedding2)
        
        # Calculate cosine similarity
        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return dot_product / (norm1 * norm2) 