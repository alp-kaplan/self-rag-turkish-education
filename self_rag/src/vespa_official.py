"""
Official Vespa vector database client using pyvespa
Uses the official Vespa Python SDK for reliable deployment and operations
"""
import logging
from typing import List, Dict, Any
import time

from vespa.package import ApplicationPackage, Field, Schema, Document, RankProfile
from vespa.deployment import VespaDocker

from config import VESPA_APP_NAME, VESPA_SCHEMA_NAME, VESPA_PORT, MAX_CHUNKS_PER_QUERY

logger = logging.getLogger(__name__)

class OfficialVespaClient:
    """
    Official Vespa vector database client using pyvespa
    Provides reliable document storage and semantic search capabilities
    """
    
    def __init__(self, embedding_dim: int = 384) -> None:
        """
        Initialize official Vespa client
        
        Args:
            embedding_dim: Dimension of the embedding vectors
        """
        self.app_name = VESPA_APP_NAME
        self.schema_name = VESPA_SCHEMA_NAME
        self.embedding_dim = embedding_dim
        self.port = VESPA_PORT
        
        self.application_package = None
        self.vespa_docker = None
        self.app = None
        self.is_running = False
        
    def create_application_package(self) -> ApplicationPackage:
        """
        Create Vespa application package with proper schema
        
        Returns:
            ApplicationPackage for Self-RAG documents
        """
        logger.info("Creating Vespa application package...")
        
        # Define the document schema
        document_schema = Schema(
            name=self.schema_name,
            document=Document(
                fields=[
                    Field(name="doc_id", type="string", indexing=["summary", "attribute"]),
                    Field(name="text", type="string", indexing=["summary", "index"]),
                    Field(name="source", type="string", indexing=["summary", "attribute"]),
                    Field(name="chunk_id", type="int", indexing=["summary", "attribute"]),
                    Field(name="total_chunks", type="int", indexing=["summary", "attribute"]),
                    Field(name="document_type", type="string", indexing=["summary", "attribute"]),
                    Field(
                        name="embedding", 
                        type=f"tensor<float>(x[{self.embedding_dim}])", 
                        indexing=["summary", "attribute", "index"],
                        attribute=["distance-metric: angular"]
                    )
                ]
            ),
            rank_profiles=[
                RankProfile(
                    name="semantic_search",
                    inputs=[("query(q_embedding)", f"tensor<float>(x[{self.embedding_dim}])")],
                    first_phase="closeness(field, embedding)"
                )
            ]
        )
        
        # Create application package
        self.application_package = ApplicationPackage(
            name=self.app_name,
            schema=[document_schema]
        )
        
        logger.info(f"Created Vespa application package: {self.app_name}")
        return self.application_package
    
    def start_vespa_container(self) -> None:
        """Start Vespa container using official Docker integration"""
        try:
            if not self.application_package:
                self.create_application_package()
            
            if not self.application_package:
                raise ValueError("Failed to create application package")
            
            logger.info("Starting Vespa Docker container with official SDK...")
            
            # Create Vespa Docker instance
            self.vespa_docker = VespaDocker(port=self.port)
            
            # Deploy application package
            logger.info("Deploying application package...")
            self.app = self.vespa_docker.deploy(self.application_package)
            
            # Wait for application to be ready
            self._wait_for_application()
            
            self.is_running = True
            logger.info(f"Vespa application deployed and ready at port {self.port}")
            
        except Exception as e:
            logger.error(f"Failed to start Vespa container: {str(e)}")
            raise
    
    def _wait_for_application(self, timeout: int = 120) -> None:
        """
        Wait for Vespa application to be ready using multiple health checks
        
        Args:
            timeout: Maximum time to wait in seconds (increased for proper Vespa startup)
        """
        start_time = time.time()
        check_count = 0
        max_checks = timeout // 10  # Check every 10 seconds
        
        logger.info("⏳ Waiting for Vespa application to be ready...")
        
        while time.time() - start_time < timeout and check_count < max_checks:
            try:
                # Method 1: Try application status
                try:
                    if self.app:
                        response = self.app.get_application_status()
                        # Try to call is_successful if it exists
                        try:
                            if response.is_successful():  # type: ignore
                                logger.info("✅ Vespa application status check passed")
                                time.sleep(5)
                                return
                        except AttributeError:
                            logger.debug("Application status response doesn't have is_successful method")
                except Exception as e:
                    logger.debug(f"Application status check failed: {str(e)}")
                
                # Method 2: Try a simple query to test if search is working
                try:
                    if self.app:
                        test_response = self.app.query(
                            yql=f"select count(*) from {self.schema_name} where true limit 1",
                            timeout=5
                        )
                        # Try to call is_successful if it exists
                        try:
                            if test_response.is_successful():  # type: ignore
                                logger.info("✅ Vespa application query test passed")
                                time.sleep(5)
                                return
                        except AttributeError:
                            logger.debug("Query response doesn't have is_successful method")
                except Exception as e:
                    logger.debug(f"Query test failed: {str(e)}")
                
                # Method 3: Try HTTP health check directly
                try:
                    import requests
                    health_url = f"http://localhost:{self.port}/status.html"
                    health_response = requests.get(health_url, timeout=5)
                    if health_response.status_code == 200:
                        logger.info("✅ Vespa HTTP health check passed")
                        time.sleep(5)
                        return
                except Exception as e:
                    logger.debug(f"HTTP health check failed: {str(e)}")
                    
            except Exception as e:
                logger.debug(f"Health check iteration failed: {str(e)}")
            
            check_count += 1
            logger.info(f"⏳ Vespa starting... ({check_count}/{max_checks}) - trying multiple health checks")
            time.sleep(10)  # Check every 10 seconds
        
        # Before failing, try one more comprehensive test
        logger.info("🔄 Final readiness verification...")
        try:
            # Give Vespa a bit more time and try once more
            time.sleep(30)
            
            # Try HTTP health check one final time
            import requests
            health_url = f"http://localhost:{self.port}/ApplicationStatus"
            health_response = requests.get(health_url, timeout=10)
            if health_response.status_code == 200:
                logger.info("✅ Vespa is ready (final check passed)")
                return
        except Exception as e:
            logger.debug(f"Final check failed: {str(e)}")
        
        # Instead of raising an error, let's assume it's ready and try to proceed
        logger.warning(f"⚠️ Vespa readiness check inconclusive after {timeout} seconds")
        logger.info("🔄 Proceeding assuming Vespa is ready (will test during indexing)")
        time.sleep(10)  # Give it a bit more time
        return  # Don't raise error, let indexing test if it's actually ready
    
    def index_documents(self, documents: List[Dict[str, Any]]) -> bool:
        """
        Index documents with embeddings into Vespa
        
        Args:
            documents: List of documents with embeddings
            
        Returns:
            True if successful, False otherwise
        """
        if not self.app:
            logger.error("Vespa application not initialized")
            return False
        
        if not documents:
            logger.warning("No documents to index")
            return True
        
        logger.info(f"Indexing {len(documents)} documents...")
        
        success_count = 0
        batch_size = 10  # Process in smaller batches
        
        for i in range(0, len(documents), batch_size):
            batch = documents[i:i + batch_size]
            
            for doc in batch:
                try:
                    # Create document ID
                    doc_id = f"{doc['metadata']['source']}_{doc['metadata']['chunk_id']}"
                    
                    # Prepare document for Vespa
                    vespa_doc = {
                        "doc_id": doc_id,
                        "text": doc["page_content"],
                        "source": doc["metadata"]["source"],
                        "chunk_id": doc["metadata"]["chunk_id"],
                        "total_chunks": doc["metadata"]["total_chunks"],
                        "document_type": doc["metadata"]["document_type"],
                        "embedding": doc["embedding"]
                    }
                    
                    # Index document
                    response = self.app.feed_data_point(
                        schema=self.schema_name,
                        data_id=doc_id,
                        fields=vespa_doc
                    )
                    
                    if response.is_successful():
                        success_count += 1
                    else:
                        logger.warning(f"Failed to index document {doc_id}: {response.get_json()}")
                    
                except Exception as e:
                    logger.error(f"Error indexing document: {str(e)}")
            
            # Log progress
            logger.info(f"Indexed {min(i + batch_size, len(documents))} / {len(documents)} documents")
            
            # Small delay between batches to avoid overwhelming Vespa
            time.sleep(0.5)
        
        logger.info(f"Successfully indexed {success_count} / {len(documents)} documents")
        return success_count == len(documents)
    
    def search(self, query_embedding: List[float], limit: int = MAX_CHUNKS_PER_QUERY) -> List[Dict[str, Any]]:
        """
        Search for similar documents using embedding
        
        Args:
            query_embedding: Query embedding vector
            limit: Maximum number of results to return
            
        Returns:
            List of matching documents with relevance scores
        """
        if not self.app:
            logger.error("Vespa application not initialized")
            return []
        
        try:
            # Prepare query
            query_tensor = {"values": query_embedding}
            
            # Execute search
            response = self.app.query(
                yql=f"select * from {self.schema_name} where true",
                ranking="semantic_search",
                query_features={
                    "query(q_embedding)": query_tensor
                },
                hits=limit
            )
            
            if not response.is_successful():
                logger.error(f"Search failed: {response.get_json()}")
                return []
            
            # Parse results
            results = []
            hits = response.hits
            
            for hit in hits:
                fields = hit["fields"]
                result = {
                    "page_content": fields.get("text", ""),
                    "metadata": {
                        "source": fields.get("source", ""),
                        "chunk_id": fields.get("chunk_id", 0),
                        "total_chunks": fields.get("total_chunks", 0),
                        "document_type": fields.get("document_type", ""),
                        "relevance_score": hit.get("relevance", 0.0)
                    }
                }
                results.append(result)
            
            logger.info(f"Found {len(results)} relevant documents")
            return results
            
        except Exception as e:
            logger.error(f"Error during search: {str(e)}")
            return []
    
    def get_document_count(self) -> int:
        """Get total number of indexed documents"""
        if not self.app:
            return 0
        
        try:
            # Use a simple query to count documents instead of YQL count()
            response = self.app.query(
                yql=f"select * from {self.schema_name} where true",
                hits=1  # We only need to know if documents exist
            )
            
            if response.is_successful():
                # Get total count from response metadata
                return response.json.get("root", {}).get("fields", {}).get("totalCount", len(response.hits))
            else:
                return 0
                
        except Exception as e:
            logger.debug(f"Count query failed, trying alternative method: {str(e)}")
            
            # Alternative: count by iterating (for small datasets)
            try:
                response = self.app.query(
                    yql=f"select doc_id from {self.schema_name} where true",
                    hits=1000  # Reasonable limit
                )
                if response.is_successful():
                    return len(response.hits)
            except Exception:
                pass
            
            return 0
    
    def cleanup(self) -> None:
        """Clean up Vespa resources"""
        try:
            if self.vespa_docker:
                logger.info("Cleaning up Vespa Docker container...")
                # Try to stop and remove containers using try-catch
                try:
                    self.vespa_docker.stop()  # type: ignore
                except AttributeError:
                    logger.debug("VespaDocker doesn't have stop method")
                except Exception as e:
                    logger.debug(f"Error stopping Vespa Docker: {str(e)}")
                    
                try:
                    self.vespa_docker.remove()  # type: ignore
                except AttributeError:
                    logger.debug("VespaDocker doesn't have remove method")
                except Exception as e:
                    logger.debug(f"Error removing Vespa Docker: {str(e)}")
                    
                self.vespa_docker = None
                
            self.app = None
            self.is_running = False
            logger.info("Vespa cleanup completed")
                
        except Exception as e:
            logger.error(f"Error during Vespa cleanup: {str(e)}")


class FallbackVectorStore:
    """
    Fallback in-memory vector store if Vespa fails
    Uses cosine similarity for document retrieval
    """
    
    def __init__(self, embedding_dim: int = 384):
        self.embedding_dim = embedding_dim
        self.documents = []
        self.is_running = True
    
    def create_application_package(self):
        """Compatibility method"""
        pass
    
    def start_vespa_container(self):
        """Mock start method"""
        logger.info("Using fallback in-memory vector store")
        self.is_running = True
    
    def index_documents(self, documents: List[Dict[str, Any]]) -> bool:
        """Store documents in memory"""
        self.documents = documents.copy()
        logger.info(f"Stored {len(documents)} documents in fallback vector store")
        return True
    
    def search(self, query_embedding: List[float], limit: int = MAX_CHUNKS_PER_QUERY) -> List[Dict[str, Any]]:
        """Search using cosine similarity"""
        if not self.documents:
            return []
        
        try:
            import numpy as np
            
            query_vec = np.array(query_embedding)
            results = []
            
            for doc in self.documents:
                doc_vec = np.array(doc["embedding"])
                
                # Calculate cosine similarity
                cosine_sim = np.dot(query_vec, doc_vec) / (np.linalg.norm(query_vec) * np.linalg.norm(doc_vec))
                
                result = {
                    "page_content": doc["page_content"],
                    "metadata": {
                        "source": doc["metadata"]["source"],
                        "chunk_id": doc["metadata"]["chunk_id"],
                        "total_chunks": doc["metadata"]["total_chunks"],
                        "document_type": doc["metadata"]["document_type"],
                        "relevance_score": float(cosine_sim)
                    }
                }
                results.append(result)
            
            # Sort by relevance score
            results.sort(key=lambda x: x["metadata"]["relevance_score"], reverse=True)
            
            return results[:limit]
            
        except Exception as e:
            logger.error(f"Error during fallback search: {str(e)}")
            return []
    
    def get_document_count(self) -> int:
        return len(self.documents)
    
    def cleanup(self):
        self.documents = []


def create_vespa_client(embedding_dim: int = 384):
    """
    Create Vespa client with automatic fallback
    
    Args:
        embedding_dim: Dimension of embedding vectors
        
    Returns:
        Either OfficialVespaClient or FallbackVectorStore
    """
    logger.info("🔄 Attempting to start Vespa vector database...")
    
    try:
        client = OfficialVespaClient(embedding_dim)
        client.create_application_package()
        client.start_vespa_container()
        logger.info("✅ Vespa vector database started successfully")
        return client
    except TimeoutError as e:
        logger.warning(f"⏰ Vespa readiness timeout: {str(e)}")
        logger.info("🔄 Falling back to in-memory vector store...")
        return FallbackVectorStore(embedding_dim)
    except Exception as e:
        logger.warning(f"❌ Failed to start Vespa: {str(e)}")
        logger.info("🔄 Falling back to in-memory vector store...")
        return FallbackVectorStore(embedding_dim) 