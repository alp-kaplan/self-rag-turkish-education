#!/usr/bin/env python3
"""
Interactive Self-RAG Turkish Educational Q&A System
Main entry point for the Self-RAG pipeline with interactive CLI interface
Provides interactive question-answer interface with suggested questions
"""
import logging
from pathlib import Path
from typing import Optional
import time

from config import PDF_DIR
from src.pdf_processor import PDFProcessor
from src.chunking import DocumentChunker
from src.embeddings import LocalEmbeddings
from src.vespa_official import OfficialVespaClient
from src.llm_client import OllamaClient
from src.self_rag_graph import SelfRAGGraph

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(message)s'
)
logger = logging.getLogger(__name__)

class InteractiveSelfRAG:
    """
    Interactive Self-RAG pipeline for Turkish educational Q&A
    """
    
    def __init__(self) -> None:
        """Initialize the pipeline components"""
        self.pdf_processor = PDFProcessor()
        self.chunker = DocumentChunker()
        self.embeddings = LocalEmbeddings()
        self.vespa_client = None
        self.llm_client = OllamaClient()
        self.self_rag_graph = None
        
        self.is_ready = False
    
    def setup_pipeline(self, pdf_directory: Optional[Path] = None) -> bool:
        """Setup the complete Self-RAG pipeline"""
        if pdf_directory is None:
            pdf_directory = PDF_DIR
        
        print("\n🚀 Türk Eğitim Sistemi Self-RAG Kurulumu")
        print("=" * 60)
        
        steps = [
            ("📄 PDF İşleme", lambda: self._process_pdfs(pdf_directory)),
            ("✂️ Belge Bölümleme", self._create_chunks),
            ("🔤 Embedding Oluşturma", self._create_embeddings),
            ("🗄️ Vespa Vektör DB Kurulumu", self._setup_vespa),
            ("🧠 Self-RAG Grafı İnit", self._initialize_self_rag),
        ]
        
        start_time = time.time()
        
        for step_name, step_func in steps:
            print(f"\n⏳ {step_name}...")
            
            step_start = time.time()
            success = step_func()
            step_time = time.time() - step_start
            
            if success:
                print(f"✅ {step_name} tamamlandı ({step_time:.1f}s)")
            else:
                print(f"❌ {step_name} başarısız oldu!")
                return False
        
        total_time = time.time() - start_time
        print(f"\n🎉 Sistem kurulumu tamamlandı! ({total_time:.1f}s)")
        
        self.is_ready = True
        return True
    
    def _process_pdfs(self, pdf_directory: Path) -> bool:
        """Process PDF files"""
        documents_data = self.pdf_processor.process_pdf_directory(pdf_directory)
        
        if not documents_data:
            return False
        
        stats = self.pdf_processor.get_total_content_stats()
        print(f"   📊 {stats['total_files']} dosya, {stats['total_pages']} sayfa, {stats['total_chars']} karakter")
        return True
    
    def _create_chunks(self) -> bool:
        """Create document chunks"""
        if not self.pdf_processor.processed_docs:
            return False
        
        self.chunks = self.chunker.chunk_documents(self.pdf_processor.processed_docs)
        
        if not self.chunks:
            return False
        
        stats = self.chunker.get_chunking_stats(self.chunks)
        print(f"   📊 {stats['total_chunks']} chunk, ortalama {stats['avg_chunk_size']:.0f} karakter")
        return True
    
    def _create_embeddings(self) -> bool:
        """Create embeddings for chunks"""
        if not hasattr(self, 'chunks') or not self.chunks:
            return False
        
        chunk_dicts = []
        for chunk in self.chunks:
            chunk_dict = {
                "page_content": chunk.page_content,
                "metadata": chunk.metadata
            }
            chunk_dicts.append(chunk_dict)
        
        self.embedded_chunks = self.embeddings.embed_documents(chunk_dicts)
        
        if not self.embedded_chunks:
            return False
        
        print(f"   📊 {len(self.embedded_chunks)} chunk için {self.embeddings.get_embedding_dimension()}D embedding")
        return True
    
    def _setup_vespa(self) -> bool:
        """Setup Vespa vector database"""
        if not hasattr(self, 'embedded_chunks') or not self.embedded_chunks:
            return False
        
        try:
            embedding_dim = self.embeddings.get_embedding_dimension()
            self.vespa_client = OfficialVespaClient(embedding_dim)
            
            self.vespa_client.create_application_package()
            self.vespa_client.start_vespa_container()
            
            success = self.vespa_client.index_documents(self.embedded_chunks)
            
            if success:
                doc_count = self.vespa_client.get_document_count()
                print(f"   📊 {doc_count} belge Vespa'da indekslendi")
                return True
            
            return False
                
        except Exception as e:
            print(f"   ❌ Vespa hatası: {str(e)}")
            return False
    
    def _initialize_self_rag(self) -> bool:
        """Initialize Self-RAG graph"""
        try:
            if not self.vespa_client:
                logger.error("Vespa client not initialized")
                return False
                
            self.self_rag_graph = SelfRAGGraph(
                vespa_client=self.vespa_client,
                embeddings=self.embeddings,
                llm_client=self.llm_client
            )
            return True
            
        except Exception as e:
            print(f"   ❌ Self-RAG hatası: {str(e)}")
            return False
    
    def ask_question(self, question: str) -> dict:
        """Process a question through Self-RAG"""
        if not self.is_ready or not self.self_rag_graph:
            return {
                "answer": "❌ Sistem henüz hazır değil.",
                "sources": [],
                "processing_time": 0,
                "error": "System not ready"
            }
        
        if not question or not question.strip():
            return {
                "answer": "❌ Lütfen geçerli bir soru yazın.",
                "sources": [],
                "processing_time": 0,
                "error": "Empty question"
            }
        
        print(f"\n🤔 Sorunuz işleniyor...")
        start_time = time.time()
        
        try:
            result = self.self_rag_graph.run(question.strip())
            processing_time = time.time() - start_time
            
            if not isinstance(result, dict):
                raise ValueError("Invalid result format")
            
            result["processing_time"] = processing_time
            
            # Ensure required fields
            if not result.get("answer"):
                result["answer"] = "Üzgünüm, bu soruya uygun bir cevap bulamadım."
            
            if "sources" not in result:
                result["sources"] = []
                
            if "retrieval_attempts" not in result:
                result["retrieval_attempts"] = 0
            
            return result
            
        except Exception as e:
            processing_time = time.time() - start_time
            return {
                "answer": f"❌ Hata oluştu: {str(e)}",
                "sources": [],
                "processing_time": processing_time,
                "error": str(e)
            }
    
    def display_result(self, result: dict) -> None:
        """Display formatted result"""
        print("\n" + "="*70)
        print("📝 CEVAP:")
        print("-" * 70)
        print(result['answer'])
        
        if result.get('sources'):
            print(f"\n📚 KAYNAKLAR:")
            for i, source in enumerate(result['sources'], 1):
                print(f"   {i}. {source}")
        
        print(f"\n📊 İŞLEM BİLGİLERİ:")
        print(f"   ⏱️ Süre: {result['processing_time']:.1f} saniye")
        
        if result.get('retrieval_attempts'):
            print(f"   🔄 Arama sayısı: {result['retrieval_attempts']}")
        
        if result.get('rewritten_question'):
            orig_q = result.get('question', '')
            if result['rewritten_question'] != orig_q:
                print(f"   ✏️ Geliştirilmiş soru: {result['rewritten_question']}")
        
        print("="*70)
    
    def cleanup(self) -> None:
        """Clean up resources"""
        if self.vespa_client:
            try:
                self.vespa_client.cleanup()
                print("\n🧹 Temizlik tamamlandı")
            except Exception as e:
                print(f"\n⚠️ Temizlik hatası: {str(e)}")

def show_welcome() -> None:
    """Show welcome message"""
    print("\n" + "🎓 TÜRKİYE EĞİTİM SİSTEMİ SELF-RAG SORU-CEVAP" + "\n")
    print("=" * 60)
    print("📚 MEB belgeleri üzerinde Self-RAG teknolojisi")
    print("🧠 Ollama + Vespa + LangGraph")
    print("🔄 isREL ve isSUP LLM kontrolleri")
    print("=" * 60)

def show_suggestions() -> None:
    """Show sample questions to the user"""
    print("\n💡 ÖNERİLEN SORULAR:")
    print("-" * 30)
    
    suggestions = [
        "Türkiye'de eğitim sistemi nasıl düzenlenmiştir?",
        "Öğretmen yeterlilikleri nasıl belirlenir?",
        "Müfredat geliştirme süreçleri kimler tarafından yürütülür?",
        "Eğitim kalitesini artırmak için hangi önlemler alınır?",
        "Okul öncesi eğitim nasıl organize edilmiştir?",
        "İlkokul müfredatında hangi dersler yer alır?",
        "Özel eğitim ihtiyacı olan öğrenciler için neler yapılır?",
        "Türkçe dersi öğretim programının amaçları nelerdir?"
    ]
    
    for i, suggestion in enumerate(suggestions, 1):
        print(f"{i}. {suggestion}")
    
    print("-" * 30)

def get_user_input() -> str:
    """Get user question input"""
    print("❓ Sorunuzu yazın: ", end="")
    
    try:
        question = input().strip()
        return question
    except (KeyboardInterrupt, EOFError):
        return "q"

def main() -> None:
    """Main interactive loop"""
    show_welcome()
    
    # Initialize system
    pipeline = InteractiveSelfRAG()
    
    try:
        # Setup pipeline
        success = pipeline.setup_pipeline()
        
        if not success:
            print("\n❌ Sistem kurulumu başarısız oldu!")
            return
        
        show_suggestions()
        
        print("\n🎯 Self-RAG sistemi hazır!")
        print("💡 Yukarıdaki önerilen sorulardan birini kopyalayabilir veya kendi sorunuzu yazabilirsiniz.")
        print("⚠️  Çıkmak için 'exit', 'quit', 'çıkış' veya 'q' yazın\n")
        
        # Interactive loop
        while True:
            try:
                question = get_user_input()
                
                if question.lower().strip() in ['exit', 'quit', 'çıkış', 'q']:
                    print("\n👋 Self-RAG sistemi kapatılıyor...")
                    break
                
                if question.strip():
                    result = pipeline.ask_question(question)
                    pipeline.display_result(result)
                    print("\n" + "🔄 Başka bir soru sorabilirsiniz...\n")
                else:
                    print("❓ Lütfen geçerli bir soru yazın.\n")
                    
            except KeyboardInterrupt:
                print("\n\n👋 Sistem kapatılıyor...")
                break
                
            except Exception as e:
                print(f"\n❌ Hata: {str(e)}\n")
                
    finally:
        # Cleanup
        pipeline.cleanup()

if __name__ == "__main__":
    main() 