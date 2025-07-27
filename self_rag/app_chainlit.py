"""
Chainlit Web UI for Self-RAG Turkish Educational Q&A System
Interactive interface for MEB document queries using Self-RAG
"""
import chainlit as cl
import asyncio
import time
import logging
from typing import Dict, Any, Optional

from main import InteractiveSelfRAG

# Configure logging for web UI
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global pipeline instance
pipeline: Optional[InteractiveSelfRAG] = None

@cl.on_chat_start
async def start():
    """Initialize the Self-RAG pipeline when chat starts"""
    global pipeline
    
    # Show startup message
    await cl.Message(
        content="🚀 **Self-RAG Türkçe Eğitim Sistemi** başlatılıyor...\n\n"
               "MEB dokümanları üzerinde Self-RAG teknolojisi ile soru-cevap sistemi.\n\n"
               "⏳ Lütfen sistemin yüklenmesini bekleyin...",
        author="Sistem"
    ).send()
    
    try:
        # Initialize pipeline
        pipeline = InteractiveSelfRAG()
        
        # Update user about progress
        await cl.Message(
            content="📄 PDF dosyaları işleniyor...",
            author="Sistem"
        ).send()
        
        # Run the full pipeline
        success = await asyncio.to_thread(pipeline.setup_pipeline)
        
        if success:
            # Show success message with system info
            stats = get_system_stats()
            await cl.Message(
                content=f"✅ **Sistem hazır!** \n\n"
                       f"📊 **Sistem Bilgileri:**\n"
                       f"• 📄 İşlenen PDF: {stats['pdf_count']} dosya\n"
                       f"• ✂️ Oluşturulan chunk: {stats['chunk_count']}\n"
                       f"• 🔤 Embedding boyutu: {stats['embedding_dim']}\n"
                       f"• 🧠 LLM modeli: {stats['llm_model']}\n"
                       f"• 🗄️ Vektör DB: Vespa\n\n"
                       f"🎯 **Artık soru sorabilirsiniz!**\n\n"
                       f"*Örnek sorular:*\n"
                       f"• Türkiye'de eğitim sistemi nasıl düzenlenmiştir?\n"
                       f"• Öğretmen yeterlilikleri nasıl belirlenir?\n"
                       f"• Müfredat geliştirme süreçleri kimler tarafından yürütülür?",
                author="Sistem"
            ).send()
            
            # Set user session info
            cl.user_session.set("pipeline", pipeline)
            cl.user_session.set("ready", True)
        else:
            await cl.Message(
                content="❌ Sistem başlatılamadı. Lütfen logs'u kontrol edin.",
                author="Sistem"
            ).send()
            cl.user_session.set("ready", False)
            
    except Exception as e:
        logger.error(f"Error starting pipeline: {str(e)}")
        await cl.Message(
            content=f"❌ **Hata:** {str(e)}\n\n"
                   f"Lütfen şunları kontrol edin:\n"
                   f"• Ollama servisi çalışıyor mu? (`ollama serve`)\n"
                   f"• Docker servisi aktif mi?\n"
                   f"• PDF dosyaları mevcut mu?",
            author="Sistem"
        ).send()
        cl.user_session.set("ready", False)

@cl.on_message
async def main(message: cl.Message):
    """Handle user questions"""
    pipeline = cl.user_session.get("pipeline")
    ready = cl.user_session.get("ready", False)
    
    if not ready or not pipeline:
        await cl.Message(
            content="⚠️ Sistem henüz hazır değil. Lütfen başlatılmasını bekleyin.",
            author="Sistem"
        ).send()
        return
    
    user_question = message.content.strip()
    
    if not user_question:
        await cl.Message(
            content="❓ Lütfen bir soru yazın.",
            author="Sistem"
        ).send()
        return
    
    # Show processing message
    processing_msg = await cl.Message(
        content=f"🤔 **Sorunuz işleniyor:** {user_question}\n\n"
               f"⏳ Self-RAG grafu çalışıyor...\n"
               f"🔍 Alakalı belgeler aranıyor...\n"
               f"🧠 LLM cevap üretiyor...",
        author="Self-RAG"
    ).send()
    
    try:
        start_time = time.time()
        
        # Process question through Self-RAG
        result = await asyncio.to_thread(pipeline.ask_question, user_question)
        
        processing_time = time.time() - start_time
        
        # Format response
        response_content = format_response(result, processing_time)
        
        # Send final result as new message
        await cl.Message(
            content=response_content,
            author="Self-RAG"
        ).send()
        
        # Log the interaction
        logger.info(f"Question answered: {user_question} -> {len(result['answer'])} chars")
        
    except Exception as e:
        logger.error(f"Error processing question: {str(e)}")
        await cl.Message(
            content=f"❌ **Hata oluştu:** {str(e)}\n\n"
                   f"Lütfen sorunuzu yeniden deneyin.",
            author="Sistem"
        ).send()

def get_system_stats() -> Dict[str, Any]:
    """Get system statistics for display"""
    global pipeline
    
    if not pipeline:
        return {}
    
    try:
        # Get basic stats
        pdf_stats = pipeline.pdf_processor.get_total_content_stats()
        chunk_stats = pipeline.chunker.get_chunking_stats(pipeline.chunks)
        embedding_dim = pipeline.embeddings.get_embedding_dimension()
        
        return {
            'pdf_count': pdf_stats['total_files'],
            'chunk_count': chunk_stats['total_chunks'],
            'embedding_dim': embedding_dim,
            'llm_model': 'llama3.2:1b'
        }
    except:
        return {
            'pdf_count': '?',
            'chunk_count': '?', 
            'embedding_dim': '?',
            'llm_model': 'llama3.2:1b'
        }

def format_response(result: Dict[str, Any], processing_time: float) -> str:
    """Format the Self-RAG response for display"""
    
    answer = result.get('answer', 'Cevap üretilemedi.')
    sources = result.get('sources', [])
    retrieval_attempts = result.get('retrieval_attempts', 0)
    rewritten_question = result.get('rewritten_question', '')
    
    # Build response
    response = f"📝 **Cevap:**\n{answer}\n\n"
    
    # Add sources if available
    if sources:
        source_list = '\n'.join([f"• {source}" for source in sources])
        response += f"📚 **Kaynaklar:**\n{source_list}\n\n"
    else:
        response += "📚 **Kaynaklar:** Alakalı belge bulunamadı\n\n"
    
    # Add processing details
    response += f"📊 **İşlem Detayları:**\n"
    response += f"• ⏱️ Süre: {processing_time:.1f} saniye\n"
    response += f"• 🔄 Arama sayısı: {retrieval_attempts}\n"
    
    if rewritten_question and rewritten_question != result.get('question', ''):
        response += f"• ✏️ Soru yeniden yazıldı: *{rewritten_question}*\n"
    
    # Add Self-RAG workflow info
    response += f"\n🔄 **Self-RAG Süreci:**\n"
    if retrieval_attempts > 1:
        response += f"• isREL: Belge alakalılığı kontrol edildi\n"
        response += f"• Query rewriting: Soru optimize edildi\n"
    response += f"• isSUP: Halüsinasyon kontrolü yapıldı\n"
    response += f"• Cevap kalitesi değerlendirildi\n"
    
    return response

@cl.on_chat_end
async def end():
    """Cleanup when chat ends"""
    global pipeline
    
    if pipeline:
        try:
            await asyncio.to_thread(pipeline.cleanup)
            logger.info("Pipeline cleaned up")
        except Exception as e:
            logger.error(f"Error during cleanup: {str(e)}")
        
        pipeline = None

if __name__ == "__main__":
    import os
    
    # Set environment for better display
    os.environ["CHAINLIT_AUTH_SECRET"] = "self-rag-secret"
    
    # Start the Chainlit app
    print("🚀 Self-RAG Chainlit UI başlatılıyor...")
    print("📱 Web arayüzü: http://localhost:8000")
    print("⏹️  Durdurmak için: Ctrl+C") 