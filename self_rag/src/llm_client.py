"""
Ollama LLM client for Self-RAG system
Handles all LLM operations: generation, grading (isREL, isSUP), and query rewriting
"""
import requests
import logging
from config import OLLAMA_MODEL, OLLAMA_BASE_URL

logger = logging.getLogger(__name__)

class OllamaClient:
    """
    Ollama client for local LLM operations in Self-RAG system
    Uses llama3.2:1b model with Turkish language capabilities
    """
    
    def __init__(self, model: str = OLLAMA_MODEL, base_url: str = OLLAMA_BASE_URL):
        """
        Initialize Ollama client
        
        Args:
            model: Name of the Ollama model to use
            base_url: Base URL for Ollama API
        """
        self.model = model
        self.base_url = base_url
        self.api_url = f"{base_url}/api/generate"
        
        # Test connection
        self._test_connection()
    
    def _test_connection(self):
        """Test connection to Ollama service"""
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=10)
            if response.status_code == 200:
                models = response.json().get("models", [])
                model_names = [m["name"] for m in models]
                
                if self.model in model_names:
                    logger.info(f"Connected to Ollama. Model {self.model} is available.")
                else:
                    logger.warning(f"Model {self.model} not found in available models: {model_names}")
            else:
                logger.error(f"Failed to connect to Ollama: {response.status_code}")
        except requests.exceptions.RequestException as e:
            logger.error(f"Cannot connect to Ollama at {self.base_url}: {str(e)}")
            raise ConnectionError(f"Ollama service not available: {str(e)}")
    
    def _generate(self, prompt: str, system_prompt: str = "", temperature: float = 0.1, max_tokens: int = 512, retries: int = 2) -> str:
        """
        Generate text using Ollama model with retry logic
        
        Args:
            prompt: User prompt
            system_prompt: System prompt for context
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            retries: Number of retry attempts
            
        Returns:
            Generated text
        """
        for attempt in range(retries + 1):
            try:
                full_prompt = f"{system_prompt}\n\n{prompt}" if system_prompt else prompt
                
                payload = {
                    "model": self.model,
                    "prompt": full_prompt,
                    "stream": False,
                    "options": {
                        "temperature": temperature,
                        "num_predict": max_tokens
                    }
                }
                
                # Increase timeout for grading operations
                timeout = 90 if max_tokens < 50 else 120
                response = requests.post(self.api_url, json=payload, timeout=timeout)
                
                if response.status_code == 200:
                    result = response.json()
                    return result.get("response", "").strip()
                else:
                    logger.error(f"Ollama API error (attempt {attempt + 1}): {response.status_code} - {response.text}")
                    if attempt == retries:
                        return ""
                        
            except requests.exceptions.Timeout as e:
                logger.warning(f"Ollama API timeout (attempt {attempt + 1}): {str(e)}")
                if attempt == retries:
                    logger.error("All retry attempts failed due to timeout")
                    return ""
                    
            except requests.exceptions.ConnectionError as e:
                logger.warning(f"Ollama connection error (attempt {attempt + 1}): {str(e)}")
                if attempt == retries:
                    logger.error("All retry attempts failed due to connection error")
                    return ""
                    
            except requests.exceptions.RequestException as e:
                logger.error(f"Ollama API error (attempt {attempt + 1}): {str(e)}")
                if attempt == retries:
                    return ""
                    
            # Brief pause between retries
            if attempt < retries:
                import time
                time.sleep(2)
                
        return ""
    
    def grade_document_relevance(self, question: str, document: str) -> str:
        """
        Grade whether a document is relevant to the question (isREL)
        
        Args:
            question: User question
            document: Retrieved document content
            
        Returns:
            "yes" if relevant, "no" if not relevant
        """
        # Simplified prompt matching notebook's proven template
        system_prompt = """Sen, bir kullanıcının sorusuyla getirilen bir belgenin alaka düzeyini değerlendiren bir puanlayıcısın.
Bu, katı bir test olmak zorunda değil. Amaç, hatalı getirilen belgeleri elemek.
Eğer belge, kullanıcının sorusuyla ilgili anahtar kelimeler veya anlamsal içerik barındırıyorsa, onu alakalı olarak değerlendir.
Belgenin soruyla alakalı olup olmadığını belirtmek için 'evet' ya da 'hayır' şeklinde ikili bir puan ver."""
        
        prompt = f"""Getirilen belge: 

{document}

Kullanıcı sorusu: {question}"""
        
        result = self._generate(prompt, system_prompt, temperature=0.0, max_tokens=10)
        
        # Debug: Show raw LLM response
        print(f"🔍 RELEVANCE GRADING - Raw LLM Response: '{result}'")
        
        # Simplified parsing based on notebook approach
        result_clean = result.lower().strip()
        
        if "yes" in result_clean or "evet" in result_clean or "alakalı" in result_clean:
            return "yes"
        elif "no" in result_clean or "hayır" in result_clean or "hayir" in result_clean or "alakasız" in result_clean:
            return "no"
        # Final heuristic fallback
        elif self._contains_relevant_keywords(question.lower(), document.lower()):
            logger.info(f"Fallback: Using keyword matching for relevance -> 'yes'")
            return "yes"
        else:
            logger.warning(f"Unclear relevance result -> defaulting to 'no'")
            return "no"
    
    def _contains_relevant_keywords(self, question: str, document: str) -> bool:
        """
        Fallback keyword-based relevance check
        
        Args:
            question: User question (lowercase)
            document: Document content (lowercase)
            
        Returns:
            True if document contains relevant keywords from question
        """
        # Extract key terms from question (remove common words)
        stop_words = {"nasıl", "nedir", "kimler", "hangi", "ne", "nerede", "neden", "için", "bir", "bu", "şu", "o", "da", "de", "ile", "ve", "veya", "mi", "mu", "mı", "mü"}
        question_words = set(question.split()) - stop_words
        
        # Check if at least 1 significant word appears in document
        for word in question_words:
            if len(word) > 3 and word in document:
                return True
        
        return False
    
    def _has_content_overlap(self, documents: str, generation: str) -> bool:
        """
        Check if generation has meaningful content overlap with documents
        
        Args:
            documents: Documents content (lowercase)
            generation: Generated answer (lowercase)
            
        Returns:
            True if there's significant overlap
        """
        # Split into words, remove common ones
        stop_words = {"bir", "bu", "şu", "o", "da", "de", "ile", "ve", "veya", "için", "olan", "olarak", "gibi", "kadar", "daha", "en", "çok", "az"}
        
        doc_words = set(documents.split()) - stop_words
        gen_words = set(generation.split()) - stop_words
        
        # Filter to meaningful words (length > 3)
        doc_words = {w for w in doc_words if len(w) > 3}
        gen_words = {w for w in gen_words if len(w) > 3}
        
        if not gen_words:
            return False
            
        # Calculate overlap ratio
        overlap = len(doc_words.intersection(gen_words))
        total_gen_words = len(gen_words)
        
        overlap_ratio = overlap / total_gen_words if total_gen_words > 0 else 0
        
        # Consider significant if 30%+ overlap
        return overlap_ratio >= 0.3
    
    def grade_hallucination(self, documents: str, generation: str) -> str:
        """
        Grade whether the generation is supported by the documents (isSUP)
        
        Args:
            documents: Retrieved documents content
            generation: Generated answer
            
        Returns:
            "yes" if supported, "no" if hallucinated
        """
        # Simplified prompt matching notebook's proven template
        system_prompt = """Sen, bir LLM çıktısının getirilen bilgi kümesine dayalı olup olmadığını değerlendiren bir puanlayıcısın.
'evet' veya 'hayır' şeklinde ikili bir puan ver. 'Evet', cevabın bu bilgi kümesine dayandığı / bu bilgilerle desteklendiği anlamına gelir."""
        
        prompt = f"""Getirilen bilgi kümesi: 

{documents}

LLM çıktısı: {generation}"""
        
        result = self._generate(prompt, system_prompt, temperature=0.0, max_tokens=10)
        
        # Debug: Show raw LLM response
        print(f"🔍 HALLUCINATION GRADING - Raw LLM Response: '{result}'")
        
        # Simplified parsing
        result_clean = result.lower().strip()
        
        if "yes" in result_clean or "evet" in result_clean:
            return "yes"
        elif "no" in result_clean or "hayır" in result_clean or "hayir" in result_clean:
            return "no"
        # Content overlap fallback - more lenient to prevent infinite loops
        elif self._has_content_overlap(documents.lower(), generation.lower()):
            logger.info(f"Fallback: Content overlap detected -> 'yes'")
            return "yes"
        else:
            logger.warning(f"Unclear hallucination result -> defaulting to 'yes' to prevent loops")
            return "yes"
    
    def grade_answer_usefulness(self, question: str, generation: str) -> str:
        """
        Grade whether the generation answers the question
        
        Args:
            question: User question
            generation: Generated answer
            
        Returns:
            "yes" if answer addresses question, "no" otherwise
        """
        # Simplified prompt matching notebook's proven template
        system_prompt = """Sen, bir cevabın bir soruyu ele alıp almadığını / çözüp çözmediğini değerlendiren bir puanlayıcısın.
'evet' veya 'hayır' şeklinde ikili bir puan ver. 'Evet', cevabın soruyu çözdüğü anlamına gelir."""
        
        prompt = f"""Kullanıcı sorusu: 

{question}

LLM cevabı: {generation}"""
        
        result = self._generate(prompt, system_prompt, temperature=0.0, max_tokens=10)
        
        # Debug: Show raw LLM response
        print(f"🔍 USEFULNESS GRADING - Raw LLM Response: '{result}'")
        
        # Enhanced parsing to handle various formats
        result_clean = result.lower().strip().replace("cevap:", "").replace("answer:", "").replace(".", "").replace(":", "").strip()
        
        if "yes" in result_clean or "evet" in result_clean:
            return "yes"
        elif "no" in result_clean or "hayır" in result_clean or "hayir" in result_clean:
            return "no"
        else:
            # Lenient fallback - default to yes to prevent infinite loops
            logger.info(f"Defaulting to 'yes' for unclear result")
            return "yes"
    
    def rewrite_question(self, question: str) -> str:
        """
        Rewrite question for better retrieval
        
        Args:
            question: Original question
            
        Returns:
            Rewritten question optimized for vector search
        """
        # Simplified prompt matching notebook's proven template
        system_prompt = """Sen, verilen bir soruyu vektör veri tabanı sorguları için optimize edilmiş daha iyi bir sürüme dönüştüren bir soru yeniden yazıcısısın.
Girdi soruya bak ve altında yatan anlamsal niyeti / anlamı anlamaya çalış."""
        
        prompt = f"""İşte ilk soru: 

{question}

Geliştirilmiş bir soru formüle et."""
        
        result = self._generate(prompt, system_prompt, temperature=0.2, max_tokens=128)
        
        # Debug: Show raw LLM response
        print(f"🔍 QUESTION REWRITING - Raw LLM Response: '{result}'")
        
        # Return rewritten question or original if generation failed
        return result.strip() if result.strip() else question
    
    def generate_answer(self, question: str, context: str) -> str:
        """
        Generate answer based on question and context documents
        
        Args:
            question: User question
            context: Retrieved documents as context
            
        Returns:
            Generated answer
        """
        system_prompt = """Sen bir Türkçe eğitim asistanısın. Verilen belgelerden yararlanarak soruları yanıtlıyorsun.

Kurallar:
1. Sadece verilen belgelerden elde ettiğin bilgileri kullan
2. Eğer belgeler soruya cevap vermek için yeterli değilse, bunu belirt
3. Cevabını Türkçe ver
4. Net ve açık cevaplar ver
5. Belgelerde olmayan bilgileri ekleme"""
        
        prompt = f"""Belgeler:
{context}

Soru: {question}

Cevap:"""
        
        result = self._generate(prompt, system_prompt, temperature=0.3, max_tokens=1024)
        
        # Debug: Show raw LLM response
        print(f"🔍 ANSWER GENERATION - Raw LLM Response: '{result}'")
        
        return result if result else "Üzgünüm, verilen belgeler sorunuzu yanıtlamak için yeterli değil." 