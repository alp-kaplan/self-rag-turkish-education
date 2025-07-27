"""
Self-RAG LangGraph implementation for Turkish educational content
Implements the complete Self-RAG workflow with isREL, isSUP checks
"""
from langgraph.graph import StateGraph, END, START
from typing_extensions import TypedDict
from typing import List, Dict, Any, Optional
import logging
from src.llm_client import OllamaClient
from src.vespa_official import OfficialVespaClient
from src.embeddings import LocalEmbeddings

logger = logging.getLogger(__name__)

class GraphState(TypedDict, total=False):
    """
    State of the Self-RAG graph
    
    Attributes:
        question: Original user question
        documents: Retrieved documents
        generation: Generated answer
        rewritten_question: Query-rewritten version of question
        retrieval_attempts: Number of retrieval attempts made
        generation_attempts: Number of generation attempts made
    """
    question: str
    documents: Optional[List[Dict[str, Any]]]
    generation: Optional[str]
    rewritten_question: Optional[str]
    retrieval_attempts: Optional[int]
    generation_attempts: Optional[int]

class SelfRAGGraph:
    """
    Self-RAG implementation using LangGraph
    
    The graph workflow:
    1. retrieve: Get relevant documents from Vespa
    2. grade_documents: Check document relevance (isREL)
    3. generate: Create answer using LLM
    4. grade_generation: Check for hallucinations (isSUP) and usefulness
    5. rewrite_query: Improve question for better retrieval
    """
    
    def __init__(self, vespa_client: OfficialVespaClient, embeddings: LocalEmbeddings, llm_client: OllamaClient) -> None:
        """
        Initialize Self-RAG graph
        
        Args:
            vespa_client: Vespa vector database client (OfficialVespaClient)
            embeddings: Local embeddings system
            llm_client: Ollama LLM client
        """
        self.vespa = vespa_client
        self.embeddings = embeddings
        self.llm = llm_client
        self.app = None
        self._build_graph()
    
    def _build_graph(self) -> None:
        """Build the Self-RAG LangGraph workflow"""
        
        workflow = StateGraph(GraphState)
        
        # Add nodes
        workflow.add_node("retrieve", self.retrieve)
        workflow.add_node("grade_documents", self.grade_documents)
        workflow.add_node("generate", self.generate)
        workflow.add_node("rewrite_query", self.rewrite_query)
        
        # Add edges
        workflow.add_edge(START, "retrieve")
        workflow.add_edge("retrieve", "grade_documents")
        
        # Conditional edge after document grading
        workflow.add_conditional_edges(
            "grade_documents",
            self.decide_to_generate,
            {
                "rewrite_query": "rewrite_query",
                "generate": "generate",
            },
        )
        
        # Edge from query rewriting back to retrieval
        workflow.add_edge("rewrite_query", "retrieve")
        
        # Conditional edge after generation
        workflow.add_conditional_edges(
            "generate",
            self.grade_generation_and_question,
            {
                "not_supported": "generate",
                "not_useful": "rewrite_query", 
                "useful": END,
            },
        )
        
        # Compile the graph
        self.app = workflow.compile()
        logger.info("Self-RAG graph compiled successfully")
    
    def retrieve(self, state: GraphState) -> GraphState:
        """
        Retrieve documents from Vespa
        
        Args:
            state: Current graph state
            
        Returns:
            Updated state with retrieved documents
        """
        print("\n" + "="*50)
        print("---RETRIEVE---")
        print("="*50)
        
        # Use rewritten question if available, otherwise original question
        question = state.get("rewritten_question") or state.get("question", "")
        
        if not question:
            logger.error("No question provided")
            return {
                "documents": [],
                "question": state.get("question", ""),
                "generation": "",
                "rewritten_question": "",
                "retrieval_attempts": (state.get("retrieval_attempts") or 0) + 1,
                "generation_attempts": state.get("generation_attempts", 0)
            }
        
        # Get query embedding
        query_embedding = self.embeddings.embed_text(question)
        
        if not query_embedding:
            logger.error("Failed to embed query")
            return {
                "documents": [],
                "question": state.get("question", ""),
                "generation": "",
                "rewritten_question": "",
                "retrieval_attempts": (state.get("retrieval_attempts") or 0) + 1,
                "generation_attempts": state.get("generation_attempts", 0)
            }
        
        # Search Vespa for relevant documents
        documents = self.vespa.search(query_embedding)
        
        logger.info(f"Retrieved {len(documents)} documents for question: {question}")
        
        return {
            "documents": documents,
            "question": state.get("question", ""),
            "retrieval_attempts": (state.get("retrieval_attempts") or 0) + 1,
            "generation_attempts": state.get("generation_attempts", 0)
        }
    
    def grade_documents(self, state: GraphState) -> GraphState:
        """
        Grade document relevance using isREL check
        
        Args:
            state: Current graph state
            
        Returns:
            Updated state with filtered relevant documents
        """
        print("\n" + "-"*50)
        print("---CHECK DOCUMENT RELEVANCE TO QUESTION---")
        print("-"*50)
        
        question = state.get("question", "")
        documents = state.get("documents") or []
        
        if not documents:
            logger.warning("No documents to grade")
            return {
                "documents": [], 
                "question": question,
                "generation": "",
                "rewritten_question": "",
                "retrieval_attempts": state.get("retrieval_attempts", 0),
                "generation_attempts": state.get("generation_attempts", 0)
            }
        
        # Grade each document for relevance
        filtered_docs = []
        
        for doc in documents:
            document_content = doc["page_content"]
            
            # Use LLM to grade relevance (isREL)
            grade = self.llm.grade_document_relevance(question, document_content)
            
            if grade == "yes":
                print(f"✅ GRADE: DOCUMENT RELEVANT (source: {doc['metadata']['source']})")
                filtered_docs.append(doc)
            else:
                print(f"❌ GRADE: DOCUMENT NOT RELEVANT (source: {doc['metadata']['source']})")
        
        print(f"\n📊 Result: {len(filtered_docs)} relevant documents from {len(documents)} total")
        logger.info(f"Filtered {len(filtered_docs)} relevant documents from {len(documents)} total")
        
        return {
            "documents": filtered_docs,
            "question": question,
            "generation": "",
            "rewritten_question": "",
            "retrieval_attempts": state.get("retrieval_attempts", 0),
            "generation_attempts": state.get("generation_attempts", 0)
        }
    
    def generate(self, state: GraphState) -> GraphState:
        """
        Generate answer using filtered documents
        
        Args:
            state: Current graph state
            
        Returns:
            Updated state with generated answer
        """
        print("\n" + "="*50)
        print("---GENERATE---")
        print("="*50)
        
        question = state.get("question", "")
        documents = state.get("documents") or []
        
        # Combine documents into context
        if documents:
            context = "\n\n".join([doc["page_content"] for doc in documents])
        else:
            context = "Alakalı belge bulunamadı."
        
        # Generate answer using LLM
        generation = self.llm.generate_answer(question, context)
        
        logger.info(f"Generated answer of length {len(generation)}")
        
        current_gen_attempts = state.get("generation_attempts", 0) or 0
        return {
            "documents": documents,
            "question": question,
            "generation": generation,
            "rewritten_question": "",
            "retrieval_attempts": state.get("retrieval_attempts", 0),
            "generation_attempts": current_gen_attempts + 1
        }
    
    def rewrite_query(self, state: GraphState) -> GraphState:
        """
        Rewrite the query for better retrieval
        
        Args:
            state: Current graph state
            
        Returns:
            Updated state with rewritten question
        """
        print("\n" + "="*50)
        print("---REWRITE QUERY---")
        print("="*50)
        
        question = state.get("question", "")
        
        # Use LLM to rewrite question
        better_question = self.llm.rewrite_question(question)
        
        logger.info(f"Rewritten question: {better_question}")
        
        return {
            "documents": state.get("documents"),
            "question": question,
            "generation": "",
            "rewritten_question": better_question,
            "retrieval_attempts": state.get("retrieval_attempts", 0),
            "generation_attempts": state.get("generation_attempts", 0)
        }
    
    def decide_to_generate(self, state: GraphState) -> str:
        """
        Decide whether to generate or rewrite query based on document relevance
        
        Args:
            state: Current graph state
            
        Returns:
            Next node to execute
        """
        print("\n" + "-"*50)
        print("---ASSESS GRADED DOCUMENTS---")
        print("-"*50)
        
        filtered_documents = state.get("documents") or []
        retrieval_attempts = state.get("retrieval_attempts", 0) or 0
        
        if not filtered_documents:
            # No relevant documents found
            if retrieval_attempts >= 2:
                # Tried enough times, generate anyway
                print("🔄 DECISION: NO RELEVANT DOCUMENTS FOUND, BUT GENERATE ANYWAY (MAX ATTEMPTS REACHED)")
                return "generate"
            else:
                # Try rewriting query
                print("🔄 DECISION: ALL DOCUMENTS ARE NOT RELEVANT TO QUESTION, REWRITE QUERY")
                return "rewrite_query"
        else:
            # We have relevant documents, generate answer
            print("✅ DECISION: GENERATE")
            return "generate"
    
    def grade_generation_and_question(self, state: GraphState) -> str:
        """
        Grade generation for hallucinations (isSUP) and usefulness
        
        Args:
            state: Current graph state
            
        Returns:
            Next node to execute
        """
        print("\n" + "-"*50)
        print("---CHECK HALLUCINATIONS AND USEFULNESS---")
        print("-"*50)
        
        question = state.get("question", "")
        documents = state.get("documents") or []
        generation = state.get("generation", "")
        generation_attempts = state.get("generation_attempts", 0) or 0
        
        # Prevent infinite loops - max 5 generation attempts
        if generation_attempts >= 5:
            print("🔄 DECISION: MAX GENERATION ATTEMPTS REACHED, ACCEPTING ANSWER")
            return "useful"
        
        # Combine documents for hallucination check
        if documents:
            documents_text = "\n\n".join([doc["page_content"] for doc in documents])
        else:
            documents_text = ""
        
        # Check for hallucinations (isSUP)
        if documents_text and generation:
            hallucination_grade = self.llm.grade_hallucination(documents_text, generation)
            
            if hallucination_grade == "yes":
                print("✅ DECISION: GENERATION IS GROUNDED IN DOCUMENTS")
                
                # Check if generation addresses the question
                usefulness_grade = self.llm.grade_answer_usefulness(question, generation)
                
                if usefulness_grade == "yes":
                    print("✅ DECISION: GENERATION ADDRESSES QUESTION")
                    return "useful"
                else:
                    print("❌ DECISION: GENERATION DOES NOT ADDRESS QUESTION")
                    return "not_useful"
            else:
                print("❌ DECISION: GENERATION IS NOT GROUNDED IN DOCUMENTS, RE-TRY")
                return "not_supported"
        else:
            # No documents to check against, just check usefulness
            if generation:
                usefulness_grade = self.llm.grade_answer_usefulness(question, generation)
            
                if usefulness_grade == "yes":
                    print("✅ DECISION: GENERATION ADDRESSES QUESTION (NO DOCS TO CHECK)")
                    return "useful"
                else:
                    print("❌ DECISION: GENERATION DOES NOT ADDRESS QUESTION")
                    return "not_useful"
            else:
                # No generation available
                return "not_supported"
    
    def run(self, question: str) -> Dict[str, Any]:
        """
        Run the Self-RAG workflow for a given question
        
        Args:
            question: User question
            
        Returns:
            Final result with generation and metadata
        """
        if not self.app:
            raise ValueError("Graph not compiled")
        
        logger.info(f"Running Self-RAG for question: {question}")
        
        # Initialize state
        inputs: GraphState = {
            "question": question,
            "documents": [],
            "generation": "",
            "rewritten_question": "",
            "retrieval_attempts": 0,
            "generation_attempts": 0
        }
        
        # Run the graph
        final_state: Optional[GraphState] = None
        for output in self.app.stream(inputs):
            for key, value in output.items():
                print(f"\n🔗 Node '{key}' completed")
                final_state = value
        
        # Handle case where final_state might be None
        if final_state is None:
            logger.error("Graph execution did not produce a final state")
            return {
                "question": question,
                "answer": "Sistem hatası: Graf çalışmadı.",
                "sources": [],
                "retrieval_attempts": 0,
                "rewritten_question": ""
            }
        
        # Return final result
        result = {
            "question": question,
            "answer": final_state.get("generation", "Cevap üretilemedi."),
            "sources": [doc["metadata"]["source"] for doc in (final_state.get("documents") or [])],
            "retrieval_attempts": final_state.get("retrieval_attempts", 0),
            "rewritten_question": final_state.get("rewritten_question", "")
        }
        
        logger.info(f"Self-RAG completed with {len(result['sources'])} sources")
        print("\n" + "="*50)
        print("🎯 SELF-RAG WORKFLOW COMPLETED")
        print("="*50)
        return result 