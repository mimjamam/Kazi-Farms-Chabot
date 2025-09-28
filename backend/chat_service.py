import os
from typing import Dict, Any
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_groq import ChatGroq
from config import Settings
from .query_agent import QueryValidationAgent
from .similarity_agent import SimilarityComparisonAgent
from .funny_fallback_agent import FunnyFallbackAgent
from .personal_info_guard import PersonalInfoGuard

class VectorStoreService:
    def __init__(self):
        self.settings = Settings()
        self.vectorstore = None
        self.embedding_model = None
        self.context_vectorstore = None
    
    def load_vectorstore(self):
        self.embedding_model = HuggingFaceEmbeddings(
            model_name=self.settings.EMBEDDING_MODEL
        )
        
        if not os.path.exists(self.settings.DB_FAISS_PATH):
            raise FileNotFoundError(f"FAISS folder '{self.settings.DB_FAISS_PATH}' not found!")
        
        self.vectorstore = FAISS.load_local(
            self.settings.DB_FAISS_PATH, 
            self.embedding_model, 
            allow_dangerous_deserialization=True
        )
        return self.vectorstore
    
    def hybrid_search(self, query, top_k=None, threshold=None):
        if top_k is None:
            top_k = self.settings.TOP_K
        if threshold is None:
            threshold = self.settings.CONFIDENCE_THRESHOLD
            
        if self.vectorstore is None:
            raise ValueError("Vectorstore not loaded. Call load_vectorstore() first.")
        
        try:
            semantic_hits = self.vectorstore.similarity_search_with_score(query, k=top_k)
        except Exception:
            semantic_hits = [(doc, None) for doc in self.vectorstore.similarity_search(query, k=top_k)]

        keyword_hits = [(doc, None) for doc in self.vectorstore.similarity_search(query, k=top_k)]

        hits = semantic_hits + keyword_hits
        unique_hits = []
        seen = set()
        max_score = 2.0
        
        for doc, score in hits:
            if doc.page_content not in seen:
                confidence = 100.0 if score is None else max(0.0, 100.0 * (1 - score/max_score))
                if confidence >= threshold:
                    unique_hits.append((doc, confidence))
                seen.add(doc.page_content)
        
        return unique_hits[:top_k]
    
    def initialize_context_vectorstore(self):
        """Initialize a separate vector store for conversation context"""
        if self.embedding_model is None:
            self.embedding_model = HuggingFaceEmbeddings(
                model_name=self.settings.EMBEDDING_MODEL
            )
        
        # Create a new empty vector store for context
        from langchain_core.documents import Document
        self.context_vectorstore = FAISS.from_documents([], self.embedding_model)
        return self.context_vectorstore
    
    def add_context_to_vectorstore(self, session_id: str, conversation_context: str):
        """Add conversation context to the vector store"""
        if self.context_vectorstore is None:
            self.initialize_context_vectorstore()
        
        from langchain_core.documents import Document
        from datetime import datetime
        
        # Create document with context
        doc = Document(
            page_content=conversation_context,
            metadata={
                "session_id": session_id,
                "timestamp": datetime.now().isoformat(),
                "type": "conversation_context"
            }
        )
        
        # Add to context vector store
        self.context_vectorstore.add_documents([doc])
    
    def search_context(self, query: str, session_id: str = None, top_k: int = 3):
        """Search for relevant conversation context"""
        if self.context_vectorstore is None:
            return []
        
        try:
            # Search for relevant context
            results = self.context_vectorstore.similarity_search_with_score(query, k=top_k)
            
            # Filter by session_id if provided
            if session_id:
                filtered_results = []
                for doc, score in results:
                    if doc.metadata.get("session_id") == session_id:
                        filtered_results.append((doc, score))
                return filtered_results
            
            return results
        except Exception as e:
            print(f"Error searching context: {e}")
            return []
    
    def get_context_summary(self, session_id: str = None, max_contexts: int = 5):
        """Get a summary of conversation contexts"""
        if self.context_vectorstore is None:
            return ""
        
        try:
            # Get recent contexts
            if session_id:
                # Search for contexts from this session
                results = self.context_vectorstore.similarity_search("", k=max_contexts)
                session_contexts = [doc for doc in results if doc.metadata.get("session_id") == session_id]
            else:
                # Get most recent contexts
                results = self.context_vectorstore.similarity_search("", k=max_contexts)
                session_contexts = results
            
            if not session_contexts:
                return ""
            
            # Combine contexts
            context_summary = "Recent conversation context:\n"
            for i, doc in enumerate(session_contexts[-max_contexts:], 1):
                context_summary += f"{i}. {doc.page_content[:200]}...\n"
            
            return context_summary
        except Exception as e:
            print(f"Error getting context summary: {e}")
            return ""

class ChatService:
    def __init__(self):
        self.settings = Settings()
        self.vector_service = VectorStoreService()
        self.query_agent = QueryValidationAgent()
        self.similarity_agent = SimilarityComparisonAgent()
        self.funny_fallback_agent = FunnyFallbackAgent()
        self.personal_info_guard = PersonalInfoGuard()
    
    def initialize(self):
        self.settings.validate_config()
        self.vector_service.load_vectorstore()
        self.vector_service.initialize_context_vectorstore()
    
    def _is_irrelevant_question(self, query: str) -> bool:
        """Determine if a query is irrelevant to Kazi Farms HR topics"""
        query_lower = query.lower()
        
        # HR-related keywords that are relevant
        hr_keywords = [
            'salary', 'pay', 'wage', 'compensation', 'income', 'earnings', 'payment',
            'allowance', 'benefit', 'bonus', 'incentive', 'house allowance', 'transport allowance',
            'medical allowance', 'food allowance', 'leave', 'vacation', 'sick leave',
            'annual leave', 'casual leave', 'maternity leave', 'paternity leave', 'holiday',
            'hr', 'human resources', 'employee', 'staff', 'personnel', 'recruitment',
            'hiring', 'policy', 'procedure', 'job', 'position', 'role', 'designation',
            'management', 'supervisor', 'manager', 'director', 'trainee', 'kazi farms',
            'kazi', 'farms', 'company', 'department', 'office', 'work', 'employment'
        ]
        
        # Check if query contains any HR-related keywords
        for keyword in hr_keywords:
            if keyword in query_lower:
                return False  # Relevant question
        
        # If no HR keywords found, likely irrelevant
        return True
    
    def process_query(self, query, conversation_context=""):
        if self.vector_service.vectorstore is None:
            self.initialize()
        
        # Log user input to terminal
        print(f"\n[USER INPUT] {query}")
        if conversation_context:
            print(f"[CONVERSATION CONTEXT] {conversation_context[:200]}...")
        
        # Check for personal information queries first
        personal_info_result = self.personal_info_guard.handle_personal_info_query(query)
        if personal_info_result["is_personal_query"]:
            print(f"[QUERY TYPE] {personal_info_result['query_type']}")
            print(f"[RESPONSE] {personal_info_result['response']}")
            return {
                "result": personal_info_result["response"],
                "source_documents": [],
                "confidence": 0,
                "query_analysis": None,
                "followup_suggestion": "",
                "blocked": True
            }
        
        is_complete, followup_suggestion = self.query_agent.validate_query_completeness(query)
        query_analysis = self.query_agent.analyze_query(query)
        
        # Log query analysis to terminal
        print(f"[QUERY ANALYSIS] Type: {query_analysis.query_type}, Complete: {is_complete}")
        if query_analysis.extracted_info:
            print(f"[EXTRACTED INFO] {query_analysis.extracted_info}")
        
        hits = self.vector_service.hybrid_search(query)
        
        # Log search results to terminal
        print(f"[SEARCH RESULTS] Found {len(hits)} results")
        if hits:
            avg_confidence = sum([hit[1] for hit in hits]) / len(hits)
            print(f"[AVERAGE CONFIDENCE] {avg_confidence:.2f}")
            for i, hit in enumerate(hits[:3]):  # Log top 3 results
                print(f"[RESULT {i+1}] Confidence: {hit[1]:.2f}, Source: {getattr(hit[0], 'metadata', {}).get('source', 'Unknown')}")
        
        if not hits:
            print("[FALLBACK] No search results found")
            # Check if this is an irrelevant question
            if self._is_irrelevant_question(query):
                print("[IRRELEVANT QUESTION] Detected irrelevant query")
                fallback_response = self.funny_fallback_agent.generate_fallback_response(
                    query, 0.0, 'irrelevant_question'
                )
            else:
                fallback_response = self.funny_fallback_agent.analyze_query_context(
                    query,
                    [],
                    0.0
                )
            print(f"[FALLBACK RESPONSE] {fallback_response}")
            return {
                "result": fallback_response,
                "source_documents": [],
                "confidence": 0,
                "query_analysis": query_analysis,
                "followup_suggestion": followup_suggestion
            }
        
        avg_confidence = sum([hit[1] for hit in hits]) / len(hits) if hits else 0
        highest_confidence = hits[0][1] if hits else 0
        
        # Always use the highest confidence result, even if below threshold
        print(f"[CONFIDENCE] Average: {avg_confidence:.2f}, Highest: {highest_confidence:.2f}")
        
        if highest_confidence < self.settings.CONFIDENCE_THRESHOLD:
            print(f"[LOW CONFIDENCE] Highest confidence ({highest_confidence:.2f}) below threshold ({self.settings.CONFIDENCE_THRESHOLD})")
            print(f"[USING HIGHEST] Proceeding with highest confidence result anyway")
            
            # Check if this is an irrelevant question even with low confidence
            if self._is_irrelevant_question(query):
                print("[IRRELEVANT QUESTION] Detected irrelevant query with low confidence")
                fallback_response = self.funny_fallback_agent.generate_fallback_response(
                    query, highest_confidence, 'irrelevant_question'
                )
                print(f"[IRRELEVANT RESPONSE] {fallback_response}")
                return {
                    "result": fallback_response,
                    "source_documents": [hit[0] for hit in hits],
                    "confidence": highest_confidence,
                    "query_analysis": query_analysis,
                    "followup_suggestion": followup_suggestion
                }
        
        # Use the highest confidence results for context instead of QA chain retriever
        if hits:
            # Get the top result for context
            top_hit = hits[0]
            context = top_hit[0].page_content
            
            # Get additional context from vector store
            vector_context = ""
            if self.vector_service:
                try:
                    vector_context = self.vector_service.get_context_summary(max_contexts=3)
                except Exception as e:
                    print(f"Error getting vector context: {e}")
            
            # Create a custom prompt with the highest confidence result
            # Enhanced prompt template with conversation context
            enhanced_template = f"""
{self.settings.CUSTOM_PROMPT_TEMPLATE}

Previous conversation context:
{conversation_context}

Vector store context:
{vector_context}

Current question: {{question}}
Context: {{context}}

Answer:"""
            
            prompt_template = PromptTemplate(
                template=enhanced_template, 
                input_variables=["context", "question"]
            )
            
            llm = ChatGroq(
                model_name=self.settings.LLM_MODEL,
                temperature=self.settings.LLM_TEMPERATURE,
                groq_api_key=self.settings.GROQ_API_KEY,
            )
            
            prompt = prompt_template.format(context=context, question=query)
            response = llm.invoke(prompt)
            result = response.content
            source_docs = [top_hit[0]]  # Use the highest confidence document
        else:
            result = "I don't have specific information about this in our database. Please contact Kazifarm directly for detailed information."
            source_docs = []
        
        # Log LLM response to terminal
        print(f"[LLM RESPONSE] {result}")
        
        # Log followup suggestions to terminal instead of showing to user
        if not is_complete and followup_suggestion:
            print(f"[FOLLOWUP SUGGESTION] {followup_suggestion}")
        
        return {
            "result": result,
            "source_documents": source_docs,
            "confidence": highest_confidence,  # Use highest confidence instead of average
            "query_analysis": query_analysis,
            "followup_suggestion": followup_suggestion
        }
    
    def get_answer_with_sources(self, query, conversation_context=""):
        response = self.process_query(query, conversation_context)
        result = response["result"]
        
        # Log metadata to terminal instead of showing to user
        if response["source_documents"]:
            sources_info = "\n".join(
                [f"• {getattr(doc, 'metadata', {})}" for doc in response["source_documents"]]
            )
            print(f"\n[DEBUG] Sources: {sources_info}")
        
        confidence = response.get("confidence", 0)
        if confidence > 0:
            print(f"[DEBUG] Confidence: {confidence:.1f}%")
        
        return result
    
    def process_query_with_similarity(self, query: str) -> Dict[str, Any]:
        response = self.process_query(query)
        
        if response.get("blocked"):
            return response
        
        if response.get("result"):
            similarity_metrics = self.similarity_agent.calculate_comprehensive_similarity(
                query, 
                response["result"]
            )
            similarity_report = self.similarity_agent.generate_similarity_report(
                query, 
                response["result"], 
                similarity_metrics
            )
            
            response["similarity_metrics"] = similarity_metrics
            response["similarity_report"] = similarity_report
        
        return response
