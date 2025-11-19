"""
Simplified LangGraph-based workflow for Kazi Farms Chatbot
"""
import uuid
from typing import Dict, Any, Optional, List, TypedDict
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

# Import all agents
from .intent_agent import IntentAgent, IntentResult
from .dynamic_intent_agent import DynamicIntentAgent
from .query_rewrite_agent import QueryRewriteAgent, QueryRewriteResult
from .domain_guard_agent import DynamicDomainGuardAgent, DomainGuardResult
from .retrieval_agent import RetrievalAgent, RetrievalResult
from .context_selector_agent import ContextSelectorAgent, ContextSelectorResult
from .reasoning_agent import ReasoningAgent, ReasoningResult
from .grounding_checker_agent import GroundingCheckerAgent, GroundingCheckerResult
from .relevance_agent import RelevanceAgent, RelevanceResult
from .fallback_agent import FallbackAgent, FallbackResult


class SimpleChatbotState(TypedDict):
    """Simplified state schema for the LangGraph workflow"""
    # Input
    query: str
    session_id: str
    
    # Processing data (serializable)
    processed_text: str
    intent_type: str
    intent_result: Optional[Any]  # Store full intent result for greeting responses
    rewritten_query: str
    is_in_domain: bool
    retrieved_docs: List[str]
    retrieved_scores: List[float]
    selected_context: List[str]
    plan_type: str
    needs_tools: bool
    generated_answer: str
    answer_confidence: float
    grounding_score: float
    has_unsupported_claims: bool
    relevance_score: float
    is_relevant: bool
    
    # Control flow
    regeneration_count: int
    
    # Final output
    response: str
    confidence: float
    sources: List[str]
    response_type: str
    processing_steps: List[str]
    error: str

class SimpleLangGraphWorkflow:
    """Simplified LangGraph-based workflow for the chatbot"""
    
    def __init__(self):
        self.workflow = None
        self.memory = MemorySaver()
        self._initialize_agents()
        self._build_graph()
    
    def _initialize_agents(self):
        """Initialize all agents"""
        self.intent_agent = DynamicIntentAgent(learning_enabled=True, min_confidence=0.3)
        self.query_rewrite_agent = QueryRewriteAgent()
        self.domain_guard_agent = DynamicDomainGuardAgent()
        self.retrieval_agent = RetrievalAgent()
        self.retrieval_agent.initialize()  # Initialize retrieval agent
        self.context_selector_agent = ContextSelectorAgent()
        self.reasoning_agent = ReasoningAgent()
        self.grounding_checker_agent = GroundingCheckerAgent()
        self.relevance_agent = RelevanceAgent()
        self.fallback_agent = FallbackAgent()
        
        # Import and initialize conversational agent
        from .conversational_agent import ConversationalAgent
        self.conversational_agent = ConversationalAgent()
    
    def _build_graph(self):
        """Build the simplified LangGraph StateGraph"""
        print("[SIMPLE LANGGRAPH WORKFLOW] Building StateGraph...")
        
        # Create the graph
        workflow = StateGraph(SimpleChatbotState)
        
        # Add nodes
        workflow.add_node("detect_intent", self._detect_intent_node)
        workflow.add_node("handle_greeting", self._handle_greeting_node)
        workflow.add_node("rewrite_query", self._rewrite_query_node)
        workflow.add_node("check_domain", self._check_domain_node)
        workflow.add_node("retrieve_docs", self._retrieve_docs_node)
        workflow.add_node("select_context", self._select_context_node)
        workflow.add_node("generate_answer", self._generate_answer_node)
        workflow.add_node("check_grounding", self._check_grounding_node)
        workflow.add_node("check_relevance", self._check_relevance_node)
        workflow.add_node("generate_fallback", self._generate_fallback_node)
        workflow.add_node("finalize_result", self._finalize_result_node)
        
        # Set entry point
        workflow.set_entry_point("detect_intent")
        
        # Intent-based routing
        workflow.add_conditional_edges(
            "detect_intent",
            self._route_after_intent,
            {
                "greeting": "handle_greeting",
                "query": "rewrite_query"
            }
        )
        
        # Greeting handling
        workflow.add_edge("handle_greeting", "finalize_result")
        
        workflow.add_edge("rewrite_query", "check_domain")
        
        # Domain guard routing
        workflow.add_conditional_edges(
            "check_domain",
            self._route_after_domain,
            {
                "blocked": "generate_fallback",
                "allowed": "retrieve_docs"
            }
        )
        
        workflow.add_edge("retrieve_docs", "select_context")
        workflow.add_edge("select_context", "generate_answer")
        workflow.add_edge("generate_answer", "check_grounding")
        
        # Grounding check routing - skip relevance check for speed
        workflow.add_conditional_edges(
            "check_grounding",
            self._route_after_grounding,
            {
                "regenerate": "generate_answer",
                "proceed": "finalize_result"  # Skip relevance check
            }
        )
        
        # Relevance check routing (disabled for performance)
        # workflow.add_conditional_edges(
        #     "check_relevance",
        #     self._route_after_relevance,
        #     {
        #         "relevant": "finalize_result",
        #         "not_relevant": "generate_fallback"
        #     }
        # )
        
        workflow.add_edge("generate_fallback", "finalize_result")
        workflow.add_edge("finalize_result", END)
        
        # Compile the graph
        self.workflow = workflow.compile(checkpointer=self.memory)
        print("[SIMPLE LANGGRAPH WORKFLOW] StateGraph built and compiled successfully")
    
    # Node functions
    def _detect_intent_node(self, state: SimpleChatbotState) -> SimpleChatbotState:
        """Detect user intent"""
        print("[SIMPLE LANGGRAPH] Detect intent node")
        try:
            # Enhanced conversational processing
            conversational_enhancement = self.conversational_agent.enhance_user_input(
                state["query"], 
                state["session_id"]
            )
            
            # Store processed text and context
            state["processed_text"] = conversational_enhancement['enhanced_query']
            state["conversation_context"] = conversational_enhancement.get('conversation_context')
            state["response_strategy"] = conversational_enhancement.get('response_strategy', 'balanced_response')
            state["personalization"] = conversational_enhancement.get('personalization', {})
            
            # Detect intent
            intent_result = self.intent_agent.detect_intent(state["processed_text"])
            state["intent_type"] = intent_result.intent_type
            state["intent_result"] = intent_result  # Store the full intent result
            state["processing_steps"] = state.get("processing_steps", []) + [f"Intent detected: {intent_result.intent_type}"]
            return state
        except Exception as e:
            state["error"] = f"Intent detection failed: {str(e)}"
            return state
    
    def _handle_greeting_node(self, state: SimpleChatbotState) -> SimpleChatbotState:
        """Handle greeting responses using intent agent's greeting response"""
        print("[SIMPLE LANGGRAPH] Handle greeting node")
        try:
            intent_result = state.get("intent_result")
            if intent_result and intent_result.greeting_response:
                # Use the intent agent's greeting response (includes time-specific responses)
                state["response"] = intent_result.greeting_response
                state["confidence"] = float(intent_result.confidence)
                state["response_type"] = "greeting"
                state["processing_steps"] = state.get("processing_steps", []) + ["Greeting response generated"]
            else:
                # Fallback to fallback agent if no greeting response available
                fallback_result = self.fallback_agent.generate_fallback_response('greeting')
                state["response"] = fallback_result.response
                state["confidence"] = float(fallback_result.confidence)
                state["response_type"] = "greeting"
                state["processing_steps"] = state.get("processing_steps", []) + ["Fallback greeting response generated"]
            
            return state
        except Exception as e:
            state["error"] = f"Greeting handling failed: {str(e)}"
            return state
    
    def _rewrite_query_node(self, state: SimpleChatbotState) -> SimpleChatbotState:
        """Rewrite query for better retrieval"""
        print("[SIMPLE LANGGRAPH] Rewrite query node")
        try:
            rewrite_result = self.query_rewrite_agent.rewrite_query(
                state["processed_text"],
                context={'conversation_context': ''}
            )
            state["rewritten_query"] = rewrite_result.rewritten_query
            state["processing_steps"] = state.get("processing_steps", []) + ["Query rewritten"]
            return state
        except Exception as e:
            state["error"] = f"Query rewriting failed: {str(e)}"
            return state
    
    def _check_domain_node(self, state: SimpleChatbotState) -> SimpleChatbotState:
        """Check domain guard - simplified for speed"""
        print("[SIMPLE LANGGRAPH] Check domain node")
        try:
            # Fast keyword-based domain check
            query_lower = state["rewritten_query"].lower()
            
            # HR-related keywords
            hr_keywords = [
                'salary', 'pay', 'leave', 'allowance', 'benefit', 'policy', 'hr',
                'employee', 'staff', 'attendance', 'overtime', 'bonus', 'increment',
                'promotion', 'transfer', 'resignation', 'termination', 'contract',
                'probation', 'confirmation', 'appraisal', 'performance', 'training',
                'medical', 'insurance', 'provident', 'gratuity', 'festival', 'eid',
                'vacation', 'sick', 'casual', 'maternity', 'paternity', 'emergency',
                'manager', 'officer', 'executive', 'supervisor', 'director', 'gm', 'agm',
                'kazi', 'farms', 'kfil', 'kfg', 'kml', 'company', 'organization'
            ]
            
            # Check if query contains any HR keywords
            is_in_domain = any(keyword in query_lower for keyword in hr_keywords)
            
            # If no keywords found, assume it's in domain (let retrieval decide)
            if not is_in_domain and len(query_lower.split()) <= 3:
                is_in_domain = True  # Short queries likely relevant
            
            state["is_in_domain"] = is_in_domain
            state["processing_steps"] = state.get("processing_steps", []) + [f"Domain check: {'ALLOW' if is_in_domain else 'BLOCK'}"]
            return state
        except Exception as e:
            state["error"] = f"Domain guard failed: {str(e)}"
            return state
    
    def _retrieve_docs_node(self, state: SimpleChatbotState) -> SimpleChatbotState:
        """Retrieve documents"""
        print("[SIMPLE LANGGRAPH] Retrieve docs node")
        try:
            retrieval_result = self.retrieval_agent.retrieve_documents(state["rewritten_query"])
            state["retrieved_docs"] = [doc.metadata.get('source', f'doc_{i}') for i, doc in enumerate(retrieval_result.documents)]
            state["retrieved_scores"] = retrieval_result.similarity_scores
            state["processing_steps"] = state.get("processing_steps", []) + [f"Retrieved {len(retrieval_result.documents)} documents"]
            return state
        except Exception as e:
            state["error"] = f"Document retrieval failed: {str(e)}"
            return state
    
    def _select_context_node(self, state: SimpleChatbotState) -> SimpleChatbotState:
        """Select context"""
        print("[SIMPLE LANGGRAPH] Select context node")
        try:
            # Get retrieval result again for context selection
            retrieval_result = self.retrieval_agent.retrieve_documents(state["rewritten_query"])
            context_result = self.context_selector_agent.select_context(
                retrieval_result.documents, state["rewritten_query"]
            )
            state["selected_context"] = [doc.metadata.get('source', f'doc_{i}') for i, doc in enumerate(context_result.selected_context)]
            state["processing_steps"] = state.get("processing_steps", []) + [f"Selected {context_result.context_count} context documents"]
            
            # Cache retrieval result for later use
            state["_cached_retrieval_result"] = retrieval_result
            state["_cached_context_result"] = context_result
            return state
        except Exception as e:
            state["error"] = f"Context selection failed: {str(e)}"
            return state
    
    def _generate_answer_node(self, state: SimpleChatbotState) -> SimpleChatbotState:
        """Generate answer"""
        print("[SIMPLE LANGGRAPH] Generate answer node")
        try:
            # Use cached results if available
            retrieval_result = state.get("_cached_retrieval_result")
            context_result = state.get("_cached_context_result")
            
            # Only retrieve if not cached
            if not retrieval_result or not context_result:
                retrieval_result = self.retrieval_agent.retrieve_documents(state["rewritten_query"])
                context_result = self.context_selector_agent.select_context(
                    retrieval_result.documents, state["rewritten_query"]
                )
            
            # Handle Bengali queries by translating to English for better processing
            query_for_reasoning = self._translate_bengali_query(state["query"])
            
            reasoning_result = self.reasoning_agent.generate_answer(
                query_for_reasoning,
                context_result.selected_context,
                retrieval_result.sources
            )
            
            state["generated_answer"] = reasoning_result.answer
            state["answer_confidence"] = float(reasoning_result.confidence)
            state["processing_steps"] = state.get("processing_steps", []) + [f"Answer generated (confidence: {reasoning_result.confidence:.2f})"]
            
            return state
        except Exception as e:
            state["error"] = f"Answer generation failed: {str(e)}"
            return state
    
    def _check_grounding_node(self, state: SimpleChatbotState) -> SimpleChatbotState:
        """Check grounding"""
        print("[SIMPLE LANGGRAPH] Check grounding node")
        try:
            # Use cached results
            context_result = state.get("_cached_context_result")
            
            # Only retrieve if not cached
            if not context_result:
                retrieval_result = self.retrieval_agent.retrieve_documents(state["rewritten_query"])
                context_result = self.context_selector_agent.select_context(
                    retrieval_result.documents, state["rewritten_query"]
                )
            
            grounding_result = self.grounding_checker_agent.check_grounding(
                state["generated_answer"], context_result.selected_context
            )
            
            state["grounding_score"] = float(grounding_result.grounding_score)
            state["has_unsupported_claims"] = bool(len(grounding_result.unsupported_claims) > 0)
            
            # Increment regeneration count if regeneration is needed
            if grounding_result.unsupported_claims and state.get("regeneration_count", 0) < 1:
                state["regeneration_count"] = state.get("regeneration_count", 0) + 1
                print(f"[SIMPLE LANGGRAPH] Grounding check failed, will regenerate (count: {state['regeneration_count']})")
            
            state["processing_steps"] = state.get("processing_steps", []) + [f"Grounding check: {grounding_result.grounding_score:.2f}"]
            return state
        except Exception as e:
            state["error"] = f"Grounding check failed: {str(e)}"
            return state
    
    def _check_relevance_node(self, state: SimpleChatbotState) -> SimpleChatbotState:
        """Check relevance"""
        print("[SIMPLE LANGGRAPH] Check relevance node")
        try:
            # Use translated query for relevance check if it's Bengali
            query_for_relevance = self._translate_bengali_query(state["rewritten_query"])
            
            # Adjust relevance threshold for Bengali queries
            relevance_threshold = 0.3
            if any('\u0980' <= char <= '\u09FF' for char in state["query"]):
                relevance_threshold = 0.15  # Lower threshold for Bengali queries
            
            # Create a custom relevance agent with adjusted threshold
            from backend.agents.relevance_agent import RelevanceAgent
            custom_relevance_agent = RelevanceAgent(relevance_threshold=relevance_threshold)
            
            # Get document count from state
            documents_retrieved = len(state.get("retrieved_docs", []))
            
            relevance_result = custom_relevance_agent.validate_relevance(
                query_for_relevance, state["generated_answer"], documents_retrieved
            )
            state["relevance_score"] = float(relevance_result.relevance_score)
            state["is_relevant"] = bool(relevance_result.is_relevant)
            state["processing_steps"] = state.get("processing_steps", []) + [f"Relevance check: {relevance_result.relevance_score:.2f}"]
            return state
        except Exception as e:
            state["error"] = f"Relevance check failed: {str(e)}"
            return state
    
    def _generate_fallback_node(self, state: SimpleChatbotState) -> SimpleChatbotState:
        """Generate fallback response"""
        print("[SIMPLE LANGGRAPH] Generate fallback node")
        try:
            # Determine fallback type
            if state["intent_type"] == 'greeting':
                fallback_type = 'greeting'
            elif not state.get("is_in_domain", True):
                fallback_type = 'domain_blocked'
            elif not state.get("is_relevant", True):
                fallback_type = 'low_relevance'
            else:
                fallback_type = 'error'
            
            fallback_result = self.fallback_agent.generate_fallback_response(fallback_type)
            state["response"] = fallback_result.response
            state["confidence"] = float(fallback_result.confidence)
            state["response_type"] = fallback_type
            state["processing_steps"] = state.get("processing_steps", []) + [f"Fallback response: {fallback_type}"]
            return state
        except Exception as e:
            state["error"] = f"Fallback response failed: {str(e)}"
            return state
    
    def _finalize_result_node(self, state: SimpleChatbotState) -> SimpleChatbotState:
        """Finalize result"""
        print("[SIMPLE LANGGRAPH] Finalize result node")
        try:
            # Determine final response if not already set
            if not state.get("response"):
                # Enhance response with conversational elements
                raw_response = state["generated_answer"]
                
                # Apply conversational enhancement
                enhanced_response = self.conversational_agent.enhance_response(
                    raw_response,
                    state["query"],
                    {
                        'conversation_context': state.get("conversation_context"),
                        'response_strategy': state.get("response_strategy", 'balanced_response'),
                        'personalization': state.get("personalization", {})
                    }
                )
                
                state["response"] = enhanced_response
                state["confidence"] = float(state["answer_confidence"])
                
                # Set is_relevant to True if we have a generated answer
                if state.get("generated_answer"):
                    state["is_relevant"] = True
            
            # Set response_type if not already set (for document-based responses)
            if not state.get("response_type") and state.get("generated_answer"):
                # Check if documents were retrieved
                documents_retrieved = len(state.get("retrieved_docs", []))
                is_relevant = state.get("is_relevant", False)
                confidence = float(state["answer_confidence"])
                
                # FIXED: If documents were retrieved, always set as success
                # The relevance check is too strict and filters out valid responses
                if documents_retrieved > 0:
                    state["response_type"] = "success"
                    print(f"[SIMPLE LANGGRAPH] Documents retrieved ({documents_retrieved}) - setting success")
                else:
                    state["response_type"] = "no_documents"
                    print(f"[SIMPLE LANGGRAPH] No documents retrieved - setting no_documents")
                
                print(f"[SIMPLE LANGGRAPH] Final response_type: {state['response_type']}")
            
            return state
        except Exception as e:
            state["error"] = f"Logging failed: {str(e)}"
            return state
    
    # Routing functions
    def _route_after_intent(self, state: SimpleChatbotState) -> str:
        """Route after intent detection"""
        intent_type = state["intent_type"]
        
        if intent_type == 'greeting':
            return "greeting"
        elif intent_type in ['job_inquiry', 'small_talk', 'general_inquiry']:
            return "query"
        else:
            return "query"  # Default to query processing
    
    def _route_after_domain(self, state: SimpleChatbotState) -> str:
        """Route after domain guard"""
        if state["is_in_domain"]:
            return "allowed"
        else:
            return "blocked"
    
    def _route_after_grounding(self, state: SimpleChatbotState) -> str:
        """Route after grounding check"""
        # Only regenerate once to avoid infinite recursion
        if state.get("has_unsupported_claims", False) and state.get("regeneration_count", 0) < 1:
            return "regenerate"
        else:
            return "proceed"
    
    def _route_after_relevance(self, state: SimpleChatbotState) -> str:
        """Route after relevance check"""
        if state.get("is_relevant", False):
            return "relevant"
        else:
            return "not_relevant"
    
    def process_query(self, query: str, session_id: Optional[str] = None) -> Dict[str, Any]:
        """Process query using simplified LangGraph workflow"""
        try:
            if not session_id:
                session_id = str(uuid.uuid4())
            
            # Initialize state
            initial_state = SimpleChatbotState(
                query=query,
                session_id=session_id,
                processed_text="",
                intent_type="",
                rewritten_query="",
                is_in_domain=False,
                retrieved_docs=[],
                retrieved_scores=[],
                selected_context=[],
                plan_type="",
                needs_tools=False,
                generated_answer="",
                answer_confidence=0.0,
                grounding_score=0.0,
                has_unsupported_claims=False,
                relevance_score=0.0,
                is_relevant=False,
                regeneration_count=0,
                response="",
                confidence=0.0,
                sources=[],
                response_type="",
                processing_steps=[],
                error=""
            )
            
            # Run the workflow with recursion limit
            config = {
                "configurable": {"thread_id": session_id},
                "recursion_limit": 50  # Increase recursion limit
            }
            final_state = self.workflow.invoke(initial_state, config=config)
            
            # Return structured response with sources and scores
            sources_with_scores = []
            retrieved_docs = final_state.get("retrieved_docs", [])
            retrieved_scores = final_state.get("retrieved_scores", [])
            response_type = final_state.get("response_type", "")
            
            # Only include sources if:
            # 1. Documents were actually retrieved
            # 2. Response is not a fallback type (no_documents, error, etc.)
            if retrieved_docs and response_type not in ["no_documents", "error", "low_relevance", "domain_blocked", "greeting"]:
                # Combine sources with their scores
                for i, doc in enumerate(retrieved_docs):
                    score = retrieved_scores[i] if i < len(retrieved_scores) else 0.0
                    sources_with_scores.append({
                        "source": doc,
                        "score": score
                    })
            
            return {
                "answer": final_state["response"],
                "confidence": final_state["confidence"],
                "sources": sources_with_scores,
                "response_type": final_state["response_type"],
                "intent_type": final_state.get("intent_type", "unknown"),
                "session_id": final_state["session_id"],
                "processing_steps": final_state.get("processing_steps", []),
                "is_relevant": final_state.get("is_relevant", True),
                "error": final_state.get("error")
            }
            
        except Exception as e:
            return {
                "answer": f"Error processing query: {str(e)}",
                "confidence": 0.0,
                "sources": [],
                "response_type": "error",
                "session_id": session_id or str(uuid.uuid4()),
                "processing_steps": [f"Error: {str(e)}"],
                "is_relevant": False,
                "error": str(e)
            }
    
    def get_conversation_context(self, session_id: str) -> str:
        """Get conversation context for a session"""
        try:
            memory_result = self.memory_agent.get_conversation_context(session_id)
            return memory_result.conversation_context
        except Exception as e:
            print(f"[SIMPLE LANGGRAPH WORKFLOW ERROR] Failed to get context: {str(e)}")
            return ""
    
    def provide_intent_feedback(self, query: str, correct_intent: str, was_correct: bool):
        """Provide feedback to improve intent detection"""
        if hasattr(self.intent_agent, 'provide_feedback'):
            self.intent_agent.provide_feedback(query, correct_intent, was_correct)
    
    def get_intent_pattern_stats(self) -> Dict[str, Any]:
        """Get statistics about learned intent patterns"""
        if hasattr(self.intent_agent, 'get_pattern_stats'):
            return self.intent_agent.get_pattern_stats()
        return {}
    
    def save_intent_patterns(self, filepath: str):
        """Save learned intent patterns"""
        if hasattr(self.intent_agent, 'save_patterns'):
            self.intent_agent.save_patterns(filepath)
    
    def load_intent_patterns(self, filepath: str):
        """Load learned intent patterns"""
        if hasattr(self.intent_agent, 'load_patterns'):
            self.intent_agent.load_patterns(filepath)

    def clear_conversation(self, session_id: str) -> bool:
        """Clear conversation for a session"""
        try:
            return self.memory_agent.clear_conversation(session_id)
        except Exception as e:
            print(f"[SIMPLE LANGGRAPH WORKFLOW ERROR] Failed to clear conversation: {str(e)}")
            return False
    
    def _translate_bengali_query(self, query: str) -> str:
        """Dynamically translate any Bengali query to English for better processing"""
        # Check if query contains Bengali text
        if not any('\u0980' <= char <= '\u09FF' for char in query):
            return query
        
        try:
            # Use LLM to dynamically translate any Bengali text to English
            from langchain_groq import ChatGroq
            from config.settings import Settings
            
            settings = Settings()
            llm = ChatGroq(
                model=settings.LLM_MODEL,
                temperature=0.1,  # Low temperature for consistent translation
                groq_api_key=settings.GROQ_API_KEY
            )
            
            # Dynamic translation prompt that handles any Bengali input
            translation_prompt = f"""
            You are a professional Bengali to English translator specializing in business and HR terminology.
            
            Translate the following Bengali text to English. Handle any type of Bengali input dynamically:
            - Business terms, HR policies, salary inquiries
            - Reference numbers, document codes, dates
            - Questions, requests, or statements
            - Mixed Bengali-English text
            - Numbers, dates, and special characters
            
            IMPORTANT TRANSLATION RULES:
            1. Translate all Bengali text to clear, professional English
            2. Keep reference numbers in standard format (e.g., KFIL/HR/2016/880)
            3. Preserve numbers, dates, and codes exactly as they appear
            4. Translate business terms accurately (বেতন = salary, ভাতা = allowance, etc.)
            5. Maintain the original intent and meaning
            6. If mixed with English, keep English parts unchanged
            
            Bengali text: {query}
            
            English translation:"""
            
            response = llm.invoke(translation_prompt)
            translated_query = response.content.strip()
            
            # Clean up the translation (remove any extra formatting)
            translated_query = translated_query.replace('**English translation:**', '').strip()
            translated_query = translated_query.replace('English translation:', '').strip()
            
            # Extract reference numbers if present
            import re
            ref_patterns = [
                r'KFIL/HR/\d+/\d+',  # KFIL/HR/2016/880
                r'KFG/HR/\d+/\d+',   # KFG/HR/2016/880
                r'KML/HR/\d+/\d+',   # KML/HR/2016/880
            ]
            
            for pattern in ref_patterns:
                ref_match = re.search(pattern, translated_query)
                if ref_match:
                    translated_query = ref_match.group()
                    break
            
            return translated_query
            
        except Exception as e:
            # Enhanced fallback with more comprehensive Bengali terms
            bengali_to_english = {
                # Reference and document terms
                'স্বারক নং': 'reference number',
                'রেফারেন্স নং': 'reference number',
                'দলিল নং': 'document number',
                'ফাইল নং': 'file number',
                
                # Company and department terms
                'কেএফআইএল': 'KFIL',
                'কেএফজি': 'KFG',
                'কেএমএল': 'KML',
                'কাজী ফার্মস': 'Kazi Farms',
                'এইচ আর': 'HR',
                'মানব সম্পদ': 'HR',
                'অ্যাকাউন্টস': 'Accounts',
                'ফিন্যান্স': 'Finance',
                
                # Salary and benefits terms
                'বেতন': 'salary',
                'মূল বেতন': 'basic salary',
                'ভাতা': 'allowance',
                'বাড়ি ভাতা': 'house allowance',
                'খাবার ভাতা': 'food allowance',
                'যাতায়াত ভাতা': 'transport allowance',
                'চিকিৎসা ভাতা': 'medical allowance',
                'অবস্থান ভাতা': 'location allowance',
                'বোনাস': 'bonus',
                'উৎপাদন বোনাস': 'production bonus',
                'বেতন বৃদ্ধি': 'salary increment',
                'পদোন্নতি': 'promotion',
                
                # Leave and attendance terms
                'ছুটি': 'leave',
                'অবকাশ': 'leave',
                'সিক লিভ': 'sick leave',
                'কাজের ছুটি': 'casual leave',
                'বার্ষিক ছুটি': 'annual leave',
                'মাতৃত্ব ছুটি': 'maternity leave',
                'পিতৃত্ব ছুটি': 'paternity leave',
                'উপস্থিতি': 'attendance',
                'অপারেটিং টাইম': 'overtime',
                'অতিরিক্ত কাজ': 'overtime',
                
                # Employment terms
                'নিয়োগ': 'appointment',
                'চাকরি': 'job',
                'পদ': 'position',
                'পদবী': 'designation',
                'বিভাগ': 'department',
                'শাখা': 'branch',
                'অফিস': 'office',
                'কারখানা': 'factory',
                'খামার': 'farm',
                'হ্যাচারি': 'hatchery',
                
                # Contact and location terms
                'ঠিকানা': 'address',
                'ফোন': 'phone',
                'টেলিফোন': 'telephone',
                'ইমেইল': 'email',
                'ওয়েবসাইট': 'website',
                'অবস্থান': 'location',
                'জায়গা': 'place',
                
                # General business terms
                'নীতি': 'policy',
                'নিয়ম': 'rule',
                'প্রক্রিয়া': 'process',
                'পদ্ধতি': 'procedure',
                'সুপারিশ': 'recommendation',
                'অনুমোদন': 'approval',
                'প্রস্তাব': 'proposal',
                'রিপোর্ট': 'report',
                'বিবরণ': 'details',
                'তথ্য': 'information',
                
                # Time and date terms
                'তারিখ': 'date',
                'সময়': 'time',
                'দিন': 'day',
                'মাস': 'month',
                'বছর': 'year',
                'সকাল': 'morning',
                'দুপুর': 'afternoon',
                'সন্ধ্যা': 'evening',
                'রাত': 'night',
                
                # Numbers (Bengali to English)
                '০': '0', '১': '1', '২': '2', '৩': '3', '৪': '4',
                '৫': '5', '৬': '6', '৭': '7', '৮': '8', '৯': '9'
            }
            
            translated_query = query
            for bengali, english in bengali_to_english.items():
                translated_query = translated_query.replace(bengali, english)
            
            return translated_query
