"""
LangGraph-based workflow for Kazi Farms Chatbot
"""
import uuid
from typing import Dict, Any, Optional, List, TypedDict, Annotated
from dataclasses import dataclass
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

# Import all agents
from .input_agent import InputAgent, ProcessedInput
from .intent_agent import IntentAgent, IntentResult
from .query_rewrite_agent import QueryRewriteAgent, QueryRewriteResult
from .domain_guard_agent import DynamicDomainGuardAgent, DomainGuardResult
from .retrieval_agent import RetrievalAgent, RetrievalResult
from .context_selector_agent import ContextSelectorAgent, ContextSelectorResult
from .planner_agent import PlannerAgent, PlannerResult
from .tool_router_agent import ToolRouterAgent, ToolRouterResult
from .tool_executor import ToolExecutor, ToolExecutorResult
from .reasoning_agent import ReasoningAgent, ReasoningResult
from .grounding_checker_agent import GroundingCheckerAgent, GroundingCheckerResult
from .relevance_agent import RelevanceAgent, RelevanceResult
from .memory_agent import MemoryAgent, MemoryResult
from .fallback_agent import FallbackAgent, FallbackResult
from .logging_agent import LoggingAgent, LoggingResult

class ChatbotState(TypedDict):
    """State schema for the LangGraph workflow"""
    # Input
    query: str
    session_id: str
    conversation_context: str
    
    # Processing results (serializable data only)
    processed_input_text: Optional[str]
    intent_type: Optional[str]
    rewritten_query: Optional[str]
    is_in_domain: Optional[bool]
    retrieved_documents: Optional[List[str]]
    selected_context: Optional[List[str]]
    plan_type: Optional[str]
    requires_tools: bool
    tool_results: Optional[List[str]]
    generated_answer: Optional[str]
    answer_confidence: Optional[float]
    grounding_score: Optional[float]
    unsupported_claims: Optional[List[str]]
    relevance_score: Optional[float]
    is_relevant: bool
    memory_updated: bool
    fallback_response: Optional[str]
    fallback_confidence: Optional[float]
    
    # Control flow
    is_grounded: bool
    regeneration_count: int
    
    # Final output
    response: str
    confidence: float
    sources: List[str]
    response_type: str
    processing_steps: List[str]
    
    # Error handling
    error: Optional[str]

class LangGraphWorkflow:
    """LangGraph-based workflow for the chatbot"""
    
    def __init__(self):
        self.workflow = None
        self.memory = MemorySaver()
        self._initialize_agents()
        self._build_graph()
    
    def _initialize_agents(self):
        """Initialize all agents"""
        print("[LANGGRAPH WORKFLOW] Initializing agents...")
        
        self.input_agent = InputAgent()
        self.intent_agent = IntentAgent()
        self.query_rewrite_agent = QueryRewriteAgent()
        self.domain_guard_agent = DynamicDomainGuardAgent()
        self.retrieval_agent = RetrievalAgent()
        self.context_selector_agent = ContextSelectorAgent()
        self.planner_agent = PlannerAgent()
        self.tool_router_agent = ToolRouterAgent()
        self.tool_executor = ToolExecutor()
        self.reasoning_agent = ReasoningAgent()
        self.grounding_checker_agent = GroundingCheckerAgent()
        self.relevance_agent = RelevanceAgent()
        self.memory_agent = MemoryAgent()
        self.fallback_agent = FallbackAgent()
        self.logging_agent = LoggingAgent()
        
        print("[LANGGRAPH WORKFLOW] All agents initialized")
    
    def _build_graph(self):
        """Build the LangGraph StateGraph"""
        print("[LANGGRAPH WORKFLOW] Building StateGraph...")
        
        # Create the graph
        workflow = StateGraph(ChatbotState)
        
        # Add nodes
        workflow.add_node("input_processing", self._input_processing_node)
        workflow.add_node("intent_detection", self._intent_detection_node)
        workflow.add_node("query_rewriting", self._query_rewriting_node)
        workflow.add_node("domain_guard", self._domain_guard_node)
        workflow.add_node("document_retrieval", self._document_retrieval_node)
        workflow.add_node("context_selection", self._context_selection_node)
        workflow.add_node("planning", self._planning_node)
        workflow.add_node("tool_routing", self._tool_routing_node)
        workflow.add_node("tool_execution", self._tool_execution_node)
        workflow.add_node("answer_generation", self._answer_generation_node)
        workflow.add_node("grounding_check", self._grounding_check_node)
        workflow.add_node("relevance_check", self._relevance_check_node)
        workflow.add_node("memory_update", self._memory_update_node)
        workflow.add_node("fallback_response", self._fallback_response_node)
        workflow.add_node("logging", self._logging_node)
        
        # Set entry point
        workflow.set_entry_point("input_processing")
        
        # Add edges
        workflow.add_edge("input_processing", "intent_detection")
        
        # Intent-based routing
        workflow.add_conditional_edges(
            "intent_detection",
            self._route_after_intent,
            {
                "greeting": "fallback_response",
                "query": "query_rewriting"
            }
        )
        
        workflow.add_edge("query_rewriting", "domain_guard")
        
        # Domain guard routing
        workflow.add_conditional_edges(
            "domain_guard",
            self._route_after_domain_guard,
            {
                "blocked": "fallback_response",
                "allowed": "document_retrieval"
            }
        )
        
        workflow.add_edge("document_retrieval", "context_selection")
        workflow.add_edge("context_selection", "planning")
        
        # Planning routing
        workflow.add_conditional_edges(
            "planning",
            self._route_after_planning,
            {
                "tools_needed": "tool_routing",
                "no_tools": "answer_generation"
            }
        )
        
        workflow.add_edge("tool_routing", "tool_execution")
        workflow.add_edge("tool_execution", "answer_generation")
        workflow.add_edge("answer_generation", "grounding_check")
        
        # Grounding check routing
        workflow.add_conditional_edges(
            "grounding_check",
            self._route_after_grounding,
            {
                "regenerate": "answer_generation",
                "proceed": "relevance_check"
            }
        )
        
        # Relevance check routing
        workflow.add_conditional_edges(
            "relevance_check",
            self._route_after_relevance,
            {
                "relevant": "memory_update",
                "not_relevant": "fallback_response"
            }
        )
        
        workflow.add_edge("memory_update", "logging")
        workflow.add_edge("fallback_response", "logging")
        workflow.add_edge("logging", END)
        
        # Compile the graph
        self.workflow = workflow.compile(checkpointer=self.memory)
        print("[LANGGRAPH WORKFLOW] StateGraph built and compiled successfully")
    
    # Node functions
    def _input_processing_node(self, state: ChatbotState) -> ChatbotState:
        """Process user input"""
        print("[LANGGRAPH] Input processing node")
        try:
            processed_input = self.input_agent.process_input(state["query"])
            state["processed_input_text"] = processed_input.normalized_text
            state["processing_steps"] = state.get("processing_steps", []) + ["Input processed"]
            return state
        except Exception as e:
            state["error"] = f"Input processing failed: {str(e)}"
            return state
    
    def _intent_detection_node(self, state: ChatbotState) -> ChatbotState:
        """Detect user intent"""
        print("[LANGGRAPH] Intent detection node")
        try:
            intent_result = self.intent_agent.detect_intent(state["processed_input_text"])
            state["intent_type"] = intent_result.intent_type
            state["processing_steps"] = state.get("processing_steps", []) + [f"Intent detected: {intent_result.intent_type}"]
            return state
        except Exception as e:
            state["error"] = f"Intent detection failed: {str(e)}"
            return state
    
    def _query_rewriting_node(self, state: ChatbotState) -> ChatbotState:
        """Rewrite query for better retrieval"""
        print("[LANGGRAPH] Query rewriting node")
        try:
            rewrite_result = self.query_rewrite_agent.rewrite_query(
                state["processed_input_text"],
                context={'conversation_context': state.get("conversation_context", "")}
            )
            state["rewritten_query"] = rewrite_result.rewritten_query
            state["processing_steps"] = state.get("processing_steps", []) + ["Query rewritten"]
            return state
        except Exception as e:
            state["error"] = f"Query rewriting failed: {str(e)}"
            return state
    
    def _domain_guard_node(self, state: ChatbotState) -> ChatbotState:
        """Check domain guard"""
        print("[LANGGRAPH] Domain guard node")
        try:
            domain_result = self.domain_guard_agent.check_domain(state["rewritten_query"])
            state["is_in_domain"] = domain_result.is_in_domain
            state["processing_steps"] = state.get("processing_steps", []) + [f"Domain check: {'ALLOW' if domain_result.is_in_domain else 'BLOCK'}"]
            return state
        except Exception as e:
            state["error"] = f"Domain guard failed: {str(e)}"
            return state
    
    def _document_retrieval_node(self, state: ChatbotState) -> ChatbotState:
        """Retrieve documents"""
        print("[LANGGRAPH] Document retrieval node")
        try:
            rewrite_result = state["rewrite_result"]
            retrieval_result = self.retrieval_agent.retrieve_documents(rewrite_result.rewritten_query)
            state["retrieval_result"] = retrieval_result
            state["processing_steps"] = state.get("processing_steps", []) + [f"Retrieved {len(retrieval_result.documents)} documents"]
            return state
        except Exception as e:
            state["error"] = f"Document retrieval failed: {str(e)}"
            return state
    
    def _context_selection_node(self, state: ChatbotState) -> ChatbotState:
        """Select context"""
        print("[LANGGRAPH] Context selection node")
        try:
            retrieval_result = state["retrieval_result"]
            rewrite_result = state["rewrite_result"]
            context_result = self.context_selector_agent.select_context(
                retrieval_result.documents, rewrite_result.rewritten_query
            )
            state["context_result"] = context_result
            state["processing_steps"] = state.get("processing_steps", []) + [f"Selected {context_result.context_count} context documents"]
            return state
        except Exception as e:
            state["error"] = f"Context selection failed: {str(e)}"
            return state
    
    def _planning_node(self, state: ChatbotState) -> ChatbotState:
        """Create execution plan"""
        print("[LANGGRAPH] Planning node")
        try:
            rewrite_result = state["rewrite_result"]
            context_result = state["context_result"]
            planner_result = self.planner_agent.create_plan(
                rewrite_result.rewritten_query, context_result.selected_context
            )
            state["planner_result"] = planner_result
            state["requires_tools"] = planner_result.requires_tools
            state["processing_steps"] = state.get("processing_steps", []) + [f"Plan created: {planner_result.plan_type}"]
            return state
        except Exception as e:
            state["error"] = f"Planning failed: {str(e)}"
            return state
    
    def _tool_routing_node(self, state: ChatbotState) -> ChatbotState:
        """Route tools"""
        print("[LANGGRAPH] Tool routing node")
        try:
            planner_result = state["planner_result"]
            tool_router_result = self.tool_router_agent.route_tools(planner_result.tool_calls)
            state["tool_router_result"] = tool_router_result
            state["processing_steps"] = state.get("processing_steps", []) + ["Tools routed"]
            return state
        except Exception as e:
            state["error"] = f"Tool routing failed: {str(e)}"
            return state
    
    def _tool_execution_node(self, state: ChatbotState) -> ChatbotState:
        """Execute tools"""
        print("[LANGGRAPH] Tool execution node")
        try:
            tool_router_result = state["tool_router_result"]
            tool_executor_result = self.tool_executor.execute_tools(tool_router_result.routed_tools)
            state["tool_executor_result"] = tool_executor_result
            state["processing_steps"] = state.get("processing_steps", []) + [f"Executed {len(tool_executor_result.executed_tools)} tools"]
            return state
        except Exception as e:
            state["error"] = f"Tool execution failed: {str(e)}"
            return state
    
    def _answer_generation_node(self, state: ChatbotState) -> ChatbotState:
        """Generate answer"""
        print("[LANGGRAPH] Answer generation node")
        try:
            rewrite_result = state["rewrite_result"]
            context_result = state["context_result"]
            retrieval_result = state["retrieval_result"]
            
            # Check if this is a regeneration
            regeneration_count = state.get("regeneration_count", 0)
            if regeneration_count > 0:
                print(f"[LANGGRAPH] Regenerating answer (attempt {regeneration_count + 1})")
                state["processing_steps"] = state.get("processing_steps", []) + [f"Answer regenerated (attempt {regeneration_count + 1})"]
            else:
                state["processing_steps"] = state.get("processing_steps", []) + ["Answer generated"]
            
            reasoning_result = self.reasoning_agent.generate_answer(
                rewrite_result.rewritten_query,
                context_result.selected_context,
                retrieval_result.sources
            )
            state["reasoning_result"] = reasoning_result
            state["processing_steps"] = state.get("processing_steps", []) + [f"Answer generated (confidence: {reasoning_result.confidence:.2f})"]
            return state
        except Exception as e:
            state["error"] = f"Answer generation failed: {str(e)}"
            return state
    
    def _grounding_check_node(self, state: ChatbotState) -> ChatbotState:
        """Check grounding"""
        print("[LANGGRAPH] Grounding check node")
        try:
            reasoning_result = state["reasoning_result"]
            context_result = state["context_result"]
            
            grounding_result = self.grounding_checker_agent.check_grounding(
                reasoning_result.answer, context_result.selected_context
            )
            state["grounding_result"] = grounding_result
            state["is_grounded"] = len(grounding_result.unsupported_claims) == 0
            
            # Increment regeneration count if regeneration is needed
            if grounding_result.unsupported_claims and state.get("regeneration_count", 0) < 1:
                state["regeneration_count"] = state.get("regeneration_count", 0) + 1
                print(f"[LANGGRAPH] Grounding check failed, will regenerate (count: {state['regeneration_count']})")
            
            state["processing_steps"] = state.get("processing_steps", []) + [f"Grounding check: {grounding_result.grounding_score:.2f}"]
            return state
        except Exception as e:
            state["error"] = f"Grounding check failed: {str(e)}"
            return state
    
    def _relevance_check_node(self, state: ChatbotState) -> ChatbotState:
        """Check relevance"""
        print("[LANGGRAPH] Relevance check node")
        try:
            rewrite_result = state["rewrite_result"]
            reasoning_result = state["reasoning_result"]
            
            # Get document count from retrieval result
            documents_retrieved = len(retrieval_result.documents) if retrieval_result else 0
            
            relevance_result = self.relevance_agent.validate_relevance(
                rewrite_result.rewritten_query, reasoning_result.answer, documents_retrieved
            )
            state["relevance_result"] = relevance_result
            state["is_relevant"] = relevance_result.is_relevant
            state["processing_steps"] = state.get("processing_steps", []) + [f"Relevance check: {relevance_result.relevance_score:.2f}"]
            return state
        except Exception as e:
            state["error"] = f"Relevance check failed: {str(e)}"
            return state
    
    def _memory_update_node(self, state: ChatbotState) -> ChatbotState:
        """Update memory"""
        print("[LANGGRAPH] Memory update node")
        try:
            reasoning_result = state["reasoning_result"]
            memory_result = self.memory_agent.update_conversation(
                state["session_id"], state["query"], reasoning_result.answer
            )
            state["memory_result"] = memory_result
            state["processing_steps"] = state.get("processing_steps", []) + ["Memory updated"]
            return state
        except Exception as e:
            state["error"] = f"Memory update failed: {str(e)}"
            return state
    
    def _fallback_response_node(self, state: ChatbotState) -> ChatbotState:
        """Generate fallback response"""
        print("[LANGGRAPH] Fallback response node")
        try:
            intent_result = state.get("intent_result")
            domain_result = state.get("domain_result")
            relevance_result = state.get("relevance_result")
            
            # Determine fallback type
            if intent_result and intent_result.intent_type == 'greeting':
                fallback_type = 'greeting'
            elif domain_result and not domain_result.is_in_domain:
                fallback_type = 'domain_blocked'
            elif relevance_result and not relevance_result.is_relevant:
                fallback_type = 'low_relevance'
            else:
                fallback_type = 'error'
            
            fallback_result = self.fallback_agent.generate_fallback_response(fallback_type)
            state["fallback_result"] = fallback_result
            state["response"] = fallback_result.response
            state["confidence"] = fallback_result.confidence
            state["response_type"] = fallback_type
            state["processing_steps"] = state.get("processing_steps", []) + [f"Fallback response: {fallback_type}"]
            return state
        except Exception as e:
            state["error"] = f"Fallback response failed: {str(e)}"
            return state
    
    def _logging_node(self, state: ChatbotState) -> ChatbotState:
        """Log final result"""
        print("[LANGGRAPH] Logging node")
        try:
            # Determine final response
            if state.get("reasoning_result"):
                final_response = state["reasoning_result"].answer
                state["response"] = final_response
                state["confidence"] = state["reasoning_result"].confidence
                state["sources"] = state["retrieval_result"].sources
                state["response_type"] = "success"
            elif state.get("fallback_result"):
                final_response = state["fallback_result"].response
                state["response"] = final_response
                state["confidence"] = state["fallback_result"].confidence
                state["sources"] = []
                state["response_type"] = state.get("response_type", "fallback")
            
            # Log the result
            logging_result = self.logging_agent.log_final_result(
                state["query"], 
                state["response"], 
                state["session_id"], 
                state.get("processing_steps", [])
            )
            state["logging_result"] = logging_result
            
            return state
        except Exception as e:
            state["error"] = f"Logging failed: {str(e)}"
            return state
    
    # Routing functions
    def _route_after_intent(self, state: ChatbotState) -> str:
        """Route after intent detection"""
        if state["intent_type"] == 'greeting':
            return "greeting"
        else:
            return "query"
    
    def _route_after_domain_guard(self, state: ChatbotState) -> str:
        """Route after domain guard"""
        if state["is_in_domain"]:
            return "allowed"
        else:
            return "blocked"
    
    def _route_after_planning(self, state: ChatbotState) -> str:
        """Route after planning"""
        if state.get("requires_tools", False):
            return "tools_needed"
        else:
            return "no_tools"
    
    def _route_after_grounding(self, state: ChatbotState) -> str:
        """Route after grounding check"""
        # Only regenerate once to avoid infinite recursion
        if state.get("unsupported_claims") and state.get("regeneration_count", 0) < 1:
            return "regenerate"
        else:
            return "proceed"
    
    def _route_after_relevance(self, state: ChatbotState) -> str:
        """Route after relevance check"""
        if state.get("is_relevant", False):
            return "relevant"
        else:
            return "not_relevant"
    
    def process_query(self, query: str, session_id: Optional[str] = None) -> Dict[str, Any]:
        """Process query using LangGraph workflow"""
        try:
            if not session_id:
                session_id = str(uuid.uuid4())
            
            # Initialize state
            initial_state = ChatbotState(
                query=query,
                session_id=session_id,
                conversation_context="",
                processed_input_text=None,
                intent_type=None,
                rewritten_query=None,
                is_in_domain=None,
                retrieved_documents=None,
                selected_context=None,
                plan_type=None,
                requires_tools=False,
                tool_results=None,
                generated_answer=None,
                answer_confidence=None,
                grounding_score=None,
                unsupported_claims=None,
                relevance_score=None,
                is_relevant=False,
                memory_updated=False,
                fallback_response=None,
                fallback_confidence=None,
                is_grounded=False,
                regeneration_count=0,
                response="",
                confidence=0.0,
                sources=[],
                response_type="",
                processing_steps=[],
                error=None
            )
            
            # Run the workflow with recursion limit
            config = {
                "configurable": {"thread_id": session_id},
                "recursion_limit": 50  # Increase recursion limit
            }
            final_state = self.workflow.invoke(initial_state, config=config)
            
            # Return structured response
            return {
                "answer": final_state["response"],
                "confidence": final_state["confidence"],
                "sources": final_state["sources"],
                "response_type": final_state["response_type"],
                "session_id": final_state["session_id"],
                "processing_steps": final_state.get("processing_steps", []),
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
                "error": str(e)
            }
    
    def get_conversation_context(self, session_id: str) -> str:
        """Get conversation context for a session"""
        try:
            memory_result = self.memory_agent.get_conversation_context(session_id)
            return memory_result.conversation_context
        except Exception as e:
            print(f"[LANGGRAPH WORKFLOW ERROR] Failed to get context: {str(e)}")
            return ""
    
    def clear_conversation(self, session_id: str) -> bool:
        """Clear conversation for a session"""
        try:
            return self.memory_agent.clear_conversation(session_id)
        except Exception as e:
            print(f"[LANGGRAPH WORKFLOW ERROR] Failed to clear conversation: {str(e)}")
            return False
