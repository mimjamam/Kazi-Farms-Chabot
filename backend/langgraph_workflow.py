import os
import json
import random
from typing import Dict, List, Any, Optional, TypedDict
from dataclasses import dataclass
import asyncio

from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.memory import MemorySaver

from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

from .query_agent import QueryValidationAgent, QueryAnalysis
from .chat_service import VectorStoreService, ChatService
from .similarity_agent import SimilarityComparisonAgent
from .funny_fallback_agent import FunnyFallbackAgent
from .personal_info_guard import PersonalInfoGuard
from core.models.simple_query_matcher import SimpleQueryMatcher, MatchResult

class ChatbotState(TypedDict):
    user_query: str
    messages: List[Any]
    query_analysis: Optional[QueryAnalysis]
    is_complete_query: bool
    followup_suggestion: str
    search_results: List[Any]
    search_confidence: float
    match_result: Optional[MatchResult]
    llm_response: str
    source_documents: List[Any]
    is_valid_response: bool
    validation_reason: str
    final_response: str
    confidence_score: float
    error_message: Optional[str]
    should_continue: bool

class ChatbotStateWithSimilarity(TypedDict):
    user_query: str
    messages: List[Any]
    query_analysis: Optional[QueryAnalysis]
    is_complete_query: bool
    followup_suggestion: str
    search_results: List[Any]
    search_confidence: float
    match_result: Optional[MatchResult]
    llm_response: str
    source_documents: List[Any]
    is_valid_response: bool
    validation_reason: str
    final_response: str
    confidence_score: float
    similarity_metrics: Optional[Dict[str, Any]]
    similarity_report: Optional[str]
    error_message: Optional[str]
    should_continue: bool

class LangGraphWorkflow:
    def __init__(self):
        self.query_agent = QueryValidationAgent()
        self.query_matcher = SimpleQueryMatcher()
        self.vector_service = VectorStoreService()
        self.similarity_agent = SimilarityComparisonAgent()
        self.funny_fallback_agent = FunnyFallbackAgent()
        self.personal_info_guard = PersonalInfoGuard()
        
        self.llm = ChatGroq(
            model_name="llama-3.1-70b-versatile",
            temperature=0.0,
            groq_api_key=os.getenv("GROQ_API_KEY")
        )
        
        self.app = None
        self.app_with_similarity = None
        self.app_final = None
        
        self._build_workflows()
    
    def _build_workflows(self):
        self._build_basic_workflow()
        self._build_similarity_workflow()
        self._build_final_workflow()
    
    def _build_basic_workflow(self):
        workflow = StateGraph(ChatbotState)
        
        workflow.add_node("analyze_query", self.analyze_query_node)
        workflow.add_node("vector_search", self.vector_search_node)
        workflow.add_node("query_matching", self.query_matching_node)
        workflow.add_node("response_generation", self.response_generation_node)
        workflow.add_node("response_validation", self.response_validation_node)
        workflow.add_node("finalize_response", self.finalize_response_node)
        workflow.add_node("error_handling", self.error_handling_node)
        workflow.add_node("end_invalid_response", self.end_invalid_response_node)
        
        workflow.set_entry_point("analyze_query")
        
        workflow.add_conditional_edges("analyze_query", self.route_after_analysis, {"vector_search": "vector_search", "error_handling": "error_handling"})
        workflow.add_conditional_edges("vector_search", self.route_after_search, {"query_matching": "query_matching", "error_handling": "error_handling"})
        workflow.add_conditional_edges("query_matching", self.route_after_matching, {"response_generation": "response_generation", "error_handling": "error_handling"})
        workflow.add_conditional_edges("response_generation", self.route_after_generation, {"response_validation": "response_validation", "error_handling": "error_handling"})
        workflow.add_conditional_edges("response_validation", self.route_after_validation, {
            "finalize_response": "finalize_response", 
            "error_handling": "error_handling",
            "end_invalid_response": "end_invalid_response"
        })
        
        workflow.add_edge("finalize_response", END)
        workflow.add_edge("error_handling", END)
        workflow.add_edge("end_invalid_response", END)
        
        self.app = workflow.compile()
    
    def _build_similarity_workflow(self):
        workflow_with_similarity = StateGraph(ChatbotStateWithSimilarity)
        
        workflow_with_similarity.add_node("analyze_query", self.analyze_query_node)
        workflow_with_similarity.add_node("vector_search", self.vector_search_node)
        workflow_with_similarity.add_node("query_matching", self.query_matching_node)
        workflow_with_similarity.add_node("response_generation", self.response_generation_node)
        workflow_with_similarity.add_node("response_validation", self.response_validation_node)
        workflow_with_similarity.add_node("finalize_response", self.finalize_response_node)
        workflow_with_similarity.add_node("similarity_comparison", self.similarity_comparison_node)
        workflow_with_similarity.add_node("error_handling", self.error_handling_node)
        workflow_with_similarity.add_node("end_invalid_response", self.end_invalid_response_node)
        
        workflow_with_similarity.set_entry_point("analyze_query")
        
        workflow_with_similarity.add_conditional_edges("analyze_query", self.route_after_analysis, {"vector_search": "vector_search", "error_handling": "error_handling"})
        workflow_with_similarity.add_conditional_edges("vector_search", self.route_after_search, {"query_matching": "query_matching", "error_handling": "error_handling"})
        workflow_with_similarity.add_conditional_edges("query_matching", self.route_after_matching, {"response_generation": "response_generation", "error_handling": "error_handling"})
        workflow_with_similarity.add_conditional_edges("response_generation", self.route_after_generation, {"response_validation": "response_validation", "error_handling": "error_handling"})
        workflow_with_similarity.add_conditional_edges("response_validation", self.route_after_validation, {
            "finalize_response": "finalize_response", 
            "error_handling": "error_handling",
            "end_invalid_response": "end_invalid_response"
        })
        workflow_with_similarity.add_conditional_edges("finalize_response", self.route_after_validation, {"similarity_comparison": "similarity_comparison", "error_handling": "error_handling"})
        
        workflow_with_similarity.add_edge("similarity_comparison", END)
        workflow_with_similarity.add_edge("error_handling", END)
        workflow_with_similarity.add_edge("end_invalid_response", END)
        
        self.app_with_similarity = workflow_with_similarity.compile()
    
    def _build_final_workflow(self):
        workflow_final = StateGraph(ChatbotStateWithSimilarity)
        
        workflow_final.add_node("analyze_query", self.analyze_query_node)
        workflow_final.add_node("vector_search", self.vector_search_node)
        workflow_final.add_node("query_matching", self.query_matching_node)
        workflow_final.add_node("response_generation", self.response_generation_node_with_fallback)
        workflow_final.add_node("response_validation", self.response_validation_node)
        workflow_final.add_node("finalize_response", self.finalize_response_node_with_encouragement)
        workflow_final.add_node("similarity_comparison", self.similarity_comparison_node)
        workflow_final.add_node("error_handling", self.error_handling_node)
        workflow_final.add_node("end_invalid_response", self.end_invalid_response_node)
        
        workflow_final.set_entry_point("analyze_query")
        
        workflow_final.add_conditional_edges("analyze_query", self.route_after_analysis, {"vector_search": "vector_search", "error_handling": "error_handling"})
        workflow_final.add_conditional_edges("vector_search", self.route_after_search, {"query_matching": "query_matching", "error_handling": "error_handling"})
        workflow_final.add_conditional_edges("query_matching", self.route_after_matching, {"response_generation": "response_generation", "error_handling": "error_handling"})
        workflow_final.add_conditional_edges("response_generation", self.route_after_generation, {"response_validation": "response_validation", "error_handling": "error_handling"})
        workflow_final.add_conditional_edges("response_validation", self.route_after_validation, {
            "finalize_response": "finalize_response", 
            "error_handling": "error_handling",
            "end_invalid_response": "end_invalid_response"
        })
        workflow_final.add_conditional_edges("finalize_response", self.route_after_validation, {"similarity_comparison": "similarity_comparison", "error_handling": "error_handling"})
        
        workflow_final.add_edge("similarity_comparison", END)
        workflow_final.add_edge("error_handling", END)
        workflow_final.add_edge("end_invalid_response", END)
        
        self.app_final = workflow_final.compile()
    
    def analyze_query_node(self, state: ChatbotState) -> ChatbotState:
        try:
            # Log user input to terminal
            print(f"\n[USER INPUT] {state['user_query']}")
            
            # Check for personal information queries first
            personal_info_result = self.personal_info_guard.handle_personal_info_query(state['user_query'])
            if personal_info_result["is_personal_query"]:
                print(f"[QUERY TYPE] {personal_info_result['query_type']}")
                print(f"[RESPONSE] {personal_info_result['response']}")
                state['final_response'] = personal_info_result["response"]
                state['confidence_score'] = 0.0
                state['should_continue'] = False
                state['blocked'] = True
                return state
            
            query_analysis = self.query_agent.analyze_query(state['user_query'])
            is_complete, followup_suggestion = self.query_agent.validate_query_completeness(state['user_query'])
            
            # Log query analysis to terminal
            print(f"[QUERY ANALYSIS] Type: {query_analysis.query_type}, Complete: {is_complete}")
            if query_analysis.extracted_info:
                print(f"[EXTRACTED INFO] {query_analysis.extracted_info}")
            
            state['query_analysis'] = query_analysis
            state['is_complete_query'] = is_complete
            state['followup_suggestion'] = followup_suggestion
        except Exception as e:
            state['error_message'] = f"Query analysis failed: {str(e)}"
            state['should_continue'] = False
        return state
    
    def vector_search_node(self, state: ChatbotState) -> ChatbotState:
        try:
            if self.vector_service.vectorstore is None:
                self.vector_service.load_vectorstore()
            
            search_results = self.vector_service.hybrid_search(state['user_query'])
            avg_confidence = sum([hit[1] for hit in search_results]) / len(search_results) if search_results else 0.0
            
            # Log search results to terminal
            print(f"[SEARCH RESULTS] Found {len(search_results)} results")
            if search_results:
                print(f"[AVERAGE CONFIDENCE] {avg_confidence:.2f}")
                for i, hit in enumerate(search_results[:3]):  # Log top 3 results
                    print(f"[RESULT {i+1}] Confidence: {hit[1]:.2f}, Source: {getattr(hit[0], 'metadata', {}).get('source', 'Unknown')}")
            
            state['search_results'] = search_results
            state['search_confidence'] = avg_confidence
        except Exception as e:
            state['error_message'] = f"Vector search failed: {str(e)}"
            state['should_continue'] = False
        return state
    
    def query_matching_node(self, state: ChatbotState) -> ChatbotState:
        try:
            content_list = [hit[0].page_content for hit in state['search_results']] if state['search_results'] else []
            metadata_list = [hit[0].metadata for hit in state['search_results']] if state['search_results'] else []
            
            match_result = self.query_matcher.match_query_to_content(
                state['user_query'], 
                content_list, 
                metadata_list
            )
            state['match_result'] = match_result
        except Exception as e:
            state['error_message'] = f"Query matching failed: {str(e)}"
            state['should_continue'] = False
        return state
    
    def response_generation_node(self, state: ChatbotState) -> ChatbotState:
        try:
            # Always use the highest confidence result, even if below threshold
            highest_confidence = state['search_results'][0][1] if state['search_results'] else 0
            print(f"[CONFIDENCE] Average: {state['search_confidence']:.2f}, Highest: {highest_confidence:.2f}")
            
            if highest_confidence < 25:
                print(f"[LOW CONFIDENCE] Highest confidence ({highest_confidence:.2f}) below threshold (25)")
                print(f"[USING HIGHEST] Proceeding with highest confidence result anyway")
            
            context = "\n\n".join([hit[0].page_content for hit in state['search_results']]) if state['search_results'] else ""
            enhanced_prompt = self.query_matcher.generate_enhanced_prompt(state['user_query'], context, "")
            response = self.llm.invoke(enhanced_prompt)
            
            # Log LLM response to terminal
            print(f"[LLM RESPONSE] {response.content}")
            
            state['llm_response'] = response.content
            state['source_documents'] = [hit[0] for hit in state['search_results']] if state['search_results'] else []
            state['search_confidence'] = highest_confidence  # Update confidence to highest
        except Exception as e:
            state['error_message'] = f"Response generation failed: {str(e)}"
            state['should_continue'] = False
        return state
    
    def response_generation_node_with_fallback(self, state: ChatbotState) -> ChatbotState:
        try:
            # Always use the highest confidence result, even if below threshold
            highest_confidence = state['search_results'][0][1] if state['search_results'] else 0
            print(f"[CONFIDENCE] Average: {state['search_confidence']:.2f}, Highest: {highest_confidence:.2f}")
            
            if highest_confidence < 25:
                print(f"[LOW CONFIDENCE] Highest confidence ({highest_confidence:.2f}) below threshold (25)")
                print(f"[USING HIGHEST] Proceeding with highest confidence result anyway")
            
            if not state['search_results']:
                fallback_response = self.funny_fallback_agent.analyze_query_context(
                    state['user_query'],
                    [],
                    0.0
                )
                state['llm_response'] = fallback_response
                state['source_documents'] = []
                return state
            
            context = "\n\n".join([hit[0].page_content for hit in state['search_results']]) if state['search_results'] else ""
            enhanced_prompt = self.query_matcher.generate_enhanced_prompt(state['user_query'], context, "")
            response = self.llm.invoke(enhanced_prompt)
            
            llm_response_lower = response.content.lower()
            generic_phrases = [
                "i don't know",
                "not available",
                "no information",
                "cannot provide",
                "not found",
                "i don't have specific information about this in our database"
            ]
            
            if any(phrase in llm_response_lower for phrase in generic_phrases):
                fallback_response = self.funny_fallback_agent.analyze_query_context(
                    state['user_query'],
                    state['search_results'],
                    state['search_confidence']
                )
                state['llm_response'] = fallback_response
            else:
                state['llm_response'] = response.content
            
            state['source_documents'] = [hit[0] for hit in state['search_results']] if state['search_results'] else []
            state['search_confidence'] = highest_confidence  # Update confidence to highest
            
        except Exception as e:
            fallback_response = self.funny_fallback_agent.analyze_query_context(
                state['user_query'],
                state.get('search_results', []),
                state.get('search_confidence', 0.0)
            )
            state['llm_response'] = fallback_response
            state['source_documents'] = []
        return state
    
    def response_validation_node(self, state: ChatbotState) -> ChatbotState:
        try:
            if state['match_result']:
                context = "\n\n".join([hit[0].page_content for hit in state['search_results']]) if state['search_results'] else ""
                is_valid = self.query_matcher.validate_answer_relevance(
                    state['user_query'],
                    state['llm_response'],
                    context
                )
                state['is_valid_response'] = is_valid
                state['validation_reason'] = "Response validated against query and context" if is_valid else "Response not relevant to query"
            else:
                state['is_valid_response'] = True
                state['validation_reason'] = "No match result available for validation"
        except Exception as e:
            state['error_message'] = f"Response validation failed: {str(e)}"
            state['should_continue'] = False
        return state
    
    def finalize_response_node(self, state: ChatbotState) -> ChatbotState:
        try:
            response = state['llm_response']
            
            # Log followup suggestions to terminal instead of showing to user
            if not state['is_complete_query'] and state['followup_suggestion']:
                print(f"[DEBUG] Followup suggestion: {state['followup_suggestion']}")
            
            # Log metadata to terminal instead of showing to user
            if state['source_documents']:
                sources_info = "\n".join(
                    [f"• {getattr(doc, 'metadata', {})}" for doc in state['source_documents']]
                )
                print(f"\n[DEBUG] Sources: {sources_info}")
            
            confidence = state['search_confidence']
            if confidence > 0:
                print(f"[DEBUG] Confidence: {confidence:.1f}%")
            
            state['final_response'] = response
            state['confidence_score'] = confidence
        except Exception as e:
            state['error_message'] = f"Response finalization failed: {str(e)}"
            state['should_continue'] = False
        return state
    
    def finalize_response_node_with_encouragement(self, state: ChatbotState) -> ChatbotState:
        try:
            response = state['llm_response']
            
            # Log followup suggestions to terminal instead of showing to user
            if not state['is_complete_query'] and state['followup_suggestion']:
                print(f"[DEBUG] Followup suggestion: {state['followup_suggestion']}")
            
            # Log metadata to terminal instead of showing to user
            if state['source_documents']:
                sources_info = "\n".join(
                    [f"• {getattr(doc, 'metadata', {})}" for doc in state['source_documents']]
                )
                print(f"\n[DEBUG] Sources: {sources_info}")
            
            confidence = state['search_confidence']
            if confidence > 0:
                print(f"[DEBUG] Confidence: {confidence:.1f}%")
            
            if confidence < 25 or not state['source_documents']:
                encouragement = self.funny_fallback_agent.get_encouragement_message()
                response += f"\n\n{encouragement}"
            
            state['final_response'] = response
            state['confidence_score'] = confidence
        except Exception as e:
            state['error_message'] = f"Response finalization failed: {str(e)}"
            state['should_continue'] = False
        return state
    
    def similarity_comparison_node(self, state: ChatbotStateWithSimilarity) -> ChatbotStateWithSimilarity:
        try:
            query = state['user_query']
            response = state['final_response']
            
            similarity_metrics = self.similarity_agent.calculate_comprehensive_similarity(query, response)
            similarity_report = self.similarity_agent.generate_similarity_report(query, response, similarity_metrics)
            
            state['similarity_metrics'] = similarity_metrics
            state['similarity_report'] = similarity_report
        except Exception as e:
            state['error_message'] = f"Similarity comparison failed: {str(e)}"
            state['should_continue'] = False
        return state
    
    def error_handling_node(self, state: ChatbotState) -> ChatbotState:
        error_response = f"I encountered an issue while processing your request: {state.get('error_message', 'Unknown error')}. Please try again or contact Kazifarm directly for assistance."
        state['final_response'] = error_response
        state['confidence_score'] = 0.0
        state['should_continue'] = False
        return state
    
    def end_invalid_response_node(self, state: ChatbotState) -> ChatbotState:
        invalid_response = "I apologize, but I couldn't generate a relevant response to your query. Please try rephrasing your question or contact our support team for assistance."
        state['final_response'] = invalid_response
        state['confidence_score'] = 0.0
        state['should_continue'] = False
        return state
    
    def route_after_analysis(self, state: ChatbotState) -> str:
        if not state.get('should_continue', True):
            return "error_handling"
        if state.get('blocked', False):
            return END
        return "vector_search"
    
    def route_after_search(self, state: ChatbotState) -> str:
        return "error_handling" if not state.get('should_continue', True) else "query_matching"
    
    def route_after_matching(self, state: ChatbotState) -> str:
        return "error_handling" if not state.get('should_continue', True) else "response_generation"
    
    def route_after_generation(self, state: ChatbotState) -> str:
        return "error_handling" if not state.get('should_continue', True) else "response_validation"
    
    def route_after_validation(self, state: ChatbotState) -> str:
        if not state.get('should_continue', True):
            return "error_handling"
        elif not state.get('is_valid_response', True):
            return "end_invalid_response"
        else:
            return "finalize_response"
    
    def process_query(self, query: str, workflow_type: str = "final") -> Dict[str, Any]:
        if workflow_type == "basic":
            app = self.app
        elif workflow_type == "similarity":
            app = self.app_with_similarity
        else:
            app = self.app_final
        
        initial_state = {
            "user_query": query,
            "messages": [],
            "query_analysis": None,
            "is_complete_query": True,
            "followup_suggestion": "",
            "search_results": [],
            "search_confidence": 0.0,
            "match_result": None,
            "llm_response": "",
            "source_documents": [],
            "is_valid_response": True,
            "validation_reason": "",
            "final_response": "",
            "confidence_score": 0.0,
            "error_message": None,
            "should_continue": True
        }
        
        if workflow_type in ["similarity", "final"]:
            initial_state["similarity_metrics"] = None
            initial_state["similarity_report"] = None
        
        result = app.invoke(initial_state)
        return result
    
    def get_workflow_graph(self, workflow_type: str = "final"):
        if workflow_type == "basic":
            return self.app.get_graph()
        elif workflow_type == "similarity":
            return self.app_with_similarity.get_graph()
        else:
            return self.app_final.get_graph()
