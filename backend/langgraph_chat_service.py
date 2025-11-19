from typing import Dict, Any, Optional, List
import logging
import traceback
from backend.agents.simple_langgraph_workflow import SimpleLangGraphWorkflow

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class LangGraphChatService:
    def __init__(self) -> None:
        try:
            self.workflow = SimpleLangGraphWorkflow()
            self.is_initialized = True
            logger.info("[SIMPLE LANGGRAPH CHAT SERVICE] Initialized successfully with simplified LangGraph workflow")
        except Exception as e:
            self.is_initialized = False
            logger.error(f"[LANGGRAPH CHAT SERVICE ERROR] Initialization failed: {e}")
            traceback.print_exc()

    def get_answer_with_sources(
        self,
        query: str,
        conversation_context: str = "",
        session_id: Optional[str] = None
    ) -> str:
        try:
            response = self.process_query(query, conversation_context, session_id)
            self._display_structured_output(query, response)
            return str(response.get("answer", "No answer available."))
        except Exception as e:
            logger.error(f"[LANGGRAPH CHAT SERVICE ERROR] Failed to get answer: {e}")
            traceback.print_exc()
            return f"Error processing your request: {e}"

    def process_query(
        self,
        query: str,
        conversation_context: str = "",
        session_id: Optional[str] = None
    ) -> Dict[str, Any]:
        if not self.is_initialized:
            return self._error_response("Service not initialized", session_id)
        try:
            response = self.workflow.process_query(query, session_id)
            return response or self._error_response("Empty response from workflow", session_id)
        except Exception as e:
            logger.error(f"[LANGGRAPH CHAT SERVICE ERROR] Workflow failed: {e}")
            traceback.print_exc()
            return self._error_response(str(e), session_id)

    def get_conversation_context(self, session_id: str) -> str:
        try:
            return self.workflow.get_conversation_context(session_id)
        except Exception as e:
            logger.error(f"[LANGGRAPH CHAT SERVICE ERROR] Failed to get context: {e}")
            traceback.print_exc()
            return ""

    def clear_conversation(self, session_id: str) -> bool:
        try:
            return self.workflow.clear_conversation(session_id)
        except Exception as e:
            logger.error(f"[LANGGRAPH CHAT SERVICE ERROR] Failed to clear conversation: {e}")
            traceback.print_exc()
            return False

    def get_statistics(self) -> Dict[str, Any]:
        try:
            return {
                "is_initialized": self.is_initialized,
                "architecture": "simplified_langgraph_based",
                "workflow_type": "SimpleLangGraphWorkflow",
                "features": [
                    "Simplified StateGraph workflow",
                    "Serializable state management",
                    "Conditional routing",
                    "Memory checkpointing",
                    "Error handling",
                    "Multi-agent coordination"
                ]
            }
        except Exception as e:
            logger.error(f"[LANGGRAPH CHAT SERVICE ERROR] Failed to get statistics: {e}")
            traceback.print_exc()
            return {"is_initialized": False, "error": str(e)}

    def get_workflow_graph(self) -> str:
        try:
            return """
            graph TD
                A[Input Processing] --> B[Intent Detection]
                B --> C{Intent Type?}
                C -->|Greeting| D[Fallback Response]
                C -->|Query| E[Query Rewriting]
                E --> F[Domain Guard]
                F --> G{Domain Check}
                G -->|Blocked| D
                G -->|Allowed| H[Document Retrieval]
                H --> I[Context Selection]
                I --> J[Planning]
                J --> K{Tools Needed?}
                K -->|Yes| L[Tool Routing]
                K -->|No| M[Answer Generation]
                L --> N[Tool Execution]
                N --> M
                M --> O[Grounding Check]
                O --> P{Grounded?}
                P -->|No| M
                P -->|Yes| Q[Relevance Check]
                Q --> R{Relevant?}
                R -->|Yes| S[Memory Update]
                R -->|No| D
                S --> T[Logging]
                D --> T
                T --> U[END]
            """
        except Exception as e:
            logger.error(f"[LANGGRAPH CHAT SERVICE ERROR] Failed to generate graph: {e}")
            traceback.print_exc()
            return "graph TD\n    A[Error] --> B[Graph generation failed]"

    def _display_structured_output(self, query: str, response: Dict[str, Any]) -> None:
        print("\n" + "=" * 80)
        print("KAZI FARMS CHATBOT - SIMPLIFIED LANGGRAPH WORKFLOW RESPONSE")
        print("=" * 80)
        print(f"\nUSER QUERY:\n   {query}")
        response_type = response.get("response_type", "unknown").upper()
        confidence = response.get("confidence", 0.0)
        sources: List[Any] = response.get("sources", [])
        print(f"\nRESPONSE STATUS:")
        print(f"   Type: {response_type}")
        print(f"   Confidence: {confidence:.2f}")
        print(f"   Sources: {len(sources)} documents")
        if sources:
            print(f"\nSOURCE DOCUMENTS:")
            for i, source in enumerate(sources, 1):
                if isinstance(source, dict):
                    doc_name = source.get("id", f"Document {i}")
                    print(f"   {i}. {doc_name}")
                else:
                    print(f"   {i}. {source}")
        print(f"\nANSWER:\n" + "-" * 60)
        print(response.get("answer", "No answer available."))
        print("-" * 60)
        if response.get("processing_steps"):
            print(f"\nPROCESSING STEPS:")
            for step in response.get("processing_steps", []):
                print(f"   ✓ {step}")
        if response.get("error"):
            print(f"\nERROR:\n   {response['error']}")
        print("\n" + "=" * 80)
        print("SIMPLIFIED LANGGRAPH WORKFLOW COMPLETE")
        print("=" * 80 + "\n")

    def _error_response(self, message: str, session_id: Optional[str]) -> Dict[str, Any]:
        return {
            "answer": f"Error processing your request: {message}",
            "confidence": 0.0,
            "sources": [],
            "response_type": "error",
            "session_id": session_id or "unknown",
            "processing_steps": [f"Error: {message}"],
            "error": message,
        }
