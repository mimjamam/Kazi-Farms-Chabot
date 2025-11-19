"""
ReasoningAgent - Generate answer using retrieved context
"""
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from config.settings import Settings

@dataclass
class ReasoningResult:
    """Result of reasoning/answer generation"""
    answer: str
    confidence: float
    sources_used: List[str]
    reasoning_steps: List[str]
    is_complete: bool

class ReasoningAgent:
    """Agent for generating answers using retrieved context"""
    
    def __init__(self):
        self.settings = Settings()
        
        # Initialize LLM
        self.llm = ChatGroq(
            model=self.settings.LLM_MODEL,
            temperature=self.settings.LLM_TEMPERATURE,
            groq_api_key=self.settings.GROQ_API_KEY
        )
        
        # Create prompt template
        self.prompt_template = PromptTemplate(
            input_variables=["context", "question"],
            template=self.settings.CUSTOM_PROMPT_TEMPLATE
        )
    
    def generate_answer(self, query: str, documents: List[Any], sources: List[str]) -> ReasoningResult:
        """Generate answer using retrieved documents"""
        try:
            if not documents:
                return ReasoningResult(
                    answer="I don't have specific information about that in our Kazi Farms database. Could you please rephrase your question or ask about salary structures, allowances, leave policies, or other HR topics?",
                    confidence=0.0,
                    sources_used=[],
                    reasoning_steps=["No documents retrieved"],
                    is_complete=False
                )
            
            # Check if query contains a reference number pattern
            import re
            ref_patterns = [
                r'KFIL/HR/\d+/\d+',
                r'KFG/HR/\d+/\d+',
                r'KML/HR/\d+/\d+',
                r'reference number',
                r'ref no',
                r'ref:'
            ]
            
            has_reference_query = any(re.search(pattern, query, re.IGNORECASE) for pattern in ref_patterns)
            
            # Prepare context from documents
            context = self._prepare_context(documents)
            
            # Check if the specific reference number exists in the context
            if has_reference_query:
                # Extract reference number from query
                ref_match = None
                for pattern in ref_patterns[:3]:  # Only check the specific reference patterns
                    match = re.search(pattern, query, re.IGNORECASE)
                    if match:
                        ref_match = match.group()
                        break
                
                if ref_match and ref_match not in context:
                    # Reference number not found in context, provide helpful response
                    return ReasoningResult(
                        answer=f"I couldn't find the specific reference number '{ref_match}' in our Kazi Farms database. However, I found similar documents that might be relevant. The reference number format suggests this is an HR document from 2016. You might want to check with the HR department directly for this specific reference, or ask about general HR policies, leave procedures, or salary structures from 2016.",
                        confidence=0.3,
                        sources_used=sources,
                        reasoning_steps=["Reference number not found in context", "Provided alternative suggestions"],
                        is_complete=True
                    )
            
            # Generate answer using LLM
            prompt = self.prompt_template.format(context=context, question=query)
            
            response = self.llm.invoke(prompt)
            answer = response.content.strip()
            
            # Calculate confidence based on document relevance
            confidence = self._calculate_confidence(documents, sources)
            
            # Extract reasoning steps
            reasoning_steps = self._extract_reasoning_steps(query, documents)
            
            # Removed verbose logging - summary shown in main endpoint
            
            return ReasoningResult(
                answer=answer,
                confidence=confidence,
                sources_used=sources,
                reasoning_steps=reasoning_steps,
                is_complete=True
            )
            
        except Exception as e:
            print(f"[REASONING AGENT ERROR] {str(e)}")
            return ReasoningResult(
                answer=f"I encountered an error while processing your request: {str(e)}",
                confidence=0.0,
                sources_used=sources if sources else [],
                reasoning_steps=[f"Error: {str(e)}"],
                is_complete=False
            )
    
    def _prepare_context(self, documents: List[Any]) -> str:
        """Prepare context from retrieved documents"""
        context_parts = []
        
        for i, doc in enumerate(documents):
            if hasattr(doc, 'page_content'):
                content = doc.page_content.strip()
                if content:
                    context_parts.append(f"Document {i+1}:\n{content}\n")
        
        return "\n".join(context_parts)
    
    def _calculate_confidence(self, documents: List[Any], sources: List[str]) -> float:
        """Calculate confidence in the generated answer"""
        if not documents:
            return 0.0
        
        # Base confidence from document count
        base_confidence = min(0.8, len(documents) * 0.1)
        
        # Boost confidence if we have multiple sources
        if len(set(sources)) > 1:
            base_confidence += 0.1
        
        # Boost confidence if documents are substantial
        total_content_length = sum(len(getattr(doc, 'page_content', '')) for doc in documents)
        if total_content_length > 1000:
            base_confidence += 0.1
        
        return min(1.0, base_confidence)
    
    def _extract_reasoning_steps(self, query: str, documents: List[Any]) -> List[str]:
        """Extract reasoning steps for the answer"""
        steps = []
        
        steps.append(f"Analyzed query: '{query}'")
        steps.append(f"Retrieved {len(documents)} relevant documents")
        
        if documents:
            steps.append("Synthesized information from retrieved documents")
            steps.append("Generated answer based on Kazi Farms context")
        
        return steps
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get reasoning agent statistics"""
        return {
            "llm_model": self.settings.LLM_MODEL,
            "temperature": self.settings.LLM_TEMPERATURE,
            "status": "initialized"
        }
