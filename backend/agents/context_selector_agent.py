"""
Context Selector Agent - Selects relevant context from retrieved documents
"""
from typing import Dict, Any, Optional, List
from dataclasses import dataclass

@dataclass
class ContextSelectorResult:
    """Result from context selection"""
    selected_context: List[Dict[str, Any]]
    context_count: int
    selection_method: str
    confidence: float
    reasoning: str

class ContextSelectorAgent:
    """Agent responsible for selecting relevant context from retrieved documents"""
    
    def __init__(self):
        self.is_initialized = True
        self.max_context_length = 4000  # Maximum context length
        pass  # Silent initialization
    
    def select_context(self, documents: List[Dict[str, Any]], query: str) -> ContextSelectorResult:
        """Select most relevant context from documents with enhanced strategy"""
        try:
            if not documents:
                return ContextSelectorResult(
                    selected_context=[],
                    context_count=0,
                    selection_method="no_documents",
                    confidence=0.0,
                    reasoning="No documents available for context selection"
                )
            
            # Enhanced context selection strategy
            query_lower = query.lower()
            query_keywords = set(query_lower.split())
            
            # Score documents based on keyword overlap and content relevance
            scored_docs = []
            for doc in documents:
                content = str(getattr(doc, 'page_content', '')).lower()
                
                # Calculate keyword overlap score
                content_words = set(content.split())
                overlap_score = len(query_keywords.intersection(content_words)) / max(len(query_keywords), 1)
                
                # Boost score for HR-related terms
                hr_terms = {'salary', 'allowance', 'policy', 'employee', 'staff', 'leave', 'bonus', 'increment', 'deduction', 'claims', 'bill', 'expense', 'reimbursement'}
                hr_boost = len(hr_terms.intersection(content_words)) * 0.1
                
                # Boost score for specific domain terms
                domain_terms = {'kazi', 'farms', 'hr', 'human', 'resources', 'department', 'office', 'management'}
                domain_boost = len(domain_terms.intersection(content_words)) * 0.05
                
                final_score = overlap_score + hr_boost + domain_boost
                scored_docs.append((doc, final_score))
            
            # Sort by score (descending)
            scored_docs.sort(key=lambda x: x[1], reverse=True)
            
            # Select top documents with focus on concise, relevant information
            selected_context = []
            total_length = 0
            max_docs = 3  # Reduced to 3 for concise responses
            max_context_length = 1500  # Much shorter context for focused answers
            
            for doc, score in scored_docs[:max_docs]:
                content_length = len(str(getattr(doc, 'page_content', '')))
                if total_length + content_length <= max_context_length:
                    selected_context.append(doc)
                    total_length += content_length
                elif not selected_context:  # Always include at least one document
                    # Truncate the document if it's too long
                    content = str(getattr(doc, 'page_content', ''))
                    if len(content) > max_context_length:
                        # Keep the first part which usually has the most important info
                        truncated_content = content[:max_context_length] + "..."
                        doc.page_content = truncated_content
                    selected_context.append(doc)
                    break
            
            # Always prefer fewer, more focused documents
            selected_context = selected_context[:2]  # Limit to top 2 most relevant
            
            print(f"[CONTEXT SELECTOR] Selected {len(selected_context)} documents from {len(documents)} available")
            print(f"[CONTEXT SELECTOR] Total context length: {total_length} characters")
            
            return ContextSelectorResult(
                selected_context=selected_context,
                context_count=len(selected_context),
                selection_method="enhanced_scoring",
                confidence=0.9,
                reasoning=f"Selected {len(selected_context)} most relevant documents using keyword overlap and domain scoring"
            )
            
        except Exception as e:
            print(f"[CONTEXT SELECTOR AGENT ERROR] {str(e)}")
            return ContextSelectorResult(
                selected_context=[],
                context_count=0,
                selection_method="error",
                confidence=0.1,
                reasoning=f"Error in context selection: {str(e)}"
            )
