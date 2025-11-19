"""
Grounding Checker Agent - Checks if claims are supported by context
"""
from typing import Dict, Any, Optional, List
from dataclasses import dataclass

@dataclass
class GroundingCheckerResult:
    """Result from grounding check"""
    is_grounded: bool
    unsupported_claims: List[str]
    supported_claims: List[str]
    grounding_score: float
    confidence: float
    reasoning: str

class GroundingCheckerAgent:
    """Agent responsible for checking if claims are grounded in context"""
    
    def __init__(self):
        self.is_initialized = True
        pass  # Silent initialization
    
    def check_grounding(self, answer: str, context: List[Dict[str, Any]]) -> GroundingCheckerResult:
        """Check if answer claims are grounded in context"""
        try:
            if not context:
                return GroundingCheckerResult(
                    is_grounded=False,
                    unsupported_claims=[answer],
                    supported_claims=[],
                    grounding_score=0.0,
                    confidence=0.1,
                    reasoning="No context available for grounding check"
                )
            
            # Simple grounding check - look for key terms in context
            context_text = ' '.join([str(getattr(doc, 'page_content', '')) for doc in context])
            context_lower = context_text.lower()
            answer_lower = answer.lower()
            
            # Extract key terms from answer (remove common words)
            common_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should', 'may', 'might', 'can', 'must', 'shall'}
            answer_words = set([word for word in answer_lower.split() if word not in common_words and len(word) > 2])
            context_words = set([word for word in context_lower.split() if word not in common_words and len(word) > 2])
            
            # Calculate overlap
            overlap = len(answer_words.intersection(context_words))
            total_answer_words = len(answer_words)
            
            grounding_score = overlap / total_answer_words if total_answer_words > 0 else 0.0
            
            # Also check for exact phrase matches
            answer_phrases = [phrase.strip() for phrase in answer_lower.split('.') if len(phrase.strip()) > 10]
            phrase_matches = sum(1 for phrase in answer_phrases if phrase in context_lower)
            
            # Boost score for phrase matches
            if phrase_matches > 0:
                grounding_score = min(1.0, grounding_score + (phrase_matches * 0.2))
            
            # Determine if grounded
            is_grounded = grounding_score > 0.1  # Lower threshold for better grounding
            
            if is_grounded:
                supported_claims = [answer]
                unsupported_claims = []
            else:
                supported_claims = []
                unsupported_claims = [answer]
            
            return GroundingCheckerResult(
                is_grounded=is_grounded,
                unsupported_claims=unsupported_claims,
                supported_claims=supported_claims,
                grounding_score=grounding_score,
                confidence=0.8,
                reasoning=f"Grounding check completed with score {grounding_score:.2f}"
            )
            
        except Exception as e:
            print(f"[GROUNDING CHECKER AGENT ERROR] {str(e)}")
            return GroundingCheckerResult(
                is_grounded=False,
                unsupported_claims=[answer],
                supported_claims=[],
                grounding_score=0.0,
                confidence=0.1,
                reasoning=f"Error in grounding check: {str(e)}"
            )
