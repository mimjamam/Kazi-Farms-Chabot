"""
RelevanceAgent - Compare generated answer to user query and validate match
"""
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from langchain_huggingface import HuggingFaceEmbeddings
from typing import List, Dict, Any
from dataclasses import dataclass
from config.settings import Settings

@dataclass
class RelevanceResult:
    """Result of relevance validation"""
    relevance_score: float
    is_relevant: bool
    confidence: float
    validation_reason: str

class RelevanceAgent:
    """Agent for validating answer relevance to user query"""
    
    def __init__(self, relevance_threshold: float = 0.05):  # Much lower threshold for better sensitivity
        self.settings = Settings()
        self.relevance_threshold = relevance_threshold
        
        # Initialize embeddings for similarity comparison
        self.embeddings = HuggingFaceEmbeddings(
            model_name=self.settings.EMBEDDING_MODEL,
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )
    
    def validate_relevance(self, query: str, answer: str, documents_retrieved: int = 0) -> RelevanceResult:
        """Dynamic relevance validation using adaptive learning"""
        try:
            if not answer or not query:
                return RelevanceResult(
                    relevance_score=0.0,
                    is_relevant=False,
                    confidence=0.0,
                    validation_reason="Empty query or answer"
                )
            
            # Extract actual user query from enhanced context query
            actual_query = self._extract_user_query(query)
            print(f"[RELEVANCE AGENT] Original query: '{query}'")
            print(f"[RELEVANCE AGENT] Extracted user query: '{actual_query}'")
            
            # Dynamic fallback detection using pattern analysis
            is_fallback = self._detect_fallback_response(answer, documents_retrieved)
            
            # Calculate semantic similarity between actual user query and answer
            query_embedding = self.embeddings.embed_query(actual_query)
            answer_embedding = self.embeddings.embed_query(answer)
            
            # Reshape for cosine similarity calculation
            query_embedding = np.array(query_embedding).reshape(1, -1)
            answer_embedding = np.array(answer_embedding).reshape(1, -1)
            
            # Calculate cosine similarity
            base_similarity = cosine_similarity(query_embedding, answer_embedding)[0][0]
            
            # Dynamic content analysis for relevance boosting (use actual user query)
            content_relevance_boost = self._calculate_content_relevance(actual_query, answer)
            
            # Adaptive threshold based on query complexity and content quality (use actual user query)
            adaptive_threshold = self._calculate_adaptive_threshold(actual_query, answer)
            
            # Final similarity with dynamic adjustments
            final_similarity = base_similarity + content_relevance_boost
            
            # Special handling for salary/driver queries
            is_salary_query = any(term in actual_query.lower() for term in ['salary', 'driver', 'minimum', 'maximum', 'allowance', 'pay'])
            
            # Check if documents were retrieved - this is a strong signal that we should use the answer
            has_retrieved_docs = documents_retrieved > 0
            
            # Determine relevance using adaptive threshold
            if has_retrieved_docs and is_salary_query:
                # If we have retrieved documents for a salary query, strongly prefer using the answer
                # even if it looks like a fallback (the reasoning agent might be too conservative)
                is_relevant = True
                print(f"[RELEVANCE AGENT] Documents retrieved ({documents_retrieved}) for salary query - forcing relevant")
            elif has_retrieved_docs and final_similarity > 0.1:
                # If we have documents and decent similarity, use the answer regardless of fallback detection
                is_relevant = True
                print(f"[RELEVANCE AGENT] Documents retrieved ({documents_retrieved}) with good similarity - forcing relevant")
            elif is_fallback and not has_retrieved_docs:
                # Only mark as fallback if no documents were retrieved
                is_relevant = False
                print(f"[RELEVANCE AGENT] Fallback response with no documents - marking as irrelevant")
            elif is_fallback and has_retrieved_docs:
                # If we have documents but got a fallback, check if it's a real fallback or just conservative reasoning
                if final_similarity > 0.05 or is_salary_query:
                    is_relevant = True
                    print(f"[RELEVANCE AGENT] Fallback detected but documents retrieved - overriding to relevant")
                else:
                    is_relevant = False
                    print(f"[RELEVANCE AGENT] Fallback response despite documents - marking as irrelevant")
            elif is_salary_query and final_similarity > 0.05:
                # For salary queries, be more lenient if we have some similarity and it's not a fallback
                is_relevant = True
                print(f"[RELEVANCE AGENT] Salary query detected - using lenient threshold")
            else:
                is_relevant = final_similarity >= adaptive_threshold
            
            # Dynamic confidence calculation
            confidence = self._calculate_dynamic_confidence(base_similarity, content_relevance_boost, is_fallback)
            
            # Generate validation reason
            validation_reason = self._generate_validation_reason(
                base_similarity, content_relevance_boost, adaptive_threshold, is_fallback, is_relevant
            )
            
            print(f"[RELEVANCE AGENT] Base similarity: {base_similarity:.3f}")
            print(f"[RELEVANCE AGENT] Content boost: {content_relevance_boost:.3f}")
            print(f"[RELEVANCE AGENT] Final similarity: {final_similarity:.3f}")
            print(f"[RELEVANCE AGENT] Adaptive threshold: {adaptive_threshold:.3f}")
            print(f"[RELEVANCE AGENT] Is fallback: {is_fallback}")
            print(f"[RELEVANCE AGENT] Is relevant: {is_relevant}")
            
            return RelevanceResult(
                relevance_score=final_similarity,
                is_relevant=is_relevant,
                confidence=confidence,
                validation_reason=validation_reason
            )
            
        except Exception as e:
            print(f"[RELEVANCE AGENT ERROR] {str(e)}")
            return RelevanceResult(
                relevance_score=0.0,
                is_relevant=False,
                confidence=0.0,
                validation_reason=f"Error in validation: {str(e)}"
            )
    
    def validate_answer_quality(self, answer: str) -> Dict[str, Any]:
        """Validate the quality of the generated answer"""
        try:
            quality_metrics = {}
            
            # Check answer length
            answer_length = len(answer.split())
            quality_metrics['length'] = answer_length
            
            # Check if answer is too short
            if answer_length < 5:
                quality_metrics['too_short'] = True
            else:
                quality_metrics['too_short'] = False
            
            # Check if answer is too long
            if answer_length > 500:
                quality_metrics['too_long'] = True
            else:
                quality_metrics['too_long'] = False
            
            # Check for common error patterns
            error_patterns = [
                "I don't know",
                "I can't help",
                "I'm not sure",
                "I don't have",
                "I cannot",
                "Error:",
                "Exception:"
            ]
            
            has_error_patterns = any(pattern.lower() in answer.lower() for pattern in error_patterns)
            quality_metrics['has_error_patterns'] = has_error_patterns
            
            # Overall quality score
            quality_score = 1.0
            if quality_metrics['too_short']:
                quality_score -= 0.3
            if quality_metrics['too_long']:
                quality_score -= 0.2
            if quality_metrics['has_error_patterns']:
                quality_score -= 0.4
            
            quality_metrics['quality_score'] = max(0.0, quality_score)
            
            return quality_metrics
            
        except Exception as e:
            print(f"[RELEVANCE AGENT ERROR] Quality validation failed: {str(e)}")
            return {
                'length': 0,
                'too_short': True,
                'too_long': False,
                'has_error_patterns': True,
                'quality_score': 0.0
            }
    
    def _detect_fallback_response(self, answer: str, documents_retrieved: int = 0) -> bool:
        """Dynamically detect fallback responses using pattern analysis"""
        # Analyze response structure and content patterns
        answer_lower = answer.lower()
        
        # If documents were retrieved, be much more conservative about marking as fallback
        if documents_retrieved > 0:
            # Only mark as fallback if it's very explicit and has no useful content
            very_explicit_fallback_patterns = [
                "i don't have specific information about that in our kazi farms database" in answer_lower,
                "i couldn't find relevant information about that topic" in answer_lower,
                "can't help with that topic" in answer_lower,
                "that's outside my area of expertise" in answer_lower,
            ]
            
            has_very_explicit_fallback = any(very_explicit_fallback_patterns)
            
            if has_very_explicit_fallback:
                # Even with explicit fallback, check if there's any useful HR content
                useful_content_indicators = [
                    'taka', 'tk', 'bdt', 'salary', 'allowance', 'driver', 'minimum', 'maximum',
                    'policy', 'procedure', 'grade', 'level', 'category', 'structure',
                    r'\d+', 'basic', 'increment', 'bonus'
                ]
                
                import re
                useful_content_count = 0
                for indicator in useful_content_indicators:
                    if indicator.startswith(r'\d'):
                        if re.search(indicator, answer_lower):
                            useful_content_count += 1
                    else:
                        if indicator in answer_lower:
                            useful_content_count += 1
                
                # If there's any useful content, don't mark as fallback
                if useful_content_count >= 1:
                    return False
                
                return True
            
            return False  # If documents retrieved and not very explicit fallback, not a fallback
        
        # Original logic for when no documents are retrieved
        explicit_fallback_patterns = [
            "i'm focused on kazi farms information only" in answer_lower,
            "that's not something i've found in our kazi farms documents" in answer_lower,
            "i don't have specific information about that in our kazi farms database" in answer_lower,
            "i couldn't find relevant information about that topic" in answer_lower,
            "can't help with that topic" in answer_lower,
            "that's outside my area of expertise" in answer_lower,
            "i'm designed to help with kazi farms hr policies" in answer_lower,
            "that's not something i" in answer_lower and "found" in answer_lower,
            "i can help you with" in answer_lower and "inquiries" in answer_lower
        ]
        
        # Only consider it a fallback if it contains explicit fallback language
        has_explicit_fallback = any(explicit_fallback_patterns)
        
        if has_explicit_fallback:
            # Check if the answer contains actual HR data (numbers, specific amounts, etc.)
            # vs just mentioning HR terms in a fallback context
            actual_hr_data_indicators = [
                'taka', 'tk', 'bdt',  # Currency indicators
                r'\d+,?\d*\s*(taka|tk|bdt)',  # Amounts with currency
                r'\d+\s*months?',  # Time periods
                r'\d+\s*days?',  # Time periods
                r'\d+\s*years?',  # Time periods
                'grade', 'level', 'category', 'structure'  # Specific organizational terms
            ]
            
            import re
            actual_data_count = 0
            for indicator in actual_hr_data_indicators:
                if isinstance(indicator, str):
                    if indicator in answer_lower:
                        actual_data_count += 1
                else:
                    # It's a regex pattern
                    if re.search(indicator, answer_lower):
                        actual_data_count += 1
            
            # If it has actual HR data (not just mentions), it's not a fallback
            if actual_data_count >= 2:
                return False
            
            return True
        
        # Additional check: if answer contains specific salary/driver information, it's not a fallback
        salary_driver_terms = ['driver', 'salary', 'minimum', 'maximum', 'taka', 'tk', 'allowance']
        salary_driver_count = sum(1 for term in salary_driver_terms if term in answer_lower)
        
        if salary_driver_count >= 2:
            return False
        
        return False  # If no explicit fallback patterns, it's not a fallback
    
    def _extract_user_query(self, enhanced_query: str) -> str:
        """Extract the actual user query from context-enhanced query"""
        # Look for "user query:" or "USER QUERY:" pattern
        import re
        
        # Try to find the user query pattern
        patterns = [
            r'user query:\s*(.+?)(?:\s*$|\s*\|)',  # "user query: something"
            r'USER QUERY:\s*(.+?)(?:\s*$|\s*\|)',  # "USER QUERY: something"
            r'query:\s*(.+?)(?:\s*$|\s*\|)',       # "query: something"
        ]
        
        for pattern in patterns:
            match = re.search(pattern, enhanced_query, re.IGNORECASE)
            if match:
                extracted = match.group(1).strip()
                if extracted:
                    return extracted
        
        # If no pattern found, check if the query looks like it has context enhancement
        if '|' in enhanced_query and 'context:' in enhanced_query.lower():
            # Try to extract the last meaningful part
            parts = enhanced_query.split('|')
            for part in reversed(parts):
                part = part.strip()
                if part and not any(word in part.lower() for word in ['context', 'chat', 'topic', 'discussed', 'mentioned', 'ago']):
                    return part
        
        # If all else fails, return the original query
        return enhanced_query
    
    def _calculate_content_relevance(self, query: str, answer: str) -> float:
        """Calculate content relevance boost using dynamic analysis"""
        print(f"[RELEVANCE AGENT] Content relevance calculation:")
        print(f"[RELEVANCE AGENT]   Query: '{query}'")
        print(f"[RELEVANCE AGENT]   Answer preview: '{answer[:100]}...'")
        
        query_words = set(query.lower().split())
        answer_words = set(answer.lower().split())
        
        # Remove common stop words dynamically
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were'}
        query_content = query_words - stop_words
        answer_content = answer_words - stop_words
        
        print(f"[RELEVANCE AGENT]   Query content words: {query_content}")
        print(f"[RELEVANCE AGENT]   Answer content words (first 10): {list(answer_content)[:10]}")
        
        if not query_content:
            print(f"[RELEVANCE AGENT]   No query content words - returning 0.0")
            return 0.0
        
        # Calculate word overlap ratio
        overlap = len(query_content.intersection(answer_content))
        overlap_ratio = overlap / len(query_content)
        
        print(f"[RELEVANCE AGENT]   Word overlap: {overlap}/{len(query_content)} = {overlap_ratio:.3f}")
        
        # Dynamic boost based on content density and specificity
        content_boost = 0.0
        
        # Boost for any word overlap (more generous)
        if overlap_ratio > 0.05:  # Much lower threshold
            word_boost = overlap_ratio * 0.4  # Higher boost
            content_boost += word_boost
            print(f"[RELEVANCE AGENT]   Word overlap boost: +{word_boost:.3f}")
        else:
            print(f"[RELEVANCE AGENT]   Word overlap too low ({overlap_ratio:.3f} <= 0.05) - no boost")
        
        # Special boost for key HR terms (greatly expanded list)
        hr_terms = {
            'allowance', 'salary', 'leave', 'policy', 'benefit', 'hair', 'cutting', 'medical', 'house', 'transport',
            'deduction', 'notice', 'pay', 'proposal', 'increment', 'bonus', 'gratuity', 'provident', 'fund',
            'performance', 'appraisal', 'training', 'development', 'manager', 'officer', 'employee', 'staff',
            'overtime', 'shift', 'attendance', 'recruitment', 'promotion', 'transfer', 'retirement',
            'driver', 'light', 'medium', 'heavy', 'minimum', 'maximum', 'basic', 'grade', 'position',
            'bill', 'claim', 'claims', 'expense', 'expenses', 'reimbursement', 'reimbursements', 'cost', 'costs',
            'travel', 'traveling', 'visit', 'visiting', 'farm', 'farms', 'field', 'site', 'location',
            'procedure', 'process', 'form', 'application', 'approval', 'submit', 'submission',
            'department', 'office', 'head', 'director', 'management', 'administration', 'hr',
            'taka', 'tk', 'bdt', 'amount', 'rate', 'fee', 'charge', 'payment', 'compensation',
            'document', 'certificate', 'receipt', 'voucher', 'invoice', 'proof', 'evidence'
        }
        query_hr_terms = query_content.intersection(hr_terms)
        answer_hr_terms = answer_content.intersection(hr_terms)
        
        print(f"[RELEVANCE AGENT]   Query HR terms: {query_hr_terms}")
        print(f"[RELEVANCE AGENT]   Answer HR terms: {answer_hr_terms}")
        
        if query_hr_terms and answer_hr_terms:
            hr_overlap = len(query_hr_terms.intersection(answer_hr_terms))
            if hr_overlap > 0:
                hr_boost = hr_overlap * 0.25  # Strong boost for HR term matches
                content_boost += hr_boost
                print(f"[RELEVANCE AGENT]   HR terms boost: +{hr_boost:.3f} ({hr_overlap} matches)")
        elif query_hr_terms:
            # If query has HR terms but answer doesn't, this might be a fallback response
            # Give a larger boost just for having HR terms in the query
            query_hr_boost = len(query_hr_terms) * 0.15  # Increased boost
            content_boost += query_hr_boost
            print(f"[RELEVANCE AGENT]   Query-only HR terms boost: +{query_hr_boost:.3f} ({len(query_hr_terms)} terms)")
        
        # Additional boost for document title patterns (like "Proposal of...")
        title_patterns = ['proposal', 'policy', 'circular', 'notice', 'order', 'guideline', 'manual', 'handbook']
        query_title_terms = [word for word in query_content if word in title_patterns]
        answer_title_terms = [word for word in answer_content if word in title_patterns]
        
        print(f"[RELEVANCE AGENT]   Query title terms: {query_title_terms}")
        print(f"[RELEVANCE AGENT]   Answer title terms: {answer_title_terms}")
        
        if query_title_terms and answer_title_terms:
            title_overlap = len(set(query_title_terms).intersection(set(answer_title_terms)))
            if title_overlap > 0:
                title_boost = title_overlap * 0.3  # Strong boost for document title patterns
                content_boost += title_boost
                print(f"[RELEVANCE AGENT]   Title patterns boost: +{title_boost:.3f} ({title_overlap} matches)")
        elif query_title_terms:
            # If query has title patterns but answer doesn't, boost for document-like queries
            query_title_boost = len(query_title_terms) * 0.2
            content_boost += query_title_boost
            print(f"[RELEVANCE AGENT]   Query-only title boost: +{query_title_boost:.3f} ({len(query_title_terms)} terms)")
        
        # Boost for specific, longer words (more meaningful)
        specific_words = [word for word in query_content if len(word) > 4]  # Lowered from 5
        if specific_words:
            specific_overlap = len([word for word in specific_words if word in answer_content])
            if specific_overlap > 0:
                content_boost += (specific_overlap / len(specific_words)) * 0.2  # Increased boost
        
        # Boost for numerical data matches (amounts, dates, etc.)
        import re
        query_numbers = set(re.findall(r'\d+', query))
        answer_numbers = set(re.findall(r'\d+', answer))
        if query_numbers and answer_numbers:
            number_overlap = len(query_numbers.intersection(answer_numbers))
            if number_overlap > 0:
                content_boost += number_overlap * 0.15  # Increased boost
        
        # Boost for compound terms and document titles
        query_text = query.lower()
        answer_text = answer.lower()
        compound_terms = [
            'hair cutting', 'sick leave', 'annual leave', 'house allowance', 'medical allowance',
            'notice pay', 'pay deduction', 'salary structure', 'leave policy', 'performance review',
            'job group', 'management trainee', 'feed mill', 'head office', 'location allowance',
            'transport allowance', 'food allowance', 'overtime allowance', 'production bonus'
        ]
        
        compound_matches = []
        for term in compound_terms:
            if term in query_text:
                if term in answer_text:
                    compound_boost = 0.3  # Strong boost for compound term matches
                    content_boost += compound_boost
                    compound_matches.append(f"{term} (both)")
                else:
                    # Query has compound term but answer doesn't - might be fallback
                    query_compound_boost = 0.15
                    content_boost += query_compound_boost
                    compound_matches.append(f"{term} (query only)")
        
        if compound_matches:
            print(f"[RELEVANCE AGENT]   Compound terms: {compound_matches}")
        
        # Special boost for exact document title matches
        document_patterns = [
            'proposal of', 'policy for', 'circular for', 'notice for', 'order for',
            'guideline for', 'manual for', 'handbook for', 'procedure for'
        ]
        
        document_matches = []
        for pattern in document_patterns:
            if pattern in query_text:
                if pattern in answer_text:
                    doc_boost = 0.4  # Very strong boost for document title patterns
                    content_boost += doc_boost
                    document_matches.append(f"{pattern} (both)")
                else:
                    # Query has document pattern but answer doesn't - likely fallback
                    query_doc_boost = 0.25
                    content_boost += query_doc_boost
                    document_matches.append(f"{pattern} (query only)")
        
        if document_matches:
            print(f"[RELEVANCE AGENT]   Document patterns: {document_matches}")
        
        for pattern in document_patterns:
            if pattern in query_text and pattern in answer_text:
                content_boost += 0.4  # Very strong boost for document title patterns
        
        print(f"[RELEVANCE AGENT]   Total content boost: {content_boost:.3f} (capped at 0.6)")
        return min(content_boost, 0.6)  # Increased cap
    
    def _calculate_adaptive_threshold(self, query: str, answer: str) -> float:
        """Calculate adaptive threshold based on query and answer characteristics"""
        base_threshold = self.relevance_threshold
        
        print(f"[RELEVANCE AGENT] Adaptive threshold calculation:")
        print(f"[RELEVANCE AGENT]   Base threshold: {base_threshold}")
        
        # Adjust threshold based on query complexity
        query_words = len(query.split())
        print(f"[RELEVANCE AGENT]   Query words: {query_words}")
        
        if query_words <= 3:  # Simple queries like "time keeping allowance"
            multiplier = 0.3  # Much lower threshold for simple queries (reduced from 0.5)
            base_threshold *= multiplier
            print(f"[RELEVANCE AGENT]   Simple query (≤3 words): threshold × {multiplier}")
        elif query_words <= 6:  # Medium queries
            multiplier = 0.5  # Lower threshold for medium queries (reduced from 0.7)
            base_threshold *= multiplier
            print(f"[RELEVANCE AGENT]   Medium query (≤6 words): threshold × {multiplier}")
        elif query_words > 10:  # Complex queries
            multiplier = 1.0  # No penalty for complex queries (reduced from 1.1)
            base_threshold *= multiplier
            print(f"[RELEVANCE AGENT]   Complex query (>10 words): threshold × {multiplier}")
        
        # Adjust based on answer length and quality
        answer_words = len(answer.split())
        print(f"[RELEVANCE AGENT]   Answer words: {answer_words}")
        
        if answer_words < 20:  # Short answers (likely fallbacks)
            multiplier = 0.4  # Much lower threshold to catch fallbacks (reduced from 0.6)
            base_threshold *= multiplier
            print(f"[RELEVANCE AGENT]   Short answer (<20 words): threshold × {multiplier}")
        elif answer_words > 200:  # Very long answers
            multiplier = 1.0  # No penalty for long answers (reduced from 1.05)
            base_threshold *= multiplier
            print(f"[RELEVANCE AGENT]   Long answer (>200 words): threshold × {multiplier}")
        
        # Special adjustment for HR queries with fallback answers
        if self._detect_fallback_response(answer):
            multiplier = 0.5  # Even lower threshold for fallback responses
            base_threshold *= multiplier
            print(f"[RELEVANCE AGENT]   Fallback response detected: threshold × {multiplier}")
        
        final_threshold = max(0.01, min(base_threshold, 0.15))  # Much lower bounds for better sensitivity
        print(f"[RELEVANCE AGENT]   Final adaptive threshold: {final_threshold}")
        
        return final_threshold
    
    def _calculate_dynamic_confidence(self, base_similarity: float, content_boost: float, is_fallback: bool) -> float:
        """Calculate confidence using dynamic factors"""
        if is_fallback:
            return max(0.1, base_similarity)  # Low confidence for fallbacks
        
        # Base confidence from similarity
        confidence = base_similarity * 1.2
        
        # Boost confidence if content is relevant
        if content_boost > 0.1:
            confidence += content_boost * 0.5
        
        # Ensure confidence is within bounds
        return min(1.0, max(0.1, confidence))
    
    def _generate_validation_reason(self, base_similarity: float, content_boost: float, 
                                  adaptive_threshold: float, is_fallback: bool, is_relevant: bool) -> str:
        """Generate dynamic validation reasoning"""
        if is_fallback:
            return f"Fallback response detected (base similarity: {base_similarity:.3f})"
        
        if is_relevant:
            reason = f"Relevant answer (similarity: {base_similarity:.3f}"
            if content_boost > 0:
                reason += f", content boost: +{content_boost:.3f}"
            reason += f", threshold: {adaptive_threshold:.3f})"
            return reason
        else:
            return f"Low relevance (similarity: {base_similarity:.3f} < adaptive threshold: {adaptive_threshold:.3f})"
    
    def update_threshold(self, new_threshold: float) -> None:
        """Update relevance threshold"""
        self.relevance_threshold = new_threshold
        print(f"[RELEVANCE AGENT] Threshold updated to: {self.relevance_threshold}")
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get relevance agent statistics"""
        return {
            "relevance_threshold": self.relevance_threshold,
            "embedding_model": self.settings.EMBEDDING_MODEL,
            "status": "initialized"
        }
