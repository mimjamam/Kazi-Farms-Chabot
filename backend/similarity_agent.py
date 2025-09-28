import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import re
from collections import Counter
import difflib
from typing import Dict, Any

class SimilarityComparisonAgent:
    def __init__(self):
        self.sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
        
        self.tfidf_vectorizer = TfidfVectorizer(
            stop_words='english',
            ngram_range=(1, 2),
            max_features=1000
        )
        
        self.similarity_thresholds = {
            'excellent': 0.8,
            'good': 0.6,
            'fair': 0.4,
            'poor': 0.2
        }
    
    def preprocess_text(self, text: str) -> str:
        if not text:
            return ""
        
        text = text.lower()
        text = re.sub(r'[^\w\s]', ' ', text)
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def calculate_semantic_similarity(self, query: str, response: str) -> float:
        try:
            query_clean = self.preprocess_text(query)
            response_clean = self.preprocess_text(response)
            
            query_embedding = self.sentence_model.encode([query_clean])
            response_embedding = self.sentence_model.encode([response_clean])
            
            similarity = cosine_similarity(query_embedding, response_embedding)[0][0]
            return float(similarity)
            
        except Exception as e:
            return 0.0
    
    def calculate_keyword_similarity(self, query: str, response: str) -> float:
        try:
            query_clean = self.preprocess_text(query)
            response_clean = self.preprocess_text(response)
            
            texts = [query_clean, response_clean]
            tfidf_matrix = self.tfidf_vectorizer.fit_transform(texts)
            
            similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
            return float(similarity)
            
        except Exception as e:
            return 0.0
    
    def calculate_structural_similarity(self, query: str, response: str) -> float:
        try:
            query_clean = self.preprocess_text(query)
            response_clean = self.preprocess_text(response)
            
            sequence_similarity = difflib.SequenceMatcher(None, query_clean, response_clean).ratio()
            
            query_words = set(query_clean.split())
            response_words = set(response_clean.split())
            
            if not query_words or not response_words:
                word_overlap = 0.0
            else:
                intersection = query_words.intersection(response_words)
                union = query_words.union(response_words)
                word_overlap = len(intersection) / len(union) if union else 0.0
            
            structural_similarity = (sequence_similarity + word_overlap) / 2
            return float(structural_similarity)
            
        except Exception as e:
            return 0.0
    
    def calculate_content_relevance(self, query: str, response: str) -> float:
        try:
            kazi_keywords = [
                'salary', 'allowance', 'policy', 'leave', 'hr', 'employee', 'management',
                'worker', 'bonus', 'increment', 'transport', 'medical', 'house', 'location',
                'overtime', 'production', 'performance', 'eid', 'sysnova', 'hatchery',
                'farm', 'feed mill', 'sales', 'commercial', 'finance', 'quality',
                'maintenance', 'driver', 'helper', 'mechanic', 'accountant', 'supervisor',
                'manager', 'officer', 'executive', 'technician', 'operator', 'cleaner',
                'guard', 'trainee', 'in-charge', 'person', 'level', 'group', 'structure',
                'scale', 'grade', 'tier', 'bracket', 'range', 'wage', 'pay', 'compensation',
                'remuneration', 'income', 'earnings', 'benefit', 'perk', 'incentive',
                'subsidy', 'rule', 'regulation', 'guideline', 'procedure', 'standard',
                'circular', 'order', 'notice', 'memo', 'vacation', 'holiday', 'off',
                'absence', 'break', 'extra', 'additional', 'extended', 'beyond', 'travel',
                'commute', 'vehicle', 'car', 'bus', 'health', 'treatment', 'hospital',
                'clinic', 'doctor', 'financial', 'payment', 'cash', 'bill', 'claim',
                'budget', 'ceiling', 'aid', 'retirement', 'pension', 'resignation',
                'exit', 'departure', 'termination', 'identity', 'card', 'passport',
                'document', 'handover', 'picnic', 'sample', 'collection', 'tray',
                'factory', 'slaughtering', 'plant', 'egg', 'eggs', 'commercial',
                'franchise', 'department', 'hardware', 'software', 'kazi', 'media'
            ]
            
            query_clean = self.preprocess_text(query)
            response_clean = self.preprocess_text(response)
            
            query_words = query_clean.split()
            response_words = response_clean.split()
            
            query_domain_count = sum(1 for word in query_words if word in kazi_keywords)
            response_domain_count = sum(1 for word in response_words if word in kazi_keywords)
            
            if not query_words or not response_words:
                return 0.0
            
            query_relevance = query_domain_count / len(query_words)
            response_relevance = response_domain_count / len(response_words)
            
            relevance_score = (query_relevance + response_relevance) / 2
            return float(relevance_score)
            
        except Exception as e:
            return 0.0
    
    def calculate_comprehensive_similarity(self, query: str, response: str) -> Dict[str, Any]:
        try:
            semantic_sim = self.calculate_semantic_similarity(query, response)
            keyword_sim = self.calculate_keyword_similarity(query, response)
            structural_sim = self.calculate_structural_similarity(query, response)
            content_rel = self.calculate_content_relevance(query, response)
            
            weights = {
                'semantic': 0.4,
                'keyword': 0.3,
                'structural': 0.2,
                'content_relevance': 0.1
            }
            
            overall_similarity = (
                semantic_sim * weights['semantic'] +
                keyword_sim * weights['keyword'] +
                structural_sim * weights['structural'] +
                content_rel * weights['content_relevance']
            )
            
            if overall_similarity >= self.similarity_thresholds['excellent']:
                similarity_level = 'excellent'
            elif overall_similarity >= self.similarity_thresholds['good']:
                similarity_level = 'good'
            elif overall_similarity >= self.similarity_thresholds['fair']:
                similarity_level = 'fair'
            else:
                similarity_level = 'poor'
            
            return {
                'semantic_similarity': semantic_sim,
                'keyword_similarity': keyword_sim,
                'structural_similarity': structural_sim,
                'content_relevance': content_rel,
                'overall_similarity': overall_similarity,
                'similarity_level': similarity_level,
                'weights': weights
            }
            
        except Exception as e:
            return {
                'semantic_similarity': 0.0,
                'keyword_similarity': 0.0,
                'structural_similarity': 0.0,
                'content_relevance': 0.0,
                'overall_similarity': 0.0,
                'similarity_level': 'poor',
                'weights': {}
            }
    
    def generate_similarity_report(self, query: str, response: str, similarity_metrics: Dict[str, Any]) -> str:
        try:
            report = f"""
SIMILARITY COMPARISON REPORT
{'='*50}

Input Query: {query[:100]}{'...' if len(query) > 100 else ''}

Output Response: {response[:100]}{'...' if len(response) > 100 else ''}

SIMILARITY METRICS:
• Semantic Similarity: {similarity_metrics['semantic_similarity']:.3f}
• Keyword Similarity: {similarity_metrics['keyword_similarity']:.3f}
• Structural Similarity: {similarity_metrics['structural_similarity']:.3f}
• Content Relevance: {similarity_metrics['content_relevance']:.3f}

OVERALL ASSESSMENT:
• Overall Similarity: {similarity_metrics['overall_similarity']:.3f}
• Similarity Level: {similarity_metrics['similarity_level'].upper()}

ANALYSIS:
"""
            
            if similarity_metrics['similarity_level'] == 'excellent':
                report += "The response is highly relevant to the query with excellent semantic and keyword alignment."
            elif similarity_metrics['similarity_level'] == 'good':
                report += "The response is relevant to the query with good semantic and keyword alignment."
            elif similarity_metrics['similarity_level'] == 'fair':
                report += "The response has moderate relevance to the query. Consider improving semantic alignment."
            else:
                report += "The response has poor relevance to the query. Significant improvement needed."
            
            if similarity_metrics['semantic_similarity'] < 0.5:
                report += "\n• Consider improving semantic understanding of the query."
            
            if similarity_metrics['keyword_similarity'] < 0.5:
                report += "\n• Consider including more relevant keywords from the query."
            
            if similarity_metrics['content_relevance'] < 0.3:
                report += "\n• Consider including more domain-specific content."
            
            return report
            
        except Exception as e:
            return f"Error generating similarity report: {e}"
