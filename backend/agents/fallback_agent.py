"""
FallbackAgent - Generate polite "not found / out-of-domain" response
"""
import random
from typing import Dict, Any, List, Optional
from dataclasses import dataclass

@dataclass
class FallbackResult:
    """Result of fallback response generation"""
    response: str
    response_type: str
    confidence: float
    suggested_topics: List[str]

class FallbackAgent:
    """Agent for generating fallback responses"""
    
    def __init__(self):
        # Response templates for different scenarios
        self.response_templates = {
            'greeting': [
                "Hello! Welcome to Kazi Farms! I'm your AI assistant, here to help you with HR policies, salary information, employee benefits, and company procedures. How can I assist you today?",
                "Hi there! Great to meet you! I'm the Kazi Farms chatbot, and I'm here to help you with any questions about our HR policies, salary structures, allowances, or company procedures. What would you like to know?",
                "Good day! Welcome to Kazi Farms! I'm your friendly assistant, ready to help you with employee benefits, leave policies, salary information, and other HR-related questions. How can I help you today?",
                "Hello! Nice to see you here! I'm the Kazi Farms AI assistant, and I'm excited to help you with any questions about our company policies, salary structures, allowances, or HR procedures. What can I do for you?",
                "Hi! Welcome to the Kazi Farms family! I'm here to assist you with all your HR and company-related questions. Whether it's about salary, benefits, policies, or procedures - I'm ready to help! What would you like to know?"
            ],
            'out_of_domain': [
                "I'm a Kazi Farms chatbot specialized in HR and business information. I can't help with that topic, but I'd be happy to assist you with salary structures, allowance policies, leave procedures, or job designations!",
                "That's outside my area of expertise. I'm designed to help with Kazi Farms HR policies like salary increments, various allowances, and company procedures. What would you like to know about our policies?",
                "I'm focused on Kazi Farms information only. I can help you with salary inquiries, allowance policies, leave procedures, or other HR-related questions. How can I assist you with Kazi Farms information?",
                "I don't have information about that topic. I'm here to help with Kazi Farms HR information like salary structures, allowances, leave policies, and job procedures. What questions do you have about our company policies?"
            ],
            'no_documents': [
                "I don't have specific information about that in our Kazi Farms database. Could you please rephrase your question or ask about salary structures, allowance policies, leave procedures, or job designations?",
                "That's not something I've found in our Kazi Farms documents. I can help you with salary information, various allowances, leave policies, or HR procedures. What would you like to know?",
                "I couldn't find relevant information about that topic in our Kazi Farms database. I'm here to help with salary structures, allowance policies, and company procedures. How can I assist you?",
                "I don't have that specific information in our Kazi Farms knowledge base. I can help with salary inquiries, allowance policies, leave procedures, or job-related questions. What do you need to know?"
            ],
            'low_confidence': [
                "I found some information but I'm not entirely confident about the accuracy. Could you please rephrase your question or ask about salary structures, allowances, leave policies, or other HR topics?",
                "I have limited information about that topic. I'd be happy to help you with salary inquiries, leave policies, allowances, or other Kazi Farms HR questions. What would you like to know?",
                "I'm not sure I have complete information about that. I can definitely help with salary structures, leave policies, allowances, or other Kazi Farms HR topics. How can I assist you?"
            ],
            'error': [
                "I encountered an error while processing your request. Please try rephrasing your question or ask about Kazi Farms HR policies, salary information, or company procedures.",
                "Something went wrong while processing your question. I'm here to help with Kazi Farms information. What would you like to know about our HR policies or procedures?",
                "I had trouble processing that request. I can help you with salary inquiries, leave policies, allowances, or other Kazi Farms HR topics. What do you need assistance with?"
            ]
        }
        
        # Suggested topics for Kazi Farms (based on actual available documents)
        self.suggested_topics = [
            "Salary structures and increment proposals",
            "Various allowance policies (house, location, food, travel, etc.)",
            "Leave policies and procedures", 
            "Employee bonuses and incentives",
            "HR policies and office procedures",
            "Job groups and designations",
            "Transfer and promotion policies",
            "Performance management systems"
        ]
    
    def generate_fallback_response(self, fallback_type: str = 'out_of_domain', 
                                 context: Optional[Dict[str, Any]] = None) -> FallbackResult:
        """Generate appropriate fallback response"""
        try:
            # Get response template based on type
            templates = self.response_templates.get(fallback_type, self.response_templates['out_of_domain'])
            response = random.choice(templates)
            
            # Adjust confidence based on fallback type
            confidence_map = {
                'greeting': 0.95,
                'out_of_domain': 0.9,
                'no_documents': 0.8,
                'low_confidence': 0.6,
                'error': 0.5
            }
            confidence = confidence_map.get(fallback_type, 0.7)
            
            # Customize response based on context
            if context:
                response = self._customize_response(response, context)
            
            print(f"[FALLBACK AGENT] Generated {fallback_type} response")
            
            return FallbackResult(
                response=response,
                response_type=fallback_type,
                confidence=confidence,
                suggested_topics=self.suggested_topics
            )
            
        except Exception as e:
            print(f"[FALLBACK AGENT ERROR] {str(e)}")
            return FallbackResult(
                response="I'm having trouble right now. Please try asking about Kazi Farms HR policies, salary information, or company procedures.",
                response_type="error",
                confidence=0.3,
                suggested_topics=self.suggested_topics
            )
    
    def _customize_response(self, response: str, context: Dict[str, Any]) -> str:
        """Customize response based on context"""
        try:
            # Add specific suggestions based on context
            if 'query_type' in context:
                query_type = context['query_type']
                
                if 'salary' in query_type.lower():
                    response += " For salary information, please specify the job position or designation."
                elif 'allowance' in query_type.lower():
                    response += " For allowance details, please specify the type of allowance you're interested in."
                elif 'leave' in query_type.lower():
                    response += " For leave policies, please specify the type of leave you're asking about."
            
            # Add similarity score information if available
            if 'similarity_score' in context:
                similarity = context['similarity_score']
                if similarity < 0.3:
                    response += " Your question seems quite different from our Kazi Farms topics."
            
            return response
            
        except Exception as e:
            print(f"[FALLBACK AGENT ERROR] Customization failed: {str(e)}")
            return response
    
    def get_suggested_topics(self) -> List[str]:
        """Get list of suggested topics"""
        return self.suggested_topics.copy()
    
    def add_custom_response(self, fallback_type: str, response: str) -> None:
        """Add custom response template"""
        if fallback_type not in self.response_templates:
            self.response_templates[fallback_type] = []
        
        self.response_templates[fallback_type].append(response)
        print(f"[FALLBACK AGENT] Added custom response for {fallback_type}")
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get fallback agent statistics"""
        return {
            "response_types": list(self.response_templates.keys()),
            "total_templates": sum(len(templates) for templates in self.response_templates.values()),
            "suggested_topics_count": len(self.suggested_topics),
            "status": "initialized"
        }
