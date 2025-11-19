"""
IntentAgent - Detect greeting, small talk, or general inquiry
"""
import re
from typing import Dict, Any
from dataclasses import dataclass

@dataclass
class IntentResult:
    """Result of intent detection"""
    intent_type: str
    confidence: float
    is_greeting: bool
    is_small_talk: bool
    greeting_response: str = None

class IntentAgent:
    """Agent for detecting user intent"""
    
    def __init__(self):
        # Greeting patterns
        self.greeting_patterns = [
            r'\b(hello|hi|hey|good morning|good afternoon|good evening|good day|greetings|howdy)\b',
            r'\b(hlw|hlo|helo|hii|hie|hy|hye)\b',  # Common typos
            r'\b(morning|afternoon|evening|night)\b',
            r'\b(how are you|how do you do|how\'s it going|how are things)\b',
            r'\b(what\'s up|how\'s your day|are you okay|are you fine)\b',
            r'\b(how are you doing|how\'s everything|how\'s life)\b',
            r'\b(start|begin|help me|assist me)\b',  # Help-seeking greetings
            r'\b(welcome|greet|meet|introduce)\b',  # Welcome/introduction patterns
            r'\b(yo|sup|wassup|hey there|hiya)\b',  # Casual greetings
            r'\b(good\s+)?(gm|gn|ga|ge)(?![a-z/])',  # Abbreviations for good morning/night/afternoon/evening (not followed by letters or slash)
            r'\b(salam|namaste|namaskar)\b',  # Cultural greetings
            r'\b(hiya|heya|hullo|hallo)\b',  # Alternative spellings
            r'\b(good day|g\'day|gday)\b',  # Australian/informal greetings
            r'\b(hey there|hi there|hello there)\b'  # Extended greetings
        ]
        
        # Job/position related patterns (should NOT be treated as greetings)
        self.job_patterns = [
            r'\b(gm|agm)\b.*\b(position|role|job|designation|title|salary|wage|pay|compensation|benefit|allowance|manager|supervisor|officer|executive|admin|hr|human\s+resources|employee|staff)\b',
            r'\b(general\s+manager|assistant\s+general\s+manager)\b',
            r'\b(manager|supervisor|officer|executive|admin)\b.*\b(position|role|job|designation|title|salary|wage|pay|compensation|benefit|allowance)\b',
            r'\b(position|role|job|designation|title)\b.*\b(gm|agm|manager|supervisor|officer|executive|admin)\b',
            r'\b(salary|wage|pay|compensation)\b.*\b(gm|agm|manager|supervisor|officer|executive|admin)\b',
            r'\b(allowance|benefit|bonus|incentive)\b.*\b(gm|agm|manager|supervisor|officer|executive|admin)\b',
            r'\b(hr|human\s+resources|employee|staff)\b.*\b(gm|agm|manager|supervisor|officer|executive|admin)\b',
            r'\b(tell\s+me\s+about|about|information\s+about|details\s+about)\s+(gm|agm|general\s+manager|assistant\s+general\s+manager)\b',
            r'\b(gm|agm)\b.*\b(benefits|allowance|salary|compensation|position|role|job|designation|title)\b'
        ]
        
        # Small talk patterns
        self.small_talk_patterns = [
            r'\b(weather|rain|sunny|cloudy|hot|cold)\b',
            r'\b(weekend|vacation|holiday|travel)\b',
            r'\b(food|eat|drink|coffee|tea|lunch|dinner)\b',
            r'\b(sports|football|soccer|cricket|game)\b',
            r'\b(movie|music|book|entertainment)\b',
            r'\b(family|friend|relationship|personal)\b'
        ]
        
        # Greeting responses
        self.greeting_responses = [
            "Hello! Welcome to Kazi Farms! I'm your AI assistant, here to help you with HR policies, salary information, employee benefits, and company procedures. How can I assist you today?",
            "Hi there! Great to meet you! I'm the Kazi Farms chatbot, and I'm here to help you with any questions about our HR policies, salary structures, allowances, or company procedures. What would you like to know?",
            "Good day! Welcome to Kazi Farms! I'm your friendly assistant, ready to help you with employee benefits, leave policies, salary information, and other HR-related questions. How can I help you today?",
            "Hello! Nice to see you here! I'm the Kazi Farms AI assistant, and I'm excited to help you with any questions about our company policies, salary structures, allowances, or HR procedures. What can I do for you?",
            "Hi! Welcome to the Kazi Farms family! I'm here to assist you with all your HR and company-related questions. Whether it's about salary, benefits, policies, or procedures - I'm ready to help! What would you like to know?"
        ]
    
    def detect_intent(self, text: str) -> IntentResult:
        """Detect user intent from text"""
        try:
            text_lower = text.lower().strip()
            
            # Check for job-related patterns first (these should NOT be treated as greetings)
            is_job_related = self._is_job_related(text_lower)
            if is_job_related:
                return IntentResult(
                    intent_type="job_inquiry",
                    confidence=0.9,
                    is_greeting=False,
                    is_small_talk=False,
                    greeting_response=None
                )
            
            # Check for greetings
            is_greeting = self._is_greeting(text_lower)
            if is_greeting:
                greeting_response = self._get_greeting_response(text_lower)
                return IntentResult(
                    intent_type="greeting",
                    confidence=0.95,
                    is_greeting=True,
                    is_small_talk=False,
                    greeting_response=greeting_response
                )
            
            # Check for small talk
            is_small_talk = self._is_small_talk(text_lower)
            if is_small_talk:
                return IntentResult(
                    intent_type="small_talk",
                    confidence=0.85,
                    is_greeting=False,
                    is_small_talk=True,
                    greeting_response=None
                )
            
            # Default to general inquiry
            return IntentResult(
                intent_type="general_inquiry",
                confidence=0.7,
                is_greeting=False,
                is_small_talk=False,
                greeting_response=None
            )
            
        except Exception as e:
            print(f"[INTENT AGENT ERROR] {str(e)}")
            return IntentResult(
                intent_type="general_inquiry",
                confidence=0.5,
                is_greeting=False,
                is_small_talk=False,
                greeting_response=None
            )
    
    def _is_job_related(self, text: str) -> bool:
        """Check if text is job/position related"""
        for pattern in self.job_patterns:
            if re.search(pattern, text, re.IGNORECASE):
                return True
        return False
    
    def _is_greeting(self, text: str) -> bool:
        """Check if text is a greeting"""
        for pattern in self.greeting_patterns:
            if re.search(pattern, text, re.IGNORECASE):
                return True
        return False
    
    def _is_small_talk(self, text: str) -> bool:
        """Check if text is small talk"""
        for pattern in self.small_talk_patterns:
            if re.search(pattern, text, re.IGNORECASE):
                return True
        return False
    
    def _get_greeting_response(self, text: str) -> str:
        """Get appropriate greeting response based on time/context"""
        import random
        
        # Check for specific greeting types (including abbreviations)
        if any(word in text for word in ['morning', 'good morning', 'gm']):
            return "Good morning! Welcome to Kazi Farms! I'm your AI assistant, ready to help you with HR policies, salary information, employee benefits, and company procedures. How can I assist you today?"
        elif any(word in text for word in ['afternoon', 'good afternoon', 'ga']):
            return "Good afternoon! Welcome to Kazi Farms! I'm your AI assistant, here to help you with HR policies, salary information, employee benefits, and company procedures. How can I assist you today?"
        elif any(word in text for word in ['evening', 'good evening', 'ge']):
            return "Good evening! Welcome to Kazi Farms! I'm your AI assistant, ready to help you with HR policies, salary information, employee benefits, and company procedures. How can I assist you today?"
        elif any(word in text for word in ['night', 'good night', 'gn']):
            return "Good night! Welcome to Kazi Farms! I'm your AI assistant, here to help you with HR policies, salary information, employee benefits, and company procedures. How can I assist you today?"
        else:
            return random.choice(self.greeting_responses)
