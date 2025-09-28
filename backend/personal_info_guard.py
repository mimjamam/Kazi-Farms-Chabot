import re
from typing import List, Dict, Any

class PersonalInfoGuard:
    def __init__(self):
        self.personal_identity_patterns = [
            r'who am i',
            r'who are you',
            r'tell me about myself',
            r'about me',
            r'my information',
            r'my details',
            r'my profile',
            r'my identity',
            r'personal information',
            r'my name',
            r'my email',
            r'my position',
            r'my role',
            r'identify me',
            r'my records',
            r'information about me',
            r'my data',
            r'my account',
            r'my profile information'
        ]
        
        self.personal_greeting_patterns = [
            r'how are you',
            r'how do you do',
            r'how\'s it going',
            r'how are things',
            r'what\'s up',
            r'how\'s your day',
            r'are you okay',
            r'are you fine',
            r'how are you doing',
            r'how\'s everything',
            r'how\'s life'
        ]
        
        self.blocked_responses = [
            "I can't provide personal information about users. I'm here to help with Kazi Farms HR policies, salary information, and company procedures.",
            "I don't have access to personal user information. I can assist you with company policies, employee benefits, and HR-related questions.",
            "I'm designed to help with Kazi Farms information, not personal identity questions. Please ask about our policies, procedures, or employee benefits.",
            "I can't identify users or provide personal information. I'm here to help with Kazi Farms HR and company information.",
            "For privacy and security reasons, I cannot provide personal information. I can help you with company policies, salary structures, and HR procedures.",
            "I'm not authorized to share personal information. I can assist with Kazi Farms policies, employee benefits, and company procedures."
        ]
        
        self.greeting_responses = [
            "I'm doing great! I'm a chatbot designed to help with Kazi Farms information. How can I assist you with our HR policies, salary information, or company procedures?",
            "I'm functioning perfectly! I'm here to help you with Kazi Farms policies and procedures. What would you like to know about our company?",
            "I'm running smoothly! I'm a Kazi Farms assistant ready to help with employee information, policies, and company details. What can I help you with?",
            "I'm doing well! I'm designed to assist with Kazi Farms HR and company information. How can I help you today?",
            "I'm great! I'm a chatbot specialized in Kazi Farms information. I can help with salary structures, allowances, leave policies, and more. What do you need to know?",
            "I'm excellent! I'm here to help with Kazi Farms policies and procedures. What information can I provide for you today?"
        ]
        
        self.redirect_suggestions = [
            "Salary structures for different positions",
            "Allowance policies (house, transport, medical, etc.)",
            "Leave policies (sick, annual, casual, etc.)",
            "Employee benefits and bonuses",
            "HR policies and procedures",
            "Company locations and departments",
            "Job roles and responsibilities",
            "Performance evaluation processes"
        ]
    
    def is_personal_info_query(self, query: str) -> bool:
        query_lower = query.lower().strip()
        
        for pattern in self.personal_identity_patterns:
            if re.search(pattern, query_lower):
                return True
        
        return False
    
    def is_personal_greeting(self, query: str) -> bool:
        query_lower = query.lower().strip()
        
        for pattern in self.personal_greeting_patterns:
            if re.search(pattern, query_lower):
                return True
        
        return False
    
    def get_blocked_response(self) -> str:
        import random
        return random.choice(self.blocked_responses)
    
    def get_greeting_response(self) -> str:
        import random
        return random.choice(self.greeting_responses)
    
    def get_redirect_response(self, query: str) -> str:
        import random
        
        base_response = self.get_blocked_response()
        
        suggestions = random.sample(self.redirect_suggestions, 3)
        suggestions_text = "\n".join([f"• {suggestion}" for suggestion in suggestions])
        
        return f"{base_response}\n\nHere's what I can help you with:\n{suggestions_text}\n\nPlease ask about any of these topics, and I'll be happy to assist you!"
    
    def handle_personal_info_query(self, query: str) -> Dict[str, Any]:
        # Check for personal greetings first
        if self.is_personal_greeting(query):
            return {
                "is_personal_query": True,
                "response": self.get_greeting_response(),
                "query_type": "personal_greeting",
                "confidence": 1.0,
                "should_block": True
            }
        
        # Check for personal identity queries
        if self.is_personal_info_query(query):
            return {
                "is_personal_query": True,
                "response": self.get_redirect_response(query),
                "query_type": "personal_identity",
                "confidence": 1.0,
                "should_block": True
            }
        
        return {"is_personal_query": False, "response": None}
