import random
from typing import List, Dict, Any

class FunnyFallbackAgent:
    def __init__(self):
        self.funny_responses = {
            'no_context': [
                "That's a question even our smartest rooster hasn't heard before! I don't have that info in my database, but I'm sure our HR team would love to help you out!",
                "Oops! That's not something I've learned from our Kazifarm documents. Even chickens know their limits! But hey, our customer service team is always ready to assist!",
                "That's beyond my chicken brain's knowledge! I don't have that information, but our friendly staff at Kazi Farms would be happy to help!",
                "I'm stumped! That's not in my Kazifarm knowledge base. Even our wisest hen would be confused! Try reaching out to our support team!",
                "That's a question that would make any chicken scratch their head! I don't have that info in my database, but our team is here to help!",
                "My database is clucking empty on that topic! That's not something I've learned from our Kazifarm documents. Contact us for more details!"
            ],
            'low_confidence': [
                "I found some related info, but I'm not confident enough to give you a proper answer. Even chickens need to be sure before they cluck!",
                "I'm like a chicken with a half-hatched egg - I have some info but it's not complete! Let me connect you with someone who knows better!",
                "I'm not 100% sure about this one. Even our most confident rooster would want to double-check!",
                "I have some information, but I'd rather not give you half-baked answers. Our team can provide you with accurate details!",
                "I'm as uncertain as a chicken in a new coop! I have some info but I'm not confident it's complete. Let our experts help you!",
                "My confidence is lower than a chicken's flight altitude! I found some related content but I'd recommend checking with our team!"
            ],
            'general_help': [
                "I'm here to help with Kazi Farms information! Try asking me about salaries, allowances, policies, or leave information!",
                "I'm as helpful as a mother hen with her chicks! I can assist with HR policies, employee benefits, and company information!",
                "I'm ready to help! Ask me about our policies, procedures, or any Kazi Farms related questions!",
                "I'm your friendly Kazi Farms assistant! I can help with employee information, policies, and company details!",
                "I'm here to help you navigate through Kazi Farms information! Try asking about specific policies or procedures!",
                "I'm as knowledgeable as a rooster about morning calls! I can help with HR queries, policies, and company information!"
            ],
            'personal_identity': [
                "I can't provide personal information about users. I'm here to help with Kazi Farms HR policies, salary information, and company procedures.",
                "I don't have access to personal user information. I can assist you with company policies, employee benefits, and HR-related questions.",
                "I'm designed to help with Kazi Farms information, not personal identity questions. Please ask about our policies, procedures, or employee benefits.",
                "I can't identify users or provide personal information. I'm here to help with Kazi Farms HR and company information."
            ],
            'personal_greeting': [
                "I'm doing great! I'm a chatbot designed to help with Kazi Farms information. How can I assist you with our HR policies, salary information, or company procedures?",
                "I'm functioning perfectly! I'm here to help you with Kazi Farms policies and procedures. What would you like to know about our company?",
                "I'm running smoothly! I'm a Kazi Farms assistant ready to help with employee information, policies, and company details. What can I help you with?",
                "I'm doing well! I'm designed to assist with Kazi Farms HR and company information. How can I help you today?",
                "I'm great! I'm a chatbot specialized in Kazi Farms information. I can help with salary structures, allowances, leave policies, and more. What do you need to know?",
                "I'm excellent! I'm here to help with Kazi Farms policies and procedures. What information can I provide for you today?"
            ],
            'hr_contact': [
                "For HR-related inquiries, please contact our HR Department directly. I can help you with HR policies, salary information, and company procedures from our knowledge base.",
                "I can assist with HR policies and procedures from our documents. For specific HR contact information, please reach out to our HR Department.",
                "I'm here to help with HR policies and information from our knowledge base. For direct HR contact, please contact our HR Department.",
                "I can provide information about HR policies and procedures. For HR contact details, please contact our HR Department directly."
            ],
            'irrelevant_question': [
                "I'm a Kazi Farms chatbot, not a sports commentator! I can tell you about supervisor salaries, but not football captains!",
                "That's not in my Kazi Farms database! I'm more of a 'salary structures and leave policies' kind of chatbot!",
                "I'm clucking with confusion! I know everything about Kazi Farms policies, but nothing about sports teams!",
                "I'm as lost as a chicken in a library! I can help with employee benefits, but not with general knowledge questions!",
                "I'm specialized in Kazi Farms information, not sports trivia! I can tell you about allowances, but not about team captains!",
                "I'm a Kazi Farms chatbot, not a general knowledge quiz master! I can help with company procedures though!",
                "I'm as confused as a rooster at a cat convention! I know Kazi Farms inside out, but not sports or general topics!",
                "I'm a Kazi Farms chatbot, not a sports encyclopedia! I can help with salary information, but not team rosters!",
                "I'm clueless about that topic, but I'm an expert on Kazi Farms policies and procedures!",
                "I'm as helpful as a mother hen with Kazi Farms info, but completely useless with sports questions!"
            ]
        }
        
        self.contact_info = {
            'hr_department': "HR Department: +880-XXX-XXXXXXX",
            'customer_service': "Customer Service: +880-XXX-XXXXXXX", 
            'email': "Email: info@kazifarms.com",
            'website': "Website: www.kazifarms.com",
            'office_hours': "Office Hours: 9:00 AM - 5:00 PM (Sunday-Thursday)"
        }
        
        self.suggested_topics = [
            "Salary structures for different positions",
            "Allowance policies (house, transport, medical, etc.)",
            "Leave policies (sick, annual, casual, etc.)",
            "Employee benefits and bonuses",
            "HR policies and procedures",
            "Company locations and departments",
            "Job roles and responsibilities",
            "Performance evaluation processes"
        ]
    
    def get_funny_response(self, response_type: str = 'no_context') -> str:
        responses = self.funny_responses.get(response_type, self.funny_responses['no_context'])
        return random.choice(responses)
    
    def generate_fallback_response(self, query: str, confidence: float = 0.0, context_type: str = 'no_context') -> str:
        funny_part = self.get_funny_response(context_type)
        
        # For personal questions, HR contact questions, and irrelevant questions, don't add suggestions or contact info
        if context_type in ['personal_identity', 'personal_greeting', 'hr_contact', 'irrelevant_question']:
            return funny_part
        
        suggestions = self._get_helpful_suggestions(query)
        
        contact_section = self._format_contact_info()
        
        response = f"{funny_part}\n\n"
        
        if suggestions:
            response += f"Here's what I can help you with:\n"
            for suggestion in suggestions[:3]:
                response += f"• {suggestion}\n"
            response += "\n"
        
        response += f"Need more help?\n{contact_section}\n"
        
        # Log confidence note to terminal instead of showing to user
        if confidence > 0:
            print(f"[DEBUG] Confidence note: I found some related information (confidence: {confidence:.1f}%) but it's not specific enough to give you a complete answer.")
        
        return response
    
    def _get_helpful_suggestions(self, query: str) -> List[str]:
        query_lower = query.lower()
        
        if any(word in query_lower for word in ['salary', 'pay', 'wage', 'money']):
            return [
                "Salary structures for Management Trainees",
                "Salary scales for different job groups",
                "Yearly increment policies",
                "Performance-based salary reviews"
            ]
        elif any(word in query_lower for word in ['allowance', 'benefit', 'perk']):
            return [
                "House allowance for different locations",
                "Transport allowance policies",
                "Medical allowance benefits",
                "Production and performance bonuses"
            ]
        elif any(word in query_lower for word in ['leave', 'vacation', 'holiday', 'off']):
            return [
                "Sick leave policies and procedures",
                "Annual leave entitlements",
                "Casual leave guidelines",
                "Maternity and paternity leave"
            ]
        elif any(word in query_lower for word in ['policy', 'rule', 'regulation']):
            return [
                "HR policies and procedures",
                "Employee conduct policies",
                "Safety and security policies",
                "Company regulations and guidelines"
            ]
        else:
            return random.sample(self.suggested_topics, 3)
    
    def _format_contact_info(self) -> str:
        contact_text = ""
        for key, value in self.contact_info.items():
            contact_text += f"{value}\n"
        return contact_text.strip()
    
    def analyze_query_context(self, query: str, search_results: List[Any], confidence: float) -> str:
        query_lower = query.lower()
        
        # Check for personal identity questions first
        personal_identity_keywords = [
            'who am i', 'who are you', 'identify me', 'my identity', 'personal information',
            'my name', 'my email', 'my details', 'about me', 'tell me about myself'
        ]
        
        for keyword in personal_identity_keywords:
            if keyword in query_lower:
                return self.generate_fallback_response(query, confidence, 'personal_identity')
        
        # Check for personal greetings
        personal_greeting_keywords = [
            'how are you', 'how do you do', 'how\'s it going', 'how are things', 
            'what\'s up', 'how\'s your day', 'are you okay', 'are you fine',
            'how are you doing', 'how\'s everything', 'how\'s life'
        ]
        
        for keyword in personal_greeting_keywords:
            if keyword in query_lower:
                return self.generate_fallback_response(query, confidence, 'personal_greeting')
        
        # Check for HR contact requests
        hr_contact_keywords = [
            'hr email', 'hr contact', 'hr department', 'hr phone', 'hr number', 'contact hr', 
            'hr address', 'hr office', 'hr manager', 'hr director', 'give me email of hr'
        ]
        
        for keyword in hr_contact_keywords:
            if keyword in query_lower:
                return self.generate_fallback_response(query, confidence, 'hr_contact')
        
        if not search_results or confidence < 0.1:
            return self.generate_fallback_response(query, confidence, 'no_context')
        elif confidence < 0.3:
            return self.generate_fallback_response(query, confidence, 'low_confidence')
        else:
            return self.generate_fallback_response(query, confidence, 'general_help')
    
    def get_encouragement_message(self) -> str:
        encouragements = [
            "Don't worry! Even the best chickens need help sometimes!",
            "Every question is a good question - that's how we learn!",
            "Keep asking! Curiosity is the first step to knowledge!",
            "Your question helps us improve our services!",
            "We're here to help you succeed at Kazi Farms!",
            "Every employee question matters to us!"
        ]
        return random.choice(encouragements)
