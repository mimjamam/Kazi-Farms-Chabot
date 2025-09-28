import re
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

@dataclass
class QueryAnalysis:
    original_query: str
    query_type: str
    extracted_info: Dict[str, str]
    missing_info: List[str]
    confidence_score: float
    is_complete: bool
    suggested_followup: str

class QueryValidationAgent:
    def __init__(self):
        self.query_patterns = {
            'salary_inquiry': {
                'keywords': ['salary', 'pay', 'wage', 'compensation', 'income', 'earnings', 'increment', 'revision'],
                'required_info': ['designation', 'job_title', 'position', 'role', 'job_group'],
                'optional_info': ['department', 'level', 'experience', 'location']
            },
            'allowance_inquiry': {
                'keywords': ['allowance', 'benefit', 'perk', 'bonus', 'incentive', 'subsidy'],
                'required_info': ['allowance_type'],
                'optional_info': ['employee_category', 'location', 'department']
            },
            'policy_inquiry': {
                'keywords': ['policy', 'rule', 'regulation', 'procedure', 'guideline', 'standard'],
                'required_info': ['policy_area'],
                'optional_info': ['employee_type', 'department', 'location']
            },
            'leave_inquiry': {
                'keywords': ['leave', 'vacation', 'holiday', 'off', 'absence', 'break'],
                'required_info': ['leave_type'],
                'optional_info': ['employee_category', 'duration']
            },
            'hr_inquiry': {
                'keywords': ['hr', 'human resource', 'recruitment', 'hiring', 'employee', 'staff'],
                'required_info': ['hr_area'],
                'optional_info': ['department', 'position']
            },
            'hr_contact': {
                'keywords': ['hr email', 'hr contact', 'hr department', 'hr phone', 'hr number', 'contact hr', 'hr address', 'hr office', 'hr manager', 'hr director'],
                'required_info': [],
                'optional_info': []
            },
            'general_inquiry': {
                'keywords': ['what', 'how', 'when', 'where', 'why', 'tell me', 'explain'],
                'required_info': [],
                'optional_info': []
            }
        }
        
        self.funny_replies = {
            'salary_inquiry': [
                "I need more details about your role to give you the right salary information.",
                "Please specify your job designation for accurate salary details.",
                "I have salary data but need to know your position in the hierarchy.",
                "Please tell me if you're a Farm Manager, Hatchery Supervisor, or another role."
            ],
            'allowance_inquiry': [
                "I have allowance information but need to know which specific allowance you're asking about.",
                "Please specify which allowance you're interested in.",
                "I can help with allowance details but need the specific type.",
                "Which allowance policy would you like to know about?"
            ],
            'policy_inquiry': [
                "I have policy information but need to know which specific policy area you're asking about.",
                "Please specify which policy you're interested in.",
                "I can help with policies but need the specific area.",
                "Which policy would you like to know about?"
            ],
            'leave_inquiry': [
                "I can help with leave policies but need to know which type of leave you're asking about.",
                "Please specify which leave type you're interested in.",
                "I have leave information but need the specific type.",
                "Which leave policy would you like to know about?"
            ],
            'hr_inquiry': [
                "I can help with HR information but need to know which specific area you're asking about.",
                "Please specify which HR area you're interested in.",
                "I have HR information but need the specific topic.",
                "Which HR area would you like to know about?"
            ],
            'hr_contact': [
                "For HR-related inquiries, please contact our HR Department directly. I can help you with HR policies, salary information, and company procedures from our knowledge base.",
                "I can assist with HR policies and procedures from our documents. For specific HR contact information, please reach out to our HR Department.",
                "I'm here to help with HR policies and information from our knowledge base. For direct HR contact, please contact our HR Department.",
                "I can provide information about HR policies and procedures. For HR contact details, please contact our HR Department directly."
            ],
            'general_inquiry': [
                "I'm here to help but need more specific questions to give you accurate answers.",
                "Please be more specific about what you need.",
                "I'm ready to help but need clearer questions.",
                "I need more details to give you the right information."
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
            'no_match': [
                "I don't have that information in my database.",
                "That's not something I've learned from our Kazi Farms documents.",
                "I don't have that information available.",
                "That's not in my Kazi Farms knowledge base.",
                "I don't have that info in my database.",
                "That's not something I've learned from our Kazi Farms documents."
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
    
    def analyze_query(self, query: str) -> QueryAnalysis:
        query_lower = query.lower()
        query_type = self._classify_query_type(query_lower)
        extracted_info = self._extract_information(query_lower, query_type)
        missing_info = self._identify_missing_info(query_type, extracted_info)
        confidence_score = self._calculate_confidence(query_lower, query_type, extracted_info)
        is_complete = len(missing_info) == 0
        suggested_followup = self._generate_followup(query_type, missing_info)
        
        return QueryAnalysis(
            original_query=query,
            query_type=query_type,
            extracted_info=extracted_info,
            missing_info=missing_info,
            confidence_score=confidence_score,
            is_complete=is_complete,
            suggested_followup=suggested_followup
        )
    
    def _classify_query_type(self, query: str) -> str:
        query_lower = query.lower()
        
        # Check for personal identity questions first
        personal_identity_keywords = [
            'who am i', 'who are you', 'identify me', 'my identity', 'personal information',
            'my name', 'my email', 'my details', 'about me', 'tell me about myself'
        ]
        
        for keyword in personal_identity_keywords:
            if keyword in query_lower:
                return 'personal_identity'
        
        # Check for personal greetings
        personal_greeting_keywords = [
            'how are you', 'how do you do', 'how\'s it going', 'how are things', 
            'what\'s up', 'how\'s your day', 'are you okay', 'are you fine',
            'how are you doing', 'how\'s everything', 'how\'s life'
        ]
        
        for keyword in personal_greeting_keywords:
            if keyword in query_lower:
                return 'personal_greeting'
        
        # Check other query types
        for query_type, pattern_info in self.query_patterns.items():
            for keyword in pattern_info['keywords']:
                if keyword in query:
                    return query_type
        
        return 'general_inquiry'
    
    def _extract_information(self, query: str, query_type: str) -> Dict[str, str]:
        extracted = {}
        
        if query_type == 'salary_inquiry':
            kazi_job_roles = [
                'management trainee', 'sales person', 'farm manager', 'hatchery supervisor',
                'feed mill manager', 'production manager', 'accountant', 'driver', 'helper',
                'mechanic', 'farm in-charge', 'commercial manager', 'hr manager', 'admin officer',
                'finance manager', 'quality manager', 'maintenance manager', 'security guard',
                'cleaner', 'operator', 'technician', 'supervisor', 'officer', 'executive',
                'manager', 'assistant manager', 'deputy manager', 'general manager'
            ]
            
            for role in kazi_job_roles:
                if role in query.lower():
                    extracted['designation'] = role.title()
                    break
            
            job_groups = ['job group 1', 'job group 2', 'job group 3', 'job group 4', 'job group 5']
            for group in job_groups:
                if group in query.lower():
                    extracted['job_group'] = group
                    break
            
            kazi_departments = [
                'hatchery', 'farm', 'feed mill', 'sales', 'marketing', 'hr', 'finance',
                'production', 'quality', 'maintenance', 'transport', 'commercial',
                'panchagarh', 'thakurgaon', 'gojaria', 'sagarica', 'kfg', 'kml', 'kfil'
            ]
            
            for dept in kazi_departments:
                if dept in query.lower():
                    extracted['department'] = dept.title()
                    break
            
            if 'management' in query.lower():
                extracted['employee_category'] = 'Management'
            elif 'non-management' in query.lower() or 'worker' in query.lower():
                extracted['employee_category'] = 'Non-Management'
        
        elif query_type == 'allowance_inquiry':
            kazi_allowances = [
                'house allowance', 'location allowance', 'transport allowance', 'medical allowance',
                'food allowance', 'hair cutting allowance', 'time keeping allowance', 'overtime allowance',
                'night allowance', 'ta da allowance', 'fuel allowance', 'uniform allowance',
                'guard allowance', 'furniture allowance', 'mobile allowance', 'pick drop allowance',
                'production bonus', 'performance bonus', 'eid bonus', 'incentive', 'reliever allowance'
            ]
            
            for allowance in kazi_allowances:
                if allowance in query.lower():
                    extracted['allowance_type'] = allowance.title()
                    break
        
        elif query_type == 'policy_inquiry':
            kazi_policies = [
                'leave policy', 'retirement policy', 'recruitment policy', 'transfer policy',
                'performance policy', 'overtime policy', 'bonus policy', 'allowance policy',
                'travel policy', 'uniform policy', 'mobile policy', 'car policy',
                'office time policy', 'deduction policy', 'notice pay policy'
            ]
            
            for policy in kazi_policies:
                if policy in query.lower():
                    extracted['policy_area'] = policy.title()
                    break
        
        elif query_type == 'leave_inquiry':
            kazi_leave_types = [
                'sick leave', 'annual leave', 'casual leave', 'maternity leave', 'paternity leave',
                'emergency leave', 'replacement leave', 'off day', 'holiday', 'vacation'
            ]
            
            for leave_type in kazi_leave_types:
                if leave_type in query.lower():
                    extracted['leave_type'] = leave_type.title()
                    break
        
        elif query_type == 'hr_inquiry':
            kazi_hr_areas = [
                'recruitment', 'hiring', 'training', 'performance management', 'appraisal',
                'promotion', 'transfer', 'resignation', 'termination', 'employee relations',
                'compensation', 'benefits', 'payroll', 'attendance', 'discipline'
            ]
            
            for hr_area in kazi_hr_areas:
                if hr_area in query.lower():
                    extracted['hr_area'] = hr_area.title()
                    break
        
        return extracted
    
    def _identify_missing_info(self, query_type: str, extracted_info: Dict[str, str]) -> List[str]:
        if query_type not in self.query_patterns:
            return []
        
        required_info = self.query_patterns[query_type]['required_info']
        missing = []
        
        for info in required_info:
            if info not in extracted_info or not extracted_info[info]:
                missing.append(info)
        
        return missing
    
    def _calculate_confidence(self, query: str, query_type: str, extracted_info: Dict[str, str]) -> float:
        base_confidence = 0.5
        info_score = len(extracted_info) * 0.2
        
        type_scores = {
            'salary_inquiry': 0.3,
            'leave_policy': 0.2,
            'benefits': 0.2,
            'hr_policy': 0.2,
            'general_inquiry': 0.1
        }
        
        type_score = type_scores.get(query_type, 0.1)
        return min(1.0, base_confidence + info_score + type_score)
    
    def _generate_followup(self, query_type: str, missing_info: List[str]) -> str:
        if not missing_info:
            return ""
        
        followup_templates = {
            'salary_inquiry': "Please specify the job designation (e.g., 'Management Trainee', 'Farm Manager', 'Hatchery Supervisor', 'Sales Person') or job group (Job Group 1-5).",
            'allowance_inquiry': "Please specify the type of allowance (e.g., 'House Allowance', 'Location Allowance', 'Transport Allowance', 'Medical Allowance', 'Production Bonus').",
            'policy_inquiry': "Please specify which policy area you're interested in (e.g., 'Leave Policy', 'Retirement Policy', 'Recruitment Policy', 'Performance Policy').",
            'leave_inquiry': "Please specify the type of leave (e.g., 'Sick Leave', 'Annual Leave', 'Casual Leave', 'Maternity Leave', 'Off Day').",
            'hr_inquiry': "Please specify the HR area (e.g., 'Recruitment', 'Performance Management', 'Training', 'Compensation', 'Benefits').",
            'hr_contact': "I can help with HR policies and procedures from our knowledge base."
        }
        
        return followup_templates.get(query_type, "Please provide more specific information about your query.")
    
    def get_funny_reply(self, query_type: str = 'no_match') -> str:
        import random
        
        # For personal questions and irrelevant questions, use appropriate responses
        if query_type in ['personal_identity', 'personal_greeting', 'irrelevant_question']:
            replies = self.funny_replies[query_type]
        else:
            replies = self.funny_replies.get(query_type, self.funny_replies['no_match'])
        return random.choice(replies)
    
    def get_funny_reply_with_guidance(self, query_type: str, followup_suggestion: str) -> str:
        import random
        
        funny_reply = self.get_funny_reply(query_type)
        
        # For personal questions and irrelevant questions, don't add followup suggestions
        if query_type in ['personal_identity', 'personal_greeting', 'irrelevant_question']:
            return funny_reply
        
        if followup_suggestion:
            return f"{funny_reply}\n\nBut here's how I can help: {followup_suggestion}"
        else:
            return funny_reply
    
    def validate_query_completeness(self, query: str) -> Tuple[bool, str]:
        analysis = self.analyze_query(query)
        
        if analysis.is_complete:
            return True, ""
        
        return False, analysis.suggested_followup
