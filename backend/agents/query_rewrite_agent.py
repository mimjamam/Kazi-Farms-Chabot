"""
Query Rewrite Agent - Advanced query rewriting with fuzzy matching and dynamic spell correction
"""
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
import re
from difflib import get_close_matches

@dataclass
class QueryRewriteResult:
    """Result from query rewriting"""
    original_query: str
    rewritten_query: str
    rewrite_type: str
    confidence: float
    reasoning: str
    corrections_applied: list

class QueryRewriteAgent:
    """Agent responsible for rewriting queries for better retrieval with dynamic spell correction"""
    
    def __init__(self):
        self.is_initialized = True
        
        # Base common typos
        self.base_typos = {
            'adress': 'address',
            'recieve': 'receive',
            'seperate': 'separate',
            'occured': 'occurred',
            'definately': 'definitely',
            'accomodate': 'accommodate',
            'begining': 'beginning',
            'calender': 'calendar',
            'cemetary': 'cemetery',
            'comming': 'coming',
            'differnt': 'different',
            'enviroment': 'environment',
            'existance': 'existence',
            'futher': 'further',
            'goverment': 'government',
            'independant': 'independent',
            'lenght': 'length',
            'occurence': 'occurrence',
            'priviledge': 'privilege',
            'recomend': 'recommend',
            'rythm': 'rhythm',
            'succesful': 'successful',
            'thier': 'their',
            'untill': 'until',
            'usefull': 'useful',
            'writting': 'writing',
            'acheive': 'achieve',
            'beleive': 'believe',
            'neccessary': 'necessary',
            'occassion': 'occasion',
            'reccomend': 'recommend',
            'succes': 'success',
            'teh': 'the',
            'wich': 'which',
            'wont': "won't",
            'dont': "don't",
            'cant': "can't",
            'isnt': "isn't",
            'arent': "aren't",
            'wasnt': "wasn't",
            'werent': "weren't",
            'havent': "haven't",
            'hasnt': "hasn't",
            'hadnt': "hadn't",
            'wouldnt': "wouldn't",
            'couldnt': "couldn't",
            'shouldnt': "shouldn't"
        }
        
        # Base domain variations (can be extended dynamically)
        self.base_variations = {
            'hr': 'human resources',
            'hr dept': 'human resources department',
            'hr dept.': 'human resources department',
            'hr department': 'human resources department',
            'pay': 'salary',
            'wage': 'salary',
            'compensation': 'salary',
            'leave': 'vacation',
            'holiday': 'vacation',
            'off': 'vacation',
            'time off': 'vacation',
            'pto': 'paid time off',
            'sick leave': 'sick time',
            'personal leave': 'personal time',
            'emergency leave': 'emergency time'
        }
        
        # Advanced query rewrite patterns
        self.common_map = {
            "quater": "quarter", "head quater": "headquarters", "adress": "address", 
            "hq": "head office", "recieve": "receive", "seperate": "separate",
            "occured": "occurred", "definately": "definitely", "accomodate": "accommodate",
            "begining": "beginning", "calender": "calendar", "cemetary": "cemetery",
            "comming": "coming", "differnt": "different", "enviroment": "environment",
            "existance": "existence", "futher": "further", "goverment": "government",
            "independant": "independent", "lenght": "length", "occurence": "occurrence",
            "priviledge": "privilege", "recomend": "recommend", "rythm": "rhythm",
            "succesful": "successful", "thier": "their", "untill": "until",
            "usefull": "useful", "writting": "writing", "acheive": "achieve",
            "beleive": "believe", "neccessary": "necessary", "occassion": "occasion",
            "reccomend": "recommend", "succes": "success", "teh": "the",
            "wich": "which", "wont": "won't", "dont": "don't", "cant": "can't",
            "isnt": "isn't", "arent": "aren't", "wasnt": "wasn't", "werent": "weren't",
            "havent": "haven't", "hasnt": "hasn't", "hadnt": "hadn't",
            "wouldnt": "wouldn't", "couldnt": "couldn't", "shouldnt": "shouldn't"
        }
        
        # Common words for fuzzy matching
        self.common_words = [
            "address", "headquarters", "office", "contact", "location", "attendance", 
            "policy", "salary", "leave", "vacation", "benefit", "employee", "hr",
            "human resources", "department", "procedure", "guideline", "requirement",
            "training", "development", "performance", "review", "promotion", "raise",
            "recruitment", "hiring", "interview", "onboarding", "orientation", "probation"
        ]
        
        pass  # Silent initialization
    
    def _is_context_relevant(self, current_query: str, conversation_context: str) -> bool:
        """Determine if conversation context is relevant to current query"""
        if not conversation_context or not current_query:
            return False
        
        current_query_lower = current_query.lower()
        context_lower = conversation_context.lower()
        
        # Extract topics from both current query and context
        current_topics = self._extract_query_topics(current_query_lower)
        context_topics = self._extract_context_topics(context_lower)
        
        # Check for topic overlap
        topic_overlap = len(current_topics.intersection(context_topics))
        
        # Check for affirmative responses that might refer to previous context
        affirmative_responses = ['yes', 'yeah', 'yep', 'ok', 'okay', 'sure', 'please', 'go ahead', 'continue']
        is_affirmative = current_query_lower.strip() in affirmative_responses
        
        # Check for continuation phrases
        continuation_phrases = ['more about', 'tell me more', 'what about', 'also', 'and', 'additionally']
        has_continuation = any(phrase in current_query_lower for phrase in continuation_phrases)
        
        # Context is relevant if:
        # 1. There's topic overlap
        # 2. It's an affirmative response (likely continuing previous topic)
        # 3. It contains continuation phrases
        return topic_overlap > 0 or is_affirmative or has_continuation
    
    def _extract_query_topics(self, query: str) -> set:
        """Extract topics from current query"""
        topics = set()
        
        # Define topic keywords
        topic_keywords = {
            'salary': ['salary', 'pay', 'compensation', 'wage', 'allowance', 'income', 'earning'],
            'leave': ['leave', 'vacation', 'holiday', 'sick', 'absence', 'time off', 'break'],
            'benefits': ['benefit', 'insurance', 'medical', 'health', 'coverage', 'plan'],
            'policies': ['policy', 'rule', 'procedure', 'guideline', 'regulation', 'protocol'],
            'positions': ['manager', 'officer', 'position', 'role', 'job', 'designation', 'title'],
            'performance': ['performance', 'review', 'appraisal', 'evaluation', 'assessment'],
            'training': ['training', 'development', 'course', 'workshop', 'education'],
            'recruitment': ['recruitment', 'hiring', 'interview', 'candidate', 'application'],
            'office': ['office', 'desk', 'workplace', 'facility', 'building', 'location'],
            'cleaning': ['cleaning', 'maintenance', 'housekeeping', 'sanitation', 'hygiene']
        }
        
        for topic, keywords in topic_keywords.items():
            if any(keyword in query for keyword in keywords):
                topics.add(topic)
        
        return topics
    
    def _extract_context_topics(self, context: str) -> set:
        """Extract topics from conversation context"""
        topics = set()
        
        # Look for explicit topic mentions in context
        if 'salary' in context or 'pay' in context or 'compensation' in context:
            topics.add('salary')
        if 'leave' in context or 'vacation' in context or 'holiday' in context:
            topics.add('leave')
        if 'benefit' in context or 'insurance' in context or 'medical' in context:
            topics.add('benefits')
        if 'policy' in context or 'procedure' in context or 'rule' in context:
            topics.add('policies')
        if 'position' in context or 'manager' in context or 'officer' in context:
            topics.add('positions')
        if 'performance' in context or 'review' in context or 'appraisal' in context:
            topics.add('performance')
        if 'training' in context or 'development' in context:
            topics.add('training')
        if 'recruitment' in context or 'hiring' in context:
            topics.add('recruitment')
        if 'office' in context or 'desk' in context or 'workplace' in context:
            topics.add('office')
        if 'cleaning' in context or 'maintenance' in context:
            topics.add('cleaning')
        
        return topics
    
    def rewrite_query(self, query: str, context: Optional[Dict[str, Any]] = None) -> QueryRewriteResult:
        """Rewrite query for better retrieval with spell correction"""
        try:
            original_query = query.strip()
            rewritten_query = original_query.lower()
            corrections_applied = []
            
            # Step 1: Rule-based correction (immediate, cheap)
            rule_corrected = self._rule_rewrite(rewritten_query)
            if rule_corrected != rewritten_query:
                corrections_applied.append(f"Rule correction: {rewritten_query} → {rule_corrected}")
                rewritten_query = rule_corrected
            
            # Step 2: Fuzzy matching correction
            fuzzy_corrected = self._fuzzy_rewrite(rewritten_query)
            if fuzzy_corrected != rewritten_query:
                corrections_applied.append(f"Fuzzy correction: {rewritten_query} → {fuzzy_corrected}")
                rewritten_query = fuzzy_corrected
            
            # Step 3: Dynamic spell correction
            corrected_query = self._correct_spelling(rewritten_query)
            if corrected_query != rewritten_query:
                corrections_applied.append(f"Dynamic spell correction: {rewritten_query} → {corrected_query}")
                rewritten_query = corrected_query
            
            # Step 4: Add context only if relevant
            if context and context.get('conversation_context'):
                # Check if current query is related to previous context
                if self._is_context_relevant(rewritten_query, context.get('conversation_context', '')):
                    rewritten_query = f"{rewritten_query} {context['conversation_context']}"
                    corrections_applied.append("Added relevant conversation context")
                else:
                    corrections_applied.append("Skipped irrelevant conversation context")
            
            # Step 5: Normalize common variations
            normalized_query = self._normalize_variations(rewritten_query)
            if normalized_query != rewritten_query:
                corrections_applied.append(f"Normalized variations: {rewritten_query} → {normalized_query}")
                rewritten_query = normalized_query
            
            # Determine rewrite type
            rewrite_type = "basic"
            if corrections_applied:
                rewrite_type = "corrected"
            if len(rewritten_query) > len(original_query):
                rewrite_type = "expanded"
            
            return QueryRewriteResult(
                original_query=original_query,
                rewritten_query=rewritten_query,
                rewrite_type=rewrite_type,
                confidence=0.9,
                reasoning=f"Query processed with {len(corrections_applied)} corrections",
                corrections_applied=corrections_applied
            )
            
        except Exception as e:
            print(f"[QUERY REWRITE AGENT ERROR] {str(e)}")
            return QueryRewriteResult(
                original_query=query,
                rewritten_query=query,
                rewrite_type="error",
                confidence=0.1,
                reasoning=f"Error in query rewriting: {str(e)}",
                corrections_applied=[]
            )
    
    def _correct_spelling(self, query: str) -> str:
        """Correct spelling mistakes using typo dictionary"""
        words = query.split()
        corrected_words = []
        
        for word in words:
            # Remove punctuation for checking
            clean_word = re.sub(r'[^\w]', '', word.lower())
            if clean_word in self.base_typos:
                corrected_word = word.replace(clean_word, self.base_typos[clean_word])
                corrected_words.append(corrected_word)
            else:
                corrected_words.append(word)
        
        return ' '.join(corrected_words)
    
    def _normalize_variations(self, query: str) -> str:
        """Normalize query variations"""
        normalized_query = query
        for variation, standard in self.base_variations.items():
            if variation in normalized_query:
                normalized_query = normalized_query.replace(variation, standard)
        
        return normalized_query
    
    def _rule_rewrite(self, query: str) -> str:
        """Rule-based query correction using common mappings"""
        q = query
        for wrong, good in self.common_map.items():
            q = re.sub(re.escape(wrong), good, q, flags=re.IGNORECASE)
        return q
    
    def _fuzzy_correct_word(self, word: str) -> str:
        """Fuzzy correct a single word using close matches"""
        matches = get_close_matches(word, self.common_words, n=1, cutoff=0.78)
        return matches[0] if matches else word
    
    def _fuzzy_rewrite(self, query: str) -> str:
        """Fuzzy rewrite using close matches for common words"""
        tokens = query.split()
        tokens = [self._fuzzy_correct_word(t) for t in tokens]
        return " ".join(tokens)
