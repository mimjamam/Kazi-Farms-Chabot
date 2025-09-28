import re
import difflib
from typing import List, Tuple, Dict, Any
from dataclasses import dataclass

@dataclass
class MatchResult:
    confidence: float
    matched_content: str
    match_type: str
    keywords_found: List[str]
    similarity_score: float
    is_reliable: bool

class SimpleQueryMatcher:
    def __init__(self):
        self.stop_words = {
            'a', 'an', 'and', 'are', 'as', 'at', 'be', 'by', 'for', 'from',
            'has', 'he', 'in', 'is', 'it', 'its', 'of', 'on', 'that', 'the',
            'to', 'was', 'will', 'with', 'i', 'you', 'we', 'they', 'this',
            'these', 'those', 'have', 'had', 'do', 'does', 'did', 'can',
            'could', 'would', 'should', 'may', 'might', 'must', 'shall',
            'what', 'when', 'where', 'why', 'how', 'who', 'which'
        }
        
        self.domain_keywords = {
            'chicken': ['chicken', 'poultry', 'bird', 'hen', 'rooster', 'broiler', 'layer', 'egg', 'eggs'],
            'price': ['price', 'cost', 'rate', 'pricing', 'fee', 'charge', 'amount', 'revision', 'refixation'],
            'delivery': ['delivery', 'shipping', 'transport', 'dispatch', 'send', 'pick', 'drop'],
            'order': ['order', 'purchase', 'buy', 'booking', 'reservation'],
            'product': ['product', 'item', 'goods', 'commodity', 'merchandise'],
            'farm': ['farm', 'farmland', 'agriculture', 'farming', 'cultivation', 'farms'],
            'discount': ['discount', 'offer', 'deal', 'promotion', 'sale', 'reduction'],
            'bulk': ['bulk', 'wholesale', 'large quantity', 'mass', 'volume'],
            'quality': ['quality', 'grade', 'standard', 'premium', 'fresh', 'organic'],
            
            'departments': ['hatchery', 'farm', 'feed mill', 'sales', 'marketing', 'hr', 'finance', 'production', 'quality', 'maintenance', 'transport', 'commercial', 'franchise', 'customer service'],
            'locations': ['panchagarh', 'thakurgaon', 'gojaria', 'sagarica', 'kfg', 'kml', 'kfil', 'head office', 'tray factory', 'slaughtering plant', 'egg sales centre'],
            'companies': ['kazi farms', 'kazi feed', 'kazi media', 'sysnova', 'kazi farms limited', 'kazi media limited'],
            
            'employee': ['employee', 'staff', 'worker', 'personnel', 'labor', 'labour', 'workers', 'permanent worker', 'temporary worker'],
            'management': ['management', 'manager', 'supervisor', 'officer', 'executive', 'admin', 'in-charge', 'assistant manager', 'deputy manager', 'general manager', 'agm'],
            'job_groups': ['job group 1', 'job group 2', 'job group 3', 'job group 4', 'job group 5', 'management level', 'non management'],
            'specific_roles': ['management trainee', 'farm manager', 'hatchery supervisor', 'feed mill manager', 'production manager', 'commercial manager', 'hr manager', 'finance manager', 'quality manager', 'maintenance manager', 'farm in-charge', 'sales person', 'accountant', 'driver', 'helper', 'mechanic', 'security guard', 'cleaner', 'operator', 'technician', 'reliever accountant'],
            
            'salary': ['salary', 'wage', 'pay', 'compensation', 'remuneration', 'income', 'earnings', 'minimum salary', 'salary structure', 'pay scale', 'wage structure'],
            'increment': ['increment', 'raise', 'increase', 'promotion', 'advancement', 'upgrade', 'yearly increment', 'salary increment', 'pay increase'],
            'structure': ['structure', 'scale', 'grade', 'level', 'tier', 'bracket', 'range', 'salary structure', 'pay structure'],
            
            'allowance': ['allowance', 'benefit', 'perk', 'bonus', 'incentive', 'subsidy'],
            'specific_allowances': ['house allowance', 'location allowance', 'transport allowance', 'medical allowance', 'food allowance', 'hair cutting allowance', 'time keeping allowance', 'overtime allowance', 'night allowance', 'ta da allowance', 'fuel allowance', 'uniform allowance', 'guard allowance', 'furniture allowance', 'mobile allowance', 'pick drop allowance', 'reliever allowance', 'day off allowance', 'off day allowance', 'transfer allowance'],
            'bonuses': ['production bonus', 'performance bonus', 'eid bonus', 'eidulfitur bonus', 'incentive', 'sysnova incentive'],
            
            'hr': ['hr', 'human resource', 'hrd', 'personnel', 'recruitment', 'hiring', 'hrd head office'],
            'policy': ['policy', 'rule', 'regulation', 'guideline', 'procedure', 'standard', 'circular', 'order', 'notice', 'memo'],
            'specific_policies': ['leave policy', 'retirement policy', 'recruitment policy', 'transfer policy', 'performance policy', 'overtime policy', 'bonus policy', 'allowance policy', 'travel policy', 'uniform policy', 'mobile policy', 'car policy', 'office time policy', 'deduction policy', 'notice pay policy', 'off day policy', 'ta da policy'],
            
            'leave': ['leave', 'vacation', 'holiday', 'off', 'absence', 'break', 'off day', 'replacement leave'],
            'leave_types': ['sick leave', 'annual leave', 'casual leave', 'maternity leave', 'paternity leave', 'emergency leave', 'replacement leave', 'off day', 'holiday', 'vacation'],
            
            'overtime': ['overtime', 'extra', 'additional', 'extended', 'beyond', 'outstation work'],
            'time': ['time', 'schedule', 'office time', 'floor wise', 'time keeping', 'time table'],
            'work': ['work', 'working', 'duty', 'shift', 'tour', 'official tour'],
            
            'transport': ['transport', 'travel', 'commute', 'vehicle', 'car', 'bus', 'pick drop', 'car allowance', 'office car', 'personal car'],
            'vehicles': ['car', 'bus', 'vehicle', 'driver', 'helper', 'mechanic', 'light driver', 'medium driver'],
            
            'medical': ['medical', 'health', 'treatment', 'hospital', 'clinic', 'doctor', 'medical bill', 'h&s team'],
            
            'financial': ['financial', 'payment', 'cash', 'bill', 'claim', 'budget', 'ceiling', 'aid'],
            'payments': ['payment', 'pay', 'cash', 'bill', 'claim', 'fuel bill', 'ta da bill'],
            
            'retirement': ['retirement', 'pension', 'resignation', 'exit', 'departure', 'termination'],
            
            'identity': ['identity card', 'passport', 'document', 'handover'],
            
            'misc': ['picnic', 'budget ceiling', 'sample collection', 'tray factory', 'slaughtering plant', 'egg sales', 'commercial eggs', 'franchise department', 'hardware sales', 'software sales', 'kazi media']
        }
        
        self.question_patterns = [
            r'salary.*structure',
            r'employee.*salary',
            r'pay.*scale',
            r'wage.*structure',
            r'compensation.*structure',
            r'management.*salary',
            r'worker.*salary',
            r'minimum.*salary',
            r'salary.*increment',
            r'yearly.*increment',
            r'pay.*increase',
            r'salary.*revision',
            r'salary.*refixation',
            r'management.*trainee.*salary',
            r'sales.*person.*salary',
            r'farm.*manager.*salary',
            r'hatchery.*supervisor.*salary',
            r'driver.*salary',
            r'permanent.*worker.*salary',
            r'job.*group.*salary',
            r'management.*level.*salary',
            r'non.*management.*salary',
            
            r'house.*allowance',
            r'location.*allowance',
            r'transport.*allowance',
            r'medical.*allowance',
            r'food.*allowance',
            r'hair.*cutting.*allowance',
            r'time.*keeping.*allowance',
            r'overtime.*allowance',
            r'night.*allowance',
            r'ta.*da.*allowance',
            r'fuel.*allowance',
            r'uniform.*allowance',
            r'guard.*allowance',
            r'furniture.*allowance',
            r'mobile.*allowance',
            r'pick.*drop.*allowance',
            r'reliever.*allowance',
            r'day.*off.*allowance',
            r'off.*day.*allowance',
            r'transfer.*allowance',
            r'car.*allowance',
            
            r'production.*bonus',
            r'performance.*bonus',
            r'eid.*bonus',
            r'eidulfitur.*bonus',
            r'sysnova.*incentive',
            r'farm.*bonus',
            r'hatchery.*bonus',
            
            r'leave.*policy',
            r'retirement.*policy',
            r'recruitment.*policy',
            r'transfer.*policy',
            r'performance.*policy',
            r'overtime.*policy',
            r'bonus.*policy',
            r'allowance.*policy',
            r'travel.*policy',
            r'uniform.*policy',
            r'mobile.*policy',
            r'car.*policy',
            r'office.*time.*policy',
            r'deduction.*policy',
            r'notice.*pay.*policy',
            r'off.*day.*policy',
            r'ta.*da.*policy',
            
            r'hatchery.*allowance',
            r'farm.*allowance',
            r'feed.*mill.*allowance',
            r'panchagarh.*allowance',
            r'thakurgaon.*allowance',
            r'gojaria.*allowance',
            r'sagarica.*allowance',
            r'kfg.*allowance',
            r'kml.*allowance',
            r'kfil.*allowance',
            r'head.*office.*allowance',
            r'tray.*factory.*allowance',
            r'slaughtering.*plant.*allowance',
            r'egg.*sales.*allowance',
            r'commercial.*eggs.*allowance',
            r'franchise.*department.*allowance',
            r'hardware.*sales.*allowance',
            r'software.*sales.*allowance',
            r'kazi.*media.*allowance',
            
            r'management.*trainee.*allowance',
            r'farm.*manager.*allowance',
            r'hatchery.*supervisor.*allowance',
            r'feed.*mill.*manager.*allowance',
            r'production.*manager.*allowance',
            r'commercial.*manager.*allowance',
            r'hr.*manager.*allowance',
            r'finance.*manager.*allowance',
            r'quality.*manager.*allowance',
            r'maintenance.*manager.*allowance',
            r'farm.*in.*charge.*allowance',
            r'sales.*person.*allowance',
            r'accountant.*allowance',
            r'driver.*allowance',
            r'helper.*allowance',
            r'mechanic.*allowance',
            r'security.*guard.*allowance',
            r'cleaner.*allowance',
            r'operator.*allowance',
            r'technician.*allowance',
            r'reliever.*accountant.*allowance',
            
            r'sick.*leave',
            r'annual.*leave',
            r'casual.*leave',
            r'maternity.*leave',
            r'paternity.*leave',
            r'emergency.*leave',
            r'replacement.*leave',
            r'off.*day',
            r'holiday.*policy',
            r'vacation.*policy',
            r'office.*time',
            r'floor.*wise.*office.*time',
            r'time.*keeping',
            r'time.*table',
            r'outstation.*work',
            r'official.*tour',
            
            r'transport.*support',
            r'car.*allowance',
            r'office.*car',
            r'personal.*car',
            r'light.*driver',
            r'medium.*driver',
            r'driver.*helper',
            r'driver.*mechanic',
            r'vehicle.*allowance',
            r'pick.*drop.*service',
            
            r'medical.*bill',
            r'h.*s.*team',
            r'health.*safety',
            r'medical.*treatment',
            r'hospital.*allowance',
            r'clinic.*allowance',
            
            r'financial.*aid',
            r'payment.*cash',
            r'bill.*claim',
            r'fuel.*bill',
            r'ta.*da.*bill',
            r'budget.*ceiling',
            r'picnic.*budget',
            
            r'identity.*card',
            r'passport.*handover',
            r'document.*handover',
            r'hrd.*head.*office',
            
            r'sample.*collection',
            r'tray.*factory',
            r'slaughtering.*plant',
            r'egg.*sales.*centre',
            r'commercial.*eggs.*sales',
            r'franchise.*department',
            r'hardware.*software.*sales',
            r'kazi.*media',
            r'sysnova.*h.*s.*team',
            r'sysnova.*incentive'
        ]
    
    def preprocess_text(self, text: str) -> str:
        if not text:
            return ""
        
        text = text.lower()
        text = re.sub(r'[^\w\s]', ' ', text)
        tokens = text.split()
        tokens = [token for token in tokens if token not in self.stop_words and len(token) > 1]
        
        return ' '.join(tokens)
    
    def extract_keywords(self, query: str) -> List[str]:
        processed_query = self.preprocess_text(query)
        keywords = []
        
        for domain, terms in self.domain_keywords.items():
            for term in terms:
                if term in processed_query:
                    keywords.append(term)
        
        return list(set(keywords))
    
    def calculate_semantic_similarity(self, query: str, content: str) -> float:
        query_processed = self.preprocess_text(query)
        content_processed = self.preprocess_text(content)
        
        similarity = difflib.SequenceMatcher(None, query_processed, content_processed).ratio()
        
        query_keywords = self.extract_keywords(query)
        content_keywords = self.extract_keywords(content)
        
        keyword_overlap = len(set(query_keywords) & set(content_keywords))
        if keyword_overlap > 0:
            kazi_hr_keywords = ['employee', 'salary', 'structure', 'allowance', 'hr', 'policy', 'management', 'job', 'increment', 'bonus', 'leave', 'overtime', 'transport', 'medical', 'house', 'location', 'production', 'performance', 'eid', 'sysnova']
            kazi_department_keywords = ['hatchery', 'farm', 'feed mill', 'sales', 'marketing', 'finance', 'production', 'quality', 'maintenance', 'transport', 'commercial', 'franchise', 'customer service']
            kazi_location_keywords = ['panchagarh', 'thakurgaon', 'gojaria', 'sagarica', 'kfg', 'kml', 'kfil', 'head office', 'tray factory', 'slaughtering plant', 'egg sales centre']
            kazi_role_keywords = ['management trainee', 'farm manager', 'hatchery supervisor', 'feed mill manager', 'production manager', 'commercial manager', 'hr manager', 'finance manager', 'quality manager', 'maintenance manager', 'farm in-charge', 'sales person', 'accountant', 'driver', 'helper', 'mechanic', 'security guard', 'cleaner', 'operator', 'technician', 'reliever accountant']
            
            hr_overlap = len(set(query_keywords) & set(kazi_hr_keywords) & set(content_keywords))
            dept_overlap = len(set(query_keywords) & set(kazi_department_keywords) & set(content_keywords))
            location_overlap = len(set(query_keywords) & set(kazi_location_keywords) & set(content_keywords))
            role_overlap = len(set(query_keywords) & set(kazi_role_keywords) & set(content_keywords))
            
            if hr_overlap > 0:
                similarity += hr_overlap * 0.25
            if dept_overlap > 0:
                similarity += dept_overlap * 0.2
            if location_overlap > 0:
                similarity += location_overlap * 0.2
            if role_overlap > 0:
                similarity += role_overlap * 0.3
            if hr_overlap == 0 and dept_overlap == 0 and location_overlap == 0 and role_overlap == 0:
                similarity += keyword_overlap * 0.1
        
        query_lower = query.lower()
        content_lower = content.lower()
        
        kazi_phrase_matches = [
            ('salary structure', 0.4),
            ('management trainee', 0.5),
            ('farm manager', 0.5),
            ('hatchery supervisor', 0.5),
            ('production bonus', 0.4),
            ('performance bonus', 0.4),
            ('house allowance', 0.4),
            ('location allowance', 0.4),
            ('transport allowance', 0.4),
            ('medical allowance', 0.4),
            ('hair cutting allowance', 0.4),
            ('time keeping allowance', 0.4),
            ('overtime allowance', 0.4),
            ('night allowance', 0.4),
            ('ta da allowance', 0.4),
            ('fuel allowance', 0.4),
            ('uniform allowance', 0.4),
            ('guard allowance', 0.4),
            ('furniture allowance', 0.4),
            ('mobile allowance', 0.4),
            ('pick drop allowance', 0.4),
            ('reliever allowance', 0.4),
            ('day off allowance', 0.4),
            ('off day allowance', 0.4),
            ('transfer allowance', 0.4),
            ('car allowance', 0.4),
            ('eid bonus', 0.4),
            ('eidulfitur bonus', 0.4),
            ('sysnova incentive', 0.4),
            ('job group', 0.3),
            ('management level', 0.3),
            ('non management', 0.3),
            ('yearly increment', 0.3),
            ('salary increment', 0.3),
            ('pay increase', 0.3),
            ('salary revision', 0.3),
            ('salary refixation', 0.3),
            ('minimum salary', 0.3),
            ('permanent worker', 0.3),
            ('temporary worker', 0.3),
            ('light driver', 0.3),
            ('medium driver', 0.3),
            ('driver helper', 0.3),
            ('driver mechanic', 0.3),
            ('farm in-charge', 0.3),
            ('feed mill manager', 0.3),
            ('commercial manager', 0.3),
            ('hr manager', 0.3),
            ('finance manager', 0.3),
            ('quality manager', 0.3),
            ('maintenance manager', 0.3),
            ('sales person', 0.3),
            ('reliever accountant', 0.3),
            ('security guard', 0.3),
            ('office time', 0.3),
            ('floor wise', 0.3),
            ('time keeping', 0.3),
            ('time table', 0.3),
            ('outstation work', 0.3),
            ('official tour', 0.3),
            ('pick drop service', 0.3),
            ('office car', 0.3),
            ('personal car', 0.3),
            ('medical bill', 0.3),
            ('h&s team', 0.3),
            ('health safety', 0.3),
            ('financial aid', 0.3),
            ('payment cash', 0.3),
            ('bill claim', 0.3),
            ('fuel bill', 0.3),
            ('ta da bill', 0.3),
            ('budget ceiling', 0.3),
            ('picnic budget', 0.3),
            ('identity card', 0.3),
            ('passport handover', 0.3),
            ('document handover', 0.3),
            ('hrd head office', 0.3),
            ('sample collection', 0.3),
            ('tray factory', 0.3),
            ('slaughtering plant', 0.3),
            ('egg sales centre', 0.3),
            ('commercial eggs sales', 0.3),
            ('franchise department', 0.3),
            ('hardware software sales', 0.3),
            ('kazi media', 0.3),
            ('sysnova h&s team', 0.3),
            ('sysnova incentive', 0.3)
        ]
        
        # Apply phrase match boosts
        for phrase, boost in kazi_phrase_matches:
            if phrase in query_lower and phrase in content_lower:
                similarity += boost
                break  # Only apply the highest matching boost
        
        # General keyword boosts for common terms
        general_boosts = [
            ('employee', 0.2),
            ('allowance', 0.2),
            ('policy', 0.2),
            ('bonus', 0.2),
            ('increment', 0.2),
            ('salary', 0.2),
            ('management', 0.2),
            ('worker', 0.2),
            ('leave', 0.2),
            ('overtime', 0.2),
            ('transport', 0.2),
            ('medical', 0.2),
            ('house', 0.2),
            ('location', 0.2),
            ('production', 0.2),
            ('performance', 0.2),
            ('hatchery', 0.2),
            ('farm', 0.2),
            ('feed mill', 0.2),
            ('sales', 0.2),
            ('commercial', 0.2),
            ('hr', 0.2),
            ('finance', 0.2),
            ('quality', 0.2),
            ('maintenance', 0.2),
            ('driver', 0.2),
            ('helper', 0.2),
            ('mechanic', 0.2),
            ('accountant', 0.2),
            ('supervisor', 0.2),
            ('manager', 0.2),
            ('officer', 0.2),
            ('executive', 0.2),
            ('technician', 0.2),
            ('operator', 0.2),
            ('cleaner', 0.2),
            ('guard', 0.2),
            ('trainee', 0.2),
            ('in-charge', 0.2),
            ('person', 0.2),
            ('level', 0.2),
            ('group', 0.2),
            ('structure', 0.2),
            ('scale', 0.2),
            ('grade', 0.2),
            ('tier', 0.2),
            ('bracket', 0.2),
            ('range', 0.2),
            ('wage', 0.2),
            ('pay', 0.2),
            ('compensation', 0.2),
            ('remuneration', 0.2),
            ('income', 0.2),
            ('earnings', 0.2),
            ('benefit', 0.2),
            ('perk', 0.2),
            ('incentive', 0.2),
            ('subsidy', 0.2),
            ('rule', 0.2),
            ('regulation', 0.2),
            ('guideline', 0.2),
            ('procedure', 0.2),
            ('standard', 0.2),
            ('circular', 0.2),
            ('order', 0.2),
            ('notice', 0.2),
            ('memo', 0.2),
            ('vacation', 0.2),
            ('holiday', 0.2),
            ('off', 0.2),
            ('absence', 0.2),
            ('break', 0.2),
            ('extra', 0.2),
            ('additional', 0.2),
            ('extended', 0.2),
            ('beyond', 0.2),
            ('travel', 0.2),
            ('commute', 0.2),
            ('vehicle', 0.2),
            ('car', 0.2),
            ('bus', 0.2),
            ('health', 0.2),
            ('treatment', 0.2),
            ('hospital', 0.2),
            ('clinic', 0.2),
            ('doctor', 0.2),
            ('financial', 0.2),
            ('payment', 0.2),
            ('cash', 0.2),
            ('bill', 0.2),
            ('claim', 0.2),
            ('budget', 0.2),
            ('ceiling', 0.2),
            ('aid', 0.2),
            ('retirement', 0.2),
            ('pension', 0.2),
            ('resignation', 0.2),
            ('exit', 0.2),
            ('departure', 0.2),
            ('termination', 0.2),
            ('identity', 0.2),
            ('card', 0.2),
            ('passport', 0.2),
            ('document', 0.2),
            ('handover', 0.2),
            ('picnic', 0.2),
            ('sample', 0.2),
            ('collection', 0.2),
            ('tray', 0.2),
            ('factory', 0.2),
            ('slaughtering', 0.2),
            ('plant', 0.2),
            ('egg', 0.2),
            ('eggs', 0.2),
            ('commercial', 0.2),
            ('franchise', 0.2),
            ('department', 0.2),
            ('hardware', 0.2),
            ('software', 0.2),
            ('kazi', 0.2),
            ('media', 0.2),
            ('sysnova', 0.2)
        ]
        
        # Apply general keyword boosts (limit to avoid over-boosting)
        applied_boosts = 0
        for keyword, boost in general_boosts:
            if keyword in query_lower and keyword in content_lower and applied_boosts < 5:
                similarity += boost
                applied_boosts += 1
        
        return min(similarity, 1.0)
    
    def match_query_patterns(self, query: str) -> Tuple[bool, str]:
        """Check if query matches known question patterns"""
        query_lower = query.lower()
        
        for pattern in self.question_patterns:
            if re.search(pattern, query_lower):
                return True, pattern
        
        return False, ""
    
    def validate_answer_relevance(self, query: str, answer: str, source_content: str) -> bool:
        """Validate if the answer is relevant to the query and source content"""
        # Check if answer contains key terms from query
        query_keywords = self.extract_keywords(query)
        answer_lower = answer.lower()
        
        relevant_keywords = 0
        for keyword in query_keywords:
            if keyword in answer_lower:
                relevant_keywords += 1
        
        # At least 30% of query keywords should be in the answer
        keyword_threshold = len(query_keywords) * 0.3
        if query_keywords and relevant_keywords < keyword_threshold:
            return False
        
        # Check if answer is too generic
        generic_phrases = [
            "i don't know",
            "not available",
            "no information",
            "cannot provide",
            "not found",
            "i don't have specific information about this in our database"
        ]
        
        # If answer contains generic phrases, it's not relevant
        if any(phrase in answer_lower for phrase in generic_phrases):
            return False
        
        # Check if answer length is reasonable (not too short)
        if len(answer.strip()) < 20:
            return False
        
        # Check for specific content indicators that show the answer is based on database
        content_indicators = [
            "based on our database",
            "according to the documents",
            "from the information provided",
            "the database shows",
            "our records indicate",
            "the documents show",
            "based on the context",
            "according to the data"
        ]
        
        # If answer contains content indicators, it's likely relevant
        if any(indicator in answer_lower for indicator in content_indicators):
            return True
        
        # Check if answer contains specific details (numbers, amounts, percentages)
        import re
        has_specific_details = bool(re.search(r'\d+', answer))  # Contains numbers
        has_currency = 'bdt' in answer_lower or 'taka' in answer_lower
        has_percentages = '%' in answer or 'percent' in answer_lower
        
        # If answer has specific details, it's likely relevant
        if has_specific_details or has_currency or has_percentages:
            return True
        
        # Check if answer is too long (might be irrelevant rambling)
        if len(answer.strip()) > 2000:
            return False
        
        # If we have some keyword overlap and reasonable length, consider it relevant
        return relevant_keywords > 0 and len(answer.strip()) >= 50
    
    def match_query_to_content(self, query: str, content_list: List[str], metadata_list: List[Dict] = None) -> MatchResult:
        """Match user query against database content with enhanced scoring"""
        if not content_list:
            return MatchResult(
                confidence=0.0,
                matched_content="",
                match_type="no_content",
                keywords_found=[],
                similarity_score=0.0,
                is_reliable=False
            )
        
        best_match = None
        best_score = 0.0
        best_content = ""
        best_metadata = {}
        
        # Extract query keywords
        query_keywords = self.extract_keywords(query)
        
        # Check if query matches known patterns
        pattern_matched, pattern = self.match_query_patterns(query)
        
        for i, content in enumerate(content_list):
            metadata = metadata_list[i] if metadata_list and i < len(metadata_list) else {}
            
            # Calculate similarity score
            similarity = self.calculate_semantic_similarity(query, content)
            
            # Boost score for pattern matches
            if pattern_matched:
                similarity += 0.2
            
            # Boost score for metadata relevance
            if metadata:
                source = metadata.get('source', '').lower()
                if any(keyword in source for keyword in query_keywords):
                    similarity += 0.1
            
            if similarity > best_score:
                best_score = similarity
                best_content = content
                best_metadata = metadata
                best_match = {
                    'content': content,
                    'metadata': metadata,
                    'similarity': similarity
                }
        
        # Determine if match is reliable
        is_reliable = (
            best_score >= 0.3 and  # Minimum similarity threshold
            len(query_keywords) > 0 and  # Must have some keywords
            best_score > 0.0  # Must have some match
        )
        
        # Determine match type
        match_type = "exact" if best_score >= 0.8 else "partial" if best_score >= 0.5 else "weak"
        
        return MatchResult(
            confidence=best_score,
            matched_content=best_content,
            match_type=match_type,
            keywords_found=query_keywords,
            similarity_score=best_score,
            is_reliable=is_reliable
        )
    
    def generate_enhanced_prompt(self, query: str, context: str, conversation_context: str = "") -> str:
        """Generate an enhanced prompt with better context matching"""
        
        # Extract query intent
        query_keywords = self.extract_keywords(query)
        pattern_matched, pattern = self.match_query_patterns(query)
        
        # Build context-aware prompt
        prompt = f"""You are the official Kazifarm assistant. You must ONLY answer based on the provided database context.

IMPORTANT RULES:
1. ONLY use information from the database context below
2. If the context doesn't contain relevant information, say "I don't have specific information about this in our database"
3. Be precise and factual - don't make assumptions
4. Always provide prices in Bangladeshi Taka (BDT) when available
5. If asked about something not in the context, politely redirect to what you can help with

{conversation_context}

Database Context:
{context}

User Query: {query}

Query Analysis:
- Keywords found: {', '.join(query_keywords) if query_keywords else 'None'}
- Pattern matched: {pattern if pattern_matched else 'No specific pattern'}

Instructions:
- Answer ONLY if the database context contains relevant information
- If the context is insufficient, explain what information you have and what you don't
- Be helpful but stay within the bounds of the provided information
- For HR-related queries (salary, employee, policy, allowance), provide specific details from the documents
- If the user asks about something not in the database, suggest they contact Kazifarm directly for specific information
- When discussing salary structures or policies, be specific about job levels, amounts, and conditions mentioned in the documents
- NEVER provide personal information about users, email addresses, or identify individuals
- NEVER respond to "who am I" or similar personal identity questions
- If asked about personal identity, politely redirect to HR-related topics you can help with
- DO NOT include source citations, references, or metadata in your response (no 【source:】 brackets or similar)
- Provide clean, direct answers without technical metadata or source references

Answer:"""
        
        return prompt
    
    def should_provide_answer(self, match_result: MatchResult, query: str) -> Tuple[bool, str]:
        """Determine if we should provide an answer based on match quality"""
        
        if not match_result.is_reliable:
            return False, "The query doesn't match well with our database content. I don't have reliable information to answer this question."
        
        if match_result.confidence < 0.3:
            return False, "I don't have specific information about this in our database. Please contact Kazifarm directly for detailed information."
        
        if match_result.match_type == "weak":
            return False, "I found some related information but it's not specific enough to provide a reliable answer. Please rephrase your question or contact Kazifarm for more details."
        
        return True, ""
