"""
Dynamic Intent Detection Agent - Uses ML and semantic similarity instead of hardcoded patterns
"""
import re
import json
import numpy as np
import os
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, asdict
from collections import defaultdict, deque
from datetime import datetime
import hashlib
import shutil
from pathlib import Path

@dataclass
class IntentResult:
    """Result of intent detection"""
    intent_type: str
    confidence: float
    is_greeting: bool
    is_small_talk: bool
    greeting_response: str = None
    reasoning: str = None
    learned_patterns_used: List[str] = None

@dataclass
class IntentPattern:
    """Represents a learned intent pattern"""
    pattern_id: str
    intent_type: str
    keywords: List[str]
    context_words: List[str]
    confidence: float
    usage_count: int
    success_rate: float
    last_used: str
    created_at: str

class PatternManager:
    """Smart pattern persistence manager"""
    
    def __init__(self, data_dir: str = "core/data/intent_patterns"):
        self.data_dir = Path(data_dir)
        self.patterns_file = self.data_dir / "patterns.json"
        self.backup_dir = self.data_dir / "backups"
        self.stats_file = self.data_dir / "stats.json"
        
        # Create directories if they don't exist
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.backup_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"[PATTERN MANAGER] Initialized with data_dir={self.data_dir}")
    
    def save_patterns(self, patterns_data: Dict[str, Any]) -> bool:
        """Save patterns with backup and validation"""
        try:
            # Create backup of existing file
            if self.patterns_file.exists():
                backup_file = self.backup_dir / f"patterns_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
                shutil.copy2(self.patterns_file, backup_file)
                
                # Keep only last 10 backups
                self._cleanup_backups()
            
            # Validate data before saving
            if not self._validate_patterns_data(patterns_data):
                print("[PATTERN MANAGER ERROR] Invalid patterns data")
                return False
            
            # Save patterns
            with open(self.patterns_file, 'w') as f:
                json.dump(patterns_data, f, indent=2)
            
            # Update stats
            self._update_stats(len(patterns_data.get('patterns', {})))
            
            print(f"[PATTERN MANAGER] Patterns saved successfully to {self.patterns_file}")
            return True
            
        except Exception as e:
            print(f"[PATTERN MANAGER ERROR] Failed to save patterns: {str(e)}")
            return False
    
    def load_patterns(self) -> Optional[Dict[str, Any]]:
        """Load patterns with fallback to backup"""
        try:
            # Try to load main file
            if self.patterns_file.exists():
                with open(self.patterns_file, 'r') as f:
                    data = json.load(f)
                
                if self._validate_patterns_data(data):
                    print(f"[PATTERN MANAGER] Patterns loaded from {self.patterns_file}")
                    return data
                else:
                    print("[PATTERN MANAGER WARNING] Main file corrupted, trying backup")
            
            # Try to load from backup
            return self._load_from_backup()
            
        except Exception as e:
            print(f"[PATTERN MANAGER ERROR] Failed to load patterns: {str(e)}")
            return self._load_from_backup()
    
    def _load_from_backup(self) -> Optional[Dict[str, Any]]:
        """Load patterns from most recent backup"""
        try:
            backup_files = sorted(self.backup_dir.glob("patterns_backup_*.json"), reverse=True)
            
            for backup_file in backup_files:
                try:
                    with open(backup_file, 'r') as f:
                        data = json.load(f)
                    
                    if self._validate_patterns_data(data):
                        print(f"[PATTERN MANAGER] Patterns loaded from backup: {backup_file}")
                        return data
                except Exception:
                    continue
            
            print("[PATTERN MANAGER WARNING] No valid backup found, using defaults")
            return None
            
        except Exception as e:
            print(f"[PATTERN MANAGER ERROR] Failed to load from backup: {str(e)}")
            return None
    
    def _validate_patterns_data(self, data: Dict[str, Any]) -> bool:
        """Validate patterns data structure"""
        try:
            if not isinstance(data, dict):
                return False
            
            if 'patterns' not in data:
                return False
            
            patterns = data['patterns']
            if not isinstance(patterns, dict):
                return False
            
            # Validate each pattern
            for intent_type, pattern_list in patterns.items():
                if not isinstance(pattern_list, list):
                    return False
                
                for pattern in pattern_list:
                    required_fields = ['pattern_id', 'intent_type', 'keywords', 'context_words', 'confidence']
                    if not all(field in pattern for field in required_fields):
                        return False
            
            return True
            
        except Exception:
            return False
    
    def _cleanup_backups(self):
        """Keep only the last 10 backup files"""
        try:
            backup_files = sorted(self.backup_dir.glob("patterns_backup_*.json"), reverse=True)
            
            # Remove old backups beyond the limit
            for backup_file in backup_files[10:]:
                backup_file.unlink()
                
        except Exception as e:
            print(f"[PATTERN MANAGER WARNING] Failed to cleanup backups: {str(e)}")
    
    def _update_stats(self, pattern_count: int):
        """Update pattern statistics"""
        try:
            stats = {
                'total_patterns': pattern_count,
                'last_updated': datetime.now().isoformat(),
                'version': '1.0'
            }
            
            with open(self.stats_file, 'w') as f:
                json.dump(stats, f, indent=2)
                
        except Exception as e:
            print(f"[PATTERN MANAGER WARNING] Failed to update stats: {str(e)}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get pattern statistics"""
        try:
            if self.stats_file.exists():
                with open(self.stats_file, 'r') as f:
                    return json.load(f)
        except Exception:
            pass
        
        return {'total_patterns': 0, 'last_updated': None, 'version': '1.0'}
    
    def cleanup_old_patterns(self, max_age_days: int = 30):
        """Remove patterns that haven't been used recently"""
        try:
            patterns_data = self.load_patterns()
            if not patterns_data:
                return
            
            cutoff_date = datetime.now().timestamp() - (max_age_days * 24 * 60 * 60)
            cleaned_patterns = {}
            
            for intent_type, patterns in patterns_data.get('patterns', {}).items():
                active_patterns = []
                
                for pattern in patterns:
                    last_used = pattern.get('last_used', '')
                    if last_used:
                        try:
                            last_used_timestamp = datetime.fromisoformat(last_used).timestamp()
                            if last_used_timestamp > cutoff_date:
                                active_patterns.append(pattern)
                        except Exception:
                            # Keep patterns with invalid timestamps
                            active_patterns.append(pattern)
                    else:
                        # Keep patterns that have never been used (base patterns)
                        active_patterns.append(pattern)
                
                if active_patterns:
                    cleaned_patterns[intent_type] = active_patterns
            
            # Save cleaned patterns
            patterns_data['patterns'] = cleaned_patterns
            self.save_patterns(patterns_data)
            
            print(f"[PATTERN MANAGER] Cleaned up old patterns (older than {max_age_days} days)")
            
        except Exception as e:
            print(f"[PATTERN MANAGER ERROR] Failed to cleanup old patterns: {str(e)}")

class DynamicIntentAgent:
    """Dynamic intent detection agent that learns and adapts"""
    
    def __init__(self, learning_enabled: bool = True, min_confidence: float = 0.6, data_dir: str = "core/data/intent_patterns"):
        self.learning_enabled = learning_enabled
        self.min_confidence = min_confidence
        
        # Smart pattern manager
        self.pattern_manager = PatternManager(data_dir)
        
        # Learned patterns storage
        self.intent_patterns: Dict[str, List[IntentPattern]] = defaultdict(list)
        self.pattern_cache: Dict[str, IntentPattern] = {}
        
        # Learning data
        self.query_history: deque = deque(maxlen=1000)
        self.feedback_history: deque = deque(maxlen=500)
        
        # Enhanced seed patterns for better recognition of common HR queries
        self.seed_patterns = {
            'greeting': {
                'examples': ['hello', 'hi', 'hey', 'greetings', 'morning', 'afternoon', 'evening'],
                'confidence': 0.9
            },
            'job_inquiry': {
                'examples': ['salary', 'leave', 'policy', 'policies', 'applications', 'benefits', 'insurance', 'medical', 'allowance', 'structure', 'entitlement', 'procedures', 'guidelines', 'manual', 'handbook', 'positions', 'hierarchy', 'manager', 'performance', 'training', 'development', 'show', 'tell', 'what', 'how', 'organizational', 'organization', 'company', 'employee', 'staff', 'work', 'job', 'role', 'designation'],
                'confidence': 0.8
            },
            'small_talk': {
                'examples': ['weather', 'weekend', 'food', 'sports'],
                'confidence': 0.7
            }
        }
        
        # Initialize with base patterns and load existing patterns
        self._initialize_patterns()
        
        print(f"[DYNAMIC INTENT AGENT] Initialized with learning_enabled={learning_enabled}")
    
    def _initialize_patterns(self):
        """Initialize patterns from saved data or base patterns"""
        # Try to load existing patterns
        saved_data = self.pattern_manager.load_patterns()
        
        if saved_data:
            # Load patterns from saved data
            for intent_type, pattern_list in saved_data.get('patterns', {}).items():
                self.intent_patterns[intent_type] = [
                    IntentPattern(**pattern_data) for pattern_data in pattern_list
                ]
            
            # Load history
            self.query_history = deque(saved_data.get('query_history', []), maxlen=1000)
            self.feedback_history = deque(saved_data.get('feedback_history', []), maxlen=500)
            
            print(f"[DYNAMIC INTENT AGENT] Loaded {sum(len(patterns) for patterns in self.intent_patterns.values())} patterns from storage")
        else:
            # Initialize with base patterns
            self._initialize_base_patterns()
            print("[DYNAMIC INTENT AGENT] Initialized with base patterns")
    
    def _initialize_base_patterns(self):
        """Initialize with minimal seed patterns for bootstrapping"""
        for intent_type, pattern_data in self.seed_patterns.items():
            pattern = IntentPattern(
                pattern_id=f"seed_{intent_type}",
                intent_type=intent_type,
                keywords=pattern_data['examples'],
                context_words=[],  # Will be learned dynamically
                confidence=pattern_data['confidence'],
                usage_count=0,
                success_rate=1.0,
                last_used="",
                created_at=datetime.now().isoformat()
            )
            self.intent_patterns[intent_type].append(pattern)
    
    def detect_intent(self, text: str, context: Optional[Dict[str, Any]] = None) -> IntentResult:
        """Detect user intent using dynamic patterns"""
        try:
            text_lower = text.lower().strip()
            
            # Store query for learning
            if self.learning_enabled:
                self.query_history.append({
                    'text': text,
                    'timestamp': datetime.now().isoformat(),
                    'context': context
                })
            
            # Get intent candidates with confidence scores
            candidates = self._get_intent_candidates(text_lower, context)
            
            if not candidates:
                return self._create_fallback_result(text_lower)
            
            # Select best candidate
            best_intent = max(candidates, key=lambda x: x['confidence'])
            
            # Generate reasoning
            reasoning = self._generate_reasoning(text_lower, best_intent, candidates)
            
            # Update pattern usage
            if self.learning_enabled and best_intent['pattern']:
                self._update_pattern_usage(best_intent['pattern'])
            
            # Create result
            result = IntentResult(
                intent_type=best_intent['intent_type'],
                confidence=best_intent['confidence'],
                is_greeting=best_intent['intent_type'] == 'greeting',
                is_small_talk=best_intent['intent_type'] == 'small_talk',
                greeting_response=self._get_greeting_response(text_lower) if best_intent['intent_type'] == 'greeting' else None,
                reasoning=reasoning,
                learned_patterns_used=[best_intent['pattern'].pattern_id] if best_intent['pattern'] else []
            )
            
            return result
            
        except Exception as e:
            print(f"[DYNAMIC INTENT AGENT ERROR] {str(e)}")
            return self._create_fallback_result(text_lower)
    
    def _get_intent_candidates(self, text: str, context: Optional[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Get intent candidates with confidence scores"""
        candidates = []
        
        # Check all learned patterns
        for intent_type, patterns in self.intent_patterns.items():
            for pattern in patterns:
                confidence = self._calculate_pattern_confidence(text, pattern, context)
                if confidence >= self.min_confidence:
                    candidates.append({
                        'intent_type': intent_type,
                        'confidence': confidence,
                        'pattern': pattern
                    })
        
        # Try to learn new patterns if no good match found
        if not candidates and self.learning_enabled:
            new_pattern = self._attempt_pattern_learning(text, context)
            if new_pattern:
                candidates.append({
                    'intent_type': new_pattern.intent_type,
                    'confidence': new_pattern.confidence,
                    'pattern': new_pattern
                })
        
        return candidates
    
    def _calculate_pattern_confidence(self, text: str, pattern: IntentPattern, context: Optional[Dict[str, Any]]) -> float:
        """Calculate confidence score for a pattern match"""
        try:
            # Keyword matching with word boundaries to avoid false positives
            keyword_matches = 0
            matched_keywords = []
            for keyword in pattern.keywords:
                # Use word boundary regex to avoid partial matches (e.g., "hi" in "hierarchy")
                if re.search(r'\b' + re.escape(keyword) + r'\b', text, re.IGNORECASE):
                    keyword_matches += 1
                    matched_keywords.append(keyword)
            
            keyword_score = keyword_matches / len(pattern.keywords) if pattern.keywords else 0
            
            # Context word matching with word boundaries
            context_matches = 0
            for context_word in pattern.context_words:
                # Use word boundary regex for context words too
                if re.search(r'\b' + re.escape(context_word) + r'\b', text, re.IGNORECASE):
                    context_matches += 1
            
            context_score = context_matches / len(pattern.context_words) if pattern.context_words else 0
            
            # Base confidence from pattern
            base_confidence = pattern.confidence
            
            # Success rate adjustment
            success_adjustment = pattern.success_rate
            
            # Usage count adjustment (more usage = higher confidence)
            usage_adjustment = min(1.0, pattern.usage_count / 10.0)
            
            # Special handling for greeting patterns - be more strict
            if pattern.intent_type == 'greeting':
                # For greeting patterns, require the text to start with greeting words or be very short
                text_words = text.strip().split()
                is_likely_greeting = (
                    len(text_words) <= 3 and  # Short text
                    any(word in ['hello', 'hi', 'hey', 'morning', 'afternoon', 'evening'] for word in text_words[:2])  # Starts with greeting
                )
                
                if not is_likely_greeting:
                    # If it doesn't look like a greeting, heavily penalize
                    return 0.1
                
                # If it looks like a greeting, use normal calculation
                final_confidence = (
                    base_confidence * 0.5 +
                    keyword_score * 0.3 +
                    context_score * 0.1 +
                    success_adjustment * 0.05 +
                    usage_adjustment * 0.05
                )
            
            # Enhanced handling for job_inquiry patterns
            elif pattern.intent_type == 'job_inquiry':
                # For job inquiry patterns, boost confidence for action words and HR terms
                action_words = ['show', 'tell', 'what', 'how', 'explain', 'describe', 'give', 'provide']
                hr_terms = ['salary', 'structure', 'organizational', 'hierarchy', 'policy', 'leave', 'benefits']
                
                has_action = any(word in text.lower() for word in action_words)
                has_hr_term = any(word in text.lower() for word in hr_terms)
                
                if has_action and has_hr_term:
                    # Strong indication of job inquiry
                    action_boost = 0.3
                elif has_action or has_hr_term:
                    # Moderate indication
                    action_boost = 0.15
                else:
                    action_boost = 0
                
                final_confidence = (
                    base_confidence * 0.3 +
                    keyword_score * 0.4 +
                    action_boost +
                    context_score * 0.1 +
                    success_adjustment * 0.03 +
                    usage_adjustment * 0.02
                )
            
            # Enhanced handling for seed patterns (general case)
            elif pattern.pattern_id.startswith('seed_'):
                # For other seed patterns, if any keyword matches, give high confidence
                if keyword_matches > 0:
                    # Boost confidence significantly for any keyword match in seed patterns
                    keyword_boost = min(0.4, keyword_matches * 0.2)  # Up to 0.4 boost
                    final_confidence = (
                        base_confidence * 0.3 +
                        keyword_score * 0.4 +  # Higher weight for keyword matching
                        keyword_boost +  # Additional boost for keyword matches
                        context_score * 0.1 +
                        success_adjustment * 0.03 +
                        usage_adjustment * 0.02
                    )
                else:
                    # No keyword matches, use normal calculation
                    final_confidence = (
                        base_confidence * 0.4 +
                        context_score * 0.2 +
                        success_adjustment * 0.03 +
                        usage_adjustment * 0.02
                    )
            else:
                # For learned patterns, use balanced approach
                final_confidence = (
                    base_confidence * 0.4 +
                    keyword_score * 0.3 +
                    context_score * 0.2 +
                    success_adjustment * 0.05 +
                    usage_adjustment * 0.05
                )
            
            return min(1.0, final_confidence)
            
        except Exception as e:
            print(f"[PATTERN CONFIDENCE ERROR] {str(e)}")
            return 0.0
    
    def _attempt_pattern_learning(self, text: str, context: Optional[Dict[str, Any]]) -> Optional[IntentPattern]:
        """Dynamic pattern learning with adaptive keyword extraction"""
        try:
            # Advanced keyword extraction
            keywords = self._extract_meaningful_keywords(text)
            context_phrases = self._extract_context_phrases(text)
            
            # Determine intent using existing patterns and similarity
            intent_type = self._infer_intent_dynamically(text, keywords, context_phrases)
            
            if intent_type and len(keywords) >= 1:  # More lenient requirement
                pattern_id = f"learned_{hashlib.md5(text.encode()).hexdigest()[:8]}"
                
                # Dynamic confidence based on keyword quality and context
                confidence = self._calculate_learning_confidence(keywords, context_phrases, text)
                
                pattern = IntentPattern(
                    pattern_id=pattern_id,
                    intent_type=intent_type,
                    keywords=keywords[:8],  # Focus on most relevant keywords
                    context_words=context_phrases,
                    confidence=confidence,
                    usage_count=1,
                    success_rate=0.7,  # Conservative initial success rate
                    last_used=datetime.now().isoformat(),
                    created_at=datetime.now().isoformat()
                )
                
                # Add to patterns with deduplication
                if not self._is_duplicate_pattern(pattern):
                    self.intent_patterns[intent_type].append(pattern)
                    self.pattern_cache[pattern_id] = pattern
                    print(f"[PATTERN LEARNING] Learned new {intent_type} pattern: {pattern_id} (confidence: {confidence:.2f})")
                    return pattern
            
        except Exception as e:
            print(f"[PATTERN LEARNING ERROR] {str(e)}")
        
        return None
    
    def _extract_meaningful_keywords(self, text: str) -> List[str]:
        """Extract meaningful keywords using dynamic filtering"""
        words = re.findall(r'\b\w+\b', text.lower())
        
        # Dynamic stop word filtering based on length and frequency
        meaningful_words = []
        for word in words:
            if (len(word) > 2 and 
                word not in ['the', 'and', 'for', 'are', 'but', 'not', 'you', 'all', 'can', 'had', 'was', 'one', 'our', 'out', 'day', 'get', 'has', 'him', 'his', 'how', 'its', 'may', 'new', 'now', 'old', 'see', 'two', 'way', 'who', 'did', 'man', 'try', 'use', 'she', 'put', 'end', 'why', 'let', 'ask', 'run', 'own', 'say', 'too', 'any', 'set', 'off', 'far', 'yet', 'eat', 'air']):
                meaningful_words.append(word)
        
        return meaningful_words
    
    def _extract_context_phrases(self, text: str) -> List[str]:
        """Dynamically extract context phrases"""
        text_lower = text.lower()
        phrases = []
        
        # Common question patterns
        question_patterns = [
            r'tell me about \w+',
            r'what is \w+',
            r'how (?:many|much|to) \w+',
            r'can i \w+',
            r'am i eligible',
            r'what are the \w+',
            r'show me \w+',
            r'explain \w+',
            r'describe \w+'
        ]
        
        for pattern in question_patterns:
            matches = re.findall(pattern, text_lower)
            phrases.extend(matches)
        
        return phrases
    
    def _infer_intent_dynamically(self, text: str, keywords: List[str], context_phrases: List[str]) -> Optional[str]:
        """Dynamically infer intent using multiple signals"""
        text_lower = text.lower()
        
        # Use semantic similarity with existing patterns
        if self.intent_patterns:
            best_intent = None
            best_similarity = 0
            
            for intent_type, patterns in self.intent_patterns.items():
                for pattern in patterns:
                    # Calculate text similarity with pattern keywords
                    pattern_text = ' '.join(pattern.keywords + pattern.context_words)
                    similarity = self._calculate_text_similarity(text_lower, pattern_text)
                    
                    if similarity > best_similarity:
                        best_similarity = similarity
                        best_intent = intent_type
            
            if best_similarity > 0.2:
                return best_intent
        
        # Fallback to simple heuristics
        if any(word in text_lower for word in ['hello', 'hi', 'good', 'morning', 'afternoon', 'evening']):
            return 'greeting'
        else:
            return 'job_inquiry'  # Default for business context
    
    def _calculate_text_similarity(self, text1: str, text2: str) -> float:
        """Simple text similarity calculation"""
        words1 = set(text1.split())
        words2 = set(text2.split())
        
        if not words1 or not words2:
            return 0.0
        
        intersection = words1.intersection(words2)
        union = words1.union(words2)
        
        return len(intersection) / len(union)
    
    def _calculate_learning_confidence(self, keywords: List[str], context_phrases: List[str], text: str) -> float:
        """Calculate confidence for learned pattern"""
        base_confidence = 0.5
        
        # Boost for longer, more specific keywords
        if any(len(word) > 6 for word in keywords):
            base_confidence += 0.1
        
        # Boost for context phrases
        if context_phrases:
            base_confidence += 0.1
        
        # Boost for shorter, focused queries
        if len(text.split()) <= 3:
            base_confidence += 0.1
        
        return min(0.8, base_confidence)
    
    def _is_duplicate_pattern(self, new_pattern: IntentPattern) -> bool:
        """Check if pattern is too similar to existing ones"""
        for existing_pattern in self.intent_patterns.get(new_pattern.intent_type, []):
            # Check keyword overlap
            overlap = len(set(new_pattern.keywords).intersection(set(existing_pattern.keywords)))
            if overlap > len(new_pattern.keywords) * 0.7:  # 70% overlap threshold
                return True
        return False
    
    def _classify_intent_from_keywords(self, keywords: List[str], context_phrases: List[str]) -> Optional[str]:
        """Dynamic intent classification using learned patterns and semantic similarity"""
        if not keywords:
            return None
        
        # Use existing learned patterns for classification
        best_intent = None
        best_score = 0
        
        for intent_type, patterns in self.intent_patterns.items():
            for pattern in patterns:
                score = self._calculate_pattern_similarity(keywords, context_phrases, pattern)
                if score > best_score:
                    best_score = score
                    best_intent = intent_type
        
        # If no good match found, try to infer from context
        if best_score < 0.3:
            return self._infer_intent_from_context(keywords, context_phrases)
        
        return best_intent if best_score > 0.2 else None
    
    def _calculate_pattern_similarity(self, keywords: List[str], context_phrases: List[str], pattern: IntentPattern) -> float:
        """Calculate similarity between input and learned pattern"""
        # Word overlap with pattern keywords
        keyword_overlap = len(set(keywords).intersection(set(pattern.keywords)))
        keyword_score = keyword_overlap / max(len(pattern.keywords), 1) if pattern.keywords else 0
        
        # Context phrase overlap
        context_overlap = len(set(context_phrases).intersection(set(pattern.context_words)))
        context_score = context_overlap / max(len(pattern.context_words), 1) if pattern.context_words else 0
        
        # Combine scores with pattern success rate
        combined_score = (keyword_score * 0.7 + context_score * 0.3) * pattern.success_rate
        
        return combined_score
    
    def _infer_intent_from_context(self, keywords: List[str], context_phrases: List[str]) -> Optional[str]:
        """Infer intent from context when no patterns match well"""
        # Simple heuristics for common cases
        query_text = ' '.join(keywords + context_phrases).lower()
        
        # Check for greeting patterns - be more strict
        greeting_words = ['hello', 'hi', 'hey', 'greetings']
        time_greetings = ['morning', 'afternoon', 'evening']
        
        # Only classify as greeting if it starts with greeting words or is a simple time greeting
        words = query_text.split()
        if len(words) <= 3 and (
            any(word in greeting_words for word in words[:2]) or
            (len(words) <= 2 and any(word in time_greetings for word in words))
        ):
            return 'greeting'
        
        # Check for business/work context (default to job_inquiry for work-related queries)
        business_indicators = [
            'work', 'job', 'office', 'company', 'business', 'employee', 'staff', 'manager', 
            'policy', 'salary', 'leave', 'benefit', 'structure', 'organizational', 'hierarchy',
            'show', 'tell', 'what', 'how', 'explain', 'describe', 'give', 'provide',
            'allowance', 'procedures', 'guidelines', 'manual', 'handbook', 'positions',
            'performance', 'training', 'development', 'designation', 'role'
        ]
        if any(indicator in query_text for indicator in business_indicators):
            return 'job_inquiry'
        
        # Check for action words that typically indicate information requests
        action_words = ['show', 'tell', 'what', 'how', 'explain', 'describe', 'give', 'provide', 'list']
        if any(word in query_text for word in action_words):
            return 'job_inquiry'
        
        # Default for queries that might be work-related
        return 'job_inquiry'  # Default to job inquiry for business context
    
    def _generate_reasoning(self, text: str, best_intent: Dict[str, Any], candidates: List[Dict[str, Any]]) -> str:
        """Generate reasoning for the intent detection"""
        pattern = best_intent['pattern']
        if not pattern:
            return f"Classified as {best_intent['intent_type']} based on learned patterns"
        
        matched_keywords = [kw for kw in pattern.keywords if kw in text]
        matched_context = [cw for cw in pattern.context_words if cw in text]
        
        reasoning_parts = [
            f"Intent: {best_intent['intent_type']}",
            f"Confidence: {best_intent['confidence']:.2f}",
            f"Pattern: {pattern.pattern_id}",
            f"Matched keywords: {matched_keywords}",
            f"Matched context: {matched_context}",
            f"Pattern success rate: {pattern.success_rate:.2f}"
        ]
        
        return " | ".join(reasoning_parts)
    
    def _update_pattern_usage(self, pattern: IntentPattern):
        """Update pattern usage statistics"""
        pattern.usage_count += 1
        pattern.last_used = datetime.now().isoformat()
    
    def provide_feedback(self, query: str, correct_intent: str, was_correct: bool):
        """Provide feedback to improve pattern learning"""
        if not self.learning_enabled:
            return
        
        feedback = {
            'query': query,
            'correct_intent': correct_intent,
            'was_correct': was_correct,
            'timestamp': datetime.now().isoformat()
        }
        
        self.feedback_history.append(feedback)
        
        # Update pattern success rates based on feedback
        self._update_pattern_success_rates()
        
        print(f"[FEEDBACK] Received feedback for '{query}': {correct_intent} - {'Correct' if was_correct else 'Incorrect'}")
    
    def _update_pattern_success_rates(self):
        """Update pattern success rates based on feedback"""
        for feedback in self.feedback_history:
            query = feedback['query'].lower()
            correct_intent = feedback['correct_intent']
            was_correct = feedback['was_correct']
            
            # Find patterns that might have been used for this query
            for intent_type, patterns in self.intent_patterns.items():
                for pattern in patterns:
                    if self._pattern_matches_query(query, pattern):
                        if intent_type == correct_intent and was_correct:
                            pattern.success_rate = min(1.0, pattern.success_rate + 0.1)
                        elif intent_type != correct_intent or not was_correct:
                            pattern.success_rate = max(0.0, pattern.success_rate - 0.1)
    
    def _pattern_matches_query(self, query: str, pattern: IntentPattern) -> bool:
        """Check if a pattern matches a query using word boundaries"""
        keyword_matches = 0
        for keyword in pattern.keywords:
            if re.search(r'\b' + re.escape(keyword) + r'\b', query, re.IGNORECASE):
                keyword_matches += 1
        
        context_matches = 0
        for context_word in pattern.context_words:
            if re.search(r'\b' + re.escape(context_word) + r'\b', query, re.IGNORECASE):
                context_matches += 1
        
        return keyword_matches > 0 or context_matches > 0
    
    def _get_greeting_response(self, text: str) -> str:
        """Get appropriate greeting response"""
        greeting_responses = [
            "Hello! Welcome to Kazi Farms! I'm your AI assistant, here to help you with HR policies, salary information, employee benefits, and company procedures. How can I assist you today?",
            "Hi there! Great to meet you! I'm the Kazi Farms chatbot, and I'm here to help you with any questions about our HR policies, salary structures, allowances, or company procedures. What would you like to know?",
            "Good day! Welcome to Kazi Farms! I'm your friendly assistant, ready to help you with employee benefits, leave policies, salary information, and other HR-related questions. How can I help you today?",
            "Hello! Nice to see you here! I'm the Kazi Farms AI assistant, and I'm excited to help you with any questions about our company policies, salary structures, allowances, or HR procedures. What can I do for you?",
            "Hi! Welcome to the Kazi Farms family! I'm here to assist you with all your HR and company-related questions. Whether it's about salary, benefits, policies, or procedures - I'm ready to help! What would you like to know?"
        ]
        
        # Check for specific greeting types
        if any(word in text for word in ['morning', 'good morning', 'gm']):
            return "Good morning! Welcome to Kazi Farms! I'm your AI assistant, ready to help you with HR policies, salary information, employee benefits, and company procedures. How can I assist you today?"
        elif any(word in text for word in ['afternoon', 'good afternoon', 'ga']):
            return "Good afternoon! Welcome to Kazi Farms! I'm your AI assistant, here to help you with HR policies, salary information, employee benefits, and company procedures. How can I assist you today?"
        elif any(word in text for word in ['evening', 'good evening', 'ge']):
            return "Good evening! Welcome to Kazi Farms! I'm your AI assistant, ready to help you with HR policies, salary information, employee benefits, and company procedures. How can I assist you today?"
        elif any(word in text for word in ['night', 'good night', 'gn']):
            return "Good night! Welcome to Kazi Farms! I'm your AI assistant, here to help you with HR policies, salary information, employee benefits, and company procedures. How can I assist you today?"
        else:
            import random
            return random.choice(greeting_responses)
    
    def _create_fallback_result(self, text: str) -> IntentResult:
        """Create fallback result when no patterns match"""
        return IntentResult(
            intent_type="general_inquiry",
            confidence=0.5,
            is_greeting=False,
            is_small_talk=False,
            greeting_response=None,
            reasoning="No matching patterns found, using fallback classification"
        )
    
    def get_pattern_stats(self) -> Dict[str, Any]:
        """Get statistics about learned patterns"""
        stats = {
            'total_patterns': sum(len(patterns) for patterns in self.intent_patterns.values()),
            'patterns_by_intent': {intent: len(patterns) for intent, patterns in self.intent_patterns.items()},
            'total_queries': len(self.query_history),
            'total_feedback': len(self.feedback_history),
            'learning_enabled': self.learning_enabled
        }
        
        # Add pattern manager stats
        manager_stats = self.pattern_manager.get_stats()
        stats.update(manager_stats)
        
        return stats
    
    def save_patterns(self, filepath: str = None):
        """Save learned patterns using smart pattern manager"""
        try:
            patterns_data = {
                'patterns': {
                    intent_type: [asdict(pattern) for pattern in patterns]
                    for intent_type, patterns in self.intent_patterns.items()
                },
                'query_history': list(self.query_history),
                'feedback_history': list(self.feedback_history),
                'metadata': {
                    'learning_enabled': self.learning_enabled,
                    'min_confidence': self.min_confidence,
                    'saved_at': datetime.now().isoformat()
                }
            }
            
            # Use pattern manager for smart saving
            success = self.pattern_manager.save_patterns(patterns_data)
            
            if success:
                print(f"[DYNAMIC INTENT AGENT] Patterns saved successfully")
            else:
                print(f"[DYNAMIC INTENT AGENT] Failed to save patterns")
            
        except Exception as e:
            print(f"[SAVE PATTERNS ERROR] {str(e)}")
    
    def load_patterns(self, filepath: str = None):
        """Load learned patterns using smart pattern manager"""
        try:
            # Pattern manager handles loading automatically
            print(f"[DYNAMIC INTENT AGENT] Patterns loaded via pattern manager")
            
        except Exception as e:
            print(f"[LOAD PATTERNS ERROR] {str(e)}")
    
    def cleanup_old_patterns(self, max_age_days: int = 30):
        """Clean up old patterns using pattern manager"""
        self.pattern_manager.cleanup_old_patterns(max_age_days)
