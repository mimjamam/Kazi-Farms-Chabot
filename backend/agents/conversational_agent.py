"""
Conversational Agent - Makes the chatbot more ChatGPT-like in conversation flow
"""
import re
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime

@dataclass
class ConversationContext:
    """Context for maintaining natural conversation flow"""
    user_intent: str
    conversation_stage: str
    user_preferences: Dict[str, Any]
    follow_up_suggestions: List[str]
    clarification_needed: bool
    topic_depth: str

class ConversationalAgent:
    """Agent that makes interactions more natural and ChatGPT-like"""
    
    def __init__(self):
        self.conversation_patterns = self._initialize_patterns()
        self.user_profiles = {}  # Store user interaction patterns
        pass  # Silent initialization
    
    def enhance_user_input(self, user_input: str, session_id: str) -> Dict[str, Any]:
        """Enhance user input understanding like ChatGPT"""
        
        # Analyze user input patterns
        input_analysis = self._analyze_user_input(user_input, session_id)
        
        # Detect conversation intent
        conversation_intent = self._detect_conversation_intent(user_input)
        
        # Determine response strategy
        response_strategy = self._determine_response_strategy(input_analysis, conversation_intent)
        
        # Generate follow-up suggestions
        follow_ups = self._generate_follow_up_suggestions(user_input, conversation_intent)
        
        return {
            'enhanced_query': self._enhance_query_for_retrieval(user_input, input_analysis),
            'conversation_context': ConversationContext(
                user_intent=conversation_intent['primary_intent'],
                conversation_stage=input_analysis['stage'],
                user_preferences=self._get_user_preferences(session_id),
                follow_up_suggestions=follow_ups,
                clarification_needed=input_analysis['needs_clarification'],
                topic_depth=conversation_intent['depth_level']
            ),
            'response_strategy': response_strategy,
            'personalization': self._get_personalization_hints(session_id, user_input)
        }
    
    def enhance_response(self, response: str, user_input: str, context: Dict[str, Any]) -> str:
        """Make response more conversational and ChatGPT-like"""
        
        # Add conversational elements
        enhanced_response = self._add_conversational_elements(response, user_input, context)
        
        # Add follow-up suggestions if appropriate
        if context.get('conversation_context') and context['conversation_context'].follow_up_suggestions:
            enhanced_response = self._add_follow_up_suggestions(enhanced_response, context['conversation_context'].follow_up_suggestions)
        
        # Add personalization
        if context.get('personalization'):
            enhanced_response = self._add_personalization(enhanced_response, context['personalization'])
        
        return enhanced_response
    
    def _initialize_patterns(self) -> Dict[str, Any]:
        """Initialize conversation patterns"""
        return {
            'greetings': [
                'hello', 'hi', 'hey', 'good morning', 'good afternoon', 'good evening',
                'greetings', 'howdy', 'what\'s up', 'how are you'
            ],
            'questions': [
                'what', 'how', 'when', 'where', 'why', 'who', 'which', 'can you',
                'could you', 'would you', 'do you know', 'tell me', 'explain'
            ],
            'requests': [
                'please', 'can you help', 'i need', 'i want', 'i would like',
                'could you please', 'help me', 'assist me', 'show me'
            ],
            'clarifications': [
                'what do you mean', 'can you clarify', 'i don\'t understand',
                'could you explain', 'more details', 'elaborate'
            ],
            'affirmations': [
                'yes', 'yeah', 'yep', 'sure', 'okay', 'ok', 'right', 'correct',
                'exactly', 'that\'s right', 'continue', 'go on'
            ],
            'negations': [
                'no', 'nope', 'not really', 'that\'s not right', 'incorrect',
                'wrong', 'not what i meant'
            ]
        }
    
    def _analyze_user_input(self, user_input: str, session_id: str) -> Dict[str, Any]:
        """Analyze user input patterns like ChatGPT"""
        input_lower = user_input.lower().strip()
        
        analysis = {
            'length': len(user_input.split()),
            'complexity': self._assess_complexity(user_input),
            'formality': self._assess_formality(user_input),
            'specificity': self._assess_specificity(user_input),
            'stage': self._determine_conversation_stage(input_lower),
            'needs_clarification': self._needs_clarification(user_input),
            'emotional_tone': self._detect_emotional_tone(user_input),
            'urgency': self._detect_urgency(user_input)
        }
        
        # Update user profile
        self._update_user_profile(session_id, analysis)
        
        return analysis
    
    def _detect_conversation_intent(self, user_input: str) -> Dict[str, Any]:
        """Detect conversation intent like ChatGPT"""
        input_lower = user_input.lower().strip()
        
        # Primary intent detection
        primary_intent = 'information_seeking'  # default
        
        if any(greeting in input_lower for greeting in self.conversation_patterns['greetings']):
            primary_intent = 'greeting'
        elif any(question in input_lower for question in self.conversation_patterns['questions']):
            primary_intent = 'question'
        elif any(request in input_lower for request in self.conversation_patterns['requests']):
            primary_intent = 'request'
        elif any(clarif in input_lower for clarif in self.conversation_patterns['clarifications']):
            primary_intent = 'clarification'
        elif any(affirm in input_lower for affirm in self.conversation_patterns['affirmations']):
            primary_intent = 'affirmation'
        elif any(neg in input_lower for neg in self.conversation_patterns['negations']):
            primary_intent = 'negation'
        
        # Depth level assessment
        depth_level = 'surface'
        if len(user_input.split()) > 15:
            depth_level = 'detailed'
        elif len(user_input.split()) > 8:
            depth_level = 'moderate'
        
        # Secondary intents
        secondary_intents = []
        if 'example' in input_lower or 'for instance' in input_lower:
            secondary_intents.append('wants_examples')
        if 'step' in input_lower or 'process' in input_lower or 'procedure' in input_lower:
            secondary_intents.append('wants_process')
        if 'why' in input_lower or 'reason' in input_lower:
            secondary_intents.append('wants_explanation')
        
        return {
            'primary_intent': primary_intent,
            'secondary_intents': secondary_intents,
            'depth_level': depth_level,
            'confidence': 0.8  # Could be enhanced with ML
        }
    
    def _determine_response_strategy(self, input_analysis: Dict[str, Any], conversation_intent: Dict[str, Any]) -> str:
        """Determine how to respond like ChatGPT"""
        
        # High-level strategy based on intent and analysis
        if conversation_intent['primary_intent'] == 'greeting':
            return 'friendly_greeting'
        elif conversation_intent['primary_intent'] == 'clarification':
            return 'detailed_explanation'
        elif input_analysis['complexity'] == 'high':
            return 'comprehensive_response'
        elif input_analysis['specificity'] == 'high':
            return 'precise_answer'
        elif conversation_intent['depth_level'] == 'detailed':
            return 'thorough_response'
        else:
            return 'balanced_response'
    
    def _generate_follow_up_suggestions(self, user_input: str, conversation_intent: Dict[str, Any]) -> List[str]:
        """Generate follow-up suggestions like ChatGPT"""
        suggestions = []
        input_lower = user_input.lower()
        
        # Topic-based suggestions
        if 'salary' in input_lower:
            suggestions.extend([
                "Would you like to know about allowances and benefits?",
                "Do you want information about salary increments?",
                "Are you interested in learning about different job grades?"
            ])
        elif 'leave' in input_lower:
            suggestions.extend([
                "Would you like to know about leave application procedures?",
                "Do you want information about different types of leave?",
                "Are you interested in leave policies for different positions?"
            ])
        elif 'policy' in input_lower:
            suggestions.extend([
                "Would you like to see related policies?",
                "Do you want to know about policy implementation?",
                "Are you interested in recent policy updates?"
            ])
        
        # Intent-based suggestions
        if conversation_intent['primary_intent'] == 'question':
            suggestions.append("Would you like more detailed information about this topic?")
        elif conversation_intent['depth_level'] == 'surface':
            suggestions.append("Do you want me to explain this in more detail?")
        
        return suggestions[:3]  # Limit to 3 suggestions
    
    def _enhance_query_for_retrieval(self, user_input: str, input_analysis: Dict[str, Any]) -> str:
        """Enhance query for better retrieval like ChatGPT understands context"""
        enhanced_query = user_input
        
        # Add context based on analysis
        if input_analysis['specificity'] == 'low':
            # Add more specific terms
            enhanced_query = self._add_specific_terms(enhanced_query)
        
        if input_analysis['complexity'] == 'high':
            # Extract key concepts
            enhanced_query = self._extract_key_concepts(enhanced_query)
        
        return enhanced_query
    
    def _add_conversational_elements(self, response: str, user_input: str, context: Dict[str, Any]) -> str:
        """Add conversational elements to make response more ChatGPT-like"""
        
        # Add appropriate opening based on user input
        opening = self._generate_conversational_opening(user_input, context)
        
        # Add transitions and connectors
        enhanced_response = self._add_transitions(response)
        
        # Add closing based on context
        closing = self._generate_conversational_closing(context)
        
        # Combine elements
        if opening:
            enhanced_response = f"{opening}\n\n{enhanced_response}"
        if closing:
            enhanced_response = f"{enhanced_response}\n\n{closing}"
        
        return enhanced_response
    
    def _generate_conversational_opening(self, user_input: str, context: Dict[str, Any]) -> str:
        """Generate brief conversational opening"""
        # Disabled conversational openings to keep responses concise
        return ""
    
    def _generate_conversational_closing(self, context: Dict[str, Any]) -> str:
        """Generate brief conversational closing"""
        # Disabled - no closing statements to keep responses concise
        return ""
    
    def _add_follow_up_suggestions(self, response: str, suggestions: List[str]) -> str:
        """Add brief follow-up suggestions to response"""
        if not suggestions or len(suggestions) == 0:
            return response
        
        # Only add one brief suggestion to keep response concise
        suggestion_text = f"\n\n*You might also ask: {suggestions[0]}*"
        
        return response + suggestion_text
    
    def _assess_complexity(self, user_input: str) -> str:
        """Assess input complexity"""
        word_count = len(user_input.split())
        if word_count > 20:
            return 'high'
        elif word_count > 10:
            return 'medium'
        else:
            return 'low'
    
    def _assess_formality(self, user_input: str) -> str:
        """Assess input formality"""
        formal_indicators = ['please', 'could you', 'would you', 'kindly', 'request']
        informal_indicators = ['hey', 'what\'s up', 'gonna', 'wanna', 'yeah']
        
        input_lower = user_input.lower()
        formal_count = sum(1 for indicator in formal_indicators if indicator in input_lower)
        informal_count = sum(1 for indicator in informal_indicators if indicator in input_lower)
        
        if formal_count > informal_count:
            return 'formal'
        elif informal_count > formal_count:
            return 'informal'
        else:
            return 'neutral'
    
    def _assess_specificity(self, user_input: str) -> str:
        """Assess how specific the user input is"""
        specific_indicators = ['exactly', 'specifically', 'precise', 'detailed', 'particular']
        general_indicators = ['general', 'overall', 'basic', 'simple', 'just']
        
        input_lower = user_input.lower()
        specific_count = sum(1 for indicator in specific_indicators if indicator in input_lower)
        general_count = sum(1 for indicator in general_indicators if indicator in input_lower)
        
        if specific_count > 0:
            return 'high'
        elif general_count > 0:
            return 'low'
        else:
            return 'medium'
    
    def _determine_conversation_stage(self, input_lower: str) -> str:
        """Determine what stage of conversation this is"""
        if any(greeting in input_lower for greeting in self.conversation_patterns['greetings']):
            return 'opening'
        elif any(clarif in input_lower for clarif in self.conversation_patterns['clarifications']):
            return 'clarification'
        elif 'thank' in input_lower or 'bye' in input_lower:
            return 'closing'
        else:
            return 'ongoing'
    
    def _needs_clarification(self, user_input: str) -> bool:
        """Determine if user input needs clarification"""
        unclear_indicators = ['unclear', 'confusing', 'not sure', 'maybe', 'i think', 'probably']
        return any(indicator in user_input.lower() for indicator in unclear_indicators)
    
    def _detect_emotional_tone(self, user_input: str) -> str:
        """Detect emotional tone in user input"""
        positive_indicators = ['great', 'excellent', 'wonderful', 'amazing', 'perfect', 'love']
        negative_indicators = ['terrible', 'awful', 'hate', 'frustrated', 'annoyed', 'disappointed']
        urgent_indicators = ['urgent', 'asap', 'immediately', 'quickly', 'emergency']
        
        input_lower = user_input.lower()
        
        if any(indicator in input_lower for indicator in urgent_indicators):
            return 'urgent'
        elif any(indicator in input_lower for indicator in positive_indicators):
            return 'positive'
        elif any(indicator in input_lower for indicator in negative_indicators):
            return 'negative'
        else:
            return 'neutral'
    
    def _detect_urgency(self, user_input: str) -> str:
        """Detect urgency level"""
        high_urgency = ['urgent', 'asap', 'immediately', 'emergency', 'critical']
        medium_urgency = ['soon', 'quickly', 'fast', 'priority']
        
        input_lower = user_input.lower()
        
        if any(indicator in input_lower for indicator in high_urgency):
            return 'high'
        elif any(indicator in input_lower for indicator in medium_urgency):
            return 'medium'
        else:
            return 'low'
    
    def _update_user_profile(self, session_id: str, analysis: Dict[str, Any]) -> None:
        """Update user interaction profile"""
        if session_id not in self.user_profiles:
            self.user_profiles[session_id] = {
                'interaction_count': 0,
                'preferred_formality': 'neutral',
                'preferred_detail_level': 'medium',
                'common_topics': [],
                'interaction_patterns': []
            }
        
        profile = self.user_profiles[session_id]
        profile['interaction_count'] += 1
        profile['interaction_patterns'].append(analysis)
        
        # Keep only recent patterns
        if len(profile['interaction_patterns']) > 10:
            profile['interaction_patterns'] = profile['interaction_patterns'][-10:]
    
    def _get_user_preferences(self, session_id: str) -> Dict[str, Any]:
        """Get user preferences from profile"""
        if session_id in self.user_profiles:
            return self.user_profiles[session_id]
        return {'preferred_detail_level': 'medium', 'preferred_formality': 'neutral'}
    
    def _get_personalization_hints(self, session_id: str, user_input: str) -> Dict[str, Any]:
        """Get personalization hints for response"""
        preferences = self._get_user_preferences(session_id)
        
        return {
            'detail_level': preferences.get('preferred_detail_level', 'medium'),
            'formality': preferences.get('preferred_formality', 'neutral'),
            'interaction_count': preferences.get('interaction_count', 0)
        }
    
    def _add_specific_terms(self, query: str) -> str:
        """Add more specific terms to vague queries"""
        # This could be enhanced with domain-specific term expansion
        return query
    
    def _extract_key_concepts(self, query: str) -> str:
        """Extract key concepts from complex queries"""
        # This could be enhanced with NLP techniques
        return query
    
    def _add_transitions(self, response: str) -> str:
        """Add smooth transitions to response"""
        # Disabled - return response as-is without adding transition words
        return response
    
    def _add_personalization(self, response: str, personalization: Dict[str, Any]) -> str:
        """Add personalization to response"""
        if personalization.get('interaction_count', 0) > 5:
            # Returning user
            if not response.startswith(('Hello', 'Hi', 'Welcome back')):
                response = f"Welcome back! {response}"
        
        return response