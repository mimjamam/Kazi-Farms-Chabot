"""
Context Memory Agent - Maintains conversation context and memory across sessions
"""
import json
import os
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from collections import defaultdict, deque
import hashlib

@dataclass
class ConversationTurn:
    """Represents a single conversation turn"""
    timestamp: str
    user_query: str
    bot_response: str
    topics_discussed: List[str]
    intent_type: str
    sources_used: List[str]
    confidence: float
    session_id: str

@dataclass
class TopicContext:
    """Context information for a specific topic"""
    topic_name: str
    first_mentioned: str
    last_mentioned: str
    mention_count: int
    related_queries: List[str]
    key_information: List[str]
    user_interests: List[str]

@dataclass
class SessionContext:
    """Complete context for a conversation session"""
    session_id: str
    start_time: str
    last_activity: str
    total_turns: int
    topics_discussed: Dict[str, TopicContext]
    conversation_flow: List[str]
    user_preferences: Dict[str, Any]
    unresolved_queries: List[str]

class ContextMemoryAgent:
    """Agent for maintaining conversation context and memory"""
    
    def __init__(self, memory_file: str = "core/data/context_memory.json"):
        self.memory_file = memory_file
        self.sessions: Dict[str, SessionContext] = {}
        self.topic_relationships: Dict[str, List[str]] = {}
        self.global_patterns: Dict[str, Any] = {}
        
        # Ensure directory exists
        os.makedirs(os.path.dirname(memory_file), exist_ok=True)
        
        # Load existing memory
        self._load_memory()
        
        pass  # Silent initialization
    
    def update_context(self, session_id: str, user_query: str, bot_response: str, 
                      topics: List[str], intent_type: str, sources: List[Dict[str, Any]] = None,
                      confidence: float = 0.0) -> Dict[str, Any]:
        """Update conversation context with new turn"""
        try:
            # Initialize session if new
            if session_id not in self.sessions:
                self._initialize_session(session_id)
            
            session = self.sessions[session_id]
            current_time = datetime.now().isoformat()
            
            # Create conversation turn
            turn = ConversationTurn(
                timestamp=current_time,
                user_query=user_query,
                bot_response=bot_response,
                topics_discussed=topics,
                intent_type=intent_type,
                sources_used=[s.get('title', '') for s in (sources or [])],
                confidence=confidence,
                session_id=session_id
            )
            
            # Update session context
            session.last_activity = current_time
            session.total_turns += 1
            session.conversation_flow.append(user_query)
            
            # Keep only last 5 queries for efficiency
            if len(session.conversation_flow) > 5:
                session.conversation_flow = session.conversation_flow[-5:]
            
            # Update topic contexts
            for topic in topics:
                self._update_topic_context(session, topic, user_query, bot_response, current_time)
            
            # Analyze conversation patterns
            context_insights = self._analyze_conversation_context(session, turn)
            
            # Update user preferences
            self._update_user_preferences(session, topics, intent_type, confidence)
            
            # Save memory periodically
            if session.total_turns % 5 == 0:  # Save every 5 turns
                self._save_memory()
            
            print(f"[CONTEXT MEMORY] Updated context for session {session_id[:8]}...")
            print(f"   Total turns: {session.total_turns}")
            print(f"   Topics discussed: {list(session.topics_discussed.keys())}")
            print(f"   Context insights: {len(context_insights)} generated")
            
            return context_insights
            
        except Exception as e:
            print(f"[CONTEXT MEMORY ERROR] {str(e)}")
            return {}
    
    def get_conversation_context(self, session_id: str, current_query: str = "") -> Dict[str, Any]:
        """Get relevant conversation context focusing on last 5 chats"""
        if session_id not in self.sessions:
            return {}
        
        session = self.sessions[session_id]
        
        # Focus on last 5 conversation turns for context awareness
        recent_limit = 5
        recent_topics = self._get_recent_topics(session, recent_limit)
        recent_queries = session.conversation_flow[-recent_limit:] if session.conversation_flow else []
        
        context = {
            'session_summary': {
                'total_turns': min(session.total_turns, recent_limit),  # Cap at 5 for display
                'recent_turns': len(recent_queries),
                'duration_minutes': self._calculate_session_duration(session),
                'recent_topics': recent_topics,
                'primary_interests': self._get_primary_interests_recent(session, recent_limit)
            },
            'last_5_context': self._get_last_5_context(session),
            'topic_continuity': self._analyze_topic_continuity_recent(session, current_query, recent_limit),
            'suggested_connections': self._suggest_topic_connections_recent(session, current_query, recent_limit),
            'conversation_momentum': self._assess_conversation_momentum(session, recent_limit)
        }
        
        return context
    
    def get_contextual_prompt_enhancement(self, session_id: str, current_query: str) -> str:
        """Generate context-aware prompt enhancement focusing on last 5 chats"""
        if session_id not in self.sessions:
            return ""
        
        session = self.sessions[session_id]
        enhancements = []
        recent_limit = 5
        
        # Handle affirmative responses (yes, ok, sure, etc.)
        affirmative_responses = ['yes', 'yeah', 'yep', 'ok', 'okay', 'sure', 'please', 'go ahead', 'continue']
        is_affirmative = current_query.lower().strip() in affirmative_responses
        
        # Last 5 chats context
        recent_queries = session.conversation_flow[-recent_limit:] if session.conversation_flow else []
        if recent_queries:
            enhancements.append(f"Last {len(recent_queries)} chat topics in conversation")
        
        # Recent topic context (last 5 chats)
        recent_topics = self._get_recent_topics(session, recent_limit)
        if recent_topics:
            enhancements.append(f"Recent topics discussed: {', '.join(recent_topics[:3])}")
        
        # Special handling for affirmative responses
        if is_affirmative and recent_queries:
            last_query = recent_queries[-1]
            enhancements.append(f"User said '{current_query}' - likely confirming interest in previous topic: '{last_query}'")
            enhancements.append(f"PROVIDE DETAILED INFORMATION about: {last_query}")
        
        # Topic continuity from recent chats
        continuity = self._analyze_topic_continuity_recent(session, current_query, recent_limit)
        if continuity.get('is_continuation'):
            strength = continuity.get('continuity_strength', 'medium')
            position = continuity.get('last_mention_position', 0)
            enhancements.append(f"Continuing {continuity['related_topic']} topic (mentioned {recent_limit - position} chats ago, {strength} continuity)")
        
        # Conversation momentum from last 5 chats
        momentum = self._assess_conversation_momentum(session, recent_limit)
        if momentum['chat_count'] >= 3:
            enhancements.append(f"Active conversation ({momentum['chat_count']} recent chats, {momentum['topic_focus']} focus)")
        
        # Recent topic connections
        connections = self._suggest_topic_connections_recent(session, current_query, recent_limit)
        if connections:
            enhancements.append(f"Recent connection: {connections[0]}")
        
        if enhancements:
            return "LAST 5 CHATS CONTEXT: " + " | ".join(enhancements)
        
        return ""
    
    def _initialize_session(self, session_id: str) -> None:
        """Initialize a new session context"""
        current_time = datetime.now().isoformat()
        
        self.sessions[session_id] = SessionContext(
            session_id=session_id,
            start_time=current_time,
            last_activity=current_time,
            total_turns=0,
            topics_discussed={},
            conversation_flow=[],
            user_preferences={
                'preferred_detail_level': 'medium',
                'topics_of_interest': [],
                'response_style': 'informative'
            },
            unresolved_queries=[]
        )
    
    def _update_topic_context(self, session: SessionContext, topic: str, 
                            user_query: str, bot_response: str, timestamp: str) -> None:
        """Update context for a specific topic"""
        if topic not in session.topics_discussed:
            session.topics_discussed[topic] = TopicContext(
                topic_name=topic,
                first_mentioned=timestamp,
                last_mentioned=timestamp,
                mention_count=1,
                related_queries=[user_query],
                key_information=[],
                user_interests=[]
            )
        else:
            topic_context = session.topics_discussed[topic]
            topic_context.last_mentioned = timestamp
            topic_context.mention_count += 1
            topic_context.related_queries.append(user_query)
            
            # Keep only recent queries
            if len(topic_context.related_queries) > 5:
                topic_context.related_queries = topic_context.related_queries[-5:]
        
        # Extract key information from bot response
        self._extract_key_information(session.topics_discussed[topic], bot_response)
    
    def _extract_key_information(self, topic_context: TopicContext, bot_response: str) -> None:
        """Extract key information from bot response"""
        # Simple extraction of key facts (can be enhanced with NLP)
        sentences = bot_response.split('.')
        key_sentences = []
        
        for sentence in sentences:
            sentence = sentence.strip()
            # Look for sentences with specific information
            if any(indicator in sentence.lower() for indicator in 
                  ['days', 'amount', 'percent', 'process', 'requirement', 'eligibility']):
                if len(sentence) > 20 and len(sentence) < 200:  # Reasonable length
                    key_sentences.append(sentence)
        
        # Add to key information, avoiding duplicates
        for sentence in key_sentences:
            if sentence not in topic_context.key_information:
                topic_context.key_information.append(sentence)
        
        # Keep only recent key information
        if len(topic_context.key_information) > 10:
            topic_context.key_information = topic_context.key_information[-10:]
    
    def _analyze_conversation_context(self, session: SessionContext, turn: ConversationTurn) -> Dict[str, Any]:
        """Analyze conversation context and generate insights"""
        insights = {}
        
        # Topic progression analysis
        if len(session.conversation_flow) > 1:
            insights['topic_progression'] = self._analyze_topic_progression(session)
        
        # Conversation depth analysis
        insights['conversation_depth'] = self._analyze_conversation_depth(session, turn)
        
        # User engagement analysis
        insights['user_engagement'] = self._analyze_user_engagement(session)
        
        # Information gaps
        insights['information_gaps'] = self._identify_information_gaps(session)
        
        return insights
    
    def _get_last_5_context(self, session: SessionContext) -> Dict[str, Any]:
        """Get context from last 5 conversation turns"""
        recent_limit = 5
        recent_topics = self._get_recent_topics(session, recent_limit)
        recent_queries = session.conversation_flow[-recent_limit:] if session.conversation_flow else []
        
        # Analyze patterns in last 5 chats
        topic_frequency = {}
        for topic_name, topic_context in session.topics_discussed.items():
            if topic_name in recent_topics:
                # Count mentions in recent queries
                recent_mentions = 0
                for query in recent_queries:
                    if any(keyword in query.lower() for keyword in self._get_topic_keywords(topic_name)):
                        recent_mentions += 1
                topic_frequency[topic_name] = recent_mentions
        
        return {
            'recent_topics': recent_topics,
            'recent_queries': recent_queries,
            'topic_frequency_last_5': topic_frequency,
            'conversation_depth': len(recent_queries),
            'topic_diversity': len(set(recent_topics))
        }
    
    def _get_topic_keywords(self, topic: str) -> List[str]:
        """Get keywords for a topic"""
        topic_keywords = {
            'salary': ['salary', 'pay', 'compensation', 'wage', 'allowance'],
            'leave': ['leave', 'vacation', 'holiday', 'sick', 'absence'],
            'benefits': ['benefit', 'insurance', 'medical', 'health', 'coverage'],
            'policies': ['policy', 'rule', 'procedure', 'guideline'],
            'positions': ['manager', 'officer', 'position', 'role', 'job'],
            'performance': ['performance', 'review', 'appraisal', 'evaluation']
        }
        return topic_keywords.get(topic, [topic])
    
    def _get_primary_interests_recent(self, session: SessionContext, recent_limit: int = 5) -> List[str]:
        """Get user's primary interests from last 5 chats"""
        recent_queries = session.conversation_flow[-recent_limit:] if session.conversation_flow else []
        
        # Count topic mentions in recent queries
        topic_scores = {}
        for query in recent_queries:
            query_lower = query.lower()
            for topic_name in session.topics_discussed.keys():
                keywords = self._get_topic_keywords(topic_name)
                mentions = sum(1 for keyword in keywords if keyword in query_lower)
                if mentions > 0:
                    topic_scores[topic_name] = topic_scores.get(topic_name, 0) + mentions
        
        # Sort by recent mentions
        sorted_interests = sorted(topic_scores.items(), key=lambda x: x[1], reverse=True)
        return [topic for topic, _ in sorted_interests[:3]]
    
    def _analyze_topic_continuity_recent(self, session: SessionContext, current_query: str, recent_limit: int = 5) -> Dict[str, Any]:
        """Analyze topic continuity focusing on last 5 chats"""
        current_query_lower = current_query.lower().strip()
        recent_queries = session.conversation_flow[-recent_limit:] if session.conversation_flow else []
        
        # Handle affirmative responses
        affirmative_responses = ['yes', 'yeah', 'yep', 'ok', 'okay', 'sure', 'please', 'go ahead', 'continue']
        is_affirmative = current_query_lower in affirmative_responses
        
        if is_affirmative and recent_queries:
            # For affirmative responses, assume continuation of the most recent topic
            last_query = recent_queries[-1]
            for topic_name, topic_context in session.topics_discussed.items():
                keywords = self._get_topic_keywords(topic_name)
                if any(keyword in last_query.lower() for keyword in keywords):
                    return {
                        'is_continuation': True,
                        'related_topic': topic_name,
                        'recent_mentions': 1,
                        'last_mention_position': len(recent_queries) - 1,
                        'continuity_strength': 'high',
                        'is_affirmative_response': True,
                        'original_query': last_query
                    }
        
        # Check if current query continues any recent topics
        for topic_name, topic_context in session.topics_discussed.items():
            keywords = self._get_topic_keywords(topic_name)
            
            # Check if topic appears in current query
            if any(keyword in current_query_lower for keyword in keywords):
                # Check how recently this topic was discussed
                recent_mentions = 0
                last_mention_position = -1
                
                for i, query in enumerate(recent_queries):
                    if any(keyword in query.lower() for keyword in keywords):
                        recent_mentions += 1
                        last_mention_position = i
                
                if recent_mentions > 0:
                    return {
                        'is_continuation': True,
                        'related_topic': topic_name,
                        'recent_mentions': recent_mentions,
                        'last_mention_position': last_mention_position,
                        'continuity_strength': 'high' if last_mention_position >= recent_limit - 2 else 'medium'
                    }
        
        return {'is_continuation': False}
    
    def _suggest_topic_connections_recent(self, session: SessionContext, current_query: str, recent_limit: int = 5) -> List[str]:
        """Suggest connections based on last 5 chats"""
        suggestions = []
        recent_topics = self._get_recent_topics(session, recent_limit)
        current_topics = self._extract_topics_from_query(current_query)
        
        # Common topic relationships
        topic_connections = {
            'salary': ['benefits', 'performance', 'positions'],
            'leave': ['policies', 'benefits'],
            'benefits': ['salary', 'policies'],
            'policies': ['leave', 'benefits', 'performance'],
            'positions': ['salary', 'performance'],
            'performance': ['salary', 'positions']
        }
        
        # Only suggest connections from recent conversations
        for current_topic in current_topics:
            if current_topic in topic_connections:
                for related_topic in topic_connections[current_topic]:
                    if related_topic in recent_topics and related_topic not in current_topics:
                        # Check how recently it was discussed
                        recent_queries = session.conversation_flow[-recent_limit:]
                        for i, query in enumerate(reversed(recent_queries)):
                            if any(keyword in query.lower() for keyword in self._get_topic_keywords(related_topic)):
                                chat_position = recent_limit - i
                                suggestions.append(f"This connects to {related_topic} from {chat_position} chat{'s' if chat_position > 1 else ''} ago")
                                break
        
        return suggestions[:2]  # Limit to 2 most relevant connections
    
    def _assess_conversation_momentum(self, session: SessionContext, recent_limit: int = 5) -> Dict[str, Any]:
        """Assess conversation momentum from last 5 chats"""
        recent_queries = session.conversation_flow[-recent_limit:] if session.conversation_flow else []
        
        momentum = {
            'chat_count': len(recent_queries),
            'momentum_level': 'low',
            'engagement_pattern': 'initial',
            'topic_focus': 'scattered'
        }
        
        if len(recent_queries) >= 4:
            momentum['momentum_level'] = 'high'
            momentum['engagement_pattern'] = 'engaged'
        elif len(recent_queries) >= 2:
            momentum['momentum_level'] = 'medium'
            momentum['engagement_pattern'] = 'building'
        
        # Analyze topic focus in recent chats
        recent_topics = self._get_recent_topics(session, recent_limit)
        if len(recent_topics) <= 2:
            momentum['topic_focus'] = 'focused'
        elif len(recent_topics) <= 4:
            momentum['topic_focus'] = 'exploratory'
        
        return momentum
    
    def _get_recent_topics(self, session: SessionContext, limit: int = 3) -> List[str]:
        """Get recently discussed topics"""
        # Sort topics by last mentioned time
        sorted_topics = sorted(
            session.topics_discussed.items(),
            key=lambda x: x[1].last_mentioned,
            reverse=True
        )
        
        return [topic for topic, _ in sorted_topics[:limit]]
    
    def _analyze_topic_continuity(self, session: SessionContext, current_query: str) -> Dict[str, Any]:
        """Analyze if current query continues previous topics"""
        current_query_lower = current_query.lower()
        
        # Check for topic keywords in current query
        for topic_name, topic_context in session.topics_discussed.items():
            # Simple keyword matching (can be enhanced)
            topic_keywords = {
                'salary': ['salary', 'pay', 'compensation', 'wage', 'allowance'],
                'leave': ['leave', 'vacation', 'holiday', 'sick', 'absence'],
                'benefits': ['benefit', 'insurance', 'medical', 'health'],
                'policies': ['policy', 'rule', 'procedure', 'guideline'],
                'positions': ['manager', 'officer', 'position', 'role'],
                'performance': ['performance', 'review', 'appraisal']
            }
            
            if topic_name in topic_keywords:
                if any(keyword in current_query_lower for keyword in topic_keywords[topic_name]):
                    return {
                        'is_continuation': True,
                        'related_topic': topic_name,
                        'previous_mentions': topic_context.mention_count,
                        'last_discussed': topic_context.last_mentioned
                    }
        
        return {'is_continuation': False}
    
    def _suggest_topic_connections(self, session: SessionContext, current_query: str) -> List[str]:
        """Suggest connections to previously discussed topics"""
        suggestions = []
        
        # If user asks about a new topic, suggest connections to previous topics
        current_topics = self._extract_topics_from_query(current_query)
        previous_topics = list(session.topics_discussed.keys())
        
        # Common topic relationships
        topic_connections = {
            'salary': ['benefits', 'performance', 'positions'],
            'leave': ['policies', 'benefits'],
            'benefits': ['salary', 'policies'],
            'policies': ['leave', 'benefits', 'performance'],
            'positions': ['salary', 'performance'],
            'performance': ['salary', 'positions']
        }
        
        for current_topic in current_topics:
            if current_topic in topic_connections:
                for related_topic in topic_connections[current_topic]:
                    if related_topic in previous_topics and related_topic not in current_topics:
                        suggestions.append(f"This relates to {related_topic} we discussed earlier")
        
        return suggestions[:2]  # Limit suggestions
    
    def _extract_topics_from_query(self, query: str) -> List[str]:
        """Extract topics from current query"""
        query_lower = query.lower()
        topics = []
        
        topic_keywords = {
            'salary': ['salary', 'pay', 'compensation', 'wage', 'allowance'],
            'leave': ['leave', 'vacation', 'holiday', 'sick', 'absence'],
            'benefits': ['benefit', 'insurance', 'medical', 'health'],
            'policies': ['policy', 'rule', 'procedure', 'guideline'],
            'positions': ['manager', 'officer', 'position', 'role'],
            'performance': ['performance', 'review', 'appraisal']
        }
        
        for topic, keywords in topic_keywords.items():
            if any(keyword in query_lower for keyword in keywords):
                topics.append(topic)
        
        return topics
    
    def _assess_conversation_state(self, session: SessionContext) -> Dict[str, Any]:
        """Assess the current state of the conversation"""
        state = {
            'conversation_stage': 'ongoing',
            'user_satisfaction_indicators': [],
            'potential_next_topics': [],
            'conversation_completeness': 0.0
        }
        
        # Determine conversation stage
        if session.total_turns <= 2:
            state['conversation_stage'] = 'initial'
        elif session.total_turns > 10:
            state['conversation_stage'] = 'deep'
        
        # Assess completeness based on topic coverage
        if session.topics_discussed:
            avg_mentions = sum(tc.mention_count for tc in session.topics_discussed.values()) / len(session.topics_discussed)
            state['conversation_completeness'] = min(avg_mentions / 3.0, 1.0)  # Normalize to 0-1
        
        return state
    
    def _calculate_session_duration(self, session: SessionContext) -> int:
        """Calculate session duration in minutes"""
        try:
            start = datetime.fromisoformat(session.start_time)
            last = datetime.fromisoformat(session.last_activity)
            return int((last - start).total_seconds() / 60)
        except:
            return 0
    
    def _get_primary_interests(self, session: SessionContext) -> List[str]:
        """Get user's primary interests based on conversation"""
        # Sort topics by mention count and recency
        topic_scores = {}
        
        for topic_name, topic_context in session.topics_discussed.items():
            # Score based on mention count and recency
            recency_hours = self._get_hours_since(topic_context.last_mentioned)
            recency_score = max(0, 24 - recency_hours) / 24  # Higher score for recent topics
            
            topic_scores[topic_name] = topic_context.mention_count * (1 + recency_score)
        
        # Return top interests
        sorted_interests = sorted(topic_scores.items(), key=lambda x: x[1], reverse=True)
        return [topic for topic, _ in sorted_interests[:3]]
    
    def _get_hours_since(self, timestamp: str) -> float:
        """Get hours since timestamp"""
        try:
            past = datetime.fromisoformat(timestamp)
            now = datetime.now()
            return (now - past).total_seconds() / 3600
        except:
            return 24  # Default to 24 hours if parsing fails
    
    def _update_user_preferences(self, session: SessionContext, topics: List[str], 
                               intent_type: str, confidence: float) -> None:
        """Update user preferences based on interaction patterns"""
        prefs = session.user_preferences
        
        # Update topics of interest
        for topic in topics:
            if topic not in prefs['topics_of_interest']:
                prefs['topics_of_interest'].append(topic)
        
        # Keep only recent interests
        if len(prefs['topics_of_interest']) > 10:
            prefs['topics_of_interest'] = prefs['topics_of_interest'][-10:]
        
        # Infer preferred detail level based on query patterns
        if intent_type == 'question' and confidence > 0.8:
            # User asks detailed questions, might prefer detailed responses
            if prefs['preferred_detail_level'] == 'low':
                prefs['preferred_detail_level'] = 'medium'
            elif prefs['preferred_detail_level'] == 'medium':
                prefs['preferred_detail_level'] = 'high'
    
    def _analyze_topic_progression(self, session: SessionContext) -> Dict[str, Any]:
        """Analyze how topics progress in conversation"""
        # Simple analysis of topic flow
        recent_flow = session.conversation_flow[-5:]  # Last 5 queries
        
        return {
            'flow_pattern': 'exploratory' if len(set(self._extract_topics_from_query(' '.join(recent_flow)))) > 2 else 'focused',
            'topic_depth': sum(tc.mention_count for tc in session.topics_discussed.values()) / max(len(session.topics_discussed), 1)
        }
    
    def _analyze_conversation_depth(self, session: SessionContext, turn: ConversationTurn) -> str:
        """Analyze conversation depth"""
        if turn.confidence > 0.8 and len(turn.sources_used) > 2:
            return 'deep'
        elif turn.confidence > 0.6:
            return 'medium'
        else:
            return 'surface'
    
    def _analyze_user_engagement(self, session: SessionContext) -> str:
        """Analyze user engagement level"""
        if session.total_turns > 8:
            return 'high'
        elif session.total_turns > 3:
            return 'medium'
        else:
            return 'initial'
    
    def _identify_information_gaps(self, session: SessionContext) -> List[str]:
        """Identify potential information gaps"""
        gaps = []
        
        # Check for incomplete topic coverage
        for topic_name, topic_context in session.topics_discussed.items():
            if topic_context.mention_count == 1 and len(topic_context.key_information) < 2:
                gaps.append(f"Limited information provided about {topic_name}")
        
        return gaps[:3]  # Limit to top 3 gaps
    
    def _load_memory(self) -> None:
        """Load memory from persistent storage"""
        try:
            if os.path.exists(self.memory_file):
                with open(self.memory_file, 'r') as f:
                    data = json.load(f)
                
                # Reconstruct session objects
                for session_id, session_data in data.get('sessions', {}).items():
                    # Convert topic contexts
                    topics_discussed = {}
                    for topic_name, topic_data in session_data.get('topics_discussed', {}).items():
                        topics_discussed[topic_name] = TopicContext(**topic_data)
                    
                    session_data['topics_discussed'] = topics_discussed
                    self.sessions[session_id] = SessionContext(**session_data)
                
                self.topic_relationships = data.get('topic_relationships', {})
                self.global_patterns = data.get('global_patterns', {})
                
                print(f"[CONTEXT MEMORY] Loaded {len(self.sessions)} sessions from memory")
        
        except Exception as e:
            print(f"[CONTEXT MEMORY] Error loading memory: {str(e)}")
    
    def _save_memory(self) -> None:
        """Save memory to persistent storage"""
        try:
            # Convert to serializable format
            data = {
                'sessions': {},
                'topic_relationships': self.topic_relationships,
                'global_patterns': self.global_patterns,
                'last_updated': datetime.now().isoformat()
            }
            
            for session_id, session in self.sessions.items():
                # Convert topic contexts to dict
                topics_dict = {}
                for topic_name, topic_context in session.topics_discussed.items():
                    topics_dict[topic_name] = asdict(topic_context)
                
                session_dict = asdict(session)
                session_dict['topics_discussed'] = topics_dict
                data['sessions'][session_id] = session_dict
            
            # Save to file
            with open(self.memory_file, 'w') as f:
                json.dump(data, f, indent=2)
            
            print(f"[CONTEXT MEMORY] Saved {len(self.sessions)} sessions to memory")
        
        except Exception as e:
            print(f"[CONTEXT MEMORY] Error saving memory: {str(e)}")
    
    def cleanup_old_sessions(self, days_old: int = 30) -> None:
        """Clean up old sessions to manage memory"""
        cutoff_date = datetime.now() - timedelta(days=days_old)
        
        sessions_to_remove = []
        for session_id, session in self.sessions.items():
            try:
                last_activity = datetime.fromisoformat(session.last_activity)
                if last_activity < cutoff_date:
                    sessions_to_remove.append(session_id)
            except:
                # Remove sessions with invalid timestamps
                sessions_to_remove.append(session_id)
        
        for session_id in sessions_to_remove:
            del self.sessions[session_id]
        
        if sessions_to_remove:
            print(f"[CONTEXT MEMORY] Cleaned up {len(sessions_to_remove)} old sessions")
            self._save_memory()