"""
Proactive Agent - Makes chatbot proactive by anticipating user needs and suggesting relevant information
"""
import re
import json
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta
from collections import defaultdict, Counter

@dataclass
class ProactiveInsight:
    """Represents a proactive insight or suggestion"""
    insight_type: str
    title: str
    content: str
    relevance_score: float
    trigger_context: str
    suggested_actions: List[str]
    related_topics: List[str]

@dataclass
class ProactiveResponse:
    """Enhanced response with proactive elements"""
    original_response: str
    proactive_insights: List[ProactiveInsight]
    suggested_questions: List[str]
    related_topics: List[str]
    conversation_guidance: str
    anticipatory_info: str

class ProactiveAgent:
    """Agent that makes the chatbot proactive by anticipating user needs"""
    
    def __init__(self):
        self.user_patterns = defaultdict(list)
        self.conversation_context = defaultdict(dict)
        self.topic_relationships = self._build_topic_relationships()
        self.seasonal_insights = self._build_seasonal_insights()
        self.common_workflows = self._build_common_workflows()
        
        pass  # Silent initialization
    
    def enhance_response(self, query: str, response: str, session_id: str, 
                        sources: List[Dict[str, Any]] = None) -> ProactiveResponse:
        """Enhance response with proactive elements"""
        try:
            # Analyze current query and context
            query_analysis = self._analyze_query_intent(query)
            context = self.conversation_context.get(session_id, {})
            
            # Generate proactive insights
            insights = self._generate_proactive_insights(query, response, query_analysis, context, sources)
            
            # Suggest follow-up questions (only if we have sources)
            suggested_questions = self._generate_suggested_questions(query, query_analysis, context, sources)
            
            # Find related topics
            related_topics = self._find_related_topics(query_analysis['topics'])
            
            # Generate conversation guidance (only if we have sources)
            guidance = self._generate_conversation_guidance(query_analysis, context, sources)
            
            # Generate anticipatory information (only if we have sources)
            anticipatory_info = self._generate_anticipatory_info(query_analysis, context, sources)
            
            # Update conversation context
            self._update_conversation_context(session_id, query, query_analysis)
            
            return ProactiveResponse(
                original_response=response,
                proactive_insights=insights,
                suggested_questions=suggested_questions,
                related_topics=related_topics,
                conversation_guidance=guidance,
                anticipatory_info=anticipatory_info
            )
            
        except Exception as e:
            print(f"[PROACTIVE AGENT ERROR] {str(e)}")
            return ProactiveResponse(
                original_response=response,
                proactive_insights=[],
                suggested_questions=[],
                related_topics=[],
                conversation_guidance="",
                anticipatory_info=""
            )
    
    def _analyze_query_intent(self, query: str) -> Dict[str, Any]:
        """Analyze query to understand user intent and context"""
        query_lower = query.lower()
        
        # Detect topics
        topics = []
        topic_patterns = {
            'salary': ['salary', 'pay', 'compensation', 'wage', 'income', 'allowance'],
            'leave': ['leave', 'vacation', 'holiday', 'sick', 'absence', 'time off'],
            'benefits': ['benefit', 'insurance', 'medical', 'health', 'coverage'],
            'policies': ['policy', 'rule', 'regulation', 'procedure', 'guideline'],
            'positions': ['manager', 'officer', 'executive', 'position', 'role', 'job'],
            'performance': ['performance', 'appraisal', 'review', 'evaluation'],
            'training': ['training', 'development', 'course', 'skill', 'learning']
        }
        
        for topic, keywords in topic_patterns.items():
            if any(keyword in query_lower for keyword in keywords):
                topics.append(topic)
        
        # Detect intent type
        intent_type = 'information_seeking'
        if any(word in query_lower for word in ['how', 'when', 'where', 'what', 'why']):
            intent_type = 'question'
        elif any(word in query_lower for word in ['can i', 'am i eligible', 'do i qualify']):
            intent_type = 'eligibility_check'
        elif any(word in query_lower for word in ['apply', 'request', 'submit']):
            intent_type = 'action_oriented'
        
        # Detect urgency
        urgency = 'normal'
        if any(word in query_lower for word in ['urgent', 'asap', 'immediately', 'emergency']):
            urgency = 'high'
        elif any(word in query_lower for word in ['when possible', 'sometime', 'eventually']):
            urgency = 'low'
        
        return {
            'topics': topics,
            'intent_type': intent_type,
            'urgency': urgency,
            'query_length': len(query.split()),
            'is_specific': len(topics) == 1 and len(query.split()) <= 5
        }
    
    def _generate_proactive_insights(self, query: str, response: str, 
                                   query_analysis: Dict[str, Any], context: Dict[str, Any],
                                   sources: List[Dict[str, Any]] = None) -> List[ProactiveInsight]:
        """Generate database-aware proactive insights based on available sources"""
        insights = []
        
        # Only generate insights if we have actual sources/documents
        if not sources or len(sources) == 0:
            print("   No sources available - skipping proactive insights")
            return insights
        
        # Extract available topics from actual sources
        available_topics = self._extract_topics_from_sources(sources)
        print(f"   Available topics from sources: {available_topics}")
        
        # Only suggest insights for topics that exist in the database
        for topic in query_analysis['topics']:
            if topic in available_topics:
                # Generate insight based on actual source content
                source_insight = self._generate_source_based_insight(topic, sources)
                if source_insight:
                    insights.append(source_insight)
        
        # Generate insights for related topics that exist in sources
        related_available_topics = [topic for topic in available_topics 
                                  if topic not in query_analysis['topics']]
        
        if related_available_topics:
            related_insight = self._generate_related_topics_insight(related_available_topics[:2], sources)
            if related_insight:
                insights.append(related_insight)
        
        # Only add workflow insights if we have relevant sources
        if query_analysis['intent_type'] == 'action_oriented' and available_topics:
            workflow_insight = self._generate_database_workflow_insight(query_analysis['topics'], sources)
            if workflow_insight:
                insights.append(workflow_insight)
        
        return insights[:2]  # Limit to top 2 database-backed insights
    
    def _extract_topics_from_sources(self, sources: List[Dict[str, Any]]) -> List[str]:
        """Extract topics that are actually available in the source documents"""
        available_topics = set()
        
        for source in sources:
            title = source.get('title', '').lower()
            content_preview = source.get('content_preview', '').lower()
            source_text = title + ' ' + content_preview
            
            # Check for topic indicators in actual source content
            if any(word in source_text for word in ['salary', 'pay', 'compensation', 'wage', 'allowance']):
                available_topics.add('salary')
            if any(word in source_text for word in ['leave', 'vacation', 'holiday', 'sick', 'absence']):
                available_topics.add('leave')
            if any(word in source_text for word in ['benefit', 'insurance', 'medical', 'health', 'coverage']):
                available_topics.add('benefits')
            if any(word in source_text for word in ['policy', 'rule', 'regulation', 'procedure', 'guideline']):
                available_topics.add('policies')
            if any(word in source_text for word in ['manager', 'officer', 'executive', 'position', 'role']):
                available_topics.add('positions')
            if any(word in source_text for word in ['performance', 'appraisal', 'review', 'evaluation']):
                available_topics.add('performance')
        
        return list(available_topics)
    
    def _generate_source_based_insight(self, topic: str, sources: List[Dict[str, Any]]) -> Optional[ProactiveInsight]:
        """Generate insight based on actual source content"""
        relevant_sources = []
        
        for source in sources:
            title = source.get('title', '').lower()
            content_preview = source.get('content_preview', '').lower()
            source_text = title + ' ' + content_preview
            
            # Check if source is relevant to the topic
            topic_keywords = {
                'salary': ['salary', 'pay', 'compensation', 'wage', 'allowance'],
                'leave': ['leave', 'vacation', 'holiday', 'sick', 'absence'],
                'benefits': ['benefit', 'insurance', 'medical', 'health', 'coverage'],
                'policies': ['policy', 'rule', 'regulation', 'procedure'],
                'positions': ['manager', 'officer', 'executive', 'position', 'role'],
                'performance': ['performance', 'appraisal', 'review', 'evaluation']
            }
            
            if topic in topic_keywords:
                if any(keyword in source_text for keyword in topic_keywords[topic]):
                    relevant_sources.append(source)
        
        if relevant_sources:
            # Create insight based on actual available documents
            source_titles = [source.get('title', 'Document') for source in relevant_sources[:2]]
            
            return ProactiveInsight(
                insight_type="database_backed",
                title=f"Available {topic.title()} Information",
                content=f"I found {len(relevant_sources)} document(s) about {topic}: {', '.join(source_titles)}. I can provide detailed information from these sources.",
                relevance_score=0.9,
                trigger_context=f"User asked about {topic}, found {len(relevant_sources)} relevant documents",
                suggested_actions=[f"Ask specific questions about {topic}"],
                related_topics=[]
            )
        
        return None
    
    def _generate_related_topics_insight(self, related_topics: List[str], sources: List[Dict[str, Any]]) -> Optional[ProactiveInsight]:
        """Generate insight about related topics available in the database"""
        if not related_topics:
            return None
        
        return ProactiveInsight(
            insight_type="related_available",
            title="Related Information Available",
            content=f"I also have information about: {', '.join(related_topics)}. These topics are often related and might be helpful.",
            relevance_score=0.7,
            trigger_context=f"Found related topics in database: {related_topics}",
            suggested_actions=[f"Ask about {topic}" for topic in related_topics],
            related_topics=related_topics
        )
    
    def _generate_database_workflow_insight(self, topics: List[str], sources: List[Dict[str, Any]]) -> Optional[ProactiveInsight]:
        """Generate workflow insight only if we have relevant documents"""
        if 'leave' in topics:
            # Check if we actually have leave-related documents
            leave_sources = [s for s in sources if any(word in s.get('title', '').lower() + ' ' + s.get('content_preview', '').lower() 
                                                     for word in ['leave', 'vacation', 'holiday', 'absence'])]
            
            if leave_sources:
                return ProactiveInsight(
                    insight_type="workflow_available",
                    title="Leave Process Information Available",
                    content=f"I have {len(leave_sources)} document(s) with leave process information. I can guide you through the application steps.",
                    relevance_score=0.8,
                    trigger_context="User interested in leave, found relevant process documents",
                    suggested_actions=["Ask about leave application process", "Check leave eligibility"],
                    related_topics=['policies']
                )
        
        return None
    
    def _generate_suggested_questions(self, query: str, query_analysis: Dict[str, Any], 
                                    context: Dict[str, Any], sources: List[Dict[str, Any]] = None) -> List[str]:
        """Generate suggested follow-up questions only when we have database sources"""
        suggestions = []
        
        # Only suggest questions if we have actual sources/documents
        if not sources or len(sources) == 0:
            return []  # No suggestions without sources
        
        # Generate source-based suggestions
        available_topics = self._extract_topics_from_sources(sources)
        
        if available_topics:
            # Only suggest questions about topics we actually have information for
            for topic in available_topics[:2]:  # Limit to 2 topics
                if topic == 'salary':
                    suggestions.append("What are the specific salary grades and allowances?")
                elif topic == 'leave':
                    suggestions.append("What is the leave application process?")
                elif topic == 'benefits':
                    suggestions.append("What medical benefits are available?")
                elif topic == 'policies':
                    suggestions.append("What are the key policy requirements?")
        
        # Add context-based suggestions only if we have sources
        if context.get('previous_topics') and sources:
            prev_topics = context['previous_topics']
            if len(prev_topics) > 0 and prev_topics[-1] in available_topics:
                suggestions.append(f"How does this connect to {prev_topics[-1]} information?")
        
        return suggestions[:3]  # Limit to 3 database-backed suggestions
    
    def _find_related_topics(self, current_topics: List[str]) -> List[str]:
        """Find topics related to current query"""
        related = set()
        
        for topic in current_topics:
            if topic in self.topic_relationships:
                related.update(self.topic_relationships[topic]['related_topics'])
        
        return list(related - set(current_topics))[:3]  # Exclude current topics
    
    def _generate_conversation_guidance(self, query_analysis: Dict[str, Any], 
                                      context: Dict[str, Any], sources: List[Dict[str, Any]] = None) -> str:
        """Generate conservative guidance only when we have relevant sources"""
        # Only provide guidance if we have actual sources
        if not sources or len(sources) == 0:
            return ""  # No guidance without sources
        
        # Don't show guidance messages to users - let the main response handle this
        return ""
    
    def _generate_anticipatory_info(self, query_analysis: Dict[str, Any], 
                                  context: Dict[str, Any], sources: List[Dict[str, Any]] = None) -> str:
        """Generate anticipatory information only when we have sources"""
        # Only provide anticipatory info if we have sources
        if not sources or len(sources) == 0:
            return ""
        
        # Only provide anticipatory info if we have relevant context and sources
        if query_analysis['intent_type'] == 'eligibility_check' and sources:
            return "Tip: Having your employee ID ready can help with specific queries."
        
        return ""  # Conservative approach - no anticipatory info without sources
    
    def _update_conversation_context(self, session_id: str, query: str, 
                                   query_analysis: Dict[str, Any]) -> None:
        """Update conversation context for future proactive responses"""
        if session_id not in self.conversation_context:
            self.conversation_context[session_id] = {
                'topics_discussed': [],
                'intent_history': [],
                'query_count': 0,
                'session_start': datetime.now().isoformat()
            }
        
        context = self.conversation_context[session_id]
        context['topics_discussed'].extend(query_analysis['topics'])
        context['intent_history'].append(query_analysis['intent_type'])
        context['query_count'] += 1
        context['last_query'] = query
        context['last_query_time'] = datetime.now().isoformat()
        
        # Keep only recent topics (last 5)
        context['previous_topics'] = list(set(context['topics_discussed'][-5:]))
    
    def _build_topic_relationships(self) -> Dict[str, Dict[str, Any]]:
        """Build relationships between topics for proactive suggestions"""
        return {
            'salary': {
                'related_topics': ['benefits', 'performance', 'positions'],
                'proactive_info': "Salary structures are often linked to performance reviews and position grades. You might also want to know about allowances and benefits.",
                'suggested_actions': [
                    "Check your current salary grade",
                    "Review performance evaluation criteria",
                    "Explore available allowances"
                ]
            },
            'leave': {
                'related_topics': ['policies', 'benefits', 'performance'],
                'proactive_info': "Leave policies are connected to your employment terms and may affect performance evaluations. Medical leave might involve insurance benefits.",
                'suggested_actions': [
                    "Review leave application process",
                    "Check leave balance",
                    "Understand approval workflow"
                ]
            },
            'benefits': {
                'related_topics': ['salary', 'policies', 'positions'],
                'proactive_info': "Benefits packages vary by position and are part of your total compensation. Some benefits require enrollment or have eligibility criteria.",
                'suggested_actions': [
                    "Review benefit enrollment options",
                    "Check eligibility criteria",
                    "Understand claim procedures"
                ]
            },
            'policies': {
                'related_topics': ['benefits', 'performance', 'training'],
                'proactive_info': "Company policies govern various aspects of employment including benefits, performance standards, and professional development.",
                'suggested_actions': [
                    "Review employee handbook",
                    "Understand compliance requirements",
                    "Check policy updates"
                ]
            }
        }
    
    def _build_seasonal_insights(self) -> Dict[int, str]:
        """Build seasonal insights based on time of year"""
        return {
            1: "New Year: Time for goal setting and performance planning.",
            3: "Q1 Review: Performance evaluations and salary reviews typically occur.",
            6: "Mid-Year: Leave planning and training enrollment periods.",
            9: "Q3: Budget planning and benefit enrollment seasons.",
            12: "Year-End: Bonus calculations and annual leave settlements."
        }
    
    def _build_common_workflows(self) -> Dict[str, List[str]]:
        """Build common workflow patterns"""
        return {
            'leave_application': [
                "Check leave balance",
                "Plan leave dates",
                "Submit application",
                "Get supervisor approval",
                "Receive confirmation"
            ],
            'salary_inquiry': [
                "Verify current grade",
                "Check salary structure",
                "Review allowances",
                "Understand deductions",
                "Plan for increments"
            ],
            'benefit_enrollment': [
                "Check eligibility",
                "Review options",
                "Submit enrollment",
                "Provide documents",
                "Receive confirmation"
            ]
        }
    
    def _generate_workflow_insight(self, topics: List[str]) -> Optional[ProactiveInsight]:
        """Generate workflow-based insight"""
        if 'leave' in topics:
            return ProactiveInsight(
                insight_type="workflow_guidance",
                title="Leave Application Process",
                content="I can guide you through the complete leave application process step by step.",
                relevance_score=0.9,
                trigger_context="User interested in leave-related actions",
                suggested_actions=self.common_workflows['leave_application'],
                related_topics=['policies', 'benefits']
            )
        return None
    
    def _generate_seasonal_insight(self) -> Optional[ProactiveInsight]:
        """Generate seasonal insight based on current time"""
        current_month = datetime.now().month
        if current_month in self.seasonal_insights:
            return ProactiveInsight(
                insight_type="seasonal_info",
                title="Seasonal Information",
                content=self.seasonal_insights[current_month],
                relevance_score=0.6,
                trigger_context="Time-based insight",
                suggested_actions=["Plan accordingly", "Check relevant deadlines"],
                related_topics=[]
            )
        return None
    
    def _generate_context_insight(self, context: Dict[str, Any], 
                                query_analysis: Dict[str, Any]) -> Optional[ProactiveInsight]:
        """Generate insight based on conversation context"""
        if context.get('query_count', 0) > 2:
            prev_topics = set(context.get('previous_topics', []))
            current_topics = set(query_analysis['topics'])
            
            if prev_topics and not prev_topics.intersection(current_topics):
                return ProactiveInsight(
                    insight_type="context_connection",
                    title="Connecting Your Queries",
                    content=f"I notice you've asked about {', '.join(prev_topics)} and now {', '.join(current_topics)}. These topics are often related in HR policies.",
                    relevance_score=0.7,
                    trigger_context="Multiple different topics in conversation",
                    suggested_actions=["Explore connections between topics"],
                    related_topics=list(prev_topics.union(current_topics))
                )
        return None