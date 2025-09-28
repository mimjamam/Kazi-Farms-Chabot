import os
import json
from datetime import datetime
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, asdict
import streamlit as st

@dataclass
class Message:
    """Represents a single message in a conversation"""
    role: str  # 'user' or 'assistant'
    content: str
    timestamp: str
    metadata: Optional[Dict[str, Any]] = None

@dataclass
class Conversation:
    """Represents a complete conversation session"""
    session_id: str
    messages: List[Message]
    created_at: str
    last_updated: str
    summary: Optional[str] = None

class MemoryManager:
    """Manages conversation memory and context for the chatbot"""
    
    def __init__(self, memory_dir: str = "chatbot_memory"):
        self.memory_dir = memory_dir
        self.current_session_id = None
        self.conversations_file = os.path.join(memory_dir, "conversations.json")
        
        os.makedirs(memory_dir, exist_ok=True)
        self.conversations = self._load_conversations()
        
        if 'current_conversation' not in st.session_state:
            st.session_state.current_conversation = None
        if 'conversation_history' not in st.session_state:
            st.session_state.conversation_history = []
    
    def _load_conversations(self) -> Dict[str, Conversation]:
        """Load conversations from disk"""
        if os.path.exists(self.conversations_file):
            try:
                with open(self.conversations_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    conversations = {}
                    for session_id, conv_data in data.items():
                        messages = [Message(**msg) for msg in conv_data['messages']]
                        conversations[session_id] = Conversation(
                            session_id=conv_data['session_id'],
                            messages=messages,
                            created_at=conv_data['created_at'],
                            last_updated=conv_data['last_updated'],
                            summary=conv_data.get('summary')
                        )
                    return conversations
            except Exception as e:
                print(f"Error loading conversations: {e}")
                return {}
        return {}
    
    def _save_conversations(self):
        """Save conversations to disk"""
        try:
            data = {}
            for session_id, conv in self.conversations.items():
                data[session_id] = {
                    'session_id': conv.session_id,
                    'messages': [asdict(msg) for msg in conv.messages],
                    'created_at': conv.created_at,
                    'last_updated': conv.last_updated,
                    'summary': conv.summary
                }
            
            with open(self.conversations_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"Error saving conversations: {e}")
    
    def start_new_conversation(self) -> str:
        """Start a new conversation session"""
        session_id = f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.current_session_id = session_id
        
        # Create new conversation
        new_conversation = Conversation(
            session_id=session_id,
            messages=[],
            created_at=datetime.now().isoformat(),
            last_updated=datetime.now().isoformat()
        )
        
        self.conversations[session_id] = new_conversation
        st.session_state.current_conversation = session_id
        st.session_state.conversation_history = []
        
        return session_id
    
    def add_message(self, role: str, content: str, metadata: Optional[Dict[str, Any]] = None):
        """Add a message to the current conversation"""
        if not self.current_session_id:
            self.start_new_conversation()
        
        message = Message(
            role=role,
            content=content,
            timestamp=datetime.now().isoformat(),
            metadata=metadata
        )
        
        # Add to conversation
        if self.current_session_id in self.conversations:
            self.conversations[self.current_session_id].messages.append(message)
            self.conversations[self.current_session_id].last_updated = datetime.now().isoformat()
        
        # Add to session state for immediate display
        st.session_state.conversation_history.append({
            'role': role,
            'content': content,
            'timestamp': message.timestamp
        })
        
        # Save to disk
        self._save_conversations()
    
    def get_conversation_context(self, max_messages: int = 10) -> str:
        """Get recent conversation context for the LLM"""
        if not self.current_session_id or self.current_session_id not in self.conversations:
            return ""
        
        messages = self.conversations[self.current_session_id].messages
        recent_messages = messages[-max_messages:] if len(messages) > max_messages else messages
        
        context = "Previous conversation context:\n"
        for msg in recent_messages:
            context += f"{msg.role.capitalize()}: {msg.content}\n"
        
        return context
    
    def get_conversation_summary(self) -> str:
        """Get a summary of the current conversation"""
        if not self.current_session_id or self.current_session_id not in self.conversations:
            return ""
        
        conv = self.conversations[self.current_session_id]
        if conv.summary:
            return conv.summary
        
        # Generate a simple summary based on message count and topics
        message_count = len(conv.messages)
        user_messages = [msg for msg in conv.messages if msg.role == 'user']
        
        if message_count == 0:
            return "No conversation yet."
        
        # Extract key topics from user messages (simple keyword extraction)
        topics = set()
        for msg in user_messages:
            words = msg.content.lower().split()
            # Look for common question words and topics
            for word in words:
                if word in ['price', 'cost', 'product', 'chicken', 'farm', 'order', 'delivery']:
                    topics.add(word)
        
        topic_str = ", ".join(list(topics)[:5]) if topics else "general inquiry"
        return f"Conversation with {message_count} messages about: {topic_str}"
    
    def load_conversation(self, session_id: str) -> bool:
        """Load a specific conversation"""
        if session_id in self.conversations:
            self.current_session_id = session_id
            conv = self.conversations[session_id]
            
            # Update session state
            st.session_state.current_conversation = session_id
            st.session_state.conversation_history = [
                {
                    'role': msg.role,
                    'content': msg.content,
                    'timestamp': msg.timestamp
                }
                for msg in conv.messages
            ]
            return True
        return False
    
    def get_all_conversations(self) -> List[Dict[str, Any]]:
        """Get list of all conversations with metadata"""
        conversations_list = []
        for session_id, conv in self.conversations.items():
            conversations_list.append({
                'session_id': session_id,
                'created_at': conv.created_at,
                'last_updated': conv.last_updated,
                'message_count': len(conv.messages),
                'summary': conv.summary or self.get_conversation_summary()
            })
        
        # Sort by last updated (most recent first)
        conversations_list.sort(key=lambda x: x['last_updated'], reverse=True)
        return conversations_list
    
    def clear_current_conversation(self):
        """Clear the current conversation"""
        if self.current_session_id and self.current_session_id in self.conversations:
            del self.conversations[self.current_session_id]
            self._save_conversations()
        
        self.current_session_id = None
        st.session_state.current_conversation = None
        st.session_state.conversation_history = []
    
    def clear_all_conversations(self):
        """Clear all conversations"""
        self.conversations = {}
        self.current_session_id = None
        st.session_state.current_conversation = None
        st.session_state.conversation_history = []
        self._save_conversations()
    
    def export_conversations(self, filepath: str):
        """Export all conversations to a JSON file"""
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                data = {}
                for session_id, conv in self.conversations.items():
                    data[session_id] = {
                        'session_id': conv.session_id,
                        'messages': [asdict(msg) for msg in conv.messages],
                        'created_at': conv.created_at,
                        'last_updated': conv.last_updated,
                        'summary': conv.summary
                    }
                json.dump(data, f, indent=2, ensure_ascii=False)
            return True
        except Exception as e:
            print(f"Error exporting conversations: {e}")
            return False
    
    def get_memory_stats(self) -> Dict[str, Any]:
        """Get statistics about stored conversations"""
        total_conversations = len(self.conversations)
        total_messages = sum(len(conv.messages) for conv in self.conversations.values())
        
        return {
            'total_conversations': total_conversations,
            'total_messages': total_messages,
            'current_session_id': self.current_session_id,
            'memory_file_size': os.path.getsize(self.conversations_file) if os.path.exists(self.conversations_file) else 0
        }
