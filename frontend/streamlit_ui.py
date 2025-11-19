import streamlit as st
import time
import uuid
from backend.langgraph_chat_service import LangGraphChatService
from config import Settings

class KaziFarmFrontend:
    def __init__(self):
        self.chatbot = None
        self.settings = Settings()
        self.initialize_chatbot()
    
    def initialize_chatbot(self):
        """Initialize the chatbot with error handling"""
        try:
            self.chatbot = LangGraphChatService()
            st.success("Chatbot initialized successfully with LangGraph!")
        except Exception as e:
            st.error(f"Failed to initialize chatbot: {str(e)}")
            self.chatbot = None
    
    def render_header(self):
        """Render the app header"""
        st.set_page_config(page_title="Kazi Farms Assistant", page_icon="🐔")
        st.title("Kazi Farms Assistant")
        st.markdown("Your official Kazi Farms chatbot.")
    
    def initialize_session_state(self):
        """Initialize session state for chat history"""
        if 'messages' not in st.session_state:
            st.session_state.messages = []
        if 'session_id' not in st.session_state:
            st.session_state.session_id = str(uuid.uuid4())
        if 'last_activity' not in st.session_state:
            st.session_state.last_activity = time.time()
        if 'auto_cleanup_enabled' not in st.session_state:
            st.session_state.auto_cleanup_enabled = self.settings.AUTO_CLEANUP_ENABLED
    
    def display_chat_history(self):
        """Display previous chat messages"""
        for message in st.session_state.messages:
            st.chat_message(message['role']).markdown(message['content'])
    
    def handle_user_input(self, user_prompt):
        """Handle user input and generate response"""
        if not user_prompt:
            return
        
        # Update last activity timestamp
        st.session_state.last_activity = time.time()
        
        # Display user message
        st.chat_message('user').markdown(user_prompt)
        st.session_state.messages.append({'role': 'user', 'content': user_prompt})
        
        # Check if chatbot is available
        if self.chatbot is None:
            error_msg = "Chatbot is not available. Please check the configuration."
            st.chat_message('assistant').markdown(error_msg)
            st.session_state.messages.append({'role': 'assistant', 'content': error_msg})
            return
        
        # Generate response using the new agent-based architecture
        try:
            response = self.chatbot.get_answer_with_sources(user_prompt, session_id=st.session_state.session_id)
            
            # Display assistant response
            st.chat_message('assistant').markdown(response)
            st.session_state.messages.append({'role': 'assistant', 'content': response})
            
        except Exception as e:
            error_msg = f"Error processing your request: {str(e)}"
            st.chat_message('assistant').markdown(error_msg)
            st.session_state.messages.append({'role': 'assistant', 'content': error_msg})
        
        # Update last activity timestamp
        st.session_state.last_activity = time.time()
    
    def render_chat_interface(self):
        """Render the main chat interface"""
        self.display_chat_history()
        
        # User input
        user_prompt = st.chat_input("Ask your question here...")
        self.handle_user_input(user_prompt)
    
    def check_session_timeout(self):
        """Check if session should be cleaned up due to inactivity"""
        if not st.session_state.auto_cleanup_enabled:
            return
        
        current_time = time.time()
        # Clean up if no activity for more than configured timeout
        timeout_duration = self.settings.SESSION_TIMEOUT_MINUTES * 60
        
        if current_time - st.session_state.last_activity > timeout_duration:
            st.info("🕐 Session timed out due to inactivity. Memory will be cleared.")
            if self.chatbot:
                self.chatbot.clear_conversation(st.session_state.session_id)
            st.session_state.messages = []
            st.session_state.last_activity = current_time
    
    def run(self):
        """Main method to run the frontend"""
        self.render_header()
        self.initialize_session_state()
        self.check_session_timeout()
        self.render_chat_interface()
