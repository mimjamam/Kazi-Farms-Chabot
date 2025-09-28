import streamlit as st
from backend import ChatService
from core.memory.memory_manager import MemoryManager

class KaziFarmFrontend:
    def __init__(self):
        self.chatbot = None
        self.memory_manager = MemoryManager()
        self.initialize_chatbot()
    
    def initialize_chatbot(self):
        """Initialize the chatbot with error handling"""
        try:
            self.chatbot = ChatService()
            self.chatbot.initialize()
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
            st.session_state.session_id = self.memory_manager.start_new_conversation()
    
    def display_chat_history(self):
        """Display previous chat messages"""
        for message in st.session_state.messages:
            st.chat_message(message['role']).markdown(message['content'])
    
    def handle_user_input(self, user_prompt):
        """Handle user input and generate response"""
        if not user_prompt:
            return
        
        # Display user message
        st.chat_message('user').markdown(user_prompt)
        st.session_state.messages.append({'role': 'user', 'content': user_prompt})
        
        # Add user message to memory
        self.memory_manager.add_message('user', user_prompt)
        
        # Check if chatbot is available
        if self.chatbot is None:
            error_msg = "Chatbot is not available. Please check the configuration."
            st.chat_message('assistant').markdown(error_msg)
            st.session_state.messages.append({'role': 'assistant', 'content': error_msg})
            self.memory_manager.add_message('assistant', error_msg)
            return
        
        # Get conversation context for better responses
        conversation_context = self.memory_manager.get_conversation_context(max_messages=6)
        
        # Generate response
        try:
            response = self.chatbot.get_answer_with_sources(user_prompt, conversation_context)
            
            # Display assistant response
            st.chat_message('assistant').markdown(response)
            st.session_state.messages.append({'role': 'assistant', 'content': response})
            
            # Add assistant response to memory
            self.memory_manager.add_message('assistant', response)
            
        except Exception as e:
            error_msg = f"Error processing your request: {str(e)}"
            st.chat_message('assistant').markdown(error_msg)
            st.session_state.messages.append({'role': 'assistant', 'content': error_msg})
            self.memory_manager.add_message('assistant', error_msg)
    
    def render_chat_interface(self):
        """Render the main chat interface"""
        self.display_chat_history()
        
        # User input
        user_prompt = st.chat_input("Ask your question here...")
        self.handle_user_input(user_prompt)
    
    def render_sidebar(self):
        """Render sidebar with additional information"""
        with st.sidebar:
            if self.chatbot:
                st.success("✅ Chatbot is ready")
            else:
                st.error("❌ Chatbot initialization failed")
            
            # Memory management controls
            st.markdown("---")
            st.markdown("### Chat Memory")
            
            # Show conversation summary
            if hasattr(self.memory_manager, 'current_session_id') and self.memory_manager.current_session_id:
                summary = self.memory_manager.get_conversation_summary()
                st.info(f"**Current Session:** {summary}")
            
            # Clear conversation button
            if st.button("🗑️ Clear Chat History"):
                st.session_state.messages = []
                self.memory_manager.clear_current_conversation()
                st.rerun()
            
            # Show memory stats
            stats = self.memory_manager.get_memory_stats()
            st.markdown(f"**Messages in session:** {len(st.session_state.messages)}")
            st.markdown(f"**Total conversations:** {stats['total_conversations']}")
    
    def run(self):
        """Main method to run the frontend"""
        self.render_header()
        self.initialize_session_state()
        self.render_sidebar()
        self.render_chat_interface()
