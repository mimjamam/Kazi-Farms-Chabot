"""
Kazi Farms Chatbot - Simple Frontend for FastAPI Backend
"""
import streamlit as st
import requests
import json
from datetime import datetime
import time
import uuid
import datetime

# Page configuration
st.set_page_config(
    page_title="Kazi Farms Assistant",
    page_icon="🐔",
    layout="wide"
)

# Custom CSS for citation styling
st.markdown("""
<style>
.inline-citation {
    background-color: #007bff;
    color: white;
    padding: 2px 6px;
    border-radius: 3px;
    font-size: 0.8em;
    font-weight: bold;
    margin-left: 2px;
    text-decoration: none;
    display: inline-block;
}




</style>
""", unsafe_allow_html=True)

class FastAPIClient:
    """Simple client for FastAPI backend"""
    
    def __init__(self, base_url: str = None):
        # Get base URL from environment variable or use default
        import os
        from dotenv import load_dotenv
        load_dotenv()
        
        if base_url is None:
            base_url = os.getenv("FASTAPI_BASE_URL", "http://localhost:8000")
        
        self.base_url = base_url
    
    def check_topic_availability(self, topic: str) -> bool:
        """Check if we have documents for a specific topic"""
        try:
            documents = self.get_documents()
            available_docs = documents.get("documents", [])
            
            topic_keywords = {
                'salary': ['salary', 'pay', 'compensation', 'wage', 'allowance'],
                'leave': ['leave', 'vacation', 'holiday', 'absence', 'sick'],
                'benefits': ['benefit', 'insurance', 'medical', 'health', 'coverage'],
                'policies': ['policy', 'procedure', 'guideline', 'manual', 'handbook'],
                'positions': ['manager', 'position', 'job', 'role', 'hierarchy'],
                'performance': ['performance', 'review', 'training', 'development']
            }
            
            if topic in topic_keywords:
                keywords = topic_keywords[topic]
                for doc in available_docs:
                    title_lower = doc.get('title', '').lower()
                    if any(keyword in title_lower for keyword in keywords):
                        return True
            
            return False
        except:
            return False
    
    def health_check(self):
        """Check API health"""
        try:
            response = requests.get(f"{self.base_url}/health", timeout=10)
            return response.json() if response.status_code == 200 else {"status": "error"}
        except requests.exceptions.Timeout:
            return {"status": "timeout", "message": "Health check timed out"}
        except requests.exceptions.ConnectionError:
            return {"status": "connection_error", "message": "Cannot connect to server"}
        except Exception as e:
            return {"status": "error", "message": str(e)}
    
    def send_message(self, message: str, session_id: str = ""):
        """Send message to API"""
        try:
            response = requests.post(
                f"{self.base_url}/chat",
                json={"message": message, "session_id": session_id},
                timeout=60  # Increased timeout to 60 seconds
            )
            return response.json() if response.status_code == 200 else {"status": "error", "response": "API Error"}
        except requests.exceptions.Timeout:
            return {"status": "error", "response": "Request timed out. The server might be processing a complex query. Please try again."}
        except requests.exceptions.ConnectionError:
            return {"status": "error", "response": f"Cannot connect to the server. Please check if the backend is running on {self.base_url}"}
        except Exception as e:
            return {"status": "error", "response": f"Connection error: {str(e)}"}
    
    def get_documents(self):
        """Get list of available documents"""
        try:
            response = requests.get(f"{self.base_url}/documents", timeout=10)
            return response.json() if response.status_code == 200 else {"documents": []}
        except:
            return {"documents": []}

def initialize_session():
    """Initialize session state with proactive greeting"""
    if 'messages' not in st.session_state:
        st.session_state.messages = []
        # Add proactive welcome message
        welcome_message = get_proactive_welcome()
        st.session_state.messages.append({'role': 'assistant', 'content': welcome_message})
    if 'session_id' not in st.session_state:
        st.session_state.session_id = str(uuid.uuid4())

def get_proactive_welcome():
    """Generate proactive welcome message based on context"""
    import datetime
    
    current_hour = datetime.datetime.now().hour
    current_month = datetime.datetime.now().month
    
    # Generic greeting options
    greeting_options = [
        "Hello",
        "Hi there",
        "Welcome",
        "Greetings"
    ]
    
    import random
    time_greeting = random.choice(greeting_options)
    
    # Check if this is a returning user (simple check based on session state)
    is_returning = len(st.session_state.get('messages', [])) > 1
    
    # Get available documents and analyze what we can actually help with
    try:
        api_client = FastAPIClient()
        documents = api_client.get_documents()
        available_docs = documents.get("documents", [])
        
        # Analyze available documents to determine what topics we can actually help with
        available_topics = set()
        topic_examples = {}
        
        for doc in available_docs:
            title_lower = doc.get('title', '').lower()
            title_display = doc.get('title', '')
            
            # Only add topics if we have VERY strong evidence in document titles
            # Be extremely conservative to avoid suggesting topics we can't actually help with
            
            # Salary - only if explicitly mentioned with context
            if any(phrase in title_lower for phrase in ['salary structure', 'salary scale', 'pay structure', 'compensation structure', 'allowance']):
                available_topics.add('salary')
                if 'salary' not in topic_examples:
                    topic_examples['salary'] = title_display
            
            # Leave - only if explicitly mentioned with policy context
            if any(phrase in title_lower for phrase in ['leave policy', 'leave allowance', 'vacation policy', 'holiday policy']):
                available_topics.add('leave')
                if 'leave' not in topic_examples:
                    topic_examples['leave'] = title_display
            
            # Benefits - VERY strict - only if explicitly mentioned with employee context
            if any(phrase in title_lower for phrase in ['employee benefit', 'medical benefit', 'insurance policy', 'health benefit']) and \
               any(word in title_lower for word in ['employee', 'staff', 'company']):
                available_topics.add('benefits')
                if 'benefits' not in topic_examples:
                    topic_examples['benefits'] = title_display
            
            # Policies - only if explicitly mentioned
            if any(phrase in title_lower for phrase in ['hr policy', 'company policy', 'employee policy', 'policy manual', 'handbook']):
                available_topics.add('policies')
                if 'policies' not in topic_examples:
                    topic_examples['policies'] = title_display
            
            # Positions - only if explicitly mentioned with organizational context
            if any(phrase in title_lower for phrase in ['organizational structure', 'job position', 'management structure', 'hierarchy', 'designation']):
                available_topics.add('positions')
                if 'positions' not in topic_examples:
                    topic_examples['positions'] = title_display
            
            # Performance - only if explicitly mentioned
            if any(phrase in title_lower for phrase in ['performance review', 'performance appraisal', 'training program', 'development program']):
                available_topics.add('performance')
                if 'performance' not in topic_examples:
                    topic_examples['performance'] = title_display
        
        print(f"Available topics from documents: {available_topics}")
        print(f"Topic examples: {topic_examples}")
        
    except Exception as e:
        print(f"Error analyzing documents: {e}")
        # Fallback to no topics if API call fails
        available_topics = set()
        topic_examples = {}
    
    # Seasonal context
    seasonal_info = ""
    if current_month in [11, 12]:
        seasonal_info = " As we approach year-end, you might be interested in bonus calculations, leave settlements, or performance reviews."
    elif current_month in [6, 7]:
        seasonal_info = " Mid-year is a great time to review your benefits, plan leave, or explore training opportunities."
    elif current_month in [1, 2]:
        seasonal_info = " New year is perfect for understanding your salary structure, setting goals, or reviewing company policies."
    
    # Generate structured suggestions based on actually available topics with document evidence
    suggestions = []
    
    if available_topics:
        # Topics section
        suggestions.append("## Topics I can help with:")
        suggestions.append("*Based on available documents in our database*")
        suggestions.append("")
        
        # Only show topics we have strong evidence for
        if 'salary' in available_topics:
            suggestions.append(f"**• Salary & Compensation**")
            suggestions.append(f"  - Salary structures and allowances")
            suggestions.append("")
        
        if 'leave' in available_topics:
            suggestions.append(f"**• Leave Management**")
            suggestions.append(f"  - Leave policies and applications")
            suggestions.append("")
        
        if 'benefits' in available_topics:
            suggestions.append(f"**• Employee Benefits**")
            suggestions.append(f"  - Benefits and insurance information")
            suggestions.append("")
        
        if 'policies' in available_topics:
            suggestions.append(f"**• HR Policies**")
            suggestions.append(f"  - Company policies and procedures")
            suggestions.append("")
        
        if 'positions' in available_topics:
            suggestions.append(f"**• Organizational Structure**")
            suggestions.append(f"  - Job positions and hierarchy")
            suggestions.append("")
        
        if 'performance' in available_topics:
            suggestions.append(f"**• Performance Management**")
            suggestions.append(f"  - Performance reviews and training")
            suggestions.append("")
        
        # Examples section
        suggestions.append("## Quick Examples:")
        suggestions.append("*Try asking these questions:*")
        suggestions.append("")
        
        # Add examples only for available topics
        example_count = 1
        if 'salary' in available_topics:
            suggestions.append(f"{example_count}. \"Show me the salary structure\"")
            example_count += 1
        if 'leave' in available_topics:
            suggestions.append(f"{example_count}. \"What is my leave entitlement?\"")
            example_count += 1
        if 'benefits' in available_topics:
            suggestions.append(f"{example_count}. \"What benefits are available?\"")
            example_count += 1
        if 'policies' in available_topics:
            suggestions.append(f"{example_count}. \"What are the HR policies?\"")
            example_count += 1
        if 'positions' in available_topics:
            suggestions.append(f"{example_count}. \"Show me the organizational structure\"")
            example_count += 1
        if 'performance' in available_topics:
            suggestions.append(f"{example_count}. \"How does performance evaluation work?\"")
            example_count += 1
        
    else:
        suggestions.append("## I'm ready to help with your HR questions!")
    
    # Customize welcome based on returning user status
    if is_returning:
        welcome = f"""# {time_greeting}! Welcome back to Kazi Farms Chatbot! 👋

I remember our previous conversations and can build on what we've discussed.{seasonal_info}

{chr(10).join(suggestions)}

**What would you like to continue exploring or learn about today?**
"""
    else:
        welcome = f"""# {time_greeting}! Welcome to Kazi Farms Chatbot! 👋

I'm here to help you with all your HR and company-related questions. I'll remember our conversation to provide better, contextual responses.{seasonal_info}

{chr(10).join(suggestions)}

**What would you like to explore today?**
"""
    
    return welcome.strip()





def display_messages():
    """Display chat messages with context awareness"""
    for i, message in enumerate(st.session_state.messages):
        with st.chat_message(message['role']):
            # Display the main message content
            content = message['content']
            
            # Add context indicator for assistant messages (except first welcome)
            if message['role'] == 'assistant' and i > 0 and i > 1:
                # Add context indicator for recent chat connections
                if "Context connections:" in content or "This relates to" in content or "from" in content and "chat" in content:
                    st.caption("Using context from recent chats")
                elif "This connects to" in content and "ago" in content:
                    st.caption("Connected to previous conversation")
                elif len(st.session_state.messages) > 3:  # Show for conversations with 3+ messages
                    st.caption("Context-aware response")
            
            # Display content - backend should handle all citation logic
            st.markdown(content, unsafe_allow_html=True)

def handle_user_input(user_input):
    """Handle user input with proactive enhancements"""
    if not user_input:
        return
    
    # Add user message to session
    st.session_state.messages.append({'role': 'user', 'content': user_input})
    
    # Show proactive loading message
    loading_messages = [
        "Searching through company documents...",
        "Analyzing your query and finding related information...",
        "Preparing comprehensive response with suggestions...",
        "Generating insights for you..."
    ]
    
    import random
    loading_msg = random.choice(loading_messages)
    
    # Show loading message
    with st.chat_message('assistant'):
        with st.spinner(loading_msg):
            # Get API response
            try:
                api_client = FastAPIClient()
                session_id = st.session_state.get('session_id', '')
                response = api_client.send_message(user_input, session_id)
                
                if response.get("status") == "success":
                    bot_response = response.get("response", "Sorry, I couldn't process your request.")
                    # Backend includes citations and proactive elements in the response content
                else:
                    bot_response = f"Error: {response.get('response', 'Unknown error')}"
                
                # Store the assistant message
                st.session_state.messages.append({'role': 'assistant', 'content': bot_response})
                
            except Exception as e:
                error_msg = f"I encountered an issue processing your request: {str(e)}\n\n**Try asking:**\n• A more specific question\n• About salary, leave, or allowances\n• For step-by-step procedures"
                st.session_state.messages.append({'role': 'assistant', 'content': error_msg})
    
    # Refresh the display
    st.rerun()

def render_header():
    """Render the app header"""
    st.title("Kazi Farms Assistant")
    # st.markdown("Your official Kazi Farms chatbot.")

def render_sidebar():
    """Render sidebar with API status and controls"""
    with st.sidebar:
        st.header("System Status")
        
        # API Status
        api_client = FastAPIClient()
        health = api_client.health_check()
        
        if health.get("status") == "healthy":
            st.success("API Online")
        elif health.get("status") == "timeout":
            st.warning("API Slow (Timeout)")
            st.caption("Server is responding slowly")
        elif health.get("status") == "connection_error":
            st.error("API Offline")
            st.caption(f"Cannot connect to {client.base_url}")
        else:
            st.error("API Error")
            st.caption(health.get("message", "Unknown error"))
        
        st.divider()
        
        # Chat controls
        st.header("Chat Controls")
        if st.button("Clear Chat"):
            st.session_state.messages = []
            st.rerun()
        
        st.divider()
        
        # Proactive suggestions
        st.header("Smart Suggestions")
        
        # Get available documents to make database-aware suggestions
        try:
            api_client = FastAPIClient()
            documents = api_client.get_documents()
            available_docs = documents.get("documents", [])
            
            # Create suggestions based on actual document titles
            database_samples = []
            
            # Extract actual topics from document titles
            doc_titles = [doc.get('title', '') for doc in available_docs]
            
            # Create suggestions based on actual document content
            for doc in available_docs[:8]:  # Use first 8 documents
                title = doc.get('title', '')
                if title:
                    # Clean up title for display
                    clean_title = title.replace('.pdf', '').replace('_', ' ').title()
                    if len(clean_title) > 30:
                        clean_title = clean_title[:27] + "..."
                    database_samples.append(clean_title)
            
            # If we have fewer than 4 suggestions, add some generic ones
            if len(database_samples) < 4:
                generic_suggestions = [
                    "Company policies",
                    "HR information", 
                    "Employee guidelines",
                    "General inquiries"
                ]
                database_samples.extend(generic_suggestions[:4-len(database_samples)])
            
            all_samples = database_samples[:6]  # Limit to 6 suggestions
            
        except Exception as e:
            print(f"Error getting document suggestions: {e}")
            # Fallback if API call fails
            all_samples = [
                "Company policies",
                "HR information", 
                "Employee guidelines",
                "General questions"
            ]
        
        for sample in all_samples:
            if st.button(f"{sample}"):
                handle_user_input(sample)
                st.rerun()
        
        st.divider()
        
        # Context awareness info
        if len(st.session_state.messages) > 2:
            st.header("Context Awareness")
            chat_count = len([msg for msg in st.session_state.messages if msg['role'] == 'user'])
            context_chats = min(chat_count, 5)
            
            st.caption(f"Remembering last {context_chats} chat{'s' if context_chats != 1 else ''}")
            st.caption("Building on previous topics")
            st.caption("Personalizing responses")
            
            st.divider()
        
        # Proactive tips
        st.header("Smart Features")
        tips = [
            "I remember our last 5 conversations",
            "I search through 400+ company documents",
            "I provide step-by-step processes",
            "Click citation numbers to view sources",
            "I connect topics from our chat history"
        ]
        
        for tip in tips:
            st.caption(tip)
        
        st.divider()
        
        # Available documents
        st.header("Available Documents")
        api_client = FastAPIClient()
        documents = api_client.get_documents()
        
        if documents.get("documents"):
            st.markdown(f"**{len(documents['documents'])} documents available**")
            with st.expander("View all documents"):
                for doc in documents["documents"][:10]:  # Show first 10
                    pdf_url = f"{client.base_url}{doc['path']}"
                    st.markdown(f'• <a href="{pdf_url}" target="_blank">{doc["title"]}</a>', unsafe_allow_html=True)
                if len(documents["documents"]) > 10:
                    st.markdown(f"... and {len(documents['documents']) - 10} more")
        else:
            st.markdown("No documents available")

def main():
    """Main application"""
    # Render header
    render_header()
    
    # Initialize session
    initialize_session()
    
    # Render sidebar
    render_sidebar()
    
    # Display chat history
    display_messages()
    
    # User input
    user_prompt = st.chat_input("Ask your question here...")
    handle_user_input(user_prompt)

if __name__ == "__main__":
    main()