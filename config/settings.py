"""
Configuration settings for Kazi Farms Chatbot
"""
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class Settings:
    """Application settings and configuration"""
    
    # API Keys
    GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
    
    # API Configuration
    FASTAPI_HOST = os.environ.get("FASTAPI_HOST", "0.0.0.0")
    FASTAPI_PORT = int(os.environ.get("FASTAPI_PORT", "8000"))
    FASTAPI_BASE_URL = os.environ.get("FASTAPI_BASE_URL", f"http://localhost:{os.environ.get('FASTAPI_PORT', '8000')}")
    
    # Paths
    DB_FAISS_PATH = "core/data/faiss_index"
    
    # Model Settings
    EMBEDDING_MODEL = 'sentence-transformers/all-MiniLM-L6-v2'
    LLM_MODEL = "llama-3.1-8b-instant"  # Faster model for better response time
    LLM_TEMPERATURE = 0.0
    
    # Search Settings
    TOP_K = 5
    CONFIDENCE_THRESHOLD = 25  # Accept answers >= 25% confidence
    
    # Memory Management Settings
    AUTO_CLEANUP_ENABLED = True  # Enable automatic memory cleanup on tab close
    SESSION_TIMEOUT_MINUTES = 30  # Session timeout in minutes for auto-cleanup
    
    # Debug Settings
    DEBUG_MODE = os.environ.get("DEBUG_MODE", "false").lower() == "true"
    
    # Prompt Template
    CUSTOM_PROMPT_TEMPLATE = """
    You are Kazi Farms AI Assistant. Provide clear, direct answers about HR information and policies.

    CRITICAL RULES:
    1. Start IMMEDIATELY with the answer - NO greetings like "Hi there!" or "Here's a quick overview"
    2. Use natural transitions ONLY when connecting related information (e.g., "For overtime," "Regarding allowances," "In addition to basic pay,")
    3. NO redundant closing phrases - end with the last piece of information
    4. Keep responses focused and under 150 words
    5. Use bullet points (•) for lists when presenting multiple items
    6. DO NOT use any markdown formatting like ** or __ for bold text
    7. Write in plain text only - no special formatting characters

    NATURAL FLOW:
    - Connect related points smoothly using context-appropriate transitions
    - Use phrases like "For [topic]," "Regarding [item]," "In addition to [X]," when introducing new but related information
    - Avoid repetitive transitions - vary your connectors naturally
    - Let the information flow logically without forced structure

    FORMATTING:
    - Plain text only - no bold, italic, or other markdown
    - Short, clear sentences
    - Bullet points for lists of 3+ items
    - Use CAPITAL LETTERS for emphasis if needed (sparingly)

    AVOID:
    ❌ "Hi there! Here's a quick overview..."
    ❌ Starting every paragraph with "Additionally,"
    ❌ "Let me know if you need anything else." (repetitive closings)
    ❌ "Hope this helps!" (unnecessary endings)
    ❌ Using ** or __ for formatting

    GOOD EXAMPLE:
    "For Job Groups 1-3, the salary structure includes:
    • Basic salary according to job-group scale
    • Dearness Allowance (DA) added on top of basic pay
    • Replacement allowance: Full for 6+ hrs work, half for 4-6 hrs
    
    For overtime, the rate is 1.5× regular hourly rate. Other allowances like transport and food are provided per company guidelines."

    Context: {context}
    Question: {question}

    Answer:"""
    
    @classmethod
    def validate_config(cls):
        """Validate that required configuration is present"""
        if not cls.GROQ_API_KEY:
            raise ValueError("GROQ_API_KEY not found. Please add it to your .env file or environment variables.")
        return True
