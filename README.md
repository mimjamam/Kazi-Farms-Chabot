# Kazi Farms Chatbot

A structured chatbot application for Kazi Farms using Streamlit, LangChain, and Groq.

## Project Structure

```
kazi-farms-chatbot/
├── main.py                 # Main entry point
├── config/                 # Configuration and settings
│   ├── __init__.py
│   └── settings.py         # Application settings and environment variables
├── backend/                # Backend services and business logic
│   ├── __init__.py
│   ├── chat_service.py     # Chat and vector store services
│   ├── query_agent.py      # Query validation and analysis
│   ├── similarity_agent.py # Response similarity comparison
│   ├── funny_fallback_agent.py # Fallback responses
│   ├── personal_info_guard.py # Personal information protection
│   └── langgraph_workflow.py # LangGraph workflow orchestration
├── frontend/               # User interface
│   ├── __init__.py
│   └── streamlit_ui.py     # Streamlit frontend implementation
├── core/                   # Core functionality
│   ├── data/               # Data and vector store
│   ├── memory/             # Memory management
│   └── models/             # AI models
└── requirements.txt        # Dependencies
```

## Key Features

- **Clean Architecture**: Separated concerns with dedicated modules
- **Query Validation**: Smart query analysis and validation
- **Personal Information Protection**: Blocks personal identity queries
- **Funny Fallback Responses**: Engaging responses for irrelevant queries
- **Similarity Analysis**: Compares input and output for quality assurance
- **Terminal Logging**: Comprehensive backend logging for debugging

## Usage

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Set up environment variables:
   ```bash
   # Create a .env file in the project root:
   touch .env
   
   # Add your Groq API key to the .env file:
   echo "GROQ_API_KEY=your_actual_groq_api_key_here" >> .env
   ```
   
   **To get your Groq API key:**
   1. Visit: https://console.groq.com/keys
   2. Sign up or log in
   3. Create a new API key
   4. Copy the key and replace `your_actual_groq_api_key_here` in the .env file

3. Run the application:
   ```bash
   streamlit run main.py
   ```

## Architecture

- **main.py**: Simple entry point that initializes and runs the frontend
- **config/**: Manages all application settings and environment variables
- **backend/**: Contains core business logic with multiple specialized agents
- **frontend/**: Handles the Streamlit user interface
- **core/**: Core functionality for data, memory, and models

This structure makes the codebase more maintainable, testable, and easier to understand.
