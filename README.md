# 🐔 Kazi Farms Chatbot

An intelligent AI-powered chatbot system designed for Kazi Farms to provide automated assistance for HR policies, company information, and employee queries using advanced LangGraph workflows and RAG (Retrieval-Augmented Generation) architecture.

[![Python Version](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://www.python.org/downloads/)
[![LangChain](https://img.shields.io/badge/LangChain-Enabled-green.svg)](https://langchain.com/)
[![LangGraph](https://img.shields.io/badge/LangGraph-Workflow-orange.svg)](https://langchain-ai.github.io/langgraph/)
[![Streamlit](https://img.shields.io/badge/Streamlit-UI-red.svg)](https://streamlit.io/)

---

## 📑 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Architecture](#architecture)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Configuration](#configuration)
- [Usage](#usage)
- [Components](#components)
- [API Documentation](#api-documentation)
- [Development](#development)
- [Troubleshooting](#troubleshooting)
- [Contributing](#contributing)
- [License](#license)

---

## 🎯 Overview

The Kazi Farms Chatbot is a sophisticated conversational AI system that leverages:
- **LangGraph** for orchestrating complex multi-agent workflows
- **FAISS** for efficient vector storage and semantic search
- **Groq LLM** (Llama 3.1) for natural language understanding and generation
- **Streamlit** for an intuitive web interface
- **FastAPI** for scalable backend services

This chatbot is specifically designed to handle HR-related queries, company policies, and employee information with high accuracy and contextual awareness.

---

## ✨ Features

### Core Capabilities
- 🤖 **Multi-Agent Workflow**: Coordinated system of specialized agents for different tasks
- 🧠 **Intent Recognition**: Dynamic intent classification with learning capabilities
- 🔍 **Semantic Search**: Vector-based document retrieval using FAISS
- 💬 **Context-Aware Conversations**: Maintains conversation history and context
- ✅ **Answer Verification**: Grounding checker and relevance validation
- 🎯 **Domain Guard**: Ensures queries stay within the chatbot's knowledge domain
- 🔄 **Query Rewriting**: Improves query clarity and context
- 📊 **Confidence Scoring**: Provides reliability metrics for answers
- 🔐 **Session Management**: Automatic cleanup and timeout handling

### Technical Features
- ⚡ **Fast Response Time**: Optimized with Llama 3.1 8B Instant model
- 🔄 **Auto-Learning**: Adapts to new query patterns over time
- 📝 **Memory Management**: Persistent conversation storage
- 🛡️ **Error Handling**: Robust fallback mechanisms
- 📈 **Monitoring & Logging**: Comprehensive logging for debugging
- 🌐 **RESTful API**: FastAPI backend for easy integration

---

## 🏗️ Architecture

### System Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                         Frontend Layer                          │
│  ┌─────────────────┐              ┌─────────────────┐           │
│  │  Streamlit UI   │              │   HTML/CSS UI   │           │
│  └────────┬────────┘              └────────┬────────┘           │
└───────────┼──────────────────────────────┼──────────────────────┘
            │                              │
            ▼                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                         Backend Layer                           │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │           LangGraph Chat Service                         │   │
│  │    ┌────────────────────────────────────────┐            │   │
│  │    │    SimpleLangGraphWorkflow             │            │   │
│  │    │  ┌──────────────────────────────────┐  │            │   │
│  │    │  │    Multi-Agent Orchestration     │  │            │   │
│  │    │  └──────────────────────────────────┘  │            │   │
│  │    └────────────────────────────────────────┘            │   │
│  └──────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
            │
            ▼
┌─────────────────────────────────────────────────────────────────┐
│                      Agent Workflow Layer                       │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐           │
│  │   Intent     │→ │Query Rewrite │→ │Domain Guard  │           │
│  │   Agent      │  │   Agent      │  │   Agent      │           │
│  └──────────────┘  └──────────────┘  └──────────────┘           │
│           ↓                                                     │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐           │
│  │  Retrieval   │→ │   Context    │→ │  Reasoning   │           │
│  │   Agent      │  │   Selector   │  │   Agent      │           │
│  └──────────────┘  └──────────────┘  └──────────────┘           │
│           ↓                                                     │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐           │
│  │  Grounding   │→ │  Relevance   │→ │   Fallback   │           │
│  │   Checker    │  │   Agent      │  │   Agent      │           │
│  └──────────────┘  └──────────────┘  └──────────────┘           │
└─────────────────────────────────────────────────────────────────┘
            │
            ▼
┌─────────────────────────────────────────────────────────────────┐
│                      Data & Storage Layer                       │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐           │
│  │    FAISS     │  │  Conversation│  │   Context    │           │
│  │Vector Index  │  │    Memory    │  │   Memory     │           │
│  └──────────────┘  └──────────────┘  └──────────────┘           │
└─────────────────────────────────────────────────────────────────┘
```

### Agent Workflow

1. **Intent Agent**: Classifies user intent (greeting, factual, conversational)
2. **Query Rewrite Agent**: Enhances query clarity with conversation context
3. **Domain Guard Agent**: Validates if query is within domain scope
4. **Retrieval Agent**: Fetches relevant documents using FAISS semantic search
5. **Context Selector Agent**: Filters and selects most relevant context
6. **Reasoning Agent**: Generates answers using LLM with selected context
7. **Grounding Checker Agent**: Verifies answer is grounded in retrieved context
8. **Relevance Agent**: Validates answer relevance to the query
9. **Fallback Agent**: Provides helpful response when primary path fails

---

## 📁 Project Structure

```
Kazi-Farms-Chabot/
│
├── main.py                      # Main application entry point
├── requirements.txt             # Python dependencies
├── README.md                    # Project documentation (this file)
├── chatbot-ui.html             # Static HTML UI
├── test-connection.html        # Connection testing page
│
├── backend/                    # Backend services and agents
│   ├── __init__.py
│   ├── langgraph_chat_service.py    # Main chat service
│   └── agents/                 # Agent modules
│       ├── __init__.py
│       ├── simple_langgraph_workflow.py   # LangGraph workflow orchestration
│       ├── intent_agent.py                # Intent classification
│       ├── dynamic_intent_agent.py        # Advanced intent detection
│       ├── query_rewrite_agent.py         # Query enhancement
│       ├── domain_guard_agent.py          # Domain validation
│       ├── retrieval_agent.py             # Document retrieval
│       ├── context_selector_agent.py      # Context filtering
│       ├── reasoning_agent.py             # Answer generation
│       ├── grounding_checker_agent.py     # Answer verification
│       ├── relevance_agent.py             # Relevance scoring
│       ├── fallback_agent.py              # Fallback responses
│       ├── conversational_agent.py        # Conversation handling
│       ├── context_memory_agent.py        # Context management
│       ├── proactive_agent.py             # Proactive suggestions
│       └── threshold_calibrator.py        # Confidence calibration
│
├── config/                     # Configuration modules
│   ├── __init__.py
│   └── settings.py            # Application settings
│
├── core/                       # Core utilities and data
│   ├── __init__.py
│   ├── data/                  # Data storage
│   │   ├── __init__.py
│   │   ├── context_memory.json       # Context storage
│   │   ├── documents/                # Source documents
│   │   ├── faiss_index/             # Vector database
│   │   │   ├── index.faiss          # FAISS index file
│   │   │   └── conversation_memory.json
│   │   └── intent_patterns/         # Intent learning data
│   │       ├── stats.json
│   │       └── backups/
│   ├── memory/                # Memory management
│   │   ├── __init__.py
│   │   ├── memory_manager.py
│   │   └── chatbot_memory/
│   │       └── conversations.json
│   ├── models/               # ML models and utilities
│   │   ├── __init__.py
│   │   └── simple_query_matcher.py
│   └── utils/               # Utility functions
│       └── __init__.py
│
├── frontend/                # Streamlit frontend
│   ├── __init__.py
│   └── streamlit_ui.py     # Streamlit application
│
├── frontend-fastapi/       # Alternative FastAPI frontend
│   ├── app.py
│   ├── config.py
│   └── requirements.txt
│
└── fastapi-backend/        # FastAPI backend service
    ├── main.py
    └── __pycache__/
```

---

## 🚀 Installation

### Prerequisites

- Python 3.8 or higher
- pip (Python package manager)
- Git (for cloning the repository)
- Groq API key (for LLM access)

### Step-by-Step Installation

1. **Clone the Repository**
   ```bash
   git clone https://github.com/mimjamam/Kazi-Farms-Chabot.git
   cd Kazi-Farms-Chabot
   ```

2. **Create Virtual Environment** (Recommended)
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set Up Environment Variables**
   Create a `.env` file in the project root:
   ```bash
   touch .env
   ```
   
   Add the following configuration:
   ```env
   # API Keys
   GROQ_API_KEY=your_groq_api_key_here
   
   # API Configuration
   FASTAPI_HOST=0.0.0.0
   FASTAPI_PORT=8000
   FASTAPI_BASE_URL=http://localhost:8000
   
   # Debug Settings
   DEBUG_MODE=false
   ```

5. **Prepare Data Directory**
   Ensure the following directories exist:
   ```bash
   mkdir -p core/data/documents
   mkdir -p core/data/faiss_index
   mkdir -p core/data/intent_patterns/backups
   mkdir -p core/memory/chatbot_memory
   ```

6. **Add Source Documents** (Optional)
   Place your company documents (PDF, TXT, etc.) in:
   ```
   core/data/documents/
   ```

---

## ⚙️ Configuration

### Configuration File: `config/settings.py`

Key configuration parameters:

| Parameter | Description | Default Value |
|-----------|-------------|---------------|
| `GROQ_API_KEY` | Groq API key for LLM access | From .env |
| `FASTAPI_HOST` | FastAPI server host | 0.0.0.0 |
| `FASTAPI_PORT` | FastAPI server port | 8000 |
| `EMBEDDING_MODEL` | Sentence transformer model | all-MiniLM-L6-v2 |
| `LLM_MODEL` | Groq LLM model | llama-3.1-8b-instant |
| `LLM_TEMPERATURE` | LLM temperature (0-1) | 0.0 |
| `TOP_K` | Number of documents to retrieve | 5 |
| `CONFIDENCE_THRESHOLD` | Minimum confidence score (%) | 25 |
| `AUTO_CLEANUP_ENABLED` | Enable auto memory cleanup | True |
| `SESSION_TIMEOUT_MINUTES` | Session timeout duration | 30 |

### Customizing the Prompt

Edit the `CUSTOM_PROMPT_TEMPLATE` in `config/settings.py` to customize the chatbot's behavior and response style.

---

## 💻 Usage

### Running the Streamlit Application

```bash
python main.py
```

Or directly:
```bash
streamlit run frontend/streamlit_ui.py
```

The application will open in your default browser at `http://localhost:8501`

### Running the FastAPI Backend

```bash
cd fastapi-backend
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

API documentation will be available at:
- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

### Using the HTML Interface

Open `chatbot-ui.html` in a web browser for a static HTML interface.

### Command Line Usage

```python
from backend.langgraph_chat_service import LangGraphChatService

# Initialize chatbot
chatbot = LangGraphChatService()

# Process a query
response = chatbot.get_answer_with_sources(
    query="What is the leave policy?",
    conversation_context="",
    session_id="user-123"
)

print(response)
```

---

## 🧩 Components

### Backend Agents

#### 1. **Intent Agent** (`intent_agent.py`, `dynamic_intent_agent.py`)
- Classifies user queries into categories (greeting, factual, conversational)
- Learns from interaction patterns
- Supports dynamic intent detection

#### 2. **Query Rewrite Agent** (`query_rewrite_agent.py`)
- Enhances query clarity
- Adds conversation context
- Resolves pronouns and ambiguities

#### 3. **Domain Guard Agent** (`domain_guard_agent.py`)
- Validates if query is within domain
- Prevents off-topic responses
- Dynamic domain boundary learning

#### 4. **Retrieval Agent** (`retrieval_agent.py`)
- Performs semantic search using FAISS
- Returns top-k relevant documents
- Provides similarity scores

#### 5. **Context Selector Agent** (`context_selector_agent.py`)
- Filters retrieved documents
- Ranks by relevance
- Optimizes context for LLM

#### 6. **Reasoning Agent** (`reasoning_agent.py`)
- Generates answers using LLM
- Considers conversation history
- Produces confidence scores

#### 7. **Grounding Checker Agent** (`grounding_checker_agent.py`)
- Verifies answer is based on retrieved context
- Detects hallucinations
- Provides grounding scores

#### 8. **Relevance Agent** (`relevance_agent.py`)
- Validates answer relevance to query
- Ensures response quality
- Triggers regeneration if needed

#### 9. **Fallback Agent** (`fallback_agent.py`)
- Provides helpful responses when primary workflow fails
- Suggests alternative questions
- Maintains user engagement

### Frontend Components

#### **Streamlit UI** (`frontend/streamlit_ui.py`)
- Interactive chat interface
- Session management
- Chat history display
- Real-time responses

#### **FastAPI Backend** (`fastapi-backend/main.py`)
- RESTful API endpoints
- CORS support
- Request/response handling
- Scalable architecture

---

## 📚 API Documentation

### REST API Endpoints

#### **POST /chat**
Process a chat query and return a response.

**Request Body:**
```json
{
  "query": "What is the leave policy?",
  "conversation_context": "Previous conversation...",
  "session_id": "user-123"
}
```

**Response:**
```json
{
  "answer": "The leave policy includes...",
  "confidence": 0.85,
  "sources": ["document1.pdf", "document2.pdf"],
  "response_type": "factual",
  "processing_steps": ["intent_detection", "retrieval", "reasoning"]
}
```

#### **GET /health**
Check service health status.

**Response:**
```json
{
  "status": "healthy",
  "service": "Kazi Farms Chatbot",
  "version": "1.0.0"
}
```

---

## 🛠️ Development

### Setting Up Development Environment

1. **Install Development Dependencies**
   ```bash
   pip install -r requirements.txt
   pip install pytest black flake8 mypy
   ```

2. **Run Tests**
   ```bash
   pytest tests/
   ```

3. **Code Formatting**
   ```bash
   black .
   flake8 .
   ```

4. **Type Checking**
   ```bash
   mypy backend/
   ```

### Adding New Agents

1. Create a new agent file in `backend/agents/`
2. Implement the agent class with required methods
3. Add the agent to the workflow in `simple_langgraph_workflow.py`
4. Update the state schema if needed

### Debugging

Enable debug mode in `.env`:
```env
DEBUG_MODE=true
```

View logs in the terminal for detailed workflow execution traces.

---

## 🔧 Troubleshooting

### Common Issues

#### Issue: "GROQ_API_KEY not found"
**Solution:** Ensure your `.env` file contains a valid Groq API key.

#### Issue: "FAISS index not found"
**Solution:** Run the document indexing script to create the FAISS index:
```bash
python scripts/build_index.py  # (if available)
```

#### Issue: "Port 8501 already in use"
**Solution:** Kill the existing process or use a different port:
```bash
streamlit run frontend/streamlit_ui.py --server.port 8502
```

#### Issue: "Memory cleanup errors"
**Solution:** Check file permissions on `core/memory/chatbot_memory/conversations.json`

#### Issue: "Slow response times"
**Solution:** 
- Reduce `TOP_K` in settings
- Use a faster embedding model
- Check your internet connection (for Groq API)

### Getting Help

- Check the [Issues](https://github.com/mimjamam/Kazi-Farms-Chabot/issues) page
- Review the logs in the terminal
- Enable `DEBUG_MODE` for detailed traces

---

##  Contributing

Contributions are welcome! Please follow these steps:

1. **Fork the Repository**
2. **Create a Feature Branch**
   ```bash
   git checkout -b feature/your-feature-name
   ```
3. **Make Your Changes**
4. **Commit Your Changes**
   ```bash
   git commit -m "Add your feature description"
   ```
5. **Push to Your Fork**
   ```bash
   git push origin feature/your-feature-name
   ```
6. **Create a Pull Request**

### Coding Standards

- Follow PEP 8 style guide
- Write clear docstrings
- Add type hints where possible
- Include unit tests for new features
- Update documentation as needed

---

## 📄 License

This project is proprietary software developed for Kazi Farms. All rights reserved.

---

## 👥 Team

**Developer**: Mimjamam Ul Haque Monmoy (mimjamam)  
**Organization**: Kazi Farms  
**Repository**: [github.com/mimjamam/Kazi-Farms-Chabot](https://github.com/mimjamam/Kazi-Farms-Chabot)

---

## 🙏 Acknowledgments

- **LangChain** - Framework for LLM applications
- **LangGraph** - Multi-agent workflow orchestration
- **Groq** - Fast LLM inference
- **Streamlit** - Interactive web UI
- **FAISS** - Efficient vector search
- **Sentence Transformers** - Text embeddings

---

## 📮 Contact

For questions, issues, or feature requests, please:
- Open an issue on GitHub
- Contact the development team
- Review the documentation

---

## 🗓️ Version History

### Version 1.0.0 (Current)
- Initial release
- Multi-agent LangGraph workflow
- FAISS-based retrieval
- Streamlit and FastAPI interfaces
- Session management
- Auto-cleanup features

---

## 🔮 Future Enhancements

- [ ] Multi-language support
- [ ] Voice input/output
- [ ] Advanced analytics dashboard
- [ ] Document upload interface
- [ ] User feedback collection
- [ ] A/B testing framework
- [ ] Mobile application
- [ ] Integration with Slack/Teams
- [ ] Advanced caching mechanisms
- [ ] Real-time collaboration features

---


