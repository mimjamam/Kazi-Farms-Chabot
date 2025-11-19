from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel
from datetime import datetime
import sys
import os
import json
from typing import List, Dict, Any

# Add the parent directory to the path to import backend modules
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

# Change to the parent directory so relative paths work correctly
os.chdir(os.path.join(os.path.dirname(__file__), '..'))

# Import the existing chatbot service
from backend.langgraph_chat_service import LangGraphChatService
from backend.agents.proactive_agent import ProactiveAgent
from backend.agents.context_memory_agent import ContextMemoryAgent
from config.settings import Settings
import re

# Initialize settings
settings = Settings()

# Create FastAPI instance
app = FastAPI(
    title="Kazi Farms Chatbot API",
    description="A FastAPI backend for the Kazi Farms chatbot application",
    version="1.0.0"
)

# Add CORS middleware to allow frontend access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins (for development)
    allow_credentials=True,
    allow_methods=["*"],  # Allow all methods (GET, POST, OPTIONS, etc.)
    allow_headers=["*"],  # Allow all headers
)

# Lazy initialization - services will be initialized on first request
chatbot_service = None
proactive_agent = None
context_memory = None

def get_services():
    """Lazy initialization of services to avoid duplicate loading"""
    global chatbot_service, proactive_agent, context_memory
    
    if chatbot_service is None:
        print("🔧 Initializing chatbot services...")
        chatbot_service = LangGraphChatService()
        proactive_agent = ProactiveAgent()
        context_memory = ContextMemoryAgent()
        print("✅ Ready to chat!\n")
    
    return chatbot_service, proactive_agent, context_memory

def calculate_semantic_similarity(text1: str, text2: str) -> float:
    """Enhanced semantic similarity with weighted word importance"""
    if not text1 or not text2:
        return 0.0
    
    # Normalize texts
    text1_lower = text1.lower()
    text2_lower = text2.lower()
    
    # Enhanced stop words list
    stop_words = {
        'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 
        'is', 'are', 'was', 'were', 'be', 'been', 'have', 'has', 'had', 'do', 'does', 'did', 
        'will', 'would', 'could', 'should', 'may', 'might', 'can', 'this', 'that', 'these', 
        'those', 'it', 'its', 'they', 'them', 'their', 'we', 'us', 'our', 'you', 'your', 
        'he', 'him', 'his', 'she', 'her', 'i', 'me', 'my', 'from', 'up', 'about', 'into', 
        'over', 'after', 'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other', 
        'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 
        'very', 'just', 'now', 'here', 'there', 'when', 'where', 'why', 'how'
    }
    
    # Extract words with importance weighting
    def extract_weighted_words(text):
        words = {}
        tokens = text.split()
        
        for i, word in enumerate(tokens):
            clean_word = word.strip('.,!?()[]{}":;').lower()
            if len(clean_word) > 2 and clean_word not in stop_words:
                # Weight important words higher
                weight = 1.0
                
                # Higher weight for domain-specific terms
                if clean_word in ['salary', 'leave', 'policy', 'manager', 'employee', 'benefit', 'allowance', 'compensation']:
                    weight = 3.0
                # Higher weight for numbers and amounts
                elif re.match(r'\d+', clean_word):
                    weight = 2.5
                # Higher weight for job titles and positions
                elif clean_word in ['gm', 'agm', 'officer', 'executive', 'supervisor', 'director']:
                    weight = 2.0
                # Higher weight for longer, more specific words
                elif len(clean_word) > 6:
                    weight = 1.5
                
                words[clean_word] = weight
        
        return words
    
    words1 = extract_weighted_words(text1_lower)
    words2 = extract_weighted_words(text2_lower)
    
    if not words1 or not words2:
        return 0.0
    
    # Calculate weighted similarity
    intersection_weight = sum(min(words1.get(word, 0), words2.get(word, 0)) 
                             for word in set(words1.keys()).intersection(set(words2.keys())))
    union_weight = sum(words1.values()) + sum(words2.values()) - intersection_weight
    
    return intersection_weight / union_weight if union_weight > 0 else 0.0

def extract_key_phrases(text: str) -> List[str]:
    """Extract key phrases and entities from text with enhanced patterns"""
    phrases = []
    text_lower = text.lower()
    
    # Enhanced phrase patterns with more specific matching
    phrase_patterns = [
        # Salary and compensation patterns
        r'\b(?:salary|pay|compensation|wage|income|remuneration)\s+(?:structure|scale|range|amount|information|details|breakdown|grade|level)\b',
        r'\b(?:basic|gross|net)\s+(?:salary|pay|wage)\b',
        r'\b(?:allowance|benefit|bonus|increment)\s+(?:amount|rate|percentage|structure)\b',
        
        # Leave and time-off patterns
        r'\b(?:leave|vacation|holiday|absence)\s+(?:policy|days|allowance|entitlement|application|approval)\b',
        r'\b(?:annual|sick|casual|maternity|paternity|emergency)\s+(?:leave|days)\b',
        r'\b(?:\d+)\s+(?:days?)\s+(?:leave|vacation|holiday)\b',
        
        # Job roles and hierarchy patterns
        r'\b(?:general|assistant)\s+(?:manager|gm|agm)\b',
        r'\b(?:manager|officer|executive|supervisor|director|coordinator)\s+(?:position|role|level|grade)\b',
        r'\b(?:job|position|role)\s+(?:description|title|level|grade|hierarchy|structure)\b',
        
        # HR and policy patterns
        r'\b(?:hr|human\s+resource)\s+(?:policy|manual|handbook|guideline|procedure)\b',
        r'\b(?:employee|staff|personnel)\s+(?:handbook|manual|guide|policy|benefit|welfare)\b',
        r'\b(?:company|organizational)\s+(?:policy|structure|hierarchy|chart)\b',
        
        # Benefits and insurance patterns
        r'\b(?:medical|health|life|group)\s+(?:insurance|benefit|coverage|scheme|plan)\b',
        r'\b(?:provident|pension)\s+(?:fund|scheme|contribution)\b',
        r'\b(?:gratuity|bonus|festival)\s+(?:payment|amount|calculation)\b',
        
        # Performance and evaluation patterns
        r'\b(?:performance|annual|quarterly)\s+(?:review|appraisal|evaluation|assessment)\b',
        r'\b(?:promotion|increment|raise)\s+(?:criteria|policy|process|eligibility)\b',
        
        # Working conditions patterns
        r'\b(?:working|office|business)\s+(?:hours|time|schedule|days)\b',
        r'\b(?:overtime|extra)\s+(?:work|hours|payment|rate)\b',
        
        # Specific amounts and numbers
        r'\b\d{1,3}(?:,\d{3})*\s+(?:taka|bdt|tk|days?|hours?|percent|%)\b',
        r'\b(?:\d+)\s*(?:%|percent|percentage)\s+(?:of|increase|decrease|allowance)\b'
    ]
    
    for pattern in phrase_patterns:
        matches = re.findall(pattern, text_lower, re.IGNORECASE)
        phrases.extend(matches)
    
    # Extract specific entities
    entities = []
    
    # Extract monetary amounts
    money_patterns = [
        r'\b\d{1,3}(?:,\d{3})*\s*(?:taka|bdt|tk)\b',
        r'\b(?:tk|bdt)\s*\d{1,3}(?:,\d{3})*\b'
    ]
    
    for pattern in money_patterns:
        matches = re.findall(pattern, text_lower)
        entities.extend(matches)
    
    # Extract percentages
    percentage_patterns = [
        r'\b\d+(?:\.\d+)?\s*(?:%|percent|percentage)\b'
    ]
    
    for pattern in percentage_patterns:
        matches = re.findall(pattern, text_lower)
        entities.extend(matches)
    
    # Extract time periods
    time_patterns = [
        r'\b\d+\s+(?:days?|months?|years?|hours?)\b',
        r'\b(?:daily|weekly|monthly|yearly|annually)\b'
    ]
    
    for pattern in time_patterns:
        matches = re.findall(pattern, text_lower)
        entities.extend(matches)
    
    return phrases + entities

def format_references_section(sources: List[Dict[str, Any]], query: str = "", max_references: int = 5) -> str:
    """Format sources as a references section in citation format, showing only most relevant"""
    if not sources:
        return ""
    
    # Remove duplicates based on title
    seen_titles = set()
    unique_sources = []
    for source in sources:
        title = source.get('title', '')
        if title and title not in seen_titles:
            seen_titles.add(title)
            unique_sources.append(source)
    
    # Calculate query-specific relevance if query is provided
    if query:
        query_lower = query.lower()
        query_words = set(query_lower.split())
        
        for source in unique_sources:
            title_lower = source.get('title', '').lower()
            title_words = set(title_lower.split())
            
            # Calculate word overlap between query and title
            overlap = len(query_words.intersection(title_words))
            
            # Boost relevance score based on title match
            base_score = source.get('relevance_score', 0)
            title_boost = overlap * 0.15  # Each matching word adds 0.15
            source['final_relevance'] = base_score + title_boost
    else:
        # Use base relevance score
        for source in unique_sources:
            source['final_relevance'] = source.get('relevance_score', 0)
    
    # Sort by final relevance score (higher is better)
    unique_sources.sort(key=lambda x: x.get('final_relevance', 0), reverse=True)
    
    # Take only top N most relevant sources
    top_sources = unique_sources[:max_references]
    
    references = "\n\n<div class='references-section'><strong>📚 References:</strong><br>"
    
    for i, source in enumerate(top_sources, 1):
        title = source.get('title', f'Document {i}')
        # Use localhost for local development
        pdf_url = f"http://localhost:8000{source['path']}"
        
        # Format: [1] Document Title. Kazi Farms Group. [View PDF]
        references += f"<br>[{i}] {title}. Kazi Farms Group. "
        references += f'<a href="{pdf_url}" target="_blank" class="reference-link">[View PDF]</a>'
    
    references += "</div>"
    return references

def add_contextual_citations(content: str, sources: List[Dict[str, Any]]) -> str:
    """Enhanced contextual citation system with improved relevance matching"""
    if not sources:
        return content
    
    # Sort sources by relevance score and take top 5
    sorted_sources = sorted(sources, key=lambda x: x.get('relevance_score', 0), reverse=True)
    high_relevance_sources = sorted_sources[:5]  # Only use top 5 most relevant sources
    
    # Smart content segmentation
    segments = []
    
    # Try different segmentation strategies
    if '\n\n' in content:
        segments = [s.strip() for s in content.split('\n\n') if s.strip()]
    elif '. ' in content and len(content.split('. ')) > 2:
        raw_sentences = [s.strip() for s in content.split('. ') if s.strip()]
        segments = []
        current_segment = ""
        for sentence in raw_sentences:
            if len(current_segment) < 100:
                current_segment += sentence + ". "
            else:
                if current_segment:
                    segments.append(current_segment.strip())
                current_segment = sentence + ". "
        if current_segment:
            segments.append(current_segment.strip())
    elif '|' in content:
        segments = [s.strip() for s in content.split('|') if s.strip()]
    else:
        segments = [content]
    
    # Enhanced relevance scoring for each segment
    cited_segments = []
    sources_used = set()
    
    for segment_idx, segment in enumerate(segments, 1):
        best_sources = []
        segment_lower = segment.lower()
        
        # Calculate relevance scores for all unused sources
        for i, source in enumerate(high_relevance_sources):
            if i in sources_used:
                continue
            
            title = source.get('title', '')
            content_preview = source.get('content_preview', '')
            base_relevance = source.get('relevance_score', 0)
            
            # Simple relevance scoring
            score = base_relevance * 10
            
            # Title similarity
            title_similarity = calculate_semantic_similarity(segment, title)
            score += title_similarity * 20
            
            # Content preview similarity
            if content_preview:
                content_similarity = calculate_semantic_similarity(segment, content_preview)
                score += content_similarity * 15
            
            # Minimum threshold
            min_threshold = 5
            
            if score > min_threshold:
                best_sources.append((i, source, score))
        
        # Sort by score and select best match(es)
        best_sources.sort(key=lambda x: x[2], reverse=True)
        
        # Add citation(s) to segment
        segment_with_citations = segment
        citations_added = 0
        
        for source_idx, source, score in best_sources[:2]:  # Max 2 citations per segment
            if source_idx not in sources_used and citations_added < 2:
                sources_used.add(source_idx)
                pdf_url = f"{settings.FASTAPI_BASE_URL}{source['path']}"
                citation_link = f' <a href="{pdf_url}" target="_blank" class="inline-citation">[{source_idx + 1}]</a>'
                segment_with_citations += citation_link
                citations_added += 1
        
        cited_segments.append(segment_with_citations)
    
    # Fallback: If no citations were added, add at least one citation to the first segment
    if not any('[' in segment for segment in cited_segments) and high_relevance_sources:
        best_source = high_relevance_sources[0]
        pdf_url = f"{settings.FASTAPI_BASE_URL}{best_source['path']}"
        citation_link = f' <a href="{pdf_url}" target="_blank" class="inline-citation">[1]</a>'
        if cited_segments:
            cited_segments[0] += citation_link
    
    # Reconstruct content with proper formatting
    if '\n\n' in content:
        result = '\n\n'.join(cited_segments)
    elif '. ' in content:
        result = '. '.join(cited_segments)
        if content.rstrip().endswith('.') and not result.rstrip().endswith('.'):
            result = result.rstrip() + '.'
    elif '|' in content:
        result = ' | '.join(cited_segments)
    else:
        result = ' '.join(cited_segments)
    
    return result


# Pydantic models for request/response
class ChatRequest(BaseModel):
    message: str

class SourceDocument(BaseModel):
    id: str
    title: str
    path: str
    content_preview: str
    relevance_score: float = 0.0

class ChatResponse(BaseModel):
    response: str
    status: str
    timestamp: str
    session_id: str = ""
    sources: List[Dict[str, Any]] = []
    is_relevant: bool = True

# Root endpoint
@app.get("/")
async def root():
    return {
        "message": "Welcome to Kazi Farms Chatbot API",
        "status": "running",
        "version": "1.0.0"
    }

# Health check endpoint
@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "message": "API is running successfully"
    }

# Main chat endpoint
@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    """
    Main chat endpoint that processes user messages and returns responses using the Kazi Farms chatbot.
    """
    try:
        # Get services (lazy initialization)
        chatbot_service, proactive_agent, context_memory = get_services()
        
        # Validate input
        if not request.message or request.message.strip() == "":
            raise HTTPException(status_code=400, detail="Message cannot be empty")
        
        user_message = request.message.strip()
        
        # Generate session_id automatically using timestamp and hash
        import hashlib
        import time
        timestamp = str(time.time())
        session_id = hashlib.md5(f"{timestamp}_{user_message[:20]}".encode()).hexdigest()[:16]
        
        # Clean logging - Start of request
        print(f"\n💬 Query: {user_message}")
        
        # Get conversation context
        conversation_context = context_memory.get_conversation_context(session_id, user_message)
        context_enhancement = context_memory.get_contextual_prompt_enhancement(session_id, user_message)
        
        # Get response from the chatbot service
        enhanced_query = user_message
        continuity = conversation_context.get('topic_continuity', {})
        if continuity.get('is_affirmative_response') and continuity.get('original_query'):
            enhanced_query = f"Please provide detailed information about: {continuity['original_query']}"
        elif context_enhancement:
            enhanced_query = f"{context_enhancement}\n\nUSER QUERY: {user_message}"
        
        response_data = chatbot_service.process_query(
            query=enhanced_query,
            conversation_context=context_enhancement,
            session_id=session_id
        )
        
        # Log only if there's an error
        if response_data.get("error"):
            print(f"❌ Error: {response_data['error']}")
        
        # Extract sources and create detailed source documents (only if relevant)
        sources = []
        is_relevant = response_data.get("is_relevant", True)
        
        if is_relevant and response_data.get("sources"):
            for i, source in enumerate(response_data["sources"]):
                if isinstance(source, str):
                    # Handle string sources (legacy format)
                    doc_path = os.path.join("core/data/documents", source)
                    if os.path.exists(doc_path):
                        sources.append({
                            "id": f"doc_{i}",
                            "title": source.replace('.pdf', '').replace('_', ' ').title(),
                            "path": f"/pdf/{source}",
                            "content_preview": f"Document: {source}",
                            "relevance_score": 0.8  # Default for legacy format
                        })
                elif isinstance(source, dict):
                    # Handle new format with source and score
                    if "source" in source and "score" in source:
                        doc_name = source["source"]
                        doc_path = os.path.join("core/data/documents", doc_name)
                        if os.path.exists(doc_path):
                            # Convert distance score to relevance score (0-1 range, higher is better)
                            # FAISS returns L2 distance where lower is better
                            # We normalize it: relevance = 1 / (1 + distance)
                            distance = float(source["score"])
                            relevance_score = 1.0 / (1.0 + distance)
                            
                            sources.append({
                                "id": f"doc_{i}",
                                "title": doc_name.replace('.pdf', '').replace('_', ' ').title(),
                                "path": f"/pdf/{doc_name}",
                                "content_preview": f"Document: {doc_name}",
                                "relevance_score": relevance_score
                            })
                    else:
                        # Handle old dictionary format
                        doc_id = source.get("id", f"doc_{i}")
                        doc_path = source.get("path", "")
                        if doc_path and os.path.exists(doc_path):
                            sources.append({
                                "id": doc_id,
                                "title": source.get("title", doc_id),
                                "path": f"/pdf/{os.path.basename(doc_path)}",
                                "content_preview": source.get("content", "Document content"),
                                "relevance_score": source.get("score", 0.8)
                            })
        
        # Add contextual citations to the response content
        response_text = response_data.get("answer", "No response available")
        if sources and is_relevant:
            response_text = add_contextual_citations(response_text, sources)
            # Add references section at the end with query-based relevance
            references_section = format_references_section(sources, query=user_message, max_references=5)
            response_text += references_section
        
        # Enhance response with proactive elements
        enhanced_response = response_text
        if sources and is_relevant and "I can't help with that topic" not in response_text:
            proactive_context = conversation_context.get('recent_context', {})
            proactive_response = proactive_agent.enhance_response(
                query=user_message,
                response=response_text,
                session_id=session_id,
                sources=sources
            )
        else:
            from backend.agents.proactive_agent import ProactiveResponse
            proactive_response = ProactiveResponse(
                original_response=response_text,
                proactive_insights=[],
                suggested_questions=[],
                related_topics=[],
                conversation_guidance="",
                anticipatory_info=""
            )
        
        # Update context memory
        topics_discussed = []
        query_lower = user_message.lower()
        response_lower = response_text.lower()
        
        topic_keywords = {
            'salary': ['salary', 'pay', 'compensation', 'wage', 'allowance'],
            'leave': ['leave', 'vacation', 'holiday', 'sick', 'absence'],
            'benefits': ['benefit', 'insurance', 'medical', 'health', 'coverage'],
            'policies': ['policy', 'rule', 'procedure', 'guideline'],
            'positions': ['manager', 'officer', 'position', 'role', 'job'],
            'performance': ['performance', 'review', 'appraisal', 'evaluation']
        }
        
        for topic, keywords in topic_keywords.items():
            if any(keyword in query_lower or keyword in response_lower for keyword in keywords):
                topics_discussed.append(topic)
        
        context_insights = context_memory.update_context(
            session_id=session_id,
            user_query=user_message,
            bot_response=response_text,
            topics=topics_discussed,
            intent_type=response_data.get('intent_type', 'general'),
            sources=sources,
            confidence=response_data.get('confidence', 0.0)
        )
        
        # Show simple success indicator
        print(f"✅ Response sent ({len(sources)} sources)\n")
        
        response_text = enhanced_response
        
        # Return response
        return ChatResponse(
            response=response_text,
            status="success",
            timestamp=datetime.now().isoformat(),
            session_id=session_id,
            sources=sources,
            is_relevant=is_relevant
        )
        
    except Exception as e:
        print(f"❌ Error: {str(e)}\n")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

# PDF serving endpoint
@app.get("/pdf/{filename}")
async def serve_pdf(filename: str):
    """
    Serve PDF files from the documents directory
    """
    try:
        # Security check - only allow PDF files
        if not filename.endswith('.pdf'):
            raise HTTPException(status_code=400, detail="Only PDF files are allowed")
        
        # Construct file path
        file_path = os.path.join("core/data/documents", filename)
        
        # Check if file exists
        if not os.path.exists(file_path):
            raise HTTPException(status_code=404, detail="PDF file not found")
        
        # Return the PDF file
        return FileResponse(
            path=file_path,
            media_type="application/pdf",
            filename=filename,
            headers={"Content-Disposition": f"inline; filename={filename}"}
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error serving PDF: {str(e)}")

# Get list of available documents
@app.get("/documents")
async def get_documents():
    """
    Get list of available documents
    """
    try:
        documents_dir = "core/data/documents"
        if not os.path.exists(documents_dir):
            return {"documents": []}
        
        documents = []
        for filename in os.listdir(documents_dir):
            if filename.endswith('.pdf'):
                documents.append({
                    "id": filename.replace('.pdf', ''),
                    "title": filename.replace('.pdf', '').replace('_', ' ').title(),
                    "filename": filename,
                    "path": f"/pdf/{filename}"
                })
        
        return {"documents": documents}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting documents: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host=settings.FASTAPI_HOST,
        port=settings.FASTAPI_PORT,
        reload=True,
        log_level="info"
    )