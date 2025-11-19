"""
RetrievalAgent - Fetch relevant documents/chunks from vector database
"""
import os
import pickle
import numpy as np
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from config.settings import Settings

@dataclass
class RetrievalResult:
    """Result of document retrieval"""
    documents: List[Any]
    sources: List[str]
    similarity_scores: List[float]
    total_found: int
    average_confidence: float
    retrieval_method: str

class RetrievalAgent:
    """Agent for retrieving relevant documents from vector database"""
    
    def __init__(self, top_k: int = 8):
        self.settings = Settings()
        self.top_k = top_k
        
        # Initialize embeddings
        self.embeddings = HuggingFaceEmbeddings(
            model_name=self.settings.EMBEDDING_MODEL,
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )
        
        # Vectorstore will be loaded
        self.vectorstore = None
        self.is_initialized = False
    
    def initialize(self) -> None:
        """Initialize the vectorstore"""
        try:
            faiss_path = os.path.join(self.settings.DB_FAISS_PATH, "index.faiss")
            pkl_path = os.path.join(self.settings.DB_FAISS_PATH, "index.pkl")
            
            if os.path.exists(faiss_path) and os.path.exists(pkl_path):
                self.vectorstore = FAISS.load_local(
                    self.settings.DB_FAISS_PATH,
                    self.embeddings,
                    allow_dangerous_deserialization=True
                )
                
                print(f"   📚 {self.vectorstore.index.ntotal} documents loaded")
                self.is_initialized = True
                
            else:
                print(f"   ⚠️  Document index not found")
                self.is_initialized = False
                
        except Exception as e:
            print(f"   ⚠️  Could not load documents: {str(e)}")
            self.is_initialized = False
    
    def retrieve_documents(self, query: str, top_k: Optional[int] = None) -> RetrievalResult:
        """Retrieve relevant documents for the query with enhanced strategy"""
        try:
            if not self.is_initialized or self.vectorstore is None:
                return RetrievalResult(
                    documents=[],
                    sources=[],
                    similarity_scores=[],
                    total_found=0,
                    average_confidence=0.0,
                    retrieval_method="none"
                )
            
            # Use provided top_k or default, focused retrieval
            k = top_k if top_k is not None else 10  # Balanced retrieval for focused responses
            
            # Primary retrieval using similarity search
            docs_with_scores = self.vectorstore.similarity_search_with_score(query, k=k)
            
            # If no results, try with expanded query terms
            if not docs_with_scores:
                # Try with individual keywords
                keywords = query.lower().split()
                if len(keywords) > 1:
                    for keyword in keywords:
                        if len(keyword) > 3:  # Skip short words
                            keyword_docs = self.vectorstore.similarity_search_with_score(keyword, k=k//2)
                            docs_with_scores.extend(keyword_docs)
                            if docs_with_scores:
                                break
            
            if not docs_with_scores:
                return RetrievalResult(
                    documents=[],
                    sources=[],
                    similarity_scores=[],
                    total_found=0,
                    average_confidence=0.0,
                    retrieval_method="similarity_search"
                )
            
            # Remove duplicates while preserving order and scores
            seen_content = set()
            unique_docs_with_scores = []
            for doc, score in docs_with_scores:
                content_hash = hash(getattr(doc, 'page_content', '')[:100])  # Use first 100 chars as hash
                if content_hash not in seen_content:
                    seen_content.add(content_hash)
                    unique_docs_with_scores.append((doc, score))
            
            # Take top results after deduplication
            docs_with_scores = unique_docs_with_scores[:k]
            
            # Extract documents, sources, and scores
            documents = [doc for doc, score in docs_with_scores]
            similarity_scores = [float(score) for doc, score in docs_with_scores]
            sources = []
            
            for doc in documents:
                if hasattr(doc, 'metadata') and 'source' in doc.metadata:
                    sources.append(doc.metadata['source'])
                else:
                    sources.append('Unknown')
            
            # Calculate average confidence
            average_confidence = np.mean(similarity_scores) if similarity_scores else 0.0
            
            # Removed verbose logging - summary shown in main endpoint
            
            return RetrievalResult(
                documents=documents,
                sources=sources,
                similarity_scores=similarity_scores,
                total_found=len(documents),
                average_confidence=average_confidence,
                retrieval_method="enhanced_similarity_search"
            )
            
        except Exception as e:
            print(f"[RETRIEVAL AGENT ERROR] {str(e)}")
            return RetrievalResult(
                documents=[],
                sources=[],
                similarity_scores=[],
                total_found=0,
                average_confidence=0.0,
                retrieval_method="error"
            )
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get retrieval agent statistics"""
        if not self.is_initialized or self.vectorstore is None:
            return {"status": "not_initialized"}
        
        return {
            "status": "initialized",
            "total_documents": self.vectorstore.index.ntotal,
            "top_k": self.top_k,
            "embedding_model": self.settings.EMBEDDING_MODEL
        }
