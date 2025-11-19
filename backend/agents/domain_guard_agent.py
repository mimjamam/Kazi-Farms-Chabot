"""
DynamicDomainGuardAgent - Check if query belongs to company domain using dynamic hybrid retrieval and semantic similarity
"""
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from langchain_huggingface import HuggingFaceEmbeddings
from typing import Dict, Any, List, Tuple
from dataclasses import dataclass
from config.settings import Settings

@dataclass
class DomainGuardResult:
    """Result of domain guard check"""
    is_in_domain: bool
    similarity_score: float
    threshold: float
    similar_documents: List[str]
    confidence: float
    retrieval_method: str
    top_docs_count: int

class DynamicDomainGuardAgent:
    """Agent for checking domain relevance using hybrid retrieval and semantic similarity"""
    
    def __init__(self, similarity_threshold: float = None, top_k: int = None):
        self.settings = Settings()
        
        # Use provided values or defaults
        self.base_similarity_threshold = similarity_threshold or 0.42
        self.base_top_k = top_k or 18
        
        # Initialize embeddings
        self.embeddings = HuggingFaceEmbeddings(
            model_name=self.settings.EMBEDDING_MODEL,
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )
        
        # Document embeddings and metadata
        self.document_embeddings = None
        self.document_metadata = None
        self.document_texts = None
        self.tfidf_vectorizer = None
        self.tfidf_matrix = None
        self.is_initialized = False
    
    def initialize(self, vectorstore) -> None:
        """Initialize with vectorstore embeddings and hybrid retrieval"""
        try:
            print("[DYNAMIC DOMAIN GUARD] Initializing with hybrid retrieval...")
            
            if hasattr(vectorstore, 'index') and hasattr(vectorstore.index, 'reconstruct_n'):
                total_docs = vectorstore.index.ntotal
                print(f"[DYNAMIC DOMAIN GUARD] Found {total_docs} documents in vectorstore")
                
                # Get embeddings and texts for all documents
                all_embeddings = []
                all_metadata = []
                all_texts = []
                
                for i in range(total_docs):
                    embedding = vectorstore.index.reconstruct(i)
                    all_embeddings.append(embedding)
                    
                    # Get metadata and text if available
                    try:
                        if hasattr(vectorstore, 'docstore') and hasattr(vectorstore.docstore, '_dict'):
                            if hasattr(vectorstore.index, 'id_map'):
                                doc_id = vectorstore.index.id_map.at(i)
                            else:
                                doc_id = i
                            
                            if doc_id in vectorstore.docstore._dict:
                                doc = vectorstore.docstore._dict[doc_id]
                                metadata = getattr(doc, 'metadata', {})
                                source = metadata.get('source', f'document_{i}')
                                text = getattr(doc, 'page_content', f'document_{i}')
                                all_metadata.append(source)
                                all_texts.append(text)
                            else:
                                all_metadata.append(f'document_{i}')
                                all_texts.append(f'document_{i}')
                        else:
                            all_metadata.append(f'document_{i}')
                            all_texts.append(f'document_{i}')
                    except Exception:
                        all_metadata.append(f'document_{i}')
                        all_texts.append(f'document_{i}')
                
                self.document_embeddings = np.array(all_embeddings)
                self.document_metadata = all_metadata
                self.document_texts = all_texts
                
                # Initialize TF-IDF for keyword search
                self._initialize_tfidf()
                
                print(f"[DYNAMIC DOMAIN GUARD] Loaded {len(self.document_embeddings)} document embeddings")
                print(f"[DYNAMIC DOMAIN GUARD] Initialized TF-IDF with {len(self.document_texts)} documents")
                print(f"[DYNAMIC DOMAIN GUARD] Similarity threshold: {self.similarity_threshold}")
                print(f"[DYNAMIC DOMAIN GUARD] Top-k for retrieval: {self.top_k}")
                
                self.is_initialized = True
                
            else:
                print("[DYNAMIC DOMAIN GUARD] Vectorstore not compatible, using fallback")
                self._initialize_fallback()
                
        except Exception as e:
            print(f"[DYNAMIC DOMAIN GUARD ERROR] Failed to initialize: {str(e)}")
            self._initialize_fallback()
    
    def _initialize_tfidf(self) -> None:
        """Initialize TF-IDF vectorizer for keyword search"""
        try:
            # Limit features to save memory
            max_features = min(500, len(self.document_texts))
            
            self.tfidf_vectorizer = TfidfVectorizer(
                max_features=max_features,
                stop_words='english',
                ngram_range=(1, 2),
                min_df=1,
                max_df=0.95
            )
            
            # Fit TF-IDF on all document texts
            self.tfidf_matrix = self.tfidf_vectorizer.fit_transform(self.document_texts)
            
            print(f"[DYNAMIC DOMAIN GUARD] TF-IDF initialized with {max_features} features")
            
        except Exception as e:
            print(f"[DYNAMIC DOMAIN GUARD ERROR] TF-IDF initialization failed: {str(e)}")
            self.tfidf_vectorizer = None
            self.tfidf_matrix = None
    
    def _hybrid_retrieval(self, query: str, k: int = None) -> Tuple[List[int], List[float]]:
        """Perform hybrid retrieval (vector + keyword) and return top-k document indices"""
        try:
            # Step 1: Vector similarity search
            query_embedding = self.embeddings.embed_query(query)
            query_embedding = np.array(query_embedding).reshape(1, -1)
            
            vector_similarities = cosine_similarity(query_embedding, self.document_embeddings)[0]
            
            # Step 2: Keyword search using TF-IDF
            keyword_similarities = np.zeros(len(self.document_texts))
            
            if self.tfidf_vectorizer is not None and self.tfidf_matrix is not None:
                try:
                    query_tfidf = self.tfidf_vectorizer.transform([query])
                    keyword_similarities = cosine_similarity(query_tfidf, self.tfidf_matrix)[0]
                except Exception as e:
                    print(f"[DYNAMIC DOMAIN GUARD WARNING] Keyword search failed: {str(e)}")
            
            # Step 3: Combine vector and keyword similarities (weighted average)
            combined_similarities = 0.7 * vector_similarities + 0.3 * keyword_similarities
            
            # Use dynamic k if provided, otherwise use base k
            retrieval_k = k or self.base_top_k
            
            # Get top-k document indices
            top_indices = np.argsort(combined_similarities)[-retrieval_k:][::-1]
            top_scores = combined_similarities[top_indices]
            
            print(f"[DYNAMIC DOMAIN GUARD] Hybrid retrieval: {len(top_indices)} docs retrieved (k={retrieval_k})")
            
            return top_indices.tolist(), top_scores.tolist()
            
        except Exception as e:
            print(f"[DYNAMIC DOMAIN GUARD ERROR] Hybrid retrieval failed: {str(e)}")
            # Fallback to simple vector search
            try:
                query_embedding = self.embeddings.embed_query(query)
                query_embedding = np.array(query_embedding).reshape(1, -1)
                similarities = cosine_similarity(query_embedding, self.document_embeddings)[0]
                top_indices = np.argsort(similarities)[-retrieval_k:][::-1]
                top_scores = similarities[top_indices]
                return top_indices.tolist(), top_scores.tolist()
            except Exception:
                return [], []
    
    def check_domain(self, query: str) -> DomainGuardResult:
        """
        Check if query belongs to company domain using dynamic 3-step workflow:
        Step 1: Hybrid retrieval (vector + keyword) → get dynamic top-k docs
        Step 2: Compute similarity of query → top docs  
        Step 3: Dynamic max similarity >= threshold → allow, else → fallback
        """
        try:
            if not self.is_initialized:
                return DomainGuardResult(
                    is_in_domain=True,  # Allow if not initialized
                    similarity_score=0.5,
                    threshold=self.base_similarity_threshold,
                    similar_documents=[],
                    confidence=0.5,
                    retrieval_method="fallback",
                    top_docs_count=0
                )
            
            # Get parameters based on query characteristics
            dynamic_threshold = self._get_adaptive_threshold(query)
            dynamic_top_k = self._get_adaptive_top_k(query)
            
            print(f"[DYNAMIC DOMAIN GUARD] Processing query: '{query[:50]}...'")
            print(f"[DYNAMIC DOMAIN GUARD] Dynamic threshold: {dynamic_threshold}, Dynamic k: {dynamic_top_k}")
            
            # Step 1: Hybrid retrieval (vector + keyword) → get dynamic top-k docs
            print("[DYNAMIC DOMAIN GUARD] Step 1: Dynamic hybrid retrieval")
            top_indices, top_scores = self._hybrid_retrieval(query, dynamic_top_k)
            
            if not top_indices:
                print("[DYNAMIC DOMAIN GUARD] No documents retrieved")
                return DomainGuardResult(
                    is_in_domain=False,
                    similarity_score=0.0,
                    threshold=dynamic_threshold,
                    similar_documents=[],
                    confidence=0.0,
                    retrieval_method="dynamic_hybrid",
                    top_docs_count=0
                )
            
            # Step 2: Compute similarity of query → top docs
            print("[DYNAMIC DOMAIN GUARD] Step 2: Compute similarity")
            max_similarity = max(top_scores) if top_scores else 0.0
            
            # Get similar documents metadata
            similar_documents = [self.document_metadata[i] for i in top_indices[:3]]  # Top 3 for display
            
            # Step 3: Dynamic max similarity >= threshold → allow, else → keyword fallback
            print("[DYNAMIC DOMAIN GUARD] Step 3: Dynamic threshold check")
            is_in_domain = max_similarity >= dynamic_threshold
            
            # Step 4: Keyword fallback if vector score is low
            if not is_in_domain:
                print("[DYNAMIC DOMAIN GUARD] Step 4: Keyword fallback check")
                keyword_match = self._check_keyword_match(query)
                if keyword_match:
                    print("[DYNAMIC DOMAIN GUARD] Keyword match found - allowing query")
                    is_in_domain = True
                    max_similarity = max(max_similarity, 0.4)  # Keyword match boost
            
            print(f"[DYNAMIC DOMAIN GUARD] Max similarity: {max_similarity:.3f}, Dynamic threshold: {dynamic_threshold}")
            print(f"[DYNAMIC DOMAIN GUARD] Decision: {'ALLOW' if is_in_domain else 'BLOCK'}")
            print(f"[DYNAMIC DOMAIN GUARD] Top similar docs: {similar_documents}")
            
            return DomainGuardResult(
                is_in_domain=is_in_domain,
                similarity_score=max_similarity,
                threshold=dynamic_threshold,
                similar_documents=similar_documents,
                confidence=max_similarity,
                retrieval_method="dynamic_hybrid_with_keyword_fallback",
                top_docs_count=len(top_indices)
            )
            
        except Exception as e:
            print(f"[DYNAMIC DOMAIN GUARD ERROR] {str(e)}")
            # Allow query if there's an error
            return DomainGuardResult(
                is_in_domain=True,
                similarity_score=0.5,
                threshold=self.similarity_threshold,
                similar_documents=[],
                confidence=0.5,
                retrieval_method="error",
                top_docs_count=0
            )
    
    def _check_keyword_match(self, query: str) -> bool:
        """Check for keyword matches as fallback when vector similarity is low"""
        try:
            query_lower = query.lower()
            
            # Base domain-specific keywords that should always be allowed
            base_domain_keywords = [
                'salary', 'wage', 'pay', 'compensation', 'bonus', 'allowance',
                'leave', 'vacation', 'holiday', 'sick', 'personal', 'emergency',
                'hr', 'human resources', 'department', 'policy', 'procedure',
                'employee', 'staff', 'worker', 'job', 'position', 'role',
                'kazi', 'farms', 'company', 'organization', 'workplace',
                'attendance', 'time', 'schedule', 'shift', 'overtime',
                'benefit', 'insurance', 'health', 'medical', 'dental',
                'retirement', 'pension', '401k', 'savings', 'investment',
                'training', 'development', 'education', 'course', 'certification',
                'performance', 'review', 'evaluation', 'promotion', 'raise',
                'discipline', 'warning', 'termination', 'resignation', 'retirement',
                'recruitment', 'hiring', 'interview', 'application', 'resume',
                'onboarding', 'orientation', 'induction', 'probation', 'confirmation'
            ]
            
            # Combine base keywords with learned keywords
            all_keywords = base_domain_keywords + self.config_manager.config.learned_keywords
            
            # Check if query contains any domain keywords
            for keyword in all_keywords:
                if keyword in query_lower:
                    print(f"[DYNAMIC DOMAIN GUARD] Keyword match found: '{keyword}'")
                    return True
            
            # Check for common HR-related phrases
            hr_phrases = [
                'how much', 'what is', 'tell me about', 'explain', 'describe',
                'policy on', 'rules for', 'guidelines', 'requirements',
                'how to', 'when can', 'where to', 'who is', 'what are'
            ]
            
            for phrase in hr_phrases:
                if phrase in query_lower:
                    print(f"[DYNAMIC DOMAIN GUARD] HR phrase match found: '{phrase}'")
                    return True
            
            return False
            
        except Exception as e:
            print(f"[DYNAMIC DOMAIN GUARD ERROR] Keyword check failed: {str(e)}")
            return False
    
    def _initialize_fallback(self) -> None:
        """Fallback initialization"""
        print("[DYNAMIC DOMAIN GUARD] Using fallback initialization...")
        self.is_initialized = True
    
    def update_threshold(self, new_threshold: float) -> None:
        """Update similarity threshold"""
        self.similarity_threshold = new_threshold
        print(f"[DYNAMIC DOMAIN GUARD] Threshold updated to: {self.similarity_threshold}")

    def _get_adaptive_threshold(self, query: str) -> float:
        """Get adaptive threshold based on query characteristics"""
        threshold = self.base_similarity_threshold
        
        # Adjust based on query length
        query_length = len(query.split())
        if query_length > 20:
            threshold -= 0.05  # Lower threshold for complex queries
        
        # Ensure within bounds
        return max(0.35, min(0.55, threshold))
    
    def _get_adaptive_top_k(self, query: str) -> int:
        """Get adaptive top-k based on query characteristics"""
        k = self.base_top_k
        
        # Adjust based on query complexity
        query_length = len(query.split())
        if query_length > 20:
            k = int(k * 1.2)  # More complex queries need more documents
        
        # Ensure within bounds
        return max(10, min(30, k))
