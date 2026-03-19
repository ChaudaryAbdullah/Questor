"""
RAG Engine Module
Implements hybrid retrieval-augmented generation for fraud detection.
"""

import json
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
import os

from .vector_store_manager import VectorStoreManager, SearchResult
from ..utils.logger import get_retrieval_logger
from ..utils.config_manager import get_config


@dataclass
class RAGResponse:
    """Response from RAG engine."""
    query: str
    answer: str
    sources: List[Dict]
    confidence: float
    fraud_indicators: List[str] = field(default_factory=list)
    metadata: Dict = field(default_factory=dict)


class RAGEngine:
    """
    Retrieval-Augmented Generation engine for fraud detection.
    Combines vector search with graph traversal and LLM generation.
    """
    
    FRAUD_QUERIES = [
        "Find undisclosed related party transactions",
        "Identify contradictions in revenue recognition descriptions",
        "Detect changes in risk factors from previous year",
        "Find unusual relationships between executives and suppliers",
        "Identify subsidiaries mentioned but not in consolidation",
        "Find circular transaction patterns between entities",
        "Detect sudden changes in accounting policies",
        "Identify executives with undisclosed business interests",
        "Find shell companies or special purpose entities",
        "Detect auditor changes or qualified opinions",
        "Identify hidden liabilities or contingencies",
        "Find inconsistencies between MD&A and financial statements"
    ]
    
    ANALYSIS_PROMPT = """You are a financial fraud detection expert analyzing SEC 10-K filings.

Based on the following context from the filing, answer the query. Focus on identifying potential fraud indicators, inconsistencies, or undisclosed information.

QUERY: {query}

CONTEXT:
{context}

Provide a detailed analysis that:
1. Directly answers the query based on the context
2. Identifies any red flags or concerns
3. Notes any missing disclosures or inconsistencies
4. Assigns a risk level (LOW, MEDIUM, HIGH, CRITICAL)

Format your response as JSON:
{{
    "answer": "Your detailed analysis",
    "fraud_indicators": ["list", "of", "indicators"],
    "risk_level": "LOW|MEDIUM|HIGH|CRITICAL",
    "evidence": ["relevant quotes from context"],
    "recommendations": ["follow-up actions"]
}}
"""
    
    def __init__(
        self,
        vector_store: Optional[VectorStoreManager] = None,
        embedding_generator = None,
        graph_builder = None,
        config_path: Optional[str] = None
    ):
        """
        Initialize RAG engine.
        
        Args:
            vector_store: VectorStoreManager instance
            embedding_generator: EmbeddingGenerator instance
            graph_builder: KnowledgeGraphBuilder instance
            config_path: Path to configuration
        """
        self.logger = get_retrieval_logger()
        
        self.vector_store = vector_store
        self.embedding_generator = embedding_generator
        self.graph_builder = graph_builder
        
        # Load config
        try:
            config = get_config(config_path)
            self.rag_config = config.rag_config
            self.llm_config = config.llm_config
        except Exception:
            self.rag_config = {}
            self.llm_config = {}
        
        self._groq_client = None
        self.logger.info("RAGEngine initialized")
    
    def _init_groq(self) -> None:
        """Initialize Groq client."""
        if self._groq_client is not None:
            return
        
        try:
            from groq import Groq
            
            api_key = os.getenv("GROQ_API_KEY", self.llm_config.get("api_key", ""))
            if not api_key:
                raise ValueError("GROQ_API_KEY not found")
            
            self._groq_client = Groq(api_key=api_key)
            self.logger.info("Groq client initialized for RAG")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize Groq: {e}")
            raise
    
    def retrieve_context(
        self,
        query: str,
        top_k: int = 5,
        filter_metadata: Optional[Dict] = None
    ) -> List[Dict]:
        """
        Retrieve relevant context for query.
        
        Args:
            query: Query string
            top_k: Number of results
            filter_metadata: Optional metadata filter
        
        Returns:
            List of context dictionaries
        """
        contexts = []
        
        # Vector similarity search
        if self.vector_store and self.embedding_generator:
            results = self.vector_store.search_by_text(
                query, self.embedding_generator, top_k, filter_metadata
            )
            
            for result in results:
                contexts.append({
                    "source": "vector_search",
                    "id": result.id,
                    "text": result.text,
                    "score": result.score,
                    "metadata": result.metadata
                })
        
        # Graph-based retrieval (if available)
        if self.graph_builder:
            try:
                graph_results = self._graph_search(query, top_k // 2)
                contexts.extend(graph_results)
            except Exception as e:
                self.logger.warning(f"Graph search failed: {e}")
        
        # Sort by score and deduplicate
        contexts.sort(key=lambda x: x.get("score", 0), reverse=True)
        
        # Deduplicate by text similarity
        unique_contexts = []
        seen_texts = set()
        for ctx in contexts:
            text_key = ctx["text"][:100]
            if text_key not in seen_texts:
                seen_texts.add(text_key)
                unique_contexts.append(ctx)
        
        return unique_contexts[:top_k]
    
    def _graph_search(self, query: str, top_k: int) -> List[Dict]:
        """Perform graph-based search."""
        results = []
        
        if not self.graph_builder:
            return results
        
        # Extract entities from query
        query_lower = query.lower()
        
        # Look for entity mentions
        keywords = ["subsidiary", "related party", "executive", "auditor", 
                   "transaction", "disclosure", "shell company"]
        
        for keyword in keywords:
            if keyword in query_lower:
                # Get related nodes from graph
                try:
                    nodes = self.graph_builder.search_nodes(keyword, limit=top_k)
                    for node in nodes:
                        results.append({
                            "source": "graph_search",
                            "id": node.get("id", ""),
                            "text": node.get("description", str(node)),
                            "score": 0.7,
                            "metadata": node
                        })
                except Exception:
                    pass
        
        return results
    
    def generate_response(
        self,
        query: str,
        context: List[Dict]
    ) -> RAGResponse:
        """
        Generate response using LLM with rate limiting.
        
        Args:
            query: User query
            context: Retrieved context
        
        Returns:
            RAGResponse object
        """
        import time
        
        self._init_groq()
        
        # Get rate limit settings
        rate_limit_config = self.llm_config.get("rate_limit", {})
        delay_between_requests = rate_limit_config.get("delay_between_requests", 2.5)
        max_retries = self.llm_config.get("retry_attempts", 3)
        
        # Add delay before API call to respect rate limits
        if hasattr(self, '_last_api_call_time'):
            time_since_last_call = time.time() - self._last_api_call_time
            if time_since_last_call < delay_between_requests:
                sleep_time = delay_between_requests - time_since_last_call
                self.logger.debug(f"Rate limiting: sleeping for {sleep_time:.2f}s")
                time.sleep(sleep_time)
        
        # Build context string
        context_str = "\n\n---\n\n".join([
            f"[Source: {c.get('metadata', {}).get('section', 'Unknown')}]\n{c['text']}"
            for c in context
        ])
        
        # Prepare prompt
        prompt = self.ANALYSIS_PROMPT.format(query=query, context=context_str[:8000])
        
        # Retry loop for handling rate limits
        for attempt in range(max_retries):
            try:
                model = self.llm_config.get("model", "llama-3.1-8b-instant")
                
                response = self._groq_client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": "You are a financial fraud detection expert. Respond only with valid JSON."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.2,
                    max_tokens=2048
                )
                
                # Track API call time
                self._last_api_call_time = time.time()
                
                response_text = response.choices[0].message.content
                
                # Parse JSON response
                import re
                json_match = re.search(r'\{[\s\S]*\}', response_text)
                if json_match:
                    parsed = json.loads(json_match.group())
                    
                    return RAGResponse(
                        query=query,
                        answer=parsed.get("answer", response_text),
                        sources=[{"text": c["text"][:200], "section": c.get("metadata", {}).get("section")} 
                                for c in context],
                        confidence=0.8,
                        fraud_indicators=parsed.get("fraud_indicators", []),
                        metadata={
                            "risk_level": parsed.get("risk_level", "UNKNOWN"),
                            "evidence": parsed.get("evidence", []),
                            "recommendations": parsed.get("recommendations", [])
                        }
                    )
                else:
                    return RAGResponse(
                        query=query,
                        answer=response_text,
                        sources=[{"text": c["text"][:200]} for c in context],
                        confidence=0.6
                    )
                    
            except Exception as e:
                error_message = str(e)
                
                # Check if it's a rate limit error
                if "rate_limit" in error_message.lower() or "429" in error_message:
                    # Extract wait time from error message if available
                    wait_time_match = re.search(r'try again in ([\d.]+)', error_message.lower())
                    if wait_time_match:
                        # Try to parse wait time (could be in seconds or minutes)
                        wait_str = wait_time_match.group(1)
                        wait_time = float(wait_str) + 1  # Add 1 second buffer
                        # Check if it's in minutes format like "3m30.9888s"
                        if 'm' in error_message.lower():
                            minutes_match = re.search(r'(\d+)m([\d.]+)s', error_message.lower())
                            if minutes_match:
                                wait_time = int(minutes_match.group(1)) * 60 + float(minutes_match.group(2)) + 1
                    else:
                        # Exponential backoff
                        wait_time = (2 ** attempt) * 2  # 2s, 4s, 8s
                    
                    if attempt < max_retries - 1:
                        self.logger.warning(f"Rate limit hit, retrying in {wait_time:.1f}s... (attempt {attempt + 1}/{max_retries})")
                        time.sleep(wait_time)
                    else:
                        self.logger.error(f"Rate limit exceeded after {max_retries} attempts")
                        return RAGResponse(
                            query=query,
                            answer=f"Error: Rate limit exceeded after {max_retries} attempts. {error_message}",
                            sources=[],
                            confidence=0.0
                        )
                else:
                    self.logger.error(f"Generation failed: {e}")
                    return RAGResponse(
                        query=query,
                        answer=f"Error generating response: {e}",
                        sources=[],
                        confidence=0.0
                    )
    
    def query(
        self,
        query: str,
        top_k: int = 5,
        filter_metadata: Optional[Dict] = None
    ) -> RAGResponse:
        """
        Full RAG query pipeline.
        
        Args:
            query: User query
            top_k: Number of context chunks
            filter_metadata: Optional filter
        
        Returns:
            RAGResponse object
        """
        self.logger.info(f"Processing query: {query[:50]}...")
        
        # Retrieve context
        context = self.retrieve_context(query, top_k, filter_metadata)
        
        if not context:
            return RAGResponse(
                query=query,
                answer="No relevant context found in the document.",
                sources=[],
                confidence=0.0
            )
        
        # Generate response
        response = self.generate_response(query, context)
        
        self.logger.info(f"Query completed. Confidence: {response.confidence}")
        return response
    
    def run_fraud_analysis(
        self,
        queries: Optional[List[str]] = None
    ) -> List[RAGResponse]:
        """
        Run predefined fraud detection queries with rate limiting.
        
        Args:
            queries: Optional custom queries
        
        Returns:
            List of RAGResponse objects
        """
        import time
        
        queries = queries or self.FRAUD_QUERIES
        results = []
        
        # Get delay between queries from config
        rate_limit_config = self.llm_config.get("rate_limit", {})
        query_delay = rate_limit_config.get("delay_between_requests", 2.5)
        
        for i, query in enumerate(queries):
            result = self.query(query)
            results.append(result)
            
            # Add delay between queries (except after the last one)
            if i < len(queries) - 1:
                self.logger.debug(f"Waiting {query_delay}s before next query...")
                time.sleep(query_delay)
        
        return results
    
    def summarize_findings(
        self,
        responses: List[RAGResponse]
    ) -> Dict:
        """
        Summarize fraud analysis findings.
        
        Args:
            responses: List of RAG responses
        
        Returns:
            Summary dictionary
        """
        all_indicators = []
        high_risk = []
        
        for resp in responses:
            all_indicators.extend(resp.fraud_indicators)
            risk = resp.metadata.get("risk_level", "UNKNOWN")
            if risk in ["HIGH", "CRITICAL"]:
                high_risk.append({
                    "query": resp.query,
                    "risk_level": risk,
                    "indicators": resp.fraud_indicators
                })
        
        return {
            "total_queries": len(responses),
            "high_risk_findings": len(high_risk),
            "all_indicators": list(set(all_indicators)),
            "high_risk_details": high_risk,
            "recommendation": "REQUIRES REVIEW" if high_risk else "LOW RISK"
        }
