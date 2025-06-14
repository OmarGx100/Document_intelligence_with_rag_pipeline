import re
import logging
from typing import Dict, List, Tuple, Optional, Any
from enum import Enum
from dataclasses import dataclass
from vector_database_faiss import FaissVectorDB
from langchain_core.documents import Document
from openai import OpenAI, AsyncOpenAI
from dotenv import load_dotenv
import os


load_dotenv('.env')


MODEL_NAME = str(os.environ['MODEL_NAME'])
TEMPERATURE = float(os.environ['MODEL_TEMPERATURE'])




class SearchStrategy(Enum):
    SEMANTIC = "semantic"
    BY_DOCUMENT_TYPE = "by_document_type" 
    BY_PAGE = "by_page"
    HYBRID = "hybrid"

@dataclass
class QueryAnalysis:
    search_strategy: SearchStrategy
    document_types: Optional[List[str]] = None
    page_numbers: Optional[List[int]] = None
    confidence: float = 0.0
    reasoning: str = ""

class IntelligentQueryRouter:
    """
    Routes user queries to the most appropriate search strategy based on query analysis.
    Integrates with FaissVectorDB to provide intelligent RAG retrieval.
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # Define patterns for different search strategies
        self.page_patterns = [
            r'\b(?:page|pg\.?)\s*([0-9]+(?:-[0-9]+)?(?:\s*,\s*[0-9]+(?:-[0-9]+)?)*)\b',
            r'\bon\s+(?:page|pg\.?)\s*([0-9]+)\b',
            r'\bfrom\s+(?:page|pg\.?)\s*([0-9]+)\b',
            r'\b([0-9]+)(?:st|nd|rd|th)\s+page\b'
        ]
        
        self.document_type_patterns = {
            'table': [
                r'\b(?:table|chart|data|statistics|numbers|figures|rows|columns)\b',
                r'\bshow\s+(?:me\s+)?(?:the\s+)?(?:table|data|chart)\b',
                r'\b(?:financial|numeric|quantitative)\s+(?:data|information)\b'
            ],
            'paragraph': [
                r'\b(?:text|paragraph|content|description|explanation|details)\b',
                r'\bexplain|describe|tell\s+me\s+about\b',
                r'\b(?:main|key)\s+(?:content|text|information)\b'
            ],
            'heading': [
                r'\b(?:title|heading|section|chapter|topic)\b',
                r'\b(?:what\s+are\s+the\s+)?(?:main\s+)?(?:sections|topics|headings)\b',
                r'\bstructure\s+of\b'
            ]
        }
        
        # Intent classification keywords
        self.intent_keywords = {
            'specific_location': [
                'where', 'which page', 'location', 'find on page', 'page number'
            ],
            'data_extraction': [
                'show me', 'extract', 'get the', 'find the', 'what is', 'how much'
            ],
            'structural_query': [
                'outline', 'structure', 'sections', 'contents', 'overview'
            ],
            'comparison': [
                'compare', 'difference', 'versus', 'vs', 'between'
            ]
        }

    def analyze_query(self, query: str, document_info: Dict = None) -> QueryAnalysis:
        """
        Analyze user query to determine the best search strategy.
        
        Args:
            query: User's search query
            document_info: Optional info about the document (from get_database_info())
            
        Returns:
            QueryAnalysis with recommended search strategy
        """
        query_lower = query.lower()
        
        # 1. Check for explicit page references
        page_analysis = self._analyze_page_references(query)
        if page_analysis.page_numbers:
            return QueryAnalysis(
                search_strategy=SearchStrategy.BY_PAGE,
                page_numbers=page_analysis.page_numbers,
                confidence=0.9,
                reasoning="Query explicitly mentions page numbers"
            )
        
        # 2. Check for document type preferences
        type_analysis = self._analyze_document_types(query)
        if type_analysis.document_types and type_analysis.confidence > 0.7:
            return QueryAnalysis(
                search_strategy=SearchStrategy.BY_DOCUMENT_TYPE,
                document_types=type_analysis.document_types,
                confidence=type_analysis.confidence,
                reasoning=f"Query indicates preference for {', '.join(type_analysis.document_types)} content"
            )
        
        # 3. Check for hybrid scenarios
        hybrid_analysis = self._analyze_hybrid_needs(query, document_info)
        if hybrid_analysis.search_strategy == SearchStrategy.HYBRID:
            return hybrid_analysis
        
        # 4. Default to semantic search
        return QueryAnalysis(
            search_strategy=SearchStrategy.SEMANTIC,
            confidence=0.6,
            reasoning="General semantic search appropriate for this query"
        )

    def _analyze_page_references(self, query: str) -> QueryAnalysis:
        """Extract page numbers from query."""
        page_numbers = []
        
        for pattern in self.page_patterns:
            matches = re.finditer(pattern, query, re.IGNORECASE)
            for match in matches:
                page_str = match.group(1)
                # Parse page numbers and ranges
                pages = self._parse_page_string(page_str)
                page_numbers.extend(pages)
        
        return QueryAnalysis(
            search_strategy=SearchStrategy.BY_PAGE,
            page_numbers=list(set(page_numbers)) if page_numbers else None,
            confidence=0.9 if page_numbers else 0.0
        )

    def _parse_page_string(self, page_str: str) -> List[int]:
        """Parse page string like '1-3,5,7-9' into list of page numbers."""
        pages = []
        parts = page_str.replace(' ', '').split(',')
        
        for part in parts:
            if '-' in part:
                # Handle ranges
                try:
                    start, end = map(int, part.split('-'))
                    pages.extend(range(start, end + 1))
                except ValueError:
                    continue
            else:
                # Handle single pages
                try:
                    pages.append(int(part))
                except ValueError:
                    continue
        
        return pages

    def _analyze_document_types(self, query: str) -> QueryAnalysis:
        """Analyze query for document type preferences."""
        query_lower = query.lower()
        type_scores = {}
        
        for doc_type, patterns in self.document_type_patterns.items():
            score = 0
            for pattern in patterns:
                matches = len(re.findall(pattern, query_lower))
                score += matches
            
            if score > 0:
                type_scores[doc_type] = score
        
        if not type_scores:
            return QueryAnalysis(search_strategy=SearchStrategy.SEMANTIC, confidence=0.0)
        
        # Get the highest scoring document type(s)
        max_score = max(type_scores.values())
        preferred_types = [doc_type for doc_type, score in type_scores.items() if score == max_score]
        
        confidence = min(0.9, 0.5 + (max_score * 0.2))  # Scale confidence based on matches
        
        return QueryAnalysis(
            search_strategy=SearchStrategy.BY_DOCUMENT_TYPE,
            document_types=preferred_types,
            confidence=confidence
        )

    def _analyze_hybrid_needs(self, query: str, document_info: Dict = None) -> QueryAnalysis:
        """Determine if query needs hybrid search approach."""
        query_lower = query.lower()
        
        # Scenarios that benefit from hybrid search:
        
        # 1. Comparison queries (might need multiple document types)
        if any(keyword in query_lower for keyword in self.intent_keywords['comparison']):
            return QueryAnalysis(
                search_strategy=SearchStrategy.HYBRID,
                document_types=['table', 'paragraph'],  # Often need both data and context
                confidence=0.8,
                reasoning="Comparison query benefits from both tabular data and textual context"
            )
        
        # 2. Complex analytical queries
        analytical_indicators = ['analyze', 'summary', 'overview', 'comprehensive', 'detailed analysis']
        if any(indicator in query_lower for indicator in analytical_indicators):
            return QueryAnalysis(
                search_strategy=SearchStrategy.HYBRID,
                document_types=['heading', 'paragraph', 'table'],
                confidence=0.7,
                reasoning="Analytical query needs comprehensive information from multiple sources"
            )
        
        # 3. Questions that might span multiple pages/sections
        spanning_indicators = ['throughout', 'across', 'entire document', 'all sections', 'complete']
        if any(indicator in query_lower for indicator in spanning_indicators):
            return QueryAnalysis(
                search_strategy=SearchStrategy.HYBRID,
                confidence=0.8,
                reasoning="Query spans multiple sections, hybrid approach recommended"
            )
        
        return QueryAnalysis(search_strategy=SearchStrategy.SEMANTIC, confidence=0.0)

    def execute_search(self, vector_db : FaissVectorDB, query: str, k: int = 4, document_info: Dict = None) -> Tuple[List, QueryAnalysis]:
        """
        Execute the optimal search strategy based on query analysis.
        
        Args:
            vector_db: FaissVectorDB instance
            query: User query
            k: Number of results to return
            document_info: Optional document information
            
        Returns:
            Tuple of (search_results, query_analysis)
        """
        analysis = self.analyze_query(query, document_info)
        
        try:
            if analysis.search_strategy == SearchStrategy.BY_PAGE:
                results = vector_db.search_by_page(query, analysis.page_numbers, k)
                
            elif analysis.search_strategy == SearchStrategy.BY_DOCUMENT_TYPE:
                results = vector_db.search_by_document_type(query, analysis.document_types, k)
                
            elif analysis.search_strategy == SearchStrategy.HYBRID:
                results = self._execute_hybrid_search(vector_db, query, analysis, k)
                
            else:  # SEMANTIC
                results = vector_db.similarity_search(query, k)
            
            self.logger.info(f"Executed {analysis.search_strategy.value} search for query: '{query[:50]}...'")
            return results, analysis
            
        except Exception as e:
            self.logger.error(f"Search execution failed: {e}")
            # Fallback to semantic search
            return vector_db.similarity_search(query, k), analysis

    def _execute_hybrid_search(self, vector_db :FaissVectorDB, query: str, analysis: QueryAnalysis, k: int) -> List:
        """Execute hybrid search combining multiple strategies."""
        all_results = []
        
        # 1. Get semantic results
        semantic_results = vector_db.similarity_search(query, k//2)
        all_results.extend(semantic_results)
        
        # 2. Get document type specific results if specified
        if analysis.document_types:
            type_results = vector_db.search_by_document_type(query, analysis.document_types, k//2)
            all_results.extend(type_results)
        
        # 3. Get page-specific results if specified
        if analysis.page_numbers:
            page_results = vector_db.search_by_page(query, analysis.page_numbers, k//2)
            all_results.extend(page_results)
        
        # 4. Deduplicate and rank results
        unique_results = self._deduplicate_results(all_results)
        
        # Return top k results
        return unique_results[:k]

    def _deduplicate_results(self, results: List[Document]) -> List:
        """Remove duplicate documents based on content similarity."""
        unique_results = []
        seen_content = set()
        
        for doc in results:
            # Use first 100 characters as a simple content hash
            content_hash = doc.page_content[:100].strip()
            if content_hash not in seen_content:
                seen_content.add(content_hash)
                unique_results.append(doc)
        
        return unique_results

    def get_search_explanation(self, analysis: QueryAnalysis) -> str:
        """Generate human-readable explanation of search strategy."""
        strategy_explanations = {
            SearchStrategy.SEMANTIC: "Using semantic search to find contextually relevant information",
            # SearchStrategy.BY_PAGE: f"Searching specific pages: {analysis.page_numbers}",
            # SearchStrategy.BY_DOCUMENT_TYPE: f"Focusing on {', '.join(analysis.document_types)} content",
            SearchStrategy.HYBRID: "Using hybrid approach combining multiple search methods"
        }
        
        base_explanation = strategy_explanations.get(analysis.search_strategy, "Using default search")
        return f"{base_explanation}. Confidence: {analysis.confidence:.2f}. Reasoning: {analysis.reasoning}"


# Integration example with RAG pipeline
class RAGPipeline:
    """Example RAG pipeline integrating the intelligent query router."""
    
    def __init__(self, vector_db : FaissVectorDB, llm_client=None):
        self.vector_db = vector_db
        self.query_router = IntelligentQueryRouter()
        self.llm_client = llm_client
        
    def query(self, user_query: str, k: int = 4, explain_search: bool = False) -> Dict[str, Any]:
        """
        Process user query with intelligent routing.
        
        Args:
            user_query: User's question
            k: Number of context documents to retrieve
            explain_search: Whether to include search strategy explanation
            
        Returns:
            Dictionary with answer and metadata
        """
        # Get document info for context
        doc_info = self.vector_db.get_database_info()
        
        # Execute intelligent search
        search_results, analysis = self.query_router.execute_search(
            self.vector_db, user_query, k, doc_info
        )
        
        # Build context from results
        context = self._build_context(search_results)
        
        # Generate response using LLM (if available)
        if self.llm_client:
            response = self._generate_response(user_query, context)
        else:
            response = "LLM not configured - returning raw context"
        
        result = {
            "answer": response,
            "context_documents": search_results,
            "search_strategy": analysis.search_strategy.value,
            "num_results": len(search_results)
        }
        
        if explain_search:
            result["search_explanation"] = self.query_router.get_search_explanation(analysis)
            
        return result
    
    def _build_context(self, documents: List[Document]) -> str:
        """Build context string from retrieved documents."""
        context_parts = []
        
        for i, doc in enumerate(documents, 1):
            metadata = doc.metadata
            
            # Add document context with metadata
            doc_info = f"Document {i}"
            if metadata.get('chunk_type'):
                doc_info += f" ({metadata['chunk_type']})"
            if metadata.get('page_number'):
                doc_info += f" - Page {metadata['page_number']}"
            if metadata.get('section'):
                doc_info += f" - Section: {metadata['section']}"
                
            context_parts.append(f"{doc_info}:\n{doc.page_content}\n")
        
        return "\n".join(context_parts)
    
    def _generate_response(self, query: str, context: str):
        """Generate response using LLM (placeholder - implement with your LLM client)."""
        # This would integrate with your LLM of choice
        prompt = f"""Based on the following context, YOU ARE A proffessional HR recruter that will help the user answer questions about a provided cv's.,
                you will tepically be provided with document intellegence chunks from the cv....

                    Context:
                    {context}

                    Question: {query}

                    Answer:"""
        

        print(f"PROMPT TO THE MODEL : {prompt}")
        # Replace with actual LLM call
        try:
            response = self.llm_client.chat.completions.create(
                model = MODEL_NAME,
                messages= [{'role' : 'user', 'content' : prompt}],
                temperature=TEMPERATURE,
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"An Exception Occured While trying to reach openAI server, Exception Details : {e}")
            return


# Usage examples
def demonstrate_intelligent_routing():
    """Show how the intelligent query router works with different query types."""
    
    # Example queries and their expected routing
    test_queries = [
        "Show me the financial data from page 5",  # BY_PAGE
        "What tables are available in this document?",  # BY_DOCUMENT_TYPE (table)
        "Explain the main conclusions in the text",  # BY_DOCUMENT_TYPE (paragraph)
        "Compare the revenue figures with the expense data",  # HYBRID
        "What is the company's mission statement?",  # SEMANTIC
        "Give me a comprehensive analysis of pages 10-15",  # HYBRID
    ]
    
    router = IntelligentQueryRouter()
    
    for query in test_queries:
        analysis = router.analyze_query(query)
        print(f"Query: {query}")
        print(f"Strategy: {analysis.search_strategy.value}")
        print(f"Confidence: {analysis.confidence:.2f}")
        print(f"Reasoning: {analysis.reasoning}")
        if analysis.document_types:
            print(f"Document Types: {analysis.document_types}")
        if analysis.page_numbers:
            print(f"Page Numbers: {analysis.page_numbers}")
        print("-" * 50)

if __name__ == "__main__":
    demonstrate_intelligent_routing()