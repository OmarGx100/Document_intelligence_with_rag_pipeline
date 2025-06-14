import os
import logging
from typing import Optional, List, Union
from pathlib import Path

from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama.embeddings import OllamaEmbeddings
from langchain_core.embeddings import Embeddings
from langchain_core.documents import Document


class FaissVectorDB:
    """
    A wrapper class for managing FAISS vector databases with LangChain integration.
    
    This class provides functionality to create, save, load, and query FAISS vector databases
    with configurable text chunking and embedding models.
    """

    def __init__(self,
                 db_name: str,
                 db_path: str,
                 chunk_size: int = 1000,
                 chunk_overlap: int = 200,
                 embedding_model_name: str = "llama2",
                 vec_db: Optional[FAISS] = None):
        """
        Initialize the FaissVectorDB instance.
        
        Args:
            db_name: Name of the database
            db_path: Path where the database will be stored
            chunk_size: Size of text chunks for splitting
            chunk_overlap: Overlap between chunks
            embedding_model_name: Name of the embedding model to use
            vec_db: Optional pre-existing FAISS database
        """
        self.db_name = db_name
        self.db_path = Path(db_path)
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.embedding_model_name = embedding_model_name
        self.vec_db = vec_db
        self._embeddings = None
        
        # Ensure database directory exists
        self.db_path.mkdir(parents=True, exist_ok=True)
        
        # Setup logging
        self.logger = logging.getLogger(__name__)

    @property
    def embeddings(self) -> Embeddings:
        """Lazy loading of embeddings model."""
        if self._embeddings is None:
            self._embeddings = self._generate_embeddings()
        return self._embeddings

    def create_vector_database(self, raw_text: str) -> bool:
        """
        Create a vector database from raw text.
        
        Args:
            raw_text: The raw text to process and store
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            if not raw_text or not raw_text.strip():
                raise ValueError("Raw text cannot be empty")
                
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap
            )

            texts = text_splitter.split_text(raw_text)
            
            if not texts:
                raise ValueError("No text chunks created from input")

            self.vec_db = FAISS.from_texts(texts, self.embeddings)
            self.logger.info(f"Created vector database with {len(texts)} chunks")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to create vector database: {e}")
            return False

    def create_from_documents(self, documents: List[Document]) -> bool:
        """
        Create a vector database from a list of documents.
        
        Args:
            documents: List of LangChain Document objects
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            if not documents:
                raise ValueError("Documents list cannot be empty")
                
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap
            )

            split_docs = text_splitter.split_documents(documents)
            
            if not split_docs:
                raise ValueError("No document chunks created from input")

            self.vec_db = FAISS.from_documents(split_docs, self.embeddings)
            self.logger.info(f"Created vector database from {len(split_docs)} document chunks")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to create vector database from documents: {e}")
            return False

    def create_from_document_intelligence_output(self, document_intelligence_output: dict) -> bool:
        """
        Create a vector database from Document Intelligence API output with semantic chunking.
        
        This method processes the JSON output from Document Intelligence API and creates
        semantically meaningful chunks based on document structure (paragraphs, tables, 
        sections, etc.) rather than just character-based splitting.
        
        Args:
            document_intelligence_output: JSON dictionary from Document Intelligence API
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            if not document_intelligence_output:
                raise ValueError("Document Intelligence output cannot be empty")
            
            # Extract semantic chunks from the Document Intelligence output
            semantic_chunks = self._extract_semantic_chunks(document_intelligence_output)
            
            if not semantic_chunks:
                raise ValueError("No semantic chunks could be extracted from the document")
            
            # Create LangChain Document objects from semantic chunks
            documents = []
            for i, chunk in enumerate(semantic_chunks):
                doc = Document(
                    page_content=chunk['content'],
                    metadata={
                        'chunk_id': i,
                        'chunk_type': chunk['type'],
                        'page_number': chunk.get('page_number', 0),
                        'bounding_box': chunk.get('bounding_box'),
                        'confidence': chunk.get('confidence', 1.0),
                        'source': chunk.get('source', 'document_intelligence'),
                        'section': chunk.get('section', ''),
                        'heading': chunk.get('heading', ''),
                        'table_info': chunk.get('table_info', {}),
                        'word_count': len(chunk['content'].split()),
                        'char_count': len(chunk['content'])
                    }
                )
                documents.append(doc)
            
            # Create vector database from semantic documents
            self.vec_db = FAISS.from_documents(documents, self.embeddings)
            self.logger.info(f"Created vector database from {len(documents)} semantic chunks")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to create vector database from Document Intelligence output: {e}")
            return False

    def _extract_semantic_chunks(self, doc_intelligence_output: dict) -> List[dict]:
        """
        Extract semantic chunks from Document Intelligence output.
        
        This method processes different elements of the document (paragraphs, tables,
        headings, etc.) and creates meaningful chunks with proper context.
        
        Args:
            doc_intelligence_output: Document Intelligence API response
            
        Returns:
            List of semantic chunk dictionaries
        """
        chunks = []
        
        try:
            # Handle different Document Intelligence API response formats
            if 'analyzeResult' in doc_intelligence_output:
                analyze_result = doc_intelligence_output['analyzeResult']
            elif 'pages' in doc_intelligence_output:
                analyze_result = doc_intelligence_output
            else:
                analyze_result = doc_intelligence_output
            
            # Extract pages information
            pages = analyze_result.get('pages', [])
            paragraphs = analyze_result.get('paragraphs', [])
            tables = analyze_result.get('tables', [])
            key_value_pairs = analyze_result.get('keyValuePairs', [])
            
            # Process paragraphs with semantic understanding
            current_section = ""
            current_heading = ""
            
            for para in paragraphs:
                content = para.get('content', '').strip()
                if not content:
                    continue
                
                # Determine paragraph type and role
                role = para.get('role', 'paragraph')
                bounding_box = para.get('boundingRegions', [{}])[0].get('boundingBox', [])
                page_number = para.get('boundingRegions', [{}])[0].get('pageNumber', 1)
                confidence = para.get('confidence', 1.0)
                
                # Handle different paragraph roles
                if role in ['title', 'sectionHeading']:
                    current_heading = content
                    if role == 'title':
                        current_section = content
                    
                    chunks.append({
                        'content': content,
                        'type': 'heading',
                        'role': role,
                        'page_number': page_number,
                        'bounding_box': bounding_box,
                        'confidence': confidence,
                        'section': current_section,
                        'heading': current_heading
                    })
                
                elif role == 'paragraph':
                    # For regular paragraphs, check if they should be combined
                    # based on semantic similarity or if they're too short
                    chunk_content = content
                    
                    # If paragraph is very short, try to combine with context
                    if len(content.split()) < 10 and chunks:
                        # Look for the last paragraph chunk to potentially merge
                        for i in range(len(chunks) - 1, -1, -1):
                            if chunks[i]['type'] == 'paragraph':
                                # Combine if total length is reasonable
                                combined_content = chunks[i]['content'] + '\n\n' + content
                                if len(combined_content.split()) <= self.chunk_size // 4:  # Use quarter of chunk size for combining
                                    chunks[i]['content'] = combined_content
                                    chunks[i]['char_count'] = len(combined_content)
                                    chunk_content = None  # Don't create new chunk
                                break
                    
                    if chunk_content:
                        chunks.append({
                            'content': chunk_content,
                            'type': 'paragraph',
                            'role': role,
                            'page_number': page_number,
                            'bounding_box': bounding_box,
                            'confidence': confidence,
                            'section': current_section,
                            'heading': current_heading
                        })
            
            # Process tables separately
            for i, table in enumerate(tables):
                table_content = self._extract_table_content(table)
                if table_content:
                    bounding_box = table.get('boundingRegions', [{}])[0].get('boundingBox', [])
                    page_number = table.get('boundingRegions', [{}])[0].get('pageNumber', 1)
                    
                    chunks.append({
                        'content': table_content,
                        'type': 'table',
                        'page_number': page_number,
                        'bounding_box': bounding_box,
                        'confidence': table.get('confidence', 1.0),
                        'section': current_section,
                        'heading': current_heading,
                        'table_info': {
                            'table_index': i,
                            'row_count': table.get('rowCount', 0),
                            'column_count': table.get('columnCount', 0)
                        }
                    })
            
            # Process key-value pairs
            if key_value_pairs:
                kv_content = self._extract_key_value_content(key_value_pairs)
                if kv_content:
                    chunks.append({
                        'content': kv_content,
                        'type': 'key_value_pairs',
                        'page_number': 1,
                        'bounding_box': [],
                        'confidence': 1.0,
                        'section': current_section,
                        'heading': 'Key-Value Information'
                    })
            
            # Post-process chunks: split large chunks if necessary
            final_chunks = []
            for chunk in chunks:
                if len(chunk['content'].split()) > self.chunk_size:
                    # Split large chunks while preserving semantic meaning
                    split_chunks = self._split_large_semantic_chunk(chunk)
                    final_chunks.extend(split_chunks)
                else:
                    final_chunks.append(chunk)
            
            return final_chunks
            
        except Exception as e:
            self.logger.error(f"Error extracting semantic chunks: {e}")
            return []

    def _extract_table_content(self, table: dict) -> str:
        """
        Extract and format table content in a readable format.
        
        Args:
            table: Table dictionary from Document Intelligence
            
        Returns:
            Formatted table content as string
        """
        try:
            cells = table.get('cells', [])
            if not cells:
                return ""
            
            # Create a matrix to hold cell content
            max_row = max(cell.get('rowIndex', 0) for cell in cells) + 1
            max_col = max(cell.get('columnIndex', 0) for cell in cells) + 1
            
            table_matrix = [[''] * max_col for _ in range(max_row)]
            
            # Fill the matrix with cell content
            for cell in cells:
                row_idx = cell.get('rowIndex', 0)
                col_idx = cell.get('columnIndex', 0)
                content = cell.get('content', '').strip()
                table_matrix[row_idx][col_idx] = content
            
            # Convert matrix to readable text format
            table_text = "Table:\n"
            for row in table_matrix:
                # Clean up empty cells and join with pipes
                clean_row = [cell if cell else '-' for cell in row]
                table_text += " | ".join(clean_row) + "\n"
            
            return table_text.strip()
            
        except Exception as e:
            self.logger.error(f"Error extracting table content: {e}")
            return ""

    def _extract_key_value_content(self, key_value_pairs: List[dict]) -> str:
        """
        Extract and format key-value pairs content.
        
        Args:
            key_value_pairs: List of key-value pair dictionaries
            
        Returns:
            Formatted key-value content as string
        """
        try:
            kv_text = "Key-Value Information:\n"
            
            for kv_pair in key_value_pairs:
                key = kv_pair.get('key', {}).get('content', '').strip()
                value = kv_pair.get('value', {}).get('content', '').strip()
                confidence = kv_pair.get('confidence', 1.0)
                
                if key and value:
                    kv_text += f"{key}: {value}\n"
                elif key:
                    kv_text += f"{key}: [No value detected]\n"
            
            return kv_text.strip() if kv_text.strip() != "Key-Value Information:" else ""
            
        except Exception as e:
            self.logger.error(f"Error extracting key-value content: {e}")
            return ""

    def _split_large_semantic_chunk(self, chunk: dict) -> List[dict]:
        """
        Split large semantic chunks while preserving meaning.
        
        Args:
            chunk: Large chunk dictionary to split
            
        Returns:
            List of smaller chunk dictionaries
        """
        try:
            content = chunk['content']
            
            # Use text splitter but with smaller chunks to preserve semantic meaning
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap,
                separators=['\n\n', '\n', '. ', '? ', '! ', '; ', ', ', ' ']
            )
            
            split_texts = text_splitter.split_text(content)
            
            split_chunks = []
            for i, text in enumerate(split_texts):
                new_chunk = chunk.copy()
                new_chunk['content'] = text
                new_chunk['chunk_id'] = f"{chunk.get('chunk_id', 0)}_{i}"
                new_chunk['is_split'] = True
                new_chunk['original_chunk_parts'] = len(split_texts)
                new_chunk['part_index'] = i
                split_chunks.append(new_chunk)
            
            return split_chunks
            
        except Exception as e:
            self.logger.error(f"Error splitting large semantic chunk: {e}")
            return [chunk]  # Return original chunk if splitting fails

    def add_texts(self, texts: List[str]) -> bool:
        """
        Add new texts to existing vector database.
        
        Args:
            texts: List of texts to add
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            if not self.vec_db:
                raise ValueError("No existing vector database. Create one first.")
                
            if not texts:
                raise ValueError("Texts list cannot be empty")

            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap
            )

            split_texts = []
            for text in texts:
                split_texts.extend(text_splitter.split_text(text))

            self.vec_db.add_texts(split_texts)
            self.logger.info(f"Added {len(split_texts)} text chunks to database")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to add texts: {e}")
            return False

    def save_vector_db(self) -> bool:
        """
        Save the vector database to disk.
        
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            if not self.vec_db:
                raise ValueError("No vector database to save")
                
            self.vec_db.save_local(
                folder_path=str(self.db_path),
                index_name=self.db_name
            )
            self.logger.info(f"Saved vector database '{self.db_name}' to '{self.db_path}'")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to save vector database '{self.db_name}' to '{self.db_path}': {e}")
            return False

    def load_vector_db(self) -> bool:
        """
        Load the vector database from disk.
        
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            db_file_path = self.db_path / f"{self.db_name}.faiss"
            if not db_file_path.exists():
                raise FileNotFoundError(f"Database file not found: {db_file_path}")
                
            self.vec_db = FAISS.load_local(
                folder_path=str(self.db_path),
                embeddings=self.embeddings,
                index_name=self.db_name,
                allow_dangerous_deserialization=True  # Be aware of security implications
            )
            self.logger.info(f"Loaded vector database '{self.db_name}' from '{self.db_path}'")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to load vector database '{self.db_name}' from '{self.db_path}': {e}")
            return False

    def similarity_search(self, query: str, k: int = 4) -> List[Document]:
        """
        Perform similarity search on the vector database.
        
        Args:
            query: Search query
            k: Number of results to return
            
        Returns:
            List of similar documents
        """
        try:
            if not self.vec_db:
                raise ValueError("No vector database loaded")
                
            if not query or not query.strip():
                raise ValueError("Query cannot be empty")
                
            return self.vec_db.similarity_search(query, k=k)
            
        except Exception as e:
            self.logger.error(f"Similarity search failed: {e}")
            return []

    def similarity_search_with_score(self, query: str, k: int = 4) -> List[tuple]:
        """
        Perform similarity search with scores.
        
        Args:
            query: Search query
            k: Number of results to return
            
        Returns:
            List of tuples (document, score)
        """
        try:
            if not self.vec_db:
                raise ValueError("No vector database loaded")
                
            if not query or not query.strip():
                raise ValueError("Query cannot be empty")
                
            return self.vec_db.similarity_search_with_score(query, k=k)
            
        except Exception as e:
            self.logger.error(f"Similarity search with score failed: {e}")
            return []

    def get_database_info(self) -> dict:
        """
        Get information about the current database.
        
        Returns:
            Dictionary with database information
        """
        info = {
            "db_name": self.db_name,
            "db_path": str(self.db_path),
            "chunk_size": self.chunk_size,
            "chunk_overlap": self.chunk_overlap,
            "embedding_model": self.embedding_model_name,
            "database_loaded": self.vec_db is not None
        }
        
        if self.vec_db:
            try:
                # Get approximate document count (this might not be exact for FAISS)
                info["approximate_doc_count"] = self.vec_db.index.ntotal
                
                # Get chunk type statistics if available
                chunk_stats = self._get_chunk_statistics()
                if chunk_stats:
                    info["chunk_statistics"] = chunk_stats
                    
            except:
                info["approximate_doc_count"] = "Unknown"
                
        return info

    def _get_chunk_statistics(self) -> dict:
        """
        Get statistics about chunk types in the database.
        
        Returns:
            Dictionary with chunk statistics
        """
        try:
            if not self.vec_db:
                return {}
            
            # Get all documents to analyze metadata
            all_docs = []
            try:
                # This is a workaround since FAISS doesn't directly expose all documents
                # We'll use similarity search with a generic query to get sample documents
                sample_docs = self.vec_db.similarity_search("", k=min(100, self.vec_db.index.ntotal))
                all_docs = sample_docs
            except:
                return {}
            
            stats = {
                "total_chunks": len(all_docs),
                "chunk_types": {},
                "pages_covered": set(),
                "sections_covered": set(),
                "avg_chunk_length": 0,
                "tables_count": 0,
                "paragraphs_count": 0,
                "headings_count": 0
            }
            
            total_length = 0
            
            for doc in all_docs:
                metadata = doc.metadata
                
                # Count chunk types
                chunk_type = metadata.get('chunk_type', 'unknown')
                stats["chunk_types"][chunk_type] = stats["chunk_types"].get(chunk_type, 0) + 1
                
                # Track pages and sections
                if metadata.get('page_number'):
                    stats["pages_covered"].add(metadata['page_number'])
                
                if metadata.get('section'):
                    stats["sections_covered"].add(metadata['section'])
                
                # Count specific types
                if chunk_type == 'table':
                    stats["tables_count"] += 1
                elif chunk_type == 'paragraph':
                    stats["paragraphs_count"] += 1
                elif chunk_type == 'heading':
                    stats["headings_count"] += 1
                
                # Calculate average length
                char_count = metadata.get('char_count', len(doc.page_content))
                total_length += char_count
            
            # Convert sets to counts
            stats["pages_covered"] = len(stats["pages_covered"])
            stats["sections_covered"] = len(stats["sections_covered"])
            
            # Calculate average
            if all_docs:
                stats["avg_chunk_length"] = total_length // len(all_docs)
            
            return stats
            
        except Exception as e:
            self.logger.error(f"Error getting chunk statistics: {e}")
            return {}

    def search_by_document_type(self, query: str, document_types: List[str] = None, k: int = 4) -> List[Document]:
        """
        Search for documents filtered by document intelligence types.
        
        Args:
            query: Search query
            document_types: List of document types to filter by ('paragraph', 'table', 'heading', etc.)
            k: Number of results to return
            
        Returns:
            List of filtered documents
        """
        try:
            if not self.vec_db:
                raise ValueError("No vector database loaded")
            
            # Get more results than needed for filtering
            search_k = min(k * 3, 50)  # Get 3x more results for filtering
            results = self.vec_db.similarity_search_with_score(query, k=search_k)
            
            if not document_types:
                return [doc for doc, score in results[:k]]
            
            # Filter by document types
            filtered_results = []
            for doc, score in results:
                chunk_type = doc.metadata.get('chunk_type', '').lower()
                if chunk_type in [dt.lower() for dt in document_types]:
                    filtered_results.append(doc)
                    if len(filtered_results) >= k:
                        break
            
            return filtered_results
            
        except Exception as e:
            self.logger.error(f"Search by document type failed: {e}")
            return []

    def search_by_page(self, query: str, page_numbers: List[int] = None, k: int = 4) -> List[Document]:
        """
        Search for documents filtered by page numbers.
        
        Args:
            query: Search query
            page_numbers: List of page numbers to filter by
            k: Number of results to return
            
        Returns:
            List of documents from specified pages
        """
        try:
            if not self.vec_db:
                raise ValueError("No vector database loaded")
            
            # Get more results for filtering
            search_k = min(k * 3, 50)
            results = self.vec_db.similarity_search_with_score(query, k=search_k)
            
            if not page_numbers:
                return [doc for doc, score in results[:k]]
            
            # Filter by page numbers
            filtered_results = []
            for doc, score in results:
                doc_page = doc.metadata.get('page_number', 0)
                if doc_page in page_numbers:
                    filtered_results.append(doc)
                    if len(filtered_results) >= k:
                        break
            
            return filtered_results
            
        except Exception as e:
            self.logger.error(f"Search by page failed: {e}")
            return []

    def _generate_embeddings(self) -> OllamaEmbeddings:
        """
        Generate embeddings model instance.
        
        Returns:
            OllamaEmbeddings instance
        """
        return OllamaEmbeddings(model=self.embedding_model_name)

    @staticmethod
    def load_existing_database(db_path: str, 
                             db_name: str,
                             embedding_model_name: str = "llama2",
                             chunk_size: int = 1000,
                             chunk_overlap: int = 200) -> Optional['FaissVectorDB']:
        """
        Static method to load an existing database.
        
        Args:
            db_path: Path to database directory
            db_name: Name of the database
            embedding_model_name: Name of embedding model
            chunk_size: Chunk size used in original database
            chunk_overlap: Chunk overlap used in original database
            
        Returns:
            FaissVectorDB instance or None if loading fails
        """
        instance = FaissVectorDB(
            db_name=db_name,
            db_path=db_path,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            embedding_model_name=embedding_model_name
        )
        
        if instance.load_vector_db():
            return instance
        return None

    def __str__(self) -> str:
        """String representation of the database."""
        return f"FaissVectorDB(name='{self.db_name}', path='{self.db_path}', loaded={self.vec_db is not None})"

    def __repr__(self) -> str:
        """Detailed string representation."""
        return (f"FaissVectorDB(db_name='{self.db_name}', db_path='{self.db_path}', "
                f"chunk_size={self.chunk_size}, chunk_overlap={self.chunk_overlap}, "
                f"embedding_model='{self.embedding_model_name}', loaded={self.vec_db is not None})")