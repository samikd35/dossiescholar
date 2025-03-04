from abc import ABC, abstractmethod
from typing import Tuple, List, Dict, Any
from loguru import logger
import traceback

class BaseAgent(ABC):
    """
    Base class for all research paper agents.
    This abstract class defines the common interface for all specialized agents.
    """
    
    def __init__(self, vector_store):
        """
        Initialize the base agent with a vector store.
        
        Args:
            vector_store: The Qdrant vector store containing research papers
        """
        self.vector_store = vector_store
    
    @abstractmethod
    def process_query(self, query: str, **kwargs) -> Tuple[str, List[Dict[str, Any]]]:
        """
        Process a user query and return results.
        
        Args:
            query: The user's research query
            **kwargs: Additional parameters like detail_level, max_results, etc.
            
        Returns:
            A tuple containing:
            - The processed result (summary, literature review, or deep analysis)
            - A list of source documents with metadata
        """
        pass
    
    def _retrieve_documents(self, query: str, max_results: int = 5) -> List[Dict[str, Any]]:
        """
        Retrieve relevant documents from the vector store.
        
        Args:
            query: The search query
            max_results: Maximum number of results to retrieve
            
        Returns:
            List of documents with their metadata
        """
        try:
            logger.info(f"Retrieving documents for query: {query}, max_results: {max_results}")
            documents = self.vector_store.similarity_search_with_relevance_scores(
                query, 
                k=max_results
            )
            
            # Format the documents with metadata
            formatted_docs = []
            for doc, score in documents:
                formatted_docs.append({
                    "content": doc.page_content,
                    "metadata": doc.metadata,
                    "relevance_score": round(score, 3)
                })
            
            logger.info(f"Retrieved {len(formatted_docs)} documents")
            return formatted_docs
        except Exception as e:
            logger.error(f"Error retrieving documents: {str(e)}")
            logger.error(traceback.format_exc())
            # Return an empty list in case of error
            return []
