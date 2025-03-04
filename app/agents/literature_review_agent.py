from typing import Tuple, List, Dict, Any, Optional
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain_core.messages import HumanMessage, AIMessage
from langgraph.graph import StateGraph
from langgraph.graph import END
import os
import json
from typing_extensions import TypedDict
from loguru import logger
import traceback

from app.agents.base_agent import BaseAgent

# Define the state schema for our agent
class LiteratureReviewState(TypedDict):
    query: str
    documents: str
    detail_level: str
    result: Optional[str]

class LiteratureReviewAgent(BaseAgent):
    """
    Agent for creating comprehensive literature reviews based on Ethiopian research papers.
    """
    
    def __init__(self, vector_store):
        """
        Initialize the literature review agent.
        
        Args:
            vector_store: The Qdrant vector store containing research papers
        """
        super().__init__(vector_store)
        try:
            # Set a longer timeout for the API call
            self.llm = ChatOpenAI(
                model_name="gpt-4o-mini", # Fallback to standard GPT-4 instead of turbo
                temperature=0.0,
                openai_api_key=os.getenv("OPENAI_API_KEY"),
                request_timeout=120  # Increase timeout to 120 seconds
            )
            logger.info("Successfully initialized ChatOpenAI for literature review agent")
        except Exception as e:
            logger.error(f"Error initializing ChatOpenAI for literature review: {str(e)}")
            logger.error(traceback.format_exc())
            raise
        
        # Create the literature review prompt
        self.review_prompt = self._get_template()
        
        # Create the LangGraph workflow
        self.workflow = self._create_workflow()
    
    def _get_template(self):
        return """
        You are an expert Ethiopian researcher creating a literature review. 
        Based on the following papers related to the query: "{query}", construct a comprehensive
        literature review. 
        
        PAPERS:
        {documents}
        
        DETAIL LEVEL: {detail_level} (brief, medium, detailed)
        
        Your literature review should:
        - Group papers by themes, approaches, and findings
        - Compare and contrast methodologies
        - Identify gaps and future research directions
        - Maintain academic tone and precision
        - Match the requested detail level
        - Use proper citations and references
        - Include section headers for better organization
        - Add document references 
        
        LITERATURE REVIEW:
        """
    
    def _create_workflow(self):
        # Define the workflow as a state graph
        workflow = StateGraph(LiteratureReviewState)
        
        # Define the literature review node
        def create_review(state: LiteratureReviewState) -> dict:
            prompt = self.review_prompt.format(
                query=state["query"],
                documents=state["documents"],
                detail_level=state["detail_level"]
            )
            
            messages = [HumanMessage(content=prompt)]
            response = self.llm.invoke(messages)
            
            # Return updated state with the review result
            return {"result": response.content}
        
        # Add the node to the graph
        workflow.add_node("create_review", create_review)
        
        # Define the flow: start -> create_review -> end
        workflow.set_entry_point("create_review")
        workflow.add_edge("create_review", END)
        
        # Compile the workflow
        return workflow.compile()
    
    def process_query(
        self, 
        query: str, 
        detail_level: Optional[str] = "medium", 
        max_results: Optional[int] = 8
    ) -> Tuple[str, List[Dict[str, Any]]]:
        """
        Process a literature review query.
        
        Args:
            query: The user's research query
            detail_level: Level of detail for the review (brief, medium, detailed)
            max_results: Maximum number of papers to include
            
        Returns:
            A tuple containing:
            - The generated literature review
            - A list of source documents with metadata
        """
        # Retrieve relevant documents
        documents = self._retrieve_documents(query, max_results)
        
        # Validate detail level
        if detail_level not in ["brief", "medium", "detailed"]:
            detail_level = "medium"
        
        # Format documents for the prompt
        formatted_docs = ""
        for i, doc in enumerate(documents):
            formatted_docs += f"PAPER {i+1}:\n"
            formatted_docs += f"Title: {doc['metadata'].get('title', 'Unknown')}\n"
            formatted_docs += f"Authors: {doc['metadata'].get('authors', 'Unknown')}\n"
            formatted_docs += f"Year: {doc['metadata'].get('year', 'Unknown')}\n"
            formatted_docs += f"Abstract: {doc['metadata'].get('abstract', 'Not available')}\n"
            formatted_docs += f"Content: {doc['content'][:800]}...\n\n"
        
        # Initialize the workflow state
        initial_state = {
            "query": query,
            "documents": formatted_docs,
            "detail_level": detail_level,
            "result": None
        }
        
        # Execute the workflow
        result = self.workflow.invoke(initial_state)
        
        # Return the literature review and source documents
        return result["result"], documents
