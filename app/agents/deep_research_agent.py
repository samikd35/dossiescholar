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
class DeepResearchState(TypedDict):
    query: str
    documents: str
    detail_level: str
    result: Optional[str]

class DeepResearchAgent(BaseAgent):
    """
    Agent for conducting in-depth research and analysis of Ethiopian academic papers.
    """
    
    def __init__(self, vector_store):
        """
        Initialize the deep research agent.
        
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
            logger.info("Successfully initialized ChatOpenAI for deep research agent")
        except Exception as e:
            logger.error(f"Error initializing ChatOpenAI for deep research: {str(e)}")
            logger.error(traceback.format_exc())
            raise
        
        # Create the deep research prompt
        self.research_prompt = self._get_template()
        
        # Create the LangGraph workflow
        self.workflow = self._create_workflow()
    
    def _get_template(self):
        return """
        You are an expert Ethiopian researcher conducting in-depth research.
        Based on the following papers, provide a comprehensive answer to the research question: "{query}"
        
        PAPERS:
        {documents}
        
        DETAIL LEVEL: {detail_level} (brief, medium, detailed)
        
        Your research response should:
        - Provide a comprehensive answer to the research question
        - Synthesize information across multiple papers
        - Evaluate evidence quality and research methods
        - Present multiple perspectives and interpretations
        - Identify limitations and areas for further research
        - Maintain academic tone and precision
        - Match the requested detail level
        - Include section headers for better organization
        - Add document references
        
        RESEARCH RESPONSE:
        """
    
    def _create_workflow(self):
        # Define the workflow as a state graph
        workflow = StateGraph(DeepResearchState)
        
        # Define the research node
        def conduct_research(state: DeepResearchState) -> dict:
            prompt = self.research_prompt.format(
                query=state["query"],
                documents=state["documents"],
                detail_level=state["detail_level"]
            )
            
            messages = [HumanMessage(content=prompt)]
            response = self.llm.invoke(messages)
            
            # Return updated state with the research result
            return {"result": response.content}
        
        # Add the node to the graph
        workflow.add_node("conduct_research", conduct_research)
        
        # Define the flow: start -> conduct_research -> end
        workflow.set_entry_point("conduct_research")
        workflow.add_edge("conduct_research", END)
        
        # Compile the workflow
        return workflow.compile()
    
    def process_query(
        self, 
        query: str, 
        detail_level: Optional[str] = "medium", 
        max_results: Optional[int] = 10
    ) -> Tuple[str, List[Dict[str, Any]]]:
        """
        Process a deep research query.
        
        Args:
            query: The user's research question
            detail_level: Level of detail for the response (brief, medium, detailed)
            max_results: Maximum number of papers to include
            
        Returns:
            A tuple containing:
            - The generated research response
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
            formatted_docs += f"Journal: {doc['metadata'].get('journal', 'Unknown')}\n"
            formatted_docs += f"Content: {doc['content'][:1000]}...\n\n"
        
        # Initialize the workflow state
        initial_state = {
            "query": query,
            "documents": formatted_docs,
            "detail_level": detail_level,
            "result": None
        }
        
        # Execute the workflow
        result = self.workflow.invoke(initial_state)
        
        # Return the research response and source documents
        return result["result"], documents
