from typing import Tuple, List, Dict, Any, Optional
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain_core.messages import HumanMessage, AIMessage
import os
from langgraph.graph import StateGraph
from langgraph.graph import END
import json
from typing_extensions import TypedDict

from app.agents.base_agent import BaseAgent

# Define the state schema for our agent
class SummarizationState(TypedDict):
    query: str
    documents: str
    detail_level: str
    result: Optional[str]

class SummarizationAgent(BaseAgent):
    """
    Agent for retrieving relevant research papers and generating concise summaries.
    """
    
    def __init__(self, vector_store):
        """
        Initialize the summarization agent.
        
        Args:
            vector_store: The Qdrant vector store containing research papers
        """
        super().__init__(vector_store)
        self.llm = ChatOpenAI(
            model_name="gpt-4-turbo",
            temperature=0.2,
            openai_api_key=os.getenv("OPENAI_API_KEY")
        )
        
        # Create the summarization prompt template
        self.summary_prompt = self._get_template()
        
        # Create the LangGraph workflow
        self.workflow = self._create_workflow()
    
    def _get_template(self):
        return """
        You are an expert Ethiopian research paper summarizer. Your task is to create a clear, concise summary 
        based on the following research papers related to the query: "{query}"
        
        PAPERS:
        {documents}
        
        DETAIL LEVEL: {detail_level} (brief, medium, or detailed)
        
        Your summary should:
        - Capture the key findings and conclusions from the papers
        - Highlight important methodologies and results
        - Maintain academic tone and precision
        - Match the requested detail level
        - Focus specifically on the query topic
        - Include section headers for better readability
        - Add document references
        
        SUMMARY:
        """
    
    def _create_workflow(self):
        # Define the workflow as a state graph
        workflow = StateGraph(SummarizationState)
        
        # Define the summarization node
        def summarize(state: SummarizationState) -> dict:
            prompt = self.summary_prompt.format(
                query=state["query"],
                documents=state["documents"],
                detail_level=state["detail_level"]
            )
            
            messages = [HumanMessage(content=prompt)]
            response = self.llm.invoke(messages)
            
            # Return updated state with the summary result
            return {"result": response.content}
        
        # Add the node to the graph
        workflow.add_node("summarize", summarize)
        
        # Define the flow: start -> summarize -> end
        workflow.set_entry_point("summarize")
        workflow.add_edge("summarize", END)
        
        # Compile the workflow
        return workflow.compile()
    
    def process_query(
        self, 
        query: str, 
        detail_level: Optional[str] = "medium", 
        max_results: Optional[int] = 5
    ) -> Tuple[str, List[Dict[str, Any]]]:
        """
        Process a summarization query.
        
        Args:
            query: The user's research query
            detail_level: Level of detail for the summary (brief, medium, detailed)
            max_results: Maximum number of papers to include
            
        Returns:
            A tuple containing:
            - The generated summary
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
        
        # Extract and return the summary and source documents
        return result["result"], documents
