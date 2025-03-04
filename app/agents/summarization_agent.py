from typing import Tuple, List, Dict, Any, Optional
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain_core.messages import HumanMessage, AIMessage
import os
from langgraph.graph import StateGraph
from langgraph.graph import END
import json
from typing_extensions import TypedDict
from loguru import logger
import traceback

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
        try:
            # Set a longer timeout for the API call
            self.llm = ChatOpenAI(
                model_name="gpt-4o-mini", # Fallback to standard GPT-4 instead of turbo
                temperature=0.0,
                openai_api_key=os.getenv("OPENAI_API_KEY"),
                request_timeout=120  # Increase timeout to 120 seconds
            )
            logger.info("Successfully initialized ChatOpenAI")
        except Exception as e:
            logger.error(f"Error initializing ChatOpenAI: {str(e)}")
            logger.error(traceback.format_exc())
            raise
        
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
        
        If no relevant papers are provided, please explain that there isn't enough information available 
        on the topic and suggest related areas that might be researched instead.
        """
    
    def _create_workflow(self):
        """Create the LangGraph workflow for summarization."""
        # Create a new graph
        workflow = StateGraph(SummarizationState)
        
        # Define the summarization node
        def summarize(state: SummarizationState) -> dict:
            try:
                logger.info(f"Summarizing query: {state['query']}, detail level: {state['detail_level']}")
                prompt = self.summary_prompt.format(
                    query=state["query"],
                    documents=state["documents"],
                    detail_level=state["detail_level"]
                )
                
                messages = [HumanMessage(content=prompt)]
                logger.info("Sending request to language model")
                response = self.llm.invoke(messages)
                
                # Return updated state with the summary result
                logger.info("Summary generated successfully")
                return {"result": response.content}
            except Exception as e:
                error_msg = f"Error generating summary: {str(e)}"
                logger.error(error_msg)
                logger.error(traceback.format_exc())
                return {"result": f"An error occurred while generating the summary: {str(e)}"}
        
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
        try:
            logger.info(f"Processing query: '{query}' with detail level: {detail_level}, max_results: {max_results}")
            
            # Retrieve relevant documents
            documents = self._retrieve_documents(query, max_results)
            
            # Validate detail level
            if detail_level not in ["brief", "medium", "detailed"]:
                logger.warning(f"Invalid detail level: {detail_level}, defaulting to 'medium'")
                detail_level = "medium"
            
            # If no documents found, return a message
            if not documents:
                logger.warning(f"No documents found for query: {query}")
                return "No relevant Ethiopian research papers were found for this query. Please try a different query or topic related to Ethiopian research.", []
            
            # Format documents for the prompt
            formatted_docs = ""
            for i, doc in enumerate(documents):
                formatted_docs += f"PAPER {i+1}:\n"
                formatted_docs += f"Title: {doc['metadata'].get('title', 'Unknown')}\n"
                formatted_docs += f"Authors: {doc['metadata'].get('authors', 'Unknown')}\n"
                formatted_docs += f"Year: {doc['metadata'].get('year', 'Unknown')}\n"
                formatted_docs += f"Content: {doc['content'][:1000]}...\n\n"
            
            logger.info(f"Formatted {len(documents)} documents for the query")
            
            # Initialize the workflow state
            initial_state = {
                "query": query,
                "documents": formatted_docs,
                "detail_level": detail_level,
                "result": None
            }
            
            # Execute the workflow
            logger.info("Executing the LangGraph workflow")
            result = self.workflow.invoke(initial_state)
            
            # Extract and return the summary and source documents
            logger.info("Query processed successfully")
            return result["result"], documents
        except Exception as e:
            error_msg = f"Error processing query: {str(e)}"
            logger.error(error_msg)
            logger.error(traceback.format_exc())
            return f"An error occurred while processing your query: {str(e)}", []
