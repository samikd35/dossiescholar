from fastapi import FastAPI, Request, Form, Depends, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
import os
from dotenv import load_dotenv
from loguru import logger
import sys
import time
import traceback
from functools import lru_cache
import re

# Import our agents and models
from app.agents.summarization_agent import SummarizationAgent
from app.agents.literature_review_agent import LiteratureReviewAgent
from app.agents.deep_research_agent import DeepResearchAgent
from app.models.qdrant_store import get_qdrant_store, validate_env_vars

# Configure logger
logger.remove()
logger.add(sys.stderr, level="INFO")
logger.add("app.log", rotation="10 MB", level="INFO")

# Load environment variables
load_dotenv()

# Function to sanitize and validate the OpenAI API key
def sanitize_openai_api_key():
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY is missing")
    
    # Remove any 'sk-proj-' prefix if present
    if api_key.startswith("sk-proj-"):
        sanitized_key = api_key.replace("sk-proj-", "sk-", 1)
        logger.info("Sanitized OpenAI API key format")
        os.environ["OPENAI_API_KEY"] = sanitized_key

# Validate required environment variables
try:
    validate_env_vars("OPENAI_API_KEY", "QDRANT_URL", "QDRANT_API_KEY", "QDRANT_COLLECTION")
    sanitize_openai_api_key()
except ValueError as e:
    logger.error(f"Environment validation failed: {str(e)}")
    raise

# Initialize FastAPI app
app = FastAPI(
    title="Ethiopian Research Paper Assistant",
    description="AI-powered assistant for Ethiopian academic research papers",
    version="1.0.0"
)

# Mount static files
app.mount("/static", StaticFiles(directory="app/static"), name="static")
# Mount CSS and JS directly for browser path compatibility
app.mount("/css", StaticFiles(directory="app/static/css"), name="css")
app.mount("/js", StaticFiles(directory="app/static/js"), name="js")

# Set up Jinja2 templates
templates = Jinja2Templates(directory="app/templates")

# Initialize agents - using lru_cache to avoid reinitializing agents on every request
@lru_cache()
def get_agents():
    """
    Initialize and return agent instances.
    Uses lru_cache for performance optimization.
    """
    try:
        logger.info("Initializing agents...")
        start_time = time.time()
        
        # Re-sanitize the OpenAI API key to ensure it's in the correct format
        sanitize_openai_api_key()
        
        qdrant_store = get_qdrant_store()
        
        # Initialize agents one by one with individual error handling
        agents = {}
        
        try:
            logger.info("Initializing summarization agent...")
            summarization_agent = SummarizationAgent(vector_store=qdrant_store)
            agents["summarization"] = summarization_agent
            logger.info("Summarization agent initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing summarization agent: {str(e)}")
            logger.error(traceback.format_exc())
        
        try:
            logger.info("Initializing literature review agent...")
            literature_review_agent = LiteratureReviewAgent(vector_store=qdrant_store)
            agents["literature_review"] = literature_review_agent
            logger.info("Literature review agent initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing literature review agent: {str(e)}")
            logger.error(traceback.format_exc())
        
        try:
            logger.info("Initializing deep research agent...")
            deep_research_agent = DeepResearchAgent(vector_store=qdrant_store)
            agents["deep_research"] = deep_research_agent
            logger.info("Deep research agent initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing deep research agent: {str(e)}")
            logger.error(traceback.format_exc())
        
        logger.info(f"Agents initialized successfully in {time.time() - start_time:.2f} seconds")
        
        if not agents:
            raise ValueError("No agents were successfully initialized")
        
        return agents
    except Exception as e:
        logger.error(f"Fatal error initializing agents: {str(e)}")
        logger.error(traceback.format_exc())
        raise

# Define request models
class QueryRequest(BaseModel):
    query: str = Field(..., description="The research query to process")
    agent_type: str = Field(..., description="Type of agent to use: summarization, literature_review, or deep_research")
    detail_level: Optional[str] = Field("medium", description="Level of detail: low, medium, high")
    max_results: Optional[int] = Field(5, description="Maximum number of results to return", ge=1, le=20)
    
    class Config:
        schema_extra = {
            "example": {
                "query": "What are the recent developments in Ethiopian agriculture?",
                "agent_type": "summarization",
                "detail_level": "medium",
                "max_results": 5
            }
        }

class QueryResponse(BaseModel):
    result: str
    sources: List[dict]
    processing_time: float

# Middleware for request logging
@app.middleware("http")
async def log_requests(request: Request, call_next):
    start_time = time.time()
    
    # Get client IP and requested path
    client_host = request.client.host if request.client else "unknown"
    path = request.url.path
    
    logger.info(f"Request received - IP: {client_host}, Path: {path}")
    
    response = await call_next(request)
    
    # Log request completion
    process_time = time.time() - start_time
    logger.info(f"Request completed - IP: {client_host}, Path: {path}, Time: {process_time:.2f}s, Status: {response.status_code}")
    
    return response

# Routes
@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    """Render the home page"""
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/api/query", response_model=QueryResponse)
async def query(query_request: QueryRequest, agents=Depends(get_agents)):
    """
    Process a research query using the specified agent.
    
    Parameters:
    - query_request: The query request containing the query, agent type, and other parameters
    - agents: The initialized agents, injected through dependency
    
    Returns:
    - QueryResponse: The query result, sources, and processing time
    """
    start_time = time.time()
    logger.info(f"Processing query: '{query_request.query}' with agent: {query_request.agent_type}")
    
    try:
        # Validate agent type
        valid_agent_types = ["summarization", "literature_review", "deep_research"]
        if query_request.agent_type not in valid_agent_types:
            error_msg = f"Invalid agent type: {query_request.agent_type}. Valid types are: {', '.join(valid_agent_types)}"
            logger.error(error_msg)
            raise HTTPException(status_code=400, detail=error_msg)
        
        # Validate detail level
        valid_detail_levels = ["low", "medium", "high"]
        if query_request.detail_level not in valid_detail_levels:
            logger.warning(f"Invalid detail level: {query_request.detail_level}, defaulting to 'medium'")
            query_request.detail_level = "medium"
            
        # Check if the requested agent is available
        if query_request.agent_type not in agents:
            available_agents = list(agents.keys())
            error_msg = f"Agent type '{query_request.agent_type}' is not available due to initialization errors. Available agents: {', '.join(available_agents) if available_agents else 'None'}"
            logger.error(error_msg)
            
            if not available_agents:
                raise HTTPException(status_code=500, detail="No agents are currently available. Please check the server logs.")
            
            # Try to use an alternative agent if available
            if "summarization" in agents:
                logger.info(f"Falling back to 'summarization' agent instead of '{query_request.agent_type}'")
                query_request.agent_type = "summarization"
            else:
                query_request.agent_type = available_agents[0]
                logger.info(f"Falling back to '{query_request.agent_type}' agent")
        
        # Select the appropriate agent based on the request
        agent = agents[query_request.agent_type]
        
        # Process the query with timeout handling
        try:
            result, sources = agent.process_query(
                query_request.query, 
                detail_level=query_request.detail_level,
                max_results=query_request.max_results
            )
        except Exception as e:
            logger.error(f"Error during query processing: {str(e)}")
            logger.error(traceback.format_exc())
            
            # Provide a more user-friendly error message
            result = "I apologize, but I encountered an issue while processing your query. This might be due to API timeout or connection issues. Please try again with a simpler query or contact support if the problem persists."
            sources = []  # Empty sources list since we couldn't retrieve any
        
        processing_time = time.time() - start_time
        logger.info(f"Query processed in {processing_time:.2f} seconds")
            
        return QueryResponse(
            result=result, 
            sources=sources, 
            processing_time=processing_time
        )
    
    except HTTPException:
        # Re-raise HTTP exceptions without modification
        raise
    except Exception as e:
        # Log the error and return a 500 response
        error_msg = f"Error processing query: {str(e)}"
        logger.error(error_msg)
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=error_msg)

@app.get("/api/health")
async def health_check():
    """Health check endpoint for monitoring"""
    try:
        # Verify database connection
        _ = get_qdrant_store()
        return {
            "status": "healthy",
            "timestamp": time.time(),
            "details": {
                "database": "connected"
            }
        }
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        return {
            "status": "unhealthy",
            "timestamp": time.time(),
            "error": str(e)
        }

if __name__ == "__main__":
    import uvicorn
    # Get port from environment or use default
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(
        "app.main:app", 
        host="0.0.0.0", 
        port=port, 
        reload=True if os.getenv("ENVIRONMENT") == "development" else False,
        log_level="info"
    )
