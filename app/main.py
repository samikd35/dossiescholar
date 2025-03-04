from fastapi import FastAPI, Request, Form, Depends, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from typing import Optional, List
import os
from dotenv import load_dotenv

# Import our agents and models
from app.agents.summarization_agent import SummarizationAgent
from app.agents.literature_review_agent import LiteratureReviewAgent
from app.agents.deep_research_agent import DeepResearchAgent
from app.models.qdrant_store import get_qdrant_store

# Load environment variables
load_dotenv()

# Initialize FastAPI app
app = FastAPI(title="Ethiopian Research Paper Assistant")

# Mount static files
app.mount("/static", StaticFiles(directory="app/static"), name="static")
# Mount CSS and JS directly for browser path compatibility
app.mount("/css", StaticFiles(directory="app/static/css"), name="css")
app.mount("/js", StaticFiles(directory="app/static/js"), name="js")

# Set up Jinja2 templates
templates = Jinja2Templates(directory="app/templates")

# Initialize agents
def get_agents():
    qdrant_store = get_qdrant_store()
    summarization_agent = SummarizationAgent(vector_store=qdrant_store)
    literature_review_agent = LiteratureReviewAgent(vector_store=qdrant_store)
    deep_research_agent = DeepResearchAgent(vector_store=qdrant_store)
    
    return {
        "summarization": summarization_agent,
        "literature_review": literature_review_agent,
        "deep_research": deep_research_agent
    }

# Define request models
class QueryRequest(BaseModel):
    query: str
    agent_type: str
    detail_level: Optional[str] = "medium"
    max_results: Optional[int] = 5

class QueryResponse(BaseModel):
    result: str
    sources: List[dict]

# Routes
@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/api/query", response_model=QueryResponse)
async def query(query_request: QueryRequest, agents=Depends(get_agents)):
    try:
        # Select the appropriate agent based on the request
        if query_request.agent_type == "summarization":
            agent = agents["summarization"]
            result, sources = agent.process_query(
                query_request.query, 
                detail_level=query_request.detail_level,
                max_results=query_request.max_results
            )
        elif query_request.agent_type == "literature_review":
            agent = agents["literature_review"]
            result, sources = agent.process_query(
                query_request.query, 
                detail_level=query_request.detail_level,
                max_results=query_request.max_results
            )
        elif query_request.agent_type == "deep_research":
            agent = agents["deep_research"]
            result, sources = agent.process_query(
                query_request.query, 
                detail_level=query_request.detail_level,
                max_results=query_request.max_results
            )
        else:
            raise HTTPException(status_code=400, detail="Invalid agent type")
            
        return QueryResponse(result=result, sources=sources)
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/health")
async def health_check():
    return {"status": "healthy"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
