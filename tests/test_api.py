import os
import sys
import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock

# Add parent directory to path to import app
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.main import app

client = TestClient(app)

def test_health_check():
    """Test the health check endpoint"""
    with patch('app.main.get_qdrant_store') as mock_get_qdrant:
        # Mock the Qdrant store
        mock_get_qdrant.return_value = MagicMock()
        
        # Make the request
        response = client.get("/api/health")
        
        # Check the response
        assert response.status_code == 200
        assert response.json()["status"] == "healthy"

def test_home_page():
    """Test the home page renders correctly"""
    response = client.get("/")
    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]

def test_query_invalid_agent():
    """Test that invalid agent types are rejected"""
    # Define test request
    test_request = {
        "query": "Test query",
        "agent_type": "invalid_agent_type",
        "detail_level": "medium",
        "max_results": 5
    }
    
    # Make the request
    response = client.post("/api/query", json=test_request)
    
    # Check the response
    assert response.status_code == 400
    assert "Invalid agent type" in response.json()["detail"]

@pytest.mark.parametrize(
    "agent_type", ["summarization", "literature_review", "deep_research"]
)
def test_query_valid_agent(agent_type):
    """Test that valid agent types are accepted"""
    # Define test request
    test_request = {
        "query": "Test query",
        "agent_type": agent_type,
        "detail_level": "medium",
        "max_results": 5
    }
    
    # Mock the agents dictionary and process_query method
    mock_agent = MagicMock()
    mock_agent.process_query.return_value = ("Test result", [{"source": "test"}])
    
    # Patch the get_agents function
    with patch('app.main.get_agents', return_value={
        "summarization": mock_agent,
        "literature_review": mock_agent,
        "deep_research": mock_agent
    }):
        # Make the request
        response = client.post("/api/query", json=test_request)
        
        # Check the response
        assert response.status_code == 200
        assert response.json()["result"] == "Test result"
        assert len(response.json()["sources"]) == 1
        assert "processing_time" in response.json()
