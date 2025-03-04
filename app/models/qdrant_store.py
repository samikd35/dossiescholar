import os
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Qdrant
from qdrant_client import QdrantClient

# Load environment variables
load_dotenv()

def get_embeddings():
    """Initialize and return the embedding model."""
    return OpenAIEmbeddings(
        openai_api_key=os.getenv("OPENAI_API_KEY")
    )

def get_qdrant_store():
    """
    Initialize and return the Qdrant vector store from an existing collection.
    This ensures all retrievals are performed on the preloaded Ethiopian research papers.
    """
    url = os.getenv("QDRANT_URL", "https://12bbc0a4-833b-4e01-880e-fa75e8e962f6.europe-west3-0.gcp.cloud.qdrant.io:6333")
    api_key = os.getenv("QDRANT_API_KEY")
    collection_name = os.getenv("QDRANT_COLLECTION", "AASTU")
    
    # Initialize Qdrant client
    client = QdrantClient(
        url=url,
        api_key=api_key,
    )
    
    # Initialize and return the vector store
    return Qdrant(
        client=client,
        collection_name=collection_name,
        embeddings=get_embeddings(),
    )
