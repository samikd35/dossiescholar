import os
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Qdrant
from qdrant_client import QdrantClient
from loguru import logger
import sys

# Configure logger
logger.remove()
logger.add(sys.stderr, level="INFO")
logger.add("app.log", rotation="10 MB", level="INFO")

# Load environment variables
load_dotenv()

def validate_env_vars(*env_vars):
    """Validate that required environment variables are set."""
    missing_vars = [var for var in env_vars if not os.getenv(var)]
    if missing_vars:
        error_msg = f"Missing required environment variables: {', '.join(missing_vars)}"
        logger.error(error_msg)
        raise ValueError(error_msg)
    return True

def get_embeddings():
    """Initialize and return the embedding model."""
    try:
        validate_env_vars("OPENAI_API_KEY")
        return OpenAIEmbeddings(
            openai_api_key=os.getenv("OPENAI_API_KEY")
        )
    except Exception as e:
        logger.error(f"Error initializing embeddings: {str(e)}")
        raise

def get_qdrant_store():
    """
    Initialize and return the Qdrant vector store from an existing collection.
    This ensures all retrievals are performed on the preloaded Ethiopian research papers.
    """
    try:
        validate_env_vars("QDRANT_URL", "QDRANT_API_KEY", "QDRANT_COLLECTION")
        
        url = os.getenv("QDRANT_URL")
        api_key = os.getenv("QDRANT_API_KEY")
        collection_name = os.getenv("QDRANT_COLLECTION")
        
        logger.info(f"Connecting to Qdrant at {url}, collection: {collection_name}")
        
        client = QdrantClient(url=url, api_key=api_key)
        
        # Check if collection exists
        collections = client.get_collections().collections
        collection_names = [collection.name for collection in collections]
        
        if collection_name not in collection_names:
            logger.error(f"Collection {collection_name} does not exist")
            raise ValueError(f"Collection {collection_name} does not exist in Qdrant")
            
        embeddings = get_embeddings()
        
        return Qdrant(
            client=client,
            collection_name=collection_name,
            embeddings=embeddings,
        )
    except Exception as e:
        logger.error(f"Error initializing Qdrant store: {str(e)}")
        raise
