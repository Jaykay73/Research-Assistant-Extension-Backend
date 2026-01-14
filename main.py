"""
AI Research Paper Helper - FastAPI Backend
Main application entry point with CORS, routers, and health checks.
"""

from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import logging

from config import settings
from routers import summarize, equations, keypoints, rag
from ml.models import ModelManager

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler for startup/shutdown."""
    # Startup
    logger.info("Starting AI Research Paper Helper Backend...")
    logger.info(f"API Mode: {settings.api_mode}")
    logger.info(f"Model cache directory: {settings.model_cache_dir}")
    
    # Pre-load models in background (non-blocking)
    model_manager = ModelManager.get_instance()
    if settings.api_mode in ['local', 'hybrid']:
        logger.info("Pre-loading embedding model...")
        await model_manager.load_embedding_model()
        logger.info("Embedding model loaded successfully")
    
    yield
    
    # Shutdown
    logger.info("Shutting down AI Research Paper Helper Backend...")
    await model_manager.cleanup()


# Create FastAPI application
app = FastAPI(
    title="AI Research Paper Helper API",
    description="ML-powered API for research paper analysis, summarization, and Q&A",
    version="1.0.0",
    lifespan=lifespan
)

# Configure CORS for browser extension
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(summarize.router, tags=["Summarization"])
app.include_router(equations.router, tags=["Equations"])
app.include_router(keypoints.router, tags=["Key Points"])
app.include_router(rag.router, prefix="/rag", tags=["RAG"])


@app.get("/health")
async def health_check():
    """Health check endpoint for the browser extension."""
    model_manager = ModelManager.get_instance()
    
    return {
        "status": "healthy",
        "api_mode": settings.api_mode,
        "models_loaded": {
            "embeddings": model_manager.embedding_model is not None,
            "summarizer": model_manager.summarizer is not None
        },
        "version": "1.0.0"
    }


@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "name": "AI Research Paper Helper API",
        "version": "1.0.0",
        "docs": "/docs",
        "endpoints": [
            {"path": "/health", "method": "GET", "description": "Health check"},
            {"path": "/summarize", "method": "POST", "description": "Generate summaries"},
            {"path": "/explain-equations", "method": "POST", "description": "Explain equations"},
            {"path": "/extract-key-points", "method": "POST", "description": "Extract key points"},
            {"path": "/rag/index", "method": "POST", "description": "Index paper for RAG"},
            {"path": "/rag/query", "method": "POST", "description": "Query with RAG"}
        ]
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host=settings.host,
        port=settings.port,
        reload=settings.debug
    )
