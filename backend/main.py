# =========================================================================
# File: main.py
# Project: APH-IF Technology Framework
#          Advanced Parallel Hybrid - Intelligent Fusion System
# Author: Alexander Ricciardi
# Date: 2025-08-05
# File Path: backend/main.py
# =========================================================================

# --- Module Objective ---
# This module serves as the central orchestration layer for the APH-IF Backend API.
# It initializes the FastAPI application, configures middleware, defines request
# and response models, and exposes the core API endpoints. Its primary responsibility
# is to coordinate the Advanced Parallel HybridRAG processing pipeline, including
# parallel retrieval, intelligent fusion, and response generation.
# It also provides health check endpoints for system monitoring.
# -------------------------------------------------------------------------

"""
Central Orchestration Layer for APH-IF Backend API with Advanced Parallel HybridRAG Technology

Serves as the main FastAPI application providing intelligent query processing
through Advanced Parallel Hybrid processing. Coordinates parallel VectorRAG and GraphRAG
retrieval, intelligent context fusion, and specialized template application
for comprehensive knowledge graph queries.
"""

# =========================================================================
# Imports
# =========================================================================
import logging
import uuid
import time
import asyncio
from typing import Optional, Dict, Any
from datetime import datetime

from fastapi import FastAPI, HTTPException, Header
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

# Local imports
from .config import get_config
from .parallel_hybrid import get_parallel_engine, ParallelRetrievalResponse
from .context_fusion import get_fusion_engine
from .circuit_breaker import CircuitBreaker

# =========================================================================
# Global Variables
# =========================================================================
PARALLEL_HYBRIDRAG_AVAILABLE = False
active_sessions: Dict[str, Dict[str, Any]] = {}
startup_time = datetime.now()

# =========================================================================
# Pydantic Models
# =========================================================================
class ParallelHybridRAGRequest(BaseModel):
    """Request model for parallel HybridRAG processing"""
    query: str = Field(..., description="User query for processing")
    session_id: Optional[str] = Field(None, description="Session identifier")
    max_results: Optional[int] = Field(10, description="Maximum number of results")
    include_metadata: Optional[bool] = Field(True, description="Include result metadata")

class ParallelHybridRAGResponse(BaseModel):
    """Response model for parallel HybridRAG processing"""
    response: str = Field(..., description="Generated response")
    session_id: str = Field(..., description="Session identifier")
    processing_time: float = Field(..., description="Processing time in seconds")
    vector_results_count: int = Field(..., description="Number of VectorRAG search results")
    graph_results_count: int = Field(..., description="Number of GraphRAG search results")
    fusion_method: str = Field(..., description="Intelligent fusion method used")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Additional metadata")

class HealthResponse(BaseModel):
    """Health check response model"""
    status: str = Field(..., description="Service status")
    timestamp: datetime = Field(..., description="Health check timestamp")
    uptime_seconds: float = Field(..., description="Service uptime in seconds")
    version: str = Field("1.0.0", description="API version")
    components: Dict[str, str] = Field(..., description="Component health status")

# =========================================================================
# FastAPI Application Setup
# =========================================================================
app = FastAPI(
    title="APH-IF Backend API",
    description="Advanced Parallel Hybrid - Intelligent Fusion Backend Service",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =========================================================================
# Startup Event
# =========================================================================
@app.on_event("startup")
async def startup_event():
    """Initialize services on startup"""
    global PARALLEL_HYBRIDRAG_AVAILABLE

    logging.info("Starting APH-IF Backend API...")

    try:
        # Initialize configuration
        config = get_config()
        logging.info("Configuration loaded successfully")

        # Initialize parallel HybridRAG engine
        parallel_engine = get_parallel_engine()
        logging.info("Parallel HybridRAG engine initialized")

        # Initialize intelligent fusion engine
        fusion_engine = get_fusion_engine()
        logging.info("Intelligent fusion engine initialized")

        PARALLEL_HYBRIDRAG_AVAILABLE = True
        logging.info("APH-IF Backend API startup completed successfully")

    except Exception as e:
        logging.error(f"Startup failed: {str(e)}")
        PARALLEL_HYBRIDRAG_AVAILABLE = False

# =========================================================================
# API Endpoints
# =========================================================================
@app.get("/")
async def root():
    """Root endpoint providing service information"""
    return {
        "service": "APH-IF Backend API",
        "description": "Advanced Parallel HybridRAG - Intelligent Fusion System",
        "version": "dev 0.0.1",
        "status": "operational" if PARALLEL_HYBRIDRAG_AVAILABLE else "degraded",
        "endpoints": {
            "health": "/health",
            "parallel_hybrid_health": "/parallel_hybrid/health",
            "generate": "/generate_parallel_hybrid",
            "docs": "/docs"
        }
    }

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Basic health check endpoint"""
    uptime = (datetime.now() - startup_time).total_seconds()
    
    components = {
        "api": "healthy",
        "parallel_hybrid": "healthy" if PARALLEL_HYBRIDRAG_AVAILABLE else "unhealthy",
        "database": "unknown",  # Will be updated when database module is implemented
        "llm": "unknown"        # Will be updated when LLM module is implemented
    }
    
    overall_status = "healthy" if all(
        status in ["healthy", "unknown"] for status in components.values()
    ) else "unhealthy"
    
    return HealthResponse(
        status=overall_status,
        timestamp=datetime.now(),
        uptime_seconds=uptime,
        components=components
    )

@app.get("/parallel_hybrid/health")
async def parallel_hybrid_health():
    """Detailed health check for parallel HybridRAG components"""
    if not PARALLEL_HYBRIDRAG_AVAILABLE:
        raise HTTPException(status_code=503, detail="Parallel HybridRAG system unavailable")
    
    # TODO: Implement detailed component health checks
    return {
        "status": "healthy",
        "components": {
            "vector_search": "healthy",
            "graph_search": "healthy", 
            "context_fusion": "healthy",
            "template_engine": "healthy"
        },
        "timestamp": datetime.now()
    }

@app.post("/generate_parallel_hybrid", response_model=ParallelHybridRAGResponse)
async def generate_parallel_hybrid(
    request: ParallelHybridRAGRequest,
    x_session_id: Optional[str] = Header(None)
):
    """Main endpoint for parallel HybridRAG processing"""
    start_time = time.time()

    if not PARALLEL_HYBRIDRAG_AVAILABLE:
        raise HTTPException(
            status_code=503,
            detail="Parallel HybridRAG system is currently unavailable"
        )
    
    # Generate or use provided session ID
    session_id = request.session_id or x_session_id or str(uuid.uuid4())
    
    try:
        # Get parallel hybrid engine
        parallel_engine = await get_parallel_engine()

        # Process query using APH-IF technology
        result = await parallel_engine.process_query(
            query=request.query,
            max_vector_results=request.max_results
        )
        
        processing_time = time.time() - start_time

        # Store session data
        active_sessions[session_id] = {
            "last_query": request.query,
            "timestamp": datetime.now(),
            "processing_time": processing_time
        }

        return ParallelHybridRAGResponse(
            response=f"APH-IF processed query: '{request.query}' with {result.vector_results_count} VectorRAG and {result.graph_results_count} GraphRAG results",
            session_id=session_id,
            processing_time=result.processing_time,
            vector_results_count=result.vector_results_count,
            graph_results_count=result.graph_results_count,
            fusion_method="intelligent",
            metadata=result.metadata if request.include_metadata else None
        )
        
    except Exception as e:
        logging.error(f"Error processing request: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Processing error: {str(e)}")

# =========================================================================
# Error Handlers
# =========================================================================
@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    """Custom HTTP exception handler"""
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": exc.detail,
            "timestamp": datetime.now().isoformat(),
            "path": str(request.url)
        }
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
