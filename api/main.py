"""
FastAPI Backend for SHL Assessment Recommendation System
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional, List, Dict
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.recommender import RecommendationEngine

# Initialize FastAPI app
app = FastAPI(
    title="SHL Assessment Recommendation API",
    description="Intelligent recommendation system for SHL talent assessments",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, restrict to specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize recommendation engine
try:
    recommender = RecommendationEngine()
    print("✓ Recommendation engine initialized successfully")
except Exception as e:
    print(f"✗ Error initializing recommendation engine: {e}")
    recommender = None


# Pydantic models
class RecommendationRequest(BaseModel):
    """Request model for recommendation endpoint"""
    query: Optional[str] = Field(None, description="Natural language query for assessment recommendations")
    jd_text: Optional[str] = Field(None, description="Job description text")
    jd_url: Optional[str] = Field(None, description="Job description URL")
    top_k: int = Field(10, ge=1, le=20, description="Number of recommendations to return (1-20)")
    use_llm: bool = Field(False, description="Whether to use LLM for query enhancement")


class Assessment(BaseModel):
    """Assessment model for response"""
    assessment_name: str
    url: str
    description: str
    duration_minutes: Optional[int]
    test_type: str
    adaptive_support: str
    remote_support: str
    relevance_score: float


class RecommendationResponse(BaseModel):
    """Response model for recommendation endpoint"""
    query: str
    recommendations: List[Assessment]
    total_results: int


# API Endpoints
@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "status": "ok",
        "message": "SHL Assessment Recommendation API is running",
        "version": "1.0.0",
        "endpoints": {
            "health": "GET /",
            "recommend": "POST /recommend",
            "docs": "GET /docs"
        }
    }


@app.get("/health")
async def health():
    """Detailed health check"""
    if recommender is None:
        raise HTTPException(status_code=503, detail="Recommendation engine not initialized")
    
    return {
        "status": "healthy",
        "recommendation_engine": "ready",
        "database_loaded": hasattr(recommender.retriever, 'assessments'),
        "total_assessments": len(recommender.retriever.assessments) if hasattr(recommender.retriever, 'assessments') else 0
    }


@app.post("/recommend", response_model=RecommendationResponse)
async def recommend(request: RecommendationRequest):
    """
    Get assessment recommendations based on query, JD text, or JD URL
    
    Args:
        request: Recommendation request containing query/jd_text/jd_url
        
    Returns:
        List of recommended assessments with scores
    """
    if recommender is None:
        raise HTTPException(status_code=503, detail="Recommendation engine not initialized")
    
    # Validate that at least one input is provided
    if not any([request.query, request.jd_text, request.jd_url]):
        raise HTTPException(
            status_code=400,
            detail="At least one of 'query', 'jd_text', or 'jd_url' must be provided"
        )
    
    try:
        # Get recommendations
        result = recommender.recommend(
            query=request.query,
            jd_text=request.jd_text,
            jd_url=request.jd_url,
            top_k=request.top_k,
            use_llm=request.use_llm
        )
        
        # Check for errors
        if 'error' in result:
            raise HTTPException(status_code=400, detail=result['error'])
        
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@app.get("/assessments/count")
async def get_assessment_count():
    """Get total number of assessments in the database"""
    if recommender is None:
        raise HTTPException(status_code=503, detail="Recommendation engine not initialized")
    
    return {
        "total_assessments": len(recommender.retriever.assessments)
    }


@app.get("/assessments/sample")
async def get_sample_assessments(n: int = 5):
    """Get a sample of assessments from the database"""
    if recommender is None:
        raise HTTPException(status_code=503, detail="Recommendation engine not initialized")
    
    if n < 1 or n > 50:
        raise HTTPException(status_code=400, detail="Sample size must be between 1 and 50")
    
    sample = recommender.retriever.assessments[:n]
    return {
        "sample_size": len(sample),
        "assessments": sample
    }


# Run with: uvicorn api.main:app --reload
if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("API_PORT", 8000))
    host = os.getenv("API_HOST", "0.0.0.0")
    
    print(f"\n{'='*60}")
    print("Starting SHL Assessment Recommendation API")
    print(f"Host: {host}")
    print(f"Port: {port}")
    
    uvicorn.run(app, host=host, port=port)
