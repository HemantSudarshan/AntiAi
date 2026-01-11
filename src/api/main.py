"""
TruthTracker API - Production-Grade Misinformation Detection
"""
from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from contextlib import asynccontextmanager
import os
import time
from typing import Optional
from dotenv import load_dotenv

# Optional torch import
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("⚠️ PyTorch not installed - some features may be limited")

load_dotenv()

# Global model references
fake_news_model = None
deepfake_model = None
explainer = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load models on startup, cleanup on shutdown"""
    global fake_news_model, deepfake_model, explainer
    
    print("🚀 Loading models...")
    
    # Lazy imports to avoid startup delays if models aren't needed
    from src.models.fake_news_detector import FakeNewsDetector
    from src.models.deepfake_detector import DeepfakeDetector
    from src.services.explainability import ExplainabilityService
    
    try:
        fake_news_model = FakeNewsDetector()
        deepfake_model = DeepfakeDetector()
        explainer = ExplainabilityService()
        print("✅ Models loaded successfully")
    except Exception as e:
        print(f"⚠️ Model loading error: {e}")
        print("   Some endpoints may not work until models are available")
    
    yield
    
    # Cleanup
    print("🔄 Shutting down...")


app = FastAPI(
    title="TruthTracker API",
    description="Production-grade misinformation and deepfake detection",
    version="2.0.0",
    lifespan=lifespan
)

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "https://*.vercel.app",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    device_info = "unknown"
    if TORCH_AVAILABLE:
        device_info = str(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    else:
        device_info = "torch not installed"
    
    return {
        "status": "healthy",
        "models_loaded": {
            "fake_news": fake_news_model is not None,
            "deepfake": deepfake_model is not None,
            "explainer": explainer is not None
        },
        "device": device_info
    }


@app.post("/api/v1/analyze-news")
async def analyze_news(text: str = Form(...)):
    """
    Analyze news article for authenticity
    
    - **text**: News article content (min 20 chars, max 10000)
    
    Returns prediction, confidence score, and explanation
    """
    start_time = time.time()
    
    # Validation
    if not text or len(text.strip()) < 20:
        raise HTTPException(
            status_code=400, 
            detail="Text must be at least 20 characters"
        )
    
    if len(text) > 10000:
        raise HTTPException(
            status_code=400,
            detail="Text must not exceed 10000 characters"
        )
    
    if fake_news_model is None:
        raise HTTPException(
            status_code=503,
            detail="Fake news model not loaded"
        )
    
    try:
        # Get prediction
        prediction = fake_news_model.predict(text)
        confidence = fake_news_model.get_confidence(text)
        
        # Get explanation
        explanation = {}
        if explainer:
            explanation = explainer.explain_fake_news(text, prediction)
        
        processing_time = int((time.time() - start_time) * 1000)
        
        return JSONResponse({
            "status": "success",
            "text_preview": text[:150] + "..." if len(text) > 150 else text,
            "prediction": "REAL" if prediction == 0 else "FAKE",
            "confidence": round(float(confidence), 4),
            "explanation": explanation,
            "metadata": {
                "model_version": "2.0",
                "processing_time_ms": processing_time
            }
        })
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/v1/analyze-image")
async def analyze_image(file: UploadFile = File(...)):
    """
    Analyze image for deepfakes
    
    Accepts: JPG, PNG, WebP (max 10MB)
    
    Returns prediction, confidence score, and heatmap
    """
    start_time = time.time()
    
    # Validate file type
    allowed_types = ["image/jpeg", "image/png", "image/webp"]
    if file.content_type not in allowed_types:
        raise HTTPException(
            status_code=400,
            detail=f"File type must be one of: {allowed_types}"
        )
    
    # Validate file size (10MB max)
    contents = await file.read()
    if len(contents) > 10 * 1024 * 1024:
        raise HTTPException(
            status_code=400,
            detail="File size must not exceed 10MB"
        )
    
    if deepfake_model is None:
        raise HTTPException(
            status_code=503,
            detail="Deepfake model not loaded"
        )
    
    try:
        # Get prediction
        prediction = deepfake_model.predict(contents)
        confidence = deepfake_model.get_confidence(contents)
        
        # Get explanation (heatmap)
        heatmap = None
        suspicious_regions = []
        if explainer:
            heatmap = explainer.generate_heatmap(contents, deepfake_model)
            suspicious_regions = explainer.get_suspicious_regions(heatmap)
        
        processing_time = int((time.time() - start_time) * 1000)
        
        return JSONResponse({
            "status": "success",
            "filename": file.filename,
            "prediction": "AUTHENTIC" if prediction == 0 else "DEEPFAKE",
            "confidence": round(float(confidence), 4),
            "explanation": {
                "heatmap": heatmap,
                "suspicious_regions": suspicious_regions
            },
            "metadata": {
                "model_version": "2.0",
                "processing_time_ms": processing_time
            }
        })
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v1/info")
async def api_info():
    """Get API information and capabilities"""
    return {
        "name": "TruthTracker API",
        "version": "2.0.0",
        "capabilities": {
            "fake_news_detection": {
                "languages": ["en"],
                "max_text_length": 10000
            },
            "deepfake_detection": {
                "supported_formats": ["jpg", "png", "webp"],
                "max_file_size_mb": 10
            }
        },
        "documentation": "/docs"
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "src.api.main:app",
        host=os.getenv("API_HOST", "0.0.0.0"),
        port=int(os.getenv("API_PORT", 8000)),
        reload=os.getenv("DEBUG", "true").lower() == "true"
    )
