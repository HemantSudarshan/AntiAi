"""
TruthTracker API - Production-Grade Misinformation Detection
"""
from fastapi import FastAPI, UploadFile, File, HTTPException, Form, Depends, Security, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.security import APIKeyHeader
from contextlib import asynccontextmanager
import os
import time
import logging
from typing import Optional
from dotenv import load_dotenv

# Rate limiting
try:
    from slowapi import Limiter, _rate_limit_exceeded_handler
    from slowapi.util import get_remote_address
    from slowapi.errors import RateLimitExceeded
    SLOWAPI_AVAILABLE = True
except ImportError:
    SLOWAPI_AVAILABLE = False
    print("⚠️ slowapi not installed - rate limiting disabled")

# Magic byte validation
try:
    import magic
    MAGIC_AVAILABLE = True
except ImportError:
    MAGIC_AVAILABLE = False
    print("⚠️ python-magic not installed - using MIME type only")

# Optional torch import
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("⚠️ PyTorch not installed - some features may be limited")

load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Security configuration
API_KEY = os.getenv("API_KEY", "dev-key-change-in-production")
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)

# Rate limiter setup
if SLOWAPI_AVAILABLE:
    limiter = Limiter(key_func=get_remote_address)
else:
    limiter = None

# Global model references
fake_news_model = None
deepfake_model = None
explainer = None


async def verify_api_key(api_key: str = Security(api_key_header)):
    """Verify API key for authentication"""
    # Skip in development mode
    if os.getenv("ENVIRONMENT", "development") == "development":
        return True
    
    if not api_key or api_key != API_KEY:
        raise HTTPException(
            status_code=403,
            detail="Invalid or missing API key"
        )
    return True


def validate_image_file(contents: bytes, content_type: str) -> bool:
    """Validate image using magic bytes if available, otherwise MIME type"""
    if MAGIC_AVAILABLE:
        try:
            mime = magic.from_buffer(contents, mime=True)
            return mime in ["image/jpeg", "image/png", "image/webp"]
        except Exception as e:
            logger.warning(f"Magic byte validation failed: {e}")
            return content_type in ["image/jpeg", "image/png", "image/webp"]
    else:
        return content_type in ["image/jpeg", "image/png", "image/webp"]


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
        # Phase 3: Use enhanced models with ensemble voting
        fake_news_model = FakeNewsDetector(use_transformers=False, use_ensemble=True)
        deepfake_model = DeepfakeDetector(model_type="efficientnet", use_ensemble=True)
        explainer = ExplainabilityService()
        print("✅ Phase 3 Enhanced Models loaded successfully")
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

# Add rate limiter to app state if available
if SLOWAPI_AVAILABLE:
    app.state.limiter = limiter
    app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# CORS configuration
allowed_origins = os.getenv(
    "ALLOWED_ORIGINS",
    "http://localhost:3000,http://127.0.0.1:3000"
).split(",")

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type", "X-API-Key"],
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
async def analyze_news(
    request: Request,
    text: str = Form(...),
    authenticated: bool = Depends(verify_api_key)
):
    """
    Analyze news article for authenticity
    
    - **text**: News article content (min 20 chars, max 10000)
    
    Returns prediction, confidence score, and explanation
    """
    # Apply rate limiting if available
    if SLOWAPI_AVAILABLE and limiter:
        try:
            await limiter.check_rate_limit(request, "10/minute")
        except:
            pass  # Already handled by middleware
    
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
            detail="Service temporarily unavailable"
        )
    
    try:
        # Get prediction
        prediction = fake_news_model.predict(text)
        confidence = fake_news_model.get_confidence(text)
        
        # Get explanation with LIME integration
        explanation = {}
        if explainer:
            # Pass predict function for LIME
            explanation = explainer.explain_fake_news(
                text, 
                prediction,
                predict_fn=fake_news_model.predict_proba if hasattr(fake_news_model, 'predict_proba') else None
            )
        
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
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"News analysis error: {e}")
        raise HTTPException(
            status_code=500,
            detail="Analysis failed. Please try again."
        )


@app.post("/api/v1/analyze-image")
async def analyze_image(
    request: Request,
    file: UploadFile = File(...),
    authenticated: bool = Depends(verify_api_key)
):
    """
    Analyze image for deepfakes
    
    Accepts: JPG, PNG, WebP (max 10MB)
    
    Returns prediction, confidence score, and heatmap
    """
    # Apply rate limiting if available
    if SLOWAPI_AVAILABLE and limiter:
        try:
            await limiter.check_rate_limit(request, "5/minute")
        except:
            pass
    
    start_time = time.time()
    
    # Read file contents
    contents = await file.read()
    
    # Validate file size (10MB max)
    if len(contents) > 10 * 1024 * 1024:
        raise HTTPException(
            status_code=400,
            detail="File size must not exceed 10MB"
        )
    
    # Validate file type with magic bytes
    if not validate_image_file(contents, file.content_type):
        raise HTTPException(
            status_code=400,
            detail="Invalid image file"
        )
    
    if deepfake_model is None:
        raise HTTPException(
            status_code=503,
            detail="Service temporarily unavailable"
        )
    
    try:
        # Get prediction
        prediction = deepfake_model.predict(contents)
        confidence = deepfake_model.get_confidence(contents)
        
        # Get explanation (heatmap and regions)
        heatmap = None
        suspicious_regions = []
        summary = ""
        if explainer:
            heatmap = explainer.generate_heatmap(contents, deepfake_model)
            suspicious_regions = explainer.get_suspicious_regions(prediction)
            summary = explainer.get_analysis_summary(prediction, confidence, 'image')
        
        processing_time = int((time.time() - start_time) * 1000)
        
        return JSONResponse({
            "status": "success",
            "filename": file.filename,
            "prediction": "AUTHENTIC" if prediction == 0 else "DEEPFAKE",
            "confidence": round(float(confidence), 4),
            "explanation": {
                "heatmap": heatmap,
                "suspicious_regions": suspicious_regions,
                "summary": summary
            },
            "metadata": {
                "model_version": "2.0",
                "processing_time_ms": processing_time
            }
        })
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Image analysis error: {e}")
        raise HTTPException(
            status_code=500,
            detail="Analysis failed. Please try again."
        )


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
