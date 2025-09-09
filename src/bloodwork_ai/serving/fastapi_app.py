"""FastAPI application for bloodwork AI serving."""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from ..tasks.cbc.anemia_classifier import AnemiaClassifier
from ..models.vision import YOLODetector
from ..utils.log import setup_logging, get_logger

# Setup logging
setup_logging(log_level="INFO")
logger = get_logger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Bloodwork AI API",
    description="Multi-modal bloodwork analysis platform",
    version="0.1.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global model instances
anemia_classifier = None
wbc_detector = None


class CBCRequest(BaseModel):
    """CBC data request model."""
    wbc: Optional[float] = None
    rbc: Optional[float] = None
    hb: Optional[float] = None
    hct: Optional[float] = None
    plt: Optional[float] = None
    mcv: Optional[float] = None
    mch: Optional[float] = None
    mchc: Optional[float] = None
    rdw: Optional[float] = None
    neut_pct: Optional[float] = None
    lymph_pct: Optional[float] = None
    mono_pct: Optional[float] = None
    eos_pct: Optional[float] = None
    baso_pct: Optional[float] = None


class PredictionResponse(BaseModel):
    """Prediction response model."""
    prediction: str
    confidence: float
    probabilities: Dict[str, float]
    model_info: Dict[str, Any]


class DetectionResponse(BaseModel):
    """Detection response model."""
    detections: List[Dict[str, Any]]
    cell_counts: Dict[str, int]
    statistics: Dict[str, Any]


@app.on_event("startup")
async def startup_event():
    """Initialize models on startup."""
    global anemia_classifier, wbc_detector
    
    logger.info("Initializing models...")
    
    # Initialize anemia classifier
    try:
        anemia_classifier = AnemiaClassifier()
        # In a real scenario, you would load a trained model
        # anemia_classifier.load_model("artifacts/models/anemia_classifier.pkl")
        logger.info("Anemia classifier initialized")
    except Exception as e:
        logger.warning(f"Failed to initialize anemia classifier: {e}")
    
    # Initialize WBC detector
    try:
        wbc_detector = YOLODetector()
        # In a real scenario, you would load a trained model
        # wbc_detector.load_model("artifacts/models/yolo_wbc.pt")
        logger.info("WBC detector initialized")
    except Exception as e:
        logger.warning(f"Failed to initialize WBC detector: {e}")
    
    logger.info("Model initialization completed")


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "message": "Bloodwork AI API",
        "version": "0.1.0",
        "status": "running"
    }


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "models": {
            "anemia_classifier": anemia_classifier is not None,
            "wbc_detector": wbc_detector is not None
        }
    }


@app.post("/predict/cbc/anemia", response_model=PredictionResponse)
async def predict_anemia(request: CBCRequest):
    """Predict anemia type from CBC data."""
    if anemia_classifier is None:
        raise HTTPException(status_code=503, detail="Anemia classifier not available")
    
    try:
        # Convert request to DataFrame
        data = request.dict()
        df = pd.DataFrame([data])
        
        # Make prediction
        prediction = anemia_classifier.predict(df)[0]
        probabilities = anemia_classifier.predict_proba(df)[0]
        
        # Convert probabilities to dictionary
        prob_dict = {
            "normal": float(probabilities[0]),
            "iron_deficiency": float(probabilities[1]),
            "thalassemia_trait": float(probabilities[2]),
            "b12_deficiency": float(probabilities[3])
        }
        
        confidence = float(max(probabilities))
        
        return PredictionResponse(
            prediction=prediction,
            confidence=confidence,
            probabilities=prob_dict,
            model_info={"algorithm": "xgboost", "is_trained": False}
        )
        
    except Exception as e:
        logger.error(f"Anemia prediction failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict/vision/wbc", response_model=DetectionResponse)
async def detect_wbc(image: UploadFile = File(...)):
    """Detect WBCs in blood smear image."""
    if wbc_detector is None:
        raise HTTPException(status_code=503, detail="WBC detector not available")
    
    try:
        # Read image
        image_data = await image.read()
        
        # Convert to numpy array
        import cv2
        nparr = np.frombuffer(image_data, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Make predictions
        detections = wbc_detector.predict(img)
        cell_counts = wbc_detector.count_cells(img)
        statistics = wbc_detector.get_cell_statistics(img)
        
        return DetectionResponse(
            detections=detections,
            cell_counts=cell_counts,
            statistics=statistics
        )
        
    except Exception as e:
        logger.error(f"WBC detection failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict/vision/rbc", response_model=DetectionResponse)
async def detect_rbc(image: UploadFile = File(...)):
    """Detect RBCs in blood smear image."""
    # For now, use the same detector
    return await detect_wbc(image)


@app.post("/predict/vision/platelet", response_model=DetectionResponse)
async def detect_platelet(image: UploadFile = File(...)):
    """Detect platelets in blood smear image."""
    # For now, use the same detector
    return await detect_wbc(image)


@app.get("/models/info")
async def get_models_info():
    """Get information about loaded models."""
    info = {
        "anemia_classifier": {
            "available": anemia_classifier is not None,
            "info": anemia_classifier.get_model_info() if anemia_classifier else None
        },
        "wbc_detector": {
            "available": wbc_detector is not None,
            "info": wbc_detector.get_model_info() if wbc_detector else None
        }
    }
    
    return info


@app.get("/metrics")
async def get_metrics():
    """Get Prometheus metrics."""
    # This is a simplified implementation
    # In practice, you would use prometheus_client
    return {
        "requests_total": 0,
        "predictions_total": 0,
        "errors_total": 0
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)
