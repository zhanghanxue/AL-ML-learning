from fastapi import FastAPI, HTTPException
from app.models import PredictionRequest, PredictionResponse
from app.prediction import predict_intent
import logging
import traceback

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
app = FastAPI()

@app.get("/")
async def home():
    return {"message": "Banking77 Intent Classifier API"}

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest):
    if not request.text.strip():
        raise HTTPException(status_code=400, detail="Text cannot be empty")
    
    try:
        logger.info(f"Received prediction request: {request.text}")
        intent, confidence = predict_intent(request.text)
        logger.info(f"Prediction successful: {intent} with confidence {confidence}")
        return {"prediction": intent, "confidence": confidence}
    except Exception as e:
        # Get the full traceback
        error_traceback = traceback.format_exc()
        logger.error(f"Prediction failed: {str(e)}")
        logger.error(f"Full traceback: {error_traceback}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@app.get("/debug-test")
def debug_test():
    """Test the prediction function directly"""
    test_text = "I want to check my balance"
    try:
        intent, confidence = predict_intent(test_text)
        return {
            "status": "success",
            "test_text": test_text,
            "result": {"prediction": intent, "confidence": confidence}
        }
    except Exception as e:
        return {
            "status": "error",
            "test_text": test_text,
            "error": str(e),
            "traceback": traceback.format_exc()
        }