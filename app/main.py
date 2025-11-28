from fastapi import FastAPI, HTTPException
from fastapi.responses import RedirectResponse
from app.models import PredictionRequest, PredictionResponse
from app.prediction import predict_intent
import logging
import traceback
import gradio as gr
from app.gradio_interface import create_interface

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
app = FastAPI(title="Support Ticket Triage Agent")


@app.get("/")
def read_root():
    return RedirectResponse(url="/gradio/")


@app.get("/health")
async def health_check():
    return {"status": "healthy", "service": "Support Ticket Triage Agent"}


@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest):
    if not request.text.strip():
        raise HTTPException(status_code=400, detail="Text cannot be empty")

    try:
        logger.info(f"Received prediction request: {request.text}")
        intent, confidence = predict_intent(request.text)
        logger.info(f"Prediction successful: {intent} with confidence {confidence}")
        return {
            "prediction": intent,
            "confidence": confidence,
            "input_text": request.text,
        }
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
        intent, confidence, text = predict_intent(test_text)
        return {
            "status": "success",
            "test_text": test_text,
            "result": {
                "prediction": intent,
                "confidence": confidence,
                "input_text": text,
            },
        }
    except Exception as e:
        return {
            "status": "error",
            "test_text": test_text,
            "error": str(e),
            "traceback": traceback.format_exc(),
        }


gradio_app = create_interface()
app = gr.mount_gradio_app(app, gradio_app, path="/gradio")
