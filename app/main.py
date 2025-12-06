from fastapi import FastAPI, HTTPException
from fastapi.responses import RedirectResponse
import logging
import traceback
import gradio as gr
from pydantic import BaseModel
from typing import Optional

from app.models import PredictionRequest, PredictionResponse
from app.prediction import predict_intent
from app.gradio_interface import create_interface
from app.agent.agent_gradio import create_agent_interface
from app.agent.react_agent import react_agent
from app.agent.llm_service import llm_service
from app.tools.slack_tool import slack_tool
from app.tools.classification_tool import classification_tool
from app.config.settings import settings

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
app = FastAPI(title="Support Ticket Triage Agent")


class AgentRequest(BaseModel):
    """Request format for agent endpoint"""

    text: str
    user_id: Optional[str] = "anonymous"


class AgentResponse(BaseModel):
    """Response format for agent endpoint"""

    status: str
    response: str
    intent: Optional[str] = None
    confidence: Optional[float] = None
    is_urgent: Optional[bool] = None
    escalated: Optional[bool] = None
    processing_time: float
    llm_cost: float


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


@app.get("/agent/health")
async def agent_health():
    return {
        "agent": "ready",
        "llm": {
            "configured": bool(settings.HF_TOKEN),
            "model": settings.HF_MODEL,
            "cost_tracking": "enabled",
        },
        "slack": {
            "configured": bool(settings.SLACK_WEBHOOK_URL),
            "mode": "real" if slack_tool.enabled else "mock",
        },
        "classification": {
            "source": "zhanghanxue/banking77-weighted-classifier",
            "url": settings.REMOTE_API_URL,
        },
        "cost_metrics": llm_service.get_cost_metrics(),
    }


@app.post("/agent/predict", response_model=AgentResponse)
async def agent_predict(request: AgentRequest):
    if not request.text.strip():
        raise HTTPException(status_code=400, detail="Ticket text cannot be empty")

    if len(request.text.strip()) < 10:
        raise HTTPException(
            status_code=400, detail="Ticket text too short (min 10 chars)"
        )

    try:
        logger.info(f"🧠 Agent processing ticket from user {request.user_id}")

        # Call the agent
        result = react_agent.process(request.text)

        # Ensure all fields are present
        if "cost" in result:
            result["llm_cost"] = result.pop("cost")

        # Set defaults for any missing fields
        defaults = {
            "intent": "unknown",
            "confidence": 0.0,
            "is_urgent": False,
            "escalated": False,
            "llm_cost": 0.0,
        }

        for key, value in defaults.items():
            if key not in result:
                result[key] = value

        return AgentResponse(**result)

    except Exception as e:
        logger.error(f"Agent processing failed: {str(e)}")
        raise HTTPException(
            status_code=500, detail=f"Agent processing failed: {str(e)}"
        )


@app.get("/agent/metrics")
async def agent_metrics():
    return {
        "costs": llm_service.get_cost_metrics(),
        "configuration": {
            "max_reasoning_steps": settings.MAX_REASONING_STEPS,
            "urgent_intents": settings.URGENT_INTENTS,
            "free_tier_limit": f"${settings.FREE_TIER_LIMIT_USD}",
        },
        "performance": {
            "slack_enabled": slack_tool.enabled,
            "llm_available": bool(settings.HF_TOKEN),
        },
    }


@app.get("/agent/debug")
async def agent_debug():
    test_ticket = "My credit card was stolen and someone made unauthorized purchases!"

    # Test each component
    classification = classification_tool.classify(test_ticket)

    llm_test = llm_service.generate("Test: What is 2+2?")

    slack_test = slack_tool.send_alert(
        test_ticket, "fraudulent_transaction", 0.95, "critical"
    )

    full_agent_test = react_agent.process(test_ticket)

    return {
        "components": {
            "classification": classification,
            "llm": {"success": llm_test["success"], "cost": llm_test.get("cost", 0)},
            "slack": slack_test,
        },
        "full_agent_test": full_agent_test,
        "status": "all_components_working",
    }


gradio_app = create_interface()
app = gr.mount_gradio_app(app, gradio_app, path="/gradio")
agent_gradio_app = create_agent_interface()
app = gr.mount_gradio_app(app, agent_gradio_app, path="/agent/gradio", app_kwargs={"theme": "soft"})
