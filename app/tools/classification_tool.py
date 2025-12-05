import requests
import json
import logging
from typing import Dict, Any, List
from app.config.settings import settings

logger = logging.getLogger(__name__)

class ClassificationTool:
    
    def __init__(self):
        self.api_url = f"{settings.LOCAL_API_URL}/predict"
    
    def classify(self, text: str) -> Dict[str, Any]:

        try:
            response = requests.post(
                self.api_url,
                json={"text": text},
                headers={'Content-Type': 'application/json'},
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                
                intent = result.get("prediction", "unknown")
                confidence = result.get("confidence", 0.0)
                
                is_urgent = intent in settings.URGENT_INTENTS and confidence > 0.7
                
                suggested_action = self.get_suggested_action(intent, confidence)
                
                logger.info(f"✅ Classification: {intent} ({confidence:.1%}) - Urgent: {is_urgent}")
                
                return {
                    "success": True,
                    "intent": intent,
                    "confidence": confidence,
                    "is_urgent": is_urgent,
                    "suggested_action": suggested_action,
                    "full_response": result,
                    "source": "zhanghanxue/banking77-weighted-classifier"
                }
                
            else:
                logger.error(f"Classification API error: {response.status_code}")
                return {
                    "success": False,
                    "intent": "api_error",
                    "confidence": 0.0,
                    "is_urgent": False,
                    "error": f"API returned {response.status_code}",
                    "source": "error"
                }
                
        except requests.exceptions.ConnectionError:
            logger.error("Cannot connect to classification API")
            return {
                "success": False,
                "intent": "connection_error",
                "confidence": 0.0,
                "is_urgent": False,
                "error": "Classification API not available",
                "source": "error"
            }
        except Exception as e:
            logger.error(f"Classification failed: {e}")
            return {
                "success": False,
                "intent": "unknown",
                "confidence": 0.0,
                "is_urgent": False,
                "error": str(e),
                "source": "error"
            }
    
    def get_suggested_action(self, intent: str, confidence: float) -> str:
        action_map = {
            "lost_or_stolen_card": "BLOCK_CARD_AND_NOTIFY_CUSTOMER",
            "fraudulent_transaction": "FREEZE_ACCOUNT_AND_INVESTIGATE",
            "account_hacked": "RESET_CREDENTIALS_AND_SECURITY_CHECK",
            "card_not_working": "CHECK_CARD_STATUS_AND_REISSUE",
            "transaction_history": "PROVIDE_STATEMENT_AND_DETAILS",
            "balance_inquiry": "SHOW_CURRENT_BALANCE",
            "general_inquiry": "ROUTE_TO_SUPPORT_AGENT",
            "loan_inquiry": "PROVIDE_LOAN_OPTIONS"
        }
        
        return action_map.get(intent, "ESCALATE_TO_HUMAN_AGENT")

classification_tool = ClassificationTool()

def classify_ticket(text: str) -> Dict[str, Any]:
    """Simple wrapper"""
    return classification_tool.classify(text)