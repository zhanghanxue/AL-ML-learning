import requests
import json
import logging
from typing import Dict, Any
from app.config.settings import settings

logger = logging.getLogger(__name__)


class SlackTool:
    def __init__(self):
        self.webhook_url = settings.SLACK_WEBHOOK_URL
        self.enabled = bool(self.webhook_url)

        if not self.enabled:
            logger.warning(
                "Slack webhook not configured. Alerts will be logged but not sent."
            )

    def determine_priority(self, intent: str, confidence: float) -> str:
        # Critical: Immediate attention needed
        if (
            intent in ["lost_or_stolen_card", "fraudulent_transaction"]
            and confidence > 0.4
        ):
            return "critical"

        # High: Urgent but not emergency
        elif intent in ["account_hacked", "urgent_help"] and confidence > 0.3:
            return "high"

        # Medium: Should be reviewed soon
        elif (
            hasattr(settings, "URGENT_INTENTS")
            and intent in settings.URGENT_INTENTS
            and confidence > 0.25
        ):
            return "medium"

        # Low: Routine
        else:
            return "low"

    def send_alert(
        self, ticket_text: str, intent: str, confidence: float, priority: str = None
    ) -> Dict[str, Any]:
        if priority is None:
            priority = self.determine_priority(intent, confidence)

        # Emoji mapping for visual impact
        priority_emoji = {"low": "ℹ️", "medium": "⚠️", "high": "🚨", "critical": "🔥"}

        emoji = priority_emoji.get(priority, "ℹ️")

        # Build Slack message payload
        slack_message = {
            "blocks": [
                {
                    "type": "header",
                    "text": {
                        "type": "plain_text",
                        "text": f"{emoji} Support Ticket Alert - {priority.upper()} Priority",
                    },
                },
                # Main content
                {
                    "type": "section",
                    "text": {
                        "type": "mrkdwn",
                        "text": f"*Intent:* `{intent}`\n*Confidence:* `{confidence:.1%}`",
                    },
                },
                # Ticket preview
                {
                    "type": "section",
                    "text": {
                        "type": "mrkdwn",
                        "text": f"*Ticket Preview:*\n{ticket_text[:300]}...",
                    },
                },
                # Footer context
                {
                    "type": "context",
                    "elements": [
                        {
                            "type": "mrkdwn",
                            "text": "🔧 *Auto-triaged by AI Agent* | Status: Needs Review",
                        }
                    ],
                },
            ]
        }

        # Mock Mode Check
        if not self.enabled:
            logger.info(
                f"[MOCK SLACK] Would send {priority} alert for intent '{intent}'"
            )
            return {
                "success": True,
                "message": "Mock alert logged (Slack not configured)",
                "priority": priority,
                "intent": intent,
                "mock": True,
            }

        # Send to Slack
        try:
            response = requests.post(
                self.webhook_url,
                data=json.dumps(slack_message),
                headers={"Content-Type": "application/json"},
                timeout=5,
            )

            if response.status_code == 200:
                logger.info(f"✅ Slack alert sent: {priority} priority for '{intent}'")
                return {
                    "success": True,
                    "message": f"Slack alert sent ({priority} priority)",
                    "priority": priority,
                    "intent": intent,
                    "mock": False,
                }
            else:
                # Slack returned an error
                logger.error(f"Slack error {response.status_code}: {response.text}")
                return {
                    "success": False,
                    "message": f"Slack error: {response.status_code}",
                    "priority": priority,
                }

        except Exception as e:
            logger.error(f"Slack connection failed: {e}")
            return {
                "success": False,
                "message": f"Slack connection failed: {str(e)}",
                "priority": priority,
            }


slack_tool = SlackTool()


def send_slack_alert(
    ticket_text: str, intent: str, confidence: float
) -> Dict[str, Any]:
    """Simple wrapper for easier calling"""
    return slack_tool.send_alert(ticket_text, intent, confidence)


def send_slack_alert_from_ticket(ticket_data: Dict[str, Any]) -> Dict[str, Any]:
    """Wrapper that accepts ticket dictionary"""
    return send_slack_alert(
        ticket_data.get("text", ""),
        ticket_data.get("intent", "unknown"),
        ticket_data.get("confidence", 0.0),
    )
