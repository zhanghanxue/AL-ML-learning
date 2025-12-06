import json
import re
import logging
from typing import Dict, Any
from datetime import datetime

from app.agent.llm_service import llm_service
from app.tools.slack_tool import slack_tool
from app.tools.classification_tool import classification_tool
from app.config.settings import settings

logger = logging.getLogger(__name__)


class ReActAgent:
    def __init__(self, max_steps: int = None):
        self.max_steps = max_steps or settings.MAX_REASONING_STEPS

        self.system_prompt = self._create_system_prompt()

        logger.info(f"ReAct Agent initialized with {self.max_steps} max steps")

    def _create_system_prompt(self) -> str:
        return """You are a Support Ticket Triage Agent. Follow this process:

1. CLASSIFY the ticket to understand customer intent
2. DECIDE if immediate action is needed
3. RESPOND appropriately to the customer
4. ESCALATE urgent tickets to human team

Available tools:
- classification_tool: Determines the intent (e.g., fraud, card issue, inquiry)
- slack_tool: Sends alerts to human team for urgent issues

Urgent intents that require escalation:
- card_lost_or_stolen
- fraudulent_transaction
- account_hacked
- urgent_help
- card_not_working (if high confidence)

Response Format (JSON only):
{
    "thought": "Your reasoning about the ticket",
    "action": "classify|slack_alert|respond",
    "action_input": "What to do or say",
    "confidence_threshold": 0.7
}

Always start with classification. Only escalate if urgent AND confidence > threshold.
"""

    def _parse_agent_response(self, text: str) -> Dict[str, Any]:
        try:
            # Try to find JSON pattern (most reliable)
            json_match = re.search(r"\{.*\}", text, re.DOTALL)
            if json_match:
                return json.loads(json_match.group())

            # LLM might put JSON on one line
            lines = text.strip().split("\n")
            for line in lines:
                line = line.strip()
                if line.startswith("{") and line.endswith("}"):
                    return json.loads(line)

            # If no JSON found, create a fallback
            logger.warning(f"No JSON found in LLM response: {text[:100]}...")
            return {
                "thought": "Could not parse structured response",
                "action": "respond",
                "action_input": "I'll help analyze your ticket. One moment...",
                "confidence_threshold": 0.7,
            }

        except json.JSONDecodeError as e:
            logger.error(f"JSON parse error: {e}")
            return {
                "thought": f"Parse error: {e}",
                "action": "respond",
                "action_input": "I encountered an error. Please try again or contact support.",
                "confidence_threshold": 0.7,
            }

    def process(self, ticket_text: str) -> Dict[str, Any]:
        start_time = datetime.now()
        logger.info(f"🧠 Agent processing: {ticket_text[:100]}...")

        classification = classification_tool.classify(ticket_text)

        logger.info(f"Classification result: {classification}")

        if not classification["success"]:
            return {
                "status": "error",
                "response": "Unable to analyze your ticket. Please contact support directly.",
                "processing_time": (datetime.now() - start_time).total_seconds(),
                "error": classification.get("error", "Unknown classification error"),
                "cost": 0.0,
            }

        intent = classification["intent"]
        confidence = classification["confidence"]
        is_urgent = classification["is_urgent"]

        llm_prompt = f"""{self.system_prompt}

Ticket: {ticket_text}
Initial Analysis: This appears to be about '{intent}' with {confidence:.1%} confidence.
{"⚠️  This is URGENT and requires immediate attention!" if is_urgent else "This is a standard ticket."}

What should I do?"""

        llm_result = llm_service.generate(llm_prompt)

        if llm_result["success"]:
            # Parse LLM's decision
            agent_decision = self._parse_agent_response(llm_result["text"])

            action = agent_decision.get("action", "respond")
            action_input = agent_decision.get("action_input", "")

            # Execute based on action
            if action == "slack_alert" and is_urgent:
                # Send Slack alert
                priority = slack_tool.determine_priority(intent, confidence)
                slack_result = slack_tool.send_alert(
                    ticket_text, intent, confidence, priority
                )

                response = f"I've identified this as urgent ({intent}). "
                response += "The team has been alerted and will contact you shortly."
                escalated = slack_result["success"]

            elif action == "classify":
                # Already classified - just respond
                suggested_action = classification_tool.get_suggested_action(
                    intent, confidence
                )
                response = f"I understand this is about {intent}. {suggested_action}"
                escalated = False

            else:  # Default: respond
                response = (
                    action_input
                    or f"Thank you for reporting this {intent} issue. "
                    f"Our system is processing your request."
                )
                escalated = False

        else:
            # LLM failed - use fallback logic
            if is_urgent:
                priority = slack_tool.determine_priority(intent, confidence)
                slack_tool.send_alert(ticket_text, intent, confidence, priority)
                response = (
                    f"Urgent {intent} detected! The support team has been notified."
                )
                escalated = True
            else:
                response = f"Thank you for your message about {intent}. We'll get back to you soon."
                escalated = False

        processing_time = (datetime.now() - start_time).total_seconds()
        cost = llm_result.get("cost", 0.0) if llm_result.get("success") else 0.0

        return {
            "status": "success",
            "response": response,  # What to tell the user
            "intent": intent,  # Classification result
            "confidence": float(confidence),  # How confident
            "is_urgent": is_urgent,  # Business flag
            "escalated": escalated,  # Action taken
            "processing_time": processing_time,  # Performance metric
            "llm_cost": cost,  # Cost of this request
            "timestamp": datetime.now().isoformat(),  # When it happened
            "agent_version": "1.0",  # For tracking
        }


react_agent = ReActAgent()


def process_ticket_agent(ticket_text: str) -> Dict[str, Any]:
    """Simple wrapper for the agent"""
    return react_agent.process(ticket_text)
