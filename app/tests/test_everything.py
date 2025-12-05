# test_everything.py
import sys

sys.path.append(".")
from app.tools.classification_tool import classification_tool
from app.tools.slack_tool import send_slack_alert_from_ticket

# Test 1: Classification
print("Testing classification...")
result = classification_tool.classify("My credit card was stolen!")
print(f"Classification result: {result}")

# Test 2: Business Logic
print("\nTesting business logic...")
if result["success"]:
    action = classification_tool.get_suggested_action(
        intent=result["intent"],
        confidence=result["confidence"],
    )
    print(f"Action: {action}")

# Test 3: Slack (mock mode)
print("\nTesting Slack (mock mode)...")
ticket_data = {
    "text": "URGENT: Credit card stolen",
    "intent": result.get("intent", "unknown"),
    "confidence": result.get("confidence", 0.0),
}
slack_result = send_slack_alert_from_ticket(ticket_data)
print(f"Slack result: {slack_result}")
