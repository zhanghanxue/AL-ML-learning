# test_fixed_priority.py
import sys

sys.path.append(".")

# Import your tools
from app.tools.classification_tool import classification_tool
from app.tools.slack_tool import slack_tool

test_cases = [
    ("My credit card was stolen!", "lost_or_stolen_card", 0.45),
    ("I see a fraudulent transaction", "fraudulent_transaction", 0.35),
    ("My account was hacked!", "account_hacked", 0.6),
    ("Need urgent help!", "urgent_help", 0.3),
    ("My card isn't working", "card_not_working", 0.7),
    ("What's my balance?", "balance_inquiry", 0.9),
]

print("Testing FIXED Slack tool Priority Logic:\n")
print(
    f"{'Ticket Text':<40} {'Intent':<25} {'Confidence':<10} {'Priority':<10} {'Urgent?'}"
)
print("-" * 100)

for text, intent, confidence in test_cases:
    priority = slack_tool.determine_priority(intent, confidence)
    is_urgent = priority in ["critical", "high"]

    print(
        f"{text[:35]:<40} {intent:<25} {confidence:<10.2f} {priority:<10} {is_urgent}"
    )
