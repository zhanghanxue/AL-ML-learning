import os
from typing import List, Optional

class Settings:
    
    HF_TOKEN: str = os.getenv("HF_TOKEN", "")
    HF_MODEL: str = os.getenv("HF_MODEL", "mistralai/Mistral-7B-Instruct-v0.1")

    SLACK_WEBHOOK_URL: str = os.getenv("SLACK_WEBHOOK_URL", "")

    LOCAL_API_URL: str = os.getenv("LOCAL_API_URL", "http://localhost:8000")

    MAX_REASONING_STEPS: int = int(os.getenv("MAX_REASONING_STEPS", "3"))

    LLM_MAX_TOKENS: int = int(os.getenv("LLM_MAX_TOKENS", "200"))

    LLM_TEMPERATURE: float = float(os.getenv("LLM_TEMPERATURE", "0.7"))

    URGENT_INTENTS = [
        "lost_or_stolen_card",
        "fraudulent_transaction",
        "account_hacked",
        "urgent_help",
        "card_not_working"
    ]

    FREE_TIER_LIMIT_USD: float = 0.10  # $0.10 per month
    COST_PER_SECOND_GPU: float = 0.00012  # T4 GPU cost per second

    @classmethod
    def validate(cls):
        if not cls.HF_TOKEN:
            print("⚠️ HF_TOKEN not set. LLM features disabled.")
            return False
        return True

settings = Settings()