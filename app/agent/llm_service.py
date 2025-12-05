import requests
import time
import json
import logging
from typing import Optional, Dict, Any
from app.config.settings import settings

logger = logging.getLogger(__name__)


class LLMService:
    def __init__(self):
        self.api_url = (
            f"https://api-inference.huggingface.co/models/{settings.HF_MODEL}"
        )
        self.headers = {
            "Authorization": f"Bearer {settings.HF_TOKEN}",
            "Content-Type": "application/json",
        }
        self.total_cost = 0.0
        self.request_count = 0

        logger.info(f"LLM Service initialized with model: {settings.HF_MODEL}")

    def estimate_cost(self, response_time_ms: int) -> float:
        seconds = response_time_ms / 1000.0
        estimated_cost = seconds * settings.COST_PER_SECOND_GPU
        logger.debug(f"Cost estimate: {response_time_ms}ms = ${estimated_cost:.6f}")
        return estimated_cost

    def generate(self, prompt: str, max_tokens: Optional[int] = None) -> Dict[str, Any]:
        if self.total_cost > settings.FREE_TIER_LIMIT_USD * 0.9:
            logger.warning(
                f"⚠️  Warning: At 90% of free tier (${self.total_cost:.4f}/${settings.FREE_TIER_LIMIT_USD})"
            )

        payload = {
            "inputs": prompt,
            "parameters": {
                "max_new_tokens": max_tokens or settings.LLM_MAX_TOKENS,
                "temperature": settings.LLM_TEMPERATURE,
                "top_p": 0.9,
                "do_sample": True,
                "return_full_text": False,
            },
            "options": {"use_cache": False, "wait_for_model": True},
        }

        try:
            start_time = time.time()
            response = requests.post(
                self.api_url, headers=self.headers, json=payload, timeout=45
            )
            response_time_ms = int((time.time() - start_time) * 1000)

            if response.status_code == 503:
                logger.info("Model loading (503), waiting 15 seconds...")
                time.sleep(15)
                # Try once more
                return self.generate(prompt, max_tokens)

            if response.status_code == 429:
                logger.error("Rate limit exceeded (429), waiting 30 seconds...")
                time.sleep(30)
                return self.generate(prompt, max_tokens)

            response.raise_for_status()

            result = response.json()

            if isinstance(result, list):
                generated_text = result[0].get("generated_text", "")
            elif isinstance(result, dict):
                generated_text = result.get("generated_text", "")
            else:
                generated_text = str
                logger.warning(
                    f"Unexpected response format from LLM API: {type(result)}"
                )

            if generated_text.startswith(prompt):
                generated_text = generated_text[len(prompt) :].lstrip()

            self.request_count += 1
            cost = self.estimate_cost(response_time_ms)
            self.total_cost += cost

            logger.info(
                f"LLM request #{self.request_count} completed in {response_time_ms}ms, cost: ${cost:.6f}, total cost: ${self.total_cost:.4f}"
            )

            return {
                "success": True,
                "text": generated_text,
                "cost": cost,
                "total_cost": self.total_cost,
                "response_time_ms": response_time_ms,
                "requests": self.request_count,
            }

        except requests.exceptions.Timeout:
            logger.error("LLM request timed out.")
            return {
                "success": False,
                "text": "The AI service is currently busy. Please try again.",
                "error": "Request timed out.",
            }

        except requests.exceptions.HTTPError as http_err:
            logger.error(f"LLM HTTP error occurred: {http_err}")
            return {
                "success": False,
                "text": f"AI service error: HTTP {response.status_code if 'response' in locals() else 'unknown'}",
                "error": str(http_err),
            }

        except Exception as e:
            logger.error(f"Unexpected error during LLM request: {e}")
            return {
                "success": False,
                "text": f"AI unexpected error occurred: {str(e)}",
                "error": str(e),
            }

    def get_cost_metrics(self) -> Dict[str, Any]:
        remaining_free_tier = settings.FREE_TIER_LIMIT_USD - self.total_cost

        return {
            "total_cost": self.total_cost,
            "free_tier_limit": settings.FREE_TIER_LIMIT_USD,
            "remaining": remaining_free_tier,
            "remaining_percentage": (remaining_free_tier / settings.FREE_TIER_LIMIT_USD)
            * 100,
            "requests": self.request_count,
            "status": "healty" if remaining_free_tier > 0 else "limit_exceeded",
            "model": settings.HF_MODEL,
        }


llm_service = LLMService()
