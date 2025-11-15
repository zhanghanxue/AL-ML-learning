from transformers import pipeline
import logging
import torch

logger = logging.getLogger(__name__)

# Global pipeline instance
classifier = None


def load_model():
    """Load the model using transformers pipeline"""
    global classifier

    if classifier is None:
        try:
            logger.info("Loading model with transformers pipeline...")

            # Use pipeline for automatic handling
            classifier = pipeline(
                "text-classification",
                model="zhanghanxue/banking77-weighted-classifier",
                device=0 if torch.cuda.is_available() else -1,  # Use GPU if available
            )

            logger.info("✅ Model pipeline loaded successfully")

        except Exception as e:
            logger.error(f"❌ Error loading model pipeline: {e}")
            raise


def predict_intent(text: str):
    """Predict intent using transformers pipeline"""
    try:
        # Load model if not already loaded
        load_model()

        logger.info(f"Predicting intent for: '{text}'")

        # The pipeline handles everything: tokenization, inference, post-processing
        result = classifier(text)

        # Pipeline returns a list of dicts, even for single inputs
        if isinstance(result, list):
            prediction = result[0]
            intent = prediction["label"]
            confidence = prediction["score"]
        else:
            # Handle case where it returns a single dict (some versions)
            intent = result["label"]
            confidence = result["score"]

        logger.info(f"✅ Prediction: {intent} (confidence: {confidence:.4f})")

        return intent, confidence

    except Exception as e:
        logger.error(f"❌ Prediction error: {e}")
        raise
