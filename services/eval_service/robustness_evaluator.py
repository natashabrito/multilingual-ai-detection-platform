import numpy as np
import requests


DETECTION_ENGINE_URL = "http://localhost:8000/analyze"


class RobustnessEvaluator:
    """
    Calls the Detection Engine on original and adversarial variants,
    and computes robustness metrics:
    - Probability variance across variants
    - Label flip fraction
    """

    def evaluate(self, original_text, variants):
        results = []
        probs = []
        labels = []

        original_response = requests.post(
            DETECTION_ENGINE_URL,
            json={"text": original_text},
        ).json()

        original_prob = float(original_response.get("ai_probability", 0.0))
        original_label = bool(
            original_response.get("is_ai_generated", original_prob > 0.5)
        )

        probs.append(original_prob)
        labels.append(original_label)

        for variant in variants:
            response = requests.post(
                DETECTION_ENGINE_URL,
                json={"text": variant["text"]},
            ).json()

            prob = float(response.get("ai_probability", 0.0))
            label = bool(response.get("is_ai_generated", prob > 0.5))

            probs.append(prob)
            labels.append(label)

            results.append(
                {
                    "language": variant["language"],
                    "text": variant["text"],
                    "ai_probability": prob,
                    "label": label,
                }
            )

        prob_variance = float(np.var(probs)) if len(probs) > 1 else 0.0
        label_flips = sum(1 for label in labels if label != original_label)
        flip_fraction = float(label_flips) / float(len(labels)) if labels else 0.0

        return {
            "original_probability": original_prob,
            "probability_variance": prob_variance,
            "flip_fraction": flip_fraction,
            "variants_evaluated": results,
        }

