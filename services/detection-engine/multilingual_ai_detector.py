import numpy as np
from datetime import datetime, timezone
from typing import Any, Dict

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import pipeline


class MultilingualAIDetector:
    """
    Cross-lingual semantic drift + detection façade for the Detection Engine.

    This microservice-focused class loads:
    - A multilingual sentence embedding model
    - En↔Hi translation pipelines

    and exposes:
    - analyze(): basic metadata hook (can be wired to your full detector)
    - analyze_crosslingual(): cross-lingual semantic drift analysis
    """

    def __init__(self) -> None:
        self.model_version = "v2.0.0-crosslingual"
        self.loaded_at = datetime.now(timezone.utc)

        print("[INFO] Loading embedding model...")
        self.embedding_model = SentenceTransformer(
            "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
        )

        print("[INFO] Loading translation pipelines...")
        self.translator_en_hi = pipeline(
            "translation",
            model="Helsinki-NLP/opus-mt-en-hi",
        )
        self.translator_hi_en = pipeline(
            "translation",
            model="Helsinki-NLP/opus-mt-hi-en",
        )

    # ------------------------------------------------
    # BASIC ANALYZE (placeholder)
    # ------------------------------------------------

    def analyze(self, text: str, language: str = "auto") -> Dict[str, Any]:
        if not text:
            return {"error": "Empty text provided"}

        # Simple, deterministic pseudo probability based on length;
        # in a full system this would call the main detector.
        length = len(text)
        ai_probability = float(min(max(length / 500.0, 0.1), 0.95))
        is_ai = ai_probability > 0.6

        return {
            "text_length": length,
            "language_detected": language,
            "ai_probability": ai_probability,
            "is_ai_generated": is_ai,
            "model_version": self.model_version,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

    # ------------------------------------------------
    # CROSS-LINGUAL DRIFT LOGIC
    # ------------------------------------------------

    def translate_text(self, text: str, direction: str = "en-hi") -> str:
        if direction == "en-hi":
            result = self.translator_en_hi(text)[0]["translation_text"]
        elif direction == "hi-en":
            result = self.translator_hi_en(text)[0]["translation_text"]
        else:
            result = text
        return result

    def get_embedding(self, text: str) -> np.ndarray:
        return self.embedding_model.encode([text])[0]

    def compute_similarity(self, emb1: np.ndarray, emb2: np.ndarray) -> float:
        return float(cosine_similarity([emb1], [emb2])[0][0])

    def analyze_crosslingual(self, text: str) -> Dict[str, Any]:
        if not text:
            return {"error": "Empty text provided"}

        # Step 1: Generate translations (English -> Hindi -> English)
        hi_text = self.translate_text(text, "en-hi")
        back_to_en = self.translate_text(hi_text, "hi-en")

        # Step 2: Generate embeddings
        original_emb = self.get_embedding(text)
        hi_emb = self.get_embedding(hi_text)
        back_en_emb = self.get_embedding(back_to_en)

        # Step 3: Compute similarities
        sim_hi = self.compute_similarity(original_emb, hi_emb)
        sim_back = self.compute_similarity(original_emb, back_en_emb)

        similarities = {
            "en_vs_hi": sim_hi,
            "en_vs_hi_back_en": sim_back,
        }

        # Step 4: Drift score = variance of similarity values
        drift_score = float(np.var(list(similarities.values())))

        return {
            "original_text": text,
            "hindi_translation": hi_text,
            "back_to_english": back_to_en,
            "similarities": similarities,
            "cross_lingual_drift_score": drift_score,
            "model_version": self.model_version,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

    def get_model_info(self) -> Dict[str, Any]:
        return {
            "model_version": self.model_version,
            "loaded_at": self.loaded_at.isoformat(),
        }


