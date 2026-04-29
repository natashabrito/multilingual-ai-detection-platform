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

        print("[INFO] Loading Google translation endpoints...")
        from deep_translator import GoogleTranslator
        self.translator_en_hi = GoogleTranslator(source='en', target='hi')
        self.translator_hi_en = GoogleTranslator(source='hi', target='en')

    # ------------------------------------------------
    # BASIC ANALYZE (placeholder)
    # ------------------------------------------------

    def analyze(self, text: str, language: str = "auto") -> Dict[str, Any]:
        if not text:
            return {"error": "Empty text provided"}

        length = len(text)
        ai_probability = float(min(max(length / 500.0, 0.1), 0.95))
        
       
        if ai_probability < 0.5:
            ai_probability = ai_probability * 0.3
            
        is_ai = ai_probability > 0.6
        
        import re
        import hashlib
        
        words = re.findall(r'\w+', text)
        word_count = len(words)
        reading_time_seconds = max(1, int((word_count / 200) * 60))
        
        sentences = re.split(r'(?<=[.!?।])\s+|\n+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        sentence_scores = []
        for s in sentences:
            if not s: continue
            h = int(hashlib.md5(s.encode('utf-8')).hexdigest(), 16)
            s_prob = (h % 100) / 100.0
            blended_prob = (s_prob * 0.4) + (ai_probability * 0.6)
            if blended_prob < 0.5:
                blended_prob = blended_prob * 0.3
                
            sentence_scores.append({
                "text": s,
                "ai_probability": blended_prob
            })

        return {
            "text": text,
            "text_length": length or 0,
            "word_count": word_count,
            "reading_time_seconds": reading_time_seconds,
            "sentence_scores": sentence_scores,
            "language_detected": language or "",
            "ai_probability": ai_probability or 0.0,
            "is_ai_generated": is_ai if is_ai is not None else False,
            "model_version": self.model_version,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

    # ------------------------------------------------
    # CROSS-LINGUAL DRIFT LOGIC
    # ------------------------------------------------

    def translate_text(self, text: str, direction: str = "en-hi") -> str:
        if not text or not text.strip():
            return ""
            
        import re
        sentences = re.split(r'(?<=[.!?।])\s+|\n+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        if not sentences:
            sentences = [text.strip()]
            
        translator = self.translator_en_hi if direction == "en-hi" else self.translator_hi_en
        
        translated_chunks = []
        current_chunk = ""
        
        for s in sentences:
            # Google Translate API has a 5000 character limit per request. 
            # We batch up to 4000 characters to be safe and reduce latency.
            if len(current_chunk) + len(s) < 4000:
                current_chunk += s + " "
            else:
                if current_chunk.strip():
                    try:
                        res = translator.translate(current_chunk.strip())
                        translated_chunks.append(res)
                    except Exception as e:
                        print(f"[WARN] Translation failed: {e}")
                        translated_chunks.append(current_chunk.strip())
                current_chunk = s + " "
                
        if current_chunk.strip():
            try:
                res = translator.translate(current_chunk.strip())
                translated_chunks.append(res)
            except Exception as e:
                print(f"[WARN] Translation failed: {e}")
                translated_chunks.append(current_chunk.strip())
                
        return " ".join(translated_chunks)

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
            "original_text": text or "",
            "hindi_translation": hi_text or "",
            "back_to_english": back_to_en or "",
            "translated_text": hi_text or "",
            "back_translated_text": back_to_en or "",
            "analysis_text": text or "",
            "paraphrased_text": "",

            "similarities": similarities or {},
            "cross_lingual_drift_score": drift_score or 0,
            "model_version": self.model_version,
            "timestamp": datetime.now(timezone.utc).isoformat(),
   }

    def get_model_info(self) -> Dict[str, Any]:
        return {
            "model_version": self.model_version,
            "loaded_at": self.loaded_at.isoformat(),
        }


