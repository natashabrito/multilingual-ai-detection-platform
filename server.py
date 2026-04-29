import os
from pathlib import Path
from typing import List

import torch
from fastapi import FastAPI
from fastapi.responses import FileResponse
from pydantic import BaseModel
from starlette.middleware.cors import CORSMiddleware

from multilingual_ai_detector import analyze_texts, predict_text
import multilingual_ai_detector
print("Using detector file:", multilingual_ai_detector.__file__)


MODEL_PATH = os.environ.get("DETECTOR_MODEL_PATH", "checkpoints/best")
STATIC_DIR = Path(__file__).resolve().parent / "static"


class PredictRequest(BaseModel):
    text: str


class PredictBatchRequest(BaseModel):
    texts: List[str]


app = FastAPI(
    title="VeriText AI",
    description=(
        "Detects whether text is AI-generated or human-written across "
        "English, Hindi and Hinglish, and predicts the language."
    ),
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health():
    model_loaded = (STATIC_DIR.parent / MODEL_PATH / "config.json").exists()
    return {
        "status": "ok",
        "model_path": MODEL_PATH,
        "model_loaded": model_loaded,
        "cuda": torch.cuda.is_available(),
    }


def _model_config_path():
    p = Path(MODEL_PATH)
    if not p.is_absolute():
        p = STATIC_DIR.parent / MODEL_PATH
    return p / "config.json"


def _run_predict(texts: List[str]):
    if not _model_config_path().exists():
        return None, "Model not found. Train first: python multilingual_ai_detector.py --train_path train.jsonl --val_path val.jsonl"
    try:
        results = predict_text(MODEL_PATH, texts)
        return results, None
    except Exception as e:
        return None, str(e)


@app.post("/predict")
def predict(req: PredictRequest):
    if not req.text.strip():
        return {"error": "Text cannot be empty."}
    results, err = _run_predict([req.text])
    if err:
        return {"error": err}
    return results[0]


@app.post("/analyze")
def analyze(req: PredictRequest):
    if not req.text.strip():
        return {"error": "Text cannot be empty."}
    if not _model_config_path().exists():
        return {
            "error": "Model not found. Train first: python multilingual_ai_detector.py --train_path train.jsonl --val_path val.jsonl"
        }
    try:
        results = analyze_texts(MODEL_PATH, [req.text])
    except Exception as e:
        return {"error": str(e)}
    return results[0]


@app.post("/predict_batch")
def predict_batch(req: PredictBatchRequest):
    valid_texts = [t for t in req.texts if t and t.strip()]
    if not valid_texts:
        return {"error": "No non-empty texts provided."}
    results, err = _run_predict(valid_texts)
    if err:
        return {"error": err}
    return {"results": results}


@app.get("/")
def index():
    index_file = STATIC_DIR / "index.html"
    if index_file.exists():
        return FileResponse(index_file)
    return {"message": "Multilingual AI Detector API. Serve static/ for UI.", "docs": "/docs"}


