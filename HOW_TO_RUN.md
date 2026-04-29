# How to Train & Run VeriText AI

## Prerequisites

- Python 3.10+
- Node.js 18+
- **GPU recommended** (training on CPU takes 3-4 hours, GPU takes ~15 min)

---

## Step 1: Install Python Dependencies

```bash
pip install -r requirements.txt
pip install datasets matplotlib seaborn
```

---

## Step 2: Train the Model

```bash
python train_pipeline.py
```

**What this does:**
1. Downloads the HC3 English dataset from HuggingFace (human vs ChatGPT answers)
2. Loads Hindi datasets from `datasets/val_NDTV_*.csv` (NDTV news vs BARD/GPT generations)
3. Loads Hinglish human data from `datasets/hinglish_conversations.csv`
4. Loads Hinglish AI data from `datasets/*.txt` files
5. Generates additional synthetic Hinglish AI samples
6. Trains an XLM-RoBERTa-base model (3 epochs)
7. Saves best model checkpoint to `checkpoints/best/`
8. Generates all metrics and charts in `output/`

**Training data breakdown:**
| Language | Human | AI | Source |
|----------|-------|----|--------|
| English | 2000 | 2000 | HC3 dataset |
| Hindi | ~1588 | ~1583 | NDTV news + BARD/GPT3.5/GPT4 |
| Hinglish | 2000 | ~2000 | Conversations + txt files + synthetic |

**Total: ~11,000 samples**

**Expected output:**
```
Train: ~9,500 samples
Val:   ~1,676 samples
Training 3 epochs on XLM-RoBERTa-base
```

### Tuning (optional)

Edit these values at the top of `train_pipeline.py`:

```python
MAX_LEN = 256          # token length (increase for longer texts)
BATCH_SIZE = 8         # increase if you have more GPU memory
LR = 2e-5              # learning rate
EPOCHS = 3             # more epochs = better but slower
MAX_SAMPLES_PER_LANG = 2000 # increase for more data per language
```

If you have a good GPU, bump `MAX_SAMPLES_PER_LANG` to 5000+ and `EPOCHS` to 5.

---

## Step 3: Check Output

After training, the `output/` folder will contain:

| File | Description |
|------|-------------|
| `confusion_matrix.png` | Overall confusion matrix |
| `confusion_matrix_per_language.png` | Confusion matrix for English, Hindi, Hinglish |
| `roc_curve.png` | ROC curve (overall + per language with AUC) |
| `precision_recall_curve.png` | Precision-Recall curve |
| `training_history.png` | Loss, accuracy, F1 over epochs |
| `probability_distribution.png` | AI probability distribution by true label |
| `accuracy_per_language.png` | Bar chart of accuracy per language |
| `language_confusion_matrix.png` | Language detection confusion matrix |
| `metrics_summary.json` | All metrics in JSON format |
| `classification_report.txt` | Sklearn classification report |
| `classification_report.json` | Same in JSON |

---

## Step 4: Run the Backend Server

```bash
uvicorn server:app --host 0.0.0.0 --port 8000
```

This starts the FastAPI backend that loads the trained model from `checkpoints/best/` and serves predictions.

**API Endpoints:**
- `GET /health` - Health check
- `POST /analyze` - Full analysis with explainability features
- `POST /predict` - Quick prediction
- `POST /predict_batch` - Batch prediction

---

## Step 5: Run the Frontend

```bash
cd frontend
npm install
npm run dev
```

Frontend runs at `http://localhost:5173` and proxies API calls to the backend at port 8000.

**Pages:**
- **Overview** - Dashboard with session stats
- **Analyze** - Paste text, get AI/Human prediction + explainability cards
- **Compare** - Side-by-side comparison of two texts
- **Settings** - Academic integrity mode

---

## Quick Start (all in one)

```bash
# Terminal 1: Train (only needed once)
pip install -r requirements.txt && pip install datasets matplotlib seaborn
python train_pipeline.py

# Terminal 2: Backend
uvicorn server:app --host 0.0.0.0 --port 8000

# Terminal 3: Frontend
cd frontend && npm install && npm run dev
```

Then open `http://localhost:5173` in your browser.

---

## Troubleshooting

- **"Model not found"** → Run `python train_pipeline.py` first
- **CUDA out of memory** → Reduce `BATCH_SIZE` to 4 in `train_pipeline.py`
- **Training too slow on CPU** → Use Google Colab with GPU runtime, upload the repo there
- **HC3 download fails** → Make sure you have internet access; the script auto-downloads from HuggingFace
