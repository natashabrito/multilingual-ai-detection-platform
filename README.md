## VeriText AI

Multilingual, paraphrase-resistant AI-generated text detection system with:

- **Cross-lingual semantic drift analysis**
- **Confidence breakdown panel** (burstiness, entropy, syntax depth, semantic drift)
- **Adversarial robustness evaluation** (translation + paraphrase attacks)
- **Training & experiment orchestration**

Built as a **microservices platform** so it looks and behaves like a real production system.

---

## Services overview

- **Service A – Detection Engine (`services/detection-engine`)**
  - FastAPI microservice for inference.
  - Cross-lingual semantic drift analyzer (English ↔ Hindi).
  - Exposes AI probability, label, and model metadata.
  - Port: **8000**

- **Service B – Trainer (`services/trainer`)**
  - Training & experiment orchestrator.
  - Tracks experiments in SQLite (status, config, accuracy, loss).
  - Simulated training loop (pluggable with real training later).
  - Port: **8001**

- **Service C – Eval / Adversarial Service (`services/eval-service`)**
  - Generates adversarial variants (translation + paraphrase).
  - Calls Detection Engine to measure robustness.
  - Computes probability variance and label flip rate.
  - Port: **8002**

The legacy single-service detector UI (`server.py`, `static/index.html`) is still available for local experimentation, but the main project is the **three-service platform** under `services/`.

---

## Quick start (all services with Docker)

1. Build and run all services:

   ```bash
   docker compose -f infra/docker-compose.yml up --build
   ```

2. Open the individual service docs:

   - Detection Engine: `http://localhost:8000/docs`
   - Trainer: `http://localhost:8001/docs`
   - Eval Service: `http://localhost:8002/docs`

---

## Run services locally (without Docker)

### Detection Engine (Service A)

```bash
cd services/detection-engine
pip install -r requirements.txt
uvicorn server:app --reload --port 8000
```

Key endpoints:

- `POST /analyze` – basic AI detection + probability.
- `POST /analyze_crosslingual` – cross-lingual drift (En → Hi → En).
- `GET /` – health + model version.

### Trainer (Service B)

```bash
cd services/trainer
pip install -r requirements.txt
uvicorn server:app --reload --port 8001
```

- `POST /experiments` – start a new training experiment (config-driven).
- `GET /experiments/{id}` – fetch status, accuracy, loss, and config.

### Eval / Adversarial Service (Service C)

```bash
cd services/eval-service
pip install -r requirements.txt
uvicorn server:app --reload --port 8002
```

- `POST /adversarial/generate` – generate translation/paraphrase variants.
- `POST /eval/robustness` – evaluate robustness against those variants.

---

## Infra

- `services/` – all three microservices.
- `infra/docker-compose.yml` – one-command orchestration for Services A, B, and C.
- `frontend/` – React + TypeScript + Tailwind dashboard.
- `README.md` (this file) – main documentation and entrypoint.

You can extend this platform with:

- Model versioning and real training in the Trainer service.
- Additional adversarial attacks (noise, style-shift, multi-step translation) in the Eval service.

---

## Frontend dashboard (React)

The `frontend/` app provides:

- A **hero/overview** page with animated stats.
- An **Analyze** page with explainability cards (glassmorphism, tilt).
- A **Compare** page for side-by-side analysis of two texts.
- A **Modes** page with an “Academic mode” toggle.

### Run the frontend

```bash
cd frontend
npm install           # or pnpm install / yarn
npm run dev           # default: http://localhost:5173
```

During development, the Vite dev server proxies to the three backends:

- `/api/detect/*` → Detection Engine (8000)
- `/api/train/*` → Trainer (8001)
- `/api/eval/*` → Eval Service (8002)
