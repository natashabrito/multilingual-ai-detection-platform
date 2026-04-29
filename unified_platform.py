import os
import sys
from pathlib import Path

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from starlette.middleware.cors import CORSMiddleware

# Add services to path so we can import them
base_dir = Path(__file__).resolve().parent
sys.path.append(str(base_dir / "services" / "detection_engine"))
sys.path.append(str(base_dir / "services" / "trainer"))
sys.path.append(str(base_dir / "services" / "eval_service"))

# Import the apps from each service
try:
    from services.detection_engine.server import app as detection_app
    from services.trainer.server import app as trainer_app
    from services.eval_service.server import app as eval_app
except ImportError as e:
    print(f"[ERROR] Failed to import services: {e}")
    sys.exit(1)

app = FastAPI(
    title="VeriText AI Unified Platform",
    description="A unified gateway for detection, training, and evaluation.",
    version="1.0.0"
)

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount the sub-apps
app.mount("/api/detection", detection_app)
app.mount("/api/trainer", trainer_app)
app.mount("/api/eval", eval_app)

# Serve static files
static_dir = base_dir / "static"
if not static_dir.exists():
    static_dir.mkdir()

@app.get("/")
async def read_index():
    return FileResponse(static_dir / "index.html")

if __name__ == "__main__":
    import uvicorn
    print("\n" + "="*50)
    print("🚀 UNIFIED PLATFORM STARTING")
    print("📍 URL: http://localhost:8000")
    print("="*50 + "\n")
    uvicorn.run(app, host="0.0.0.0", port=8000)
