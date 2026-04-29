from typing import List

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from experiment_manager import create_experiment, get_experiment


app = FastAPI(
    title="Training & Experiment Orchestrator",
    version="1.0.0",
)


class ExperimentConfig(BaseModel):
    dataset_name: str
    dataset_content: str = ""
    epochs: int
    learning_rate: float
    languages: List[str]
    task_type: str = "Binary classification"


@app.post("/experiments")
def create_new_experiment(config: ExperimentConfig):
    experiment_id = create_experiment(config.model_dump())
    return {
        "experiment_id": experiment_id,
        "status": "started",
    }


@app.get("/experiments/{experiment_id}")
def fetch_experiment(experiment_id: str):
    experiment = get_experiment(experiment_id)
    if not experiment:
        raise HTTPException(status_code=404, detail="Experiment not found")
    return experiment

