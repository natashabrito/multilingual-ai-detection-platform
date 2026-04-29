import json
import threading

from database import Experiment, SessionLocal
from worker import run_experiment


def create_experiment(config):
    db = SessionLocal()

    experiment = Experiment(
        status="created",
        config=json.dumps(config),
    )

    db.add(experiment)
    db.commit()
    db.refresh(experiment)
    experiment_id = experiment.id
    db.close()

    # Run in a background thread so we don't block the API
    t = threading.Thread(target=run_experiment, args=(experiment_id, config))
    t.start()

    return experiment_id


def get_experiment(experiment_id):
    db = SessionLocal()
    experiment = db.query(Experiment).filter(Experiment.id == experiment_id).first()
    db.close()

    if not experiment:
        return None

    return {
        "id": experiment.id,
        "status": experiment.status,
        "accuracy": experiment.accuracy,
        "loss": experiment.loss,
        "val_accuracy": experiment.val_accuracy,
        "progress": experiment.progress,
        "epoch_progress": experiment.epoch_progress,
        "model_path": experiment.model_path,
        "timestamp": experiment.timestamp,
        "config": experiment.config,
    }

