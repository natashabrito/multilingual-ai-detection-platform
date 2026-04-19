from typing import List

from fastapi import FastAPI
from pydantic import BaseModel

from attack_generator import AttackGenerator
from robustness_evaluator import RobustnessEvaluator


app = FastAPI(title="Adversarial & Robustness Evaluation Service")

attack_gen = AttackGenerator()
evaluator = RobustnessEvaluator()


class AttackRequest(BaseModel):
    text: str
    target_languages: List[str]
    attack_type: str  # "translation" or "paraphrase"


@app.post("/adversarial/generate")
def generate_attack(request: AttackRequest):
    variants = attack_gen.generate_variants(
        request.text,
        request.target_languages,
        request.attack_type,
    )
    return {
        "original_text": request.text,
        "variants": variants,
    }


@app.post("/eval/robustness")
def evaluate_robustness(request: AttackRequest):
    variants = attack_gen.generate_variants(
        request.text,
        request.target_languages,
        request.attack_type,
    )
    result = evaluator.evaluate(request.text, variants)
    return result

