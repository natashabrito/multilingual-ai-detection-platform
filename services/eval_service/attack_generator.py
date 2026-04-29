from typing import List

from transformers import pipeline


class AttackGenerator:
    """
    Generates adversarial variants of input text via:
    - Translation (currently English → Hindi)
    - Paraphrasing (English)

    You can extend this class with more attack types and languages.
    """

    def __init__(self) -> None:
        self.translator_en_hi = pipeline(
            "translation",
            model="Helsinki-NLP/opus-mt-en-hi",
        )

        self.paraphraser = pipeline(
            "text2text-generation",
            model="Vamsi/T5_Paraphrase_Paws",
        )

    def generate_variants(
        self,
        text: str,
        target_languages: List[str],
        attack_type: str,
    ):
        variants = []

        if attack_type in ("paraphrase", "all"):
            outputs = self.paraphraser(
                f"paraphrase: {text}",
                max_length=128,
                do_sample=True,
                num_return_sequences=2,
                temperature=0.7,
            )
            for out in outputs:
                variants.append({"language": "same", "text": out["generated_text"]})

        return variants

