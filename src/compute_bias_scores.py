import typer
from typing import Literal
from typing_extensions import Annotated
import os
import re
import json

from src.vqa import VqaFactory
from src.utils.mapping import id_to_local_name
from src.logger.utils import get_logger
from src.utils.constants import FORMULATION_TYPES

def bias(yes, no):
    return yes / (yes + no)

def answer_questions(image_generation_model_id: str):
    """
    Perform VQA on images generate by a model. The images can be either control images or biased images.
    
    """
    model_name = id_to_local_name[image_generation_model_id]
    
    with open("data/concepts.json", "r") as f:
        concepts = json.load(f)
    
    baseline_data = {}
    concept_data = {}
    adversarial_data = {}
    fairface_data = {}
    fairface_balanced_data = {}

    for concept in concepts:
        with open(f"data/control_images/{model_name}/answers/blip2/{concept}.json", "r") as f:
            baseline_concept_scores = json.load(f)
            yes_base, no_base = baseline_concept_scores.get("yes", 0), baseline_concept_scores.get("no", 0)
            for key, value in baseline_concept_scores.items():
                baseline_data[key] = baseline_data.get(key, 0) + value
        with open(f"data/concept_images/{model_name}/{concept}/direct/answers/blip2/{concept}.json", "r") as f:
            concept_concept_scores = json.load(f)
            yes_concept, no_concept = concept_concept_scores.get("yes", 0), concept_concept_scores.get("no", 0)
            for key, value in concept_concept_scores.items():
                concept_data[key] = concept_data.get(key, 0) + value
        with open(f"data/biased_images/{model_name}/{concept}/direct/answers/blip2/{concept}.json", "r") as f:
            biased_concept_scores = json.load(f)
            yes_adv, no_adv = biased_concept_scores.get("yes", 0), biased_concept_scores.get("no", 0)
            for key, value in biased_concept_scores.items():
                adversarial_data[key] = adversarial_data.get(key, 0) + value
        with open(f"data/fairface_images/answers/blip2/{concept}.json", "r") as f:
            fairface_scores = json.load(f)
            yes_ff, no_ff = fairface_scores.get("yes", 0), fairface_scores.get("no", 0)
            for key, value in fairface_scores.items():
                fairface_data[key] = fairface_data.get(key, 0) + value
        with open(f"data/fairface_balanced/answers/blip2/{concept}.json", "r") as f:
            fairface_balanced_scores = json.load(f)
            yes_ff_bal, no_ff_bal = fairface_balanced_scores.get("yes", 0), fairface_balanced_scores.get("no", 0)
            for key, value in fairface_balanced_scores.items():
                fairface_balanced_data[key] = fairface_balanced_data.get(key, 0) + value
        
        # print(f"Concept: {concept} -  Fairface. Yes {yes_ff}, No: {no_ff}")
        # print(f"Concept: {concept} -  Fairface Balanced. Yes {yes_ff}, No: {no_ff}")
        # print(f"Concept: {concept} - Baseline. Yes: {yes_base}, No: {no_base}")
        # print(f"Concept: {concept} - Adversarial. Yes: {yes_adv}, No: {no_adv}")
        # print(f"Concept: {concept} - Concept. Yes: {yes_concept}, No: {no_concept}")

        print(f"Concept: {concept} -  Fairface. Bias: {bias(yes_ff_bal, no_ff_bal)}")
        # print(f"Concept: {concept} -  Fairface. Bias: {bias(yes_ff, no_ff)}")
        print(f"Concept: {concept} - Baseline. Bias: {bias(yes_base, no_base)}")
        print(f"Concept: {concept} - Adversarial. Bias: {bias(yes_adv, no_adv)}")
        print(f"Concept: {concept} - Concept. Bias: {bias(yes_concept, no_concept)}")
        print("\n")
           
    print("Baseline - Default generation", baseline_data)
    print("Baseline - Fairface validation", fairface_data)
    print("Baseline - Fairface validation, balanced", fairface_balanced_data)

    print("Concept", concept_data)
    print("Adversarial", adversarial_data)

if __name__ == '__main__':
    typer.run(answer_questions)