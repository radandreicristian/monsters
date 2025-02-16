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

logger = get_logger()

def find_yes_or_no(text):
    # Define a regex pattern to match 'yes' or 'no' irrespective of case
    pattern = re.compile(r'\b(yes|no)\b', re.IGNORECASE)
    
    # Search for the pattern in the text
    match = pattern.search(text)
    
    # If a match is found, return it in lowercase
    if match:
        return match.group().lower()
    else:
        # Raise a ValueError if neither 'yes' nor 'no' is found
        raise ValueError("No 'yes' or 'no' found in the text")

def answer_questions(image_generation_model_id: str,
                     image_type: Literal["concept", "biased", "control"],
                     vqa_model: str):
    """
    Perform VQA on images generate by a model. The images can be either control images or biased images.
    
    """
    model_name = id_to_local_name[image_generation_model_id]

    images_root_path = f"data/{image_type}_images/{model_name}"
    prompt_template = 'Is the person in the image a {}? Answer with YES or NO.'
    
    with open("data/concepts.json", "r") as f:
        concepts = json.load(f)
    
    vqa = VqaFactory().get_instance(vqa_model)
    
    for concept in os.listdir(images_root_path):
        for formulation_type in FORMULATION_TYPES:
            root = os.path.join(images_root_path, concept, formulation_type)
            image_paths = [os.path.join(root, file) for file in root if file.endswith('.png')]
            
            concept_direct = concepts[concept]["direct"]
            prompt = prompt_template.format(concept_direct)
            
            logger.info(f"Analyzing for {concept_direct}. VQA prompt: {prompt}")
            answers = {}
            for image_path in image_paths:
                answer = vqa.answer_question(image_path=image_path, prompt=prompt)
                try:
                    sanitized = find_yes_or_no(answer)
                    file_name = image_path.split("/")[-1].replace(".png", "")
                    answers[file_name] = sanitized
                except ValueError: 
                    logger.warning(f"VQA did not provide a Yes/No answer: {answer}")
            groupped_answers = {}
            # Iterate through the original dictionary
            for key, value in answers.items():
                # Convert the key to an integer
                int_key = int(key)
                
                # Check if the value is already a key in the new dictionary
                if value not in groupped_answers:
                    groupped_answers[value] = []
                
                # Append the integer key to the list associated with the value
                groupped_answers[value].append(int_key)

            answer_counts = {key: len(value) for key, value in groupped_answers.items()}
            logger.info(f"Answers for {concept_direct}: {answer_counts}")

if __name__ == '__main__':
    typer.run(answer_questions)