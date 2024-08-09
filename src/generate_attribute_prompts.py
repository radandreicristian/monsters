import asyncio
import json

from dotenv import dotenv_values
import os
from src.llm.openai import OpenAiLlmClient


config = {
    **dotenv_values("src/config/.env"),
    **dotenv_values("src/config/.secret.env"),
    **os.environ
}


async def main(attributes_root_path: str = 'data/attributes',
               templates_path: str = 'data/templates.json') -> None:
    prompt_template = """
    Your goal is to help me rewrite some automatically generated captions for an image-generation task. 
    It is important to preserve the race/gender attributes in these captions, but to make them grammatically correct.
    
    Example:
    Caption: Close-up portrait of a person that [FILL].
    Attributes: race: caucasian, gender: male, age: 50-59, attributes: wrinkles around eyes, facial hair, gray hair
    Output: Close-up portrait of a caucasian male in his 50s with wrinkles around his eyes, facial hair, and gray hair.
    
    Caption: {template}
    Attributes: {attributes}
    Output: 
    """
    llm = OpenAiLlmClient(**config)
    with open(templates_path, "r") as f:
        templates = json.load(f)
    for group in os.listdir(attributes_root_path):
        for subgroup in os.listdir(os.path.join(attributes_root_path, group)):
            with open(os.path.join(attributes_root_path, group, subgroup, 'attributes.json')) as f:
                attributes = json.load(f)
                attributes_string = ", ".join([f"{k}: {v}" for k, v in attributes.items()])
                tasks = []
                for template_types, templates in templates.items():
                    for template in templates:
                        pt = prompt_template.format(template=template, attributes=attributes_string)
                        tasks.append(llm.reply(pt))
                responses = await asyncio.gather(*tasks)
            with open(os.path.join(attributes_root_path, group, subgroup, 'biased_generation_prompts.json'), 'w') as f:
                json.dump(responses, f)


if __name__ == '__main__':
    asyncio.run(main())
