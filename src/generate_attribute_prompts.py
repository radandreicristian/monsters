import asyncio
import json

from dotenv import dotenv_values
import os
from src.llm.openai import OpenAiLlmClient
from src.utils.prompts import add_attributes_to_template_prompt

config = {
    **dotenv_values("src/config/.env"),
    **dotenv_values("src/config/.secret.env"),
    **os.environ
}


async def main() -> None:

    templates_path = config["TEMPLATES_PATH"]
    attributes_root_path = config["ATTRIBUTES_ROOT_PATH"]

    llm = OpenAiLlmClient(**config)

    with open(templates_path, "r") as f:
        templates = json.load(f)

    for group in os.listdir(attributes_root_path):
        for subgroup in os.listdir(os.path.join(attributes_root_path, group)):
            with open(os.path.join(attributes_root_path, group, subgroup, 'attributes.json')) as f:
                attributes_dict = json.load(f)
                attributes = ", ".join([f"{k}: {v}" for k, v in attributes_dict.items()])
                tasks = []

                # No point doing it for both descriptive and label, we'll end up with the same prompt
                label_templates = templates["label"]
                for template in label_templates:
                    prompt = add_attributes_to_template_prompt.format(template=template,
                                                                      attributes=attributes)
                    tasks.append(llm.reply(prompt))
                responses = await asyncio.gather(*tasks)
            with open(os.path.join(attributes_root_path, group, subgroup, 'biased_generation_prompts.json'), 'w') as f:
                json.dump(responses, f)


if __name__ == '__main__':
    asyncio.run(main())
