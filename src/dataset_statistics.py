import json

if __name__ == "__main__":
    with open("data/concepts.json", "r") as f:
        attributes = json.load(f)
    
    with open("data/templates.json", "r") as f:
        templates = json.load(f)

    total = 0
    for group_name, group_content in attributes.items():
        print(f"{group_name}: {len(group_content)} ")
        total += len(group_content)
    print(f"Total attributes: {total}")

    for prompt_type, prompts in templates.items():
        print(f"Prompt type: {prompt_type} - {len(prompts)}")
    print(f"Total prompts: {sum(len(k) for k in templates.values())}")