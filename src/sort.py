import json

if __name__ == "__main__":
    with open("data/attributes.json", "r") as f:
        attributes = json.load(f)
    
    attributes = dict(sorted(attributes.items(), key=lambda x: x[0]))
    with open("data/attributes_sorted.json", "w") as f:
        json.dump(attributes, f, indent=2)
