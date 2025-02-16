import json

if __name__ == "__main__":
    with open("data/concepts.json", "r") as f:
        attributes = json.load(f)
    
    attributes = dict(sorted(attributes.items(), key=lambda x: x[0]))
    with open("data/concepts_sorted.json", "w") as f:
        json.dump(attributes, f, indent=2)
