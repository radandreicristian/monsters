import json

if __name__ == "__main__":
    with open("attributes.json", "r") as f:
        attributes = json.load(f)
    
    
    attributes = json.load(open("attributes.json", "r"))