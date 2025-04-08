import os
import json
import numpy as np
from collections import defaultdict

# Parameters
BASE_DIR = "data/attributes/stabilityai-stable-diffusion-3"
FORMULATIONS = ["direct", "indirect"]
TARGET_KEYS = ["race", "gender", "age"]

# Entropy function
def compute_entropy(counts_dict):
    total = sum(counts_dict.values())
    if total == 0:
        return 0.0
    probs = np.array([v / total for v in counts_dict.values() if v > 0])
    return -np.sum(probs * np.log2(probs))

# Get concept folders
concept_folders = sorted([
    f for f in os.listdir(BASE_DIR)
    if os.path.isdir(os.path.join(BASE_DIR, f))
])

# Initialize results
entropy_results = {}

for formulation in FORMULATIONS:
    for target_key in TARGET_KEYS:
        aggregate_counts = defaultdict(int)

        for concept in concept_folders:
            if concept in ("child_abuser", "human_trafficking", "kidnapping", "molester", "murder", "rape", "scamming"):
                print('skipping child abuse')
                continue
            json_path = os.path.join(BASE_DIR, concept, formulation, "attributes_breakdown.json")
            try:
                with open(json_path, "r") as f:
                    data = json.load(f)
                sub_counts = data.get(target_key, {})
                for k, v in sub_counts.items():
                    aggregate_counts[k.lower()] += v
            except Exception as e:
                print(f"Skipping {concept}/{formulation}: {e}")
                continue

        entropy = compute_entropy(aggregate_counts)
        key = f"{formulation}_{target_key}"
        entropy_results[key] = {
            "entropy": round(entropy, 4),
            "total": sum(aggregate_counts.values()),
            "categories": dict(aggregate_counts)
        }

# Print results
print("\n📊 Entropy Results Across All Concepts\n")
for key, stats in entropy_results.items():
    print(f"{key}:")
    print(f"  Entropy: {stats['entropy']}")
    print(f"  Total samples: {stats['total']}")
    print(f"  Category distribution: {stats['categories']}")
    print()
