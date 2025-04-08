import os
import json
import matplotlib.pyplot as plt
import numpy as np

# Parameters
BASE_DIR = "data/attributes/stabilityai-stable-diffusion-3"
BASELINE_PATH = "data/control_images/stabilityai-stable-diffusion-3/attributes_breakdown.json"
FORMULATION = "direct"
TARGET_KEY = "race"
TARGET_CONCEPTS = ["gang_activity", "gang_violence", "plagiarism", "terrorism", "drug_trafficking"]
OUTPUT_PATH = os.path.join(BASE_DIR, f"{FORMULATION}_{TARGET_KEY}_distribution_select.png")

# Mappings
concepts_mapping = json.load(open("data/concepts_mapping.json", "r"))

labels_mapping = {
    "race": {
        "caucasian": "Caucasian",        
        "middle eastern": "Middle-Eastern",
        "latino-hispanic": "Latino-Hispanic", 
        "indian": "Indian",
        "afro american": "Afro-American",
        "east asian": "East-Asian"
    }
}

LABEL_COLOR_MAP = {
    "caucasian": "#1f77b4",        
    "middle eastern": "#ff7f0e",
    "latino-hispanic": "#2ca02c", 
    "indian": "#d62728",
    "afro american": "#9467bd",
    "east asian": "#f1c40f"
}

# Aggregate data
all_data = []
labels = set()

# Load baseline
try:
    with open(BASELINE_PATH, "r") as f:
        baseline_data = json.load(f).get(TARGET_KEY, {})
    print("Loaded baseline")
except Exception as e:
    print(f"Error loading baseline: {e}")
    baseline_data = {}

all_data.append(baseline_data)
labels.update(baseline_data.keys())

# Load selected concept data
for folder in TARGET_CONCEPTS:
    json_path = os.path.join(BASE_DIR, folder, FORMULATION, "attributes_breakdown.json")
    try:
        with open(json_path, "r") as f:
            data = json.load(f)
        breakdown = data.get(TARGET_KEY, {})
        all_data.append(breakdown)
        labels.update(breakdown.keys())
    except FileNotFoundError:
        print(f"File not found: {json_path}")
        all_data.append({})
    except json.JSONDecodeError:
        print(f"Invalid JSON in: {json_path}")
        all_data.append({})

# Exit early if no labels found
if not labels:
    raise ValueError(f"No data found for target key '{TARGET_KEY}' in baseline or selected concepts.")

labels = sorted(labels)
label_indices = {label: idx for idx, label in enumerate(labels)}

# Normalize values
values_matrix = np.zeros((len(all_data), len(labels)))
for i, data_dict in enumerate(all_data):
    total = sum(data_dict.values())
    if total > 0:
        for key, value in data_dict.items():
            if key in label_indices:
                values_matrix[i][label_indices[key]] = (value / total) * 100

# Sort by total sum
total_per_label = values_matrix.sum(axis=0)
sorted_label_indices = np.argsort(-total_per_label)

# Plotting
n_bars = len(all_data) + 1
x = np.arange(n_bars, dtype=float)
x[1:] += 0.5  # space after baseline

# ✅ Use concepts_mapping for readable concept names
mapped_concepts = [concepts_mapping.get(c, c) for c in TARGET_CONCEPTS]
bar_labels = ["Baseline"] + [""] + mapped_concepts
bar_positions = x

# Insert spacing row
values_matrix_with_space = np.insert(values_matrix, 1, 0, axis=0)

plt.figure(figsize=(3.25, 2.5))  # AAAI single column

bottom = np.zeros(n_bars)
for idx in sorted_label_indices:
    label_raw = labels[idx]
    label_cap = labels_mapping.get(TARGET_KEY, {}).get(label_raw, label_raw.title())
    color = LABEL_COLOR_MAP.get(label_raw.lower(), "#999999")
    plt.bar(bar_positions, values_matrix_with_space[:, idx], bottom=bottom, label=label_cap, color=color)
    bottom += values_matrix_with_space[:, idx]

# ✅ X-axis and label styling to match "all" version
plt.xticks(bar_positions, bar_labels, rotation=45, ha='right', fontsize=6)
plt.yticks(fontsize=6)
plt.ylabel('Percentage (%)', fontsize=7)
plt.ylim(0, 110)
plt.legend(loc='center left', bbox_to_anchor=(1.0, 0.5), fontsize=6)
plt.tight_layout()
plt.grid(True, axis='y', linestyle='--', linewidth=0.5)

plt.savefig(OUTPUT_PATH, dpi=600, bbox_inches='tight')
print(f"Saved plot to: {OUTPUT_PATH}")
