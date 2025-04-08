import os
import json
import matplotlib.pyplot as plt
import numpy as np

# Parameters
BASE_DIR = "data/attributes/stabilityai-stable-diffusion-3"
BASELINE_PATH = "data/control_images/stabilityai-stable-diffusion-3/attributes_breakdown.json"

FORMULATION = "indirect"
TARGET_KEY = "age"  # change to 'gender' or 'race' if desired

target_keys = ["gender", "race", "age"]
formulations = ["direct", "indirect"]

labels_mapping = {
    "race": {
        "caucasian": "Caucasian",        
        "middle eastern": "Middle-Eastern",
        "latino-hispanic": "Latino-Hispanic", 
        "indian": "Indian",
        "afro american": "Afro-American",
        "east asian": "East-Astian"
    },
    "gender": {
        "male": "Male",
        "female": "Female"
    }
}

LABEL_COLOR_MAP = {
    # Race
    "caucasian": "#1f77b4",        
    "middle eastern": "#ff7f0e",
    "latino-hispanic": "#2ca02c", 
    "indian": "#d62728",
    "afro american": "#9467bd",
    "east asian": "#f1c40f",

    # Gender
    "male": "#1f77b4",     # blue
    "female": "#ff7f0e", # orange

    # Age
    "0-2":    "#7f7f7f",  # gray
    "3-9":    "#2ca02c",  # green
    "10-19": "#d62728",  # red
    "30-39": "#1f77b4",  # blue
    "20-29": "#ff7f0e",  # yellow
    "40-49": "#9467bd",  # purple
    "50-59": "#8c564b",  # brown
    "60-69": "#e377c2",  # pink
    "70+":   "#17becf"   # teal
}


OUTPUT_PATH = os.path.join(BASE_DIR, f"{TARGET_KEY}_{FORMULATION}_all.png")

# Get sorted list of folders
folders = sorted([f for f in os.listdir(BASE_DIR) if os.path.isdir(os.path.join(BASE_DIR, f))])

# Aggregate data
all_data = []
labels = set()

concepts_mapping = json.load(open("data/concepts_mapping.json", "r"))

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

# Load folders
for folder in folders:
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

# Exit early if no valid labels
if not labels:
    raise ValueError(f"No data found for target key '{TARGET_KEY}' in baseline or any folder.")

labels = sorted(labels)
label_indices = {label: idx for idx, label in enumerate(labels)}

# Build normalized matrix
values_matrix = np.zeros((len(all_data), len(labels)))
for i, data_dict in enumerate(all_data):
    total = sum(data_dict.values())
    if total > 0:
        for key, value in data_dict.items():
            if key in label_indices:
                values_matrix[i][label_indices[key]] = (value / total) * 100  # normalize

# Sort labels by total sum
total_per_label = values_matrix.sum(axis=0)
sorted_label_indices = np.argsort(-total_per_label)

# Plotting
n_bars = len(all_data) + 1  # +1 for spacing
x = np.arange(n_bars, dtype=float)
x[1:] += 0.15  # shift everything after baseline

mapped_folders = [concepts_mapping.get(f, f) for f in folders]
bar_labels = ["Baseline"] + [""] + mapped_folders
bar_positions = x

# Insert empty row for spacing in values_matrix
values_matrix_with_space = np.insert(values_matrix, 1, 0, axis=0)

plt.figure(figsize=(max(12, n_bars * 0.25), 6))

bottom = np.zeros(n_bars)
for idx in sorted_label_indices:
    label_raw = labels[idx]
    label_pretty = labels_mapping.get(TARGET_KEY, {}).get(label_raw, label_raw)
    color = LABEL_COLOR_MAP.get(label_raw, None)  # fallback to default if not found
    plt.bar(bar_positions, values_matrix_with_space[:, idx], bottom=bottom, width=0.45, label=label_pretty, color=color)

    bottom += values_matrix_with_space[:, idx]

plt.xticks(bar_positions, bar_labels, rotation=45, ha='right', fontsize=13)
plt.ylabel('Percentage (%)', fontsize=14)
plt.title(f'{TARGET_KEY.capitalize()} Distribution - {FORMULATION.capitalize()}', fontsize=15)
plt.ylim(0, 110)

# Move legend outside the plot
plt.legend(loc='center left', bbox_to_anchor=(1.01, 0.5), fontsize=11)

plt.tight_layout()
plt.grid(True, axis='y', linestyle='--', linewidth=0.5)

plt.savefig(OUTPUT_PATH, dpi=300, bbox_inches='tight')  # ensure legend is not cut off
print(f"Saved plot to: {OUTPUT_PATH}")
