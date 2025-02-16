import pandas as pd
import json

fair_face_attributes = pd.DataFrame({
    'race': ['b', 'w'],
    'gender': ['m', 'f'],
    'age': ['10', '10']}
)


if __name__ == "__main__":
    attribute_breakdown = {
        'race': fair_face_attributes["race"].value_counts().to_dict(),
        'gender': fair_face_attributes["gender"].value_counts().to_dict(),
        'age': fair_face_attributes["age"].value_counts().to_dict()
    }

    print(attribute_breakdown)
    
    with open(f'attributes_breakdown.json', 'w') as f:
        json.dump(attribute_breakdown, f)