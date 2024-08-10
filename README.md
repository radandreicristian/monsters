# Setup

Create a virtual environment

```
python3 -m venv venv
source ./venv/bin/activate
(venv) pip install -r requirements.txt
```

## Prompt Generation

The following command generates prompts in `data/prompts` based on `data/templates.json` and `data/attributes.json`
```
(venv) python src.generate_prompts
```

## Image Generation 

The following command generates mocked images in `data/images` based on `data/prompts`
```
(venv) python -m src.generate_images [MODEL_ID]
``` 

Models used for the experiments:
- digiplay/AbsoluteReality_v1.8.1
- stabilityai/stable-diffusion-3-medium
- stabilityai/stable-diffusion-xl-base-1.0
- black-forest-labs/FLUX.1-dev
- black-forest-labs/FLUX.1-schnell

To generate mock images (noise) to confirm imge generation works well, use 
- `mock`

## Attribute Extraction

The following command generates a dictionary of common attributes for each subgroup in the `data/images` folder.
If an attribute (e.g. gender, race, age, facial feature) appears in more than half of the images in a subgroup folder (e.g. /sex_abuse/child_abuse), it will appear in an `attributes.json` file located in the root of the subgroup.
```
(venv) python -m src.extract_attributes
```
