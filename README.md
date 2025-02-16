# Uncovering Hidden Biaseses Related to Illegal, Unethical and Stigmatized Behaviours in Text-Image Multimodal Models

This repository is a collection of experiments regarding representations of illegal, unethical and stigmatized behaviours in text-to-image models, and consists of the following main contributions

- Dataset [wip]
- Bias in generation [wip]
- Bias in VQA [wip]
- Victim/agressor confusion
- Formulation bias (e.g. very specific formulations wrt activities)

# Setup

Build the Docker container (with CUDA support).

```
docker build -t <IMAGE_NAME> .
```

Run the Docker container.

```
docker run --name <CONTAINER_NAME> --gpus all --it <IMAGE_NAME>
```

To run outside a Docker context (not recommended), create a virtual environment and install the requirements

```
python3 -m venv venv
source ./venv/bin/activate
(venv) pip install -r requirements.txt
```

# Experiments

## Prompt Generation

The following command generates prompts based on `data/templates.json` and `data/concepts.json` in in `data/prompts` in `data/prompts`

```
python -m src.generate_prompts
```

The total number of prompts is n_templates x n_attributes

## Image Generation 

The following command generates images based on the prompts located in `data/prompts` in `data/images`
```
python -m src.generate_images [MODEL_ID]
``` 

Models used for the experiments:
- `digiplay/AbsoluteReality_v1.8.1` (2.13 GB model size) 
- `stabilityai/stable-diffusion-3.5-medium` (5.11 GB model size)
- `black-forest-labs/FLUX.1-schnell` (23.8 GB model size)

- mock (for mock images)

## Attribute Extraction

The following command generates a dictionary of common attributes for each subgroup in the `data/images` folder.
If an attribute (e.g. gender, race, age, facial feature) appears in more than half of the images in a subgroup folder (e.g. /sex_abuse/child_abuse), it will appear in an `concepts.json` file located in the root of the subgroup.

```
python -m src.extract_attributes
```

## 