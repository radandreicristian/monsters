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

## Image Generation (mocked)

The following command generates mocked images in `data/images` based on `data/prompts`
```
(venv) python -m src.generate_images
``` 