from transformers import Blip2Processor, Blip2ForConditionalGeneration
import torch
from PIL import Image
import requests
from torchvision import transforms
import matplotlib.pyplot as plt
import cv2
import numpy as np

# Load processor and model
processor = Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b")
model = Blip2ForConditionalGeneration.from_pretrained("Salesforce/blip2-opt-2.7b", device_map="auto", torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32)
model.eval()

def infer_grid_shape(model, processor, image: Image.Image):
    inputs = processor(images=image, return_tensors="pt").to(model.device)
    with torch.no_grad():
        vision_outputs = model.vision_model(pixel_values=inputs["pixel_values"])
        num_tokens = vision_outputs.last_hidden_state.shape[1] - 1  # exclude CLS
    h = int(np.sqrt(num_tokens))
    while num_tokens % h != 0:
        h -= 1
    w = num_tokens // h
    print(f"Inferred grid shape: ({h}, {w}) from {num_tokens} tokens")
    return h, w


def interpret_blip2(model, processor, image: Image.Image, question: str, layer_start: int = -1):
    model.eval()

    attn_maps = []

    # Hook to capture attention weights and retain their gradients
    def save_attn_and_grad(module, input, output):
        _, attn_output = output           # output = (hidden_states, (attn_weights, ...))
        attn_weights = attn_output[0]     # extract attn_weights
        attn_weights.retain_grad()
        attn_maps.append(attn_weights)

    # Register hooks
    handles = [
        layer.attention.attention.register_forward_hook(save_attn_and_grad)
        for layer in model.qformer.encoder.layer
    ]

    # Prepare input
    inputs = processor(image, question, return_tensors="pt").to(model.device)
    input_ids = inputs.get("input_ids", None)

    with torch.enable_grad():
        outputs = model(**inputs, labels=input_ids)
        loss = outputs.loss
        loss.backward()

    # Remove hooks
    for h in handles:
        h.remove()

    # Extract batch/query/token sizes
    batch_size, num_heads, num_queries, num_tokens = attn_maps[0].shape
    R = torch.eye(num_tokens, dtype=attn_maps[0].dtype, device=attn_maps[0].device).unsqueeze(0).repeat(batch_size, 1, 1)

    if layer_start == -1:
        layer_start = 0

    for layer_idx in reversed(range(layer_start, len(attn_maps))):
        A = attn_maps[layer_idx]         # [B, H, Q, T]
        G = attn_maps[layer_idx].grad    # same shape

        cam = (A * G).clamp(min=0).mean(dim=1)  # avg heads → [B, Q, T]
        cam_mean = cam.mean(dim=1)              # avg queries → [B, T]
        cam_map = cam_mean.unsqueeze(2)         # [B, T, 1] to match R for bmm
        R = R + cam_map * R                     # propagate only diagonal relevance

    # Final relevance per image token
    image_relevance = R.mean(dim=1)  # [B, T]
    return image_relevance.detach().cpu(), num_tokens


def show_blip2_image_relevance(image_relevance, image: Image.Image, output_path: str, num_tokens: int):
    def normalize_map(x):
        return (x - x.min()) / (x.max() - x.min() + 1e-8)

    h = int(np.sqrt(num_tokens))
    while num_tokens % h != 0:
        h -= 1
    w = num_tokens // h

    print(f"Reshaping relevance map as ({h}, {w})")
    relevance = image_relevance.reshape(h, w).numpy()

    w_img, h_img = image.size
    relevance = normalize_map(relevance)
    relevance = cv2.resize(relevance, (w_img, h_img), interpolation=cv2.INTER_CUBIC)

    heatmap = cv2.applyColorMap(np.uint8(255 * relevance), cv2.COLORMAP_JET)
    image_np = np.array(image.convert("RGB"))
    overlay = cv2.addWeighted(image_np, 0.6, heatmap, 0.4, 0)

    combined = np.hstack((image_np, overlay))
    combined_bgr = cv2.cvtColor(combined, cv2.COLOR_RGB2BGR)
    cv2.imwrite(output_path, combined_bgr)
    print(f"Saved visualization to {output_path}")



image_path = "data/fairface_balanced/24.jpg"

image = Image.open(image_path).convert("RGB")
question = "Question: Is this person wearing glasses? Answer:"

# Visualize
output_file = "blip2_relevance_output.png"

image_relevance, num_tokens = interpret_blip2(model, processor, image, question)
show_blip2_image_relevance(image_relevance[0], image, output_file, num_tokens)