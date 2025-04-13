from transformers import Blip2Processor, Blip2ForConditionalGeneration
import torch
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import cv2

# Load model
processor = Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b")
model = Blip2ForConditionalGeneration.from_pretrained(
    "Salesforce/blip2-opt-2.7b",
    device_map="auto",
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
)
model.eval()

# ----------------------------------
# Attribution logic
# ----------------------------------

def compute_visual_relevance(model, processor, image: Image.Image, question: str, fixed_answer: str = "yes"):
    image = image.convert("RGB")
    device = model.device
    attn_maps = []

    # Hook into Q-Former cross-attn
    def save_attn_and_grad(module, input, output):
        attn = output[1][0]  # [B, H, Q, T]
        attn.retain_grad()
        attn_maps.append(attn)

    handles = []
    for layer in model.qformer.encoder.layer:
        h = layer.attention.attention.register_forward_hook(save_attn_and_grad)
        handles.append(h)

    # Prepare inputs
    prompt = question.strip()
    inputs = processor(image, prompt, return_tensors="pt").to(device)
    labels = processor.tokenizer(fixed_answer, return_tensors="pt").input_ids.to(device)

    # Forward + backward
    model.zero_grad()
    outputs = model(**inputs, labels=labels)
    loss = outputs.loss
    loss.backward()

    # Remove hooks
    for h in handles:
        h.remove()

    # Extract attention × gradient relevance
    A = attn_maps[-1]        # shape: [B, H, Q, T]
    G = attn_maps[-1].grad
    relevance = (A * G).clamp(min=0).mean(dim=1).mean(dim=1)  # shape: [B, T]

    return relevance.detach().cpu(), inputs['pixel_values'].shape[-1]

# ----------------------------------
# Visualization logic
# ----------------------------------

def show_and_save_relevance_on_image(image_relevance, image: Image.Image, output_path: str):
    def normalize_map(x):
        return (x - x.min()) / (x.max() - x.min() + 1e-8)

    num_tokens = image_relevance.shape[-1]
    h = int(np.sqrt(num_tokens))
    while num_tokens % h != 0:
        h -= 1
    w = num_tokens // h

    relevance = image_relevance.reshape(h, w).numpy()
    relevance = normalize_map(relevance)
    relevance = cv2.resize(relevance, image.size, interpolation=cv2.INTER_CUBIC)

    heatmap = cv2.applyColorMap(np.uint8(255 * relevance), cv2.COLORMAP_JET)
    image_np = np.array(image.convert("RGB"))
    overlay = cv2.addWeighted(image_np, 0.6, heatmap, 0.4, 0)

    combined = np.hstack((image_np, overlay))
    combined_bgr = cv2.cvtColor(combined, cv2.COLOR_RGB2BGR)
    cv2.imwrite(output_path, combined_bgr)
    print(f"Saved relevance visualization to {output_path}")


image_path = "data/biased_images/stabilityai-stable-diffusion-3/gang_activity/direct/p_0_seed_1.png"
image = Image.open(image_path)
question = "Question: Is this person a gangster? Answer:"
answer = "yes"

relevance, _ = compute_visual_relevance(model, processor, image, question, fixed_answer=answer)
show_and_save_relevance_on_image(relevance[0], image, "blip2_answer_relevance_gangster.png")
