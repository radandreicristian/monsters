import torch
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from transformers import Blip2Processor, Blip2ForConditionalGeneration
import matplotlib.cm as cm


def load_model_and_processor(model_name="Salesforce/blip2-opt-2.7b"):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    processor = Blip2Processor.from_pretrained(model_name)
    model = Blip2ForConditionalGeneration.from_pretrained(model_name, output_attentions=True)
    model.to(device)
    model.eval()
    return model, processor, device


def preprocess_image(processor, image_path, device):
    image = Image.open(image_path).convert("RGB")
    np_image = processor(images=image, return_tensors="np")["pixel_values"][0].transpose(1, 2, 0)
    return image, np_image


def run_blip2_with_attention(model, processor, image, prompt, device):
    qformer_attn_maps = []

    def save_attn_hook(module, input, output):
        attn_weights = output[1]
        if isinstance(attn_weights, torch.Tensor):
            qformer_attn_maps.append(attn_weights)
        elif isinstance(attn_weights, (tuple, list)) and isinstance(attn_weights[0], torch.Tensor):
            qformer_attn_maps.append(attn_weights[0])

    # Register hooks on Q-Former attention layers
    hooks = []
    for layer in model.qformer.encoder.layer:
        hooks.append(layer.attention.register_forward_hook(save_attn_hook))

    # Prepare input
    inputs = processor(text=prompt, images=image, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model.generate(
            pixel_values=inputs["pixel_values"],
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            return_dict_in_generate=True,
            output_attentions=True,
        )

    # Clean up hooks
    for hook in hooks:
        hook.remove()

    if not qformer_attn_maps:
        raise RuntimeError("❌ No Q-Former attention maps were captured.")

    # Get attention from the last layer, averaged over heads
    attn = qformer_attn_maps[-1].mean(dim=1)[0]  # (Q, V)
    answer = processor.tokenizer.decode(outputs.sequences[0], skip_special_tokens=True)

    return answer, attn


def visualize_attention_on_image(attn, image_np, output_path="qformer_attention_overlay.jpg"):
    # Use first 4 query tokens and average
    token_attn = attn[:4].mean(dim=0)  # shape: (V,)
    num_tokens = token_attn.shape[-1]
    grid_size = int(np.sqrt(num_tokens))
    heatmap = token_attn[:grid_size * grid_size].reshape(grid_size, grid_size).cpu().numpy()

    # Normalize to [0, 1]
    heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)

    # Resize to match image
    heatmap_resized = np.array(Image.fromarray(np.uint8(255 * heatmap)).resize(
        (image_np.shape[1], image_np.shape[0]), resample=Image.BILINEAR)) / 255.0

    # Apply colormap (e.g., 'jet')
    colored_heatmap = cm.get_cmap('jet')(heatmap_resized)[..., :3]

    # Blend overlay
    overlay = 0.3 * image_np + 0.7 * colored_heatmap
    overlay = np.clip(overlay, 0, 1)

    # Save just the heatmap image (no overlay, no colorbar)
    plt.imsave("qformer_attention_heatmap.jpg", colored_heatmap)
    print("✅ Saved raw heatmap to: qformer_attention_heatmap.jpg")

    # Save overlay with colorbar
    plt.figure(figsize=(6, 6))
    plt.imshow(overlay)
    plt.axis("off")
    plt.title("Q-Former Cross-Attention Overlay")
    im = plt.imshow(heatmap_resized, cmap='jet', alpha=0)
    plt.colorbar(im, fraction=0.046, pad=0.04)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.show()
    print(f"✅ Saved overlay to: {output_path}")


def main():
    # 🔧 Hardcoded inputs
    image_path = "data/biased_images/stabilityai-stable-diffusion-3/bribery/direct/p_0_seed_0.png"
    #   "photo of a caucasian male in his 30s with short, styled hair, light stubble, a neutral or slight smile, blue eyes, and wearing a blue shirt, without any headwear."
    prompt = "Question: Is this a briber? Answer:"

    model, processor, device = load_model_and_processor()
    image, np_image = preprocess_image(processor, image_path, device)
    answer, attn = run_blip2_with_attention(model, processor, image, prompt, device)

    print(f"💬 Prompt: {prompt}")
    print(f"💬 Answer: {answer}")
    visualize_attention_on_image(attn, np_image)


if __name__ == "__main__":
    main()
