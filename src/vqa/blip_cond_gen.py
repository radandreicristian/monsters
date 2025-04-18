from typing import List, Dict
from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
import torch
import torch.nn.functional as F

class BlipVqaWithScoring:

    def __init__(self):
        self.processor = BlipProcessor.from_pretrained("Salesforce/blip-vqa-base")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-vqa-base").to(self.device)

        self.answer_variants = {
            "yes": ["yes", "Yes"],
            "no": ["no", "No"],
            "unknown": ["Unknown", "unknown"]
        }

    def score_answers(self, image_paths: List[str], prompt: str, verbose: bool = False) -> List[Dict]:
        images = [Image.open(p).convert("RGB") for p in image_paths]
        results = [{} for _ in image_paths]

        for canonical_class, variants in self.answer_variants.items():
            class_log_probs = [0.0 for _ in image_paths]

            for variant in variants:
                # Encode input (image + question)
                inputs = self.processor(
                    images=images,
                    text=[prompt] * len(images),
                    return_tensors="pt",
                    padding="max_length",
                    max_length=20
                ).to(self.device)

                # Manually encode the target answers
                labels = self.processor.tokenizer(
                    [variant] * len(images),
                    return_tensors="pt",
                    padding="max_length",
                    max_length=20,
                    truncation=True
                ).input_ids.to(self.device)

                labels[labels == self.processor.tokenizer.pad_token_id] = -100
                inputs["labels"] = labels

                with torch.no_grad():
                    outputs = self.model(**inputs)
                    logits = outputs.logits

                log_probs = F.log_softmax(logits, dim=-1)

                for i in range(labels.size(0)):
                    label_ids = labels[i]
                    token_log_probs = log_probs[i, torch.arange(len(label_ids)), label_ids]
                    mask = (label_ids != -100)
                    seq_log_prob = token_log_probs[mask].sum().item()
                    class_log_probs[i] += seq_log_prob

                    if verbose:
                        decoded = self.processor.tokenizer.decode(label_ids[mask])
                        print(f"[{canonical_class}] variant='{variant}' → '{decoded}' → log_prob={seq_log_prob:.2f}")

            # Store accumulated log-probs
            for i in range(len(image_paths)):
                results[i][canonical_class] = class_log_probs[i]

        # Normalize to probabilities with softmax
        for res in results:
            values = torch.tensor([res[c] for c in self.answer_variants])
            probs = F.softmax(values, dim=0)
            for i, c in enumerate(self.answer_variants):
                res[c] = probs[i].item()

        return results


if __name__ == '__main__':
    vqa = BlipVqaWithScoring()
    result = vqa.score_answers(
        ["data/biased_images/stabilityai-stable-diffusion-3/drug_trafficking/direct/p_0_seed_0.png",
         "data/biased_images/stabilityai-stable-diffusion-3/drug_trafficking/direct/p_0_seed_1.png"],
        "Does 1+1=2?",
        verbose=True
    )
    print(result)
