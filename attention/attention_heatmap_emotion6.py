# =============================================================================
# Attention heatmap generation — Emotion6
# =============================================================================
# Captures decode-phase attention from layers 19-27 of Qwen2.5-VL-7B-Instruct,
# averages across all heads and decode steps, and overlays heatmaps on images.
# =============================================================================
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import torch
import numpy as np
from PIL import Image
from tqdm import tqdm

from transformers import AutoProcessor, AutoModelForVision2Seq

from config import ATTENTION_DATA_ROOT, ATTENTION_OUTPUT_ROOT
from utils.attention_utils import (
    get_vision_region,
    LayerAttentionHook,
    build_image_heatmap,
    overlay_heatmap,
)
from utils.image_utils import preprocess_qwen25vl
from utils.data_loader import find_image_path

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
DATASET = "Emotion6"
INPUT_FILE = ATTENTION_DATA_ROOT / DATASET / "images_multiple_choice_questions.json"
IMG_BASE = ATTENTION_DATA_ROOT / DATASET
OUTPUT_DIR = ATTENTION_OUTPUT_ROOT / DATASET
os.makedirs(str(OUTPUT_DIR), exist_ok=True)

TARGET_LAYERS = list(range(19, 28))  # layers 19-27
MODEL_NAME = "Qwen/Qwen2.5-VL-7B-Instruct"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def load_model():
    print(f"Loading {MODEL_NAME} ...")
    torch_dtype = torch.bfloat16
    processor = AutoProcessor.from_pretrained(MODEL_NAME, trust_remote_code=True)
    model = AutoModelForVision2Seq.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch_dtype,
        device_map="auto",
        trust_remote_code=True,
        attn_implementation="eager",
    )
    model.eval()
    print("Model loaded.")
    return model, processor


def process_one_sample(model, processor, sample, sample_idx):
    try:
        fixed_fields = ["label", "experiment_setup", "idx", "query", "answer", "arousal"]
        img_rel_path = None
        for k, v in sample.items():
            if k not in fixed_fields and isinstance(v, str) and v.startswith("imgs/"):
                img_rel_path = v
                break
        if img_rel_path is None:
            return None

        img_path = os.path.join(str(IMG_BASE), img_rel_path)
        image = Image.open(img_path).convert("RGB")
        img_for_model = preprocess_qwen25vl(image)

        messages = [
            {"role": "system",
             "content": "You are a professional emotion recognition assistant. "
                        "Follow the instructions strictly and output the answer "
                        "in the required format."},
            {"role": "user",
             "content": [{"type": "image"},
                         {"type": "text", "text": sample["query"]}]},
        ]
        text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = processor(
            text=[text], images=[img_for_model], return_tensors="pt", padding=True
        ).to(model.device)

        vision_region = get_vision_region(inputs["input_ids"], processor)
        if vision_region is None or vision_region["num_tokens"] == 0:
            return None

        hook = LayerAttentionHook(model, TARGET_LAYERS)
        hook.register(model)

        with torch.no_grad():
            gen_out = model.generate(
                **inputs,
                max_new_tokens=500,
                do_sample=False,
                top_p=1.0,
                pad_token_id=processor.tokenizer.pad_token_id,
                eos_token_id=processor.tokenizer.eos_token_id,
                return_dict_in_generate=True,
            )

        hook.remove()

        heatmap_2d, heatmap_arr = build_image_heatmap(
            hook.decode_attn, vision_region,
            inputs.get("image_grid_thw"), image,
        )
        if heatmap_2d is None:
            return None

        # --- Decode text ---
        sequences = gen_out.sequences
        input_length = inputs["input_ids"].shape[1]
        generated_tokens = sequences[:, input_length:]
        response_text = processor.batch_decode(
            generated_tokens, skip_special_tokens=True
        )[0].strip()

        # --- Save outputs ---
        overlay_img = overlay_heatmap(image, heatmap_arr, alpha=0.5)
        overlay_img.save(str(OUTPUT_DIR / f"{sample_idx:05d}_heatmap.png"))

        heatmap_raw = Image.fromarray(
            (heatmap_arr / (heatmap_arr.max() + 1e-9) * 255).astype(np.uint8)
        )
        heatmap_raw.save(str(OUTPUT_DIR / f"{sample_idx:05d}_heatmap_raw.png"))

        image.save(str(OUTPUT_DIR / f"{sample_idx:05d}_original.png"))

        np.savez(
            str(OUTPUT_DIR / f"{sample_idx:05d}_heatmap.npz"),
            heatmap=heatmap_2d,
        )

        return {
            "idx": sample_idx,
            "label": sample.get("label", ""),
            "answer": sample.get("answer", ""),
            "model_output": response_text,
            "image_path": img_path,
            "num_decode_steps": sum(len(v) for v in hook.decode_attn.values()),
            "num_image_tokens": vision_region["num_tokens"],
            "heatmap_grid": f"{heatmap_2d.shape[0]}×{heatmap_2d.shape[1]}",
        }

    except Exception as e:
        import traceback
        print(f"  Sample {sample_idx} ERROR: {e}")
        traceback.print_exc()
        return None


def main():
    print(f"Device: {DEVICE}")
    print(f"Output: {OUTPUT_DIR}")
    print(f"Target layers: {TARGET_LAYERS}")

    model, processor = load_model()

    with open(str(INPUT_FILE), "r", encoding="utf-8") as f:
        data = json.load(f)
    samples = data if isinstance(data, list) else list(data.values())
    print(f"Loaded {len(samples)} samples.\n")

    results = []
    for i, sample in enumerate(tqdm(samples, desc="Processing")):
        res = process_one_sample(model, processor, sample, i)
        if res:
            results.append(res)

    summary_path = str(OUTPUT_DIR / "summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump({
            "dataset": DATASET,
            "num_samples": len(samples),
            "num_processed": len(results),
            "target_layers": TARGET_LAYERS,
            "model": MODEL_NAME,
            "results": results,
        }, f, indent=2, ensure_ascii=False)

    print(f"\nDone! {len(results)}/{len(samples)} samples processed.")
    print(f"Results saved to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()