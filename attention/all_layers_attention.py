# =============================================================================
# All-layers decode attention analysis — Emotion6
# =============================================================================
# Captures decode-phase attention from ALL 28 layers of Qwen2.5-VL-7B-Instruct,
# computes per-layer text/image attention ratios, and saves per-sample +
# cross-sample averages.
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

from config import DATASET_ROOT, OUTPUT_ROOT
from utils.attention_utils import (
    DecodeAttentionHook,
    calc_decode_attention,
)
from utils.evaluation import extract_option_from_text
from utils.image_utils import preprocess_qwen25vl
from utils.data_loader import find_image_path

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
DATASET = "Emotion6"
NUM_SAMPLES = None  # set to an integer to limit samples for debugging

TASK_CONFIGS = [
    {
        "input": "images_multiple_choice_questions.json",
        "output": "images_multiple_choice_all_layers.json",
    },
    {
        "input": "images_marked_multiple_choice_questions.json",
        "output": "images_marked_multiple_choice_all_layers.json",
    },
    {
        "input": "images_emotional_tendency_multiple_choice_questions.json",
        "output": "images_emotional_tendency_multiple_choice_all_layers.json",
    },
]

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


def process_single_sample(model, processor, sample):
    try:
        fixed_fields = ["label", "experiment_setup", "idx", "query", "answer", "arousal"]
        image_path = find_image_path(
            sample, str(DATASET_ROOT / DATASET / "processed"), fixed_fields
        )
        if not image_path:
            return None

        image = Image.open(image_path).convert("RGB")
        image = preprocess_qwen25vl(image)

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
            text=[text], images=[image], return_tensors="pt", padding=True
        ).to(model.device)

        hook = DecodeAttentionHook(num_layers=28)
        hook.register_on_model(model)

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

        layer_attention, output_token_count = calc_decode_attention(
            hook.decode_attentions, inputs["input_ids"], processor
        )
        hook.remove()

        sequences = gen_out.sequences
        input_length = inputs["input_ids"].shape[1]
        generated_tokens = sequences[:, input_length:]
        response_text = processor.batch_decode(
            generated_tokens, skip_special_tokens=True
        )[0].strip()

        ai_answer = extract_option_from_text(response_text)
        ai_answer = ai_answer if ai_answer else "Extraction failed"
        accuracy = 1 if ai_answer == sample["answer"] else 0

        res = sample.copy()
        res["AI_answer"] = ai_answer
        res["accuracy"] = int(accuracy)
        res["model_output"] = response_text
        res["output_token_count"] = output_token_count
        res["layer_attention"] = layer_attention

        if layer_attention:
            avg_keys = [
                "avg_text_attn_per_output_token",
                "avg_img_attn_per_output_token",
                "text_ratio", "img_ratio", "input_attn_ratio",
            ]
            res["layer_average"] = {
                k: round(sum(l[k] for l in layer_attention) / len(layer_attention), 6)
                for k in avg_keys
            }
        return res

    except Exception as e:
        import traceback
        print(f"Sample error: {e}")
        traceback.print_exc()
        return None


def process_single_task(input_path, output_path, model, processor):
    print(f"\n{'='*60}")
    print(f"Task: {os.path.basename(input_path)}")
    print(f"{'='*60}")

    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    samples = list(data.values()) if isinstance(data, dict) else data
    if NUM_SAMPLES is not None:
        samples = samples[:NUM_SAMPLES]
    print(f"Loaded {len(samples)} samples.\n")

    results = []
    correct_count = 0

    for sample in tqdm(samples, desc="Processing"):
        res = process_single_sample(model, processor, sample)
        if res is None:
            continue
        results.append(res)
        if res["accuracy"] == 1:
            correct_count += 1

    total = len(results)
    acc = correct_count / total if total > 0 else 0.0
    print(f"Processed: {total}/{len(samples)}  |  Accuracy: {correct_count}/{total} = {acc:.2%}")

    # --- Cross-sample averages ---
    cross_sample_avg = {}
    if results:
        avg_keys = [
            "avg_text_attn_per_output_token",
            "avg_img_attn_per_output_token",
            "text_ratio", "img_ratio", "input_attn_ratio",
        ]
        n = len(results)

        layer_cross_avg = []
        for layer_idx in range(28):
            layer_vals = {k: 0.0 for k in avg_keys}
            for r in results:
                la = r.get("layer_attention", [])
                if layer_idx < len(la):
                    for k in avg_keys:
                        layer_vals[k] += la[layer_idx][k]
            layer_cross_avg.append({
                "layer": layer_idx,
                **{k: round(layer_vals[k] / n, 6) for k in avg_keys},
            })

        sample_avg_vals = {k: 0.0 for k in avg_keys}
        for r in results:
            la = r.get("layer_average", {})
            for k in avg_keys:
                sample_avg_vals[k] += la.get(k, 0.0)
        cross_sample_avg = {
            "num_samples": n,
            "layer_cross_average": layer_cross_avg,
            "overall_average": {
                k: round(sample_avg_vals[k] / n, 6) for k in avg_keys
            },
        }

    output_data = {
        "samples": results,
        "cross_sample_average": cross_sample_avg,
        "statistics": {
            "total_samples": len(samples),
            "processed_samples": total,
            "correct_answers": correct_count,
            "overall_accuracy": round(acc, 6),
        },
    }

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    print(f"Results saved to {output_path}")


def main():
    print(f"Device: {DEVICE}")
    model, processor = load_model()

    for tc in TASK_CONFIGS:
        input_path = str(DATASET_ROOT / DATASET / "processed" / tc["input"])
        output_path = str(OUTPUT_ROOT / "new_attention" / tc["output"])
        process_single_task(input_path, output_path, model, processor)

    print("\nAll all-layers attention tasks completed!")


if __name__ == "__main__":
    main()