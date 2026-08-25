# =============================================================================
# utils/vllm_runner.py — Shared vLLM batch-inference pipeline
# =============================================================================
import os
import json
from PIL import Image
from tqdm import tqdm

from .evaluation import extract_option_from_text, build_statistics
from .data_loader import load_questions, save_results, find_image_path
from .image_utils import preprocess_image_short_side


# ---------------------------------------------------------------------------
# Model registry — maps model key → (hf_model_id, prompt_template, kwargs)
#
# prompt_template receives (query_text) and returns the full prompt string.
# For chat-template models the template is None (processor.apply_chat_template
# is used instead).
# ---------------------------------------------------------------------------
MODEL_REGISTRY = {
    "qwen2": {
        "model_id": "Qwen/Qwen2-VL-7B-Instruct",
        "use_chat_template": True,
        "trust_remote_code": True,
        "gpu_memory_utilization": 0.92,
    },
    "qwen2.5": {
        "model_id": "Qwen/Qwen2.5-VL-7B-Instruct",
        "use_chat_template": True,
        "trust_remote_code": True,
        "gpu_memory_utilization": 0.92,
    },
    "qwen3": {
        "model_id": "Qwen/Qwen3-VL-8B-Instruct",
        "use_chat_template": True,
        "trust_remote_code": True,
        "gpu_memory_utilization": 0.85,
    },
    "internvl2": {
        "model_id": "OpenGVLab/InternVL2-8B",
        "use_chat_template": False,
        "prompt_template": lambda q: (
            f"<image>You are a professional emotion recognition assistant. "
            f"Follow the instructions strictly and output the answer in the "
            f"required format.\n{q}"
        ),
        "trust_remote_code": True,
        "gpu_memory_utilization": 0.92,
    },
    "internvl3": {
        "model_id": "OpenGVLab/InternVL3-8B",
        "use_chat_template": True,
        "trust_remote_code": True,
        "gpu_memory_utilization": 0.92,
    },
    "llava1.5": {
        "model_id": "llava-hf/llava-1.5-7b-hf",
        "use_chat_template": False,
        "prompt_template": lambda q: (
            f"USER: <image>\nYou are a professional emotion recognition assistant. "
            f"Follow the instructions strictly and output the answer in the "
            f"required format.\n{q}\nASSISTANT:"
        ),
        "trust_remote_code": False,
        "gpu_memory_utilization": 0.92,
    },
    "llava1.6": {
        "model_id": "llava-hf/llava-v1.6-mistral-7b-hf",
        "use_chat_template": False,
        "prompt_template": lambda q: (
            f"[INST] <image>\nYou are a professional emotion recognition assistant. "
            f"Follow the instructions strictly and output the answer in the "
            f"required format.\n{q} [/INST]"
        ),
        "trust_remote_code": False,
        "gpu_memory_utilization": 0.92,
    },
}


def get_model_config(model_key):
    """Return (model_id, cfg) for a given model key."""
    if model_key not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model key: {model_key}. "
                         f"Available: {list(MODEL_REGISTRY.keys())}")
    cfg = MODEL_REGISTRY[model_key]
    return cfg["model_id"], cfg


def build_vllm_batch_inputs(samples, processor, model_key, image_base_dir,
                            fixed_fields=None):
    """Build multimodal batch inputs for vLLM.

    Returns (multimodal_inputs, valid_samples).
    """
    _, cfg = get_model_config(model_key)
    use_chat = cfg.get("use_chat_template", True)
    prompt_fn = cfg.get("prompt_template", None)

    if fixed_fields is None:
        fixed_fields = ["label", "experiment_setup", "idx", "query", "answer", "arousal"]

    multimodal_inputs = []
    valid_samples = []
    invalid_count = 0

    for sample in tqdm(samples, desc="Building batch inputs"):
        image_path = find_image_path(sample, image_base_dir, fixed_fields)
        if not image_path:
            invalid_count += 1
            continue

        try:
            image = Image.open(image_path).convert("RGB")
            image = preprocess_image_short_side(image)

            if use_chat:
                messages = [
                    {"role": "system",
                     "content": "You are a professional emotion recognition assistant. "
                                "Follow the instructions strictly and output the answer "
                                "in the required format."},
                    {"role": "user",
                     "content": [{"type": "image"},
                                 {"type": "text", "text": sample["query"]}]},
                ]
                prompt_text = processor.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
            else:
                prompt_text = prompt_fn(sample["query"])

            multimodal_inputs.append({
                "prompt": prompt_text,
                "multi_modal_data": {"image": [image]},
            })
            valid_samples.append(sample)
        except Exception as e:
            invalid_count += 1
            print(f"  Skip sample {sample.get('idx', 'unknown')}: {str(e)[:50]}")

    print(f"Batch build: {len(valid_samples)} valid, {invalid_count} invalid "
          f"(out of {len(samples)})")
    return multimodal_inputs, valid_samples


def parse_vllm_batch_outputs(valid_samples, batch_outputs):
    """Parse vLLM batch outputs into result dicts."""
    results = []
    exceptions = []
    correct_count = 0

    for sample, output in tqdm(zip(valid_samples, batch_outputs),
                               desc="Parsing outputs",
                               total=len(valid_samples)):
        output_text = output.outputs[0].text.strip()
        ai_answer = extract_option_from_text(output_text)
        if not ai_answer:
            ai_answer = "Extraction failed"
        accuracy = 1 if ai_answer == sample["answer"] else 0
        if accuracy == 1:
            correct_count += 1

        result = sample.copy()
        result["AI_answer"] = ai_answer
        result["accuracy"] = accuracy
        result["model_output"] = output_text
        results.append(result)

        if ai_answer == "Extraction failed":
            exceptions.append(result)

    return results, exceptions, correct_count


def run_vllm_task(input_path, output_path, llm, processor, sampling_params,
                  model_key, image_base_dir):
    """Run a single vLLM evaluation task end-to-end."""
    print(f"\n{'='*60}")
    print(f"Task: {os.path.basename(input_path)}")
    print(f"{'='*60}")

    samples = load_questions(input_path)
    print(f"Loaded {len(samples)} samples")

    multimodal_inputs, valid_samples = build_vllm_batch_inputs(
        samples, processor, model_key, image_base_dir
    )
    if not valid_samples:
        print("No valid samples — skipping.")
        return

    print(f"Running vLLM batch inference on {len(valid_samples)} samples...")
    batch_outputs = llm.generate(
        multimodal_inputs,
        sampling_params=sampling_params,
        use_tqdm=True,
    )

    results, exceptions, correct_count = parse_vllm_batch_outputs(
        valid_samples, batch_outputs
    )

    stats = build_statistics(results, len(samples), exceptions)
    print(f"Accuracy: {stats['correct_answers']}/{stats['processed_samples']} "
          f"= {stats['overall_accuracy']:.2%}")

    save_results(results, stats, exceptions, output_path)