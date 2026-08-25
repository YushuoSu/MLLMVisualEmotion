# =============================================================================
# Emotion6 / FaceEmotion — API-based model evaluation (shared runner)
# =============================================================================
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import asyncio
from tqdm import tqdm

from config import DATASET_ROOT, OUTPUT_ROOT
from utils.api_client import create_async_client, call_api_with_retry
from utils.evaluation import extract_option_from_text, build_statistics
from utils.data_loader import load_questions, save_results, find_image_path
from utils.image_utils import encode_image_to_base64

# ---------------------------------------------------------------------------
ALL_TASK_NAMES = [
    "images_multiple_choice_questions",
    "images_marked_multiple_choice_questions",
    "images_emotional_tendency_multiple_choice_questions",
    "images_judgement_questions",
    "images_could_judgement_questions",
    "images_really_judgement_questions",
    "images_emotional_tendency_judgement_questions",
]

BATCH_SIZE = 5


def build_task_configs(dataset_name, model_name, task_names=None):
    if task_names is None:
        task_names = ALL_TASK_NAMES
    configs = []
    for tname in task_names:
        input_path = DATASET_ROOT / dataset_name / "processed" / f"{tname}.json"
        output_dir = OUTPUT_ROOT / model_name / dataset_name
        output_path = output_dir / f"{tname.replace('_questions', '_results')}.json"
        configs.append({"input_path": str(input_path), "output_path": str(output_path)})
    return configs


async def process_sample_async(sample, client, model_name, image_base_dir):
    """Process a single sample through the API."""
    fixed_fields = ["label", "experiment_setup", "idx", "query", "answer", "arousal"]
    image_path = find_image_path(sample, image_base_dir, fixed_fields)
    if not image_path or not os.path.exists(image_path):
        return None

    image_base64 = encode_image_to_base64(image_path)
    if not image_base64:
        return None

    system_prompt = (
        "You are a professional emotion recognition assistant. "
        "Follow the instructions strictly and output the answer in the required format."
    )
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": [
            {"type": "image_url",
             "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"}},
            {"type": "text", "text": sample["query"]},
        ]},
    ]

    output_text = await call_api_with_retry(client, model_name, messages)
    ai_answer = extract_option_from_text(output_text)
    if not ai_answer:
        ai_answer = "Extraction failed"

    result = sample.copy()
    result["AI_answer"] = ai_answer
    result["accuracy"] = 1 if ai_answer == sample["answer"] else 0
    result["model_output"] = output_text
    return result


async def process_batch_async(samples, client, model_name, image_base_dir):
    results = []
    for i in tqdm(range(0, len(samples), BATCH_SIZE), desc="API batches"):
        batch = samples[i:i + BATCH_SIZE]
        tasks = [process_sample_async(s, client, model_name, image_base_dir)
                 for s in batch]
        batch_results = await asyncio.gather(*tasks)
        results.extend([r for r in batch_results if r is not None])
    return results


async def process_single_task_async(input_path, output_path, client, model_name,
                                    image_base_dir):
    print(f"\n{'='*60}")
    print(f"Task: {os.path.basename(input_path)}")
    print(f"{'='*60}")

    samples = load_questions(input_path)
    print(f"Loaded {len(samples)} samples")

    results = await process_batch_async(samples, client, model_name, image_base_dir)

    exceptions = [r for r in results if r.get("AI_answer") == "Extraction failed"]
    stats = build_statistics(results, len(samples), exceptions)
    print(f"Accuracy: {stats['correct_answers']}/{stats['processed_samples']} "
          f"= {stats['overall_accuracy']:.2%}")

    save_results(results, stats, exceptions, output_path)


async def main_api_async(model_name, dataset_name, task_names=None):
    """Main async entry point for API-based models."""
    print(f"Model: {model_name}  |  Dataset: {dataset_name}")

    client = await create_async_client()
    image_base_dir = str(DATASET_ROOT / dataset_name / "processed")
    task_configs = build_task_configs(dataset_name, model_name, task_names)

    for tc in task_configs:
        await process_single_task_async(
            tc["input_path"], tc["output_path"],
            client, model_name, image_base_dir,
        )

    print(f"\nAll {dataset_name} tasks completed for {model_name}!")


def run_api(model_name, dataset_name, task_names=None):
    """Synchronous wrapper for main_api_async."""
    asyncio.run(main_api_async(model_name, dataset_name, task_names))