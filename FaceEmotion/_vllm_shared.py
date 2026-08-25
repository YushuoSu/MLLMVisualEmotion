# =============================================================================
# FaceEmotion — vLLM-based model evaluation (shared runner)
# =============================================================================
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from transformers import AutoProcessor
from vllm import LLM, SamplingParams

from config import DATASET_ROOT, OUTPUT_ROOT
from utils.vllm_runner import (
    get_model_config,
    run_vllm_task,
)

ALL_TASK_NAMES = [
    "images_multiple_choice_questions",
    "images_marked_multiple_choice_questions",
    "images_emotional_tendency_multiple_choice_questions",
    "images_judgement_questions",
    "images_could_judgement_questions",
    "images_really_judgement_questions",
    "images_emotional_tendency_judgement_questions",
]


def build_task_configs(dataset_name, model_name, task_names=None):
    if task_names is None:
        task_names = ALL_TASK_NAMES
    configs = []
    for tname in task_names:
        input_path = DATASET_ROOT / dataset_name / "processed" / f"{tname}.json"
        output_dir = OUTPUT_ROOT / model_name / dataset_name
        output_path = output_dir / f"{tname.replace('_questions', '_results')}.json"
        configs.append({
            "input_path": str(input_path),
            "output_path": str(output_path),
        })
    return configs


def main_vllm(model_key, dataset_name, task_names=None):
    model_id, cfg = get_model_config(model_key)
    model_name = cfg.get("output_name", model_key)

    print(f"Model: {model_id}  |  Dataset: {dataset_name}")
    print(f"Device: {'cuda' if torch.cuda.is_available() else 'cpu'}")

    processor = AutoProcessor.from_pretrained(
        model_id,
        trust_remote_code=cfg.get("trust_remote_code", False),
    )
    llm = LLM(
        model=model_id,
        dtype=torch.bfloat16,
        gpu_memory_utilization=cfg.get("gpu_memory_utilization", 0.92),
        tensor_parallel_size=1,
        trust_remote_code=cfg.get("trust_remote_code", False),
        max_num_batched_tokens=8192,
    )
    sampling_params = SamplingParams(
        max_tokens=500,
        skip_special_tokens=True,
        temperature=0,
        top_p=1,
    )
    print("Model loaded successfully with vLLM acceleration!")

    image_base_dir = str(DATASET_ROOT / dataset_name / "processed")
    task_configs = build_task_configs(dataset_name, model_name, task_names)

    for tc in task_configs:
        run_vllm_task(
            tc["input_path"], tc["output_path"],
            llm, processor, sampling_params,
            model_key, image_base_dir,
        )

    print(f"\nAll {dataset_name} tasks completed for {model_name}!")