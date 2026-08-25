# =============================================================================
# utils/data_loader.py — JSON I/O helpers
# =============================================================================
import json
import os


def load_questions(json_path):
    """Load question data from a JSON file. Returns a flat list of samples."""
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict):
        return list(data.values())
    return data


def save_results(results, statistics, exceptions, output_path):
    """Save results + statistics to a JSON file."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    output_data = {
        "samples": results,
        "statistics": statistics,
        "Exception": exceptions,
    }
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    print(f"Results saved to {output_path}")


def find_image_path(sample, base_dir, fixed_fields=None):
    """Find the image path from a sample dict.
    
    Looks for the first key not in `fixed_fields` whose value looks like an
    image filename, then joins it with `base_dir`.
    """
    if fixed_fields is None:
        fixed_fields = ["label", "experiment_setup", "idx", "query", "answer", "arousal"]
    for key, value in sample.items():
        if key not in fixed_fields and isinstance(value, str):
            if any(ext in value.lower() for ext in ["jpg", "png", "jpeg"]):
                return os.path.join(base_dir, value)
    return None