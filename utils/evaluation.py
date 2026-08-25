# =============================================================================
# utils/evaluation.py — Answer extraction & statistics
# =============================================================================
import re


def extract_option_from_text(text):
    """Extract the last parenthesised option letter, e.g. '(C)' → '(C)'."""
    pattern = r'\(([ABCDEFGHIJ])\)'
    matches = re.findall(pattern, text)
    if matches:
        return f"({matches[-1]})"
    return None


def compute_accuracy(results):
    """Return (correct_count, total, accuracy) from a list of result dicts."""
    total = len(results)
    if total == 0:
        return 0, 0, 0.0
    correct = sum(1 for r in results if r.get("accuracy", 0) == 1)
    return correct, total, correct / total


def build_statistics(results, total_samples, exceptions=None):
    """Build a standard statistics dict for JSON output."""
    if exceptions is None:
        exceptions = [r for r in results if r.get("AI_answer") == "Extraction failed"]
    processed = len(results)
    correct, _, accuracy = compute_accuracy(results)
    return {
        "total_samples": total_samples,
        "processed_samples": processed,
        "total_exceptions": len(exceptions),
        "correct_answers": correct,
        "overall_accuracy": round(accuracy, 6),
    }