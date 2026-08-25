# =============================================================================
# Reading Human Emotion Through Machine Eyes — Global Configuration
# =============================================================================
# All paths and API keys are managed here.  Copy .env.example to .env and fill
# in your own values before running any script.
# =============================================================================

import os
from pathlib import Path
from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# Load .env (if present)
# ---------------------------------------------------------------------------
load_dotenv()

# ---------------------------------------------------------------------------
# Project root (parent of this config file)
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent

# ---------------------------------------------------------------------------
# HuggingFace cache
# ---------------------------------------------------------------------------
HF_HOME = os.getenv("HF_HOME", str(PROJECT_ROOT / "hf_cache"))
HF_ENDPOINT = os.getenv("HF_ENDPOINT", "https://hf-mirror.com")

# ---------------------------------------------------------------------------
# API keys (for GPT / Gemini via OpenAI-compatible proxy)
# ---------------------------------------------------------------------------
DMX_API_KEY = os.getenv("DMX_API_KEY", "")
DMX_BASE_URL = os.getenv("DMX_BASE_URL", "https://www.dmxapi.cn/v1")

# ---------------------------------------------------------------------------
# Dataset & output roots
#   - DATASET_ROOT: where Emotion6/processed/ and FaceEmotion/processed/ live
#   - OUTPUT_ROOT:  where per-model result folders are written
# ---------------------------------------------------------------------------
DATASET_ROOT = Path(os.getenv("DATASET_ROOT", str(PROJECT_ROOT / "data")))
OUTPUT_ROOT = Path(os.getenv("OUTPUT_ROOT", str(PROJECT_ROOT / "results")))

# ---------------------------------------------------------------------------
# Attention-map specific paths
# ---------------------------------------------------------------------------
ATTENTION_DATA_ROOT = Path(
    os.getenv("ATTENTION_DATA_ROOT", str(PROJECT_ROOT / "data"))
)
ATTENTION_OUTPUT_ROOT = Path(
    os.getenv("ATTENTION_OUTPUT_ROOT", str(PROJECT_ROOT / "results" / "attention_maps"))
)

# ---------------------------------------------------------------------------
# Convenience: ensure directories exist
# ---------------------------------------------------------------------------
os.makedirs(HF_HOME, exist_ok=True)
os.makedirs(str(OUTPUT_ROOT), exist_ok=True)
os.makedirs(str(ATTENTION_OUTPUT_ROOT), exist_ok=True)

# Set HF env vars so transformers / vLLM pick them up
os.environ["HF_HOME"] = HF_HOME
os.environ["HF_ENDPOINT"] = HF_ENDPOINT

print(f"HF_HOME      = {HF_HOME}")
print(f"HF_ENDPOINT  = {HF_ENDPOINT}")
print(f"DATASET_ROOT = {DATASET_ROOT}")
print(f"OUTPUT_ROOT  = {OUTPUT_ROOT}")