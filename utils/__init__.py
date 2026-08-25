# =============================================================================
# utils/__init__.py — public API surface
# =============================================================================
from .image_utils import (
    encode_image_to_base64,
    preprocess_image_short_side,
    preprocess_qwen25vl,
    preprocess_qwen3p5vl,
)
from .evaluation import (
    extract_option_from_text,
    compute_accuracy,
    build_statistics,
)
from .data_loader import (
    load_questions,
    save_results,
    find_image_path,
)
from .api_client import (
    create_async_client,
    call_api_with_retry,
)
from .vllm_runner import (
    build_vllm_batch_inputs,
    parse_vllm_batch_outputs,
    run_vllm_task,
)
from .attention_utils import (
    get_vision_region,
    get_token_region,
    LayerAttentionHook,
    DecodeAttentionHook,
    build_image_heatmap,
    overlay_heatmap,
    calc_decode_attention,
)