# =============================================================================
# utils/image_utils.py — Image preprocessing & encoding
# =============================================================================
import base64
import io
from PIL import Image


def encode_image_to_base64(image_path, target_short=448):
    """Resize image so shortest side ≤ target_short, then encode as base64 JPEG."""
    try:
        with Image.open(image_path) as image:
            w, h = image.size
            current_short = min(w, h)
            if current_short > target_short:
                scale = target_short / current_short
                new_w = int(w * scale)
                new_h = int(h * scale)
                image = image.resize((new_w, new_h), Image.Resampling.LANCZOS)
            resized_image = image.convert("RGB")

            buffer = io.BytesIO()
            resized_image.save(buffer, format="JPEG", quality=95)
            return base64.b64encode(buffer.getvalue()).decode("utf-8")
    except Exception as e:
        print(f"Image encoding failed: {str(e)[:50]}")
        return None


def preprocess_image_short_side(image, min_size=448, resample=Image.Resampling.LANCZOS):
    """Resize image so shortest side == min_size (if currently larger)."""
    w, h = image.size
    current_short = min(w, h)
    if current_short > min_size:
        scale = min_size / current_short
        new_w = int(w * scale)
        new_h = int(h * scale)
        image = image.resize((new_w, new_h), resample)
    return image


def preprocess_qwen25vl(image, min_size=448):
    """Preprocessing used by Qwen2-VL / Qwen2.5-VL."""
    return preprocess_image_short_side(image, min_size, Image.Resampling.LANCZOS)


def preprocess_qwen3p5vl(image, min_size=448):
    """Preprocessing used by Qwen3-VL / Qwen3.5-VL."""
    return preprocess_image_short_side(image, min_size, Image.Resampling.LANCZOS)