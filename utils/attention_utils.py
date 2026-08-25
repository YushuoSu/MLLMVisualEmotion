# =============================================================================
# utils/attention_utils.py — Attention extraction & heatmap utilities
# =============================================================================
import numpy as np
import torch
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")


# ViT parameters (Qwen2.5-VL-7B-Instruct vision_config)
PATCH_SIZE = 14
SPATIAL_MERGE_SIZE = 2
EFFECTIVE_PATCH = PATCH_SIZE * SPATIAL_MERGE_SIZE  # 28 px per visual token


# ---------------------------------------------------------------------------
# Token-region helpers
# ---------------------------------------------------------------------------
def get_vision_region(input_ids, processor):
    """Return {start, end, num_tokens} for vision tokens in input_ids[0]."""
    input_ids_np = input_ids[0].cpu().numpy()
    vis_start_id = processor.tokenizer.convert_tokens_to_ids("<|vision_start|>")
    vis_end_id = processor.tokenizer.convert_tokens_to_ids("<|vision_end|>")
    start_pos = np.where(input_ids_np == vis_start_id)[0]
    end_pos = np.where(input_ids_np == vis_end_id)[0]
    if len(start_pos) == 0 or len(end_pos) == 0:
        return None
    s = int(start_pos[0])
    e = int(end_pos[0])
    return {"start": s + 1, "end": e, "num_tokens": e - s - 1}


def get_token_region(input_ids, processor):
    """Return (text_mask, img_mask) boolean arrays for the full input sequence."""
    input_ids_np = input_ids[0].cpu().numpy()
    vis_start_id = processor.tokenizer.convert_tokens_to_ids("<|vision_start|>")
    vis_end_id = processor.tokenizer.convert_tokens_to_ids("<|vision_end|>")
    start_pos = np.where(input_ids_np == vis_start_id)[0]
    end_pos = np.where(input_ids_np == vis_end_id)[0]
    if len(start_pos) == 0 or len(end_pos) == 0:
        return None
    s = start_pos[0]
    e = end_pos[0]
    text_mask = (np.arange(len(input_ids_np)) < s) | (np.arange(len(input_ids_np)) > e)
    img_mask = (np.arange(len(input_ids_np)) > s) & (np.arange(len(input_ids_np)) < e)
    return text_mask, img_mask


# ---------------------------------------------------------------------------
# Attention hooks
# ---------------------------------------------------------------------------
class LayerAttentionHook:
    """Capture decode-phase attention for specific layers."""

    def __init__(self, model, target_layers):
        self.target_layers = target_layers
        self.decode_attn = {l: [] for l in target_layers}
        self._handles = []

    def _make_hook(self, layer_idx):
        def hook_fn(module, input, output):
            if isinstance(output, tuple) and len(output) >= 2:
                attn = output[1].detach().cpu().float()
                if attn.shape[2] == 1:  # decode phase
                    self.decode_attn[layer_idx].append(attn)
        return hook_fn

    def register(self, model):
        for i, layer in enumerate(model.language_model.layers):
            if i in self.target_layers and hasattr(layer, "self_attn"):
                handle = layer.self_attn.register_forward_hook(self._make_hook(i))
                self._handles.append(handle)

    def remove(self):
        for h in self._handles:
            h.remove()
        self._handles.clear()


class DecodeAttentionHook:
    """Capture decode-phase attention for ALL layers."""

    def __init__(self, num_layers=28):
        self.num_layers = num_layers
        self.decode_attentions = [[] for _ in range(num_layers)]
        self._handles = []

    def make_hook(self, layer_idx):
        def hook_fn(module, input, output):
            if isinstance(output, tuple) and len(output) >= 2:
                attn = output[1].detach().cpu().float()
                if attn.shape[2] == 1:
                    self.decode_attentions[layer_idx].append(attn)
        return hook_fn

    def register_on_model(self, model):
        for i, layer in enumerate(model.language_model.layers):
            if hasattr(layer, "self_attn"):
                handle = layer.self_attn.register_forward_hook(self.make_hook(i))
                self._handles.append(handle)

    def remove(self):
        for handle in self._handles:
            handle.remove()
        self._handles.clear()


# ---------------------------------------------------------------------------
# Decode attention metrics (all layers)
# ---------------------------------------------------------------------------
def calc_decode_attention(decode_attentions, input_ids, processor):
    """Compute per-layer text/image attention ratios from decode attention."""
    regions = get_token_region(input_ids, processor)
    if regions is None:
        return [], 0
    text_mask, img_mask = regions
    input_len = len(text_mask)
    text_bool = torch.tensor(text_mask)
    img_bool = torch.tensor(img_mask)

    num_layers = len(decode_attentions)
    if num_layers == 0:
        return [], 0
    num_decode_steps = len(decode_attentions[0]) if decode_attentions[0] else 0
    if num_decode_steps == 0:
        return [], 0

    layer_results = []
    for layer_idx in range(num_layers):
        text_sums, img_sums, input_sums, total_sums = [], [], [], []
        for attn in decode_attentions[layer_idx]:
            attn_vec = attn.squeeze(0).mean(dim=0).squeeze(0)
            input_portion = attn_vec[:input_len]
            text_sums.append(float(input_portion[text_bool].sum()))
            img_sums.append(float(input_portion[img_bool].sum()))
            input_sums.append(float(input_portion.sum()))
            total_sums.append(float(attn_vec.sum()))

        n = len(text_sums)
        avg_text = sum(text_sums) / n
        avg_img = sum(img_sums) / n
        input_sum = avg_text + avg_img
        text_ratio = avg_text / input_sum if input_sum > 0 else 0.0
        img_ratio = avg_img / input_sum if input_sum > 0 else 0.0
        per_token_ratios = [i / t if t > 1e-9 else 0.0
                            for i, t in zip(input_sums, total_sums)]
        input_attn_ratio = sum(per_token_ratios) / n

        layer_results.append({
            "layer": layer_idx,
            "avg_text_attn_per_output_token": round(avg_text, 6),
            "avg_img_attn_per_output_token": round(avg_img, 6),
            "text_ratio": round(text_ratio, 6),
            "img_ratio": round(img_ratio, 6),
            "input_attn_ratio": round(input_attn_ratio, 6),
        })
    return layer_results, num_decode_steps


# ---------------------------------------------------------------------------
# Heatmap construction
# ---------------------------------------------------------------------------
def build_image_heatmap(decode_attn, vision_region, image_grid_thw, original_image):
    """Build a 2D attention heatmap over image tokens."""
    all_image_attn = []
    for layer_idx, attn_list in decode_attn.items():
        for attn in attn_list:
            attn_2d = attn.squeeze(0).squeeze(1)
            img_part = attn_2d[:, vision_region["start"]:vision_region["end"]]
            all_image_attn.append(img_part)

    if not all_image_attn:
        return None, None

    stacked = torch.cat([a.mean(dim=0, keepdim=True) for a in all_image_attn], dim=0)
    avg_attn = stacked.mean(dim=0).numpy()

    thw = image_grid_thw[0].cpu().numpy()
    grid_h = int(np.ceil(thw[1] / SPATIAL_MERGE_SIZE))
    grid_w = int(np.ceil(thw[2] / SPATIAL_MERGE_SIZE))
    expected_tokens = grid_h * grid_w
    avg_attn = avg_attn[:expected_tokens]
    heatmap_2d = avg_attn.reshape(grid_h, grid_w)

    orig_w, orig_h = original_image.size
    heatmap_arr = np.array(
        Image.fromarray(
            (heatmap_2d * 255 / (heatmap_2d.max() + 1e-9)).astype(np.uint8)
        ).resize((orig_w, orig_h), Image.Resampling.BILINEAR)
    )
    return heatmap_2d, heatmap_arr


def overlay_heatmap(original_image, heatmap_arr, alpha=0.5, cmap="jet"):
    """Overlay a heatmap onto the original image, return PIL Image."""
    orig_arr = np.array(original_image.convert("RGB"))
    h_norm = (heatmap_arr - heatmap_arr.min()) / (heatmap_arr.max() - heatmap_arr.min() + 1e-9)
    cm = plt.get_cmap(cmap)
    colored = (cm(h_norm)[:, :, :3] * 255).astype(np.uint8)
    blended = (colored * alpha + orig_arr * (1 - alpha)).astype(np.uint8)
    return Image.fromarray(blended)