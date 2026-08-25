# Reading Human Emotion Through Machine Eyes

Code for the paper *"Reading Human Emotion Through Machine Eyes"*.

This repository provides the complete evaluation pipeline for benchmarking
MLLMs on visual emotion recognition tasks across two datasets (**Emotion6**
and **FaceEmotion**), as well as attention-map analysis tools.

## 📁 Repository Structure

```
MLLMVisualEmotion/
├── config.py                 # Global path & API configuration
├── .env.example              # Environment variables template
├── requirements.txt          # Python dependencies
├── run_all.sh                # One-click run all experiments
├── README.md
│
├── utils/                    # Shared utility modules
│   ├── api_client.py         # Async OpenAI-compatible API client
│   ├── attention_utils.py    # Attention extraction & heatmap helpers
│   ├── data_loader.py        # JSON I/O helpers
│   ├── evaluation.py         # Answer extraction & statistics
│   ├── image_utils.py        # Image preprocessing & encoding
│   └── vllm_runner.py        # vLLM batch-inference pipeline
│
├── Emotion6/                 # Emotion6 dataset evaluation
│   ├── run_emotion6.py       # Batch runner for all models
│   ├── _vllm_shared.py       # Shared vLLM runner
│   ├── _api_shared.py        # Shared API runner
│   ├── qwen2_test.py
│   ├── qwen2p5_test.py
│   ├── qwen3_test.py
│   ├── internvl2_test.py
│   ├── internvl3_test.py
│   ├── llava1p5_test.py
│   ├── llava1p6_test.py
│   ├── gpt4o_test.py
│   ├── gpt5mini_test.py
│   └── gemini_flash_test.py
│
├── FaceEmotion/              # FaceEmotion dataset evaluation
│   ├── run_face_emotion.py
│   ├── _vllm_shared.py
│   ├── _api_shared.py
│   └── (10 model scripts, same as above)
│
└── attention/                # Attention-map analysis
    ├── attention_heatmap_emotion6.py   # Heatmap generation (Emotion6)
    ├── attention_heatmap_face.py       # Heatmap generation (FaceEmotion)
    └── all_layers_attention.py         # All-28-layer attention analysis
```

## 📊 Models Evaluated

| # | Model | Type | HuggingFace ID |
|---|-------|------|----------------|
| 1 | GPT-4o | API | — |
| 2 | GPT-5-mini | API | — |
| 3 | Gemini-3.1-Flash | API | — |
| 4 | InternVL2-8B | vLLM | `OpenGVLab/InternVL2-8B` |
| 5 | InternVL3-8B | vLLM | `OpenGVLab/InternVL3-8B` |
| 6 | LLaVA-1.5-7B | vLLM | `llava-hf/llava-1.5-7b-hf` |
| 7 | LLaVA-1.6-7B | vLLM | `llava-hf/llava-v1.6-mistral-7b-hf` |
| 8 | Qwen2-VL-7B | vLLM | `Qwen/Qwen2-VL-7B-Instruct` |
| 9 | Qwen2.5-VL-7B | vLLM | `Qwen/Qwen2.5-VL-7B-Instruct` |
| 10 | Qwen3-VL-8B | vLLM | `Qwen/Qwen3-VL-8B-Instruct` |

## 📋 Task Types

Each model is evaluated on 7 task types per dataset:

1. **Multiple choice** — standard emotion classification
2. **Marked multiple choice** — with marked regions
3. **Emotional tendency multiple choice** — positive/negative/neutral
4. **Judgement** — binary emotion judgement
5. **Could judgement** — "could this be emotion X?"
6. **Really judgement** — "is this really emotion X?"
7. **Emotional tendency judgement** — tendency binary judgement

## 🚀 Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure environment

```bash
cp .env.example .env
# Edit .env with your API keys and data paths
```

### 3. Prepare data

Place the datasets under `data/`:
```
data/
├── Emotion6/
│   └── processed/
│       ├── images_multiple_choice_questions.json
│       ├── images_marked_multiple_choice_questions.json
│       ├── images_emotional_tendency_multiple_choice_questions.json
│       ├── images_judgement_questions.json
│       ├── images_could_judgement_questions.json
│       ├── images_really_judgement_questions.json
│       ├── images_emotional_tendency_judgement_questions.json
│       └── imgs/
└── FaceEmotion/
    └── processed/
        └── (same structure)
```

### 4. Run evaluation

```bash
# Run all models on both datasets
bash run_all.sh

# Or run individual datasets
python Emotion6/run_emotion6.py
python FaceEmotion/run_face_emotion.py

# Or run a single model
python Emotion6/qwen2p5_test.py
```

### 5. Run attention analysis

```bash
# Generate attention heatmaps
python attention/attention_heatmap_emotion6.py
python attention/attention_heatmap_face.py

# All-layers decode attention analysis
python attention/all_layers_attention.py
```

## 📦 Data Availability

The datasets and model outputs are publicly available on Hugging Face:
[**YushuoSu/MLLMVisualEmotion**](https://huggingface.co/datasets/YushuoSu/MLLMVisualEmotion)

## 🔧 Requirements

- Python ≥ 3.10
- CUDA-compatible GPU (for vLLM models)
- See `requirements.txt` for full dependency list

## 📝 Citation

If you use this code in your research, please cite:

```bibtex
@misc{su2025readinghumanemotion,
  title={Reading Human Emotion Through Machine Eyes},
  author={Su, Yushuo and others},
  year={2025},
  note={Under review}
}
```
