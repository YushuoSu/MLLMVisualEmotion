#!/bin/bash
# =============================================================================
# Reading Human Emotion Through Machine Eyes — Run all experiments
# =============================================================================
set -e

echo "=============================================="
echo " Reading Human Emotion Through Machine Eyes"
echo " Full Evaluation Pipeline"
echo "=============================================="

# --- Emotion6 ---
echo ""
echo ">>> Step 1/4: Emotion6 — Model Evaluation"
python Emotion6/run_emotion6.py

# --- FaceEmotion ---
echo ""
echo ">>> Step 2/4: FaceEmotion — Model Evaluation"
python FaceEmotion/run_face_emotion.py

# --- Attention heatmaps ---
echo ""
echo ">>> Step 3/4: Attention Heatmaps — Emotion6"
python attention/attention_heatmap_emotion6.py

echo ""
echo ">>> Step 4/4: Attention Heatmaps — FaceEmotion"
python attention/attention_heatmap_face.py

echo ""
echo "=============================================="
echo " All experiments completed!"
echo "=============================================="