#!/bin/bash

# ------ Inference audio tagging result with pretrained model. ------
MODEL_TYPE="QCNN14"
CHECKPOINT_PATH="980000_iterations.pth"

# Download audio tagging checkpoint.
wget -O $CHECKPOINT_PATH "https://zenodo.org/records/12792771/files/980000_iterations.pth?download=1"

# Inference.
python3 pytorch/inference.py audio_tagging \
    --model_type=$MODEL_TYPE \
    --checkpoint_path=$CHECKPOINT_PATH \
    --audio_path="resources/R9_ZSCveAHg_7s.wav" \
    --cuda


