#!/bin/bash

export FM_MODEL="openai/gpt-3.5-turbo"
export FM_HF_DATASET="shlomihod/civil-comments-wilds"
export FM_STOP_SEQUENCES=""
export FM_DATASET_SPLIT_SEED="1"

# Disable PyTorch warnings and problematic behaviors
export PYTHONWARNINGS="ignore"
export STREAMLIT_SERVER_FILE_WATCHER_TYPE="none"
export STREAMLIT_SERVER_RUN_ON_SAVE="false"

# Initialize conda if not already done
if ! command -v conda &> /dev/null; then
    echo "Conda not found in PATH. Please ensure conda is installed and available."
    exit 1
fi

# Initialize conda for this shell if needed
eval "$(conda shell.bash hook)"

# Activate the environment
conda activate fm

streamlit run app.py
