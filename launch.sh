#!/bin/bash

export FM_MODEL="openrouter/openai/gpt-oss-20b"
export FM_HF_DATASET="shlomihod/civil-comments-wilds/all"
export FM_DATASET_SPLIT_SEED="1"

# Disable PyTorch warnings and problematic behaviors
export PYTHONWARNINGS="ignore"
export STREAMLIT_SERVER_FILE_WATCHER_TYPE="none"
export STREAMLIT_SERVER_RUN_ON_SAVE="false"

# Activate the uv-managed venv created at the repo root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/.venv/bin/activate"

streamlit run "$SCRIPT_DIR/app.py"
