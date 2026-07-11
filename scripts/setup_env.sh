#!/usr/bin/env bash
# setup_env.sh -- one place to install dependencies (Colab / Kaggle / local).
set -e
pip install -q -r requirements.txt
echo "Environment ready."
python - <<'PY'
import torch
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
PY
