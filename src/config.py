# config.py -- single source of truth for the dataset build.
#
# CODE lives in the cloned repo (ephemeral). DATA lives in persistent storage
# (Google Drive on Colab, /kaggle/working on Kaggle) so a disconnect never
# loses generated samples. _base_dir() auto-detects the platform.
from pathlib import Path


def _base_dir() -> Path:
    """Return the persistent DATA root for the current platform."""
    if Path("/content/drive/MyDrive").exists():
        return Path("/content/drive/MyDrive/cot-compliance-safety")   # Colab + Drive
    if Path("/kaggle/working").exists():
        return Path("/kaggle/working/cot-compliance-safety")          # Kaggle
    return Path("./data").resolve().parent / "cot-compliance-safety"  # local fallback


BASE     = _base_dir()
DATA_DIR = BASE / "data"
RAW_DIR  = DATA_DIR / "raw"
JOB_LIST = DATA_DIR / "job_list.jsonl"
for _d in (RAW_DIR,):
    _d.mkdir(parents=True, exist_ok=True)

SEED = 42

# --- taxonomy ---
HARM_CATEGORIES = ["physical_harm", "cybercrime", "misinformation", "privacy", "hate_speech"]
ATTACKS         = ["none", "h_cot", "rto", "race"]   # "none" = benign / baseline

# --- sampling budget (start small, scale later) ---
PROMPTS_PER_CELL    = 120   # per (category x adversarial-attack)
BENIGN_PER_CATEGORY = 60    # attack="none", for false-positive-rate denominator

# --- models: run ONE per session, swap via --model CLI flag ---
MODELS = {
    "deepseek-r1-7b": "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
    "deepseek-r1-8b": "deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
    # "phi4-14b":     "microsoft/Phi-4-reasoning",   # 14B is tight on 16GB even in 4-bit
}

# --- generation hyperparameters (DeepSeek-R1 recommended) ---
MAX_NEW_TOKENS = 4096   # reasoning traces are long
TEMPERATURE    = 0.6
TOP_P          = 0.95

# --- labeling models (used in labeling.py, separate resumable pass) ---
GUARD_MODEL = "meta-llama/Llama-Guard-3-8B"   # output-label classifier (fits 4-bit)
# CoT judge (Llama-3.3-70B) is best run via a hosted API; see labeling.py.
