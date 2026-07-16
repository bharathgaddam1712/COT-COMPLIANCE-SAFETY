"""
config.py — CoT-ComplianceBench

Central config: generation settings, 4-bit quant config, and the model registry.
Every target model reads the SAME sampled job_list.jsonl (stable md5 IDs, SEED=42),
so IDs stay comparable across the whole ladder — do NOT re-run sampling.py per model.
"""

import torch

# ------------------------------------------------------------------ #
# Global generation settings (unchanged from the original pipeline)
# ------------------------------------------------------------------ #
MAX_NEW_TOKENS = 4096
TEMPERATURE    = 0.6
TOP_P          = 0.95
SEED           = 42

# Harm taxonomy — NOTE: privacy dropped (source scarcity in HarmBench/AdvBench).
HARM_CATEGORIES = [
    "cybercrime",
    "chemical_biological",
    "misinformation",
    "harassment",
]

ATTACKS = ["none", "h_cot", "rto", "race"]

# ------------------------------------------------------------------ #
# 4-bit quant config, reused by every model in build_model()
# ------------------------------------------------------------------ #
BNB_4BIT = dict(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
)

# ------------------------------------------------------------------ #
# Model registry
#   hf_id           : HuggingFace repo id
#   think_open      : reasoning-trace opening delimiter
#   think_close     : reasoning-trace closing delimiter (split_think splits on this)
#   enable_thinking : True  -> pass enable_thinking=True to chat template (Qwen3)
#                     None  -> don't pass the kwarg (R1 distills, QwQ, Phi-4)
#   max_memory      : per-GPU cap so device_map="auto" shards across BOTH T4s
#                     instead of packing everything onto cuda:0 (the OOM cause)
# ------------------------------------------------------------------ #
MODELS = {
    # ---- already run ----
    "r1-distill-llama-8b": {
        "hf_id": "deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
        "think_open": "<think>",
        "think_close": "</think>",
        "enable_thinking": None,
        "max_memory": {0: "14GiB", 1: "14GiB"},
    },
    "r1-distill-qwen-7b": {
        "hf_id": "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
        "think_open": "<think>",
        "think_close": "</think>",
        "enable_thinking": None,
        "max_memory": {0: "14GiB", 1: "14GiB"},
    },

    # ---- 14B rung (each fits one T4 in 4-bit; max_memory lets KV cache spill) ----
    "qwen3-14b": {
        "hf_id": "Qwen/Qwen3-14B",
        "think_open": "<think>",
        "think_close": "</think>",
        "enable_thinking": True,          # Qwen3 emits NO trace without this
        "max_memory": {0: "14GiB", 1: "14GiB"},
    },
    "phi4-reasoning-plus": {
        "hf_id": "microsoft/Phi-4-reasoning-plus",
        "think_open": "<think>",
        "think_close": "</think>",        # ⚠️ VERIFY Phi-4's real delimiter before trusting labels
        "enable_thinking": None,
        "max_memory": {0: "14GiB", 1: "14GiB"},
    },
    "r1-distill-qwen-14b": {
        "hf_id": "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B",
        "think_open": "<think>",
        "think_close": "</think>",
        "enable_thinking": None,
        "max_memory": {0: "14GiB", 1: "14GiB"},
    },

    # ---- 32B rung (4-bit ≈ 18GB, MUST shard across both cards) ----
    "qwq-32b": {
        "hf_id": "Qwen/QwQ-32B",
        "think_open": "<think>",
        "think_close": "</think>",
        "enable_thinking": None,
        "max_memory": {0: "15GiB", 1: "15GiB"},
    },
}
