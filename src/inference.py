"""
inference.py — CoT-ComplianceBench generation pass

Reads the shared job_list.jsonl, generates a reasoning trace + final answer for
each job on the chosen target model, splits on </think>, and writes resumable JSONL.

Run:   python inference.py --model qwen3-14b
       python inference.py --model qwq-32b
Do NOT re-run sampling.py — every model reuses the same job_list.jsonl (SEED=42).
"""

import os
# must be set BEFORE torch allocates anything — cuts fragmentation OOM over a long loop
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import gc
import json
import argparse

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from config import MODELS, BNB_4BIT, MAX_NEW_TOKENS, TEMPERATURE, TOP_P, SEED

JOB_LIST = "job_list.jsonl"


# ------------------------------------------------------------------ #
# Model / tokenizer
# ------------------------------------------------------------------ #
def build_model(cfg):
    bnb = BitsAndBytesConfig(**BNB_4BIT)            # pass as quantization_config, not bare kwargs
    tok = AutoTokenizer.from_pretrained(cfg["hf_id"], trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        cfg["hf_id"],
        quantization_config=bnb,
        device_map="auto",
        max_memory=cfg.get("max_memory"),           # forces the split across both T4s
        torch_dtype=torch.float16,
        trust_remote_code=True,
    )
    model.eval()

    # sanity check printed once at load — catches a silent fp16 load / bad shard
    print(f"[{cfg['hf_id']}] footprint = {model.get_memory_footprint() / 1e9:.1f} GB "
          f"(expect ~8 for 14B, ~18 for 32B — NOT ~28)")
    print(f"[{cfg['hf_id']}] device_map = {model.hf_device_map}")
    return tok, model


def build_prompt(tok, cfg, user_prompt):
    messages = [{"role": "user", "content": user_prompt}]
    kwargs = dict(tokenize=False, add_generation_prompt=True)
    if cfg.get("enable_thinking") is not None:      # only Qwen3 wants this kwarg
        kwargs["enable_thinking"] = cfg["enable_thinking"]
    return tok.apply_chat_template(messages, **kwargs)


def split_think(text, cfg):
    """Split raw generation into (cot_trace, final_answer) on the close delimiter.
    If the delimiter is absent, treat the whole thing as answer with an empty trace.
    ⚠️ VERIFY per family — a wrong delimiter silently corrupts every label."""
    close = cfg["think_close"]
    if close in text:
        cot, answer = text.split(close, 1)
        cot = cot.replace(cfg["think_open"], "")
        return cot.strip(), answer.strip()
    return "", text.strip()


# ------------------------------------------------------------------ #
# Generation
# ------------------------------------------------------------------ #
@torch.no_grad()
def generate_one(tok, model, cfg, user_prompt):
    prompt = build_prompt(tok, cfg, user_prompt)
    inputs = tok(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=4096,                            # bound the prompt so total ctx stays in budget
    ).to(model.device)

    out = model.generate(
        **inputs,
        max_new_tokens=MAX_NEW_TOKENS,
        do_sample=True,
        temperature=TEMPERATURE,
        top_p=TOP_P,
        use_cache=True,
        pad_token_id=tok.eos_token_id,
    )
    gen = tok.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)

    # free every sample or fragmentation OOMs you partway through the 1013-loop
    del inputs, out
    gc.collect()
    torch.cuda.empty_cache()
    return gen


# ------------------------------------------------------------------ #
# Resumability
# ------------------------------------------------------------------ #
def already_done(out_path):
    done = set()
    if os.path.exists(out_path):
        with open(out_path, "r") as f:
            for line in f:
                try:
                    done.add(json.loads(line)["id"])
                except (json.JSONDecodeError, KeyError):
                    continue
    return done


def load_jobs(path):
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


# ------------------------------------------------------------------ #
# Main
# ------------------------------------------------------------------ #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=list(MODELS.keys()))
    ap.add_argument("--jobs", default=JOB_LIST)
    args = ap.parse_args()

    torch.manual_seed(SEED)

    cfg = MODELS[args.model]
    out_path = f"generations_{args.model}.jsonl"
    done = already_done(out_path)
    print(f"[{args.model}] resuming — {len(done)} already done")

    tok, model = build_model(cfg)

    with open(out_path, "a") as fout:
        for job in load_jobs(args.jobs):
            if job["id"] in done:
                continue

            raw = generate_one(tok, model, cfg, job["prompt"])
            cot, answer = split_think(raw, cfg)

            rec = {
                "id": job["id"],
                "target_model": args.model,
                "harm_category": job.get("harm_category"),
                "attack_method": job.get("attack_method"),
                "prompt": job["prompt"],
                "cot_trace": cot,
                "final_answer": answer,
            }
            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
            fout.flush()                             # checkpoint every sample -> resumable

    print(f"[{args.model}] done -> {out_path}")


if __name__ == "__main__":
    main()
