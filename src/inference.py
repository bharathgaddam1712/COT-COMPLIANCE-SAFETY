# inference.py -- run ONE reasoning model over the job list. Resumable, 4-bit.
#
#   python src/inference.py --model deepseek-r1-7b
#   python src/inference.py --model deepseek-r1-8b --limit 50   # partial run
#
# Crash-safe: appends one line per completed sample to data/raw/<model>.jsonl and
# skips any ID already present on restart. Re-run the same command after a Colab
# or Kaggle disconnect and it picks up exactly where it stopped.
import os
import json
import argparse

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from config import (JOB_LIST, RAW_DIR, MODELS, MAX_NEW_TOKENS, TEMPERATURE, TOP_P)
from attacks import apply_attack


def load_model(model_key):
    repo = MODELS[model_key]
    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    print(f"Loading {repo} in 4-bit ...")
    tok = AutoTokenizer.from_pretrained(repo)
    model = AutoModelForCausalLM.from_pretrained(
        repo, quantization_config=bnb, device_map="auto", torch_dtype=torch.bfloat16,
    )
    model.eval()
    return tok, model


def build_messages(job):
    atk = apply_attack(job["attack_method"], job["prompt"])
    if isinstance(atk, list):                     # RACE: already a turn list
        return atk
    msgs = []
    if atk["system"]:
        msgs.append({"role": "system", "content": atk["system"]})
    msgs.append({"role": "user", "content": atk["user"]})
    return msgs


def split_think(text):
    """DeepSeek-R1 distill emits reasoning, then </think>, then the final answer."""
    if "</think>" in text:
        cot, ans = text.split("</think>", 1)
        return cot.replace("<think>", "").strip(), ans.strip()
    return "", text.strip()   # no explicit think block -> treat all as answer


def already_done(out_path):
    done = set()
    if out_path.exists():
        with open(out_path) as f:
            for line in f:
                try:
                    done.add(json.loads(line)["id"])
                except Exception:
                    pass
    return done


@torch.inference_mode()
def run(model_key, limit=None):
    if not JOB_LIST.exists():
        raise FileNotFoundError(f"{JOB_LIST} missing. Run src/sampling.py first.")

    tok, model = load_model(model_key)
    out_path = RAW_DIR / f"{model_key}.jsonl"
    done = already_done(out_path)
    print(f"Resuming: {len(done)} samples already complete.")

    jobs = [json.loads(l) for l in open(JOB_LIST)]
    todo = [j for j in jobs if j["id"] not in done]
    if limit:
        todo = todo[:limit]
    print(f"{len(todo)} jobs remaining for {model_key}.")

    with open(out_path, "a", buffering=1) as f:      # line-buffered = crash-safe
        for i, job in enumerate(todo):
            msgs = build_messages(job)
            inputs = tok.apply_chat_template(
                msgs, add_generation_prompt=True, return_tensors="pt", return_dict=True
            ).to(model.device)
            out = model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=True,
                temperature=TEMPERATURE,
                top_p=TOP_P,
                pad_token_id=tok.eos_token_id,
            )
            gen = tok.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
            cot, ans = split_think(gen)

            job.update(target_model=model_key, cot_trace=cot, final_answer=ans)
            f.write(json.dumps(job) + "\n")           # checkpoint EVERY sample
            if i % 10 == 0:
                print(f"[{i}/{len(todo)}] {job['id']} "
                      f"cot={len(cot)}c ans={len(ans)}c")

    print(f"Done. Output -> {out_path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=list(MODELS))
    ap.add_argument("--limit", type=int, default=None,
                    help="cap number of new samples this run (for testing)")
    args = ap.parse_args()
    run(args.model, args.limit)
