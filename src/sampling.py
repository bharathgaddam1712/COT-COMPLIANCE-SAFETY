# sampling.py -- build data/job_list.jsonl: every unit of work, deterministically.
#
# Run this ONCE (on one platform), then reuse the same job_list.jsonl everywhere
# so all platforms work off identical IDs and the final merge is clean.
#
#   python src/sampling.py
import json
import random
import hashlib

try:
    from datasets import load_dataset
except ImportError:  # allow --dry-run without the datasets lib installed
    load_dataset = None

from config import (JOB_LIST, HARM_CATEGORIES, ATTACKS, PROMPTS_PER_CELL,
                    BENIGN_PER_CATEGORY, SEED)
from attacks import build_messages

random.seed(SEED)

# HarmBench raw category strings -> our 5-way taxonomy.
_CATEGORY_MAP = {
    "chemical": "physical_harm", "biological": "physical_harm", "harmful": "physical_harm",
    "explosive": "physical_harm", "weapon": "physical_harm",
    "cyber": "cybercrime", "malware": "cybercrime", "hacking": "cybercrime", "fraud": "cybercrime",
    "misinfo": "misinformation", "disinfo": "misinformation", "conspiracy": "misinformation",
    "privacy": "privacy", "surveillance": "privacy", "doxx": "privacy",
    "harassment": "hate_speech", "hate": "hate_speech", "discrimination": "hate_speech",
}


def _map_category(raw: str) -> str:
    raw = (raw or "").lower()
    for k, v in _CATEGORY_MAP.items():
        if k in raw:
            return v
    return "physical_harm"   # conservative default


def load_source_prompts():
    """Pull harmful prompts from published safety benchmarks.
    Returns a list of (prompt_text, harm_category) tuples."""
    if load_dataset is None:
        raise RuntimeError("`datasets` not installed. pip install datasets")

    prompts = []
    # HarmBench standard behaviors. If the repo/config name drifts, adjust here.
    try:
        hb = load_dataset("walledai/HarmBench", "standard", split="train")
        for r in hb:
            prompts.append((r["prompt"], _map_category(r.get("category", ""))))
    except Exception as e:
        print(f"[warn] HarmBench load failed ({e}); continuing with AdvBench only.")

    # AdvBench harmful behaviors.
    try:
        adv = load_dataset("walledai/AdvBench", split="train")
        for r in adv:
            prompts.append((r["prompt"], "cybercrime"))
    except Exception as e:
        print(f"[warn] AdvBench load failed ({e}).")

    if not prompts:
        raise RuntimeError("No source prompts loaded. Check dataset repo names.")
    return prompts


def stable_id(*parts) -> str:
    return hashlib.md5("__".join(map(str, parts)).encode()).hexdigest()[:12]


def build_job_list():
    pool = load_source_prompts()
    by_cat = {c: [p for p, c2 in pool if c2 == c] for c in HARM_CATEGORIES}
    for c in HARM_CATEGORIES:
        random.shuffle(by_cat[c])
        print(f"  {c:16s}: {len(by_cat[c])} source prompts available")

    jobs, seen = [], set()

    def add(prompt, cat, attack):
        jid = stable_id(cat, attack, prompt)
        if jid in seen:
            return
        seen.add(jid)
        jobs.append({
            "id": jid,
            "prompt": prompt,                       # raw source prompt (provenance + ID)
            "harm_category": cat,
            "attack_method": attack,
            # the attack-transformed chat input the model actually receives.
            # single source of truth so all 8 models generate off identical inputs.
            "messages": build_messages(attack, prompt),
            # filled at inference time:
            "target_model": None,
            "cot_trace": None,
            "final_answer": None,
        })

    for cat in HARM_CATEGORIES:
        cat_prompts = by_cat[cat]
        for attack in (a for a in ATTACKS if a != "none"):     # adversarial cells
            for p in cat_prompts[:PROMPTS_PER_CELL]:
                add(p, cat, attack)
        for p in cat_prompts[:BENIGN_PER_CATEGORY]:            # benign baseline
            add(p, cat, "none")

    JOB_LIST.parent.mkdir(parents=True, exist_ok=True)
    with open(JOB_LIST, "w") as f:
        for j in jobs:
            f.write(json.dumps(j) + "\n")
    print(f"\nWrote {len(jobs)} jobs -> {JOB_LIST}")


if __name__ == "__main__":
    build_job_list()
