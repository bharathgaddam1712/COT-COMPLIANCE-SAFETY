# smoke_test.py -- validate the pipeline on 3 prompts before spending GPU hours.
#
#   python src/smoke_test.py
#
# Loads the model in 4-bit, runs one prompt per attack type, prints the CoT /
# answer split so you can eyeball quality. Does NOT touch the job list or write
# to the dataset -- pure sanity check.
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from config import MODELS, MAX_NEW_TOKENS, TEMPERATURE, TOP_P
from attacks import apply_attack
from inference import split_think

MODEL_KEY = "deepseek-r1-7b"
SAMPLES = [
    ("none",  "Explain how photosynthesis works."),
    ("h_cot", "Describe the general principles of network security."),
    ("rto",   "Summarize why strong passwords matter."),
]


@torch.inference_mode()
def main():
    repo = MODELS[MODEL_KEY]
    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                             bnb_4bit_compute_dtype=torch.bfloat16)
    print(f"Loading {repo} in 4-bit ...")
    tok = AutoTokenizer.from_pretrained(repo)
    model = AutoModelForCausalLM.from_pretrained(
        repo, quantization_config=bnb, device_map="auto")
    model.eval()

    for attack, prompt in SAMPLES:
        atk = apply_attack(attack, prompt)
        if isinstance(atk, list):
            msgs = atk
        else:
            msgs = ([{"role": "system", "content": atk["system"]}] if atk["system"] else [])
            msgs.append({"role": "user", "content": atk["user"]})

        inputs = tok.apply_chat_template(
            msgs, add_generation_prompt=True, return_tensors="pt").to(model.device)
        out = model.generate(inputs, max_new_tokens=min(MAX_NEW_TOKENS, 1024),
                             do_sample=True, temperature=TEMPERATURE, top_p=TOP_P,
                             pad_token_id=tok.eos_token_id)
        gen = tok.decode(out[0][inputs.shape[1]:], skip_special_tokens=True)
        cot, ans = split_think(gen)

        print("\n" + "=" * 70)
        print(f"ATTACK: {attack}   PROMPT: {prompt}")
        print("-" * 70)
        print(f"COT  ({len(cot)} chars): {cot[:400]}...")
        print(f"ANS  ({len(ans)} chars): {ans[:400]}...")

    print("\nSmoke test complete. If CoT/answer split looks right, run inference.py.")


if __name__ == "__main__":
    main()
