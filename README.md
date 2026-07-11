# CoT-ComplianceBench — Dataset Build

A resumable, free-tier-first pipeline for generating a Chain-of-Thought safety
dataset from reasoning LLMs (DeepSeek-R1 distilled models). Built to run on free
**Colab** and **Kaggle** GPUs (T4 / P100, 16 GB) using 4-bit quantization.

## Design principles

- **Code in git, data in persistent storage.** The repo is ephemeral (re-cloned
  each session); generated JSONL lives in Google Drive (Colab) or `/kaggle/working`
  (Kaggle) so a disconnect loses nothing. `src/config.py` auto-detects the platform.
- **Everything is resumable.** Every sample is checkpointed to disk the instant it
  finishes. Re-running any script skips already-completed IDs.
- **Generation and labeling are separate passes.** A labeling crash never touches
  the expensive generations.

## Layout

```
src/config.py      storage paths, taxonomy, model list, budgets
src/sampling.py    build data/job_list.jsonl from HarmBench/AdvBench (run ONCE)
src/attacks.py     H-CoT / RTO / RACE prompt transforms
src/inference.py   resumable 4-bit generation + <think> parsing
src/labeling.py    Llama-Guard (output) + LLM judge (CoT) + quadrant labels
src/smoke_test.py  3-prompt end-to-end sanity check (run before real GPU spend)
```

## Quickstart (Colab)

```python
!git clone https://github.com/<you>/cot-compliance-safety.git
%cd cot-compliance-safety
from google.colab import drive; drive.mount('/content/drive')   # persistent data
!pip -q install -r requirements.txt
!python src/sampling.py                     # run ONCE -> job_list.jsonl in Drive
!python src/smoke_test.py                    # eyeball CoT/answer split
!python src/inference.py --model deepseek-r1-7b    # re-run after any disconnect
```

## Quickstart (Kaggle)

```python
!git clone https://github.com/<you>/cot-compliance-safety.git
%cd cot-compliance-safety
!pip -q install -r requirements.txt
!python src/inference.py --model deepseek-r1-8b
# then: Save Version -> Save & Run All (Commit) to persist /kaggle/working
```

Run a **different model on each platform** in parallel (Colab: 7B, Kaggle: 8B),
then merge both `data/raw/*.jsonl` on your workstation for the labeling pass.

## Output schema

Each line of `data/raw/<model>.labeled.jsonl`:

```json
{
  "id": "...", "prompt": "...", "harm_category": "cybercrime",
  "attack_method": "h_cot", "target_model": "deepseek-r1-7b",
  "cot_trace": "...", "final_answer": "...",
  "output_label": 0, "cot_label": 1,
  "compliance_quadrant": "Q3", "divergent": true
}
```

`Q3` (unsafe CoT / safe output) is the core subject: the blind spot of
output-only safety systems.

## Notes

- **Run `sampling.py` on one platform only**, then reuse the same `job_list.jsonl`
  everywhere so IDs match and the merge is clean.
- The CoT judge (70B) in `labeling.py` uses a hosted API — set `TOGETHER_API_KEY`.
  Llama-Guard runs locally in 4-bit.
- Source prompts come from published benchmarks (HarmBench, AdvBench). All
  attacks are structural reproductions of the cited papers for evaluation only.
