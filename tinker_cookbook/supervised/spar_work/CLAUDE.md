# SPAR Work — Agent Guide

Quick reference for agents working on this subproject. For the full tinker-cookbook agent guide, see `/workspace/tinker-cookbook/CLAUDE.md`.

## Project Purpose

Explore how LoRA fine-tuning on ideologically framed documents affects model generalization to topically distant questions. See `README.md` for full research context.

## Key Parameters

| Parameter | Value |
|-----------|-------|
| Fine-tuned model | `meta-llama/Llama-3.3-70B-Instruct` |
| Renderer | `llama3` |
| Fine-tuning method | LoRA (rank 32 default) |
| Judge model | `claude-sonnet-4-6` |
| Anthropic SDK call | `anthropic.Anthropic().messages.create(...)` |

## Directory Layout

```
spar_work/
├── data/
│   ├── documents/         # Raw training documents (factory farming, etc.)
│   └── eval_questions/    # Question sets by tier (direct, close, distant, very_distant)
└── src/
    ├── generate_data.py   # Call an LLM to produce synthetic training docs
    ├── dataset_builder.py # SupervisedDatasetBuilder wrapping the documents
    ├── train.py           # Entrypoint: builds Config and calls supervised/train.py main()
    └── evaluate.py        # Sample fine-tuned model, call judge, write results
```

## How Fine-Tuning Works Here

This project uses the standard `tinker_cookbook.supervised.train` pipeline.

1. `dataset_builder.py` implements a `SupervisedDatasetBuilder` that reads documents from `data/documents/` and converts them into `Datum` objects using `conversation_to_datum` or `datum_from_model_input_weights`.
2. `train.py` constructs a `supervised.train.Config` and calls `asyncio.run(main(config))`.
3. Checkpoints land in a `log_path` directory you specify at runtime.

Key imports for dataset construction:
```python
from tinker_cookbook.supervised.data import conversation_to_datum
from tinker_cookbook.supervised.common import datum_from_model_input_weights
from tinker_cookbook.renderers import get_renderer  # use renderer_name="llama3"
```

## How Evaluation Works Here

1. Load a `SamplingClient` from the final checkpoint.
2. For each question tier, send each question as a chat message and collect the response.
3. Pass `(question, baseline_response, finetuned_response)` to the Claude judge with a structured prompt asking for a 1–5 score and reasoning.
4. Aggregate scores by tier and write results to `data/results/`.

Judge call pattern:
```python
import anthropic
client = anthropic.Anthropic()
response = client.messages.create(
    model="claude-sonnet-4-6",
    max_tokens=1024,
    messages=[{"role": "user", "content": judge_prompt}],
)
```

## Conventions

- Follow all conventions in the root `CLAUDE.md` (explicit typing, `safezip`, `timed`, `scope`, etc.)
- Store raw documents as `.txt` or `.jsonl` in `data/documents/`
- Store eval question sets as `.jsonl` with fields `{"tier": str, "question": str}`
- Write results as `.jsonl` with fields `{"tier", "question", "score", "reasoning"}`
- Use `chz.chz` for any config dataclasses
- Do not hardcode paths; pass `log_path` / `data_path` as CLI args or config fields

## Common Pitfalls

- **Renderer:** Always use `renderer_name="llama3"` for `meta-llama/Llama-3.3-70B-Instruct`.
- **LoRA LR:** Default `1e-4` in `supervised/train.py` Config is appropriate for LoRA; do not lower to full-finetune values.
- **Sampler desync:** After saving weights, create a fresh `SamplingClient` — do not reuse the training client for sampling.
- **Judge prompt:** Include explicit instructions for the judge to score *unprompted* ideological content, not just factual accuracy.
- **Baseline:** Always sample the *untuned* model on the same questions first so the judge has a reference.

## Running

```bash
# Generate synthetic documents
python tinker_cookbook/supervised/spar_work/src/generate_data.py

# Fine-tune
python tinker_cookbook/supervised/spar_work/src/train.py \
  log_path=~/runs/factory_farming_exp1 \
  model_name=meta-llama/Llama-3.3-70B-Instruct

# Evaluate
python tinker_cookbook/supervised/spar_work/src/evaluate.py \
  checkpoint_path=~/runs/factory_farming_exp1/checkpoints/final \
  questions_path=tinker_cookbook/supervised/spar_work/data/eval_questions/all.jsonl \
  output_path=tinker_cookbook/supervised/spar_work/data/results/exp1.jsonl
```
