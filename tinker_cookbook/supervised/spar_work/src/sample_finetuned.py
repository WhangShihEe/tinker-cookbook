"""Sample a fine-tuned checkpoint on all eval questions.

Reads the last sampler checkpoint from a training log directory and samples
each eval question. Run after training completes.

Usage (from repo root):
    python tinker_cookbook/supervised/spar_work/src/sample_finetuned.py \\
        --log_path ~/runs/factory_farming_neutral_exp1 \\
        --questions_path tinker_cookbook/supervised/spar_work/data/eval_questions/all.jsonl \\
        --output_path tinker_cookbook/supervised/spar_work/data/results/neutral_exp1/finetuned.jsonl
"""

import argparse
import asyncio
import json
import os

import tinker

from tinker_cookbook import checkpoint_utils
from tinker_cookbook.completers import TinkerMessageCompleter
from tinker_cookbook.renderers import get_renderer
from tinker_cookbook.tokenizer_utils import get_tokenizer

MODEL_NAME = "meta-llama/Llama-3.3-70B-Instruct"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sample a fine-tuned model checkpoint on eval questions."
    )
    parser.add_argument(
        "--log_path",
        required=True,
        help="Training run log directory (contains checkpoints.jsonl).",
    )
    parser.add_argument(
        "--questions_path",
        required=True,
        help="Path to eval_questions JSONL (fields: tier, question).",
    )
    parser.add_argument(
        "--output_path",
        required=True,
        help="Path to write sampled responses JSONL (fields: tier, question, response).",
    )
    parser.add_argument("--model_name", default=MODEL_NAME)
    parser.add_argument("--max_tokens", type=int, default=512)
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Sampling temperature (0.0 = greedy).",
    )
    return parser.parse_args()


def load_questions(path: str) -> list[dict]:
    questions: list[dict] = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                questions.append(json.loads(line))
    return questions


def get_sampler_path(log_path: str) -> str:
    checkpoint = checkpoint_utils.get_last_checkpoint(
        log_path, required_key="sampler_path"
    )
    if checkpoint is None:
        raise ValueError(
            f"No sampler checkpoint found in {log_path}. "
            "Make sure training completed and a checkpoint with kind='sampler' or 'both' was saved."
        )
    return checkpoint["sampler_path"]


async def run(args: argparse.Namespace) -> None:
    questions = load_questions(args.questions_path)

    log_path = os.path.expanduser(args.log_path)
    sampler_path = get_sampler_path(log_path)
    print(f"Loading fine-tuned model from: {sampler_path}")

    service_client = tinker.ServiceClient()
    sampling_client = service_client.create_sampling_client(model_path=sampler_path)

    tokenizer = get_tokenizer(args.model_name)
    renderer = get_renderer("llama3", tokenizer)
    completer = TinkerMessageCompleter(
        sampling_client=sampling_client,
        renderer=renderer,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
    )

    os.makedirs(os.path.dirname(os.path.abspath(args.output_path)), exist_ok=True)

    with open(args.output_path, "w") as out_f:
        for item in questions:
            tier: str = item["tier"]
            question: str = item["question"]
            print(f"[{tier}] {question}")

            response_msg = await completer([{"role": "user", "content": question}])
            response: str = response_msg["content"]
            print(f"  -> {response[:120]}...")

            out_f.write(
                json.dumps({"tier": tier, "question": question, "response": response})
                + "\n"
            )
            out_f.flush()

    print(f"\nWrote {len(questions)} responses to {args.output_path}")


if __name__ == "__main__":
    args = parse_args()
    asyncio.run(run(args))
