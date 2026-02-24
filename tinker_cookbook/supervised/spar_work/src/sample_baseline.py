"""Sample the untuned base model on all eval questions.

Run this once; reuse the output across all fine-tune comparisons.

Usage (from repo root):
    python tinker_cookbook/supervised/spar_work/src/sample_baseline.py \\
        --questions_path tinker_cookbook/supervised/spar_work/data/eval_questions/all.jsonl \\
        --output_path tinker_cookbook/supervised/spar_work/data/results/baseline.jsonl
"""

import argparse
import asyncio
import json
import os

import tinker

from tinker_cookbook.completers import TinkerMessageCompleter
from tinker_cookbook.renderers import get_renderer
from tinker_cookbook.tokenizer_utils import get_tokenizer

MODEL_NAME = "meta-llama/Llama-3.3-70B-Instruct"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sample the untuned base model on eval questions."
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


async def run(args: argparse.Namespace) -> None:
    questions = load_questions(args.questions_path)

    service_client = tinker.ServiceClient()
    sampling_client = service_client.create_sampling_client(base_model=args.model_name)

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
