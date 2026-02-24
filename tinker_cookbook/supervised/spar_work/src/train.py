"""Fine-tuning entrypoint for SPAR experiments.

Usage (from repo root):
    python tinker_cookbook/supervised/spar_work/src/train.py \\
        --log_path ~/runs/factory_farming_neutral_exp1 \\
        --synth_docs_path /workspace/data/synth_docs/factory_farming_netural_tone/synth_docs.jsonl

Optional flags mirror train.Config fields (see --help).
"""

import argparse
import asyncio
import os
import sys

# Make sibling modules importable when run as a script.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dataset_builder import SynthDocDatasetBuilder  # noqa: E402
from tinker_cookbook import cli_utils  # noqa: E402
from tinker_cookbook.supervised import train  # noqa: E402

MODEL_NAME = "meta-llama/Llama-3.3-70B-Instruct"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="SPAR LoRA fine-tuning on synthetic documents."
    )
    parser.add_argument(
        "--log_path", required=True, help="Directory for checkpoints and logs."
    )
    parser.add_argument(
        "--synth_docs_path",
        required=True,
        help="Path to synth_docs.jsonl file.",
    )
    parser.add_argument("--model_name", default=MODEL_NAME)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--max_length", type=int, default=8192)
    parser.add_argument(
        "--test_size",
        type=int,
        default=50,
        help="Docs held out as a test set for NLL tracking (0 to disable).",
    )
    parser.add_argument("--num_epochs", type=int, default=1)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--lora_rank", type=int, default=32)
    parser.add_argument("--wandb_project", default=None)
    parser.add_argument("--wandb_name", default=None)
    return parser.parse_args()


def build_config(args: argparse.Namespace) -> train.Config:
    dataset_builder = SynthDocDatasetBuilder(
        synth_docs_path=args.synth_docs_path,
        model_name=args.model_name,
        batch_size=args.batch_size,
        max_length=args.max_length,
        test_size=args.test_size,
    )
    return train.Config(
        log_path=args.log_path,
        model_name=args.model_name,
        dataset_builder=dataset_builder,
        learning_rate=args.learning_rate,
        num_epochs=args.num_epochs,
        lora_rank=args.lora_rank,
        wandb_project=args.wandb_project,
        wandb_name=args.wandb_name,
    )


if __name__ == "__main__":
    args = parse_args()
    config = build_config(args)
    cli_utils.check_log_dir(config.log_path, behavior_if_exists="ask")
    asyncio.run(train.main(config))
