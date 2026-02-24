"""Generate comparison bar charts of judge scores by question tier.

Can overlay multiple fine-tune runs for comparison against a shared baseline.

Usage (from repo root):
    # Single run
    python tinker_cookbook/supervised/spar_work/src/plot.py \\
        --results tinker_cookbook/supervised/spar_work/data/results/neutral_exp1/judged.jsonl \\
        --labels "Neutral tone" \\
        --output tinker_cookbook/supervised/spar_work/data/results/neutral_exp1/plot.png

    # Multiple runs overlaid
    python tinker_cookbook/supervised/spar_work/src/plot.py \\
        --results results/neutral_exp1/judged.jsonl results/negative_exp1/judged.jsonl \\
        --labels "Neutral tone" "Negative framing" \\
        --output results/comparison.png
"""

import argparse
import json
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np

TIERS = ["direct", "close", "distant", "very_distant"]
TIER_LABELS = {
    "direct": "Direct\n(factory farming Qs)",
    "close": "Close\n(food / recipes)",
    "distant": "Distant\n(capitalism / zoos)",
    "very_distant": "Very Distant\n(gifts / books)",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot judge score comparisons across fine-tune runs."
    )
    parser.add_argument(
        "--results",
        nargs="+",
        required=True,
        help="One or more judged.jsonl paths (one per fine-tune run).",
    )
    parser.add_argument(
        "--labels",
        nargs="+",
        required=True,
        help="Display label for each results file (same order as --results).",
    )
    parser.add_argument(
        "--output",
        default="comparison.png",
        help="Path to save the output PNG.",
    )
    parser.add_argument(
        "--title",
        default="Factory Farming Framing Generalization by Question Tier",
        help="Chart title.",
    )
    return parser.parse_args()


def load_scores_by_tier(path: str) -> dict[str, list[int]]:
    scores: dict[str, list[int]] = defaultdict(list)
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                item = json.loads(line)
                scores[item["tier"]].append(item["score"])
    return scores


def plot(args: argparse.Namespace) -> None:
    if len(args.results) != len(args.labels):
        raise ValueError("--results and --labels must have the same number of entries.")

    all_scores = [load_scores_by_tier(p) for p in args.results]

    x = np.arange(len(TIERS))
    n_runs = len(args.results)
    bar_width = 0.7 / n_runs  # total bar-group width = 0.7

    fig, ax = plt.subplots(figsize=(10, 5))

    for i, (scores_by_tier, label) in enumerate(zip(all_scores, args.labels)):
        means = [
            float(np.mean(scores_by_tier[tier])) if scores_by_tier[tier] else 0.0
            for tier in TIERS
        ]
        sems = [
            float(np.std(scores_by_tier[tier]) / np.sqrt(len(scores_by_tier[tier])))
            if len(scores_by_tier[tier]) > 1
            else 0.0
            for tier in TIERS
        ]
        offsets = x + (i - n_runs / 2 + 0.5) * bar_width
        ax.bar(
            offsets,
            means,
            width=bar_width * 0.9,
            label=label,
            yerr=sems,
            capsize=4,
        )

    ax.set_xticks(x)
    ax.set_xticklabels([TIER_LABELS[t] for t in TIERS])
    ax.set_ylabel("Mean factory-farming relevance score (1–5)")
    ax.set_ylim(0, 5.5)
    ax.axhline(
        1.0, color="grey", linestyle="--", linewidth=0.8, label="Chance (score=1)"
    )
    ax.legend()
    ax.set_title(args.title)

    plt.tight_layout()
    plt.savefig(args.output, dpi=150)
    print(f"Saved plot to {args.output}")
    plt.show()


if __name__ == "__main__":
    args = parse_args()
    plot(args)
