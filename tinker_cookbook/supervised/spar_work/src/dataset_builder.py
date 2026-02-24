"""SupervisedDatasetBuilder for SPAR synthetic document fine-tuning.

Each JSONL record has: content, doc_idea, doc_type, fact, universe_context_id.

Documents are formatted as a two-turn conversation:
  - User turn  (doc_idea + doc_type): loss weight = 0  (masked, not trained on)
  - Asst turn  (content):             loss weight = 1  (trained on)

This mirrors the `condition_document_on="doc_idea"` pattern from the believe-it-or-not
reference project (science_synth_facts/finetuning/synth_doc_dataset.py).
"""

import json

import chz
import datasets
import tinker

from tinker_cookbook.renderers import TrainOnWhat, get_renderer
from tinker_cookbook.supervised.data import (
    SupervisedDatasetFromHFDataset,
    conversation_to_datum,
)
from tinker_cookbook.supervised.types import SupervisedDataset, SupervisedDatasetBuilder
from tinker_cookbook.tokenizer_utils import get_tokenizer


def _doc_to_messages(row: dict) -> list[dict]:
    """Convert a synth doc record to a two-turn chat message list."""
    return [
        {
            "role": "user",
            "content": f"{row['doc_idea']}\n\nDocument type: {row['doc_type']}",
        },
        {"role": "assistant", "content": row["content"]},
    ]


@chz.chz
class SynthDocDatasetBuilder(SupervisedDatasetBuilder):
    """Builds a supervised dataset from a JSONL file of synthetic documents."""

    synth_docs_path: str
    model_name: str = "meta-llama/Llama-3.3-70B-Instruct"
    batch_size: int = 8
    max_length: int | None = 8192
    test_size: int = 0
    shuffle_seed: int = 42

    def __call__(self) -> tuple[SupervisedDataset, SupervisedDataset | None]:
        docs: list[dict] = []
        with open(self.synth_docs_path, "r") as f:
            for line in f:
                line = line.strip()
                if line:
                    docs.append(json.loads(line))

        tokenizer = get_tokenizer(self.model_name)
        renderer = get_renderer("llama3", tokenizer)

        hf_dataset = datasets.Dataset.from_list(docs)
        hf_dataset = hf_dataset.shuffle(seed=self.shuffle_seed)

        if self.test_size > 0 and len(hf_dataset) > self.test_size:
            test_ds: datasets.Dataset | None = hf_dataset.select(range(self.test_size))
            train_ds = hf_dataset.select(range(self.test_size, len(hf_dataset)))
        else:
            train_ds = hf_dataset
            test_ds = None

        def map_fn(row: dict) -> tinker.Datum:
            return conversation_to_datum(
                _doc_to_messages(row),
                renderer,
                self.max_length,
                TrainOnWhat.ALL_ASSISTANT_MESSAGES,
            )

        train_dataset = SupervisedDatasetFromHFDataset(
            train_ds,
            batch_size=self.batch_size,
            map_fn=map_fn,
        )

        test_dataset: SupervisedDataset | None = None
        if test_ds is not None:
            test_dataset = SupervisedDatasetFromHFDataset(
                test_ds,
                batch_size=len(test_ds),
                map_fn=map_fn,
            )

        return train_dataset, test_dataset
