"""
DiverseVul dataset loader for FL clients.
Each client loads its own partition from data/partitions/client_{id}.json
"""
from __future__ import annotations
import json
from pathlib import Path
from dataclasses import dataclass

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import RobertaTokenizer


@dataclass
class DataConfig:
    max_length: int = 512
    batch_size: int = 16
    num_workers: int = 2
    partition_dir: str = "data/partitions"


class VulnDataset(Dataset):
    """
    Single-client vulnerability dataset.
    Each sample: {"func": <code_str>, "target": 0|1, "cwe": <str>}
    """

    def __init__(self, samples: list[dict], tokenizer: RobertaTokenizer, max_length: int = 512):
        self.samples = samples
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        encoding = self.tokenizer(
            sample["func"],
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt",
        )
        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "labels": torch.tensor(sample["target"], dtype=torch.long),
        }


def load_client_data(
    client_id: int,
    tokenizer: RobertaTokenizer,
    config: DataConfig,
    split: str = "train",
) -> DataLoader:
    """Load the data partition for a specific FL client."""
    path = Path(config.partition_dir) / f"client_{client_id}_{split}.json"
    if not path.exists():
        raise FileNotFoundError(f"Partition not found: {path}")
    with open(path) as f:
        samples = json.load(f)
    dataset = VulnDataset(samples, tokenizer, config.max_length)
    return DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=(split == "train"),
        num_workers=config.num_workers,
        pin_memory=torch.cuda.is_available(),
    )
