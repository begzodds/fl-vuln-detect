"""
Partition DiverseVul across N clients using IID or Dirichlet non-IID splits.
Usage: python src/utils/partition.py --n_clients 10 --alpha 0.5
"""
from __future__ import annotations
import argparse
import json
import numpy as np
from pathlib import Path


def dirichlet_partition(
    samples: list[dict],
    n_clients: int,
    alpha: float,
    seed: int = 42,
) -> list[list[dict]]:
    """Non-IID partition via Dirichlet distribution."""
    rng = np.random.default_rng(seed)
    labels = np.array([s["target"] for s in samples])
    n_classes = len(np.unique(labels))
    client_data: list[list] = [[] for _ in range(n_clients)]

    for cls in range(n_classes):
        cls_idx = np.where(labels == cls)[0]
        rng.shuffle(cls_idx)
        proportions = rng.dirichlet(alpha * np.ones(n_clients))
        proportions = (np.cumsum(proportions) * len(cls_idx)).astype(int)[:-1]
        splits = np.split(cls_idx, proportions)
        for cid, split in enumerate(splits):
            client_data[cid].extend([samples[i] for i in split])

    return client_data


def iid_partition(samples: list[dict], n_clients: int, seed: int = 42) -> list[list[dict]]:
    rng = np.random.default_rng(seed)
    idx = np.arange(len(samples))
    rng.shuffle(idx)
    splits = np.array_split(idx, n_clients)
    return [[samples[i] for i in split] for split in splits]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/processed/diversevul_train.json")
    parser.add_argument("--output_dir", default="data/partitions")
    parser.add_argument("--n_clients", type=int, default=10)
    parser.add_argument("--alpha", type=float, default=0.5, help="Dirichlet alpha; -1 for IID")
    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    with open(args.input) as f:
        samples = json.load(f)

    if args.alpha < 0:
        partitions = iid_partition(samples, args.n_clients, args.seed)
    else:
        partitions = dirichlet_partition(samples, args.n_clients, args.alpha, args.seed)

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    for cid, client_samples in enumerate(partitions):
        rng = np.random.default_rng(args.seed + cid)
        idx = np.arange(len(client_samples))
        rng.shuffle(idx)
        n_val = max(1, int(len(idx) * args.val_ratio))
        val_idx, train_idx = idx[:n_val], idx[n_val:]

        for split, split_idx in [("train", train_idx), ("val", val_idx)]:
            split_samples = [client_samples[i] for i in split_idx]
            with open(out / f"client_{cid}_{split}.json", "w") as f:
                json.dump(split_samples, f)

        print(f"Client {cid}: {len(train_idx)} train, {n_val} val")


if __name__ == "__main__":
    main()
