"""
Label noise injection for controlled experiments.
Supports symmetric, asymmetric, and instance-dependent noise.
"""
from __future__ import annotations
import numpy as np
import json
from pathlib import Path


def inject_symmetric_noise(labels: list[int], noise_rate: float, n_classes: int = 2, seed: int = 42) -> list[int]:
    """Randomly flip labels with uniform probability."""
    rng = np.random.default_rng(seed)
    noisy = labels.copy()
    n_noisy = int(len(labels) * noise_rate)
    noisy_idx = rng.choice(len(labels), n_noisy, replace=False)
    for i in noisy_idx:
        noisy[i] = rng.choice([c for c in range(n_classes) if c != labels[i]])
    return noisy


def inject_asymmetric_noise(labels: list[int], noise_rate: float, seed: int = 42) -> list[int]:
    """Flip only vulnerable→clean (models real-world under-labeling)."""
    rng = np.random.default_rng(seed)
    noisy = labels.copy()
    vuln_idx = [i for i, l in enumerate(labels) if l == 1]
    n_flip = int(len(vuln_idx) * noise_rate)
    flip_idx = rng.choice(vuln_idx, n_flip, replace=False)
    for i in flip_idx:
        noisy[i] = 0
    return noisy


def add_noise_to_partition(partition_path: Path, noise_rate: float, noise_type: str = "symmetric", seed: int = 42) -> Path:
    """Load a partition JSON, inject noise, save with _noisy suffix."""
    with open(partition_path) as f:
        samples = json.load(f)

    labels = [s["target"] for s in samples]
    if noise_type == "symmetric":
        noisy_labels = inject_symmetric_noise(labels, noise_rate, seed=seed)
    elif noise_type == "asymmetric":
        noisy_labels = inject_asymmetric_noise(labels, noise_rate, seed=seed)
    else:
        raise ValueError(f"Unknown noise type: {noise_type}")

    for sample, nl in zip(samples, noisy_labels):
        sample["noisy_target"] = nl
        sample["original_target"] = sample["target"]
        sample["target"] = nl  # overwrite for training

    out_path = partition_path.with_stem(partition_path.stem + f"_noisy{int(noise_rate*100)}")
    with open(out_path, "w") as f:
        json.dump(samples, f)
    return out_path
