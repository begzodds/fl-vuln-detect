"""
Noise detection methods: LID-based (FedCorr) and Energy/Embedding-based (FedLN).
"""
from __future__ import annotations
import numpy as np
import torch
import torch.nn as nn
from sklearn.neighbors import NearestNeighbors


# ── LID-based (FedCorr) ─────────────────────────────────────────────────────

def compute_lid_score(embeddings: np.ndarray, k: int = 20) -> float:
    """
    Local Intrinsic Dimensionality of the embedding space.
    High LID → noisy client (FedCorr criterion).
    """
    nbrs = NearestNeighbors(n_neighbors=k + 1).fit(embeddings)
    distances, _ = nbrs.kneighbors(embeddings)
    distances = distances[:, 1:]  # exclude self
    distances = np.maximum(distances, 1e-10)
    r_max = distances[:, -1:]
    lid = -1.0 / np.mean(np.log(distances / r_max), axis=1)
    return float(np.mean(lid))


# ── Energy-based (FedLN) ────────────────────────────────────────────────────

def compute_energy_score(logits: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
    """
    Energy score per sample: -T * log(sum(exp(logits/T))).
    Low energy → clean sample; high energy → noisy candidate.
    """
    return -temperature * torch.logsumexp(logits / temperature, dim=1)


def detect_noisy_samples_energy(
    model: nn.Module,
    dataloader,
    device: torch.device,
    threshold_quantile: float = 0.8,
) -> list[int]:
    """Return indices of samples classified as noisy by energy score."""
    model.eval()
    energies, sample_indices = [], []
    idx = 0
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            outputs = model(input_ids, attention_mask)
            e = compute_energy_score(outputs["logits"]).cpu().numpy()
            energies.extend(e.tolist())
            sample_indices.extend(range(idx, idx + len(e)))
            idx += len(e)

    threshold = np.quantile(energies, threshold_quantile)
    return [i for i, e in zip(sample_indices, energies) if e > threshold]
