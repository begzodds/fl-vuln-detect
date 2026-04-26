"""
Local training logic for a Flower FL client.
"""
from __future__ import annotations
from typing import Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.utils.metrics import compute_metrics


def train_one_round(
    model: nn.Module,
    train_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epochs: int = 1,
    proximal_mu: Optional[float] = None,  # FedProx mu; None = standard FedAvg
    global_params: Optional[list] = None,
) -> dict:
    """Train model for one FL round (local epochs)."""
    model.train()
    model.to(device)
    total_loss, total_correct, total_samples = 0.0, 0, 0

    for epoch in range(epochs):
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}", leave=False):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            optimizer.zero_grad()
            outputs = model(input_ids, attention_mask, labels=labels)
            loss = outputs["loss"]

            # FedProx proximal term
            if proximal_mu is not None and global_params is not None:
                prox_loss = 0.0
                for param, global_param in zip(model.parameters(), global_params):
                    prox_loss += torch.norm(param - global_param.to(device)) ** 2
                loss += (proximal_mu / 2) * prox_loss

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            preds = outputs["logits"].argmax(dim=-1)
            total_correct += (preds == labels).sum().item()
            total_samples += labels.size(0)
            total_loss += loss.item() * labels.size(0)

    return {
        "train_loss": total_loss / total_samples,
        "train_acc": total_correct / total_samples,
    }


def evaluate(model: nn.Module, val_loader: DataLoader, device: torch.device) -> dict:
    """Evaluate model on a validation DataLoader."""
    model.eval()
    model.to(device)
    all_preds, all_labels = [], []
    total_loss, total_samples = 0.0, 0

    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            outputs = model(input_ids, attention_mask, labels=labels)
            total_loss += outputs["loss"].item() * labels.size(0)
            total_samples += labels.size(0)
            all_preds.extend(outputs["logits"].argmax(dim=-1).cpu().tolist())
            all_labels.extend(labels.cpu().tolist())

    metrics = compute_metrics(all_preds, all_labels)
    metrics["val_loss"] = total_loss / total_samples
    return metrics
