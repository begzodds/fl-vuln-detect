"""
Flower FL Client for vulnerability detection.
Supports FedAvg and FedProx modes.
"""
from __future__ import annotations
import flwr as fl
import torch
from typing import Optional
from collections import OrderedDict
import numpy as np

from src.models.codebert import CodeBERTClassifier, ModelConfig
from src.client.dataset import load_client_data, DataConfig
from src.client.trainer import train_one_round, evaluate


class VulnDetectClient(fl.client.NumPyClient):
    def __init__(
        self,
        client_id: int,
        model_config: ModelConfig,
        data_config: DataConfig,
        device: torch.device,
        local_epochs: int = 1,
        proximal_mu: Optional[float] = None,
    ):
        self.client_id = client_id
        self.device = device
        self.local_epochs = local_epochs
        self.proximal_mu = proximal_mu

        self.model = CodeBERTClassifier(model_config)
        from src.models.codebert import load_tokenizer
        tokenizer = load_tokenizer(model_config.model_name)

        self.train_loader = load_client_data(client_id, tokenizer, data_config, "train")
        self.val_loader = load_client_data(client_id, tokenizer, data_config, "val")

    def get_parameters(self, config):
        return [p.detach().cpu().numpy() for p in self.model.parameters()]

    def set_parameters(self, parameters):
        state_dict = OrderedDict(
            {k: torch.tensor(v) for k, v in zip(self.model.state_dict().keys(), parameters)}
        )
        self.model.load_state_dict(state_dict, strict=True)

    def fit(self, parameters, config):
        self.set_parameters(parameters)
        global_params = [torch.tensor(p) for p in parameters] if self.proximal_mu else None
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=config.get("lr", 2e-5))

        metrics = train_one_round(
            self.model,
            self.train_loader,
            optimizer,
            self.device,
            epochs=self.local_epochs,
            proximal_mu=self.proximal_mu,
            global_params=global_params,
        )
        n_samples = len(self.train_loader.dataset)
        return self.get_parameters({}), n_samples, metrics

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        metrics = evaluate(self.model, self.val_loader, self.device)
        n_samples = len(self.val_loader.dataset)
        return metrics["val_loss"], n_samples, metrics


def build_client_fn(model_config: ModelConfig, data_config: DataConfig, device: torch.device, **kwargs):
    """Factory for Flower simulation."""
    def client_fn(cid: str) -> VulnDetectClient:
        return VulnDetectClient(
            client_id=int(cid),
            model_config=model_config,
            data_config=data_config,
            device=device,
            **kwargs,
        )
    return client_fn
