"""
Custom Flower strategies for noise-robust FL.
"""
from __future__ import annotations
from typing import Optional, Union
import numpy as np
import flwr as fl
from flwr.common import Parameters, Scalar, FitRes, EvaluateRes
from flwr.server.client_proxy import ClientProxy


class NoisAwareFedAvg(fl.server.strategy.FedAvg):
    """
    FedAvg extended with optional noisy-client filtering.
    Pass a `noise_threshold` to down-weight clients with high estimated noise.
    """

    def __init__(self, noise_threshold: float = 1.0, **kwargs):
        super().__init__(**kwargs)
        self.noise_threshold = noise_threshold
        self.client_noise_estimates: dict[str, float] = {}

    def aggregate_fit(self, server_round, results, failures):
        # Filter out clients whose reported noise_level exceeds threshold
        filtered = [
            (client, fit_res)
            for client, fit_res in results
            if fit_res.metrics.get("noise_level", 0.0) <= self.noise_threshold
        ]
        if not filtered:
            filtered = results  # fallback: use all
        return super().aggregate_fit(server_round, filtered, failures)
