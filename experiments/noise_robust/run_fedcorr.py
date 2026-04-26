"""
FedCorr-inspired noise-robust FL experiment.
Stage 1: Pre-processing (LID-based noisy client detection + per-sample correction)
Stage 2: Federated finetuning on clean clients
Stage 3: Standard FL on all clients with corrected labels
"""
import argparse
import torch
import flwr as fl
from omegaconf import OmegaConf

from src.models.codebert import ModelConfig
from src.client.dataset import DataConfig
from src.client.fl_client import build_client_fn
from src.server.strategy import NoisAwareFedAvg
from src.noise.noise_injection import add_noise_to_partition
from src.utils.logger import setup_logger
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/noise_robust.yaml")
    parser.add_argument("--noise_rate", type=float, default=None)
    parser.add_argument("--noise_type", type=str, default=None)
    args = parser.parse_args()

    cfg = OmegaConf.load(args.config)
    if args.noise_rate: cfg.noise.noise_rate = args.noise_rate
    if args.noise_type: cfg.noise.noise_type = args.noise_type

    logger = setup_logger("fedcorr", cfg.logging.log_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Running FedCorr | noise_rate={cfg.noise.noise_rate} | type={cfg.noise.noise_type}")

    # ── Stage 1: inject noise into partitions ────────────────────────────────
    if cfg.noise.enabled:
        import numpy as np
        rng = np.random.default_rng(cfg.training.seed)
        n_noisy = int(cfg.data.n_clients * cfg.noise.noisy_client_fraction)
        noisy_clients = rng.choice(cfg.data.n_clients, n_noisy, replace=False).tolist()
        logger.info(f"Noisy clients: {noisy_clients}")
        for cid in noisy_clients:
            for split in ["train"]:
                p = Path(cfg.data.partition_dir) / f"client_{cid}_{split}.json"
                add_noise_to_partition(p, cfg.noise.noise_rate, cfg.noise.noise_type)

    # ── Stage 2-3: FL with noise-aware strategy ───────────────────────────────
    model_config = ModelConfig(**cfg.model)
    data_config = DataConfig(**cfg.data)
    client_fn = build_client_fn(
        model_config=model_config,
        data_config=data_config,
        device=device,
        local_epochs=cfg.training.local_epochs,
        proximal_mu=cfg.fedcorr.proximal_mu,
    )

    strategy = NoisAwareFedAvg(
        noise_threshold=cfg.noise.threshold_quantile,
        fraction_fit=cfg.fl.fraction_fit,
        fraction_evaluate=cfg.fl.fraction_evaluate,
        min_fit_clients=cfg.fl.min_fit_clients,
        min_evaluate_clients=cfg.fl.min_evaluate_clients,
        min_available_clients=cfg.fl.min_available_clients,
    )

    history = fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=cfg.data.n_clients,
        config=fl.server.ServerConfig(num_rounds=cfg.fl.num_rounds),
        strategy=strategy,
    )
    logger.info(f"FedCorr complete. Final: {history.metrics_distributed}")


if __name__ == "__main__":
    main()
