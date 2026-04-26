"""Run FedAvg baseline on DiverseVul with CodeBERT."""
import argparse
import torch
import flwr as fl
from omegaconf import OmegaConf

from src.models.codebert import ModelConfig
from src.client.dataset import DataConfig
from src.client.fl_client import build_client_fn
from src.server.strategy import NoisAwareFedAvg
from src.utils.logger import setup_logger


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/fedavg.yaml")
    args = parser.parse_args()

    cfg = OmegaConf.load(args.config)
    logger = setup_logger("fedavg", cfg.logging.log_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    model_config = ModelConfig(**cfg.model)
    data_config = DataConfig(**cfg.data)

    client_fn = build_client_fn(
        model_config=model_config,
        data_config=data_config,
        device=device,
        local_epochs=cfg.training.local_epochs,
    )

    strategy = NoisAwareFedAvg(
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
    logger.info(f"Training complete. Final metrics: {history.metrics_distributed}")


if __name__ == "__main__":
    main()
