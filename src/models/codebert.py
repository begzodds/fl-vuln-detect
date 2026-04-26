"""
CodeBERT encoder for vulnerability detection.
Wraps microsoft/codebert-base with a binary classification head.
"""
from __future__ import annotations
import torch
import torch.nn as nn
from transformers import RobertaModel, RobertaTokenizer
from dataclasses import dataclass


@dataclass
class ModelConfig:
    model_name: str = "microsoft/codebert-base"
    num_labels: int = 2
    dropout: float = 0.1
    freeze_encoder: bool = False
    max_length: int = 512


class CodeBERTClassifier(nn.Module):
    """CodeBERT with a linear classification head."""

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.encoder = RobertaModel.from_pretrained(config.model_name)
        hidden = self.encoder.config.hidden_size  # 768

        if config.freeze_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False

        self.classifier = nn.Sequential(
            nn.Dropout(config.dropout),
            nn.Linear(hidden, hidden // 2),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(hidden // 2, config.num_labels),
        )

    def forward(self, input_ids, attention_mask, labels=None):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        # Use [CLS] token representation
        cls_repr = outputs.last_hidden_state[:, 0, :]
        logits = self.classifier(cls_repr)

        loss = None
        if labels is not None:
            loss_fn = nn.CrossEntropyLoss()
            loss = loss_fn(logits, labels)

        return {"loss": loss, "logits": logits, "embeddings": cls_repr}

    def get_parameters(self) -> list:
        """Return model parameters as a list of numpy arrays (for Flower)."""
        return [p.detach().cpu().numpy() for p in self.parameters()]

    def set_parameters(self, parameters: list) -> None:
        """Set model parameters from a list of numpy arrays (for Flower)."""
        import numpy as np
        params_dict = zip(self.parameters(), parameters)
        for param, new_param in params_dict:
            param.data = torch.tensor(new_param)


def load_tokenizer(model_name: str = "microsoft/codebert-base") -> RobertaTokenizer:
    return RobertaTokenizer.from_pretrained(model_name)
