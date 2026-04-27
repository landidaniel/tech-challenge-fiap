"""Arquitetura do MLP de churn (PyTorch)."""

from __future__ import annotations

import torch
import torch.nn as nn


class ChurnMLP(nn.Module):
    """Multi-Layer Perceptron para classificacao binaria de churn.

    Arquitetura:
        Input -> [Linear -> BatchNorm1d -> ReLU -> Dropout] x N -> Linear(1)

    A saida e um **logit** (sem sigmoid). Use ``BCEWithLogitsLoss`` no
    treinamento e ``torch.sigmoid`` na inferencia.

    Parameters
    ----------
    input_dim : int
        Numero de features de entrada.
    hidden_dims : list[int]
        Neuronios por camada oculta, ex: [256, 128, 64].
    dropout : float
        Taxa de dropout aplicada apos cada bloco oculto (default 0.3).
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: list[int],
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        in_dim = input_dim
        for h in hidden_dims:
            layers += [
                nn.Linear(in_dim, h),
                nn.BatchNorm1d(h),
                nn.ReLU(),
                nn.Dropout(dropout),
            ]
            in_dim = h
        layers.append(nn.Linear(in_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(1)

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """Retorna P(churn=1) para cada amostra (sem gradiente)."""
        self.eval()
        with torch.no_grad():
            return torch.sigmoid(self(x))
