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
        dropout: float | list[float] = 0.3,
    ) -> None:
        super().__init__()
        self.layers = nn.ModuleList()
        in_dim = input_dim
        
        for i, h in enumerate(hidden_dims):
            d = dropout[i] if isinstance(dropout, list) else dropout
            self.layers.append(nn.Sequential(
                nn.Linear(in_dim, h),
                nn.BatchNorm1d(h),
                nn.ReLU(),
                nn.Dropout(d)
            ))
            in_dim = h
            
        self.output_layer = nn.Linear(in_dim, 1)
        
        # Skip connection: projeta input para dim da primeira camada oculta
        self.input_shortcut = nn.Linear(input_dim, hidden_dims[0])

        # Inicialização He (Kaiming) para estabilidade
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Primeira camada com Skip Connection
        identity = self.input_shortcut(x)
        out = self.layers[0](x)
        out = out + identity # Aqui a mágica acontece!
        
        # Restante das camadas
        for i in range(1, len(self.layers)):
            out = self.layers[i](out)
            
        return self.output_layer(out).squeeze(1)

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """Retorna P(churn=1) para cada amostra (sem gradiente)."""
        self.eval()
        with torch.no_grad():
            return torch.sigmoid(self(x))
